# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flat THD kernels for boundary-safe packed Hyena prediction.

These kernels intentionally implement inference-only operators. Training uses the
autograd-backed bucketed reference until equivalent packed backward kernels are
available and validated.
"""

from dataclasses import dataclass

import torch


try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


@dataclass(frozen=True)
class ModalChunkMetadata:
    """Compact chunk schedule for a flat segmented modal recurrence."""

    chunk_starts: torch.Tensor
    chunk_lengths: torch.Tensor
    chunk_sequence_ids: torch.Tensor
    chunked_sequence_ids: torch.Tensor
    unchunked_sequence_ids: torch.Tensor
    sequence_chunk_offsets: torch.Tensor
    continuation_chunk_ids: torch.Tensor


_MAX_SIGNED_INT32 = torch.iinfo(torch.int32).max


def _requires_64bit_indexing(*element_counts: int) -> bool:
    """Return whether a flat tensor span can exceed Triton's signed-i32 offsets."""
    return max(element_counts, default=0) > _MAX_SIGNED_INT32


@dataclass(frozen=True)
class ModalPoles:
    """Precomputed fp32 ``[groups, order]`` modal poles shared by every packed modal kernel."""

    decay: torch.Tensor
    log_decay: torch.Tensor


def modal_poles(gamma: torch.Tensor, poles_parameter: torch.Tensor) -> ModalPoles:
    """Reduce Evo2's modal ``(gamma, p)`` parameters to per-mode decay and its log.

    ``log_decay`` is the quantity ``ImplicitModalFilter.get_logp`` feeds the unpacked
    recurrence; ``decay`` is its exponential. The token recurrences read ``decay``
    directly, while the chunk carry needs ``exp(log_decay * chunk_length)`` because
    recovering the log from a decay near one would lose precision for slow modes.
    Callers compute this once per layer and reuse it across prefill and decode.
    """
    if gamma.shape != poles_parameter.shape or gamma.ndim != 2:
        raise ValueError("gamma and poles_parameter must share a [groups, order] shape")
    log_decay = -torch.exp(poles_parameter.to(torch.float32)) * torch.exp(gamma.to(torch.float32))
    return ModalPoles(decay=torch.exp(log_decay).contiguous(), log_decay=log_decay.contiguous())


def _validate_modal_poles(poles: ModalPoles, expected_shape: tuple[int, int], device: torch.device) -> None:
    for tensor in (poles.decay, poles.log_decay):
        if (
            tensor.shape != expected_shape
            or tensor.dtype != torch.float32
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError("modal poles must be contiguous fp32 [groups,16] on the input device; use modal_poles")


try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

PACKED_CAUSAL_CONV_AVAILABLE = causal_conv1d_fn is not None or TRITON_AVAILABLE


if TRITON_AVAILABLE:
    # The ``.to(tl.bfloat16).to(tl.float32)`` round trips below are unnecessary: all
    # accumulation is fp32, and the subquadratic-ops kernels never round intermediates
    # yet remain compatible with the PyTorch path. They mimic PyTorch's BF16
    # intermediate materialization and can be removed.

    @triton.jit
    def _segmented_causal_conv1d_kernel(
        input,
        weight,
        sequence_ids,
        output,
        total_tokens: tl.constexpr,
        channels: tl.constexpr,
        taps: tl.constexpr,
        group_width: tl.constexpr,
        use_64bit_indexing: tl.constexpr,
        block_tokens: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        """Small causal projection FIR with explicit packed-boundary checks."""
        token = tl.program_id(0) * block_tokens + tl.arange(0, block_tokens)
        if use_64bit_indexing:
            token = token.to(tl.int64)
        channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
        token_grid = token[:, None]
        channel_grid = channel[None, :]
        valid = (token_grid < total_tokens) & (channel_grid < channels)
        current_sequence = tl.load(
            sequence_ids + token,
            mask=token < total_tokens,
            other=-1,
        )[:, None]
        filter_index = channel_grid // group_width
        accumulator = tl.zeros((block_tokens, block_channels), dtype=tl.float32)

        for lag in tl.static_range(0, taps):
            source_token = token - lag
            source_sequence = tl.load(
                sequence_ids + source_token,
                mask=(source_token >= 0) & (token < total_tokens),
                other=-2,
            )[:, None]
            source_grid = source_token[:, None]
            source_valid = valid & (source_sequence == current_sequence)
            value = tl.load(
                input + source_grid * channels + channel_grid,
                mask=source_valid,
                other=0.0,
            ).to(tl.float32)
            tap = tl.load(
                weight + filter_index * taps + (taps - 1 - lag),
                mask=channel_grid < channels,
                other=0.0,
            ).to(tl.float32)
            accumulator += value * tap

        tl.store(output + token_grid * channels + channel_grid, accumulator, mask=valid)

    @triton.jit
    def _segmented_fir_from_projection_kernel(
        projection,
        recurrent_input,
        weight,
        local_positions,
        diagonal,
        output,
        total_tokens: tl.constexpr,
        channels: tl.constexpr,
        taps: tl.constexpr,
        group_width: tl.constexpr,
        flip_filter: tl.constexpr,
        pregate: tl.constexpr,
        postgate: tl.constexpr,
        has_diagonal: tl.constexpr,
        materialized_input: tl.constexpr,
        use_64bit_indexing: tl.constexpr,
        block_tokens: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        token = tl.program_id(0) * block_tokens + tl.arange(0, block_tokens)
        if use_64bit_indexing:
            token = token.to(tl.int64)
        channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
        token_grid = token[:, None]
        channel_grid = channel[None, :]
        valid = (token_grid < total_tokens) & (channel_grid < channels)
        position = tl.load(local_positions + token, mask=token < total_tokens, other=0)[:, None]
        accumulator = tl.zeros((block_tokens, block_channels), dtype=tl.float32)

        for lag in tl.static_range(0, taps):
            source_token = token_grid - lag
            lag_valid = valid & (position >= lag)
            if materialized_input:
                value = tl.load(
                    recurrent_input + source_token * channels + channel_grid,
                    mask=lag_valid,
                    other=0.0,
                ).to(tl.float32)
            else:
                projection_base = source_token * (3 * channels) + 3 * channel_grid
                x2 = tl.load(projection + projection_base + 1, mask=lag_valid, other=0.0)
                value = tl.load(projection + projection_base + 2, mask=lag_valid, other=0.0)
                if pregate:
                    # Match PyTorch BF16 materialization of x2 * value before convolution.
                    value = (x2 * value).to(tl.bfloat16).to(tl.float32)
            filter_index = channel_grid // group_width
            if flip_filter:
                weight_offset = taps - 1 - lag
            else:
                weight_offset = lag
            tap = tl.load(weight + filter_index * taps + weight_offset, mask=channel_grid < channels, other=0.0)
            accumulator += value * tap

        current_base = token_grid * (3 * channels) + 3 * channel_grid
        if materialized_input:
            current_value = tl.load(
                recurrent_input + token_grid * channels + channel_grid,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
        else:
            current_x2 = tl.load(projection + current_base + 1, mask=valid, other=0.0)
            current_value = tl.load(projection + current_base + 2, mask=valid, other=0.0)
            if pregate:
                current_value = (current_x2 * current_value).to(tl.bfloat16).to(tl.float32)
        if has_diagonal:
            diagonal_value = tl.load(diagonal + channel_grid, mask=channel_grid < channels, other=0.0)
            accumulator += current_value * diagonal_value

        # Existing Evo2 operators round the convolution to the activation dtype
        # before applying the outer gate.
        accumulator = accumulator.to(tl.bfloat16).to(tl.float32)
        if postgate:
            outer_gate = tl.load(projection + current_base, mask=valid, other=0.0)
            accumulator *= outer_gate
        tl.store(output + token_grid * channels + channel_grid, accumulator, mask=valid)

    @triton.jit
    def _segmented_modal_from_projection_kernel(
        projection,
        diagonal,
        residues,
        decays,
        cu_seqlens,
        selected_sequence_ids,
        output,
        final_state_out,
        channels: tl.constexpr,
        order: tl.constexpr,
        group_width: tl.constexpr,
        write_final_state: tl.constexpr,
        use_selected_sequence_ids: tl.constexpr,
        use_64bit_indexing: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        sequence_index = tl.program_id(0)
        if use_selected_sequence_ids:
            sequence = tl.load(selected_sequence_ids + sequence_index)
        else:
            sequence = sequence_index
        if use_64bit_indexing:
            sequence = sequence.to(tl.int64)
        channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
        valid_channel = channel < channels
        filter_index = channel // group_width
        mode = tl.arange(0, order)
        parameter_offset = filter_index[:, None] * order + mode[None, :]
        parameter_mask = valid_channel[:, None]
        residue = tl.load(residues + parameter_offset, mask=parameter_mask, other=0.0).to(tl.float32)
        decay = tl.load(decays + parameter_offset, mask=parameter_mask, other=0.0)
        state = tl.zeros((block_channels, order), dtype=tl.float32)
        sequence_start = tl.load(cu_seqlens + sequence)
        sequence_end = tl.load(cu_seqlens + sequence + 1)
        if use_64bit_indexing:
            sequence_start = sequence_start.to(tl.int64)
            sequence_end = sequence_end.to(tl.int64)

        for token in tl.range(sequence_start, sequence_end):
            projection_base = token * (3 * channels) + 3 * channel
            x1 = tl.load(projection + projection_base, mask=valid_channel, other=0.0).to(tl.float32)
            x2 = tl.load(projection + projection_base + 1, mask=valid_channel, other=0.0).to(tl.float32)
            value = tl.load(projection + projection_base + 2, mask=valid_channel, other=0.0).to(tl.float32)
            recurrent_input = (x2 * value).to(tl.bfloat16).to(tl.float32)
            state = recurrent_input[:, None] + decay * state
            convolved = tl.sum(residue * state, axis=1)
            diagonal_value = tl.load(diagonal + channel, mask=valid_channel, other=0.0).to(tl.float32)
            mixed = (convolved + recurrent_input * diagonal_value).to(tl.bfloat16).to(tl.float32)
            tl.store(output + token * channels + channel, mixed * x1, mask=valid_channel)

        if write_final_state:
            final_offset = (sequence * channels + channel[:, None]) * order + mode[None, :]
            tl.store(final_state_out + final_offset, state, mask=parameter_mask)

    @triton.jit
    def _fused_hyena_decode_from_projection_kernel(
        projection,
        projection_state,
        projection_weight,
        mixer_state,
        mixer_weight,
        diagonal,
        residues,
        decays,
        output,
        projection_state_stride_batch,
        projection_state_stride_channel,
        projection_state_stride_tap,
        mixer_state_stride_batch,
        mixer_state_stride_channel,
        mixer_state_stride_tap,
        channels: tl.constexpr,
        projection_taps: tl.constexpr,
        projection_group_width: tl.constexpr,
        mixer_taps: tl.constexpr,
        mixer_group_width: tl.constexpr,
        operator_kind: tl.constexpr,
        modal_order: tl.constexpr,
        has_diagonal: tl.constexpr,
        diagonal_group_width: tl.constexpr,
        block_channels: tl.constexpr,
        projection_ring_block: tl.constexpr,
        mixer_ring_block: tl.constexpr,
    ):
        """Fuse the shared projection FIR with one single-token Hyena recurrence.

        Every ring buffer is read into registers once, then shifted by storing those
        registers one tap lower after a CTA barrier. Reloading the neighbouring tap in
        place would race: lanes in other warps (including replicated lanes when a tile
        is smaller than the CTA) may already have overwritten it.
        """
        batch = tl.program_id(0)
        channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
        valid_channel = channel < channels
        x1 = tl.zeros((block_channels,), dtype=tl.float32)
        x2 = tl.zeros((block_channels,), dtype=tl.float32)
        value = tl.zeros((block_channels,), dtype=tl.float32)
        projection_ring_index = tl.arange(0, projection_ring_block)
        projection_ring_valid = projection_ring_index[None, :] < projection_taps - 1

        for feature in tl.static_range(0, 3):
            projected_channel = 3 * channel + feature
            projection_group = projected_channel // projection_group_width
            current = tl.load(
                projection + batch * (3 * channels) + projected_channel,
                mask=valid_channel,
                other=0.0,
            ).to(tl.float32)
            accumulator = current * tl.load(
                projection_weight + projection_group * projection_taps + projection_taps - 1,
                mask=valid_channel,
                other=0.0,
            ).to(tl.float32)
            state_base = batch * projection_state_stride_batch + projected_channel * projection_state_stride_channel
            ring_mask = valid_channel[:, None] & projection_ring_valid
            ring_offset = state_base[:, None] + projection_ring_index[None, :] * projection_state_stride_tap
            ring = tl.load(projection_state + ring_offset, mask=ring_mask, other=0.0).to(tl.float32)
            ring_tap = tl.load(
                projection_weight + projection_group[:, None] * projection_taps + projection_ring_index[None, :],
                mask=ring_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(ring * ring_tap, axis=1)

            tl.debug_barrier()
            shift_mask = ring_mask & (projection_ring_index[None, :] >= 1)
            tl.store(projection_state + ring_offset - projection_state_stride_tap, ring, mask=shift_mask)
            tl.store(
                projection_state + state_base + (projection_taps - 2) * projection_state_stride_tap,
                current,
                mask=valid_channel,
            )
            projected_value = accumulator.to(tl.bfloat16).to(tl.float32)
            if feature == 0:
                x1 = projected_value
            elif feature == 1:
                x2 = projected_value
            else:
                value = projected_value
        recurrent_input = (x2 * value).to(tl.bfloat16).to(tl.float32)
        diagonal_value = tl.zeros((block_channels,), dtype=tl.float32)
        if has_diagonal:
            diagonal_value = tl.load(
                diagonal + channel // diagonal_group_width,
                mask=valid_channel,
                other=0.0,
            ).to(tl.float32)
        mixer_state_base = batch * mixer_state_stride_batch + channel * mixer_state_stride_channel

        if operator_kind < 2:
            mixer_group = channel // mixer_group_width
            state_index = tl.arange(0, mixer_ring_block)
            ring_valid = state_index[None, :] < mixer_taps - 1
            state_mask = valid_channel[:, None] & ring_valid
            state_offset = mixer_state_base[:, None] + state_index[None, :] * mixer_state_stride_tap
            state_value = tl.load(mixer_state + state_offset, mask=state_mask, other=0.0).to(tl.float32)
            if operator_kind == 0:
                current_tap_index = mixer_taps - 1
                tap_index = state_index
            else:
                current_tap_index = 0
                tap_index = mixer_taps - 1 - state_index
            mixed = recurrent_input * tl.load(
                mixer_weight + mixer_group * mixer_taps + current_tap_index,
                mask=valid_channel,
                other=0.0,
            ).to(tl.float32)
            tap = tl.load(
                mixer_weight + mixer_group[:, None] * mixer_taps + tap_index[None, :],
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)
            mixed += tl.sum(state_value * tap, axis=1)
            mixed += diagonal_value * recurrent_input

            tl.debug_barrier()
            shift_mask = state_mask & (state_index[None, :] >= 1)
            tl.store(mixer_state + state_offset - mixer_state_stride_tap, state_value, mask=shift_mask)
            tl.store(
                mixer_state + mixer_state_base + (mixer_taps - 2) * mixer_state_stride_tap,
                recurrent_input,
                mask=valid_channel,
            )
        else:
            mixer_group = channel // mixer_group_width
            mode = tl.arange(0, modal_order)
            parameter_offset = mixer_group[:, None] * modal_order + mode[None, :]
            parameter_mask = valid_channel[:, None]
            residue = tl.load(residues + parameter_offset, mask=parameter_mask, other=0.0).to(tl.float32)
            decay = tl.load(decays + parameter_offset, mask=parameter_mask, other=0.0)
            state_offset = mixer_state_base[:, None] + mode[None, :] * mixer_state_stride_tap
            state = tl.load(mixer_state + state_offset, mask=parameter_mask, other=0.0).to(tl.float32)
            state = recurrent_input[:, None] + decay * state
            tl.debug_barrier()
            tl.store(mixer_state + state_offset, state, mask=parameter_mask)
            mixed = tl.sum(residue * state, axis=1) + diagonal_value * recurrent_input

        mixed = mixed.to(tl.bfloat16).to(tl.float32)
        tl.store(
            output + batch * channels + channel,
            x1 * mixed,
            mask=valid_channel,
        )

    @triton.jit
    def _segmented_modal_chunk_summarize_kernel(
        projection,
        diagonal,
        residues,
        decays,
        chunk_starts,
        chunk_lengths,
        chunk_sequence_ids,
        output,
        chunk_states,
        channels: tl.constexpr,
        order: tl.constexpr,
        group_width: tl.constexpr,
        use_64bit_indexing: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        """Summarize every chunk, emitting output only for each sequence's first chunk."""
        chunk = tl.program_id(0)
        if use_64bit_indexing:
            chunk = chunk.to(tl.int64)
        channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
        valid_channel = channel < channels
        filter_index = channel // group_width
        mode = tl.arange(0, order)
        parameter_offset = filter_index[:, None] * order + mode[None, :]
        parameter_mask = valid_channel[:, None]
        decay = tl.load(decays + parameter_offset, mask=parameter_mask, other=0.0)
        sequence_id = tl.load(chunk_sequence_ids + chunk)
        previous_sequence_id = tl.load(chunk_sequence_ids + chunk - 1, mask=chunk > 0, other=-1)
        is_first_chunk = sequence_id != previous_sequence_id
        residue = tl.load(
            residues + parameter_offset,
            mask=parameter_mask & is_first_chunk,
            other=0.0,
        ).to(tl.float32)
        diagonal_value = tl.load(
            diagonal + channel,
            mask=valid_channel & is_first_chunk,
            other=0.0,
        ).to(tl.float32)
        state = tl.zeros((block_channels, order), dtype=tl.float32)
        chunk_start = tl.load(chunk_starts + chunk)
        chunk_end = chunk_start + tl.load(chunk_lengths + chunk)
        if use_64bit_indexing:
            chunk_start = chunk_start.to(tl.int64)
            chunk_end = chunk_end.to(tl.int64)

        for token in tl.range(chunk_start, chunk_end):
            projection_base = token * (3 * channels) + 3 * channel
            x2 = tl.load(projection + projection_base + 1, mask=valid_channel, other=0.0).to(tl.float32)
            value = tl.load(projection + projection_base + 2, mask=valid_channel, other=0.0).to(tl.float32)
            recurrent_input = (x2 * value).to(tl.bfloat16).to(tl.float32)
            state = recurrent_input[:, None] + decay * state
            if is_first_chunk:
                x1 = tl.load(projection + projection_base, mask=valid_channel, other=0.0).to(tl.float32)
                convolved = tl.sum(residue * state, axis=1)
                mixed = (convolved + recurrent_input * diagonal_value).to(tl.bfloat16).to(tl.float32)
                tl.store(output + token * channels + channel, mixed * x1, mask=valid_channel)

        state_offset = (chunk * channels + channel[:, None]) * order + mode[None, :]
        tl.store(chunk_states + state_offset, state, mask=parameter_mask)

    @triton.jit
    def _segmented_modal_chunk_propagate_kernel(
        log_decays,
        chunk_lengths,
        sequence_chunk_offsets,
        chunked_sequence_ids,
        chunk_states,
        final_state_out,
        channels: tl.constexpr,
        order: tl.constexpr,
        group_width: tl.constexpr,
        write_final_state: tl.constexpr,
        use_64bit_indexing: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        """Scan compact chunk summaries and replace each with its initial carry."""
        sequence = tl.program_id(0)
        channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
        valid_channel = channel < channels
        filter_index = channel // group_width
        mode = tl.arange(0, order)
        parameter_offset = filter_index[:, None] * order + mode[None, :]
        parameter_mask = valid_channel[:, None]
        decay_exponent = tl.load(log_decays + parameter_offset, mask=parameter_mask, other=0.0)
        state = tl.zeros((block_channels, order), dtype=tl.float32)
        first_chunk = tl.load(sequence_chunk_offsets + sequence)
        last_chunk = tl.load(sequence_chunk_offsets + sequence + 1)
        if use_64bit_indexing:
            first_chunk = first_chunk.to(tl.int64)
            last_chunk = last_chunk.to(tl.int64)

        for chunk in tl.range(first_chunk, last_chunk):
            state_offset = (chunk * channels + channel[:, None]) * order + mode[None, :]
            summary = tl.load(chunk_states + state_offset, mask=parameter_mask, other=0.0).to(tl.float32)
            initial_state = state
            chunk_length = tl.load(chunk_lengths + chunk).to(tl.float32)
            state = summary + tl.exp(decay_exponent * chunk_length) * state
            tl.store(chunk_states + state_offset, initial_state, mask=parameter_mask)

        if write_final_state:
            output_sequence = tl.load(chunked_sequence_ids + sequence)
            if use_64bit_indexing:
                output_sequence = output_sequence.to(tl.int64)
            final_offset = (output_sequence * channels + channel[:, None]) * order + mode[None, :]
            tl.store(final_state_out + final_offset, state, mask=parameter_mask)

    @triton.jit
    def _segmented_modal_chunk_continue_kernel(
        projection,
        diagonal,
        residues,
        decays,
        chunk_starts,
        chunk_lengths,
        continuation_chunk_ids,
        output,
        chunk_states,
        channels: tl.constexpr,
        order: tl.constexpr,
        group_width: tl.constexpr,
        use_64bit_indexing: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        """Re-evaluate only non-initial chunks from their propagated carry."""
        continuation = tl.program_id(0)
        chunk = tl.load(continuation_chunk_ids + continuation)
        if use_64bit_indexing:
            chunk = chunk.to(tl.int64)
        channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
        valid_channel = channel < channels
        filter_index = channel // group_width
        mode = tl.arange(0, order)
        parameter_offset = filter_index[:, None] * order + mode[None, :]
        parameter_mask = valid_channel[:, None]
        residue = tl.load(residues + parameter_offset, mask=parameter_mask, other=0.0).to(tl.float32)
        decay = tl.load(decays + parameter_offset, mask=parameter_mask, other=0.0)
        state_offset = (chunk * channels + channel[:, None]) * order + mode[None, :]
        state = tl.load(chunk_states + state_offset, mask=parameter_mask, other=0.0).to(tl.float32)
        chunk_start = tl.load(chunk_starts + chunk)
        chunk_end = chunk_start + tl.load(chunk_lengths + chunk)
        if use_64bit_indexing:
            chunk_start = chunk_start.to(tl.int64)
            chunk_end = chunk_end.to(tl.int64)

        for token in tl.range(chunk_start, chunk_end):
            projection_base = token * (3 * channels) + 3 * channel
            x1 = tl.load(projection + projection_base, mask=valid_channel, other=0.0).to(tl.float32)
            x2 = tl.load(projection + projection_base + 1, mask=valid_channel, other=0.0).to(tl.float32)
            value = tl.load(projection + projection_base + 2, mask=valid_channel, other=0.0).to(tl.float32)
            recurrent_input = (x2 * value).to(tl.bfloat16).to(tl.float32)
            state = recurrent_input[:, None] + decay * state
            convolved = tl.sum(residue * state, axis=1)
            diagonal_value = tl.load(diagonal + channel, mask=valid_channel, other=0.0).to(tl.float32)
            mixed = (convolved + recurrent_input * diagonal_value).to(tl.bfloat16).to(tl.float32)
            tl.store(output + token * channels + channel, mixed * x1, mask=valid_channel)


def modal_chunk_metadata_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    chunk_size: int,
) -> ModalChunkMetadata:
    """Create a compact no-padding chunk schedule for a segmented recurrence.

    This performs one scalar device synchronization to size the compact arrays.
    Callers should cache the returned metadata across all long-Hyena layers.
    """
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.dtype != torch.int32
        or not cu_seqlens.is_cuda
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("cu_seqlens must be contiguous CUDA int32 [N+1]")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must describe at least one sequence")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    all_chunk_counts = torch.div(lengths + chunk_size - 1, chunk_size, rounding_mode="floor")
    if bool((all_chunk_counts == 0).any().item()):
        raise ValueError("cu_seqlens must describe strictly positive sequence lengths")

    sequence_ids = torch.arange(lengths.numel(), dtype=torch.int32, device=cu_seqlens.device)
    chunked_mask = all_chunk_counts > 1
    chunked_sequence_ids = sequence_ids[chunked_mask].contiguous()
    unchunked_sequence_ids = sequence_ids[~chunked_mask].contiguous()
    chunk_counts = all_chunk_counts[chunked_mask].contiguous()
    chunked_sequence_count = chunked_sequence_ids.numel()
    total_chunks = int(chunk_counts.sum().item())

    sequence_chunk_offsets = torch.empty(
        chunked_sequence_count + 1,
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    sequence_chunk_offsets[0] = 0
    torch.cumsum(chunk_counts, dim=0, dtype=torch.int32, out=sequence_chunk_offsets[1:])
    chunk_sequence_ids = torch.repeat_interleave(
        chunked_sequence_ids,
        chunk_counts.to(torch.int64),
        output_size=total_chunks,
    )
    chunk_ids = torch.arange(total_chunks, dtype=torch.int32, device=cu_seqlens.device)
    repeated_chunk_offsets = torch.repeat_interleave(
        sequence_chunk_offsets[:-1],
        chunk_counts.to(torch.int64),
        output_size=total_chunks,
    )
    chunk_indices = chunk_ids - repeated_chunk_offsets
    chunk_starts = cu_seqlens[chunk_sequence_ids.to(torch.int64)] + chunk_indices * chunk_size
    sequence_ends = cu_seqlens[chunk_sequence_ids.to(torch.int64) + 1]
    chunk_lengths = torch.minimum(sequence_ends - chunk_starts, torch.full_like(chunk_starts, chunk_size))

    continuation_counts = chunk_counts - 1
    continuation_count = total_chunks - chunked_sequence_count
    if continuation_count:
        compact_sequence_ids = torch.arange(
            chunked_sequence_count,
            dtype=torch.int32,
            device=cu_seqlens.device,
        )
        continuation_sequence_ids = torch.repeat_interleave(
            compact_sequence_ids,
            continuation_counts.to(torch.int64),
            output_size=continuation_count,
        )
        continuation_offsets = torch.empty_like(sequence_chunk_offsets)
        continuation_offsets[0] = 0
        torch.cumsum(continuation_counts, dim=0, dtype=torch.int32, out=continuation_offsets[1:])
        continuation_ids = torch.arange(
            continuation_count,
            dtype=torch.int32,
            device=cu_seqlens.device,
        )
        continuation_indices = continuation_ids - torch.repeat_interleave(
            continuation_offsets[:-1],
            continuation_counts.to(torch.int64),
            output_size=continuation_count,
        )
        continuation_chunk_ids = (
            sequence_chunk_offsets[continuation_sequence_ids.to(torch.int64)] + continuation_indices + 1
        )
    else:
        continuation_chunk_ids = torch.empty(0, dtype=torch.int32, device=cu_seqlens.device)

    return ModalChunkMetadata(
        chunk_starts=chunk_starts.contiguous(),
        chunk_lengths=chunk_lengths.contiguous(),
        chunk_sequence_ids=chunk_sequence_ids.contiguous(),
        chunked_sequence_ids=chunked_sequence_ids,
        unchunked_sequence_ids=unchunked_sequence_ids,
        sequence_chunk_offsets=sequence_chunk_offsets,
        continuation_chunk_ids=continuation_chunk_ids.contiguous(),
    )


def local_positions_from_cu_seqlens(cu_seqlens: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """Build zero-based positions for every physical packed segment without a host sync."""
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_cuda:
        raise ValueError("cu_seqlens must be CUDA int32 [N+1]")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    starts = torch.repeat_interleave(cu_seqlens[:-1], lengths.to(torch.int64), output_size=total_tokens)
    return torch.arange(total_tokens, dtype=torch.int32, device=cu_seqlens.device) - starts


def sequence_ids_from_cu_seqlens(cu_seqlens: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """Build the packed sequence id consumed by causal-conv's boundary reset."""
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_cuda:
        raise ValueError("cu_seqlens must be CUDA int32 [N+1]")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    sequence_ids = torch.arange(cu_seqlens.numel() - 1, dtype=torch.int32, device=cu_seqlens.device)
    return torch.repeat_interleave(sequence_ids, lengths.to(torch.int64), output_size=total_tokens)


def fused_hyena_decode_from_projection(
    projection: torch.Tensor,
    projection_state: torch.Tensor,
    projection_weight: torch.Tensor,
    mixer_state: torch.Tensor,
    mixer_weight: torch.Tensor,
    diagonal: torch.Tensor | None,
    residues: torch.Tensor | None,
    poles: ModalPoles | None,
    *,
    projection_group_width: int,
    mixer_group_width: int,
    operator: str,
    diagonal_group_width: int = 1,
) -> torch.Tensor:
    """Fuse one Hyena layer's projection FIR and single-token mixer recurrence.

    ``projection`` is the interleaved ``[x1, x2, v]`` dense projection for one decode
    position per request. Both recurrent state views are updated in place, preserving
    their aliases into MCore's dynamic-context state slots. ``residues`` and the
    precomputed ``modal_poles`` output are only read by the modal operator and may be
    ``None`` otherwise.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused Hyena decode")
    if operator not in {"short", "medium", "modal"}:
        raise ValueError(f"Unsupported Hyena decode operator {operator!r}")
    if (
        projection.ndim != 2
        or not projection.is_cuda
        or projection.dtype != torch.bfloat16
        or not projection.is_contiguous()
        or projection.shape[1] % 3 != 0
    ):
        raise ValueError("projection must be contiguous CUDA BF16 [B,3C]")
    batch, projection_channels = projection.shape
    channels = projection_channels // 3
    if (
        projection_state.ndim != 3
        or projection_state.shape[:2] != (batch, projection_channels)
        or projection_state.dtype != torch.float32
        or not projection_state.is_cuda
    ):
        raise ValueError("projection_state must be CUDA FP32 [B,3C,Kproj-1]")
    if (
        projection_weight.ndim != 2
        or not projection_weight.is_cuda
        or not projection_weight.is_contiguous()
        or projection_weight.shape[0] * projection_group_width != projection_channels
        or projection_weight.shape[1] - 1 != projection_state.shape[2]
    ):
        raise ValueError("projection_weight and projection_group_width must match projection_state")
    if projection_weight.shape[1] < 2 or projection_weight.shape[1] > 4:
        raise ValueError("fused decode supports projection FIR widths from 2 through 4")
    if (
        mixer_state.ndim != 3
        or mixer_state.shape[:2] != (batch, channels)
        or mixer_state.dtype != torch.float32
        or not mixer_state.is_cuda
    ):
        raise ValueError("mixer_state must be CUDA FP32 [B,C,K]")
    has_diagonal = diagonal is not None
    if has_diagonal:
        if (
            diagonal.ndim != 1
            or not diagonal.is_cuda
            or not diagonal.is_contiguous()
            or diagonal_group_width <= 0
            or diagonal.numel() * diagonal_group_width != channels
        ):
            raise ValueError("diagonal and diagonal_group_width must cover C CUDA channels")
    else:
        # Triton still needs a pointer argument, but the constexpr branch never loads it.
        diagonal = projection

    operator_kind = {"short": 0, "medium": 1, "modal": 2}[operator]
    if operator_kind < 2:
        if (
            mixer_weight.ndim != 2
            or not mixer_weight.is_cuda
            or not mixer_weight.is_contiguous()
            or mixer_weight.shape[0] * mixer_group_width != channels
            or mixer_weight.shape[1] - 1 != mixer_state.shape[2]
        ):
            raise ValueError("mixer_weight and mixer_group_width must match FIR mixer_state")
        mixer_taps = mixer_weight.shape[1]
    else:
        if mixer_state.shape[2] != 16:
            raise ValueError("modal mixer_state must have order 16")
        expected_parameter_shape = (channels // mixer_group_width, 16)
        if (
            residues is None
            or residues.shape != expected_parameter_shape
            or not residues.is_cuda
            or not residues.is_contiguous()
        ):
            raise ValueError("modal residues must be contiguous CUDA [groups,16]")
        if poles is None:
            raise ValueError("modal decode requires precomputed modal_poles")
        _validate_modal_poles(poles, expected_parameter_shape, projection.device)
        mixer_taps = 1
    # Triton still needs pointer arguments for the FIR operators, but the constexpr branch never loads them.
    residues_argument = residues if residues is not None else projection
    decays_argument = poles.decay if poles is not None else projection

    if operator_kind < 2 and mixer_taps > 128:
        raise ValueError("fused decode supports FIR mixers up to 128 taps")
    output = torch.empty((batch, channels), dtype=projection.dtype, device=projection.device)
    # Measured on GB300 with graph-captured launches at C=4096: the short FIR gains from
    # four warps, while the wide medium ring and the modal state tiles reduce fastest
    # within one warp (four warps cost 2x on medium at batch 32).
    block_channels = {0: 64, 1: 16, 2: 16}[operator_kind]
    num_warps = 4 if operator_kind == 0 else 1
    projection_ring_block = triton.next_power_of_2(projection_weight.shape[1] - 1)
    mixer_ring_block = triton.next_power_of_2(mixer_taps - 1) if operator_kind < 2 else 1
    grid = (batch, triton.cdiv(channels, block_channels))
    _fused_hyena_decode_from_projection_kernel[grid](
        projection,
        projection_state,
        projection_weight,
        mixer_state,
        mixer_weight,
        diagonal,
        residues_argument,
        decays_argument,
        output,
        *projection_state.stride(),
        *mixer_state.stride(),
        channels=channels,
        projection_taps=projection_weight.shape[1],
        projection_group_width=projection_group_width,
        mixer_taps=mixer_taps,
        mixer_group_width=mixer_group_width,
        operator_kind=operator_kind,
        modal_order=16,
        has_diagonal=has_diagonal,
        diagonal_group_width=diagonal_group_width,
        block_channels=block_channels,
        projection_ring_block=projection_ring_block,
        mixer_ring_block=mixer_ring_block,
        num_warps=num_warps,
    )
    return output


def segmented_causal_conv1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    sequence_ids: torch.Tensor,
    *,
    group_width: int,
) -> torch.Tensor:
    """Run a width-2..4 causal FIR on flat ``[T,C]`` input with segment resets.

    Inference uses a Triton kernel with size-gated 64-bit pointer arithmetic.
    Grad-enabled execution uses ``causal_conv1d`` through a view-only transpose
    and retains the extension's native backward. Neither path pads or reshuffles.
    """
    if input.ndim != 2 or not input.is_cuda or not input.is_contiguous():
        raise ValueError("input must be contiguous CUDA [T,C]")
    if weight.ndim != 2 or not weight.is_cuda or not weight.is_contiguous():
        raise ValueError("weight must be contiguous CUDA [groups,taps]")
    if not 2 <= weight.shape[1] <= 4:
        raise ValueError("causal_conv1d only supports widths from 2 through 4")
    if weight.shape[0] * group_width != input.shape[1]:
        raise ValueError("weight groups and group_width must partition the input channels")
    if sequence_ids.shape != (input.shape[0],) or sequence_ids.dtype != torch.int32:
        raise ValueError("sequence_ids must be int32 [T]")
    if not torch.is_grad_enabled():
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for packed projection FIR inference")
        output = torch.empty_like(input)
        block_tokens = 8
        block_channels = 128
        grid = (triton.cdiv(input.shape[0], block_tokens), triton.cdiv(input.shape[1], block_channels))
        _segmented_causal_conv1d_kernel[grid](
            input,
            weight,
            sequence_ids,
            output,
            total_tokens=input.shape[0],
            channels=input.shape[1],
            taps=weight.shape[1],
            group_width=group_width,
            use_64bit_indexing=_requires_64bit_indexing(input.numel(), output.numel()),
            block_tokens=block_tokens,
            block_channels=block_channels,
            num_warps=4,
        )
        return output

    if causal_conv1d_fn is None:
        raise RuntimeError("causal_conv1d is required for packed projection FIR backward")
    expanded_weight = weight.repeat_interleave(group_width, dim=0)
    channel_last_view = input.T.unsqueeze(0)
    output = causal_conv1d_fn(
        channel_last_view,
        expanded_weight,
        bias=None,
        seq_idx=sequence_ids.unsqueeze(0),
        activation=None,
    )
    flat_output = output.squeeze(0).T
    if not flat_output.is_contiguous():
        flat_output = flat_output.contiguous()
    return flat_output


def segmented_tail(
    input: torch.Tensor,
    cu_seqlens: torch.Tensor,
    tail_length: int,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gather zero-left-padded segment tails as contiguous fp32 ``[N,C,K]`` state."""
    if input.ndim != 2 or not input.is_cuda or not input.is_contiguous():
        raise ValueError("input must be contiguous CUDA [T,C]")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_cuda:
        raise ValueError("cu_seqlens must be CUDA int32 [N+1]")
    if tail_length <= 0:
        raise ValueError(f"tail_length must be positive, got {tail_length}")
    starts = cu_seqlens[:-1, None].to(torch.int64)
    ends = cu_seqlens[1:, None].to(torch.int64)
    offsets = torch.arange(tail_length, device=input.device, dtype=torch.int64)
    source_positions = ends - tail_length + offsets
    valid = source_positions >= starts
    safe_positions = source_positions.clamp(min=0, max=max(0, input.shape[0] - 1))
    gathered = input[safe_positions]
    gathered = gathered * valid[..., None]
    return gathered.permute(0, 2, 1).to(output_dtype).contiguous()


def segmented_fir_from_projection(
    projection: torch.Tensor,
    weight: torch.Tensor,
    local_positions: torch.Tensor,
    *,
    group_width: int,
    flip_filter: bool,
    pregate: bool = True,
    postgate: bool = True,
    diagonal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a segmented FIR directly from interleaved ``[x1, x2, v]`` projections."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for packed Hyena inference kernels")
    if projection.ndim != 2 or not projection.is_cuda or not projection.is_contiguous():
        raise ValueError("projection must be contiguous CUDA [T, 3C]")
    if projection.shape[1] % 3 != 0:
        raise ValueError("projection feature dimension must be divisible by three")
    if weight.ndim != 2 or not weight.is_cuda or not weight.is_contiguous():
        raise ValueError("weight must be contiguous CUDA [groups, taps]")
    total_tokens = projection.shape[0]
    channels = projection.shape[1] // 3
    if weight.shape[0] * group_width != channels:
        raise ValueError("weight groups and group_width must partition the projection channels")
    if local_positions.shape != (total_tokens,) or not local_positions.is_cuda:
        raise ValueError("local_positions must be CUDA [T]")
    if diagonal is not None and diagonal.shape != (channels,):
        raise ValueError("diagonal must have one value per output channel")

    taps = weight.shape[1]
    materialized_input = taps >= 64
    recurrent_input = projection
    if materialized_input:
        split_projection = projection.view(total_tokens, channels, 3)
        recurrent_input = split_projection[..., 2]
        if pregate:
            recurrent_input = split_projection[..., 1] * recurrent_input
        if not recurrent_input.is_contiguous():
            recurrent_input = recurrent_input.contiguous()
    if taps >= 64:
        block_tokens, block_channels, num_warps = 1, 128, 4
    else:
        block_tokens, block_channels, num_warps = 8, 64, 4
    output = torch.empty((total_tokens, channels), dtype=projection.dtype, device=projection.device)
    use_64bit_indexing = _requires_64bit_indexing(
        projection.numel(),
        recurrent_input.numel(),
        output.numel(),
    )
    diagonal_argument = diagonal if diagonal is not None else weight
    grid = (triton.cdiv(total_tokens, block_tokens), triton.cdiv(channels, block_channels))
    _segmented_fir_from_projection_kernel[grid](
        projection,
        recurrent_input,
        weight,
        local_positions,
        diagonal_argument,
        output,
        total_tokens=total_tokens,
        channels=channels,
        taps=taps,
        group_width=group_width,
        flip_filter=flip_filter,
        pregate=pregate,
        postgate=postgate,
        has_diagonal=diagonal is not None,
        materialized_input=materialized_input,
        use_64bit_indexing=use_64bit_indexing,
        block_tokens=block_tokens,
        block_channels=block_channels,
        num_warps=num_warps,
    )
    return output


def segmented_modal_from_projection(
    projection: torch.Tensor,
    diagonal: torch.Tensor,
    residues: torch.Tensor,
    poles: ModalPoles,
    cu_seqlens: torch.Tensor,
    *,
    group_width: int,
    final_state_out: torch.Tensor | None = None,
    chunk_metadata: ModalChunkMetadata | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Apply Evo2's 16-mode recurrence independently within every packed segment.

    ``poles`` is the precomputed ``modal_poles`` output shared by every launch
    below. Supplying ``chunk_metadata`` (or the convenience ``chunk_size``
    argument) parallelizes long sequences without padding, sorting, or moving token
    rows. The compact FP32 chunk summaries are overwritten in-place with each
    chunk's initial carry before continuation chunks are evaluated.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for packed Hyena inference kernels")
    if projection.ndim != 2 or not projection.is_cuda or not projection.is_contiguous():
        raise ValueError("projection must be contiguous CUDA [T, 3C]")
    total_tokens, projected_channels = projection.shape
    if projected_channels % 3 != 0:
        raise ValueError("projection feature dimension must be divisible by three")
    channels = projected_channels // 3
    if diagonal.shape != (channels,):
        raise ValueError("diagonal must have one value per output channel")
    if residues.ndim != 2 or residues.shape[1] != 16:
        raise ValueError("packed modal inference currently requires exactly 16 modes")
    _validate_modal_poles(poles, tuple(residues.shape), projection.device)
    if residues.shape[0] * group_width != channels:
        raise ValueError("modal groups and group_width must partition the projection channels")
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.dtype != torch.int32
        or not cu_seqlens.is_cuda
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("cu_seqlens must be contiguous CUDA int32 [N+1]")
    sequence_count = cu_seqlens.numel() - 1
    if final_state_out is not None and (
        final_state_out.shape != (sequence_count, channels, 16)
        or final_state_out.dtype != torch.float32
        or not final_state_out.is_cuda
        or not final_state_out.is_contiguous()
    ):
        raise ValueError("final_state_out must be contiguous CUDA fp32 [N,C,16]")
    if chunk_metadata is not None and chunk_size is not None:
        raise ValueError("pass either chunk_metadata or chunk_size, not both")
    if chunk_size is not None:
        chunk_metadata = modal_chunk_metadata_from_cu_seqlens(cu_seqlens, chunk_size=chunk_size)

    output = torch.empty((total_tokens, channels), dtype=projection.dtype, device=projection.device)
    final_state_argument = final_state_out if final_state_out is not None else output
    use_64bit_indexing = _requires_64bit_indexing(
        projection.numel(),
        output.numel(),
        0 if final_state_out is None else final_state_out.numel(),
    )
    block_channels = 64
    if chunk_metadata is not None:
        metadata_tensors = (
            chunk_metadata.chunk_starts,
            chunk_metadata.chunk_lengths,
            chunk_metadata.chunk_sequence_ids,
            chunk_metadata.chunked_sequence_ids,
            chunk_metadata.unchunked_sequence_ids,
            chunk_metadata.sequence_chunk_offsets,
            chunk_metadata.continuation_chunk_ids,
        )
        if any(
            tensor.dtype != torch.int32
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != projection.device
            for tensor in metadata_tensors
        ):
            raise ValueError("modal chunk metadata must contain contiguous CUDA int32 tensors on the input device")
        total_chunks = chunk_metadata.chunk_starts.numel()
        if chunk_metadata.chunk_lengths.shape != (total_chunks,):
            raise ValueError("chunk_lengths must have one entry per chunk")
        if chunk_metadata.chunk_sequence_ids.shape != (total_chunks,):
            raise ValueError("chunk_sequence_ids must have one entry per chunk")
        chunked_sequence_count = chunk_metadata.chunked_sequence_ids.numel()
        unchunked_sequence_count = chunk_metadata.unchunked_sequence_ids.numel()
        if chunked_sequence_count + unchunked_sequence_count != sequence_count:
            raise ValueError("chunked and unchunked sequence ids must partition the packed batch")
        if chunk_metadata.sequence_chunk_offsets.shape != (chunked_sequence_count + 1,):
            raise ValueError("sequence_chunk_offsets must have one more entry than the chunked sequence count")

        if unchunked_sequence_count:
            unchunked_grid = (unchunked_sequence_count, triton.cdiv(channels, block_channels))
            _segmented_modal_from_projection_kernel[unchunked_grid](
                projection,
                diagonal,
                residues,
                poles.decay,
                cu_seqlens,
                chunk_metadata.unchunked_sequence_ids,
                output,
                final_state_argument,
                channels=channels,
                order=16,
                group_width=group_width,
                write_final_state=final_state_out is not None,
                use_selected_sequence_ids=True,
                use_64bit_indexing=use_64bit_indexing,
                block_channels=block_channels,
                num_warps=4,
            )

        chunk_states = torch.empty(
            total_chunks,
            channels,
            16,
            dtype=torch.float32,
            device=projection.device,
        )
        use_64bit_indexing = use_64bit_indexing or _requires_64bit_indexing(chunk_states.numel())
        if total_chunks:
            chunk_grid = (total_chunks, triton.cdiv(channels, block_channels))
            _segmented_modal_chunk_summarize_kernel[chunk_grid](
                projection,
                diagonal,
                residues,
                poles.decay,
                chunk_metadata.chunk_starts,
                chunk_metadata.chunk_lengths,
                chunk_metadata.chunk_sequence_ids,
                output,
                chunk_states,
                channels=channels,
                order=16,
                group_width=group_width,
                use_64bit_indexing=use_64bit_indexing,
                block_channels=block_channels,
                num_warps=4,
            )
        if chunked_sequence_count:
            sequence_grid = (chunked_sequence_count, triton.cdiv(channels, block_channels))
            _segmented_modal_chunk_propagate_kernel[sequence_grid](
                poles.log_decay,
                chunk_metadata.chunk_lengths,
                chunk_metadata.sequence_chunk_offsets,
                chunk_metadata.chunked_sequence_ids,
                chunk_states,
                final_state_argument,
                channels=channels,
                order=16,
                group_width=group_width,
                write_final_state=final_state_out is not None,
                use_64bit_indexing=use_64bit_indexing,
                block_channels=block_channels,
                num_warps=4,
            )
        continuation_count = chunk_metadata.continuation_chunk_ids.numel()
        if continuation_count:
            continuation_grid = (continuation_count, triton.cdiv(channels, block_channels))
            _segmented_modal_chunk_continue_kernel[continuation_grid](
                projection,
                diagonal,
                residues,
                poles.decay,
                chunk_metadata.chunk_starts,
                chunk_metadata.chunk_lengths,
                chunk_metadata.continuation_chunk_ids,
                output,
                chunk_states,
                channels=channels,
                order=16,
                group_width=group_width,
                use_64bit_indexing=use_64bit_indexing,
                block_channels=block_channels,
                num_warps=4,
            )
        return output

    grid = (sequence_count, triton.cdiv(channels, block_channels))
    _segmented_modal_from_projection_kernel[grid](
        projection,
        diagonal,
        residues,
        poles.decay,
        cu_seqlens,
        cu_seqlens,
        output,
        final_state_argument,
        channels=channels,
        order=16,
        group_width=group_width,
        write_final_state=final_state_out is not None,
        use_selected_sequence_ids=False,
        use_64bit_indexing=use_64bit_indexing,
        block_channels=block_channels,
        num_warps=4,
    )
    return output
