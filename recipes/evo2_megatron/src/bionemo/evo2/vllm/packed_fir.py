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

"""Boundary-safe packed causal FIR operations for Evo2."""

import argparse
import json
import statistics
from itertools import pairwise
from pathlib import Path

import torch
import triton
import triton.language as tl


def select_fir_path(*, direct_ms: float, bucketed_ms: float) -> str:
    """Select bucketing only when its measured median is at least five percent faster."""
    if direct_ms <= 0 or bucketed_ms <= 0:
        raise ValueError("FIR benchmark durations must be positive")
    return "bucketed" if bucketed_ms <= direct_ms * 0.95 else "direct"


def select_production_fir_path(
    *,
    num_requests: int,
    total_tokens: int,
    max_query_len: int,
    taps: int,
) -> str:
    """Select the measured fast path only for equal long 128-tap segments."""
    if num_requests < 1 or total_tokens < 1 or max_query_len < 1 or taps < 2:
        raise ValueError("FIR production path dimensions must be positive")
    equal_length = total_tokens == num_requests * max_query_len
    return "equal_length_conv" if taps == 128 and max_query_len >= 1_024 and equal_length else "direct"


def _expand_channels(values: torch.Tensor, *, channels: int, group_size: int, name: str) -> torch.Tensor:
    if values.shape[0] == channels:
        return values
    if values.shape[0] * group_size != channels:
        raise ValueError(f"{name} does not partition the activation channels with group_size")
    return values.repeat_interleave(group_size, dim=0)


def _validate_shapes(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    group_size: int,
) -> int:
    if x.ndim != 2:
        raise ValueError("x must have shape [total_tokens, channels]")
    if weight.ndim != 2 or weight.shape[1] < 2:
        raise ValueError("weight must have shape [filters, taps] with at least two taps")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")

    channels = x.shape[1]
    if weight.shape[0] * group_size != channels:
        raise ValueError("weight filters do not partition the activation channels with group_size")
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] not in (weight.shape[0], channels):
            raise ValueError("bias must contain one value per filter or activation channel")
        if bias.device != x.device:
            raise ValueError("bias and x must be on the same device")
    if weight.device != x.device or state_cache.device != x.device:
        raise ValueError("x, weight, and state_cache must be on the same device")

    taps = weight.shape[1]
    if state_cache.ndim != 3 or state_cache.shape[1] != channels or state_cache.shape[2] < taps - 1:
        raise ValueError("state_cache must have shape [blocks, channels, at_least_taps_minus_one]")
    if state_cache.dtype != torch.float32:
        raise ValueError("state_cache must use float32 recurrent state")
    if query_start_loc.ndim != 1:
        raise ValueError("query_start_loc must be one-dimensional")

    num_requests = query_start_loc.numel() - 1
    if num_requests < 0 or state_indices.shape != (num_requests,) or has_initial_state.shape != (num_requests,):
        raise ValueError("packed metadata lengths must agree")
    if query_start_loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_start_loc must use an integer dtype")
    if state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("state_indices must use an integer dtype")
    if has_initial_state.dtype != torch.bool:
        raise ValueError("has_initial_state must use bool dtype")
    if x.is_cuda and (
        query_start_loc.device != x.device or state_indices.device != x.device or has_initial_state.device != x.device
    ):
        raise ValueError("packed CUDA metadata and activations must be on the same device")
    return num_requests


def _validate_metadata(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    group_size: int,
) -> tuple[list[int], list[int]]:
    _validate_shapes(
        x,
        weight,
        bias,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        group_size,
    )
    starts = [int(value) for value in query_start_loc.tolist()]
    if not starts or starts[0] != 0 or starts[-1] != x.shape[0]:
        raise ValueError("query_start_loc must span every packed token")
    if any(end < start for start, end in pairwise(starts)):
        raise ValueError("query_start_loc must be monotonically nondecreasing")

    slots = [int(value) for value in state_indices.tolist()]
    if any(slot < 0 or slot >= state_cache.shape[0] for slot in slots):
        raise ValueError("state index references an invalid cache block")
    nonzero_slots = [slot for slot in slots if slot != 0]
    if len(nonzero_slots) != len(set(nonzero_slots)):
        raise ValueError("nonzero state indices must be unique within a packed call")
    return starts, slots


def packed_fir_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    group_size: int = 1,
    gated_bias: bool = False,
    flip_filter: bool = False,
) -> torch.Tensor:
    """Evaluate independent FIR segments and update their terminal states.

    This scalar implementation is a correctness oracle and diagnostic fallback. Production
    prefill and decode use the packed CUDA dispatch rather than this per-request/token loop.
    Cache block zero is a permanent null block: it is treated as zero initial state and is
    never mutated. Only the first ``taps - 1`` columns of each valid cache block belong to
    this FIR; uniform-cache padding is preserved exactly.
    """
    starts, slots = _validate_metadata(
        x,
        weight,
        bias,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        group_size,
    )
    channels = x.shape[1]
    taps = weight.shape[1]
    filters = _expand_channels(weight, channels=channels, group_size=group_size, name="weight").float()
    if flip_filter:
        filters = filters.flip(-1)
    channel_bias = (
        None if bias is None else _expand_channels(bias, channels=channels, group_size=group_size, name="bias").float()
    )
    output = torch.empty_like(x)

    for request_index, (start, end) in enumerate(pairwise(starts)):
        slot = slots[request_index]
        if slot != 0 and bool(has_initial_state[request_index]):
            history = state_cache[slot, :, : taps - 1].clone()
        else:
            history = torch.zeros((channels, taps - 1), dtype=torch.float32, device=x.device)

        for token_index in range(start, end):
            current = x[token_index].float()
            value = filters[:, -1] * current + torch.sum(history * filters[:, :-1], dim=-1)
            if channel_bias is not None:
                value = value + (channel_bias * current if gated_bias else channel_bias)
            output[token_index].copy_(value.to(x.dtype))
            history = torch.cat((history[:, 1:], current[:, None]), dim=-1)

        if slot != 0 and end > start:
            state_cache[slot, :, : taps - 1].copy_(history)

    return output


@triton.jit
def _packed_causal_fir_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    state_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    channels,
    stride_x_token,
    stride_x_channel,
    stride_weight_filter,
    stride_weight_tap,
    stride_bias,
    stride_state_block,
    stride_state_channel,
    stride_state_tap,
    stride_query_start_loc,
    stride_state_indices,
    stride_has_initial_state,
    stride_output_token,
    stride_output_channel,
    kernel_size: tl.constexpr,
    max_query_len: tl.constexpr,
    group_size: tl.constexpr,
    has_bias: tl.constexpr,
    bias_per_channel: tl.constexpr,
    gated_bias: tl.constexpr,
    flip_filter: tl.constexpr,
    block_c: tl.constexpr,
):
    request_index = tl.program_id(0)
    channel_offsets = tl.program_id(1) * block_c + tl.arange(0, block_c)
    channel_mask = channel_offsets < channels
    request_start = tl.load(query_start_loc_ptr + request_index * stride_query_start_loc).to(tl.int64)
    request_end = tl.load(query_start_loc_ptr + (request_index + 1) * stride_query_start_loc).to(tl.int64)
    request_length = request_end - request_start
    cache_slot = tl.load(state_indices_ptr + request_index * stride_state_indices).to(tl.int64)
    load_initial_state = tl.load(
        has_initial_state_ptr + request_index * stride_has_initial_state,
    ).to(tl.int1)
    filter_offsets = channel_offsets // group_size

    for token_offset in tl.range(0, max_query_len):
        token_mask = token_offset < request_length
        current = tl.load(
            x_ptr + (request_start + token_offset) * stride_x_token + channel_offsets * stride_x_channel,
            mask=channel_mask & token_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.zeros((block_c,), dtype=tl.float32)

        for tap in tl.static_range(0, kernel_size):
            relative_index = token_offset + tap - (kernel_size - 1)
            from_request = relative_index >= 0
            request_value = tl.load(
                x_ptr + (request_start + relative_index) * stride_x_token + channel_offsets * stride_x_channel,
                mask=channel_mask & token_mask & from_request,
                other=0.0,
            ).to(tl.float32)
            cached_value = tl.load(
                state_ptr
                + cache_slot * stride_state_block
                + channel_offsets * stride_state_channel
                + (relative_index + kernel_size - 1) * stride_state_tap,
                mask=(channel_mask & token_mask & (~from_request) & load_initial_state & (cache_slot != 0)),
                other=0.0,
            ).to(tl.float32)
            value = tl.where(from_request, request_value, cached_value)
            weight_tap = kernel_size - 1 - tap if flip_filter else tap
            filter_value = tl.load(
                weight_ptr + filter_offsets * stride_weight_filter + weight_tap * stride_weight_tap,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += value * filter_value

        if has_bias:
            bias_offsets = channel_offsets if bias_per_channel else filter_offsets
            bias_value = tl.load(
                bias_ptr + bias_offsets * stride_bias,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += bias_value * current if gated_bias else bias_value
        tl.store(
            output_ptr
            + (request_start + token_offset) * stride_output_token
            + channel_offsets * stride_output_channel,
            accumulator,
            mask=channel_mask & token_mask,
        )

    write_state = (cache_slot != 0) & (request_length > 0)
    for state_tap in tl.range(0, kernel_size - 1):
        source_index = request_length - (kernel_size - 1) + state_tap
        from_request = source_index >= 0
        request_value = tl.load(
            x_ptr + (request_start + source_index) * stride_x_token + channel_offsets * stride_x_channel,
            mask=channel_mask & write_state & from_request,
            other=0.0,
        ).to(tl.float32)
        cached_value = tl.load(
            state_ptr
            + cache_slot * stride_state_block
            + channel_offsets * stride_state_channel
            + (source_index + kernel_size - 1) * stride_state_tap,
            mask=channel_mask & write_state & (~from_request) & load_initial_state,
            other=0.0,
        ).to(tl.float32)
        terminal_value = tl.where(from_request, request_value, cached_value)
        tl.store(
            state_ptr
            + cache_slot * stride_state_block
            + channel_offsets * stride_state_channel
            + state_tap * stride_state_tap,
            terminal_value,
            mask=channel_mask & write_state,
        )


@triton.jit
def _packed_long_causal_fir_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    state_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    channels,
    stride_x_token,
    stride_x_channel,
    stride_weight_filter,
    stride_weight_tap,
    stride_bias,
    stride_state_block,
    stride_state_channel,
    stride_state_tap,
    stride_query_start_loc,
    stride_state_indices,
    stride_has_initial_state,
    stride_output_token,
    stride_output_channel,
    kernel_size: tl.constexpr,
    group_size: tl.constexpr,
    has_bias: tl.constexpr,
    bias_per_channel: tl.constexpr,
    gated_bias: tl.constexpr,
    flip_filter: tl.constexpr,
    block_t: tl.constexpr,
    block_c: tl.constexpr,
):
    request_index = tl.program_id(0)
    channel_offsets = tl.program_id(1) * block_c + tl.arange(0, block_c)
    token_offsets = tl.program_id(2) * block_t + tl.arange(0, block_t)
    channel_mask = channel_offsets < channels
    request_start = tl.load(query_start_loc_ptr + request_index * stride_query_start_loc).to(tl.int64)
    request_end = tl.load(query_start_loc_ptr + (request_index + 1) * stride_query_start_loc).to(tl.int64)
    request_length = request_end - request_start
    token_mask = token_offsets < request_length
    cache_slot = tl.load(state_indices_ptr + request_index * stride_state_indices).to(tl.int64)
    load_initial_state = tl.load(
        has_initial_state_ptr + request_index * stride_has_initial_state,
    ).to(tl.int1)
    filter_offsets = channel_offsets // group_size
    accumulator = tl.zeros((block_t, block_c), dtype=tl.float32)

    for tap in tl.range(0, kernel_size, num_stages=2):
        relative_indices = token_offsets + tap - (kernel_size - 1)
        from_request = relative_indices >= 0
        request_values = tl.load(
            x_ptr
            + (request_start + relative_indices[:, None]) * stride_x_token
            + channel_offsets[None, :] * stride_x_channel,
            mask=token_mask[:, None] & from_request[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        cached_values = tl.load(
            state_ptr
            + cache_slot * stride_state_block
            + channel_offsets[None, :] * stride_state_channel
            + (relative_indices[:, None] + kernel_size - 1) * stride_state_tap,
            mask=(
                token_mask[:, None]
                & (~from_request[:, None])
                & channel_mask[None, :]
                & load_initial_state
                & (cache_slot != 0)
            ),
            other=0.0,
        ).to(tl.float32)
        values = tl.where(from_request[:, None], request_values, cached_values)
        weight_tap = kernel_size - 1 - tap if flip_filter else tap
        filter_values = tl.load(
            weight_ptr + filter_offsets * stride_weight_filter + weight_tap * stride_weight_tap,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += values * filter_values[None, :]

    current = tl.load(
        x_ptr
        + (request_start + token_offsets[:, None]) * stride_x_token
        + channel_offsets[None, :] * stride_x_channel,
        mask=token_mask[:, None] & channel_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if has_bias:
        bias_offsets = channel_offsets if bias_per_channel else filter_offsets
        bias_value = tl.load(
            bias_ptr + bias_offsets * stride_bias,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += bias_value[None, :] * current if gated_bias else bias_value[None, :]
    tl.store(
        output_ptr
        + (request_start + token_offsets[:, None]) * stride_output_token
        + channel_offsets[None, :] * stride_output_channel,
        accumulator,
        mask=token_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def _packed_fir_update_state_kernel(
    x_ptr,
    state_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    channels,
    stride_x_token,
    stride_x_channel,
    stride_state_block,
    stride_state_channel,
    stride_state_tap,
    stride_query_start_loc,
    stride_state_indices,
    stride_has_initial_state,
    kernel_size: tl.constexpr,
    block_c: tl.constexpr,
):
    request_index = tl.program_id(0)
    channel_offsets = tl.program_id(1) * block_c + tl.arange(0, block_c)
    channel_mask = channel_offsets < channels
    request_start = tl.load(query_start_loc_ptr + request_index * stride_query_start_loc).to(tl.int64)
    request_end = tl.load(query_start_loc_ptr + (request_index + 1) * stride_query_start_loc).to(tl.int64)
    request_length = request_end - request_start
    cache_slot = tl.load(state_indices_ptr + request_index * stride_state_indices).to(tl.int64)
    load_initial_state = tl.load(
        has_initial_state_ptr + request_index * stride_has_initial_state,
    ).to(tl.int1)
    write_state = (cache_slot != 0) & (request_length > 0)

    for state_tap in tl.range(0, kernel_size - 1, num_stages=2):
        source_index = request_length - (kernel_size - 1) + state_tap
        from_request = source_index >= 0
        request_value = tl.load(
            x_ptr + (request_start + source_index) * stride_x_token + channel_offsets * stride_x_channel,
            mask=channel_mask & write_state & from_request,
            other=0.0,
        ).to(tl.float32)
        cached_value = tl.load(
            state_ptr
            + cache_slot * stride_state_block
            + channel_offsets * stride_state_channel
            + (source_index + kernel_size - 1) * stride_state_tap,
            mask=channel_mask & write_state & (~from_request) & load_initial_state,
            other=0.0,
        ).to(tl.float32)
        terminal_value = tl.where(from_request, request_value, cached_value)
        tl.store(
            state_ptr
            + cache_slot * stride_state_block
            + channel_offsets * stride_state_channel
            + state_tap * stride_state_tap,
            terminal_value,
            mask=channel_mask & write_state,
        )


def _packed_equal_length_causal_fir(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    num_requests: int,
    query_len: int,
    group_size: int,
    gated_bias: bool,
    flip_filter: bool,
) -> torch.Tensor:
    """Run the measured grouped-convolution path with recurrent history."""
    channels = x.shape[1]
    taps = weight.shape[1]
    slots = state_indices.to(torch.int64)
    gathered_state = state_cache.index_select(0, slots)[:, :, : taps - 1]
    valid_history = has_initial_state[:, None, None] & slots.ne(0)[:, None, None]
    history = torch.where(valid_history, gathered_state, torch.zeros_like(gathered_state))
    sequence = x.reshape(num_requests, query_len, channels).transpose(1, 2).float()
    padded_sequence = torch.cat((history, sequence), dim=-1)

    expanded_weight = _expand_channels(
        weight,
        channels=channels,
        group_size=group_size,
        name="weight",
    ).float()
    if flip_filter:
        expanded_weight = expanded_weight.flip(-1)
    convolution = torch.conv1d(
        padded_sequence,
        expanded_weight.contiguous().unsqueeze(1),
        groups=channels,
    )
    if bias is not None:
        expanded_bias = _expand_channels(
            bias,
            channels=channels,
            group_size=group_size,
            name="bias",
        ).float()[None, :, None]
        convolution = convolution + (expanded_bias * sequence if gated_bias else expanded_bias)
    output = convolution.transpose(1, 2).reshape(-1, channels).to(x.dtype)

    block_channels = 16
    state_grid = (num_requests, triton.cdiv(channels, block_channels))
    _packed_fir_update_state_kernel[state_grid](
        x,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        channels,
        x.stride(0),
        x.stride(1),
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(2),
        query_start_loc.stride(0),
        state_indices.stride(0),
        has_initial_state.stride(0),
        kernel_size=taps,
        block_c=block_channels,
        num_warps=4,
    )
    return output


def packed_causal_fir(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    group_size: int = 1,
    gated_bias: bool = False,
    flip_filter: bool = False,
    max_query_len: int,
) -> torch.Tensor:
    """Run segmented FIR kernels over packed prefill or decode requests of any positive length."""
    num_requests = _validate_shapes(
        x,
        weight,
        bias,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        group_size,
    )
    if isinstance(max_query_len, bool) or not isinstance(max_query_len, int) or max_query_len < 1:
        raise ValueError("max_query_len must be a positive integer")
    if not x.is_cuda:
        return packed_fir_reference(
            x,
            weight,
            bias,
            state_cache,
            query_start_loc,
            state_indices,
            has_initial_state,
            group_size=group_size,
            gated_bias=gated_bias,
            flip_filter=flip_filter,
        )

    taps = weight.shape[1]
    production_path = select_production_fir_path(
        num_requests=num_requests,
        total_tokens=x.shape[0],
        max_query_len=max_query_len,
        taps=taps,
    )
    if production_path == "equal_length_conv":
        return _packed_equal_length_causal_fir(
            x,
            weight,
            bias,
            state_cache,
            query_start_loc,
            state_indices,
            has_initial_state,
            num_requests=num_requests,
            query_len=max_query_len,
            group_size=group_size,
            gated_bias=gated_bias,
            flip_filter=flip_filter,
        )

    output = torch.empty_like(x)
    bias_tensor = x if bias is None else bias
    if max_query_len <= 32:
        block_channels = 32 if taps >= 128 else 128
        grid = (num_requests, triton.cdiv(x.shape[1], block_channels))
        _packed_causal_fir_kernel[grid](
            x,
            weight,
            bias_tensor,
            state_cache,
            query_start_loc,
            state_indices,
            has_initial_state,
            output,
            x.shape[1],
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            bias_tensor.stride(0),
            state_cache.stride(0),
            state_cache.stride(1),
            state_cache.stride(2),
            query_start_loc.stride(0),
            state_indices.stride(0),
            has_initial_state.stride(0),
            output.stride(0),
            output.stride(1),
            kernel_size=taps,
            max_query_len=max_query_len,
            group_size=group_size,
            has_bias=bias is not None,
            bias_per_channel=bias is not None and bias.shape[0] == x.shape[1],
            gated_bias=gated_bias,
            flip_filter=flip_filter,
            block_c=block_channels,
            num_warps=4,
        )
        return output

    block_tokens = 8 if taps >= 128 else 16
    block_channels = 16 if taps >= 128 else 32
    output_grid = (
        num_requests,
        triton.cdiv(x.shape[1], block_channels),
        triton.cdiv(max_query_len, block_tokens),
    )
    _packed_long_causal_fir_kernel[output_grid](
        x,
        weight,
        bias_tensor,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        output,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        bias_tensor.stride(0),
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(2),
        query_start_loc.stride(0),
        state_indices.stride(0),
        has_initial_state.stride(0),
        output.stride(0),
        output.stride(1),
        kernel_size=taps,
        group_size=group_size,
        has_bias=bias is not None,
        bias_per_channel=bias is not None and bias.shape[0] == x.shape[1],
        gated_bias=gated_bias,
        flip_filter=flip_filter,
        block_t=block_tokens,
        block_c=block_channels,
        num_warps=4,
    )
    state_grid = (num_requests, triton.cdiv(x.shape[1], block_channels))
    _packed_fir_update_state_kernel[state_grid](
        x,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        x.shape[1],
        x.stride(0),
        x.stride(1),
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(2),
        query_start_loc.stride(0),
        state_indices.stride(0),
        has_initial_state.stride(0),
        kernel_size=taps,
        block_c=block_channels,
        num_warps=4,
    )
    return output


def _parse_prompt_lengths(value: str) -> tuple[int, ...]:
    try:
        if ":" in value:
            lower_text, upper_text = value.split(":", maxsplit=1)
            lower, upper = int(lower_text), int(upper_text)
            lengths = tuple(range(lower, upper + 1))
        else:
            lengths = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "prompt lengths must be an inclusive range or comma-separated integers"
        ) from error
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("FIR prompt lengths must be positive")
    return lengths


def _exact_length_buckets(
    lengths: list[int],
    device: torch.device,
) -> tuple[tuple[int, torch.Tensor, torch.Tensor], ...]:
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    requests_by_length: dict[int, list[int]] = {}
    for request_index, length in enumerate(lengths):
        requests_by_length.setdefault(length, []).append(request_index)

    buckets = []
    for length, request_indices in sorted(requests_by_length.items()):
        token_indices = [
            token_index
            for request_index in request_indices
            for token_index in range(starts[request_index], starts[request_index + 1])
        ]
        buckets.append(
            (
                length,
                torch.tensor(request_indices, dtype=torch.int64, device=device),
                torch.tensor(token_indices, dtype=torch.int64, device=device),
            )
        )
    return tuple(buckets)


def _bucketed_fir_benchmark_candidate(
    x: torch.Tensor,
    convolution_weight: torch.Tensor,
    channel_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_slots: torch.Tensor,
    buckets: tuple[tuple[int, torch.Tensor, torch.Tensor], ...],
    *,
    gated_bias: bool,
) -> torch.Tensor:
    """Benchmark-only exact-length candidate; production uses the selected direct kernel."""
    channels = x.shape[1]
    taps = convolution_weight.shape[-1]
    output = torch.empty_like(x)
    state_view = state_cache[:, :, : taps - 1]
    for length, request_indices, token_indices in buckets:
        request_count = request_indices.numel()
        slots = state_slots.index_select(0, request_indices)
        initial_state = torch.zeros(
            (request_count, channels, taps - 1),
            dtype=torch.float32,
            device=x.device,
        )
        sequence = x.index_select(0, token_indices).reshape(request_count, length, channels).transpose(1, 2).float()
        padded_sequence = torch.cat((initial_state, sequence), dim=-1)
        bucket_output = torch.conv1d(padded_sequence, convolution_weight, groups=channels)
        expanded_bias = channel_bias[None, :, None]
        bucket_output = bucket_output + (expanded_bias * sequence if gated_bias else expanded_bias)
        output.index_copy_(0, token_indices, bucket_output.transpose(1, 2).reshape(-1, channels).to(x.dtype))
        state_view.index_copy_(0, slots, padded_sequence[:, :, -(taps - 1) :])
    return output


def _benchmark_fir(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the packed FIR benchmark requires CUDA")
    if args.channels % args.group_size:
        raise ValueError("channels must be divisible by group size")

    device = torch.device("cuda")
    lengths = [args.prompt_lengths[index % len(args.prompt_lengths)] for index in range(args.batch_size)]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device=device)
    state_indices = torch.arange(1, args.batch_size + 1, dtype=torch.int32, device=device)
    state_slots = state_indices.to(torch.int64)
    has_initial_state = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
    buckets = _exact_length_buckets(lengths, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    results = []

    for taps in args.taps:
        x = torch.randn((starts[-1], args.channels), dtype=torch.bfloat16, device=device, generator=generator)
        weight = torch.randn(
            (args.channels // args.group_size, taps),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        bias = torch.randn(
            (args.channels // args.group_size,),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        state_cache = torch.zeros(
            (args.batch_size + 1, args.channels, max(127, taps - 1)),
            dtype=torch.float32,
            device=device,
        )
        gated_bias = taps == 128
        flip_filter = taps == 128
        expanded_weight = weight.repeat_interleave(args.group_size, dim=0).float()
        if flip_filter:
            expanded_weight = expanded_weight.flip(-1)
        convolution_weight = expanded_weight.contiguous().unsqueeze(1)
        channel_bias = bias.repeat_interleave(args.group_size, dim=0).float()

        def direct() -> torch.Tensor:
            return packed_causal_fir(
                x,
                weight,
                bias,
                state_cache,
                query_start_loc,
                state_indices,
                has_initial_state,
                group_size=args.group_size,
                gated_bias=gated_bias,
                flip_filter=flip_filter,
                max_query_len=max(args.prompt_lengths),
            )

        def bucketed() -> torch.Tensor:
            return _bucketed_fir_benchmark_candidate(
                x,
                convolution_weight,
                channel_bias,
                state_cache,
                state_slots,
                buckets,
                gated_bias=gated_bias,
            )

        direct_trials = []
        bucketed_trials = []
        for trial in range(args.trials):
            candidates = ((direct, direct_trials), (bucketed, bucketed_trials))
            if trial % 2:
                candidates = tuple(reversed(candidates))
            for candidate, timings in candidates:
                timings.append(float(triton.testing.do_bench(candidate, warmup=args.warmup, rep=args.repetitions)))

        direct_ms = statistics.median(direct_trials)
        bucketed_ms = statistics.median(bucketed_trials)
        results.append(
            {
                "taps": taps,
                "production_path": select_production_fir_path(
                    num_requests=args.batch_size,
                    total_tokens=starts[-1],
                    max_query_len=max(lengths),
                    taps=taps,
                ),
                "production_trials_ms": direct_trials,
                "production_median_ms": direct_ms,
                "benchmark_candidate_trials_ms": bucketed_trials,
                "benchmark_candidate_median_ms": bucketed_ms,
                "direct_trials_ms": direct_trials,
                "bucketed_trials_ms": bucketed_trials,
                "direct_median_ms": direct_ms,
                "bucketed_median_ms": bucketed_ms,
                "selected_path": select_fir_path(direct_ms=direct_ms, bucketed_ms=bucketed_ms),
            }
        )

    return {
        "schema_version": 2,
        "dispatch_under_test": "packed_causal_fir production dispatch",
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "request_prompt_lengths": lengths,
        "channels": args.channels,
        "group_size": args.group_size,
        "dtype": "bfloat16",
        "state_dtype": "float32",
        "warmup_ms": args.warmup,
        "repetitions_ms": args.repetitions,
        "trials": args.trials,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the packed FIR crossover benchmark CLI."""
    parser = argparse.ArgumentParser(description="Benchmark Evo2 packed FIR dispatch candidates")
    parser.add_argument("--benchmark", action="store_true", help="run the CUDA benchmark")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--prompt-lengths", type=_parse_prompt_lengths, default=_parse_prompt_lengths("4:12"))
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument(
        "--taps", type=lambda value: tuple(int(item) for item in value.split(",")), default=(3, 7, 128)
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=25, help="Triton benchmark warmup in milliseconds")
    parser.add_argument("--repetitions", type=int, default=100, help="Triton benchmark measurement in milliseconds")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.benchmark:
        parser.error("--benchmark is required")
    if args.batch_size < 1 or args.channels < 1 or args.group_size < 1 or args.trials < 1:
        parser.error("batch size, channels, group size, and trials must be positive")
    if not args.taps or any(taps < 2 for taps in args.taps):
        parser.error("every FIR tap count must be at least two")

    report = _benchmark_fir(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
