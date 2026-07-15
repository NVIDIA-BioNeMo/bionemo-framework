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

"""Boundary-safe packed modal recurrence operations for Evo2 HCL layers."""

import argparse
import json
import statistics
from itertools import pairwise
from pathlib import Path

import torch
import triton
import triton.language as tl


def _validate_shapes(
    recurrent_input: torch.Tensor,
    gate: torch.Tensor,
    decay: torch.Tensor,
    residues: torch.Tensor,
    diagonal: torch.Tensor,
    state_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    state_size: int,
) -> int:
    if recurrent_input.ndim != 2:
        raise ValueError("recurrent_input must have shape [total_tokens, channels]")
    if gate.shape != recurrent_input.shape:
        raise ValueError("gate must have the same shape as recurrent_input")
    if isinstance(state_size, bool) or not isinstance(state_size, int) or state_size <= 0:
        raise ValueError("state_size must be a positive integer")

    channels = recurrent_input.shape[1]
    if decay.shape != (channels, state_size):
        raise ValueError("decay must have shape [channels, state_size]")
    if residues.shape != (channels, state_size):
        raise ValueError("residues must have shape [channels, state_size]")
    if diagonal.shape != (channels,):
        raise ValueError("diagonal must have shape [channels]")
    if not all(torch.is_floating_point(tensor) for tensor in (recurrent_input, gate, decay, residues, diagonal)):
        raise ValueError("modal activations and coefficients must use floating-point dtypes")

    data_tensors = (gate, decay, residues, diagonal, state_cache)
    if any(tensor.device != recurrent_input.device for tensor in data_tensors):
        raise ValueError("modal activations, coefficients, and state cache must be on the same device")
    if state_cache.ndim != 3 or state_cache.shape[1] != channels or state_cache.shape[2] < state_size:
        raise ValueError("state_cache must have shape [blocks, channels, at_least_state_size]")
    if state_cache.dtype != torch.float32:
        raise ValueError("state_cache must use float32 recurrent state")

    if query_start_loc.ndim != 1 or query_start_loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_start_loc must be a one-dimensional integer tensor")
    num_requests = query_start_loc.numel() - 1
    if num_requests < 0 or state_indices.shape != (num_requests,) or has_initial_state.shape != (num_requests,):
        raise ValueError("packed metadata lengths must agree")
    if state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("state_indices must use an integer dtype")
    if has_initial_state.dtype != torch.bool:
        raise ValueError("has_initial_state must use bool dtype")
    if any(tensor.device != recurrent_input.device for tensor in (query_start_loc, state_indices, has_initial_state)):
        raise ValueError("packed metadata and activations must be on the same device")

    return num_requests


def _validate_metadata(
    recurrent_input: torch.Tensor,
    gate: torch.Tensor,
    decay: torch.Tensor,
    residues: torch.Tensor,
    diagonal: torch.Tensor,
    state_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    state_size: int,
) -> tuple[list[int], list[int]]:
    _validate_shapes(
        recurrent_input,
        gate,
        decay,
        residues,
        diagonal,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        state_size,
    )

    starts = [int(value) for value in query_start_loc.tolist()]
    if not starts or starts[0] != 0 or starts[-1] != recurrent_input.shape[0]:
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


def packed_iir_reference(
    recurrent_input: torch.Tensor,
    gate: torch.Tensor,
    decay: torch.Tensor,
    residues: torch.Tensor,
    diagonal: torch.Tensor,
    state_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    state_size: int = 16,
) -> torch.Tensor:
    """Evaluate independent HCL recurrences and update their terminal states.

    ``recurrent_input`` is Evo2's ``x2 * v`` stream and ``gate`` is the ``x1`` stream after
    accounting for the historical argument swap in the Megatron implementation. The recurrence,
    coefficient reduction, and persistent cache all use fp32. Cache block zero is treated as zero
    initial state and is never mutated; padding beyond ``state_size`` is preserved exactly.

    This scalar implementation is a correctness oracle and diagnostic fallback. Production prefill
    and decode use the packed CUDA dispatch rather than this per-request/token loop.
    """
    starts, slots = _validate_metadata(
        recurrent_input,
        gate,
        decay,
        residues,
        diagonal,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        state_size,
    )
    channels = recurrent_input.shape[1]
    decay = decay.float()
    residues = residues.float()
    diagonal = diagonal.float()
    output = torch.empty_like(recurrent_input)

    for request_index, (start, end) in enumerate(pairwise(starts)):
        slot = slots[request_index]
        if slot != 0 and bool(has_initial_state[request_index]):
            state = state_cache[slot, :, :state_size].clone()
        else:
            state = torch.zeros((channels, state_size), dtype=torch.float32, device=recurrent_input.device)

        for token_index in range(start, end):
            drive = recurrent_input[token_index].float()
            state.mul_(decay).add_(drive[:, None])
            mixed = torch.sum(residues * state, dim=-1) + diagonal * drive
            output[token_index].copy_((gate[token_index].float() * mixed).to(output.dtype))

        if slot != 0 and end > start:
            state_cache[slot, :, :state_size].copy_(state)

    return output


@triton.jit
def _packed_modal_iir_kernel(
    recurrent_input_ptr,
    gate_ptr,
    decay_ptr,
    residues_ptr,
    diagonal_ptr,
    state_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    query_offset,
    channels,
    stride_input_token,
    stride_input_channel,
    stride_gate_token,
    stride_gate_channel,
    stride_decay_channel,
    stride_decay_state,
    stride_residues_channel,
    stride_residues_state,
    stride_diagonal,
    stride_cache_block,
    stride_cache_channel,
    stride_cache_state,
    stride_query_start_loc,
    stride_state_indices,
    stride_has_initial_state,
    stride_output_token,
    stride_output_channel,
    state_size: tl.constexpr,
    block_state: tl.constexpr,
    max_query_len: tl.constexpr,
    block_c: tl.constexpr,
):
    request_index = tl.program_id(0)
    channel_offsets = tl.program_id(1) * block_c + tl.arange(0, block_c)
    state_offsets = tl.arange(0, block_state)
    channel_mask = channel_offsets < channels
    modal_mask = channel_mask[:, None] & (state_offsets[None, :] < state_size)

    request_start = tl.load(query_start_loc_ptr + request_index * stride_query_start_loc).to(tl.int64)
    request_end = tl.load(query_start_loc_ptr + (request_index + 1) * stride_query_start_loc).to(tl.int64)
    request_length = request_end - request_start
    cache_slot = tl.load(state_indices_ptr + request_index * stride_state_indices).to(tl.int64)
    has_initial_state = tl.load(
        has_initial_state_ptr + request_index * stride_has_initial_state,
    ).to(tl.int1)
    load_initial_state = has_initial_state | (query_offset > 0)

    modal_offsets = channel_offsets[:, None] * stride_decay_channel + state_offsets[None, :] * stride_decay_state
    decay = tl.load(decay_ptr + modal_offsets, mask=modal_mask, other=0.0).to(tl.float32)
    residue_offsets = (
        channel_offsets[:, None] * stride_residues_channel + state_offsets[None, :] * stride_residues_state
    )
    residues = tl.load(residues_ptr + residue_offsets, mask=modal_mask, other=0.0).to(tl.float32)
    diagonal = tl.load(diagonal_ptr + channel_offsets * stride_diagonal, mask=channel_mask, other=0.0).to(tl.float32)
    cache_offsets = (
        cache_slot * stride_cache_block
        + channel_offsets[:, None] * stride_cache_channel
        + state_offsets[None, :] * stride_cache_state
    )
    state = tl.load(
        state_ptr + cache_offsets,
        mask=modal_mask & load_initial_state & (cache_slot != 0),
        other=0.0,
    ).to(tl.float32)

    for token_offset in tl.static_range(0, max_query_len):
        token_is_real = query_offset + token_offset < request_length
        token_index = request_start + query_offset + token_offset
        drive = tl.load(
            recurrent_input_ptr + token_index * stride_input_token + channel_offsets * stride_input_channel,
            mask=channel_mask & token_is_real,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            gate_ptr + token_index * stride_gate_token + channel_offsets * stride_gate_channel,
            mask=channel_mask & token_is_real,
            other=0.0,
        ).to(tl.float32)
        advanced_state = state * decay + drive[:, None]
        state = tl.where(token_is_real, advanced_state, state)
        mixed = tl.sum(residues * state, axis=1) + diagonal * drive
        tl.store(
            output_ptr + token_index * stride_output_token + channel_offsets * stride_output_channel,
            gate * mixed,
            mask=channel_mask & token_is_real,
        )

    tl.store(
        state_ptr + cache_offsets,
        state,
        mask=modal_mask & (cache_slot != 0) & (request_length > query_offset),
    )


@triton.jit
def _packed_modal_iir_kernel_long(
    recurrent_input_ptr,
    gate_ptr,
    decay_ptr,
    residues_ptr,
    diagonal_ptr,
    state_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    max_query_len,
    channels,
    stride_input_token,
    stride_input_channel,
    stride_gate_token,
    stride_gate_channel,
    stride_decay_channel,
    stride_decay_state,
    stride_residues_channel,
    stride_residues_state,
    stride_diagonal,
    stride_cache_block,
    stride_cache_channel,
    stride_cache_state,
    stride_query_start_loc,
    stride_state_indices,
    stride_has_initial_state,
    stride_output_token,
    stride_output_channel,
    state_size: tl.constexpr,
    block_state: tl.constexpr,
    block_c: tl.constexpr,
):
    request_index = tl.program_id(0)
    channel_offsets = tl.program_id(1) * block_c + tl.arange(0, block_c)
    state_offsets = tl.arange(0, block_state)
    channel_mask = channel_offsets < channels
    modal_mask = channel_mask[:, None] & (state_offsets[None, :] < state_size)

    request_start = tl.load(query_start_loc_ptr + request_index * stride_query_start_loc).to(tl.int64)
    request_end = tl.load(query_start_loc_ptr + (request_index + 1) * stride_query_start_loc).to(tl.int64)
    request_length = request_end - request_start
    cache_slot = tl.load(state_indices_ptr + request_index * stride_state_indices).to(tl.int64)
    load_initial_state = tl.load(
        has_initial_state_ptr + request_index * stride_has_initial_state,
    ).to(tl.int1)

    modal_offsets = channel_offsets[:, None] * stride_decay_channel + state_offsets[None, :] * stride_decay_state
    decay = tl.load(decay_ptr + modal_offsets, mask=modal_mask, other=0.0).to(tl.float32)
    residue_offsets = (
        channel_offsets[:, None] * stride_residues_channel + state_offsets[None, :] * stride_residues_state
    )
    residues = tl.load(residues_ptr + residue_offsets, mask=modal_mask, other=0.0).to(tl.float32)
    diagonal = tl.load(diagonal_ptr + channel_offsets * stride_diagonal, mask=channel_mask, other=0.0).to(tl.float32)
    cache_offsets = (
        cache_slot * stride_cache_block
        + channel_offsets[:, None] * stride_cache_channel
        + state_offsets[None, :] * stride_cache_state
    )
    state = tl.load(
        state_ptr + cache_offsets,
        mask=modal_mask & load_initial_state & (cache_slot != 0),
        other=0.0,
    ).to(tl.float32)

    for token_offset in tl.range(0, max_query_len, num_stages=1):
        token_is_real = token_offset < request_length
        token_index = request_start + token_offset
        drive = tl.load(
            recurrent_input_ptr + token_index * stride_input_token + channel_offsets * stride_input_channel,
            mask=channel_mask & token_is_real,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            gate_ptr + token_index * stride_gate_token + channel_offsets * stride_gate_channel,
            mask=channel_mask & token_is_real,
            other=0.0,
        ).to(tl.float32)
        advanced_state = state * decay + drive[:, None]
        state = tl.where(token_is_real, advanced_state, state)
        mixed = tl.sum(residues * state, axis=1) + diagonal * drive
        tl.store(
            output_ptr + token_index * stride_output_token + channel_offsets * stride_output_channel,
            gate * mixed,
            mask=channel_mask & token_is_real,
        )

    tl.store(
        state_ptr + cache_offsets,
        state,
        mask=modal_mask & (cache_slot != 0) & (request_length > 0),
    )


def packed_modal_iir(
    recurrent_input: torch.Tensor,
    gate: torch.Tensor,
    decay: torch.Tensor,
    residues: torch.Tensor,
    diagonal: torch.Tensor,
    state_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    state_size: int = 16,
    max_query_len: int,
) -> torch.Tensor:
    """Run one segmented modal-recurrence kernel over packed prefill or decode requests."""
    num_requests = _validate_shapes(
        recurrent_input,
        gate,
        decay,
        residues,
        diagonal,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        state_size,
    )
    if isinstance(max_query_len, bool) or not isinstance(max_query_len, int) or max_query_len < 1:
        raise ValueError("max_query_len must be a positive integer")
    if state_size > 32:
        raise ValueError("the direct HCL kernel supports state_size up to 32")
    if not recurrent_input.is_cuda:
        return packed_iir_reference(
            recurrent_input,
            gate,
            decay,
            residues,
            diagonal,
            state_cache,
            query_start_loc,
            state_indices,
            has_initial_state,
            state_size=state_size,
        )

    output = torch.empty_like(recurrent_input)
    block_channels = 32
    block_state = 1 << (state_size - 1).bit_length()
    grid = (num_requests, triton.cdiv(recurrent_input.shape[1], block_channels))
    kernel_args = (
        recurrent_input,
        gate,
        decay,
        residues,
        diagonal,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        output,
    )
    kernel_strides = (
        recurrent_input.shape[1],
        recurrent_input.stride(0),
        recurrent_input.stride(1),
        gate.stride(0),
        gate.stride(1),
        decay.stride(0),
        decay.stride(1),
        residues.stride(0),
        residues.stride(1),
        diagonal.stride(0),
        state_cache.stride(0),
        state_cache.stride(1),
        state_cache.stride(2),
        query_start_loc.stride(0),
        state_indices.stride(0),
        has_initial_state.stride(0),
        output.stride(0),
        output.stride(1),
    )
    kernel_meta = {
        "state_size": state_size,
        "block_state": block_state,
        "block_c": block_channels,
        "num_warps": 4,
    }
    if max_query_len <= 32:
        _packed_modal_iir_kernel[grid](
            *kernel_args,
            0,
            *kernel_strides,
            max_query_len=max_query_len,
            **kernel_meta,
        )
    else:
        _packed_modal_iir_kernel_long[grid](
            *kernel_args,
            max_query_len,
            *kernel_strides,
            **kernel_meta,
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
        raise argparse.ArgumentTypeError("prompt lengths must be positive")
    return lengths


def _benchmark_iir(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the packed HCL benchmark requires CUDA")

    device = torch.device("cuda")
    lengths = [args.prompt_lengths[index % len(args.prompt_lengths)] for index in range(args.batch_size)]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device=device)
    decode_start_loc = torch.arange(args.batch_size + 1, dtype=torch.int32, device=device)
    state_indices = torch.arange(1, args.batch_size + 1, dtype=torch.int32, device=device)
    prefill_has_initial_state = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
    decode_has_initial_state = torch.ones(args.batch_size, dtype=torch.bool, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    recurrent_input = torch.randn(
        (starts[-1], args.channels),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gate = torch.randn(recurrent_input.shape, dtype=torch.bfloat16, device=device, generator=generator)
    decode_input = torch.randn(
        (args.batch_size, args.channels),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    decode_gate = torch.randn(decode_input.shape, dtype=torch.bfloat16, device=device, generator=generator)
    decay = torch.exp(-(torch.rand((args.channels, args.state_size), device=device, generator=generator) * 0.2 + 0.01))
    residues = torch.randn((args.channels, args.state_size), device=device, generator=generator) * 0.05
    diagonal = torch.randn((args.channels,), device=device, generator=generator) * 0.05
    state_cache = torch.zeros(
        (args.batch_size + 1, args.channels, max(127, args.state_size)),
        dtype=torch.float32,
        device=device,
    )

    def prefill() -> torch.Tensor:
        return packed_modal_iir(
            recurrent_input,
            gate,
            decay,
            residues,
            diagonal,
            state_cache,
            query_start_loc,
            state_indices,
            prefill_has_initial_state,
            state_size=args.state_size,
            max_query_len=max(lengths),
        )

    def decode() -> torch.Tensor:
        return packed_modal_iir(
            decode_input,
            decode_gate,
            decay,
            residues,
            diagonal,
            state_cache,
            decode_start_loc,
            state_indices,
            decode_has_initial_state,
            state_size=args.state_size,
            max_query_len=1,
        )

    prefill_trials = []
    decode_trials = []
    for trial in range(args.trials):
        candidates = ((prefill, prefill_trials), (decode, decode_trials))
        if trial % 2:
            candidates = tuple(reversed(candidates))
        for candidate, timings in candidates:
            timings.append(float(triton.testing.do_bench(candidate, warmup=args.warmup, rep=args.repetitions)))

    return {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "request_prompt_lengths": lengths,
        "channels": args.channels,
        "state_size": args.state_size,
        "dtype": "bfloat16",
        "coefficient_dtype": "float32",
        "state_dtype": "float32",
        "warmup_ms": args.warmup,
        "repetitions_ms": args.repetitions,
        "trials": args.trials,
        "prefill_trials_ms": prefill_trials,
        "prefill_median_ms": statistics.median(prefill_trials),
        "decode_trials_ms": decode_trials,
        "decode_median_ms": statistics.median(decode_trials),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the packed HCL benchmark CLI."""
    parser = argparse.ArgumentParser(description="Benchmark Evo2 packed HCL prefill and decode")
    parser.add_argument("--benchmark", action="store_true", help="run the CUDA benchmark")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--prompt-lengths", type=_parse_prompt_lengths, default=_parse_prompt_lengths("4:12"))
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=25, help="Triton benchmark warmup in milliseconds")
    parser.add_argument("--repetitions", type=int, default=100, help="Triton benchmark measurement in milliseconds")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.benchmark:
        parser.error("--benchmark is required")
    if args.batch_size < 1 or args.channels < 1 or args.state_size < 1 or args.trials < 1:
        parser.error("batch size, channels, state size, and trials must be positive")
    if args.state_size > 32:
        parser.error("the packed HCL kernel supports state size up to 32")

    report = _benchmark_iir(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
