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

from itertools import pairwise

import torch


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
