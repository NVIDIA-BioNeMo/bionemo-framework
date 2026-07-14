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

from itertools import pairwise

import torch


def _expand_channels(values: torch.Tensor, *, channels: int, group_size: int, name: str) -> torch.Tensor:
    if values.shape[0] == channels:
        return values
    if values.shape[0] * group_size != channels:
        raise ValueError(f"{name} does not partition the activation channels with group_size")
    return values.repeat_interleave(group_size, dim=0)


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
