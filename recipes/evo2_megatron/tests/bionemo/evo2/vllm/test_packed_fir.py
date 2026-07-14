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

import pytest
import torch

from bionemo.evo2.models.megatron.hyena.engine import step_fir
from bionemo.evo2.vllm.packed_fir import packed_fir_reference


PROMPT_LENGTHS = list(range(4, 13)) * 10 + [4, 5, 6, 7, 8, 9]


def _query_start_loc(lengths):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)


def _expanded(values, channels, group_size):
    if values is None:
        return None
    if values.shape[0] == channels:
        return values
    return values.repeat_interleave(group_size, dim=0)


def _scalar_oracle(
    x,
    weight,
    bias,
    initial_cache,
    query_start_loc,
    state_indices,
    has_initial_state,
    *,
    group_size,
    gated_bias,
    flip_filter,
):
    expected_output = torch.empty_like(x)
    expected_cache = initial_cache.clone()
    taps = weight.shape[-1]
    channels = x.shape[-1]
    expanded_weight = _expanded(weight, channels, group_size)
    expanded_bias = _expanded(bias, channels, group_size)

    for request_index in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[request_index])
        end = int(query_start_loc[request_index + 1])
        slot = int(state_indices[request_index])
        if slot != 0 and bool(has_initial_state[request_index]):
            state = initial_cache[slot, :, : taps - 1].clone().unsqueeze(0)
        else:
            state = torch.zeros((1, channels, taps - 1), dtype=torch.float32)

        for token_index in range(start, end):
            value, state = step_fir(
                u=x[token_index].unsqueeze(0),
                fir_state=state,
                weight=expanded_weight,
                bias=expanded_bias,
                gated_bias=gated_bias,
                flip_filter=flip_filter,
            )
            expected_output[token_index].copy_(value[0])

        if slot != 0 and end > start:
            expected_cache[slot, :, : taps - 1].copy_(state[0])

    return expected_output, expected_cache


@pytest.mark.parametrize(
    ("taps", "group_size", "gated_bias", "flip_filter", "populated_state"),
    [
        (3, 1, False, False, False),
        (3, 16, False, False, True),
        (7, 1, False, False, True),
        (7, 16, False, False, False),
        (128, 1, True, True, True),
        (128, 16, True, True, False),
    ],
)
def test_packed_fir_matches_independent_scalar_oracle(
    taps,
    group_size,
    gated_bias,
    flip_filter,
    populated_state,
):
    generator = torch.Generator().manual_seed(1701 + taps + group_size)
    channels = 32
    num_filters = channels // group_size
    query_start_loc = _query_start_loc(PROMPT_LENGTHS)
    total_tokens = int(query_start_loc[-1])
    x = torch.randn((total_tokens, channels), generator=generator)
    weight = torch.randn((num_filters, taps), generator=generator) * 0.05
    bias = torch.randn((num_filters,), generator=generator) * 0.05

    state_indices = torch.arange(len(PROMPT_LENGTHS), 0, -1, dtype=torch.int64)
    state_indices[::17] = 0
    has_initial_state = torch.zeros(len(PROMPT_LENGTHS), dtype=torch.bool)
    if populated_state:
        has_initial_state[1::3] = True

    cache_width = max(127, taps - 1) + 5
    state_cache = torch.full((len(PROMPT_LENGTHS) + 8, channels, cache_width), 13.25, dtype=torch.float32)
    original_cache = state_cache.clone()
    expected_output, expected_cache = _scalar_oracle(
        x,
        weight,
        bias,
        original_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        group_size=group_size,
        gated_bias=gated_bias,
        flip_filter=flip_filter,
    )

    output = packed_fir_reference(
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

    torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(state_cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(state_cache[0], original_cache[0], rtol=0, atol=0)


def test_empty_segments_and_null_slots_do_not_mutate_cache():
    x = torch.tensor([[2.0, -3.0]])
    weight = torch.tensor([[0.25, 0.5, 1.0], [-0.5, 0.25, 0.75]])
    state_cache = torch.full((3, 2, 7), 9.0)
    original_cache = state_cache.clone()

    output = packed_fir_reference(
        x,
        weight,
        None,
        state_cache,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 0, 2]),
        torch.tensor([False, True, True]),
        group_size=1,
        gated_bias=False,
        flip_filter=False,
    )

    torch.testing.assert_close(output, torch.tensor([[2.0, -2.25]]))
    torch.testing.assert_close(state_cache, original_cache, rtol=0, atol=0)


@pytest.mark.parametrize("target_position", [0, 1])
def test_request_matches_when_run_alone_or_inside_a_batch(target_position):
    generator = torch.Generator().manual_seed(90210)
    target = torch.randn((8, 8), generator=generator)
    other = torch.randn((5, 8), generator=generator) * 1000
    weight = torch.randn((4, 7), generator=generator) * 0.1
    bias = torch.randn((4,), generator=generator) * 0.1
    initial_cache = torch.randn((4, 8, 11), generator=generator)

    single_cache = initial_cache.clone()
    single_output = packed_fir_reference(
        target,
        weight,
        bias,
        single_cache,
        torch.tensor([0, len(target)]),
        torch.tensor([1]),
        torch.tensor([True]),
        group_size=2,
        gated_bias=False,
        flip_filter=False,
    )

    requests = [target, other]
    if target_position == 1:
        requests.reverse()
    lengths = [len(request) for request in requests]
    batched_cache = initial_cache.clone()
    batched_output = packed_fir_reference(
        torch.cat(requests),
        weight,
        bias,
        batched_cache,
        _query_start_loc(lengths),
        torch.tensor([1 if index == target_position else 2 for index in range(2)]),
        torch.tensor([index == target_position for index in range(2)]),
        group_size=2,
        gated_bias=False,
        flip_filter=False,
    )
    target_start = sum(lengths[:target_position])

    torch.testing.assert_close(
        batched_output[target_start : target_start + len(target)],
        single_output,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(batched_cache[1], single_cache[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("state_indices", "message"),
    [
        (torch.tensor([1, 1]), "unique"),
        (torch.tensor([1, 4]), "cache block"),
    ],
)
def test_invalid_state_indices_are_rejected(state_indices, message):
    with pytest.raises(ValueError, match=message):
        packed_fir_reference(
            torch.ones((2, 2)),
            torch.ones((2, 3)),
            None,
            torch.zeros((3, 2, 2)),
            torch.tensor([0, 1, 2]),
            state_indices,
            torch.zeros(2, dtype=torch.bool),
            group_size=1,
            gated_bias=False,
            flip_filter=False,
        )
