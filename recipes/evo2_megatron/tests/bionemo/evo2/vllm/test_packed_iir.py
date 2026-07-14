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

from bionemo.evo2.models.megatron.hyena.engine import step_iir
from bionemo.evo2.vllm.packed_iir import packed_iir_reference


PROMPT_LENGTHS = list(range(4, 13)) * 10 + [4, 5, 6, 7, 8, 9]


def _query_start_loc(lengths):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)


def _scalar_oracle(
    x2,
    v,
    gate,
    decay,
    residues,
    diagonal,
    initial_cache,
    query_start_loc,
    state_indices,
    has_initial_state,
):
    expected_output = torch.empty_like(x2)
    expected_cache = initial_cache.clone()
    state_size = decay.shape[1]
    log_decay = decay.log().unsqueeze(-1)

    for request_index in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[request_index])
        end = int(query_start_loc[request_index + 1])
        slot = int(state_indices[request_index])
        if slot != 0 and bool(has_initial_state[request_index]):
            state = initial_cache[slot, :, :state_size].clone().unsqueeze(0)
        else:
            state = torch.zeros((1, x2.shape[1], state_size), dtype=torch.float32)

        for token_index in range(start, end):
            output, state = step_iir(
                x2=gate[token_index].unsqueeze(0),
                x1=x2[token_index].unsqueeze(0),
                v=v[token_index].unsqueeze(0),
                D=diagonal,
                residues=residues,
                poles=log_decay,
                iir_state=state,
            )
            expected_output[token_index].copy_(output[0].to(expected_output.dtype))

        if slot != 0 and end > start:
            expected_cache[slot, :, :state_size].copy_(state[0])

    return expected_output, expected_cache


@pytest.mark.parametrize("populated_state", [False, True])
def test_packed_iir_matches_independent_scalar_oracle(populated_state):
    generator = torch.Generator().manual_seed(7100 + populated_state)
    channels = 16
    state_size = 16
    query_start_loc = _query_start_loc(PROMPT_LENGTHS)
    total_tokens = int(query_start_loc[-1])
    x2 = torch.randn((total_tokens, channels), generator=generator).to(torch.bfloat16)
    v = torch.randn((total_tokens, channels), generator=generator).to(torch.bfloat16)
    gate = torch.randn((total_tokens, channels), generator=generator).to(torch.bfloat16)
    recurrent_input = x2 * v
    decay = torch.exp(-(torch.rand((channels, state_size), generator=generator) * 0.2 + 0.01))
    residues = torch.randn((channels, state_size), generator=generator) * 0.05
    diagonal = torch.randn((channels,), generator=generator) * 0.05

    state_indices = torch.arange(len(PROMPT_LENGTHS), 0, -1, dtype=torch.int64)
    state_indices[::17] = 0
    has_initial_state = torch.zeros(len(PROMPT_LENGTHS), dtype=torch.bool)
    if populated_state:
        has_initial_state[1::3] = True
    initial_cache = torch.randn((len(PROMPT_LENGTHS) + 8, channels, 127), generator=generator)
    initial_cache[0].fill_(7.0)
    initial_cache[-1].fill_(-11.0)
    expected_output, expected_cache = _scalar_oracle(
        x2,
        v,
        gate,
        decay,
        residues,
        diagonal,
        initial_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
    )
    actual_cache = initial_cache.clone()

    actual_output = packed_iir_reference(
        recurrent_input,
        gate,
        decay,
        residues,
        diagonal,
        actual_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        state_size=state_size,
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache[0], initial_cache[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_cache[-1], initial_cache[-1], rtol=0, atol=0)
    torch.testing.assert_close(actual_cache[:, :, state_size:], initial_cache[:, :, state_size:], rtol=0, atol=0)


def test_empty_segments_and_null_slots_do_not_mutate_iir_cache():
    recurrent_input = torch.tensor([[2.0, -3.0]])
    gate = torch.tensor([[0.5, 2.0]])
    decay = torch.full((2, 2), 0.75)
    residues = torch.tensor([[0.25, 0.5], [-0.5, 0.25]])
    diagonal = torch.tensor([0.1, -0.2])
    state_cache = torch.full((3, 2, 7), 9.0)
    original_cache = state_cache.clone()

    output = packed_iir_reference(
        recurrent_input,
        gate,
        decay,
        residues,
        diagonal,
        state_cache,
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 0, 2]),
        torch.tensor([False, True, True]),
        state_size=2,
    )

    torch.testing.assert_close(output, torch.tensor([[0.85, 2.7]]))
    torch.testing.assert_close(state_cache, original_cache, rtol=0, atol=0)


@pytest.mark.parametrize("split", range(1, 12))
def test_packed_iir_chunk_splits_match_one_full_request(split):
    generator = torch.Generator().manual_seed(1212)
    length = 12
    channels = 8
    state_size = 4
    recurrent_input = torch.randn((length, channels), generator=generator).to(torch.bfloat16)
    gate = torch.randn((length, channels), generator=generator).to(torch.bfloat16)
    decay = torch.exp(-(torch.rand((channels, state_size), generator=generator) * 0.2 + 0.01))
    residues = torch.randn((channels, state_size), generator=generator) * 0.05
    diagonal = torch.randn((channels,), generator=generator) * 0.05
    initial_cache = torch.randn((2, channels, 7), generator=generator)

    full_cache = initial_cache.clone()
    full_output = packed_iir_reference(
        recurrent_input,
        gate,
        decay,
        residues,
        diagonal,
        full_cache,
        torch.tensor([0, length]),
        torch.tensor([1]),
        torch.tensor([True]),
        state_size=state_size,
    )

    split_cache = initial_cache.clone()
    first_output = packed_iir_reference(
        recurrent_input[:split],
        gate[:split],
        decay,
        residues,
        diagonal,
        split_cache,
        torch.tensor([0, split]),
        torch.tensor([1]),
        torch.tensor([True]),
        state_size=state_size,
    )
    second_output = packed_iir_reference(
        recurrent_input[split:],
        gate[split:],
        decay,
        residues,
        diagonal,
        split_cache,
        torch.tensor([0, length - split]),
        torch.tensor([1]),
        torch.tensor([True]),
        state_size=state_size,
    )

    torch.testing.assert_close(torch.cat((first_output, second_output)), full_output, rtol=0, atol=0)
    torch.testing.assert_close(split_cache, full_cache, rtol=0, atol=0)


@pytest.mark.parametrize("target_position", [0, 1])
def test_iir_request_matches_when_run_alone_or_inside_a_batch(target_position):
    generator = torch.Generator().manual_seed(13579)
    channels = 8
    state_size = 4
    target_input = torch.randn((8, channels), generator=generator).to(torch.bfloat16)
    target_gate = torch.randn((8, channels), generator=generator).to(torch.bfloat16)
    other_input = torch.randn((5, channels), generator=generator).to(torch.bfloat16) * 1000
    other_gate = torch.randn((5, channels), generator=generator).to(torch.bfloat16) * 1000
    decay = torch.exp(-(torch.rand((channels, state_size), generator=generator) * 0.2 + 0.01))
    residues = torch.randn((channels, state_size), generator=generator) * 0.05
    diagonal = torch.randn((channels,), generator=generator) * 0.05
    initial_cache = torch.randn((3, channels, 7), generator=generator)

    single_cache = initial_cache.clone()
    single_output = packed_iir_reference(
        target_input,
        target_gate,
        decay,
        residues,
        diagonal,
        single_cache,
        torch.tensor([0, len(target_input)]),
        torch.tensor([1]),
        torch.tensor([True]),
        state_size=state_size,
    )

    inputs = [target_input, other_input]
    gates = [target_gate, other_gate]
    if target_position == 1:
        inputs.reverse()
        gates.reverse()
    lengths = [len(value) for value in inputs]
    state_indices = torch.tensor([1 if index == target_position else 2 for index in range(2)])
    has_initial_state = torch.tensor([index == target_position for index in range(2)])
    batched_cache = initial_cache.clone()
    batched_output = packed_iir_reference(
        torch.cat(inputs),
        torch.cat(gates),
        decay,
        residues,
        diagonal,
        batched_cache,
        _query_start_loc(lengths),
        state_indices,
        has_initial_state,
        state_size=state_size,
    )
    target_start = sum(lengths[:target_position])

    torch.testing.assert_close(
        batched_output[target_start : target_start + len(target_input)],
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
def test_invalid_iir_state_indices_are_rejected(state_indices, message):
    with pytest.raises(ValueError, match=message):
        packed_iir_reference(
            torch.ones((2, 2)),
            torch.ones((2, 2)),
            torch.full((2, 2), 0.5),
            torch.ones((2, 2)),
            torch.ones(2),
            torch.zeros((3, 2, 2)),
            torch.tensor([0, 1, 2]),
            state_indices,
            torch.zeros(2, dtype=torch.bool),
            state_size=2,
        )
