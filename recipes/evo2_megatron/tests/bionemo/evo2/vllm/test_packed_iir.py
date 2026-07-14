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
from bionemo.evo2.vllm.packed_iir import packed_iir_reference, packed_modal_iir


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production packed HCL kernel")
@pytest.mark.parametrize(
    ("channels", "real_lengths"),
    [
        (1920, [1] * 96),
        (4096, PROMPT_LENGTHS),
    ],
)
def test_packed_modal_iir_matches_reference_and_reuses_slots(channels, real_lengths):
    device = torch.device("cuda")
    state_size = 16
    graph_batch_size = 128
    generator = torch.Generator(device=device).manual_seed(9100 + channels)
    lengths = [*real_lengths, *([0] * (graph_batch_size - len(real_lengths)))]
    query_start_loc = _query_start_loc(lengths).to(device=device, dtype=torch.int32)
    total_tokens = int(query_start_loc[-1])
    recurrent_input = torch.randn(
        (total_tokens, channels),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate = torch.randn(recurrent_input.shape, device=device, dtype=torch.bfloat16, generator=generator)
    decay = torch.exp(-(torch.rand((channels, state_size), device=device, generator=generator) * 0.2 + 0.01))
    residues = torch.randn((channels, state_size), device=device, generator=generator) * 0.05
    diagonal = torch.randn((channels,), device=device, generator=generator) * 0.05
    real_slots = torch.arange(len(real_lengths), 0, -1, device=device, dtype=torch.int32)
    real_slots[::17] = 0
    state_indices = torch.cat(
        (real_slots, torch.zeros(graph_batch_size - len(real_lengths), device=device, dtype=torch.int32))
    )
    has_initial_state = state_indices.ne(0) & (torch.arange(graph_batch_size, device=device) % 2 == 0)
    initial_cache = torch.randn(
        (len(real_lengths) + 8, channels, 127),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    initial_cache[0].fill_(7.0)
    initial_cache[-1].fill_(-11.0)
    expected_cache = initial_cache.clone()
    actual_cache = initial_cache.clone()

    expected_output = packed_iir_reference(
        recurrent_input,
        gate,
        decay,
        residues,
        diagonal,
        expected_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        state_size=state_size,
    )
    actual_output = packed_modal_iir(
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
        max_query_len=max(real_lengths),
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=2e-5, atol=2e-5)

    next_input = torch.randn(recurrent_input.shape, device=device, dtype=torch.bfloat16, generator=generator)
    next_gate = torch.randn(gate.shape, device=device, dtype=torch.bfloat16, generator=generator)
    next_has_initial_state = state_indices.ne(0)
    expected_next = packed_iir_reference(
        next_input,
        next_gate,
        decay,
        residues,
        diagonal,
        expected_cache,
        query_start_loc,
        state_indices,
        next_has_initial_state,
        state_size=state_size,
    )
    actual_next = packed_modal_iir(
        next_input,
        next_gate,
        decay,
        residues,
        diagonal,
        actual_cache,
        query_start_loc,
        state_indices,
        next_has_initial_state,
        state_size=state_size,
        max_query_len=max(real_lengths),
    )

    torch.testing.assert_close(actual_next, expected_next, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_cache[0], initial_cache[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_cache[-1], initial_cache[-1], rtol=0, atol=0)
    torch.testing.assert_close(actual_cache[:, :, state_size:], initial_cache[:, :, state_size:], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production packed HCL kernel")
@pytest.mark.parametrize("target_position", [0, 1])
def test_packed_modal_iir_request_matches_alone_or_inside_batch(target_position):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(24680)
    channels = 32
    state_size = 16
    target_input = torch.randn((8, channels), device=device, dtype=torch.bfloat16, generator=generator)
    target_gate = torch.randn((8, channels), device=device, dtype=torch.bfloat16, generator=generator)
    other_input = torch.randn((5, channels), device=device, dtype=torch.bfloat16, generator=generator) * 1000
    other_gate = torch.randn((5, channels), device=device, dtype=torch.bfloat16, generator=generator) * 1000
    decay = torch.exp(-(torch.rand((channels, state_size), device=device, generator=generator) * 0.2 + 0.01))
    residues = torch.randn((channels, state_size), device=device, generator=generator) * 0.05
    diagonal = torch.randn((channels,), device=device, generator=generator) * 0.05
    initial_cache = torch.randn((3, channels, 127), device=device, generator=generator)

    single_cache = initial_cache.clone()
    single_output = packed_modal_iir(
        target_input,
        target_gate,
        decay,
        residues,
        diagonal,
        single_cache,
        torch.tensor([0, len(target_input)], device=device, dtype=torch.int32),
        torch.tensor([1], device=device, dtype=torch.int32),
        torch.tensor([True], device=device),
        state_size=state_size,
        max_query_len=len(target_input),
    )

    inputs = [target_input, other_input]
    gates = [target_gate, other_gate]
    if target_position == 1:
        inputs.reverse()
        gates.reverse()
    lengths = [len(value) for value in inputs]
    query_start_loc = _query_start_loc(lengths).to(device=device, dtype=torch.int32)
    state_indices = torch.tensor(
        [1 if index == target_position else 2 for index in range(2)],
        device=device,
        dtype=torch.int32,
    )
    has_initial_state = torch.tensor([index == target_position for index in range(2)], device=device)
    batched_cache = initial_cache.clone()
    batched_output = packed_modal_iir(
        torch.cat(inputs),
        torch.cat(gates),
        decay,
        residues,
        diagonal,
        batched_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        state_size=state_size,
        max_query_len=len(target_input),
    )
    target_start = sum(lengths[:target_position])

    torch.testing.assert_close(
        batched_output[target_start : target_start + len(target_input)],
        single_output,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(batched_cache[1], single_cache[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for long-prefill HCL coverage")
def test_packed_modal_iir_long_prefill_matches_reference():
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(97531)
    lengths = [33, 65, 17]
    channels = 32
    state_size = 16
    query_start_loc = _query_start_loc(lengths).to(device=device, dtype=torch.int32)
    recurrent_input = torch.randn(
        (int(query_start_loc[-1]), channels),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate = torch.randn(recurrent_input.shape, device=device, dtype=torch.bfloat16, generator=generator)
    decay = torch.exp(-(torch.rand((channels, state_size), device=device, generator=generator) * 0.2 + 0.01))
    residues = torch.randn((channels, state_size), device=device, generator=generator) * 0.05
    diagonal = torch.randn((channels,), device=device, generator=generator) * 0.05
    state_indices = torch.tensor([3, 1, 2], device=device, dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, True], device=device)
    initial_cache = torch.randn((4, channels, 127), device=device, generator=generator)
    expected_cache = initial_cache.clone()
    actual_cache = initial_cache.clone()

    expected = packed_iir_reference(
        recurrent_input,
        gate,
        decay,
        residues,
        diagonal,
        expected_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        state_size=state_size,
    )
    actual = packed_modal_iir(
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
        max_query_len=max(lengths),
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for long-prefill HCL coverage")
def test_null_padding_tokens_do_not_advance_a_real_request_state():
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(86420)
    target_length = 12
    padding_length = 64
    channels = 32
    state_size = 16
    target_input = torch.randn((target_length, channels), device=device, dtype=torch.bfloat16, generator=generator)
    target_gate = torch.randn(target_input.shape, device=device, dtype=torch.bfloat16, generator=generator)
    decay = torch.exp(-(torch.rand((channels, state_size), device=device, generator=generator) * 0.2 + 0.01))
    residues = torch.randn((channels, state_size), device=device, generator=generator) * 0.05
    diagonal = torch.randn((channels,), device=device, generator=generator) * 0.05
    initial_cache = torch.randn((2, channels, 127), device=device, generator=generator)
    initial_cache[0].fill_(6.0)

    target_cache = initial_cache.clone()
    target_output = packed_modal_iir(
        target_input,
        target_gate,
        decay,
        residues,
        diagonal,
        target_cache,
        torch.tensor([0, target_length], device=device, dtype=torch.int32),
        torch.tensor([1], device=device, dtype=torch.int32),
        torch.tensor([True], device=device),
        state_size=state_size,
        max_query_len=target_length,
    )

    padded_cache = initial_cache.clone()
    padded_output = packed_modal_iir(
        torch.cat((target_input, torch.zeros((padding_length, channels), device=device, dtype=torch.bfloat16))),
        torch.cat((target_gate, torch.zeros((padding_length, channels), device=device, dtype=torch.bfloat16))),
        decay,
        residues,
        diagonal,
        padded_cache,
        torch.tensor([0, target_length, target_length + padding_length], device=device, dtype=torch.int32),
        torch.tensor([1, 0], device=device, dtype=torch.int32),
        torch.tensor([True, False], device=device),
        state_size=state_size,
        max_query_len=padding_length,
    )

    torch.testing.assert_close(padded_output[:target_length], target_output, rtol=0, atol=0)
    torch.testing.assert_close(padded_cache[1], target_cache[1], rtol=0, atol=0)
    torch.testing.assert_close(padded_cache[0], initial_cache[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for torch.compile HCL coverage")
def test_packed_modal_iir_matches_torch_compile():
    device = torch.device("cuda")
    batch_size = 16
    channels = 256
    state_size = 16
    generator = torch.Generator(device=device).manual_seed(3333)
    recurrent_input = torch.randn((batch_size * 4, channels), device=device, dtype=torch.bfloat16, generator=generator)
    gate = torch.randn(recurrent_input.shape, device=device, dtype=torch.bfloat16, generator=generator)
    decay = torch.exp(-(torch.rand((channels, state_size), device=device, generator=generator) * 0.2 + 0.01))
    residues = torch.randn((channels, state_size), device=device, generator=generator) * 0.05
    diagonal = torch.randn((channels,), device=device, generator=generator) * 0.05
    query_start_loc = torch.arange(0, batch_size * 4 + 1, 4, device=device, dtype=torch.int32)
    state_indices = torch.arange(1, batch_size + 1, device=device, dtype=torch.int32)
    has_initial_state = torch.ones(batch_size, device=device, dtype=torch.bool)
    initial_cache = torch.randn((batch_size + 1, channels, 127), device=device, generator=generator)

    def run(input_tensor, gate_tensor, cache):
        return packed_modal_iir(
            input_tensor,
            gate_tensor,
            decay,
            residues,
            diagonal,
            cache,
            query_start_loc,
            state_indices,
            has_initial_state,
            state_size=state_size,
            max_query_len=4,
        )

    eager_cache = initial_cache.clone()
    expected = run(recurrent_input, gate, eager_cache)
    compiled_cache = initial_cache.clone()
    compiled = torch.compile(run, fullgraph=True)
    actual = compiled(recurrent_input, gate, compiled_cache)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled_cache, eager_cache, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
def test_packed_modal_iir_cuda_graph_replay_uses_static_buffers():
    device = torch.device("cuda")
    batch_size = 128
    channels = 4096
    state_size = 16
    generator = torch.Generator(device=device).manual_seed(4444)
    recurrent_input = torch.randn((batch_size, channels), device=device, dtype=torch.bfloat16, generator=generator)
    gate = torch.randn(recurrent_input.shape, device=device, dtype=torch.bfloat16, generator=generator)
    decay = torch.exp(-(torch.rand((channels, state_size), device=device, generator=generator) * 0.2 + 0.01))
    residues = torch.randn((channels, state_size), device=device, generator=generator) * 0.05
    diagonal = torch.randn((channels,), device=device, generator=generator) * 0.05
    state_cache = torch.randn((97, channels, 127), device=device, generator=generator)
    state_cache[0].fill_(5.0)
    query_start_loc = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
    state_indices = torch.cat(
        (
            torch.arange(1, 97, device=device, dtype=torch.int32),
            torch.zeros(32, device=device, dtype=torch.int32),
        )
    )
    has_initial_state = state_indices.ne(0)

    packed_modal_iir(
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
        max_query_len=1,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = packed_modal_iir(
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
            max_query_len=1,
        )

    output_pointer = output.data_ptr()
    first_output = output.clone()
    first_state = state_cache[1].clone()
    null_state = state_cache[0].clone()
    recurrent_input.copy_(
        torch.randn(recurrent_input.shape, device=device, dtype=torch.bfloat16, generator=generator) + 3
    )
    graph.replay()
    torch.cuda.synchronize()

    assert output.data_ptr() == output_pointer
    assert not torch.equal(output, first_output)
    assert not torch.equal(state_cache[1], first_state)
    torch.testing.assert_close(state_cache[0], null_state, rtol=0, atol=0)


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
