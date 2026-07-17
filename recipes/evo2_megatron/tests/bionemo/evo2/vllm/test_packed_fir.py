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

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as functional

import bionemo.evo2.vllm.packed_fir as packed_fir_module
from bionemo.evo2.models.megatron.hyena.engine import step_fir
from bionemo.evo2.vllm.packed_fir import (
    get_fir_route_stats,
    packed_causal_fir,
    packed_fir_reference,
    reset_fir_route_stats,
    select_fir_path,
    select_production_fir_path,
)


PROMPT_LENGTHS = list(range(4, 13)) * 10 + [4, 5, 6, 7, 8, 9]


def test_route_stats_can_record_inside_opaque_compiler_runtime(monkeypatch):
    telemetry_context = getattr(packed_fir_module, "fir_route_telemetry_context", nullcontext)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    reset_fir_route_stats()

    with telemetry_context():
        packed_fir_module._record_fir_route(
            "equal_length_conv",
            num_requests=3,
            total_tokens=75_000,
            max_query_len=25_000,
            taps=128,
        )

    assert get_fir_route_stats() == {"equal_length_conv": {"calls": 1, "requests": 3, "tokens": 75_000}}


def _query_start_loc(lengths):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)


def _expanded(values, channels, group_size):
    if values is None:
        return None
    if values.shape[0] == channels:
        return values
    return values.repeat_interleave(group_size, dim=0)


def test_bucketed_fir_requires_a_five_percent_measured_speedup():
    assert select_fir_path(direct_ms=1.0, bucketed_ms=0.96) == "direct"
    assert select_fir_path(direct_ms=1.0, bucketed_ms=0.94) == "bucketed"


def test_production_fir_uses_grouped_convolution_only_for_proven_equal_long_128_tap_segments():
    assert (
        select_production_fir_path(
            num_requests=2,
            total_tokens=50_000,
            max_query_len=25_000,
            taps=128,
        )
        == "equal_length_conv"
    )
    assert (
        select_production_fir_path(
            num_requests=2,
            total_tokens=49_999,
            max_query_len=25_000,
            taps=128,
        )
        == "direct"
    )
    assert (
        select_production_fir_path(
            num_requests=1,
            total_tokens=25_000,
            max_query_len=25_000,
            taps=7,
        )
        == "direct"
    )
    assert (
        select_production_fir_path(
            num_requests=1,
            total_tokens=511,
            max_query_len=511,
            taps=128,
        )
        == "direct"
    )


def test_packed_causal_fir_writes_into_caller_output_buffer():
    generator = torch.Generator().manual_seed(1704)
    x = torch.randn((2, 4), generator=generator)
    weight = torch.randn((4, 3), generator=generator)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int64)
    state_indices = torch.tensor([1, 2], dtype=torch.int64)
    has_initial_state = torch.tensor([False, False])
    initial_cache = torch.randn((3, 4, 2), generator=generator)
    expected_cache = initial_cache.clone()
    actual_cache = initial_cache.clone()
    expected = packed_fir_reference(
        x,
        weight,
        None,
        expected_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
    )
    output = torch.full_like(x, torch.nan)

    actual = packed_causal_fir(
        x,
        weight,
        None,
        actual_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        max_query_len=1,
        out=output,
    )

    if actual is not output:
        raise AssertionError("packed FIR did not return the caller-owned output buffer")
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production packed FIR kernel")
@pytest.mark.parametrize(
    ("channels", "taps", "group_size", "gated_bias", "flip_filter", "state_width"),
    [
        (12288, 3, 16, False, False, 2),
        (4096, 7, 16, False, False, 127),
        (4096, 128, 16, True, True, 127),
    ],
)
def test_packed_causal_fir_matches_reference_and_reuses_slots(
    channels,
    taps,
    group_size,
    gated_bias,
    flip_filter,
    state_width,
):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(4000 + taps)
    real_request_count = len(PROMPT_LENGTHS)
    padded_request_count = 128
    lengths = [*PROMPT_LENGTHS, *([0] * (padded_request_count - real_request_count))]
    query_start_loc = _query_start_loc(lengths).to(device=device, dtype=torch.int32)
    total_tokens = int(query_start_loc[-1])
    num_filters = channels // group_size
    x = torch.randn((total_tokens, channels), device=device, dtype=torch.bfloat16, generator=generator)
    weight = torch.randn((num_filters, taps), device=device, dtype=torch.bfloat16, generator=generator) * 0.025
    bias = torch.randn((num_filters,), device=device, dtype=torch.bfloat16, generator=generator) * 0.025

    real_slots = torch.arange(real_request_count, 0, -1, device=device, dtype=torch.int32)
    real_slots[::17] = 0
    state_indices = torch.cat(
        (real_slots, torch.zeros(padded_request_count - real_request_count, device=device, dtype=torch.int32))
    )
    has_initial_state = state_indices.ne(0) & (torch.arange(padded_request_count, device=device) % 2 == 0)
    initial_cache = torch.randn(
        (real_request_count + 8, channels, state_width),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    initial_cache[0].fill_(7.0)
    initial_cache[-1].fill_(-11.0)
    expected_cache = initial_cache.clone()
    actual_cache = initial_cache.clone()
    expected_output = packed_fir_reference(
        x,
        weight,
        bias,
        expected_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        group_size=group_size,
        gated_bias=gated_bias,
        flip_filter=flip_filter,
    )
    actual_output = packed_causal_fir(
        x,
        weight,
        bias,
        actual_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        group_size=group_size,
        gated_bias=gated_bias,
        flip_filter=flip_filter,
        max_query_len=12,
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=2e-5, atol=2e-5)

    next_x = torch.randn(x.shape, device=device, dtype=torch.bfloat16, generator=generator)
    next_has_initial_state = state_indices.ne(0)
    expected_next = packed_fir_reference(
        next_x,
        weight,
        bias,
        expected_cache,
        query_start_loc,
        state_indices,
        next_has_initial_state,
        group_size=group_size,
        gated_bias=gated_bias,
        flip_filter=flip_filter,
    )
    actual_next = packed_causal_fir(
        next_x,
        weight,
        bias,
        actual_cache,
        query_start_loc,
        state_indices,
        next_has_initial_state,
        group_size=group_size,
        gated_bias=gated_bias,
        flip_filter=flip_filter,
        max_query_len=12,
    )

    torch.testing.assert_close(actual_next, expected_next, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_cache[0], initial_cache[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_cache[-1], initial_cache[-1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production packed FIR kernel")
@pytest.mark.parametrize(
    ("taps", "gated_bias", "flip_filter"),
    [(3, False, False), (7, False, False), (128, True, True)],
)
def test_packed_causal_fir_handles_mixed_irregular_lengths_across_internal_tiles(
    taps,
    gated_bias,
    flip_filter,
):
    device = torch.device("cuda")
    lengths = [1, 31, 32, 33, 47, 64, 65, 127, 129, 257, 511, 0]
    channels = 16
    group_size = 4
    generator = torch.Generator(device=device).manual_seed(12000 + taps)
    query_start_loc = _query_start_loc(lengths).to(device=device, dtype=torch.int64)
    x = torch.randn(
        (int(query_start_loc[-1]), channels),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = (
        torch.randn(
            (channels // group_size, taps),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    bias = (
        torch.randn(
            (channels // group_size,),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    state_indices = torch.tensor([1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 0], device=device, dtype=torch.int64)
    has_initial_state = torch.tensor(
        [True, False, True, True, False, True, False, True, True, False, True, True],
        device=device,
    )
    initial_cache = torch.randn(
        (12, channels, max(127, taps - 1) + 3),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    initial_cache[0].fill_(17.0)
    expected_cache = initial_cache.clone()
    actual_cache = initial_cache.clone()

    expected = packed_fir_reference(
        x,
        weight,
        bias,
        expected_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        group_size=group_size,
        gated_bias=gated_bias,
        flip_filter=flip_filter,
    )
    actual = packed_causal_fir(
        x,
        weight,
        bias,
        actual_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        group_size=group_size,
        gated_bias=gated_bias,
        flip_filter=flip_filter,
        max_query_len=max(lengths),
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_cache[0], initial_cache[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production packed FIR kernel")
def test_packed_causal_fir_long_prefill_matches_independent_convolution_oracle():
    device = torch.device("cuda")
    length = 8193
    channels = 8
    taps = 128
    group_size = 2
    generator = torch.Generator(device=device).manual_seed(25000)
    x = torch.randn((length, channels), device=device, dtype=torch.bfloat16, generator=generator)
    weight = (
        torch.randn(
            (channels // group_size, taps),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    )
    bias = (
        torch.randn(
            (channels // group_size,),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    )
    state_cache = torch.randn((2, channels, taps + 5), device=device, dtype=torch.float32, generator=generator)
    original_cache = state_cache.clone()
    history = original_cache[1, :, : taps - 1]
    expanded_weight = weight.repeat_interleave(group_size, dim=0).float().flip(-1)
    expanded_bias = bias.repeat_interleave(group_size, dim=0).float()
    padded = torch.cat((history, x.float().transpose(0, 1)), dim=-1)
    expected = functional.conv1d(
        padded.unsqueeze(0),
        expanded_weight[:, None, :],
        groups=channels,
    ).squeeze(0)
    expected = (expected + expanded_bias[:, None] * x.float().transpose(0, 1)).transpose(0, 1).to(x.dtype)

    actual = packed_causal_fir(
        x,
        weight,
        bias,
        state_cache,
        torch.tensor([0, length], device=device, dtype=torch.int64),
        torch.tensor([1], device=device, dtype=torch.int64),
        torch.tensor([True], device=device),
        group_size=group_size,
        gated_bias=True,
        flip_filter=True,
        max_query_len=length,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_cache[1, :, : taps - 1], padded[:, -(taps - 1) :], rtol=0, atol=0)
    torch.testing.assert_close(state_cache[1, :, taps - 1 :], original_cache[1, :, taps - 1 :], rtol=0, atol=0)
    torch.testing.assert_close(state_cache[0], original_cache[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for torch.compile FIR coverage")
def test_packed_causal_fir_long_kernel_is_torch_compile_compatible():
    device = torch.device("cuda")
    lengths = [1_024, 1_024]
    channels = 16
    taps = 128
    group_size = 4
    generator = torch.Generator(device=device).manual_seed(31415)
    query_start_loc = _query_start_loc(lengths).to(device=device, dtype=torch.int32)
    x = torch.randn(
        (int(query_start_loc[-1]), channels),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = (
        torch.randn(
            (channels // group_size, taps),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    )
    bias = (
        torch.randn(
            (channels // group_size,),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    )
    state_indices = torch.arange(1, len(lengths) + 1, device=device, dtype=torch.int32)
    has_initial_state = torch.tensor([True, False], device=device)
    initial_cache = torch.randn(
        (len(lengths) + 1, channels, taps - 1),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    def run(input_tensor, cache):
        return packed_causal_fir(
            input_tensor,
            weight,
            bias,
            cache,
            query_start_loc,
            state_indices,
            has_initial_state,
            group_size=group_size,
            gated_bias=True,
            flip_filter=True,
            max_query_len=max(lengths),
        )

    eager_cache = initial_cache.clone()
    expected = run(x, eager_cache)
    compiled_cache = initial_cache.clone()
    compiled = torch.compile(run, fullgraph=True)
    actual = compiled(x, compiled_cache)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled_cache, eager_cache, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
@pytest.mark.parametrize("lengths", ([65, 47], [1_024, 1_024]))
def test_packed_causal_fir_long_kernel_is_cuda_graph_compatible(lengths):
    device = torch.device("cuda")
    channels = 32
    taps = 128
    group_size = 4
    generator = torch.Generator(device=device).manual_seed(27182)
    query_start_loc = _query_start_loc(lengths).to(device=device, dtype=torch.int32)
    x = torch.randn(
        (int(query_start_loc[-1]), channels),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = (
        torch.randn(
            (channels // group_size, taps),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    )
    bias = (
        torch.randn(
            (channels // group_size,),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    )
    state_cache = torch.randn((3, channels, 127), device=device, dtype=torch.float32, generator=generator)
    state_cache[0].fill_(19.0)
    state_indices = torch.tensor([1, 0], device=device, dtype=torch.int32)
    has_initial_state = torch.tensor([True, False], device=device)

    def run():
        return packed_causal_fir(
            x,
            weight,
            bias,
            state_cache,
            query_start_loc,
            state_indices,
            has_initial_state,
            group_size=group_size,
            gated_bias=True,
            flip_filter=True,
            max_query_len=max(lengths),
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()

    output_pointer = output.data_ptr()
    first_output = output.clone()
    first_state = state_cache[1].clone()
    null_state = state_cache[0].clone()
    x.copy_(torch.randn(x.shape, device=device, dtype=torch.bfloat16, generator=generator) + 3)
    graph.replay()
    torch.cuda.synchronize()

    assert output.data_ptr() == output_pointer
    assert not torch.equal(output, first_output)
    assert not torch.equal(state_cache[1], first_state)
    torch.testing.assert_close(state_cache[0], null_state, rtol=0, atol=0)


def _independent_long_convolution(
    sequences,
    histories,
    weight,
    bias,
    *,
    group_size,
):
    channels = sequences.shape[-1]
    expanded_weight = weight.repeat_interleave(group_size, dim=0).float().flip(-1)
    expanded_bias = bias.repeat_interleave(group_size, dim=0).float()
    padded = torch.cat((histories, sequences.transpose(1, 2).float()), dim=-1)
    output = functional.conv1d(padded, expanded_weight[:, None, :], groups=channels)
    return (output + expanded_bias[None, :, None] * sequences.transpose(1, 2).float()).transpose(1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production packed FIR kernel")
def test_equal_long_path_matches_exact_25000_independent_oracle_for_mixed_state_slots():
    device = torch.device("cuda")
    request_count = 3
    length = 25_000
    channels = 4
    taps = 128
    group_size = 2
    generator = torch.Generator(device=device).manual_seed(2500025000)
    sequences = torch.randn(
        (request_count, length, channels),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = (
        torch.randn((channels // group_size, taps), device=device, dtype=torch.bfloat16, generator=generator) * 0.01
    )
    bias = torch.randn((channels // group_size,), device=device, dtype=torch.bfloat16, generator=generator) * 0.01
    state_cache = torch.randn((4, channels, 135), device=device, dtype=torch.float32, generator=generator)
    original_cache = state_cache.clone()
    state_indices = torch.tensor([2, 1, 0], device=device, dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, True], device=device)
    histories = torch.stack(
        (
            original_cache[2, :, : taps - 1],
            torch.zeros_like(original_cache[1, :, : taps - 1]),
            torch.zeros_like(original_cache[0, :, : taps - 1]),
        )
    )
    expected = _independent_long_convolution(
        sequences,
        histories,
        weight,
        bias,
        group_size=group_size,
    ).to(torch.bfloat16)

    reset_fir_route_stats()
    actual = packed_causal_fir(
        sequences.reshape(-1, channels),
        weight,
        bias,
        state_cache,
        torch.arange(0, (request_count + 1) * length, length, device=device, dtype=torch.int32),
        state_indices,
        has_initial_state,
        group_size=group_size,
        gated_bias=True,
        flip_filter=True,
        max_query_len=length,
    ).reshape_as(sequences)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        state_cache[2, :, : taps - 1], sequences[0, -(taps - 1) :].transpose(0, 1).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        state_cache[1, :, : taps - 1], sequences[1, -(taps - 1) :].transpose(0, 1).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(state_cache[0], original_cache[0], rtol=0, atol=0)
    stats = get_fir_route_stats()
    assert stats["equal_length_conv"] == {"calls": 1, "requests": 3, "tokens": 75_000}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production packed FIR kernel")
def test_equal_long_path_matches_independent_oracle_across_two_continuations():
    device = torch.device("cuda")
    length = 1_537
    channels = 4
    taps = 128
    group_size = 2
    generator = torch.Generator(device=device).manual_seed(15371537)
    sequence = torch.randn((2, length, channels), device=device, dtype=torch.bfloat16, generator=generator)
    weight = (
        torch.randn((channels // group_size, taps), device=device, dtype=torch.bfloat16, generator=generator) * 0.01
    )
    bias = torch.randn((channels // group_size,), device=device, dtype=torch.bfloat16, generator=generator) * 0.01
    state_cache = torch.randn((2, channels, 127), device=device, dtype=torch.float32, generator=generator)
    initial_history = state_cache[1].clone().unsqueeze(0)
    expected = (
        _independent_long_convolution(
            sequence.reshape(1, 2 * length, channels),
            initial_history,
            weight,
            bias,
            group_size=group_size,
        )
        .to(torch.bfloat16)
        .reshape(2, length, channels)
    )
    metadata = torch.tensor([0, length], device=device, dtype=torch.int32)

    reset_fir_route_stats()
    outputs = [
        packed_causal_fir(
            continuation,
            weight,
            bias,
            state_cache,
            metadata,
            torch.tensor([1], device=device, dtype=torch.int32),
            torch.tensor([True], device=device),
            group_size=group_size,
            gated_bias=True,
            flip_filter=True,
            max_query_len=length,
        )
        for continuation in sequence
    ]

    torch.testing.assert_close(torch.stack(outputs), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_cache[1], sequence[-1, -(taps - 1) :].transpose(0, 1).float(), rtol=0, atol=0)
    assert get_fir_route_stats()["equal_length_conv"] == {
        "calls": 2,
        "requests": 2,
        "tokens": 2 * length,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production packed FIR kernel")
def test_route_stats_report_ragged_long_fallback_without_semantic_padding():
    device = torch.device("cuda")
    lengths = [1_024, 1_025]
    channels = 4
    generator = torch.Generator(device=device).manual_seed(2049)
    starts = _query_start_loc(lengths).to(device=device, dtype=torch.int32)
    reset_fir_route_stats()

    packed_causal_fir(
        torch.randn((sum(lengths), channels), device=device, dtype=torch.bfloat16, generator=generator),
        torch.randn((channels // 2, 128), device=device, dtype=torch.bfloat16, generator=generator) * 0.01,
        torch.zeros((channels // 2,), device=device, dtype=torch.bfloat16),
        torch.zeros((3, channels, 127), device=device, dtype=torch.float32),
        starts,
        torch.tensor([1, 2], device=device, dtype=torch.int32),
        torch.zeros(2, device=device, dtype=torch.bool),
        group_size=2,
        gated_bias=True,
        flip_filter=True,
        max_query_len=max(lengths),
    )

    stats = get_fir_route_stats()
    assert stats["direct"] == {"calls": 1, "requests": 2, "tokens": sum(lengths)}
    assert stats["fallback_reasons"] == {"ragged_or_chunked": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the production packed FIR kernel")
@pytest.mark.parametrize("taps", [7, 128])
@pytest.mark.parametrize("target_position", [0, 1])
def test_packed_causal_fir_request_matches_alone_or_inside_batch(taps, target_position):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(8100 + taps)
    channels = 32
    target = torch.randn((8, channels), device=device, dtype=torch.bfloat16, generator=generator)
    other = torch.randn((5, channels), device=device, dtype=torch.bfloat16, generator=generator) * 1000
    weight = torch.randn((channels // 2, taps), device=device, dtype=torch.bfloat16, generator=generator) * 0.01
    bias = torch.randn((channels // 2,), device=device, dtype=torch.bfloat16, generator=generator) * 0.01
    initial_cache = torch.randn((3, channels, 127), device=device, dtype=torch.float32, generator=generator)

    single_query_start_loc = torch.tensor([0, len(target)], device=device, dtype=torch.int32)
    single_state_indices = torch.tensor([1], device=device, dtype=torch.int32)
    single_cache = initial_cache.clone()
    single_output = packed_causal_fir(
        target,
        weight,
        bias,
        single_cache,
        single_query_start_loc,
        single_state_indices,
        torch.tensor([True], device=device),
        group_size=2,
        gated_bias=False,
        flip_filter=False,
        max_query_len=len(target),
    )

    requests = [target, other]
    if target_position == 1:
        requests.reverse()
    lengths = [len(request) for request in requests]
    query_start_loc = _query_start_loc(lengths).to(device=device, dtype=torch.int32)
    state_indices = torch.tensor(
        [1 if request_index == target_position else 2 for request_index in range(2)],
        device=device,
        dtype=torch.int32,
    )
    has_initial_state = torch.tensor(
        [request_index == target_position for request_index in range(2)],
        device=device,
    )
    batched_cache = initial_cache.clone()
    batched_output = packed_causal_fir(
        torch.cat(requests),
        weight,
        bias,
        batched_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        group_size=2,
        gated_bias=False,
        flip_filter=False,
        max_query_len=len(target),
    )
    target_start = sum(lengths[:target_position])

    torch.testing.assert_close(
        batched_output[target_start : target_start + len(target)],
        single_output,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(batched_cache[1], single_cache[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
def test_packed_causal_fir_cuda_graph_replay_uses_static_buffers():
    device = torch.device("cuda")
    batch_size = 128
    channels = 4096
    taps = 7
    generator = torch.Generator(device=device).manual_seed(6060)
    x = torch.randn((batch_size, channels), device=device, dtype=torch.bfloat16, generator=generator)
    weight = torch.randn((channels // 16, taps), device=device, dtype=torch.bfloat16, generator=generator)
    bias = torch.randn((channels // 16,), device=device, dtype=torch.bfloat16, generator=generator)
    state_cache = torch.randn((batch_size + 1, channels, 127), device=device, generator=generator)
    state_cache[0].fill_(5.0)
    query_start_loc = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
    state_indices = torch.cat(
        (
            torch.arange(1, 97, device=device, dtype=torch.int32),
            torch.zeros(32, device=device, dtype=torch.int32),
        )
    )
    has_initial_state = state_indices.ne(0)

    packed_causal_fir(
        x,
        weight,
        bias,
        state_cache,
        query_start_loc,
        state_indices,
        has_initial_state,
        group_size=16,
        gated_bias=False,
        flip_filter=False,
        max_query_len=1,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = packed_causal_fir(
            x,
            weight,
            bias,
            state_cache,
            query_start_loc,
            state_indices,
            has_initial_state,
            group_size=16,
            gated_bias=False,
            flip_filter=False,
            max_query_len=1,
        )

    output_pointer = output.data_ptr()
    first_output = output.clone()
    first_state = state_cache[1].clone()
    null_state = state_cache[0].clone()
    x.copy_(torch.randn(x.shape, device=device, dtype=torch.bfloat16, generator=generator) + 3)
    graph.replay()
    torch.cuda.synchronize()

    assert output.data_ptr() == output_pointer
    assert not torch.equal(output, first_output)
    assert not torch.equal(state_cache[1], first_state)
    torch.testing.assert_close(state_cache[0], null_state, rtol=0, atol=0)
