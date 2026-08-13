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

import json
import logging
from types import SimpleNamespace

import pytest
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.megatron.megatron_generation import (
    MegatronGeneration,
    _adapter_requires_all_workers,
    _load_generation_adapter,
)

import bionemo.evo2_phage_gen.nemo_rl_evo2_generation as evo2_generation
from bionemo.evo2_phage_gen.nemo_rl_evo2_generation import (
    Evo2GenerationResult,
    Evo2MegatronGenerationAdapter,
    _PromptTokenProxy,
    resume_generation_call_offset,
    should_use_evo2_native_batched_generation,
)


class _Tokenizer:
    vocab_size = 8

    def tokenize(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def test_prompt_token_proxy_preserves_nemo_rl_prompt_ids_and_delegates_other_text():
    tokenizer = _Tokenizer()
    proxy = _PromptTokenProxy(tokenizer, [[11, 12], [21, 22, 23]])

    assert proxy.tokenize(proxy.prompts[0]) == [11, 12]
    assert proxy.tokenize(proxy.prompts[1]) == [21, 22, 23]
    assert proxy.tokenize("AC") == [65, 67]
    assert proxy.detokenize([65, 67]) == "AC"
    assert proxy.vocab_size == 8


def test_nemo_rl_generation_adapter_loader_imports_configured_adapter():
    config = {
        "mcore_generation_config": {
            "generation_adapter": ("bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"),
            "generation_adapter_config": {"seed": 13},
        }
    }

    adapter = _load_generation_adapter(config)

    assert isinstance(adapter, Evo2MegatronGenerationAdapter)
    assert adapter.config["seed"] == 13
    assert _adapter_requires_all_workers(adapter, config)


def test_should_use_evo2_native_batched_generation_requires_evo2_batch_and_model():
    cfg = {
        "generation": {
            "mcore_generation_config": {
                "prompt_batch_size": 8,
                "generation_adapter": ("bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"),
            }
        }
    }
    evo2_model = SimpleNamespace(decoder=SimpleNamespace(hyena_state_shapes_per_request=lambda: None))
    wrapped_evo2_model = SimpleNamespace(module=SimpleNamespace(module=evo2_model))
    non_evo2_model = SimpleNamespace(decoder=SimpleNamespace())

    assert not should_use_evo2_native_batched_generation(
        {"generation": {"mcore_generation_config": {"prompt_batch_size": 8}}},
        evo2_model,
        batch_size=8,
    )
    assert should_use_evo2_native_batched_generation(cfg, evo2_model, batch_size=8)
    assert should_use_evo2_native_batched_generation(cfg, wrapped_evo2_model, batch_size=8)
    assert should_use_evo2_native_batched_generation(cfg, evo2_model, batch_size=1)
    assert not should_use_evo2_native_batched_generation(
        {"generation": {"mcore_generation_config": {"prompt_batch_size": 1}}},
        evo2_model,
        batch_size=8,
    )
    assert not should_use_evo2_native_batched_generation(cfg, non_evo2_model, batch_size=8)


def test_evo2_adapter_rng_seed_advances_and_records_trace(caplog, capsys):
    adapter = Evo2MegatronGenerationAdapter({"seed": 17, "seed_stride": 101})
    worker = SimpleNamespace(rank=0, cfg={"generation": {"mcore_generation_config": {}}})

    with caplog.at_level(logging.INFO, logger=evo2_generation.__name__):
        assert adapter._next_seed(worker) == 17
        assert adapter._next_seed(worker) == 118
    assert worker._evo2_generation_rng_trace == [
        {
            "rank": 0,
            "data_parallel_rank": 0,
            "data_parallel_size": 1,
            "tensor_parallel_rank": 0,
            "tensor_parallel_size": 1,
            "call_index": 0,
            "seed_index": 0,
            "seed": 17,
            "base_seed": 17,
            "seed_stride": 101,
        },
        {
            "rank": 0,
            "data_parallel_rank": 0,
            "data_parallel_size": 1,
            "tensor_parallel_rank": 0,
            "tensor_parallel_size": 1,
            "call_index": 1,
            "seed_index": 1,
            "seed": 118,
            "base_seed": 17,
            "seed_stride": 101,
        },
    ]
    trace_lines = caplog.messages
    assert len(trace_lines) == 2
    for line, expected in zip(trace_lines, worker._evo2_generation_rng_trace, strict=True):
        assert line.startswith("EVO2_SEED_TRACE ")
        payload = line.removeprefix("EVO2_SEED_TRACE ")
        assert json.loads(payload) == expected
        assert payload == json.dumps(expected, sort_keys=True)
    assert capsys.readouterr().out == ""


def test_evo2_adapter_rng_seed_continues_from_configured_call_offset():
    adapter = Evo2MegatronGenerationAdapter({"seed": 17, "seed_stride": 101, "call_index_offset": 2})
    worker = SimpleNamespace(
        rank=0,
        data_parallel_rank=0,
        dp_size=2,
        cfg={"generation": {"mcore_generation_config": {}}},
    )

    assert adapter._next_seed(worker) == 421
    assert adapter._next_seed(worker) == 623
    assert [entry["call_index"] for entry in worker._evo2_generation_rng_trace] == [2, 3]
    assert [entry["seed_index"] for entry in worker._evo2_generation_rng_trace] == [4, 6]


@pytest.mark.parametrize(
    ("completed_steps", "val_period", "val_at_start", "expected"),
    [(0, 10, False, 0), (30, 10, False, 33), (30, 0, False, 30), (30, 10, True, 34)],
)
def test_evo2_resume_call_offset_counts_prior_train_and_validation_generations(
    completed_steps, val_period, val_at_start, expected
):
    assert resume_generation_call_offset(completed_steps, val_period=val_period, val_at_start=val_at_start) == expected


def test_evo2_adapter_shares_tp_seed_and_separates_dp_and_successive_calls():
    adapter = Evo2MegatronGenerationAdapter({"seed": 17, "seed_stride": 101})
    dp0_tp0 = SimpleNamespace(
        rank=0,
        data_parallel_rank=0,
        dp_size=2,
        tensor_parallel_rank=0,
        tp_size=2,
        cfg={"generation": {"mcore_generation_config": {}}},
    )
    dp0_tp1 = SimpleNamespace(
        rank=1,
        data_parallel_rank=0,
        dp_size=2,
        tensor_parallel_rank=1,
        tp_size=2,
        cfg={"generation": {"mcore_generation_config": {}}},
    )
    dp1_tp0 = SimpleNamespace(
        rank=2,
        data_parallel_rank=1,
        dp_size=2,
        tensor_parallel_rank=0,
        tp_size=2,
        cfg={"generation": {"mcore_generation_config": {}}},
    )

    assert adapter._next_seed(dp0_tp0) == 17
    assert adapter._next_seed(dp0_tp1) == 17
    assert adapter._next_seed(dp1_tp0) == 118
    assert adapter._next_seed(dp0_tp0) == 219
    assert adapter._next_seed(dp0_tp1) == 219
    assert adapter._next_seed(dp1_tp0) == 320
    assert [entry["seed_index"] for entry in dp0_tp0._evo2_generation_rng_trace] == [0, 2]
    assert [entry["seed_index"] for entry in dp0_tp1._evo2_generation_rng_trace] == [0, 2]
    assert [entry["seed_index"] for entry in dp1_tp0._evo2_generation_rng_trace] == [1, 3]


def test_evo2_adapter_broadcasts_implicit_base_seed_from_model_parallel_leader(monkeypatch):
    from megatron.core import parallel_state

    adapter = Evo2MegatronGenerationAdapter()
    current_tp_rank = {"value": 0}
    leader_seed = {"value": None}
    initial_seeds = iter([111, 999])
    model_parallel_group = object()
    workers = [SimpleNamespace(rank=rank, cfg={"generation": {"mcore_generation_config": {}}}) for rank in range(2)]

    monkeypatch.setattr(torch, "initial_seed", lambda: next(initial_seeds))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: current_tp_rank["value"])
    monkeypatch.setattr(torch.distributed, "get_backend", lambda _group: "gloo")
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: current_tp_rank["value"])
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_model_parallel_group", lambda: model_parallel_group)
    monkeypatch.setattr(parallel_state, "get_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        parallel_state,
        "get_context_parallel_group",
        lambda: pytest.fail("CP=1 must not request a context-parallel collective group"),
    )
    monkeypatch.setattr(
        parallel_state,
        "get_context_parallel_global_ranks",
        lambda: pytest.fail("CP=1 must not request context-parallel source ranks"),
    )

    def _broadcast(seed_tensor, *, src, group):
        assert src == 0
        assert group is model_parallel_group
        if current_tp_rank["value"] == 0:
            leader_seed["value"] = int(seed_tensor.item())
        else:
            seed_tensor.fill_(leader_seed["value"])

    monkeypatch.setattr(torch.distributed, "broadcast", _broadcast)

    current_tp_rank["value"] = 0
    leader_result = adapter._next_seed(workers[0])
    current_tp_rank["value"] = 1
    peer_result = adapter._next_seed(workers[1])

    assert leader_result == peer_result == 111
    assert workers[0]._evo2_generation_rng_trace[0]["base_seed"] == 111
    assert workers[1]._evo2_generation_rng_trace[0]["base_seed"] == 111


def test_evo2_adapter_broadcasts_implicit_base_seed_across_model_and_context_parallel_groups(monkeypatch):
    from megatron.core import parallel_state

    adapter = Evo2MegatronGenerationAdapter({"seed_stride": 101})
    current = {"dp": 0, "cp": 0, "mp": 0}
    model_group_values = {}
    context_group_values = {}
    broadcast_groups = []
    initial_seeds = iter([100, 999, 300, 777, 500, 888, 700, 666])
    workers = [SimpleNamespace(rank=rank, cfg={"generation": {"mcore_generation_config": {}}}) for rank in range(8)]

    def _global_rank(dp_rank, cp_rank, mp_rank):
        return dp_rank * 4 + cp_rank * 2 + mp_rank

    def _model_group():
        return ("model", current["dp"], current["cp"])

    def _context_group():
        return ("context", current["dp"], current["mp"])

    monkeypatch.setattr(torch, "initial_seed", lambda: next(initial_seeds))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_rank",
        lambda: _global_rank(current["dp"], current["cp"], current["mp"]),
    )
    monkeypatch.setattr(torch.distributed, "get_backend", lambda _group: "gloo")
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", lambda: current["dp"])
    monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: current["mp"])
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: current["cp"])
    monkeypatch.setattr(parallel_state, "get_model_parallel_group", _model_group)
    monkeypatch.setattr(
        parallel_state,
        "get_model_parallel_src_rank",
        lambda: _global_rank(current["dp"], current["cp"], 0),
    )
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_context_parallel_group", _context_group)
    monkeypatch.setattr(
        parallel_state,
        "get_context_parallel_global_ranks",
        lambda: [
            _global_rank(current["dp"], 0, current["mp"]),
            _global_rank(current["dp"], 1, current["mp"]),
        ],
    )

    def _broadcast(seed_tensor, *, src, group):
        broadcast_groups.append(group)
        current_rank = _global_rank(current["dp"], current["cp"], current["mp"])
        values = model_group_values if group[0] == "model" else context_group_values
        if current_rank == src:
            values[group] = int(seed_tensor.item())
        else:
            seed_tensor.fill_(values[group])

    monkeypatch.setattr(torch.distributed, "broadcast", _broadcast)

    seeds = {}
    worker_idx = 0
    for dp_rank in range(2):
        for cp_rank in range(2):
            for mp_rank in range(2):
                current.update(dp=dp_rank, cp=cp_rank, mp=mp_rank)
                seeds[(dp_rank, cp_rank, mp_rank)] = adapter._next_seed(workers[worker_idx])
                worker_idx += 1

    assert {seed for (dp_rank, _cp, _mp), seed in seeds.items() if dp_rank == 0} == {100}
    assert {seed for (dp_rank, _cp, _mp), seed in seeds.items() if dp_rank == 1} == {601}
    assert {worker._evo2_generation_rng_trace[0]["base_seed"] for worker in workers[:4]} == {100}
    assert {worker._evo2_generation_rng_trace[0]["base_seed"] for worker in workers[4:]} == {500}
    assert sum(group[0] == "model" for group in broadcast_groups) == 8
    assert sum(group[0] == "context" for group in broadcast_groups) == 8


def test_evo2_native_generation_reseeds_cached_sampling_rng_for_each_adapter_call():
    cached_rng = object()
    native_dynamic = SimpleNamespace(evo2_seed=17, sampling_rng=cached_rng)

    evo2_generation._reseed_evo2_native_dynamic(native_dynamic, 118)

    assert native_dynamic.evo2_seed == 118
    assert native_dynamic.sampling_rng is None


@pytest.mark.parametrize(
    ("configured_size", "tensor_parallel_size", "expected_size"),
    [(48, 1, 48), (48, 2, 48), (48, 5, 50), (96, 7, 98), (96, 8, 96)],
)
def test_evo2_native_decode_capacity_is_rounded_up_to_tensor_parallel_multiple(
    configured_size, tensor_parallel_size, expected_size
):
    worker = SimpleNamespace(
        cfg={
            "megatron_cfg": {"tensor_model_parallel_size": tensor_parallel_size},
            "generation": {"mcore_generation_config": {"prompt_batch_size": configured_size}},
        }
    )

    assert evo2_generation._evo2_native_batched_decode_size(worker) == expected_size


@pytest.mark.parametrize(
    ("request_count", "tensor_parallel_size", "expected_size"),
    [(48, 1, 48), (48, 5, 50), (96, 7, 98), (96, 8, 96)],
)
def test_nemo_worker_request_capacity_is_rounded_up_to_tensor_parallel_multiple(
    request_count, tensor_parallel_size, expected_size
):
    from nemo_rl.models.generation.megatron.megatron_worker import (
        _round_up_request_capacity,
    )

    assert _round_up_request_capacity(request_count, tensor_parallel_size) == expected_size


def test_evo2_adapter_emits_replicated_batched_data_from_every_model_parallel_rank(monkeypatch):
    adapter = Evo2MegatronGenerationAdapter({"seed": 17})
    data = SimpleNamespace(size=2)
    prompt_tokens = torch.tensor([[11, 12], [21, 22]])
    prompt_lengths = torch.tensor([2, 2])
    sampling_params = [SimpleNamespace(num_tokens_to_generate=2)]
    parsed = object()
    group_timings = {
        "prefill_elapsed_s": 0.25,
        "decode_elapsed_s": 0.75,
        "generation_elapsed_s": 1.0,
    }
    generated = [
        Evo2GenerationResult(
            prompt_tokens=prompt_tokens[0],
            generated_tokens=[65, 67],
            generated_log_probs=[-0.1, -0.2],
            timings=group_timings,
        ),
        Evo2GenerationResult(
            prompt_tokens=prompt_tokens[1],
            generated_tokens=[67, 65],
            generated_log_probs=[-0.3, -0.4],
            timings=group_timings,
        ),
    ]

    worker = SimpleNamespace(
        rank=1,
        cfg={
            "generation": {
                "mcore_generation_config": {
                    "prompt_batch_size": 8,
                    "generation_adapter": (
                        "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"
                    ),
                }
            }
        },
        model=SimpleNamespace(decoder=SimpleNamespace(hyena_state_shapes_per_request=lambda: None)),
        _prepare_data_for_generation=lambda _data, _greedy: (
            prompt_tokens,
            prompt_lengths,
            sampling_params,
        ),
        _parse_result_to_batched_data_dict=lambda _data, _result: parsed,
    )

    monkeypatch.setattr(
        evo2_generation,
        "generate_evo2_native_batched",
        lambda *args, **kwargs: generated,
    )

    assert adapter.generate_worker(worker, data=data, greedy=False) is parsed
    assert worker._evo2_generation_timing["timing/train/generation/evo2_prefill_elapsed_s"] == 0.25
    assert worker._evo2_generation_timing["timing/train/generation/evo2_decode_elapsed_s"] == 0.75
    assert worker._evo2_generation_timing["timing/train/generation/evo2_generation_elapsed_s"] == 1.0

    monkeypatch.setattr(
        evo2_generation,
        "generate_evo2_native_batched",
        lambda *args, **kwargs: generated[:1],
    )
    with pytest.raises(RuntimeError, match="returned 1 results for 2 prompts"):
        adapter.generate_worker(worker, data=data, greedy=False)


def test_evo2_adapter_aggregates_cold_and_multi_group_timings_by_stable_group_id(monkeypatch):
    adapter = Evo2MegatronGenerationAdapter({"seed": 17})
    prompt_tokens = torch.tensor([[11, 12], [21, 22], [31, 32], [41, 42]])
    prompt_lengths = torch.tensor([2, 2, 2, 2])
    sampling_params = [SimpleNamespace(num_tokens_to_generate=2)]
    group_zero = {
        "timing_scope": "native_generation_group",
        "timing_group_id": "native-call-00000000-group-00000000",
        "timing_request_count": 2,
        "engine_setup_elapsed_s": 1.0,
        "context_setup_elapsed_s": 2.0,
        "cuda_graph_capture_elapsed_s": 3.0,
        "prefill_elapsed_s": 4.0,
        "decode_elapsed_s": 5.0,
        "generation_elapsed_s": 9.0,
        "total_elapsed_s": 15.0,
    }
    group_one = {
        "timing_scope": "native_generation_group",
        "timing_group_id": "native-call-00000000-group-00000002",
        "timing_request_count": 2,
        "engine_setup_elapsed_s": 0.0,
        "context_setup_elapsed_s": 0.0,
        "cuda_graph_capture_elapsed_s": 0.0,
        "prefill_elapsed_s": 6.0,
        "decode_elapsed_s": 7.0,
        "generation_elapsed_s": 13.0,
        "total_elapsed_s": 13.0,
    }
    group_zero_memory = {
        "engine_setup_peak_allocated_bytes": 100,
        "engine_setup_peak_reserved_bytes": 110,
        "context_setup_peak_allocated_bytes": 200,
        "context_setup_peak_reserved_bytes": 210,
        "cuda_graph_capture_peak_allocated_bytes": 300,
        "cuda_graph_capture_peak_reserved_bytes": 310,
        "prefill_peak_allocated_bytes": 400,
        "prefill_peak_reserved_bytes": 410,
        "decode_peak_allocated_bytes": 500,
        "decode_peak_reserved_bytes": 510,
        "generation_peak_allocated_bytes": 500,
        "generation_peak_reserved_bytes": 510,
        "total_peak_allocated_bytes": 500,
        "total_peak_reserved_bytes": 510,
    }
    group_one_memory = {
        "engine_setup_peak_allocated_bytes": 0,
        "engine_setup_peak_reserved_bytes": 0,
        "context_setup_peak_allocated_bytes": 0,
        "context_setup_peak_reserved_bytes": 0,
        "cuda_graph_capture_peak_allocated_bytes": 0,
        "cuda_graph_capture_peak_reserved_bytes": 0,
        "prefill_peak_allocated_bytes": 600,
        "prefill_peak_reserved_bytes": 610,
        "decode_peak_allocated_bytes": 550,
        "decode_peak_reserved_bytes": 560,
        "generation_peak_allocated_bytes": 600,
        "generation_peak_reserved_bytes": 610,
        "total_peak_allocated_bytes": 600,
        "total_peak_reserved_bytes": 610,
    }
    generated = [
        Evo2GenerationResult(
            prompt_tokens=prompt_tokens[idx],
            generated_tokens=[65, 67],
            generated_log_probs=[-0.1, -0.2],
            timings=dict(group_zero if idx < 2 else group_one),
            memory=dict(group_zero_memory if idx < 2 else group_one_memory),
        )
        for idx in range(4)
    ]
    worker = SimpleNamespace(
        rank=0,
        cfg={
            "generation": {
                "mcore_generation_config": {
                    "prompt_batch_size": 2,
                    "generation_adapter": (
                        "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"
                    ),
                }
            }
        },
        model=SimpleNamespace(decoder=SimpleNamespace(hyena_state_shapes_per_request=lambda: None)),
        _prepare_data_for_generation=lambda _data, _greedy: (
            prompt_tokens,
            prompt_lengths,
            sampling_params,
        ),
        _parse_result_to_batched_data_dict=lambda _data, result: result,
    )
    monkeypatch.setattr(evo2_generation, "generate_evo2_native_batched", lambda *args, **kwargs: generated)

    adapter.generate_worker(worker, data=SimpleNamespace(size=4))

    timing = worker._evo2_generation_timing
    assert timing["timing/train/generation/evo2_engine_setup_elapsed_s"] == 1.0
    assert timing["timing/train/generation/evo2_context_setup_elapsed_s"] == 2.0
    assert timing["timing/train/generation/evo2_cuda_graph_capture_elapsed_s"] == 3.0
    assert timing["timing/train/generation/evo2_prefill_elapsed_s"] == 10.0
    assert timing["timing/train/generation/evo2_decode_elapsed_s"] == 12.0
    assert timing["timing/train/generation/evo2_generation_elapsed_s"] == 22.0
    assert timing["timing/train/generation/evo2_total_elapsed_s"] == 28.0
    expected_memory_metrics = {
        "engine_setup_peak_allocated_bytes": 100,
        "engine_setup_peak_reserved_bytes": 110,
        "context_setup_peak_allocated_bytes": 200,
        "context_setup_peak_reserved_bytes": 210,
        "cuda_graph_capture_peak_allocated_bytes": 300,
        "cuda_graph_capture_peak_reserved_bytes": 310,
        "prefill_peak_allocated_bytes": 600,
        "prefill_peak_reserved_bytes": 610,
        "decode_peak_allocated_bytes": 550,
        "decode_peak_reserved_bytes": 560,
        "generation_peak_allocated_bytes": 600,
        "generation_peak_reserved_bytes": 610,
        "total_peak_allocated_bytes": 600,
        "total_peak_reserved_bytes": 610,
    }
    for metric_name, expected_value in expected_memory_metrics.items():
        assert timing[f"memory/train/generation/evo2_{metric_name}"] == expected_value


def test_evo2_adapter_forwards_exact_generation_controls(monkeypatch):
    adapter = Evo2MegatronGenerationAdapter({"ignore_eos": True, "strict_generation": True})
    prompt_tokens = torch.tensor([[11, 12], [21, 22]])
    prompt_lengths = torch.tensor([2, 2])
    sampling_params = [SimpleNamespace(num_tokens_to_generate=2)] * 2
    forwarded = {}

    worker = SimpleNamespace(
        rank=0,
        cfg={
            "generation": {
                "mcore_generation_config": {
                    "prompt_batch_size": 2,
                    "generation_adapter": (
                        "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"
                    ),
                }
            }
        },
        model=SimpleNamespace(decoder=SimpleNamespace(hyena_state_shapes_per_request=lambda: None)),
        megatron_tokenizer=_Tokenizer(),
        _evo2_native_dynamic_components=SimpleNamespace(forward_model=object(), evo2_seed=0, sampling_rng=None),
        _prepare_data_for_generation=lambda _data, _greedy: (
            prompt_tokens,
            prompt_lengths,
            sampling_params,
        ),
        _parse_result_to_batched_data_dict=lambda _data, result: result,
    )

    def _fake_generate_native_dynamic(*args, **kwargs):
        forwarded.update(kwargs)
        return [
            SimpleNamespace(
                prompt_tokens=prompt_tokens[idx].tolist(),
                generated_tokens=[65, 67],
                generated_log_probs=[-0.1, -0.2],
                memory={
                    "generation_peak_allocated_bytes": 123,
                    "generation_peak_reserved_bytes": 456,
                },
            )
            for idx in range(2)
        ]

    monkeypatch.setattr("bionemo.evo2.run.infer._generate_native_dynamic", _fake_generate_native_dynamic)

    results = adapter.generate_worker(worker, data=SimpleNamespace(size=2))

    assert len(results) == 2
    assert forwarded["ignore_eos"] is True
    assert forwarded["strict_generation"] is True
    assert results[0].memory == {
        "generation_peak_allocated_bytes": 123,
        "generation_peak_reserved_bytes": 456,
    }


def test_megatron_generation_shards_adapter_input_across_dp_and_gathers_in_order():
    class _ShardingAnnotations:
        def get_axis_size(self, axis):
            assert axis == "data_parallel"
            return 2

    class _WorkerGroup:
        def __init__(self):
            self.sharding_annotations = _ShardingAnnotations()
            self.call = None
            self.future_bundle = object()
            self.outputs = [
                BatchedDataDict(
                    {
                        "output_ids": torch.tensor([[11, 12, 13], [21, 22, 23]]),
                        "generation_lengths": torch.tensor([1, 1]),
                        "unpadded_sequence_lengths": torch.tensor([3, 3]),
                        "logprobs": torch.tensor([[0.0, 0.0, -0.1], [0.0, 0.0, -0.2]]),
                        "truncated": torch.tensor([False, False]),
                    }
                ),
                BatchedDataDict(
                    {
                        "output_ids": torch.tensor([[31, 32, 33, 34], [41, 42, 43, 44]]),
                        "generation_lengths": torch.tensor([2, 2]),
                        "unpadded_sequence_lengths": torch.tensor([4, 4]),
                        "logprobs": torch.tensor([[0.0, 0.0, -0.3, -0.4], [0.0, 0.0, -0.5, -0.6]]),
                        "truncated": torch.tensor([False, False]),
                    }
                ),
            ]

        def run_all_workers_sharded_data(self, method_name, **kwargs):
            self.call = (method_name, kwargs)
            return self.future_bundle

        def get_all_worker_results(self, future_bundle):
            assert future_bundle is self.future_bundle
            return self.outputs

    worker_group = _WorkerGroup()
    generation = object.__new__(MegatronGeneration)
    generation.cfg = {
        "_pad_token_id": 99,
        "mcore_generation_config": {
            "generation_adapter": ("bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter")
        },
    }
    generation._owns_policy = False
    generation._policy = SimpleNamespace(worker_group=worker_group)
    generation._generation_adapter = _load_generation_adapter(generation.cfg)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "input_lengths": torch.tensor([2, 2, 2, 2]),
        }
    )

    result = generation.generate(data)

    assert worker_group.call is not None
    method_name, call = worker_group.call
    assert method_name == "generate_with_adapter"
    assert call["in_sharded_axes"] == ["data_parallel"]
    assert call["replicate_on_axes"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    assert call["output_is_replicated"] == call["replicate_on_axes"]
    assert call["common_kwargs"] == {"greedy": False}
    assert call["data"][0]["input_ids"].tolist() == [[1, 2], [3, 4]]
    assert call["data"][1]["input_ids"].tolist() == [[5, 6], [7, 8]]
    assert result["output_ids"].tolist() == [
        [11, 12, 13, 99],
        [21, 22, 23, 99],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
    ]
    assert result["generation_lengths"].tolist() == [1, 1, 2, 2]

    too_small = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )
    with pytest.raises(ValueError, match=r"batch size 1.*data-parallel size 2"):
        generation.generate(too_small)

    worker_group.outputs = worker_group.outputs[:1]
    with pytest.raises(RuntimeError, match="expected 2 data-parallel results, received 1"):
        generation.generate(data)


def test_megatron_generation_balances_uneven_dp_shards_without_empty_replicas():
    class _ShardingAnnotations:
        @staticmethod
        def get_axis_size(axis):
            assert axis == "data_parallel"
            return 4

    class _WorkerGroup:
        def __init__(self):
            self.sharding_annotations = _ShardingAnnotations()
            self.shards = None

        def run_all_workers_sharded_data(self, method_name, **kwargs):
            assert method_name == "generate_with_adapter"
            self.shards = kwargs["data"]
            return object()

        def get_all_worker_results(self, _future_bundle):
            outputs = []
            for shard in self.shards:
                input_ids = shard["input_ids"]
                shard_size = input_ids.size(0)
                outputs.append(
                    BatchedDataDict(
                        {
                            "output_ids": torch.cat(
                                [input_ids, torch.full((shard_size, 1), 9, dtype=torch.long)], dim=1
                            ),
                            "generation_lengths": torch.ones(shard_size, dtype=torch.long),
                            "unpadded_sequence_lengths": torch.full((shard_size,), 3, dtype=torch.long),
                            "logprobs": torch.zeros((shard_size, 3)),
                            "truncated": torch.zeros(shard_size, dtype=torch.bool),
                        }
                    )
                )
            return outputs

    worker_group = _WorkerGroup()
    generation = object.__new__(MegatronGeneration)
    generation.cfg = {
        "_pad_token_id": 99,
        "mcore_generation_config": {
            "generation_adapter": ("bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter")
        },
    }
    generation._owns_policy = False
    generation._policy = SimpleNamespace(worker_group=worker_group)
    generation._generation_adapter = _load_generation_adapter(generation.cfg)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]]),
            "input_lengths": torch.full((5,), 2, dtype=torch.long),
        }
    )

    result = generation.generate(data)

    assert [shard.size for shard in worker_group.shards] == [2, 1, 1, 1]
    assert [shard["input_ids"][:, 0].tolist() for shard in worker_group.shards] == [[0, 1], [2], [3], [4]]
    assert result["output_ids"][:, 0].tolist() == [0, 1, 2, 3, 4]
