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
from unittest.mock import Mock

import pytest
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.megatron.megatron_generation import (
    MegatronGeneration,
    _adapter_requires_all_workers,
    _load_generation_adapter,
)
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

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


@pytest.mark.parametrize(("precision", "prepared"), [("fp8", True), ("fp4", True), ("bf16", False)])
def test_quantized_rollout_prepares_rows_outside_inference_mode(monkeypatch, precision, prepared):
    calls = []

    def prepare(model, config):
        calls.append((model, config, torch.is_inference_mode_enabled()))
        return True

    monkeypatch.setattr("bionemo.evo2.run.low_precision.prepare_model_for_quantized_inference", prepare)
    config = SimpleNamespace(
        fp8="e4m3" if precision == "fp8" else None,
        fp4="nvfp4" if precision == "fp4" else None,
    )
    model = SimpleNamespace(config=config)

    with torch.inference_mode():
        evo2_generation._prepare_evo2_quantized_inference(model)

    assert bool(calls) is prepared
    if prepared:
        assert calls == [(model, config, False)]


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


def test_finish_generation_keeps_native_engine_across_rollout_cycles(monkeypatch):
    adapter = Evo2MegatronGenerationAdapter({"seed": 17})
    assert adapter.bypasses_persistent_mcore_engine is True
    context = SimpleNamespace(reset_count=0)
    context.reset = lambda: setattr(context, "reset_count", context.reset_count + 1)
    model = SimpleNamespace()
    native_dynamic = SimpleNamespace(
        forward_model=model,
        shared_dyn_ctx=context,
        evo2_seed=0,
        sampling_rng=None,
    )
    worker = SimpleNamespace(
        cfg={
            "megatron_cfg": {"tensor_model_parallel_size": 1},
            "generation": {
                "mcore_generation_config": {
                    "max_model_len": 64,
                    "cuda_graph_impl": "local",
                    "prompt_batch_size": 2,
                }
            },
        },
        model=model,
        megatron_tokenizer=_Tokenizer(),
        _evo2_native_dynamic_components=native_dynamic,
    )
    prompt_tokens = torch.tensor([[11], [21]])
    prompt_lengths = torch.tensor([1, 1])
    sampling_params = [SimpleNamespace(num_tokens_to_generate=1)] * 2

    def _fake_generate(_components, prompts, **_kwargs):
        return [
            SimpleNamespace(
                prompt_tokens=[token_id],
                generated_tokens=[65],
                generated_log_probs=[-0.1],
                finish_reason="length",
                stopped_on_eos=False,
                truncated=True,
                timings={},
                memory={},
            )
            for token_id, _prompt in zip((11, 21), prompts, strict=True)
        ]

    monkeypatch.setattr("bionemo.evo2.run.infer.generate", _fake_generate)
    monkeypatch.setattr(
        "bionemo.evo2.run.infer._setup_native_dynamic_components",
        lambda **_kwargs: pytest.fail("the second rollout rebuilt the native engine"),
    )

    for _ in range(2):
        results = evo2_generation.generate_evo2_native_batched(
            worker,
            prompt_tokens,
            prompt_lengths,
            sampling_params,
        )
        assert [result.generated_tokens for result in results] == [[65], [65]]
        adapter.finish_worker(worker)

    assert worker._evo2_native_dynamic_components is native_dynamic
    assert context.reset_count == 2


def test_nemo_worker_bypasses_generic_engine_for_evo2_adapter(monkeypatch):
    from nemo_rl.models.generation.megatron import megatron_worker

    adapter = SimpleNamespace(
        bypasses_persistent_mcore_engine=True,
        finish_worker=Mock(),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(flash_decode=True),
        eval=Mock(),
        rotary_pos_emb=None,
    )
    worker = MegatronGenerationMixin()
    worker.rank = 0
    worker.model = model
    worker.cfg = {
        "generation": {"mcore_generation_config": {"cuda_graph_impl": "local"}},
    }
    worker.is_generation_colocated = True
    worker.should_disable_forward_pre_hook = False
    worker._inference_engine_initialized = True
    worker._inference_engine_asleep = False
    worker._load_generation_adapter = lambda: adapter
    worker._sleep = Mock()
    worker._initialize_inference_engine = Mock()
    worker._run_async_coordinator_start = Mock()
    worker._wake = Mock()
    graph_toggles = []

    monkeypatch.setattr(megatron_worker, "unwrap_model", lambda value: value)
    monkeypatch.setattr(megatron_worker, "log_gpu_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(megatron_worker, "toggle_cuda_graphs", lambda _model, *, set_to: graph_toggles.append(set_to))
    monkeypatch.setattr(megatron_worker.gc, "collect", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    worker.prepare_for_generation()
    worker.finish_generation()

    assert model.config.flash_decode is False
    model.eval.assert_called_once_with()
    assert graph_toggles == ["local", "none"]
    worker._sleep.assert_not_called()
    worker._initialize_inference_engine.assert_not_called()
    worker._run_async_coordinator_start.assert_not_called()
    worker._wake.assert_not_called()
    adapter.finish_worker.assert_called_once_with(worker)


def test_graph_adapter_preserves_captured_model_storage():
    adapter = Evo2MegatronGenerationAdapter({"seed": 17})
    worker = SimpleNamespace(
        _evo2_native_dynamic_components=None,
        _load_generation_adapter=lambda: adapter,
    )
    assert not adapter.requires_persistent_model_storage(worker)

    worker._evo2_native_dynamic_components = SimpleNamespace(
        cuda_graphs_enabled=True,
        shared_dyn_ctx=object(),
        cuda_graph_replay_verified=False,
        static_contexts={},
    )
    assert not adapter.requires_persistent_model_storage(worker)

    worker._evo2_native_dynamic_components.cuda_graph_replay_verified = True
    assert adapter.requires_persistent_model_storage(worker)
    assert MegatronGenerationMixin._generation_adapter_requires_persistent_model_storage(worker)

    worker._evo2_native_dynamic_components.shared_dyn_ctx = None
    worker._evo2_native_dynamic_components.cuda_graph_replay_verified = False
    worker._evo2_native_dynamic_components.static_contexts = {
        (2, 64): SimpleNamespace(evo2_static_cuda_graph_replay_verified=False)
    }
    assert not adapter.requires_persistent_model_storage(worker)

    static_context = worker._evo2_native_dynamic_components.static_contexts[(2, 64)]
    static_context.evo2_static_cuda_graph_replay_verified = True
    assert adapter.requires_persistent_model_storage(worker)

    worker._evo2_native_dynamic_components.cuda_graphs_enabled = False
    assert not adapter.requires_persistent_model_storage(worker)


@pytest.mark.parametrize(
    ("precision_kind", "vortex_style_fp8", "expected"),
    [
        ("bf16", False, False),
        ("fp8", False, True),
        ("fp8-all-layers", False, True),
        ("mxfp8", False, True),
        ("nvfp4", False, True),
        ("bf16", True, True),
    ],
)
def test_graph_adapter_recaptures_quantized_graphs_after_refit(precision_kind, vortex_style_fp8, expected):
    adapter = Evo2MegatronGenerationAdapter({"seed": 17})
    native_dynamic = SimpleNamespace(
        cuda_graphs_enabled=True,
        shared_dyn_ctx=object(),
        cuda_graph_replay_verified=True,
        static_contexts={},
        precision_kind=precision_kind,
        hyena_model=SimpleNamespace(config=SimpleNamespace(vortex_style_fp8=vortex_style_fp8)),
        cuda_graph_force_recapture=False,
    )
    worker = SimpleNamespace(_evo2_native_dynamic_components=native_dynamic)

    assert adapter.model_refit_complete(worker) is expected
    assert native_dynamic.cuda_graph_force_recapture is expected


@pytest.mark.parametrize(("storage_required", "expected_move_params"), [(False, True), (True, False)])
def test_refit_offload_respects_graph_storage(monkeypatch, storage_required, expected_move_params):
    calls = []
    model = SimpleNamespace(eval=lambda: calls.append(("eval",)))
    worker = SimpleNamespace(
        model=model,
        _generation_adapter_requires_persistent_model_storage=lambda: storage_required,
        move_model=lambda moved_model, device, **kwargs: (
            calls.append(("move", moved_model, device, kwargs)) or moved_model
        ),
        offload_before_refit=lambda: calls.append(("offload-before-refit",)),
        _generation_adapter_model_refit_complete=lambda: calls.append(("refit-complete",)),
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch, "randn", lambda *_args, **_kwargs: SimpleNamespace(cuda=lambda: None))

    MegatronPolicyWorkerImpl.offload_after_refit(worker)

    move_call = calls[0]
    assert move_call[:3] == ("move", model, "cpu")
    assert move_call[3]["move_params"] is expected_move_params
    assert calls[1:] == [("eval",), ("offload-before-refit",), ("refit-complete",)]


def test_evo2_adapter_aggregates_cold_and_multi_group_timings_by_stable_group_id(monkeypatch):
    adapter = Evo2MegatronGenerationAdapter({"seed": 17})
    prompt_tokens = torch.tensor([[11, 12], [21, 22], [31, 32], [41, 42]])
    prompt_lengths = torch.tensor([2, 2, 2, 2])
    sampling_params = [SimpleNamespace(num_tokens_to_generate=2)]
    group_zero = {
        "timing_scope": "native_generation_group",
        "timing_group_id": "native-call-00000000-group-00000000",
        "timing_request_count": 2,
        "phase_timing_exact": True,
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
        "phase_timing_exact": True,
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
    assert timing["timing/train/generation/evo2_phase_timing_exact"] == 1.0
    assert timing["timing/train/generation/evo2_generation_completion_tokens"] == 8.0
    assert timing["timing/train/generation/evo2_decode_completion_tokens"] == 4.0
    assert timing["timing/train/generation/evo2_generation_completion_tokens_per_s"] == pytest.approx(8 / 22)
    assert timing["timing/train/generation/evo2_decode_completion_tokens_per_s"] == pytest.approx(4 / 12)
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


def test_evo2_adapter_forwards_generation_controls(monkeypatch):
    adapter = Evo2MegatronGenerationAdapter(
        {"ignore_eos": True, "preserve_eos_token": True, "strict_generation": True}
    )
    prompt_tokens = torch.tensor([[11, 0], [21, 22]])
    prompt_lengths = torch.tensor([1, 2])
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

    def _fake_generate(*args, **kwargs):
        forwarded.update(kwargs)
        components, prompts = args
        forwarded["prompt_token_ids"] = [components.tokenizer.tokenize(prompt) for prompt in prompts]
        return [
            SimpleNamespace(
                prompt_tokens=prompt_tokens[idx, : prompt_lengths[idx]].tolist(),
                generated_tokens=[65, 0],
                generated_log_probs=[-0.1, -0.2],
                finish_reason="stop",
                stopped_on_eos=True,
                memory={
                    "generation_peak_allocated_bytes": 123,
                    "generation_peak_reserved_bytes": 456,
                },
            )
            for idx in range(2)
        ]

    monkeypatch.setattr("bionemo.evo2.run.infer.generate", _fake_generate)

    results = adapter.generate_worker(worker, data=SimpleNamespace(size=2))

    assert len(results) == 2
    assert worker._evo2_native_dynamic_components.use_torch_inference_mode is False
    assert forwarded["ignore_eos"] is True
    assert forwarded["preserve_eos_token"] is True
    assert forwarded["strict_generation"] is True
    assert forwarded["inference_backend"] == "dynamic"
    assert forwarded["evo2_batched_decode_size"] == 2
    assert forwarded["prompt_token_ids"] == [[11], [21, 22]]
    assert results[0].generated_tokens == [65, 0]
    assert results[0].generated_log_probs == [-0.1, -0.2]
    assert results[0].stopped_on_eos is True
    assert results[0].memory == {
        "generation_peak_allocated_bytes": 123,
        "generation_peak_reserved_bytes": 456,
    }


@pytest.mark.parametrize(
    ("fp8", "fp4", "expected_scope"),
    [
        (None, None, "block"),
        ("hybrid", None, "layer"),
        (None, "nvfp4", "layer"),
    ],
)
def test_adapter_resolves_quantized_graph_scope_before_setup(monkeypatch, fp8, fp4, expected_scope):
    prompt_tokens = torch.tensor([[11, 12], [21, 22]])
    prompt_lengths = torch.tensor([2, 2])
    sampling_params = [SimpleNamespace(num_tokens_to_generate=2, top_k=5, top_p=0.999)] * 2
    setup_kwargs = {}
    native_dynamic = SimpleNamespace(forward_model=object(), evo2_seed=0, sampling_rng=None)
    worker = SimpleNamespace(
        cfg={
            "generation": {
                "mcore_generation_config": {
                    "max_model_len": 5632,
                    "prompt_batch_size": 2,
                    "cuda_graph_impl": "local",
                    "inference_cuda_graph_scope": "block",
                }
            }
        },
        model=SimpleNamespace(
            config=SimpleNamespace(fp8=fp8, fp4=fp4),
            decoder=SimpleNamespace(hyena_state_shapes_per_request=lambda: None),
        ),
        megatron_tokenizer=_Tokenizer(),
    )

    def fake_setup(**kwargs):
        setup_kwargs.update(kwargs)
        return native_dynamic

    monkeypatch.setattr(evo2_generation, "_prepare_evo2_quantized_inference", lambda _model: None)
    monkeypatch.setattr("bionemo.evo2.run.infer._setup_native_dynamic_components", fake_setup)
    monkeypatch.setattr(
        "bionemo.evo2.run.infer.generate",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                prompt_tokens=prompt_tokens[index, : prompt_lengths[index]].tolist(),
                generated_tokens=[65],
                generated_log_probs=[-0.1],
                finish_reason="length",
                stopped_on_eos=False,
                memory={},
            )
            for index in range(2)
        ],
    )

    evo2_generation.generate_evo2_native_batched(worker, prompt_tokens, prompt_lengths, sampling_params)

    assert setup_kwargs["cuda_graphs_enabled"] is True
    assert setup_kwargs["cuda_graph_scope"] == expected_scope


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
