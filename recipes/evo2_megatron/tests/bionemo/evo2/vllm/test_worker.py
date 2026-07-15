# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from types import SimpleNamespace

import torch
from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

from bionemo.evo2.vllm import nemo_generation_worker as nemo_generation_worker_module
from bionemo.evo2.vllm import plugin as plugin_module
from bionemo.evo2.vllm import runner
from bionemo.evo2.vllm.nemo_generation_worker import Evo2NemoRlGenerationWorkerImpl
from bionemo.evo2.vllm.nemo_worker import Evo2NemoRlVllmWorkerExtension
from bionemo.evo2.vllm.worker import Evo2VllmWorkerExtension


def test_named_worker_extension_delegates_proof_state_without_callable_rpc(monkeypatch) -> None:
    worker = Evo2VllmWorkerExtension()
    monkeypatch.setattr(runner, "reset_vllm_worker_proof_state", lambda owner: {"owner": owner})
    monkeypatch.setattr(runner, "snapshot_vllm_worker_proof_state", lambda owner: {"snapshot": owner})

    assert worker.reset_evo2_proof_state() == {"owner": worker}
    assert worker.snapshot_evo2_proof_state() == {"snapshot": worker}


def test_nemo_worker_extension_composes_refit_and_evo2_proof_controls() -> None:
    assert issubclass(Evo2NemoRlVllmWorkerExtension, VllmInternalWorkerExtension)
    assert issubclass(Evo2NemoRlVllmWorkerExtension, Evo2VllmWorkerExtension)


def test_nemo_worker_extension_records_actual_incremental_refit_chunks(monkeypatch) -> None:
    load_calls = []

    def fake_load_weights(self, weights):
        load_calls.append(tuple(name for name, _ in weights))

    monkeypatch.setattr(VllmInternalWorkerExtension, "_load_weights", fake_load_weights)
    loader = SimpleNamespace(
        completed_transactions=1,
        required_parameter_names=frozenset({"a", "b"}),
        _loaded_parameter_names={"a"},
        _pending_fc1={},
        _started=True,
        _complete=False,
        _consumed=True,
    )
    worker = object.__new__(Evo2NemoRlVllmWorkerExtension)
    worker.model_runner = SimpleNamespace(model=SimpleNamespace(_weight_loader=loader))

    assert worker.reset_evo2_refit_proof_state("refit-1") == {"phase": "refit-1"}
    worker._load_weights([("a", torch.ones(2)), ("b", torch.ones(3))])
    loader.completed_transactions = 2
    loader._loaded_parameter_names = {"a", "b"}
    loader._complete = True

    proof = worker.snapshot_evo2_refit_proof_state("refit-1")

    assert load_calls == [("a", "b")]
    assert proof["phase"] == "refit-1"
    assert proof["chunk_count"] == 1
    assert proof["chunks"][0]["tensor_count"] == 2
    assert proof["chunks"][0]["tensor_bytes"] == 20
    assert proof["loader"]["completed_transactions"] == 2
    assert proof["loader"]["loaded_parameter_count"] == 2
    assert proof["loader"]["required_parameter_count"] == 2
    assert proof["loader"]["complete"] is True


def test_nemo_generation_actor_registers_evo2_before_nemo_model_preflight(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(plugin_module, "register", lambda: events.append("register-evo2"))
    monkeypatch.setattr(
        VllmGenerationWorkerImpl,
        "__init__",
        lambda self, *args, **kwargs: events.append(("base-init", args, kwargs)),
    )

    worker = object.__new__(Evo2NemoRlGenerationWorkerImpl)
    worker.__init__("bundle", seed=42)

    assert events == ["register-evo2", ("base-init", ("bundle",), {"seed": 42})]


def test_nemo_generation_worker_records_outer_graph_and_inner_route_proof(monkeypatch) -> None:
    rpc_calls = []
    monkeypatch.setattr(
        nemo_generation_worker_module,
        "resolved_config_snapshot",
        lambda config: {"resolved": config},
    )

    class FakeLLM:
        llm_engine = SimpleNamespace(
            logger_manager=SimpleNamespace(stat_loggers=[]),
            vllm_config="engine-config",
        )

        def collective_rpc(self, method, args=tuple()):
            rpc_calls.append((method, args))
            return [{"rank": 0, "fir_routes": {"equal_length_conv": {"calls": 9}}}]

    worker = object.__new__(Evo2NemoRlGenerationWorkerImpl)
    worker.llm = FakeLLM()
    worker._attach_evo2_proof_recorder()
    worker.reset_evo2_proof_phase("steady-0")
    worker._evo2_cudagraph_recorder.record(
        SimpleNamespace(
            cudagraph_stats=SimpleNamespace(
                num_unpadded_tokens=48,
                num_padded_tokens=48,
                num_paddings=0,
                runtime_mode="CUDAGraphMode.FULL",
            ),
            prefix_cache_stats=SimpleNamespace(
                preempted_requests=0,
                preempted_queries=0,
                preempted_hits=0,
            ),
            num_running_reqs=48,
            num_waiting_reqs=0,
            num_skipped_waiting_reqs=0,
        ),
        SimpleNamespace(
            num_preempted_reqs=0,
            prompt_token_stats=SimpleNamespace(computed=384, cached_tokens=0, total=384),
        ),
    )

    proof = worker.snapshot_evo2_proof_phase("steady-0")

    assert len(worker.llm.llm_engine.logger_manager.stat_loggers) == 1
    assert proof["phase"] == "steady-0"
    assert proof["cudagraph_observations"][0]["num_unpadded_tokens"] == 48
    assert proof["cudagraph_summary"][0]["count"] == 1
    assert proof["scheduler_observations"][0]["preemption_events"] == 0
    assert proof["scheduler_observations"][0]["prompt_tokens_computed"] == 384
    assert proof["resolved_config"] == {"resolved": "engine-config"}
    assert proof["worker_proof"][0]["fir_routes"]["equal_length_conv"]["calls"] == 9
    assert rpc_calls == [
        ("reset_evo2_proof_state", ()),
        ("snapshot_evo2_proof_state", ()),
    ]


def test_nemo_generation_worker_exposes_internal_refit_transaction_proof(monkeypatch) -> None:
    monkeypatch.setattr(
        nemo_generation_worker_module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_actor_id=lambda: "actor-123"),
    )

    class FakeLLM:
        llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(
                model="/checkpoint",
                architectures=("Evo2ForCausalLM",),
                dtype="bfloat16",
                max_model_len=16,
            )
        )

        def collective_rpc(self, method, args=tuple()):
            if method == "reset_evo2_refit_proof_state":
                return [{"phase": args[0]}] * 2
            if method == "snapshot_evo2_refit_proof_state":
                return [{"phase": args[0], "chunk_count": 53}] * 2
            assert method == "report_device_id"
            return ["GPU-a", "GPU-b"]

    worker = object.__new__(Evo2NemoRlGenerationWorkerImpl)
    worker.llm = FakeLLM()

    assert worker.reset_evo2_refit_phase("refit-1")["phase"] == "refit-1"
    proof = worker.snapshot_evo2_refit_phase("refit-1")

    assert proof["actor"]["ray_actor_id"] == "actor-123"
    assert proof["model"]["architectures"] == ["Evo2ForCausalLM"]
    assert proof["device_uuids"] == ["GPU-a", "GPU-b"]
    assert [item["chunk_count"] for item in proof["worker_proof"]] == [53, 53]
