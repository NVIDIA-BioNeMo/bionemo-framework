# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from types import SimpleNamespace

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
            )
        ),
        None,
    )

    proof = worker.snapshot_evo2_proof_phase("steady-0")

    assert len(worker.llm.llm_engine.logger_manager.stat_loggers) == 1
    assert proof["phase"] == "steady-0"
    assert proof["cudagraph_observations"][0]["num_unpadded_tokens"] == 48
    assert proof["cudagraph_summary"][0]["count"] == 1
    assert proof["resolved_config"] == {"resolved": "engine-config"}
    assert proof["worker_proof"][0]["fir_routes"]["equal_length_conv"]["calls"] == 9
    assert rpc_calls == [
        ("reset_evo2_proof_state", ()),
        ("snapshot_evo2_proof_state", ()),
    ]
