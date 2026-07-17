# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

from bionemo.evo2.vllm import nemo_generation_worker as nemo_generation_worker_module
from bionemo.evo2.vllm import plugin as plugin_module
from bionemo.evo2.vllm import runner
from bionemo.evo2.vllm.nemo_generation_worker import Evo2NemoRlGenerationWorkerImpl
from bionemo.evo2.vllm.nemo_proof_worker import Evo2NemoRlProofVllmWorkerExtension
from bionemo.evo2.vllm.nemo_worker import Evo2NemoRlVllmWorkerExtension
from bionemo.evo2.vllm.worker import Evo2VllmWorkerExtension


def test_named_worker_extension_delegates_proof_state_without_callable_rpc(monkeypatch) -> None:
    worker = Evo2VllmWorkerExtension()
    monkeypatch.setattr(
        runner,
        "reset_vllm_worker_proof_state",
        lambda owner, reset_prefix_sources: {
            "owner": owner,
            "reset_prefix_sources": reset_prefix_sources,
        },
    )
    monkeypatch.setattr(runner, "snapshot_vllm_worker_proof_state", lambda owner: {"snapshot": owner})

    assert worker.reset_evo2_proof_state() == {"owner": worker, "reset_prefix_sources": True}
    assert worker.reset_evo2_proof_state(False) == {
        "owner": worker,
        "reset_prefix_sources": False,
    }
    assert worker.snapshot_evo2_proof_state() == {"snapshot": worker}


def _rank_local_output(req_ids, sampled_token_ids, chosen_logprobs):
    token_rows = []
    logprob_rows = []
    for request_tokens, request_logprobs in zip(sampled_token_ids, chosen_logprobs, strict=True):
        for token_id, logprob in zip(request_tokens, request_logprobs, strict=True):
            token_rows.append([token_id, (token_id + 1) % 512])
            logprob_rows.append([logprob, logprob - 1.0])
    return SimpleNamespace(
        req_ids=list(req_ids),
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[list(tokens) for tokens in sampled_token_ids],
        logprobs=SimpleNamespace(
            logprob_token_ids=np.asarray(token_rows, dtype=np.int32),
            logprobs=np.asarray(logprob_rows, dtype=np.float32),
            sampled_token_ranks=np.zeros(len(token_rows), dtype=np.int32),
            cu_num_generated_tokens=None,
        ),
    )


def test_named_worker_extension_derives_independent_tp_evidence_from_model_runner_output() -> None:
    outputs = [
        _rank_local_output(("req-0", "req-1"), ((65,), (66,)), ((-0.1,), (-0.2,))),
        _rank_local_output(("req-1", "req-0"), ((68,), (67,)), ((-0.4,), (-0.3,))),
    ]
    worker = Evo2VllmWorkerExtension()
    worker.execute_model = lambda scheduler_output: pytest.fail(
        "compiled-DAG TP execution must not depend on worker.execute_model"
    )
    worker.model_runner = SimpleNamespace(
        execute_model=lambda scheduler_output, intermediate_tensors=None: outputs.pop(0),
        sample_tokens=lambda grammar_output: pytest.fail(
            f"unexpected sample_tokens fallback: {grammar_output!r}"
        ),
    )
    worker.begin_evo2_rank_local_generation_evidence(
        phase="steady-0.wave-000",
        expected_envelope_sha256="a" * 64,
        expected_request_count=2,
        expected_max_new_tokens=2,
    )

    worker.model_runner.execute_model(SimpleNamespace(), None)
    worker.model_runner.execute_model(SimpleNamespace(), None)
    evidence = worker.snapshot_evo2_rank_local_generation_evidence()

    assert evidence["source"] == "rank_local_model_runner_execute_or_sample"
    assert evidence["request_count"] == 2
    assert evidence["generated_token_count"] == 4
    assert [row["vllm_request_id"] for row in evidence["requests"]] == ["req-0", "req-1"]
    assert [row["token_count"] for row in evidence["requests"]] == [2, 2]
    assert len({row["selected_stream_sha256"] for row in evidence["requests"]}) == 2
    assert outputs == []


def test_named_worker_extension_rejects_missing_rank_local_chosen_logprob() -> None:
    output = _rank_local_output(("req-0",), ((65,),), ((-0.1,),))
    output.logprobs.logprob_token_ids[0] = [66, 67]
    worker = Evo2VllmWorkerExtension()
    worker.model_runner = SimpleNamespace(
        execute_model=lambda scheduler_output, intermediate_tensors=None: output,
        sample_tokens=lambda grammar_output: None,
    )
    worker.begin_evo2_rank_local_generation_evidence(
        phase="steady-0.wave-000",
        expected_envelope_sha256="a" * 64,
        expected_request_count=1,
        expected_max_new_tokens=1,
    )

    with pytest.raises(RuntimeError, match="chosen token"):
        worker.model_runner.execute_model(SimpleNamespace(), None)


def test_named_worker_extension_observes_compiled_dag_sample_tokens_fallback() -> None:
    output = _rank_local_output(("req-0",), ((65,),), ((0.0,),))
    worker = Evo2VllmWorkerExtension()
    worker.model_runner = SimpleNamespace(
        execute_model=lambda scheduler_output, intermediate_tensors=None: None,
        sample_tokens=lambda grammar_output: output,
    )
    worker.begin_evo2_rank_local_generation_evidence(
        phase="steady-0.wave-000",
        expected_envelope_sha256="a" * 64,
        expected_request_count=1,
        expected_max_new_tokens=1,
    )

    assert worker.model_runner.execute_model(SimpleNamespace(), None) is None
    assert worker.model_runner.sample_tokens(SimpleNamespace()) is output
    evidence = worker.snapshot_evo2_rank_local_generation_evidence()

    assert evidence["execution_call_count"] == 1
    assert evidence["generated_token_count"] == 1


def test_named_worker_extension_aborts_incomplete_epoch_and_retries_without_stale_rows() -> None:
    incomplete = _rank_local_output(("req-0",), ((65,),), ((-0.1,),))
    complete = _rank_local_output(("req-1",), ((66,),), ((-0.2,),))
    outputs = [incomplete, complete]
    worker = Evo2VllmWorkerExtension()
    worker.model_runner = SimpleNamespace(
        execute_model=lambda scheduler_output, intermediate_tensors=None: outputs.pop(0),
        sample_tokens=lambda grammar_output: None,
    )
    worker.begin_evo2_rank_local_generation_evidence(
        phase="steady-0.wave-000",
        expected_envelope_sha256="a" * 64,
        expected_request_count=2,
        expected_max_new_tokens=1,
    )
    worker.model_runner.execute_model(SimpleNamespace(), None)

    with pytest.raises(RuntimeError, match="request count is incomplete"):
        worker.snapshot_evo2_rank_local_generation_evidence()

    assert worker.abort_evo2_rank_local_generation_evidence() == {
        "tp_rank": 0,
        "aborted": True,
        "phase": "steady-0.wave-000",
    }
    worker.begin_evo2_rank_local_generation_evidence(
        phase="steady-0.wave-001",
        expected_envelope_sha256="b" * 64,
        expected_request_count=1,
        expected_max_new_tokens=1,
    )
    worker.model_runner.execute_model(SimpleNamespace(), None)
    evidence = worker.snapshot_evo2_rank_local_generation_evidence()

    assert evidence["phase"] == "steady-0.wave-001"
    assert evidence["request_order"] == ["req-1"]
    assert outputs == []


def test_named_worker_extension_reset_aborts_active_rank_local_epoch(monkeypatch) -> None:
    delegated = []
    monkeypatch.setattr(
        runner,
        "reset_vllm_worker_proof_state",
        lambda owner, reset_prefix_sources: delegated.append(
            (owner, reset_prefix_sources)
        )
        or {"reset": True},
    )
    worker = Evo2VllmWorkerExtension()
    worker.model_runner = SimpleNamespace(
        execute_model=lambda scheduler_output, intermediate_tensors=None: None,
        sample_tokens=lambda grammar_output: None,
    )
    worker.begin_evo2_rank_local_generation_evidence(
        phase="cold-generation.wave-000",
        expected_envelope_sha256="a" * 64,
        expected_request_count=1,
        expected_max_new_tokens=1,
    )

    assert worker.reset_evo2_proof_state(False) == {"reset": True}
    assert worker.abort_evo2_rank_local_generation_evidence() == {
        "tp_rank": 0,
        "aborted": False,
        "phase": None,
    }
    assert delegated == [(worker, False)]


def test_nemo_worker_extension_excludes_proof_controls_from_normal_actor() -> None:
    if not issubclass(Evo2NemoRlVllmWorkerExtension, VllmInternalWorkerExtension):
        raise AssertionError("normal Evo2 NeMo worker must retain the supported refit adapter")
    if issubclass(Evo2NemoRlVllmWorkerExtension, Evo2VllmWorkerExtension):
        raise AssertionError("proof-disabled actor must not load the Evo2 proof extension")


def test_normal_nemo_worker_import_does_not_load_proof_modules() -> None:
    code = """
import sys
from bionemo.evo2.vllm.nemo_worker import Evo2NemoRlVllmWorkerExtension

if Evo2NemoRlVllmWorkerExtension.__module__ != "bionemo.evo2.vllm.nemo_worker":
    raise RuntimeError("normal worker resolved from an unexpected module")
for module_name in (
    "bionemo.evo2.vllm.worker",
    "bionemo.evo2.vllm.nemo_proof_worker",
):
    if module_name in sys.modules:
        raise RuntimeError(f"normal worker imported proof-only module: {module_name}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"normal worker import isolation failed:\n{result.stdout}\n{result.stderr}")


def test_nemo_proof_worker_extension_composes_refit_and_evo2_proof_controls() -> None:
    if not issubclass(Evo2NemoRlProofVllmWorkerExtension, VllmInternalWorkerExtension):
        raise AssertionError("proof worker must retain the supported refit adapter")
    if not issubclass(Evo2NemoRlProofVllmWorkerExtension, Evo2VllmWorkerExtension):
        raise AssertionError("proof worker must expose the opt-in Evo2 proof controls")


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
    worker = object.__new__(Evo2NemoRlProofVllmWorkerExtension)
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
    worker._evo2_proof_enabled = True
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
    worker.reset_evo2_proof_phase("steady-0.wave-001")
    assert rpc_calls == [
        ("reset_evo2_proof_state", (True,)),
        ("snapshot_evo2_proof_state", ()),
        ("reset_evo2_proof_state", (False,)),
    ]


def test_nemo_generation_worker_does_not_attach_proof_recorder_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(VllmGenerationWorkerImpl, "post_init", lambda self: "base-post-init")
    worker = object.__new__(Evo2NemoRlGenerationWorkerImpl)
    worker.cfg = {"evo2_collect_proof": False}
    worker.llm = SimpleNamespace(llm_engine=SimpleNamespace(logger_manager=SimpleNamespace(stat_loggers=[])))

    assert worker.post_init() == "base-post-init"
    assert worker.llm.llm_engine.logger_manager.stat_loggers == []
    assert not hasattr(worker, "_evo2_cudagraph_recorder")
    with pytest.raises(RuntimeError, match="disabled"):
        worker.reset_evo2_proof_phase("steady-0")


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
