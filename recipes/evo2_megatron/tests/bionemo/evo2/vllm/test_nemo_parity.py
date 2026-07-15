# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

import bionemo.evo2.vllm.nemo_parity as nemo_parity
from bionemo.evo2.vllm.benchmark import WorkloadManifest
from bionemo.evo2.vllm.nemo_parity import (
    build_duplicate_prompt_manifest,
    build_parser,
    compare_full_vocab_evidence,
    run_production_refit,
)
from bionemo.evo2.vllm.refit import IndexedSafetensorsLayout, IpcChunkPlan
from bionemo.evo2.vllm.runner import PeakMemoryMonitor


DATA = Path(__file__).with_name("data") / "gdpo_mixed_96.json"


def test_tp2_refit_parity_cli_requires_reproducible_paths_and_knobs(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "/checkpoint",
            "--manifest",
            str(DATA),
            "--output",
            str(tmp_path / "proof.json"),
            "--log-path",
            str(tmp_path / "proof.log"),
            "--optimization-level",
            "2",
            "--performance-mode",
            "balanced",
        ]
    )

    assert args.checkpoint == Path("/checkpoint")
    assert args.refit_buffer_size_bytes == 512 * 1024 * 1024
    assert args.num_logprobs == 512
    assert args.max_new_tokens == 4


def test_duplicate_prompt_manifest_preserves_tokens_with_unique_ids() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 1).with_max_new_tokens(4)

    duplicate = build_duplicate_prompt_manifest(manifest, request_count=2, request_id_prefix="placement")

    assert [request.request_id for request in duplicate.requests] == ["placement-000", "placement-001"]
    assert [request.prompt_token_ids for request in duplicate.requests] == [
        manifest.requests[0].prompt_token_ids,
        manifest.requests[0].prompt_token_ids,
    ]
    assert duplicate.max_new_tokens == 4


def test_full_vocab_comparison_retains_all_steps_and_reports_worst_coordinate() -> None:
    reference = {
        "shape": [1, 2, 4],
        "coverage_counts": [[4, 4]],
        "chosen_token_oracle_passed": True,
        "logprobs": [[[-4.0, -3.0, -2.0, -1.0], [-1.0, -2.0, -3.0, -4.0]]],
    }
    candidate = {
        **reference,
        "logprobs": [[[-4.0, -3.0, -2.0, -0.98], [-1.0, -2.0, -3.0, -4.0]]],
    }

    comparison = compare_full_vocab_evidence(reference, candidate)

    assert comparison["shape"] == [1, 2, 4]
    assert comparison["max_abs_logprob_error"] == pytest.approx(0.02)
    assert comparison["worst_coordinate"] == {"request_index": 0, "step": 0, "token_id": 3}
    assert comparison["top1_identity"] is True


def test_logprob_evidence_summary_records_coverage_support_and_chosen_values() -> None:
    evidence = {
        "shape": [2, 4, 512],
        "coverage_counts": [[512] * 4] * 2,
        "finite_support_counts": [[4] * 4] * 2,
        "negative_infinity_counts": [[508] * 4] * 2,
        "expected_finite_support": 4,
        "chosen_token_oracle_passed": True,
        "chosen_token_in_finite_support": True,
        "chosen_token_logprobs": [[-1.0, -2.0, -3.0, -4.0]] * 2,
    }

    summary = nemo_parity._logprob_evidence_summary(evidence)

    assert summary == {
        "shape": [2, 4, 512],
        "returned_candidate_ids_per_step": 512,
        "finite_support_size_per_step": 4,
        "negative_infinity_exclusions_per_step": 508,
        "chosen_token_in_finite_support": True,
        "chosen_token_oracle_passed": True,
        "chosen_token_logprobs": [[-1.0, -2.0, -3.0, -4.0]] * 2,
    }


def test_final_compilation_gate_rejects_drift_in_last_stochastic_phase() -> None:
    initialized = {
        "num_models_seen": 1,
        "num_backend_compilations": 12,
        "num_inductor_compiles": 12,
        "num_eager_compiles": 0,
        "num_gpu_runner_capture_triggers": 1,
        "num_cudagraph_captured": 24,
        "stock_torch_compile_count": 0,
    }
    initialized_proof = {
        "worker_proof": [
            {"compilation": initialized},
            {"compilation": initialized},
        ]
    }
    stable_final = {
        "engines": [
            {
                "worker_proof": [
                    {"compilation": initialized},
                    {"compilation": initialized},
                ]
            }
        ]
    }
    final_phase = SimpleNamespace(phase="stochastic-2", wave_proofs=(stable_final,))

    summary = nemo_parity.validate_final_compilation_stability(initialized_proof, final_phase)

    assert summary == {"passed": True, "final_phase": "stochastic-2", "tp_worker_count": 2}

    drifted = {
        **stable_final,
        "engines": [
            {
                "worker_proof": [
                    {"compilation": initialized},
                    {"compilation": {**initialized, "num_inductor_compiles": 13}},
                ]
            }
        ],
    }
    with pytest.raises(AssertionError, match="recompile"):
        nemo_parity.validate_final_compilation_stability(
            initialized_proof,
            SimpleNamespace(phase="stochastic-2", wave_proofs=(drifted,)),
        )


def test_production_refit_runs_both_uuid_streams_against_real_generation_contract() -> None:
    layout = IndexedSafetensorsLayout(root=Path("/checkpoint"), tensors=())
    plan = IpcChunkPlan(buffer_size_bytes=2_048, per_buffer_capacity_bytes=1_024, chunks=())
    events = []

    class FakeWorkerGroup:
        def run_all_workers_single_data(self, method_name, *, run_rank_0_only_axes, phase):
            events.append(("rpc", method_name, phase, tuple(run_rank_0_only_axes)))
            return [{"phase": phase}]

    class FakeGeneration:
        device_uuids: ClassVar = [["GPU-a", "GPU-b"]]
        worker_group = FakeWorkerGroup()

        def prepare_refit_info(self, state_dict_info):
            events.append(("prepare", state_dict_info))

        def update_weights_via_ipc_zmq(self):
            events.append(("consumers",))
            return ["consumer-future"]

    def fake_stream(layout, **kwargs):
        events.append(("producer-ready", kwargs["expected_device_uuid"]))
        kwargs["ready_event"].set()
        assert kwargs["start_event"].wait(1.0)
        events.append(("producer-stream", kwargs["expected_device_uuid"]))
        return {
            "device_index": kwargs["device_index"],
            "device_uuid": kwargs["expected_device_uuid"],
            "stream_s": 0.5,
        }

    times = iter((10.0, 10.25, 11.0, 13.0))
    result = run_production_refit(
        generation=FakeGeneration(),
        layout=layout,
        plan=plan,
        phase="refit-1",
        device_index_by_uuid={"GPU-a": 0, "GPU-b": 1},
        ray_get=lambda futures: [True for _ in futures],
        memory_monitor_factory=lambda: PeakMemoryMonitor(lambda: (1_000, 2_000)),
        stream_fn=fake_stream,
        clock=lambda: next(times),
    )

    assert result["phase"] == "refit-1"
    assert result["device_uuids"] == ["GPU-a", "GPU-b"]
    assert result["consumer_results"] == [True]
    assert [item["device_uuid"] for item in result["producer_results"]] == ["GPU-a", "GPU-b"]
    assert result["timing"]["prepare_refit_s"] == 0.25
    assert result["timing"]["transfer_wall_s"] == 2.0
    assert result["peak_device_memory_bytes"] == [1_000, 2_000]
    assert events.index(("consumers",)) < events.index(("producer-stream", "GPU-a"))
