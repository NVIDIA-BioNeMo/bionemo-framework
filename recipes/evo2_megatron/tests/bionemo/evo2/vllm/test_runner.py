# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import copy
import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import bionemo.evo2.vllm.runner as runner
from bionemo.evo2.vllm.benchmark import GenerationRecord, WorkloadManifest, records_from_vllm_outputs
from bionemo.evo2.vllm.runner import (
    CUDAGraphProofRecorder,
    PeakMemoryMonitor,
    build_request_sampling_params,
    prepare_workload,
    request_seed,
    run_generation_phase,
    validate_full_decode_proof,
)


DATA = __import__("pathlib").Path(__file__).with_name("data") / "gdpo_mixed_96.json"


def _scheduler_stats(
    *,
    unpadded: int,
    padded: int,
    mode: str,
):
    return SimpleNamespace(
        cudagraph_stats=SimpleNamespace(
            num_unpadded_tokens=unpadded,
            num_padded_tokens=padded,
            num_paddings=padded - unpadded,
            runtime_mode=mode,
        ),
        prefix_cache_stats=SimpleNamespace(
            preempted_requests=0,
            preempted_queries=0,
            preempted_hits=0,
        ),
        num_running_reqs=unpadded,
        num_waiting_reqs=0,
        num_skipped_waiting_reqs=0,
    )


def _iteration_stats(
    *,
    preempted: int = 0,
    computed: int = 0,
    cached_tokens: int = 0,
    total: int = 0,
):
    return SimpleNamespace(
        num_preempted_reqs=preempted,
        prompt_token_stats=SimpleNamespace(computed=computed, cached_tokens=cached_tokens, total=total),
    )


def test_cudagraph_recorder_persists_phase_and_runtime_mode_without_log_resets() -> None:
    recorder = CUDAGraphProofRecorder()
    recorder.start_phase("steady-0")
    recorder.record(
        _scheduler_stats(unpadded=768, padded=768, mode="CUDAGraphMode.PIECEWISE"),
        None,
        engine_idx=0,
    )
    recorder.record(
        _scheduler_stats(unpadded=96, padded=96, mode="CUDAGraphMode.FULL"),
        None,
        engine_idx=0,
    )
    recorder.record(None, None, engine_idx=0)

    assert recorder.observations == [
        {
            "phase": "steady-0",
            "engine_index": 0,
            "num_unpadded_tokens": 768,
            "num_padded_tokens": 768,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.PIECEWISE",
        },
        {
            "phase": "steady-0",
            "engine_index": 0,
            "num_unpadded_tokens": 96,
            "num_padded_tokens": 96,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        },
    ]


def test_scheduler_capacity_proof_retains_and_rejects_preemption_recompute() -> None:
    recorder = CUDAGraphProofRecorder()
    recorder.start_phase("capacity.wave-000")
    recorder.record(
        SimpleNamespace(
            cudagraph_stats=None,
            prefix_cache_stats=SimpleNamespace(
                preempted_requests=1,
                preempted_queries=100,
                preempted_hits=20,
            ),
            num_running_reqs=20,
            num_waiting_reqs=76,
            num_skipped_waiting_reqs=0,
        ),
        SimpleNamespace(
            num_preempted_reqs=1,
            prompt_token_stats=SimpleNamespace(
                computed=80,
                cached_tokens=20,
                total=100,
            ),
        ),
        engine_idx=0,
    )

    proof = runner.scheduler_capacity_proof_summary(
        recorder.scheduler_observations,
        phase="capacity.wave-000",
        global_wave_size=20,
        max_num_seqs=20,
    )

    assert proof["preemption_events"] == 1
    assert proof["recompute_events"] == 1
    assert proof["preempted_prompt_recomputed_tokens"] == 80
    assert proof["prompt_tokens_computed"] == 80
    assert proof["batch_fit_without_preemption"] is False
    with pytest.raises(AssertionError, match="preempt"):
        runner.validate_scheduler_capacity_proof(proof)


def test_scheduler_capacity_proof_rejects_missing_iteration_telemetry() -> None:
    proof = runner.scheduler_capacity_proof_summary(
        [],
        phase="capacity.wave-000",
        global_wave_size=20,
        max_num_seqs=20,
    )

    with pytest.raises(AssertionError, match="scheduler telemetry"):
        runner.validate_scheduler_capacity_proof(proof)


def test_profile_from_cli_preserves_physical_wave_and_per_engine_ceiling(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA)
    args = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            "/checkpoint",
            "--manifest",
            str(DATA),
            "--topology",
            "tp2",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.92",
            "--global-wave-size",
            "20",
            "--max-num-seqs",
            "20",
            "--output",
            str(tmp_path / "proof.json"),
        ]
    )

    profile = runner.profile_from_args(args, manifest)

    assert profile.global_wave_size == 20
    assert profile.per_engine_batch_size == 20
    assert profile.resolved_max_num_seqs == 20
    assert profile.gdpo_waves_to_96 == 5


def test_speed_lane_rejects_minimal_self_attested_proof_artifact(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA)
    proof_path = tmp_path / "proof.json"
    common = [
        "--backend",
        "vllm",
        "--checkpoint",
        "/checkpoint",
        "--manifest",
        str(DATA),
        "--topology",
        "tp2",
        "--max-num-batched-tokens",
        "32768",
        "--gpu-memory-utilization",
        "0.95",
        "--optimization-level",
        "3",
        "--performance-mode",
        "throughput",
    ]
    proof_args = runner.build_parser().parse_args([*common, "--proof", "--output", str(proof_path)])
    speed_args = runner.build_parser().parse_args(
        [
            *common,
            "--linked-proof-artifact",
            str(proof_path),
            "--output",
            str(tmp_path / "speed.json"),
        ]
    )

    assert runner.benchmark_mode_from_args(proof_args) == "proof"
    assert runner.benchmark_mode_from_args(speed_args) == "speed"
    proof_contract = runner.build_benchmark_contract(
        proof_args,
        manifest,
        runner.profile_from_args(proof_args, manifest),
    )
    speed_contract = runner.build_benchmark_contract(
        speed_args,
        manifest,
        runner.profile_from_args(speed_args, manifest),
    )
    assert proof_contract == speed_contract
    assert "proof" not in proof_contract["profile"]

    proof_path.write_text(
        json.dumps(
            {
                "benchmark_mode": "proof",
                "benchmark_contract": proof_contract,
                "benchmark_contract_sha256": runner.benchmark_contract_sha256(proof_contract),
                "proof_status": {"passed": True},
                "invocation": {"exit_status": 0},
            }
        )
    )

    with pytest.raises(AssertionError, match="phase evidence"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=speed_contract,
        )


def test_benchmark_mode_rejects_unlinked_speed_or_doubly_attested_proof(tmp_path) -> None:
    common = [
        "--backend",
        "vllm",
        "--checkpoint",
        "/checkpoint",
        "--manifest",
        str(DATA),
        "--topology",
        "tp2",
        "--max-num-batched-tokens",
        "16384",
        "--gpu-memory-utilization",
        "0.92",
        "--output",
        str(tmp_path / "result.json"),
    ]

    with pytest.raises(ValueError, match="linked proof artifact"):
        runner.benchmark_mode_from_args(runner.build_parser().parse_args(common))
    with pytest.raises(ValueError, match="cannot link"):
        runner.benchmark_mode_from_args(
            runner.build_parser().parse_args(
                [*common, "--proof", "--linked-proof-artifact", str(tmp_path / "proof.json")]
            )
        )


def test_benchmark_instrumentation_contract_distinguishes_proof_and_speed() -> None:
    proof = runner.benchmark_instrumentation_contract("proof")
    speed = runner.benchmark_instrumentation_contract("speed")

    assert proof == {
        "scheduler_callbacks_during_generation": True,
        "worker_proof_rpcs": True,
        "prefix_clone_instrumentation": True,
        "peak_memory_polling_during_generation": True,
        "post_generation_exact_output_validation": True,
    }
    assert speed == {
        "scheduler_callbacks_during_generation": False,
        "worker_proof_rpcs": False,
        "prefix_clone_instrumentation": False,
        "peak_memory_polling_during_generation": False,
        "post_generation_exact_output_validation": True,
    }


def test_full_decode_proof_requires_full_unpadded_replay_and_rejects_fallback() -> None:
    observations = [
        {
            "phase": "steady-0",
            "engine_index": 0,
            "num_unpadded_tokens": 768,
            "num_padded_tokens": 768,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.NONE",
        },
        *[
            {
                "phase": "steady-0",
                "engine_index": 0,
                "num_unpadded_tokens": 96,
                "num_padded_tokens": 96,
                "num_paddings": 0,
                "runtime_mode": "CUDAGraphMode.FULL",
            }
            for _ in range(2)
        ],
    ]

    validate_full_decode_proof(
        observations,
        phase="steady-0",
        batch_size=96,
        max_new_tokens=3,
    )

    with pytest.raises(AssertionError, match="NONE"):
        validate_full_decode_proof(
            [
                {
                    **observations[-1],
                    "runtime_mode": "CUDAGraphMode.NONE",
                }
            ],
            phase="steady-0",
            batch_size=96,
            max_new_tokens=2,
        )
    with pytest.raises(AssertionError, match="FULL"):
        validate_full_decode_proof(
            observations[:1],
            phase="steady-0",
            batch_size=96,
            max_new_tokens=3,
        )
    with pytest.raises(AssertionError, match="padding"):
        validate_full_decode_proof(
            [
                {
                    **observations[-1],
                    "num_padded_tokens": 128,
                    "num_paddings": 32,
                }
            ],
            phase="steady-0",
            batch_size=96,
            max_new_tokens=2,
        )


def test_full_decode_proof_allows_staggered_admission_with_full_global_batch() -> None:
    observations = [
        {
            "phase": "cold-generation",
            "engine_index": 0,
            "num_unpadded_tokens": 4,
            "num_padded_tokens": 4,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.PIECEWISE",
        },
        {
            "phase": "cold-generation",
            "engine_index": 0,
            "num_unpadded_tokens": 6,
            "num_padded_tokens": 8,
            "num_paddings": 2,
            "runtime_mode": "CUDAGraphMode.PIECEWISE",
        },
        {
            "phase": "cold-generation",
            "engine_index": 0,
            "num_unpadded_tokens": 2,
            "num_padded_tokens": 2,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        },
        {
            "phase": "cold-generation",
            "engine_index": 0,
            "num_unpadded_tokens": 1,
            "num_padded_tokens": 1,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        },
    ]

    validate_full_decode_proof(
        observations,
        phase="cold-generation",
        batch_size=2,
        max_new_tokens=3,
    )

    steady_observations = [{**item, "phase": "steady-0"} for item in observations]
    validate_full_decode_proof(
        steady_observations,
        phase="steady-0",
        batch_size=2,
        max_new_tokens=3,
    )

    with pytest.raises(AssertionError, match="global batch"):
        validate_full_decode_proof(
            [item for item in steady_observations if item["num_unpadded_tokens"] != 2],
            phase="steady-0",
            batch_size=2,
            max_new_tokens=3,
        )


def test_long_full_decode_proof_rejects_missing_work_and_serialization() -> None:
    def full_observation(unpadded: int) -> dict[str, object]:
        return {
            "phase": "steady-0",
            "engine_index": 0,
            "num_unpadded_tokens": unpadded,
            "num_padded_tokens": unpadded,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        }

    batched = [full_observation(10) for _ in range(98)]
    validate_full_decode_proof(
        batched,
        phase="steady-0",
        batch_size=10,
        max_new_tokens=100,
    )

    with pytest.raises(AssertionError, match="coverage"):
        validate_full_decode_proof(
            batched[:40],
            phase="steady-0",
            batch_size=10,
            max_new_tokens=100,
        )

    with pytest.raises(AssertionError, match="occupancy"):
        validate_full_decode_proof(
            [full_observation(10), *[full_observation(1) for _ in range(980)]],
            phase="steady-0",
            batch_size=10,
            max_new_tokens=100,
        )


def test_full_decode_proof_summary_persists_long_run_coverage_and_occupancy() -> None:
    observations = [
        {
            "phase": "steady-0",
            "engine_index": 0,
            "num_unpadded_tokens": 10,
            "num_padded_tokens": 10,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        }
        for _ in range(98)
    ]

    summary = runner.full_decode_proof_summary(
        observations,
        phase="steady-0",
        batch_size=10,
        max_new_tokens=100,
    )

    assert summary["expected_decode_tokens"] == 990
    assert summary["full_decode_tokens"] == 980
    assert summary["coverage_fraction"] == pytest.approx(980 / 990)
    assert summary["full_dispatch_count"] == 98
    assert summary["maximum_full_batch"] == 10
    assert summary["average_full_batch_occupancy"] == 10
    assert summary["occupancy_fraction"] == 1.0
    assert summary["global_batch_hit"] is True
    assert summary["full_decode_unpadded"] is True
    assert summary["long_run_gates_applied"] is True
    assert summary["passed"] is True


def test_peak_memory_monitor_records_each_device_maximum() -> None:
    samples = iter(((100, 200), (150, 180), (125, 240)))
    monitor = PeakMemoryMonitor(lambda: next(samples))

    monitor.sample_now()
    monitor.sample_now()
    monitor.sample_now()

    assert monitor.peak_device_memory_bytes == (150, 240)


def test_peak_memory_monitor_rejects_device_count_changes() -> None:
    samples = iter(((100, 200), (150,)))
    monitor = PeakMemoryMonitor(lambda: next(samples))
    monitor.sample_now()

    with pytest.raises(RuntimeError, match="device count"):
        monitor.sample_now()


def test_request_seeds_encode_call_and_dp_stream_coordinates() -> None:
    tp2_call0 = [
        request_seed(
            42,
            call_index=0,
            dp_rank=0,
            dp_size=1,
            request_index_in_stream=index,
        )
        for index in range(96)
    ]
    tp2_call1 = [
        request_seed(
            42,
            call_index=1,
            dp_rank=0,
            dp_size=1,
            request_index_in_stream=index,
        )
        for index in range(96)
    ]
    dp2_call0_rank0 = [
        request_seed(
            42,
            call_index=0,
            dp_rank=0,
            dp_size=2,
            request_index_in_stream=index,
        )
        for index in range(48)
    ]
    dp2_call0_rank1 = [
        request_seed(
            42,
            call_index=0,
            dp_rank=1,
            dp_size=2,
            request_index_in_stream=index,
        )
        for index in range(48)
    ]

    assert tp2_call0 == list(range(42, 138))
    assert tp2_call1 == list(range(1_000_045, 1_000_141))
    assert dp2_call0_rank0 == list(range(42, 90))
    assert dp2_call0_rank1 == list(range(1_000_045, 1_000_093))
    assert set(tp2_call0).isdisjoint(tp2_call1)
    assert set(dp2_call0_rank0).isdisjoint(dp2_call0_rank1)


def test_request_sampling_params_consume_persisted_stream_seeds() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    records = runner.build_request_execution_records(
        manifest,
        global_request_offset=48,
        dp_rank=1,
        dp_size=2,
        call_index=7,
    )

    params = build_request_sampling_params(
        manifest,
        sampling_params_factory=SimpleNamespace,
        execution_records=records,
    )

    assert [param.seed for param in params] == [record.seed for record in records]
    assert [param.seed for param in params] == [15_000_087, 15_000_088]
    assert all(param.max_tokens == 3 and param.min_tokens == 3 for param in params)
    assert all(param.detokenize is False and param.logprobs == 0 for param in params)


def test_request_execution_records_persist_round_rank_call_and_global_seed() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)

    records = runner.build_request_execution_records(
        manifest,
        global_request_offset=48,
        dp_rank=1,
        dp_size=2,
        call_index=7,
    )

    assert [record.to_dict() for record in records] == [
        {
            "execution_uid": "round=7/call=7/global=48/dp=1/request=gdpo-000",
            "request_id": "gdpo-000",
            "global_request_index": 48,
            "generation_round": 7,
            "dp_rank": 1,
            "call_index": 7,
            "seed": 15_000_087,
        },
        {
            "execution_uid": "round=7/call=7/global=49/dp=1/request=gdpo-001",
            "request_id": "gdpo-001",
            "global_request_index": 49,
            "generation_round": 7,
            "dp_rank": 1,
            "call_index": 7,
            "seed": 15_000_088,
        },
    ]


def test_full_output_artifact_round_trips_every_token_logprob_and_seed(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=48,
        dp_rank=1,
        dp_size=2,
        call_index=9,
    )
    output_path = tmp_path / "steady-0.outputs.jsonl.gz"

    metadata = runner.write_full_output_artifact(
        output_path,
        manifest=manifest,
        outputs=_fake_outputs(manifest),
        execution_records=execution_records,
    )

    with gzip.open(output_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert rows[0]["request_id"] == "gdpo-000"
    assert rows[0]["generation_round"] == 9
    assert rows[0]["dp_rank"] == 1
    assert rows[0]["call_index"] == 9
    assert rows[0]["global_request_index"] == 48
    assert rows[0]["seed"] == execution_records[0].seed
    assert rows[0]["execution_uid"] == "round=9/call=9/global=48/dp=1/request=gdpo-000"
    assert rows[0]["requested_max_tokens"] == 3
    assert rows[0]["requested_prompt_tokens"] == len(manifest.requests[0].prompt_token_ids)
    assert rows[0]["requested_new_tokens"] == 3
    assert rows[0]["requested_total_tokens"] == len(manifest.requests[0].prompt_token_ids) + 3
    assert rows[0]["observed_prompt_tokens"] == len(manifest.requests[0].prompt_token_ids)
    assert rows[0]["observed_new_tokens"] == 3
    assert rows[0]["observed_total_tokens"] == len(manifest.requests[0].prompt_token_ids) + 3
    assert rows[0]["finish_reason"] == "length"
    assert rows[0]["stop_reason"] is None
    assert rows[0]["stopped_on_eos"] is False
    assert rows[0]["output_token_ids"] == [65, 67, 71]
    assert rows[0]["chosen_token_logprobs"] == pytest.approx([-0.1, -0.1, -0.1])
    assert rows[1]["output_token_ids"] == [66, 68, 72]
    assert metadata == {
        "schema_version": 2,
        "format": "jsonl",
        "compression": "gzip",
        "path": str(output_path.resolve()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "size_bytes": output_path.stat().st_size,
        "request_count": 2,
        "generated_token_count": 6,
    }


def test_backend_neutral_full_output_artifact_accepts_generation_records(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=48,
        dp_rank=1,
        dp_size=2,
        call_index=9,
    )
    records = records_from_vllm_outputs(manifest, _fake_outputs(manifest))
    output_path = tmp_path / "nemo-steady-0.outputs.jsonl.gz"

    metadata = runner.write_full_generation_records_artifact(
        output_path,
        records=records,
        execution_records=execution_records,
    )

    with gzip.open(output_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert [row["request_id"] for row in rows] == ["gdpo-000", "gdpo-001"]
    assert rows[0]["prompt_token_ids"] == list(manifest.requests[0].prompt_token_ids)
    assert rows[0]["output_token_ids"] == [65, 67, 71]
    assert rows[0]["chosen_token_logprobs"] == pytest.approx([-0.1, -0.1, -0.1])
    assert metadata["generated_token_count"] == 6


def test_phase_output_artifact_paths_are_phase_and_replica_specific(tmp_path) -> None:
    root = tmp_path / "benchmark.json"

    assert runner.phase_output_artifact_path(root, phase="steady-0") == (
        tmp_path / "benchmark.json.steady-0.outputs.jsonl.gz"
    )
    assert runner.phase_output_artifact_path(root, phase="steady-0", dp_rank=1) == (
        tmp_path / "benchmark.json.steady-0.dp1.outputs.jsonl.gz"
    )


def test_same_stem_different_suffix_namespaces_cannot_alias_or_run_simultaneously(tmp_path) -> None:
    json_output = tmp_path / "proof.json"
    yaml_output = tmp_path / "proof.yaml"

    marker = runner.reserve_output_namespace(json_output)
    with pytest.raises(FileExistsError, match="proof.inprogress"):
        runner.reserve_output_namespace(yaml_output)

    assert runner.phase_output_artifact_path(json_output, phase="steady-0") != (
        runner.phase_output_artifact_path(yaml_output, phase="steady-0")
    )
    runner.complete_output_namespace(marker, output_path=json_output, require_final_artifact=False)


def test_output_namespace_reservation_refuses_stale_final_sidecar_or_active_run(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    unrelated = tmp_path / "unrelated.steady-0.outputs.jsonl.gz"
    unrelated.write_bytes(b"unrelated")

    marker = runner.reserve_output_namespace(output)

    assert marker.is_file()
    with pytest.raises(FileExistsError, match="namespace"):
        runner.reserve_output_namespace(output)

    runner.complete_output_namespace(marker, output_path=output, require_final_artifact=False)
    output.write_text("stale success\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="benchmark.json"):
        runner.reserve_output_namespace(output)

    output.unlink()
    sidecar = runner.phase_output_artifact_path(output, phase="steady-0")
    sidecar.write_bytes(b"stale sidecar")
    with pytest.raises(FileExistsError, match="steady-0"):
        runner.reserve_output_namespace(output)
    assert unrelated.read_bytes() == b"unrelated"


def test_json_artifact_writer_refuses_to_overwrite_prior_success(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    runner.write_json_artifact(output, {"run": 1})

    with pytest.raises(FileExistsError, match="benchmark.json"):
        runner.write_json_artifact(output, {"run": 2})

    assert json.loads(output.read_text()) == {"run": 1}


def test_context_preflight_only_cli_writes_proof_without_launching_generation(
    tmp_path,
    monkeypatch,
) -> None:
    from bionemo.evo2.vllm.config import Evo2Config

    checkpoint = tmp_path / "checkpoint"
    Evo2Config(max_position_embeddings=10_240).save_pretrained(checkpoint)
    output = tmp_path / "preflight.json"
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    monkeypatch.setattr(
        runner,
        "run_tp2_benchmark",
        lambda *args, **kwargs: pytest.fail("preflight-only mode launched generation"),
    )

    exit_status = runner.main(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(DATA),
            "--topology",
            "tp2",
            "--max-model-len",
            "50000",
            "--max-num-batched-tokens",
            "32768",
            "--gpu-memory-utilization",
            "0.92",
            "--context-preflight-only",
            "--output",
            str(output),
        ]
    )

    artifact = json.loads(output.read_text())
    assert exit_status == 0
    assert artifact["task"] == "evo2-vllm-context-length-preflight"
    assert artifact["profile"]["max_model_len"] == 50_000
    assert artifact["context_length_preflight"]["checkpoint_declared_max_position_embeddings"] == 10_240
    assert artifact["context_length_preflight"]["requested_max_model_len"] == 50_000
    assert artifact["context_length_preflight"]["resolved_max_model_len"] == 50_000
    assert artifact["context_length_preflight"]["workload_max_total_tokens"] == 6_001
    assert artifact["context_length_preflight"]["workload_fits_resolved_max_model_len"] is True
    assert artifact["context_length_preflight"]["workload_headroom_tokens"] == 43_999
    assert not output.with_name("preflight.inprogress").exists()


def test_context_preflight_only_cli_rejects_model_len_shorter_than_manifest(
    tmp_path,
) -> None:
    from bionemo.evo2.vllm.config import Evo2Config

    checkpoint = tmp_path / "checkpoint"
    Evo2Config(max_position_embeddings=10_240).save_pretrained(checkpoint)
    output = tmp_path / "undersized.json"

    with pytest.raises(ValueError, match="max_model_len=16.*workload max_total_tokens=6001"):
        runner.main(
            [
                "--backend",
                "vllm",
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(DATA),
                "--topology",
                "tp2",
                "--max-model-len",
                "16",
                "--max-num-batched-tokens",
                "32768",
                "--gpu-memory-utilization",
                "0.92",
                "--context-preflight-only",
                "--output",
                str(output),
            ]
        )

    assert not output.exists()
    assert output.with_name("undersized.inprogress").is_file()


def test_checkpoint_provenance_hashes_actual_indexed_weight_shards(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    index = checkpoint / "model.safetensors.index.json"
    manifest_path = checkpoint / "manifest.json"
    shard_a = checkpoint / "model-00001-of-00002.safetensors"
    shard_b = checkpoint / "model-00002-of-00002.safetensors"
    tokenizer = checkpoint / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir()
    config.write_text("{}\n")
    shard_a.write_bytes(b"first-real-shard")
    shard_b.write_bytes(b"second-real-shard")
    tokenizer.write_text('{"vocab_size": 512}\n')
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard_a.stat().st_size + shard_b.stat().st_size},
                "weight_map": {"a": shard_a.name, "b": shard_b.name},
            }
        )
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            }
        )
    )

    first = runner.checkpoint_provenance(checkpoint)

    assert first["checkpoint_sha256"]
    assert first["indexed_weight_bytes"] == shard_a.stat().st_size + shard_b.stat().st_size
    assert [item["path"] for item in first["indexed_weight_shards"]] == [shard_a.name, shard_b.name]
    assert first["indexed_weight_shards"][0]["sha256"] == hashlib.sha256(shard_a.read_bytes()).hexdigest()
    assert first["manifest_digest_verification"] == {"config": True, "index": True}
    assert "tokenizer/tokenizer.json" in {item["path"] for item in first["files"]}

    shard_b.write_bytes(b"changed-second-real-shard")
    second = runner.checkpoint_provenance(checkpoint)
    assert second["checkpoint_sha256"] != first["checkpoint_sha256"]


def test_source_provenance_records_head_dirty_diff_and_actual_source_tree(tmp_path) -> None:
    source = tmp_path / "src" / "model.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    pycache = source.parent / "__pycache__"
    pycache.mkdir()
    (pycache / "model.cpython-313.pyc").write_bytes(b"transient-bytecode")
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "src/model.py", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Evo2 Test",
            "-c",
            "user.email=evo2@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )

    clean = runner.source_provenance(repository=tmp_path, source_roots=(source.parent,))
    assert len(clean["git_head"]) == 40
    assert clean["git_dirty"] is False
    assert clean["source_file_count"] == 1
    runner.source_provenance(
        repository=tmp_path,
        source_roots=(source.parent,),
        require_clean=True,
    )

    source.write_text("VALUE = 2\n")
    dirty = runner.source_provenance(repository=tmp_path, source_roots=(source.parent,))
    assert dirty["git_head"] == clean["git_head"]
    assert dirty["git_dirty"] is True
    assert dirty["dirty_fingerprint_sha256"] != clean["dirty_fingerprint_sha256"]
    assert dirty["source_tree_sha256"] != clean["source_tree_sha256"]
    with pytest.raises(RuntimeError, match="dirty source repository"):
        runner.source_provenance(
            repository=tmp_path,
            source_roots=(source.parent,),
            require_clean=True,
        )


def test_package_installation_provenance_hashes_source_binary_and_metadata(tmp_path) -> None:
    package_root = tmp_path / "site-packages" / "vllm"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("__version__ = '0.20.0'\n")
    (package_root / "_C.abi3.so").write_bytes(b"compiled-v1")
    pycache = package_root / "__pycache__"
    pycache.mkdir()
    (pycache / "ignored.pyc").write_bytes(b"transient")
    metadata = tmp_path / "site-packages" / "vllm-0.20.0.dist-info" / "RECORD"
    metadata.parent.mkdir()
    metadata.write_text("vllm/__init__.py,,\n")

    first = runner.package_installation_provenance(
        package_root,
        distribution_name="vllm",
        distribution_version="0.20.0",
        metadata_paths=(metadata,),
        require_binary=True,
    )

    assert first["source_file_count"] == 1
    assert first["binary_file_count"] == 1
    assert first["metadata_file_count"] == 1
    assert first["package_file_count"] == 2
    assert first["source_files"][0]["path"] == "__init__.py"
    assert first["binary_files"][0]["path"] == "_C.abi3.so"

    (package_root / "_C.abi3.so").write_bytes(b"compiled-v2")
    second = runner.package_installation_provenance(
        package_root,
        distribution_name="vllm",
        distribution_version="0.20.0",
        metadata_paths=(metadata,),
        require_binary=True,
    )
    assert second["installation_sha256"] != first["installation_sha256"]


def test_package_installation_provenance_rejects_missing_binary(tmp_path) -> None:
    package_root = tmp_path / "vllm"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("\n")

    with pytest.raises(RuntimeError, match="compiled binary"):
        runner.package_installation_provenance(
            package_root,
            distribution_name="vllm",
            distribution_version="0.20.0",
            require_binary=True,
        )


def test_gpu_hardware_provenance_and_memory_headroom_are_exact(monkeypatch) -> None:
    gib = 1024**3

    class FakeNvml:
        initialized = False

        @classmethod
        def nvmlInit(cls):  # noqa: N802
            cls.initialized = True

        @staticmethod
        def nvmlSystemGetDriverVersion():  # noqa: N802
            return b"570.86.15"

        @staticmethod
        def nvmlDeviceGetCount():  # noqa: N802
            return 2

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):  # noqa: N802
            return index

        @staticmethod
        def nvmlDeviceGetUUID(handle):  # noqa: N802
            return f"GPU-uuid-{handle}".encode()

        @staticmethod
        def nvmlDeviceGetName(handle):  # noqa: N802
            return f"NVIDIA H100 {handle}".encode()

        @staticmethod
        def nvmlDeviceGetPciInfo(handle):  # noqa: N802
            return SimpleNamespace(busId=f"00000000:0{handle + 1}:00.0".encode())

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):  # noqa: N802
            total = (80 + handle) * gib
            return SimpleNamespace(total=total, used=3 * gib, free=total - 3 * gib)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    hardware = runner.gpu_hardware_provenance(
        nvml_module=FakeNvml,
        expected_device_count=2,
    )

    assert FakeNvml.initialized is True
    assert hardware["driver_version"] == "570.86.15"
    assert hardware["cuda_visible_devices"] == "0,1"
    assert [device["uuid"] for device in hardware["devices"]] == ["GPU-uuid-0", "GPU-uuid-1"]
    assert [device["pci_bus_id"] for device in hardware["devices"]] == [
        "00000000:01:00.0",
        "00000000:02:00.0",
    ]
    assert [device["total_memory_bytes"] for device in hardware["devices"]] == [80 * gib, 81 * gib]

    headroom = runner.gpu_memory_headroom_evidence(
        hardware,
        peak_device_memory_bytes=(77 * gib, 78 * gib),
    )
    assert headroom["required_headroom_bytes"] == 2 * gib
    assert [device["headroom_bytes"] for device in headroom["devices"]] == [3 * gib, 3 * gib]
    assert headroom["passed"] is True

    attestation = runner.runtime_attestation_contract(
        checkpoint={"checkpoint_sha256": "checkpoint"},
        sources={
            "bionemo": {
                "git_dirty": False,
                "git_head": "head",
                "source_tree_sha256": "tree",
            }
        },
        vllm_installation={
            "distribution_version": "0.20.0",
            "installation_sha256": "installation",
        },
        gpu_hardware=hardware,
    )
    assert attestation["gpu"]["cuda_visible_devices"] == "0,1"
    assert [device["pci_bus_id"] for device in attestation["gpu"]["devices"]] == [
        "00000000:01:00.0",
        "00000000:02:00.0",
    ]


def test_worker_gpu_identity_resolves_logical_device_to_physical_uuid_and_pci(monkeypatch) -> None:
    class FakeNvml:
        @staticmethod
        def nvmlInit():  # noqa: N802
            return None

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):  # noqa: N802
            return index

        @staticmethod
        def nvmlDeviceGetHandleByUUID(uuid):  # noqa: N802
            return int(str(uuid).rsplit("-", 1)[1])

        @staticmethod
        def nvmlDeviceGetUUID(handle):  # noqa: N802
            return f"GPU-uuid-{handle}".encode()

        @staticmethod
        def nvmlDeviceGetName(handle):  # noqa: N802
            return f"NVIDIA H100 {handle}".encode()

        @staticmethod
        def nvmlDeviceGetPciInfo(handle):  # noqa: N802
            return SimpleNamespace(busId=f"00000000:0{handle + 1}:00.0".encode())

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert runner.worker_gpu_identity(
        logical_device=0,
        nvml_module=FakeNvml,
    ) == {
        "logical_device": 0,
        "cuda_visible_devices": "1",
        "visible_device_selector": "1",
        "device_uuid": "GPU-uuid-1",
        "pci_bus_id": "00000000:02:00.0",
        "device_name": "NVIDIA H100 1",
    }

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-uuid-1")
    assert (
        runner.worker_gpu_identity(
            logical_device=0,
            nvml_module=FakeNvml,
        )["pci_bus_id"]
        == "00000000:02:00.0"
    )


def test_gpu_memory_headroom_rejects_less_than_two_gib() -> None:
    gib = 1024**3
    hardware = {
        "devices": [
            {
                "index": 0,
                "uuid": "GPU-a",
                "total_memory_bytes": 80 * gib,
            }
        ]
    }

    with pytest.raises(RuntimeError, match="2 GiB"):
        runner.gpu_memory_headroom_evidence(
            hardware,
            peak_device_memory_bytes=(79 * gib,),
        )


def test_prepare_workload_builds_exact_pressure_shape_without_mutating_manifest() -> None:
    manifest = WorkloadManifest.from_path(DATA)

    pressure = prepare_workload(
        manifest,
        request_count=3,
        uniform_prompt_length=25_000,
        request_id_prefix="pressure",
        max_new_tokens=25_000,
    )

    assert len(manifest.requests) == 96
    assert manifest.max_new_tokens == 5_989
    assert [len(request.prompt_token_ids) for request in pressure.requests] == [25_000] * 3
    assert pressure.max_new_tokens == 25_000
    assert [request.request_id for request in pressure.requests] == [
        "pressure-0000",
        "pressure-0001",
        "pressure-0002",
    ]


def test_load_source_manifest_tokenizes_hash_pinned_jsonl_and_preserves_ids(tmp_path) -> None:
    prompt_source = tmp_path / "matched.jsonl"
    prompt_source.write_text(
        '{"id":"audit_prompt10_0000","prompt":"+~GAGTTTTATC"}\n{"id":"audit_prompt10_0001","prompt":"+~GAGTTTTATC"}\n',
        encoding="utf-8",
    )
    tokenizer_json = (
        __import__("pathlib").Path(__file__).parents[4]
        / "tokenizers"
        / "nucleotide_fast_tokenizer_512"
        / "tokenizer.json"
    )
    args = SimpleNamespace(
        manifest=DATA,
        prompt_jsonl=prompt_source,
        prompt_jsonl_sha256=hashlib.sha256(prompt_source.read_bytes()).hexdigest(),
        prompt_tokenizer_json=tokenizer_json,
        expected_prompt_tokens=12,
    )

    manifest = runner.load_source_manifest(args)

    assert [request.request_id for request in manifest.requests] == [
        "audit_prompt10_0000",
        "audit_prompt10_0001",
    ]
    assert [len(request.prompt_token_ids) for request in manifest.requests] == [12, 12]
    assert manifest.prompt_source_sha256 == args.prompt_jsonl_sha256
    assert manifest.prompt_tokenizer_sha256 == hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()


def test_prepare_workload_rejects_synthetic_prompt_or_id_rewrites_for_frozen_source(tmp_path) -> None:
    prompt_source = tmp_path / "matched.jsonl"
    prompt_source.write_text('{"id":"audit-0","prompt":"ACGT"}\n', encoding="utf-8")
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text("{}\n", encoding="utf-8")
    manifest = WorkloadManifest.from_path(DATA).with_prompt_jsonl(
        prompt_source,
        tokenize=lambda prompt: tuple(map(ord, prompt)),
        tokenizer_path=tokenizer_json,
        expected_sha256=hashlib.sha256(prompt_source.read_bytes()).hexdigest(),
        expected_prompt_tokens=4,
    )

    with pytest.raises(ValueError, match="frozen prompt source"):
        prepare_workload(
            manifest,
            request_count=2,
            uniform_prompt_length=None,
            request_id_prefix="rewritten",
            max_new_tokens=5_988,
        )


def _fake_outputs(manifest: WorkloadManifest):
    outputs = []
    for index, request in enumerate(manifest.requests):
        token_ids = (65 + index, 67 + index, 71 + index)
        completion = SimpleNamespace(
            token_ids=token_ids,
            logprobs=[{token_id: SimpleNamespace(logprob=-0.1)} for token_id in token_ids],
            finish_reason="length",
            stop_reason=None,
        )
        outputs.append(
            SimpleNamespace(
                prompt_token_ids=list(request.prompt_token_ids),
                outputs=[completion],
                finished=True,
                metrics=SimpleNamespace(
                    first_token_latency=0.2,
                    first_token_ts=10.0,
                    last_token_ts=10.2,
                    num_generation_tokens=3,
                ),
            )
        )
    return outputs


def _compilation_snapshot() -> dict[str, int]:
    return {
        "num_models_seen": 1,
        "num_backend_compilations": 1,
        "num_inductor_compiles": 2,
        "num_eager_compiles": 0,
        "num_gpu_runner_capture_triggers": 1,
        "num_cudagraph_captured": 2,
        "stock_torch_compile_count": 1,
    }


def _write_valid_direct_proof_artifact(
    tmp_path,
    *,
    generation_round: int = 7,
    shared_prefix: bool = False,
) -> tuple[Path, dict, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    proof_path = tmp_path / "proof.json"
    manifest = (
        _shared_prefix_manifest() if shared_prefix else WorkloadManifest.from_path(DATA).request_slice(0, 2)
    ).with_max_new_tokens(3)
    prefix_args = ["--shared-prefix-state-reuse"] if shared_prefix else []
    args = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            "/checkpoint",
            "--manifest",
            str(DATA),
            "--topology",
            "tp2",
            "--max-model-len",
            "64",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.95",
            "--global-wave-size",
            "2",
            "--max-num-seqs",
            "2",
            "--generation-round",
            str(generation_round),
            "--warmups",
            "0",
            "--repetitions",
            "1",
            "--proof",
            *prefix_args,
            "--output",
            str(proof_path),
        ]
    )
    profile = runner.profile_from_args(args, manifest)
    compilation = _compilation_snapshot()
    hardware = {
        "driver_version": "570.86.15",
        "cuda_visible_devices": "0,1",
        "device_count": 2,
        "devices": [
            {
                "index": 0,
                "uuid": "GPU-uuid-0",
                "pci_bus_id": "00000000:01:00.0",
                "name": "NVIDIA H100 0",
                "total_memory_bytes": 80 * 1024**3,
                "initial_used_memory_bytes": 0,
                "initial_free_memory_bytes": 80 * 1024**3,
            },
            {
                "index": 1,
                "uuid": "GPU-uuid-1",
                "pci_bus_id": "00000000:02:00.0",
                "name": "NVIDIA H100 1",
                "total_memory_bytes": 81 * 1024**3,
                "initial_used_memory_bytes": 0,
                "initial_free_memory_bytes": 81 * 1024**3,
            },
        ],
    }
    checkpoint_identity = {"checkpoint_sha256": "checkpoint-sha256"}
    source_identity = {
        "git_dirty": False,
        "git_head": "bionemo-head",
        "source_tree_sha256": "bionemo-source-sha256",
    }
    vllm_identity = {
        "distribution_version": "0.20.0",
        "installation_sha256": "vllm-installation-sha256",
    }
    contract = {
        **runner.build_benchmark_contract(args, manifest, profile),
        "runtime_attestation": runner.runtime_attestation_contract(
            checkpoint=checkpoint_identity,
            sources={"bionemo": source_identity},
            vllm_installation=vllm_identity,
            gpu_hardware=hardware,
        ),
    }
    peak_memory = (70 * 1024**3, 71 * 1024**3)
    phases = []
    call_index = generation_round
    for sample_index, phase_name in enumerate(("cold-generation", "steady-0")):
        wave_phase = f"{phase_name}.wave-000"
        executions = runner.build_wave_execution_records(
            manifest,
            global_wave_size=profile.global_wave_size,
            call_index_start=call_index,
        )
        records = tuple(
            GenerationRecord(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                output_token_ids=(65 + index, 67 + index, 71 + index),
                output_logprobs=(-0.1, -0.2, -0.3),
                requested_max_tokens=manifest.max_new_tokens,
                finish_reason="length",
                stop_reason=None,
                stopped_on_eos=False,
            )
            for index, request in enumerate(manifest.requests)
        )
        sidecar = runner.write_full_generation_records_artifact(
            runner.phase_output_artifact_path(proof_path, phase=phase_name),
            records=records,
            execution_records=executions,
        )
        observations = [
            {
                "phase": wave_phase,
                "engine_index": 0,
                "num_unpadded_tokens": 2,
                "num_padded_tokens": 2,
                "num_paddings": 0,
                "runtime_mode": "CUDAGraphMode.FULL",
            },
            {
                "phase": wave_phase,
                "engine_index": 0,
                "num_unpadded_tokens": 2,
                "num_padded_tokens": 2,
                "num_paddings": 0,
                "runtime_mode": "CUDAGraphMode.FULL",
            },
        ]
        scheduler_observations = [
            {
                "phase": wave_phase,
                "engine_index": 0,
                "preemption_events": 0,
                "recompute_events": 0,
                "prefix_preempted_requests": 0,
                "prefix_preempted_queries": 0,
                "prefix_preempted_hits": 0,
                "preempted_prompt_recomputed_tokens": 0,
                "prompt_tokens_computed": sum(len(request.prompt_token_ids) for request in manifest.requests),
                "prompt_tokens_cached": 0,
                "prompt_tokens_total": sum(len(request.prompt_token_ids) for request in manifest.requests),
                "num_running_requests": len(manifest.requests),
                "num_waiting_requests": 0,
                "num_skipped_waiting_requests": 0,
            }
        ]
        full_decode = runner.full_decode_proof_summary(
            observations,
            phase=wave_phase,
            batch_size=len(manifest.requests),
            max_new_tokens=manifest.max_new_tokens,
        )
        scheduler = runner.scheduler_capacity_proof_summary(
            scheduler_observations,
            phase=wave_phase,
            global_wave_size=len(manifest.requests),
            max_num_seqs=profile.resolved_max_num_seqs,
        )
        wave = {
            "wave_index": 0,
            "start": 0,
            "stop": len(manifest.requests),
            "request_count": len(manifest.requests),
            "call_index": call_index,
            "generation_s": 1.0,
            "full_decode_proof": full_decode,
            "scheduler_capacity_proof": scheduler,
        }
        workers = [
            {
                "rank": rank,
                "device": rank,
                "logical_device": rank,
                "device_name": hardware["devices"][rank]["name"],
                "device_uuid": hardware["devices"][rank]["uuid"],
                "pci_bus_id": hardware["devices"][rank]["pci_bus_id"],
                "cuda_visible_devices": "0,1",
                "visible_device_selector": str(rank),
                "fir_routes": {
                    "direct": {"calls": 27, "requests": 54, "tokens": 108},
                    "fallback_reasons": {"short_request": 27},
                },
                "compilation": dict(compilation),
                "cuda_memory": {
                    "allocated_bytes": 1,
                    "reserved_bytes": 2,
                    "peak_allocated_bytes": 3,
                    "peak_reserved_bytes": 4,
                },
                "mamba_state_copies": {},
                "mamba_prefix_clones": (
                    _prefix_worker_stats([_prefix_clone_record(f"clone-{rank}")]) if shared_prefix else {}
                ),
            }
            for rank in range(2)
        ]
        prefix_reuse = None
        if shared_prefix:
            prefix_reuse = runner.shared_prefix_state_reuse_evidence(
                manifest,
                cached_tokens=(0, 16),
                worker_proof=workers,
                expected_worker_clone_counts=(1, 1),
                cache_block_size=16,
            )
            prefix_reuse = {
                **prefix_reuse,
                "phase_prefix_cache_reset": True,
            }
        expected_decode_tokens = full_decode["expected_decode_tokens"]
        full_decode_tokens = full_decode["full_decode_tokens"]
        phases.append(
            {
                "phase": phase_name,
                "sample": {
                    "sample_index": sample_index,
                    "generation_s": 1.0,
                    "request_count": len(manifest.requests),
                    "prompt_tokens": sum(len(request.prompt_token_ids) for request in manifest.requests),
                    "generated_tokens": len(manifest.requests) * manifest.max_new_tokens,
                    "ttft_s": [0.1, 0.1],
                    "inter_token_latency_s": [0.1, 0.1],
                    "output_lengths": [manifest.max_new_tokens] * len(manifest.requests),
                    "peak_device_memory_bytes": list(peak_memory),
                },
                "generation_call_s": [1.0],
                "wave_proofs": [wave],
                "wave_execution": runner.wave_execution_summary([wave]),
                "cudagraph_observation_count": len(observations),
                "cudagraph_observations_retained": observations,
                "cudagraph_summary": runner.summarize_cudagraph_observations(tuple(observations)),
                "outputs": [record.summary_dict() for record in records],
                "request_executions": [execution.to_dict() for execution in executions],
                "full_output_artifact": sidecar,
                "full_decode_proof": {
                    "phase": phase_name,
                    "wave_count": 1,
                    "expected_decode_tokens": expected_decode_tokens,
                    "full_decode_tokens": full_decode_tokens,
                    "coverage_fraction": full_decode_tokens / expected_decode_tokens,
                    "passed": True,
                    "waves": [full_decode],
                },
                "worker_proof": workers,
                "shared_prefix_state_reuse": prefix_reuse,
                "proof_collected": True,
                "prefix_cache_reset": shared_prefix,
            }
        )
        call_index += 1

    initialized_workers = [
        {
            "rank": rank,
            "device": rank,
            "logical_device": rank,
            "device_name": hardware["devices"][rank]["name"],
            "device_uuid": hardware["devices"][rank]["uuid"],
            "pci_bus_id": hardware["devices"][rank]["pci_bus_id"],
            "cuda_visible_devices": "0,1",
            "visible_device_selector": str(rank),
            "fir_routes": {},
            "compilation": dict(compilation),
        }
        for rank in range(2)
    ]
    artifact = {
        "schema_version": 1,
        "backend": "vllm",
        "topology": "tp2",
        "benchmark_mode": "proof",
        "benchmark_contract": contract,
        "benchmark_contract_sha256": runner.benchmark_contract_sha256(contract),
        "proof_status": {
            "passed": True,
            "phase_count": len(phases),
            "full_decode_passed": True,
            "compilation_stable": True,
        },
        "invocation": {
            "parsed_args": {
                name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
            },
            "output_artifact_path": str(proof_path.resolve()),
            "exit_status": 0,
        },
        "manifest": manifest.to_dict(),
        "manifest_sha256": manifest.sha256,
        "profile": vars(profile),
        "resolved_config": profile.expected_resolved_config(),
        "checkpoint_provenance": checkpoint_identity,
        "source_provenance": source_identity,
        "vllm_installation_provenance": vllm_identity,
        "gpu_hardware_provenance": hardware,
        "gpu_memory_headroom": runner.gpu_memory_headroom_evidence(
            hardware,
            peak_device_memory_bytes=peak_memory,
        ),
        "timing": {
            "engine_init_peak_device_memory_bytes": list(peak_memory),
        },
        "initialized_worker_proof": initialized_workers,
        "phases": phases,
    }
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")
    return proof_path, contract, artifact


def _write_valid_dp2_proof_artifact(tmp_path) -> tuple[Path, dict, dict]:
    proof_path, _, base = _write_valid_direct_proof_artifact(tmp_path)
    manifest = WorkloadManifest.from_dict(base["manifest"])
    args = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            "/checkpoint",
            "--manifest",
            str(DATA),
            "--topology",
            "dp2",
            "--max-model-len",
            "64",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.95",
            "--global-wave-size",
            "2",
            "--max-num-seqs",
            "1",
            "--generation-round",
            "0",
            "--warmups",
            "0",
            "--repetitions",
            "1",
            "--proof",
            "--output",
            str(proof_path),
        ]
    )
    profile = runner.profile_from_args(args, manifest)
    hardware = base["gpu_hardware_provenance"]
    nemo_source_identity = {
        "git_dirty": False,
        "git_head": "nemo-rl-head",
        "source_tree_sha256": "nemo-rl-source-sha256",
    }
    source_identities = {
        "bionemo": base["source_provenance"],
        "nemo_rl": nemo_source_identity,
    }
    contract = {
        **runner.build_benchmark_contract(args, manifest, profile),
        "runtime_attestation": runner.runtime_attestation_contract(
            checkpoint=base["checkpoint_provenance"],
            sources=source_identities,
            vllm_installation=base["vllm_installation_provenance"],
            gpu_hardware=hardware,
        ),
    }
    compilation = _compilation_snapshot()
    resolved = profile.expected_resolved_config()
    phases = []
    call_index = 0
    global_index = 0
    for sample_index, phase_name in enumerate(("cold-generation", "steady-0")):
        wave = runner.build_request_waves(
            request_count=len(manifest.requests),
            global_batch_size=profile.global_wave_size,
            replica_count=profile.replica_count,
        )[0]
        executions = tuple(
            record
            for shard in wave.shards
            for record in runner.build_request_execution_records(
                manifest.request_slice(shard.start, shard.stop),
                global_request_offset=global_index + shard.start,
                dp_rank=shard.replica_index,
                dp_size=profile.replica_count,
                call_index=call_index,
            )
        )
        records = tuple(
            GenerationRecord(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                output_token_ids=(65 + index, 67 + index, 71 + index),
                output_logprobs=(-0.1, -0.2, -0.3),
                requested_max_tokens=manifest.max_new_tokens,
                finish_reason="length",
                stop_reason=None,
                stopped_on_eos=False,
            )
            for index, request in enumerate(manifest.requests)
        )
        sidecar = runner.write_full_generation_records_artifact(
            runner.phase_output_artifact_path(proof_path, phase=phase_name),
            records=records,
            execution_records=executions,
        )
        wave_phase = f"{phase_name}.wave-000"
        engines = []
        for shard in wave.shards:
            observations = [
                {
                    "phase": wave_phase,
                    "engine_index": 0,
                    "num_unpadded_tokens": shard.request_count,
                    "num_padded_tokens": shard.request_count,
                    "num_paddings": 0,
                    "runtime_mode": "CUDAGraphMode.FULL",
                }
                for _ in range(2)
            ]
            scheduler_observations = [
                {
                    "phase": wave_phase,
                    "engine_index": 0,
                    "preemption_events": 0,
                    "recompute_events": 0,
                    "prefix_preempted_requests": 0,
                    "prefix_preempted_queries": 0,
                    "prefix_preempted_hits": 0,
                    "preempted_prompt_recomputed_tokens": 0,
                    "prompt_tokens_computed": len(manifest.requests[shard.start].prompt_token_ids),
                    "prompt_tokens_cached": 0,
                    "prompt_tokens_total": len(manifest.requests[shard.start].prompt_token_ids),
                    "num_running_requests": shard.request_count,
                    "num_waiting_requests": 0,
                    "num_skipped_waiting_requests": 0,
                }
            ]
            worker = {
                "rank": 0,
                "device": 0,
                "logical_device": 0,
                "device_name": hardware["devices"][shard.replica_index]["name"],
                "device_uuid": hardware["devices"][shard.replica_index]["uuid"],
                "pci_bus_id": hardware["devices"][shard.replica_index]["pci_bus_id"],
                "cuda_visible_devices": str(shard.replica_index),
                "visible_device_selector": str(shard.replica_index),
                "fir_routes": {
                    "direct": {"calls": 27, "requests": 27, "tokens": 54},
                    "fallback_reasons": {"short_request": 27},
                },
                "compilation": dict(compilation),
            }
            engines.append(
                {
                    "dp_rank": shard.replica_index,
                    "request_count": shard.request_count,
                    "full_decode_proof": runner.full_decode_proof_summary(
                        observations,
                        phase=wave_phase,
                        batch_size=shard.request_count,
                        max_new_tokens=manifest.max_new_tokens,
                    ),
                    "scheduler_capacity_proof": runner.scheduler_capacity_proof_summary(
                        scheduler_observations,
                        phase=wave_phase,
                        global_wave_size=wave.request_count,
                        engine_request_count=shard.request_count,
                        max_num_seqs=profile.resolved_max_num_seqs,
                    ),
                    "phase": wave_phase,
                    "resolved_config": resolved,
                    "cudagraph_observations": observations,
                    "cudagraph_summary": runner.summarize_cudagraph_observations(tuple(observations)),
                    "scheduler_observations": scheduler_observations,
                    "worker_proof": [worker],
                }
            )
        wave_proof = {
            "wave_index": 0,
            "phase": wave_phase,
            "start": 0,
            "stop": len(manifest.requests),
            "request_count": len(manifest.requests),
            "generation_s": 1.0,
            "reset_proof": [{"phase": wave_phase}] * 2,
            "engines": engines,
            "full_vocab_logprobs": None,
            "shared_prefix_state_reuse": None,
        }
        phases.append(
            {
                "phase": phase_name,
                "sample": {
                    "sample_index": sample_index,
                    "generation_s": 1.0,
                    "request_count": len(manifest.requests),
                    "prompt_tokens": sum(len(request.prompt_token_ids) for request in manifest.requests),
                    "generated_tokens": len(manifest.requests) * manifest.max_new_tokens,
                    "ttft_s": [0.1, 0.1],
                    "inter_token_latency_s": [0.1, 0.1],
                    "output_lengths": [manifest.max_new_tokens] * len(manifest.requests),
                    "peak_device_memory_bytes": base["phases"][sample_index]["sample"]["peak_device_memory_bytes"],
                },
                "generation_call_s": [1.0],
                "wave_execution": runner.wave_execution_summary([wave_proof]),
                "outputs": [record.summary_dict() for record in records],
                "request_executions": [record.to_dict() for record in executions],
                "full_output_artifact": sidecar,
                "waves": [wave_proof],
                "proof_collected": True,
                "prefix_cache_reset": False,
            }
        )
        call_index += 1
        global_index += len(manifest.requests)

    initialized = [
        {
            "phase": "engine-initialized",
            "resolved_config": resolved,
            "worker_proof": [
                {
                    **phases[0]["waves"][0]["engines"][dp_rank]["worker_proof"][0],
                    "fir_routes": {},
                }
            ],
        }
        for dp_rank in range(2)
    ]
    artifact = {
        **base,
        "backend": "nemo-rl-vllm",
        "topology": "dp2",
        "benchmark_contract": contract,
        "benchmark_contract_sha256": runner.benchmark_contract_sha256(contract),
        "invocation": {
            "parsed_args": {
                name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
            },
            "output_artifact_path": str(proof_path.resolve()),
            "exit_status": 0,
        },
        "profile": vars(profile),
        "source_provenance": source_identities,
        "resolved_configs": [resolved, resolved],
        "initialized_engine_proofs": initialized,
        "phases": phases,
        "proof_status": {
            "passed": True,
            "phase_count": len(phases),
            "full_decode_passed": True,
            "compilation_stable": True,
        },
    }
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")
    return proof_path, contract, artifact


def test_linked_proof_validator_accepts_complete_recomputed_direct_evidence(tmp_path) -> None:
    proof_path, contract, _ = _write_valid_direct_proof_artifact(tmp_path)

    evidence = runner.validate_linked_proof_artifact(
        proof_path,
        expected_contract=contract,
        require_memory_headroom=True,
    )

    assert evidence["artifact_path"] == str(proof_path.resolve())
    assert evidence["artifact_sha256"] == hashlib.sha256(proof_path.read_bytes()).hexdigest()


def test_linked_proof_validator_accepts_and_recomputes_dp2_engine_evidence(tmp_path) -> None:
    proof_path, contract, _ = _write_valid_dp2_proof_artifact(tmp_path)

    evidence = runner.validate_linked_proof_artifact(
        proof_path,
        expected_contract=contract,
        require_memory_headroom=True,
    )

    assert evidence["validated_evidence"]["final_worker_count"] == 2


@pytest.mark.parametrize("tamper", ("scheduler_raw", "resolved_config"))
def test_linked_proof_validator_rejects_tampered_dp2_engine_evidence(tmp_path, tamper) -> None:
    proof_path, contract, artifact = _write_valid_dp2_proof_artifact(tmp_path / tamper)
    if tamper == "scheduler_raw":
        scheduler = artifact["phases"][0]["waves"][0]["engines"][0]["scheduler_observations"][0]
        scheduler["preemption_events"] = 1
    else:
        artifact["resolved_configs"][0]["model"]["max_model_len"] += 1
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(AssertionError):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            require_memory_headroom=True,
        )


def test_linked_proof_validator_recomputes_physical_prefix_reuse(tmp_path) -> None:
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(
        tmp_path,
        shared_prefix=True,
    )
    runner.validate_linked_proof_artifact(
        proof_path,
        expected_contract=contract,
        require_memory_headroom=True,
    )

    artifact["phases"][0]["shared_prefix_state_reuse"]["cache_hit_request_count"] = 0
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(AssertionError, match="prefix"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            require_memory_headroom=True,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "manifest",
        "sidecar",
        "scheduler",
        "scheduler_prefix_preemption",
        "full_decode",
        "full_decode_derived",
        "compilation",
        "fir_route",
        "fir_unknown_route",
        "gpu_binding",
        "runtime_attestation",
        "memory",
    ),
)
def test_linked_proof_validator_recomputes_all_direct_evidence(tmp_path, tamper) -> None:
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(tmp_path / tamper)
    artifact = copy.deepcopy(artifact)
    if tamper == "manifest":
        artifact["manifest"]["name"] = "forged-manifest"
    elif tamper == "sidecar":
        artifact["phases"][0]["full_output_artifact"]["sha256"] = "0" * 64
    elif tamper == "scheduler":
        artifact["phases"][0]["wave_proofs"][0]["scheduler_capacity_proof"]["preemption_events"] = 1
    elif tamper == "scheduler_prefix_preemption":
        scheduler = artifact["phases"][0]["wave_proofs"][0]["scheduler_capacity_proof"]
        scheduler["prefix_preempted_queries"] = 1
        scheduler["prefix_preempted_hits"] = 1
    elif tamper == "full_decode":
        artifact["phases"][0]["wave_proofs"][0]["full_decode_proof"]["full_decode_tokens"] += 2
    elif tamper == "full_decode_derived":
        artifact["phases"][0]["wave_proofs"][0]["full_decode_proof"]["minimum_average_occupancy"] = 0
    elif tamper == "compilation":
        artifact["phases"][-1]["worker_proof"][0]["compilation"]["num_inductor_compiles"] += 1
    elif tamper == "fir_route":
        artifact["phases"][0]["worker_proof"][0]["fir_routes"]["fallback_reasons"] = {"ragged_or_chunked": 1}
    elif tamper == "fir_unknown_route":
        artifact["phases"][0]["worker_proof"][0]["fir_routes"]["eager_fallback"] = {
            "calls": 1,
            "requests": 2,
            "tokens": 4,
        }
    elif tamper == "gpu_binding":
        artifact["phases"][0]["worker_proof"][0]["cuda_visible_devices"] = "1,0"
    elif tamper == "runtime_attestation":
        artifact["gpu_hardware_provenance"]["devices"][0]["total_memory_bytes"] += 100 * 1024**3
        artifact["gpu_memory_headroom"] = runner.gpu_memory_headroom_evidence(
            artifact["gpu_hardware_provenance"],
            peak_device_memory_bytes=artifact["timing"]["engine_init_peak_device_memory_bytes"],
        )
    elif tamper == "memory":
        artifact["gpu_memory_headroom"]["devices"][0]["headroom_bytes"] += 1
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(AssertionError):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            require_memory_headroom=True,
        )


def test_benchmark_contract_pins_generation_round_seed_stream(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA)
    common = [
        "--backend",
        "vllm",
        "--checkpoint",
        "/checkpoint",
        "--manifest",
        str(DATA),
        "--topology",
        "tp2",
        "--max-num-batched-tokens",
        "32768",
        "--gpu-memory-utilization",
        "0.95",
        "--output",
        str(tmp_path / "proof.json"),
    ]
    first_args = runner.build_parser().parse_args([*common, "--generation-round", "7", "--proof"])
    second_args = runner.build_parser().parse_args(
        [
            *common,
            "--generation-round",
            "8",
            "--linked-proof-artifact",
            str(tmp_path / "proof.json"),
        ]
    )
    first_contract = runner.build_benchmark_contract(
        first_args,
        manifest,
        runner.profile_from_args(first_args, manifest),
    )
    second_contract = runner.build_benchmark_contract(
        second_args,
        manifest,
        runner.profile_from_args(second_args, manifest),
    )

    assert first_contract["seed_stream"] == {
        "schema_version": 1,
        "base_seed": manifest.seed,
        "generation_round": 7,
        "round_stride": 1_000_003,
        "modulus": 2**31,
    }
    assert first_contract != second_contract


def test_generation_phase_times_one_complete_batch_and_preserves_exact_outputs(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    recorder = CUDAGraphProofRecorder()
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        call_index=0,
    )

    class FakeLLM:
        def generate(self, prompts, sampling_params, *, use_tqdm):
            assert prompts == [{"prompt_token_ids": list(request.prompt_token_ids)} for request in manifest.requests]
            assert len(sampling_params) == 2
            assert use_tqdm is False
            recorder.record(
                _scheduler_stats(unpadded=2, padded=2, mode="CUDAGraphMode.FULL"),
                None,
            )
            recorder.record(
                _scheduler_stats(unpadded=2, padded=2, mode="CUDAGraphMode.FULL"),
                None,
            )
            return _fake_outputs(manifest)

    times = iter((10.0, 12.5))
    proof_events = []
    result = run_generation_phase(
        llm=FakeLLM(),
        manifest=manifest,
        sampling_params=build_request_sampling_params(
            manifest,
            sampling_params_factory=SimpleNamespace,
            execution_records=execution_records,
        ),
        phase="steady-0",
        sample_index=0,
        recorder=recorder,
        memory_monitor_factory=lambda: PeakMemoryMonitor(lambda: (1_000, 2_000)),
        execution_records=execution_records,
        full_output_path=tmp_path / "steady-0.outputs.jsonl.gz",
        reset_worker_proof=lambda: proof_events.append("reset"),
        snapshot_worker_proof=lambda: (
            {
                "rank": 0,
                "fir_routes": {"equal_length_conv": {"calls": 9, "requests": 2, "tokens": 10}},
            },
        ),
        clock=lambda: next(times),
    )

    assert result.sample.generation_s == 2.5
    assert result.sample.request_count == 2
    assert result.sample.generated_tokens == 6
    assert result.sample.peak_device_memory_bytes == (1_000, 2_000)
    assert len(result.observations) == 2
    assert [summary["output_length"] for summary in result.output_summaries] == [3, 3]
    assert proof_events == ["reset"]
    assert result.worker_proof[0]["rank"] == 0
    artifact = result.to_dict()
    assert artifact["worker_proof"][0]["fir_routes"]["equal_length_conv"]["calls"] == 9
    assert artifact["request_executions"][0]["seed"] == 42
    assert artifact["full_output_artifact"]["generated_token_count"] == 6
    assert artifact["full_decode_proof"]["passed"] is True
    assert artifact["full_decode_proof"]["full_decode_tokens"] == 4


def test_speed_generation_avoids_proof_callbacks_and_memory_polling(tmp_path) -> None:
    manifest = _shared_prefix_manifest().with_max_new_tokens(3)
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        call_index=0,
    )

    class FakeLLM:
        cache_resets = 0

        def reset_prefix_cache(self):
            self.cache_resets += 1
            return True

        def generate(self, prompts, sampling_params, *, use_tqdm):
            assert len(prompts) == len(sampling_params) == 2
            assert use_tqdm is False
            return _fake_outputs(manifest)

    llm = FakeLLM()
    times = iter((10.0, 11.0))
    result = run_generation_phase(
        llm=llm,
        manifest=manifest,
        sampling_params=build_request_sampling_params(
            manifest,
            sampling_params_factory=SimpleNamespace,
            execution_records=execution_records,
        ),
        phase="steady-0",
        sample_index=0,
        recorder=None,
        memory_monitor_factory=lambda: pytest.fail("speed lane started peak-memory polling"),
        execution_records=execution_records,
        full_output_path=tmp_path / "speed.outputs.jsonl.gz",
        reset_worker_proof=lambda: pytest.fail("speed lane reset worker proof"),
        snapshot_worker_proof=lambda: pytest.fail("speed lane snapshotted worker proof"),
        require_shared_prefix_state_reuse=True,
        collect_proof=False,
        global_wave_size=2,
        scheduler_max_num_seqs=2,
        clock=lambda: next(times),
    )

    assert llm.cache_resets == 1
    assert result.sample.generation_s == 1.0
    assert result.sample.peak_device_memory_bytes == ()
    assert result.proof_collected is False
    assert result.prefix_cache_reset is True
    assert result.observations == ()
    assert result.worker_proof == ()
    assert result.shared_prefix_state_reuse is None
    assert result.full_decode_proof is None
    assert result.wave_proofs[0]["full_decode_proof"] is None
    assert result.wave_proofs[0]["scheduler_capacity_proof"] is None
    assert result.full_output_artifact["generated_token_count"] == 6


def test_generation_phase_executes_explicit_10x96_plus_tail40_calls(tmp_path) -> None:
    manifest = (
        WorkloadManifest.from_path(DATA).with_request_count(1_000, request_id_prefix="audit").with_max_new_tokens(3)
    )
    recorder = CUDAGraphProofRecorder()
    calls = []

    class FakeLLM:
        def generate(self, prompts, sampling_params, *, use_tqdm):
            start = sum(calls)
            wave_index = len(calls)
            request_count = len(prompts)
            calls.append(request_count)
            wave_manifest = manifest.request_slice(start, start + request_count)
            assert prompts == [
                {"prompt_token_ids": list(request.prompt_token_ids)} for request in wave_manifest.requests
            ]
            assert [params.seed for params in sampling_params] == [
                request_seed(
                    42,
                    call_index=7 + wave_index,
                    dp_rank=0,
                    dp_size=1,
                    request_index_in_stream=index,
                )
                for index in range(request_count)
            ]
            assert use_tqdm is False
            for _ in range(2):
                recorder.record(
                    _scheduler_stats(
                        unpadded=request_count,
                        padded=request_count,
                        mode="CUDAGraphMode.FULL",
                    ),
                    _iteration_stats(),
                )
            return _fake_outputs(wave_manifest)

    execution_records = runner.build_wave_execution_records(
        manifest,
        global_wave_size=96,
        call_index_start=7,
    )
    clock_values = iter(float(value) for value in range(22))
    result = run_generation_phase(
        llm=FakeLLM(),
        manifest=manifest,
        sampling_params=build_request_sampling_params(
            manifest,
            sampling_params_factory=SimpleNamespace,
            execution_records=execution_records,
        ),
        phase="steady-0",
        sample_index=0,
        recorder=recorder,
        memory_monitor_factory=lambda: PeakMemoryMonitor(lambda: (1_000, 2_000)),
        execution_records=execution_records,
        full_output_path=tmp_path / "steady-0.outputs.jsonl.gz",
        global_wave_size=96,
        clock=lambda: next(clock_values),
        scheduler_max_num_seqs=96,
    )

    assert calls == [96] * 10 + [40]
    assert result.generation_call_s == (1.0,) * 11
    assert [proof["request_count"] for proof in result.wave_proofs] == calls
    assert [proof["call_index"] for proof in result.wave_proofs] == list(range(7, 18))
    assert all(record.generation_round == record.call_index for record in result.request_executions)
    assert [record.seed for record in result.request_executions[:96]] == list(range(7_000_063, 7_000_159))
    assert [record.seed for record in result.request_executions[96:192]] == list(range(8_000_066, 8_000_162))
    assert [record.seed for record in result.request_executions[-40:]] == list(range(17_000_093, 17_000_133))
    assert all(proof["scheduler_capacity_proof"]["batch_fit_without_preemption"] for proof in result.wave_proofs)
    assert result.full_decode_proof["wave_count"] == 11
    assert result.full_decode_proof["expected_decode_tokens"] == 2_000
    assert result.full_decode_proof["full_decode_tokens"] == 2_000
    assert result.full_decode_proof["passed"] is True
    assert result.sample.request_count == 1_000
    assert result.full_output_artifact["request_count"] == 1_000
    assert result.to_dict()["wave_execution"]["actual_call_count"] == 11
    assert result.to_dict()["wave_execution"]["measured_waves_to_target"] == 1
    assert result.to_dict()["wave_execution"]["measured_time_to_target_s"] == 1.0


def test_wave_execution_summary_measures_five_physical_calls_to_96() -> None:
    summary = runner.wave_execution_summary(
        tuple(
            {
                "wave_index": index,
                "request_count": request_count,
                "generation_s": 1.0,
            }
            for index, request_count in enumerate((20, 20, 20, 20, 16))
        )
    )

    assert summary["target_request_count"] == 96
    assert summary["actual_call_count"] == 5
    assert summary["actual_request_count"] == 96
    assert summary["measured_waves_to_target"] == 5
    assert summary["measured_time_to_target_s"] == 5.0
    assert summary["requests_completed_at_target_boundary"] == 96


def _shared_prefix_manifest() -> WorkloadManifest:
    return (
        WorkloadManifest.from_path(DATA)
        .request_slice(0, 1)
        .with_uniform_prompt_length(
            32,
            request_count=2,
            request_id_prefix="shared",
        )
    )


def _prefix_attention_groups(
    num_computed_tokens: int,
    *,
    block_size: int = 16,
    first_block_id: int = 100,
) -> list[dict[str, object]]:
    block_ids = list(range(first_block_id, first_block_id + num_computed_tokens // block_size))
    digest = hashlib.sha256(json.dumps(block_ids, separators=(",", ":")).encode()).hexdigest()
    return [
        {
            "kv_cache_group_id": 1,
            "layer_names": ["model.layers.3.self_attention"],
            "block_size_tokens": block_size,
            "physical_block_count": len(block_ids),
            "physical_block_ids": block_ids,
            "physical_block_ids_sha256": digest,
        }
    ]


def _prefix_clone_record(
    request_id: str,
    *,
    num_computed_tokens: int = 16,
    prompt_tokens: int = 32,
    copy_entries: int = 8,
    copied_elements: int = 1_024,
    source_request_id: str = "miss",
) -> dict[str, object]:
    base_elements, remainder = divmod(copied_elements, copy_entries)
    state_copies = []
    for state_index in range(copy_entries):
        elements = base_elements + int(state_index < remainder)
        state_copies.append(
            {
                "kv_cache_group_id": 0,
                "layer_name": f"model.layers.{state_index}.mixer",
                "state_index": state_index,
                "dtype": "torch.float32",
                "state_shape": [2 * copy_entries + 1, elements],
                "block_shape": [elements],
                "source_logical_block_index": 0,
                "destination_logical_block_index": 1,
                "source_physical_block_id": 2 * state_index,
                "destination_physical_block_id": 2 * state_index + 1,
                "source_data_ptr": 10_000 + 2 * state_index,
                "destination_data_ptr": 10_001 + 2 * state_index,
                "copied_elements": elements,
                "copied_bytes": 4 * elements,
            }
        )
    source_groups = _prefix_attention_groups(num_computed_tokens)
    reused_groups = _prefix_attention_groups(num_computed_tokens)
    copied_bytes = 4 * copied_elements
    return {
        "request_id": request_id,
        "source_miss_request_id": source_request_id,
        "source_snapshot_index": 0,
        "attention_kv_identity_verified": True,
        "num_computed_tokens": num_computed_tokens,
        "prompt_tokens": prompt_tokens,
        "block_size": 16,
        "source_attention_kv_groups": source_groups,
        "reused_attention_kv_groups": reused_groups,
        "state_copies": state_copies,
        "copy_entries": copy_entries,
        "copied_elements": copied_elements,
        "copied_bytes": copied_bytes,
        "expected_copy_entries": copy_entries,
        "expected_copied_elements": copied_elements,
        "expected_copied_bytes": copied_bytes,
        "all_state_dtypes_fp32": True,
    }


def _prefix_worker_stats(
    clones: list[dict[str, object]],
    *,
    cache_miss_count: int = 1,
    source_request_id: str = "miss",
) -> dict[str, object]:
    clone = clones[0]
    source_groups = _prefix_attention_groups(int(clone["num_computed_tokens"]))
    return {
        "cache_miss_count": cache_miss_count,
        "cache_miss_request_ids": [source_request_id] if cache_miss_count else [],
        "prefix_sources": [
            {
                "request_id": source_request_id,
                "prompt_tokens": clone["prompt_tokens"],
                "snapshots": [
                    {
                        "snapshot_index": 0,
                        "num_computed_tokens_before_step": 0,
                        "num_scheduled_tokens": clone["prompt_tokens"],
                        "directly_observed_prefix_tokens": clone["num_computed_tokens"],
                        "attention_kv_groups": source_groups,
                    }
                ],
            }
        ],
        "clone_count": len(clones),
        "requests": clones,
    }


def test_generation_phase_resets_shared_prefix_cache_once_and_proves_tp2_clones(tmp_path) -> None:
    manifest = _shared_prefix_manifest().with_max_new_tokens(3)
    recorder = CUDAGraphProofRecorder()
    clone = _prefix_clone_record("clone")

    class FakeLLM:
        def __init__(self) -> None:
            self.cache_resets = 0

        def reset_prefix_cache(self):
            self.cache_resets += 1
            return True

        def generate(self, prompts, sampling_params, *, use_tqdm):
            assert len(prompts) == len(sampling_params) == 2
            assert use_tqdm is False
            for _ in range(2):
                recorder.record(
                    _scheduler_stats(unpadded=2, padded=2, mode="CUDAGraphMode.FULL"),
                    _iteration_stats(),
                )
            outputs = _fake_outputs(manifest)
            outputs[0].num_cached_tokens = 0
            outputs[1].num_cached_tokens = 16
            return outputs

    llm = FakeLLM()
    execution_records = runner.build_wave_execution_records(
        manifest,
        global_wave_size=2,
        call_index_start=0,
    )
    times = iter((10.0, 11.0))
    result = run_generation_phase(
        llm=llm,
        manifest=manifest,
        sampling_params=build_request_sampling_params(
            manifest,
            sampling_params_factory=SimpleNamespace,
            execution_records=execution_records,
        ),
        phase="steady-0",
        sample_index=0,
        recorder=recorder,
        memory_monitor_factory=lambda: PeakMemoryMonitor(lambda: (1_000, 2_000)),
        execution_records=execution_records,
        full_output_path=tmp_path / "shared.outputs.jsonl.gz",
        snapshot_worker_proof=lambda: (
            {"rank": 0, "device": 0, "mamba_prefix_clones": _prefix_worker_stats([clone])},
            {"rank": 1, "device": 1, "mamba_prefix_clones": _prefix_worker_stats([clone])},
        ),
        prefix_cache_block_size=16,
        require_shared_prefix_state_reuse=True,
        global_wave_size=2,
        scheduler_max_num_seqs=2,
        clock=lambda: next(times),
    )

    assert llm.cache_resets == 1
    assert result.shared_prefix_state_reuse["phase_prefix_cache_reset"] is True
    assert result.shared_prefix_state_reuse["cache_miss_request_count"] == 1
    assert result.shared_prefix_state_reuse["cache_hit_request_count"] == 1
    assert [worker["clone_count"] for worker in result.shared_prefix_state_reuse["worker_state_clones"]] == [1, 1]


def test_shared_prefix_state_reuse_evidence_requires_cache_hits_and_physical_copies() -> None:
    manifest = _shared_prefix_manifest()
    clone = _prefix_clone_record("1")
    evidence = runner.shared_prefix_state_reuse_evidence(
        manifest,
        cached_tokens=(0, 16),
        worker_proof=(
            {
                "rank": 0,
                "device": 0,
                "mamba_prefix_clones": _prefix_worker_stats([clone]),
            },
            {
                "rank": 1,
                "device": 1,
                "mamba_prefix_clones": _prefix_worker_stats([clone]),
            },
        ),
        expected_worker_clone_counts=(1, 1),
        cache_block_size=16,
    )

    assert evidence["identical_prompt_count"] == 2
    assert evidence["cached_tokens_by_request"] == [0, 16]
    assert evidence["cache_hit_request_count"] == 1
    assert evidence["cache_miss_request_count"] == 1
    assert evidence["logical_clone_request_count"] == 1
    assert evidence["physically_reused_prompt_tokens_per_clone"] == 16
    assert evidence["recomputed_prompt_tokens_per_clone"] == 16
    assert evidence["total_cached_prompt_tokens"] == 16
    assert evidence["scheduled_uncached_prompt_tokens"] == 48
    assert evidence["attention_kv_physical_reuse_proven"] is True
    assert evidence["physical_state_copy_proven"] is True
    assert evidence["expected_fp32_state_copy_elements_per_request"] == 1_024
    assert evidence["expected_fp32_state_copy_bytes_per_request"] == 4_096
    assert [worker["clone_count"] for worker in evidence["worker_state_clones"]] == [1, 1]
    assert evidence["worker_state_clones"][0]["requests"][0]["copied_bytes"] == 4_096


def test_shared_prefix_state_reuse_evidence_proves_exact_96x25k_clone_layout() -> None:
    manifest = (
        WorkloadManifest.from_path(DATA)
        .request_slice(0, 1)
        .with_uniform_prompt_length(25_000, request_count=96, request_id_prefix="prefix25k")
    )
    clones = [
        _prefix_clone_record(
            f"clone-{index}",
            num_computed_tokens=24_992,
            prompt_tokens=25_000,
            copy_entries=18,
            copied_elements=131_072,
        )
        for index in range(95)
    ]

    evidence = runner.shared_prefix_state_reuse_evidence(
        manifest,
        cached_tokens=(0, *((24_992,) * 95)),
        worker_proof=(
            {"rank": 0, "device": 0, "mamba_prefix_clones": _prefix_worker_stats(clones)},
            {"rank": 1, "device": 1, "mamba_prefix_clones": _prefix_worker_stats(clones)},
        ),
        expected_worker_clone_counts=(95, 95),
        cache_block_size=16,
    )

    assert evidence["cache_miss_request_count"] == 1
    assert evidence["cache_hit_request_count"] == 95
    assert evidence["logical_clone_request_count"] == 95
    assert evidence["physically_reused_prompt_tokens_per_clone"] == 24_992
    assert evidence["recomputed_prompt_tokens_per_clone"] == 8
    assert evidence["attention_kv_physical_reuse_proven"] is True
    assert [worker["clone_count"] for worker in evidence["worker_state_clones"]] == [95, 95]
    assert all(
        request["copied_bytes"] == request["expected_copied_bytes"] == 524_288
        for worker in evidence["worker_state_clones"]
        for request in worker["requests"]
    )


def test_shared_prefix_state_reuse_evidence_accepts_all_clones_after_phase_first_wave() -> None:
    manifest = _shared_prefix_manifest()
    clones = [_prefix_clone_record(f"clone-{index}", copy_entries=1, copied_elements=8) for index in range(2)]

    evidence = runner.shared_prefix_state_reuse_evidence(
        manifest,
        cached_tokens=(16, 16),
        worker_proof=(
            {
                "rank": 0,
                "device": 0,
                "mamba_prefix_clones": _prefix_worker_stats(clones, cache_miss_count=0),
            },
        ),
        expected_worker_clone_counts=(2,),
        cache_block_size=16,
        expected_cache_misses=0,
    )

    assert evidence["cache_miss_request_count"] == 0
    assert evidence["cache_hit_request_count"] == 2


def test_shared_prefix_state_reuse_evidence_fails_closed() -> None:
    manifest = _shared_prefix_manifest()
    clone = _prefix_clone_record("1", copy_entries=1, copied_elements=1)
    worker_proof = (
        {
            "rank": 0,
            "device": 0,
            "mamba_prefix_clones": _prefix_worker_stats([clone]),
        },
    )

    with pytest.raises(AssertionError, match="cached-token telemetry"):
        runner.shared_prefix_state_reuse_evidence(
            manifest,
            cached_tokens=(None, 16),
            worker_proof=worker_proof,
            expected_worker_clone_counts=(1,),
            cache_block_size=16,
        )
    with pytest.raises(AssertionError, match="exactly one cache miss"):
        runner.shared_prefix_state_reuse_evidence(
            manifest,
            cached_tokens=(0, 0),
            worker_proof=worker_proof,
            expected_worker_clone_counts=(1,),
            cache_block_size=16,
        )
    with pytest.raises(AssertionError, match="block-aligned prefix"):
        runner.shared_prefix_state_reuse_evidence(
            manifest,
            cached_tokens=(0, 15),
            worker_proof=worker_proof,
            expected_worker_clone_counts=(1,),
            cache_block_size=16,
        )
    with pytest.raises(AssertionError, match="clone count"):
        runner.shared_prefix_state_reuse_evidence(
            manifest,
            cached_tokens=(0, 16),
            worker_proof=worker_proof,
            expected_worker_clone_counts=(2,),
            cache_block_size=16,
        )
    bad_clone = {**clone, "copied_bytes": 8}
    with pytest.raises(AssertionError, match="copy bytes"):
        runner.shared_prefix_state_reuse_evidence(
            manifest,
            cached_tokens=(0, 16),
            worker_proof=(
                {
                    "rank": 0,
                    "device": 0,
                    "mamba_prefix_clones": _prefix_worker_stats([bad_clone]),
                },
            ),
            expected_worker_clone_counts=(1,),
            cache_block_size=16,
        )
    bad_attention_clone = _prefix_clone_record("1", copy_entries=1, copied_elements=1)
    bad_attention_group = bad_attention_clone["reused_attention_kv_groups"][0]
    bad_attention_group["physical_block_ids"][0] = 999
    bad_attention_group["physical_block_ids_sha256"] = hashlib.sha256(
        json.dumps(bad_attention_group["physical_block_ids"], separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(AssertionError, match="physical block IDs"):
        runner.shared_prefix_state_reuse_evidence(
            manifest,
            cached_tokens=(0, 16),
            worker_proof=(
                {
                    "rank": 0,
                    "device": 0,
                    "mamba_prefix_clones": _prefix_worker_stats([bad_attention_clone]),
                },
            ),
            expected_worker_clone_counts=(1,),
            cache_block_size=16,
        )
