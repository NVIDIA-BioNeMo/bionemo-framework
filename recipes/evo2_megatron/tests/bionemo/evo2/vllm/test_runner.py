# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import gzip
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

import bionemo.evo2.vllm.runner as runner
from bionemo.evo2.vllm.benchmark import WorkloadManifest, records_from_vllm_outputs
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
        )
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


def test_request_seeds_match_between_tp2_and_dp2_and_advance_by_round() -> None:
    tp2 = [request_seed(42, generation_round=0, global_request_index=index) for index in range(96)]
    dp2 = [
        *[request_seed(42, generation_round=0, global_request_index=index) for index in range(48)],
        *[request_seed(42, generation_round=0, global_request_index=index) for index in range(48, 96)],
    ]
    next_round = [request_seed(42, generation_round=1, global_request_index=index) for index in range(96)]

    assert tp2 == dp2
    assert len(set(tp2)) == 96
    assert set(tp2).isdisjoint(next_round)
    assert tp2[0] == 42


def test_request_sampling_params_apply_global_seed_offsets() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)

    params = build_request_sampling_params(
        manifest,
        sampling_params_factory=SimpleNamespace,
        generation_round=2,
        global_request_offset=48,
    )

    assert [param.seed for param in params] == [
        request_seed(42, generation_round=2, global_request_index=48),
        request_seed(42, generation_round=2, global_request_index=49),
    ]
    assert all(param.max_tokens == 3 and param.min_tokens == 3 for param in params)
    assert all(param.detokenize is False and param.logprobs == 0 for param in params)


def test_request_execution_records_persist_round_rank_call_and_global_seed() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)

    records = runner.build_request_execution_records(
        manifest,
        generation_round=2,
        global_request_offset=48,
        dp_rank=1,
        call_index=7,
    )

    assert [record.to_dict() for record in records] == [
        {
            "execution_uid": "round=2/call=7/global=48/dp=1/request=gdpo-000",
            "request_id": "gdpo-000",
            "global_request_index": 48,
            "generation_round": 2,
            "dp_rank": 1,
            "call_index": 7,
            "seed": request_seed(42, generation_round=2, global_request_index=48),
        },
        {
            "execution_uid": "round=2/call=7/global=49/dp=1/request=gdpo-001",
            "request_id": "gdpo-001",
            "global_request_index": 49,
            "generation_round": 2,
            "dp_rank": 1,
            "call_index": 7,
            "seed": request_seed(42, generation_round=2, global_request_index=49),
        },
    ]


def test_full_output_artifact_round_trips_every_token_logprob_and_seed(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    execution_records = runner.build_request_execution_records(
        manifest,
        generation_round=3,
        global_request_offset=48,
        dp_rank=1,
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
    assert rows[0]["generation_round"] == 3
    assert rows[0]["dp_rank"] == 1
    assert rows[0]["call_index"] == 9
    assert rows[0]["global_request_index"] == 48
    assert rows[0]["seed"] == execution_records[0].seed
    assert rows[0]["execution_uid"] == "round=3/call=9/global=48/dp=1/request=gdpo-000"
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
        generation_round=3,
        global_request_offset=48,
        dp_rank=1,
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

    source.write_text("VALUE = 2\n")
    dirty = runner.source_provenance(repository=tmp_path, source_roots=(source.parent,))
    assert dirty["git_head"] == clean["git_head"]
    assert dirty["git_dirty"] is True
    assert dirty["dirty_fingerprint_sha256"] != clean["dirty_fingerprint_sha256"]
    assert dirty["source_tree_sha256"] != clean["source_tree_sha256"]


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
        '{"id":"audit_prompt10_0000","prompt":"+~GAGTTTTATC"}\n'
        '{"id":"audit_prompt10_0001","prompt":"+~GAGTTTTATC"}\n',
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


def test_generation_phase_times_one_complete_batch_and_preserves_exact_outputs(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    recorder = CUDAGraphProofRecorder()
    execution_records = runner.build_request_execution_records(
        manifest,
        generation_round=0,
        global_request_offset=0,
        dp_rank=0,
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
            generation_round=0,
            global_request_offset=0,
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
