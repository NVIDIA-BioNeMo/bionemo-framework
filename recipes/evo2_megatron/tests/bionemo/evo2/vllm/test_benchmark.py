# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import ast
import hashlib
import inspect
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from bionemo.evo2.vllm.benchmark import (
    BenchmarkSample,
    GenerationRecord,
    WorkloadManifest,
    aggregate_samples,
    benchmark_sample_from_vllm_outputs,
    build_parser,
    build_request_waves,
    records_from_vllm_outputs,
    sampling_params_kwargs,
    summarize_vllm_outputs,
    validate_compilation_proof,
    validate_generation_records,
)
from bionemo.evo2.vllm.tokenizer_io import SnapshotBoundTokenizer


DATA = Path(__file__).with_name("data") / "gdpo_mixed_96.json"
TOKENIZER_JSON = Path(__file__).resolve().parents[4] / "tokenizers/nucleotide_fast_tokenizer_512/tokenizer.json"
BASE_MANIFEST_SHA256 = "6f85cf5681e194c6499b0ea967acdc70d6706abc41f5c579bdc785b787fc2207"
BASE_INDEX_SHA256 = "f9a37408f07c774c13b55648c40f9a4f7fe2847114f988541e5b501bf59f127f"


def test_gdpo_manifest_is_immutable_and_matches_the_production_workload() -> None:
    manifest = WorkloadManifest.from_path(DATA)

    assert manifest.schema_version == 1
    assert manifest.name == "evo2-phage-gdpo-mixed-96"
    assert len(manifest.requests) == 96
    assert len({request.request_id for request in manifest.requests}) == 96
    assert {len(request.prompt_token_ids) for request in manifest.requests} == set(range(4, 13))
    assert all(0 <= token_id < 512 for request in manifest.requests for token_id in request.prompt_token_ids)
    assert manifest.max_new_tokens == 5_989
    assert manifest.temperature == 1.0
    assert manifest.top_p == 1.0
    assert manifest.top_k == 4
    assert manifest.seed == 42
    assert manifest.dtype == "bfloat16"
    assert manifest.ignore_eos is True
    assert manifest.stop_token_ids == ()
    assert manifest.checkpoint_manifest_sha256 == BASE_MANIFEST_SHA256
    assert manifest.checkpoint_index_sha256 == BASE_INDEX_SHA256
    assert manifest.sha256 == "da98065224de1603a0f4b0103669cffdc7ba221ac64e914bf42b7ad3129c4126"


def test_manifest_validation_rejects_duplicate_ids_and_invalid_sampling() -> None:
    manifest = WorkloadManifest.from_path(DATA)

    with pytest.raises(ValueError, match="unique"):
        WorkloadManifest(
            **{
                **manifest.constructor_kwargs(),
                "requests": (manifest.requests[0], manifest.requests[0]),
            }
        )
    with pytest.raises(ValueError, match="temperature"):
        WorkloadManifest(**{**manifest.constructor_kwargs(), "temperature": -1.0})
    with pytest.raises(ValueError, match="max_new_tokens"):
        WorkloadManifest(**{**manifest.constructor_kwargs(), "max_new_tokens": 0})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("max_new_tokens", 3.0, "max_new_tokens"),
        ("max_new_tokens", True, "max_new_tokens"),
        ("temperature", 1, "temperature"),
        ("temperature", True, "temperature"),
        ("temperature", float("nan"), "temperature"),
        ("temperature", float("inf"), "temperature"),
        ("temperature", 1e-6, "temperature"),
        ("top_p", 1, "top_p"),
        ("top_p", True, "top_p"),
        ("top_p", float("nan"), "top_p"),
        ("top_p", float("inf"), "top_p"),
        ("top_k", 4.0, "top_k"),
        ("top_k", True, "top_k"),
        ("seed", 42.0, "seed"),
        ("seed", True, "seed"),
        ("ignore_eos", 1, "ignore_eos"),
        ("stop_token_ids", (1.0,), "stop_token_ids"),
        ("stop_token_ids", (True,), "stop_token_ids"),
    ),
)
def test_manifest_rejects_raw_sampling_type_coercion_and_nonfinite_values(
    field: str,
    value: object,
    message: str,
) -> None:
    manifest = WorkloadManifest.from_path(DATA)

    with pytest.raises((TypeError, ValueError), match=message):
        WorkloadManifest(**{**manifest.constructor_kwargs(), field: value})


def test_manifest_preserves_top_k_equal_to_vocab_size_as_an_explicit_policy() -> None:
    manifest = WorkloadManifest(
        **{
            **WorkloadManifest.from_path(DATA).constructor_kwargs(),
            "top_k": 512,
        }
    )

    kwargs = sampling_params_kwargs(manifest)

    assert kwargs["top_k"] == 512
    assert type(kwargs["top_k"]) is int


def test_manifest_expands_deterministically_to_full_1000_by_6000_audit() -> None:
    manifest = (
        WorkloadManifest.from_path(DATA)
        .with_request_count(1_000, request_id_prefix="audit")
        .with_max_new_tokens(6_000)
    )

    assert len(manifest.requests) == 1_000
    assert manifest.requests[0].request_id == "audit-0000"
    assert manifest.requests[-1].request_id == "audit-0999"
    assert {len(request.prompt_token_ids) for request in manifest.requests} == set(range(4, 13))
    assert len({request.request_id for request in manifest.requests}) == 1_000
    assert manifest.max_new_tokens == 6_000
    assert manifest.sha256 == (
        WorkloadManifest.from_path(DATA)
        .with_request_count(1_000, request_id_prefix="audit")
        .with_max_new_tokens(6_000)
        .sha256
    )


def test_full_audit_routes_as_10x96_plus_exact_20_per_dp_replica_tail() -> None:
    waves = build_request_waves(request_count=1_000, global_batch_size=96, replica_count=2)

    assert len(waves) == 11
    assert [wave.request_count for wave in waves] == [96] * 10 + [40]
    assert [[shard.request_count for shard in wave.shards] for wave in waves[:-1]] == [[48, 48]] * 10
    assert [shard.request_count for shard in waves[-1].shards] == [20, 20]
    assert [
        request_index for wave in waves for shard in wave.shards for request_index in range(shard.start, shard.stop)
    ] == list(range(1_000))


def test_manifest_builds_exact_long_prompts_from_real_tokens_without_padding() -> None:
    manifest = (
        WorkloadManifest.from_path(DATA)
        .with_uniform_prompt_length(
            25_000,
            request_count=3,
            request_id_prefix="pressure",
        )
        .with_max_new_tokens(25_000)
    )

    assert [request.request_id for request in manifest.requests] == [
        "pressure-0000",
        "pressure-0001",
        "pressure-0002",
    ]
    assert [len(request.prompt_token_ids) for request in manifest.requests] == [25_000] * 3
    assert all(0 <= token_id < 512 for request in manifest.requests for token_id in request.prompt_token_ids)
    assert manifest.max_total_tokens == 50_000


def test_prompt_jsonl_loader_preserves_ids_hash_and_exact_length_lanes(tmp_path) -> None:
    prompt_source = tmp_path / "frozen-prompts.jsonl"
    prompt_payload = (
        '{"id":"audit_prompt10_0000","prompt":"+~GAGTTTTATC"}\n{"id":"audit_prompt10_0001","prompt":"+~GAGTTTTATC"}\n'
    )
    prompt_source.write_text(prompt_payload, encoding="utf-8")
    tokenizer = SnapshotBoundTokenizer.from_path(TOKENIZER_JSON)
    source_sha256 = hashlib.sha256(prompt_payload.encode()).hexdigest()
    manifest = WorkloadManifest.from_path(DATA).with_prompt_jsonl(
        prompt_source,
        tokenizer=tokenizer,
        expected_sha256=source_sha256,
        expected_prompt_tokens=12,
        name="matched-audit-96",
    )

    assert [request.request_id for request in manifest.requests] == [
        "audit_prompt10_0000",
        "audit_prompt10_0001",
    ]
    assert [len(request.prompt_token_ids) for request in manifest.requests] == [12, 12]
    assert manifest.prompt_source_path == str(prompt_source.resolve())
    assert manifest.prompt_source_sha256 == source_sha256
    assert manifest.prompt_tokenizer_path == str(TOKENIZER_JSON.resolve())
    assert manifest.prompt_tokenizer_sha256 == tokenizer.source_sha256
    assert WorkloadManifest.from_dict(manifest.to_dict()) == manifest

    exact_6k_total = manifest.with_max_new_tokens(5_988)
    exact_6k_new = manifest.with_max_new_tokens(6_000)
    assert exact_6k_total.max_total_tokens == 6_000
    assert exact_6k_new.max_total_tokens == 6_012


def test_prompt_jsonl_loader_rejects_source_or_token_length_drift(tmp_path) -> None:
    prompt_source = tmp_path / "frozen-prompts.jsonl"
    prompt_source.write_text('{"id":"audit-0","prompt":"ACGT"}\n', encoding="utf-8")
    tokenizer = SnapshotBoundTokenizer.from_path(TOKENIZER_JSON)
    base = WorkloadManifest.from_path(DATA)

    with pytest.raises(ValueError, match="prompt source SHA256"):
        base.with_prompt_jsonl(
            prompt_source,
            tokenizer=tokenizer,
            expected_sha256="0" * 64,
            expected_prompt_tokens=4,
        )

    source_sha256 = hashlib.sha256(prompt_source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="expected 5 prompt tokens"):
        base.with_prompt_jsonl(
            prompt_source,
            tokenizer=tokenizer,
            expected_sha256=source_sha256,
            expected_prompt_tokens=5,
        )


def test_workload_and_prompt_admission_reject_duplicate_json_keys(tmp_path) -> None:
    manifest_path = tmp_path / "duplicate-manifest.json"
    manifest_payload = DATA.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_payload.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key 'schema_version'"):
        WorkloadManifest.from_path(manifest_path)

    prompt_source = tmp_path / "duplicate-prompts.jsonl"
    prompt_source.write_text('{"id":"audit-0","id":"foreign","prompt":"ACGT"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key 'id'"):
        WorkloadManifest.from_path(DATA).with_prompt_jsonl(
            prompt_source,
            tokenizer=SnapshotBoundTokenizer.from_path(TOKENIZER_JSON),
            expected_sha256=hashlib.sha256(prompt_source.read_bytes()).hexdigest(),
        )


def test_sample_aggregation_reports_median_p95_and_mad_without_hiding_outlier() -> None:
    samples = [
        BenchmarkSample(
            sample_index=0,
            generation_s=1.0,
            request_count=10,
            prompt_tokens=80,
            generated_tokens=100,
            ttft_s=(0.1, 0.2),
            inter_token_latency_s=(0.01, 0.02),
            output_lengths=(10,) * 10,
            peak_device_memory_bytes=(1000,),
        ),
        BenchmarkSample(
            sample_index=1,
            generation_s=2.0,
            request_count=10,
            prompt_tokens=80,
            generated_tokens=100,
            ttft_s=(0.2, 0.3),
            inter_token_latency_s=(0.02, 0.03),
            output_lengths=(10,) * 10,
            peak_device_memory_bytes=(1200,),
        ),
        BenchmarkSample(
            sample_index=2,
            generation_s=10.0,
            request_count=10,
            prompt_tokens=80,
            generated_tokens=100,
            ttft_s=(0.4, 0.5),
            inter_token_latency_s=(0.04, 0.05),
            output_lengths=(10,) * 10,
            peak_device_memory_bytes=(1800,),
        ),
    ]

    aggregate = aggregate_samples(samples)

    assert aggregate["sample_count"] == 3
    assert aggregate["generation_s"]["median"] == 2.0
    assert aggregate["generation_s"]["p95"] == pytest.approx(9.2)
    assert aggregate["generation_s"]["mad"] == 1.0
    assert aggregate["generated_tokens_per_s"]["median"] == 50.0
    assert aggregate["requests_per_s"]["median"] == 5.0
    assert aggregate["batch_prefill_s"]["median"] == 0.3
    assert aggregate["batch_decode_s"]["median"] == pytest.approx(0.27)
    assert aggregate["ttft_s"]["median"] == 0.25
    assert aggregate["inter_token_latency_s"]["p95"] == pytest.approx(0.0475)
    assert aggregate["peak_device_memory_bytes"]["max"] == 1800


def test_sample_aggregation_marks_single_token_inter_token_latency_undefined() -> None:
    sample = BenchmarkSample(
        sample_index=0,
        generation_s=1.0,
        request_count=1,
        prompt_tokens=1_024,
        generated_tokens=1,
        ttft_s=(1.0,),
        inter_token_latency_s=(),
        output_lengths=(1,),
        peak_device_memory_bytes=(1_000,),
    )

    aggregate = aggregate_samples([sample])

    assert aggregate["inter_token_latency_s"] is None


def test_unmonitored_speed_sample_retains_explicit_missing_memory_aggregate() -> None:
    sample = BenchmarkSample(
        sample_index=0,
        generation_s=1.0,
        request_count=2,
        prompt_tokens=24,
        generated_tokens=6,
        ttft_s=(0.1, 0.1),
        inter_token_latency_s=(0.01, 0.01),
        output_lengths=(3, 3),
        peak_device_memory_bytes=(),
    )

    aggregate = aggregate_samples([sample])

    if sample.peak_device_memory_bytes != () or aggregate["peak_device_memory_bytes"] is not None:
        raise AssertionError("unmonitored speed sample fabricated peak-memory evidence")


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    (
        ("generation_s", True, TypeError),
        ("generation_s", math.nan, ValueError),
        ("generation_s", math.inf, ValueError),
        ("request_count", True, TypeError),
        ("prompt_tokens", 1.0, TypeError),
        ("ttft_s", [0.1], TypeError),
        ("ttft_s", (0.1, True), TypeError),
        ("ttft_s", (0.1, math.nan), ValueError),
        ("inter_token_latency_s", (math.inf,), ValueError),
        ("output_lengths", (1.0,), TypeError),
        ("peak_device_memory_bytes", (True,), TypeError),
    ),
)
def test_benchmark_sample_rejects_numeric_aliases_and_nonfinite_timings(
    field,
    value,
    error_type,
) -> None:
    values = {
        "sample_index": 0,
        "generation_s": 1.0,
        "request_count": 1,
        "prompt_tokens": 1,
        "generated_tokens": 1,
        "ttft_s": (0.1,),
        "inter_token_latency_s": (),
        "output_lengths": (1,),
        "peak_device_memory_bytes": (1_000,),
    }
    values[field] = value

    with pytest.raises(error_type):
        BenchmarkSample(**values)


def test_generation_validation_requires_exact_ids_lengths_prompts_and_finite_logprobs() -> None:
    manifest = WorkloadManifest.from_path(DATA).with_max_new_tokens(3)
    records = tuple(
        GenerationRecord(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=(65, 67, 71),
            output_logprobs=(-0.1, -0.2, -0.3),
            requested_max_tokens=3,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        )
        for request in manifest.requests
    )

    validate_generation_records(manifest, records)

    duplicate = records[:-1] + (records[0],)
    with pytest.raises(AssertionError, match="exactly once"):
        validate_generation_records(manifest, duplicate)
    short = list(records)
    short[0] = GenerationRecord(
        request_id=short[0].request_id,
        prompt_token_ids=short[0].prompt_token_ids,
        output_token_ids=(65, 67),
        output_logprobs=(-0.1, -0.2),
        requested_max_tokens=3,
        finish_reason="length",
        stop_reason=None,
        stopped_on_eos=False,
    )
    with pytest.raises(AssertionError, match="exactly 3"):
        validate_generation_records(manifest, tuple(short))
    nonfinite = list(records)
    nonfinite[0] = GenerationRecord(
        request_id=nonfinite[0].request_id,
        prompt_token_ids=nonfinite[0].prompt_token_ids,
        output_token_ids=nonfinite[0].output_token_ids,
        output_logprobs=(-0.1, math.nan, -0.3),
        requested_max_tokens=3,
        finish_reason="length",
        stop_reason=None,
        stopped_on_eos=False,
    )
    with pytest.raises(AssertionError, match="finite"):
        validate_generation_records(manifest, tuple(nonfinite))


def test_benchmark_cli_requires_reproducible_profile_inputs(tmp_path) -> None:
    args = build_parser().parse_args(
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
            "32768",
            "--gpu-memory-utilization",
            "0.95",
            "--warmups",
            "2",
            "--repetitions",
            "5",
            "--shared-prefix-state-reuse",
            "--global-wave-size",
            "20",
            "--max-num-seqs",
            "20",
            "--common-prefix-identity-case",
            "2",
            "--linked-proof-artifact",
            str(tmp_path / "proof.json"),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.backend == "vllm"
    assert args.topology == "tp2"
    assert args.max_num_batched_tokens == 32_768
    assert args.gpu_memory_utilization == 0.95
    assert args.warmups == 2
    assert args.repetitions == 5
    assert args.proof is False
    assert args.common_prefix_identity_case == 2
    assert args.optimization_level == 2
    assert args.performance_mode == "balanced"
    assert args.load_format == "safetensors"
    assert args.request_count is None
    assert args.uniform_prompt_length is None
    assert args.generation_round == 0
    assert args.prompt_jsonl is None
    assert args.prompt_jsonl_sha256 is None
    assert args.prompt_tokenizer_json is None
    assert args.expected_prompt_tokens is None
    assert args.context_preflight_only is False
    assert args.shared_prefix_state_reuse is True
    assert args.global_wave_size == 20
    assert args.max_num_seqs == 20
    assert args.linked_proof_artifact == tmp_path / "proof.json"


def _fake_vllm_outputs(manifest: WorkloadManifest):
    outputs = []
    for index, request in enumerate(manifest.requests):
        token_ids = (65 + index, 67 + index, 71 + index)
        logprobs = [
            {token_id: SimpleNamespace(logprob=-0.1 * (position + 1))} for position, token_id in enumerate(token_ids)
        ]
        completion = SimpleNamespace(
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason="length",
            stop_reason=None,
        )
        metrics = SimpleNamespace(
            first_token_latency=0.4 + index / 100,
            first_token_ts=10.0,
            last_token_ts=10.2,
            num_generation_tokens=3,
        )
        outputs.append(
            SimpleNamespace(
                request_id=str(index),
                prompt_token_ids=list(request.prompt_token_ids),
                outputs=[completion],
                finished=True,
                metrics=metrics,
            )
        )
    return outputs


def test_vllm_output_adapter_preserves_manifest_identity_and_chosen_logprobs() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)

    records = records_from_vllm_outputs(manifest, _fake_vllm_outputs(manifest))

    validate_generation_records(manifest, records)
    assert [record.request_id for record in records] == ["gdpo-000", "gdpo-001"]
    assert records[0].output_token_ids == (65, 67, 71)
    assert records[0].output_logprobs == pytest.approx((-0.1, -0.2, -0.3))


def test_vllm_output_summary_streams_exact_lengths_and_matches_record_digest() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    outputs = _fake_vllm_outputs(manifest)

    summaries = summarize_vllm_outputs(manifest, outputs)
    records = records_from_vllm_outputs(manifest, outputs)

    assert summaries[0] == records[0].summary_dict()
    assert summaries[1]["request_id"] == "gdpo-001"
    assert summaries[1]["output_length"] == 3
    assert summaries[1]["requested_prompt_tokens"] == len(manifest.requests[1].prompt_token_ids)
    assert summaries[1]["requested_new_tokens"] == 3
    assert summaries[1]["requested_total_tokens"] == len(manifest.requests[1].prompt_token_ids) + 3
    assert summaries[1]["observed_prompt_tokens"] == len(manifest.requests[1].prompt_token_ids)
    assert summaries[1]["observed_new_tokens"] == 3
    assert summaries[1]["observed_total_tokens"] == len(manifest.requests[1].prompt_token_ids) + 3


def test_vllm_output_adapter_rejects_reordered_or_missing_logprob_outputs() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    outputs = _fake_vllm_outputs(manifest)
    outputs.reverse()

    with pytest.raises(AssertionError, match="prompt"):
        records_from_vllm_outputs(manifest, outputs)

    outputs = _fake_vllm_outputs(manifest)
    outputs[0].outputs[0].logprobs[1] = {}
    with pytest.raises(AssertionError, match="chosen-token logprob"):
        records_from_vllm_outputs(manifest, outputs)


def test_benchmark_sample_uses_vllm_request_metrics_without_engine_timing_inflation() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)

    sample = benchmark_sample_from_vllm_outputs(
        manifest,
        _fake_vllm_outputs(manifest),
        sample_index=4,
        generation_s=2.5,
        peak_device_memory_bytes=(1_000, 2_000),
    )

    assert sample.sample_index == 4
    assert sample.generation_s == 2.5
    assert sample.request_count == 2
    assert sample.prompt_tokens == sum(len(request.prompt_token_ids) for request in manifest.requests)
    assert sample.generated_tokens == 6
    assert sample.ttft_s == pytest.approx((0.4, 0.41))
    assert sample.inter_token_latency_s == pytest.approx((0.1, 0.1))
    assert sample.output_lengths == (3, 3)
    assert sample.peak_device_memory_bytes == (1_000, 2_000)
    assert sample.batch_prefill_s == pytest.approx(0.41)
    assert sample.batch_decode_s == pytest.approx(0.2)
    assert sample.to_dict()["batch_prefill_s"] == pytest.approx(0.41)
    assert sample.to_dict()["batch_decode_s"] == pytest.approx(0.2)


def test_sampling_params_match_gdpo_and_force_exact_lengths_without_detokenization() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(25_000)

    kwargs = sampling_params_kwargs(manifest)

    assert kwargs == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 4,
        "max_tokens": 25_000,
        "min_tokens": 25_000,
        "logprobs": 0,
        "stop_token_ids": [],
        "ignore_eos": True,
        "detokenize": False,
    }


def test_compilation_proof_requires_inductor_graph_capture_and_stable_warm_replay() -> None:
    initialized = {
        "num_models_seen": 1,
        "num_backend_compilations": 12,
        "num_inductor_compiles": 12,
        "num_eager_compiles": 0,
        "num_gpu_runner_capture_triggers": 1,
        "num_cudagraph_captured": 63,
        "stock_torch_compile_count": 0,
    }

    validate_compilation_proof(initialized, initialized)

    with pytest.raises(AssertionError, match="eager"):
        validate_compilation_proof({**initialized, "num_eager_compiles": 1}, initialized)
    with pytest.raises(AssertionError, match="recompile"):
        validate_compilation_proof(initialized, {**initialized, "num_inductor_compiles": 13})


def test_compilation_proof_has_no_optimization_sensitive_asserts() -> None:
    source = textwrap.dedent(inspect.getsource(validate_compilation_proof))

    assert not any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(source)))


def test_compilation_proof_rejects_counter_drift_under_optimized_python() -> None:
    source_root = Path(inspect.getfile(validate_compilation_proof)).resolve().parents[3]
    script = """
from bionemo.evo2.vllm.benchmark import validate_compilation_proof

initialized = {
    "num_models_seen": 1,
    "num_backend_compilations": 12,
    "num_inductor_compiles": 12,
    "num_eager_compiles": 0,
    "num_gpu_runner_capture_triggers": 1,
    "num_cudagraph_captured": 63,
    "stock_torch_compile_count": 0,
}
after = {**initialized, "num_inductor_compiles": 999, "num_eager_compiles": 7}
validate_compilation_proof(initialized, after)
"""
    pythonpath = os.pathsep.join(filter(None, (str(source_root), os.environ.get("PYTHONPATH"))))

    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "warm replay entered eager compilation" in result.stderr
