# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import math
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
    records_from_vllm_outputs,
    sampling_params_kwargs,
    validate_compilation_proof,
    validate_generation_records,
)


DATA = Path(__file__).with_name("data") / "gdpo_mixed_96.json"
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
    assert manifest.sha256 == "391647d48dbf06fb2b340eb55bbfc672ea768d2d948513e74dc319fd9789faa9"


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
    assert aggregate["ttft_s"]["median"] == 0.25
    assert aggregate["inter_token_latency_s"]["p95"] == pytest.approx(0.0475)
    assert aggregate["peak_device_memory_bytes"]["max"] == 1800


def test_generation_validation_requires_exact_ids_lengths_prompts_and_finite_logprobs() -> None:
    manifest = WorkloadManifest.from_path(DATA).with_max_new_tokens(3)
    records = tuple(
        GenerationRecord(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=(65, 67, 71),
            output_logprobs=(-0.1, -0.2, -0.3),
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
    )
    with pytest.raises(AssertionError, match="exactly 3"):
        validate_generation_records(manifest, tuple(short))
    nonfinite = list(records)
    nonfinite[0] = GenerationRecord(
        request_id=nonfinite[0].request_id,
        prompt_token_ids=nonfinite[0].prompt_token_ids,
        output_token_ids=nonfinite[0].output_token_ids,
        output_logprobs=(-0.1, math.nan, -0.3),
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


def _fake_vllm_outputs(manifest: WorkloadManifest):
    outputs = []
    for index, request in enumerate(manifest.requests):
        token_ids = (65 + index, 67 + index, 71 + index)
        logprobs = [
            {token_id: SimpleNamespace(logprob=-0.1 * (position + 1))}
            for position, token_id in enumerate(token_ids)
        ]
        completion = SimpleNamespace(token_ids=token_ids, logprobs=logprobs)
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
        "stop_token_ids": None,
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
