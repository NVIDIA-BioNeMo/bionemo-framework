# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bionemo.evo2.vllm.runner as runner
import bionemo.evo2.vllm.sampler as sampler_module
from bionemo.evo2.vllm.accuracy import (
    CANONICAL_7B_CHECKPOINT,
    build_canonical_identity_contract,
    build_canonical_identity_manifest,
    build_common_prefix_identity_contract,
    build_common_prefix_identity_manifest,
    build_homogeneous_identity_schedule,
    load_canonical_7b_identity_cases,
    load_common_prefix_identity_cases,
)
from bionemo.evo2.vllm.benchmark import (
    GenerationRecord,
    WorkloadManifest,
    WorkloadRequest,
    records_from_vllm_outputs,
)
from bionemo.evo2.vllm.runner import (
    CUDAGraphProofRecorder,
    PeakMemoryMonitor,
    build_request_sampling_params,
    prepare_workload,
    request_seed,
    run_generation_phase,
    validate_full_decode_proof,
)
from bionemo.evo2.vllm.tokenizer_io import SnapshotBoundTokenizer


DATA = __import__("pathlib").Path(__file__).with_name("data") / "gdpo_mixed_96.json"
PROMPTS_CSV = Path(__file__).resolve().parent.parent / "data" / "prompts.csv"
TOKENIZER_JSON = Path(__file__).resolve().parents[4] / "tokenizers/nucleotide_fast_tokenizer_512/tokenizer.json"


def _sampler_installation_provenance() -> dict:
    source_files = [
        {
            "path": f"/runtime/{relative_path}",
            "relative_path": relative_path,
            "size_bytes": index + 1,
            "sha256": sha256,
        }
        for index, (relative_path, sha256) in enumerate(sampler_module._EXPECTED_SOURCE_SHA256.items())
    ]
    return {
        "schema_version": 1,
        "executing_python": {
            "invoked_path": "/nemo-rl/.venv/bin/python",
            "resolved_path": "/python3.13",
            "resolved_sha256": "a" * 64,
            "invoked_symlink_target": "/python3.13",
            "nemo_rl_py_executables_system": "1",
        },
        "launcher_selection": {
            "nemo_rl_py_executables_system": "1",
            "selected_environment": "system-runtime",
            "executing_python_path": "/nemo-rl/.venv/bin/python",
            "isolated_worker_environment": {
                "path": "/nemo-rl/venvs/vllm-worker/bin/python",
                "exists": True,
                "status": "installed-but-bypassed",
                "attested_as_executing": False,
            },
        },
        "distributions": dict(sampler_module._EXPECTED_DISTRIBUTIONS),
        "distribution_metadata": {
            name: {"record_path": f"/runtime/{name}.dist-info/RECORD", "record_sha256": str(index) * 64}
            for index, name in enumerate(sampler_module._EXPECTED_DISTRIBUTIONS, start=1)
        },
        "loaded_modules": {},
        "required_sampler_modules_loaded": False,
        "source_files": source_files,
        "source_contract_sha256": hashlib.sha256(
            "".join(item["sha256"] for item in source_files).encode()
        ).hexdigest(),
        "flashinfer_per_row_seed_arrays_supported": False,
        "flashinfer_impact_on_selected_vllm_route": "none-route-bypassed",
        "passed": True,
    }


def _phase_with_coordinator_timing(
    manifest: WorkloadManifest,
    *,
    sample_generation_s: object = 1.0,
    generation_call_s: object = None,
    coordinator_generation_wall_s: object = 1.0,
    generation_timing_authority: object = "coordinator_monotonic_generation_wall",
) -> dict:
    call_timings = [1.0] if generation_call_s is None else generation_call_s
    return {
        "sample": {
            "sample_index": 0,
            "generation_s": sample_generation_s,
            "request_count": len(manifest.requests),
            "prompt_tokens": sum(
                len(request.prompt_token_ids) for request in manifest.requests
            ),
            "generated_tokens": len(manifest.requests) * manifest.max_new_tokens,
            "ttft_s": [0.1] * len(manifest.requests),
            "inter_token_latency_s": [0.01] * len(manifest.requests),
            "output_lengths": [manifest.max_new_tokens] * len(manifest.requests),
            "peak_device_memory_bytes": [1_000],
        },
        "generation_call_s": call_timings,
        "coordinator_generation_wall_s": coordinator_generation_wall_s,
        "generation_timing_authority": generation_timing_authority,
    }


@pytest.mark.parametrize(
    "tamper",
    (
        {"sample_generation_s": True},
        {"sample_generation_s": float("nan")},
        {"generation_call_s": [float("inf")]},
        {"coordinator_generation_wall_s": float("nan")},
        {"generation_timing_authority": "worker-reported"},
    ),
)
def test_phase_sample_rejects_nonfinite_or_noncoordinator_generation_timing(tamper) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 1).with_max_new_tokens(1)
    phase = _phase_with_coordinator_timing(manifest, **tamper)

    with pytest.raises(AssertionError):
        runner._validate_phase_sample(phase, manifest=manifest, sample_index=0)


def test_phase_sample_requires_explicit_coordinator_generation_timing_authority() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 1).with_max_new_tokens(1)
    phase = _phase_with_coordinator_timing(manifest)
    del phase["coordinator_generation_wall_s"]

    with pytest.raises(AssertionError, match="coordinator"):
        runner._validate_phase_sample(phase, manifest=manifest, sample_index=0)


def test_shared_prefix_manifest_uses_generation_prompt_token_ids_v1_digest() -> None:
    base = WorkloadManifest.from_path(DATA).request_slice(0, 2)
    manifest = replace(
        base,
        requests=(
            WorkloadRequest(request_id="shared-43-a", prompt_token_ids=(43,)),
            WorkloadRequest(request_id="shared-43-b", prompt_token_ids=(43,)),
        ),
    )

    evidence = runner.shared_prefix_manifest_evidence(manifest)

    assert evidence["prompt_token_ids_sha256"] == (
        "8fcfb284618fdd1c28d8a7022eee50831e44986fac86e48b396800bf5ba2c93b"
    )


def test_rank_local_generation_contract_uses_v1_prompt_digest() -> None:
    base = WorkloadManifest.from_path(DATA).request_slice(0, 1)
    request = WorkloadRequest(request_id="semantic-43", prompt_token_ids=(43,))
    manifest = replace(base, requests=(request,))
    execution = runner.RequestExecutionRecord(
        request_id=request.request_id,
        global_request_index=0,
        generation_round=0,
        dp_rank=0,
        call_index=0,
        seed=42,
    )
    expected_payload = {
        "schema_version": "evo2-rank-local-generation-contract/v1",
        "phase": "cold-generation",
        "semantic_namespace": "proof/cold",
        "wave_index": 0,
        "manifest_sha256": manifest.sha256,
        "requested_max_new_tokens": manifest.max_new_tokens,
        "requests": [
            {
                "request_id": request.request_id,
                "qualified_request_identity": ["proof/cold", request.request_id],
                "prompt_token_ids_sha256": (
                    "8fcfb284618fdd1c28d8a7022eee50831e44986fac86e48b396800bf5ba2c93b"
                ),
                "execution": execution.to_dict(),
            }
        ],
    }

    observed = runner.rank_local_generation_contract_sha256(
        "cold-generation",
        "proof/cold",
        0,
        manifest,
        (execution,),
    )

    assert observed == hashlib.sha256(
        json.dumps(expected_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class _RunnerTestNativeTopKSampler:
    __module__ = "vllm.v1.sample.ops.topk_topp_sampler"
    logprobs_mode = "processed_logprobs"

    def __init__(self) -> None:
        self.forward = self.forward_native

    def forward_native(self, logits, generators, k, p):
        del k, p
        q = torch.empty_like(logits)
        for row_index, generator in generators.items():
            q[row_index].exponential_(generator=generator)
        logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
        return logprobs.exp().div_(q).argmax(dim=-1), logprobs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _RunnerTestV1ModelRunner:
    __module__ = "vllm.v1.worker.gpu_model_runner"

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.sampler = SimpleNamespace(topk_topp_sampler=_RunnerTestNativeTopKSampler())


_SAMPLER_BEHAVIORAL_PREFLIGHT = [None]


def _sampler_worker_proof(
    seed_batches: tuple[tuple[int, ...], ...],
    *,
    accepted_output_token_count: int = 3,
) -> dict:
    environment = sampler_module.sampler_runtime_environment_contract()
    if _SAMPLER_BEHAVIORAL_PREFLIGHT[0] is None:
        worker = SimpleNamespace(model_runner=_RunnerTestV1ModelRunner())
        _SAMPLER_BEHAVIORAL_PREFLIGHT[0] = sampler_module.run_sampler_seed_behavioral_preflight(worker)
    installation = _sampler_installation_provenance()
    installation["required_sampler_modules_loaded"] = True
    installation["loaded_modules"] = {
        "vllm.v1.worker.gpu_model_runner": {
            "loaded": True,
            "path": "/runtime/vllm/v1/worker/gpu_model_runner.py",
            "sha256": sampler_module._EXPECTED_SOURCE_SHA256["vllm/v1/worker/gpu_model_runner.py"],
        },
        "vllm.v1.sample.ops.topk_topp_sampler": {
            "loaded": True,
            "path": "/runtime/vllm/v1/sample/ops/topk_topp_sampler.py",
            "sha256": sampler_module._EXPECTED_SOURCE_SHA256["vllm/v1/sample/ops/topk_topp_sampler.py"],
        },
        "flashinfer": {"loaded": False, "path": None, "sha256": None},
        "flashinfer.sampling": {"loaded": False, "path": None, "sha256": None},
    }
    return {
        "schema_version": 4,
        "environment_contract": environment,
        "model_runner_type": "vllm.v1.worker.gpu_model_runner._RunnerTestV1ModelRunner",
        "sampler_type": "vllm.v1.sample.ops.topk_topp_sampler._RunnerTestNativeTopKSampler",
        "selected_route": "vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler.forward_native",
        "installation": installation,
        "behavioral_preflight": copy.deepcopy(_SAMPLER_BEHAVIORAL_PREFLIGHT[0]),
        "generation_batch_descriptors": [
            {
                "descriptor_index": descriptor_index,
                "batch_rows": len(seeds),
                "generator_count": len(seeds),
                "generator_indices": list(range(len(seeds))),
                "generator_seeds": list(seeds),
                "sampler_call_count": accepted_output_token_count,
                "full_row_identity_verified": True,
                "full_row_identity_verification_mode": "generation-call-boundary-full-row",
                "initial_reference_state_sha256": [
                    hashlib.sha256(f"initial:{seed}".encode()).hexdigest() for seed in seeds
                ],
                "phase_end_state_sha256": [
                    hashlib.sha256(f"final:{seed}".encode()).hexdigest() for seed in seeds
                ],
                "persistent_generator_identity": True,
                "state_changed_from_seeded_initial": True,
                "route": "vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler.forward_native",
                "passed": True,
            }
            for descriptor_index, seeds in enumerate(seed_batches)
        ],
        "flashinfer_sampling_calls": 0,
        "passed": True,
    }


def _canonical_base_manifest() -> WorkloadManifest:
    return WorkloadManifest(
        schema_version=1,
        name="canonical-identity-base",
        source_checkpoint=CANONICAL_7B_CHECKPOINT,
        checkpoint_manifest_sha256="1" * 64,
        checkpoint_index_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        requests=(WorkloadRequest(request_id="base", prompt_token_ids=(1,)),),
        max_new_tokens=1,
        temperature=0.5,
        top_p=0.5,
        top_k=4,
        seed=7,
        dtype="bfloat16",
        ignore_eos=False,
        stop_token_ids=(0,),
    )


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
    num_generation_tokens: int = 0,
    first_token_events: int = 0,
):
    return SimpleNamespace(
        num_preempted_reqs=preempted,
        prompt_token_stats=SimpleNamespace(computed=computed, cached_tokens=cached_tokens, total=total),
        num_generation_tokens=num_generation_tokens,
        time_to_first_tokens_iter=[0.0] * first_token_events,
    )


def test_cudagraph_recorder_persists_phase_and_runtime_mode_without_log_resets() -> None:
    recorder = CUDAGraphProofRecorder()
    recorder.start_phase("steady-0")
    recorder.record(
        _scheduler_stats(unpadded=768, padded=768, mode="CUDAGraphMode.PIECEWISE"),
        _iteration_stats(computed=768, total=768, num_generation_tokens=4, first_token_events=4),
        engine_idx=0,
    )
    recorder.record(
        _scheduler_stats(unpadded=96, padded=96, mode="CUDAGraphMode.FULL"),
        _iteration_stats(num_generation_tokens=96),
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
            "request_dimensions": {
                "schema_version": 1,
                "source": "iteration-stats-bound-to-cudagraph-dispatch",
                "prefill_req_count": None,
                "decode_req_count": None,
                "token_count": 768,
                "prompt_token_count": 768,
                "first_token_event_count": 4,
            },
        },
        {
            "phase": "steady-0",
            "engine_index": 0,
            "num_unpadded_tokens": 96,
            "num_padded_tokens": 96,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
            "request_dimensions": {
                "schema_version": 1,
                "source": "iteration-stats-bound-to-cudagraph-dispatch",
                "prefill_req_count": 0,
                "decode_req_count": 96,
                "token_count": 96,
                "prompt_token_count": 0,
                "first_token_event_count": 0,
            },
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
    speed_profile = runner.profile_from_args(speed_args, manifest)
    speed_contract = runner.build_benchmark_contract(
        speed_args,
        manifest,
        speed_profile,
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
            caller_coordinates=runner.CallerCoordinateContract.from_inputs(
                manifest,
                speed_profile,
                speed_args.generation_round,
            ),
        )


def test_benchmark_mode_accepts_unlinked_speed_and_rejects_doubly_attested_proof(tmp_path) -> None:
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

    if runner.benchmark_mode_from_args(runner.build_parser().parse_args(common)) != "speed":
        raise AssertionError("unlinked proof=false execution must use the low-overhead speed path")
    with pytest.raises(ValueError, match="cannot link"):
        runner.benchmark_mode_from_args(
            runner.build_parser().parse_args(
                [*common, "--proof", "--linked-proof-artifact", str(tmp_path / "proof.json")]
            )
        )


def test_benchmark_instrumentation_contract_distinguishes_proof_and_speed() -> None:
    proof = runner.benchmark_instrumentation_contract("proof")
    speed = runner.benchmark_instrumentation_contract("speed")

    expected_proof = {
        "scheduler_callbacks_during_generation": True,
        "worker_proof_rpcs": True,
        "sampler_call_counter_during_generation": True,
        "sampler_endpoint_state_hashing_during_generation": False,
        "prefix_clone_instrumentation": True,
        "peak_memory_polling_during_generation": True,
        "post_generation_exact_output_validation": True,
    }
    expected_speed = {
        "scheduler_callbacks_during_generation": False,
        "worker_proof_rpcs": False,
        "sampler_call_counter_during_generation": False,
        "sampler_endpoint_state_hashing_during_generation": False,
        "prefix_clone_instrumentation": False,
        "peak_memory_polling_during_generation": False,
        "post_generation_exact_output_validation": True,
    }
    if proof != expected_proof or speed != expected_speed:
        raise AssertionError("proof and speed instrumentation contracts are not explicit and disjoint")


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


def test_exact_generation_progress_rejects_one_batch_decode_shortfall() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    observation = {
        "phase": "steady-0.wave-000",
        "engine_index": 0,
        "num_unpadded_tokens": 2,
        "num_padded_tokens": 2,
        "num_paddings": 0,
        "runtime_mode": "CUDAGraphMode.FULL",
    }
    exact = runner.full_decode_proof_summary(
        [dict(observation), dict(observation)],
        phase="steady-0.wave-000",
        batch_size=2,
        max_new_tokens=3,
    )
    evidence = runner.exact_generation_progress_evidence(
        manifest,
        sidecar_counts={
            "request_count": 2,
            "output_token_id_count": 6,
            "chosen_token_logprob_count": 6,
        },
        full_decode_summaries=[exact],
    )

    assert evidence == {
        "schema_version": 1,
        "request_count": 2,
        "prefill_request_count": 2,
        "first_sampled_token_count": 2,
        "decode_steps_per_request": 2,
        "expected_decode_token_updates": 4,
        "observed_full_decode_token_updates": 4,
        "retained_output_token_id_count": 6,
        "retained_chosen_token_logprob_count": 6,
        "passed": True,
    }

    tolerated_by_general_long_gate = runner.full_decode_proof_summary(
        [dict(observation)],
        phase="steady-0.wave-000",
        batch_size=2,
        max_new_tokens=3,
    )
    assert tolerated_by_general_long_gate["passed"] is True
    with pytest.raises(AssertionError, match="exact decode"):
        runner.exact_generation_progress_evidence(
            manifest,
            sidecar_counts={
                "request_count": 2,
                "output_token_id_count": 6,
                "chosen_token_logprob_count": 6,
            },
            full_decode_summaries=[tolerated_by_general_long_gate],
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("request_count", 1),
        ("output_token_id_count", 5),
        ("chosen_token_logprob_count", 5),
    ),
)
def test_exact_generation_progress_rejects_truncated_retained_outputs(field, value) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(1)
    counts = {
        "request_count": 2,
        "output_token_id_count": 2,
        "chosen_token_logprob_count": 2,
    }
    counts[field] = value

    with pytest.raises(AssertionError, match="retained"):
        runner.exact_generation_progress_evidence(
            manifest,
            sidecar_counts=counts,
            full_decode_summaries=[],
        )


@pytest.mark.parametrize("topology", ("tp2", "dp2"))
def test_exact_progress_artifacts_cover_proof_topologies_without_timed_callbacks(topology) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)

    def summary(phase: str, batch_size: int):
        observation = {
            "phase": phase,
            "engine_index": 0,
            "num_unpadded_tokens": batch_size,
            "num_padded_tokens": batch_size,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        }
        return runner.full_decode_proof_summary(
            [dict(observation), dict(observation)],
            phase=phase,
            batch_size=batch_size,
            max_new_tokens=3,
        )

    if topology == "tp2":
        physical = {"wave_proofs": [{"full_decode_proof": summary("steady-0.wave-000", 2)}]}
    else:
        physical = {
            "waves": [
                {
                    "engines": [
                        {"full_decode_proof": summary("steady-0.wave-000", 1)},
                        {"full_decode_proof": summary("steady-0.wave-000", 1)},
                    ]
                }
            ]
        }
    phases, aggregate = runner.attach_exact_generation_progress_evidence(
        [
            {
                "phase": "steady-0",
                "full_output_artifact": {
                    "request_count": 2,
                    "output_token_id_count": 6,
                    "chosen_token_logprob_count": 6,
                },
                **physical,
            }
        ],
        manifest=manifest,
        enabled=True,
        proof_collected=True,
        topology=topology,
        linked_proof_artifact=None,
    )

    assert phases[0]["exact_generation_progress"]["observed_full_decode_token_updates"] == 4
    assert aggregate["phase_count"] == 1
    assert aggregate["execution_proof_source"] == "current-proof-run"
    assert aggregate["passed"] is True


def _exact_1k_contract(topology: str, *, global_wave_size: int) -> dict:
    manifest = WorkloadManifest.from_path(DATA).with_request_count(1_000, request_id_prefix="audit")
    max_num_seqs = global_wave_size if topology == "tp2" else global_wave_size // 2
    args = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            "/checkpoint",
            "--manifest",
            str(DATA),
            "--topology",
            topology,
            "--max-model-len",
            "64",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.95",
            "--global-wave-size",
            str(global_wave_size),
            "--max-num-seqs",
            str(max_num_seqs),
            "--exact-progress-gate",
            "--proof",
            "--output",
            "/tmp/exact-1k-proof.json",
        ]
    )
    profile = runner.profile_from_args(args, manifest)
    return runner.build_benchmark_contract(args, manifest, profile)


def _mbs1_exact_1k_inputs(topology: str):
    manifest = (
        WorkloadManifest.from_path(DATA)
        .request_slice(0, 1)
        .with_request_count(1_000, request_id_prefix="mbs1")
        .with_max_new_tokens(3)
    )
    max_num_seqs = 96 if topology == "tp2" else 48
    args = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            "/checkpoint",
            "--manifest",
            str(DATA),
            "--topology",
            topology,
            "--max-model-len",
            "64",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.95",
            "--global-wave-size",
            "96",
            "--max-num-seqs",
            str(max_num_seqs),
            "--exact-progress-gate",
            "--shared-prefix-state-reuse",
            "--mbs1-exact1k-audit",
            "--proof",
            "--output",
            "/tmp/mbs1-exact-1k-proof.json",
        ]
    )
    return args, manifest, runner.profile_from_args(args, manifest)


@pytest.mark.parametrize(
    ("topology", "expected_local_counts", "expected_first_wave_misses"),
    (
        ("tp2", [[96]] * 10 + [[40]], 1),
        ("dp2", [[48, 48]] * 10 + [[20, 20]], 2),
    ),
)
def test_mbs1_exact_1k_contract_seals_one_prompt_and_physical_schedule(
    topology: str,
    expected_local_counts: list[list[int]],
    expected_first_wave_misses: int,
) -> None:
    args, manifest, profile = _mbs1_exact_1k_inputs(topology)

    contract = runner.build_benchmark_contract(args, manifest, profile)["mbs1_exact1k"]

    expected_prompt = runner.shared_prefix_manifest_evidence(manifest)
    if contract["prompt_identity"] != expected_prompt:
        raise AssertionError("MBS=1 contract did not bind the exact repeated prompt")
    if contract["semantic_request_count"] != 1_000 or contract["unique_semantic_request_ids"] is not True:
        raise AssertionError("MBS=1 contract collapsed or omitted semantic requests")
    if contract["global_call_request_counts"] != [96] * 10 + [40]:
        raise AssertionError("MBS=1 contract did not retain 10x96 plus tail40")
    if contract["per_engine_call_request_counts"] != expected_local_counts:
        raise AssertionError("MBS=1 contract did not retain exact local physical shapes")
    if contract["expected_first_wave_cache_misses"] != expected_first_wave_misses:
        raise AssertionError("MBS=1 contract did not require one primer per replica")


def test_mbs1_exact_1k_contract_rejects_distinct_prompt_diagnostic() -> None:
    args, _manifest, profile = _mbs1_exact_1k_inputs("tp2")
    distinct = WorkloadManifest.from_path(DATA).with_request_count(1_000, request_id_prefix="diagnostic")

    with pytest.raises(AssertionError, match="identical prompt"):
        runner.build_benchmark_contract(args, distinct, profile)


def test_benchmark_phase_coordinates_advance_round_calls_requests_and_seeds() -> None:
    _args, manifest, profile = _mbs1_exact_1k_inputs("tp2")

    coordinates = runner.benchmark_phase_coordinates(
        manifest,
        profile,
        generation_round_start=3,
        warmups=1,
        repetitions=2,
    )

    if [row["phase"] for row in coordinates] != [
        "cold-generation",
        "warmup-0",
        "steady-0",
        "steady-1",
    ]:
        raise AssertionError("benchmark phases changed")
    if [row["generation_round"] for row in coordinates] != [3, 4, 5, 6]:
        raise AssertionError("benchmark phases reused a semantic generation round")
    if [row["global_call_index_start"] for row in coordinates] != [33, 44, 55, 66]:
        raise AssertionError("benchmark phases reused physical call coordinates")
    if [row["global_request_index_start"] for row in coordinates] != [3_000, 4_000, 5_000, 6_000]:
        raise AssertionError("benchmark phases reused global request coordinates")
    first_records = runner.build_wave_execution_records(
        manifest,
        global_wave_size=96,
        generation_round=coordinates[0]["generation_round"],
        call_index_start=coordinates[0]["global_call_index_start"],
        global_request_index_start=coordinates[0]["global_request_index_start"],
    )
    second_records = runner.build_wave_execution_records(
        manifest,
        global_wave_size=96,
        generation_round=coordinates[1]["generation_round"],
        call_index_start=coordinates[1]["global_call_index_start"],
        global_request_index_start=coordinates[1]["global_request_index_start"],
    )
    if {record.seed for record in first_records} & {record.seed for record in second_records}:
        raise AssertionError("consecutive benchmark phases reused stochastic request seeds")
    if {record.global_request_index for record in first_records} & {
        record.global_request_index for record in second_records
    }:
        raise AssertionError("consecutive benchmark phases reused global request ownership")


def test_mbs1_exact_1k_phase_timing_separates_cold_warm_repeat_and_tail() -> None:
    _args, manifest, profile = _mbs1_exact_1k_inputs("tp2")
    coordinates = runner.benchmark_phase_coordinates(
        manifest,
        profile,
        generation_round_start=0,
        warmups=0,
        repetitions=1,
    )
    phase_coordinate = coordinates[0]
    executions = runner.build_wave_execution_records(
        manifest,
        global_wave_size=96,
        generation_round=phase_coordinate["generation_round"],
        call_index_start=phase_coordinate["global_call_index_start"],
        global_request_index_start=phase_coordinate["global_request_index_start"],
    )
    request_counts = [96] * 10 + [40]
    phase = {
        "phase": "cold-generation",
        "wave_proofs": [
            {
                "wave_index": wave_index,
                "request_count": request_count,
                "generation_round": phase_coordinate["generation_round"],
                "call_index": phase_coordinate["global_call_index_start"] + wave_index,
                "generation_s": float(wave_index + 1),
            }
            for wave_index, request_count in enumerate(request_counts)
        ],
        "request_executions": [record.to_dict() for record in executions],
        "shared_prefix_state_reuse": None,
        "prefix_cache_reset": True,
        "proof_collected": False,
    }

    evidence = runner.mbs1_exact1k_phase_evidence(
        phase,
        manifest=manifest,
        profile=profile,
        phase_coordinate=phase_coordinate,
        proof_collected=False,
        linked_proof_artifact=Path("/tmp/linked-proof.json"),
    )

    if evidence["first_96"]["generation_s"] != 1.0:
        raise AssertionError("first-96 timing did not include the first physical call")
    if evidence["first_96"]["includes_prefix_materialization"] is not True:
        raise AssertionError("first-96 timing omitted prefix materialization")
    if evidence["warmed_repeat_96"]["generation_s"] != 2.0:
        raise AssertionError("warmed repeat timing did not use the second physical call")
    if evidence["tail_40"]["generation_s"] != 11.0:
        raise AssertionError("tail timing did not use the exact final 40-request call")
    if evidence["exact_1000_generation_s"] != 66.0:
        raise AssertionError("exact-1k generation wall did not include all eleven calls")
    if evidence["token_equality_between_phases_required"] is not False:
        raise AssertionError("MBS=1 timing incorrectly required cold/warm token equality")


@pytest.mark.parametrize("topology", ("tp2", "dp2"))
def test_exact_1k_contract_rejects_noncanonical_global_wave_size(topology) -> None:
    with pytest.raises(ValueError, match="1,000-request audit requires global_wave_size=96"):
        _exact_1k_contract(topology, global_wave_size=80)


@pytest.mark.parametrize(
    "topology,expected_local_counts",
    (
        ("tp2", [[96]] * 10 + [[40]]),
        ("dp2", [[48, 48]] * 10 + [[20, 20]]),
    ),
)
def test_exact_1k_contract_freezes_10x96_plus_global40_tail(topology, expected_local_counts) -> None:
    contract = _exact_1k_contract(topology, global_wave_size=96)

    schedule = contract["exact_generation_progress"]["physical_schedule"]
    assert schedule["generation_round"] == 0
    assert schedule["global_call_index_start"] == 0
    assert schedule["global_wave_size"] == 96
    assert schedule["physical_calls_per_round"] == 11
    assert schedule["global_call_request_counts"] == [96] * 10 + [40]
    assert schedule["per_engine_call_request_counts"] == expected_local_counts
    assert [call["call_in_round"] for call in schedule["calls"]] == list(range(11))
    assert [call["global_call_index"] for call in schedule["calls"]] == list(range(11))
    assert [call["global_request_count"] for call in schedule["calls"]] == [96] * 10 + [40]
    assert [
        [replica["local_request_count"] for replica in call["replicas"]] for call in schedule["calls"]
    ] == expected_local_counts


@pytest.mark.parametrize(
    "bad_value",
    (0.0, np.int64(0), "0", False, float("nan"), float("inf")),
)
def test_exact_1k_contract_rejects_non_builtin_raw_coordinate_values(bad_value: object) -> None:
    contract = _exact_1k_contract("dp2", global_wave_size=96)
    progress = copy.deepcopy(contract["exact_generation_progress"])
    progress["physical_schedule"]["calls"][0]["replicas"][0]["dp_rank"] = bad_value

    with pytest.raises(AssertionError, match="built-in integer"):
        runner.validate_exact_generation_progress_contract(
            progress,
            manifest=WorkloadManifest.from_path(DATA).with_request_count(1_000, request_id_prefix="audit"),
            profile=runner.Evo2VllmProfile(
                topology="dp2",
                max_model_len=64,
                max_num_batched_tokens=16_384,
                gpu_memory_utilization=0.95,
                proof=True,
                global_wave_size=96,
                max_num_seqs=48,
            ),
            generation_round=0,
        )


@pytest.mark.parametrize(
    "field",
    (
        "generation_round",
        "global_call_index_start",
        "global_wave_size",
        "physical_calls_per_round",
    ),
)
def test_exact_1k_contract_rejects_integral_float_schedule_summaries(field: str) -> None:
    contract = _exact_1k_contract("tp2", global_wave_size=96)
    progress = copy.deepcopy(contract["exact_generation_progress"])
    progress["physical_schedule"][field] = float(progress["physical_schedule"][field])

    with pytest.raises(AssertionError, match="built-in integer"):
        runner.validate_exact_generation_progress_contract(
            progress,
            manifest=WorkloadManifest.from_path(DATA).with_request_count(1_000, request_id_prefix="audit"),
            profile=runner.Evo2VllmProfile(
                topology="tp2",
                max_model_len=64,
                max_num_batched_tokens=16_384,
                gpu_memory_utilization=0.95,
                proof=True,
                global_wave_size=96,
                max_num_seqs=96,
            ),
            generation_round=0,
        )


def test_exact_progress_speed_artifact_retains_counts_and_links_physical_proof(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    proof_path = tmp_path / "proof.json"
    phases, aggregate = runner.attach_exact_generation_progress_evidence(
        [
            {
                "phase": "steady-0",
                "full_output_artifact": {
                    "request_count": 2,
                    "output_token_id_count": 6,
                    "chosen_token_logprob_count": 6,
                },
                "wave_proofs": [{"full_decode_proof": None}],
            }
        ],
        manifest=manifest,
        enabled=True,
        proof_collected=False,
        topology="tp2",
        linked_proof_artifact=proof_path,
    )

    evidence = phases[0]["exact_generation_progress"]
    assert evidence["retained_output_token_id_count"] == 6
    assert evidence["observed_full_decode_token_updates"] is None
    assert evidence["execution_proof_source"] == str(proof_path.resolve())
    assert aggregate["execution_proof_source"] == str(proof_path.resolve())


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


def _seed_capacity_manifest(*, request_count: int, seed: int) -> WorkloadManifest:
    base = WorkloadManifest.from_path(DATA).request_slice(0, 1)
    prompt_token_ids = base.requests[0].prompt_token_ids
    return replace(
        base,
        requests=tuple(
            WorkloadRequest(
                request_id=f"seed-capacity-{index}",
                prompt_token_ids=prompt_token_ids,
            )
            for index in range(request_count)
        ),
        seed=seed,
    )


def _seed_capacity_profile(topology: str) -> runner.Evo2VllmProfile:
    return runner.Evo2VllmProfile(
        topology=topology,
        max_model_len=64,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.95,
        global_wave_size=96,
        max_num_seqs=96 if topology == "tp2" else 48,
    )


def test_request_seed_rejects_modulus_wrap_instead_of_reusing_zero() -> None:
    last_seed = request_seed(
        2**31 - 1,
        call_index=0,
        dp_rank=0,
        dp_size=1,
        request_index_in_stream=0,
    )
    if last_seed != 2**31 - 1:
        raise AssertionError("the final nonwrapping request seed changed")

    with pytest.raises(ValueError, match="wraparound"):
        request_seed(
            2**31,
            call_index=0,
            dp_rank=0,
            dp_size=1,
            request_index_in_stream=0,
        )


@pytest.mark.parametrize(
    ("topology", "maximum_coordinate_offset"),
    (("tp2", 10 * 1_000_003 + 39), ("dp2", 21 * 1_000_003 + 19)),
)
def test_caller_seed_capacity_accepts_last_value_and_rejects_one_over(
    topology: str,
    maximum_coordinate_offset: int,
) -> None:
    last_valid_base_seed = 2**31 - 1 - maximum_coordinate_offset
    profile = _seed_capacity_profile(topology)
    accepted = runner.CallerCoordinateContract.from_inputs(
        _seed_capacity_manifest(request_count=1_000, seed=last_valid_base_seed),
        profile,
        generation_round=0,
    )
    capacity = accepted.seed_stream_contract()["capacity_proof"]
    if capacity["maximum_pre_modulo_seed"] != 2**31 - 1:
        raise AssertionError("caller seed-capacity proof did not reach the exact last valid seed")
    if max(record.seed for record in accepted.expected_phase_executions()) != 2**31 - 1:
        raise AssertionError("execution records did not retain the exact last valid seed")

    with pytest.raises(ValueError, match="wraparound"):
        runner.CallerCoordinateContract.from_inputs(
            _seed_capacity_manifest(request_count=1_000, seed=last_valid_base_seed + 1),
            profile,
            generation_round=0,
        )


@pytest.mark.parametrize(
    ("topology", "first_local_counts", "tail_local_counts"),
    (("tp2", [96], [40]), ("dp2", [48, 48], [20, 20])),
)
def test_exact1k_seed_capacity_covers_first_second_and_tail_without_collisions(
    topology: str,
    first_local_counts: list[int],
    tail_local_counts: list[int],
) -> None:
    caller = runner.CallerCoordinateContract.from_inputs(
        _seed_capacity_manifest(request_count=1_000, seed=42),
        _seed_capacity_profile(topology),
        generation_round=0,
    )
    capacity = caller.seed_stream_contract()["capacity_proof"]
    calls = capacity["physical_calls"]
    if len(calls) != 11:
        raise AssertionError("exact-1k seed proof must contain ten full calls plus one tail")
    if [row["local_request_count"] for row in calls[0]["replicas"]] != first_local_counts:
        raise AssertionError("first full call has the wrong physical seed-stream geometry")
    if [row["local_request_count"] for row in calls[1]["replicas"]] != first_local_counts:
        raise AssertionError("second full call has the wrong physical seed-stream geometry")
    if [row["local_request_count"] for row in calls[-1]["replicas"]] != tail_local_counts:
        raise AssertionError("tail call has the wrong physical seed-stream geometry")

    seeds_by_call: dict[int, set[int]] = {}
    for record in caller.expected_phase_executions():
        seeds_by_call.setdefault(record.call_index, set()).add(record.seed)
    first = seeds_by_call[0]
    second = seeds_by_call[1]
    tail = seeds_by_call[10]
    if (len(first), len(second), len(tail)) != (96, 96, 40):
        raise AssertionError("exact-1k full/full/tail calls did not retain every unique seed")
    if first & second or first & tail or second & tail:
        raise AssertionError("exact-1k first, second, and tail calls reused a request seed")
    if max(first | second | tail) >= 2**31:
        raise AssertionError("exact-1k request seeds exceeded the admitted nonwrapping domain")


def test_tp2_seed_capacity_rejects_before_context_or_gpu_preflight(tmp_path, monkeypatch) -> None:
    output = tmp_path / "seed-capacity-preflight.json"
    args = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--manifest",
            str(DATA),
            "--topology",
            "tp2",
            "--max-model-len",
            "7000",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.92",
            "--proof",
            "--output",
            str(output),
        ]
    )
    manifest = replace(WorkloadManifest.from_path(DATA), seed=2**31)
    runner.reserve_output_namespace(output)
    forbidden_calls: list[str] = []

    def forbidden(stage: str):
        def fail(*_args, **_kwargs):
            forbidden_calls.append(stage)
            raise AssertionError(f"seed-capacity rejection reached {stage}")

        return fail

    monkeypatch.setattr(runner, "context_length_preflight", forbidden("context preflight"))
    monkeypatch.setattr(runner, "gpu_hardware_provenance", forbidden("GPU preflight"))

    with pytest.raises(ValueError, match="wraparound"):
        runner.run_tp2_benchmark(args, manifest)
    if forbidden_calls:
        raise AssertionError(f"seed-capacity rejection performed forbidden work: {forbidden_calls}")


def test_request_sampling_params_consume_persisted_stream_seeds() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    records = runner.build_request_execution_records(
        manifest,
        global_request_offset=48,
        dp_rank=1,
        dp_size=2,
        generation_round=7,
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


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("temperature", 0.0),
        ("top_p", 0.5),
        ("top_k", 0),
        ("max_tokens", 2),
        ("min_tokens", 2),
        ("logprobs", None),
        ("ignore_eos", False),
        ("detokenize", True),
        ("seed", 7),
    ),
)
def test_request_sampling_params_reject_constructor_policy_normalization(field: str, replacement: object) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 1).with_max_new_tokens(3)
    records = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )

    def mutating_factory(**kwargs):
        return SimpleNamespace(**{**kwargs, field: replacement})

    with pytest.raises(RuntimeError, match=f"sampling parameter {field}"):
        build_request_sampling_params(
            manifest,
            sampling_params_factory=mutating_factory,
            execution_records=records,
        )


def test_request_execution_records_persist_round_rank_call_and_global_seed() -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)

    records = runner.build_request_execution_records(
        manifest,
        global_request_offset=48,
        dp_rank=1,
        dp_size=2,
        generation_round=7,
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


@pytest.mark.parametrize(
    "bad_value",
    (0.0, np.int64(0), "0", False, float("nan"), float("inf")),
)
@pytest.mark.parametrize(
    "field",
    ("global_request_index", "generation_round", "dp_rank", "call_index", "seed"),
)
def test_retained_request_execution_rows_reject_non_builtin_coordinates(field: str, bad_value: object) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 1)
    expected = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    retained = [expected[0].to_dict()]
    retained[0][field] = bad_value

    with pytest.raises(AssertionError, match="built-in integer"):
        runner._validate_request_execution_rows(retained, expected=expected)


@pytest.mark.parametrize(
    "bad_value",
    (0.0, np.int64(0), "0", False, float("nan"), float("inf")),
)
def test_retained_physical_wave_rejects_non_builtin_coordinate_values(bad_value: object) -> None:
    expected = {
        "wave_index": 0,
        "start": 0,
        "stop": 96,
        "request_count": 96,
        "generation_round": 0,
        "call_index": 0,
    }
    retained = dict(expected)
    retained["wave_index"] = bad_value

    with pytest.raises(AssertionError, match="built-in integer"):
        runner._validate_exact_wave_coordinates(retained, expected=expected)


def test_full_output_artifact_round_trips_every_token_logprob_and_seed(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=48,
        dp_rank=1,
        dp_size=2,
        generation_round=9,
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
    assert rows[1]["output_token_ids"] == [67, 71, 84]
    publication_receipt = metadata["publication_receipt"]
    assert publication_receipt["final_path"] == str(output_path.resolve())
    assert publication_receipt["sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert metadata == {
        "schema_version": 2,
        "format": "jsonl",
        "compression": "gzip",
        "path": str(output_path.resolve()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "size_bytes": output_path.stat().st_size,
        "publication_receipt": publication_receipt,
        "request_count": 2,
        "generated_token_count": 6,
        "output_token_id_count": 6,
        "chosen_token_logprob_count": 6,
    }


def test_backend_neutral_full_output_artifact_accepts_generation_records(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=48,
        dp_rank=1,
        dp_size=2,
        generation_round=9,
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


def test_full_output_writer_rejects_decoded_non_dna_or_token_id_mismatch(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 1).with_max_new_tokens(3)
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    records = records_from_vllm_outputs(manifest, _fake_outputs(manifest))
    output_path = tmp_path / "invalid-dna.outputs.jsonl.gz"

    with pytest.raises(AssertionError, match="decoded output must exactly match A/C/G/T token IDs"):
        runner.write_full_generation_records_artifact(
            output_path,
            records=records,
            execution_records=execution_records,
            decode_output_token_ids=lambda _token_ids: "ACN",
        )

    assert not output_path.exists()


def test_full_output_writer_never_overwrites_foreign_sidecar_created_before_publish(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    executions = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    records = records_from_vllm_outputs(manifest, _fake_outputs(manifest))
    output = tmp_path / "foreign.outputs.jsonl.gz"
    foreign = b"foreign-sidecar\n"
    output.write_bytes(foreign)

    with pytest.raises(FileExistsError):
        runner.write_full_generation_records_artifact(
            output,
            records=records,
            execution_records=executions,
        )

    assert output.read_bytes() == foreign


def test_full_output_writer_removes_only_its_own_sidecar_after_ownership_loss(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    executions = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    records = records_from_vllm_outputs(manifest, _fake_outputs(manifest))
    output = tmp_path / "owned.outputs.jsonl.gz"
    calls = 0

    def lose_ownership_after_publish() -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("ownership lost")

    with pytest.raises(RuntimeError, match="ownership lost"):
        runner.write_full_generation_records_artifact(
            output,
            records=records,
            execution_records=executions,
            ownership_validator=lose_ownership_after_publish,
        )

    assert calls == 5
    assert not output.exists()
    assert not output.with_suffix(f"{output.suffix}.tmp").exists()


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
    with pytest.raises(FileExistsError, match=r"proof.inprogress"):
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
    with pytest.raises(FileExistsError, match=r"benchmark.json"):
        runner.reserve_output_namespace(output)

    output.unlink()
    sidecar = runner.phase_output_artifact_path(output, phase="steady-0")
    sidecar.write_bytes(b"stale sidecar")
    with pytest.raises(FileExistsError, match="steady-0"):
        runner.reserve_output_namespace(output)
    assert unrelated.read_bytes() == b"unrelated"


def test_output_namespace_rejects_foreign_replacement_inode_without_unlinking_it(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    marker = runner.reserve_output_namespace(output)
    reserved_identity = (marker.stat().st_dev, marker.stat().st_ino)
    foreign = tmp_path / "foreign-marker"
    foreign_payload = "foreign owner\n"
    foreign.write_text(foreign_payload, encoding="utf-8")
    foreign_identity = (foreign.stat().st_dev, foreign.stat().st_ino)
    assert foreign_identity != reserved_identity
    foreign.replace(marker)

    with pytest.raises(RuntimeError, match="ownership"):
        runner.require_output_namespace_reservation(output)
    with pytest.raises(RuntimeError, match="ownership"):
        runner.complete_output_namespace(marker, output_path=output, require_final_artifact=False)

    assert marker.read_text(encoding="utf-8") == foreign_payload


def test_output_namespace_rejects_self_consistent_foreign_inode_metadata(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    marker = runner.reserve_output_namespace(output)
    foreign = tmp_path / "foreign-marker"
    foreign.touch()
    foreign_stat = foreign.stat()
    foreign.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "state": "in_progress",
                "output_artifact_path": str(output.resolve()),
                "marker_device": foreign_stat.st_dev,
                "marker_inode": foreign_stat.st_ino,
            }
        ),
        encoding="utf-8",
    )
    foreign.replace(marker)

    with pytest.raises(RuntimeError, match="ownership"):
        runner.require_output_namespace_reservation(output)


def test_output_namespace_completion_rejects_unregistered_final_path(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    marker = runner.reserve_output_namespace(output)
    output.write_text("foreign but present\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="coordinator-owned final receipt"):
        runner.complete_output_namespace(marker, output_path=output)

    assert marker.exists()
    assert output.read_text(encoding="utf-8") == "foreign but present\n"
    output.unlink()
    runner.complete_output_namespace(marker, output_path=output, require_final_artifact=False)


def test_output_namespace_completion_rejects_unregistered_sidecar(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    marker = runner.reserve_output_namespace(output)
    runner.write_json_artifact(
        output,
        {"status": "passed"},
        ownership_validator=lambda: runner.require_output_namespace_reservation(output),
    )
    foreign_sidecar = runner.phase_output_artifact_path(output, phase="foreign")
    foreign_sidecar.write_bytes(b"foreign\n")

    with pytest.raises(RuntimeError, match="unregistered"):
        runner.complete_output_namespace(marker, output_path=output)

    assert marker.exists()
    assert foreign_sidecar.read_bytes() == b"foreign\n"


def test_output_namespace_completion_reopens_registered_root_and_sidecar_receipts(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    marker = runner.reserve_output_namespace(output)
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    executions = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    sidecar = runner.phase_output_artifact_path(output, phase="steady-0")
    sidecar_metadata = runner.write_full_generation_records_artifact(
        sidecar,
        records=records_from_vllm_outputs(manifest, _fake_outputs(manifest)),
        execution_records=executions,
        ownership_validator=lambda: runner.require_output_namespace_reservation(output),
    )
    root_receipt = runner.write_json_artifact(
        output,
        {"status": "passed", "sidecar": sidecar_metadata},
        ownership_validator=lambda: runner.require_output_namespace_reservation(output),
    )

    runner.complete_output_namespace(marker, output_path=output)

    assert not marker.exists()
    assert output.exists()
    assert sidecar.exists()
    assert root_receipt.final_path == str(output.resolve())
    assert sidecar_metadata["publication_receipt"]["final_path"] == str(sidecar.resolve())


def test_output_namespace_completion_rejects_foreign_replacement_of_registered_final(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    marker = runner.reserve_output_namespace(output)
    runner.write_json_artifact(
        output,
        {"status": "passed"},
        ownership_validator=lambda: runner.require_output_namespace_reservation(output),
    )
    foreign = tmp_path / "foreign.json"
    foreign.write_text("foreign\n", encoding="utf-8")
    foreign.replace(output)

    with pytest.raises(RuntimeError, match="inode"):
        runner.complete_output_namespace(marker, output_path=output)

    assert marker.exists()
    assert output.read_text(encoding="utf-8") == "foreign\n"


def test_main_does_not_publish_evidence_after_namespace_ownership_loss(tmp_path, monkeypatch) -> None:
    output = tmp_path / "ownership-loss.json"
    foreign_payload = "foreign owner\n"

    def replace_reservation(*_args, **_kwargs):
        marker = output.with_name("ownership-loss.inprogress")
        foreign = tmp_path / "foreign-marker"
        foreign.write_text(foreign_payload, encoding="utf-8")
        foreign.replace(marker)
        return {"must_not_be_published": True}

    monkeypatch.setattr(runner, "run_context_length_preflight", replace_reservation)

    with pytest.raises(RuntimeError, match="ownership"):
        runner.main(
            [
                "--backend",
                "vllm",
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--manifest",
                str(DATA),
                "--topology",
                "tp2",
                "--max-model-len",
                "7000",
                "--max-num-batched-tokens",
                "16384",
                "--gpu-memory-utilization",
                "0.92",
                "--context-preflight-only",
                "--output",
                str(output),
            ]
        )

    assert not output.exists()
    assert output.with_name("ownership-loss.inprogress").read_text(encoding="utf-8") == foreign_payload


def _gpu_preflight_failure() -> runner.GpuPreflightError:
    return runner.GpuPreflightError(
        "one-byte CUDA usable-total drift",
        evidence={
            "schema_version": 2,
            "passed": False,
            "cuda_visible_devices": "0,1",
            "expected_cuda_visible_devices": "0,1",
            "devices": [
                {
                    "logical_device_index": 0,
                    "memory": {
                        "nvml": {
                            "physical_total_bytes": 85_520_809_984,
                            "system_reserved_bytes": 351_666_176,
                            "free_bytes": 80_000_000_000,
                            "used_bytes": 5_169_143_808,
                        },
                        "cuda": {
                            "usable_total_bytes": 85_169_143_809,
                            "free_bytes": 80_000_000_000,
                        },
                        "torch_properties_total_bytes": 85_169_143_809,
                        "usable_total_relation_delta_bytes": -1,
                    },
                }
            ],
            "failure": {
                "stage": "memory-accounting",
                "message": "one-byte CUDA usable-total drift",
            },
        },
    )


def test_tp2_gpu_preflight_failure_occurs_before_vllm_import(tmp_path, monkeypatch) -> None:
    import builtins

    from bionemo.evo2.vllm.config import Evo2Config

    checkpoint = tmp_path / "checkpoint"
    Evo2Config(max_position_embeddings=10_240).save_pretrained(checkpoint)
    output = tmp_path / "gpu-preflight-order.json"
    args = runner.build_parser().parse_args(
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
            "7000",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.92",
            "--proof",
            "--output",
            str(output),
        ]
    )
    manifest = WorkloadManifest.from_path(DATA)
    runner.reserve_output_namespace(output)
    monkeypatch.setattr(runner, "context_length_preflight", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner,
        "gpu_hardware_provenance",
        lambda: (_ for _ in ()).throw(_gpu_preflight_failure()),
    )
    real_import = builtins.__import__
    imported_vllm = []

    def track_import(name, *args, **kwargs):
        if name == "vllm":
            imported_vllm.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_import)

    with pytest.raises(runner.GpuPreflightError):
        runner.run_tp2_benchmark(args, manifest)

    assert imported_vllm == []
    assert not output.exists()


def test_main_atomically_publishes_raw_gpu_preflight_failure(tmp_path, monkeypatch) -> None:
    output = tmp_path / "gpu-preflight-failure.json"
    monkeypatch.setattr(
        runner,
        "run_tp2_benchmark",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_gpu_preflight_failure()),
    )

    status = runner.main(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--manifest",
            str(DATA),
            "--topology",
            "tp2",
            "--max-model-len",
            "7000",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.92",
            "--proof",
            "--output",
            str(output),
        ]
    )

    artifact = json.loads(output.read_text())
    assert status == 1
    assert artifact["invocation"]["exit_status"] == 1
    assert artifact["failure"]["type"] == "GpuPreflightError"
    assert artifact["gpu_hardware_provenance"]["devices"][0]["memory"]["usable_total_relation_delta_bytes"] == -1
    assert not output.with_name("gpu-preflight-failure.inprogress").exists()


def test_json_artifact_writer_refuses_to_overwrite_prior_success(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    runner.write_json_artifact(output, {"run": 1})

    with pytest.raises(FileExistsError, match=r"benchmark.json"):
        runner.write_json_artifact(output, {"run": 2})

    assert json.loads(output.read_text()) == {"run": 1}


def test_json_artifact_writer_never_overwrites_foreign_output_created_before_publish(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    foreign = b"foreign-output\n"
    output.write_bytes(foreign)

    with pytest.raises(FileExistsError):
        runner.write_json_artifact(output, {"run": 1})

    assert output.read_bytes() == foreign


def test_json_writer_preserves_foreign_replacement_after_reservation_ownership_loss(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    foreign_path = tmp_path / "foreign.json"
    foreign = b"foreign-replacement\n"
    calls = 0

    def replace_published_inode_then_lose_ownership() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            foreign_path.write_bytes(foreign)
            foreign_path.replace(output)
            raise RuntimeError("ownership lost")

    with pytest.raises(RuntimeError, match="ownership lost"):
        runner.write_json_artifact(
            output,
            {"run": 1},
            ownership_validator=replace_published_inode_then_lose_ownership,
        )

    assert calls == 3
    assert output.read_bytes() == foreign
    assert not output.with_suffix(f"{output.suffix}.tmp").exists()


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

    with pytest.raises(ValueError, match=r"max_model_len=16.*workload max_total_tokens=6001"):
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
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    hardware = runner.gpu_hardware_provenance(
        nvml_module=fake_nvml,
        torch_module=fake_torch,
        expected_assignments=expected_assignments,
    )

    assert hardware["driver_version"] == "570.86.15"
    assert hardware["cuda_visible_devices"] == "0,1"
    assert hardware["passed"] is True
    assert [device["uuid"] for device in hardware["devices"]] == ["GPU-uuid-0", "GPU-uuid-1"]
    assert [device["pci_bus_id"] for device in hardware["devices"]] == [
        "00000000:01:00.0",
        "00000000:02:00.0",
    ]
    assert [device["memory"]["nvml"]["physical_total_bytes"] for device in hardware["devices"]] == [
        80 * gib,
        81 * gib,
    ]

    headroom = runner.gpu_memory_headroom_evidence(
        hardware,
        peak_device_memory_bytes=(77 * gib, 78 * gib),
    )
    assert headroom["required_headroom_bytes"] == 2 * gib
    assert [device["headroom_bytes"] for device in headroom["devices"]] == [5 * gib // 2, 5 * gib // 2]
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
        sampler_installation=_sampler_installation_provenance(),
        gpu_hardware=hardware,
    )
    assert attestation["gpu"]["cuda_visible_devices"] == "0,1"
    assert [device["pci_bus_id"] for device in attestation["gpu"]["devices"]] == [
        "00000000:01:00.0",
        "00000000:02:00.0",
    ]
    assert attestation["sampler"]["executing_python"]["invoked_path"] == "/nemo-rl/.venv/bin/python"
    assert (
        attestation["sampler"]["launcher_selection"]["isolated_worker_environment"]["status"]
        == "installed-but-bypassed"
    )


def _gpu_preflight_fakes(*, cuda_total_delta: int = 0):
    gib = 1024**3
    reserved = 512 * 1024**2

    class FakeNvml:
        __version__ = "13.590.48"
        nvmlMemory_v2 = 2  # noqa: N815

        @staticmethod
        def nvmlInit():  # noqa: N802
            return None

        @staticmethod
        def nvmlSystemGetDriverVersion():  # noqa: N802
            return b"570.86.15"

        @staticmethod
        def nvmlSystemGetCudaDriverVersion_v2():  # noqa: N802
            return 13_000

        @staticmethod
        def nvmlDeviceGetCount():  # noqa: N802
            return 2

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):  # noqa: N802
            return index

        @staticmethod
        def nvmlDeviceGetHandleByUUID(uuid):  # noqa: N802
            return int(str(uuid).rsplit("-", 1)[1])

        @staticmethod
        def nvmlDeviceGetIndex(handle):  # noqa: N802
            return handle

        @staticmethod
        def nvmlDeviceGetUUID(handle):  # noqa: N802
            return f"GPU-uuid-{handle}".encode()

        @staticmethod
        def nvmlDeviceGetName(handle):  # noqa: N802
            return f"NVIDIA H100 {handle}".encode()

        @staticmethod
        def nvmlDeviceGetPciInfo(handle):  # noqa: N802
            return SimpleNamespace(busId=f"00000000:0{handle + 1}:00.0".encode())

        @classmethod
        def nvmlDeviceGetMemoryInfo(cls, handle, *, version):  # noqa: N802
            assert version == cls.nvmlMemory_v2
            physical_total = (80 + handle) * gib
            usable_total = physical_total - reserved
            used = 3 * gib
            return SimpleNamespace(
                total=physical_total,
                reserved=reserved,
                used=used,
                free=usable_total - used,
            )

    class FakeCuda:
        @staticmethod
        def _physical_index(logical_device):
            selector = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[logical_device]
            return int(selector.rsplit("-", 1)[1]) if selector.startswith("GPU-") else int(selector)

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_properties(logical_device):
            physical_index = FakeCuda._physical_index(logical_device)
            physical_total = (80 + physical_index) * gib
            usable_total = physical_total - reserved
            return SimpleNamespace(
                name=f"NVIDIA H100 {physical_index}",
                uuid=f"GPU-uuid-{physical_index}",
                total_memory=usable_total,
            )

        @staticmethod
        def mem_get_info(logical_device):
            physical_index = FakeCuda._physical_index(logical_device)
            physical_total = (80 + physical_index) * gib
            usable_total = physical_total - reserved + cuda_total_delta
            return usable_total - 3 * gib, usable_total

    fake_torch = SimpleNamespace(
        __version__="2.10.0",
        version=SimpleNamespace(cuda="13.0"),
        cuda=FakeCuda,
    )
    expected_assignments = (
        {
            "logical_device_index": 0,
            "visible_device_selector": "0",
            "physical_index": 0,
            "uuid": "GPU-uuid-0",
            "pci_bus_id": "00000000:01:00.0",
        },
        {
            "logical_device_index": 1,
            "visible_device_selector": "1",
            "physical_index": 1,
            "uuid": "GPU-uuid-1",
            "pci_bus_id": "00000000:02:00.0",
        },
    )
    return FakeNvml, fake_torch, expected_assignments


def test_gpu_preflight_binds_assignment_and_exact_nvml_cuda_memory_semantics(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()

    hardware = runner.gpu_hardware_provenance(
        nvml_module=fake_nvml,
        torch_module=fake_torch,
        expected_assignments=expected_assignments,
    )

    assert hardware["passed"] is True
    assert hardware["expected_assignments"] == list(expected_assignments)
    assert hardware["api_versions"]["nvml_memory_info_api"] == "nvmlDeviceGetMemoryInfo_v2"
    assert hardware["api_versions"]["cuda_memory_info_api"] == "torch.cuda.mem_get_info"
    for device in hardware["devices"]:
        memory = device["memory"]
        assert (
            memory["nvml"]["physical_total_bytes"] - memory["nvml"]["system_reserved_bytes"]
            == (memory["cuda"]["usable_total_bytes"])
        )
        assert memory["torch_properties_total_bytes"] == memory["cuda"]["usable_total_bytes"]
        assert memory["usable_total_relation_delta_bytes"] == 0


def test_gpu_preflight_canonicalizes_nvml_pci_hex_case(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()
    expected_assignments = tuple(dict(item) for item in expected_assignments)
    expected_assignments[0]["pci_bus_id"] = "00000000:0a:00.0"
    expected_assignments[1]["pci_bus_id"] = "00000000:18:00.0"
    fake_nvml.nvmlDeviceGetPciInfo = staticmethod(
        lambda handle: SimpleNamespace(
            busId=("00000000:0A:00.0" if handle == 0 else "00000000:18:00.0").encode()
        )
    )

    hardware = runner.gpu_hardware_provenance(
        nvml_module=fake_nvml,
        torch_module=fake_torch,
        expected_assignments=expected_assignments,
    )

    observed = [device["pci_bus_id"] for device in hardware["devices"]]
    if observed != ["00000000:0a:00.0", "00000000:18:00.0"]:
        raise AssertionError(f"NVML PCI identities were not canonicalized: {observed!r}")


def test_gpu_preflight_retains_raw_one_byte_usable_total_disagreement(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes(cuda_total_delta=1)

    with pytest.raises(runner.GpuPreflightError) as error:
        runner.gpu_hardware_provenance(
            nvml_module=fake_nvml,
            torch_module=fake_torch,
            expected_assignments=expected_assignments,
        )

    retained = error.value.evidence
    assert retained["passed"] is False
    assert retained["devices"][0]["memory"]["usable_total_relation_delta_bytes"] == -1
    assert retained["failure"]["stage"] == "memory-accounting"


def test_gpu_preflight_retains_raw_mapping_for_wrong_visible_order(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()

    with pytest.raises(runner.GpuPreflightError) as error:
        runner.gpu_hardware_provenance(
            nvml_module=fake_nvml,
            torch_module=fake_torch,
            expected_assignments=expected_assignments,
        )

    retained = error.value.evidence
    assert retained["cuda_visible_devices"] == "1,0"
    assert [device["visible_device_selector"] for device in retained["devices"]] == ["1", "0"]
    assert [device["physical_index"] for device in retained["devices"]] == [1, 0]
    assert retained["failure"]["stage"] == "assignment"


def test_gpu_preflight_rejects_uuid_selectors_even_when_physical_mapping_matches(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-uuid-0,GPU-uuid-1")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()

    with pytest.raises(runner.GpuPreflightError) as error:
        runner.gpu_hardware_provenance(
            nvml_module=fake_nvml,
            torch_module=fake_torch,
            expected_assignments=expected_assignments,
        )

    retained = error.value.evidence
    assert [device["physical_index"] for device in retained["devices"]] == [0, 1]
    assert retained["observed_visible_device_selectors"] == ["GPU-uuid-0", "GPU-uuid-1"]
    assert retained["failure"]["stage"] == "assignment"


@pytest.mark.parametrize("tamper", ("physical_index", "uuid", "pci_bus_id"))
def test_gpu_preflight_retains_raw_physical_identity_mismatch(monkeypatch, tamper) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()
    if tamper == "physical_index":
        monkeypatch.setattr(
            fake_nvml,
            "nvmlDeviceGetIndex",
            staticmethod(lambda handle: 1 - handle),
        )
    elif tamper == "uuid":
        monkeypatch.setattr(
            fake_nvml,
            "nvmlDeviceGetUUID",
            staticmethod(lambda handle: f"GPU-wrong-{handle}".encode()),
        )
    else:
        monkeypatch.setattr(
            fake_nvml,
            "nvmlDeviceGetPciInfo",
            staticmethod(lambda handle: SimpleNamespace(busId=f"00000000:9{handle}:00.0".encode())),
        )

    with pytest.raises(runner.GpuPreflightError) as error:
        runner.gpu_hardware_provenance(
            nvml_module=fake_nvml,
            torch_module=fake_torch,
            expected_assignments=expected_assignments,
        )

    retained = error.value.evidence
    assert len(retained["devices"]) == 2
    assert len(retained["assignment_mismatches"]) == 2
    assert retained["failure"]["stage"] == "assignment"


def test_gpu_preflight_retains_raw_one_byte_nvml_sum_drift(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()
    original = fake_nvml.nvmlDeviceGetMemoryInfo

    def drifted_memory(handle, *, version):
        memory = original(handle, version=version)
        if handle != 0:
            return memory
        return SimpleNamespace(
            total=memory.total,
            reserved=memory.reserved,
            used=memory.used,
            free=memory.free + 1,
        )

    monkeypatch.setattr(fake_nvml, "nvmlDeviceGetMemoryInfo", staticmethod(drifted_memory))

    with pytest.raises(runner.GpuPreflightError) as error:
        runner.gpu_hardware_provenance(
            nvml_module=fake_nvml,
            torch_module=fake_torch,
            expected_assignments=expected_assignments,
        )

    memory = error.value.evidence["devices"][0]["memory"]["nvml"]
    assert memory["free_bytes"] + memory["used_bytes"] + memory["system_reserved_bytes"] == (
        memory["physical_total_bytes"] + 1
    )
    assert error.value.evidence["failure"]["stage"] == "memory-accounting"


def test_gpu_preflight_rejects_one_byte_below_initial_headroom(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()
    original = fake_nvml.nvmlDeviceGetMemoryInfo
    required = 2 * 1024**3

    def low_headroom_memory(handle, *, version):
        memory = original(handle, version=version)
        if handle != 0:
            return memory
        free = required - 1
        return SimpleNamespace(
            total=memory.total,
            reserved=memory.reserved,
            used=memory.total - memory.reserved - free,
            free=free,
        )

    monkeypatch.setattr(fake_nvml, "nvmlDeviceGetMemoryInfo", staticmethod(low_headroom_memory))

    with pytest.raises(runner.GpuPreflightError) as error:
        runner.gpu_hardware_provenance(
            nvml_module=fake_nvml,
            torch_module=fake_torch,
            expected_assignments=expected_assignments,
        )

    assert error.value.evidence["devices"][0]["memory"]["nvml"]["free_bytes"] == required - 1
    assert error.value.evidence["failure"]["stage"] == "memory-headroom"


def test_gpu_preflight_rejects_mutated_frozen_logical_indices_before_collection(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()
    mutated = copy.deepcopy(expected_assignments)
    mutated[0]["logical_device_index"] = 1
    monkeypatch.setattr(
        fake_nvml,
        "nvmlInit",
        staticmethod(lambda: pytest.fail("NVML collection must not start")),
    )

    with pytest.raises(ValueError, match="logical indices"):
        runner.gpu_hardware_provenance(
            nvml_module=fake_nvml,
            torch_module=fake_torch,
            expected_assignments=mutated,
        )


def _worker_gpu_proofs(hardware, *, physical_order=(0, 1)):
    workers = []
    for rank, physical_index in enumerate(physical_order):
        device = hardware["devices"][physical_index]
        workers.append(
            {
                "rank": rank,
                "logical_device": physical_index,
                "device_uuid": device["uuid"],
                "pci_bus_id": device["pci_bus_id"],
                "device_name": device["name"],
                "cuda_visible_devices": "0,1",
                "visible_device_selector": str(physical_index),
                "engine_seed": 42,
            }
        )
    return workers


def test_worker_gpu_bindings_accept_exact_physical_index_schema(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()
    hardware = runner.gpu_hardware_provenance(
        nvml_module=fake_nvml,
        torch_module=fake_torch,
        expected_assignments=expected_assignments,
    )

    runner._validate_worker_gpu_bindings(
        _worker_gpu_proofs(hardware),
        hardware=hardware,
        expected_worker_count=2,
        expected_engine_seed=42,
        expected_physical_indices=(0, 1),
    )


def test_worker_gpu_bindings_reject_rank_to_physical_assignment_swap(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_nvml, fake_torch, expected_assignments = _gpu_preflight_fakes()
    hardware = runner.gpu_hardware_provenance(
        nvml_module=fake_nvml,
        torch_module=fake_torch,
        expected_assignments=expected_assignments,
    )
    workers = _worker_gpu_proofs(hardware, physical_order=(1, 0))

    with pytest.raises(AssertionError, match="frozen physical assignment"):
        runner._validate_worker_gpu_bindings(
            workers,
            hardware=hardware,
            expected_worker_count=2,
            expected_engine_seed=42,
            expected_physical_indices=(0, 1),
        )


def test_worker_sampler_proof_uses_exact_runtime_and_physical_seed_batches(monkeypatch) -> None:
    calls = []

    def fake_validate(proof, **kwargs):
        calls.append((proof, kwargs))
        return {"passed": True}

    monkeypatch.setattr(sampler_module, "validate_sampler_proof_evidence", fake_validate)
    workers = [{"sampler": {"worker": rank}} for rank in range(2)]
    installation = sampler_module.sampler_installation_contract(_sampler_installation_provenance())
    seed_batches = ((11, 13), (17, 19))
    request_generations = tuple(
        {
            "request_id": f"request-{seed}",
            "seed": seed,
            "accepted_output_token_count": 3,
        }
        for batch in seed_batches
        for seed in batch
    )

    summaries = runner._validate_worker_sampler_evidence(
        workers,
        expected_installation=installation,
        expected_seed_batches=seed_batches,
        expected_request_generations=request_generations,
        require_generation_observations=True,
    )

    if summaries != [{"passed": True}, {"passed": True}]:
        raise AssertionError("worker sampler summaries drifted")
    if [proof for proof, _ in calls] != [{"worker": 0}, {"worker": 1}]:
        raise AssertionError("worker sampler proofs were not forwarded exactly")
    if any(call["expected_seed_batches"] != seed_batches for _, call in calls):
        raise AssertionError("physical sampler seed batches were not forwarded exactly")
    if any(call["expected_request_generations"] != request_generations for _, call in calls):
        raise AssertionError("caller-reopened request generations were not forwarded exactly")
    if any(call["require_generation_observations"] is not True for _, call in calls):
        raise AssertionError("worker sampler generation evidence was not required")


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
                "logical_device_index": 0,
                "physical_index": 0,
                "uuid": "GPU-a",
                "memory": {
                    "nvml": {
                        "physical_total_bytes": 80 * gib,
                        "system_reserved_bytes": 0,
                    },
                    "cuda": {"usable_total_bytes": 80 * gib},
                },
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


def test_parser_and_loader_build_one_executable_homogeneous_identity_case(tmp_path) -> None:
    manifest_path = tmp_path / "base.json"
    manifest_path.write_text(json.dumps(_canonical_base_manifest().to_dict()), encoding="utf-8")
    parsed = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--manifest",
            str(manifest_path),
            "--topology",
            "tp2",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.92",
            "--request-count",
            "96",
            "--request-id-prefix",
            "canonical-case3",
            "--prompt-tokenizer-json",
            str(TOKENIZER_JSON),
            "--canonical-identity-case",
            "3",
            "--canonical-prompts-csv",
            str(PROMPTS_CSV),
            "--output",
            str(tmp_path / "proof.json"),
        ]
    )

    manifest = runner.load_source_manifest(parsed)

    assert parsed.canonical_identity_case == 3
    assert len(manifest.requests) == 96
    assert {len(request.prompt_token_ids) for request in manifest.requests} == {3808}
    assert len({request.prompt_token_ids for request in manifest.requests}) == 1
    assert manifest.max_new_tokens == 500

    profile = runner.profile_from_args(parsed, manifest)
    contract = runner.build_benchmark_contract(parsed, manifest, profile)
    identity = contract["canonical_identity"]
    assert identity["case_index"] == 3
    assert identity["checkpoint"] == CANONICAL_7B_CHECKPOINT
    assert identity["schedule"]["global_request_shapes"] == [96]
    assert identity["schedule"]["engine_request_shapes"] == [[96]]


def test_parser_and_loader_build_one_physical_interleaved_mixed_identity_batch(tmp_path) -> None:
    manifest_path = tmp_path / "base.json"
    manifest_path.write_text(json.dumps(_canonical_base_manifest().to_dict()), encoding="utf-8")
    parsed = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--manifest",
            str(manifest_path),
            "--topology",
            "tp2",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.92",
            "--request-count",
            "96",
            "--global-wave-size",
            "96",
            "--max-num-seqs",
            "96",
            "--warmups",
            "0",
            "--repetitions",
            "1",
            "--request-id-prefix",
            "mixed",
            "--prompt-tokenizer-json",
            str(TOKENIZER_JSON),
            "--mixed-canonical-identity",
            "--mixed-same-engine-qualification",
            "--canonical-prompts-csv",
            str(PROMPTS_CSV),
            "--output",
            str(tmp_path / "mixed.json"),
        ]
    )

    manifest = runner.load_source_manifest(parsed)
    profile = runner.profile_from_args(parsed, manifest)
    contract = runner.build_benchmark_contract(parsed, manifest, profile)
    mixed = contract["mixed_canonical_identity"]

    if [len(request.prompt_token_ids) for request in manifest.requests] != [3269, 3528, 3080, 3808] * 24:
        raise AssertionError("mixed CLI did not preserve the four original variable prompt lengths")
    if [request.request_id for request in manifest.requests[:4]] != [
        "mixed-b96-case0-occurrence0000",
        "mixed-b96-case1-occurrence0000",
        "mixed-b96-case2-occurrence0000",
        "mixed-b96-case3-occurrence0000",
    ]:
        raise AssertionError("mixed CLI lost interleaved semantic case ownership")
    if contract["canonical_identity"] is not None or mixed["case_order"] != [0, 1, 2, 3] * 24:
        raise AssertionError("mixed CLI contract overlapped homogeneous mode or reordered cases")
    if mixed["schedule"]["global_request_shapes"] != [96] or mixed["schedule"]["engine_request_shapes"] != [[96]]:
        raise AssertionError("mixed CLI did not bind one physical TP2 B96 call")

    stage_specs = runner.mixed_same_engine_stage_specs(parsed, manifest, profile)
    if stage_specs is None or [spec["stage"] for spec in stage_specs] != ["mixed-b4", "mixed-b96"]:
        raise AssertionError("mixed B96 qualification did not retain the ordered two-stage lifecycle")
    if [len(spec["manifest"].requests) for spec in stage_specs] != [4, 96]:
        raise AssertionError("capacity-96 engine did not retain actual B4 and B96 stage call shapes")
    if [spec["schedule"].global_wave_size for spec in stage_specs] != [4, 96]:
        raise AssertionError("mixed stage call shapes were incorrectly coupled to engine capacity")
    if [spec["execution_records"][0].call_index for spec in stage_specs] != [0, 1]:
        raise AssertionError("mixed stages did not advance the physical semantic call")
    if [spec["execution_records"][0].global_request_index for spec in stage_specs] != [0, 4]:
        raise AssertionError("mixed stages did not retain disjoint caller-owned global ranges")
    b4_seeds = {record.seed for record in stage_specs[0]["execution_records"]}
    b96_seeds = {record.seed for record in stage_specs[1]["execution_records"]}
    if b4_seeds & b96_seeds:
        raise AssertionError("mixed B4 and B96 stages reused request seeds")
    phase_coordinates = contract["measurement"]["phase_coordinates"]
    if [row["phase"] for row in phase_coordinates] != ["mixed-b4", "mixed-b96"]:
        raise AssertionError("mixed benchmark contract described phases other than actual B4 then B96")
    if [row["global_call_index_start"] for row in phase_coordinates] != [0, 1]:
        raise AssertionError("mixed benchmark contract did not bind actual advancing calls")
    if [row["global_request_index_start"] for row in phase_coordinates] != [0, 4]:
        raise AssertionError("mixed benchmark contract did not bind actual disjoint request ranges")
    if contract["seed_stream"] != mixed["admission_bundle"]:
        raise AssertionError("mixed benchmark seed authority drifted from the actual stage admission")


def test_tp2_mixed_qualification_constructs_one_capacity96_engine_without_forced_proof(
    tmp_path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "base.json"
    manifest_path.write_text(json.dumps(_canonical_base_manifest().to_dict()), encoding="utf-8")
    output = tmp_path / "mixed-same-engine.json"
    args = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--manifest",
            str(manifest_path),
            "--topology",
            "tp2",
            "--max-model-len",
            "6144",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.92",
            "--request-count",
            "96",
            "--global-wave-size",
            "96",
            "--max-num-seqs",
            "96",
            "--warmups",
            "0",
            "--repetitions",
            "1",
            "--request-id-prefix",
            "mixed",
            "--prompt-tokenizer-json",
            str(TOKENIZER_JSON),
            "--mixed-canonical-identity",
            "--mixed-same-engine-qualification",
            "--canonical-prompts-csv",
            str(PROMPTS_CSV),
            "--output",
            str(output),
        ]
    )
    manifest = runner.load_source_manifest(args)
    runner.reserve_output_namespace(output)

    engine_instances = []
    stage_calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.llm_engine = SimpleNamespace(
                logger_manager=SimpleNamespace(stat_loggers=[]),
                vllm_config=object(),
            )
            engine_instances.append(self)

    class FakeMemoryMonitor:
        peak_device_memory_bytes = (10, 11)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_phase(**kwargs):
        stage_manifest = kwargs["manifest"]
        execution_records = kwargs["execution_records"]
        stage_calls.append(kwargs)
        request_count = len(stage_manifest.requests)
        sample = runner.BenchmarkSample(
            sample_index=kwargs["sample_index"],
            generation_s=1.0,
            request_count=request_count,
            prompt_tokens=sum(len(request.prompt_token_ids) for request in stage_manifest.requests),
            generated_tokens=request_count * 500,
            ttft_s=(0.1,) * request_count,
            inter_token_latency_s=(0.001,) * request_count,
            output_lengths=(500,) * request_count,
            peak_device_memory_bytes=(20, 21),
        )
        return runner.GenerationPhaseResult(
            phase=kwargs["phase"],
            sample=sample,
            generation_call_s=(1.0,),
            wave_proofs=(
                {
                    "wave_index": 0,
                    "start": 0,
                    "stop": request_count,
                    "request_count": request_count,
                    "generation_s": 1.0,
                    "full_decode_proof": {"passed": True},
                },
            ),
            observations=(),
            output_summaries=(),
            request_executions=execution_records,
            full_output_artifact={"stage": kwargs["phase"]},
            full_decode_proof={"passed": True},
            worker_proof=(),
            shared_prefix_state_reuse=None,
            proof_collected=kwargs["collect_proof"],
            prefix_cache_reset=False,
        )

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=SimpleNamespace))
    monkeypatch.setattr(runner, "context_length_preflight", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "gpu_hardware_provenance", lambda: {"devices": []})
    monkeypatch.setattr(runner, "checkpoint_provenance", lambda *_args: {})
    monkeypatch.setattr(runner, "source_provenance", lambda **_kwargs: {})
    monkeypatch.setattr(runner, "vllm_installation_provenance", lambda: {})
    monkeypatch.setattr(sampler_module, "sampler_installation_provenance", lambda **_kwargs: {})
    monkeypatch.setattr(runner, "runtime_attestation_contract", lambda **_kwargs: {})
    monkeypatch.setattr(
        runner.Evo2VllmProfile,
        "engine_kwargs",
        lambda self, **_kwargs: {"max_num_seqs": self.resolved_max_num_seqs},
    )
    monkeypatch.setattr(runner, "manifest_output_decoder", lambda _manifest: lambda ids: bytes(ids).decode())
    monkeypatch.setattr(runner, "make_nvml_memory_reader", lambda: None)
    monkeypatch.setattr(runner, "PeakMemoryMonitor", lambda _reader: FakeMemoryMonitor())
    monkeypatch.setattr(runner, "resolved_config_snapshot", lambda _config: {"cache": {"block_size": 16}})
    monkeypatch.setattr(runner, "validate_resolved_profile", lambda *_args: None)
    monkeypatch.setattr(runner, "run_generation_phase", fake_phase)
    monkeypatch.setattr(
        runner,
        "gpu_memory_headroom_evidence",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(runner, "runtime_versions", lambda: {})

    def validate_mixed_phases(**kwargs):
        phases = kwargs["phase_artifacts"]
        if [phase["phase"] for phase in phases] != ["mixed-b4", "mixed-b96"]:
            raise AssertionError("mixed scorer did not receive the exact ordered stage results")
        if kwargs["collect_physical_proof"] is not False:
            raise AssertionError("proof-disabled mixed execution forced physical-proof instrumentation")
        return phases, {"same_engine_b4_then_b96_qualified": True, "passed": True}

    monkeypatch.setattr(runner, "mixed_canonical_identity_phase_artifacts", validate_mixed_phases)

    artifact = runner.run_tp2_benchmark(args, manifest)

    if len(engine_instances) != 1:
        raise AssertionError("mixed B4/B96 qualification constructed more than one vLLM engine")
    if [call["phase"] for call in stage_calls] != ["mixed-b4", "mixed-b96"]:
        raise AssertionError("mixed B4/B96 qualification executed the wrong stage order")
    if [len(call["manifest"].requests) for call in stage_calls] != [4, 96]:
        raise AssertionError("mixed B4/B96 qualification used the wrong actual call shapes")
    if [call["global_wave_size"] for call in stage_calls] != [4, 96]:
        raise AssertionError("mixed B4/B96 stage shapes remained coupled to engine capacity")
    if [call["scheduler_max_num_seqs"] for call in stage_calls] != [96, 96]:
        raise AssertionError("mixed stages did not share one capacity-96 scheduler")
    if any(call["llm"] is not engine_instances[0] for call in stage_calls):
        raise AssertionError("mixed stages did not execute through the same vLLM engine object")
    if [call["execution_records"][0].call_index for call in stage_calls] != [0, 1]:
        raise AssertionError("mixed stages did not execute their admitted advancing calls")
    if [call["execution_records"][0].global_request_index for call in stage_calls] != [0, 4]:
        raise AssertionError("mixed stages did not execute their admitted global ranges")
    if engine_instances[0].kwargs.get("max_num_seqs") != 96:
        raise AssertionError("mixed engine was not constructed at capacity 96")
    if engine_instances[0].kwargs.get("cudagraph_metrics") is not True:
        raise AssertionError("mixed qualification did not enable supported CUDA-graph metrics")
    if "worker_extension_cls" in engine_instances[0].kwargs:
        raise AssertionError("mixed qualification installed a proof-only worker extension")
    if artifact["mixed_canonical_identity"]["same_engine_b4_then_b96_qualified"] is not True:
        raise AssertionError("mixed same-engine result was not propagated into the root artifact")


def test_mixed_same_engine_scorer_accepts_outputs_without_forced_physical_proof(tmp_path) -> None:
    cases = load_canonical_7b_identity_cases(PROMPTS_CSV)
    tokenizer = SnapshotBoundTokenizer.from_path(TOKENIZER_JSON)
    manifest = runner.build_mixed_canonical_identity_manifest(
        _canonical_base_manifest(),
        cases=cases,
        prompts_csv=PROMPTS_CSV,
        tokenizer=tokenizer,
        request_count=96,
        request_id_prefix="mixed-b96",
    )
    args = SimpleNamespace(
        mixed_canonical_identity=True,
        mixed_same_engine_qualification=True,
        canonical_prompts_csv=PROMPTS_CSV,
        prompt_tokenizer_json=TOKENIZER_JSON,
        request_id_prefix="mixed",
        warmups=0,
        repetitions=1,
        generation_round=0,
        proof=False,
    )
    profile = runner.Evo2VllmProfile(
        topology="tp2",
        max_model_len=6144,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.92,
        global_wave_size=96,
        max_num_seqs=96,
    )
    stage_specs = runner.mixed_same_engine_stage_specs(args, manifest, profile)
    phases = []
    for spec in stage_specs:
        stage_manifest = spec["manifest"]
        records = tuple(
            GenerationRecord(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                output_token_ids=tuple(cases[index % 4].target.encode("ascii")),
                output_logprobs=(-0.1,) * 500,
                requested_max_tokens=500,
                finish_reason="length",
                stop_reason=None,
                stopped_on_eos=False,
            )
            for index, request in enumerate(stage_manifest.requests)
        )
        phase = _runner_identity_phase(spec["schedule"], phase_name=spec["stage"])
        phase["request_executions"] = [record.to_dict() for record in spec["execution_records"]]
        phase["full_output_artifact"] = runner.write_full_generation_records_artifact(
            tmp_path / f"{spec['stage']}.outputs.jsonl.gz",
            records=records,
            execution_records=spec["execution_records"],
            decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
        )
        phases.append(phase)

    retained, summary = runner.mixed_canonical_identity_phase_artifacts(
        args=args,
        manifest=manifest,
        profile=profile,
        phase_artifacts=phases,
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
        collect_physical_proof=False,
    )

    if [phase["phase"] for phase in retained] != ["mixed-b4", "mixed-b96"]:
        raise AssertionError("mixed scorer reordered the admitted stage artifacts")
    if summary["same_engine_b4_then_b96_qualified"] is not True:
        raise AssertionError("mixed scorer did not qualify exact B4 then B96 evidence")
    if summary["physical_schedule_attested"] is not False:
        raise AssertionError("proof-disabled mixed scorer claimed physical schedule attestation")
    if any(phase["mixed_canonical_identity_evidence"]["physical_schedule"] is not None for phase in retained):
        raise AssertionError("proof-disabled mixed scorer retained physical proof evidence")
    if any(
        phase["mixed_canonical_identity_evidence"]["proof_only_runtime_hooks_installed"] is not False
        for phase in retained
    ):
        raise AssertionError("proof-disabled mixed scorer claimed proof-only runtime hooks")
    if [phase["request_count"] for phase in summary["phases"]] != [4, 96]:
        raise AssertionError("mixed scorer did not retain both exact stage output inventories")
    with pytest.raises(AssertionError, match="exact ordered B4 then B96"):
        runner.mixed_canonical_identity_phase_artifacts(
            args=args,
            manifest=manifest,
            profile=profile,
            phase_artifacts=[phases[0], phases[0], phases[1]],
            decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
            collect_physical_proof=False,
        )


def test_loader_builds_one_executable_common_prefix_identity_case(tmp_path) -> None:
    manifest_path = tmp_path / "base.json"
    manifest_path.write_text(json.dumps(_canonical_base_manifest().to_dict()), encoding="utf-8")
    parsed = runner.build_parser().parse_args(
        [
            "--backend",
            "vllm",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--manifest",
            str(manifest_path),
            "--topology",
            "dp2",
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            "0.92",
            "--request-count",
            "96",
            "--request-id-prefix",
            "common-case2",
            "--prompt-tokenizer-json",
            str(TOKENIZER_JSON),
            "--common-prefix-identity-case",
            "2",
            "--canonical-prompts-csv",
            str(PROMPTS_CSV),
            "--output",
            str(tmp_path / "proof.json"),
        ]
    )

    manifest = runner.load_source_manifest(parsed)
    profile = runner.profile_from_args(parsed, manifest)
    contract = runner.build_benchmark_contract(parsed, manifest, profile)

    assert len(manifest.requests) == 96
    assert {len(request.prompt_token_ids) for request in manifest.requests} == {2048}
    assert manifest.max_new_tokens == 500
    assert contract["canonical_identity"] is None
    assert contract["common_prefix_identity"]["case_index"] == 2
    assert contract["common_prefix_identity"]["serial_schedule"]["engine_request_shapes"] == [[1]]
    assert contract["common_prefix_identity"]["candidate_schedule"]["engine_request_shapes"] == [[48, 48]]


@pytest.mark.parametrize(
    "overrides",
    (
        {"request_count": None},
        {"prompt_jsonl": Path("other.jsonl")},
        {"uniform_prompt_length": 2048},
        {"max_new_tokens": 499},
    ),
)
def test_canonical_identity_loader_rejects_incompatible_workload_rewrites(tmp_path, overrides) -> None:
    manifest_path = tmp_path / "base.json"
    manifest_path.write_text(json.dumps(_canonical_base_manifest().to_dict()), encoding="utf-8")
    values = {
        "manifest": manifest_path,
        "prompt_jsonl": None,
        "prompt_jsonl_sha256": None,
        "prompt_tokenizer_json": TOKENIZER_JSON,
        "expected_prompt_tokens": None,
        "canonical_identity_case": 0,
        "canonical_prompts_csv": PROMPTS_CSV,
        "request_count": 96,
        "request_id_prefix": "canonical-case0",
        "uniform_prompt_length": None,
        "max_new_tokens": None,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="canonical identity"):
        runner.load_source_manifest(SimpleNamespace(**values))


def _runner_identity_records(manifest, target):
    return tuple(
        GenerationRecord(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=tuple(target.encode("ascii")),
            output_logprobs=(-0.1,) * 500,
            requested_max_tokens=500,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        )
        for request in manifest.requests
    )


def _runner_identity_phase(schedule, *, phase_name="identity"):
    waves = []
    observations = []
    start = 0
    for wave_index, (global_shape, engine_shapes) in enumerate(
        zip(schedule.global_request_shapes, schedule.engine_request_shapes, strict=True)
    ):
        wave_phase = f"{phase_name}.wave-{wave_index:03d}"
        stop = start + global_shape
        if schedule.topology == "tp2":
            replay_count = 499 if schedule.mixed_case_batching else 1
            observations.extend(
                {
                    "phase": wave_phase,
                    "runtime_mode": "CUDAGraphMode.FULL",
                    "num_unpadded_tokens": global_shape,
                    "num_padded_tokens": global_shape,
                    "num_paddings": 0,
                    **(
                        {
                            "request_dimensions": {
                                "schema_version": 1,
                                "source": "iteration-stats-bound-to-cudagraph-dispatch",
                                "prefill_req_count": 0,
                                "decode_req_count": global_shape,
                                "token_count": global_shape,
                                "prompt_token_count": 0,
                                "first_token_event_count": 0,
                            }
                        }
                        if schedule.mixed_case_batching
                        else {}
                    ),
                }
                for _ in range(replay_count)
            )
            waves.append(
                {
                    "wave_index": wave_index,
                    "start": start,
                    "stop": stop,
                    "request_count": global_shape,
                    "full_decode_proof": {
                        "batch_size": global_shape,
                        "max_new_tokens": 500,
                        "maximum_full_batch": global_shape,
                        "passed": True,
                    },
                    "scheduler_capacity_proof": {
                        "global_wave_size": global_shape,
                        "engine_request_count": global_shape,
                        "maximum_running_requests": global_shape,
                        "batch_fit_without_preemption": True,
                    },
                }
            )
        else:
            engines = []
            for dp_rank, engine_shape in enumerate(engine_shapes):
                engine_observations = [
                    {
                        "phase": wave_phase,
                        "runtime_mode": "CUDAGraphMode.FULL",
                        "num_unpadded_tokens": engine_shape,
                        "num_padded_tokens": engine_shape,
                        "num_paddings": 0,
                    }
                ]
                engines.append(
                    {
                        "dp_rank": dp_rank,
                        "request_count": engine_shape,
                        "cudagraph_observations": engine_observations,
                        "full_decode_proof": {
                            "batch_size": engine_shape,
                            "max_new_tokens": 500,
                            "maximum_full_batch": engine_shape,
                            "passed": True,
                        },
                        "scheduler_capacity_proof": {
                            "global_wave_size": global_shape,
                            "engine_request_count": engine_shape,
                            "maximum_running_requests": engine_shape,
                            "batch_fit_without_preemption": True,
                        },
                    }
                )
            waves.append(
                {
                    "wave_index": wave_index,
                    "start": start,
                    "stop": stop,
                    "request_count": global_shape,
                    "engines": engines,
                }
            )
        start = stop
    return {
        "phase": phase_name,
        "sample": {
            "request_count": schedule.request_count,
            "generated_tokens": schedule.request_count * 500,
            "output_lengths": [500] * schedule.request_count,
        },
        "wave_proofs": waves,
        "cudagraph_observations_retained": observations,
    }


def test_linked_identity_validator_recomputes_raw_outputs_and_physical_shapes(tmp_path) -> None:
    case = load_canonical_7b_identity_cases(PROMPTS_CSV)[0]
    manifest = build_canonical_identity_manifest(
        _canonical_base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=SnapshotBoundTokenizer.from_path(TOKENIZER_JSON),
        request_count=2,
        request_id_prefix="identity-case0",
    )
    schedule = build_homogeneous_identity_schedule(topology="tp2", request_count=2, global_wave_size=2)
    phase = _runner_identity_phase(schedule)
    executions = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    phase["full_output_artifact"] = runner.write_full_generation_records_artifact(
        tmp_path / "linked.outputs.jsonl.gz",
        records=_runner_identity_records(manifest, case.target),
        execution_records=executions,
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )
    args = SimpleNamespace(
        canonical_identity_case=0,
        canonical_prompts_csv=PROMPTS_CSV,
        prompt_tokenizer_json=TOKENIZER_JSON,
    )
    profile = SimpleNamespace(topology="tp2", global_wave_size=2)
    phases, summary = runner.canonical_identity_phase_artifacts(
        args=args,
        manifest=manifest,
        profile=profile,
        phase_artifacts=[phase],
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
        collect_physical_proof=True,
    )
    artifact = {"phases": phases, "canonical_identity": summary}
    contract = build_canonical_identity_contract(
        case=case,
        schedule=schedule,
        prompts_csv=PROMPTS_CSV,
        tokenizer_path=TOKENIZER_JSON,
    )

    recomputed = runner.validate_canonical_identity_proof_evidence(
        artifact,
        manifest=manifest,
        profile=profile,
        expected_contract={"canonical_identity": contract},
    )
    assert recomputed == summary

    artifact["phases"][0]["canonical_identity_evidence"]["outputs"]["passed"] = False
    with pytest.raises(AssertionError, match="canonical identity"):
        runner.validate_canonical_identity_proof_evidence(
            artifact,
            manifest=manifest,
            profile=profile,
            expected_contract={"canonical_identity": contract},
        )


def test_identity_phase_does_not_claim_linked_physical_attestation_without_link(tmp_path) -> None:
    case = load_canonical_7b_identity_cases(PROMPTS_CSV)[0]
    manifest = build_canonical_identity_manifest(
        _canonical_base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=SnapshotBoundTokenizer.from_path(TOKENIZER_JSON),
        request_count=1,
        request_id_prefix="identity-case0",
    )
    schedule = build_homogeneous_identity_schedule(topology="tp2", request_count=1, global_wave_size=1)
    phase = _runner_identity_phase(schedule)
    executions = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    phase["full_output_artifact"] = runner.write_full_generation_records_artifact(
        tmp_path / "unlinked.outputs.jsonl.gz",
        records=_runner_identity_records(manifest, case.target),
        execution_records=executions,
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
    )

    phases, _summary = runner.canonical_identity_phase_artifacts(
        args=SimpleNamespace(
            canonical_identity_case=0,
            canonical_prompts_csv=PROMPTS_CSV,
            prompt_tokenizer_json=TOKENIZER_JSON,
            linked_proof_artifact=None,
        ),
        manifest=manifest,
        profile=SimpleNamespace(topology="tp2", global_wave_size=1),
        phase_artifacts=[phase],
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
        collect_physical_proof=False,
    )

    evidence = phases[0]["canonical_identity_evidence"]
    if evidence["physical_schedule"] is not None:
        raise AssertionError("proof-free identity unexpectedly retained direct physical proof")
    if evidence["physical_schedule_attested_by_linked_proof"] is not False:
        raise AssertionError("proof-free identity falsely claimed linked physical attestation")


def test_common_prefix_identity_production_caller_compares_serial_and_batched_outputs(tmp_path) -> None:
    case = load_common_prefix_identity_cases(PROMPTS_CSV)[0]
    serial_manifest = build_common_prefix_identity_manifest(
        _canonical_base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=SnapshotBoundTokenizer.from_path(TOKENIZER_JSON),
        request_count=1,
        request_id_prefix="common-serial",
    )
    candidate_manifest = build_common_prefix_identity_manifest(
        _canonical_base_manifest(),
        case=case,
        prompts_csv=PROMPTS_CSV,
        tokenizer=SnapshotBoundTokenizer.from_path(TOKENIZER_JSON),
        request_count=2,
        request_id_prefix="common-candidate",
    )

    def phase_artifact(manifest, *, phase_name):
        schedule = build_homogeneous_identity_schedule(
            topology="tp2",
            request_count=len(manifest.requests),
            global_wave_size=2,
        )
        phase = _runner_identity_phase(schedule, phase_name=phase_name)
        executions = runner.build_request_execution_records(
            manifest,
            global_request_offset=0,
            dp_rank=0,
            dp_size=1,
            generation_round=0,
            call_index=0,
        )
        phase["full_output_artifact"] = runner.write_full_generation_records_artifact(
            tmp_path / f"{phase_name}.outputs.jsonl.gz",
            records=_runner_identity_records(manifest, case.target),
            execution_records=executions,
            decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
        )
        return phase

    serial_phase = phase_artifact(serial_manifest, phase_name="common-serial")
    candidate_phase = phase_artifact(candidate_manifest, phase_name="steady-0")
    args = SimpleNamespace(
        common_prefix_identity_case=0,
        canonical_prompts_csv=PROMPTS_CSV,
        prompt_tokenizer_json=TOKENIZER_JSON,
    )
    profile = SimpleNamespace(topology="tp2", global_wave_size=2)

    phases, summary = runner.common_prefix_identity_phase_artifacts(
        args=args,
        manifest=candidate_manifest,
        profile=profile,
        serial_reference_phase=serial_phase,
        phase_artifacts=[candidate_phase],
        decode_output_token_ids=lambda token_ids: bytes(token_ids).decode("ascii"),
        collect_physical_proof=True,
    )

    assert summary["passed"] is True
    assert summary["serial_reference"]["physical_schedule"]["request_count"] == 1
    assert summary["phases"][0]["outputs"]["candidate_request_count"] == 2
    assert phases[0]["common_prefix_identity_evidence"] == summary["phases"][0]

    serial_schedule = build_homogeneous_identity_schedule(topology="tp2", request_count=1, global_wave_size=2)
    candidate_schedule = build_homogeneous_identity_schedule(topology="tp2", request_count=2, global_wave_size=2)
    contract = build_common_prefix_identity_contract(
        case=case,
        serial_schedule=serial_schedule,
        candidate_schedule=candidate_schedule,
        prompts_csv=PROMPTS_CSV,
        tokenizer_path=TOKENIZER_JSON,
    )
    artifact = {
        "phases": phases,
        "common_prefix_identity": summary,
        "common_prefix_serial_reference": serial_phase,
    }
    recomputed = runner.validate_common_prefix_identity_proof_evidence(
        artifact,
        manifest=candidate_manifest,
        profile=profile,
        expected_contract={"common_prefix_identity": contract},
    )
    assert recomputed == summary

    artifact["common_prefix_identity"]["phases"][0]["outputs"]["candidate_request_count"] = 99
    with pytest.raises(AssertionError, match="common-prefix identity"):
        runner.validate_common_prefix_identity_proof_evidence(
            artifact,
            manifest=candidate_manifest,
            profile=profile,
            expected_contract={"common_prefix_identity": contract},
        )


def test_prepare_workload_rejects_synthetic_prompt_or_id_rewrites_for_frozen_source(tmp_path) -> None:
    prompt_source = tmp_path / "matched.jsonl"
    prompt_source.write_text('{"id":"audit-0","prompt":"ACGT"}\n', encoding="utf-8")
    manifest = WorkloadManifest.from_path(DATA).with_prompt_jsonl(
        prompt_source,
        tokenizer=SnapshotBoundTokenizer.from_path(TOKENIZER_JSON),
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
    dna_token_ids = (65, 67, 71, 84)
    for index, request in enumerate(manifest.requests):
        token_ids = tuple(dna_token_ids[(index + position) % len(dna_token_ids)] for position in range(3))
        completion = SimpleNamespace(
            token_ids=token_ids,
            logprobs=[{token_id: SimpleNamespace(logprob=-0.1)} for token_id in token_ids],
            finish_reason="length",
            stop_reason=None,
        )
        outputs.append(
            SimpleNamespace(
                request_id=str(index),
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


def _synthetic_gpu_hardware() -> dict:
    gib = 1024**3
    reserved = 512 * 1024**2
    assignments = [
        {
            "logical_device_index": 0,
            "visible_device_selector": "0",
            "physical_index": 0,
            "uuid": "GPU-uuid-0",
            "pci_bus_id": "00000000:01:00.0",
        },
        {
            "logical_device_index": 1,
            "visible_device_selector": "1",
            "physical_index": 1,
            "uuid": "GPU-uuid-1",
            "pci_bus_id": "00000000:02:00.0",
        },
    ]
    devices = []
    for assignment in assignments:
        physical_index = assignment["physical_index"]
        physical_total = (80 + physical_index) * gib
        usable_total = physical_total - reserved
        devices.append(
            {
                **assignment,
                "expected_visible_device_selector": assignment["visible_device_selector"],
                "name": f"NVIDIA H100 {physical_index}",
                "torch_uuid": assignment["uuid"],
                "torch_name": f"NVIDIA H100 {physical_index}",
                "memory": {
                    "nvml": {
                        "physical_total_bytes": physical_total,
                        "system_reserved_bytes": reserved,
                        "free_bytes": usable_total - gib,
                        "used_bytes": gib,
                    },
                    "cuda": {
                        "usable_total_bytes": usable_total,
                        "free_bytes": usable_total - gib,
                    },
                    "torch_properties_total_bytes": usable_total,
                    "usable_total_relation_delta_bytes": 0,
                },
            }
        )
    return {
        "schema_version": 2,
        "passed": True,
        "cuda_visible_devices": "0,1",
        "expected_cuda_visible_devices": "0,1",
        "observed_visible_device_selectors": ["0", "1"],
        "expected_assignments": assignments,
        "required_initial_headroom_bytes": 2 * gib,
        "api_versions": {
            "nvml_python_version": "13.590.48",
            "nvml_module_path": "/runtime/pynvml.py",
            "nvml_memory_info_api": "nvmlDeviceGetMemoryInfo_v2",
            "torch_version": "2.10.0",
            "torch_cuda_version": "13.0",
            "torch_module_path": "/runtime/torch/__init__.py",
            "cuda_memory_info_api": "torch.cuda.mem_get_info",
        },
        "device_count": 2,
        "cuda_device_count": 2,
        "driver_version": "570.86.15",
        "nvml_cuda_driver_version_integer": 13_000,
        "nvml_memory_info_version": 2,
        "devices": devices,
    }


def _write_valid_direct_proof_artifact(
    tmp_path,
    *,
    generation_round: int = 7,
    global_request_index_start: int = 0,
    shared_prefix: bool = False,
    exact_progress: bool = False,
) -> tuple[Path, dict, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    proof_path = tmp_path / "proof.json"
    manifest = (
        _shared_prefix_manifest() if shared_prefix else WorkloadManifest.from_path(DATA).request_slice(0, 2)
    ).with_max_new_tokens(3)
    prefix_args = ["--shared-prefix-state-reuse"] if shared_prefix else []
    progress_args = ["--exact-progress-gate"] if exact_progress else []
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
            *progress_args,
            "--output",
            str(proof_path),
        ]
    )
    profile = runner.profile_from_args(args, manifest)
    compilation = _compilation_snapshot()
    hardware = _synthetic_gpu_hardware()
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
            sampler_installation=_sampler_installation_provenance(),
            gpu_hardware=hardware,
        ),
    }
    peak_memory = (70 * 1024**3, 71 * 1024**3)
    phases = []
    call_index = generation_round
    for sample_index, phase_name in enumerate(("cold-generation", "steady-0")):
        wave_phase = f"{phase_name}.wave-000"
        executions = runner.build_request_execution_records(
            manifest,
            global_request_offset=global_request_index_start,
            dp_rank=0,
            dp_size=1,
            generation_round=generation_round,
            call_index=call_index,
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
            "generation_round": generation_round,
            "call_index": call_index,
            "generation_s": 1.0,
            "full_decode_proof": full_decode,
            "scheduler_observations": scheduler_observations,
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
                "engine_seed": manifest.seed,
                "sampler": _sampler_worker_proof((tuple(execution.seed for execution in executions),)),
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
                "generation_timing_authority": runner.COORDINATOR_GENERATION_TIMING_AUTHORITY,
                "coordinator_generation_wall_s": 1.0,
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
                "exact_generation_progress": (
                    runner.exact_generation_progress_evidence(
                        manifest,
                        sidecar_counts={
                            "request_count": sidecar["request_count"],
                            "output_token_id_count": sidecar["output_token_id_count"],
                            "chosen_token_logprob_count": sidecar["chosen_token_logprob_count"],
                        },
                        full_decode_summaries=[full_decode],
                    )
                    if exact_progress
                    else None
                ),
                "worker_proof": workers,
                "shared_prefix_state_reuse": prefix_reuse,
                "proof_collected": True,
                "prefix_cache_reset": shared_prefix,
            }
        )
    phases, exact_progress_evidence = runner.attach_exact_generation_progress_evidence(
        phases,
        manifest=manifest,
        enabled=exact_progress,
        proof_collected=True,
        topology="tp2",
        linked_proof_artifact=None,
    )
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
            "engine_seed": manifest.seed,
            "sampler": _sampler_worker_proof(()),
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
        "sampler_installation_provenance": _sampler_installation_provenance(),
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
        "exact_generation_progress": exact_progress_evidence,
    }
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")
    return proof_path, contract, artifact


def _fixture_caller_profile(topology: str) -> runner.Evo2VllmProfile:
    return runner.Evo2VllmProfile(
        topology=topology,
        max_model_len=64,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.95,
        global_wave_size=2,
        max_num_seqs=2 if topology == "tp2" else 1,
    )


def _fixture_caller_coordinates(artifact: dict, contract: dict) -> runner.CallerCoordinateContract:
    profile_data = dict(artifact["profile"])
    profile_data["proof"] = False
    return runner.CallerCoordinateContract.from_inputs(
        WorkloadManifest.from_dict(artifact["manifest"]),
        runner.Evo2VllmProfile(**profile_data),
        contract["seed_stream"]["generation_round"],
    )


def _write_valid_dp2_proof_artifact(
    tmp_path,
    *,
    generation_round: int = 0,
    global_request_index_start: int = 0,
) -> tuple[Path, dict, dict]:
    _, _, base = _write_valid_direct_proof_artifact(tmp_path / "direct-base")
    tmp_path.mkdir(parents=True, exist_ok=True)
    proof_path = tmp_path / "proof.json"
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
            str(generation_round),
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
            sampler_installation=base["sampler_installation_provenance"],
            gpu_hardware=hardware,
        ),
    }
    compilation = _compilation_snapshot()
    resolved = profile.expected_resolved_config()
    phases = []
    call_index = generation_round
    global_index = global_request_index_start
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
                generation_round=generation_round,
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
                "engine_seed": manifest.seed,
                "sampler": _sampler_worker_proof(
                    (tuple(execution.seed for execution in executions if execution.dp_rank == shard.replica_index),)
                ),
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
            "generation_round": generation_round,
            "call_index": call_index,
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
                "generation_timing_authority": runner.COORDINATOR_GENERATION_TIMING_AUTHORITY,
                "coordinator_generation_wall_s": 1.0,
                "wave_execution": runner.wave_execution_summary([wave_proof]),
                "outputs": [record.summary_dict() for record in records],
                "request_executions": [record.to_dict() for record in executions],
                "full_output_artifact": sidecar,
                "waves": [wave_proof],
                "proof_collected": True,
                "prefix_cache_reset": False,
            }
        )
    initialized = [
        {
            "phase": "engine-initialized",
            "resolved_config": resolved,
            "worker_proof": [
                {
                    **phases[0]["waves"][0]["engines"][dp_rank]["worker_proof"][0],
                    "sampler": _sampler_worker_proof(()),
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
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(tmp_path)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)

    evidence = runner.validate_linked_proof_artifact(
        proof_path,
        expected_contract=contract,
        caller_coordinates=caller_coordinates,
        require_memory_headroom=True,
    )

    if evidence["artifact_path"] != str(proof_path.resolve()):
        raise AssertionError("linked direct proof path drifted")
    if evidence["artifact_sha256"] != hashlib.sha256(proof_path.read_bytes()).hexdigest():
        raise AssertionError("linked direct proof digest drifted")


def test_linked_proof_rejects_consistent_tp2_coordinate_shift_from_external_caller_contract(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    caller_coordinates = runner.CallerCoordinateContract.from_inputs(
        manifest,
        _fixture_caller_profile("tp2"),
        7,
    )
    proof_path, shifted_contract, _ = _write_valid_direct_proof_artifact(
        tmp_path,
        generation_round=8,
        global_request_index_start=100,
    )

    with pytest.raises(AssertionError, match="caller coordinate"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=shifted_contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


def test_linked_proof_rejects_consistent_dp2_coordinate_shift_from_external_caller_contract(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    caller_coordinates = runner.CallerCoordinateContract.from_inputs(
        manifest,
        _fixture_caller_profile("dp2"),
        0,
    )
    proof_path, shifted_contract, _ = _write_valid_dp2_proof_artifact(
        tmp_path,
        generation_round=1,
        global_request_index_start=100,
    )

    with pytest.raises(AssertionError, match="caller coordinate"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=shifted_contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


def test_linked_proof_rejects_duplicate_root_and_sidecar_json_keys(tmp_path) -> None:
    root_dir = tmp_path / "duplicate-root"
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(root_dir)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
    payload = proof_path.read_text(encoding="utf-8")
    proof_path.write_text(
        '{"benchmark_mode":"foreign",' + payload.lstrip()[1:],
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key 'benchmark_mode'"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )

    sidecar_dir = tmp_path / "duplicate-sidecar"
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(sidecar_dir)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
    metadata = artifact["phases"][0]["full_output_artifact"]
    sidecar = Path(metadata["path"])
    uncompressed = gzip.decompress(sidecar.read_bytes())
    first_row = json.loads(uncompressed.splitlines()[0])
    request_field = f'"request_id":"{first_row["request_id"]}"'.encode()
    tampered = uncompressed.replace(request_field, request_field + b"," + request_field, 1)
    compressed = gzip.compress(tampered, mtime=0)
    sidecar.write_bytes(compressed)
    metadata["sha256"] = hashlib.sha256(compressed).hexdigest()
    metadata["size_bytes"] = len(compressed)
    proof_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    with pytest.raises(AssertionError, match="could not be decoded"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


def test_linked_proof_digest_remains_bound_to_captured_root_after_path_replacement(
    tmp_path,
    monkeypatch,
) -> None:
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(tmp_path)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
    original = proof_path.read_bytes()
    foreign = b'{"benchmark_mode":"foreign"}\n'
    real_reader = runner.read_json_snapshot
    replaced = False

    def replace_after_snapshot(path, *, label):
        nonlocal replaced
        snapshot = real_reader(path, label=label)
        if Path(path).resolve() == proof_path.resolve() and not replaced:
            proof_path.write_bytes(foreign)
            replaced = True
        return snapshot

    monkeypatch.setattr(runner, "read_json_snapshot", replace_after_snapshot)
    evidence = runner.validate_linked_proof_artifact(
        proof_path,
        expected_contract=contract,
        caller_coordinates=caller_coordinates,
        require_memory_headroom=True,
    )

    assert replaced is True
    assert evidence["artifact_sha256"] == hashlib.sha256(original).hexdigest()
    assert proof_path.read_bytes() == foreign


def test_linked_proof_rejects_coherent_foreign_profile_from_external_caller_contract(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    caller_profile = replace(
        _fixture_caller_profile("tp2"),
        max_num_batched_tokens=32_768,
    )
    caller_coordinates = runner.CallerCoordinateContract.from_inputs(
        manifest,
        caller_profile,
        7,
    )
    proof_path, foreign_contract, _ = _write_valid_direct_proof_artifact(tmp_path)

    with pytest.raises(AssertionError, match="caller coordinate"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=foreign_contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_tp_rank",
        "extra_tp_rank",
        "duplicate_tp_rank",
        "policy_sampling_drift",
        "profile_config_drift",
        "global_wave_size_drift",
        "row_truncation",
        "row_extension",
    ),
)
def test_linked_proof_caller_binding_rejects_tp2_admission_matrix(tmp_path, tamper) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    caller_manifest = manifest
    caller_profile = _fixture_caller_profile("tp2")
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(tmp_path / tamper)

    if tamper == "missing_tp_rank":
        artifact["phases"][0]["worker_proof"].pop()
    elif tamper == "extra_tp_rank":
        extra = copy.deepcopy(artifact["phases"][0]["worker_proof"][-1])
        extra["rank"] = 2
        artifact["phases"][0]["worker_proof"].append(extra)
    elif tamper == "duplicate_tp_rank":
        artifact["phases"][0]["worker_proof"][1]["rank"] = 0
    elif tamper == "policy_sampling_drift":
        caller_manifest = replace(manifest, top_k=manifest.top_k + 1)
    elif tamper == "profile_config_drift":
        caller_profile = replace(caller_profile, max_num_batched_tokens=32_768)
    elif tamper == "global_wave_size_drift":
        caller_profile = replace(caller_profile, global_wave_size=1, max_num_seqs=1)
    elif tamper == "row_truncation":
        artifact["phases"][0]["request_executions"].pop()
    elif tamper == "row_extension":
        artifact["phases"][0]["request_executions"].append(
            copy.deepcopy(artifact["phases"][0]["request_executions"][-1])
        )
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")
    caller_coordinates = runner.CallerCoordinateContract.from_inputs(
        caller_manifest,
        caller_profile,
        7,
    )

    with pytest.raises(AssertionError):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_dp_rank",
        "extra_dp_rank",
        "duplicate_dp_rank",
        "rank_row_truncation",
        "rank_row_extension",
    ),
)
def test_linked_proof_caller_binding_rejects_dp2_admission_matrix(tmp_path, tamper) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    caller_coordinates = runner.CallerCoordinateContract.from_inputs(
        manifest,
        _fixture_caller_profile("dp2"),
        0,
    )
    proof_path, contract, artifact = _write_valid_dp2_proof_artifact(tmp_path / tamper)
    engines = artifact["phases"][0]["waves"][0]["engines"]
    executions = artifact["phases"][0]["request_executions"]

    if tamper == "missing_dp_rank":
        engines.pop()
    elif tamper == "extra_dp_rank":
        extra = copy.deepcopy(engines[-1])
        extra["dp_rank"] = 2
        engines.append(extra)
    elif tamper == "duplicate_dp_rank":
        engines[1]["dp_rank"] = 0
    elif tamper == "rank_row_truncation":
        executions.pop(next(index for index, row in enumerate(executions) if row["dp_rank"] == 1))
    elif tamper == "rank_row_extension":
        executions.append(copy.deepcopy(next(row for row in executions if row["dp_rank"] == 1)))
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(AssertionError):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


def test_linked_proof_validator_rejects_sampler_seed_batch_outside_physical_wave(tmp_path) -> None:
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(tmp_path)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
    descriptor = artifact["phases"][0]["worker_proof"][0]["sampler"]["generation_batch_descriptors"][0]
    descriptor["generator_seeds"][0] += 1
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(AssertionError, match="batch descriptor drifted from exact requests"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


@pytest.mark.parametrize("topology", ("tp2", "dp2"))
def test_linked_sampler_admission_uses_caller_reopened_sidecar_before_generation_validation(
    tmp_path,
    monkeypatch,
    topology: str,
) -> None:
    writer = _write_valid_direct_proof_artifact if topology == "tp2" else _write_valid_dp2_proof_artifact
    proof_path, contract, artifact = writer(tmp_path / topology)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
    events = []
    original_sidecar = runner._validate_full_output_sidecar
    original_sampler = runner._validate_worker_sampler_evidence

    def validate_sidecar(*args, **kwargs):
        result = original_sidecar(*args, **kwargs)
        events.append(("sidecar", tuple(copy.deepcopy(result["request_generations"]))))
        return result

    def validate_sampler(*args, **kwargs):
        rows = tuple(copy.deepcopy(kwargs["expected_request_generations"]))
        events.append(("sampler", rows))
        return original_sampler(*args, **kwargs)

    monkeypatch.setattr(runner, "_validate_full_output_sidecar", validate_sidecar)
    monkeypatch.setattr(runner, "_validate_worker_sampler_evidence", validate_sampler)

    runner.validate_linked_proof_artifact(
        proof_path,
        expected_contract=contract,
        caller_coordinates=caller_coordinates,
        require_memory_headroom=True,
    )

    sidecar_indices = [index for index, event in enumerate(events) if event[0] == "sidecar"]
    if not sidecar_indices:
        raise AssertionError("linked sampler admission never reopened the caller sidecar")
    for phase_index, sidecar_index in enumerate(sidecar_indices):
        stop = sidecar_indices[phase_index + 1] if phase_index + 1 < len(sidecar_indices) else len(events)
        generation_events = [
            rows
            for kind, rows in events[sidecar_index + 1 : stop]
            if kind == "sampler" and rows
        ]
        if not generation_events:
            raise AssertionError("one proof phase did not validate sampler evidence after sidecar reopen")
        joined = [row for rows in generation_events for row in rows]
        sidecar_rows = list(events[sidecar_index][1])
        if sorted(joined, key=lambda row: (row["request_id"], row["seed"])) != sorted(
            sidecar_rows,
            key=lambda row: (row["request_id"], row["seed"]),
        ):
            raise AssertionError("sampler shard admission is not disjoint and complete over the caller sidecar")
        if any(row["accepted_output_token_count"] != 3 for row in joined):
            raise AssertionError("sampler accepted-token counts did not come from retained sidecar rows")


def test_linked_proof_validator_rejects_tp2_summary_without_raw_scheduler_evidence(tmp_path) -> None:
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(tmp_path)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
    wave = artifact["phases"][0]["wave_proofs"][0]
    wave.pop("scheduler_observations")
    wave["scheduler_capacity_proof"].update(
        {
            "scheduler_observation_count": 999,
            "prompt_tokens_computed": 0,
            "prompt_tokens_cached": 999_999,
            "prompt_tokens_total": 1,
            "maximum_waiting_requests": 999_999,
            "maximum_skipped_waiting_requests": 999_999,
            "passed": True,
        }
    )
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(AssertionError, match="scheduler"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


def test_linked_proof_validator_rejects_foreign_phase_scheduler_observations(tmp_path) -> None:
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(tmp_path)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
    observations = artifact["phases"][0]["wave_proofs"][0]["scheduler_observations"]
    foreign_observation = copy.deepcopy(observations[0])
    foreign_observation["phase"] = "foreign.wave-000"
    observations.append(foreign_observation)
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(AssertionError, match="scheduler"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


def test_linked_proof_validator_rejects_cuda_observations_outside_their_physical_wave(tmp_path) -> None:
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(tmp_path)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
    artifact["phases"][0]["cudagraph_observations_retained"][0]["phase"] = "foreign.wave-000"
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(AssertionError, match=r"CUDA|wave|observation"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


def test_cuda_phase_evidence_rejects_truncated_middle_observations() -> None:
    observation = {
        "phase": "steady-0.wave-000",
        "engine_index": 0,
        "num_unpadded_tokens": 2,
        "num_padded_tokens": 2,
        "num_paddings": 0,
        "runtime_mode": "CUDAGraphMode.FULL",
    }
    complete = [dict(observation) for _ in range(257)]
    phase = {
        "cudagraph_observation_count": len(complete),
        "cudagraph_observations_retained": [*complete[:128], *complete[-128:]],
        "cudagraph_summary": runner.summarize_cudagraph_observations(tuple(complete)),
    }

    with pytest.raises(AssertionError, match=r"complete|lossless|observation"):
        runner._validate_cudagraph_phase_evidence(phase, maximum_wave_size=2)


def test_generation_phase_result_serializes_every_cuda_observation_losslessly() -> None:
    observations = tuple(
        {
            "phase": "steady-0.wave-000",
            "engine_index": 0,
            "num_unpadded_tokens": 2,
            "num_padded_tokens": 2,
            "num_paddings": 0,
            "runtime_mode": "CUDAGraphMode.FULL",
        }
        for _ in range(257)
    )
    result = runner.GenerationPhaseResult(
        phase="steady-0",
        sample=SimpleNamespace(generation_s=1.0, to_dict=lambda: {}),
        generation_call_s=(1.0,),
        wave_proofs=({"wave_index": 0, "request_count": 2, "generation_s": 1.0},),
        observations=observations,
        output_summaries=(),
        request_executions=(),
        full_output_artifact={},
        full_decode_proof=None,
        worker_proof=(),
        shared_prefix_state_reuse=None,
        proof_collected=True,
        prefix_cache_reset=False,
    )

    serialized = result.to_dict()

    assert serialized["cudagraph_observation_count"] == 257
    assert serialized["cudagraph_observations_retained"] == list(observations)
    assert serialized["cudagraph_summary"][0]["count"] == 257


def test_linked_proof_validator_accepts_and_recomputes_dp2_engine_evidence(tmp_path) -> None:
    proof_path, contract, artifact = _write_valid_dp2_proof_artifact(tmp_path)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)

    evidence = runner.validate_linked_proof_artifact(
        proof_path,
        expected_contract=contract,
        caller_coordinates=caller_coordinates,
        require_memory_headroom=True,
    )

    if evidence["validated_evidence"]["final_worker_count"] != 2:
        raise AssertionError("linked DP2 proof did not retain both final workers")


@pytest.mark.parametrize("tamper", ("scheduler_raw", "resolved_config"))
def test_linked_proof_validator_rejects_tampered_dp2_engine_evidence(tmp_path, tamper) -> None:
    proof_path, contract, artifact = _write_valid_dp2_proof_artifact(tmp_path / tamper)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
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
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


def test_linked_proof_validator_recomputes_physical_prefix_reuse(tmp_path) -> None:
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(
        tmp_path,
        shared_prefix=True,
    )
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
    runner.validate_linked_proof_artifact(
        proof_path,
        expected_contract=contract,
        caller_coordinates=caller_coordinates,
        require_memory_headroom=True,
    )

    artifact["phases"][0]["shared_prefix_state_reuse"]["cache_hit_request_count"] = 0
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(AssertionError, match="prefix"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


def test_exact_progress_gate_is_linked_and_recomputed_from_raw_proof(tmp_path) -> None:
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(
        tmp_path,
        exact_progress=True,
    )
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)

    assert contract["exact_generation_progress"] == {
        "schema_version": 1,
        "request_count": 2,
        "max_new_tokens": 3,
        "expected_first_sampled_tokens": 2,
        "expected_decode_token_updates": 4,
        "expected_retained_output_token_ids": 6,
        "expected_retained_chosen_token_logprobs": 6,
    }
    runner.validate_linked_proof_artifact(
        proof_path,
        expected_contract=contract,
        caller_coordinates=caller_coordinates,
        require_memory_headroom=True,
    )

    artifact["phases"][0]["exact_generation_progress"]["observed_full_decode_token_updates"] -= 1
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(AssertionError, match="exact-generation|exact generation"):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
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
        "engine_seed",
        "phase_gpu_reassignment",
        "runtime_attestation",
        "memory",
    ),
)
def test_linked_proof_validator_recomputes_all_direct_evidence(tmp_path, tamper) -> None:
    proof_path, contract, artifact = _write_valid_direct_proof_artifact(tmp_path / tamper)
    caller_coordinates = _fixture_caller_coordinates(artifact, contract)
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
    elif tamper == "engine_seed":
        artifact["phases"][0]["worker_proof"][1]["engine_seed"] += 1
    elif tamper == "phase_gpu_reassignment":
        workers = artifact["phases"][0]["worker_proof"]
        fields = (
            "device",
            "logical_device",
            "device_name",
            "device_uuid",
            "pci_bus_id",
            "visible_device_selector",
        )
        first = {field: workers[0][field] for field in fields}
        second = {field: workers[1][field] for field in fields}
        workers[0].update(second)
        workers[1].update(first)
    elif tamper == "runtime_attestation":
        artifact["gpu_hardware_provenance"]["driver_version"] = "forged-driver"
    elif tamper == "memory":
        artifact["gpu_memory_headroom"]["devices"][0]["headroom_bytes"] += 1
    proof_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(AssertionError):
        runner.validate_linked_proof_artifact(
            proof_path,
            expected_contract=contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )


def test_benchmark_contract_pins_generation_round_seed_stream(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).with_request_count(1_000, request_id_prefix="audit")
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

    first_seed_stream = first_contract["seed_stream"]
    capacity = first_seed_stream["capacity_proof"]
    expected_seed_stream = {
        "schema_version": 3,
        "base_seed": manifest.seed,
        "generation_round": 7,
        "physical_calls_per_round": 11,
        "global_call_index_start": 77,
        "round_stride": 1_000_003,
        "modulus": 2**31,
    }
    if {key: first_seed_stream[key] for key in expected_seed_stream} != expected_seed_stream:
        raise AssertionError("generation-round seed-stream coordinates changed")
    if (
        capacity["maximum_pre_modulo_seed"] != 87_000_342
        or capacity["maximum_coordinates"]
        != {
            "call_in_round": 10,
            "global_call_index": 87,
            "dp_rank": 0,
            "request_index_in_stream": 39,
        }
        or [call["global_call_index"] for call in capacity["physical_calls"]] != list(range(77, 88))
        or [call["global_request_count"] for call in capacity["physical_calls"]] != [96] * 10 + [40]
        or capacity["passed"] is not True
    ):
        raise AssertionError("generation-round seed-capacity proof changed")
    if second_contract["seed_stream"]["global_call_index_start"] != 88:
        raise AssertionError("second generation round did not advance by eleven physical calls")
    if first_contract == second_contract:
        raise AssertionError("distinct generation rounds produced an identical benchmark contract")


def test_generation_phase_times_one_complete_batch_and_preserves_exact_outputs(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    recorder = CUDAGraphProofRecorder()
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
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
    assert artifact["full_output_artifact"]["output_token_id_count"] == 6
    assert artifact["full_output_artifact"]["chosen_token_logprob_count"] == 6
    assert artifact["full_decode_proof"]["passed"] is True
    assert artifact["full_decode_proof"]["full_decode_tokens"] == 4


def _rank_local_test_evidence(
    outputs,
    *,
    phase,
    contract_sha256,
    tp_rank,
    request_order,
):
    rows_by_id = {
        row["vllm_request_id"]: row
        for row in runner.rank_local_selected_stream_rows(outputs)
    }
    requests = [rows_by_id[request_id] for request_id in request_order]
    aggregate = hashlib.sha256(
        json.dumps(requests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "evo2-rank-local-generation-evidence/v1",
        "source": "rank_local_model_runner_execute_or_sample",
        "tp_rank": tp_rank,
        "phase": phase,
        "expected_envelope_sha256": contract_sha256,
        "request_count": len(requests),
        "generated_token_count": sum(row["token_count"] for row in requests),
        "execution_call_count": 3,
        "request_order": list(request_order),
        "requests": requests,
        "aggregate_selected_stream_sha256": aggregate,
    }


def test_generation_phase_consumes_every_tp_rank_witness_for_reordered_short_final_wave(
    tmp_path,
) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 12).with_max_new_tokens(3)
    recorder = CUDAGraphProofRecorder()
    execution_records = runner.build_wave_execution_records(
        manifest,
        global_wave_size=10,
        generation_round=4,
        call_index_start=44,
    )
    state = {"cursor": 0, "latest": None, "begin": None}
    events = []

    class FakeLLM:
        def generate(self, prompts, sampling_params, *, use_tqdm):
            request_count = len(prompts)
            start = state["cursor"]
            wave_manifest = manifest.request_slice(start, start + request_count)
            outputs = _fake_outputs(wave_manifest)
            for offset, output in enumerate(outputs):
                output.request_id = str(8 + start + offset)
            state["cursor"] += request_count
            state["latest"] = outputs
            events.append(("generate", request_count))
            for _ in range(2):
                recorder.record(
                    _scheduler_stats(
                        unpadded=request_count,
                        padded=request_count,
                        mode="CUDAGraphMode.FULL",
                    ),
                    _iteration_stats(computed=request_count, total=request_count),
                )
            return outputs

    def begin(**kwargs):
        state["begin"] = kwargs
        events.append(("begin", kwargs["expected_request_count"]))
        return tuple(
            {
                "tp_rank": tp_rank,
                "phase": kwargs["phase"],
                "expected_envelope_sha256": kwargs["contract_sha256"],
                "expected_request_count": kwargs["expected_request_count"],
                "expected_max_new_tokens": kwargs["expected_max_new_tokens"],
                "source": "rank_local_model_runner_execute_or_sample",
            }
            for tp_rank in range(2)
        )

    def snapshot():
        kwargs = state["begin"]
        outputs = state["latest"]
        request_order = [output.request_id for output in outputs]
        request_order = request_order[-1:] + request_order[:-1]
        events.append(("snapshot", len(request_order)))
        return tuple(
            _rank_local_test_evidence(
                outputs,
                phase=kwargs["phase"],
                contract_sha256=kwargs["contract_sha256"],
                tp_rank=tp_rank,
                request_order=request_order,
            )
            for tp_rank in range(2)
        )

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
        full_output_path=tmp_path / "tp-rank.outputs.jsonl.gz",
        require_rank_local_evidence=True,
        rank_local_semantic_namespace="proof-plan/steady-0",
        expected_tensor_parallel_size=2,
        begin_rank_local_evidence=begin,
        snapshot_rank_local_evidence=snapshot,
        abort_rank_local_evidence=lambda: events.append(("abort", 0)),
        global_wave_size=10,
        scheduler_max_num_seqs=10,
        clock=iter((10.0, 11.0, 20.0, 21.0)).__next__,
    )

    assert events == [
        ("begin", 10),
        ("generate", 10),
        ("snapshot", 10),
        ("begin", 2),
        ("generate", 2),
        ("snapshot", 2),
    ]
    assert [wave["request_count"] for wave in result.wave_proofs] == [10, 2]
    for wave in result.wave_proofs:
        evidence = wave["rank_local_generation_evidence"]
        assert evidence["passed"] is True
        assert evidence["tensor_parallel_ranks"] == [0, 1]
        assert [
            row["request_ordinal"] for row in evidence["semantic_streams"]
        ] == list(range(wave["request_count"]))
        assert all(
            row["qualified_request_identity"]
            == ["proof-plan/steady-0", row["local_request_id"]]
            for row in evidence["semantic_streams"]
        )


def test_generation_phase_aborts_rank_epoch_when_one_tp_witness_is_missing(
    tmp_path,
) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    recorder = CUDAGraphProofRecorder()
    execution_records = runner.build_wave_execution_records(
        manifest,
        global_wave_size=2,
        generation_round=0,
        call_index_start=0,
    )
    outputs = _fake_outputs(manifest)
    for index, output in enumerate(outputs):
        output.request_id = str(10 + index)
    contract = {}
    events = []

    class FakeLLM:
        def generate(self, prompts, sampling_params, *, use_tqdm):
            events.append("generate")
            for _ in range(2):
                recorder.record(
                    _scheduler_stats(
                        unpadded=2,
                        padded=2,
                        mode="CUDAGraphMode.FULL",
                    ),
                    _iteration_stats(computed=2, total=2),
                )
            return outputs

    def begin(**kwargs):
        contract.update(kwargs)
        events.append("begin")
        return tuple(
            {
                "tp_rank": tp_rank,
                "phase": kwargs["phase"],
                "expected_envelope_sha256": kwargs["contract_sha256"],
                "expected_request_count": kwargs["expected_request_count"],
                "expected_max_new_tokens": kwargs["expected_max_new_tokens"],
                "source": "rank_local_model_runner_execute_or_sample",
            }
            for tp_rank in range(2)
        )

    def snapshot():
        events.append("snapshot")
        return (
            _rank_local_test_evidence(
                outputs,
                phase=contract["phase"],
                contract_sha256=contract["contract_sha256"],
                tp_rank=0,
                request_order=[output.request_id for output in outputs],
            ),
        )

    with pytest.raises(RuntimeError, match="TP ranks are incomplete"):
        run_generation_phase(
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
            full_output_path=tmp_path / "missing-rank.outputs.jsonl.gz",
            require_rank_local_evidence=True,
            rank_local_semantic_namespace="proof-plan/steady-0",
            expected_tensor_parallel_size=2,
            begin_rank_local_evidence=begin,
            snapshot_rank_local_evidence=snapshot,
            abort_rank_local_evidence=lambda: events.append("abort"),
            global_wave_size=2,
            scheduler_max_num_seqs=2,
            clock=iter((10.0, 11.0)).__next__,
        )

    assert events == ["begin", "generate", "snapshot", "abort"]
    assert not (tmp_path / "missing-rank.outputs.jsonl.gz").exists()


def test_generation_phase_skips_post_generation_evidence_after_namespace_ownership_loss(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    output = tmp_path / "proof.json"
    marker = runner.reserve_output_namespace(output)
    sidecar = runner.phase_output_artifact_path(output, phase="steady-0")
    foreign = tmp_path / "foreign-marker"
    foreign_payload = "foreign owner\n"
    foreign.write_text(foreign_payload, encoding="utf-8")
    recorder = CUDAGraphProofRecorder()
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )
    proof_events = []

    class OwnershipLosingLLM:
        def generate(self, prompts, sampling_params, *, use_tqdm):
            foreign.replace(marker)
            return _fake_outputs(manifest)

    with pytest.raises(RuntimeError, match="ownership"):
        run_generation_phase(
            llm=OwnershipLosingLLM(),
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
            full_output_path=sidecar,
            namespace_output_path=output,
            reset_worker_proof=lambda: proof_events.append("reset"),
            snapshot_worker_proof=lambda: proof_events.append("snapshot"),
            clock=iter((10.0, 11.0)).__next__,
        )

    assert proof_events == ["reset"]
    assert not sidecar.exists()
    assert marker.read_text(encoding="utf-8") == foreign_payload


def test_generation_phase_registers_its_sidecar_with_the_output_namespace(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    output = tmp_path / "result.json"
    marker = runner.reserve_output_namespace(output)
    sidecar = runner.phase_output_artifact_path(output, phase="steady-0")
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )

    class FakeLLM:
        def generate(self, prompts, sampling_params, *, use_tqdm):
            return _fake_outputs(manifest)

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
        recorder=CUDAGraphProofRecorder(),
        memory_monitor_factory=lambda: PeakMemoryMonitor(lambda: (1_000, 2_000)),
        execution_records=execution_records,
        full_output_path=sidecar,
        namespace_output_path=output,
        collect_proof=False,
        clock=iter((10.0, 11.0)).__next__,
    )
    runner.write_json_artifact(
        output,
        {"phase": result.phase},
        ownership_validator=lambda: runner.require_output_namespace_reservation(output),
    )

    runner.complete_output_namespace(marker, output_path=output)

    if marker.exists() or not output.is_file() or not sidecar.is_file():
        raise AssertionError("generation phase publications did not complete as one owned namespace")


def test_speed_generation_avoids_proof_callbacks_and_memory_polling(tmp_path) -> None:
    manifest = _shared_prefix_manifest().with_max_new_tokens(3)
    execution_records = runner.build_request_execution_records(
        manifest,
        global_request_offset=0,
        dp_rank=0,
        dp_size=1,
        generation_round=0,
        call_index=0,
    )

    class FakeLLM:
        cache_resets = 0

        def reset_prefix_cache(self):
            self.cache_resets += 1
            return True

        def generate(self, prompts, sampling_params, *, use_tqdm):
            if len(prompts) != 2 or len(sampling_params) != 2 or use_tqdm is not False:
                raise AssertionError("speed lane changed the exact generation call")
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

    if llm.cache_resets != 1 or result.sample.generation_s != 1.0:
        raise AssertionError("speed lane cache reset or generation timing drifted")
    if result.sample.peak_device_memory_bytes != ():
        raise AssertionError("speed lane fabricated peak-memory observations")
    if (
        result.proof_collected is not False
        or result.prefix_cache_reset is not True
        or result.observations != ()
        or result.worker_proof != ()
        or result.shared_prefix_state_reuse is not None
        or result.full_decode_proof is not None
        or result.wave_proofs[0]["full_decode_proof"] is not None
        or result.wave_proofs[0]["scheduler_capacity_proof"] is not None
    ):
        raise AssertionError("speed lane retained proof-only instrumentation")
    if result.full_output_artifact["generated_token_count"] != 6:
        raise AssertionError("speed lane did not retain every generated token")


def test_generation_timer_excludes_worker_sampler_endpoint_snapshot(tmp_path) -> None:
    manifest = WorkloadManifest.from_path(DATA).request_slice(0, 2).with_max_new_tokens(3)
    execution_records = runner.build_wave_execution_records(
        manifest,
        global_wave_size=2,
        generation_round=0,
        call_index_start=0,
    )
    recorder = CUDAGraphProofRecorder()
    events = []

    class FakeLLM:
        def generate(self, prompts, sampling_params, *, use_tqdm):
            events.append("generate")
            for _ in range(2):
                recorder.record(
                    _scheduler_stats(unpadded=2, padded=2, mode="CUDAGraphMode.FULL"),
                    _iteration_stats(),
                )
            return _fake_outputs(manifest)

    times = iter((10.0, 11.0))

    def clock():
        events.append("clock")
        return next(times)

    def snapshot_worker_proof():
        events.append("endpoint-state-snapshot")
        return ({"sampler": "phase-end"},)

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
        full_output_path=tmp_path / "proof.outputs.jsonl.gz",
        reset_worker_proof=lambda: events.append("reset-worker-proof"),
        snapshot_worker_proof=snapshot_worker_proof,
        global_wave_size=2,
        scheduler_max_num_seqs=2,
        clock=clock,
    )

    if events != [
        "reset-worker-proof",
        "clock",
        "generate",
        "clock",
        "endpoint-state-snapshot",
    ]:
        raise AssertionError("sampler endpoint snapshot overlapped the coordinator generation timer")
    if result.sample.generation_s != 1.0:
        raise AssertionError("generation timer included post-generation sampler proof work")


def test_generation_phase_round1_executes_global_calls_11_through_21_and_replays_exactly(tmp_path) -> None:
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
                    call_index=11 + wave_index,
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
        generation_round=1,
        call_index_start=11,
    )
    assert execution_records == runner.build_wave_execution_records(
        manifest,
        global_wave_size=96,
        generation_round=1,
        call_index_start=11,
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
    assert [proof["generation_round"] for proof in result.wave_proofs] == [1] * 11
    assert [proof["call_index"] for proof in result.wave_proofs] == list(range(11, 22))
    assert {record.generation_round for record in result.request_executions} == {1}
    assert [record.seed for record in result.request_executions[:96]] == list(range(11_000_075, 11_000_171))
    assert [record.seed for record in result.request_executions[96:192]] == list(range(12_000_078, 12_000_174))
    assert [record.seed for record in result.request_executions[-40:]] == list(range(21_000_105, 21_000_145))
    assert all(proof["scheduler_capacity_proof"]["batch_fit_without_preemption"] for proof in result.wave_proofs)
    assert [len(proof["scheduler_observations"]) for proof in result.wave_proofs] == [2] * 11
    assert all(
        observation["phase"] == f"steady-0.wave-{wave_index:03d}"
        for wave_index, proof in enumerate(result.wave_proofs)
        for observation in proof["scheduler_observations"]
    )
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
    runtime_state_layout = [
        {
            key: entry[key]
            for key in (
                "kv_cache_group_id",
                "layer_name",
                "state_index",
                "dtype",
                "state_shape",
                "block_shape",
                "copied_elements",
                "copied_bytes",
            )
        }
        for entry in state_copies
    ]
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
        "runtime_state_layout": runtime_state_layout,
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
        generation_round=0,
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


def test_shared_prefix_state_reuse_evidence_rejects_empty_self_attested_state_copy() -> None:
    manifest = _shared_prefix_manifest()
    clone = _prefix_clone_record("empty")
    clone.update(
        {
            "state_copies": [],
            "copy_entries": 0,
            "copied_elements": 0,
            "copied_bytes": 0,
            "expected_copy_entries": 0,
            "expected_copied_elements": 0,
            "expected_copied_bytes": 0,
        }
    )

    with pytest.raises(AssertionError, match=r"state copy|state-copy|recurrent-state"):
        runner.shared_prefix_state_reuse_evidence(
            manifest,
            cached_tokens=(0, 16),
            worker_proof=(
                {
                    "rank": 0,
                    "device": 0,
                    "mamba_prefix_clones": _prefix_worker_stats([clone]),
                },
            ),
            expected_worker_clone_counts=(1,),
            cache_block_size=16,
        )


def test_shared_prefix_state_reuse_evidence_rejects_self_consistent_positive_layout_subset() -> None:
    manifest = _shared_prefix_manifest()
    clone = _prefix_clone_record("subset", copy_entries=2, copied_elements=8)
    retained_copy = clone["state_copies"][0]
    retained_elements = retained_copy["copied_elements"]
    retained_bytes = retained_copy["copied_bytes"]
    clone.update(
        {
            "state_copies": [retained_copy],
            "copy_entries": 1,
            "copied_elements": retained_elements,
            "copied_bytes": retained_bytes,
            "expected_copy_entries": 1,
            "expected_copied_elements": retained_elements,
            "expected_copied_bytes": retained_bytes,
        }
    )

    with pytest.raises(AssertionError, match=r"complete|runtime|layout"):
        runner.shared_prefix_state_reuse_evidence(
            manifest,
            cached_tokens=(0, 16),
            worker_proof=(
                {
                    "rank": 0,
                    "device": 0,
                    "mamba_prefix_clones": _prefix_worker_stats([clone]),
                },
            ),
            expected_worker_clone_counts=(1,),
            cache_block_size=16,
        )


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
    assert evidence["physical_prefill_prompt_tokens_by_request"] == [
        {
            "request_id": request.request_id,
            "replica_rank": 0,
            "prompt_tokens": 25_000,
            "cached_complete_block_tokens": 0 if index == 0 else 24_992,
            "physical_prefill_prompt_tokens": 25_000 if index == 0 else 8,
            "cache_status": "miss" if index == 0 else "hit",
        }
        for index, request in enumerate(manifest.requests)
    ]
    assert evidence["physical_prefill_prompt_tokens_by_replica"] == [
        {
            "replica_rank": 0,
            "request_count": 96,
            "cache_miss_request_count": 1,
            "cache_hit_request_count": 95,
            "cache_miss_physical_prefill_prompt_tokens": 25_000,
            "cache_hit_physical_prefill_prompt_tokens": 95 * 8,
            "physical_prefill_prompt_tokens_total": 25_000 + 95 * 8,
        }
    ]
    assert evidence["cache_miss_physical_prefill_prompt_tokens"] == 25_000
    assert evidence["cache_hit_physical_prefill_prompt_tokens"] == 95 * 8
    assert evidence["physical_prefill_prompt_tokens_total"] == 25_000 + 95 * 8
    assert evidence["physical_prefix_reuse_scope"] == {
        "resolved_cache_block_size_tokens": 16,
        "attention_kv_reused_complete_blocks_per_hit": 1_562,
        "attention_kv_reused_tokens_per_hit": 24_992,
        "fp32_hyena_state_clone_position_tokens": 24_992,
        "partial_block_tail_recomputed_tokens_per_hit": 8,
        "full_prompt_attention_kv_and_state_cloned": False,
    }
    assert evidence["attention_kv_physical_reuse_proven"] is True
    assert [worker["clone_count"] for worker in evidence["worker_state_clones"]] == [95, 95]
    assert all(
        request["copied_bytes"] == request["expected_copied_bytes"] == 524_288
        for worker in evidence["worker_state_clones"]
        for request in worker["requests"]
    )


def test_shared_prefix_state_reuse_evidence_reports_one_seed_prefill_per_dp_replica() -> None:
    manifest = (
        WorkloadManifest.from_path(DATA)
        .request_slice(0, 1)
        .with_uniform_prompt_length(32, request_count=4, request_id_prefix="dp-prefix")
    )
    worker_proof = tuple(
        {
            "rank": rank,
            "device": rank,
            "mamba_prefix_clones": _prefix_worker_stats(
                [_prefix_clone_record(f"clone-r{rank}", source_request_id=f"miss-r{rank}")],
                source_request_id=f"miss-r{rank}",
            ),
        }
        for rank in range(2)
    )

    evidence = runner.shared_prefix_state_reuse_evidence(
        manifest,
        cached_tokens=(0, 16, 0, 16),
        request_replica_ranks=(0, 0, 1, 1),
        worker_proof=worker_proof,
        expected_worker_clone_counts=(1, 1),
        cache_block_size=16,
        expected_cache_misses=2,
    )

    assert [
        (row["request_id"], row["replica_rank"], row["physical_prefill_prompt_tokens"])
        for row in evidence["physical_prefill_prompt_tokens_by_request"]
    ] == [
        (manifest.requests[0].request_id, 0, 32),
        (manifest.requests[1].request_id, 0, 16),
        (manifest.requests[2].request_id, 1, 32),
        (manifest.requests[3].request_id, 1, 16),
    ]
    assert evidence["physical_prefill_prompt_tokens_by_replica"] == [
        {
            "replica_rank": rank,
            "request_count": 2,
            "cache_miss_request_count": 1,
            "cache_hit_request_count": 1,
            "cache_miss_physical_prefill_prompt_tokens": 32,
            "cache_hit_physical_prefill_prompt_tokens": 16,
            "physical_prefill_prompt_tokens_total": 48,
        }
        for rank in range(2)
    ]
    assert evidence["cache_miss_request_count"] == 2
    assert evidence["cache_miss_physical_prefill_prompt_tokens"] == 64
    assert evidence["cache_hit_physical_prefill_prompt_tokens"] == 32
    assert evidence["physical_prefill_prompt_tokens_total"] == 96


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
