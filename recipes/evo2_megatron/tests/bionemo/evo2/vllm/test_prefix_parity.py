# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from bionemo.evo2.vllm.benchmark import WorkloadManifest, WorkloadRequest
from bionemo.evo2.vllm.prefix_parity import PrefixParityAcceptance, compare_prefix_artifacts


_SOURCE_SHA = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> WorkloadManifest:
    return WorkloadManifest(
        schema_version=1,
        name="prefix-parity-test",
        source_checkpoint="evo2/7b-1m:1.0",
        checkpoint_manifest_sha256="b" * 64,
        checkpoint_index_sha256="c" * 64,
        tokenizer_sha256="d" * 64,
        requests=tuple(WorkloadRequest(f"request-{index}", (1, 2, 3, 4)) for index in range(2)),
        max_new_tokens=3,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        seed=42,
        dtype="bfloat16",
        ignore_eos=True,
        stop_token_ids=(),
        prompt_source_path="/frozen/prefix.jsonl",
        prompt_source_sha256=_SOURCE_SHA,
        prompt_tokenizer_path="/frozen/tokenizer.json",
        prompt_tokenizer_sha256="e" * 64,
    )


def _write_sidecar(path: Path, *, token_delta: int = 0, logprob_delta: float = 0.0) -> dict:
    rows = []
    for index in range(2):
        output_ids = [10 + index, 11 + index, 12 + index]
        if index == 1:
            output_ids[1] += token_delta
        logprobs = [-0.1, -0.2, -0.3]
        if index == 1:
            logprobs[1] += logprob_delta
        rows.append(
            {
                "request_id": f"request-{index}",
                "execution_uid": f"round=0/call=0/global={index}/dp=0/request=request-{index}",
                "generation_round": 0,
                "call_index": 0,
                "global_request_index": index,
                "dp_rank": 0,
                "seed": 42 + index,
                "prompt_token_ids": [1, 2, 3, 4],
                "output_token_ids": output_ids,
                "chosen_token_logprobs": logprobs,
                "finish_reason": "length",
                "stop_reason": None,
                "stopped_on_eos": False,
            }
        )
    with gzip.open(path, mode="wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "schema_version": 2,
        "format": "jsonl",
        "compression": "gzip",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "request_count": 2,
        "generated_token_count": 6,
        "output_token_id_count": 6,
        "chosen_token_logprob_count": 6,
    }


def _prefix_evidence() -> dict:
    request = {
        "expected_copy_entries": 2,
        "expected_copied_elements": 8,
        "expected_copied_bytes": 32,
    }
    return {
        "prompt_tokens_per_request": 4,
        "cache_block_size": 2,
        "cached_tokens_by_request": [0, 2],
        "cache_hit_request_count": 1,
        "cache_miss_request_count": 1,
        "logical_clone_request_count": 1,
        "physically_reused_prompt_tokens_per_clone": 2,
        "recomputed_prompt_tokens_per_clone": 2,
        "total_cached_prompt_tokens": 2,
        "scheduled_uncached_prompt_tokens": 6,
        "worker_state_clones": [{"rank": rank, "clone_count": 1, "requests": [dict(request)]} for rank in range(2)],
        "rank_local_physical_clone_count": 2,
        "expected_fp32_state_copy_elements_per_request": 8,
        "expected_fp32_state_copy_bytes_per_request": 32,
        "attention_kv_physical_reuse_proven": True,
        "physical_state_copy_proven": True,
        "phase_prefix_cache_reset": True,
    }


def _write_pair(tmp_path: Path) -> tuple[Path, Path, dict[str, dict]]:
    manifest = _manifest()
    artifacts = {}
    retained_proofs = {}
    paths = []
    for label, cached in (("independent", False), ("cached", True)):
        artifact_path = tmp_path / f"{label}.json"
        proof_path = tmp_path / f"{label}.proof.json"
        proof = {
            "profile": {"proof": True, "shared_prefix_state_reuse": cached},
            "phases": [
                {
                    "phase": "steady-0",
                    "shared_prefix_state_reuse": _prefix_evidence() if cached else None,
                }
            ],
        }
        proof_path.write_text(json.dumps(proof), encoding="utf-8")
        retained_proof = {
            "artifact_path": str(proof_path.resolve()),
            "artifact_sha256": _sha256(proof_path),
            "benchmark_contract_sha256": "proof-contract",
            "proof_status": {"passed": True},
            "gpu_memory_headroom": {"passed": True},
            "validated_evidence": {"final_worker_count": 2},
        }
        retained_proofs[str(proof_path.resolve())] = retained_proof
        profile = {
            "topology": "tp2",
            "max_model_len": 50_000,
            "max_num_batched_tokens": 16_384,
            "gpu_memory_utilization": 0.95,
            "optimization_level": 2,
            "performance_mode": "balanced",
            "shared_prefix_state_reuse": cached,
            "global_wave_size": 2,
            "max_num_seqs": 2,
        }
        contract = {
            "schema_version": 2,
            "backend": "vllm",
            "topology": "tp2",
            "checkpoint": "/checkpoint",
            "load_format": "safetensors",
            "manifest_sha256": manifest.sha256,
            "profile": profile,
            "seed_stream": {"base_seed": 42, "generation_round": 0},
            "measurement": {"warmups": 0, "repetitions": 1},
            "canonical_identity": None,
            "exact_generation_progress": {"max_new_tokens": 3},
            "runtime_attestation": {"checkpoint_sha256": "checkpoint"},
        }
        sidecar = _write_sidecar(tmp_path / f"{label}.steady-0.outputs.jsonl.gz")
        artifact = {
            "schema_version": 1,
            "benchmark_mode": "speed",
            "backend": "vllm",
            "topology": "tp2",
            "benchmark_contract": contract,
            "benchmark_contract_sha256": hashlib.sha256(
                json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "instrumentation": {
                "scheduler_callbacks_during_generation": False,
                "worker_proof_rpcs": False,
                "prefix_clone_instrumentation": False,
                "peak_memory_polling_during_generation": False,
                "post_generation_exact_output_validation": True,
            },
            "linked_proof_artifact": retained_proof,
            "invocation": {"exit_status": 0},
            "manifest": manifest.to_dict(),
            "manifest_sha256": manifest.sha256,
            "profile": {**profile, "proof": False},
            "checkpoint_provenance": {"checkpoint_sha256": "checkpoint"},
            "phases": [
                {
                    "phase": "steady-0",
                    "proof_collected": False,
                    "full_output_artifact": sidecar,
                    "exact_generation_progress": {"passed": True},
                }
            ],
            "exact_generation_progress": {"passed": True},
        }
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        artifacts[label] = artifact
        paths.append(artifact_path)
    return paths[0], paths[1], retained_proofs


def _acceptance() -> PrefixParityAcceptance:
    return PrefixParityAcceptance(
        request_count=2,
        prompt_tokens=4,
        max_new_tokens=3,
        comparison_tokens=2,
        prompt_source_sha256=_SOURCE_SHA,
    )


def test_prefix_parity_compares_every_request_and_requires_physical_reuse(tmp_path) -> None:
    independent, cached, retained = _write_pair(tmp_path)

    evidence = compare_prefix_artifacts(
        independent,
        cached,
        acceptance=_acceptance(),
        proof_validator=lambda path, **_: retained[str(Path(path).resolve())],
    )

    assert evidence["compared_request_count"] == 2
    assert evidence["compared_token_count"] == 4
    assert evidence["physical_prefix_reuse"]["cache_miss_request_count"] == 1
    assert evidence["physical_prefix_reuse"]["cache_hit_request_count"] == 1
    assert evidence["passed"] is True


@pytest.mark.parametrize("tamper", ("token", "logprob", "knob", "prefix"))
def test_prefix_parity_fails_closed_on_output_contract_or_reuse_drift(tmp_path, tamper) -> None:
    independent, cached, retained = _write_pair(tmp_path)
    candidate = json.loads(cached.read_text(encoding="utf-8"))
    if tamper in {"token", "logprob"}:
        metadata = _write_sidecar(
            Path(candidate["phases"][0]["full_output_artifact"]["path"]),
            token_delta=int(tamper == "token"),
            logprob_delta=0.2 if tamper == "logprob" else 0.0,
        )
        candidate["phases"][0]["full_output_artifact"] = metadata
    elif tamper == "knob":
        candidate["benchmark_contract"]["profile"]["optimization_level"] = 3
        candidate["benchmark_contract_sha256"] = hashlib.sha256(
            json.dumps(candidate["benchmark_contract"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    else:
        proof_path = Path(candidate["linked_proof_artifact"]["artifact_path"])
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        proof["phases"][0]["shared_prefix_state_reuse"]["cache_hit_request_count"] = 0
        proof_path.write_text(json.dumps(proof), encoding="utf-8")
        candidate["linked_proof_artifact"]["artifact_sha256"] = _sha256(proof_path)
        retained[str(proof_path.resolve())] = candidate["linked_proof_artifact"]
    cached.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(AssertionError):
        compare_prefix_artifacts(
            independent,
            cached,
            acceptance=_acceptance(),
            proof_validator=lambda path, **_: retained[str(Path(path).resolve())],
        )
