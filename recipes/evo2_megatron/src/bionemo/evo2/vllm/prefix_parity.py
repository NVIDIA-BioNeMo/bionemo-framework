# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Compare optimized independent-prefill and physical-prefix-reuse artifacts."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bionemo.evo2.vllm.benchmark import WorkloadManifest
from bionemo.evo2.vllm.runner import benchmark_contract_sha256, validate_linked_proof_artifact


EXACT_25K_PROMPT_SOURCE_SHA256 = "09778b8b2254cc977088c771880da9f14361bbe1ad74e2b35ecf8e8634dabdb6"


@dataclass(frozen=True)
class PrefixParityAcceptance:
    """Immutable shape and numerical policy for one prefix differential gate."""

    request_count: int
    prompt_tokens: int
    max_new_tokens: int
    comparison_tokens: int
    prompt_source_sha256: str
    logprob_rtol: float = 2e-2
    logprob_atol: float = 5e-2

    def __post_init__(self) -> None:
        """Reject any drift from the fixed production parity protocol."""
        for name in ("request_count", "prompt_tokens", "max_new_tokens", "comparison_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.comparison_tokens > self.max_new_tokens:
            raise ValueError("comparison_tokens cannot exceed max_new_tokens")
        if len(self.prompt_source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.prompt_source_sha256
        ):
            raise ValueError("prompt_source_sha256 must be a lowercase SHA256 digest")
        if not math.isfinite(self.logprob_rtol) or not math.isfinite(self.logprob_atol):
            raise ValueError("logprob tolerances must be finite")
        if self.logprob_rtol < 0 or self.logprob_atol < 0:
            raise ValueError("logprob tolerances must be nonnegative")


EXACT_25K_PREFIX_ACCEPTANCE = PrefixParityAcceptance(
    request_count=96,
    prompt_tokens=25_000,
    max_new_tokens=25_000,
    comparison_tokens=500,
    prompt_source_sha256=EXACT_25K_PROMPT_SOURCE_SHA256,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dict(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AssertionError(f"{label} must be a JSON array")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"{label} is not valid JSON") from error
    return _require_dict(value, label=label)


def _load_speed_artifact(
    path: str | Path,
    *,
    proof_validator: Callable[..., dict[str, Any]],
) -> tuple[Path, dict[str, Any], WorkloadManifest, dict[str, Any]]:
    artifact_path = Path(path).expanduser().resolve()
    artifact = _load_json(artifact_path, label="prefix parity speed artifact")
    if (
        artifact.get("benchmark_mode") != "speed"
        or artifact.get("backend") != "vllm"
        or artifact.get("topology") != "tp2"
        or artifact.get("invocation", {}).get("exit_status") != 0
    ):
        raise AssertionError("prefix parity requires successful TP2 vLLM speed artifacts")
    instrumentation = _require_dict(artifact.get("instrumentation"), label="speed instrumentation")
    required_instrumentation = {
        "scheduler_callbacks_during_generation": False,
        "worker_proof_rpcs": False,
        "prefix_clone_instrumentation": False,
        "peak_memory_polling_during_generation": False,
        "post_generation_exact_output_validation": True,
    }
    if instrumentation != required_instrumentation:
        raise AssertionError("prefix parity speed artifact used proof overhead inside generation timing")

    contract = _require_dict(artifact.get("benchmark_contract"), label="benchmark contract")
    if artifact.get("benchmark_contract_sha256") != benchmark_contract_sha256(contract):
        raise AssertionError("prefix parity benchmark contract digest is invalid")
    try:
        manifest = WorkloadManifest.from_dict(_require_dict(artifact.get("manifest"), label="manifest"))
    except (KeyError, TypeError, ValueError) as error:
        raise AssertionError("prefix parity manifest is malformed") from error
    if artifact.get("manifest_sha256") != manifest.sha256 or contract.get("manifest_sha256") != manifest.sha256:
        raise AssertionError("prefix parity manifest does not match its benchmark contract")
    profile = _require_dict(artifact.get("profile"), label="speed profile")
    if profile.get("proof") is not False:
        raise AssertionError("prefix parity timing artifact unexpectedly enabled proof instrumentation")
    profile_contract = dict(profile)
    profile_contract.pop("proof")
    if profile_contract != contract.get("profile"):
        raise AssertionError("prefix parity speed profile does not match its benchmark contract")
    if artifact.get("exact_generation_progress", {}).get("passed") is not True:
        raise AssertionError("prefix parity speed artifact lacks exact-generation progress evidence")

    retained_proof = _require_dict(artifact.get("linked_proof_artifact"), label="linked proof evidence")
    proof_path = Path(str(retained_proof.get("artifact_path", ""))).expanduser().resolve()
    recomputed_proof = proof_validator(
        proof_path,
        expected_contract=contract,
        require_memory_headroom=True,
    )
    if retained_proof != recomputed_proof:
        raise AssertionError("prefix parity linked proof evidence failed recomputation")
    return artifact_path, artifact, manifest, retained_proof


def _validate_acceptance_manifest(
    manifest: WorkloadManifest,
    *,
    acceptance: PrefixParityAcceptance,
) -> None:
    if len(manifest.requests) != acceptance.request_count:
        raise AssertionError("prefix parity request count does not match its frozen acceptance shape")
    if manifest.max_new_tokens != acceptance.max_new_tokens:
        raise AssertionError("prefix parity new-token count does not match its frozen acceptance shape")
    if manifest.prompt_source_sha256 != acceptance.prompt_source_sha256:
        raise AssertionError("prefix parity prompt source hash does not match the frozen manifest")
    prompt_ids = [request.prompt_token_ids for request in manifest.requests]
    if any(len(prompt) != acceptance.prompt_tokens for prompt in prompt_ids):
        raise AssertionError("prefix parity prompt token count is not exact")
    if any(prompt != prompt_ids[0] for prompt in prompt_ids[1:]):
        raise AssertionError("prefix parity requires one identical physical prefix across all requests")
    if manifest.ignore_eos is not True or manifest.stop_token_ids:
        raise AssertionError("prefix parity must run to the exact requested length")


def _without_prefix_mode(contract: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(contract)
    profile = _require_dict(normalized.get("profile"), label="benchmark profile contract")
    profile.pop("shared_prefix_state_reuse", None)
    return normalized


def _validate_cross_artifact_contract(
    independent: dict[str, Any],
    cached: dict[str, Any],
    *,
    manifest: WorkloadManifest,
    acceptance: PrefixParityAcceptance,
) -> str:
    independent_contract = _require_dict(independent.get("benchmark_contract"), label="independent contract")
    cached_contract = _require_dict(cached.get("benchmark_contract"), label="cached contract")
    independent_profile = _require_dict(independent_contract.get("profile"), label="independent profile")
    cached_profile = _require_dict(cached_contract.get("profile"), label="cached profile")
    if independent_profile.get("shared_prefix_state_reuse") is not False:
        raise AssertionError("independent prefix lane enabled prefix reuse")
    if cached_profile.get("shared_prefix_state_reuse") is not True:
        raise AssertionError("cached prefix lane did not enable physical prefix reuse")
    if _without_prefix_mode(independent_contract) != _without_prefix_mode(cached_contract):
        raise AssertionError("prefix lanes differ in a knob other than physical prefix reuse")
    if independent.get("checkpoint_provenance") != cached.get("checkpoint_provenance"):
        raise AssertionError("prefix lanes do not use the exact same checkpoint/export")
    if (
        independent_profile.get("topology") != "tp2"
        or independent_profile.get("global_wave_size") != acceptance.request_count
    ):
        raise AssertionError("prefix parity did not use one exact TP2 global request wave")
    max_num_seqs = independent_profile.get("max_num_seqs")
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs < acceptance.request_count:
        raise AssertionError("prefix parity scheduler ceiling does not cover the exact physical wave")
    if independent_contract.get("manifest_sha256") != manifest.sha256:
        raise AssertionError("prefix parity normalized contract drifted from the exact manifest")
    return benchmark_contract_sha256(_without_prefix_mode(independent_contract))


def _load_sidecar_rows(
    artifact_path: Path,
    phase: dict[str, Any],
    *,
    manifest: WorkloadManifest,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = _require_dict(phase.get("full_output_artifact"), label="prefix parity output sidecar")
    if (
        metadata.get("schema_version") != 2
        or metadata.get("format") != "jsonl"
        or metadata.get("compression") != "gzip"
    ):
        raise AssertionError("prefix parity sidecar schema is unsupported")
    sidecar = Path(str(metadata.get("path", ""))).expanduser().resolve()
    if not sidecar.is_file() or metadata.get("sha256") != _sha256_file(sidecar):
        raise AssertionError(f"prefix parity sidecar hash is invalid for {artifact_path}")
    if metadata.get("size_bytes") != sidecar.stat().st_size:
        raise AssertionError("prefix parity sidecar byte count is invalid")
    expected_tokens = len(manifest.requests) * manifest.max_new_tokens
    expected_metadata = {
        "request_count": len(manifest.requests),
        "generated_token_count": expected_tokens,
        "output_token_id_count": expected_tokens,
        "chosen_token_logprob_count": expected_tokens,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise AssertionError("prefix parity sidecar retained counts are not exact")
    rows = []
    try:
        with gzip.open(sidecar, mode="rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append(_require_dict(row, label="prefix parity sidecar row"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssertionError("prefix parity sidecar could not be decoded") from error
    if len(rows) != len(manifest.requests):
        raise AssertionError("prefix parity sidecar does not cover every request")
    for request, row in zip(manifest.requests, rows, strict=True):
        output_ids = row.get("output_token_ids")
        logprobs = row.get("chosen_token_logprobs")
        if row.get("request_id") != request.request_id or row.get("prompt_token_ids") != list(
            request.prompt_token_ids
        ):
            raise AssertionError("prefix parity sidecar request identity or prompt drifted")
        if not isinstance(output_ids, list) or len(output_ids) != manifest.max_new_tokens:
            raise AssertionError("prefix parity sidecar output IDs are not exact length")
        if (
            not isinstance(logprobs, list)
            or len(logprobs) != manifest.max_new_tokens
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in logprobs
            )
        ):
            raise AssertionError("prefix parity chosen-token logprobs are malformed")
        if (
            row.get("finish_reason") != "length"
            or row.get("stop_reason") is not None
            or row.get("stopped_on_eos") is not False
        ):
            raise AssertionError("prefix parity output did not finish at the exact length boundary")
    return metadata, rows


def _compare_steady_outputs(
    independent_path: Path,
    independent: dict[str, Any],
    cached_path: Path,
    cached: dict[str, Any],
    *,
    manifest: WorkloadManifest,
    acceptance: PrefixParityAcceptance,
) -> tuple[list[dict[str, Any]], int, float]:
    independent_phases = {
        phase.get("phase"): phase
        for phase in _require_list(independent.get("phases"), label="independent phases")
        if isinstance(phase, dict) and str(phase.get("phase", "")).startswith("steady-")
    }
    cached_phases = {
        phase.get("phase"): phase
        for phase in _require_list(cached.get("phases"), label="cached phases")
        if isinstance(phase, dict) and str(phase.get("phase", "")).startswith("steady-")
    }
    if not independent_phases or independent_phases.keys() != cached_phases.keys():
        raise AssertionError("prefix parity requires matching nonempty steady timing phases")
    comparisons = []
    compared_tokens = 0
    maximum_absolute_logprob_error = 0.0
    ownership_fields = (
        "request_id",
        "execution_uid",
        "generation_round",
        "call_index",
        "global_request_index",
        "dp_rank",
        "seed",
        "prompt_token_ids",
    )
    for phase_name in sorted(independent_phases):
        independent_phase = independent_phases[phase_name]
        cached_phase = cached_phases[phase_name]
        if independent_phase.get("proof_collected") is not False or cached_phase.get("proof_collected") is not False:
            raise AssertionError("prefix parity steady timing phases unexpectedly collected proof callbacks")
        independent_metadata, independent_rows = _load_sidecar_rows(
            independent_path,
            independent_phase,
            manifest=manifest,
        )
        cached_metadata, cached_rows = _load_sidecar_rows(cached_path, cached_phase, manifest=manifest)
        seeds = []
        phase_maximum_error = 0.0
        for reference, candidate in zip(independent_rows, cached_rows, strict=True):
            if any(reference.get(field) != candidate.get(field) for field in ownership_fields):
                raise AssertionError("prefix parity request ownership or seed coordinates differ")
            seed = reference.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise AssertionError("prefix parity request seed is malformed")
            seeds.append(seed)
            reference_ids = reference["output_token_ids"][: acceptance.comparison_tokens]
            candidate_ids = candidate["output_token_ids"][: acceptance.comparison_tokens]
            if reference_ids != candidate_ids:
                raise AssertionError("cached prefix output token IDs differ from independent prefill")
            for expected, actual in zip(
                reference["chosen_token_logprobs"][: acceptance.comparison_tokens],
                candidate["chosen_token_logprobs"][: acceptance.comparison_tokens],
                strict=True,
            ):
                error = abs(float(actual) - float(expected))
                allowed = acceptance.logprob_atol + acceptance.logprob_rtol * abs(float(expected))
                if error > allowed:
                    raise AssertionError("cached prefix chosen-token logprob exceeded the established tolerance")
                phase_maximum_error = max(phase_maximum_error, error)
        if len(seeds) != len(set(seeds)):
            raise AssertionError("prefix parity requires one distinct request seed per identical prompt")
        phase_tokens = len(manifest.requests) * acceptance.comparison_tokens
        compared_tokens += phase_tokens
        maximum_absolute_logprob_error = max(maximum_absolute_logprob_error, phase_maximum_error)
        comparisons.append(
            {
                "phase": phase_name,
                "request_count": len(manifest.requests),
                "comparison_tokens_per_request": acceptance.comparison_tokens,
                "compared_token_count": phase_tokens,
                "maximum_absolute_chosen_logprob_error": phase_maximum_error,
                "independent_sidecar_sha256": independent_metadata["sha256"],
                "cached_sidecar_sha256": cached_metadata["sha256"],
                "passed": True,
            }
        )
    return comparisons, compared_tokens, maximum_absolute_logprob_error


def _validate_physical_prefix_reuse(
    independent_proof: dict[str, Any],
    cached_proof: dict[str, Any],
    *,
    acceptance: PrefixParityAcceptance,
) -> dict[str, Any]:
    if independent_proof.get("profile", {}).get("shared_prefix_state_reuse") is not False:
        raise AssertionError("independent linked proof enabled prefix reuse")
    if cached_proof.get("profile", {}).get("shared_prefix_state_reuse") is not True:
        raise AssertionError("cached linked proof did not enable prefix reuse")
    independent_phases = _require_list(independent_proof.get("phases"), label="independent proof phases")
    cached_phases = _require_list(cached_proof.get("phases"), label="cached proof phases")
    if not independent_phases or len(independent_phases) != len(cached_phases):
        raise AssertionError("prefix linked proof phase counts differ")
    if any(
        not isinstance(phase, dict) or phase.get("shared_prefix_state_reuse") is not None
        for phase in independent_phases
    ):
        raise AssertionError("independent proof retained unexpected physical prefix reuse")

    per_phase = []
    expected_hits = acceptance.request_count - 1
    for independent_phase, cached_phase in zip(independent_phases, cached_phases, strict=True):
        if independent_phase.get("phase") != cached_phase.get("phase"):
            raise AssertionError("prefix linked proof phase names differ")
        evidence = _require_dict(
            cached_phase.get("shared_prefix_state_reuse"),
            label="cached physical prefix reuse evidence",
        )
        reused_tokens = evidence.get("physically_reused_prompt_tokens_per_clone")
        expected_elements = evidence.get("expected_fp32_state_copy_elements_per_request")
        expected_bytes = evidence.get("expected_fp32_state_copy_bytes_per_request")
        if (
            evidence.get("prompt_tokens_per_request") != acceptance.prompt_tokens
            or evidence.get("cache_miss_request_count") != 1
            or evidence.get("cache_hit_request_count") != expected_hits
            or evidence.get("logical_clone_request_count") != expected_hits
            or evidence.get("attention_kv_physical_reuse_proven") is not True
            or evidence.get("physical_state_copy_proven") is not True
            or evidence.get("phase_prefix_cache_reset") is not True
        ):
            raise AssertionError("cached proof did not retain exactly one miss and every physical prefix clone")
        if (
            isinstance(reused_tokens, bool)
            or not isinstance(reused_tokens, int)
            or not 0 < reused_tokens <= acceptance.prompt_tokens
        ):
            raise AssertionError("cached proof retained an invalid physical prefix length")
        recomputed_tokens = acceptance.prompt_tokens - reused_tokens
        if (
            evidence.get("recomputed_prompt_tokens_per_clone") != recomputed_tokens
            or evidence.get("total_cached_prompt_tokens") != expected_hits * reused_tokens
            or evidence.get("scheduled_uncached_prompt_tokens")
            != acceptance.prompt_tokens + expected_hits * recomputed_tokens
        ):
            raise AssertionError("cached proof prefix work does not equal one full prefill plus block remainders")
        if (
            isinstance(expected_elements, bool)
            or not isinstance(expected_elements, int)
            or expected_elements <= 0
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes != 4 * expected_elements
        ):
            raise AssertionError("cached proof recurrent-state copy size is not exact FP32 storage")
        workers = _require_list(evidence.get("worker_state_clones"), label="prefix worker clones")
        if len(workers) != 2:
            raise AssertionError("TP2 prefix proof must cover both tensor-parallel workers")
        for worker in workers:
            if not isinstance(worker, dict) or worker.get("clone_count") != expected_hits:
                raise AssertionError("each TP2 worker must clone every cache-hit request")
            requests = _require_list(worker.get("requests"), label="worker prefix clone requests")
            if len(requests) != expected_hits or any(
                not isinstance(request, dict)
                or request.get("expected_copied_elements") != expected_elements
                or request.get("expected_copied_bytes") != expected_bytes
                for request in requests
            ):
                raise AssertionError("worker prefix clones do not match the complete FP32 state layout")
        if evidence.get("rank_local_physical_clone_count") != 2 * expected_hits:
            raise AssertionError("rank-local TP2 clone count does not match every logical prefix hit")
        per_phase.append(
            {
                "phase": cached_phase["phase"],
                "cache_miss_request_count": 1,
                "cache_hit_request_count": expected_hits,
                "full_source_prefill_tokens": acceptance.prompt_tokens,
                "physically_reused_prompt_tokens_per_clone": reused_tokens,
                "recomputed_prompt_tokens_per_clone": recomputed_tokens,
                "expected_fp32_state_copy_elements_per_request": expected_elements,
                "expected_fp32_state_copy_bytes_per_request": expected_bytes,
                "attention_kv_physical_reuse_proven": True,
                "physical_state_copy_proven": True,
                "passed": True,
            }
        )
    return {
        "phase_count": len(per_phase),
        "cache_miss_request_count": 1,
        "cache_hit_request_count": expected_hits,
        "logical_clone_request_count": expected_hits,
        "phases": per_phase,
        "passed": True,
    }


def _compare_prefix_artifacts(
    independent_artifact: str | Path,
    cached_artifact: str | Path,
    *,
    acceptance: PrefixParityAcceptance = EXACT_25K_PREFIX_ACCEPTANCE,
    proof_validator: Callable[..., dict[str, Any]] = validate_linked_proof_artifact,
) -> dict[str, Any]:
    """Internal comparator with injectable fixture contracts for focused CPU tests."""
    independent_path, independent, independent_manifest, independent_link = _load_speed_artifact(
        independent_artifact,
        proof_validator=proof_validator,
    )
    cached_path, cached, cached_manifest, cached_link = _load_speed_artifact(
        cached_artifact,
        proof_validator=proof_validator,
    )
    if (
        independent_manifest.to_dict() != cached_manifest.to_dict()
        or independent_manifest.sha256 != cached_manifest.sha256
    ):
        raise AssertionError("prefix parity lanes do not use the same exact manifest")
    _validate_acceptance_manifest(independent_manifest, acceptance=acceptance)
    normalized_contract_sha256 = _validate_cross_artifact_contract(
        independent,
        cached,
        manifest=independent_manifest,
        acceptance=acceptance,
    )
    independent_proof_path = Path(independent_link["artifact_path"])
    cached_proof_path = Path(cached_link["artifact_path"])
    if independent_link.get("artifact_sha256") != _sha256_file(independent_proof_path):
        raise AssertionError("independent physical proof hash drifted")
    if cached_link.get("artifact_sha256") != _sha256_file(cached_proof_path):
        raise AssertionError("cached physical proof hash drifted")
    physical_reuse = _validate_physical_prefix_reuse(
        _load_json(independent_proof_path, label="independent prefix proof"),
        _load_json(cached_proof_path, label="cached prefix proof"),
        acceptance=acceptance,
    )
    phase_comparisons, compared_tokens, maximum_error = _compare_steady_outputs(
        independent_path,
        independent,
        cached_path,
        cached,
        manifest=independent_manifest,
        acceptance=acceptance,
    )
    return {
        "schema_version": 1,
        "task": "evo2-vllm-exact-prefix-reuse-parity",
        "acceptance": asdict(acceptance),
        "normalized_benchmark_contract_sha256": normalized_contract_sha256,
        "manifest_sha256": independent_manifest.sha256,
        "checkpoint_provenance": independent["checkpoint_provenance"],
        "independent_artifact": {
            "path": str(independent_path),
            "sha256": _sha256_file(independent_path),
            "linked_proof_path": str(independent_proof_path),
            "linked_proof_sha256": independent_link["artifact_sha256"],
        },
        "cached_artifact": {
            "path": str(cached_path),
            "sha256": _sha256_file(cached_path),
            "linked_proof_path": str(cached_proof_path),
            "linked_proof_sha256": cached_link["artifact_sha256"],
        },
        "compared_phase_count": len(phase_comparisons),
        "compared_request_count": len(independent_manifest.requests),
        "compared_token_count": compared_tokens,
        "maximum_absolute_chosen_logprob_error": maximum_error,
        "phase_comparisons": phase_comparisons,
        "physical_prefix_reuse": physical_reuse,
        "full_raw_outputs_retained_in_source_sidecars": True,
        "passed": True,
    }


def compare_prefix_artifacts(
    independent_artifact: str | Path,
    cached_artifact: str | Path,
) -> dict[str, Any]:
    """Compare exact-25k artifacts under the immutable production acceptance contract."""
    return _compare_prefix_artifacts(
        independent_artifact,
        cached_artifact,
        acceptance=EXACT_25K_PREFIX_ACCEPTANCE,
        proof_validator=validate_linked_proof_artifact,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed exact-25k prefix differential CLI."""
    parser = argparse.ArgumentParser(description="Compare exact 25k independent and prefix-reuse Evo2 artifacts")
    parser.add_argument("--independent-artifact", type=Path, required=True)
    parser.add_argument("--cached-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write one immutable CPU-only exact-25k prefix parity artifact."""
    from bionemo.evo2.vllm.runner import (
        complete_output_namespace,
        require_output_namespace_reservation,
        reserve_output_namespace,
        write_json_artifact,
    )

    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    reservation = reserve_output_namespace(output)
    artifact = compare_prefix_artifacts(args.independent_artifact, args.cached_artifact)
    require_output_namespace_reservation(output)
    write_json_artifact(
        output,
        artifact,
        ownership_validator=lambda: require_output_namespace_reservation(output),
    )
    complete_output_namespace(reservation, output_path=output)
    return 0


__all__ = [
    "EXACT_25K_PREFIX_ACCEPTANCE",
    "PrefixParityAcceptance",
    "build_parser",
    "compare_prefix_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
