# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""End-to-end optimized vLLM benchmark and proof runner for Evo2."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import math
import os
import platform
import stat
import struct
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from nemo_rl.models.generation.interfaces import generation_prompt_token_ids_sha256

from bionemo.evo2.vllm.accuracy import (
    build_canonical_identity_contract,
    build_common_prefix_identity_contract,
    build_homogeneous_identity_schedule,
    build_mixed_canonical_identity_contract,
    build_mixed_canonical_identity_manifest,
    build_mixed_identity_schedule,
    load_canonical_7b_identity_cases,
    load_common_prefix_identity_cases,
    validate_canonical_identity_manifest,
    validate_canonical_identity_output_artifact,
    validate_common_prefix_identity_manifest,
    validate_common_prefix_identity_output_artifacts,
    validate_homogeneous_identity_phase_evidence,
    validate_mixed_canonical_identity_manifest,
    validate_mixed_canonical_identity_output_artifact,
    validate_mixed_identity_phase_evidence,
)
from bionemo.evo2.vllm.artifact_io import (
    ArtifactSnapshotError,
    PublicationReceipt,
    parse_json_bytes,
    publish_bytes_noreplace,
    publish_file_noreplace,
    read_byte_snapshot,
    read_file_digest_snapshot,
    read_json_snapshot,
    read_jsonl_snapshot,
    validate_publication_receipt,
)
from bionemo.evo2.vllm.benchmark import (
    BenchmarkSample,
    GenerationRecord,
    WorkloadManifest,
    aggregate_samples,
    benchmark_sample_from_vllm_outputs,
    build_parser,
    build_request_waves,
    exact_length_evidence,
    records_from_vllm_outputs,
    sampling_params_kwargs,
    summarize_vllm_outputs,
    validate_compilation_proof,
    validate_dna_output_token_ids,
)
from bionemo.evo2.vllm.profile import (
    Evo2VllmProfile,
    context_length_preflight,
    resolved_config_snapshot,
    validate_resolved_profile,
)
from bionemo.evo2.vllm.tokenizer_io import SnapshotBoundTokenizer


_SEED_ROUND_STRIDE = 1_000_003
_SEED_MODULUS = 2**31
_REQUIRED_GPU_HEADROOM_BYTES = 2 * 1024**3
COORDINATOR_GENERATION_TIMING_AUTHORITY = "coordinator_monotonic_generation_wall"


def _request_seed_preimage(
    base_seed: int,
    *,
    call_index: int,
    dp_rank: int,
    dp_size: int,
    request_index_in_stream: int,
) -> int:
    """Return the exact request-seed coordinate before any forbidden modulus."""
    for label, value in (
        ("base_seed", base_seed),
        ("call_index", call_index),
        ("dp_rank", dp_rank),
        ("dp_size", dp_size),
        ("request_index_in_stream", request_index_in_stream),
    ):
        if type(value) is not int:
            raise TypeError(f"{label} must be a built-in integer")
    if min(base_seed, call_index, dp_rank, request_index_in_stream) < 0 or dp_size <= 0:
        raise ValueError("seed coordinates must be nonnegative")
    if dp_rank >= dp_size:
        raise ValueError("dp_rank must be smaller than dp_size")
    if request_index_in_stream >= _SEED_ROUND_STRIDE:
        raise ValueError("request index exceeds the collision-free stream stride")
    return base_seed + (call_index * dp_size + dp_rank) * _SEED_ROUND_STRIDE + request_index_in_stream


_FROZEN_GPU_ASSIGNMENTS: tuple[dict[str, Any], ...] = (
    {
        "logical_device_index": 0,
        "visible_device_selector": "0",
        "physical_index": 0,
        "uuid": "GPU-f080a92d-e5d2-6bad-68b2-1d458c0f8337",
        "pci_bus_id": "00000000:0a:00.0",
    },
    {
        "logical_device_index": 1,
        "visible_device_selector": "1",
        "physical_index": 1,
        "uuid": "GPU-a3040cd2-4d5e-bfe0-3c75-4f8149d5d8b8",
        "pci_bus_id": "00000000:18:00.0",
    },
)
@dataclass
class _OutputNamespaceOwnership:
    output_path: Path
    marker_identity: tuple[int, int]
    parent_identity: tuple[int, int]
    publications: dict[Path, PublicationReceipt] = field(default_factory=dict)


_OUTPUT_NAMESPACE_OWNERSHIP: dict[Path, _OutputNamespaceOwnership] = {}
_OUTPUT_NAMESPACE_OWNERSHIP_LOCK = threading.Lock()


@dataclass(frozen=True)
class CallerCoordinateContract:
    """Immutable caller-owned coordinates used to validate worker evidence."""

    manifest: WorkloadManifest
    profile: Evo2VllmProfile
    generation_round: int

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, WorkloadManifest):
            raise TypeError("caller coordinate manifest must be a WorkloadManifest")
        if not isinstance(self.profile, Evo2VllmProfile):
            raise TypeError("caller coordinate profile must be an Evo2VllmProfile")
        if (
            isinstance(self.generation_round, bool)
            or not isinstance(self.generation_round, int)
            or self.generation_round < 0
        ):
            raise ValueError("caller coordinate generation round must be a nonnegative integer")
        self.seed_capacity_contract()

    @classmethod
    def from_inputs(
        cls,
        manifest: WorkloadManifest,
        profile: Evo2VllmProfile,
        generation_round: int,
    ) -> CallerCoordinateContract:
        """Freeze coordinates directly from caller-owned runtime inputs."""
        return cls(
            manifest=manifest,
            profile=profile,
            generation_round=generation_round,
        )

    @property
    def topology(self) -> str:
        return self.profile.topology

    @property
    def global_wave_size(self) -> int:
        return self.profile.global_wave_size

    @property
    def replica_count(self) -> int:
        return self.profile.replica_count

    def profile_contract(self) -> dict[str, Any]:
        """Return the complete optimized profile except proof instrumentation."""
        contract = asdict(self.profile)
        contract.pop("proof")
        return contract

    @property
    def physical_calls_per_round(self) -> int:
        """Return the exact physical wave count derived from immutable inputs."""
        return len(
            build_request_waves(
                request_count=len(self.manifest.requests),
                global_batch_size=self.global_wave_size,
                replica_count=self.replica_count,
            )
        )

    @property
    def global_call_index_start(self) -> int:
        """Return the first physical call for the semantic generation round."""
        return self.generation_round * self.physical_calls_per_round

    def seed_capacity_contract(self) -> dict[str, Any]:
        """Prove every exact physical request seed remains below the modulus."""
        waves = build_request_waves(
            request_count=len(self.manifest.requests),
            global_batch_size=self.global_wave_size,
            replica_count=self.replica_count,
        )
        global_call_index_start = self.generation_round * len(waves)
        maximum_pre_modulo_seed = -1
        maximum_coordinates: dict[str, int] | None = None
        physical_calls: list[dict[str, Any]] = []
        for wave in waves:
            global_call_index = global_call_index_start + wave.wave_index
            replicas = []
            for shard in wave.shards:
                maximum_request_index = shard.request_count - 1
                shard_maximum = _request_seed_preimage(
                    self.manifest.seed,
                    call_index=global_call_index,
                    dp_rank=shard.replica_index,
                    dp_size=self.replica_count,
                    request_index_in_stream=maximum_request_index,
                )
                replicas.append(
                    {
                        "dp_rank": shard.replica_index,
                        "local_request_count": shard.request_count,
                        "maximum_request_index_in_stream": maximum_request_index,
                        "maximum_pre_modulo_seed": shard_maximum,
                    }
                )
                if shard_maximum > maximum_pre_modulo_seed:
                    maximum_pre_modulo_seed = shard_maximum
                    maximum_coordinates = {
                        "call_in_round": wave.wave_index,
                        "global_call_index": global_call_index,
                        "dp_rank": shard.replica_index,
                        "request_index_in_stream": maximum_request_index,
                    }
            physical_calls.append(
                {
                    "call_in_round": wave.wave_index,
                    "global_call_index": global_call_index,
                    "global_request_count": wave.request_count,
                    "replicas": replicas,
                }
            )
        if maximum_coordinates is None:
            raise RuntimeError("seed-capacity proof requires at least one physical request")
        if maximum_pre_modulo_seed >= _SEED_MODULUS:
            raise ValueError(
                "request seed wraparound is forbidden: maximum pre-modulo seed "
                f"{maximum_pre_modulo_seed} reaches modulus {_SEED_MODULUS} at "
                f"{maximum_coordinates}"
            )
        return {
            "schema_version": 1,
            "round_stride": _SEED_ROUND_STRIDE,
            "modulus": _SEED_MODULUS,
            "maximum_pre_modulo_seed": maximum_pre_modulo_seed,
            "maximum_coordinates": maximum_coordinates,
            "physical_calls": physical_calls,
            "passed": True,
        }

    def seed_stream_contract(self) -> dict[str, Any]:
        """Return caller-derived stream coordinates without worker-owned input."""
        return {
            "schema_version": 3,
            "base_seed": self.manifest.seed,
            "generation_round": self.generation_round,
            "physical_calls_per_round": self.physical_calls_per_round,
            "global_call_index_start": self.global_call_index_start,
            "round_stride": _SEED_ROUND_STRIDE,
            "modulus": _SEED_MODULUS,
            "capacity_proof": self.seed_capacity_contract(),
        }

    def expected_phase_executions(self) -> tuple[RequestExecutionRecord, ...]:
        """Reconstruct every phase row from caller-owned manifest and topology."""
        if self.topology == "tp2":
            return build_wave_execution_records(
                self.manifest,
                global_wave_size=self.global_wave_size,
                generation_round=self.generation_round,
                call_index_start=self.global_call_index_start,
            )
        return _expected_dp2_executions(
            self.manifest,
            profile=self.profile,
            generation_round=self.generation_round,
            call_index_start=self.global_call_index_start,
            global_index_start=0,
        )

    def rank_row_bindings(self) -> list[dict[str, Any]]:
        """Hash exact caller-derived request IDs and coordinate rows per DP rank."""
        executions = self.expected_phase_executions()
        bindings = []
        for dp_rank in range(self.replica_count):
            rows = [execution.to_dict() for execution in executions if execution.dp_rank == dp_rank]
            request_ids = [row["request_id"] for row in rows]
            bindings.append(
                {
                    "dp_rank": dp_rank,
                    "request_count": len(rows),
                    "request_ids_sha256": hashlib.sha256(
                        json.dumps(request_ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "coordinate_rows_sha256": hashlib.sha256(
                        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
            )
        return bindings

    def summary(self) -> dict[str, Any]:
        """Return durable evidence for the independently reconstructed binding."""
        request_ids = [request.request_id for request in self.manifest.requests]
        request_ids_sha256 = hashlib.sha256(
            json.dumps(request_ids, separators=(",", ":")).encode()
        ).hexdigest()
        profile_contract = self.profile_contract()
        profile_sha256 = hashlib.sha256(
            json.dumps(profile_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "schema_version": 1,
            "manifest_sha256": self.manifest.sha256,
            "request_count": len(request_ids),
            "request_ids_sha256": request_ids_sha256,
            "base_seed": self.manifest.seed,
            "topology": self.topology,
            "global_wave_size": self.global_wave_size,
            "replica_count": self.replica_count,
            "tensor_parallel_size": self.profile.tensor_parallel_size,
            "tp_ranks": list(range(self.profile.tensor_parallel_size)),
            "dp_ranks": list(range(self.replica_count)),
            "worker_rank_pairs": [
                {"dp_rank": dp_rank, "tp_rank": tp_rank}
                for dp_rank in range(self.replica_count)
                for tp_rank in range(self.profile.tensor_parallel_size)
            ],
            "rank_row_bindings": self.rank_row_bindings(),
            "profile_sha256": profile_sha256,
            "generation_round": self.generation_round,
            "physical_calls_per_round": self.physical_calls_per_round,
            "global_call_index_start": self.global_call_index_start,
        }


class GpuPreflightError(RuntimeError):
    """Fail-closed hardware-preflight error with raw publishable evidence."""

    def __init__(self, message: str, *, evidence: dict[str, Any]) -> None:
        super().__init__(message)
        self.evidence = evidence


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def runtime_versions() -> dict[str, Any]:
    """Return exact runtime versions needed to reproduce an artifact."""
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm": _package_version("vllm"),
        "triton": _package_version("triton"),
        "transformers": _package_version("transformers"),
        "nemo_rl": _package_version("nemo-rl"),
    }


def benchmark_mode_from_args(args: Any) -> str:
    """Resolve one preflight, proof, or low-overhead speed invocation."""
    linked_proof = getattr(args, "linked_proof_artifact", None)
    if args.context_preflight_only:
        if args.proof or linked_proof is not None:
            raise ValueError("context preflight cannot enable proof or link a proof artifact")
        return "preflight"
    if args.proof:
        if linked_proof is not None:
            raise ValueError("a proof run cannot link another proof artifact")
        return "proof"
    return "speed"


def benchmark_instrumentation_contract(mode: str) -> dict[str, bool]:
    """Describe instrumentation that is active inside generation measurements."""
    if mode not in {"proof", "speed"}:
        raise ValueError(f"generation instrumentation requires proof or speed mode, got {mode!r}")
    collect_proof = mode == "proof"
    return {
        "scheduler_callbacks_during_generation": collect_proof,
        "worker_proof_rpcs": collect_proof,
        "sampler_call_counter_during_generation": collect_proof,
        "sampler_endpoint_state_hashing_during_generation": False,
        "prefix_clone_instrumentation": collect_proof,
        "peak_memory_polling_during_generation": collect_proof,
        "post_generation_exact_output_validation": True,
    }


def benchmark_phase_coordinates(
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    *,
    generation_round_start: int,
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, Any], ...]:
    """Allocate one disjoint semantic round, call range, and request range per phase."""
    if not isinstance(manifest, WorkloadManifest) or not isinstance(profile, Evo2VllmProfile):
        raise TypeError("benchmark phase coordinates require exact manifest and profile objects")
    if type(generation_round_start) is not int or generation_round_start < 0:
        raise ValueError("generation_round_start must be a nonnegative built-in integer")
    phase_specs = _phase_specs(warmups, repetitions)
    physical_calls_per_round = len(
        build_request_waves(
            request_count=len(manifest.requests),
            global_batch_size=profile.global_wave_size,
            replica_count=profile.replica_count,
        )
    )
    return tuple(
        {
            "phase": phase,
            "sample_index": sample_index,
            "generation_round": generation_round,
            "global_call_index_start": generation_round * physical_calls_per_round,
            "global_request_index_start": generation_round * len(manifest.requests),
            "physical_calls_per_round": physical_calls_per_round,
            "semantic_request_count": len(manifest.requests),
        }
        for sample_index, (phase, _phase_index) in enumerate(phase_specs)
        for generation_round in (generation_round_start + sample_index,)
    )


def mbs1_exact1k_contract(
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
) -> dict[str, Any]:
    """Seal the primary repeated-prompt 10x96+40 stochastic audit workload."""
    if len(manifest.requests) != 1_000:
        raise ValueError("MBS=1 exact-1k audit requires exactly 1,000 semantic requests")
    if profile.global_wave_size != 96:
        raise ValueError("MBS=1 exact-1k audit requires global_wave_size=96")
    if profile.shared_prefix_state_reuse is not True:
        raise ValueError("MBS=1 exact-1k audit requires physical shared-prefix state reuse")
    prompt_identity = shared_prefix_manifest_evidence(manifest)
    waves = build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=profile.global_wave_size,
        replica_count=profile.replica_count,
    )
    global_counts = [wave.request_count for wave in waves]
    local_counts = [[shard.request_count for shard in wave.shards] for wave in waves]
    expected_local = [[96]] * 10 + [[40]] if profile.topology == "tp2" else [[48, 48]] * 10 + [[20, 20]]
    if global_counts != [96] * 10 + [40] or local_counts != expected_local:
        raise AssertionError("MBS=1 exact-1k physical schedule is not 10x96 plus tail40")
    return {
        "schema_version": 1,
        "workload": "primary-homogeneous-mbs1-exact1k",
        "semantic_request_count": 1_000,
        "unique_semantic_request_ids": len({request.request_id for request in manifest.requests}) == 1_000,
        "prompt_identity": prompt_identity,
        "physical_call_count": 11,
        "global_call_request_counts": global_counts,
        "per_engine_call_request_counts": local_counts,
        "expected_first_wave_cache_misses": profile.replica_count,
        "expected_first_wave_cache_hits": 96 - profile.replica_count,
        "expected_later_wave_cache_misses": 0,
        "first_wave_includes_prefix_materialization": True,
        "second_wave_is_warmed_prefix_repeat": True,
        "semantic_padding": False,
    }


def exact_generation_progress_contract(
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    *,
    generation_round: int,
) -> dict[str, Any]:
    """Describe exact first-sample, decode-update, and retained-output counts."""
    if type(generation_round) is not int or generation_round < 0:
        raise ValueError("generation_round must be a nonnegative built-in integer")
    request_count = len(manifest.requests)
    retained_tokens = request_count * manifest.max_new_tokens
    contract: dict[str, Any] = {
        "schema_version": 1,
        "request_count": request_count,
        "max_new_tokens": manifest.max_new_tokens,
        "expected_first_sampled_tokens": request_count,
        "expected_decode_token_updates": request_count * max(0, manifest.max_new_tokens - 1),
        "expected_retained_output_token_ids": retained_tokens,
        "expected_retained_chosen_token_logprobs": retained_tokens,
    }
    if request_count == 1_000:
        if profile.global_wave_size != 96:
            raise ValueError("1,000-request audit requires global_wave_size=96")
        waves = build_request_waves(
            request_count=request_count,
            global_batch_size=profile.global_wave_size,
            replica_count=profile.replica_count,
        )
        global_call_index_start = generation_round * len(waves)
        contract["physical_schedule"] = {
            "generation_round": generation_round,
            "global_call_index_start": global_call_index_start,
            "global_wave_size": profile.global_wave_size,
            "physical_calls_per_round": len(waves),
            "global_call_request_counts": [wave.request_count for wave in waves],
            "per_engine_call_request_counts": [
                [shard.request_count for shard in wave.shards] for wave in waves
            ],
            "calls": [
                {
                    "call_in_round": wave.wave_index,
                    "global_call_index": global_call_index_start + wave.wave_index,
                    "global_start": wave.start,
                    "global_stop": wave.stop,
                    "global_request_count": wave.request_count,
                    "replicas": [
                        {
                            "dp_rank": shard.replica_index,
                            "global_start": shard.start,
                            "global_stop": shard.stop,
                            "local_request_count": shard.request_count,
                        }
                        for shard in wave.shards
                    ],
                }
                for wave in waves
            ],
        }
    return contract


def _require_builtin_integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AssertionError(f"{label} must be a built-in integer >= {minimum}")
    return value


def _require_finite_number(
    value: Any,
    *,
    label: str,
    minimum: float = 0.0,
    strictly_positive: bool = False,
) -> int | float:
    if type(value) not in (int, float):
        raise AssertionError(f"{label} must be a built-in integer or float")
    if not math.isfinite(value):
        raise AssertionError(f"{label} must be finite")
    if value < minimum or (strictly_positive and value <= minimum):
        comparator = ">" if strictly_positive else ">="
        raise AssertionError(f"{label} must be {comparator} {minimum}")
    return value


def _validate_exact_wave_coordinates(
    retained: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    """Require exact raw physical-wave coordinates without numeric aliases."""
    for field, expected_value in expected.items():
        observed_value = retained.get(field)
        if type(expected_value) is int:
            _require_builtin_integer(observed_value, label=f"physical wave {field}")
        elif type(observed_value) is not type(expected_value):
            raise AssertionError(f"physical wave {field} has the wrong raw type")
        if observed_value != expected_value:
            raise AssertionError(f"physical wave {field} does not match reconstructed geometry")


def validate_exact_generation_progress_contract(
    retained: Any,
    *,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    generation_round: int,
) -> dict[str, Any]:
    """Reconstruct exact progress and reject numeric aliases in retained raw geometry."""
    if not isinstance(retained, dict):
        raise AssertionError("exact-generation progress contract must be a JSON object")
    expected = exact_generation_progress_contract(
        manifest,
        profile,
        generation_round=generation_round,
    )
    for field in (
        "schema_version",
        "request_count",
        "max_new_tokens",
        "expected_first_sampled_tokens",
        "expected_decode_token_updates",
        "expected_retained_output_token_ids",
        "expected_retained_chosen_token_logprobs",
    ):
        _require_builtin_integer(retained.get(field), label=f"exact-generation {field}")
    if len(manifest.requests) == 1_000:
        schedule = retained.get("physical_schedule")
        if not isinstance(schedule, dict):
            raise AssertionError("exact-generation physical schedule must be a JSON object")
        for field in (
            "generation_round",
            "global_call_index_start",
            "global_wave_size",
            "physical_calls_per_round",
        ):
            _require_builtin_integer(schedule.get(field), label=f"physical schedule {field}")
        for field in ("global_call_request_counts",):
            values = schedule.get(field)
            if not isinstance(values, list):
                raise AssertionError(f"physical schedule {field} must be a JSON array")
            for index, value in enumerate(values):
                _require_builtin_integer(value, label=f"physical schedule {field}[{index}]", minimum=1)
        local_counts = schedule.get("per_engine_call_request_counts")
        if not isinstance(local_counts, list):
            raise AssertionError("per-engine request counts must be a JSON array")
        for call_index, values in enumerate(local_counts):
            if not isinstance(values, list):
                raise AssertionError("per-engine call request counts must be JSON arrays")
            for dp_rank, value in enumerate(values):
                _require_builtin_integer(
                    value,
                    label=f"per-engine call {call_index} rank {dp_rank} request count",
                    minimum=1,
                )
        calls = schedule.get("calls")
        if not isinstance(calls, list):
            raise AssertionError("physical schedule calls must be a JSON array")
        for call_position, call in enumerate(calls):
            if not isinstance(call, dict):
                raise AssertionError("physical schedule call rows must be JSON objects")
            for field in (
                "call_in_round",
                "global_call_index",
                "global_start",
                "global_stop",
                "global_request_count",
            ):
                _require_builtin_integer(
                    call.get(field),
                    label=f"physical call {call_position} {field}",
                    minimum=1 if field in {"global_stop", "global_request_count"} else 0,
                )
            replicas = call.get("replicas")
            if not isinstance(replicas, list):
                raise AssertionError("physical call replicas must be a JSON array")
            for replica_position, replica in enumerate(replicas):
                if not isinstance(replica, dict):
                    raise AssertionError("physical call replica rows must be JSON objects")
                for field in ("dp_rank", "global_start", "global_stop", "local_request_count"):
                    _require_builtin_integer(
                        replica.get(field),
                        label=f"physical call {call_position} replica {replica_position} {field}",
                        minimum=1 if field in {"global_stop", "local_request_count"} else 0,
                    )
    if retained != expected:
        raise AssertionError("exact-generation progress contract does not match reconstructed raw geometry")
    return expected


def build_benchmark_contract(
    args: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
) -> dict[str, Any]:
    """Build the optimized engine/workload identity shared by proof and speed lanes."""
    generation_round = args.generation_round
    if type(generation_round) is not int or generation_round < 0:
        raise ValueError("generation_round must be a nonnegative built-in integer")
    mixed_identity = getattr(args, "mixed_canonical_identity", False)
    if type(mixed_identity) is not bool:
        raise TypeError("mixed canonical identity flag must be a built-in boolean")
    mixed_same_engine = getattr(args, "mixed_same_engine_qualification", False)
    if type(mixed_same_engine) is not bool:
        raise TypeError("mixed same-engine qualification flag must be a built-in boolean")
    if mixed_same_engine and not mixed_identity:
        raise ValueError("mixed same-engine qualification requires mixed canonical identity")
    if sum(
        (
            getattr(args, "canonical_identity_case", None) is not None,
            getattr(args, "common_prefix_identity_case", None) is not None,
            mixed_identity,
        )
    ) > 1:
        raise ValueError("canonical, common-prefix, and mixed identity modes are mutually exclusive")
    caller_coordinates = CallerCoordinateContract.from_inputs(manifest, profile, generation_round)
    profile_contract = asdict(profile)
    profile_contract.pop("proof")
    identity_context = canonical_identity_context(args, manifest, profile)
    common_identity_context = common_prefix_identity_context(args, manifest, profile)
    mixed_identity_context_value = mixed_canonical_identity_context(args, manifest, profile)
    if mixed_same_engine:
        stage_specs = mixed_same_engine_stage_specs(args, manifest, profile)
        if stage_specs is None or mixed_identity_context_value is None:
            raise AssertionError("mixed same-engine qualification did not resolve its exact stages")
        mixed_contract = mixed_identity_context_value[2]
        admission_bundle = mixed_contract["admission_bundle"]
        phase_coordinates = tuple(
            {
                "phase": spec["stage"],
                "sample_index": sample_index,
                "generation_round": spec["execution_records"][0].generation_round,
                "global_call_index_start": spec["execution_records"][0].call_index,
                "global_request_index_start": spec["execution_records"][0].global_request_index,
                "physical_calls_per_round": 1,
                "semantic_request_count": len(spec["manifest"].requests),
                "manifest_sha256": spec["manifest"].sha256,
            }
            for sample_index, spec in enumerate(stage_specs)
        )
        seed_stream = admission_bundle
        measurement_protocol = "mixed-b4-then-b96-single-engine"
    else:
        phase_coordinates = benchmark_phase_coordinates(
            manifest,
            profile,
            generation_round_start=generation_round,
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        seed_stream = caller_coordinates.seed_stream_contract()
        measurement_protocol = "cold-warm-steady"
    mbs1_enabled = bool(getattr(args, "mbs1_exact1k_audit", False))
    if mbs1_enabled and not bool(getattr(args, "exact_progress_gate", False)):
        raise ValueError("MBS=1 exact-1k audit requires --exact-progress-gate")
    return {
        "schema_version": 4,
        "backend": args.backend,
        "topology": args.topology,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "load_format": args.load_format,
        "manifest_sha256": manifest.sha256,
        "profile": profile_contract,
        "seed_stream": seed_stream,
        "measurement": {
            "protocol": measurement_protocol,
            "warmups": int(args.warmups),
            "repetitions": int(args.repetitions),
            "phase_coordinates": list(phase_coordinates),
        },
        "mixed_same_engine_qualification": mixed_same_engine,
        "canonical_identity": None if identity_context is None else identity_context[2],
        "common_prefix_identity": None if common_identity_context is None else common_identity_context[3],
        "mixed_canonical_identity": (
            None if mixed_identity_context_value is None else mixed_identity_context_value[2]
        ),
        "exact_generation_progress": (
            exact_generation_progress_contract(
                manifest,
                profile,
                generation_round=generation_round,
            )
            if bool(getattr(args, "exact_progress_gate", False))
            else None
        ),
        "mbs1_exact1k": mbs1_exact1k_contract(manifest, profile) if mbs1_enabled else None,
    }


def canonical_identity_context(
    args: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
) -> tuple[Any, Any, dict[str, Any]] | None:
    """Resolve and validate one canonical identity case, schedule, and linked contract."""
    case_index = getattr(args, "canonical_identity_case", None)
    if case_index is None:
        return None
    prompts_csv = getattr(args, "canonical_prompts_csv", None)
    tokenizer_path = getattr(args, "prompt_tokenizer_json", None)
    if prompts_csv is None or tokenizer_path is None:
        raise ValueError("canonical identity contract requires its source and tokenizer paths")
    cases = load_canonical_7b_identity_cases(prompts_csv)
    case = cases[case_index]
    validate_canonical_identity_manifest(manifest, case=case, request_count=len(manifest.requests))
    schedule = build_homogeneous_identity_schedule(
        topology=profile.topology,
        request_count=len(manifest.requests),
        global_wave_size=profile.global_wave_size,
    )
    contract = build_canonical_identity_contract(
        case=case,
        schedule=schedule,
        prompts_csv=prompts_csv,
        tokenizer_path=tokenizer_path,
    )
    if (
        manifest.prompt_source_path != contract["prompts_csv_path"]
        or manifest.prompt_source_sha256 != contract["prompts_csv_sha256"]
        or manifest.prompt_tokenizer_path != contract["tokenizer_path"]
        or manifest.prompt_tokenizer_sha256 != contract["tokenizer_sha256"]
    ):
        raise AssertionError("canonical identity manifest provenance does not match its linked contract")
    return case, schedule, contract


def mixed_canonical_identity_context(
    args: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
) -> tuple[tuple[Any, ...], Any, dict[str, Any], dict[str, WorkloadManifest]] | None:
    """Resolve one active mixed stage plus the caller-owned B4/B96 contract."""
    if getattr(args, "mixed_canonical_identity", False) is not True:
        return None
    prompts_csv = getattr(args, "canonical_prompts_csv", None)
    tokenizer_path = getattr(args, "prompt_tokenizer_json", None)
    if prompts_csv is None or tokenizer_path is None:
        raise ValueError("mixed canonical identity requires its source and tokenizer paths")
    cases = load_canonical_7b_identity_cases(prompts_csv)
    validate_mixed_canonical_identity_manifest(manifest, cases=cases)
    tokenizer = SnapshotBoundTokenizer.from_path(tokenizer_path)
    request_id_root = getattr(args, "request_id_prefix", None)
    if type(request_id_root) is not str or not request_id_root:
        raise ValueError("mixed canonical identity requires a request-ID root")
    stage_manifests = {
        stage: build_mixed_canonical_identity_manifest(
            manifest,
            cases=cases,
            prompts_csv=prompts_csv,
            tokenizer=tokenizer,
            request_count=request_count,
            request_id_prefix=f"{request_id_root}-{'b4' if request_count == 4 else 'b96'}",
        )
        for stage, request_count in (("mixed-b4", 4), ("mixed-b96", 96))
    }
    active_stage = "mixed-b4" if len(manifest.requests) == 4 else "mixed-b96"
    if manifest.sha256 != stage_manifests[active_stage].sha256:
        raise AssertionError("active mixed manifest does not match its caller-derived stage manifest")
    schedule = build_mixed_identity_schedule(
        topology=profile.topology,
        request_count=len(manifest.requests),
        global_wave_size=profile.global_wave_size,
        request_id_prefix=f"{request_id_root}-{'b4' if len(manifest.requests) == 4 else 'b96'}",
    )
    contract = build_mixed_canonical_identity_contract(
        cases=cases,
        schedule=schedule,
        stage_manifests=stage_manifests,
        prompts_csv=prompts_csv,
        tokenizer_path=tokenizer_path,
    )
    return cases, schedule, contract, stage_manifests


def mixed_same_engine_stage_specs(
    args: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
) -> tuple[dict[str, Any], ...] | None:
    """Resolve exact B4 then B96 calls for one capacity-96 engine."""
    enabled = getattr(args, "mixed_same_engine_qualification", False)
    if type(enabled) is not bool:
        raise TypeError("mixed same-engine qualification flag must be a built-in boolean")
    if not enabled:
        return None
    context = mixed_canonical_identity_context(args, manifest, profile)
    if context is None:
        raise ValueError("mixed same-engine qualification requires mixed canonical identity")
    if profile.topology != "tp2":
        raise ValueError("mixed same-engine qualification currently requires TP2")
    if len(manifest.requests) != 96:
        raise ValueError("mixed same-engine qualification requires the B96 source manifest")
    if profile.global_wave_size != 96 or profile.resolved_max_num_seqs != 96:
        raise ValueError("mixed same-engine qualification requires one capacity-96 engine")
    if args.warmups != 0 or args.repetitions != 1:
        raise ValueError("mixed same-engine qualification executes exactly B4 then B96 without repeated phases")
    if args.generation_round != 0:
        raise ValueError("mixed same-engine qualification owns semantic calls 0 and 1")
    if args.proof:
        raise ValueError("mixed same-engine qualification must not install proof-only worker hooks")

    _cases, _active_schedule, contract, stage_manifests = context
    attempts = contract["admission_bundle"]["attempts"]
    specs = []
    for attempt in attempts:
        stage = attempt["stage"]
        stage_manifest = stage_manifests[stage]
        request_count = attempt["request_count"]
        stage_prefix = stage_manifest.requests[0].request_id.rsplit("-case", 1)[0]
        schedule = build_mixed_identity_schedule(
            topology=profile.topology,
            request_count=request_count,
            global_wave_size=request_count,
            request_id_prefix=stage_prefix,
        )
        execution_records = tuple(
            RequestExecutionRecord(**coordinate) for coordinate in attempt["execution_coordinates"]
        )
        if stage_manifest.sha256 != attempt["manifest_sha256"]:
            raise AssertionError("mixed stage manifest drifted from caller admission")
        if tuple(request.request_id for request in stage_manifest.requests) != schedule.request_ids:
            raise AssertionError("mixed stage manifest drifted from its physical schedule")
        if tuple(record.request_id for record in execution_records) != schedule.request_ids:
            raise AssertionError("mixed stage execution records drifted from caller admission")
        specs.append(
            {
                "stage": stage,
                "manifest": stage_manifest,
                "schedule": schedule,
                "execution_records": execution_records,
            }
        )
    if [spec["stage"] for spec in specs] != ["mixed-b4", "mixed-b96"]:
        raise AssertionError("mixed same-engine stages are not ordered B4 then B96")
    return tuple(specs)


def common_prefix_identity_context(
    args: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
) -> tuple[Any, Any, Any, dict[str, Any]] | None:
    """Resolve the immutable serial-vs-batched common-prefix identity contract."""
    case_index = getattr(args, "common_prefix_identity_case", None)
    if case_index is None:
        return None
    prompts_csv = getattr(args, "canonical_prompts_csv", None)
    tokenizer_path = getattr(args, "prompt_tokenizer_json", None)
    if prompts_csv is None or tokenizer_path is None:
        raise ValueError("common-prefix identity contract requires its source and tokenizer paths")
    cases = load_common_prefix_identity_cases(prompts_csv)
    case = cases[case_index]
    validate_common_prefix_identity_manifest(manifest, case=case, request_count=len(manifest.requests))
    serial_schedule = build_homogeneous_identity_schedule(
        topology=profile.topology,
        request_count=1,
        global_wave_size=profile.global_wave_size,
    )
    candidate_schedule = build_homogeneous_identity_schedule(
        topology=profile.topology,
        request_count=len(manifest.requests),
        global_wave_size=profile.global_wave_size,
    )
    contract = build_common_prefix_identity_contract(
        case=case,
        serial_schedule=serial_schedule,
        candidate_schedule=candidate_schedule,
        prompts_csv=prompts_csv,
        tokenizer_path=tokenizer_path,
    )
    if (
        manifest.prompt_source_path != contract["prompts_csv_path"]
        or manifest.prompt_source_sha256 != contract["prompts_csv_sha256"]
        or manifest.prompt_tokenizer_path != contract["tokenizer_path"]
        or manifest.prompt_tokenizer_sha256 != contract["tokenizer_sha256"]
    ):
        raise AssertionError("common-prefix identity manifest provenance does not match its linked contract")
    return case, serial_schedule, candidate_schedule, contract


def manifest_output_decoder(manifest: WorkloadManifest) -> Callable[[Sequence[int]], str] | None:
    """Build a hash-verified tokenizer decoder for retained raw output bytes."""
    tokenizer_path = manifest.prompt_tokenizer_path
    tokenizer_sha256 = manifest.prompt_tokenizer_sha256
    if tokenizer_path is None:
        if tokenizer_sha256 is not None:
            raise AssertionError("output tokenizer provenance is incomplete")
        return None
    path = Path(tokenizer_path).expanduser().resolve()
    tokenizer = SnapshotBoundTokenizer.from_path(path)
    if tokenizer_sha256 is None or tokenizer.source_sha256 != tokenizer_sha256:
        raise AssertionError("output tokenizer SHA256 does not match the workload manifest")
    return tokenizer


def canonical_identity_phase_artifacts(
    *,
    args: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    phase_artifacts: list[dict[str, Any]],
    decode_output_token_ids: Callable[[Sequence[int]], str] | None,
    collect_physical_proof: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Attach independently recomputed per-request identity evidence after generation timing."""
    context = canonical_identity_context(args, manifest, profile)
    if context is None:
        return phase_artifacts, None
    if decode_output_token_ids is None:
        raise AssertionError("canonical identity requires decoded raw output retention")
    case, schedule, contract = context
    expected_request_ids = tuple(request.request_id for request in manifest.requests)
    phase_summaries = []
    minimum_identity = 100.0
    for phase in phase_artifacts:
        outputs = validate_canonical_identity_output_artifact(
            phase["full_output_artifact"],
            case=case,
            expected_request_ids=expected_request_ids,
            decode_output_token_ids=decode_output_token_ids,
        )
        physical = (
            validate_homogeneous_identity_phase_evidence(phase, schedule=schedule) if collect_physical_proof else None
        )
        evidence = {
            "schema_version": 1,
            "outputs": outputs,
            "physical_schedule": physical,
            "physical_schedule_attested_by_linked_proof": False,
            "passed": True,
        }
        phase["canonical_identity_evidence"] = evidence
        minimum_identity = min(minimum_identity, outputs["minimum_observed_identity_percent"])
        phase_summaries.append(
            {
                "phase": phase["phase"],
                "request_count": outputs["request_count"],
                "minimum_observed_identity_percent": outputs["minimum_observed_identity_percent"],
                "physical_schedule_proven": physical is not None,
                "passed": True,
            }
        )
    return phase_artifacts, {
        "schema_version": 1,
        "contract": contract,
        "phase_count": len(phase_summaries),
        "minimum_observed_identity_percent": minimum_identity,
        "phases": phase_summaries,
        "passed": True,
    }


def mixed_canonical_identity_phase_artifacts(
    *,
    args: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    phase_artifacts: list[dict[str, Any]],
    decode_output_token_ids: Callable[[Sequence[int]], str] | None,
    collect_physical_proof: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Score every mixed row while keeping single-stage runs explicitly unqualified."""
    context = mixed_canonical_identity_context(args, manifest, profile)
    if context is None:
        return phase_artifacts, None
    if decode_output_token_ids is None:
        raise AssertionError("mixed canonical identity requires decoded raw output retention")
    cases, _schedule, contract, _stage_manifests = context
    same_engine_specs = mixed_same_engine_stage_specs(args, manifest, profile)
    if same_engine_specs is not None:
        expected_phases = [spec["stage"] for spec in same_engine_specs]
        if [phase.get("phase") for phase in phase_artifacts] != expected_phases:
            raise AssertionError("mixed same-engine phases are not exact ordered B4 then B96")
        summaries = []
        for phase, spec in zip(phase_artifacts, same_engine_specs, strict=True):
            stage_manifest = spec["manifest"]
            expected_request_ids = tuple(request.request_id for request in stage_manifest.requests)
            expected_prompts = tuple(request.prompt_token_ids for request in stage_manifest.requests)
            expected_cases = tuple(cases[index % 4] for index in range(len(stage_manifest.requests)))
            expected_executions = tuple(record.to_dict() for record in spec["execution_records"])
            outputs = validate_mixed_canonical_identity_output_artifact(
                phase["full_output_artifact"],
                cases_by_request=expected_cases,
                expected_request_ids=expected_request_ids,
                expected_prompt_token_ids=expected_prompts,
                expected_execution_coordinates=expected_executions,
                decode_output_token_ids=decode_output_token_ids,
            )
            physical = (
                validate_mixed_identity_phase_evidence(
                    phase,
                    schedule=spec["schedule"],
                    expected_execution_coordinates=expected_executions,
                )
                if collect_physical_proof
                else None
            )
            evidence = {
                "schema_version": 1,
                "stage": spec["stage"],
                "manifest_sha256": stage_manifest.sha256,
                "outputs": outputs,
                "physical_schedule": physical,
                "physical_schedule_attested_by_linked_proof": False,
                "supported_vllm_runtime_metrics": True,
                "proof_only_runtime_hooks_installed": False,
                "same_engine_b4_then_b96_qualified": True,
                "passed": True,
            }
            phase["mixed_canonical_identity_evidence"] = evidence
            summaries.append(
                {
                    "phase": spec["stage"],
                    "manifest_sha256": stage_manifest.sha256,
                    "request_count": outputs["request_count"],
                    "minimum_observed_identity_percent": outputs["minimum_observed_identity_percent"],
                    "physical_schedule_proven": physical is not None,
                    "passed": True,
                }
            )
        return phase_artifacts, {
            "schema_version": 1,
            "contract": contract,
            "phases": summaries,
            "exploratory_output_correctness_passed": True,
            "same_engine_b4_then_b96_qualified": True,
            "physical_schedule_attested": collect_physical_proof,
            "supported_vllm_runtime_metrics": True,
            "proof_only_runtime_hooks_installed": False,
            "timing_admissible_for_speed_ranking": False,
            "passed": True,
        }
    if collect_physical_proof:
        raise AssertionError("single-stage mixed runs cannot qualify physical B4-then-B96 execution")
    coordinates_by_phase = {
        row["phase"]: row
        for row in benchmark_phase_coordinates(
            manifest,
            profile,
            generation_round_start=args.generation_round,
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
    }
    expected_request_ids = tuple(request.request_id for request in manifest.requests)
    expected_prompts = tuple(request.prompt_token_ids for request in manifest.requests)
    expected_cases = tuple(cases[index % 4] for index in range(len(manifest.requests)))
    summaries = []
    for phase in phase_artifacts:
        coordinate = coordinates_by_phase.get(phase.get("phase"))
        if coordinate is None:
            raise AssertionError("mixed canonical phase lacks caller-owned phase coordinates")
        expected_executions = build_wave_execution_records(
            manifest,
            global_wave_size=profile.global_wave_size,
            generation_round=coordinate["generation_round"],
            call_index_start=coordinate["global_call_index_start"],
            global_request_index_start=coordinate["global_request_index_start"],
        )
        outputs = validate_mixed_canonical_identity_output_artifact(
            phase["full_output_artifact"],
            cases_by_request=expected_cases,
            expected_request_ids=expected_request_ids,
            expected_prompt_token_ids=expected_prompts,
            expected_execution_coordinates=tuple(record.to_dict() for record in expected_executions),
            decode_output_token_ids=decode_output_token_ids,
        )
        evidence = {
            "schema_version": 1,
            "outputs": outputs,
            "physical_schedule": None,
            "physical_schedule_attested_by_linked_proof": False,
            "same_engine_b4_then_b96_qualified": False,
            "passed": True,
        }
        phase["mixed_canonical_identity_evidence"] = evidence
        summaries.append(
            {
                "phase": phase["phase"],
                "request_count": outputs["request_count"],
                "minimum_observed_identity_percent": outputs["minimum_observed_identity_percent"],
                "passed": True,
            }
        )
    return phase_artifacts, {
        "schema_version": 1,
        "contract": contract,
        "phases": summaries,
        "exploratory_output_correctness_passed": bool(summaries),
        "same_engine_b4_then_b96_qualified": False,
        "physical_schedule_attested": False,
        "passed": bool(summaries),
    }


def common_prefix_identity_phase_artifacts(
    *,
    args: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    serial_reference_phase: dict[str, Any] | None,
    phase_artifacts: list[dict[str, Any]],
    decode_output_token_ids: Callable[[Sequence[int]], str] | None,
    collect_physical_proof: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Attach serial-vs-batched common-prefix evidence to production phases."""
    context = common_prefix_identity_context(args, manifest, profile)
    if context is None:
        if serial_reference_phase is not None:
            raise AssertionError("common-prefix serial reference exists without an enabled contract")
        return phase_artifacts, None
    if serial_reference_phase is None or decode_output_token_ids is None:
        raise AssertionError("common-prefix identity requires a serial phase and decoded raw outputs")
    case, serial_schedule, candidate_schedule, contract = context
    serial_output = serial_reference_phase.get("full_output_artifact")
    if not isinstance(serial_output, dict):
        raise AssertionError("common-prefix serial phase is missing its full output artifact")
    serial_physical = (
        validate_homogeneous_identity_phase_evidence(serial_reference_phase, schedule=serial_schedule)
        if collect_physical_proof
        else None
    )
    expected_request_ids = tuple(request.request_id for request in manifest.requests)
    summaries = []
    for phase in phase_artifacts:
        candidate_output = phase.get("full_output_artifact")
        if not isinstance(candidate_output, dict):
            raise AssertionError("common-prefix candidate phase is missing its full output artifact")
        outputs = validate_common_prefix_identity_output_artifacts(
            serial_output,
            candidate_output,
            case=case,
            expected_candidate_request_ids=expected_request_ids,
            decode_output_token_ids=decode_output_token_ids,
        )
        physical = (
            validate_homogeneous_identity_phase_evidence(phase, schedule=candidate_schedule)
            if collect_physical_proof
            else None
        )
        evidence = {
            "schema_version": 1,
            "phase": phase.get("phase"),
            "outputs": outputs,
            "physical_schedule": physical,
            "physical_schedule_attested_by_linked_proof": False,
            "passed": outputs.get("passed") is True and (physical is None or physical.get("passed") is True),
        }
        if evidence["passed"] is not True:
            raise AssertionError("common-prefix production phase failed its accuracy contract")
        phase["common_prefix_identity_evidence"] = evidence
        summaries.append(evidence)
    return phase_artifacts, {
        "schema_version": 1,
        "contract": contract,
        "serial_reference": {
            "full_output_artifact": serial_output,
            "physical_schedule": serial_physical,
            "physical_schedule_attested_by_linked_proof": False,
        },
        "phases": summaries,
        "passed": bool(summaries)
        and all(summary.get("passed") is True for summary in summaries)
        and (serial_physical is None or serial_physical.get("passed") is True),
    }


def validate_canonical_identity_proof_evidence(
    artifact: dict[str, Any],
    *,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    expected_contract: dict[str, Any],
) -> dict[str, Any] | None:
    """Recompute linked canonical identity evidence from raw sidecars and physical telemetry."""
    contract = expected_contract.get("canonical_identity")
    phases = artifact.get("phases")
    if not isinstance(phases, list):
        raise AssertionError("canonical identity proof phases are missing")
    if contract is None:
        if artifact.get("canonical_identity") is not None or any(
            isinstance(phase, dict) and phase.get("canonical_identity_evidence") is not None for phase in phases
        ):
            raise AssertionError("non-identity proof retained unexpected canonical identity evidence")
        return None
    if not isinstance(contract, dict):
        raise AssertionError("canonical identity benchmark contract is malformed")
    case_index = contract.get("case_index")
    if isinstance(case_index, bool) or not isinstance(case_index, int) or not 0 <= case_index < 4:
        raise AssertionError("canonical identity contract case index is malformed")
    try:
        cases = load_canonical_7b_identity_cases(contract["prompts_csv_path"])
        case = cases[case_index]
        schedule = build_homogeneous_identity_schedule(
            topology=profile.topology,
            request_count=len(manifest.requests),
            global_wave_size=profile.global_wave_size,
        )
        recomputed_contract = build_canonical_identity_contract(
            case=case,
            schedule=schedule,
            prompts_csv=contract["prompts_csv_path"],
            tokenizer_path=contract["tokenizer_path"],
        )
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        raise AssertionError("canonical identity contract could not be reconstructed") from error
    if contract != recomputed_contract:
        raise AssertionError("canonical identity benchmark contract failed reconstruction")
    validate_canonical_identity_manifest(manifest, case=case, request_count=len(manifest.requests))
    decoder = manifest_output_decoder(manifest)
    if decoder is None:
        raise AssertionError("canonical identity proof lacks its output tokenizer")

    expected_request_ids = tuple(request.request_id for request in manifest.requests)
    phase_summaries = []
    minimum_identity = 100.0
    for phase in phases:
        if not isinstance(phase, dict):
            raise AssertionError("canonical identity phase evidence is malformed")
        physical = validate_homogeneous_identity_phase_evidence(phase, schedule=schedule)
        outputs = validate_canonical_identity_output_artifact(
            phase.get("full_output_artifact", {}),
            case=case,
            expected_request_ids=expected_request_ids,
            decode_output_token_ids=decoder,
        )
        recomputed_phase = {
            "schema_version": 1,
            "outputs": outputs,
            "physical_schedule": physical,
            "physical_schedule_attested_by_linked_proof": False,
            "passed": True,
        }
        if phase.get("canonical_identity_evidence") != recomputed_phase:
            raise AssertionError("canonical identity phase summary does not match recomputed raw evidence")
        minimum_identity = min(minimum_identity, outputs["minimum_observed_identity_percent"])
        phase_summaries.append(
            {
                "phase": phase["phase"],
                "request_count": outputs["request_count"],
                "minimum_observed_identity_percent": outputs["minimum_observed_identity_percent"],
                "physical_schedule_proven": True,
                "passed": True,
            }
        )
    recomputed = {
        "schema_version": 1,
        "contract": recomputed_contract,
        "phase_count": len(phase_summaries),
        "minimum_observed_identity_percent": minimum_identity,
        "phases": phase_summaries,
        "passed": True,
    }
    if artifact.get("canonical_identity") != recomputed:
        raise AssertionError("canonical identity run summary does not match recomputed phase evidence")
    return recomputed


def validate_common_prefix_identity_proof_evidence(
    artifact: dict[str, Any],
    *,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    expected_contract: dict[str, Any],
) -> dict[str, Any] | None:
    """Recompute linked serial-vs-batched identity evidence from raw proof artifacts."""
    contract = expected_contract.get("common_prefix_identity")
    phases = artifact.get("phases")
    if not isinstance(phases, list):
        raise AssertionError("common-prefix identity proof phases are missing")
    if contract is None:
        if (
            artifact.get("common_prefix_identity") is not None
            or artifact.get("common_prefix_serial_reference") is not None
            or any(
                isinstance(phase, dict) and phase.get("common_prefix_identity_evidence") is not None
                for phase in phases
            )
        ):
            raise AssertionError("non-identity proof retained unexpected common-prefix identity evidence")
        return None
    if not isinstance(contract, dict):
        raise AssertionError("common-prefix identity benchmark contract is malformed")
    case_index = contract.get("case_index")
    if isinstance(case_index, bool) or not isinstance(case_index, int) or not 0 <= case_index < 4:
        raise AssertionError("common-prefix identity contract case index is malformed")
    try:
        cases = load_common_prefix_identity_cases(contract["prompts_csv_path"])
        case = cases[case_index]
        serial_schedule = build_homogeneous_identity_schedule(
            topology=profile.topology,
            request_count=1,
            global_wave_size=profile.global_wave_size,
        )
        candidate_schedule = build_homogeneous_identity_schedule(
            topology=profile.topology,
            request_count=len(manifest.requests),
            global_wave_size=profile.global_wave_size,
        )
        recomputed_contract = build_common_prefix_identity_contract(
            case=case,
            serial_schedule=serial_schedule,
            candidate_schedule=candidate_schedule,
            prompts_csv=contract["prompts_csv_path"],
            tokenizer_path=contract["tokenizer_path"],
        )
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        raise AssertionError("common-prefix identity contract could not be reconstructed") from error
    if contract != recomputed_contract:
        raise AssertionError("common-prefix identity benchmark contract failed reconstruction")
    validate_common_prefix_identity_manifest(manifest, case=case, request_count=len(manifest.requests))
    decoder = manifest_output_decoder(manifest)
    if decoder is None:
        raise AssertionError("common-prefix identity proof lacks its output tokenizer")

    serial_phase = artifact.get("common_prefix_serial_reference")
    if not isinstance(serial_phase, dict):
        raise AssertionError("common-prefix identity proof lacks its serial reference phase")
    serial_output = serial_phase.get("full_output_artifact")
    if not isinstance(serial_output, dict):
        raise AssertionError("common-prefix serial reference lacks its raw output artifact")
    serial_physical = validate_homogeneous_identity_phase_evidence(serial_phase, schedule=serial_schedule)

    expected_request_ids = tuple(request.request_id for request in manifest.requests)
    phase_summaries = []
    for phase in phases:
        if not isinstance(phase, dict):
            raise AssertionError("common-prefix identity phase evidence is malformed")
        candidate_output = phase.get("full_output_artifact")
        if not isinstance(candidate_output, dict):
            raise AssertionError("common-prefix candidate phase lacks its raw output artifact")
        physical = validate_homogeneous_identity_phase_evidence(phase, schedule=candidate_schedule)
        outputs = validate_common_prefix_identity_output_artifacts(
            serial_output,
            candidate_output,
            case=case,
            expected_candidate_request_ids=expected_request_ids,
            decode_output_token_ids=decoder,
        )
        recomputed_phase = {
            "schema_version": 1,
            "phase": phase.get("phase"),
            "outputs": outputs,
            "physical_schedule": physical,
            "physical_schedule_attested_by_linked_proof": False,
            "passed": outputs.get("passed") is True and physical.get("passed") is True,
        }
        if recomputed_phase["passed"] is not True:
            raise AssertionError("common-prefix identity phase failed recomputed gates")
        if phase.get("common_prefix_identity_evidence") != recomputed_phase:
            raise AssertionError("common-prefix identity phase summary does not match recomputed raw evidence")
        phase_summaries.append(recomputed_phase)
    recomputed = {
        "schema_version": 1,
        "contract": recomputed_contract,
        "serial_reference": {
            "full_output_artifact": serial_output,
            "physical_schedule": serial_physical,
            "physical_schedule_attested_by_linked_proof": False,
        },
        "phases": phase_summaries,
        "passed": bool(phase_summaries)
        and all(summary.get("passed") is True for summary in phase_summaries)
        and serial_physical.get("passed") is True,
    }
    if artifact.get("common_prefix_identity") != recomputed:
        raise AssertionError("common-prefix identity run summary does not match recomputed phase evidence")
    return recomputed


def benchmark_contract_sha256(contract: dict[str, Any]) -> str:
    """Return the canonical digest used to link proof and speed artifacts."""
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_linked_proof_artifact(
    path: str | Path,
    *,
    expected_contract: dict[str, Any],
    caller_coordinates: CallerCoordinateContract,
    require_memory_headroom: bool = False,
) -> dict[str, Any]:
    """Require one successful proof artifact with the exact speed-run contract."""
    if not isinstance(caller_coordinates, CallerCoordinateContract):
        raise TypeError("linked proof validation requires an external caller coordinate contract")
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"linked proof artifact is missing: {artifact_path}")
    artifact_snapshot = read_json_snapshot(artifact_path, label="linked proof artifact")
    artifact = _require_dict(artifact_snapshot.value, label="linked proof artifact")
    if artifact.get("benchmark_mode") != "proof":
        raise AssertionError("linked artifact is not a proof benchmark")
    if artifact.get("invocation", {}).get("exit_status") != 0:
        raise AssertionError("linked proof artifact did not complete successfully")
    if artifact.get("proof_status", {}).get("passed") is not True:
        raise AssertionError("linked proof artifact did not pass its proof gates")
    phases = artifact.get("phases")
    if not isinstance(phases, list) or not phases or any(not isinstance(phase, dict) for phase in phases):
        raise AssertionError("linked proof artifact is missing concrete phase evidence")
    retained_contract = artifact.get("benchmark_contract")
    if not isinstance(retained_contract, dict):
        raise AssertionError("linked proof artifact is missing its benchmark contract")
    retained_sha256 = benchmark_contract_sha256(retained_contract)
    if artifact.get("benchmark_contract_sha256") != retained_sha256:
        raise AssertionError("linked proof artifact benchmark contract digest is invalid")
    expected_sha256 = benchmark_contract_sha256(expected_contract)
    if retained_sha256 != expected_sha256 or retained_contract != expected_contract:
        raise AssertionError("linked proof artifact benchmark contract does not match the speed run")
    recomputed = _validate_linked_proof_evidence(
        artifact,
        artifact_path=artifact_path,
        expected_contract=expected_contract,
        caller_coordinates=caller_coordinates,
        require_memory_headroom=require_memory_headroom,
    )
    memory_headroom = recomputed["gpu_memory_headroom"]
    return {
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_snapshot.sha256,
        "benchmark_contract_sha256": retained_sha256,
        "proof_status": dict(artifact["proof_status"]),
        "gpu_memory_headroom": memory_headroom,
        "validated_evidence": recomputed,
    }


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


def _validate_full_decode_summary_values(
    summary: dict[str, Any],
    *,
    phase: str,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    expected_tokens = batch_size * max(0, max_new_tokens - 1)
    minimum_tokens = max(0, expected_tokens - batch_size)
    if summary.get("phase") != phase:
        raise AssertionError("FULL decode summary phase does not match its physical wave")
    if summary.get("batch_size") != batch_size or summary.get("max_new_tokens") != max_new_tokens:
        raise AssertionError("FULL decode summary workload dimensions drifted")
    if summary.get("expected_decode_tokens") != expected_tokens:
        raise AssertionError("FULL decode expected-token count is inconsistent")
    if summary.get("minimum_full_decode_tokens") != minimum_tokens:
        raise AssertionError("FULL decode minimum-token gate is inconsistent")
    observation_count = summary.get("observation_count")
    dispatch_count = summary.get("full_dispatch_count")
    full_tokens = summary.get("full_decode_tokens")
    maximum_full_batch = summary.get("maximum_full_batch")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (observation_count, dispatch_count, full_tokens, maximum_full_batch)
    ):
        raise AssertionError("FULL decode numeric evidence is malformed")
    if observation_count == 0 or (max_new_tokens > 1 and dispatch_count == 0):
        raise AssertionError("FULL decode proof retained no graph replay")
    if dispatch_count > observation_count:
        raise AssertionError("FULL decode dispatch count exceeds scheduler observations")
    if maximum_full_batch > batch_size or full_tokens > dispatch_count * batch_size:
        raise AssertionError("FULL decode occupancy exceeds the exact physical batch")
    if summary.get("eager_decode_dispatch_count") != 0:
        raise AssertionError("FULL decode proof retained eager decode dispatch")
    if summary.get("full_decode_unpadded") is not True:
        raise AssertionError("FULL decode proof retained semantic padding")
    global_batch_hit = maximum_full_batch == batch_size
    if summary.get("global_batch_hit") is not global_batch_hit:
        raise AssertionError("FULL decode global-batch hit is inconsistent")
    if max_new_tokens > 1 and not global_batch_hit:
        raise AssertionError("FULL decode proof did not hit the exact physical batch")
    coverage = full_tokens / expected_tokens if expected_tokens else 1.0
    if not math.isclose(float(summary.get("coverage_fraction", -1.0)), coverage):
        raise AssertionError("FULL decode coverage fraction is inconsistent")
    average_occupancy = full_tokens / dispatch_count if dispatch_count else 0.0
    if not math.isclose(float(summary.get("average_full_batch_occupancy", -1.0)), average_occupancy):
        raise AssertionError("FULL decode occupancy is inconsistent")
    minimum_occupancy = batch_size * 0.9
    if not math.isclose(float(summary.get("minimum_average_occupancy", -1.0)), minimum_occupancy):
        raise AssertionError("FULL decode minimum occupancy threshold is inconsistent")
    occupancy_fraction = average_occupancy / batch_size
    if not math.isclose(float(summary.get("occupancy_fraction", -1.0)), occupancy_fraction):
        raise AssertionError("FULL decode occupancy fraction is inconsistent")
    long_gate = max_new_tokens >= 32
    coverage_passed = not long_gate or full_tokens >= minimum_tokens
    occupancy_passed = not long_gate or average_occupancy >= batch_size * 0.9
    if summary.get("long_run_gates_applied") is not long_gate:
        raise AssertionError("FULL decode long-run gate selection is inconsistent")
    if summary.get("coverage_gate_passed") is not coverage_passed:
        raise AssertionError("FULL decode coverage gate result is inconsistent")
    if summary.get("occupancy_gate_passed") is not occupancy_passed:
        raise AssertionError("FULL decode occupancy gate result is inconsistent")
    if summary.get("passed") is not True or not coverage_passed or not occupancy_passed:
        raise AssertionError("FULL decode proof did not pass recomputed gates")


def _validate_cudagraph_phase_evidence(
    phase: dict[str, Any],
    *,
    maximum_wave_size: int,
) -> tuple[int, list[dict[str, Any]]]:
    observation_count = phase.get("cudagraph_observation_count")
    if isinstance(observation_count, bool) or not isinstance(observation_count, int) or observation_count <= 0:
        raise AssertionError("proof phase is missing concrete CUDA graph observations")
    retained = _require_list(
        phase.get("cudagraph_observations_retained"),
        label="retained CUDA graph observations",
    )
    if len(retained) != observation_count or any(not isinstance(item, dict) for item in retained):
        raise AssertionError("CUDA graph proof must retain complete lossless observations")
    aggregate = _require_list(phase.get("cudagraph_summary"), label="CUDA graph aggregate")
    if not aggregate or any(not isinstance(item, dict) for item in aggregate):
        raise AssertionError("CUDA graph aggregate is empty or malformed")
    aggregate_counts = {}
    total_count = 0
    full_tokens = 0
    for item in aggregate:
        count = item.get("count")
        unpadded = item.get("num_unpadded_tokens")
        padded = item.get("num_padded_tokens")
        paddings = item.get("num_paddings")
        mode = item.get("runtime_mode")
        engine_index = item.get("engine_index")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (engine_index, unpadded, padded, paddings)
            )
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(mode, str)
        ):
            raise AssertionError("CUDA graph aggregate contains malformed numeric evidence")
        key = (engine_index, mode, unpadded, padded, paddings)
        if key in aggregate_counts:
            raise AssertionError("CUDA graph aggregate contains duplicate buckets")
        aggregate_counts[key] = count
        total_count += count
        if mode.endswith("FULL"):
            if padded != unpadded or paddings != 0:
                raise AssertionError("FULL CUDA graph aggregate contains scheduler padding")
            full_tokens += count * unpadded
        if mode.endswith("NONE") and unpadded <= maximum_wave_size:
            raise AssertionError("CUDA graph aggregate contains eager decode dispatch")
    if total_count != observation_count:
        raise AssertionError("CUDA graph aggregate count does not match retained observation count")
    if summarize_cudagraph_observations(tuple(retained)) != aggregate:
        raise AssertionError("CUDA graph aggregate does not match complete raw observations")
    return full_tokens, retained


def _validate_fir_route_evidence(
    worker_proof: Sequence[dict[str, Any]],
    *,
    manifest: WorkloadManifest,
) -> None:
    prompt_lengths = [len(request.prompt_token_ids) for request in manifest.requests]
    long_equal_prefill = min(prompt_lengths) >= 1_024 and len(set(prompt_lengths)) == 1
    for worker in worker_proof:
        routes = _require_dict(worker.get("fir_routes"), label="worker FIR route evidence")
        fallback_reasons = routes.get("fallback_reasons", {})
        if not isinstance(fallback_reasons, dict):
            raise AssertionError("FIR fallback reasons must be a JSON object")
        forbidden = set(fallback_reasons) - {"short_request"}
        if forbidden:
            raise AssertionError(f"FIR production dispatch used forbidden fallback reasons: {sorted(forbidden)}")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in fallback_reasons.values()
        ):
            raise AssertionError("FIR fallback counters must be positive integers")
        route_names = set(routes) - {"fallback_reasons"}
        if not route_names:
            raise AssertionError("worker retained no production FIR route hits")
        unknown_routes = route_names - {"direct", "equal_length_conv"}
        if unknown_routes:
            raise AssertionError(f"worker retained unknown FIR routes: {sorted(unknown_routes)}")
        for route_name in route_names:
            totals = routes[route_name]
            if not isinstance(totals, dict) or any(
                isinstance(totals.get(field), bool) or not isinstance(totals.get(field), int) or totals[field] <= 0
                for field in ("calls", "requests", "tokens")
            ):
                raise AssertionError(f"FIR route {route_name!r} has malformed counters")
        if "direct" in route_names:
            if sum(fallback_reasons.values()) != routes["direct"]["calls"]:
                raise AssertionError("direct FIR route calls do not match their retained reasons")
        elif fallback_reasons:
            raise AssertionError("FIR fallback reasons were retained without direct route calls")
        if long_equal_prefill and "equal_length_conv" not in route_names:
            raise AssertionError("long equal-length prefill did not hit equal_length_conv")
        if not long_equal_prefill and "equal_length_conv" in route_names:
            raise AssertionError("short or ragged workload unexpectedly hit equal_length_conv")
        if not long_equal_prefill and "direct" not in route_names:
            raise AssertionError("short or ragged FIR workload did not hit the direct production route")
        if manifest.max_new_tokens > 1 and "direct" not in route_names:
            raise AssertionError("autoregressive decode did not hit the direct production FIR route")


def _validate_worker_gpu_bindings(
    workers: Sequence[dict[str, Any]],
    *,
    hardware: dict[str, Any],
    expected_worker_count: int,
    expected_engine_seed: int,
    expected_physical_indices: Sequence[int],
) -> None:
    devices = _require_list(hardware.get("devices"), label="GPU hardware devices")
    if len(workers) != expected_worker_count:
        raise AssertionError("worker proof count does not match the physical model topology")
    if isinstance(expected_engine_seed, bool) or not isinstance(expected_engine_seed, int) or expected_engine_seed < 0:
        raise AssertionError("expected engine seed must be a nonnegative integer")
    if len(expected_physical_indices) != expected_worker_count:
        raise AssertionError("expected physical GPU assignments must match the worker topology")
    if any(
        isinstance(physical_index, bool) or not isinstance(physical_index, int) or physical_index < 0
        for physical_index in expected_physical_indices
    ):
        raise AssertionError("expected physical GPU indices must be nonnegative integers")
    if len(set(expected_physical_indices)) != expected_worker_count:
        raise AssertionError("expected physical GPU assignments must be unique")
    physical = {}
    physical_by_index = {}
    physical_by_uuid = {}
    for device in devices:
        if not isinstance(device, dict):
            raise AssertionError("GPU hardware device provenance is malformed")
        identity = (device.get("uuid"), device.get("pci_bus_id"))
        if not all(isinstance(value, str) and value for value in identity) or identity in physical:
            raise AssertionError("GPU UUID/PCI provenance must be complete and unique")
        physical_index = device.get("physical_index")
        if (
            isinstance(physical_index, bool)
            or not isinstance(physical_index, int)
            or physical_index < 0
            or physical_index in physical_by_index
        ):
            raise AssertionError("GPU physical-index provenance must be complete and unique")
        physical[identity] = device
        physical_by_index[physical_index] = device
        physical_by_uuid[device.get("uuid")] = device
    ranks = []
    observed = set()
    for worker in workers:
        rank = worker.get("rank")
        identity = (worker.get("device_uuid"), worker.get("pci_bus_id"))
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= expected_worker_count:
            raise AssertionError("worker rank is malformed")
        if worker.get("engine_seed") != expected_engine_seed:
            raise AssertionError("every tensor-parallel worker must attest the same intended engine seed")
        if identity not in physical or identity in observed:
            raise AssertionError("worker rank is not bound to one unique physical GPU UUID/PCI identity")
        expected_device = physical_by_index.get(expected_physical_indices[rank])
        if (
            expected_device is None
            or (
                expected_device.get("uuid"),
                expected_device.get("pci_bus_id"),
            )
            != identity
        ):
            raise AssertionError("worker rank does not match its frozen physical assignment")
        if worker.get("device_name") != physical[identity].get("name"):
            raise AssertionError("worker device name does not match physical GPU provenance")
        visible = worker.get("cuda_visible_devices")
        logical_device = worker.get("logical_device")
        if (
            not isinstance(visible, str)
            or isinstance(logical_device, bool)
            or not isinstance(logical_device, int)
            or logical_device < 0
        ):
            raise AssertionError("worker proof omitted CUDA_VISIBLE_DEVICES")
        selectors = tuple(item.strip() for item in visible.split(","))
        if logical_device >= len(selectors) or not selectors[logical_device]:
            raise AssertionError("worker logical device is not present in CUDA_VISIBLE_DEVICES")
        selector = selectors[logical_device]
        if worker.get("visible_device_selector") != selector:
            raise AssertionError("worker retained an inconsistent visible-device selector")
        selected = physical_by_index.get(int(selector)) if selector.isdecimal() else physical_by_uuid.get(selector)
        if selected is None or (selected.get("uuid"), selected.get("pci_bus_id")) != identity:
            raise AssertionError("worker CUDA_VISIBLE_DEVICES selector does not match its UUID/PCI binding")
        ranks.append(rank)
        observed.add(identity)
    if sorted(ranks) != list(range(expected_worker_count)):
        raise AssertionError("worker ranks are not exact and contiguous")


def _validate_worker_rank_continuity(
    workers: Sequence[dict[str, Any]],
    initialized_workers: Sequence[dict[str, Any]],
) -> None:
    def bindings(items: Sequence[dict[str, Any]]) -> dict[int, tuple[Any, Any]]:
        return {item.get("rank"): (item.get("device_uuid"), item.get("pci_bus_id")) for item in items}

    if bindings(workers) != bindings(initialized_workers):
        raise AssertionError("worker rank moved to a different physical GPU during a proof phase")


def _validate_worker_sampler_evidence(
    workers: Sequence[dict[str, Any]],
    *,
    expected_installation: Mapping[str, Any],
    expected_seed_batches: Sequence[Sequence[int]],
    expected_request_generations: Sequence[Mapping[str, Any]],
    require_generation_observations: bool,
) -> list[dict[str, Any]]:
    from bionemo.evo2.vllm.sampler import (
        sampler_runtime_environment_contract,
        validate_sampler_proof_evidence,
    )

    expected_environment = sampler_runtime_environment_contract()
    return [
        validate_sampler_proof_evidence(
            _require_dict(worker.get("sampler"), label="worker sampler proof"),
            expected_environment=expected_environment,
            expected_installation=expected_installation,
            expected_seed_batches=expected_seed_batches,
            expected_request_generations=expected_request_generations,
            require_generation_observations=require_generation_observations,
        )
        for worker in workers
    ]


def _execution_seed_batches(
    executions: Sequence[RequestExecutionRecord],
) -> tuple[tuple[int, ...], ...]:
    call_order = []
    batches: dict[int, list[int]] = {}
    for execution in executions:
        if execution.call_index not in batches:
            call_order.append(execution.call_index)
            batches[execution.call_index] = []
        batches[execution.call_index].append(execution.seed)
    return tuple(tuple(batches[call_index]) for call_index in call_order)


def _validate_full_output_sidecar(
    metadata: dict[str, Any],
    *,
    artifact_path: Path,
    phase: str,
    manifest: WorkloadManifest,
    expected_executions: Sequence[RequestExecutionRecord],
    output_summaries: Any,
) -> dict[str, Any]:
    if metadata.get("schema_version") != 2 or metadata.get("format") != "jsonl":
        raise AssertionError("full-output sidecar schema is unsupported")
    if metadata.get("compression") != "gzip":
        raise AssertionError("full-output sidecar compression is not gzip")
    sidecar = Path(str(metadata.get("path", ""))).expanduser().resolve()
    expected_path = phase_output_artifact_path(artifact_path, phase=phase).resolve()
    if sidecar != expected_path or not sidecar.is_file():
        raise AssertionError("full-output sidecar path does not match its proof namespace")
    try:
        sidecar_snapshot = read_jsonl_snapshot(
            sidecar,
            label=f"{phase} full-output sidecar",
            compression="gzip",
        )
    except ArtifactSnapshotError as error:
        raise AssertionError("full-output sidecar could not be decoded") from error
    if metadata.get("sha256") != sidecar_snapshot.sha256:
        raise AssertionError("full-output sidecar SHA256 does not match retained bytes")
    if metadata.get("size_bytes") != sidecar_snapshot.size_bytes:
        raise AssertionError("full-output sidecar byte count is inconsistent")
    summaries = _require_list(output_summaries, label="phase output summaries")
    if len(expected_executions) != len(manifest.requests) or len(summaries) != len(manifest.requests):
        raise AssertionError("phase outputs do not cover the exact manifest")

    request_count = 0
    generated_count = 0
    request_generations = []
    seen_execution_uids = set()
    for request_count, row in enumerate(sidecar_snapshot.values, start=1):
        index = request_count - 1
        if index >= len(manifest.requests):
            raise AssertionError("full-output sidecar contains extra requests")
        if not isinstance(row, dict):
            raise AssertionError("full-output sidecar row is not a JSON object")
        request = manifest.requests[index]
        execution = expected_executions[index]
        if any(row.get(key) != value for key, value in execution.to_dict().items()):
            raise AssertionError("sidecar execution ownership or seed coordinates drifted")
        execution_uid = row.get("execution_uid")
        if execution_uid in seen_execution_uids:
            raise AssertionError("full-output sidecar contains a duplicate execution UID")
        seen_execution_uids.add(execution_uid)
        if row.get("prompt_token_ids") != list(request.prompt_token_ids):
            raise AssertionError("sidecar prompt tokens do not match the manifest")
        output_ids = row.get("output_token_ids")
        logprobs = row.get("chosen_token_logprobs")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != manifest.max_new_tokens
            or any(isinstance(token, bool) or not isinstance(token, int) for token in output_ids)
        ):
            raise AssertionError("sidecar output token IDs are malformed or not exact length")
        if (
            not isinstance(logprobs, list)
            or len(logprobs) != len(output_ids)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in logprobs
            )
        ):
            raise AssertionError("sidecar chosen-token logprobs are malformed or non-finite")
        expected_lengths = exact_length_evidence(
            prompt_tokens=len(request.prompt_token_ids),
            generated_tokens=len(output_ids),
            requested_new_tokens=manifest.max_new_tokens,
        )
        if any(row.get(key) != value for key, value in expected_lengths.items()):
            raise AssertionError("sidecar requested/observed exact-length evidence drifted")
        if (
            row.get("finish_reason") != "length"
            or row.get("stop_reason") is not None
            or row.get("stopped_on_eos") is not False
        ):
            raise AssertionError("sidecar request did not finish at the exact length boundary")
        record = GenerationRecord(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=tuple(output_ids),
            output_logprobs=tuple(float(value) for value in logprobs),
            requested_max_tokens=manifest.max_new_tokens,
            finish_reason="length",
            stop_reason=None,
            stopped_on_eos=False,
        )
        if summaries[index] != record.summary_dict():
            raise AssertionError("phase output summary does not match its full sidecar row")
        generated_count += len(output_ids)
        request_generations.append(
            {
                "request_id": execution.request_id,
                "seed": execution.seed,
                "accepted_output_token_count": len(output_ids),
            }
        )
    if request_count != len(manifest.requests):
        raise AssertionError("full-output sidecar omitted manifest requests")
    if metadata.get("request_count") != request_count:
        raise AssertionError("full-output sidecar request count is inconsistent")
    if metadata.get("generated_token_count") != generated_count:
        raise AssertionError("full-output sidecar generated-token count is inconsistent")
    if metadata.get("output_token_id_count") != generated_count:
        raise AssertionError("full-output sidecar token-ID count is inconsistent")
    if metadata.get("chosen_token_logprob_count") != generated_count:
        raise AssertionError("full-output sidecar chosen-logprob count is inconsistent")
    return {
        "request_count": request_count,
        "output_token_id_count": generated_count,
        "chosen_token_logprob_count": generated_count,
        "request_generations": request_generations,
    }


def _sidecar_request_generations_for_executions(
    sidecar_counts: Mapping[str, Any],
    executions: Sequence[RequestExecutionRecord],
) -> tuple[dict[str, Any], ...]:
    rows = _require_list(
        sidecar_counts.get("request_generations"),
        label="caller-reopened sidecar request generations",
    )
    expected_fields = {"request_id", "seed", "accepted_output_token_count"}
    by_key = {}
    for row in rows:
        if type(row) is not dict or set(row) != expected_fields:
            raise AssertionError("caller-reopened request generation fields are not exact")
        key = (row.get("request_id"), row.get("seed"))
        if key in by_key:
            raise AssertionError("caller-reopened sidecar repeated one request and seed")
        by_key[key] = row
    selected = []
    seen = set()
    for execution in executions:
        key = (execution.request_id, execution.seed)
        row = by_key.get(key)
        if row is None or key in seen:
            raise AssertionError("sampler execution does not join bijectively to the caller sidecar")
        seen.add(key)
        selected.append(dict(row))
    return tuple(selected)


def _validate_phase_sample(
    phase: dict[str, Any],
    *,
    manifest: WorkloadManifest,
    sample_index: int,
) -> tuple[int, ...]:
    sample = _require_dict(phase.get("sample"), label="phase benchmark sample")
    prompt_tokens = sum(len(request.prompt_token_ids) for request in manifest.requests)
    generated_tokens = len(manifest.requests) * manifest.max_new_tokens
    for field, expected in (
        ("sample_index", sample_index),
        ("request_count", len(manifest.requests)),
        ("prompt_tokens", prompt_tokens),
        ("generated_tokens", generated_tokens),
    ):
        observed = _require_builtin_integer(
            sample.get(field),
            label=f"phase benchmark sample {field}",
            minimum=0,
        )
        if observed != expected:
            raise AssertionError("phase benchmark sample does not match the exact workload")
    output_lengths = sample.get("output_lengths")
    if type(output_lengths) is not list or any(
        type(value) is not int or value <= 0 for value in output_lengths
    ):
        raise AssertionError("phase benchmark sample output_lengths are malformed")
    if output_lengths != [manifest.max_new_tokens] * len(manifest.requests):
        raise AssertionError("phase benchmark sample does not match the exact workload")
    for field, allow_empty in (
        ("ttft_s", False),
        ("inter_token_latency_s", True),
    ):
        timings = sample.get(field)
        if type(timings) is not list or (not allow_empty and not timings):
            raise AssertionError(f"phase benchmark sample {field} is malformed")
        for index, value in enumerate(timings):
            _require_finite_number(
                value,
                label=f"phase benchmark sample {field}[{index}]",
            )
    sample_generation_s = _require_finite_number(
        sample.get("generation_s"),
        label="phase benchmark sample generation_s",
        strictly_positive=True,
    )
    generation_calls = _require_list(phase.get("generation_call_s"), label="generation call timings")
    if not generation_calls:
        raise AssertionError("phase generation timing has no physical calls")
    for index, value in enumerate(generation_calls):
        _require_finite_number(
            value,
            label=f"generation call timings[{index}]",
            strictly_positive=True,
        )
    if phase.get("generation_timing_authority") != COORDINATOR_GENERATION_TIMING_AUTHORITY:
        raise AssertionError("phase generation timing authority is not the coordinator monotonic wall")
    coordinator_wall_s = _require_finite_number(
        phase.get("coordinator_generation_wall_s"),
        label="coordinator generation wall time",
        strictly_positive=True,
    )
    if not math.isclose(float(sample_generation_s), float(coordinator_wall_s)) or not math.isclose(
        float(coordinator_wall_s),
        float(sum(generation_calls)),
    ):
        raise AssertionError("phase generation timing does not match its coordinator physical-call wall")
    peaks = sample.get("peak_device_memory_bytes")
    if type(peaks) is not list or not peaks or any(
        type(value) is not int or value < 0 for value in peaks
    ):
        raise AssertionError("phase peak-memory evidence is malformed")
    return tuple(peaks)


def _validate_direct_phase_evidence(
    artifact: dict[str, Any],
    *,
    artifact_path: Path,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    generation_round: int,
    expected_sampler_installation: Mapping[str, Any],
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    measurement = _require_dict(
        artifact["benchmark_contract"].get("measurement"),
        label="benchmark measurement contract",
    )
    warmups = measurement.get("warmups")
    repetitions = measurement.get("repetitions")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (warmups, repetitions))
        or repetitions == 0
    ):
        raise AssertionError("benchmark measurement counts are malformed")
    expected_names = [
        "cold-generation",
        *(f"warmup-{index}" for index in range(warmups)),
        *(f"steady-{index}" for index in range(repetitions)),
    ]
    phases = _require_list(artifact.get("phases"), label="proof phases")
    if [phase.get("phase") for phase in phases] != expected_names:
        raise AssertionError("proof phases do not match the benchmark measurement contract")
    expected_phase_coordinates = benchmark_phase_coordinates(
        manifest,
        profile,
        generation_round_start=generation_round,
        warmups=warmups,
        repetitions=repetitions,
    )
    if measurement.get("phase_coordinates") != list(expected_phase_coordinates):
        raise AssertionError("benchmark measurement phase coordinates are not caller-reconstructed")
    memory_peaks = []
    final_workers = []
    initialized = _require_list(
        artifact.get("initialized_worker_proof"),
        label="initialized worker proof",
    )
    _validate_worker_sampler_evidence(
        initialized,
        expected_installation=expected_sampler_installation,
        expected_seed_batches=(),
        expected_request_generations=(),
        require_generation_observations=False,
    )
    for sample_index, (phase, phase_coordinate) in enumerate(
        zip(phases, expected_phase_coordinates, strict=True)
    ):
        if not isinstance(phase, dict) or phase.get("proof_collected") is not True:
            raise AssertionError("proof phase lacks production proof collection")
        phase_name = expected_names[sample_index]
        graph_full_tokens, graph_observations = _validate_cudagraph_phase_evidence(
            phase,
            maximum_wave_size=profile.global_wave_size,
        )
        waves = build_request_waves(
            request_count=len(manifest.requests),
            global_batch_size=profile.global_wave_size,
            replica_count=1,
        )
        expected_wave_phases = {f"{phase_name}.wave-{wave.wave_index:03d}" for wave in waves}
        if any(observation.get("phase") not in expected_wave_phases for observation in graph_observations):
            raise AssertionError("CUDA graph observations must belong to one exact physical wave")
        retained_waves = _require_list(phase.get("wave_proofs"), label="physical wave proofs")
        if len(retained_waves) != len(waves):
            raise AssertionError("physical proof wave count does not match the exact workload")
        full_summaries = []
        for wave, retained in zip(waves, retained_waves, strict=True):
            expected_phase = f"{phase_name}.wave-{wave.wave_index:03d}"
            expected_fields = {
                "wave_index": wave.wave_index,
                "start": wave.start,
                "stop": wave.stop,
                "request_count": wave.request_count,
                "generation_round": phase_coordinate["generation_round"],
                "call_index": phase_coordinate["global_call_index_start"] + wave.wave_index,
            }
            _validate_exact_wave_coordinates(retained, expected=expected_fields)
            full_decode = _require_dict(
                retained.get("full_decode_proof"),
                label="wave FULL decode proof",
            )
            recomputed_full_decode = full_decode_proof_summary(
                graph_observations,
                phase=expected_phase,
                batch_size=wave.request_count,
                max_new_tokens=manifest.max_new_tokens,
            )
            if full_decode != recomputed_full_decode:
                raise AssertionError("wave FULL decode proof does not match complete raw CUDA observations")
            _validate_full_decode_summary_values(
                full_decode,
                phase=expected_phase,
                batch_size=wave.request_count,
                max_new_tokens=manifest.max_new_tokens,
            )
            scheduler = _require_dict(
                retained.get("scheduler_capacity_proof"),
                label="wave scheduler proof",
            )
            scheduler_observations = _require_list(
                retained.get("scheduler_observations"),
                label="raw wave scheduler observations",
            )
            if not scheduler_observations or any(
                not isinstance(observation, dict) or observation.get("phase") != expected_phase
                for observation in scheduler_observations
            ):
                raise AssertionError("raw scheduler observations must belong exclusively to the physical wave")
            recomputed_scheduler = scheduler_capacity_proof_summary(
                scheduler_observations,
                phase=expected_phase,
                global_wave_size=wave.request_count,
                engine_request_count=wave.request_count,
                max_num_seqs=profile.resolved_max_num_seqs,
            )
            if scheduler != recomputed_scheduler:
                raise AssertionError("scheduler proof does not match raw wave observations")
            validate_scheduler_capacity_proof(recomputed_scheduler)
            full_summaries.append(full_decode)
        if phase.get("wave_execution") != wave_execution_summary(retained_waves):
            raise AssertionError("physical wave execution summary is inconsistent")
        phase_full = _require_dict(phase.get("full_decode_proof"), label="phase FULL decode proof")
        expected_tokens = sum(item["expected_decode_tokens"] for item in full_summaries)
        full_tokens = sum(item["full_decode_tokens"] for item in full_summaries)
        if (
            phase_full.get("phase") != phase_name
            or phase_full.get("wave_count") != len(waves)
            or phase_full.get("expected_decode_tokens") != expected_tokens
            or phase_full.get("full_decode_tokens") != full_tokens
            or phase_full.get("waves") != full_summaries
            or phase_full.get("passed") is not True
        ):
            raise AssertionError("phase FULL decode aggregate is inconsistent")
        coverage = full_tokens / expected_tokens if expected_tokens else 1.0
        if not math.isclose(float(phase_full.get("coverage_fraction", -1.0)), coverage):
            raise AssertionError("phase FULL decode aggregate coverage is inconsistent")
        if graph_full_tokens != full_tokens:
            raise AssertionError("raw CUDA graph aggregate does not match wave FULL decode totals")

        executions = build_wave_execution_records(
            manifest,
            global_wave_size=profile.global_wave_size,
            generation_round=phase_coordinate["generation_round"],
            call_index_start=phase_coordinate["global_call_index_start"],
            global_request_index_start=phase_coordinate["global_request_index_start"],
        )
        _validate_request_execution_rows(phase.get("request_executions"), expected=executions)
        sidecar_counts = _validate_full_output_sidecar(
            _require_dict(phase.get("full_output_artifact"), label="full-output sidecar metadata"),
            artifact_path=artifact_path,
            phase=phase_name,
            manifest=manifest,
            expected_executions=executions,
            output_summaries=phase.get("outputs"),
        )
        progress_contract = artifact["benchmark_contract"].get("exact_generation_progress")
        if progress_contract is None:
            if phase.get("exact_generation_progress") is not None:
                raise AssertionError("proof retained exact-generation evidence without a linked contract")
        else:
            recomputed_progress = exact_generation_progress_evidence(
                manifest,
                sidecar_counts=sidecar_counts,
                full_decode_summaries=full_summaries,
            )
            if phase.get("exact_generation_progress") != recomputed_progress:
                raise AssertionError("exact-generation progress evidence failed raw recomputation")
        memory_peaks.append(
            _validate_phase_sample(
                phase,
                manifest=manifest,
                sample_index=sample_index,
            )
        )
        workers = _require_list(phase.get("worker_proof"), label="phase worker proof")
        hardware = _require_dict(
            artifact.get("gpu_hardware_provenance"),
            label="GPU hardware provenance",
        )
        _validate_worker_gpu_bindings(
            workers,
            hardware=hardware,
            expected_worker_count=profile.tensor_parallel_size,
            expected_engine_seed=manifest.seed,
            expected_physical_indices=tuple(range(profile.tensor_parallel_size)),
        )
        _validate_worker_rank_continuity(workers, initialized)
        _validate_worker_sampler_evidence(
            workers,
            expected_installation=expected_sampler_installation,
            expected_seed_batches=_execution_seed_batches(executions),
            expected_request_generations=_sidecar_request_generations_for_executions(
                sidecar_counts,
                executions,
            ),
            require_generation_observations=True,
        )
        _validate_fir_route_evidence(workers, manifest=manifest)
        if profile.shared_prefix_state_reuse:
            retained_prefix = _require_dict(
                phase.get("shared_prefix_state_reuse"),
                label="shared-prefix state reuse evidence",
            )
            resolved = _require_dict(artifact.get("resolved_config"), label="resolved vLLM config")
            cache = _require_dict(resolved.get("cache"), label="resolved vLLM cache config")
            recomputed_prefix = shared_prefix_state_reuse_evidence(
                manifest,
                cached_tokens=_require_list(
                    retained_prefix.get("cached_tokens_by_request"),
                    label="cached-token prefix evidence",
                ),
                worker_proof=workers,
                expected_worker_clone_counts=tuple(len(manifest.requests) - 1 for _ in workers),
                cache_block_size=int(cache.get("block_size", 0)),
            )
            recomputed_prefix = {
                **recomputed_prefix,
                "phase_prefix_cache_reset": True,
            }
            if phase.get("prefix_cache_reset") is not True or retained_prefix != recomputed_prefix:
                raise AssertionError("shared-prefix physical reuse evidence failed recomputation")
        elif phase.get("shared_prefix_state_reuse") is not None or phase.get("prefix_cache_reset") is not False:
            raise AssertionError("non-prefix proof retained unexpected prefix-clone evidence")
        final_workers = workers
    hardware = _require_dict(
        artifact.get("gpu_hardware_provenance"),
        label="GPU hardware provenance",
    )
    _validate_worker_gpu_bindings(
        initialized,
        hardware=hardware,
        expected_worker_count=profile.tensor_parallel_size,
        expected_engine_seed=manifest.seed,
        expected_physical_indices=tuple(range(profile.tensor_parallel_size)),
    )
    initialized_by_rank = {worker["rank"]: worker for worker in initialized}
    final_by_rank = {worker["rank"]: worker for worker in final_workers}
    if initialized_by_rank.keys() != final_by_rank.keys():
        raise AssertionError("initialized and final worker ranks differ")
    for rank in sorted(initialized_by_rank):
        initialized_worker = initialized_by_rank[rank]
        final_worker = final_by_rank[rank]
        if (
            initialized_worker.get("device_uuid"),
            initialized_worker.get("pci_bus_id"),
        ) != (
            final_worker.get("device_uuid"),
            final_worker.get("pci_bus_id"),
        ):
            raise AssertionError("worker rank moved to a different physical GPU")
        validate_compilation_proof(
            _require_dict(initialized_worker.get("compilation"), label="initialized compilation proof"),
            _require_dict(final_worker.get("compilation"), label="final compilation proof"),
        )
    return tuple(max(values) for values in zip(*memory_peaks, strict=True)), final_workers


def _expected_dp2_executions(
    manifest: WorkloadManifest,
    *,
    profile: Evo2VllmProfile,
    generation_round: int,
    call_index_start: int,
    global_index_start: int,
) -> tuple[RequestExecutionRecord, ...]:
    records = []
    for wave in build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=profile.global_wave_size,
        replica_count=profile.replica_count,
    ):
        for shard in wave.shards:
            records.extend(
                build_request_execution_records(
                    manifest.request_slice(shard.start, shard.stop),
                    global_request_offset=global_index_start + shard.start,
                    dp_rank=shard.replica_index,
                    dp_size=profile.replica_count,
                    generation_round=generation_round,
                    call_index=call_index_start + wave.wave_index,
                )
            )
    return tuple(records)


def _validate_dp2_phase_evidence(
    artifact: dict[str, Any],
    *,
    artifact_path: Path,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    generation_round: int,
    expected_sampler_installation: Mapping[str, Any],
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    measurement = _require_dict(
        artifact["benchmark_contract"].get("measurement"),
        label="benchmark measurement contract",
    )
    warmups = measurement.get("warmups")
    repetitions = measurement.get("repetitions")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (warmups, repetitions))
        or repetitions == 0
    ):
        raise AssertionError("benchmark measurement counts are malformed")
    expected_names = [
        "cold-generation",
        *(f"warmup-{index}" for index in range(warmups)),
        *(f"steady-{index}" for index in range(repetitions)),
    ]
    phases = _require_list(artifact.get("phases"), label="proof phases")
    if [phase.get("phase") for phase in phases] != expected_names:
        raise AssertionError("DP2 proof phases do not match the measurement contract")

    call_index = generation_round * len(
        build_request_waves(
            request_count=len(manifest.requests),
            global_batch_size=profile.global_wave_size,
            replica_count=profile.replica_count,
        )
    )
    global_index = 0
    memory_peaks = []
    final_engines = []
    hardware = _require_dict(
        artifact.get("gpu_hardware_provenance"),
        label="GPU hardware provenance",
    )
    initialized_engines = _require_list(
        artifact.get("initialized_engine_proofs"),
        label="initialized DP2 engine proofs",
    )
    if len(initialized_engines) != profile.replica_count:
        raise AssertionError("initialized DP2 engine count does not match the topology")
    for initialized_engine in initialized_engines:
        _validate_worker_sampler_evidence(
            _require_list(
                initialized_engine.get("worker_proof"),
                label="initialized DP2 sampler worker proof",
            ),
            expected_installation=expected_sampler_installation,
            expected_seed_batches=(),
            expected_request_generations=(),
            require_generation_observations=False,
        )
    for sample_index, phase in enumerate(phases):
        if not isinstance(phase, dict) or phase.get("proof_collected") is not True:
            raise AssertionError("DP2 proof phase lacks production proof collection")
        phase_name = expected_names[sample_index]
        waves = build_request_waves(
            request_count=len(manifest.requests),
            global_batch_size=profile.global_wave_size,
            replica_count=profile.replica_count,
        )
        retained_waves = _require_list(phase.get("waves"), label="DP2 physical wave proofs")
        if len(retained_waves) != len(waves):
            raise AssertionError("DP2 proof wave count does not match the exact workload")
        phase_workers = []
        phase_full_summaries = []
        pending_sampler_validations = []
        for wave, retained in zip(waves, retained_waves, strict=True):
            wave_phase = f"{phase_name}.wave-{wave.wave_index:03d}"
            expected_wave = {
                "wave_index": wave.wave_index,
                "phase": wave_phase,
                "start": wave.start,
                "stop": wave.stop,
                "request_count": wave.request_count,
                "generation_round": generation_round,
                "call_index": call_index + wave.wave_index,
            }
            _validate_exact_wave_coordinates(retained, expected=expected_wave)
            engines = _require_list(retained.get("engines"), label="DP2 engine wave proofs")
            if len(engines) != len(wave.shards):
                raise AssertionError("DP2 engine proof count does not match active replicas")
            for shard, engine in zip(wave.shards, engines, strict=True):
                _require_builtin_integer(engine.get("dp_rank"), label="DP2 engine dp_rank")
                _require_builtin_integer(
                    engine.get("request_count"),
                    label="DP2 engine request_count",
                    minimum=1,
                )
                if (
                    engine.get("dp_rank") != shard.replica_index
                    or engine.get("request_count") != shard.request_count
                    or engine.get("phase") != wave_phase
                ):
                    raise AssertionError("DP2 engine ownership does not match its exact shard")
                observations = _require_list(
                    engine.get("cudagraph_observations"),
                    label="DP2 raw CUDA graph observations",
                )
                if engine.get("cudagraph_summary") != summarize_cudagraph_observations(tuple(observations)):
                    raise AssertionError("DP2 CUDA graph aggregate does not match raw observations")
                recomputed_full = full_decode_proof_summary(
                    observations,
                    phase=wave_phase,
                    batch_size=shard.request_count,
                    max_new_tokens=manifest.max_new_tokens,
                )
                if engine.get("full_decode_proof") != recomputed_full:
                    raise AssertionError("DP2 FULL decode proof does not match raw observations")
                _validate_full_decode_summary_values(
                    recomputed_full,
                    phase=wave_phase,
                    batch_size=shard.request_count,
                    max_new_tokens=manifest.max_new_tokens,
                )
                phase_full_summaries.append(recomputed_full)
                scheduler_observations = _require_list(
                    engine.get("scheduler_observations"),
                    label="DP2 raw scheduler observations",
                )
                if not scheduler_observations or any(
                    not isinstance(observation, dict) or observation.get("phase") != wave_phase
                    for observation in scheduler_observations
                ):
                    raise AssertionError("raw DP2 scheduler observations must belong exclusively to the physical wave")
                recomputed_scheduler = scheduler_capacity_proof_summary(
                    scheduler_observations,
                    phase=wave_phase,
                    global_wave_size=wave.request_count,
                    engine_request_count=shard.request_count,
                    max_num_seqs=profile.resolved_max_num_seqs,
                )
                if engine.get("scheduler_capacity_proof") != recomputed_scheduler:
                    raise AssertionError("DP2 scheduler proof does not match raw observations")
                validate_scheduler_capacity_proof(recomputed_scheduler)
                try:
                    validate_resolved_profile(
                        profile,
                        _require_dict(engine.get("resolved_config"), label="DP2 resolved config"),
                    )
                except (AssertionError, KeyError, TypeError, ValueError) as error:
                    raise AssertionError("DP2 engine resolved config drifted") from error
                workers = _require_list(engine.get("worker_proof"), label="DP2 inner worker proof")
                _validate_worker_gpu_bindings(
                    workers,
                    hardware=hardware,
                    expected_worker_count=profile.tensor_parallel_size,
                    expected_engine_seed=manifest.seed,
                    expected_physical_indices=(shard.replica_index,),
                )
                _validate_worker_rank_continuity(
                    workers,
                    _require_list(
                        initialized_engines[shard.replica_index].get("worker_proof"),
                        label="initialized DP2 inner worker proof",
                    ),
                )
                shard_executions = build_request_execution_records(
                    manifest.request_slice(shard.start, shard.stop),
                    global_request_offset=global_index + shard.start,
                    dp_rank=shard.replica_index,
                    dp_size=profile.replica_count,
                    generation_round=generation_round,
                    call_index=call_index + wave.wave_index,
                )
                pending_sampler_validations.append((workers, shard_executions))
                _validate_fir_route_evidence(workers, manifest=manifest.request_slice(wave.start, wave.stop))
                phase_workers.extend(workers)
            identities = {
                (worker.get("device_uuid"), worker.get("pci_bus_id")) for worker in phase_workers[-len(wave.shards) :]
            }
            if len(identities) != len(wave.shards):
                raise AssertionError("DP2 replicas are not bound to distinct physical GPUs")
            if profile.shared_prefix_state_reuse:
                retained_prefix = _require_dict(
                    retained.get("shared_prefix_state_reuse"),
                    label="DP2 shared-prefix evidence",
                )
                workers = [
                    worker
                    for engine in engines
                    for worker in _require_list(engine.get("worker_proof"), label="DP2 worker proof")
                ]
                cache = _require_dict(engines[0]["resolved_config"].get("cache"), label="DP2 cache config")
                recomputed_prefix = shared_prefix_state_reuse_evidence(
                    manifest.request_slice(wave.start, wave.stop),
                    cached_tokens=_require_list(
                        retained_prefix.get("cached_tokens_by_request"),
                        label="DP2 cached-token evidence",
                    ),
                    request_replica_ranks=tuple(
                        shard.replica_index for shard in wave.shards for _ in range(shard.request_count)
                    ),
                    worker_proof=workers,
                    expected_worker_clone_counts=tuple(
                        shard.request_count - int(wave.wave_index == 0) for shard in wave.shards
                    ),
                    cache_block_size=int(cache.get("block_size", 0)),
                    expected_cache_misses=len(wave.shards) if wave.wave_index == 0 else 0,
                )
                recomputed_prefix = {
                    **recomputed_prefix,
                    "phase_prefix_cache_reset_before_first_wave": True,
                }
                if retained_prefix != recomputed_prefix:
                    raise AssertionError("DP2 shared-prefix evidence failed physical recomputation")
            elif retained.get("shared_prefix_state_reuse") is not None:
                raise AssertionError("non-prefix DP2 wave retained unexpected prefix evidence")
            final_engines = engines
        if phase.get("wave_execution") != wave_execution_summary(retained_waves):
            raise AssertionError("DP2 physical wave execution summary is inconsistent")
        if phase.get("generation_call_s") != [wave["generation_s"] for wave in retained_waves]:
            raise AssertionError("DP2 generation calls do not match retained physical waves")
        executions = _expected_dp2_executions(
            manifest,
            profile=profile,
            generation_round=generation_round,
            call_index_start=call_index,
            global_index_start=global_index,
        )
        _validate_request_execution_rows(phase.get("request_executions"), expected=executions)
        sidecar_counts = _validate_full_output_sidecar(
            _require_dict(phase.get("full_output_artifact"), label="DP2 full-output sidecar"),
            artifact_path=artifact_path,
            phase=phase_name,
            manifest=manifest,
            expected_executions=executions,
            output_summaries=phase.get("outputs"),
        )
        sampler_joined_keys = set()
        for workers, shard_executions in pending_sampler_validations:
            request_generations = _sidecar_request_generations_for_executions(
                sidecar_counts,
                shard_executions,
            )
            joined_keys = {(row["request_id"], row["seed"]) for row in request_generations}
            if sampler_joined_keys & joined_keys:
                raise AssertionError("DP2 sampler validation joined one caller request more than once")
            sampler_joined_keys.update(joined_keys)
            _validate_worker_sampler_evidence(
                workers,
                expected_installation=expected_sampler_installation,
                expected_seed_batches=_execution_seed_batches(shard_executions),
                expected_request_generations=request_generations,
                require_generation_observations=True,
            )
        sidecar_keys = {
            (row["request_id"], row["seed"])
            for row in _require_list(
                sidecar_counts.get("request_generations"),
                label="DP2 caller-reopened sampler request generations",
            )
        }
        if sampler_joined_keys != sidecar_keys:
            raise AssertionError("DP2 sampler validations do not cover the complete caller sidecar")
        progress_contract = artifact["benchmark_contract"].get("exact_generation_progress")
        if progress_contract is None:
            if phase.get("exact_generation_progress") is not None:
                raise AssertionError("DP2 proof retained exact-generation evidence without a linked contract")
        else:
            recomputed_progress = exact_generation_progress_evidence(
                manifest,
                sidecar_counts=sidecar_counts,
                full_decode_summaries=phase_full_summaries,
            )
            if phase.get("exact_generation_progress") != recomputed_progress:
                raise AssertionError("DP2 exact-generation progress evidence failed raw recomputation")
        memory_peaks.append(_validate_phase_sample(phase, manifest=manifest, sample_index=sample_index))
        if phase.get("prefix_cache_reset") is not profile.shared_prefix_state_reuse:
            raise AssertionError("DP2 phase prefix-cache reset contract drifted")
    resolved_configs = _require_list(artifact.get("resolved_configs"), label="DP2 resolved configs")
    if len(resolved_configs) != profile.replica_count:
        raise AssertionError("DP2 resolved-config count does not match the topology")
    final_by_rank = {engine["dp_rank"]: engine for engine in final_engines}
    final_workers = []
    initialized_identities = set()
    for dp_rank, (initialized, resolved) in enumerate(zip(initialized_engines, resolved_configs, strict=True)):
        if initialized.get("resolved_config") != resolved:
            raise AssertionError("DP2 initialized and retained resolved configs differ")
        try:
            validate_resolved_profile(profile, _require_dict(resolved, label="DP2 resolved config"))
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            raise AssertionError("DP2 retained resolved config drifted") from error
        initialized_workers = _require_list(
            initialized.get("worker_proof"),
            label="initialized DP2 inner worker proof",
        )
        final = final_by_rank.get(dp_rank)
        if final is None:
            raise AssertionError("final DP2 wave omitted one replica")
        retained_final_workers = _require_list(final.get("worker_proof"), label="final DP2 worker proof")
        _validate_worker_gpu_bindings(
            initialized_workers,
            hardware=hardware,
            expected_worker_count=profile.tensor_parallel_size,
            expected_engine_seed=manifest.seed,
            expected_physical_indices=(dp_rank,),
        )
        initial_identity = (
            initialized_workers[0].get("device_uuid"),
            initialized_workers[0].get("pci_bus_id"),
        )
        final_identity = (
            retained_final_workers[0].get("device_uuid"),
            retained_final_workers[0].get("pci_bus_id"),
        )
        if initial_identity != final_identity or initial_identity in initialized_identities:
            raise AssertionError("DP2 replica physical GPU binding changed or overlaps")
        initialized_identities.add(initial_identity)
        validate_compilation_proof(
            _require_dict(
                initialized_workers[0].get("compilation"),
                label="initialized DP2 compilation proof",
            ),
            _require_dict(
                retained_final_workers[0].get("compilation"),
                label="final DP2 compilation proof",
            ),
        )
        final_workers.extend(retained_final_workers)
    return tuple(max(values) for values in zip(*memory_peaks, strict=True)), final_workers


def _validate_linked_proof_evidence(
    artifact: dict[str, Any],
    *,
    artifact_path: Path,
    expected_contract: dict[str, Any],
    caller_coordinates: CallerCoordinateContract,
    require_memory_headroom: bool,
) -> dict[str, Any]:
    try:
        retained_manifest = WorkloadManifest.from_dict(
            _require_dict(artifact.get("manifest"), label="proof manifest")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AssertionError("proof manifest is malformed") from error
    manifest = caller_coordinates.manifest
    if retained_manifest != manifest:
        raise AssertionError("proof manifest does not match the external caller coordinate contract")
    if (
        artifact.get("manifest_sha256") != manifest.sha256
        or expected_contract.get("manifest_sha256") != manifest.sha256
    ):
        raise AssertionError("proof manifest SHA256 does not match the benchmark contract")
    progress_contract = expected_contract.get("exact_generation_progress")
    if progress_contract is not None:
        validate_exact_generation_progress_contract(
            progress_contract,
            manifest=manifest,
            profile=caller_coordinates.profile,
            generation_round=caller_coordinates.generation_round,
        )
    profile_data = _require_dict(artifact.get("profile"), label="proof profile")
    try:
        profile = Evo2VllmProfile(**profile_data)
    except (TypeError, ValueError) as error:
        raise AssertionError("proof profile is malformed") from error
    if profile.proof is not True:
        raise AssertionError("linked proof profile did not enable proof instrumentation")
    if (
        profile.topology != caller_coordinates.topology
        or profile.global_wave_size != caller_coordinates.global_wave_size
        or profile.replica_count != caller_coordinates.replica_count
    ):
        raise AssertionError("proof topology does not match the external caller coordinate contract")
    profile_contract = asdict(profile)
    profile_contract.pop("proof")
    caller_profile_contract = caller_coordinates.profile_contract()
    if profile_contract != caller_profile_contract or expected_contract.get("profile") != caller_profile_contract:
        raise AssertionError("proof profile does not match the external caller coordinate contract")
    seed_stream = _require_dict(expected_contract.get("seed_stream"), label="seed-stream contract")
    expected_seed_stream = caller_coordinates.seed_stream_contract()
    if seed_stream != expected_seed_stream:
        raise AssertionError("benchmark seed stream does not match the external caller coordinate contract")
    generation_round = caller_coordinates.generation_round
    invocation = _require_dict(artifact.get("invocation"), label="proof invocation")
    output_path = Path(str(invocation.get("output_artifact_path", ""))).expanduser().resolve()
    if output_path != artifact_path:
        raise AssertionError("proof invocation output path does not match the linked artifact")
    parsed_args = _require_dict(invocation.get("parsed_args"), label="proof parsed arguments")
    if parsed_args.get("generation_round") != generation_round:
        raise AssertionError("proof parsed generation round does not match the seed-stream contract")
    if (
        artifact.get("topology") != caller_coordinates.topology
        or expected_contract.get("topology") != caller_coordinates.topology
    ):
        raise AssertionError("proof topology does not match its profile and benchmark contract")
    expected_runtime = _require_dict(
        expected_contract.get("runtime_attestation"),
        label="benchmark runtime attestation",
    )
    expected_sampler_installation = _require_dict(
        expected_runtime.get("sampler"),
        label="benchmark sampler installation identity",
    )
    backend = artifact.get("backend")
    if backend == "vllm" and profile.topology == "tp2":
        try:
            validate_resolved_profile(
                profile,
                _require_dict(artifact.get("resolved_config"), label="resolved vLLM config"),
            )
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            raise AssertionError("resolved vLLM config does not match the proof profile") from error
        phase_peak, final_workers = _validate_direct_phase_evidence(
            artifact,
            artifact_path=artifact_path,
            manifest=manifest,
            profile=profile,
            generation_round=generation_round,
            expected_sampler_installation=expected_sampler_installation,
        )
    elif backend == "nemo-rl-vllm" and profile.topology == "dp2":
        phase_peak, final_workers = _validate_dp2_phase_evidence(
            artifact,
            artifact_path=artifact_path,
            manifest=manifest,
            profile=profile,
            generation_round=generation_round,
            expected_sampler_installation=expected_sampler_installation,
        )
    else:
        raise AssertionError("linked proof backend/topology schema is unsupported")
    if progress_contract is None:
        if artifact.get("exact_generation_progress") is not None:
            raise AssertionError("proof retained an exact-generation aggregate without a linked contract")
    else:
        _, recomputed_progress = attach_exact_generation_progress_evidence(
            _require_list(artifact.get("phases"), label="proof phases"),
            manifest=manifest,
            enabled=True,
            proof_collected=True,
            topology=profile.topology,
            linked_proof_artifact=None,
        )
        if artifact.get("exact_generation_progress") != recomputed_progress:
            raise AssertionError("exact-generation aggregate failed phase recomputation")
    mbs1_contract = expected_contract.get("mbs1_exact1k")
    recomputed_phases, recomputed_mbs1 = attach_mbs1_exact1k_evidence(
        _require_list(artifact.get("phases"), label="proof phases"),
        manifest=manifest,
        profile=profile,
        generation_round_start=generation_round,
        warmups=_require_dict(expected_contract.get("measurement"), label="measurement")["warmups"],
        repetitions=_require_dict(expected_contract.get("measurement"), label="measurement")["repetitions"],
        enabled=mbs1_contract is not None,
        proof_collected=True,
        linked_proof_artifact=None,
    )
    if artifact.get("phases") != recomputed_phases or artifact.get("mbs1_exact1k") != recomputed_mbs1:
        raise AssertionError("MBS=1 exact-1k proof failed caller-side recomputation")
    hardware = _require_dict(
        artifact.get("gpu_hardware_provenance"),
        label="GPU hardware provenance",
    )
    retained_sources = _require_dict(
        artifact.get("source_provenance"),
        label="proof source provenance",
    )
    if backend == "vllm":
        sources = {"bionemo": retained_sources}
    else:
        expected_source_names = set(_require_dict(expected_runtime.get("sources"), label="runtime source identities"))
        if set(retained_sources) != expected_source_names:
            raise AssertionError("DP2 proof source provenance does not match the runtime contract")
        sources = {
            name: _require_dict(source, label=f"proof source provenance {name!r}")
            for name, source in retained_sources.items()
        }
    try:
        recomputed_runtime = runtime_attestation_contract(
            checkpoint=_require_dict(
                artifact.get("checkpoint_provenance"),
                label="proof checkpoint provenance",
            ),
            sources=sources,
            vllm_installation=_require_dict(
                artifact.get("vllm_installation_provenance"),
                label="proof vLLM installation provenance",
            ),
            sampler_installation=_require_dict(
                artifact.get("sampler_installation_provenance"),
                label="proof sampler installation provenance",
            ),
            gpu_hardware=hardware,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise AssertionError("proof runtime attestation could not be recomputed") from error
    if recomputed_runtime != expected_runtime:
        raise AssertionError("proof runtime provenance does not match the linked speed-run contract")
    init_peak = _require_dict(artifact.get("timing"), label="proof timing").get("engine_init_peak_device_memory_bytes")
    if (
        not isinstance(init_peak, list)
        or len(init_peak) != len(phase_peak)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in init_peak)
    ):
        raise AssertionError("engine initialization peak-memory evidence is malformed")
    peak = tuple(max(initialized, phase_value) for initialized, phase_value in zip(init_peak, phase_peak, strict=True))
    try:
        recomputed_memory = gpu_memory_headroom_evidence(
            hardware,
            peak_device_memory_bytes=peak,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise AssertionError("proof GPU memory headroom did not pass recomputation") from error
    if artifact.get("gpu_memory_headroom") != recomputed_memory:
        raise AssertionError("retained GPU memory headroom does not match recomputed peaks")
    if require_memory_headroom and recomputed_memory.get("passed") is not True:
        raise AssertionError("linked proof lacks passed GPU memory headroom")
    proof_status = _require_dict(artifact.get("proof_status"), label="proof status")
    if (
        proof_status.get("phase_count") != len(artifact["phases"])
        or proof_status.get("full_decode_passed") is not True
        or proof_status.get("compilation_stable") is not True
    ):
        raise AssertionError("top-level proof status does not match recomputed phase evidence")
    canonical_identity = validate_canonical_identity_proof_evidence(
        artifact,
        manifest=manifest,
        profile=profile,
        expected_contract=expected_contract,
    )
    common_prefix_identity = validate_common_prefix_identity_proof_evidence(
        artifact,
        manifest=manifest,
        profile=profile,
        expected_contract=expected_contract,
    )
    return {
        "manifest_sha256": manifest.sha256,
        "caller_coordinate_binding": caller_coordinates.summary(),
        "phase_count": len(artifact["phases"]),
        "final_worker_count": len(final_workers),
        "runtime_attestation": recomputed_runtime,
        "gpu_memory_headroom": recomputed_memory,
        "canonical_identity": canonical_identity,
        "common_prefix_identity": common_prefix_identity,
        "passed": True,
    }


def _file_records(root: Path, paths: Any) -> list[dict[str, Any]]:
    records = []
    for path in sorted({Path(path).resolve() for path in paths}):
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"provenance path escapes root {root}: {path}") from error
        snapshot = read_file_digest_snapshot(path, label=f"provenance file {relative.as_posix()}")
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": snapshot.size_bytes,
                "sha256": snapshot.sha256,
            }
        )
    return records


def _records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def package_installation_provenance(
    package_root: str | Path,
    *,
    distribution_name: str,
    distribution_version: str,
    metadata_paths: Sequence[str | Path] = (),
    require_binary: bool = False,
) -> dict[str, Any]:
    """Hash one installed Python package, including compiled extensions and metadata."""
    root = Path(package_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"installed package root is missing: {root}")

    def is_durable(path: Path) -> bool:
        return "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}

    package_paths = tuple(path for path in root.rglob("*") if path.is_file() and is_durable(path))
    source_suffixes = {".py", ".pyi", ".pyx", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    binary_suffixes = {".so", ".pyd", ".dll", ".dylib", ".cubin", ".fatbin"}
    source_paths = tuple(path for path in package_paths if path.suffix.lower() in source_suffixes)
    binary_paths = tuple(path for path in package_paths if path.suffix.lower() in binary_suffixes)
    if not source_paths:
        raise RuntimeError(f"installed {distribution_name} package contains no source implementation files")
    if require_binary and not binary_paths:
        raise RuntimeError(f"installed {distribution_name} package contains no compiled binary files")

    package_records = _file_records(root, package_paths)
    source_records = _file_records(root, source_paths)
    binary_records = _file_records(root, binary_paths)
    metadata_records = []
    for metadata_path in sorted({Path(path).expanduser().resolve() for path in metadata_paths}):
        snapshot = read_file_digest_snapshot(metadata_path, label=f"installed package metadata {metadata_path}")
        metadata_records.append(
            {
                "path": str(metadata_path),
                "size_bytes": snapshot.size_bytes,
                "sha256": snapshot.sha256,
            }
        )
    installation_identity = {
        "distribution_name": distribution_name,
        "distribution_version": distribution_version,
        "package_root": str(root),
        "package_files": package_records,
        "metadata_files": metadata_records,
    }
    installation_sha256 = hashlib.sha256(
        json.dumps(installation_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        **installation_identity,
        "installation_sha256": installation_sha256,
        "package_file_count": len(package_records),
        "package_bytes": sum(record["size_bytes"] for record in package_records),
        "source_file_count": len(source_records),
        "binary_file_count": len(binary_records),
        "metadata_file_count": len(metadata_records),
        "source_files": source_records,
        "binary_files": binary_records,
    }


def vllm_installation_provenance() -> dict[str, Any]:
    """Discover and hash the exact installed vLLM implementation and binaries."""
    spec = find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("installed vLLM package could not be resolved")
    package_locations = tuple(Path(path).expanduser().resolve() for path in spec.submodule_search_locations)
    if len(package_locations) != 1:
        raise RuntimeError(f"vLLM must resolve to one package root, got {package_locations}")

    installed_distribution = distribution("vllm")
    distribution_files = installed_distribution.files
    if distribution_files is None:
        raise RuntimeError("installed vLLM distribution exposes no file manifest")
    retained_metadata_names = {"INSTALLER", "METADATA", "RECORD", "WHEEL", "direct_url.json"}
    metadata_paths = []
    for entry in distribution_files:
        entry_path = Path(str(entry))
        if not any(part.endswith(".dist-info") for part in entry_path.parts):
            continue
        if entry_path.name in retained_metadata_names:
            located = Path(installed_distribution.locate_file(entry)).expanduser().resolve()
            if located.is_file():
                metadata_paths.append(located)
    if not metadata_paths:
        raise RuntimeError("installed vLLM distribution metadata could not be resolved")
    return package_installation_provenance(
        package_locations[0],
        distribution_name="vllm",
        distribution_version=installed_distribution.version,
        metadata_paths=tuple(metadata_paths),
        require_binary=True,
    )


def runtime_attestation_contract(
    *,
    checkpoint: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    vllm_installation: dict[str, Any],
    sampler_installation: dict[str, Any],
    gpu_hardware: dict[str, Any],
) -> dict[str, Any]:
    """Reduce full provenance to the immutable identities linked across benchmark lanes."""
    from bionemo.evo2.vllm.sampler import sampler_installation_contract

    source_contract = {}
    for name, source in sorted(sources.items()):
        if source.get("git_dirty") is not False:
            raise RuntimeError(f"runtime attestation source {name!r} is dirty")
        source_contract[name] = {
            "git_head": source["git_head"],
            "source_tree_sha256": source["source_tree_sha256"],
        }
    devices = gpu_hardware.get("devices")
    if gpu_hardware.get("passed") is not True or not isinstance(devices, list) or not devices:
        raise ValueError("runtime attestation requires exact GPU hardware provenance")
    if gpu_hardware.get("cuda_visible_devices") != gpu_hardware.get("expected_cuda_visible_devices"):
        raise ValueError("runtime attestation GPU visibility does not match its frozen assignment")
    return {
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "sources": source_contract,
        "vllm": {
            "distribution_version": vllm_installation["distribution_version"],
            "installation_sha256": vllm_installation["installation_sha256"],
        },
        "sampler": sampler_installation_contract(sampler_installation),
        "gpu": {
            "schema_version": gpu_hardware["schema_version"],
            "driver_version": gpu_hardware["driver_version"],
            "nvml_cuda_driver_version_integer": gpu_hardware["nvml_cuda_driver_version_integer"],
            "cuda_visible_devices": gpu_hardware.get("cuda_visible_devices"),
            "expected_cuda_visible_devices": gpu_hardware["expected_cuda_visible_devices"],
            "api_versions": gpu_hardware["api_versions"],
            "expected_assignments": gpu_hardware["expected_assignments"],
            "devices": [
                {
                    "logical_device_index": device["logical_device_index"],
                    "visible_device_selector": device["visible_device_selector"],
                    "physical_index": device["physical_index"],
                    "uuid": device["uuid"],
                    "pci_bus_id": device["pci_bus_id"],
                    "name": device["name"],
                    "torch_uuid": device["torch_uuid"],
                    "torch_name": device["torch_name"],
                    "physical_total_memory_bytes": device["memory"]["nvml"]["physical_total_bytes"],
                    "system_reserved_memory_bytes": device["memory"]["nvml"]["system_reserved_bytes"],
                    "cuda_usable_total_memory_bytes": device["memory"]["cuda"]["usable_total_bytes"],
                    "torch_properties_total_memory_bytes": device["memory"]["torch_properties_total_bytes"],
                }
                for device in devices
            ],
        },
    }


def checkpoint_provenance(checkpoint: str | Path) -> dict[str, Any]:
    """Hash the actual indexed checkpoint shards and every durable checkpoint file."""
    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {root}")
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    manifest_path = root / "manifest.json"
    for required in (config_path, index_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"checkpoint provenance requires {required}")

    config_snapshot = read_json_snapshot(config_path, label="checkpoint config")
    index_snapshot = read_json_snapshot(index_path, label="checkpoint index")
    manifest_snapshot = read_json_snapshot(manifest_path, label="checkpoint manifest")
    index = index_snapshot.value
    if not isinstance(index, dict):
        raise ValueError("checkpoint index must be a JSON object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index must contain a non-empty weight_map")
    shard_paths = []
    for shard_name in sorted(set(weight_map.values())):
        shard_path = (root / str(shard_name)).resolve()
        try:
            shard_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"checkpoint shard escapes checkpoint root: {shard_name}") from error
        shard_paths.append(shard_path)
    shard_records = _file_records(root, shard_paths)
    all_records = _file_records(root, (path for path in root.rglob("*") if path.is_file()))

    manifest = manifest_snapshot.value
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    records_by_path = {record["path"]: record for record in all_records}
    snapshot_records = {
        "config.json": config_snapshot,
        "model.safetensors.index.json": index_snapshot,
        "manifest.json": manifest_snapshot,
    }
    for relative_path, snapshot in snapshot_records.items():
        record = records_by_path.get(relative_path)
        if (
            record is None
            or record.get("sha256") != snapshot.sha256
            or record.get("size_bytes") != snapshot.size_bytes
        ):
            raise AssertionError(f"checkpoint metadata changed during provenance capture: {relative_path}")
    digest_verification = {
        "config": manifest.get("config_sha256") == config_snapshot.sha256,
        "index": manifest.get("index_sha256") == index_snapshot.sha256,
    }
    if not all(digest_verification.values()):
        raise AssertionError(f"checkpoint manifest digest verification failed: {digest_verification}")
    return {
        "path": str(root),
        "checkpoint_sha256": _records_sha256(all_records),
        "file_count": len(all_records),
        "total_file_bytes": sum(item["size_bytes"] for item in all_records),
        "indexed_weight_bytes": sum(item["size_bytes"] for item in shard_records),
        "indexed_weight_shards": shard_records,
        "files": all_records,
        "manifest_digest_verification": digest_verification,
    }


def _git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def source_provenance(
    *,
    repository: str | Path | None = None,
    source_roots: tuple[str | Path, ...] | None = None,
    require_clean: bool = False,
) -> dict[str, Any]:
    """Record git identity plus a content hash of the production vLLM source tree."""
    if repository is None:
        repository = _git_output(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel").strip()
    root = Path(repository).expanduser().resolve()
    roots = (Path(__file__).resolve().parent,) if source_roots is None else tuple(Path(path) for path in source_roots)
    source_paths = []

    def is_durable_source(path: Path) -> bool:
        return "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}

    for source_root in roots:
        resolved = source_root.expanduser().resolve()
        if resolved.is_file():
            if is_durable_source(resolved):
                source_paths.append(resolved)
        elif resolved.is_dir():
            source_paths.extend(path for path in resolved.rglob("*") if path.is_file() and is_durable_source(path))
        else:
            raise FileNotFoundError(f"source provenance root is missing: {resolved}")
    source_records = _file_records(root, source_paths)
    source_tree_sha256 = _records_sha256(source_records)
    git_head = _git_output(root, "rev-parse", "HEAD").strip()
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if require_clean and status:
        raise RuntimeError(f"dirty source repository is not benchmarkable: {root}; status={status.splitlines()}")
    tracked_diff = _git_output(root, "diff", "--binary", "HEAD", "--")
    dirty_payload = json.dumps(
        {
            "status": status,
            "tracked_diff_sha256": hashlib.sha256(tracked_diff.encode()).hexdigest(),
            "source_tree_sha256": source_tree_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "repository": str(root),
        "git_head": git_head,
        "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff.encode()).hexdigest(),
        "dirty_fingerprint_sha256": hashlib.sha256(dirty_payload).hexdigest(),
        "source_tree_sha256": source_tree_sha256,
        "source_file_count": len(source_records),
        "source_files": source_records,
    }


def prepare_workload(
    manifest: WorkloadManifest,
    *,
    request_count: int | None,
    uniform_prompt_length: int | None,
    request_id_prefix: str,
    max_new_tokens: int | None,
) -> WorkloadManifest:
    """Return an immutable exact-shape workload derived from a pinned manifest."""
    if manifest.prompt_source_path is not None and (request_count is not None or uniform_prompt_length is not None):
        raise ValueError("a frozen prompt source cannot be rewritten with synthetic request IDs or prompt lengths")
    result = manifest
    if uniform_prompt_length is not None:
        result = result.with_uniform_prompt_length(
            uniform_prompt_length,
            request_count=len(result.requests) if request_count is None else request_count,
            request_id_prefix=request_id_prefix,
        )
    elif request_count is not None:
        result = result.with_request_count(request_count, request_id_prefix=request_id_prefix)
    if max_new_tokens is not None:
        result = result.with_max_new_tokens(max_new_tokens)
    return result


def load_source_manifest(args: Any) -> WorkloadManifest:
    """Load a base manifest and optionally overlay one hash-pinned prompt JSONL source."""
    manifest = WorkloadManifest.from_path(args.manifest)
    prompt_jsonl = getattr(args, "prompt_jsonl", None)
    prompt_jsonl_sha256 = getattr(args, "prompt_jsonl_sha256", None)
    prompt_tokenizer_json = getattr(args, "prompt_tokenizer_json", None)
    expected_prompt_tokens = getattr(args, "expected_prompt_tokens", None)
    canonical_case_index = getattr(args, "canonical_identity_case", None)
    common_case_index = getattr(args, "common_prefix_identity_case", None)
    mixed_identity = getattr(args, "mixed_canonical_identity", False)
    canonical_prompts_csv = getattr(args, "canonical_prompts_csv", None)
    if type(mixed_identity) is not bool:
        raise TypeError("mixed canonical identity flag must be a built-in boolean")
    if sum((canonical_case_index is not None, common_case_index is not None, mixed_identity)) > 1:
        raise ValueError("canonical, common-prefix, and mixed identity modes are mutually exclusive")
    if mixed_identity:
        incompatible = {
            "prompt_jsonl": prompt_jsonl,
            "prompt_jsonl_sha256": prompt_jsonl_sha256,
            "expected_prompt_tokens": expected_prompt_tokens,
            "uniform_prompt_length": getattr(args, "uniform_prompt_length", None),
        }
        enabled = sorted(name for name, value in incompatible.items() if value is not None)
        if enabled:
            raise ValueError(f"mixed canonical identity is incompatible with workload rewrites: {enabled}")
        if canonical_prompts_csv is None or prompt_tokenizer_json is None:
            raise ValueError("mixed canonical identity requires --canonical-prompts-csv and --prompt-tokenizer-json")
        request_count = getattr(args, "request_count", None)
        if type(request_count) is not int or request_count not in {4, 96}:
            raise ValueError("mixed canonical identity requires --request-count 4 or 96")
        max_new_tokens = getattr(args, "max_new_tokens", None)
        if max_new_tokens not in (None, 500):
            raise ValueError("mixed canonical identity requires exactly 500 new tokens")
        tokenizer = SnapshotBoundTokenizer.from_path(prompt_tokenizer_json)
        cases = load_canonical_7b_identity_cases(canonical_prompts_csv)
        stage = "b4" if request_count == 4 else "b96"
        return build_mixed_canonical_identity_manifest(
            manifest,
            cases=cases,
            prompts_csv=canonical_prompts_csv,
            tokenizer=tokenizer,
            request_count=request_count,
            request_id_prefix=f"{args.request_id_prefix}-{stage}",
        )
    if common_case_index is not None:
        incompatible = {
            "prompt_jsonl": prompt_jsonl,
            "prompt_jsonl_sha256": prompt_jsonl_sha256,
            "expected_prompt_tokens": expected_prompt_tokens,
            "uniform_prompt_length": getattr(args, "uniform_prompt_length", None),
        }
        enabled = sorted(name for name, value in incompatible.items() if value is not None)
        if enabled:
            raise ValueError(f"common-prefix identity is incompatible with workload rewrites: {enabled}")
        if canonical_prompts_csv is None or prompt_tokenizer_json is None:
            raise ValueError("common-prefix identity requires --canonical-prompts-csv and --prompt-tokenizer-json")
        request_count = getattr(args, "request_count", None)
        if request_count is None or request_count <= 0:
            raise ValueError("common-prefix identity requires a positive --request-count")
        max_new_tokens = getattr(args, "max_new_tokens", None)
        if max_new_tokens not in (None, 500):
            raise ValueError("common-prefix identity requires exactly 500 new tokens")

        from bionemo.evo2.vllm.accuracy import (
            build_common_prefix_identity_manifest,
            load_common_prefix_identity_cases,
        )

        tokenizer = SnapshotBoundTokenizer.from_path(prompt_tokenizer_json)
        cases = load_common_prefix_identity_cases(canonical_prompts_csv)
        return build_common_prefix_identity_manifest(
            manifest,
            case=cases[common_case_index],
            prompts_csv=canonical_prompts_csv,
            tokenizer=tokenizer,
            request_count=request_count,
            request_id_prefix=args.request_id_prefix,
        )
    if canonical_case_index is not None:
        incompatible = {
            "prompt_jsonl": prompt_jsonl,
            "prompt_jsonl_sha256": prompt_jsonl_sha256,
            "expected_prompt_tokens": expected_prompt_tokens,
            "uniform_prompt_length": getattr(args, "uniform_prompt_length", None),
        }
        enabled = sorted(name for name, value in incompatible.items() if value is not None)
        if enabled:
            raise ValueError(f"canonical identity is incompatible with workload rewrites: {enabled}")
        if canonical_prompts_csv is None or prompt_tokenizer_json is None:
            raise ValueError("canonical identity requires --canonical-prompts-csv and --prompt-tokenizer-json")
        request_count = getattr(args, "request_count", None)
        if request_count is None or request_count <= 0:
            raise ValueError("canonical identity requires a positive --request-count")
        max_new_tokens = getattr(args, "max_new_tokens", None)
        if max_new_tokens not in (None, 500):
            raise ValueError("canonical identity requires exactly 500 new tokens")

        from bionemo.evo2.vllm.accuracy import (
            build_canonical_identity_manifest,
            load_canonical_7b_identity_cases as load_canonical_cases,
        )

        tokenizer = SnapshotBoundTokenizer.from_path(prompt_tokenizer_json)
        cases = load_canonical_cases(canonical_prompts_csv)
        return build_canonical_identity_manifest(
            manifest,
            case=cases[canonical_case_index],
            prompts_csv=canonical_prompts_csv,
            tokenizer=tokenizer,
            request_count=request_count,
            request_id_prefix=args.request_id_prefix,
        )
    if canonical_prompts_csv is not None:
        raise ValueError(
            "--canonical-prompts-csv requires --canonical-identity-case, "
            "--common-prefix-identity-case, or --mixed-canonical-identity"
        )
    if prompt_jsonl is None:
        if any(value is not None for value in (prompt_jsonl_sha256, prompt_tokenizer_json, expected_prompt_tokens)):
            raise ValueError("prompt JSONL provenance options require --prompt-jsonl")
        return manifest
    if prompt_jsonl_sha256 is None or prompt_tokenizer_json is None:
        raise ValueError("--prompt-jsonl requires --prompt-jsonl-sha256 and --prompt-tokenizer-json")

    tokenizer = SnapshotBoundTokenizer.from_path(prompt_tokenizer_json)
    return manifest.with_prompt_jsonl(
        prompt_jsonl,
        tokenizer=tokenizer,
        expected_sha256=prompt_jsonl_sha256,
        expected_prompt_tokens=expected_prompt_tokens,
    )


class CUDAGraphProofRecorder:
    """Persist scheduler CUDA-graph observations without periodic logger resets."""

    def __init__(self) -> None:
        """Create an empty phase-aware observation stream."""
        self._phase = "unlabeled"
        self.observations: list[dict[str, Any]] = []
        self.scheduler_observations: list[dict[str, Any]] = []

    def start_phase(self, phase: str) -> None:
        """Label subsequent scheduler observations."""
        if not phase:
            raise ValueError("CUDA graph proof phase cannot be empty")
        self._phase = phase

    def record(
        self,
        scheduler_stats: Any | None,
        iteration_stats: Any | None,
        mm_cache_stats: Any | None = None,
        engine_idx: int | None = None,
    ) -> None:
        """Record one scheduler dispatch when CUDA graph metrics are present."""
        del mm_cache_stats
        engine_index = 0 if engine_idx is None else int(engine_idx)
        if scheduler_stats is not None and iteration_stats is not None:
            prefix_stats = getattr(scheduler_stats, "prefix_cache_stats", None)
            prompt_stats = getattr(iteration_stats, "prompt_token_stats", None)
            if prefix_stats is not None and prompt_stats is not None:
                preempted_queries = int(prefix_stats.preempted_queries)
                preempted_hits = int(prefix_stats.preempted_hits)
                preemption_events = int(iteration_stats.num_preempted_reqs)
                self.scheduler_observations.append(
                    {
                        "phase": self._phase,
                        "engine_index": engine_index,
                        "preemption_events": preemption_events,
                        "recompute_events": preemption_events,
                        "prefix_preempted_requests": int(prefix_stats.preempted_requests),
                        "prefix_preempted_queries": preempted_queries,
                        "prefix_preempted_hits": preempted_hits,
                        "preempted_prompt_recomputed_tokens": preempted_queries - preempted_hits,
                        "prompt_tokens_computed": int(prompt_stats.computed),
                        "prompt_tokens_cached": int(prompt_stats.cached_tokens),
                        "prompt_tokens_total": int(prompt_stats.total),
                        "num_running_requests": int(scheduler_stats.num_running_reqs),
                        "num_waiting_requests": int(scheduler_stats.num_waiting_reqs),
                        "num_skipped_waiting_requests": int(scheduler_stats.num_skipped_waiting_reqs),
                    }
                )

        graph_stats = None if scheduler_stats is None else scheduler_stats.cudagraph_stats
        if graph_stats is None:
            return
        stats = graph_stats
        prompt_stats = None if iteration_stats is None else getattr(iteration_stats, "prompt_token_stats", None)
        prompt_token_count = None if prompt_stats is None else int(prompt_stats.total)
        first_token_event_count = (
            None if iteration_stats is None else len(getattr(iteration_stats, "time_to_first_tokens_iter", ()))
        )
        generation_token_count = (
            None if iteration_stats is None else int(getattr(iteration_stats, "num_generation_tokens", -1))
        )
        pure_decode = prompt_token_count == 0 and first_token_event_count == 0
        self.observations.append(
            {
                "phase": self._phase,
                "engine_index": engine_index,
                "num_unpadded_tokens": int(stats.num_unpadded_tokens),
                "num_padded_tokens": int(stats.num_padded_tokens),
                "num_paddings": int(stats.num_paddings),
                "runtime_mode": str(stats.runtime_mode),
                "request_dimensions": {
                    "schema_version": 1,
                    "source": "iteration-stats-bound-to-cudagraph-dispatch",
                    "prefill_req_count": 0 if pure_decode else None,
                    "decode_req_count": generation_token_count if pure_decode else None,
                    "token_count": int(stats.num_unpadded_tokens),
                    "prompt_token_count": prompt_token_count,
                    "first_token_event_count": first_token_event_count,
                },
            }
        )

    def log(self) -> None:
        """Retain observations when vLLM requests a periodic log flush."""

    def log_engine_initialized(self) -> None:
        """Satisfy the vLLM stat logger protocol."""

    def record_sleep_state(self, sleep: int = 0, level: int = 0) -> None:
        """Satisfy the vLLM stat logger protocol."""
        del sleep, level


def scheduler_capacity_proof_summary(
    observations: Sequence[dict[str, Any]],
    *,
    phase: str,
    global_wave_size: int,
    max_num_seqs: int,
    engine_request_count: int | None = None,
) -> dict[str, Any]:
    """Summarize phase-local scheduler fit without inferring absent telemetry."""
    if not phase:
        raise ValueError("scheduler proof phase cannot be empty")
    submitted_engine_requests = global_wave_size if engine_request_count is None else engine_request_count
    if global_wave_size <= 0 or max_num_seqs <= 0 or submitted_engine_requests <= 0:
        raise ValueError("global wave, engine request, and max_num_seqs counts must be positive")
    phase_observations = [item for item in observations if item.get("phase") == phase]

    def total(field: str) -> int:
        return sum(int(item[field]) for item in phase_observations)

    preemption_events = total("preemption_events")
    recompute_events = total("recompute_events")
    prefix_preempted_requests = total("prefix_preempted_requests")
    preempted_queries = total("prefix_preempted_queries")
    preempted_hits = total("prefix_preempted_hits")
    preempted_recomputed = total("preempted_prompt_recomputed_tokens")
    maximum_running = max((int(item["num_running_requests"]) for item in phase_observations), default=0)
    request_count_within_scheduler_ceiling = submitted_engine_requests <= max_num_seqs
    running_count_within_scheduler_ceiling = maximum_running <= max_num_seqs
    batch_fit_without_preemption = (
        bool(phase_observations)
        and request_count_within_scheduler_ceiling
        and running_count_within_scheduler_ceiling
        and preemption_events == 0
        and recompute_events == 0
        and prefix_preempted_requests == 0
        and preempted_queries == 0
        and preempted_hits == 0
        and preempted_recomputed == 0
    )
    return {
        "phase": phase,
        "global_wave_size": global_wave_size,
        "engine_request_count": submitted_engine_requests,
        "max_num_seqs": max_num_seqs,
        "scheduler_observation_count": len(phase_observations),
        "preemption_events": preemption_events,
        "recompute_events": recompute_events,
        "prefix_preempted_requests": prefix_preempted_requests,
        "prefix_preempted_queries": preempted_queries,
        "prefix_preempted_hits": preempted_hits,
        "preempted_prompt_recomputed_tokens": preempted_recomputed,
        "prompt_tokens_computed": total("prompt_tokens_computed"),
        "prompt_tokens_cached": total("prompt_tokens_cached"),
        "prompt_tokens_total": total("prompt_tokens_total"),
        "maximum_running_requests": maximum_running,
        "maximum_waiting_requests": max((int(item["num_waiting_requests"]) for item in phase_observations), default=0),
        "maximum_skipped_waiting_requests": max(
            (int(item["num_skipped_waiting_requests"]) for item in phase_observations), default=0
        ),
        "request_count_within_scheduler_ceiling": request_count_within_scheduler_ceiling,
        "running_count_within_scheduler_ceiling": running_count_within_scheduler_ceiling,
        "batch_fit_without_preemption": batch_fit_without_preemption,
    }


def validate_scheduler_capacity_proof(proof: dict[str, Any]) -> None:
    """Fail closed unless one submitted wave fits without preemption or recompute."""
    positive_fields = ("global_wave_size", "engine_request_count", "max_num_seqs")
    nonnegative_fields = (
        "scheduler_observation_count",
        "preemption_events",
        "recompute_events",
        "prefix_preempted_requests",
        "prefix_preempted_queries",
        "prefix_preempted_hits",
        "preempted_prompt_recomputed_tokens",
        "prompt_tokens_computed",
        "prompt_tokens_cached",
        "prompt_tokens_total",
        "maximum_running_requests",
        "maximum_waiting_requests",
        "maximum_skipped_waiting_requests",
    )
    if any(
        isinstance(proof.get(field), bool) or not isinstance(proof.get(field), int) or proof[field] <= 0
        for field in positive_fields
    ):
        raise AssertionError("scheduler wave dimensions must be positive integers")
    if any(
        isinstance(proof.get(field), bool) or not isinstance(proof.get(field), int) or proof[field] < 0
        for field in nonnegative_fields
    ):
        raise AssertionError("scheduler telemetry counters must be nonnegative integers")
    if proof["scheduler_observation_count"] <= 0:
        raise AssertionError("no scheduler telemetry was retained for the generation wave")
    request_within_ceiling = proof["engine_request_count"] <= proof["max_num_seqs"]
    running_within_ceiling = proof["maximum_running_requests"] <= proof["max_num_seqs"]
    if proof.get("request_count_within_scheduler_ceiling") is not request_within_ceiling:
        raise AssertionError("scheduler request-ceiling gate is inconsistent")
    if proof.get("running_count_within_scheduler_ceiling") is not running_within_ceiling:
        raise AssertionError("scheduler running-count gate is inconsistent")
    if not request_within_ceiling or not running_within_ceiling:
        raise AssertionError("generation wave exceeded the per-engine max_num_seqs scheduler ceiling")
    preemption_fields = (
        "preemption_events",
        "recompute_events",
        "prefix_preempted_requests",
        "prefix_preempted_queries",
        "prefix_preempted_hits",
        "preempted_prompt_recomputed_tokens",
    )
    if any(proof[field] != 0 for field in preemption_fields):
        raise AssertionError("scheduler preemption/recompute occurred during the generation wave")
    if proof["prefix_preempted_hits"] > proof["prefix_preempted_queries"]:
        raise AssertionError("scheduler preempted prefix-cache hit telemetry is inconsistent")
    expected_fit = (
        request_within_ceiling and running_within_ceiling and all(proof[field] == 0 for field in preemption_fields)
    )
    if proof.get("batch_fit_without_preemption") is not expected_fit or not expected_fit:
        raise AssertionError("generation wave did not prove scheduler fit without preemption")


def validate_full_decode_proof(
    observations: list[dict[str, Any]],
    *,
    phase: str,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    """Allow mixed prefill while requiring dense, exact FULL decode replay."""
    summary = full_decode_proof_summary(
        observations,
        phase=phase,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    if summary["observation_count"] == 0:
        raise AssertionError(f"no CUDA graph observations were recorded for {phase}")
    if summary["eager_decode_dispatch_count"]:
        raise AssertionError(f"{phase} used forbidden CUDAGraphMode.NONE decode fallback execution")
    if not summary["full_decode_unpadded"]:
        raise AssertionError(f"{phase} used semantic or scheduler padding during FULL decode")
    if max_new_tokens > 1 and not summary["global_batch_hit"]:
        raise AssertionError(f"{phase} did not execute a FULL global batch")
    if summary["long_run_gates_applied"] and not summary["coverage_gate_passed"]:
        raise AssertionError(
            f"{phase} FULL decode coverage was {summary['full_decode_tokens']}/"
            f"{summary['expected_decode_tokens']} tokens; at least "
            f"{summary['minimum_full_decode_tokens']} are required"
        )
    if summary["long_run_gates_applied"] and not summary["occupancy_gate_passed"]:
        raise AssertionError(
            f"{phase} FULL decode occupancy averaged {summary['average_full_batch_occupancy']:.3f}/"
            f"{batch_size}; at least {summary['minimum_average_occupancy']:.3f} is required"
        )


def full_decode_proof_summary(
    observations: list[dict[str, Any]],
    *,
    phase: str,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Return durable numeric evidence for exact, batched FULL decode replay."""
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    phase_observations = [item for item in observations if item["phase"] == phase]
    eager_decode = [
        item
        for item in phase_observations
        if item["runtime_mode"].endswith("NONE") and item["num_unpadded_tokens"] <= batch_size
    ]
    full_decode = [item for item in phase_observations if item["runtime_mode"].endswith("FULL")]
    full_decode_unpadded = not any(
        item["num_padded_tokens"] != item["num_unpadded_tokens"] or item["num_paddings"] != 0 for item in full_decode
    )
    full_decode_tokens = sum(item["num_unpadded_tokens"] for item in full_decode)
    expected_decode_tokens = batch_size * max(0, max_new_tokens - 1)
    minimum_full_decode_tokens = max(0, expected_decode_tokens - batch_size)
    average_batch_occupancy = full_decode_tokens / len(full_decode) if full_decode else 0.0
    minimum_average_occupancy = batch_size * 0.9
    global_batch_hit = any(item["num_unpadded_tokens"] == batch_size for item in full_decode)
    long_run_gates_applied = max_new_tokens >= 32
    coverage_gate_passed = not long_run_gates_applied or full_decode_tokens >= minimum_full_decode_tokens
    occupancy_gate_passed = not long_run_gates_applied or average_batch_occupancy >= minimum_average_occupancy
    passed = (
        bool(phase_observations)
        and not eager_decode
        and full_decode_unpadded
        and (max_new_tokens <= 1 or global_batch_hit)
        and coverage_gate_passed
        and occupancy_gate_passed
    )
    return {
        "phase": phase,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "observation_count": len(phase_observations),
        "eager_decode_dispatch_count": len(eager_decode),
        "full_dispatch_count": len(full_decode),
        "expected_decode_tokens": expected_decode_tokens,
        "full_decode_tokens": full_decode_tokens,
        "minimum_full_decode_tokens": minimum_full_decode_tokens,
        "coverage_fraction": full_decode_tokens / expected_decode_tokens if expected_decode_tokens else 1.0,
        "maximum_full_batch": max((item["num_unpadded_tokens"] for item in full_decode), default=0),
        "average_full_batch_occupancy": average_batch_occupancy,
        "minimum_average_occupancy": minimum_average_occupancy,
        "occupancy_fraction": average_batch_occupancy / batch_size,
        "global_batch_hit": global_batch_hit,
        "full_decode_unpadded": full_decode_unpadded,
        "long_run_gates_applied": long_run_gates_applied,
        "coverage_gate_passed": coverage_gate_passed,
        "occupancy_gate_passed": occupancy_gate_passed,
        "passed": passed,
    }


def exact_generation_progress_evidence(
    manifest: WorkloadManifest,
    *,
    sidecar_counts: dict[str, int],
    full_decode_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Prove exact retained outputs and every post-prefill decode update."""
    request_count = len(manifest.requests)
    retained_token_count = request_count * manifest.max_new_tokens
    expected_counts = {
        "request_count": request_count,
        "output_token_id_count": retained_token_count,
        "chosen_token_logprob_count": retained_token_count,
    }
    for field, expected in expected_counts.items():
        observed = sidecar_counts.get(field)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected:
            raise AssertionError(f"retained exact-generation field {field!r} was {observed!r}, expected {expected}")

    decode_steps_per_request = max(0, manifest.max_new_tokens - 1)
    expected_decode_updates = request_count * decode_steps_per_request
    observed_decode_updates = 0
    retained_expected_updates = 0
    for summary in full_decode_summaries:
        if not isinstance(summary, dict):
            raise AssertionError("exact decode proof contains a malformed summary")
        expected = summary.get("expected_decode_tokens")
        observed = summary.get("full_decode_tokens")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected < 0
            or isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < 0
        ):
            raise AssertionError("exact decode proof contains malformed token counters")
        if (
            observed != expected
            or summary.get("eager_decode_dispatch_count") != 0
            or summary.get("full_decode_unpadded") is not True
            or summary.get("passed") is not True
        ):
            raise AssertionError("exact decode proof omitted updates or used forbidden execution")
        retained_expected_updates += expected
        observed_decode_updates += observed
    if retained_expected_updates != expected_decode_updates or observed_decode_updates != expected_decode_updates:
        raise AssertionError("exact decode proof does not cover every request's post-prefill token update")

    return {
        "schema_version": 1,
        "request_count": request_count,
        "prefill_request_count": request_count,
        "first_sampled_token_count": request_count,
        "decode_steps_per_request": decode_steps_per_request,
        "expected_decode_token_updates": expected_decode_updates,
        "observed_full_decode_token_updates": observed_decode_updates,
        "retained_output_token_id_count": retained_token_count,
        "retained_chosen_token_logprob_count": retained_token_count,
        "passed": True,
    }


def attach_exact_generation_progress_evidence(
    phase_artifacts: Sequence[dict[str, Any]],
    *,
    manifest: WorkloadManifest,
    enabled: bool,
    proof_collected: bool,
    topology: str,
    linked_proof_artifact: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Attach exact counters after timed generation for proof and linked speed lanes."""
    artifacts = [dict(phase) for phase in phase_artifacts]
    if not enabled:
        return artifacts, None
    if topology not in {"tp2", "dp2"}:
        raise ValueError(f"unsupported exact-progress topology: {topology!r}")
    if proof_collected:
        if linked_proof_artifact is not None:
            raise ValueError("a proof run cannot link external exact-progress evidence")
        proof_source = "current-proof-run"
    else:
        if linked_proof_artifact is None:
            raise ValueError("an exact-progress speed run requires its linked proof artifact")
        proof_source = str(Path(linked_proof_artifact).expanduser().resolve())

    retained_tokens = len(manifest.requests) * manifest.max_new_tokens
    phase_summaries = []
    for phase in artifacts:
        sidecar = phase.get("full_output_artifact")
        if not isinstance(sidecar, dict):
            raise AssertionError("exact-progress phase is missing full-output sidecar metadata")
        counts = {
            "request_count": sidecar.get("request_count"),
            "output_token_id_count": sidecar.get("output_token_id_count"),
            "chosen_token_logprob_count": sidecar.get("chosen_token_logprob_count"),
        }
        if proof_collected:
            if topology == "tp2":
                waves = phase.get("wave_proofs")
                if not isinstance(waves, list):
                    raise AssertionError("TP2 exact-progress proof is missing physical waves")
                decode_summaries = [wave.get("full_decode_proof") for wave in waves if isinstance(wave, dict)]
            else:
                waves = phase.get("waves")
                if not isinstance(waves, list):
                    raise AssertionError("DP2 exact-progress proof is missing physical waves")
                decode_summaries = [
                    engine.get("full_decode_proof")
                    for wave in waves
                    if isinstance(wave, dict)
                    for engine in wave.get("engines", ())
                    if isinstance(engine, dict)
                ]
            evidence = exact_generation_progress_evidence(
                manifest,
                sidecar_counts=counts,
                full_decode_summaries=decode_summaries,
            )
        else:
            expected_counts = {
                "request_count": len(manifest.requests),
                "output_token_id_count": retained_tokens,
                "chosen_token_logprob_count": retained_tokens,
            }
            if counts != expected_counts:
                raise AssertionError("speed phase retained output counts do not match the exact workload")
            evidence = {
                "schema_version": 1,
                "request_count": len(manifest.requests),
                "prefill_request_count": len(manifest.requests),
                "first_sampled_token_count": len(manifest.requests),
                "decode_steps_per_request": max(0, manifest.max_new_tokens - 1),
                "expected_decode_token_updates": len(manifest.requests) * max(0, manifest.max_new_tokens - 1),
                "observed_full_decode_token_updates": None,
                "retained_output_token_id_count": retained_tokens,
                "retained_chosen_token_logprob_count": retained_tokens,
                "execution_proof_source": proof_source,
                "passed": True,
            }
        phase["exact_generation_progress"] = evidence
        phase_summaries.append({"phase": phase.get("phase"), "evidence": evidence})
    return artifacts, {
        "schema_version": 1,
        "phase_count": len(artifacts),
        "execution_proof_source": proof_source,
        "phases": phase_summaries,
        "passed": bool(artifacts) and all(item["evidence"].get("passed") is True for item in phase_summaries),
    }


def request_seed(
    base_seed: int,
    *,
    call_index: int,
    dp_rank: int,
    dp_size: int,
    request_index_in_stream: int,
) -> int:
    """Return one deterministic request seed from physical call and DP stream coordinates."""
    seed = _request_seed_preimage(
        base_seed,
        call_index=call_index,
        dp_rank=dp_rank,
        dp_size=dp_size,
        request_index_in_stream=request_index_in_stream,
    )
    if seed >= _SEED_MODULUS:
        raise ValueError(
            f"request seed wraparound is forbidden: pre-modulo seed {seed} reaches modulus {_SEED_MODULUS}"
        )
    return seed


def build_request_sampling_params(
    manifest: WorkloadManifest,
    *,
    sampling_params_factory: Callable[..., Any],
    execution_records: Sequence[RequestExecutionRecord],
) -> list[Any]:
    """Build exact-length sampling params from the persisted production seed records."""
    if len(execution_records) != len(manifest.requests):
        raise ValueError("execution records must align with every manifest request")
    if any(
        request.request_id != record.request_id
        for request, record in zip(manifest.requests, execution_records, strict=True)
    ):
        raise ValueError("execution record request IDs must preserve manifest order")
    common_kwargs = sampling_params_kwargs(manifest)
    params_by_request = []
    for record in execution_records:
        expected = {**common_kwargs, "seed": record.seed}
        params = sampling_params_factory(**expected)
        for field, expected_value in expected.items():
            observed_value = getattr(params, field, None)
            if type(observed_value) is not type(expected_value) or observed_value != expected_value:
                raise RuntimeError(
                    f"sampling parameter {field} changed during construction: "
                    f"expected {expected_value!r}, observed {observed_value!r}"
                )
        params_by_request.append(params)
    return params_by_request


@dataclass(frozen=True)
class RequestExecutionRecord:
    """Persist deterministic ownership and RNG coordinates for one request."""

    request_id: str
    global_request_index: int
    generation_round: int
    dp_rank: int
    call_index: int
    seed: int

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise TypeError("execution request_id must be a nonempty built-in string")
        for field in ("global_request_index", "generation_round", "dp_rank", "call_index", "seed"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise TypeError(f"execution {field} must be a nonnegative built-in integer")

    @property
    def execution_uid(self) -> str:
        """Return a phase-stable composite identity for one execution."""
        return (
            f"round={self.generation_round}/call={self.call_index}/"
            f"global={self.global_request_index}/dp={self.dp_rank}/request={self.request_id}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe execution record."""
        return {"execution_uid": self.execution_uid, **asdict(self)}


def _validate_request_execution_rows(
    retained: Any,
    *,
    expected: Sequence[RequestExecutionRecord],
) -> None:
    """Bind retained request coordinates to exact independently reconstructed rows."""
    if not isinstance(retained, list) or len(retained) != len(expected):
        raise AssertionError("retained request execution rows do not cover the exact workload")
    expected_rows = [record.to_dict() for record in expected]
    expected_fields = set(expected_rows[0]) if expected_rows else set()
    for row_index, (row, expected_row) in enumerate(zip(retained, expected_rows, strict=True)):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise AssertionError("retained request execution row fields are not exact")
        for field in ("global_request_index", "generation_round", "dp_rank", "call_index", "seed"):
            _require_builtin_integer(row.get(field), label=f"request execution row {row_index} {field}")
        for field in ("execution_uid", "request_id"):
            if type(row.get(field)) is not str:
                raise AssertionError(f"request execution row {row_index} {field} must be a built-in string")
        if row != expected_row:
            raise AssertionError("retained request execution row does not match reconstructed coordinates")


def build_request_execution_records(
    manifest: WorkloadManifest,
    *,
    global_request_offset: int,
    dp_rank: int,
    dp_size: int,
    generation_round: int,
    call_index: int,
) -> tuple[RequestExecutionRecord, ...]:
    """Build one ownership/seed record for each exact manifest request."""
    for label, value in (
        ("global_request_offset", global_request_offset),
        ("dp_rank", dp_rank),
        ("dp_size", dp_size),
        ("generation_round", generation_round),
        ("call_index", call_index),
    ):
        if type(value) is not int:
            raise TypeError(f"{label} must be a built-in integer")
    if min(global_request_offset, dp_rank, generation_round, call_index) < 0 or dp_size <= 0:
        raise ValueError("execution coordinates must be nonnegative")
    if dp_rank >= dp_size:
        raise ValueError("dp_rank must be smaller than dp_size")
    return tuple(
        RequestExecutionRecord(
            request_id=request.request_id,
            global_request_index=global_request_offset + local_index,
            generation_round=generation_round,
            dp_rank=dp_rank,
            call_index=call_index,
            seed=request_seed(
                manifest.seed,
                call_index=call_index,
                dp_rank=dp_rank,
                dp_size=dp_size,
                request_index_in_stream=local_index,
            ),
        )
        for local_index, request in enumerate(manifest.requests)
    )


def build_wave_execution_records(
    manifest: WorkloadManifest,
    *,
    global_wave_size: int,
    generation_round: int,
    call_index_start: int,
    global_request_index_start: int = 0,
) -> tuple[RequestExecutionRecord, ...]:
    """Build exact request records whose call indices match physical generation calls."""
    for label, value in (
        ("global_wave_size", global_wave_size),
        ("generation_round", generation_round),
        ("call_index_start", call_index_start),
        ("global_request_index_start", global_request_index_start),
    ):
        if type(value) is not int:
            raise TypeError(f"{label} must be a built-in integer")
    if min(generation_round, call_index_start, global_request_index_start) < 0:
        raise ValueError("generation round and call index start must be nonnegative")
    records = []
    for wave in build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=global_wave_size,
        replica_count=1,
    ):
        records.extend(
            build_request_execution_records(
                manifest.request_slice(wave.start, wave.stop),
                global_request_offset=global_request_index_start + wave.start,
                dp_rank=0,
                dp_size=1,
                generation_round=generation_round,
                call_index=call_index_start + wave.wave_index,
            )
        )
    return tuple(records)


def write_full_output_artifact(
    path: str | Path,
    *,
    manifest: WorkloadManifest,
    outputs: Any,
    execution_records: tuple[RequestExecutionRecord, ...],
    decode_output_token_ids: Callable[[Sequence[int]], str] | None = None,
    ownership_validator: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Stream every output token and chosen-token logprob to deterministic gzip JSONL."""
    records = records_from_vllm_outputs(manifest, outputs)
    return write_full_generation_records_artifact(
        path,
        records=records,
        execution_records=execution_records,
        decode_output_token_ids=decode_output_token_ids,
        ownership_validator=ownership_validator,
    )


def write_full_generation_records_artifact(
    path: str | Path,
    *,
    records: Sequence[GenerationRecord],
    execution_records: Sequence[RequestExecutionRecord],
    decode_output_token_ids: Callable[[Sequence[int]], str] | None = None,
    ownership_validator: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Persist backend-neutral exact generation records as deterministic gzip JSONL."""
    if len(execution_records) != len(records):
        raise AssertionError("execution records must align with generated outputs")
    output = Path(path).resolve()
    generated_token_count = 0
    decoded_output_byte_count = 0

    def writer(raw_handle: Any) -> None:
        nonlocal decoded_output_byte_count, generated_token_count
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for generation, execution in zip(
                    records,
                    execution_records,
                    strict=True,
                ):
                    if ownership_validator is not None:
                        ownership_validator()
                    if execution.request_id != generation.request_id:
                        raise AssertionError("execution and generation request IDs must align")
                    validate_dna_output_token_ids(
                        generation.output_token_ids,
                        request_id=generation.request_id,
                    )
                    row = {
                        **execution.to_dict(),
                        "prompt_token_ids": list(generation.prompt_token_ids),
                        "output_token_ids": list(generation.output_token_ids),
                        "chosen_token_logprobs": list(generation.output_logprobs),
                        **exact_length_evidence(
                            prompt_tokens=len(generation.prompt_token_ids),
                            generated_tokens=len(generation.output_token_ids),
                            requested_new_tokens=generation.requested_max_tokens,
                        ),
                        "finish_reason": generation.finish_reason,
                        "stop_reason": generation.stop_reason,
                        "stopped_on_eos": generation.stopped_on_eos,
                    }
                    if decode_output_token_ids is not None:
                        decoded = decode_output_token_ids(generation.output_token_ids)
                        if not isinstance(decoded, str):
                            raise TypeError("output token decoder must return text")
                        try:
                            decoded_bytes = decoded.encode("ascii")
                        except UnicodeEncodeError as error:
                            raise AssertionError(
                                "decoded output must exactly match A/C/G/T token IDs"
                            ) from error
                        if decoded_bytes != bytes(generation.output_token_ids):
                            raise AssertionError("decoded output must exactly match A/C/G/T token IDs")
                        row.update(
                            {
                                "output_text_utf8_base64": base64.b64encode(decoded_bytes).decode("ascii"),
                                "output_text_sha256": hashlib.sha256(decoded_bytes).hexdigest(),
                                "output_text_utf8_bytes": len(decoded_bytes),
                            }
                        )
                        decoded_output_byte_count += len(decoded_bytes)
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                    handle.write("\n")
                    generated_token_count += len(generation.output_token_ids)

    receipt, _ = publish_file_noreplace(
        output,
        writer,
        ownership_validator=ownership_validator,
        publication_recorder=_record_output_namespace_publication,
    )
    metadata = {
        "schema_version": 2,
        "format": "jsonl",
        "compression": "gzip",
        "path": str(output),
        "sha256": receipt.sha256,
        "size_bytes": receipt.size_bytes,
        "publication_receipt": receipt.to_dict(),
        "request_count": len(records),
        "generated_token_count": generated_token_count,
        "output_token_id_count": generated_token_count,
        "chosen_token_logprob_count": generated_token_count,
    }
    if decode_output_token_ids is not None:
        metadata.update(
            {
                "decoded_output_bytes_retained": True,
                "decoded_output_byte_count": decoded_output_byte_count,
            }
        )
    return metadata


def phase_output_artifact_path(
    root_artifact_path: str | Path,
    *,
    phase: str,
    dp_rank: int | None = None,
) -> Path:
    """Return a collision-free full-output sidecar path for one phase/replica."""
    if not phase:
        raise ValueError("phase cannot be empty")
    if dp_rank is not None and dp_rank < 0:
        raise ValueError("dp_rank must be nonnegative")
    root = Path(root_artifact_path)
    replica_suffix = "" if dp_rank is None else f".dp{dp_rank}"
    return root.with_name(f"{root.name}.{phase}{replica_suffix}.outputs.jsonl.gz")


def _output_namespace_marker_path(path: str | Path) -> Path:
    output = Path(path).resolve()
    base_name = output.name.removesuffix(output.suffix)
    return output.with_name(f"{base_name}.inprogress")


def _unlink_path_if_identity(path: Path, expected_identity: tuple[int, int]) -> bool:
    """Unlink only a regular file that is still the inode created by this process."""
    try:
        observed = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(observed.st_mode) or (observed.st_dev, observed.st_ino) != expected_identity:
        return False
    path.unlink()
    return True


def reserve_output_namespace(path: str | Path) -> Path:
    """Atomically reserve a new artifact namespace and refuse any stale outputs."""
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    marker = _output_namespace_marker_path(output)
    legacy_marker = output.with_name(f"{output.name}.inprogress")
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    sidecar_prefix = f"{output.name.removesuffix(output.suffix)}."
    collisions = [candidate for candidate in {output, temporary, marker, legacy_marker} if candidate.exists()]
    collisions.extend(
        candidate
        for candidate in output.parent.iterdir()
        if candidate.name.startswith(sidecar_prefix)
        and (candidate.name.endswith(".outputs.jsonl.gz") or candidate.name.endswith(".outputs.jsonl.gz.tmp"))
    )
    if collisions:
        names = ", ".join(sorted({candidate.name for candidate in collisions}))
        raise FileExistsError(f"output namespace already contains prior artifacts: {names}")
    parent_fd = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    marker_fd = -1
    marker_identity: tuple[int, int] | None = None
    try:
        parent_stat = os.fstat(parent_fd)
        marker_fd = os.open(
            marker.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        marker_stat = os.fstat(marker_fd)
        marker_identity = (marker_stat.st_dev, marker_stat.st_ino)
        marker_payload = (
            json.dumps(
                {
                    "schema_version": 3,
                    "state": "in_progress",
                    "output_artifact_path": str(output),
                    "parent_device": parent_stat.st_dev,
                    "parent_inode": parent_stat.st_ino,
                    "marker_device": marker_stat.st_dev,
                    "marker_inode": marker_stat.st_ino,
                    "started_unix_s": time.time(),
                    "argv": [sys.executable, *sys.argv],
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        with os.fdopen(os.dup(marker_fd), "wb") as handle:
            handle.write(marker_payload)
            handle.flush()
        os.fsync(marker_fd)
        marker_path_stat = os.stat(marker.name, dir_fd=parent_fd, follow_symlinks=False)
        if (marker_path_stat.st_dev, marker_path_stat.st_ino) != marker_identity:
            raise RuntimeError("output namespace marker changed during reservation")
        os.fsync(parent_fd)
        ownership = _OutputNamespaceOwnership(
            output_path=output,
            marker_identity=marker_identity,
            parent_identity=(parent_stat.st_dev, parent_stat.st_ino),
        )
        with _OUTPUT_NAMESPACE_OWNERSHIP_LOCK:
            if marker in _OUTPUT_NAMESPACE_OWNERSHIP:
                raise RuntimeError(f"output namespace ownership is already registered: {marker}")
            _OUTPUT_NAMESPACE_OWNERSHIP[marker] = ownership
        return marker
    except BaseException:
        if marker_identity is not None:
            try:
                if _unlink_path_if_identity(marker, marker_identity):
                    os.fsync(parent_fd)
            except BaseException:
                pass
        raise
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)
        os.close(parent_fd)


def _validate_output_namespace_ownership(marker: Path, *, output: Path) -> _OutputNamespaceOwnership:
    expected_marker = _output_namespace_marker_path(output)
    lexical_marker = Path(os.path.abspath(marker.expanduser()))
    if lexical_marker != expected_marker:
        raise ValueError("output namespace marker does not match the requested artifact")
    with _OUTPUT_NAMESPACE_OWNERSHIP_LOCK:
        ownership = _OUTPUT_NAMESPACE_OWNERSHIP.get(lexical_marker)
    if ownership is None:
        raise RuntimeError("output namespace reservation ownership is not bound to this process")
    if ownership.output_path != output:
        raise RuntimeError("output namespace reservation is bound to another output path")
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd = os.open(
            lexical_marker.parent,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        parent_stat = os.fstat(parent_fd)
        if (parent_stat.st_dev, parent_stat.st_ino) != ownership.parent_identity:
            raise RuntimeError("output namespace parent directory ownership changed")
        descriptor = os.open(
            lexical_marker.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        marker_stat = os.fstat(descriptor)
        if not stat.S_ISREG(marker_stat.st_mode):
            raise RuntimeError("output namespace reservation ownership is not a regular file")
        with os.fdopen(descriptor, mode="rb") as handle:
            descriptor = -1
            payload = parse_json_bytes(handle.read(), label="output namespace marker")
    except (OSError, ArtifactSnapshotError, TypeError) as error:
        raise RuntimeError(f"output namespace reservation ownership is invalid: {lexical_marker}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)
    expected_identity = ownership.marker_identity
    observed_identity = (marker_stat.st_dev, marker_stat.st_ino)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 3
        or payload.get("state") != "in_progress"
        or payload.get("output_artifact_path") != str(output)
        or (payload.get("parent_device"), payload.get("parent_inode")) != ownership.parent_identity
        or (payload.get("marker_device"), payload.get("marker_inode")) != expected_identity
        or observed_identity != expected_identity
    ):
        raise RuntimeError("output namespace reservation ownership metadata does not match its inode")
    try:
        path_stat = os.lstat(lexical_marker)
    except OSError as error:
        raise RuntimeError("output namespace reservation ownership was lost during validation") from error
    if (path_stat.st_dev, path_stat.st_ino) != expected_identity:
        raise RuntimeError("output namespace reservation ownership was replaced during validation")
    return ownership


def _record_output_namespace_publication(receipt: PublicationReceipt, ownership_token: Any) -> None:
    """Bind one publisher-created receipt to its coordinator-owned output namespace."""
    if ownership_token is None:
        return
    if not isinstance(ownership_token, Path):
        raise TypeError("output namespace publication token must be a Path")
    marker = Path(os.path.abspath(ownership_token.expanduser()))
    with _OUTPUT_NAMESPACE_OWNERSHIP_LOCK:
        ownership = _OUTPUT_NAMESPACE_OWNERSHIP.get(marker)
    if ownership is None:
        raise RuntimeError("cannot register a publication without coordinator-owned namespace state")
    _validate_output_namespace_ownership(marker, output=ownership.output_path)
    validate_publication_receipt(receipt)
    published_path = Path(receipt.final_path)
    is_root = published_path == ownership.output_path
    is_sidecar = (
        published_path.parent == ownership.output_path.parent
        and published_path.name.startswith(f"{ownership.output_path.name}.")
        and published_path.name.endswith(".outputs.jsonl.gz")
    )
    if not is_root and not is_sidecar:
        raise RuntimeError("published artifact path is outside its reserved output namespace")
    with _OUTPUT_NAMESPACE_OWNERSHIP_LOCK:
        current = _OUTPUT_NAMESPACE_OWNERSHIP.get(marker)
        if current is not ownership:
            raise RuntimeError("output namespace ownership changed during publication registration")
        if published_path in ownership.publications:
            raise RuntimeError("output namespace already registered this publication path")
        ownership.publications[published_path] = receipt


def _observed_output_namespace_publications(output: Path) -> set[Path]:
    sidecar_prefix = f"{output.name.removesuffix(output.suffix)}."
    observed = set()
    for candidate in output.parent.iterdir():
        if candidate == output or (
            candidate.name.startswith(sidecar_prefix)
            and (
                candidate.name.endswith(".outputs.jsonl.gz")
                or candidate.name.endswith(".outputs.jsonl.gz.tmp")
            )
        ):
            observed.add(candidate)
    return observed


def require_output_namespace_reservation(path: str | Path) -> Path:
    """Fail unless the caller reserved this exact output namespace."""
    output = Path(path).resolve()
    marker = _output_namespace_marker_path(output)
    _validate_output_namespace_ownership(marker, output=output)
    return marker


def register_output_namespace_publication(
    path: str | Path,
    receipt: PublicationReceipt,
) -> None:
    """Register one coordinator-finalized external worker publication."""
    if not isinstance(receipt, PublicationReceipt):
        raise TypeError("external namespace publication must provide a PublicationReceipt")
    marker = require_output_namespace_reservation(path)
    _record_output_namespace_publication(receipt, marker)


def complete_output_namespace(
    marker: str | Path,
    *,
    output_path: str | Path,
    require_final_artifact: bool = True,
) -> None:
    """Release one successful reservation without touching any other artifacts."""
    output = Path(output_path).resolve()
    reservation = Path(os.path.abspath(Path(marker).expanduser()))
    if reservation != _output_namespace_marker_path(output):
        raise ValueError("output namespace marker does not match the requested artifact")
    ownership = _validate_output_namespace_ownership(reservation, output=output)
    with _OUTPUT_NAMESPACE_OWNERSHIP_LOCK:
        publications = dict(ownership.publications)
    if require_final_artifact and output not in publications:
        raise RuntimeError("cannot complete an output namespace without its coordinator-owned final receipt")
    for published_path, receipt in publications.items():
        if Path(receipt.final_path) != published_path:
            raise RuntimeError("output namespace publication key and receipt path differ")
        validate_publication_receipt(receipt)
    observed_publications = _observed_output_namespace_publications(output)
    registered_publications = set(publications)
    if observed_publications != registered_publications:
        missing = sorted(str(path) for path in registered_publications - observed_publications)
        foreign = sorted(str(path) for path in observed_publications - registered_publications)
        raise RuntimeError(
            "output namespace paths differ from coordinator-owned receipts: "
            f"missing={missing}, unregistered={foreign}"
        )
    _validate_output_namespace_ownership(reservation, output=output)
    parent_fd = os.open(
        reservation.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        parent_stat = os.fstat(parent_fd)
        if (parent_stat.st_dev, parent_stat.st_ino) != ownership.parent_identity:
            raise RuntimeError("output namespace parent changed before completion")
        marker_stat = os.stat(reservation.name, dir_fd=parent_fd, follow_symlinks=False)
        if (marker_stat.st_dev, marker_stat.st_ino) != ownership.marker_identity:
            raise RuntimeError("output namespace marker changed before completion")
        os.unlink(reservation.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    for receipt in publications.values():
        validate_publication_receipt(receipt)
    with _OUTPUT_NAMESPACE_OWNERSHIP_LOCK:
        current = _OUTPUT_NAMESPACE_OWNERSHIP.get(reservation)
        if current is not ownership:
            raise RuntimeError("output namespace ownership changed during completion")
        _OUTPUT_NAMESPACE_OWNERSHIP.pop(reservation)


def shared_prefix_manifest_evidence(manifest: WorkloadManifest) -> dict[str, Any]:
    """Validate one physically reusable prompt and return its stable identity."""
    if len(manifest.requests) < 2:
        raise AssertionError("shared-prefix reuse requires at least two requests")
    prompt = manifest.requests[0].prompt_token_ids
    if not prompt:
        raise AssertionError("shared-prefix reuse requires a nonempty prompt")
    if any(request.prompt_token_ids != prompt for request in manifest.requests[1:]):
        raise AssertionError("shared-prefix reuse requires identical prompt token IDs")
    return {
        "identical_prompt_count": len(manifest.requests),
        "prompt_tokens_per_request": len(prompt),
        "prompt_token_ids_sha256": generation_prompt_token_ids_sha256(prompt),
    }


def _validated_attention_kv_groups(
    groups: Any,
    *,
    expected_prefix_tokens: int,
    expected_block_size: int,
    kind: str,
) -> list[dict[str, Any]]:
    if not isinstance(groups, list) or not groups:
        raise AssertionError(f"{kind} must retain at least one attention KV cache group")
    retained = []
    group_ids = set()
    for group in groups:
        if not isinstance(group, dict):
            raise AssertionError(f"{kind} attention KV cache group telemetry is malformed")
        group_id = group.get("kv_cache_group_id")
        if isinstance(group_id, bool) or not isinstance(group_id, int) or group_id < 0:
            raise AssertionError(f"{kind} attention KV cache group ID is malformed")
        if group_id in group_ids:
            raise AssertionError(f"{kind} attention KV cache group IDs must be unique")
        group_ids.add(group_id)
        layer_names = group.get("layer_names")
        if (
            not isinstance(layer_names, list)
            or not layer_names
            or any(not isinstance(name, str) or not name for name in layer_names)
            or len(set(layer_names)) != len(layer_names)
        ):
            raise AssertionError(f"{kind} attention KV layer ownership is malformed")
        block_size = group.get("block_size_tokens")
        if block_size != expected_block_size:
            raise AssertionError(f"{kind} attention KV block size does not match the resolved cache")
        expected_block_count = expected_prefix_tokens // expected_block_size
        if group.get("physical_block_count") != expected_block_count:
            raise AssertionError(f"{kind} attention KV physical block count is not exact")
        block_ids = group.get("physical_block_ids")
        if (
            not isinstance(block_ids, list)
            or len(block_ids) != expected_block_count
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in block_ids)
            or len(set(block_ids)) != len(block_ids)
        ):
            raise AssertionError(f"{kind} attention KV physical block IDs are malformed")
        expected_hash = hashlib.sha256(json.dumps(block_ids, separators=(",", ":")).encode()).hexdigest()
        if group.get("physical_block_ids_sha256") != expected_hash:
            raise AssertionError(f"{kind} attention KV physical block ID hash does not match retained IDs")
        retained.append(dict(group))
    return retained


def _validated_prefix_source(
    stats: dict[str, Any],
    *,
    prompt_tokens: int,
    physically_reused_tokens: int,
    cache_block_size: int,
    expected_cache_misses: int,
) -> dict[str, Any]:
    expected_worker_misses = int(expected_cache_misses > 0)
    if stats.get("cache_miss_count") != expected_worker_misses:
        raise AssertionError("physical worker cache-miss count does not match the exact wave layout")
    miss_ids = stats.get("cache_miss_request_ids")
    if not isinstance(miss_ids, list) or len(miss_ids) != expected_worker_misses:
        raise AssertionError("physical worker cache-miss request IDs are not exact")
    sources = stats.get("prefix_sources")
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
        raise AssertionError("physical worker must retain exactly one direct cache-miss source")
    source = sources[0]
    source_request_id = source.get("request_id")
    if not isinstance(source_request_id, str) or not source_request_id:
        raise AssertionError("direct cache-miss source request ID is malformed")
    if miss_ids and miss_ids != [source_request_id]:
        raise AssertionError("phase cache-miss request ID does not match the retained direct source")
    if source.get("prompt_tokens") != prompt_tokens:
        raise AssertionError("direct cache-miss source prompt length drifted")
    snapshots = source.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise AssertionError("direct cache-miss source has no physical attention block snapshots")
    previous_groups = None
    previous_prefix_tokens = 0
    retained_snapshots = []
    for snapshot_index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, dict) or snapshot.get("snapshot_index") != snapshot_index:
            raise AssertionError("direct cache-miss source snapshot indices must be exact and contiguous")
        for key in ("num_computed_tokens_before_step", "num_scheduled_tokens"):
            value = snapshot.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AssertionError(f"direct cache-miss source {key} is malformed")
        prefix_tokens = snapshot.get("directly_observed_prefix_tokens")
        if (
            isinstance(prefix_tokens, bool)
            or not isinstance(prefix_tokens, int)
            or prefix_tokens <= previous_prefix_tokens
            or prefix_tokens > physically_reused_tokens
            or prefix_tokens % cache_block_size
        ):
            raise AssertionError("direct cache-miss source prefix coverage is malformed")
        if snapshot["num_computed_tokens_before_step"] + snapshot["num_scheduled_tokens"] < prefix_tokens:
            raise AssertionError("direct cache-miss source snapshot exceeds scheduled prefill work")
        groups = _validated_attention_kv_groups(
            snapshot.get("attention_kv_groups"),
            expected_prefix_tokens=prefix_tokens,
            expected_block_size=cache_block_size,
            kind="source",
        )
        if previous_groups is not None:
            if len(previous_groups) != len(groups):
                raise AssertionError("direct cache-miss source attention group count changed")
            for previous, current in zip(previous_groups, groups, strict=True):
                for key in ("kv_cache_group_id", "layer_names", "block_size_tokens"):
                    if previous[key] != current[key]:
                        raise AssertionError("direct cache-miss source attention ownership changed")
                previous_ids = previous["physical_block_ids"]
                if previous_ids != current["physical_block_ids"][: len(previous_ids)]:
                    raise AssertionError("direct cache-miss source physical block identity changed")
        previous_groups = groups
        previous_prefix_tokens = prefix_tokens
        retained_snapshots.append({**snapshot, "attention_kv_groups": groups})
    if previous_prefix_tokens != physically_reused_tokens:
        raise AssertionError("direct cache-miss source does not cover the exact reusable prompt prefix")
    return {**source, "snapshots": retained_snapshots}


_RUNTIME_STATE_LAYOUT_FIELDS = (
    "kv_cache_group_id",
    "layer_name",
    "state_index",
    "dtype",
    "state_shape",
    "block_shape",
    "copied_elements",
    "copied_bytes",
)


def _validated_runtime_state_layout(record: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    layout = record.get("runtime_state_layout")
    if not isinstance(layout, list) or not layout:
        raise AssertionError("physical prefix clone is missing the complete runtime state layout")
    retained = []
    identities = set()
    expected_elements = 0
    expected_bytes = 0
    for entry in layout:
        if not isinstance(entry, dict):
            raise AssertionError("runtime recurrent-state layout entry is malformed")
        group_id = entry.get("kv_cache_group_id")
        layer_name = entry.get("layer_name")
        state_index = entry.get("state_index")
        if (
            isinstance(group_id, bool)
            or not isinstance(group_id, int)
            or group_id < 0
            or not isinstance(layer_name, str)
            or not layer_name
            or isinstance(state_index, bool)
            or not isinstance(state_index, int)
            or state_index < 0
        ):
            raise AssertionError("runtime recurrent-state layout ownership is malformed")
        identity = (group_id, layer_name, state_index)
        if identity in identities:
            raise AssertionError("runtime recurrent-state layout ownership must be unique")
        identities.add(identity)
        if entry.get("dtype") != "torch.float32":
            raise AssertionError("complete Evo2 runtime recurrent-state layout must be FP32")
        state_shape = entry.get("state_shape")
        block_shape = entry.get("block_shape")
        if (
            not isinstance(state_shape, list)
            or len(state_shape) < 2
            or not isinstance(block_shape, list)
            or not block_shape
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in state_shape)
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in block_shape)
            or state_shape[1:] != block_shape
        ):
            raise AssertionError("runtime recurrent-state layout tensor shape is malformed")
        copied_elements = entry.get("copied_elements")
        copied_bytes = entry.get("copied_bytes")
        if (
            isinstance(copied_elements, bool)
            or not isinstance(copied_elements, int)
            or copied_elements <= 0
            or isinstance(copied_bytes, bool)
            or not isinstance(copied_bytes, int)
            or copied_bytes <= 0
            or copied_elements != math.prod(block_shape)
            or copied_bytes != 4 * copied_elements
        ):
            raise AssertionError("runtime recurrent-state layout size is inconsistent with exact FP32 storage")
        expected_elements += copied_elements
        expected_bytes += copied_bytes
        retained.append(
            {
                **entry,
                "state_shape": list(state_shape),
                "block_shape": list(block_shape),
            }
        )
    return retained, {
        "expected_copy_entries": len(retained),
        "expected_copied_elements": expected_elements,
        "expected_copied_bytes": expected_bytes,
    }


def _validated_fp32_state_copies(
    record: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    runtime_layout, direct_layout = _validated_runtime_state_layout(record)
    copies = record.get("state_copies")
    if not isinstance(copies, list) or not copies:
        raise AssertionError("physical prefix clone must retain positive recurrent-state copy entries")
    retained = []
    identities = set()
    copied_elements = 0
    copied_bytes = 0
    for entry in copies:
        if not isinstance(entry, dict):
            raise AssertionError("recurrent-state copy entry is malformed")
        group_id = entry.get("kv_cache_group_id")
        layer_name = entry.get("layer_name")
        state_index = entry.get("state_index")
        if (
            isinstance(group_id, bool)
            or not isinstance(group_id, int)
            or group_id < 0
            or not isinstance(layer_name, str)
            or not layer_name
            or isinstance(state_index, bool)
            or not isinstance(state_index, int)
            or state_index < 0
        ):
            raise AssertionError("recurrent-state copy ownership is malformed")
        identity = (group_id, layer_name, state_index)
        if identity in identities:
            raise AssertionError("recurrent-state copy ownership must be unique per request")
        identities.add(identity)
        if entry.get("dtype") != "torch.float32":
            raise AssertionError("every physical Evo2 recurrent-state copy must be FP32")
        state_shape = entry.get("state_shape")
        block_shape = entry.get("block_shape")
        if (
            not isinstance(state_shape, list)
            or len(state_shape) < 2
            or not isinstance(block_shape, list)
            or not block_shape
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in state_shape)
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in block_shape)
            or state_shape[1:] != block_shape
        ):
            raise AssertionError("recurrent-state copy tensor shape evidence is malformed")
        integer_keys = (
            "source_logical_block_index",
            "destination_logical_block_index",
            "source_physical_block_id",
            "destination_physical_block_id",
            "source_data_ptr",
            "destination_data_ptr",
            "copied_elements",
            "copied_bytes",
        )
        if any(
            isinstance(entry.get(key), bool) or not isinstance(entry.get(key), int) or entry[key] < 0
            for key in integer_keys
        ):
            raise AssertionError("recurrent-state copy pointer, slot, or size evidence is malformed")
        if entry["source_logical_block_index"] == entry["destination_logical_block_index"]:
            raise AssertionError("recurrent-state copy logical source and destination must differ")
        if entry["source_physical_block_id"] == entry["destination_physical_block_id"]:
            raise AssertionError("recurrent-state copy physical source and destination must differ")
        if entry["source_data_ptr"] <= 0 or entry["destination_data_ptr"] <= 0:
            raise AssertionError("recurrent-state copy pointers must be concrete nonzero addresses")
        if entry["source_data_ptr"] == entry["destination_data_ptr"]:
            raise AssertionError("recurrent-state copy source and destination pointers must differ")
        if state_shape[0] <= max(entry["source_physical_block_id"], entry["destination_physical_block_id"]):
            raise AssertionError("recurrent-state physical block IDs exceed the retained tensor shape")
        if entry["copied_elements"] != math.prod(block_shape):
            raise AssertionError("recurrent-state copied elements do not match the retained block shape")
        if entry["copied_bytes"] != 4 * entry["copied_elements"]:
            raise AssertionError("recurrent-state copied bytes do not match exact FP32 storage")
        copied_elements += entry["copied_elements"]
        copied_bytes += entry["copied_bytes"]
        retained.append(dict(entry))
    retained_layout = [{key: entry[key] for key in _RUNTIME_STATE_LAYOUT_FIELDS} for entry in retained]
    runtime_by_identity = {
        (entry["kv_cache_group_id"], entry["layer_name"], entry["state_index"]): entry for entry in runtime_layout
    }
    retained_by_identity = {
        (entry["kv_cache_group_id"], entry["layer_name"], entry["state_index"]): entry for entry in retained_layout
    }
    if retained_by_identity != runtime_by_identity:
        raise AssertionError("physical state copies do not match the complete runtime recurrent-state layout")
    if (
        len(retained) != direct_layout["expected_copy_entries"]
        or copied_elements != direct_layout["expected_copied_elements"]
        or copied_bytes != direct_layout["expected_copied_bytes"]
    ):
        raise AssertionError("physical state-copy totals do not match the complete runtime layout")
    actual_fields = {
        "copy_entries": len(retained),
        "copied_elements": copied_elements,
        "copied_bytes": copied_bytes,
    }
    field_labels = {
        "copy_entries": "copy entry count",
        "copied_elements": "copy elements",
        "copied_bytes": "copy bytes",
    }
    for field, value in actual_fields.items():
        if record.get(field) != value:
            raise AssertionError(f"per-state {field_labels[field]} do not sum to the physical clone total")
    if any(record.get(field) != value for field, value in direct_layout.items()):
        raise AssertionError("self-reported state-copy expectations do not match direct per-state evidence")
    return retained, runtime_layout, direct_layout


def shared_prefix_state_reuse_evidence(
    manifest: WorkloadManifest,
    *,
    cached_tokens: Sequence[int | None],
    worker_proof: Sequence[dict[str, Any]],
    expected_worker_clone_counts: Sequence[int],
    cache_block_size: int,
    request_replica_ranks: Sequence[int] | None = None,
    expected_cache_misses: int = 1,
) -> dict[str, Any]:
    """Prove exact scheduler hits and request-scoped FP32 recurrent-state clones."""
    identity = shared_prefix_manifest_evidence(manifest)
    request_count = len(manifest.requests)
    if len(cached_tokens) != request_count:
        raise AssertionError("cached-token telemetry must cover every request")
    if request_replica_ranks is None:
        normalized_replica_ranks = [0] * request_count
    else:
        if len(request_replica_ranks) != request_count:
            raise AssertionError("replica-rank telemetry must cover every request")
        normalized_replica_ranks = []
        for value in request_replica_ranks:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AssertionError("replica-rank telemetry must contain nonnegative integers")
            normalized_replica_ranks.append(value)
    replica_ranks = sorted(set(normalized_replica_ranks))
    if replica_ranks != list(range(len(replica_ranks))):
        raise AssertionError("active scheduler replica ranks must be contiguous from zero")
    if isinstance(cache_block_size, bool) or not isinstance(cache_block_size, int) or cache_block_size <= 0:
        raise AssertionError("prefix cache block size must be a positive integer")
    if expected_cache_misses < 0 or expected_cache_misses >= request_count:
        raise AssertionError("expected cache misses must leave at least one physical prefix clone")
    if len(expected_worker_clone_counts) != len(worker_proof) or not worker_proof:
        raise AssertionError("expected worker clone counts must cover every physical worker")

    prompt_tokens = identity["prompt_tokens_per_request"]
    physically_reused_tokens = (prompt_tokens - 1) // cache_block_size * cache_block_size
    if physically_reused_tokens <= 0:
        raise AssertionError("the prompt is too short for one block-aligned prefix clone")

    normalized_counts = []
    for value in cached_tokens:
        if value is None or isinstance(value, bool) or not isinstance(value, int):
            raise AssertionError("cached-token telemetry must contain concrete integer counts")
        if not 0 <= value <= prompt_tokens:
            raise AssertionError("cached-token telemetry exceeds the exact prompt length")
        normalized_counts.append(value)

    miss_count = sum(value == 0 for value in normalized_counts)
    if miss_count != expected_cache_misses:
        qualifier = "one" if expected_cache_misses == 1 else str(expected_cache_misses)
        raise AssertionError(f"shared-prefix execution must have exactly {qualifier} cache miss")
    expected_misses_per_replica = int(expected_cache_misses > 0)
    if expected_cache_misses != expected_misses_per_replica * len(replica_ranks):
        raise AssertionError("shared-prefix cache misses must be exactly one per active scheduler replica")
    for replica_rank in replica_ranks:
        replica_counts = [
            cached
            for cached, rank in zip(normalized_counts, normalized_replica_ranks, strict=True)
            if rank == replica_rank
        ]
        replica_miss_count = sum(value == 0 for value in replica_counts)
        if replica_miss_count != expected_misses_per_replica:
            raise AssertionError("shared-prefix cache misses must be independently exact on every scheduler replica")
    hit_counts = [value for value in normalized_counts if value > 0]
    if len(hit_counts) != request_count - expected_cache_misses:
        raise AssertionError("shared-prefix execution did not clone every request after each replica's miss")
    if any(value != physically_reused_tokens for value in hit_counts):
        raise AssertionError("every cache hit must reuse the exact block-aligned prefix")

    worker_clones = []
    expected_elements_per_request = set()
    expected_bytes_per_request = set()
    for proof, expected_clone_count in zip(worker_proof, expected_worker_clone_counts, strict=True):
        if isinstance(expected_clone_count, bool) or not isinstance(expected_clone_count, int):
            raise AssertionError("expected worker clone counts must be integers")
        stats = proof.get("mamba_prefix_clones")
        if not isinstance(stats, dict):
            raise AssertionError("shared-prefix proof is missing request-scoped physical state clones")
        clone_count = stats.get("clone_count")
        requests = stats.get("requests")
        if clone_count != expected_clone_count or not isinstance(requests, list) or len(requests) != clone_count:
            raise AssertionError("physical worker prefix clone count does not match the exact request layout")
        source = _validated_prefix_source(
            stats,
            prompt_tokens=prompt_tokens,
            physically_reused_tokens=physically_reused_tokens,
            cache_block_size=cache_block_size,
            expected_cache_misses=expected_cache_misses,
        )
        source_snapshot = source["snapshots"][-1]
        source_groups = source_snapshot["attention_kv_groups"]

        retained_requests = []
        request_ids = set()
        for record in requests:
            if not isinstance(record, dict):
                raise AssertionError("request-scoped physical clone telemetry is malformed")
            request_id = record.get("request_id")
            if not isinstance(request_id, str) or not request_id or request_id in request_ids:
                raise AssertionError("physical prefix clone request IDs must be unique nonempty strings")
            request_ids.add(request_id)
            for key in (
                "num_computed_tokens",
                "prompt_tokens",
                "block_size",
                "copy_entries",
                "copied_elements",
                "copied_bytes",
                "expected_copy_entries",
                "expected_copied_elements",
                "expected_copied_bytes",
            ):
                value = record.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise AssertionError(f"physical prefix clone {key} telemetry is malformed")
            if record["num_computed_tokens"] != physically_reused_tokens:
                raise AssertionError("physical clone source position does not match the scheduler cache hit")
            if record["prompt_tokens"] != prompt_tokens or record["block_size"] != cache_block_size:
                raise AssertionError("physical clone prompt or block-size provenance drifted")
            if record.get("all_state_dtypes_fp32") is not True:
                raise AssertionError("every cloned Evo2 recurrent state must be FP32")
            if record.get("source_miss_request_id") != source["request_id"]:
                raise AssertionError("physical prefix clone does not name its direct cache-miss source")
            if record.get("source_snapshot_index") != source_snapshot["snapshot_index"]:
                raise AssertionError("physical prefix clone does not name the exact direct source snapshot")
            if record.get("attention_kv_identity_verified") is not True:
                raise AssertionError("physical prefix clone lacks attention KV identity verification")
            retained_source_groups = _validated_attention_kv_groups(
                record.get("source_attention_kv_groups"),
                expected_prefix_tokens=physically_reused_tokens,
                expected_block_size=cache_block_size,
                kind="clone source",
            )
            retained_reused_groups = _validated_attention_kv_groups(
                record.get("reused_attention_kv_groups"),
                expected_prefix_tokens=physically_reused_tokens,
                expected_block_size=cache_block_size,
                kind="clone hit",
            )
            if retained_source_groups != source_groups:
                raise AssertionError("clone source attention KV blocks do not match the direct miss snapshot")
            if retained_reused_groups != retained_source_groups:
                raise AssertionError("clone hit attention KV physical block IDs do not exactly match its source")
            state_copies, runtime_state_layout, direct_layout = _validated_fp32_state_copies(record)
            expected_elements_per_request.add(direct_layout["expected_copied_elements"])
            expected_bytes_per_request.add(direct_layout["expected_copied_bytes"])
            retained_requests.append(
                {
                    **record,
                    **direct_layout,
                    "runtime_state_layout": runtime_state_layout,
                    "source_attention_kv_groups": retained_source_groups,
                    "reused_attention_kv_groups": retained_reused_groups,
                    "state_copies": state_copies,
                }
            )

        worker_clones.append(
            {
                "rank": int(proof.get("rank", 0)),
                "device": int(proof.get("device", 0)),
                "clone_count": clone_count,
                "prefix_source": source,
                "requests": retained_requests,
            }
        )

    if len(expected_elements_per_request) != 1 or len(expected_bytes_per_request) != 1:
        raise AssertionError("all physical prefix clones must retain one exact Evo2 state-copy layout")

    per_request_physical_work = []
    for request, cached, replica_rank in zip(
        manifest.requests,
        normalized_counts,
        normalized_replica_ranks,
        strict=True,
    ):
        per_request_physical_work.append(
            {
                "request_id": request.request_id,
                "replica_rank": replica_rank,
                "prompt_tokens": prompt_tokens,
                "cached_complete_block_tokens": cached,
                "physical_prefill_prompt_tokens": prompt_tokens - cached,
                "cache_status": "miss" if cached == 0 else "hit",
            }
        )

    per_replica_physical_work = []
    for replica_rank in replica_ranks:
        rows = [row for row in per_request_physical_work if row["replica_rank"] == replica_rank]
        miss_rows = [row for row in rows if row["cache_status"] == "miss"]
        hit_rows = [row for row in rows if row["cache_status"] == "hit"]
        per_replica_physical_work.append(
            {
                "replica_rank": replica_rank,
                "request_count": len(rows),
                "cache_miss_request_count": len(miss_rows),
                "cache_hit_request_count": len(hit_rows),
                "cache_miss_physical_prefill_prompt_tokens": sum(
                    row["physical_prefill_prompt_tokens"] for row in miss_rows
                ),
                "cache_hit_physical_prefill_prompt_tokens": sum(
                    row["physical_prefill_prompt_tokens"] for row in hit_rows
                ),
                "physical_prefill_prompt_tokens_total": sum(row["physical_prefill_prompt_tokens"] for row in rows),
            }
        )

    total_prompt_tokens = prompt_tokens * request_count
    cache_miss_physical_tokens = sum(
        row["physical_prefill_prompt_tokens"] for row in per_request_physical_work if row["cache_status"] == "miss"
    )
    cache_hit_physical_tokens = sum(
        row["physical_prefill_prompt_tokens"] for row in per_request_physical_work if row["cache_status"] == "hit"
    )
    physical_prefill_tokens_total = cache_miss_physical_tokens + cache_hit_physical_tokens
    return {
        **identity,
        "cache_block_size": cache_block_size,
        "cached_tokens_by_request": normalized_counts,
        "cache_hit_request_count": len(hit_counts),
        "cache_miss_request_count": miss_count,
        "logical_clone_request_count": len(hit_counts),
        "physically_reused_prompt_tokens_per_clone": physically_reused_tokens,
        "recomputed_prompt_tokens_per_clone": prompt_tokens - physically_reused_tokens,
        "total_cached_prompt_tokens": sum(normalized_counts),
        "scheduled_uncached_prompt_tokens": total_prompt_tokens - sum(normalized_counts),
        "physical_prefill_prompt_tokens_by_request": per_request_physical_work,
        "physical_prefill_prompt_tokens_by_replica": per_replica_physical_work,
        "cache_miss_physical_prefill_prompt_tokens": cache_miss_physical_tokens,
        "cache_hit_physical_prefill_prompt_tokens": cache_hit_physical_tokens,
        "physical_prefill_prompt_tokens_total": physical_prefill_tokens_total,
        "physical_prefix_reuse_scope": {
            "resolved_cache_block_size_tokens": cache_block_size,
            "attention_kv_reused_complete_blocks_per_hit": physically_reused_tokens // cache_block_size,
            "attention_kv_reused_tokens_per_hit": physically_reused_tokens,
            "fp32_hyena_state_clone_position_tokens": physically_reused_tokens,
            "partial_block_tail_recomputed_tokens_per_hit": prompt_tokens - physically_reused_tokens,
            "full_prompt_attention_kv_and_state_cloned": physically_reused_tokens == prompt_tokens,
        },
        "worker_state_clones": worker_clones,
        "rank_local_physical_clone_count": sum(worker["clone_count"] for worker in worker_clones),
        "expected_fp32_state_copy_elements_per_request": next(iter(expected_elements_per_request)),
        "expected_fp32_state_copy_bytes_per_request": next(iter(expected_bytes_per_request)),
        "attention_kv_physical_reuse_proven": True,
        "physical_state_copy_proven": True,
    }


def wave_execution_summary(
    wave_proofs: Sequence[dict[str, Any]],
    *,
    target_request_count: int = 96,
) -> dict[str, Any]:
    """Retain actual physical calls and measured wall time needed to cover a target batch."""
    if target_request_count <= 0:
        raise ValueError("target_request_count must be positive")
    if not wave_proofs:
        raise AssertionError("wave execution summary requires at least one physical generation call")

    request_counts = []
    generation_s = []
    covered_requests = 0
    measured_waves_to_target = None
    measured_time_to_target_s = None
    requests_completed_at_target_boundary = None
    for expected_index, proof in enumerate(wave_proofs):
        if proof.get("wave_index") != expected_index:
            raise AssertionError("physical generation wave indices must be exact and contiguous")
        request_count = proof.get("request_count")
        elapsed = proof.get("generation_s")
        if isinstance(request_count, bool) or not isinstance(request_count, int) or request_count <= 0:
            raise AssertionError("physical generation wave request counts must be positive integers")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise AssertionError("physical generation wave timings must be finite and positive")
        request_counts.append(request_count)
        generation_s.append(float(elapsed))
        covered_requests += request_count
        if measured_waves_to_target is None and covered_requests >= target_request_count:
            measured_waves_to_target = expected_index + 1
            measured_time_to_target_s = sum(generation_s)
            requests_completed_at_target_boundary = covered_requests

    return {
        "target_request_count": target_request_count,
        "actual_call_count": len(wave_proofs),
        "actual_request_count": sum(request_counts),
        "call_request_counts": request_counts,
        "call_generation_s": generation_s,
        "measured_waves_to_target": measured_waves_to_target,
        "measured_time_to_target_s": measured_time_to_target_s,
        "requests_completed_at_target_boundary": requests_completed_at_target_boundary,
    }


def mbs1_exact1k_phase_evidence(
    phase: Mapping[str, Any],
    *,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    phase_coordinate: Mapping[str, Any],
    proof_collected: bool,
    linked_proof_artifact: str | Path | None,
) -> dict[str, Any]:
    """Reconstruct first-fill, warmed-repeat, tail, and exact-1k timing from physical calls."""
    contract = mbs1_exact1k_contract(manifest, profile)
    if type(proof_collected) is not bool:
        raise TypeError("proof_collected must be a built-in bool")
    if not proof_collected and linked_proof_artifact is None:
        raise ValueError("MBS=1 speed evidence requires its linked proof artifact")
    expected_coordinate_fields = {
        "phase",
        "sample_index",
        "generation_round",
        "global_call_index_start",
        "global_request_index_start",
        "physical_calls_per_round",
        "semantic_request_count",
    }
    if type(phase_coordinate) is not dict or set(phase_coordinate) != expected_coordinate_fields:
        raise AssertionError("MBS=1 phase coordinate fields are not exact")
    if phase.get("phase") != phase_coordinate["phase"]:
        raise AssertionError("MBS=1 phase name differs from its caller coordinate")
    retained_waves = phase.get("wave_proofs") if profile.topology == "tp2" else phase.get("waves")
    if type(retained_waves) is not list or len(retained_waves) != 11:
        raise AssertionError("MBS=1 phase must retain exactly eleven physical calls")
    expected_counts = [96] * 10 + [40]
    timings = []
    for wave_index, (wave, expected_count) in enumerate(zip(retained_waves, expected_counts, strict=True)):
        if type(wave) is not dict:
            raise AssertionError("MBS=1 physical wave evidence is malformed")
        elapsed = wave.get("generation_s")
        if (
            wave.get("wave_index") != wave_index
            or wave.get("request_count") != expected_count
            or wave.get("generation_round") != phase_coordinate["generation_round"]
            or wave.get("call_index") != phase_coordinate["global_call_index_start"] + wave_index
            or type(elapsed) not in (int, float)
            or isinstance(elapsed, bool)
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise AssertionError("MBS=1 physical wave coordinates or timing drifted")
        timings.append(float(elapsed))

    expected_executions = (
        build_wave_execution_records(
            manifest,
            global_wave_size=profile.global_wave_size,
            generation_round=phase_coordinate["generation_round"],
            call_index_start=phase_coordinate["global_call_index_start"],
            global_request_index_start=phase_coordinate["global_request_index_start"],
        )
        if profile.topology == "tp2"
        else _expected_dp2_executions(
            manifest,
            profile=profile,
            generation_round=phase_coordinate["generation_round"],
            call_index_start=phase_coordinate["global_call_index_start"],
            global_index_start=phase_coordinate["global_request_index_start"],
        )
    )
    _validate_request_execution_rows(phase.get("request_executions"), expected=expected_executions)

    first_misses = None
    first_hits = None
    second_misses = None
    second_hits = None
    tail_misses = None
    tail_hits = None
    if proof_collected:
        if profile.topology == "tp2":
            prefix = _require_dict(
                phase.get("shared_prefix_state_reuse"),
                label="MBS=1 shared-prefix proof",
            )
            cached = _require_list(
                prefix.get("cached_tokens_by_request"),
                label="MBS=1 cached-token rows",
            )
            if len(cached) != 1_000:
                raise AssertionError("MBS=1 cached-token proof omitted semantic requests")
            sections = (cached[:96], cached[96:192], cached[-40:])
            counts = tuple(
                (sum(value == 0 for value in section), sum(value > 0 for value in section))
                for section in sections
            )
            (first_misses, first_hits), (second_misses, second_hits), (tail_misses, tail_hits) = counts
        else:
            prefix_rows = [
                _require_dict(wave.get("shared_prefix_state_reuse"), label="MBS=1 DP2 prefix wave")
                for wave in retained_waves
            ]
            first_misses = prefix_rows[0].get("cache_miss_request_count")
            first_hits = prefix_rows[0].get("cache_hit_request_count")
            second_misses = prefix_rows[1].get("cache_miss_request_count")
            second_hits = prefix_rows[1].get("cache_hit_request_count")
            tail_misses = prefix_rows[-1].get("cache_miss_request_count")
            tail_hits = prefix_rows[-1].get("cache_hit_request_count")
        if (
            (first_misses, first_hits) != (profile.replica_count, 96 - profile.replica_count)
            or (second_misses, second_hits) != (0, 96)
            or (tail_misses, tail_hits) != (0, 40)
        ):
            raise AssertionError("MBS=1 prefix cache did not progress from primer to warm hits")
        proof_source = "current-proof-run"
    else:
        proof_source = "linked-proof-artifact"
    if phase.get("prefix_cache_reset") is not True:
        raise AssertionError("MBS=1 phase must reset prefix state before its first physical wave")
    return {
        "schema_version": 1,
        "contract": contract,
        "phase": phase_coordinate["phase"],
        "generation_round": phase_coordinate["generation_round"],
        "global_call_index_start": phase_coordinate["global_call_index_start"],
        "global_request_index_start": phase_coordinate["global_request_index_start"],
        "semantic_request_count": 1_000,
        "physical_call_count": 11,
        "first_96": {
            "generation_s": timings[0],
            "includes_prefix_materialization": True,
            "cache_miss_request_count": first_misses,
            "cache_hit_request_count": first_hits,
        },
        "warmed_repeat_96": {
            "generation_s": timings[1],
            "cache_miss_request_count": second_misses,
            "cache_hit_request_count": second_hits,
        },
        "tail_40": {
            "generation_s": timings[-1],
            "cache_miss_request_count": tail_misses,
            "cache_hit_request_count": tail_hits,
        },
        "exact_1000_generation_s": sum(timings),
        "prefix_proof_source": proof_source,
        "token_equality_between_phases_required": False,
        "passed": True,
    }


def attach_mbs1_exact1k_evidence(
    phase_artifacts: list[dict[str, Any]],
    *,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    generation_round_start: int,
    warmups: int,
    repetitions: int,
    enabled: bool,
    proof_collected: bool,
    linked_proof_artifact: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Attach caller-reconstructed primary MBS=1 timing and prefix evidence."""
    if type(enabled) is not bool:
        raise TypeError("MBS=1 evidence enablement must be a built-in bool")
    if not enabled:
        if any(phase.get("mbs1_exact1k") is not None for phase in phase_artifacts):
            raise AssertionError("disabled MBS=1 audit retained unexpected phase evidence")
        return phase_artifacts, None
    coordinates = benchmark_phase_coordinates(
        manifest,
        profile,
        generation_round_start=generation_round_start,
        warmups=warmups,
        repetitions=repetitions,
    )
    if len(phase_artifacts) != len(coordinates):
        raise AssertionError("MBS=1 phase inventory differs from its caller coordinate schedule")
    retained_phases = []
    summaries = []
    global_indices: set[int] = set()
    request_seeds: set[int] = set()
    for phase, coordinate in zip(phase_artifacts, coordinates, strict=True):
        retained = dict(phase)
        evidence = mbs1_exact1k_phase_evidence(
            retained,
            manifest=manifest,
            profile=profile,
            phase_coordinate=coordinate,
            proof_collected=proof_collected,
            linked_proof_artifact=linked_proof_artifact,
        )
        executions = _require_list(retained.get("request_executions"), label="MBS=1 request executions")
        phase_global_indices = {row.get("global_request_index") for row in executions}
        phase_request_seeds = {row.get("seed") for row in executions}
        if (
            len(phase_global_indices) != 1_000
            or len(phase_request_seeds) != 1_000
            or global_indices & phase_global_indices
            or request_seeds & phase_request_seeds
        ):
            raise AssertionError("MBS=1 benchmark phases reused caller request or RNG coordinates")
        global_indices.update(phase_global_indices)
        request_seeds.update(phase_request_seeds)
        retained["mbs1_exact1k"] = evidence
        retained_phases.append(retained)
        summaries.append(
            {
                "phase": evidence["phase"],
                "generation_round": evidence["generation_round"],
                "global_call_index_start": evidence["global_call_index_start"],
                "global_request_index_start": evidence["global_request_index_start"],
                "first_96_generation_s": evidence["first_96"]["generation_s"],
                "warmed_repeat_96_generation_s": evidence["warmed_repeat_96"]["generation_s"],
                "tail_40_generation_s": evidence["tail_40"]["generation_s"],
                "exact_1000_generation_s": evidence["exact_1000_generation_s"],
            }
        )
    return retained_phases, {
        "schema_version": 1,
        "contract": mbs1_exact1k_contract(manifest, profile),
        "phase_count": len(summaries),
        "semantic_request_count_per_phase": 1_000,
        "total_semantic_request_occurrences": len(summaries) * 1_000,
        "phase_coordinates_disjoint": True,
        "token_equality_between_phases_required": False,
        "phases": summaries,
        "passed": True,
    }


def rank_local_generation_contract_sha256(
    phase: str,
    semantic_namespace: str,
    wave_index: int,
    manifest: WorkloadManifest,
    execution_records: Sequence[RequestExecutionRecord],
) -> str:
    """Seal one physical TP wave to caller-owned prompts and RNG coordinates."""
    if type(phase) is not str or not phase:
        raise TypeError("rank-local generation phase must be a nonempty string")
    if type(semantic_namespace) is not str or not semantic_namespace:
        raise TypeError("rank-local semantic namespace must be a nonempty string")
    if type(wave_index) is not int or wave_index < 0:
        raise TypeError("rank-local wave index must be a nonnegative built-in integer")
    if len(execution_records) != len(manifest.requests) or not execution_records:
        raise ValueError("rank-local generation executions must cover one nonempty wave")
    requests = []
    for request, execution in zip(manifest.requests, execution_records, strict=True):
        if request.request_id != execution.request_id:
            raise ValueError("rank-local generation execution order differs from the manifest")
        requests.append(
            {
                "request_id": request.request_id,
                "qualified_request_identity": [
                    semantic_namespace,
                    request.request_id,
                ],
                "prompt_token_ids_sha256": generation_prompt_token_ids_sha256(
                    request.prompt_token_ids
                ),
                "execution": execution.to_dict(),
            }
        )
    payload = {
        "schema_version": "evo2-rank-local-generation-contract/v1",
        "phase": phase,
        "semantic_namespace": semantic_namespace,
        "wave_index": wave_index,
        "manifest_sha256": manifest.sha256,
        "requested_max_new_tokens": manifest.max_new_tokens,
        "requests": requests,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def rank_local_selected_stream_rows(outputs: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    """Derive selected token/logprob stream hashes from returned production outputs."""
    from bionemo.evo2.vllm.worker import selected_stream_sha256

    rows = []
    for output in outputs:
        request_id = getattr(output, "request_id", None)
        completions = getattr(output, "outputs", None)
        if type(request_id) is not str or not request_id:
            raise RuntimeError("vLLM output omitted its internal request ID")
        if type(completions) is not list or len(completions) != 1:
            raise RuntimeError("vLLM output must contain exactly one completion")
        completion = completions[0]
        token_ids = getattr(completion, "token_ids", None)
        positions = getattr(completion, "logprobs", None)
        if type(token_ids) not in (list, tuple) or any(
            type(token_id) is not int for token_id in token_ids
        ):
            raise RuntimeError("vLLM output token IDs are malformed")
        if type(positions) is not list or len(positions) != len(token_ids):
            raise RuntimeError("vLLM output chosen-token logprobs are incomplete")
        logprob_bits = []
        for token_id, position in zip(token_ids, positions, strict=True):
            if type(position) is not dict or token_id not in position:
                raise RuntimeError("vLLM output omitted a chosen-token logprob")
            value = getattr(position[token_id], "logprob", None)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError("vLLM output chosen-token logprob is not numeric")
            value = float(value)
            if not math.isfinite(value):
                raise RuntimeError("vLLM output chosen-token logprob is not finite")
            logprob_bits.append(struct.pack("<f", value).hex())
        token_id_list = list(token_ids)
        rows.append(
            {
                "vllm_request_id": request_id,
                "token_count": len(token_id_list),
                "selected_stream_sha256": selected_stream_sha256(
                    request_id,
                    token_id_list,
                    logprob_bits,
                ),
            }
        )
    if len({row["vllm_request_id"] for row in rows}) != len(rows):
        raise RuntimeError("vLLM output internal request IDs are not unique")
    return tuple(rows)


def validate_rank_local_generation_evidence(
    *,
    phase: str,
    semantic_namespace: str,
    wave_index: int,
    contract_sha256: str,
    expected_tensor_parallel_size: int,
    expected_request_count: int,
    expected_max_new_tokens: int,
    begin_evidence: Sequence[dict[str, Any]],
    rank_evidence: Sequence[dict[str, Any]],
    manifest: WorkloadManifest,
    execution_records: Sequence[RequestExecutionRecord],
    outputs: Sequence[Any],
) -> dict[str, Any]:
    """Require every TP rank to reconstruct the returned selected-token streams."""
    if type(expected_tensor_parallel_size) is not int or expected_tensor_parallel_size <= 0:
        raise TypeError("expected tensor-parallel size must be a positive built-in integer")
    expected_contract = rank_local_generation_contract_sha256(
        phase,
        semantic_namespace,
        wave_index,
        manifest,
        execution_records,
    )
    if contract_sha256 != expected_contract:
        raise RuntimeError("rank-local contract digest differs from caller-owned coordinates")
    expected_ranks = list(range(expected_tensor_parallel_size))
    if len(begin_evidence) != expected_tensor_parallel_size:
        raise RuntimeError("rank-local begin evidence does not cover every TP rank")
    begin_by_rank = {}
    expected_begin_fields = {
        "tp_rank",
        "phase",
        "expected_envelope_sha256",
        "expected_request_count",
        "expected_max_new_tokens",
        "source",
    }
    for item in begin_evidence:
        if type(item) is not dict or set(item) != expected_begin_fields:
            raise RuntimeError("rank-local begin evidence fields are not exact")
        rank = item["tp_rank"]
        if type(rank) is not int or rank in begin_by_rank:
            raise RuntimeError("rank-local begin TP rank set is malformed")
        if item != {
            "tp_rank": rank,
            "phase": phase,
            "expected_envelope_sha256": contract_sha256,
            "expected_request_count": expected_request_count,
            "expected_max_new_tokens": expected_max_new_tokens,
            "source": "rank_local_model_runner_execute_or_sample",
        }:
            raise RuntimeError("rank-local begin evidence differs from the caller contract")
        begin_by_rank[rank] = item
    if sorted(begin_by_rank) != expected_ranks:
        raise RuntimeError("rank-local begin evidence TP ranks are incomplete")

    outer_internal_rows = list(rank_local_selected_stream_rows(outputs))
    if (
        len(outer_internal_rows) != expected_request_count
        or len(manifest.requests) != expected_request_count
        or len(execution_records) != expected_request_count
        or sum(row["token_count"] for row in outer_internal_rows)
        != expected_request_count * expected_max_new_tokens
    ):
        raise RuntimeError("returned vLLM streams do not cover the sealed wave")
    semantic_rows = []
    for request_ordinal, (request, execution, stream) in enumerate(
        zip(
            manifest.requests,
            execution_records,
            outer_internal_rows,
            strict=True,
        )
    ):
        if request.request_id != execution.request_id:
            raise RuntimeError("rank-local semantic request order differs from execution records")
        semantic_rows.append(
            {
                "semantic_namespace": semantic_namespace,
                "local_request_id": request.request_id,
                "qualified_request_identity": [
                    semantic_namespace,
                    request.request_id,
                ],
                "request_ordinal": request_ordinal,
                "wave_index": wave_index,
                "global_request_index": execution.global_request_index,
                "generation_round": execution.generation_round,
                "global_call_index": execution.call_index,
                "dp_rank": execution.dp_rank,
                "seed": execution.seed,
                **stream,
            }
        )
    outer_aggregate = hashlib.sha256(
        json.dumps(semantic_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if sum(
        row["token_count"] for row in semantic_rows
    ) != expected_request_count * expected_max_new_tokens:
        raise RuntimeError("returned vLLM streams do not cover the sealed wave")

    expected_snapshot_fields = {
        "schema_version",
        "source",
        "tp_rank",
        "phase",
        "expected_envelope_sha256",
        "request_count",
        "generated_token_count",
        "execution_call_count",
        "request_order",
        "requests",
        "aggregate_selected_stream_sha256",
    }
    snapshots_by_rank = {}
    for item in rank_evidence:
        if type(item) is not dict or set(item) != expected_snapshot_fields:
            raise RuntimeError("rank-local snapshot evidence fields are not exact")
        rank = item["tp_rank"]
        if type(rank) is not int or rank in snapshots_by_rank:
            raise RuntimeError("rank-local snapshot TP rank set is malformed")
        if (
            item["schema_version"] != "evo2-rank-local-generation-evidence/v1"
            or item["source"] != "rank_local_model_runner_execute_or_sample"
            or item["phase"] != phase
            or item["expected_envelope_sha256"] != contract_sha256
            or item["request_count"] != expected_request_count
            or item["generated_token_count"]
            != expected_request_count * expected_max_new_tokens
            or type(item["execution_call_count"]) is not int
            or item["execution_call_count"] <= 0
        ):
            raise RuntimeError("rank-local TP stream evidence differs from returned output")
        request_order = item["request_order"]
        requests = item["requests"]
        if (
            type(request_order) is not list
            or any(type(request_id) is not str for request_id in request_order)
            or len(request_order) != expected_request_count
            or len(set(request_order)) != expected_request_count
            or type(requests) is not list
            or len(requests) != expected_request_count
            or [request.get("vllm_request_id") for request in requests]
            != request_order
        ):
            raise RuntimeError("rank-local TP witness request order is malformed")
        observed_by_id = {
            request["vllm_request_id"]: request
            for request in requests
            if type(request) is dict
            and set(request)
            == {"vllm_request_id", "token_count", "selected_stream_sha256"}
        }
        expected_by_id = {
            request["vllm_request_id"]: {
                "vllm_request_id": request["vllm_request_id"],
                "token_count": request["token_count"],
                "selected_stream_sha256": request["selected_stream_sha256"],
            }
            for request in outer_internal_rows
        }
        rank_aggregate = hashlib.sha256(
            json.dumps(requests, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if (
            observed_by_id != expected_by_id
            or item["aggregate_selected_stream_sha256"] != rank_aggregate
        ):
            raise RuntimeError("rank-local TP stream evidence differs from returned output")
        snapshots_by_rank[rank] = item
    if sorted(snapshots_by_rank) != expected_ranks:
        raise RuntimeError("rank-local snapshot evidence TP ranks are incomplete")
    if len({item["execution_call_count"] for item in snapshots_by_rank.values()}) != 1:
        raise RuntimeError("rank-local TP ranks observed different execution call counts")
    if len({tuple(item["request_order"]) for item in snapshots_by_rank.values()}) != 1:
        raise RuntimeError("rank-local TP ranks observed different scheduler request order")
    return {
        "schema_version": "evo2-rank-local-generation-validation/v1",
        "passed": True,
        "phase": phase,
        "semantic_namespace": semantic_namespace,
        "wave_index": wave_index,
        "contract_sha256": contract_sha256,
        "tensor_parallel_ranks": expected_ranks,
        "request_count": expected_request_count,
        "generated_token_count": expected_request_count * expected_max_new_tokens,
        "aggregate_selected_stream_sha256": outer_aggregate,
        "semantic_streams": semantic_rows,
        "rank_evidence": [snapshots_by_rank[rank] for rank in expected_ranks],
    }


@dataclass(frozen=True)
class GenerationPhaseResult:
    """One timed generation phase plus its unreset CUDA graph observations."""

    phase: str
    sample: BenchmarkSample
    generation_call_s: tuple[float, ...]
    wave_proofs: tuple[dict[str, Any], ...]
    observations: tuple[dict[str, Any], ...]
    output_summaries: tuple[dict[str, Any], ...]
    request_executions: tuple[RequestExecutionRecord, ...]
    full_output_artifact: dict[str, Any]
    full_decode_proof: dict[str, Any] | None
    worker_proof: tuple[dict[str, Any], ...]
    shared_prefix_state_reuse: dict[str, Any] | None
    proof_collected: bool
    prefix_cache_reset: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe phase record."""
        retained_observations = list(self.observations)
        return {
            "phase": self.phase,
            "sample": self.sample.to_dict(),
            "generation_call_s": list(self.generation_call_s),
            "generation_timing_authority": COORDINATOR_GENERATION_TIMING_AUTHORITY,
            "coordinator_generation_wall_s": self.sample.generation_s,
            "wave_proofs": list(self.wave_proofs),
            "wave_execution": wave_execution_summary(self.wave_proofs),
            "cudagraph_observation_count": len(self.observations),
            "cudagraph_observations_retained": retained_observations,
            "cudagraph_summary": summarize_cudagraph_observations(self.observations),
            "outputs": list(self.output_summaries),
            "request_executions": [record.to_dict() for record in self.request_executions],
            "full_output_artifact": self.full_output_artifact,
            "full_decode_proof": self.full_decode_proof,
            "worker_proof": list(self.worker_proof),
            "shared_prefix_state_reuse": self.shared_prefix_state_reuse,
            "proof_collected": self.proof_collected,
            "prefix_cache_reset": self.prefix_cache_reset,
        }


def summarize_cudagraph_observations(observations: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Aggregate graph-mode observations without emitting one row per decode token."""
    counts = Counter(
        (
            item["engine_index"],
            item["runtime_mode"],
            item["num_unpadded_tokens"],
            item["num_padded_tokens"],
            item["num_paddings"],
        )
        for item in observations
    )
    return [
        {
            "engine_index": engine_index,
            "runtime_mode": runtime_mode,
            "num_unpadded_tokens": unpadded,
            "num_padded_tokens": padded,
            "num_paddings": paddings,
            "count": count,
        }
        for (engine_index, runtime_mode, unpadded, padded, paddings), count in sorted(counts.items())
    ]


def run_generation_phase(
    *,
    llm: Any,
    manifest: WorkloadManifest,
    sampling_params: list[Any],
    phase: str,
    sample_index: int,
    recorder: CUDAGraphProofRecorder,
    memory_monitor_factory: Callable[[], PeakMemoryMonitor],
    execution_records: tuple[RequestExecutionRecord, ...],
    full_output_path: str | Path,
    namespace_output_path: str | Path | None = None,
    collect_proof: bool = True,
    reset_worker_proof: Callable[[], Any] | None = None,
    snapshot_worker_proof: Callable[[], tuple[dict[str, Any], ...]] | None = None,
    require_rank_local_evidence: bool = False,
    rank_local_semantic_namespace: str | None = None,
    expected_tensor_parallel_size: int | None = None,
    begin_rank_local_evidence: Callable[..., Sequence[dict[str, Any]]] | None = None,
    snapshot_rank_local_evidence: Callable[[], Sequence[dict[str, Any]]] | None = None,
    abort_rank_local_evidence: Callable[[], Any] | None = None,
    prefix_cache_block_size: int | None = None,
    require_shared_prefix_state_reuse: bool = False,
    global_wave_size: int | None = None,
    scheduler_max_num_seqs: int | None = None,
    decode_output_token_ids: Callable[[Sequence[int]], str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    barrier: Any | None = None,
) -> GenerationPhaseResult:
    """Time explicit offline vLLM calls while preserving one ordered phase artifact."""

    def require_namespace_ownership() -> Path | None:
        if namespace_output_path is None:
            return None
        return require_output_namespace_reservation(namespace_output_path)

    require_namespace_ownership()
    if collect_proof and recorder is None:
        raise ValueError("proof collection requires a CUDA graph recorder")
    if type(require_rank_local_evidence) is not bool:
        raise TypeError("require_rank_local_evidence must be a built-in bool")
    if require_rank_local_evidence:
        if not collect_proof:
            raise ValueError("rank-local evidence is restricted to the proof lane")
        if type(rank_local_semantic_namespace) is not str or not rank_local_semantic_namespace:
            raise ValueError("rank-local evidence requires a semantic namespace")
        if type(expected_tensor_parallel_size) is not int or expected_tensor_parallel_size <= 0:
            raise ValueError("rank-local evidence requires a positive tensor-parallel size")
        if not all(
            callable(callback)
            for callback in (
                begin_rank_local_evidence,
                snapshot_rank_local_evidence,
                abort_rank_local_evidence,
            )
        ):
            raise ValueError("rank-local evidence requires begin, snapshot, and abort callbacks")
    if len(sampling_params) != len(manifest.requests):
        raise ValueError("sampling params must align with every request")
    if len(execution_records) != len(manifest.requests):
        raise ValueError("execution records must align with every request")
    for request, params, execution in zip(manifest.requests, sampling_params, execution_records, strict=True):
        if execution.request_id != request.request_id:
            raise ValueError("execution record request IDs must preserve manifest order")
        if params.seed != execution.seed:
            raise ValueError("sampling parameter seeds must match persisted execution records")
    wave_size = len(manifest.requests) if global_wave_size is None else global_wave_size
    waves = build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=wave_size,
        replica_count=1,
    )
    if scheduler_max_num_seqs is not None and scheduler_max_num_seqs <= 0:
        raise ValueError("scheduler_max_num_seqs must be positive")
    generation_rounds = {record.generation_round for record in execution_records}
    if len(generation_rounds) != 1:
        raise ValueError("all execution records in one phase must share one semantic generation round")
    generation_round = next(iter(generation_rounds))
    call_index_start = execution_records[0].call_index
    for wave in waves:
        call_indexes = {record.call_index for record in execution_records[wave.start : wave.stop]}
        expected_call_index = call_index_start + wave.wave_index
        if call_indexes != {expected_call_index}:
            raise ValueError("execution call indices must match explicit generation waves")

    observation_start = 0 if recorder is None else len(recorder.observations)
    if collect_proof and reset_worker_proof is not None:
        reset_worker_proof()
    prefix_cache_reset = False
    if require_shared_prefix_state_reuse:
        shared_prefix_manifest_evidence(manifest)
        reset_prefix_cache = getattr(llm, "reset_prefix_cache", None)
        if reset_prefix_cache is None or reset_prefix_cache() is not True:
            raise AssertionError("shared-prefix execution requires a successful phase-local prefix-cache reset")
        prefix_cache_reset = True

    outputs = []
    generation_call_s = []
    wave_proofs = []
    latest_worker_proof: tuple[dict[str, Any], ...] = ()
    monitor_context = memory_monitor_factory() if collect_proof else _UnmonitoredMemory()
    with monitor_context as monitor:
        for wave in waves:
            wave_phase = f"{phase}.wave-{wave.wave_index:03d}"
            wave_manifest = manifest.request_slice(wave.start, wave.stop)
            wave_execution_records = execution_records[wave.start : wave.stop]
            wave_prompts = [{"prompt_token_ids": list(request.prompt_token_ids)} for request in wave_manifest.requests]
            if collect_proof:
                recorder.start_phase(wave_phase)
                wave_observation_start = len(recorder.observations)
                wave_scheduler_start = len(recorder.scheduler_observations)
            if barrier is not None:
                barrier.wait()
            rank_local_validation = None
            rank_local_started = False
            try:
                if require_rank_local_evidence:
                    wave_contract_sha256 = rank_local_generation_contract_sha256(
                        wave_phase,
                        rank_local_semantic_namespace,
                        wave.wave_index,
                        wave_manifest,
                        wave_execution_records,
                    )
                    rank_local_started = True
                    begin_evidence = tuple(
                        begin_rank_local_evidence(
                            phase=wave_phase,
                            contract_sha256=wave_contract_sha256,
                            expected_request_count=wave.request_count,
                            expected_max_new_tokens=manifest.max_new_tokens,
                        )
                    )
                begin = clock()
                wave_outputs = list(
                    llm.generate(
                        wave_prompts,
                        sampling_params[wave.start : wave.stop],
                        use_tqdm=False,
                    )
                )
                elapsed = clock() - begin
                if barrier is not None:
                    barrier.wait()
                require_namespace_ownership()
                if require_rank_local_evidence:
                    rank_evidence = tuple(snapshot_rank_local_evidence())
                    rank_local_validation = validate_rank_local_generation_evidence(
                        phase=wave_phase,
                        semantic_namespace=rank_local_semantic_namespace,
                        wave_index=wave.wave_index,
                        contract_sha256=wave_contract_sha256,
                        expected_tensor_parallel_size=expected_tensor_parallel_size,
                        expected_request_count=wave.request_count,
                        expected_max_new_tokens=manifest.max_new_tokens,
                        begin_evidence=begin_evidence,
                        rank_evidence=rank_evidence,
                        manifest=wave_manifest,
                        execution_records=wave_execution_records,
                        outputs=wave_outputs,
                    )
                    rank_local_started = False
            finally:
                if rank_local_started:
                    abort_rank_local_evidence()
            if len(wave_outputs) != wave.request_count:
                raise AssertionError("vLLM output count must match the explicit generation wave")
            if collect_proof and snapshot_worker_proof is not None:
                latest_worker_proof = snapshot_worker_proof()
            generation_call_s.append(elapsed)
            outputs.extend(wave_outputs)

            full_decode = None
            scheduler_proof = None
            scheduler_observations = None
            if collect_proof:
                full_decode = full_decode_proof_summary(
                    recorder.observations[wave_observation_start:],
                    phase=wave_phase,
                    batch_size=wave.request_count,
                    max_new_tokens=manifest.max_new_tokens,
                )
                scheduler_observations = list(recorder.scheduler_observations[wave_scheduler_start:])
                scheduler_proof = scheduler_capacity_proof_summary(
                    scheduler_observations,
                    phase=wave_phase,
                    global_wave_size=wave.request_count,
                    max_num_seqs=(wave.request_count if scheduler_max_num_seqs is None else scheduler_max_num_seqs),
                )
                if scheduler_max_num_seqs is not None:
                    validate_scheduler_capacity_proof(scheduler_proof)
            wave_proofs.append(
                {
                    "wave_index": wave.wave_index,
                    "start": wave.start,
                    "stop": wave.stop,
                    "request_count": wave.request_count,
                    "generation_round": generation_round,
                    "call_index": call_index_start + wave.wave_index,
                    "generation_s": elapsed,
                    "full_decode_proof": full_decode,
                    "scheduler_observations": scheduler_observations,
                    "scheduler_capacity_proof": scheduler_proof,
                    "rank_local_generation_evidence": rank_local_validation,
                }
            )
    require_namespace_ownership()
    worker_proof = latest_worker_proof
    shared_prefix_reuse = None
    if collect_proof and require_shared_prefix_state_reuse:
        if prefix_cache_block_size is None:
            raise AssertionError("shared-prefix proof requires the resolved cache block size")
        shared_prefix_reuse = shared_prefix_state_reuse_evidence(
            manifest,
            cached_tokens=tuple(getattr(output, "num_cached_tokens", None) for output in outputs),
            worker_proof=worker_proof,
            expected_worker_clone_counts=tuple(len(manifest.requests) - 1 for _ in worker_proof),
            cache_block_size=prefix_cache_block_size,
        )
        shared_prefix_reuse = {
            **shared_prefix_reuse,
            "phase_prefix_cache_reset": prefix_cache_reset,
        }

    output_summaries = summarize_vllm_outputs(manifest, outputs)
    sample = benchmark_sample_from_vllm_outputs(
        manifest,
        outputs,
        sample_index=sample_index,
        generation_s=sum(generation_call_s),
        peak_device_memory_bytes=monitor.peak_device_memory_bytes,
        validated_summaries=output_summaries,
    )
    require_namespace_ownership()
    full_output_artifact = write_full_output_artifact(
        full_output_path,
        manifest=manifest,
        outputs=outputs,
        execution_records=execution_records,
        decode_output_token_ids=decode_output_token_ids,
        ownership_validator=require_namespace_ownership,
    )
    full_decode_proof = None
    if collect_proof:
        expected_decode_tokens = sum(
            int(proof["full_decode_proof"]["expected_decode_tokens"]) for proof in wave_proofs
        )
        full_decode_tokens = sum(int(proof["full_decode_proof"]["full_decode_tokens"]) for proof in wave_proofs)
        full_decode_proof = {
            "phase": phase,
            "wave_count": len(wave_proofs),
            "expected_decode_tokens": expected_decode_tokens,
            "full_decode_tokens": full_decode_tokens,
            "coverage_fraction": full_decode_tokens / expected_decode_tokens if expected_decode_tokens else 1.0,
            "passed": all(proof["full_decode_proof"]["passed"] for proof in wave_proofs),
            "waves": [proof["full_decode_proof"] for proof in wave_proofs],
        }
    return GenerationPhaseResult(
        phase=phase,
        sample=sample,
        generation_call_s=tuple(generation_call_s),
        wave_proofs=tuple(wave_proofs),
        observations=(tuple(recorder.observations[observation_start:]) if collect_proof else ()),
        output_summaries=output_summaries,
        request_executions=execution_records,
        full_output_artifact=full_output_artifact,
        full_decode_proof=full_decode_proof,
        worker_proof=worker_proof,
        shared_prefix_state_reuse=shared_prefix_reuse,
        proof_collected=collect_proof,
        prefix_cache_reset=prefix_cache_reset,
    )


def reset_vllm_worker_proof_state(
    worker: Any,
    reset_prefix_sources: bool = True,
) -> dict[str, Any]:
    """Reset phase-local FIR telemetry and CUDA allocator peaks on one vLLM worker."""
    import torch

    from bionemo.evo2.vllm.model import (
        install_mamba_prefix_clone_proof_hook,
        reset_mamba_prefix_clone_stats,
        reset_mamba_state_copy_stats,
    )
    from bionemo.evo2.vllm.packed_fir import reset_fir_route_stats
    from bionemo.evo2.vllm.sampler import install_sampler_route_proof_hook

    sampler = install_sampler_route_proof_hook(worker)
    reset_fir_route_stats()
    reset_mamba_state_copy_stats()
    torch.cuda.reset_peak_memory_stats()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    install_mamba_prefix_clone_proof_hook()
    reset_mamba_prefix_clone_stats(reset_prefix_sources=reset_prefix_sources)
    return {
        "rank": int(rank),
        "device": int(torch.cuda.current_device()),
        "reset_prefix_sources": reset_prefix_sources,
        "sampler_route": sampler["selected_route"],
    }


def snapshot_vllm_worker_proof_state(worker: Any) -> dict[str, Any]:
    """Collect route, compile, and CUDA-memory evidence from one vLLM worker."""
    import torch

    from bionemo.evo2.vllm.model import (
        get_mamba_prefix_clone_stats,
        get_mamba_state_copy_stats,
    )
    from bionemo.evo2.vllm.packed_fir import get_fir_route_stats
    from bionemo.evo2.vllm.profile import compilation_counter_snapshot
    from bionemo.evo2.vllm.sampler import snapshot_sampler_route_proof

    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    gpu_identity = worker_gpu_identity(logical_device=int(device))
    engine_seed = getattr(getattr(worker, "model_config", None), "seed", None)
    if isinstance(engine_seed, bool) or not isinstance(engine_seed, int) or engine_seed < 0:
        raise RuntimeError("vLLM worker model_config.seed is unavailable or malformed")
    return {
        "rank": int(rank),
        "device": int(device),
        "engine_seed": engine_seed,
        **gpu_identity,
        "sampler": snapshot_sampler_route_proof(
            worker,
            require_generation_observations=hasattr(worker, "_evo2_sampler_batch_descriptors"),
        ),
        "fir_routes": get_fir_route_stats(),
        "mamba_state_copies": get_mamba_state_copy_stats(),
        "mamba_prefix_clones": get_mamba_prefix_clone_stats(),
        "compilation": compilation_counter_snapshot(),
        "cuda_memory": {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }


class _UnmonitoredMemory:
    """Expose the sample interface without polling during speed measurements."""

    peak_device_memory_bytes: tuple[int, ...] = ()

    def __enter__(self) -> _UnmonitoredMemory:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback


class PeakMemoryMonitor:
    """Poll low-overhead per-device used memory and retain phase-local peaks."""

    def __init__(
        self,
        read_device_memory_bytes: Callable[[], tuple[int, ...]],
        *,
        interval_s: float = 0.02,
    ) -> None:
        """Configure a monitor around one stable per-device memory reader."""
        if interval_s <= 0:
            raise ValueError("memory polling interval must be positive")
        self._read = read_device_memory_bytes
        self._interval_s = interval_s
        self._peaks: tuple[int, ...] = ()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    @property
    def peak_device_memory_bytes(self) -> tuple[int, ...]:
        """Return the maximum used bytes observed for every device."""
        with self._lock:
            return self._peaks

    def sample_now(self) -> None:
        """Read and merge one synchronous device-memory sample."""
        values = tuple(int(value) for value in self._read())
        if not values:
            raise RuntimeError("memory reader returned no devices")
        with self._lock:
            if self._peaks and len(values) != len(self._peaks):
                raise RuntimeError("memory reader device count changed during a phase")
            self._peaks = values if not self._peaks else tuple(map(max, self._peaks, values))

    def _poll(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self.sample_now()
            except BaseException as error:
                self._error = error
                self._stop_event.set()

    def __enter__(self) -> PeakMemoryMonitor:
        """Start phase-local polling."""
        self.sample_now()
        self._thread = threading.Thread(target=self._poll, name="evo2-nvml-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Stop polling and surface monitor failures."""
        del exc_type, exc_value, traceback
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.sample_now()
        if self._error is not None:
            raise RuntimeError("peak-memory polling failed") from self._error


def make_nvml_memory_reader() -> Callable[[], tuple[int, ...]]:
    """Create a stable reader for used memory on every visible physical GPU."""
    import pynvml

    pynvml.nvmlInit()
    handles = tuple(pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount()))
    return lambda: tuple(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used) for handle in handles)


def _nvml_text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def worker_gpu_identity(
    *,
    logical_device: int,
    nvml_module: Any | None = None,
) -> dict[str, Any]:
    """Resolve one CUDA logical device to an exact physical UUID and PCI address."""
    if isinstance(logical_device, bool) or not isinstance(logical_device, int) or logical_device < 0:
        raise ValueError("logical_device must be a nonnegative integer")
    if nvml_module is None:
        import pynvml as nvml_module

    nvml_module.nvmlInit()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = tuple(item.strip() for item in visible.split(",")) if visible is not None and visible.strip() else ()
    if selectors:
        if logical_device >= len(selectors) or not selectors[logical_device]:
            raise RuntimeError("CUDA logical device is not represented by CUDA_VISIBLE_DEVICES")
        selector = selectors[logical_device]
    else:
        selector = str(logical_device)
    if selector.isdecimal():
        handle = nvml_module.nvmlDeviceGetHandleByIndex(int(selector))
    elif selector.startswith("GPU-"):
        handle = nvml_module.nvmlDeviceGetHandleByUUID(selector)
    else:
        raise RuntimeError(f"unsupported CUDA_VISIBLE_DEVICES selector: {selector!r}")
    return {
        "logical_device": logical_device,
        "cuda_visible_devices": visible,
        "visible_device_selector": selector,
        "device_uuid": _nvml_text(nvml_module.nvmlDeviceGetUUID(handle)),
        "pci_bus_id": _nvml_text(nvml_module.nvmlDeviceGetPciInfo(handle).busId).lower(),
        "device_name": _nvml_text(nvml_module.nvmlDeviceGetName(handle)),
    }


def gpu_hardware_provenance(
    *,
    nvml_module: Any | None = None,
    torch_module: Any | None = None,
    expected_device_count: int = 2,
    expected_assignments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail closed on the frozen logical-to-physical GPU and memory contract."""
    if expected_device_count <= 0:
        raise ValueError("expected_device_count must be positive")
    if nvml_module is None:
        import pynvml as nvml_module
    if torch_module is None:
        import torch as torch_module

    assignments = [
        dict(item) for item in (_FROZEN_GPU_ASSIGNMENTS if expected_assignments is None else expected_assignments)
    ]
    if len(assignments) != expected_device_count:
        raise ValueError("expected GPU assignments must match expected_device_count")
    required_assignment_keys = {
        "logical_device_index",
        "visible_device_selector",
        "physical_index",
        "uuid",
        "pci_bus_id",
    }
    for logical_index, assignment in enumerate(assignments):
        if set(assignment) != required_assignment_keys:
            raise ValueError("each expected GPU assignment must use the exact frozen schema")
        if assignment["logical_device_index"] != logical_index:
            raise ValueError("expected GPU logical indices must be exact and contiguous")
        if (
            isinstance(assignment["physical_index"], bool)
            or not isinstance(assignment["physical_index"], int)
            or assignment["physical_index"] < 0
        ):
            raise ValueError("expected GPU physical indices must be nonnegative integers")
        if not all(
            isinstance(assignment[field], str) and assignment[field]
            for field in ("visible_device_selector", "uuid", "pci_bus_id")
        ):
            raise ValueError("expected GPU selector, UUID, and PCI identity must be nonempty")
        if assignment["pci_bus_id"] != assignment["pci_bus_id"].lower():
            raise ValueError("expected GPU PCI identities must use canonical lowercase text")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected_visible = ",".join(item["visible_device_selector"] for item in assignments)
    observed_selectors = tuple(item.strip() for item in visible.split(",")) if isinstance(visible, str) else ()
    assignment_mismatches: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "schema_version": 2,
        "passed": False,
        "cuda_visible_devices": visible,
        "expected_cuda_visible_devices": expected_visible,
        "observed_visible_device_selectors": list(observed_selectors),
        "expected_assignments": assignments,
        "required_initial_headroom_bytes": _REQUIRED_GPU_HEADROOM_BYTES,
        "api_versions": {
            "nvml_python_version": str(getattr(nvml_module, "__version__", _package_version("nvidia-ml-py"))),
            "nvml_module_path": (
                str(Path(nvml_module.__file__).resolve())
                if isinstance(getattr(nvml_module, "__file__", None), str)
                else None
            ),
            "nvml_memory_info_api": "nvmlDeviceGetMemoryInfo_v2",
            "torch_version": str(getattr(torch_module, "__version__", "unknown")),
            "torch_cuda_version": str(getattr(getattr(torch_module, "version", None), "cuda", "unknown")),
            "torch_module_path": (
                str(Path(torch_module.__file__).resolve())
                if isinstance(getattr(torch_module, "__file__", None), str)
                else None
            ),
            "cuda_memory_info_api": "torch.cuda.mem_get_info",
        },
        "devices": [],
    }

    def fail(stage: str, message: str) -> None:
        evidence["passed"] = False
        evidence["failure"] = {"stage": stage, "message": message}
        raise GpuPreflightError(message, evidence=evidence)

    memory_version = getattr(nvml_module, "nvmlMemory_v2", None)
    if memory_version is None:
        fail("api-capability", "pynvml does not expose nvmlMemory_v2")
    cuda_driver_query = getattr(nvml_module, "nvmlSystemGetCudaDriverVersion_v2", None)
    if not callable(cuda_driver_query):
        fail("api-capability", "pynvml does not expose nvmlSystemGetCudaDriverVersion_v2")

    initialized = False
    try:
        nvml_module.nvmlInit()
        initialized = True
        device_count = int(nvml_module.nvmlDeviceGetCount())
        cuda_device_count = int(torch_module.cuda.device_count())
        evidence["device_count"] = device_count
        evidence["cuda_device_count"] = cuda_device_count
        evidence["driver_version"] = _nvml_text(nvml_module.nvmlSystemGetDriverVersion())
        evidence["nvml_cuda_driver_version_integer"] = int(cuda_driver_query())
        evidence["nvml_memory_info_version"] = int(memory_version)
        if device_count != expected_device_count or cuda_device_count != expected_device_count:
            fail(
                "assignment",
                "NVML and CUDA device counts must both match the frozen benchmark topology",
            )
        if len(observed_selectors) != expected_device_count or any(not selector for selector in observed_selectors):
            fail(
                "assignment",
                "CUDA_VISIBLE_DEVICES must expose one nonempty selector per frozen logical device",
            )

        for assignment in assignments:
            logical_index = int(assignment["logical_device_index"])
            expected_selector = str(assignment["visible_device_selector"])
            selector = observed_selectors[logical_index]
            if selector.isdecimal():
                handle = nvml_module.nvmlDeviceGetHandleByIndex(int(selector))
            elif selector.startswith("GPU-"):
                handle = nvml_module.nvmlDeviceGetHandleByUUID(selector)
            else:
                fail("assignment", f"unsupported CUDA_VISIBLE_DEVICES selector {selector!r}")

            physical_index = int(nvml_module.nvmlDeviceGetIndex(handle))
            uuid = _nvml_text(nvml_module.nvmlDeviceGetUUID(handle))
            pci_bus_id = _nvml_text(nvml_module.nvmlDeviceGetPciInfo(handle).busId).lower()
            name = _nvml_text(nvml_module.nvmlDeviceGetName(handle))
            memory = nvml_module.nvmlDeviceGetMemoryInfo(handle, version=memory_version)
            properties = torch_module.cuda.get_device_properties(logical_index)
            torch_uuid = _nvml_text(properties.uuid)
            if not torch_uuid.startswith("GPU-"):
                torch_uuid = f"GPU-{torch_uuid}"
            cuda_free, cuda_total = torch_module.cuda.mem_get_info(logical_index)
            physical_total = int(memory.total)
            system_reserved = int(memory.reserved)
            nvml_free = int(memory.free)
            nvml_used = int(memory.used)
            cuda_free = int(cuda_free)
            cuda_total = int(cuda_total)
            properties_total = int(properties.total_memory)
            relation_delta = physical_total - system_reserved - cuda_total
            device = {
                "logical_device_index": logical_index,
                "visible_device_selector": selector,
                "expected_visible_device_selector": expected_selector,
                "physical_index": physical_index,
                "uuid": uuid,
                "pci_bus_id": pci_bus_id,
                "name": name,
                "torch_uuid": torch_uuid,
                "torch_name": str(properties.name),
                "memory": {
                    "nvml": {
                        "physical_total_bytes": physical_total,
                        "system_reserved_bytes": system_reserved,
                        "free_bytes": nvml_free,
                        "used_bytes": nvml_used,
                    },
                    "cuda": {
                        "usable_total_bytes": cuda_total,
                        "free_bytes": cuda_free,
                    },
                    "torch_properties_total_bytes": properties_total,
                    "usable_total_relation_delta_bytes": relation_delta,
                },
            }
            evidence["devices"].append(device)

            if (
                selector != expected_selector
                or physical_index != assignment["physical_index"]
                or uuid != assignment["uuid"]
                or pci_bus_id != assignment["pci_bus_id"]
                or torch_uuid != uuid
                or str(properties.name) != name
            ):
                assignment_mismatches.append(
                    {
                        "logical_device_index": logical_index,
                        "expected": dict(assignment),
                        "observed": {
                            "visible_device_selector": selector,
                            "physical_index": physical_index,
                            "uuid": uuid,
                            "pci_bus_id": pci_bus_id,
                            "torch_uuid": torch_uuid,
                            "torch_name": str(properties.name),
                            "nvml_name": name,
                        },
                    }
                )
            if (
                min(physical_total, cuda_total, properties_total) <= 0
                or min(system_reserved, nvml_free, nvml_used, cuda_free) < 0
            ):
                fail("memory-accounting", f"GPU {logical_index} returned malformed memory counters")
            if nvml_free + nvml_used + system_reserved != physical_total:
                fail(
                    "memory-accounting",
                    f"GPU {logical_index} NVML free+used+reserved does not equal physical total",
                )
            if relation_delta != 0 or properties_total != cuda_total:
                fail(
                    "memory-accounting",
                    f"GPU {logical_index} physical-reserved memory does not exactly equal CUDA/Torch usable total",
                )
            if nvml_free > cuda_total or cuda_free > cuda_total:
                fail("memory-accounting", f"GPU {logical_index} free memory exceeds usable total")
            if nvml_free < _REQUIRED_GPU_HEADROOM_BYTES or cuda_free < _REQUIRED_GPU_HEADROOM_BYTES:
                fail(
                    "memory-headroom",
                    f"GPU {logical_index} has less than 2 GiB free before engine creation",
                )
        if visible != expected_visible:
            assignment_mismatches.insert(
                0,
                {
                    "expected_cuda_visible_devices": expected_visible,
                    "observed_cuda_visible_devices": visible,
                },
            )
        if assignment_mismatches:
            evidence["assignment_mismatches"] = assignment_mismatches
            fail(
                "assignment",
                "observed logical GPUs do not match the frozen CUDA-visible physical assignments",
            )
    except GpuPreflightError:
        raise
    except BaseException as error:
        fail("collection", f"GPU provenance collection failed: {type(error).__name__}: {error}")
    finally:
        if initialized:
            shutdown = getattr(nvml_module, "nvmlShutdown", None)
            if callable(shutdown):
                shutdown()

    evidence["passed"] = True
    evidence.pop("failure", None)
    return evidence


def gpu_memory_headroom_evidence(
    hardware: dict[str, Any],
    *,
    peak_device_memory_bytes: Sequence[int],
    required_headroom_bytes: int = _REQUIRED_GPU_HEADROOM_BYTES,
) -> dict[str, Any]:
    """Require at least 2 GiB beyond the observed peak on every benchmark GPU."""
    if required_headroom_bytes <= 0:
        raise ValueError("required_headroom_bytes must be positive")
    devices = hardware.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ValueError("GPU hardware provenance must contain a nonempty device list")
    if len(devices) != len(peak_device_memory_bytes):
        raise ValueError("peak-memory samples must align with every provenance GPU")

    retained = []
    for device, peak in zip(devices, peak_device_memory_bytes, strict=True):
        memory = device.get("memory")
        nvml_memory = memory.get("nvml") if isinstance(memory, dict) else None
        cuda_memory = memory.get("cuda") if isinstance(memory, dict) else None
        if not isinstance(nvml_memory, dict) or not isinstance(cuda_memory, dict):
            raise ValueError("GPU memory provenance must retain NVML and CUDA counters")
        physical_total = nvml_memory.get("physical_total_bytes")
        system_reserved = nvml_memory.get("system_reserved_bytes")
        usable_total = cuda_memory.get("usable_total_bytes")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (physical_total, system_reserved, usable_total)
        ):
            raise ValueError("GPU physical, reserved, and usable memory must be nonnegative integers")
        if physical_total - system_reserved != usable_total or usable_total <= 0:
            raise ValueError("GPU physical-reserved memory must equal its positive CUDA usable total")
        if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
            raise ValueError("GPU peak memory must be a nonnegative integer")
        headroom = usable_total - peak
        retained.append(
            {
                "logical_device_index": int(device.get("logical_device_index", len(retained))),
                "physical_index": int(device.get("physical_index", len(retained))),
                "uuid": str(device.get("uuid", "")),
                "physical_total_memory_bytes": physical_total,
                "system_reserved_memory_bytes": system_reserved,
                "cuda_usable_total_memory_bytes": usable_total,
                "peak_used_memory_bytes": peak,
                "headroom_bytes": headroom,
            }
        )
        if headroom < required_headroom_bytes:
            raise RuntimeError(
                f"GPU {device.get('uuid', device.get('physical_index'))} has {headroom} bytes headroom; "
                "at least 2 GiB is required"
            )
    return {
        "required_headroom_bytes": required_headroom_bytes,
        "devices": retained,
        "passed": True,
    }


def _attach_cudagraph_recorder(llm: Any, recorder: CUDAGraphProofRecorder) -> None:
    manager = llm.llm_engine.logger_manager
    if manager is None:
        raise RuntimeError("vLLM stat logger manager is disabled; CUDA graph proof is unavailable")
    manager.stat_loggers.append(recorder)


def _reset_worker_proof(llm: Any) -> tuple[dict[str, Any], ...]:
    return tuple(llm.collective_rpc("reset_evo2_proof_state"))


def _snapshot_worker_proof(llm: Any) -> tuple[dict[str, Any], ...]:
    return tuple(llm.collective_rpc("snapshot_evo2_proof_state"))


def _begin_rank_local_generation_evidence(
    llm: Any,
    *,
    phase: str,
    contract_sha256: str,
    expected_request_count: int,
    expected_max_new_tokens: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        llm.collective_rpc(
            "begin_evo2_rank_local_generation_evidence",
            args=(
                phase,
                contract_sha256,
                expected_request_count,
                expected_max_new_tokens,
            ),
        )
    )


def _snapshot_rank_local_generation_evidence(
    llm: Any,
) -> tuple[dict[str, Any], ...]:
    return tuple(llm.collective_rpc("snapshot_evo2_rank_local_generation_evidence"))


def _abort_rank_local_generation_evidence(
    llm: Any,
) -> tuple[dict[str, Any], ...]:
    return tuple(llm.collective_rpc("abort_evo2_rank_local_generation_evidence"))


def _phase_specs(warmups: int, repetitions: int) -> tuple[tuple[str, int], ...]:
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be nonnegative and repetitions must be positive")
    return (
        ("cold-generation", 0),
        *((f"warmup-{index}", index + 1) for index in range(warmups)),
        *((f"steady-{index}", warmups + index + 1) for index in range(repetitions)),
    )


def profile_from_args(args: Any, manifest: WorkloadManifest) -> Evo2VllmProfile:
    """Map benchmark CLI settings to one topology-local physical engine profile."""
    profile = Evo2VllmProfile(
        topology=args.topology,
        max_model_len=args.max_model_len or manifest.max_total_tokens,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        async_scheduling=args.async_scheduling,
        proof=args.proof,
        max_concurrent_partial_prefills=args.max_concurrent_partial_prefills,
        long_prefill_chunk_tokens=args.long_prefill_chunk_tokens,
        optimization_level=args.optimization_level,
        performance_mode=args.performance_mode,
        shared_prefix_state_reuse=getattr(args, "shared_prefix_state_reuse", False),
        global_wave_size=getattr(args, "global_wave_size", 96),
        max_num_seqs=getattr(args, "max_num_seqs", None),
    )
    if profile.shared_prefix_state_reuse:
        shared_prefix_manifest_evidence(manifest)
    return profile


def run_context_length_preflight(args: Any, manifest: WorkloadManifest) -> dict[str, Any]:
    """Persist the resolved long-context contract without constructing a GPU engine."""
    require_output_namespace_reservation(args.output)
    profile = profile_from_args(args, manifest)

    preflight_begin = time.perf_counter()
    preflight = context_length_preflight(
        profile,
        model=args.checkpoint,
        workload_max_total_tokens=manifest.max_total_tokens,
        load_format=args.load_format,
    )
    preflight_s = time.perf_counter() - preflight_begin

    provenance_begin = time.perf_counter()
    source_identity = source_provenance()
    provenance_s = time.perf_counter() - provenance_begin
    return {
        "schema_version": 1,
        "task": "evo2-vllm-context-length-preflight",
        "benchmark_mode": "preflight",
        "backend": "vllm",
        "topology": args.topology,
        "versions": runtime_versions(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_provenance": source_identity,
        "invocation": {
            "argv": [sys.executable, *sys.argv],
            "parsed_args": {
                name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
            },
            "output_artifact_path": str(Path(args.output).resolve()),
            "exit_status": 0,
        },
        "manifest": manifest.to_dict(),
        "manifest_sha256": manifest.sha256,
        "profile": asdict(profile),
        "context_length_preflight": preflight,
        "timing": {
            "context_length_preflight_s": preflight_s,
            "source_provenance_s": provenance_s,
        },
    }


def run_tp2_benchmark(args: Any, manifest: WorkloadManifest) -> dict[str, Any]:
    """Run one TP2 Ray engine through cold, warm, and measured exact phases."""
    if args.topology != "tp2":
        raise ValueError("run_tp2_benchmark requires topology=tp2")
    require_output_namespace_reservation(args.output)
    profile = profile_from_args(args, manifest)
    mixed_stage_specs = mixed_same_engine_stage_specs(args, manifest, profile)
    mixed_same_engine = mixed_stage_specs is not None
    caller_coordinates = CallerCoordinateContract.from_inputs(
        manifest,
        profile,
        args.generation_round,
    )
    benchmark_mode = benchmark_mode_from_args(args)
    instrumentation = benchmark_instrumentation_contract(benchmark_mode)
    if mixed_same_engine:
        instrumentation = {
            **instrumentation,
            "scheduler_callbacks_during_generation": True,
            "supported_vllm_stat_logger": True,
            "peak_memory_polling_during_generation": True,
            "proof_only_worker_extension": False,
            "timing_admissible_for_speed_ranking": False,
        }
    preflight_begin = time.perf_counter()
    preflight = context_length_preflight(
        profile,
        model=args.checkpoint,
        workload_max_total_tokens=manifest.max_total_tokens,
        load_format=args.load_format,
    )
    preflight_s = time.perf_counter() - preflight_begin

    gpu_preflight_begin = time.perf_counter()
    gpu_identity = gpu_hardware_provenance()
    gpu_preflight_s = time.perf_counter() - gpu_preflight_begin

    vllm_import_begin = time.perf_counter()
    from vllm import LLM, SamplingParams

    vllm_import_s = time.perf_counter() - vllm_import_begin

    provenance_begin = time.perf_counter()
    checkpoint_identity = checkpoint_provenance(args.checkpoint)
    source_identity = source_provenance(require_clean=True)
    vllm_identity = vllm_installation_provenance()
    from bionemo.evo2.vllm.sampler import sampler_installation_provenance

    sampler_identity = sampler_installation_provenance(require_loaded_modules=False)
    runtime_attestation = runtime_attestation_contract(
        checkpoint=checkpoint_identity,
        sources={"bionemo": source_identity},
        vllm_installation=vllm_identity,
        sampler_installation=sampler_identity,
        gpu_hardware=gpu_identity,
    )
    benchmark_contract = {
        **build_benchmark_contract(args, manifest, profile),
        "runtime_attestation": runtime_attestation,
    }
    benchmark_contract_digest = benchmark_contract_sha256(benchmark_contract)
    linked_proof = (
        validate_linked_proof_artifact(
            args.linked_proof_artifact,
            expected_contract=benchmark_contract,
            caller_coordinates=caller_coordinates,
            require_memory_headroom=True,
        )
        if benchmark_mode == "speed" and args.linked_proof_artifact is not None
        else None
    )
    provenance_s = time.perf_counter() - provenance_begin

    engine_kwargs = profile.engine_kwargs(
        model=str(args.checkpoint),
        seed=manifest.seed,
        load_format=args.load_format,
    )
    if mixed_same_engine:
        engine_kwargs["cudagraph_metrics"] = True
    decoder_begin = time.perf_counter()
    output_decoder = manifest_output_decoder(manifest)
    output_decoder_setup_s = time.perf_counter() - decoder_begin
    memory_reader = make_nvml_memory_reader()
    recorder = CUDAGraphProofRecorder() if profile.proof or mixed_same_engine else None

    init_begin = time.perf_counter()
    with PeakMemoryMonitor(memory_reader) as init_memory:
        llm = LLM(**engine_kwargs)
    engine_init_s = time.perf_counter() - init_begin
    if recorder is not None:
        _attach_cudagraph_recorder(llm, recorder)
    resolved = resolved_config_snapshot(llm.llm_engine.vllm_config)
    validate_resolved_profile(profile, resolved)
    initialized_worker_proof = _snapshot_worker_proof(llm) if profile.proof else ()

    common_identity = common_prefix_identity_context(args, manifest, profile)
    serial_reference_result = None
    phase_coordinates = (
        ()
        if mixed_same_engine
        else benchmark_phase_coordinates(
            manifest,
            profile,
            generation_round_start=args.generation_round,
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
    )
    call_index_start = 0 if mixed_same_engine else phase_coordinates[0]["global_call_index_start"]
    if common_identity is not None:
        serial_manifest = manifest.request_slice(0, 1)
        serial_executions = build_request_execution_records(
            serial_manifest,
            global_request_offset=0,
            dp_rank=0,
            dp_size=1,
            generation_round=args.generation_round,
            call_index=call_index_start,
        )
        serial_reference_result = run_generation_phase(
            llm=llm,
            manifest=serial_manifest,
            sampling_params=build_request_sampling_params(
                serial_manifest,
                sampling_params_factory=SamplingParams,
                execution_records=serial_executions,
            ),
            phase="common-prefix-serial-reference",
            sample_index=-1,
            recorder=recorder,
            memory_monitor_factory=lambda: PeakMemoryMonitor(memory_reader),
            execution_records=serial_executions,
            full_output_path=phase_output_artifact_path(
                args.output,
                phase="common-prefix-serial-reference",
            ),
            namespace_output_path=args.output,
            collect_proof=profile.proof,
            reset_worker_proof=(lambda: _reset_worker_proof(llm)) if profile.proof else None,
            snapshot_worker_proof=(lambda: _snapshot_worker_proof(llm)) if profile.proof else None,
            require_rank_local_evidence=profile.proof,
            rank_local_semantic_namespace=(
                f"{benchmark_contract_digest}:common-prefix-serial-reference"
                if profile.proof
                else None
            ),
            expected_tensor_parallel_size=2 if profile.proof else None,
            begin_rank_local_evidence=(
                lambda **kwargs: _begin_rank_local_generation_evidence(llm, **kwargs)
                if profile.proof
                else None
            ),
            snapshot_rank_local_evidence=(
                lambda: _snapshot_rank_local_generation_evidence(llm)
                if profile.proof
                else None
            ),
            abort_rank_local_evidence=(
                lambda: _abort_rank_local_generation_evidence(llm)
                if profile.proof
                else None
            ),
            require_shared_prefix_state_reuse=profile.shared_prefix_state_reuse,
            prefix_cache_block_size=int(resolved["cache"]["block_size"]),
            global_wave_size=1,
            scheduler_max_num_seqs=profile.resolved_max_num_seqs,
            decode_output_token_ids=output_decoder,
        )

    if mixed_same_engine:
        phase_runs = tuple(
            {
                "phase": spec["stage"],
                "sample_index": sample_index,
                "manifest": spec["manifest"],
                "global_wave_size": spec["schedule"].global_wave_size,
                "execution_records": spec["execution_records"],
            }
            for sample_index, spec in enumerate(mixed_stage_specs)
        )
    else:
        phase_runs = tuple(
            {
                "phase": coordinate["phase"],
                "sample_index": coordinate["sample_index"],
                "manifest": manifest,
                "global_wave_size": profile.global_wave_size,
                "execution_records": build_wave_execution_records(
                    manifest,
                    global_wave_size=profile.global_wave_size,
                    generation_round=coordinate["generation_round"],
                    call_index_start=coordinate["global_call_index_start"],
                    global_request_index_start=coordinate["global_request_index_start"],
                ),
            }
            for coordinate in phase_coordinates
        )

    phase_results = []
    for phase_run in phase_runs:
        phase = phase_run["phase"]
        phase_manifest = phase_run["manifest"]
        execution_records = phase_run["execution_records"]
        sampling_params = build_request_sampling_params(
            phase_manifest,
            sampling_params_factory=SamplingParams,
            execution_records=execution_records,
        )
        result = run_generation_phase(
            llm=llm,
            manifest=phase_manifest,
            sampling_params=sampling_params,
            phase=phase,
            sample_index=phase_run["sample_index"],
            recorder=recorder,
            memory_monitor_factory=lambda: PeakMemoryMonitor(memory_reader),
            execution_records=execution_records,
            full_output_path=phase_output_artifact_path(args.output, phase=phase),
            namespace_output_path=args.output,
            collect_proof=profile.proof or mixed_same_engine,
            reset_worker_proof=(lambda: _reset_worker_proof(llm)) if profile.proof else None,
            snapshot_worker_proof=(lambda: _snapshot_worker_proof(llm)) if profile.proof else None,
            require_rank_local_evidence=profile.proof,
            rank_local_semantic_namespace=(
                f"{benchmark_contract_digest}:{phase}" if profile.proof else None
            ),
            expected_tensor_parallel_size=2 if profile.proof else None,
            begin_rank_local_evidence=(
                lambda **kwargs: _begin_rank_local_generation_evidence(llm, **kwargs)
                if profile.proof
                else None
            ),
            snapshot_rank_local_evidence=(
                lambda: _snapshot_rank_local_generation_evidence(llm)
                if profile.proof
                else None
            ),
            abort_rank_local_evidence=(
                lambda: _abort_rank_local_generation_evidence(llm)
                if profile.proof
                else None
            ),
            require_shared_prefix_state_reuse=profile.shared_prefix_state_reuse,
            prefix_cache_block_size=int(resolved["cache"]["block_size"]),
            global_wave_size=phase_run["global_wave_size"],
            scheduler_max_num_seqs=profile.resolved_max_num_seqs,
            decode_output_token_ids=output_decoder,
        )
        if args.proof:
            for wave_proof in result.wave_proofs:
                validate_full_decode_proof(
                    list(result.observations),
                    phase=wave_proof["full_decode_proof"]["phase"],
                    batch_size=wave_proof["request_count"],
                    max_new_tokens=phase_manifest.max_new_tokens,
                )
        phase_results.append(result)

    if profile.proof:
        final_worker_proof = phase_results[-1].worker_proof
        for initialized, final in zip(initialized_worker_proof, final_worker_proof, strict=True):
            validate_compilation_proof(initialized["compilation"], final["compilation"])
            memory_samples = [
                init_memory.peak_device_memory_bytes,
                *(
                    [serial_reference_result.sample.peak_device_memory_bytes]
                    if serial_reference_result is not None
                    else []
                ),
                *(result.sample.peak_device_memory_bytes for result in phase_results),
            ]
        peak_device_memory = tuple(max(values) for values in zip(*memory_samples, strict=True))
        memory_headroom = gpu_memory_headroom_evidence(
            gpu_identity,
            peak_device_memory_bytes=peak_device_memory,
        )
    elif mixed_same_engine:
        memory_samples = [
            init_memory.peak_device_memory_bytes,
            *(result.sample.peak_device_memory_bytes for result in phase_results),
        ]
        peak_device_memory = tuple(max(values) for values in zip(*memory_samples, strict=True))
        memory_headroom = gpu_memory_headroom_evidence(
            gpu_identity,
            peak_device_memory_bytes=peak_device_memory,
        )
    else:
        if linked_proof is not None:
            if not isinstance(linked_proof.get("gpu_memory_headroom"), dict):
                raise RuntimeError("linked speed evidence is missing GPU memory headroom")
            memory_headroom = linked_proof["gpu_memory_headroom"]
        else:
            memory_headroom = gpu_memory_headroom_evidence(
                gpu_identity,
                peak_device_memory_bytes=init_memory.peak_device_memory_bytes,
            )

    steady_results = (
        [phase_results[-1]]
        if mixed_same_engine
        else [result for result in phase_results if result.phase.startswith("steady-")]
    )
    phase_artifacts, exact_progress = attach_exact_generation_progress_evidence(
        [result.to_dict() for result in phase_results],
        manifest=manifest,
        enabled=bool(args.exact_progress_gate),
        proof_collected=profile.proof,
        topology=profile.topology,
        linked_proof_artifact=args.linked_proof_artifact,
    )
    phase_artifacts, mbs1_exact1k = attach_mbs1_exact1k_evidence(
        phase_artifacts,
        manifest=manifest,
        profile=profile,
        generation_round_start=args.generation_round,
        warmups=args.warmups,
        repetitions=args.repetitions,
        enabled=bool(getattr(args, "mbs1_exact1k_audit", False)),
        proof_collected=profile.proof,
        linked_proof_artifact=args.linked_proof_artifact,
    )
    phase_artifacts, canonical_identity = canonical_identity_phase_artifacts(
        args=args,
        manifest=manifest,
        profile=profile,
        phase_artifacts=phase_artifacts,
        decode_output_token_ids=output_decoder,
        collect_physical_proof=profile.proof,
    )
    phase_artifacts, mixed_canonical_identity = mixed_canonical_identity_phase_artifacts(
        args=args,
        manifest=manifest,
        profile=profile,
        phase_artifacts=phase_artifacts,
        decode_output_token_ids=output_decoder,
        collect_physical_proof=profile.proof,
    )
    phase_artifacts, common_prefix_identity = common_prefix_identity_phase_artifacts(
        args=args,
        manifest=manifest,
        profile=profile,
        serial_reference_phase=(None if serial_reference_result is None else serial_reference_result.to_dict()),
        phase_artifacts=phase_artifacts,
        decode_output_token_ids=output_decoder,
        collect_physical_proof=profile.proof,
    )
    waves = build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=profile.global_batch_size,
        replica_count=profile.replica_count,
    )
    return {
        "schema_version": 1,
        "backend": "vllm",
        "topology": "tp2",
        "benchmark_mode": benchmark_mode,
        "benchmark_contract": benchmark_contract,
        "benchmark_contract_sha256": benchmark_contract_digest,
        "instrumentation": instrumentation,
        "linked_proof_artifact": linked_proof,
        "proof_status": (
            {
                "passed": True,
                "phase_count": len(phase_results),
                "full_decode_passed": all(result.full_decode_proof["passed"] for result in phase_results),
                "compilation_stable": True,
            }
            if profile.proof
            else {
                "passed": None,
                "linked_proof": linked_proof,
                "post_output_validation_passed": True,
            }
        ),
        "versions": runtime_versions(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_provenance": checkpoint_identity,
        "source_provenance": source_identity,
        "vllm_installation_provenance": vllm_identity,
        "sampler_installation_provenance": sampler_identity,
        "gpu_hardware_provenance": gpu_identity,
        "gpu_memory_headroom": memory_headroom,
        "canonical_identity": canonical_identity,
        "mixed_canonical_identity": mixed_canonical_identity,
        "common_prefix_identity": common_prefix_identity,
        "common_prefix_serial_reference": (
            None if serial_reference_result is None else serial_reference_result.to_dict()
        ),
        "exact_generation_progress": exact_progress,
        "mbs1_exact1k": mbs1_exact1k,
        "invocation": {
            "argv": [sys.executable, *sys.argv],
            "parsed_args": {
                name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
            },
            "output_artifact_path": str(Path(args.output).resolve()),
            "exit_status": 0,
        },
        "manifest": manifest.to_dict(),
        "manifest_sha256": manifest.sha256,
        "profile": asdict(profile),
        "context_length_preflight": preflight,
        "engine_kwargs": engine_kwargs,
        "resolved_config": resolved,
        "execution_contract": {
            "outer_model": "torch.compile Inductor",
            "prefill": "optimized eager no_compile custom op; packed route proven per worker",
            "decode": "FULL CUDA graph replay required",
            "prefix_caching": profile.shared_prefix_state_reuse,
            "mamba_cache_mode": "align" if profile.shared_prefix_state_reuse else "none",
            "shared_prefix_state_reuse": profile.shared_prefix_state_reuse,
            "global_wave_size": profile.global_wave_size,
            "per_engine_max_num_seqs": profile.resolved_max_num_seqs,
            "gdpo_target_request_count": profile.gdpo_target_batch_size,
            "planned_waves_to_96": profile.gdpo_waves_to_96,
            "semantic_padding": False,
            "benchmark_mode": benchmark_mode,
            "timed_generation_instrumentation": instrumentation,
        },
        "request_waves": [
            {
                "wave_index": wave.wave_index,
                "start": wave.start,
                "stop": wave.stop,
                "request_count": wave.request_count,
                "shards": [asdict(shard) | {"request_count": shard.request_count} for shard in wave.shards],
            }
            for wave in waves
        ],
        "timing": {
            "context_length_preflight_s": preflight_s,
            "gpu_hardware_preflight_s": gpu_preflight_s,
            "vllm_import_s": vllm_import_s,
            "provenance_hashing_s": provenance_s,
            "output_decoder_setup_s": output_decoder_setup_s,
            "engine_init_s": engine_init_s,
            "engine_init_peak_device_memory_bytes": list(init_memory.peak_device_memory_bytes),
        },
        "initialized_worker_proof": list(initialized_worker_proof),
        "phases": phase_artifacts,
        "steady_aggregate": aggregate_samples([result.sample for result in steady_results]),
    }


def write_json_artifact(
    path: str | Path,
    artifact: dict[str, Any],
    *,
    ownership_validator: Callable[[], Any] | None = None,
) -> PublicationReceipt:
    """Write one durable, deterministic benchmark artifact."""
    payload = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return publish_bytes_noreplace(
        path,
        payload,
        ownership_validator=ownership_validator,
        publication_recorder=_record_output_namespace_publication,
    )


def _gpu_preflight_failure_artifact(
    args: Any,
    manifest: WorkloadManifest,
    error: GpuPreflightError,
) -> dict[str, Any]:
    """Build a durable failure artifact without querying the failed GPU path again."""
    return {
        "schema_version": 1,
        "task": "evo2-vllm-gpu-preflight",
        "benchmark_mode": benchmark_mode_from_args(args),
        "backend": "vllm",
        "topology": args.topology,
        "versions": runtime_versions(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "invocation": {
            "argv": [sys.executable, *sys.argv],
            "parsed_args": {
                name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
            },
            "output_artifact_path": str(Path(args.output).resolve()),
            "exit_status": 1,
        },
        "manifest": manifest.to_dict(),
        "manifest_sha256": manifest.sha256,
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "gpu_hardware_provenance": error.evidence,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the optimized exact-length vLLM benchmark CLI."""
    args = build_parser().parse_args(argv)
    benchmark_mode_from_args(args)
    reservation = reserve_output_namespace(args.output)
    if args.backend != "vllm":
        raise NotImplementedError("the MCore baseline uses its pinned backend adapter")
    source_manifest = load_source_manifest(args)
    identity_mode = (
        args.canonical_identity_case is not None
        or args.common_prefix_identity_case is not None
        or args.mixed_canonical_identity
    )
    manifest = prepare_workload(
        source_manifest,
        request_count=None if identity_mode else args.request_count,
        uniform_prompt_length=None if identity_mode else args.uniform_prompt_length,
        request_id_prefix=args.request_id_prefix,
        max_new_tokens=None if identity_mode else args.max_new_tokens,
    )
    try:
        if args.context_preflight_only:
            artifact = run_context_length_preflight(args, manifest)
        elif args.topology == "tp2":
            artifact = run_tp2_benchmark(args, manifest)
        else:
            from bionemo.evo2.vllm.nemo_runner import run_nemo_dp2_benchmark

            artifact = run_nemo_dp2_benchmark(args, manifest)
    except GpuPreflightError as error:
        artifact = _gpu_preflight_failure_artifact(args, manifest, error)
        require_output_namespace_reservation(args.output)
        write_json_artifact(
            args.output,
            artifact,
            ownership_validator=lambda: require_output_namespace_reservation(args.output),
        )
        complete_output_namespace(reservation, output_path=args.output)
        return 1
    require_output_namespace_reservation(args.output)
    write_json_artifact(
        args.output,
        artifact,
        ownership_validator=lambda: require_output_namespace_reservation(args.output),
    )
    complete_output_namespace(reservation, output_path=args.output)
    return 0


__all__ = [
    "CUDAGraphProofRecorder",
    "GenerationPhaseResult",
    "PeakMemoryMonitor",
    "RequestExecutionRecord",
    "build_request_execution_records",
    "build_request_sampling_params",
    "checkpoint_provenance",
    "complete_output_namespace",
    "full_decode_proof_summary",
    "load_source_manifest",
    "phase_output_artifact_path",
    "prepare_workload",
    "request_seed",
    "require_output_namespace_reservation",
    "register_output_namespace_publication",
    "reserve_output_namespace",
    "reset_vllm_worker_proof_state",
    "run_context_length_preflight",
    "run_generation_phase",
    "runtime_versions",
    "snapshot_vllm_worker_proof_state",
    "source_provenance",
    "summarize_cudagraph_observations",
    "validate_full_decode_proof",
    "write_full_generation_records_artifact",
    "write_full_output_artifact",
    "write_json_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
