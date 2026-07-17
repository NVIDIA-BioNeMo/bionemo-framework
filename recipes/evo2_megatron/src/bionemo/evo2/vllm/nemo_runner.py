# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Strict adapters between Evo2 benchmark manifests and production NeMo-RL generation."""

from __future__ import annotations

import math
import hashlib
import json
import re
import struct
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GENERATION_REQUEST_METADATA_KEYS,
    GENERATION_REQUEST_SEED_MODULUS,
    GENERATION_REQUEST_SEED_STRIDE,
    GenerationRequestEnvelope,
    GenerationRequestIdentity,
    GenerationRequestLedgerAdmission,
    GenerationRequestScheduleAdmission,
    generation_prompt_token_ids_sha256,
    validate_generation_request_metadata,
    validate_generation_selected_logprobs,
)
from nemo_rl.models.generation.vllm.vllm_generation import (
    validate_generation_request_envelope_result,
)

from bionemo.evo2.vllm.benchmark import (
    BenchmarkSample,
    GenerationRecord,
    WorkloadManifest,
    aggregate_samples,
    build_request_waves,
    validate_compilation_proof,
)
from bionemo.evo2.vllm.artifact_io import (
    FilePublicationPlan,
    FilePublicationReservation,
    PublicationReceipt,
    cancel_file_publication_reservation,
    finalize_reserved_publication,
    read_jsonl_snapshot,
    reserve_file_publication,
    validate_file_publication_plan,
    validate_publication_receipt,
)
from bionemo.evo2.vllm.profile import Evo2VllmProfile, context_length_preflight, validate_resolved_profile
from bionemo.evo2.vllm.nemo_publication_schema import (
    NEMO_CALLER_PROMPT_ROOT_SCHEMA_VERSION,
    NEMO_RANK_EXECUTION_OCCURRENCE_SCHEMA_VERSION,
    NEMO_RANK_PUBLICATION_OUTCOME_SCHEMA_VERSION,
    NEMO_RANK_PUBLICATION_PROOF_SCHEMA_VERSION,
    NEMO_RANK_PUBLICATION_SCHEMA_VERSION,
    NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION,
)
from bionemo.evo2.vllm.runner import (
    COORDINATOR_GENERATION_TIMING_AUTHORITY,
    CallerCoordinateContract,
    PeakMemoryMonitor,
    RequestExecutionRecord,
    attach_exact_generation_progress_evidence,
    benchmark_contract_sha256,
    benchmark_instrumentation_contract,
    benchmark_mode_from_args,
    build_benchmark_contract,
    canonical_identity_phase_artifacts,
    checkpoint_provenance,
    common_prefix_identity_context,
    common_prefix_identity_phase_artifacts,
    full_decode_proof_summary,
    gpu_hardware_provenance,
    gpu_memory_headroom_evidence,
    make_nvml_memory_reader,
    manifest_output_decoder,
    phase_output_artifact_path,
    profile_from_args,
    request_seed,
    register_output_namespace_publication,
    require_output_namespace_reservation,
    runtime_attestation_contract,
    runtime_versions,
    scheduler_capacity_proof_summary,
    shared_prefix_state_reuse_evidence,
    source_provenance,
    validate_full_decode_proof,
    validate_linked_proof_artifact,
    validate_scheduler_capacity_proof,
    vllm_installation_provenance,
    wave_execution_summary,
    write_full_generation_records_artifact,
)


_OUTPUT_METADATA_KEYS = GENERATION_REQUEST_METADATA_KEYS
_RANK_PUBLICATION_PHASE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


@dataclass(frozen=True, slots=True)
class NemoPromptPayloadRecord:
    """One deduplicated caller-tokenized prompt payload."""

    prompt_token_ids: tuple[int, ...]
    prompt_token_ids_sha256: str

    def __post_init__(self) -> None:
        if type(self.prompt_token_ids) is not tuple or not self.prompt_token_ids or any(
            type(token_id) is not int or token_id < 0 for token_id in self.prompt_token_ids
        ):
            raise TypeError("prompt payload token IDs must be a nonempty exact integer tuple")
        if type(self.prompt_token_ids_sha256) is not str or re.fullmatch(
            r"[0-9a-f]{64}", self.prompt_token_ids_sha256
        ) is None:
            raise ValueError("prompt payload digest must be a lowercase SHA256")
        if generation_prompt_token_ids_sha256(self.prompt_token_ids) != self.prompt_token_ids_sha256:
            raise ValueError("prompt payload digest differs from its caller-tokenized bytes")

    @property
    def prompt_token_count(self) -> int:
        return len(self.prompt_token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_token_ids": list(self.prompt_token_ids),
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
        }


@dataclass(frozen=True, slots=True)
class NemoSemanticPromptRequestRecord:
    """One immutable semantic request independent of execution phase replay."""

    request_id: str
    global_request_index: int
    request_index_in_dp_stream: int
    generation_round: int
    call_index: int
    dp_rank: int
    request_seed: int
    prompt_token_count: int
    prompt_token_ids_sha256: str

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise TypeError("semantic request_id must be a nonempty built-in str")
        for label, value in (
            ("global_request_index", self.global_request_index),
            ("request_index_in_dp_stream", self.request_index_in_dp_stream),
            ("generation_round", self.generation_round),
            ("call_index", self.call_index),
            ("dp_rank", self.dp_rank),
            ("request_seed", self.request_seed),
        ):
            if type(value) is not int or value < 0:
                raise TypeError(f"semantic request {label} must be a nonnegative built-in int")
        if type(self.prompt_token_count) is not int or self.prompt_token_count <= 0:
            raise TypeError("semantic request token count must be a positive built-in int")
        if type(self.prompt_token_ids_sha256) is not str or re.fullmatch(
            r"[0-9a-f]{64}", self.prompt_token_ids_sha256
        ) is None:
            raise ValueError("semantic request prompt digest must be a lowercase SHA256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class NemoRankPromptExecutionOccurrence:
    """One phase-specific publication occurrence referencing semantic requests."""

    publication_key: str
    phase: str
    wave_index: int
    generation_round: int
    call_index: int
    dp_rank: int
    active: bool
    requests: tuple[NemoSemanticPromptRequestRecord, ...]

    def __post_init__(self) -> None:
        if type(self.publication_key) is not str or not self.publication_key:
            raise TypeError("execution occurrence key must be a nonempty built-in str")
        if type(self.phase) is not str or _RANK_PUBLICATION_PHASE.fullmatch(self.phase) is None:
            raise ValueError("execution occurrence phase is invalid")
        for label, value in (
            ("wave_index", self.wave_index),
            ("generation_round", self.generation_round),
            ("call_index", self.call_index),
            ("dp_rank", self.dp_rank),
        ):
            if type(value) is not int or value < 0:
                raise TypeError(f"execution occurrence {label} must be a nonnegative built-in int")
        if type(self.active) is not bool:
            raise TypeError("execution occurrence active flag must be an exact built-in bool")
        if type(self.requests) is not tuple or any(
            type(request) is not NemoSemanticPromptRequestRecord for request in self.requests
        ):
            raise TypeError("execution occurrence requests must be an exact record tuple")
        if self.active is not bool(self.requests):
            raise ValueError("execution occurrence activity contradicts its request inventory")
        if any(
            request.generation_round != self.generation_round
            or request.call_index != self.call_index
            or request.dp_rank != self.dp_rank
            for request in self.requests
        ):
            raise ValueError("execution occurrence route differs from its semantic requests")
        local_indices = tuple(request.request_index_in_dp_stream for request in self.requests)
        if local_indices != tuple(range(len(self.requests))):
            raise ValueError("execution occurrence DP stream order is not canonical")

    @property
    def semantic_request_sha256(self) -> tuple[str, ...]:
        return tuple(request.sha256 for request in self.requests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NEMO_RANK_EXECUTION_OCCURRENCE_SCHEMA_VERSION,
            "publication_key": self.publication_key,
            "phase": self.phase,
            "wave_index": self.wave_index,
            "generation_round": self.generation_round,
            "call_index": self.call_index,
            "dp_rank": self.dp_rank,
            "active": self.active,
            "semantic_request_sha256": list(self.semantic_request_sha256),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class NemoCallerPromptAnchorRoot:
    """Coordinator-owned root for prompt payloads, semantics, and occurrences."""

    manifest_sha256: str
    external_contract_sha256: str
    base_seed: int
    data_parallel_size: int
    prompt_payload_catalog: tuple[NemoPromptPayloadRecord, ...]
    semantic_requests: tuple[NemoSemanticPromptRequestRecord, ...]
    execution_occurrences: tuple[NemoRankPromptExecutionOccurrence, ...]

    def __post_init__(self) -> None:
        for label, digest in (
            ("manifest", self.manifest_sha256),
            ("external contract", self.external_contract_sha256),
        ):
            if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"caller prompt root {label} digest must be a lowercase SHA256")
        if type(self.base_seed) is not int or not 0 <= self.base_seed < 2**31:
            raise ValueError("caller prompt root base seed must be a built-in int in [0, 2**31)")
        if type(self.data_parallel_size) is not int or self.data_parallel_size <= 0:
            raise ValueError("caller prompt root DP size must be a positive built-in int")
        if type(self.prompt_payload_catalog) is not tuple or not self.prompt_payload_catalog or any(
            type(record) is not NemoPromptPayloadRecord for record in self.prompt_payload_catalog
        ):
            raise TypeError("caller prompt payload catalog must be a nonempty exact tuple")
        if type(self.semantic_requests) is not tuple or not self.semantic_requests or any(
            type(record) is not NemoSemanticPromptRequestRecord for record in self.semantic_requests
        ):
            raise TypeError("caller semantic request catalog must be a nonempty exact tuple")
        if type(self.execution_occurrences) is not tuple or not self.execution_occurrences or any(
            type(record) is not NemoRankPromptExecutionOccurrence
            for record in self.execution_occurrences
        ):
            raise TypeError("caller execution occurrences must be a nonempty exact tuple")
        payload_digests = tuple(record.prompt_token_ids_sha256 for record in self.prompt_payload_catalog)
        semantic_digests = tuple(record.sha256 for record in self.semantic_requests)
        occurrence_keys = tuple(record.publication_key for record in self.execution_occurrences)
        if len(set(payload_digests)) != len(payload_digests):
            raise ValueError("caller prompt payload catalog contains a digest collision")
        if len(set(semantic_digests)) != len(semantic_digests):
            raise ValueError("caller semantic request catalog contains duplicate records")
        if len(set(occurrence_keys)) != len(occurrence_keys):
            raise ValueError("caller execution occurrence keys are not unique")
        payload_set = set(payload_digests)
        semantic_set = set(semantic_digests)
        if any(request.prompt_token_ids_sha256 not in payload_set for request in self.semantic_requests):
            raise ValueError("caller semantic request references a missing prompt payload")
        for request in self.semantic_requests:
            if request.dp_rank >= self.data_parallel_size:
                raise ValueError("caller semantic request DP rank exceeds the root topology")
            expected_seed = request_seed(
                self.base_seed,
                call_index=request.call_index,
                dp_rank=request.dp_rank,
                dp_size=self.data_parallel_size,
                request_index_in_stream=request.request_index_in_dp_stream,
            )
            if request.request_seed != expected_seed:
                raise ValueError("caller semantic request seed contradicts the root coordinates")
        if any(
            digest not in semantic_set
            for occurrence in self.execution_occurrences
            for digest in occurrence.semantic_request_sha256
        ):
            raise ValueError("caller execution occurrence references a missing semantic request")

    def require_occurrence(self, publication_key: str) -> NemoRankPromptExecutionOccurrence:
        if type(publication_key) is not str or not publication_key:
            raise TypeError("execution occurrence lookup key must be a nonempty built-in str")
        matches = tuple(
            occurrence
            for occurrence in self.execution_occurrences
            if occurrence.publication_key == publication_key
        )
        if len(matches) != 1:
            raise RuntimeError(f"caller prompt root has no exact execution occurrence: {publication_key}")
        return matches[0]

    def require_payload(self, prompt_sha256: str) -> NemoPromptPayloadRecord:
        if type(prompt_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", prompt_sha256) is None:
            raise TypeError("prompt payload lookup requires an exact lowercase SHA256")
        matches = tuple(
            payload
            for payload in self.prompt_payload_catalog
            if payload.prompt_token_ids_sha256 == prompt_sha256
        )
        if len(matches) != 1:
            raise RuntimeError("caller prompt root has no exact prompt payload")
        return matches[0]

    def index_by_publication_key(self) -> dict[str, NemoCallerPromptAnchorRoot]:
        """Build an exact-key coordinator lookup without copying the immutable root."""
        return {
            occurrence.publication_key: self
            for occurrence in self.execution_occurrences
            if occurrence.active
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NEMO_CALLER_PROMPT_ROOT_SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "external_contract_sha256": self.external_contract_sha256,
            "base_seed": self.base_seed,
            "data_parallel_size": self.data_parallel_size,
            "prompt_payload_catalog": [record.to_dict() for record in self.prompt_payload_catalog],
            "semantic_requests": [record.to_dict() for record in self.semantic_requests],
            "execution_occurrences": [record.to_dict() for record in self.execution_occurrences],
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _rank_publication_key(phase: str, wave_index: int, dp_rank: int) -> str:
    return f"{phase}/wave-{wave_index:03d}/dp-{dp_rank}"


def _rank_publication_path(
    output_path: str | Path,
    *,
    phase: str,
    wave_index: int,
    dp_rank: int,
) -> Path:
    output = Path(output_path).resolve()
    return output.with_name(
        f"{output.name}.rank.{phase}.wave-{wave_index:03d}.dp{dp_rank}.outputs.jsonl.gz"
    )


def validate_nemo_rank_prompt_anchor(
    envelope: dict[str, Any],
    caller_root: NemoCallerPromptAnchorRoot,
) -> None:
    """Bind one mutable transport envelope to the caller-owned prompt snapshot."""
    if type(envelope) is not dict:
        raise TypeError("rank publication envelope must be a built-in dict")
    if type(caller_root) is not NemoCallerPromptAnchorRoot:
        raise TypeError("rank prompt anchor root must be an exact immutable caller root")
    publication_key = envelope.get("publication_key")
    occurrence = caller_root.require_occurrence(publication_key)
    if not occurrence.active:
        raise RuntimeError("inactive caller occurrence cannot authorize a publication envelope")
    expected_fields = {
        "publication_key": occurrence.publication_key,
        "external_contract_sha256": caller_root.external_contract_sha256,
        "manifest_sha256": caller_root.manifest_sha256,
        "phase": occurrence.phase,
        "wave_index": occurrence.wave_index,
        "generation_round": occurrence.generation_round,
        "call_index": occurrence.call_index,
        "dp_rank": occurrence.dp_rank,
        "caller_prompt_anchor_sha256": caller_root.sha256,
        "execution_occurrence_sha256": occurrence.sha256,
    }
    for field_name, expected in expected_fields.items():
        observed = envelope.get(field_name)
        if observed != expected or type(observed) is not type(expected):
            raise RuntimeError(f"rank publication drifted from caller prompt anchor field {field_name}")
    observed_requests = envelope.get("requests")
    if type(observed_requests) is not list or len(observed_requests) != len(occurrence.requests):
        raise RuntimeError("rank publication prompt anchor request count changed")
    observed_by_semantic_sha256 = {}
    for observed in observed_requests:
        if type(observed) is not dict:
            raise RuntimeError("rank publication prompt anchor request is malformed")
        semantic_sha256 = observed.get("semantic_request_sha256")
        if type(semantic_sha256) is not str or semantic_sha256 in observed_by_semantic_sha256:
            raise RuntimeError("rank publication semantic request identity is malformed or duplicated")
        observed_by_semantic_sha256[semantic_sha256] = observed
    if set(observed_by_semantic_sha256) != set(occurrence.semantic_request_sha256):
        raise RuntimeError("rank publication semantic request inventory differs from the caller root")
    for expected in occurrence.requests:
        observed = observed_by_semantic_sha256[expected.sha256]
        expected_request = {
            "request_id": expected.request_id,
            "global_request_index": expected.global_request_index,
            "request_index_in_dp_stream": expected.request_index_in_dp_stream,
            "prompt_token_count": expected.prompt_token_count,
            "prompt_token_ids_sha256": expected.prompt_token_ids_sha256,
            "semantic_request_sha256": expected.sha256,
            "seed": expected.request_seed,
        }
        for field_name, expected_value in expected_request.items():
            observed_value = observed.get(field_name)
            if observed_value != expected_value or type(observed_value) is not type(expected_value):
                raise RuntimeError(
                    f"rank publication request drifted from caller prompt anchor field {field_name}"
                )


def validate_nemo_rank_caller_ledger(
    envelope: dict[str, Any],
    caller_ledger_admission: GenerationRequestLedgerAdmission,
) -> None:
    """Bind one rank occurrence to the coordinator-retained generation ledger."""
    if type(envelope) is not dict:
        raise TypeError("rank publication envelope must be a built-in dict")
    if type(caller_ledger_admission) is not GenerationRequestLedgerAdmission:
        raise TypeError("rank caller ledger must be an exact immutable ledger")
    phase = envelope.get("phase")
    wave_index = envelope.get("wave_index")
    generation_round = envelope.get("generation_round")
    caller_ledger_admission.require_round_and_slot(
        generation_round,
        wave_index,
    )
    schedule = caller_ledger_admission.schedule_admission
    expected_wave_start = caller_ledger_admission.global_request_index_for(
        generation_round,
        wave_index,
        0,
    )
    expected_wave_count = caller_ledger_admission.request_count_for(
        generation_round,
        wave_index,
    )
    expected_fields = {
        "generation_caller_ledger_sha256": caller_ledger_admission.sha256,
        "generation_schedule_admission_sha256": schedule.sha256,
        "generation_semantic_namespace": (
            caller_ledger_admission.semantic_namespace_for(
                generation_round,
                wave_index,
            )
        ),
        "call_index": (
            (generation_round * schedule.batch_call_slots_per_round + wave_index)
            * schedule.turns_per_call
        ),
        "dp_size": schedule.data_parallel_size,
        "global_wave_start": expected_wave_start,
        "global_wave_stop": expected_wave_start + expected_wave_count,
        "global_wave_request_count": expected_wave_count,
    }
    if (
        type(phase) is not str
        or schedule.semantic_namespace_root.rsplit("/", 1)[-1] != phase
    ):
        raise RuntimeError("rank publication phase differs from the caller ledger")
    for field_name, expected in expected_fields.items():
        observed = envelope.get(field_name)
        if observed != expected or type(observed) is not type(expected):
            raise RuntimeError(
                f"rank publication drifted from caller ledger field {field_name}"
            )
    for digest_field in (
        "generation_request_envelope_sha256",
        "generation_execution_occurrence_sha256",
    ):
        digest = envelope.get(digest_field)
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"rank publication {digest_field} is malformed")
    dp_rank = envelope.get("dp_rank")
    if type(dp_rank) is not int or not 0 <= dp_rank < schedule.data_parallel_size:
        raise RuntimeError("rank publication DP rank differs from the caller ledger")
    requests = envelope.get("requests")
    if type(requests) is not list:
        raise RuntimeError("rank publication caller-ledger requests are malformed")
    for request in requests:
        global_request_index = request.get("global_request_index")
        local_ordinal = request.get("request_index_in_dp_stream")
        if (
            type(global_request_index) is not int
            or type(local_ordinal) is not int
            or not expected_wave_start
            <= global_request_index
            < expected_wave_start + expected_wave_count
        ):
            raise RuntimeError("rank publication request is outside the caller ledger wave")
        expected_seed = (
            schedule.base_seed
            + (expected_fields["call_index"] * schedule.data_parallel_size + dp_rank)
            * GENERATION_REQUEST_SEED_STRIDE
            + local_ordinal
        ) % GENERATION_REQUEST_SEED_MODULUS
        if request.get("seed") != expected_seed:
            raise RuntimeError("rank publication request seed differs from the caller ledger")


def rank_publication_contract_sha256(envelope: dict[str, Any]) -> str:
    """Digest caller-owned rank semantics while excluding transport-owned fields."""
    if type(envelope) is not dict:
        raise TypeError("rank publication envelope must be a built-in dict")
    semantic = dict(envelope)
    semantic.pop("publication_plan", None)
    semantic.pop("envelope_sha256", None)
    payload = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rank_publication_envelope_payload(envelope: dict[str, Any]) -> bytes:
    """Serialize one rank envelope as duplicate-key-checkable canonical wire bytes."""
    if type(envelope) is not dict:
        raise TypeError("rank publication envelope must be a built-in dict")
    return json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")


def reserve_nemo_rank_publications(
    output_path: str | Path,
    *,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    phases: tuple[str, ...],
    generation_round: int,
    global_call_index_start: int,
    external_contract_sha256: str,
    caller_ledgers_by_phase: dict[str, GenerationRequestLedgerAdmission],
    global_request_index_start: int = 0,
    phase_coordinates_by_name: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    dict[str, FilePublicationReservation],
    dict[str, FilePublicationPlan],
    dict[str, dict[str, Any]],
    NemoCallerPromptAnchorRoot,
]:
    """Reserve every active phase/wave/DP sidecar before any Ray worker launch."""
    if not isinstance(manifest, WorkloadManifest):
        raise TypeError("rank publication manifest must be a WorkloadManifest")
    if not isinstance(profile, Evo2VllmProfile):
        raise TypeError("rank publication profile must be an Evo2VllmProfile")
    if type(phases) is not tuple or not phases or len(set(phases)) != len(phases):
        raise ValueError("rank publication phases must be a nonempty unique tuple")
    if any(type(phase) is not str or _RANK_PUBLICATION_PHASE.fullmatch(phase) is None for phase in phases):
        raise ValueError("rank publication phase names contain unsafe characters")
    for label, value in (
        ("generation_round", generation_round),
        ("global_call_index_start", global_call_index_start),
        ("global_request_index_start", global_request_index_start),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a nonnegative built-in integer")
    if type(external_contract_sha256) is not str or not re.fullmatch(
        r"[0-9a-f]{64}", external_contract_sha256
    ):
        raise ValueError("external contract SHA256 must be a lowercase digest")

    waves = build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=profile.global_wave_size,
        replica_count=profile.replica_count,
    )
    if phase_coordinates_by_name is None:
        phase_coordinates = {
            phase: {
                "phase": phase,
                "generation_round": generation_round,
                "global_call_index_start": global_call_index_start,
                "global_request_index_start": global_request_index_start,
                "physical_calls_per_round": len(waves),
                "semantic_request_count": len(manifest.requests),
            }
            for phase in phases
        }
    else:
        if (
            type(phase_coordinates_by_name) is not dict
            or set(phase_coordinates_by_name) != set(phases)
        ):
            raise ValueError(
                "rank publication phase-coordinate inventory must exactly match the phases"
            )
        expected_coordinate_fields = {
            "phase",
            "sample_index",
            "generation_round",
            "global_call_index_start",
            "global_request_index_start",
            "physical_calls_per_round",
            "semantic_request_count",
        }
        phase_coordinates = {}
        for phase in phases:
            coordinate = phase_coordinates_by_name[phase]
            if (
                type(coordinate) is not dict
                or set(coordinate) != expected_coordinate_fields
            ):
                raise TypeError(
                    "rank publication phase coordinates must have exact built-in fields"
                )
            integer_fields = expected_coordinate_fields - {"phase"}
            if any(
                type(coordinate[field]) is not int or coordinate[field] < 0
                for field in integer_fields
            ):
                raise TypeError(
                    "rank publication phase coordinates must contain nonnegative built-in integers"
                )
            if (
                coordinate["phase"] != phase
                or coordinate["physical_calls_per_round"] != len(waves)
                or coordinate["semantic_request_count"] != len(manifest.requests)
                or coordinate["global_call_index_start"]
                != coordinate["generation_round"] * len(waves)
            ):
                raise ValueError(
                    "rank publication phase coordinates differ from the physical workload"
                )
            phase_coordinates[phase] = dict(coordinate)
        first_coordinate = phase_coordinates[phases[0]]
        if (
            first_coordinate["generation_round"] != generation_round
            or first_coordinate["global_call_index_start"]
            != global_call_index_start
            or first_coordinate["global_request_index_start"]
            != global_request_index_start
        ):
            raise ValueError(
                "rank publication phase-coordinate root differs from the supplied starting coordinates"
            )
    _require_exact_publication_mapping(
        caller_ledgers_by_phase,
        label="rank publication caller ledgers",
    )
    if set(caller_ledgers_by_phase) != set(phases):
        raise RuntimeError("rank publication caller ledger phase inventory is incomplete")
    expected_request_counts = tuple(wave.request_count for wave in waves)
    for phase in phases:
        coordinate = phase_coordinates[phase]
        ledger = caller_ledgers_by_phase[phase]
        if type(ledger) is not GenerationRequestLedgerAdmission:
            raise TypeError("rank publication caller ledgers must be exact immutable ledgers")
        schedule = ledger.schedule_admission
        if (
            schedule.semantic_namespace_root != f"{external_contract_sha256}/{phase}"
            or schedule.base_seed != manifest.seed
            or schedule.data_parallel_size != profile.replica_count
            or schedule.batch_call_slots_per_round != len(waves)
            or schedule.turns_per_call != 1
            or ledger.generation_round_start != coordinate["generation_round"]
            or ledger.generation_round_count != 1
            or ledger.global_request_index_start
            != coordinate["global_request_index_start"]
            or ledger.request_counts_by_batch_call_slot != expected_request_counts
        ):
            raise RuntimeError("rank publication caller ledger differs from the workload")
    payloads_by_digest: dict[str, NemoPromptPayloadRecord] = {}
    for request in manifest.requests:
        prompt_token_ids = tuple(request.prompt_token_ids)
        prompt_digest = generation_prompt_token_ids_sha256(prompt_token_ids)
        payload = NemoPromptPayloadRecord(
            prompt_token_ids=prompt_token_ids,
            prompt_token_ids_sha256=prompt_digest,
        )
        existing_payload = payloads_by_digest.get(prompt_digest)
        if existing_payload is not None and existing_payload.prompt_token_ids != prompt_token_ids:
            raise RuntimeError("caller prompt digest collision maps distinct token payloads")
        payloads_by_digest[prompt_digest] = payload

    semantic_by_digest: dict[str, NemoSemanticPromptRequestRecord] = {}
    occurrences = []
    for phase in phases:
        coordinate = phase_coordinates[phase]
        phase_generation_round = coordinate["generation_round"]
        phase_call_index_start = coordinate["global_call_index_start"]
        phase_request_index_start = coordinate["global_request_index_start"]
        for wave in waves:
            call_index = phase_call_index_start + wave.wave_index
            shards_by_rank = {shard.replica_index: shard for shard in wave.shards}
            for dp_rank in range(profile.replica_count):
                key = _rank_publication_key(phase, wave.wave_index, dp_rank)
                shard = shards_by_rank.get(dp_rank)
                semantic_requests = []
                request_indices = () if shard is None else range(shard.start, shard.stop)
                for local_index, request_index in enumerate(request_indices):
                    request = manifest.requests[request_index]
                    semantic_seed = request_seed(
                        manifest.seed,
                        call_index=call_index,
                        dp_rank=dp_rank,
                        dp_size=profile.replica_count,
                        request_index_in_stream=local_index,
                    )
                    semantic_request = NemoSemanticPromptRequestRecord(
                        request_id=request.request_id,
                        global_request_index=phase_request_index_start + request_index,
                        request_index_in_dp_stream=local_index,
                        generation_round=phase_generation_round,
                        call_index=call_index,
                        dp_rank=dp_rank,
                        request_seed=semantic_seed,
                        prompt_token_count=len(request.prompt_token_ids),
                        prompt_token_ids_sha256=generation_prompt_token_ids_sha256(
                            request.prompt_token_ids
                        ),
                    )
                    existing_semantic = semantic_by_digest.get(semantic_request.sha256)
                    if existing_semantic is not None and existing_semantic != semantic_request:
                        raise RuntimeError("caller semantic request digest collision")
                    semantic_by_digest[semantic_request.sha256] = semantic_request
                    semantic_requests.append(semantic_request)
                occurrences.append(
                    NemoRankPromptExecutionOccurrence(
                        publication_key=key,
                        phase=phase,
                        wave_index=wave.wave_index,
                        generation_round=phase_generation_round,
                        call_index=call_index,
                        dp_rank=dp_rank,
                        active=shard is not None,
                        requests=tuple(semantic_requests),
                    )
                )
    caller_root = NemoCallerPromptAnchorRoot(
        manifest_sha256=manifest.sha256,
        external_contract_sha256=external_contract_sha256,
        base_seed=manifest.seed,
        data_parallel_size=profile.replica_count,
        prompt_payload_catalog=tuple(
            payloads_by_digest[digest] for digest in sorted(payloads_by_digest)
        ),
        semantic_requests=tuple(
            semantic_by_digest[digest] for digest in sorted(semantic_by_digest)
        ),
        execution_occurrences=tuple(occurrences),
    )
    reservations: dict[str, FilePublicationReservation] = {}
    plans: dict[str, FilePublicationPlan] = {}
    envelopes: dict[str, dict[str, Any]] = {}
    try:
        for phase in phases:
            coordinate = phase_coordinates[phase]
            phase_generation_round = coordinate["generation_round"]
            phase_request_index_start = coordinate["global_request_index_start"]
            caller_ledger_admission = caller_ledgers_by_phase[phase]
            for wave in waves:
                wave_manifest = manifest.request_slice(wave.start, wave.stop)
                generation_request_envelope = build_nemo_request_envelope(
                    wave_manifest=wave_manifest,
                    wave=wave,
                    caller_ledger_admission=caller_ledger_admission,
                    generation_round=phase_generation_round,
                    batch_call_slot=wave.wave_index,
                    execution_attempt_epoch=0,
                )
                for shard in wave.shards:
                    key = _rank_publication_key(phase, wave.wave_index, shard.replica_index)
                    occurrence = caller_root.require_occurrence(key)
                    requests = [
                            {
                                "request_id": semantic_request.request_id,
                                "global_request_index": semantic_request.global_request_index,
                                "request_index_in_dp_stream": (
                                    semantic_request.request_index_in_dp_stream
                                ),
                                "prompt_token_count": semantic_request.prompt_token_count,
                                "prompt_token_ids_sha256": (
                                    semantic_request.prompt_token_ids_sha256
                                ),
                                "semantic_request_sha256": semantic_request.sha256,
                                "seed": semantic_request.request_seed,
                            }
                            for semantic_request in occurrence.requests
                    ]
                    envelope = {
                        "schema_version": NEMO_RANK_PUBLICATION_SCHEMA_VERSION,
                        "publication_key": key,
                        "external_contract_sha256": external_contract_sha256,
                        "manifest_sha256": manifest.sha256,
                        "phase": phase,
                        "wave_index": wave.wave_index,
                        "generation_round": phase_generation_round,
                        "call_index": occurrence.call_index,
                        "dp_rank": shard.replica_index,
                        "dp_size": profile.replica_count,
                        "tensor_parallel_size": profile.tensor_parallel_size,
                        "global_wave_start": phase_request_index_start + wave.start,
                        "global_wave_stop": phase_request_index_start + wave.stop,
                        "global_wave_request_count": wave.request_count,
                        "expected_request_count": shard.request_count,
                        "requested_max_new_tokens": manifest.max_new_tokens,
                        "caller_prompt_anchor_sha256": caller_root.sha256,
                        "execution_occurrence_sha256": occurrence.sha256,
                        "generation_caller_ledger_sha256": (
                            caller_ledger_admission.sha256
                        ),
                        "generation_schedule_admission_sha256": (
                            caller_ledger_admission.schedule_admission.sha256
                        ),
                        "generation_semantic_namespace": (
                            generation_request_envelope.semantic_namespace
                        ),
                        "generation_request_envelope_sha256": (
                            generation_request_envelope.sha256
                        ),
                        "generation_execution_occurrence_sha256": (
                            generation_request_envelope.execution_occurrence_sha256
                        ),
                        "requests": requests,
                    }
                    validate_nemo_rank_prompt_anchor(envelope, caller_root)
                    validate_nemo_rank_caller_ledger(
                        envelope,
                        caller_ledger_admission,
                    )
                    envelope_sha256 = rank_publication_contract_sha256(envelope)
                    reservation, plan = reserve_file_publication(
                        _rank_publication_path(
                            output_path,
                            phase=phase,
                            wave_index=wave.wave_index,
                            dp_rank=shard.replica_index,
                        ),
                        external_contract_sha256=external_contract_sha256,
                        payload_contract_sha256=envelope_sha256,
                        publication_key=key,
                    )
                    envelope["publication_plan"] = plan.to_dict()
                    envelope["envelope_sha256"] = envelope_sha256
                    reservations[key] = reservation
                    plans[key] = plan
                    envelopes[key] = envelope
        return reservations, plans, envelopes, caller_root
    except BaseException as error:
        for reservation in reservations.values():
            if reservation.closed:
                continue
            try:
                cancel_file_publication_reservation(reservation)
            except BaseException as cleanup_error:
                error.add_note(f"cancelling rank publication reservation failed: {cleanup_error!r}")
        raise


_RANK_PUBLICATION_OUTCOME_FIELDS = {
    "schema_version",
    "publication_key",
    "external_contract_sha256",
    "envelope_sha256",
    "caller_prompt_anchor_sha256",
    "execution_occurrence_sha256",
    "phase",
    "wave_index",
    "generation_round",
    "call_index",
    "dp_rank",
    "publisher",
    "payload_sha256",
    "payload_size_bytes",
    "row_count",
    "request_ids",
    "semantic_request_coordinates",
    "tp_sibling_evidence",
    "provisional_receipt",
}
_RANK_SIDECAR_ROW_FIELDS = {
    "schema_version",
    "request_id",
    "engine_request_id",
    "global_request_index",
    "request_index_in_dp_stream",
    "semantic_request_sha256",
    "seed",
    "generation_round",
    "call_index",
    "dp_rank",
    "phase",
    "wave_index",
    "prompt_token_ids",
    "output_token_ids",
    "chosen_token_logprobs",
    "selected_logprob_valid_count",
    "requested_max_tokens",
    "observed_prompt_tokens",
    "observed_new_tokens",
    "observed_total_tokens",
    "finish_reason",
    "stopped_on_eos",
    "external_contract_sha256",
    "envelope_sha256",
    "caller_prompt_anchor_sha256",
    "execution_occurrence_sha256",
}


def _require_exact_publication_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact built-in dict")
    if any(type(key) is not str or not key for key in value):
        raise TypeError(f"{label} keys must be nonempty exact built-in strings")
    return value


def _expected_rank_semantic_coordinates(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": request["request_id"],
            "global_request_index": request["global_request_index"],
            "request_index_in_dp_stream": request["request_index_in_dp_stream"],
            "semantic_request_sha256": request["semantic_request_sha256"],
            "seed": request["seed"],
            "generation_round": envelope["generation_round"],
            "call_index": envelope["call_index"],
            "dp_rank": envelope["dp_rank"],
        }
        for request in envelope["requests"]
    ]


def validate_nemo_rank_publication_outcome(
    envelope: dict[str, Any],
    outcome: Any,
    *,
    caller_prompt_anchor_root: NemoCallerPromptAnchorRoot,
    caller_ledger_admission: GenerationRequestLedgerAdmission,
) -> dict[str, Any]:
    """Reopen one worker sidecar and compare it to caller-owned coordinates."""
    validate_nemo_rank_prompt_anchor(envelope, caller_prompt_anchor_root)
    validate_nemo_rank_caller_ledger(envelope, caller_ledger_admission)
    if type(outcome) is not dict or set(outcome) != _RANK_PUBLICATION_OUTCOME_FIELDS:
        raise RuntimeError("rank publication outcome fields are not exact")
    if outcome.get("schema_version") != NEMO_RANK_PUBLICATION_OUTCOME_SCHEMA_VERSION:
        raise RuntimeError("rank publication outcome schema is unsupported")
    for field_name in (
        "publication_key",
        "external_contract_sha256",
        "envelope_sha256",
        "caller_prompt_anchor_sha256",
        "execution_occurrence_sha256",
        "phase",
        "wave_index",
        "generation_round",
        "call_index",
        "dp_rank",
    ):
        if outcome.get(field_name) != envelope.get(field_name) or type(outcome.get(field_name)) is not type(
            envelope.get(field_name)
        ):
            raise RuntimeError(f"rank publication outcome drifted from caller field {field_name}")
    if outcome.get("publisher") is not True:
        raise RuntimeError("rank publication outcome is not owned by the DP engine publisher")
    expected_coordinates = _expected_rank_semantic_coordinates(envelope)
    expected_request_ids = [request["request_id"] for request in envelope["requests"]]
    observed_request_ids = outcome.get("request_ids")
    if (
        type(observed_request_ids) is not list
        or any(type(request_id) is not str or not request_id for request_id in observed_request_ids)
        or len(observed_request_ids) != len(set(observed_request_ids))
        or set(observed_request_ids) != set(expected_request_ids)
    ):
        raise RuntimeError("rank publication request IDs drifted from the caller envelope")
    observed_coordinates = outcome.get("semantic_request_coordinates")
    coordinate_fields = set(expected_coordinates[0]) | {"engine_request_id"}
    if type(observed_coordinates) is not list or len(observed_coordinates) != len(expected_coordinates):
        raise RuntimeError("rank publication semantic coordinate inventory is incomplete")
    coordinates_by_semantic_sha256 = {}
    engine_request_ids = []
    for coordinate in observed_coordinates:
        if type(coordinate) is not dict or set(coordinate) != coordinate_fields:
            raise RuntimeError("rank publication semantic coordinate fields are not exact")
        semantic_sha256 = coordinate.get("semantic_request_sha256")
        engine_request_id = coordinate.get("engine_request_id")
        if (
            type(semantic_sha256) is not str
            or semantic_sha256 in coordinates_by_semantic_sha256
            or type(engine_request_id) is not str
            or not engine_request_id
        ):
            raise RuntimeError("rank publication semantic or engine request identity is malformed")
        coordinates_by_semantic_sha256[semantic_sha256] = coordinate
        engine_request_ids.append(engine_request_id)
    if len(set(engine_request_ids)) != len(engine_request_ids):
        raise RuntimeError("rank publication engine request identities are duplicated")
    if set(coordinates_by_semantic_sha256) != {
        coordinate["semantic_request_sha256"] for coordinate in expected_coordinates
    }:
        raise RuntimeError("rank publication semantic coordinate inventory is foreign")
    for expected_coordinate in expected_coordinates:
        observed_coordinate = coordinates_by_semantic_sha256[
            expected_coordinate["semantic_request_sha256"]
        ]
        for field_name, expected_value in expected_coordinate.items():
            observed_value = observed_coordinate.get(field_name)
            if observed_value != expected_value or type(observed_value) is not type(expected_value):
                raise RuntimeError(
                    f"rank publication semantic coordinates drifted from {field_name}"
                )
    if type(outcome.get("row_count")) is not int or outcome["row_count"] != len(expected_request_ids):
        raise RuntimeError("rank publication row count is not exact")
    receipt_value = outcome.get("provisional_receipt")
    if type(receipt_value) is not dict:
        raise RuntimeError("rank publication outcome is missing its provisional receipt")
    receipt = PublicationReceipt(**receipt_value)
    validate_publication_receipt(receipt)
    plan = FilePublicationPlan(**envelope["publication_plan"])
    if receipt.final_path != plan.final_path:
        raise RuntimeError("rank publication outcome names an unplanned sidecar")
    if (
        type(outcome.get("payload_sha256")) is not str
        or outcome["payload_sha256"] != receipt.sha256
        or type(outcome.get("payload_size_bytes")) is not int
        or outcome["payload_size_bytes"] != receipt.size_bytes
    ):
        raise RuntimeError("rank publication worker payload summary differs from the actual receipt")
    siblings = outcome.get("tp_sibling_evidence")
    if type(siblings) is not list or len(siblings) != envelope["tensor_parallel_size"]:
        raise RuntimeError("rank publication TP sibling set is incomplete")
    for tp_rank, sibling in enumerate(siblings):
        if type(sibling) is not dict or set(sibling) != {
            "tp_rank",
            "publisher",
            "payload_sha256",
            "payload_size_bytes",
            "envelope_sha256",
        }:
            raise RuntimeError("rank publication TP sibling fields are not exact")
        if (
            sibling["tp_rank"] != tp_rank
            or type(sibling["tp_rank"]) is not int
            or sibling["publisher"] is not (tp_rank == 0)
            or sibling["payload_sha256"] != receipt.sha256
            or sibling["payload_size_bytes"] != receipt.size_bytes
            or sibling["envelope_sha256"] != envelope["envelope_sha256"]
        ):
            raise RuntimeError("rank publication TP siblings did not attest identical bytes")

    snapshot = read_jsonl_snapshot(receipt.final_path, label="rank-local generation sidecar", compression="gzip")
    if snapshot.sha256 != receipt.sha256 or snapshot.size_bytes != receipt.size_bytes:
        raise RuntimeError("rank publication coordinator snapshot differs from the worker receipt")
    if len(snapshot.values) != len(envelope["requests"]):
        raise RuntimeError("rank publication coordinator row count differs from the caller envelope")
    rows_by_semantic_sha256 = {}
    for row in snapshot.values:
        if type(row) is not dict or set(row) != _RANK_SIDECAR_ROW_FIELDS:
            raise RuntimeError("rank publication sidecar row fields are not exact")
        if row.get("schema_version") != NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION:
            raise RuntimeError("rank publication sidecar row schema is unsupported")
        semantic_sha256 = row.get("semantic_request_sha256")
        if type(semantic_sha256) is not str or semantic_sha256 in rows_by_semantic_sha256:
            raise RuntimeError("rank publication sidecar semantic identity is malformed or duplicated")
        rows_by_semantic_sha256[semantic_sha256] = row
    expected_semantic_sha256 = {
        request["semantic_request_sha256"] for request in envelope["requests"]
    }
    if set(rows_by_semantic_sha256) != expected_semantic_sha256:
        raise RuntimeError("rank publication sidecar semantic inventory is foreign or incomplete")
    canonical_rows = []
    for expected_request, expected_coordinates_row in zip(
        envelope["requests"], expected_coordinates, strict=True
    ):
        row = rows_by_semantic_sha256[expected_request["semantic_request_sha256"]]
        for field_name, expected_value in expected_coordinates_row.items():
            if row.get(field_name) != expected_value or type(row.get(field_name)) is not type(expected_value):
                raise RuntimeError(f"rank publication sidecar row drifted from {field_name}")
        engine_request_id = row.get("engine_request_id")
        if (
            type(engine_request_id) is not str
            or not engine_request_id
            or engine_request_id
            != coordinates_by_semantic_sha256[expected_request["semantic_request_sha256"]][
                "engine_request_id"
            ]
        ):
            raise RuntimeError("rank publication sidecar engine request identity drifted")
        if (
            row.get("phase") != envelope["phase"]
            or row.get("wave_index") != envelope["wave_index"]
            or row.get("external_contract_sha256") != envelope["external_contract_sha256"]
            or row.get("envelope_sha256") != envelope["envelope_sha256"]
            or row.get("caller_prompt_anchor_sha256") != caller_prompt_anchor_root.sha256
            or row.get("execution_occurrence_sha256")
            != envelope["execution_occurrence_sha256"]
            or row.get("semantic_request_sha256")
            != expected_request["semantic_request_sha256"]
            or row.get("requested_max_tokens") != envelope["requested_max_new_tokens"]
            or row.get("observed_prompt_tokens") != expected_request["prompt_token_count"]
            or row.get("observed_new_tokens") != envelope["requested_max_new_tokens"]
            or row.get("selected_logprob_valid_count") != envelope["requested_max_new_tokens"]
            or row.get("observed_total_tokens")
            != expected_request["prompt_token_count"] + envelope["requested_max_new_tokens"]
            or row.get("finish_reason") != "length"
            or row.get("stopped_on_eos") is not False
        ):
            raise RuntimeError("rank publication sidecar lengths or termination evidence drifted")
        prompt_token_ids = row.get("prompt_token_ids")
        output_token_ids = row.get("output_token_ids")
        chosen_logprobs = row.get("chosen_token_logprobs")
        if (
                type(prompt_token_ids) is not list
                or any(type(token_id) is not int for token_id in prompt_token_ids)
                or generation_prompt_token_ids_sha256(prompt_token_ids)
                != expected_request["prompt_token_ids_sha256"]
            or type(output_token_ids) is not list
            or len(output_token_ids) != envelope["requested_max_new_tokens"]
            or any(type(token_id) is not int for token_id in output_token_ids)
            or type(chosen_logprobs) is not list
            or len(chosen_logprobs) != len(output_token_ids)
            or any(type(value) is not float or not math.isfinite(value) for value in chosen_logprobs)
        ):
            raise RuntimeError("rank publication sidecar tokens or chosen logprobs are malformed")
        caller_payload = caller_prompt_anchor_root.require_payload(
            expected_request["prompt_token_ids_sha256"]
        )
        canonical_row = dict(row)
        canonical_row.update(
            {
                "request_id": expected_request["request_id"],
                "global_request_index": expected_request["global_request_index"],
                "request_index_in_dp_stream": expected_request[
                    "request_index_in_dp_stream"
                ],
                "semantic_request_sha256": expected_request[
                    "semantic_request_sha256"
                ],
                "prompt_token_ids": list(caller_payload.prompt_token_ids),
                "caller_prompt_anchor_sha256": caller_prompt_anchor_root.sha256,
            }
        )
        canonical_rows.append(canonical_row)
    if len({row["engine_request_id"] for row in canonical_rows}) != len(canonical_rows):
        raise RuntimeError("rank publication sidecar engine request identities are duplicated")
    return {
        "publication_key": envelope["publication_key"],
        "receipt": receipt,
        "rows": tuple(canonical_rows),
        "coordinator_snapshot_sha256": snapshot.sha256,
        "passed": True,
    }


def publish_nemo_rank_wave(
    generation: Any,
    *,
    wave: Any,
    phase: str,
    envelopes: dict[str, dict[str, Any]],
    caller_prompt_anchor_roots: dict[str, NemoCallerPromptAnchorRoot],
    caller_ledger_admission: GenerationRequestLedgerAdmission,
    ray_get: Any,
) -> tuple[dict[str, Any], ...]:
    """Invoke one post-timing publisher RPC on each active DP leader."""
    _require_exact_publication_mapping(envelopes, label="rank publication envelopes")
    _require_exact_publication_mapping(
        caller_prompt_anchor_roots,
        label="rank publication caller prompt roots",
    )
    leader_indices = generation.worker_group.dp_leader_worker_indices
    preflight = []
    for shard in wave.shards:
        key = _rank_publication_key(phase, wave.wave_index, shard.replica_index)
        if key not in envelopes:
            raise RuntimeError(f"rank publication plan is missing before launch: {key}")
        if key not in caller_prompt_anchor_roots:
            raise RuntimeError(f"rank publication caller prompt root is missing before launch: {key}")
        if shard.replica_index >= len(leader_indices):
            raise RuntimeError("rank publication DP leader topology is incomplete")
        envelope = envelopes[key]
        caller_prompt_anchor_root = caller_prompt_anchor_roots[key]
        validate_nemo_rank_prompt_anchor(envelope, caller_prompt_anchor_root)
        validate_nemo_rank_caller_ledger(envelope, caller_ledger_admission)
        plan = validate_file_publication_plan(envelope.get("publication_plan"))
        if (
            plan.publication_key != key
            or plan.external_contract_sha256 != caller_prompt_anchor_root.external_contract_sha256
            or plan.payload_contract_sha256 != envelope.get("envelope_sha256")
        ):
            raise RuntimeError("rank publication marker plan differs from the external caller root")
        expected_envelope_sha256 = envelope["envelope_sha256"]
        preflight.append(
            (
                shard.replica_index,
                envelope,
                expected_envelope_sha256,
            )
        )

    futures = []
    for replica_index, envelope, expected_envelope_sha256 in preflight:
        futures.append(
            generation.worker_group.run_single_worker_single_data(
                "publish_evo2_generation_sidecar",
                leader_indices[replica_index],
                envelope_payload=rank_publication_envelope_payload(envelope),
                expected_envelope_sha256=expected_envelope_sha256,
            )
        )
    outcomes = list(ray_get(futures))
    if len(outcomes) != len(preflight):
        raise RuntimeError("rank publication worker outcome count is incomplete")
    outcome_by_key = {}
    for outcome in outcomes:
        if type(outcome) is not dict:
            raise RuntimeError("rank publication worker returned a malformed outcome")
        publication_key = outcome.get("publication_key")
        if type(publication_key) is not str or publication_key not in envelopes:
            raise RuntimeError("rank publication worker returned a foreign publication key")
        if publication_key in outcome_by_key:
            raise RuntimeError("rank publication worker returned a duplicate publication key")
        outcome_by_key[publication_key] = outcome
    expected_keys = tuple(envelope["publication_key"] for _rank, envelope, _sha in preflight)
    if set(outcome_by_key) != set(expected_keys):
        raise RuntimeError("rank publication worker outcome inventory is incomplete")
    ordered_outcomes = []
    for _rank, envelope, _sha in preflight:
        outcome = outcome_by_key[envelope["publication_key"]]
        validate_nemo_rank_publication_outcome(
            envelope,
            outcome,
            caller_prompt_anchor_root=caller_prompt_anchor_roots[
                envelope["publication_key"]
            ],
            caller_ledger_admission=caller_ledger_admission,
        )
        ordered_outcomes.append(outcome)
    return tuple(ordered_outcomes)


def finalize_nemo_rank_publications(
    *,
    reservations: dict[str, FilePublicationReservation],
    envelopes: dict[str, dict[str, Any]],
    caller_prompt_anchor_roots: dict[str, NemoCallerPromptAnchorRoot],
    caller_ledgers_by_phase: dict[str, GenerationRequestLedgerAdmission],
    outcomes: dict[str, dict[str, Any]],
    namespace_output_path: str | Path,
) -> dict[str, Any]:
    """Coordinator-finalize every rank receipt and reconstruct global wave order."""
    _require_exact_publication_mapping(reservations, label="rank publication reservations")
    _require_exact_publication_mapping(envelopes, label="rank publication envelopes")
    _require_exact_publication_mapping(outcomes, label="rank publication outcomes")
    _require_exact_publication_mapping(
        caller_prompt_anchor_roots,
        label="rank publication caller prompt roots",
    )
    _require_exact_publication_mapping(
        caller_ledgers_by_phase,
        label="rank publication caller ledgers",
    )
    if any(
        type(root) is not NemoCallerPromptAnchorRoot
        for root in caller_prompt_anchor_roots.values()
    ):
        raise TypeError("rank publication caller prompt roots must be exact immutable roots")
    if (
        set(reservations) != set(envelopes)
        or set(outcomes) != set(envelopes)
        or set(caller_prompt_anchor_roots) != set(envelopes)
    ):
        raise RuntimeError("rank publication reservation, anchor, envelope, and outcome key sets differ")
    phases = {envelope["phase"] for envelope in envelopes.values()}
    if set(caller_ledgers_by_phase) != phases:
        raise RuntimeError("rank publication caller ledger phase inventory differs")
    for key, reservation in reservations.items():
        envelope = envelopes[key]
        caller_prompt_anchor_root = caller_prompt_anchor_roots[key]
        validate_nemo_rank_prompt_anchor(envelope, caller_prompt_anchor_root)
        validate_nemo_rank_caller_ledger(
            envelope,
            caller_ledgers_by_phase[envelope["phase"]],
        )
        if (
            reservation.plan.payload_contract_sha256 != envelope.get("envelope_sha256")
            or reservation.plan.external_contract_sha256 != envelope.get("external_contract_sha256")
            or reservation.plan.publication_key != key
        ):
            raise RuntimeError("rank publication envelope drifted from the caller-owned reservation")
    validated = {
        key: validate_nemo_rank_publication_outcome(
            envelopes[key],
            outcomes[key],
            caller_prompt_anchor_root=caller_prompt_anchor_roots[key],
            caller_ledger_admission=caller_ledgers_by_phase[
                envelopes[key]["phase"]
            ],
        )
        for key in sorted(envelopes)
    }
    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for key in sorted(envelopes):
        envelope = envelopes[key]
        grouped.setdefault((envelope["phase"], envelope["wave_index"]), []).append(
            (envelope, validated[key])
        )
    reassembled_wave_sha256 = {}
    for (phase, wave_index), parts in sorted(grouped.items()):
        caller_ledger_admission = caller_ledgers_by_phase[phase]
        expected_wave_start = caller_ledger_admission.global_request_index_for(
            parts[0][0]["generation_round"],
            wave_index,
            0,
        )
        generation_requests = []
        for rank_envelope, _proof in parts:
            for request in rank_envelope["requests"]:
                generation_requests.append(
                    GenerationRequestIdentity(
                        local_request_id=request["request_id"],
                        global_request_index=request["global_request_index"],
                        target_data_parallel_rank=rank_envelope["dp_rank"],
                        original_batch_ordinal=(
                            request["global_request_index"] - expected_wave_start
                        ),
                        immutable_local_ordinal=request[
                            "request_index_in_dp_stream"
                        ],
                        prompt_token_ids_sha256=request[
                            "prompt_token_ids_sha256"
                        ],
                    )
                )
        generation_requests.sort(key=lambda request: request.original_batch_ordinal)
        expected_count = caller_ledger_admission.request_count_for(
            parts[0][0]["generation_round"],
            wave_index,
        )
        if tuple(
            request.original_batch_ordinal for request in generation_requests
        ) != tuple(range(expected_count)):
            raise RuntimeError(
                "rank publication cannot reconstruct the complete generation envelope"
            )
        reconstructed_envelope = GenerationRequestEnvelope(
            schedule_admission=caller_ledger_admission.schedule_admission,
            caller_ledger_sha256=caller_ledger_admission.sha256,
            semantic_namespace=caller_ledger_admission.semantic_namespace_for(
                parts[0][0]["generation_round"],
                wave_index,
            ),
            generation_round=parts[0][0]["generation_round"],
            batch_call_slot=wave_index,
            turn_index=0,
            execution_attempt_epoch=0,
            requests=tuple(generation_requests),
        )
        caller_ledger_admission.require_envelope(reconstructed_envelope)
        if any(
            rank_envelope["generation_request_envelope_sha256"]
            != reconstructed_envelope.sha256
            or rank_envelope["generation_execution_occurrence_sha256"]
            != reconstructed_envelope.execution_occurrence_sha256
            for rank_envelope, _proof in parts
        ):
            raise RuntimeError(
                "rank publication generation envelope differs from caller reconstruction"
            )
        rows = [row for _envelope, proof in parts for row in proof["rows"]]
        global_indices = [row["global_request_index"] for row in rows]
        expected_start = min(envelope["global_wave_start"] for envelope, _proof in parts)
        expected_stop = max(envelope["global_wave_stop"] for envelope, _proof in parts)
        if sorted(global_indices) != list(range(expected_start, expected_stop)):
            raise RuntimeError("rank publication DP partitions are overlapping or incomplete")
        rows.sort(key=lambda row: row["global_request_index"])
        wave_payload = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows
        )
        reassembled_wave_sha256[f"{phase}/wave-{wave_index:03d}"] = hashlib.sha256(wave_payload).hexdigest()

    durable_receipts = {}
    for key in sorted(reservations):
        receipt = finalize_reserved_publication(reservations[key], validated[key]["receipt"])
        register_output_namespace_publication(namespace_output_path, receipt)
        terminal_snapshot = read_jsonl_snapshot(
            receipt.final_path,
            label="terminal rank-local generation sidecar",
            compression="gzip",
        )
        if terminal_snapshot.sha256 != receipt.sha256:
            raise RuntimeError("terminal rank publication changed after coordinator finalization")
        durable_receipts[key] = {
            **receipt.to_dict(),
            "caller_prompt_anchor_sha256": caller_prompt_anchor_roots[key].sha256,
            "execution_occurrence_sha256": (
                caller_prompt_anchor_roots[key].require_occurrence(key).sha256
            ),
        }
    result_payload = json.dumps(reassembled_wave_sha256, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": NEMO_RANK_PUBLICATION_PROOF_SCHEMA_VERSION,
        "live_standard_engine_enqueue_wait_admission": False,
        "nemo_rl_gdpo_e2e_admission": False,
        "provenance_assurance_tier": "adapter_integrity",
        "trusted_node_reproducibility_admission": False,
        "full_runtime_adversarial_admission": False,
        "full_runtime_adversarial_admission_reason": (
            "controlled clean launch without concurrent writers; no administrator-enforced read-only runtime snapshot"
        ),
        "publication_count": len(durable_receipts),
        "durable_receipts": durable_receipts,
        "caller_prompt_anchor_sha256": {
            key: caller_prompt_anchor_roots[key].sha256 for key in sorted(envelopes)
        },
        "caller_ledger_sha256": {
            phase: caller_ledgers_by_phase[phase].sha256
            for phase in sorted(caller_ledgers_by_phase)
        },
        "execution_occurrence_sha256": {
            key: caller_prompt_anchor_roots[key].require_occurrence(key).sha256
            for key in sorted(envelopes)
        },
        "reassembled_wave_sha256": reassembled_wave_sha256,
        "result_sha256": hashlib.sha256(result_payload).hexdigest(),
        "passed": True,
    }


def terminalize_failed_nemo_rank_publications(
    reservations: dict[str, FilePublicationReservation],
    *,
    primary_error: BaseException | None = None,
) -> None:
    """Cancel every unfinalized plan while preserving the attempt's primary error."""
    if type(reservations) is not dict:
        raise TypeError("rank publication reservations must be a built-in dict")
    if primary_error is not None and not isinstance(primary_error, BaseException):
        raise TypeError("primary_error must be an exception or None")
    cleanup_failures: list[tuple[str, BaseException]] = []
    for key, reservation in sorted(reservations.items()):
        if type(key) is not str or not isinstance(reservation, FilePublicationReservation):
            cleanup_failures.append((str(key), TypeError("rank publication reservation entry is malformed")))
            continue
        if reservation.closed:
            continue
        try:
            cancel_file_publication_reservation(reservation)
        except BaseException as cleanup_error:
            cleanup_failures.append((key, cleanup_error))
        if not reservation.closed:
            cleanup_failures.append((key, RuntimeError("rank publication reservation remained live after cleanup")))
    if not cleanup_failures:
        return
    if primary_error is not None:
        for key, cleanup_error in cleanup_failures:
            primary_error.add_note(f"terminalizing rank publication {key!r} failed: {cleanup_error!r}")
        return
    error = RuntimeError("one or more rank publication reservations could not be terminalized")
    for key, cleanup_error in cleanup_failures:
        error.add_note(f"terminalizing rank publication {key!r} failed: {cleanup_error!r}")
    raise error from cleanup_failures[0][1]


def cached_tokens_from_nemo_generation_output(
    outputs: BatchedDataDict,
    *,
    request_count: int,
) -> tuple[int, ...]:
    """Read exact scheduler cache hits from the recombined production output."""
    values = outputs.get("generation_num_cached_tokens")
    if not isinstance(values, torch.Tensor):
        raise AssertionError("production NeMo-RL output is missing cached-token telemetry")
    if values.ndim != 1 or values.shape[0] != request_count:
        raise AssertionError("production cached-token telemetry must align with every request")
    if values.dtype == torch.bool or torch.is_floating_point(values) or torch.is_complex(values):
        raise AssertionError("production cached-token telemetry must use an integer tensor")
    result = tuple(int(value) for value in values.tolist())
    if any(value < 0 for value in result):
        raise AssertionError("production cached-token telemetry cannot be negative")
    return result


def build_nemo_generation_config(
    profile: Evo2VllmProfile,
    manifest: WorkloadManifest,
    *,
    checkpoint: Path,
    load_format: str,
    num_logprobs: int = 0,
    require_rank_publication: bool = False,
) -> dict[str, Any]:
    """Build a complete NeMo-RL vLLM config for one exact workload."""
    if num_logprobs < 0:
        raise ValueError("num_logprobs must be nonnegative")
    if type(require_rank_publication) is not bool:
        raise TypeError("require_rank_publication must be a built-in bool")
    if profile.max_model_len < manifest.max_total_tokens:
        raise ValueError(
            f"profile max_model_len={profile.max_model_len} is smaller than "
            f"workload max_total_tokens={manifest.max_total_tokens}"
        )

    config = profile.nemo_rl_generation_config(
        load_format=load_format,
        request_seed=manifest.seed,
    )
    config.update(
        {
            "model_name": str(checkpoint.resolve()),
            "dtype": manifest.dtype,
            "max_new_tokens": manifest.max_new_tokens,
            "temperature": manifest.temperature,
            "top_p": manifest.top_p,
            "top_k": manifest.top_k,
            "stop_token_ids": list(manifest.stop_token_ids),
            "stop_strings": None,
            "ignore_eos": manifest.ignore_eos,
            "require_explicit_request_envelope": True,
            "evo2_rank_publication_required": require_rank_publication,
            "_pad_token_id": 0,
            "colocated": {
                "enabled": False,
                "resources": {"gpus_per_node": 2, "num_nodes": 1},
            },
        }
    )
    if num_logprobs:
        config["num_logprobs"] = num_logprobs
        config["vllm_kwargs"]["max_logprobs"] = num_logprobs
    return config


def build_nemo_generation_input(manifest: WorkloadManifest) -> BatchedDataDict:
    """Right-pad prompts for transport while preserving every semantic length."""
    prompt_lengths = torch.tensor(
        [len(request.prompt_token_ids) for request in manifest.requests],
        dtype=torch.long,
    )
    max_prompt_length = int(prompt_lengths.max().item())
    input_ids = torch.zeros(
        (len(manifest.requests), max_prompt_length),
        dtype=torch.long,
    )
    for row_index, request in enumerate(manifest.requests):
        prompt = torch.tensor(request.prompt_token_ids, dtype=torch.long)
        input_ids[row_index, : len(prompt)] = prompt
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": prompt_lengths,
        }
    )


def _require_aligned_tensor(
    outputs: BatchedDataDict,
    key: str,
    *,
    request_count: int,
) -> torch.Tensor:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"NeMo-RL generation output {key!r} must be a tensor")
    if value.ndim != 1 or value.shape[0] != request_count:
        raise ValueError(
            f"NeMo-RL generation output {key!r} must have shape ({request_count},), got {tuple(value.shape)}"
        )
    return value.detach().cpu()


def records_from_nemo_generation_output(
    manifest: WorkloadManifest,
    outputs: BatchedDataDict,
) -> tuple[
    tuple[GenerationRecord, ...],
    tuple[RequestExecutionRecord, ...],
    dict[str, tuple[float, ...]],
]:
    """Validate and retain exact outputs, RNG ownership, and timing from NeMo-RL."""
    request_count = len(manifest.requests)
    validate_generation_selected_logprobs(outputs)
    validate_generation_request_metadata(
        outputs,
        request_count=request_count,
        require_explicit_envelope=True,
        expected_base_seed=manifest.seed,
    )
    output_ids = outputs.get("output_ids")
    logprobs = outputs.get("logprobs")
    if not isinstance(output_ids, torch.Tensor) or output_ids.ndim != 2:
        raise TypeError("NeMo-RL generation output 'output_ids' must be a rank-2 tensor")
    if not isinstance(logprobs, torch.Tensor) or logprobs.shape != output_ids.shape:
        raise ValueError("NeMo-RL generation logprobs must match output_ids exactly")
    if output_ids.shape[0] != request_count:
        raise ValueError(f"NeMo-RL returned {output_ids.shape[0]} outputs for {request_count} requests")
    output_ids = output_ids.detach().cpu()
    logprobs = logprobs.detach().cpu()

    generation_lengths = _require_aligned_tensor(
        outputs,
        "generation_lengths",
        request_count=request_count,
    )
    expected_generation_lengths = torch.full_like(
        generation_lengths,
        manifest.max_new_tokens,
    )
    if not torch.equal(generation_lengths, expected_generation_lengths):
        raise ValueError(
            "NeMo-RL generation did not return the requested exact output length: "
            f"expected={manifest.max_new_tokens}, actual={generation_lengths.tolist()}"
        )

    unpadded_lengths = _require_aligned_tensor(
        outputs,
        "unpadded_sequence_lengths",
        request_count=request_count,
    )
    truncated = _require_aligned_tensor(outputs, "truncated", request_count=request_count)
    if not bool(torch.all(truncated)):
        raise ValueError("exact-length generation must finish every request by max-token truncation")

    metadata = {
        key: _require_aligned_tensor(outputs, key, request_count=request_count)
        for key in (
            "generation_request_seeds",
            "generation_global_request_indices",
            "generation_rounds",
            "generation_flattened_call_indices",
            "generation_dp_ranks",
        )
    }
    ttft = _require_aligned_tensor(
        outputs,
        "generation_first_token_latency_s",
        request_count=request_count,
    ).to(torch.float64)
    decode = _require_aligned_tensor(
        outputs,
        "generation_decode_s",
        request_count=request_count,
    ).to(torch.float64)
    if not bool(torch.all(torch.isfinite(ttft))) or not bool(torch.all(ttft >= 0)):
        raise ValueError("first-token latency must be finite and nonnegative")
    if not bool(torch.all(torch.isfinite(decode))) or not bool(torch.all(decode >= 0)):
        raise ValueError("decode timing must be finite and nonnegative")

    records = []
    executions = []
    for row_index, request in enumerate(manifest.requests):
        prompt_length = len(request.prompt_token_ids)
        total_length = prompt_length + manifest.max_new_tokens
        if int(unpadded_lengths[row_index]) != total_length:
            raise ValueError(
                f"request {request.request_id} has unpadded length "
                f"{int(unpadded_lengths[row_index])}, expected {total_length}"
            )
        if output_ids.shape[1] < total_length:
            raise ValueError(
                f"request {request.request_id} output width {output_ids.shape[1]} "
                f"is shorter than exact length {total_length}"
            )
        actual_prompt = tuple(int(token) for token in output_ids[row_index, :prompt_length])
        if actual_prompt != request.prompt_token_ids:
            raise ValueError(f"request {request.request_id} prompt tokens changed during generation")

        generated_ids = tuple(int(token) for token in output_ids[row_index, prompt_length:total_length])
        generated_logprobs = tuple(float(value) for value in logprobs[row_index, prompt_length:total_length])
        if not all(math.isfinite(value) for value in generated_logprobs):
            raise ValueError(f"request {request.request_id} has non-finite chosen-token logprobs")
        records.append(
            GenerationRecord(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                output_token_ids=generated_ids,
                output_logprobs=generated_logprobs,
                requested_max_tokens=manifest.max_new_tokens,
                finish_reason="length",
                stop_reason=None,
                stopped_on_eos=False,
            )
        )
        executions.append(
            RequestExecutionRecord(
                request_id=request.request_id,
                global_request_index=int(metadata["generation_global_request_indices"][row_index]),
                generation_round=int(metadata["generation_rounds"][row_index]),
                dp_rank=int(metadata["generation_dp_ranks"][row_index]),
                call_index=int(
                    metadata["generation_flattened_call_indices"][row_index]
                ),
                seed=int(metadata["generation_request_seeds"][row_index]),
            )
        )

    global_indices = [record.global_request_index for record in executions]
    seeds = [record.seed for record in executions]
    if len(global_indices) != len(set(global_indices)):
        raise ValueError("NeMo-RL returned duplicate global request ownership")
    if len(seeds) != len(set(seeds)):
        raise ValueError("NeMo-RL returned duplicate per-request seeds")

    return (
        tuple(records),
        tuple(executions),
        {
            "ttft_s": tuple(float(value) for value in ttft),
            "decode_s": tuple(float(value) for value in decode),
        },
    )


def full_vocab_logprob_evidence_from_nemo_output(
    outputs: BatchedDataDict,
    *,
    records: tuple[GenerationRecord, ...],
    require_full: bool,
    expected_finite_support: int | None = None,
) -> dict[str, Any]:
    """Validate and retain per-step processed-logprob distributions."""
    dense = outputs.get("generation_vocab_logprobs")
    counts = outputs.get("generation_logprob_counts")
    if not isinstance(dense, torch.Tensor) or dense.ndim != 3:
        raise TypeError("NeMo-RL full-vocabulary logprobs must be a rank-3 tensor")
    if not isinstance(counts, torch.Tensor) or counts.shape != dense.shape[:2]:
        raise ValueError("NeMo-RL logprob coverage counts must align with every generated position")
    if dense.shape[0] != len(records):
        raise ValueError("NeMo-RL full-vocabulary logprobs must align with every output record")

    dense = dense.detach().cpu().to(torch.float32)
    counts = counts.detach().cpu().to(torch.long)
    vocab_size = dense.shape[2]
    if expected_finite_support is None and require_full:
        expected_finite_support = vocab_size
    if expected_finite_support is not None and not 1 <= expected_finite_support <= vocab_size:
        raise ValueError("expected finite logprob support must be within the model vocabulary")

    generation_lengths = [len(record.output_token_ids) for record in records]
    if any(dense.shape[1] < length for length in generation_lengths):
        raise ValueError("NeMo-RL full-vocabulary evidence is shorter than its generated output")
    finite_count_tensor = torch.isfinite(dense).sum(dim=-1)
    valid_coordinates = [
        (row_index, step)
        for row_index, generation_length in enumerate(generation_lengths)
        for step in range(generation_length)
    ]
    reported_values = [int(counts[row_index, step]) for row_index, step in valid_coordinates]
    finite_values = [int(finite_count_tensor[row_index, step]) for row_index, step in valid_coordinates]

    if expected_finite_support == vocab_size and reported_values != finite_values:
        mismatch_coordinates = [
            (row_index, step, reported, finite)
            for (row_index, step), reported, finite in zip(
                valid_coordinates,
                reported_values,
                finite_values,
                strict=True,
            )
            if reported != finite
        ]
        row_index, step, reported, finite = mismatch_coordinates[0]
        raise AssertionError(
            f"shape={list(dense.shape)}, "
            f"mismatched_positions={len(mismatch_coordinates)}/{len(valid_coordinates)}, "
            f"reported_counts={dict(sorted(Counter(reported_values).items()))}, "
            f"finite_counts={dict(sorted(Counter(finite_values).items()))}, "
            "first_mismatch=("
            f"request={row_index}, step={step}, reported={reported}, finite={finite})"
        )

    retained: list[list[list[float | None]]] = []
    retained_counts: list[list[int]] = []
    retained_finite_counts: list[list[int]] = []
    retained_negative_infinity_counts: list[list[int]] = []
    chosen_token_ids: list[list[int]] = []
    chosen_token_logprobs: list[list[float]] = []
    for row_index, record in enumerate(records):
        generation_length = len(record.output_token_ids)
        row = dense[row_index, :generation_length]
        row_counts = counts[row_index, :generation_length]
        finite_counts = torch.isfinite(row).sum(dim=-1)
        if bool(torch.any(row_counts <= 0)) or bool(torch.any(row_counts > vocab_size)):
            raise AssertionError("logprob coverage count is outside the model vocabulary")
        if require_full and not bool(torch.all(row_counts == vocab_size)):
            raise AssertionError("full-vocabulary evidence omitted one or more token logprobs")
        if bool(torch.any(torch.isnan(row))) or bool(torch.any(torch.isposinf(row))):
            raise AssertionError("processed logprob evidence contains NaN or positive infinity")
        if expected_finite_support is not None and not bool(torch.all(finite_counts == expected_finite_support)):
            raise AssertionError(
                "processed logprob finite support does not match the configured sampling policy: "
                f"expected={expected_finite_support}, actual={finite_counts.tolist()}"
            )

        row_chosen_token_ids = []
        row_chosen_logprobs = []
        for position, (token_id, chosen_logprob) in enumerate(
            zip(record.output_token_ids, record.output_logprobs, strict=True)
        ):
            if not 0 <= token_id < vocab_size:
                raise AssertionError(f"generated token {token_id} is outside vocabulary size {vocab_size}")
            distribution_logprob = float(row[position, token_id])
            if not math.isfinite(distribution_logprob) or not math.isfinite(chosen_logprob):
                raise AssertionError("chosen token is outside finite processed support")
            if struct.pack(">d", distribution_logprob) != struct.pack(">d", float(chosen_logprob)):
                raise AssertionError(
                    "chosen-token logprob does not bitwise match its full-vocabulary distribution entry"
                )
            row_chosen_token_ids.append(token_id)
            row_chosen_logprobs.append(distribution_logprob)

        retained.append(
            [[float(value) if math.isfinite(float(value)) else None for value in position] for position in row]
        )
        retained_counts.append([int(value) for value in row_counts])
        retained_finite_counts.append([int(value) for value in finite_counts])
        retained_negative_infinity_counts.append([int(value) for value in torch.isneginf(row).sum(dim=-1)])
        chosen_token_ids.append(row_chosen_token_ids)
        chosen_token_logprobs.append(row_chosen_logprobs)

    return {
        "shape": [len(records), dense.shape[1], vocab_size],
        "coverage_counts": retained_counts,
        "finite_support_counts": retained_finite_counts,
        "negative_infinity_counts": retained_negative_infinity_counts,
        "expected_finite_support": expected_finite_support,
        "chosen_token_oracle_passed": True,
        "chosen_token_in_finite_support": True,
        "chosen_token_ids": chosen_token_ids,
        "chosen_token_logprobs": chosen_token_logprobs,
        "logprobs": retained,
    }


@dataclass(frozen=True)
class NemoGenerationPhaseResult:
    """One exact production NeMo-RL generation phase across all request waves."""

    phase: str
    caller_ledger_admission: GenerationRequestLedgerAdmission
    sample: BenchmarkSample
    generation_call_s: tuple[float, ...]
    rank_publication_s: tuple[float, ...]
    coordinator_aggregate_sidecar_s: float
    operational_phase_s: float
    output_summaries: tuple[dict[str, Any], ...]
    request_executions: tuple[RequestExecutionRecord, ...]
    full_output_artifact: dict[str, Any]
    wave_proofs: tuple[dict[str, Any], ...]
    proof_collected: bool
    prefix_cache_reset: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe production phase proof."""
        return {
            "phase": self.phase,
            "caller_ledger_admission": self.caller_ledger_admission.to_dict(),
            "caller_ledger_sha256": self.caller_ledger_admission.sha256,
            "sample": self.sample.to_dict(),
            "generation_call_s": list(self.generation_call_s),
            "generation_timing_authority": COORDINATOR_GENERATION_TIMING_AUTHORITY,
            "coordinator_generation_wall_s": self.sample.generation_s,
            "rank_publication_s": list(self.rank_publication_s),
            "coordinator_aggregate_sidecar_s": self.coordinator_aggregate_sidecar_s,
            "operational_phase_s": self.operational_phase_s,
            "wave_execution": wave_execution_summary(self.wave_proofs),
            "outputs": list(self.output_summaries),
            "request_executions": [record.to_dict() for record in self.request_executions],
            "full_output_artifact": self.full_output_artifact,
            "waves": list(self.wave_proofs),
            "proof_collected": self.proof_collected,
            "prefix_cache_reset": self.prefix_cache_reset,
        }


def _proof_rpc(
    generation: Any,
    method_name: str,
    *,
    phase: str,
    ray_get: Any,
) -> list[dict[str, Any]]:
    futures = generation.worker_group.run_all_workers_single_data(
        method_name,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        phase=phase,
    )
    return list(ray_get(futures))


def snapshot_and_validate_nemo_resolved_configs(
    generation: Any,
    *,
    profile: Evo2VllmProfile,
    ray_get: Any,
) -> list[dict[str, Any]]:
    """Snapshot every DP engine config once, outside timed generation."""
    futures = generation.worker_group.run_all_workers_single_data(
        "snapshot_evo2_resolved_config",
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )
    resolved = list(ray_get(futures))
    if len(resolved) != profile.replica_count:
        raise AssertionError("NeMo-RL resolved-config count does not match the DP topology")
    for config in resolved:
        validate_resolved_profile(profile, config)
    return resolved


def _validate_wave_execution(
    manifest: WorkloadManifest,
    executions: tuple[RequestExecutionRecord, ...],
    *,
    expected_envelope: GenerationRequestEnvelope,
) -> None:
    if len(executions) != len(manifest.requests):
        raise AssertionError("execution metadata must cover every wave request")
    actual_ids = tuple(record.request_id for record in executions)
    expected_ids = tuple(request.request_id for request in manifest.requests)
    if actual_ids != expected_ids:
        raise AssertionError("NeMo-RL output ownership changed request order")
    if len(expected_envelope.requests) != len(executions):
        raise AssertionError("request envelope does not cover every returned execution")
    for record, request in zip(
        executions, expected_envelope.requests, strict=True
    ):
        if record.request_id != request.local_request_id:
            raise AssertionError("NeMo-RL semantic request identity changed")
        if record.global_request_index != request.global_request_index:
            raise AssertionError("NeMo-RL global request index changed")
        if record.dp_rank != request.target_data_parallel_rank:
            raise AssertionError("NeMo-RL DP ownership changed")
        if record.generation_round != expected_envelope.generation_round:
            raise AssertionError("NeMo-RL semantic generation round changed")
        if record.call_index != expected_envelope.flattened_call_index:
            raise AssertionError("NeMo-RL physical generation call changed")
        if record.seed != expected_envelope.seed_for(request):
            raise AssertionError("NeMo-RL request seed changed from its immutable envelope")


def build_nemo_request_envelope(
    *,
    wave_manifest: WorkloadManifest,
    wave: Any,
    caller_ledger_admission: GenerationRequestLedgerAdmission,
    generation_round: int,
    batch_call_slot: int,
    execution_attempt_epoch: int,
) -> GenerationRequestEnvelope:
    """Build one caller-owned request envelope from manifest and physical shards."""
    request_count = len(wave_manifest.requests)
    if request_count != wave.request_count:
        raise ValueError("wave manifest does not match the physical request wave")
    route_by_position: dict[int, tuple[int, int]] = {}
    for shard in wave.shards:
        for immutable_local_ordinal, global_position in enumerate(
            range(shard.start, shard.stop)
        ):
            local_position = global_position - wave.start
            if local_position in route_by_position:
                raise RuntimeError("physical request wave assigned one row more than once")
            route_by_position[local_position] = (
                shard.replica_index,
                immutable_local_ordinal,
            )
    if sorted(route_by_position) != list(range(request_count)):
        raise RuntimeError("physical request wave does not cover every manifest row")
    requests = tuple(
        GenerationRequestIdentity(
            local_request_id=request.request_id,
            global_request_index=caller_ledger_admission.global_request_index_for(
                generation_round,
                batch_call_slot,
                local_position,
            ),
            target_data_parallel_rank=route_by_position[local_position][0],
            original_batch_ordinal=local_position,
            immutable_local_ordinal=route_by_position[local_position][1],
            prompt_token_ids_sha256=generation_prompt_token_ids_sha256(
                request.prompt_token_ids
            ),
        )
        for local_position, request in enumerate(wave_manifest.requests)
    )
    envelope = GenerationRequestEnvelope(
        schedule_admission=caller_ledger_admission.schedule_admission,
        caller_ledger_sha256=caller_ledger_admission.sha256,
        semantic_namespace=caller_ledger_admission.semantic_namespace_for(
            generation_round, batch_call_slot
        ),
        generation_round=generation_round,
        batch_call_slot=batch_call_slot,
        turn_index=0,
        execution_attempt_epoch=execution_attempt_epoch,
        requests=requests,
    )
    caller_ledger_admission.require_envelope(envelope)
    if tuple(
        request.original_batch_ordinal for request in envelope.requests
    ) != tuple(range(request_count)):
        raise RuntimeError("request envelope does not fill the caller-ledger slot in order")
    return envelope


def build_nemo_generation_caller_ledgers(
    *,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    phases: tuple[str, ...],
    generation_round: int,
    global_request_index_start: int,
    request_envelope_namespace: str,
    phase_coordinates_by_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, GenerationRequestLedgerAdmission]:
    """Freeze one external caller ledger per phase before reservation or worker work."""
    if type(phases) is not tuple or not phases or len(set(phases)) != len(phases):
        raise ValueError("caller ledger phases must be a nonempty unique tuple")
    if any(type(phase) is not str or not phase for phase in phases):
        raise TypeError("caller ledger phases must be nonempty built-in strings")
    if type(generation_round) is not int or generation_round < 0:
        raise ValueError("generation_round must be a nonnegative built-in integer")
    if type(global_request_index_start) is not int or global_request_index_start < 0:
        raise ValueError(
            "global_request_index_start must be a nonnegative built-in integer"
        )
    if (
        type(request_envelope_namespace) is not str
        or not request_envelope_namespace
    ):
        raise TypeError("request_envelope_namespace must be a nonempty built-in string")
    waves = build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=profile.global_wave_size,
        replica_count=profile.replica_count,
    )
    request_counts = tuple(wave.request_count for wave in waves)
    if phase_coordinates_by_name is None:
        phase_coordinates = {
            phase: {
                "phase": phase,
                "generation_round": generation_round,
                "global_call_index_start": generation_round * len(waves),
                "global_request_index_start": global_request_index_start,
                "physical_calls_per_round": len(waves),
                "semantic_request_count": len(manifest.requests),
            }
            for phase in phases
        }
    else:
        if type(phase_coordinates_by_name) is not dict or set(phase_coordinates_by_name) != set(phases):
            raise ValueError("caller phase-coordinate inventory must exactly match the requested phases")
        expected_fields = {
            "phase",
            "sample_index",
            "generation_round",
            "global_call_index_start",
            "global_request_index_start",
            "physical_calls_per_round",
            "semantic_request_count",
        }
        phase_coordinates = {}
        for phase in phases:
            coordinate = phase_coordinates_by_name[phase]
            if type(coordinate) is not dict or set(coordinate) != expected_fields:
                raise TypeError("caller phase coordinates must have exact built-in fields")
            integer_fields = expected_fields - {"phase"}
            if any(type(coordinate[field]) is not int or coordinate[field] < 0 for field in integer_fields):
                raise TypeError("caller phase coordinates must contain nonnegative built-in integers")
            if (
                coordinate["phase"] != phase
                or coordinate["physical_calls_per_round"] != len(waves)
                or coordinate["semantic_request_count"] != len(manifest.requests)
                or coordinate["global_call_index_start"] != coordinate["generation_round"] * len(waves)
            ):
                raise ValueError("caller phase coordinates differ from the physical workload")
            phase_coordinates[phase] = dict(coordinate)
        first = phase_coordinates[phases[0]]
        if (
            first["generation_round"] != generation_round
            or first["global_request_index_start"] != global_request_index_start
        ):
            raise ValueError("caller phase-coordinate root differs from the supplied starting coordinates")
    return {
        phase: GenerationRequestLedgerAdmission(
            schedule_admission=GenerationRequestScheduleAdmission(
                semantic_namespace_root=f"{request_envelope_namespace}/{phase}",
                base_seed=manifest.seed,
                data_parallel_size=profile.replica_count,
                batch_call_slots_per_round=len(waves),
                turns_per_call=1,
            ),
            generation_round_start=phase_coordinates[phase]["generation_round"],
            generation_round_count=1,
            global_request_index_start=phase_coordinates[phase]["global_request_index_start"],
            request_counts_by_batch_call_slot=request_counts,
        )
        for phase in phases
    }


def _preflight_nemo_generation_phase(
    *,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    phase: str,
    generation_round: int,
    global_call_index_start: int,
    global_request_index_start: int,
    request_envelope_namespace: str,
    caller_ledger_admission: GenerationRequestLedgerAdmission,
    rank_publication_envelopes: dict[str, dict[str, Any]] | None,
    rank_prompt_anchor_roots: dict[str, NemoCallerPromptAnchorRoot] | None,
) -> tuple[
    GenerationRequestLedgerAdmission,
    dict[int, tuple[WorkloadManifest, GenerationRequestEnvelope, BatchedDataDict]],
]:
    """Authorize every wave, rank, prompt, seed, and marker before provider work."""
    if (rank_publication_envelopes is None) != (rank_prompt_anchor_roots is None):
        raise ValueError("rank publication envelopes and caller prompt roots must be supplied together")
    if rank_publication_envelopes is not None:
        _require_exact_publication_mapping(
            rank_publication_envelopes,
            label="rank publication envelopes",
        )
        _require_exact_publication_mapping(
            rank_prompt_anchor_roots,
            label="rank publication caller prompt roots",
        )

    wave_contexts = {}
    waves = build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=profile.global_wave_size,
        replica_count=profile.replica_count,
    )
    if type(caller_ledger_admission) is not GenerationRequestLedgerAdmission:
        raise TypeError(
            "caller_ledger_admission must be an exact GenerationRequestLedgerAdmission"
        )
    schedule_admission = caller_ledger_admission.schedule_admission
    if (
        schedule_admission.semantic_namespace_root
        != f"{request_envelope_namespace}/{phase}"
        or schedule_admission.base_seed != manifest.seed
        or schedule_admission.data_parallel_size != profile.replica_count
        or schedule_admission.batch_call_slots_per_round != len(waves)
        or schedule_admission.turns_per_call != 1
        or caller_ledger_admission.generation_round_start != generation_round
        or caller_ledger_admission.generation_round_count != 1
        or caller_ledger_admission.global_request_index_start
        != global_request_index_start
        or caller_ledger_admission.request_counts_by_batch_call_slot
        != tuple(wave.request_count for wave in waves)
    ):
        raise ValueError("caller ledger differs from the exact phase workload")
    expected_batch_call_index_start = generation_round * len(waves)
    if global_call_index_start != expected_batch_call_index_start:
        raise ValueError(
            "phase global_call_index_start contradicts its frozen batch-call schedule"
        )
    for wave in waves:
        wave_manifest = manifest.request_slice(wave.start, wave.stop)
        request_envelope = build_nemo_request_envelope(
            wave_manifest=wave_manifest,
            wave=wave,
            caller_ledger_admission=caller_ledger_admission,
            generation_round=generation_round,
            batch_call_slot=wave.wave_index,
            execution_attempt_epoch=0,
        )
        generation_input = build_nemo_generation_input(wave_manifest)
        if rank_publication_envelopes is not None:
            active_keys = tuple(
                _rank_publication_key(phase, wave.wave_index, shard.replica_index)
                for shard in wave.shards
            )
            roots = tuple(rank_prompt_anchor_roots[key] for key in active_keys)
            if not roots or any(type(root) is not NemoCallerPromptAnchorRoot for root in roots):
                raise TypeError("phase preflight requires exact immutable caller prompt roots")
            caller_root = roots[0]
            if any(root is not caller_root for root in roots):
                raise RuntimeError("one physical wave is split across different caller roots")
            if (
                caller_root.manifest_sha256 != manifest.sha256
                or caller_root.base_seed != manifest.seed
                or caller_root.data_parallel_size != profile.replica_count
            ):
                raise RuntimeError("caller prompt root differs from manifest seed or DP topology")
            active_ranks = {shard.replica_index for shard in wave.shards}
            for dp_rank in range(profile.replica_count):
                key = _rank_publication_key(phase, wave.wave_index, dp_rank)
                occurrence = caller_root.require_occurrence(key)
                if dp_rank in active_ranks:
                    if key not in rank_publication_envelopes or key not in rank_prompt_anchor_roots:
                        raise RuntimeError("active DP occurrence is missing its publication inventory")
                    envelope = rank_publication_envelopes[key]
                    validate_nemo_rank_prompt_anchor(envelope, caller_root)
                    validate_nemo_rank_caller_ledger(
                        envelope,
                        caller_ledger_admission,
                    )
                    if (
                        envelope.get("generation_request_envelope_sha256")
                        != request_envelope.sha256
                        or envelope.get(
                            "generation_execution_occurrence_sha256"
                        )
                        != request_envelope.execution_occurrence_sha256
                    ):
                        raise RuntimeError(
                            "rank publication differs from the admitted generation envelope"
                        )
                    validate_file_publication_plan(envelope["publication_plan"])
                elif occurrence.active or occurrence.requests:
                    raise RuntimeError("inactive DP occurrence is not exact zero-work inventory")
                elif key in rank_publication_envelopes or key in rank_prompt_anchor_roots:
                    raise RuntimeError("inactive DP occurrence unexpectedly owns publication inventory")

            semantic_by_global_index = {
                request.global_request_index: request
                for key in active_keys
                for request in caller_root.require_occurrence(key).requests
            }
            expected_global_indices = tuple(
                global_request_index_start + wave.start + index
                for index in range(len(wave_manifest.requests))
            )
            if set(semantic_by_global_index) != set(expected_global_indices):
                raise RuntimeError("caller semantic records do not cover the exact physical wave")
            for local_index, request in enumerate(wave_manifest.requests):
                global_index = expected_global_indices[local_index]
                semantic = semantic_by_global_index[global_index]
                payload = caller_root.require_payload(semantic.prompt_token_ids_sha256)
                if (
                    semantic.request_id != request.request_id
                    or payload.prompt_token_ids != tuple(request.prompt_token_ids)
                    or semantic.generation_round != generation_round
                    or semantic.call_index != global_call_index_start + wave.wave_index
                ):
                    raise RuntimeError("caller semantic prompt differs from the phase input")
        wave_contexts[wave.wave_index] = (
            wave_manifest,
            request_envelope,
            generation_input,
        )
    return caller_ledger_admission, wave_contexts


def run_nemo_generation_phase(
    *,
    generation: Any,
    manifest: WorkloadManifest,
    profile: Evo2VllmProfile,
    phase: str,
    sample_index: int,
    generation_round: int,
    global_call_index_start: int,
    global_request_index_start: int,
    caller_ledger_admission: GenerationRequestLedgerAdmission,
    full_output_path: str | Path,
    namespace_output_path: str | Path | None = None,
    memory_monitor_factory: Any,
    ray_get: Any | None = None,
    clock: Any | None = None,
    greedy: bool = False,
    require_full_vocab_logprobs: bool = False,
    expected_finite_logprob_support: int | None = None,
    decode_output_token_ids: Any | None = None,
    rank_publication_envelopes: dict[str, dict[str, Any]] | None = None,
    rank_prompt_anchor_roots: dict[str, NemoCallerPromptAnchorRoot] | None = None,
    request_envelope_namespace: str,
) -> NemoGenerationPhaseResult:
    """Run exact NeMo-RL waves with per-engine graph, route, seed, and ownership proof."""

    def require_namespace_ownership() -> Path | None:
        if namespace_output_path is None:
            return None
        return require_output_namespace_reservation(namespace_output_path)

    require_namespace_ownership()
    if not phase:
        raise ValueError("phase cannot be empty")
    if type(request_envelope_namespace) is not str or not request_envelope_namespace:
        raise ValueError("request_envelope_namespace must be a nonempty built-in string")
    if min(generation_round, global_call_index_start, global_request_index_start) < 0:
        raise ValueError("generation and request coordinates must be nonnegative")
    caller_ledger_admission, wave_contexts = _preflight_nemo_generation_phase(
        manifest=manifest,
        profile=profile,
        phase=phase,
        generation_round=generation_round,
        global_call_index_start=global_call_index_start,
        global_request_index_start=global_request_index_start,
        request_envelope_namespace=request_envelope_namespace,
        caller_ledger_admission=caller_ledger_admission,
        rank_publication_envelopes=rank_publication_envelopes,
        rank_prompt_anchor_roots=rank_prompt_anchor_roots,
    )
    if ray_get is None:
        import ray

        ray_get = ray.get
    if clock is None:
        import time

        clock = time.perf_counter
    operational_clock = __import__("time").perf_counter
    operational_phase_begin = operational_clock()

    waves = build_request_waves(
        request_count=len(manifest.requests),
        global_batch_size=profile.global_wave_size,
        replica_count=profile.replica_count,
    )
    all_records: list[GenerationRecord] = []
    all_executions: list[RequestExecutionRecord] = []
    all_qualified_engine_request_ids: list[tuple[int, str]] = []
    all_ttft: list[float] = []
    all_decode: list[float] = []
    generation_call_s = []
    rank_publication_s = []
    wave_proofs = []
    prefix_cache_reset = False
    if profile.shared_prefix_state_reuse:
        if generation.invalidate_kv_cache() is not True:
            raise AssertionError("shared-prefix NeMo-RL phase requires a successful cache invalidation")
        prefix_cache_reset = True

    monitor_context = (
        memory_monitor_factory() if profile.proof else nullcontext(SimpleNamespace(peak_device_memory_bytes=()))
    )
    with monitor_context as monitor:
        for wave in waves:
            wave_manifest, request_envelope, generation_input = wave_contexts[
                wave.wave_index
            ]
            wave_phase = f"{phase}.wave-{wave.wave_index:03d}"
            expected_dp_ranks = tuple(
                request.target_data_parallel_rank
                for request in request_envelope.requests
            )
            caller_ledger_admission.require_envelope(request_envelope)
            reset_proof = None
            if profile.proof:
                reset_proof = _proof_rpc(
                    generation,
                    "reset_evo2_proof_phase",
                    phase=wave_phase,
                    ray_get=ray_get,
                )
            begin = clock()
            outputs = generation.generate(
                generation_input,
                greedy=greedy,
                request_envelope=request_envelope,
            )
            generation_call_s.append(clock() - begin)
            require_namespace_ownership()
            validate_generation_request_envelope_result(
                outputs,
                request_envelope,
            )
            engine_request_ids = outputs["generation_engine_request_ids"]
            qualified_engine_request_ids = [
                (request.target_data_parallel_rank, engine_request_id)
                for request, engine_request_id in zip(
                    request_envelope.requests,
                    engine_request_ids,
                    strict=True,
                )
            ]
            if set(qualified_engine_request_ids).intersection(
                all_qualified_engine_request_ids
            ):
                raise RuntimeError(
                    "engine request IDs were reused within one DP replica across physical calls"
                )
            all_qualified_engine_request_ids.extend(qualified_engine_request_ids)
            engine_proofs = []
            if profile.proof:
                engine_proofs = _proof_rpc(
                    generation,
                    "snapshot_evo2_proof_phase",
                    phase=wave_phase,
                    ray_get=ray_get,
                )
            records, executions, timings = records_from_nemo_generation_output(wave_manifest, outputs)
            validate_generation_request_metadata(
                outputs,
                request_count=wave.request_count,
                require_explicit_envelope=True,
                expected_base_seed=manifest.seed,
                expected_data_parallel_size=profile.replica_count,
            )
            full_vocab_logprobs = (
                full_vocab_logprob_evidence_from_nemo_output(
                    outputs,
                    records=records,
                    require_full=True,
                    expected_finite_support=expected_finite_logprob_support,
                )
                if require_full_vocab_logprobs
                else None
            )

            _validate_wave_execution(
                wave_manifest,
                executions,
                expected_envelope=request_envelope,
            )

            if profile.proof and len(engine_proofs) != profile.replica_count:
                raise AssertionError(
                    f"wave {wave.wave_index} returned {len(engine_proofs)} engine proofs for "
                    f"{profile.replica_count} physical DP replicas"
                )
            validated_engine_proofs = []
            inactive_engine_proofs = []
            if profile.proof:
                active_engine_proofs = engine_proofs[: len(wave.shards)]
                for shard, engine_proof in zip(wave.shards, active_engine_proofs, strict=True):
                    if engine_proof.get("phase") != wave_phase:
                        raise AssertionError("engine proof phase does not match its request wave")
                    observations = list(engine_proof.get("cudagraph_observations", ()))
                    proof_summary = full_decode_proof_summary(
                        observations,
                        phase=wave_phase,
                        batch_size=shard.request_count,
                        max_new_tokens=manifest.max_new_tokens,
                    )
                    scheduler_proof = scheduler_capacity_proof_summary(
                        list(engine_proof.get("scheduler_observations", ())),
                        phase=wave_phase,
                        global_wave_size=wave.request_count,
                        engine_request_count=shard.request_count,
                        max_num_seqs=profile.resolved_max_num_seqs,
                    )
                    validate_scheduler_capacity_proof(scheduler_proof)
                    validate_full_decode_proof(
                        observations,
                        phase=wave_phase,
                        batch_size=shard.request_count,
                        max_new_tokens=manifest.max_new_tokens,
                    )
                    validated_engine_proofs.append(
                        {
                            "dp_rank": shard.replica_index,
                            "request_count": shard.request_count,
                            "full_decode_proof": proof_summary,
                            "scheduler_capacity_proof": scheduler_proof,
                            **engine_proof,
                        }
                    )
                for dp_rank, engine_proof in enumerate(
                    engine_proofs[len(wave.shards) :],
                    start=len(wave.shards),
                ):
                    if engine_proof.get("phase") != wave_phase:
                        raise AssertionError("inactive engine proof phase does not match its request wave")
                    if engine_proof.get("cudagraph_observations") or engine_proof.get("scheduler_observations"):
                        raise AssertionError("inactive DP replica executed scheduler or CUDA-graph work")
                    inactive_engine_proofs.append(
                        {
                            "dp_rank": dp_rank,
                            "request_count": 0,
                            "inactive": True,
                            **engine_proof,
                        }
                    )
            shared_prefix_reuse = None
            if profile.proof and profile.shared_prefix_state_reuse:
                shared_prefix_reuse = shared_prefix_state_reuse_evidence(
                    wave_manifest,
                    cached_tokens=cached_tokens_from_nemo_generation_output(
                        outputs,
                        request_count=wave.request_count,
                    ),
                    request_replica_ranks=expected_dp_ranks,
                    worker_proof=tuple(
                        worker for engine in validated_engine_proofs for worker in _inner_worker_proofs(engine)
                    ),
                    expected_worker_clone_counts=tuple(
                        shard.request_count - int(wave.wave_index == 0) for shard in wave.shards
                    ),
                    cache_block_size=int(validated_engine_proofs[0]["resolved_config"]["cache"]["block_size"]),
                    expected_cache_misses=len(wave.shards) if wave.wave_index == 0 else 0,
                )
                shared_prefix_reuse = {
                    **shared_prefix_reuse,
                    "phase_prefix_cache_reset_before_first_wave": prefix_cache_reset,
                }

            rank_publication_outcomes = []
            if rank_publication_envelopes is not None:
                publication_begin = operational_clock()
                rank_publication_outcomes = list(
                    publish_nemo_rank_wave(
                        generation,
                        wave=wave,
                        phase=phase,
                        envelopes=rank_publication_envelopes,
                        caller_prompt_anchor_roots=rank_prompt_anchor_roots,
                        caller_ledger_admission=caller_ledger_admission,
                        ray_get=ray_get,
                    )
                )
                rank_publication_s.append(operational_clock() - publication_begin)

            wave_proofs.append(
                {
                    "wave_index": wave.wave_index,
                    "phase": wave_phase,
                    "start": wave.start,
                    "stop": wave.stop,
                    "request_count": wave.request_count,
                    "generation_round": generation_round,
                    "call_index": global_call_index_start + wave.wave_index,
                    "batch_call_slot": request_envelope.batch_call_slot,
                    "turn_index": request_envelope.turn_index,
                    "caller_ledger_sha256": caller_ledger_admission.sha256,
                    "request_envelope_sha256": request_envelope.sha256,
                    "execution_occurrence_sha256": (
                        request_envelope.execution_occurrence_sha256
                    ),
                    "engine_request_occurrences": [
                        {
                            "dp_rank": dp_rank,
                            "engine_request_id": engine_request_id,
                            "execution_occurrence_sha256": (
                                request_envelope.execution_occurrence_sha256
                            ),
                        }
                        for dp_rank, engine_request_id in qualified_engine_request_ids
                    ],
                    "generation_s": generation_call_s[-1],
                    "reset_proof": reset_proof,
                    "engines": validated_engine_proofs,
                    "inactive_engines": inactive_engine_proofs,
                    "full_vocab_logprobs": full_vocab_logprobs,
                    "shared_prefix_state_reuse": shared_prefix_reuse,
                    "rank_publication_outcomes": rank_publication_outcomes,
                }
            )
            all_records.extend(records)
            all_executions.extend(executions)
            all_ttft.extend(timings["ttft_s"])
            all_decode.extend(timings["decode_s"])

    if tuple(record.request_id for record in all_records) != tuple(
        request.request_id for request in manifest.requests
    ):
        raise AssertionError("production waves did not return every manifest request exactly once")
    global_indices = [record.global_request_index for record in all_executions]
    seeds = [record.seed for record in all_executions]
    if len(global_indices) != len(set(global_indices)) or len(seeds) != len(set(seeds)):
        raise AssertionError("production phase has duplicate request ownership or RNG streams")
    if len(all_qualified_engine_request_ids) != len(
        set(all_qualified_engine_request_ids)
    ):
        raise AssertionError("production phase reused an engine request ID within a DP replica")

    output_lengths = tuple(len(record.output_token_ids) for record in all_records)
    sample = BenchmarkSample(
        sample_index=sample_index,
        generation_s=sum(generation_call_s),
        request_count=len(all_records),
        prompt_tokens=sum(len(record.prompt_token_ids) for record in all_records),
        generated_tokens=sum(output_lengths),
        ttft_s=tuple(all_ttft),
        inter_token_latency_s=(
            tuple(
                decode_s / (output_length - 1)
                for decode_s, output_length in zip(all_decode, output_lengths, strict=True)
            )
            if manifest.max_new_tokens > 1
            else ()
        ),
        output_lengths=output_lengths,
        peak_device_memory_bytes=monitor.peak_device_memory_bytes,
    )
    require_namespace_ownership()
    aggregate_sidecar_begin = operational_clock()
    full_output_artifact = write_full_generation_records_artifact(
        full_output_path,
        records=all_records,
        execution_records=all_executions,
        decode_output_token_ids=decode_output_token_ids,
        ownership_validator=require_namespace_ownership,
    )
    coordinator_aggregate_sidecar_s = operational_clock() - aggregate_sidecar_begin
    operational_phase_s = operational_clock() - operational_phase_begin
    return NemoGenerationPhaseResult(
        phase=phase,
        caller_ledger_admission=caller_ledger_admission,
        sample=sample,
        generation_call_s=tuple(generation_call_s),
        rank_publication_s=tuple(rank_publication_s),
        coordinator_aggregate_sidecar_s=coordinator_aggregate_sidecar_s,
        operational_phase_s=operational_phase_s,
        output_summaries=tuple(record.summary_dict() for record in all_records),
        request_executions=tuple(all_executions),
        full_output_artifact=full_output_artifact,
        wave_proofs=tuple(wave_proofs),
        proof_collected=profile.proof,
        prefix_cache_reset=prefix_cache_reset,
    )


def _inner_worker_proofs(engine_proof: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    workers = engine_proof.get("worker_proof")
    if not isinstance(workers, list) or not workers:
        raise AssertionError("production engine proof is missing internal vLLM worker evidence")
    return tuple(workers)


def run_nemo_dp2_benchmark(args: Any, manifest: WorkloadManifest) -> dict[str, Any]:
    """Launch the actual NeMo-RL TP1/DP2 path and retain exact production evidence."""
    if args.topology != "dp2":
        raise ValueError("run_nemo_dp2_benchmark requires topology=dp2")
    clock = __import__("time").perf_counter
    output_path = Path(args.output).resolve()
    require_output_namespace_reservation(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile_from_args(args, manifest)
    caller_coordinates = CallerCoordinateContract.from_inputs(
        manifest,
        profile,
        args.generation_round,
    )
    benchmark_mode = benchmark_mode_from_args(args)
    instrumentation = benchmark_instrumentation_contract(benchmark_mode)
    preflight_begin = clock()
    preflight = context_length_preflight(
        profile,
        model=args.checkpoint,
        workload_max_total_tokens=manifest.max_total_tokens,
        load_format=args.load_format,
    )
    preflight_s = clock() - preflight_begin

    gpu_preflight_begin = clock()
    gpu_identity = gpu_hardware_provenance()
    gpu_preflight_s = clock() - gpu_preflight_begin

    import_begin = __import__("time").perf_counter()
    import nemo_rl
    import ray
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration

    import_s = __import__("time").perf_counter() - import_begin
    provenance_begin = clock()
    checkpoint_identity = checkpoint_provenance(args.checkpoint)
    bionemo_source_identity = source_provenance(require_clean=True)
    nemo_package = Path(nemo_rl.__file__).resolve().parent
    nemo_source_identity = source_provenance(
        repository=nemo_package.parent,
        source_roots=(
            nemo_package / "models/generation/interfaces.py",
            nemo_package / "models/generation/vllm/config.py",
            nemo_package / "models/generation/vllm/vllm_generation.py",
            nemo_package / "models/generation/vllm/vllm_worker.py",
        ),
        require_clean=True,
    )
    vllm_identity = vllm_installation_provenance()
    from bionemo.evo2.vllm.sampler import sampler_installation_provenance

    sampler_identity = sampler_installation_provenance(require_loaded_modules=False)
    runtime_attestation = runtime_attestation_contract(
        checkpoint=checkpoint_identity,
        sources={
            "bionemo": bionemo_source_identity,
            "nemo_rl": nemo_source_identity,
        },
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
    provenance_s = clock() - provenance_begin

    phase_specs = (
        ("cold-generation", 0),
        *((f"warmup-{index}", index + 1) for index in range(args.warmups)),
        *((f"steady-{index}", args.warmups + index + 1) for index in range(args.repetitions)),
    )
    calls_per_generation_round = len(
        build_request_waves(
            request_count=len(manifest.requests),
            global_batch_size=profile.global_wave_size,
            replica_count=profile.replica_count,
        )
    )
    global_call_index_start = args.generation_round * calls_per_generation_round
    serial_identity_context = common_prefix_identity_context(args, manifest, profile)
    phase_ledgers = build_nemo_generation_caller_ledgers(
        manifest=manifest,
        profile=profile,
        phases=tuple(phase for phase, _sample_index in phase_specs),
        generation_round=args.generation_round,
        global_request_index_start=0,
        request_envelope_namespace=benchmark_contract_digest,
    )
    serial_phase_ledger = (
        None
        if serial_identity_context is None
        else build_nemo_generation_caller_ledgers(
            manifest=manifest.request_slice(0, 1),
            profile=profile,
            phases=("common-prefix-serial-reference",),
            generation_round=args.generation_round,
            global_request_index_start=0,
            request_envelope_namespace=benchmark_contract_digest,
        )["common-prefix-serial-reference"]
    )

    config = build_nemo_generation_config(
        profile,
        manifest,
        checkpoint=args.checkpoint,
        load_format=args.load_format,
        require_rank_publication=True,
    )
    decoder_begin = clock()
    output_decoder = manifest_output_decoder(manifest)
    output_decoder_setup_s = clock() - decoder_begin
    memory_reader = make_nvml_memory_reader()

    ray_dir_suffix = __import__("hashlib").sha256(str(output_path).encode()).hexdigest()[:10]
    ray_log_dir = Path("/tmp") / f"e2ray-{ray_dir_suffix}"
    ray_log_dir.mkdir(parents=True, exist_ok=True)

    rank_reservation_begin = clock()
    rank_reservations: dict[str, FilePublicationReservation] = {}
    rank_plans: dict[str, FilePublicationPlan] = {}
    rank_envelopes: dict[str, dict[str, Any]] = {}
    rank_prompt_anchor_roots: dict[str, NemoCallerPromptAnchorRoot] = {}
    try:
        rank_reservations, rank_plans, rank_envelopes, rank_prompt_root = (
            reserve_nemo_rank_publications(
                output_path,
                manifest=manifest,
                profile=profile,
                phases=tuple(phase for phase, _sample_index in phase_specs),
                generation_round=args.generation_round,
                global_call_index_start=global_call_index_start,
                external_contract_sha256=benchmark_contract_digest,
                caller_ledgers_by_phase=phase_ledgers,
            )
        )
        rank_prompt_anchor_roots.update(rank_prompt_root.index_by_publication_key())
        if serial_identity_context is not None:
            serial_reservations, serial_plans, serial_envelopes, serial_prompt_root = (
                reserve_nemo_rank_publications(
                    output_path,
                    manifest=manifest.request_slice(0, 1),
                    profile=profile,
                    phases=("common-prefix-serial-reference",),
                    generation_round=args.generation_round,
                    global_call_index_start=args.generation_round,
                    external_contract_sha256=benchmark_contract_digest,
                    caller_ledgers_by_phase={
                        "common-prefix-serial-reference": serial_phase_ledger
                    },
                )
            )
            if set(rank_reservations) & set(serial_reservations):
                raise RuntimeError("serial and batched rank publication plans overlap")
            rank_reservations.update(serial_reservations)
            rank_plans.update(serial_plans)
            rank_envelopes.update(serial_envelopes)
            rank_prompt_anchor_roots.update(
                serial_prompt_root.index_by_publication_key()
            )
    except BaseException as reservation_error:
        terminalize_failed_nemo_rank_publications(rank_reservations, primary_error=reservation_error)
        raise
    rank_publication_reservation_s = clock() - rank_reservation_begin

    cluster = None
    generation = None
    primary_error: BaseException | None = None
    try:
        ray_begin = clock()
        init_ray(log_dir=str(ray_log_dir))
        ray_init_s = clock() - ray_begin
        cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[2],
            use_gpus=True,
            max_colocated_worker_groups=1,
            num_gpus_per_node=2,
            name=f"evo2-vllm-dp2-{output_path.stem}",
        )
        engine_begin = clock()
        with PeakMemoryMonitor(memory_reader) as init_memory:
            generation = VllmGeneration(
                cluster,
                config,
                name_prefix=f"evo2_vllm_dp2_{output_path.stem}",
            )
        engine_init_s = clock() - engine_begin
        resolved_begin = clock()
        resolved_configs = snapshot_and_validate_nemo_resolved_configs(
            generation,
            profile=profile,
            ray_get=ray.get,
        )
        resolved_config_snapshot_s = clock() - resolved_begin

        initialized_phase = "engine-initialized"
        initialized_reset = None
        initialized_proofs = []
        if profile.proof:
            initialized_reset = _proof_rpc(
                generation,
                "reset_evo2_proof_phase",
                phase=initialized_phase,
                ray_get=ray.get,
            )
            initialized_proofs = _proof_rpc(
                generation,
                "snapshot_evo2_proof_phase",
                phase=initialized_phase,
                ray_get=ray.get,
            )
            if len(initialized_proofs) != profile.replica_count:
                raise AssertionError("NeMo-RL did not launch exactly two independent DP engines")
            for proof in initialized_proofs:
                validate_resolved_profile(profile, proof["resolved_config"])

        phase_results = []
        serial_reference_result = None
        if serial_identity_context is not None:
            serial_reference_result = run_nemo_generation_phase(
                generation=generation,
                manifest=manifest.request_slice(0, 1),
                profile=profile,
                phase="common-prefix-serial-reference",
                sample_index=-1,
                generation_round=args.generation_round,
                global_call_index_start=args.generation_round,
                global_request_index_start=0,
                caller_ledger_admission=serial_phase_ledger,
                full_output_path=phase_output_artifact_path(
                    output_path,
                    phase="common-prefix-serial-reference",
                ),
                namespace_output_path=output_path,
                memory_monitor_factory=lambda: PeakMemoryMonitor(memory_reader),
                ray_get=ray.get,
                clock=clock,
                decode_output_token_ids=output_decoder,
                rank_publication_envelopes=rank_envelopes,
                rank_prompt_anchor_roots=rank_prompt_anchor_roots,
                request_envelope_namespace=benchmark_contract_digest,
            )
        for sample_index, (phase, _) in enumerate(phase_specs):
            phase_results.append(
                run_nemo_generation_phase(
                    generation=generation,
                    manifest=manifest,
                    profile=profile,
                    phase=phase,
                    sample_index=sample_index,
                    generation_round=args.generation_round,
                    global_call_index_start=global_call_index_start,
                    global_request_index_start=0,
                    caller_ledger_admission=phase_ledgers[phase],
                    full_output_path=phase_output_artifact_path(output_path, phase=phase),
                    namespace_output_path=output_path,
                    memory_monitor_factory=lambda: PeakMemoryMonitor(memory_reader),
                    ray_get=ray.get,
                    clock=clock,
                    decode_output_token_ids=output_decoder,
                    rank_publication_envelopes=rank_envelopes,
                    rank_prompt_anchor_roots=rank_prompt_anchor_roots,
                    request_envelope_namespace=benchmark_contract_digest,
                )
            )

        if profile.proof:
            final_engine_proofs = phase_results[-1].wave_proofs[-1]["engines"]
            for initialized_engine, final_engine in zip(
                initialized_proofs,
                final_engine_proofs,
                strict=True,
            ):
                initialized_workers = _inner_worker_proofs(initialized_engine)
                final_workers = _inner_worker_proofs(final_engine)
                if len(initialized_workers) != len(final_workers):
                    raise AssertionError("internal vLLM worker topology changed during generation")
                for initialized_worker, final_worker in zip(
                    initialized_workers,
                    final_workers,
                    strict=True,
                ):
                    validate_compilation_proof(
                        initialized_worker["compilation"],
                        final_worker["compilation"],
                    )
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
        else:
            if linked_proof is None or not isinstance(linked_proof.get("gpu_memory_headroom"), dict):
                raise RuntimeError("speed lane is missing linked proof GPU memory headroom evidence")
            memory_headroom = linked_proof["gpu_memory_headroom"]

        steady_results = [result for result in phase_results if result.phase.startswith("steady-")]
        phase_artifacts, exact_progress = attach_exact_generation_progress_evidence(
            [result.to_dict() for result in phase_results],
            manifest=manifest,
            enabled=bool(args.exact_progress_gate),
            proof_collected=profile.proof,
            topology=profile.topology,
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
        phase_artifacts, common_prefix_identity = common_prefix_identity_phase_artifacts(
            args=args,
            manifest=manifest,
            profile=profile,
            serial_reference_phase=(None if serial_reference_result is None else serial_reference_result.to_dict()),
            phase_artifacts=phase_artifacts,
            decode_output_token_ids=output_decoder,
            collect_physical_proof=profile.proof,
        )
        rank_outcomes = {}
        publication_phase_results = (
            *((serial_reference_result,) if serial_reference_result is not None else ()),
            *phase_results,
        )
        for result in publication_phase_results:
            for wave_proof in result.wave_proofs:
                for outcome in wave_proof["rank_publication_outcomes"]:
                    key = outcome["publication_key"]
                    if key in rank_outcomes:
                        raise RuntimeError(f"duplicate rank publication outcome: {key}")
                    rank_outcomes[key] = outcome
        rank_terminalization_begin = clock()
        rank_publication_proof = finalize_nemo_rank_publications(
            reservations=rank_reservations,
            envelopes=rank_envelopes,
            caller_prompt_anchor_roots=rank_prompt_anchor_roots,
            caller_ledgers_by_phase={
                **phase_ledgers,
                **(
                    {}
                    if serial_phase_ledger is None
                    else {
                        "common-prefix-serial-reference": serial_phase_ledger
                    }
                ),
            },
            outcomes=rank_outcomes,
            namespace_output_path=output_path,
        )
        rank_publication_terminalization_s = clock() - rank_terminalization_begin
        pure_generation_s = sum(result.sample.generation_s for result in publication_phase_results)
        rank_publication_worker_s = sum(sum(result.rank_publication_s) for result in publication_phase_results)
        coordinator_aggregate_sidecar_s = sum(
            result.coordinator_aggregate_sidecar_s for result in publication_phase_results
        )
        operational_generation_phases_s = sum(result.operational_phase_s for result in publication_phase_results)
        return {
            "schema_version": 1,
            "backend": "nemo-rl-vllm",
            "topology": "dp2",
            "benchmark_mode": benchmark_mode,
            "benchmark_contract": benchmark_contract,
            "benchmark_contract_sha256": benchmark_contract_digest,
            "instrumentation": instrumentation,
            "linked_proof_artifact": linked_proof,
            "proof_status": (
                {
                    "passed": True,
                    "phase_count": len(phase_results),
                    "full_decode_passed": True,
                    "compilation_stable": True,
                }
                if profile.proof
                else {
                    "passed": None,
                    "attested_by_linked_proof": linked_proof,
                }
            ),
            "versions": runtime_versions(),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_provenance": checkpoint_identity,
            "source_provenance": {
                "bionemo": bionemo_source_identity,
                "nemo_rl": nemo_source_identity,
            },
            "vllm_installation_provenance": vllm_identity,
            "sampler_installation_provenance": sampler_identity,
            "gpu_hardware_provenance": gpu_identity,
            "gpu_memory_headroom": memory_headroom,
            "provenance_assurance": {
                "tier": rank_publication_proof["provenance_assurance_tier"],
                "trusted_node_reproducibility": rank_publication_proof[
                    "trusted_node_reproducibility_admission"
                ],
                "full_runtime_adversarial_admission": rank_publication_proof[
                    "full_runtime_adversarial_admission"
                ],
                "full_runtime_adversarial_admission_reason": rank_publication_proof[
                    "full_runtime_adversarial_admission_reason"
                ],
            },
            "rank_publication_proof": rank_publication_proof,
            "rank_publication_plans": {key: plan.to_dict() for key, plan in sorted(rank_plans.items())},
            "rank_publication_envelopes": {
                key: envelope for key, envelope in sorted(rank_envelopes.items())
            },
            "canonical_identity": canonical_identity,
            "common_prefix_identity": common_prefix_identity,
            "common_prefix_serial_reference": (
                None if serial_reference_result is None else serial_reference_result.to_dict()
            ),
            "exact_generation_progress": exact_progress,
            "invocation": {
                "argv": [__import__("sys").executable, *__import__("sys").argv],
                "parsed_args": {
                    name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
                },
                "output_artifact_path": str(output_path),
                "ray_log_dir": str(ray_log_dir),
                "exit_status": 0,
            },
            "manifest": manifest.to_dict(),
            "manifest_sha256": manifest.sha256,
            "profile": asdict(profile),
            "context_length_preflight": preflight,
            "nemo_generation_config": config,
            "resolved_configs": resolved_configs,
            "execution_contract": {
                "production_path": "nemo_rl.models.generation.vllm.VllmGeneration",
                "replicas": 2,
                "tensor_parallel_size_per_replica": 1,
                "global_batch_size": profile.global_batch_size,
                "global_wave_size": profile.global_wave_size,
                "per_engine_batch_size": profile.per_engine_batch_size,
                "per_engine_max_num_seqs": profile.resolved_max_num_seqs,
                "gdpo_target_request_count": profile.gdpo_target_batch_size,
                "planned_waves_to_96": profile.gdpo_waves_to_96,
                "semantic_padding": False,
                "prefix_caching": profile.shared_prefix_state_reuse,
                "mamba_cache_mode": "align" if profile.shared_prefix_state_reuse else "none",
                "shared_prefix_state_reuse": profile.shared_prefix_state_reuse,
                "benchmark_mode": benchmark_mode,
                "timed_generation_instrumentation": instrumentation,
                "rank_local_publication_required": True,
                "rank_publication_outside_pure_generation_timer": True,
            },
            "timing": {
                "context_length_preflight_s": preflight_s,
                "gpu_hardware_preflight_s": gpu_preflight_s,
                "imports_s": import_s,
                "provenance_hashing_s": provenance_s,
                "ray_init_s": ray_init_s,
                "engine_init_s": engine_init_s,
                "resolved_config_snapshot_s": resolved_config_snapshot_s,
                "output_decoder_setup_s": output_decoder_setup_s,
                "rank_publication_reservation_s": rank_publication_reservation_s,
                "pure_model_generation_s": pure_generation_s,
                "rank_publication_worker_and_validation_s": rank_publication_worker_s,
                "coordinator_aggregate_sidecar_s": coordinator_aggregate_sidecar_s,
                "rank_publication_terminalization_s": rank_publication_terminalization_s,
                "operational_generation_phases_s": operational_generation_phases_s,
                "engine_init_peak_device_memory_bytes": list(init_memory.peak_device_memory_bytes),
            },
            "initialized_reset": initialized_reset,
            "initialized_engine_proofs": initialized_proofs,
            "phases": phase_artifacts,
            "steady_aggregate": aggregate_samples([result.sample for result in steady_results]),
        }
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_failures: list[tuple[str, BaseException]] = []
        for label, cleanup in (
            ("generation shutdown", None if generation is None else generation.shutdown),
            ("cluster shutdown", None if cluster is None else cluster.shutdown),
            ("Ray shutdown", ray.shutdown),
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except BaseException as cleanup_error:
                cleanup_failures.append((label, cleanup_error))
        try:
            terminalize_failed_nemo_rank_publications(
                rank_reservations,
                primary_error=primary_error,
            )
        except BaseException as cleanup_error:
            cleanup_failures.append(("rank publication terminalization", cleanup_error))
        if cleanup_failures:
            if primary_error is not None:
                for label, cleanup_error in cleanup_failures:
                    primary_error.add_note(f"{label} failed: {cleanup_error!r}")
            else:
                cleanup_error = RuntimeError("benchmark cleanup failed")
                for label, failure in cleanup_failures:
                    cleanup_error.add_note(f"{label} failed: {failure!r}")
                raise cleanup_error from cleanup_failures[0][1]


__all__ = [
    "NemoGenerationPhaseResult",
    "build_nemo_generation_config",
    "build_nemo_generation_input",
    "full_vocab_logprob_evidence_from_nemo_output",
    "records_from_nemo_generation_output",
    "rank_publication_contract_sha256",
    "rank_publication_envelope_payload",
    "reserve_nemo_rank_publications",
    "run_nemo_dp2_benchmark",
    "run_nemo_generation_phase",
    "snapshot_and_validate_nemo_resolved_configs",
    "terminalize_failed_nemo_rank_publications",
]
