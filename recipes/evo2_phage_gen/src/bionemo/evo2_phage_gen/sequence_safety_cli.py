# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line interface for first-order phage sequence filters."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

import yaml

from bionemo.evo2_phage_gen import sequence_safety_adapters as safety_adapters
from bionemo.evo2_phage_gen.design_scope import (
    DesignObjective,
    HostDomain,
    HostEvidence,
    ObjectiveDirection,
    ObjectiveEndpoint,
    ObjectiveKind,
    ScopeDecision,
    evaluate_host_evidence,
    validate_design_scope,
)
from bionemo.evo2_phage_gen.external_assets import _validate_staged_safety_manifest
from bionemo.evo2_phage_gen.sequence_safety import (
    GenomeSafetyResult,
    SafetyClassResult,
    SafetyState,
    aggregate_safety_state,
    load_phage_safety_policy,
)
from bionemo.evo2_phage_gen.sequence_safety_adapters import (
    PHROGS_HOMOLOGY_POLICY_V1,
    TOXIN_HOMOLOGY_POLICY_V2,
    AdapterResult,
    AssetProvenanceError,
    GenomeInput,
    NormalizedSafetyFinding,
    ORFArtifacts,
    ORFQueryRecord,
    ToolPin,
    build_amrfinder_command,
    build_diamond_command,
    build_phrogs_command,
    prepare_orf_artifacts_checked,
    run_amrfinder,
    run_phrogs,
    run_toxin_diamond,
    validate_tool_pin,
)
from bionemo.evo2_phage_gen.sequence_safety_batch import (
    AMRFINDER_SPLIT_POLICY_ID,
    AMRFINDER_SPLIT_POLICY_SHA256,
    DIAMOND_SPLIT_POLICY_ID,
    DIAMOND_SPLIT_POLICY_SHA256,
    BatchAdapterExecution,
    BatchedORFInputs,
    run_amrfinder_batch,
    run_toxin_diamond_batch,
    split_amrfinder_batch_output,
    split_diamond_batch_output,
)


CLI_ID = "evo2-phage-sequence-safety"
CLI_VERSION = "1"
MANIFEST_SCHEMA_VERSION = 1
_VIRULENCE_REASON = "AMRFINDER_PLUS_VIRULENCE_SUPPLEMENTAL"
_AMRFINDER_POLICY_DESCRIPTOR = {
    "policy_id": "amrfinder-curated-thresholds-v4.2.7",
    "amrfinder_release": "amrfinder_v4.2.7",
    "curated_identity_threshold_overrides": False,
    "amr_type_action": "FAIL",
    "plus_virulence_action": "SUPPLEMENTAL_TOXIN_EVIDENCE",
    "nucleotide_only_protein_id": "NA",
    "nucleotide_only_evidence_path": safety_adapters._AMRFINDER_NUCLEOTIDE_EVIDENCE_PATH,
    "nucleotide_only_methods": sorted(safety_adapters._AMRFINDER_NUCLEOTIDE_METHODS),
    "nucleotide_only_binding": "exact_contig_sequence_sha256_coordinates_strand",
}


def _canonical_mapping_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


_ADAPTER_POLICY_DESCRIPTORS = {
    "amr": _AMRFINDER_POLICY_DESCRIPTOR,
    "toxin": TOXIN_HOMOLOGY_POLICY_V2.to_dict(),
    "lysogeny": PHROGS_HOMOLOGY_POLICY_V1.to_dict(),
}
_AMRFINDER_POLICY = (
    _AMRFINDER_POLICY_DESCRIPTOR["policy_id"],
    _canonical_mapping_sha256(_AMRFINDER_POLICY_DESCRIPTOR),
)
_ADAPTER_POLICIES = {
    "amr": _AMRFINDER_POLICY,
    "toxin": (TOXIN_HOMOLOGY_POLICY_V2.policy_id, TOXIN_HOMOLOGY_POLICY_V2.sha256),
    "lysogeny": (PHROGS_HOMOLOGY_POLICY_V1.policy_id, PHROGS_HOMOLOGY_POLICY_V1.sha256),
}
_CLAIM_BOUNDARY = {
    "label": "EMA-draft-aligned first-order sequence filters",
    "pass_meaning": (
        "PASS means no qualifying hit under the pinned first-order sequence filters; it is not proof of strict "
        "lysis, absence of all harmful functions, product safety, regulatory compliance, or EMA acceptance."
    ),
    "required_follow_up": [
        "unknown, remote, divergent, and nonhomologous ORFs",
        "complete annotation, genome mapping, taxonomy, and comparative genomics",
        "construct identity, integrity, and genetic stability",
        "experimental host range and lytic phenotype",
        "transducing capacity",
        "production-strain toxins, virulence factors, and process impurities",
    ],
}
_ORF_FALLBACK_DESCRIPTOR = {
    "policy_id": "six-frame-fallback-v1",
    "minimum_amino_acids": 8,
    "frames": [-3, -2, -1, 1, 2, 3],
    "circular_origin_search": True,
    "deduplicate_primary_calls": True,
}
_ORF_FALLBACK_SHA256 = _canonical_mapping_sha256(_ORF_FALLBACK_DESCRIPTOR)
_ORF_PROVENANCE_FILENAME = "orf_provenance.json"


class CLIValidationError(ValueError):
    """An input cannot be interpreted at the CLI trust boundary."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIValidationError(message)


@dataclass(frozen=True)
class LoadedToolPin:
    """One authenticated tool pin plus the identity of its reviewed pin file."""

    tool: str
    pin: ToolPin
    pin_file_path: Path
    pin_file_sha256: str


@dataclass(frozen=True)
class FastaRecord:
    """One FASTA record with separate byte-exact and normalized scan representations."""

    sequence_id: str
    original_bytes: bytes
    normalized_sequence: str


@dataclass(frozen=True)
class ScannedRecord:
    """One input record and its independently aggregated adapter result."""

    input_index: int
    sequence_id: str
    result: GenomeSafetyResult
    adapters: Mapping[str, AdapterResult]


@dataclass(frozen=True)
class BatchSafetyResult:
    """Ordered per-record outcomes and the strict combined batch state."""

    state: SafetyState
    records: tuple[ScannedRecord, ...]


@dataclass(frozen=True)
class BatchedScanExecution:
    """Ordered record results plus the real commands shared across each batch."""

    batch: BatchSafetyResult
    shared_executions: tuple[BatchAdapterExecution, ...]


@dataclass(frozen=True)
class LoadedSafetyAssetManifest:
    """A structurally strict and path-verified Task 2 safety asset manifest."""

    manifest: Mapping[str, object]
    manifest_path: Path
    manifest_sha256: str
    recipe_path: Path
    recipe_sha256: str


@dataclass(frozen=True)
class ValidatedORFProvenance:
    """Reconstructed Task 3 artifacts and their authenticated query inventory."""

    artifacts: ORFArtifacts | None
    query_index: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class ReplayedORFEvidence:
    """Exact Task 3 replay artifacts and ordered query records."""

    artifact_bytes: Mapping[str, bytes]
    query_records: tuple[ORFQueryRecord, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CLIValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict_json(path: Path, *, label: str) -> object:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CLIValidationError(f"{label} must be a non-symlink regular file")
    try:
        return _load_strict_json_text(path.read_text(), label=label)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CLIValidationError(f"cannot read {label}: {error}") from error


def _load_strict_json_text(value: str, *, label: str) -> object:
    """Decode JSON without duplicate-key shadowing or non-standard finite values."""
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CLIValidationError(f"{label} contains non-finite value {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise CLIValidationError(f"{label} is not valid JSON: {error}") from error


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise CLIValidationError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_strict_yaml_bytes(payload: bytes, *, label: str) -> object:
    try:
        text = payload.decode("utf-8")
        for token in yaml.scan(text):
            if isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken)):
                raise CLIValidationError(f"{label} must not contain YAML aliases or anchors")
        return yaml.load(text, Loader=_UniqueKeySafeLoader)
    except CLIValidationError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CLIValidationError(f"cannot read {label}: {error}") from error


def _load_strict_yaml(path: Path, *, label: str) -> object:
    return _load_strict_yaml_bytes(_read_regular_file_bytes(Path(path), label=label), label=label)


def load_safety_asset_manifest(
    path: Path,
    *,
    validator=_validate_staged_safety_manifest,
) -> LoadedSafetyAssetManifest:
    """Wrap Task 2's strongest validator with a strict public file boundary."""
    manifest_path = Path(path).absolute()
    manifest_bytes = _read_regular_file_bytes(manifest_path, label="safety asset manifest")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    payload = _load_strict_yaml_bytes(manifest_bytes, label="safety asset manifest")
    manifest = _strict_payload(
        payload,
        name="safety asset manifest top-level",
        keys=frozenset({"schema_version", "recipe", "amrfinder_plus", "toxin_reference", "phrogs_v4"}),
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 2:
        raise CLIValidationError("safety asset manifest has an unsupported schema version")
    recipe = _strict_payload(
        manifest["recipe"],
        name="safety asset manifest recipe",
        keys=frozenset({"path", "sha256"}),
    )
    recipe_path_value = recipe["path"]
    recipe_sha256 = recipe["sha256"]
    if not isinstance(recipe_path_value, str) or not Path(recipe_path_value).is_absolute():
        raise CLIValidationError("safety asset recipe path must be absolute")
    if not isinstance(recipe_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", recipe_sha256):
        raise CLIValidationError("safety asset recipe SHA-256 is invalid")
    recipe_path = Path(recipe_path_value)
    if recipe_path.is_symlink() or not recipe_path.is_file():
        raise CLIValidationError("safety asset recipe must be a non-symlink regular file")
    try:
        observed_recipe_sha256 = _sha256_file(recipe_path)
    except OSError as error:
        raise CLIValidationError(f"cannot hash safety asset recipe: {error}") from error
    if observed_recipe_sha256 != recipe_sha256:
        raise CLIValidationError("safety asset recipe digest drift")
    try:
        mutable_manifest = dict(manifest)
        validator(mutable_manifest, verify_asset_paths=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CLIValidationError(f"safety asset manifest validation failed: {error}") from error
    if _sha256_regular_file(manifest_path, label="safety asset manifest") != manifest_sha256:
        raise CLIValidationError("safety asset manifest changed during validation")
    return LoadedSafetyAssetManifest(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        recipe_path=recipe_path,
        recipe_sha256=recipe_sha256,
    )


def load_tool_pin_file(
    path: Path,
    *,
    expected_tool: str,
    runner=subprocess.run,
    timeout: float = 300.0,
) -> LoadedToolPin:
    """Load and execute a strict operator-reviewed tool identity assertion."""
    pin_path = Path(path).absolute()
    payload = _strict_payload(
        _load_strict_json(pin_path, label=f"{expected_tool} tool pin"),
        name=f"{expected_tool} tool pin",
        keys=frozenset({"schema_version", "tool", "path", "sha256", "version", "version_args"}),
    )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise CLIValidationError(f"{expected_tool} tool pin has an unsupported schema version")
    if payload["tool"] != expected_tool:
        raise CLIValidationError(f"expected a {expected_tool} tool pin")
    executable = payload["path"]
    digest = payload["sha256"]
    version = payload["version"]
    version_args = payload["version_args"]
    if not isinstance(executable, str) or not Path(executable).is_absolute():
        raise CLIValidationError(f"{expected_tool} tool path must be absolute")
    if not isinstance(digest, str) or not isinstance(version, str):
        raise CLIValidationError(f"{expected_tool} tool digest/version must be strings")
    if (
        not isinstance(version_args, list)
        or not version_args
        or not all(isinstance(argument, str) and argument for argument in version_args)
    ):
        raise CLIValidationError(f"{expected_tool} version_args must be a non-empty string list")
    tool_path = Path(executable)
    if tool_path.is_symlink():
        raise CLIValidationError(f"pinned tool path must not be a symlink: {tool_path}")
    try:
        pin = ToolPin(
            path=tool_path,
            sha256=digest,
            version=version,
            version_args=tuple(version_args),
        )
        validate_tool_pin(pin, runner=runner, timeout=timeout)
    except (AssetProvenanceError, OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        raise CLIValidationError(f"pinned tool validation failed for {expected_tool}: {error}") from error
    return LoadedToolPin(
        tool=expected_tool,
        pin=pin,
        pin_file_path=pin_path,
        pin_file_sha256=_sha256_file(pin_path),
    )


def parse_fasta_records(path: Path) -> tuple[FastaRecord, ...]:
    """Parse a strict FASTA while retaining the exact bytes of each record."""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise CLIValidationError(f"cannot read input FASTA: {error}") from error
    lines = payload.splitlines(keepends=True)
    records: list[FastaRecord] = []
    seen_ids: set[str] = set()
    current: list[bytes] = []

    def finish_record() -> None:
        if not current:
            return
        header = current[0].rstrip(b"\r\n")
        if not header.startswith(b">"):
            raise CLIValidationError("FASTA record does not start with a header")
        header_body = header[1:].strip()
        if not header_body:
            raise CLIValidationError("FASTA header is empty")
        try:
            sequence_id = header_body.split(None, 1)[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise CLIValidationError("FASTA identifiers must be ASCII") from error
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", sequence_id):
            raise CLIValidationError(f"FASTA record ID is not byte-stable: {sequence_id!r}")
        if sequence_id in seen_ids:
            raise CLIValidationError(f"duplicate FASTA record ID: {sequence_id}")
        sequence_bytes = b"".join(b"".join(current[1:]).split())
        if not sequence_bytes:
            raise CLIValidationError(f"FASTA record has no sequence: {sequence_id}")
        try:
            normalized_sequence = sequence_bytes.decode("ascii").upper()
        except UnicodeDecodeError as error:
            raise CLIValidationError(f"FASTA sequence must be ASCII: {sequence_id}") from error
        if not re.fullmatch(r"[ACGTN]+", normalized_sequence):
            raise CLIValidationError(f"FASTA sequence contains unsupported symbols: {sequence_id}")
        records.append(
            FastaRecord(
                sequence_id=sequence_id,
                original_bytes=b"".join(current),
                normalized_sequence=normalized_sequence,
            )
        )
        seen_ids.add(sequence_id)

    for line in lines:
        if line.startswith(b">"):
            finish_record()
            current = [line]
        else:
            if not current:
                raise CLIValidationError("FASTA bytes precede the first header")
            current.append(line)
    finish_record()
    if not records:
        raise CLIValidationError("input FASTA batch is empty")
    return tuple(records)


def partition_fasta_records(
    records: Sequence[FastaRecord],
    states_by_id: Mapping[str, SafetyState],
) -> dict[SafetyState, bytes]:
    """Partition exact source record bytes once, preserving relative input order."""
    record_ids = [record.sequence_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise CLIValidationError("input FASTA contains duplicate record IDs")
    if set(states_by_id) != set(record_ids):
        raise CLIValidationError("manifest record IDs do not exactly match the input FASTA")
    partitions: dict[SafetyState, bytearray] = {state: bytearray() for state in SafetyState}
    for record in records:
        try:
            state = SafetyState(states_by_id[record.sequence_id])
        except (KeyError, ValueError) as error:
            raise CLIValidationError(f"invalid safety state for record {record.sequence_id}") from error
        partitions[state].extend(record.original_bytes)
    return {state: bytes(payload) for state, payload in partitions.items()}


def scan_records(
    records: Sequence[FastaRecord],
    *,
    scanner,
    host_domain: HostDomain,
    strict_lysis: bool = False,
    max_workers: int = 1,
) -> BatchSafetyResult:
    """Run isolated record bundles with bounded concurrency and retain input order."""
    if type(max_workers) is not int or max_workers < 1:
        raise CLIValidationError("record workers must be a positive integer")

    def scan_one(indexed_record: tuple[int, FastaRecord]) -> ScannedRecord:
        input_index, record = indexed_record
        adapters = dict(scanner(record, input_index))
        result = aggregate_adapter_results(
            adapters,
            host_domain=host_domain,
            strict_lysis=strict_lysis,
        )
        return ScannedRecord(
            input_index=input_index,
            sequence_id=record.sequence_id,
            result=result,
            adapters=adapters,
        )

    indexed_records = tuple(enumerate(records))
    if max_workers == 1:
        scanned = [scan_one(indexed_record) for indexed_record in indexed_records]
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="phage-safety") as executor:
            scanned = list(executor.map(scan_one, indexed_records))
    states = {record.result.state for record in scanned}
    if SafetyState.FAIL in states:
        state = SafetyState.FAIL
    elif SafetyState.INDETERMINATE in states or not scanned:
        state = SafetyState.INDETERMINATE
    else:
        state = SafetyState.PASS
    return BatchSafetyResult(state=state, records=tuple(scanned))


def _available_cpu_slots() -> int:
    """Return the scheduler/container CPU affinity when available."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _resolve_record_workers(
    *,
    requested: int,
    record_count: int,
    tool_threads: int,
    cpu_slots: int,
) -> int:
    """Admit record workers only when their per-tool thread budget fits the CPU allocation."""
    for name, value in (
        ("requested record workers", requested),
        ("record count", record_count),
        ("tool threads", tool_threads),
        ("CPU slots", cpu_slots),
    ):
        if type(value) is not int or value < 1:
            raise CLIValidationError(f"{name} must be a positive integer")
    resolved = min(requested, record_count)
    if resolved * tool_threads > cpu_slots:
        raise CLIValidationError(
            f"record workers ({resolved}) x tool threads ({tool_threads}) exceed available CPU slots ({cpu_slots})"
        )
    return resolved


def _orf_indeterminate_adapters(
    *,
    host_domain: HostDomain,
    strict_lysis: bool,
    reason_codes: tuple[str, ...],
) -> dict[str, AdapterResult]:
    applicability = _applicability(host_domain, strict_lysis=strict_lysis)
    policy_identities = {
        "amr": _AMRFINDER_POLICY,
        "toxin": (TOXIN_HOMOLOGY_POLICY_V2.policy_id, TOXIN_HOMOLOGY_POLICY_V2.sha256),
        "lysogeny": (PHROGS_HOMOLOGY_POLICY_V1.policy_id, PHROGS_HOMOLOGY_POLICY_V1.sha256),
    }
    adapters: dict[str, AdapterResult] = {}
    for safety_class, required in applicability.items():
        policy_id, policy_sha256 = policy_identities[safety_class]
        adapters[safety_class] = AdapterResult(
            class_result=SafetyClassResult(
                safety_class=safety_class,
                state=SafetyState.INDETERMINATE,
                required=required,
                reason_codes=reason_codes or ("ORF_PREPARATION_INDETERMINATE",),
            ),
            policy_id=policy_id,
            policy_sha256=policy_sha256,
        )
    return adapters


def _default_orf_generation_identity() -> dict[str, object]:
    """Return the exact Task 3 implementation and installed predictor identity."""
    implementation_path = Path(safety_adapters.__file__).resolve()
    try:
        predictor_version = importlib.metadata.version("pyrodigal-gv")
    except importlib.metadata.PackageNotFoundError:
        predictor_version = "UNAVAILABLE"
    return {
        "predictor": "pyrodigal-gv",
        "predictor_version": predictor_version,
        "entry_point": "bionemo.evo2_phage_gen.sequence_safety_adapters:prepare_orf_artifacts_checked",
        "implementation_path": str(implementation_path),
        "implementation_sha256": _sha256_file(implementation_path),
    }


def _orf_query_provenance_record(query: ORFQueryRecord) -> dict[str, object]:
    return {
        "query_id": query.query_id,
        "sequence_id": query.sequence_id,
        "start": query.start,
        "end": query.end,
        "strand": query.strand,
        "frame": query.frame,
        "evidence_path": query.evidence_path,
        "nucleotide_length": len(query.nucleotide),
        "nucleotide_sha256": hashlib.sha256(query.nucleotide.encode()).hexdigest(),
        "protein_length": len(query.protein),
        "protein_sha256": hashlib.sha256(query.protein.encode()).hexdigest(),
    }


def _write_orf_provenance(
    record_root: Path,
    *,
    artifacts: ORFArtifacts | None,
    generation_identity: Mapping[str, object],
    preparation_state: SafetyState = SafetyState.PASS,
    reason_codes: Sequence[str] = (),
) -> Path:
    """Write the immutable predictor/fallback, artifact, and query inventory."""
    record_root = Path(record_root)
    _validate_json_value(generation_identity, label="ORF generation identity")
    artifact_records: dict[str, object] = {}
    queries: list[dict[str, object]] = []
    if artifacts is not None:
        for role in ("genomes_fna", "proteins_faa", "proteins_fna", "proteins_gff", "all_queries_faa"):
            artifact_path = Path(getattr(artifacts, role))
            try:
                relative = artifact_path.relative_to(record_root)
            except ValueError as error:
                raise CLIValidationError(f"ORF artifact {role} leaves its record root") from error
            if artifact_path.is_symlink() or not artifact_path.is_file():
                raise CLIValidationError(f"ORF artifact {role} is missing")
            artifact_records[role] = {
                "path": relative.as_posix(),
                "sha256": _sha256_file(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
            }
        queries.extend(_orf_query_provenance_record(query) for query in artifacts.query_records)
    query_inventory_sha256 = _canonical_mapping_sha256({"queries": queries})
    identity = dict(generation_identity)
    payload = {
        "schema_version": 1,
        "preparation_state": preparation_state.value,
        "reason_codes": list(reason_codes),
        "generation_identity": identity,
        "generation_identity_sha256": _canonical_mapping_sha256(identity),
        "fallback_policy": _ORF_FALLBACK_DESCRIPTOR,
        "fallback_policy_sha256": _ORF_FALLBACK_SHA256,
        "artifacts": artifact_records,
        "queries": queries,
        "query_inventory_sha256": query_inventory_sha256,
    }
    path = record_root / _ORF_PROVENANCE_FILENAME
    _write_json_fsync(path, payload)
    return path


def run_default_adapter_bundle(
    record: FastaRecord,
    input_index: int,
    *,
    work_root: Path,
    asset_manifest: Mapping[str, object],
    diamond_pin: ToolPin,
    mmseqs_pin: ToolPin,
    host_domain: HostDomain,
    strict_lysis: bool = False,
    threads: int = 1,
    timeout: float = 300.0,
    circular: bool = True,
    orf_generation_identity: Mapping[str, object] | None = None,
    predictor=None,
    runner=None,
    prepare_orfs=prepare_orf_artifacts_checked,
    amr_adapter=run_amrfinder,
    toxin_adapter=run_toxin_diamond,
    phrogs_adapter=run_phrogs,
) -> dict[str, AdapterResult]:
    """Prepare coordinated ORFs and run all applicable Task 3 adapters for one record."""
    selected_runner = subprocess.run if runner is None else runner
    work_dir = Path(work_root) / f"{input_index:06d}-{record.sequence_id}"
    if work_dir.exists() or work_dir.is_symlink():
        raise CLIValidationError(f"record work directory already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    preparation = prepare_orfs(
        (GenomeInput(record.sequence_id, record.normalized_sequence, circular=circular),),
        work_dir,
        predictor=predictor,
        minimum_fallback_amino_acids=8,
    )
    identity = _default_orf_generation_identity() if orf_generation_identity is None else orf_generation_identity
    if preparation.state is not SafetyState.PASS or preparation.artifacts is None:
        _write_orf_provenance(
            work_dir,
            artifacts=None,
            generation_identity=identity,
            preparation_state=SafetyState.INDETERMINATE,
            reason_codes=preparation.reason_codes,
        )
        return _orf_indeterminate_adapters(
            host_domain=host_domain,
            strict_lysis=strict_lysis,
            reason_codes=tuple(preparation.reason_codes),
        )
    for section in ("amrfinder_plus", "toxin_reference", "phrogs_v4"):
        if not isinstance(asset_manifest.get(section), Mapping):
            return _orf_indeterminate_adapters(
                host_domain=host_domain,
                strict_lysis=strict_lysis,
                reason_codes=("SAFETY_ASSET_MANIFEST_SECTION_MISSING",),
            )
    applicability = _applicability(host_domain, strict_lysis=strict_lysis)
    artifacts = preparation.artifacts
    _write_orf_provenance(
        work_dir,
        artifacts=artifacts,
        generation_identity=identity,
    )
    return {
        "amr": amr_adapter(
            artifacts,
            manifest_section=asset_manifest["amrfinder_plus"],
            work_dir=work_dir,
            threads=threads,
            required=applicability["amr"],
            runner=selected_runner,
            timeout=timeout,
        ),
        "toxin": toxin_adapter(
            artifacts,
            manifest_section=asset_manifest["toxin_reference"],
            tool_pin=diamond_pin,
            work_dir=work_dir,
            threads=threads,
            required=applicability["toxin"],
            runner=selected_runner,
            timeout=timeout,
        ),
        "lysogeny": phrogs_adapter(
            artifacts,
            manifest_section=asset_manifest["phrogs_v4"],
            tool_pin=mmseqs_pin,
            host_domain=host_domain,
            work_dir=work_dir,
            threads=threads,
            strict_lysis=strict_lysis,
            runner=selected_runner,
            timeout=timeout,
        ),
    }


def run_default_batched_adapter_bundles(
    records: Sequence[FastaRecord],
    *,
    work_root: Path,
    shared_root: Path,
    asset_manifest: Mapping[str, object],
    diamond_pin: ToolPin,
    mmseqs_pin: ToolPin,
    host_domain: HostDomain,
    strict_lysis: bool = False,
    threads: int = 8,
    batch_size: int = 8,
    batch_workers: int = 1,
    phrogs_threads: int = 1,
    phrogs_workers: int = 8,
    timeout: float = 300.0,
    circular: bool = True,
    orf_generation_identity: Mapping[str, object] | None = None,
    predictor=None,
    runner=None,
    prepare_orfs=prepare_orf_artifacts_checked,
    amr_batch_adapter=run_amrfinder_batch,
    toxin_batch_adapter=run_toxin_diamond_batch,
    phrogs_adapter=run_phrogs,
    phrogs_asset_validator=safety_adapters._prepare_validated_phrogs_assets,
    phrogs_asset_revalidator=safety_adapters._revalidate_phrogs_assets,
) -> BatchedScanExecution:
    """Run exact independent record parsing around shared AMR/toxin commands and per-record PHROGs."""
    for label, value in (
        ("threads", threads),
        ("batch size", batch_size),
        ("batch workers", batch_workers),
        ("PHROGs threads", phrogs_threads),
        ("PHROGs workers", phrogs_workers),
    ):
        if type(value) is not int or value < 1:
            raise CLIValidationError(f"{label} must be a positive integer")
    selected_records = tuple(records)
    if not selected_records:
        raise CLIValidationError("batched scan requires at least one record")
    selected_runner = subprocess.run if runner is None else runner
    record_root = Path(work_root)
    execution_root = Path(shared_root)
    if record_root.exists() or record_root.is_symlink() or execution_root.exists() or execution_root.is_symlink():
        raise CLIValidationError("batched scan output roots must not already exist")
    record_root.mkdir(parents=True)
    execution_root.mkdir(parents=True)
    identity = _default_orf_generation_identity() if orf_generation_identity is None else orf_generation_identity
    applicability = _applicability(host_domain, strict_lysis=strict_lysis)
    phrogs_section = asset_manifest.get("phrogs_v4")
    validated_phrogs_assets = None
    if isinstance(phrogs_section, Mapping):
        try:
            validated_phrogs_assets = phrogs_asset_validator(phrogs_section)
        except (AssetProvenanceError, KeyError, OSError, TypeError, ValueError):
            validated_phrogs_assets = None
    indexed = tuple(enumerate(selected_records))
    groups = tuple(indexed[start : start + batch_size] for start in range(0, len(indexed), batch_size))

    def run_group(group_item: tuple[int, tuple[tuple[int, FastaRecord], ...]]):
        group_index, group = group_item
        adapters_by_index: dict[int, dict[str, AdapterResult]] = {}
        artifacts_by_record: list[tuple[str, ORFArtifacts]] = []
        output_roots: dict[str, Path] = {}
        indices_by_record: dict[str, int] = {}
        for input_index, record in group:
            output_dir = record_root / f"{input_index:06d}-{record.sequence_id}"
            if output_dir.exists() or output_dir.is_symlink():
                raise CLIValidationError(f"record work directory already exists: {output_dir}")
            output_dir.mkdir()
            preparation = prepare_orfs(
                (GenomeInput(record.sequence_id, record.normalized_sequence, circular=circular),),
                output_dir,
                predictor=predictor,
                minimum_fallback_amino_acids=8,
            )
            if preparation.state is not SafetyState.PASS or preparation.artifacts is None:
                _write_orf_provenance(
                    output_dir,
                    artifacts=None,
                    generation_identity=identity,
                    preparation_state=SafetyState.INDETERMINATE,
                    reason_codes=preparation.reason_codes,
                )
                adapters_by_index[input_index] = _orf_indeterminate_adapters(
                    host_domain=host_domain,
                    strict_lysis=strict_lysis,
                    reason_codes=tuple(preparation.reason_codes),
                )
                continue
            _write_orf_provenance(output_dir, artifacts=preparation.artifacts, generation_identity=identity)
            artifacts_by_record.append((record.sequence_id, preparation.artifacts))
            output_roots[record.sequence_id] = output_dir
            indices_by_record[record.sequence_id] = input_index

        shared: list[BatchAdapterExecution] = []
        if artifacts_by_record:
            missing_sections = tuple(
                section
                for section in ("amrfinder_plus", "toxin_reference", "phrogs_v4")
                if not isinstance(asset_manifest.get(section), Mapping)
            )
            if missing_sections:
                for record_id, _ in artifacts_by_record:
                    adapters_by_index[indices_by_record[record_id]] = _orf_indeterminate_adapters(
                        host_domain=host_domain,
                        strict_lysis=strict_lysis,
                        reason_codes=("SAFETY_ASSET_MANIFEST_SECTION_MISSING",),
                    )
            else:
                group_id = f"batch-{group_index:06d}"
                amr_execution = amr_batch_adapter(
                    tuple(artifacts_by_record),
                    manifest_section=asset_manifest["amrfinder_plus"],
                    work_dir=execution_root / f"{group_id}-amr",
                    record_output_roots=output_roots,
                    threads=threads,
                    required=applicability["amr"],
                    runner=selected_runner,
                    timeout=timeout,
                )
                toxin_execution = toxin_batch_adapter(
                    tuple(artifacts_by_record),
                    manifest_section=asset_manifest["toxin_reference"],
                    tool_pin=diamond_pin,
                    work_dir=execution_root / f"{group_id}-toxin",
                    record_output_roots=output_roots,
                    threads=threads,
                    required=applicability["toxin"],
                    runner=selected_runner,
                    timeout=timeout,
                )
                shared.extend((amr_execution, toxin_execution))
                amr_by_record = dict(amr_execution.record_results)
                toxin_by_record = dict(toxin_execution.record_results)

                def scan_phrogs(item: tuple[str, ORFArtifacts]) -> tuple[str, AdapterResult]:
                    record_id, artifacts = item
                    return (
                        record_id,
                        phrogs_adapter(
                            artifacts,
                            manifest_section=asset_manifest["phrogs_v4"],
                            tool_pin=mmseqs_pin,
                            host_domain=host_domain,
                            work_dir=output_roots[record_id],
                            threads=phrogs_threads,
                            strict_lysis=strict_lysis,
                            runner=selected_runner,
                            timeout=timeout,
                            _validated_assets=validated_phrogs_assets,
                        ),
                    )

                if phrogs_workers == 1:
                    phrogs_results = tuple(scan_phrogs(item) for item in artifacts_by_record)
                else:
                    with ThreadPoolExecutor(
                        max_workers=min(phrogs_workers, len(artifacts_by_record)),
                        thread_name_prefix=f"{group_id}-phrogs",
                    ) as executor:
                        phrogs_results = tuple(executor.map(scan_phrogs, artifacts_by_record))
                phrogs_by_record = dict(phrogs_results)
                if not (set(amr_by_record) == set(toxin_by_record) == set(phrogs_by_record) == set(indices_by_record)):
                    raise CLIValidationError("batched adapter result inventories differ")
                for record_id in indices_by_record:
                    adapters_by_index[indices_by_record[record_id]] = {
                        "amr": amr_by_record[record_id],
                        "toxin": toxin_by_record[record_id],
                        "lysogeny": phrogs_by_record[record_id],
                    }

        scanned: list[ScannedRecord] = []
        for input_index, record in group:
            output_dir = record_root / f"{input_index:06d}-{record.sequence_id}"
            adapters = _trusted_adapter_bundle(adapters_by_index[input_index], record_root=output_dir)
            result = aggregate_adapter_results(
                adapters,
                host_domain=host_domain,
                strict_lysis=strict_lysis,
            )
            scanned.append(
                ScannedRecord(
                    input_index=input_index,
                    sequence_id=record.sequence_id,
                    result=result,
                    adapters=adapters,
                )
            )
        return tuple(scanned), tuple(shared)

    group_items = tuple(enumerate(groups))
    if batch_workers == 1:
        group_results = tuple(run_group(item) for item in group_items)
    else:
        with ThreadPoolExecutor(
            max_workers=min(batch_workers, len(groups)), thread_name_prefix="safety-batch"
        ) as executor:
            group_results = tuple(executor.map(run_group, group_items))
    scanned_records = tuple(record for scanned, _ in group_results for record in scanned)
    shared_executions = tuple(execution for _, shared in group_results for execution in shared)
    if validated_phrogs_assets is not None:
        try:
            phrogs_asset_revalidator(phrogs_section, validated_phrogs_assets)
        except (AssetProvenanceError, KeyError, OSError, TypeError, ValueError) as error:
            raise CLIValidationError("PHROGs assets changed during batched execution") from error
    states = {record.result.state for record in scanned_records}
    if SafetyState.FAIL in states:
        state = SafetyState.FAIL
    elif SafetyState.INDETERMINATE in states:
        state = SafetyState.INDETERMINATE
    else:
        state = SafetyState.PASS
    return BatchedScanExecution(
        batch=BatchSafetyResult(state=state, records=scanned_records),
        shared_executions=shared_executions,
    )


def _default_adapter_replayer(
    safety_class: str,
    *,
    normalized_output: Path,
    artifacts: ORFArtifacts,
    asset_manifest: Mapping[str, object],
    diamond_pin: ToolPin,
    mmseqs_pin: ToolPin,
    host_domain: HostDomain,
    strict_lysis: bool,
    required: bool,
    _validated_phrogs_assets=None,
) -> AdapterResult:
    """Replay the exact Task 3 validated parser for one completed attempt."""
    if safety_class == "amr":
        return safety_adapters._parse_amrfinder_output_validated(
            normalized_output,
            artifacts=artifacts,
            manifest_section=asset_manifest["amrfinder_plus"],
            required=required,
        )
    if safety_class == "toxin":
        return safety_adapters._parse_toxin_diamond_output_validated(
            normalized_output,
            artifacts=artifacts,
            manifest_section=asset_manifest["toxin_reference"],
            tool_pin=diamond_pin,
            required=required,
            policy=TOXIN_HOMOLOGY_POLICY_V2,
        )
    if safety_class == "lysogeny":
        return safety_adapters._parse_phrogs_output_validated(
            normalized_output,
            artifacts=artifacts,
            manifest_section=asset_manifest["phrogs_v4"],
            tool_pin=mmseqs_pin,
            host_domain=host_domain,
            strict_lysis=strict_lysis,
            policy=PHROGS_HOMOLOGY_POLICY_V1,
            _validated_assets=_validated_phrogs_assets,
        )
    raise CLIValidationError(f"unsupported adapter replay class: {safety_class}")


def _default_orf_replayer(record: FastaRecord, *, circular: bool) -> ReplayedORFEvidence:
    """Rerun Task 3 ORF preparation from authenticated genome bytes."""
    with tempfile.TemporaryDirectory(prefix="evo2-phage-orf-replay-") as temporary:
        replay = prepare_orf_artifacts_checked(
            (GenomeInput(record.sequence_id, record.normalized_sequence, circular=circular),),
            Path(temporary),
            minimum_fallback_amino_acids=8,
        )
        if replay.state is not SafetyState.PASS or replay.artifacts is None:
            raise CLIValidationError("ORF predictor replay did not complete successfully")
        artifacts = replay.artifacts
        roles = ("genomes_fna", "proteins_faa", "proteins_fna", "proteins_gff", "all_queries_faa")
        return ReplayedORFEvidence(
            artifact_bytes={role: Path(getattr(artifacts, role)).read_bytes() for role in roles},
            query_records=tuple(artifacts.query_records),
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _environment_provenance() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


@dataclass(frozen=True)
class CLIRuntime:
    """Injectable process boundaries; production defaults remain fully functional."""

    record_scanner: Callable[..., Mapping[str, AdapterResult]] = run_default_adapter_bundle
    batch_scanner: Callable[..., BatchedScanExecution] = run_default_batched_adapter_bundles
    asset_loader: Callable[..., LoadedSafetyAssetManifest] = load_safety_asset_manifest
    tool_pin_loader: Callable[..., LoadedToolPin] = load_tool_pin_file
    clock: Callable[[], datetime] = _utc_now
    environment_collector: Callable[[], Mapping[str, object]] = _environment_provenance
    orf_identity_collector: Callable[[], Mapping[str, object]] = _default_orf_generation_identity
    orf_replayer: Callable[..., ReplayedORFEvidence] = _default_orf_replayer
    adapter_replayer: Callable[..., AdapterResult] = _default_adapter_replayer
    phrogs_asset_validator: Callable[..., object] = safety_adapters._prepare_validated_phrogs_assets
    phrogs_asset_revalidator: Callable[..., None] = safety_adapters._revalidate_phrogs_assets
    command_runner: Callable[..., object] = subprocess.run
    replace: Callable[[Path, Path], object] = os.replace


def _normalize_requested_host_domain(requested: HostDomain, evidence: HostEvidence) -> HostDomain:
    domains = evidence.replication_host_domains
    if domains == frozenset({HostDomain.BACTERIA, HostDomain.ARCHAEA}):
        observed = HostDomain.BACTERIA_AND_ARCHAEA
    elif domains == frozenset({HostDomain.BACTERIA_AND_ARCHAEA}):
        observed = HostDomain.BACTERIA_AND_ARCHAEA
    elif domains == frozenset({HostDomain.BACTERIA}):
        observed = HostDomain.BACTERIA
    elif domains == frozenset({HostDomain.ARCHAEA}):
        observed = HostDomain.ARCHAEA
    else:
        raise CLIValidationError("host evidence does not resolve to one supported prokaryotic profile")
    if requested is not observed:
        raise CLIValidationError("host profile conflicts with the supplied host evidence")
    return observed


def _trusted_adapter_bundle(
    adapters: Mapping[str, AdapterResult],
    *,
    record_root: Path,
) -> dict[str, AdapterResult]:
    """Downgrade unbound measured outcomes before aggregation or publication."""
    unknown = sorted(set(adapters) - set(_ADAPTER_POLICIES))
    if unknown:
        raise CLIValidationError(f"unknown adapter safety classes: {unknown}")
    trusted: dict[str, AdapterResult] = {}
    for safety_class in ("amr", "toxin", "lysogeny"):
        adapter = adapters.get(safety_class)
        if adapter is None:
            continue
        expected_policy = _ADAPTER_POLICIES[safety_class]
        reason: str | None = None
        if (adapter.policy_id, adapter.policy_sha256) != expected_policy:
            reason = "ADAPTER_POLICY_PROVENANCE_MISMATCH"
        elif adapter.class_result.state in {SafetyState.PASS, SafetyState.FAIL}:
            if not adapter.command or not adapter.raw_output_path or not adapter.raw_output_sha256:
                reason = "ADAPTER_MEASURED_EVIDENCE_INCOMPLETE"
            else:
                output_path = Path(adapter.raw_output_path)
                try:
                    output_path.relative_to(record_root)
                except ValueError:
                    reason = "ADAPTER_OUTPUT_PATH_OUTSIDE_RECORD_ROOT"
                if reason is None:
                    if output_path.is_symlink() or not output_path.is_file():
                        reason = "ADAPTER_OUTPUT_MISSING"
                    elif _sha256_file(output_path) != adapter.raw_output_sha256:
                        reason = "ADAPTER_OUTPUT_DIGEST_DRIFT"
                raw_names = {
                    "toxin": "toxin_diamond.raw.tsv",
                    "lysogeny": "phrogs.raw.tsv",
                }
                raw_name = raw_names.get(safety_class)
                if reason is None and raw_name is not None:
                    raw_command_output = record_root / raw_name
                    if raw_command_output.is_symlink() or not raw_command_output.is_file():
                        reason = "ADAPTER_RAW_COMMAND_OUTPUT_MISSING"
        if reason is None:
            trusted[safety_class] = adapter
        else:
            trusted[safety_class] = replace(
                adapter,
                class_result=SafetyClassResult(
                    safety_class=safety_class,
                    state=SafetyState.INDETERMINATE,
                    required=adapter.class_result.required,
                    reason_codes=(reason,),
                ),
            )
    return trusted


def _relative_artifact(path_value: str | None, *, root: Path) -> dict[str, object] | None:
    if path_value is None:
        return None
    path = Path(path_value)
    try:
        relative = path.relative_to(root)
    except ValueError:
        return {"path": str(path), "sha256": _sha256_file(path) if path.is_file() else None, "owned": False}
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(path) if path.is_file() else None,
        "owned": True,
    }


def _canonicalize_command(command: Sequence[str], *, root: Path) -> list[str]:
    """Replace transient owned absolute paths with one replayable output-root token."""
    canonical: list[str] = []
    absolute_root = root.absolute()
    for argument in command:
        if not isinstance(argument, str) or not argument:
            raise CLIValidationError("adapter command arguments must be non-empty strings")
        candidate = Path(argument)
        if candidate.is_absolute():
            try:
                relative = candidate.relative_to(absolute_root)
            except ValueError:
                canonical.append(argument)
            else:
                canonical.append(f"@OUTPUT_ROOT/{relative.as_posix()}")
        else:
            canonical.append(argument)
    return canonical


def _adapter_execution_status(adapter: AdapterResult) -> str:
    if not adapter.command:
        return "NOT_STARTED"
    if any(reason.startswith("ADAPTER_") for reason in adapter.class_result.reason_codes):
        return "FAILED"
    if adapter.raw_output_path is not None and adapter.raw_output_sha256 is not None:
        return "COMPLETED_AND_PARSED"
    return "FAILED"


def _serialize_adapter_attempt(
    safety_class: str,
    adapter: AdapterResult,
    *,
    root: Path,
    record_root: Path,
) -> dict[str, object]:
    raw_names = {
        "amr": "amrfinder.tsv",
        "toxin": "toxin_diamond.raw.tsv",
        "lysogeny": "phrogs.raw.tsv",
    }
    payload = {
        "safety_class": safety_class,
        "execution_status": _adapter_execution_status(adapter),
        "state": adapter.class_result.state.value,
        "reason_codes": list(adapter.class_result.reason_codes),
        "policy_id": adapter.policy_id,
        "policy_sha256": adapter.policy_sha256,
        "command_cwd": "@OUTPUT_ROOT",
        "command": _canonicalize_command(adapter.command, root=root),
        "normalized_output": _relative_artifact(adapter.raw_output_path, root=root),
        "raw_command_output": _relative_artifact(str(record_root / raw_names[safety_class]), root=root),
        "primary_findings": [finding.to_dict() for finding in adapter.class_result.findings],
        "supplemental_findings": [finding.to_dict() for finding in adapter.supplemental_findings],
    }
    if adapter.shared_execution_id is not None:
        payload["shared_execution_id"] = adapter.shared_execution_id
    return payload


def _serialize_shared_execution(
    execution: BatchAdapterExecution,
    *,
    root: Path,
    record_indices: Mapping[str, int],
) -> dict[str, object]:
    """Serialize one real shared command separately from its per-record parser results."""
    if set(record_indices) != set(execution.record_ids):
        raise CLIValidationError("shared execution record indices do not match its inventory")
    artifacts = execution.inputs.artifacts
    input_paths = {
        role: Path(getattr(artifacts, role))
        for role in ("genomes_fna", "proteins_faa", "proteins_fna", "proteins_gff", "all_queries_faa")
    }
    inputs = {role: _relative_artifact(str(path), root=root) for role, path in input_paths.items()}
    if any(value is None or not value["owned"] for value in inputs.values()):
        raise CLIValidationError("shared execution inputs must be owned output artifacts")
    raw_output = _relative_artifact(
        None if execution.raw_output_path is None else str(execution.raw_output_path),
        root=root,
    )
    if raw_output is None or not raw_output["owned"]:
        raise CLIValidationError("completed shared execution output must be an owned artifact")
    return {
        "execution_id": execution.batch_id,
        "safety_class": execution.safety_class,
        "record_ids": list(execution.record_ids),
        "record_indices": [record_indices[record_id] for record_id in execution.record_ids],
        "command_cwd": "@OUTPUT_ROOT",
        "command": _canonicalize_command(execution.command, root=root),
        "inputs": inputs,
        "raw_command_output": raw_output,
        "split_policy": {
            "policy_id": execution.split_policy_id,
            "policy_sha256": execution.split_policy_sha256,
        },
    }


def _validate_shared_executions(
    value: object,
    *,
    root: Path,
    input_records: Sequence[FastaRecord],
    assets: LoadedSafetyAssetManifest,
    diamond: LoadedToolPin,
    threads: int,
    batch_size: int,
) -> dict[str, dict[str, object]]:
    """Authenticate each shared command independently from its later per-record parser results."""
    if not isinstance(value, list):
        raise CLIValidationError("shared_executions must be a list")
    expected_ids = [record.sequence_id for record in input_records]
    validated: dict[str, dict[str, object]] = {}
    class_groups: set[tuple[str, tuple[int, ...]]] = set()
    for index, item in enumerate(value):
        row = _strict_payload(
            item,
            name=f"shared_execution[{index}]",
            keys=frozenset(
                {
                    "execution_id",
                    "safety_class",
                    "record_ids",
                    "record_indices",
                    "command_cwd",
                    "command",
                    "inputs",
                    "raw_command_output",
                    "split_policy",
                }
            ),
        )
        execution_id = row["execution_id"]
        safety_class = row["safety_class"]
        if not isinstance(execution_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", execution_id) is None:
            raise CLIValidationError("shared execution ID is invalid")
        if execution_id in validated or safety_class not in {"amr", "toxin"}:
            raise CLIValidationError("shared execution identity or class is invalid")
        if row["command_cwd"] != "@OUTPUT_ROOT":
            raise CLIValidationError("shared execution cwd is invalid")
        record_ids = _string_list(row["record_ids"], label="shared execution record_ids")
        record_indices = row["record_indices"]
        if (
            not isinstance(record_indices, list)
            or not record_indices
            or any(type(item_index) is not int for item_index in record_indices)
            or record_indices != sorted(set(record_indices))
            or len(record_indices) > batch_size
            or len(record_ids) != len(record_indices)
            or any(item_index < 0 or item_index >= len(input_records) for item_index in record_indices)
            or record_ids != [expected_ids[item_index] for item_index in record_indices]
            or len({item_index // batch_size for item_index in record_indices}) != 1
        ):
            raise CLIValidationError("shared execution record inventory is invalid")
        group_key = (str(safety_class), tuple(record_indices))
        if group_key in class_groups:
            raise CLIValidationError("duplicate shared execution for one record group and safety class")
        class_groups.add(group_key)
        expected_prefix = f"shared-executions/{execution_id}/"
        inputs = _strict_payload(
            row["inputs"],
            name="shared execution inputs",
            keys=frozenset({"genomes_fna", "proteins_faa", "proteins_fna", "proteins_gff", "all_queries_faa"}),
        )
        input_paths: dict[str, Path] = {}
        for role, artifact in inputs.items():
            path = _validate_owned_artifact(
                artifact,
                root=root,
                label=f"shared execution {role}",
                required=True,
                prefix=expected_prefix + "inputs/",
            )
            assert path is not None
            input_paths[role] = path
        raw_path = _validate_owned_artifact(
            row["raw_command_output"],
            root=root,
            label="shared execution raw output",
            required=True,
            prefix=expected_prefix,
        )
        assert raw_path is not None
        split_policy = _strict_payload(
            row["split_policy"],
            name="shared execution split policy",
            keys=frozenset({"policy_id", "policy_sha256"}),
        )
        expected_split = {
            "amr": {
                "policy_id": AMRFINDER_SPLIT_POLICY_ID,
                "policy_sha256": AMRFINDER_SPLIT_POLICY_SHA256,
            },
            "toxin": {
                "policy_id": DIAMOND_SPLIT_POLICY_ID,
                "policy_sha256": DIAMOND_SPLIT_POLICY_SHA256,
            },
        }[str(safety_class)]
        if dict(split_policy) != expected_split:
            raise CLIValidationError("shared execution split policy drift")
        command = _string_list(row["command"], label="shared execution command", unique=False)
        if safety_class == "amr":
            section = assets.manifest.get("amrfinder_plus")
            if not isinstance(section, Mapping):
                raise CLIValidationError("AMRFinder asset section is missing")
            expected_command = build_amrfinder_command(
                amrfinder=Path(_asset_string(section.get("binary_path"), label="AMRFinder binary")),
                genomes_fna=input_paths["genomes_fna"],
                proteins_faa=input_paths["proteins_faa"],
                proteins_gff=input_paths["proteins_gff"],
                database_dir=Path(_asset_string(section.get("database_path"), label="AMRFinder database")),
                blast_bin_dir=Path(_asset_string(section.get("blastx_path"), label="AMRFinder BLASTX")).parent,
                hmmer_bin_dir=Path(_asset_string(section.get("hmmsearch_path"), label="AMRFinder HMM search")).parent,
                threads=threads,
                output_tsv=raw_path,
            )
        else:
            toxin = assets.manifest.get("toxin_reference")
            toxin_files = None if not isinstance(toxin, Mapping) else toxin.get("files")
            database = None if not isinstance(toxin_files, Mapping) else toxin_files.get("diamond_database")
            if not isinstance(database, Mapping):
                raise CLIValidationError("toxin database asset section is missing")
            expected_command = build_diamond_command(
                diamond=diamond.pin.path,
                queries_faa=input_paths["all_queries_faa"],
                database=Path(_asset_string(database.get("path"), label="toxin database")),
                output_tsv=raw_path,
                threads=threads,
            )
        if command != _canonicalize_command(expected_command, root=root):
            raise CLIValidationError("shared execution command drift")
        validated[execution_id] = {
            "safety_class": safety_class,
            "record_ids": tuple(record_ids),
            "record_indices": tuple(record_indices),
            "command": tuple(command),
            "input_paths": input_paths,
            "raw_path": raw_path,
        }
    return validated


def _validate_shared_execution_record_bindings(
    shared_executions: Mapping[str, Mapping[str, object]],
    *,
    root: Path,
    record_artifacts: Mapping[str, ORFArtifacts],
    record_index: Mapping[str, int],
) -> None:
    """Rebuild every combined input and per-record split from authenticated record artifacts."""
    grouped_classes: dict[tuple[int, ...], set[str]] = {}
    for execution_id, execution in shared_executions.items():
        record_ids = tuple(execution["record_ids"])
        indices = tuple(execution["record_indices"])
        if any(
            record_id not in record_artifacts or record_index.get(record_id) != index
            for record_id, index in zip(record_ids, indices, strict=True)
        ):
            raise CLIValidationError("shared execution references a record without measured ORF evidence")
        grouped_classes.setdefault(indices, set()).add(str(execution["safety_class"]))
        inputs = execution["input_paths"]
        assert isinstance(inputs, Mapping)
        for role in ("genomes_fna", "proteins_faa", "proteins_fna", "all_queries_faa"):
            expected = b"".join(
                Path(getattr(record_artifacts[record_id], role)).read_bytes() for record_id in record_ids
            )
            if Path(inputs[role]).read_bytes() != expected:
                raise CLIValidationError(f"shared execution {role} bytes differ from its record inputs")
        gff_payload = bytearray(b"##gff-version 3\n")
        for record_id in record_ids:
            source = record_artifacts[record_id].proteins_gff.read_bytes()
            header, separator, remainder = source.partition(b"\n")
            if header != b"##gff-version 3" or separator != b"\n":
                raise CLIValidationError("record GFF is not canonical during shared execution replay")
            gff_payload.extend(remainder)
        if Path(inputs["proteins_gff"]).read_bytes() != bytes(gff_payload):
            raise CLIValidationError("shared execution GFF bytes differ from its record inputs")
        query_records = tuple(query for record_id in record_ids for query in record_artifacts[record_id].query_records)
        batched = BatchedORFInputs(
            artifacts=ORFArtifacts(
                genomes_fna=Path(inputs["genomes_fna"]),
                proteins_faa=Path(inputs["proteins_faa"]),
                proteins_fna=Path(inputs["proteins_fna"]),
                proteins_gff=Path(inputs["proteins_gff"]),
                all_queries_faa=Path(inputs["all_queries_faa"]),
                query_records=query_records,
            ),
            record_ids=record_ids,
            query_owners=tuple((query.query_id, query.sequence_id) for query in query_records),
        )
        with tempfile.TemporaryDirectory(prefix=f"evo2-safety-{execution_id}-split-") as temporary:
            if execution["safety_class"] == "amr":
                split = split_amrfinder_batch_output(
                    Path(execution["raw_path"]), batched=batched, output_root=Path(temporary) / "split"
                )
                filename = "amrfinder.tsv"
            else:
                split = split_diamond_batch_output(
                    Path(execution["raw_path"]), batched=batched, output_root=Path(temporary) / "split"
                )
                filename = "toxin_diamond.raw.tsv"
            for record_id, index in zip(record_ids, indices, strict=True):
                recorded = root / "records" / f"{index:06d}-{record_id}" / filename
                if split[record_id].read_bytes() != recorded.read_bytes():
                    raise CLIValidationError("shared execution split differs from recorded per-record evidence")
    if any(classes != {"amr", "toxin"} for classes in grouped_classes.values()):
        raise CLIValidationError("shared execution groups must contain exactly AMR and toxin commands")


def _record_reason_codes(result: GenomeSafetyResult) -> list[str]:
    return list(dict.fromkeys(reason for class_result in result.class_results for reason in class_result.reason_codes))


def _serialize_orf_provenance(*, record_root: Path, root: Path) -> dict[str, object]:
    path = record_root / _ORF_PROVENANCE_FILENAME
    if not path.is_file() or path.is_symlink():
        return {
            "artifact": None,
            "preparation_state": SafetyState.INDETERMINATE.value,
            "generation_identity_sha256": None,
            "query_inventory_sha256": None,
        }
    payload = _load_strict_json(path, label="ORF provenance")
    strict = _strict_payload(
        payload,
        name="ORF provenance",
        keys=frozenset(
            {
                "schema_version",
                "preparation_state",
                "reason_codes",
                "generation_identity",
                "generation_identity_sha256",
                "fallback_policy",
                "fallback_policy_sha256",
                "artifacts",
                "queries",
                "query_inventory_sha256",
            }
        ),
    )
    return {
        "artifact": _relative_artifact(str(path), root=root),
        "preparation_state": strict["preparation_state"],
        "generation_identity_sha256": strict["generation_identity_sha256"],
        "query_inventory_sha256": strict["query_inventory_sha256"],
    }


def _serialize_scanned_record(
    scanned: ScannedRecord,
    record: FastaRecord,
    *,
    root: Path,
    evidence: HostEvidence,
    host_domain: HostDomain,
    strict_lysis: bool,
    circular: bool,
) -> dict[str, object]:
    record_root = root / "records" / f"{scanned.input_index:06d}-{record.sequence_id}"
    evidence_mapping = evidence.to_dict()
    evidence_digest = hashlib.sha256(
        json.dumps(evidence_mapping, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    class_by_name = {result.safety_class: result for result in scanned.result.class_results}
    complete_adapters = dict(scanned.adapters)
    for safety_class in ("amr", "toxin", "lysogeny"):
        if safety_class not in complete_adapters:
            policy_id, policy_sha256 = _ADAPTER_POLICIES[safety_class]
            complete_adapters[safety_class] = AdapterResult(
                class_result=class_by_name[safety_class],
                policy_id=policy_id,
                policy_sha256=policy_sha256,
            )
    return {
        "record_id": record.sequence_id,
        "input_index": scanned.input_index,
        "sequence_sha256": hashlib.sha256(record.normalized_sequence.encode()).hexdigest(),
        "original_record_sha256": hashlib.sha256(record.original_bytes).hexdigest(),
        "sequence_length": len(record.normalized_sequence),
        "circular": circular,
        "host_evidence": evidence_mapping,
        "host_evidence_sha256": evidence_digest,
        "resolved_host_profile": host_domain.value,
        "strict_lysis": strict_lysis,
        "applicability": _applicability(host_domain, strict_lysis=strict_lysis),
        "orf_provenance": _serialize_orf_provenance(record_root=record_root, root=root),
        "state": scanned.result.state.value,
        "reason_codes": _record_reason_codes(scanned.result),
        "class_results": [result.to_dict() for result in scanned.result.class_results],
        "adapter_attempts": [
            _serialize_adapter_attempt(
                safety_class,
                complete_adapters[safety_class],
                root=root,
                record_root=record_root,
            )
            for safety_class in ("amr", "toxin", "lysogeny")
        ],
    }


def _tool_manifest_record(loaded: LoadedToolPin) -> dict[str, object]:
    return {
        "tool": loaded.tool,
        "path": str(loaded.pin.path),
        "sha256": loaded.pin.sha256,
        "version": loaded.pin.version,
        "version_args": list(loaded.pin.version_args),
        "pin_file_path": str(loaded.pin_file_path),
        "pin_file_sha256": loaded.pin_file_sha256,
    }


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CLIValidationError("runtime clock must return a timezone-aware timestamp")
    return value.isoformat()


def _write_json_fsync(path: Path, value: Mapping[str, object]) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("wb") as destination:
        destination.write(encoded)
        destination.flush()
        os.fsync(destination.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_owned_regular_files(root: Path) -> None:
    """Durably flush every regular output before its digest enters a manifest."""
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise CLIValidationError(f"cannot open owned output for fsync: {path}: {error}") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise CLIValidationError(f"owned output is not a regular file: {path}")
            os.fsync(descriptor)
        except OSError as error:
            raise CLIValidationError(f"cannot fsync owned output: {path}: {error}") from error
        finally:
            os.close(descriptor)


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CLIValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_json_value(value: object, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool)) or type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CLIValidationError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CLIValidationError(f"{label} contains a non-string mapping key")
            _validate_json_value(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, label=f"{label}[{index}]")
        return
    raise CLIValidationError(f"{label} contains a non-JSON value")


def _reject_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise CLIValidationError(f"{label} path must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise CLIValidationError(f"{label} path contains a symlink component")
        except FileNotFoundError:
            return
        except OSError as error:
            raise CLIValidationError(f"cannot inspect {label} path: {error}") from error


def _sha256_regular_file(path: Path, *, label: str) -> str:
    _reject_symlink_components(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CLIValidationError(f"cannot open {label}: {error}") from error
    digest = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CLIValidationError(f"{label} must be a regular file")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_regular_file_bytes(path: Path, *, label: str) -> bytes:
    """Read one immutable byte snapshot through a single no-follow descriptor."""
    path = Path(path).absolute()
    _reject_symlink_components(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CLIValidationError(f"cannot open {label}: {error}") from error
    chunks: list[bytes] = []
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CLIValidationError(f"{label} must be a regular file")
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _external_file(path_value: object, *, label: str, expected_sha256: object) -> Path:
    if not isinstance(path_value, str):
        raise CLIValidationError(f"{label} path must be a string")
    path = Path(path_value)
    expected = _require_digest(expected_sha256, label=f"{label} SHA-256")
    if _sha256_regular_file(path, label=label) != expected:
        raise CLIValidationError(f"{label} digest drift")
    return path


def _owned_path(root: Path, path_value: object, *, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value or "\\" in path_value:
        raise CLIValidationError(f"{label} must be a normalized relative POSIX path")
    relative = PurePosixPath(path_value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise CLIValidationError(f"{label} must stay beneath the manifest root")
    candidate = root.joinpath(*relative.parts)
    _reject_symlink_components(root.absolute(), label="manifest root")
    current = root
    for part in relative.parts:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise CLIValidationError(f"{label} contains a symlink component")
        except FileNotFoundError:
            break
        except OSError as error:
            raise CLIValidationError(f"cannot inspect {label}: {error}") from error
    return candidate


def _validate_owned_artifact(
    value: object,
    *,
    root: Path,
    label: str,
    required: bool,
    prefix: str | None = None,
) -> Path | None:
    if value is None:
        if required:
            raise CLIValidationError(f"{label} is required for a measured adapter result")
        return None
    artifact = _strict_payload(
        value,
        name=label,
        keys=frozenset({"path", "sha256", "owned"}),
    )
    if artifact["owned"] is not True:
        raise CLIValidationError(f"{label} must be owned by the published output")
    path = _owned_path(root, artifact["path"], label=f"{label}.path")
    if prefix is not None and not str(artifact["path"]).startswith(prefix):
        raise CLIValidationError(f"{label} crosses its record boundary")
    digest = artifact["sha256"]
    if digest is None and not required:
        if path.exists() or path.is_symlink():
            raise CLIValidationError(f"{label} exists without a recorded digest")
        return None
    expected = _require_digest(digest, label=f"{label}.sha256")
    if _sha256_regular_file(path, label=label) != expected:
        raise CLIValidationError(f"{label} digest drift")
    return path


def _string_list(value: object, *, label: str, unique: bool = True) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CLIValidationError(f"{label} must be a string list")
    if unique and len(value) != len(set(value)):
        raise CLIValidationError(f"{label} contains duplicates")
    return value


def _manifest_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise CLIValidationError(f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise CLIValidationError(f"{label} is not an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CLIValidationError(f"{label} must include a timezone")
    return parsed


def _validate_cli_identity(value: object) -> None:
    identity = _strict_payload(
        value,
        name="cli_identity",
        keys=frozenset({"name", "version", "entry_point", "source_path", "source_sha256"}),
    )
    if identity["name"] != CLI_ID or identity["version"] != CLI_VERSION:
        raise CLIValidationError("CLI identity mismatch")
    if identity["entry_point"] != "bionemo.evo2_phage_gen.sequence_safety_cli:main":
        raise CLIValidationError("CLI entry-point identity mismatch")
    expected_source = Path(__file__).resolve()
    if identity["source_path"] != str(expected_source):
        raise CLIValidationError("CLI source path mismatch")
    expected_sha256 = _sha256_regular_file(expected_source, label="CLI source")
    if identity["source_sha256"] != expected_sha256:
        raise CLIValidationError("CLI source digest mismatch")


def _validate_policy_record(value: object) -> None:
    record = _strict_payload(
        value,
        name="policy",
        keys=frozenset({"path", "raw_sha256", "policy_id", "canonical_sha256"}),
    )
    path = _external_file(record["path"], label="policy", expected_sha256=record["raw_sha256"])
    try:
        _load_strict_yaml(path, label="policy")
        policy = load_phage_safety_policy(path)
    except (OSError, TypeError, ValueError) as error:
        raise CLIValidationError(f"policy validation failed: {error}") from error
    if record["policy_id"] != policy.policy_id or record["canonical_sha256"] != policy.sha256:
        raise CLIValidationError("policy canonical identity drift")


def _validate_adapter_policy_records(value: object) -> None:
    if not isinstance(value, list):
        raise CLIValidationError("adapter_policies must be a list")
    expected = [
        {
            "safety_class": safety_class,
            "policy_id": policy_id,
            "policy_sha256": policy_sha256,
            "descriptor": _ADAPTER_POLICY_DESCRIPTORS[safety_class],
        }
        for safety_class, (policy_id, policy_sha256) in _ADAPTER_POLICIES.items()
    ]
    if value != expected:
        raise CLIValidationError("adapter policy identities are missing, duplicate, unknown, or mixed")


def _validate_asset_manifest_record(
    value: object,
    *,
    runtime: CLIRuntime,
) -> LoadedSafetyAssetManifest:
    record = _strict_payload(
        value,
        name="safety_asset_manifest",
        keys=frozenset({"path", "sha256", "recipe_path", "recipe_sha256"}),
    )
    path = _external_file(
        record["path"],
        label="safety asset manifest",
        expected_sha256=record["sha256"],
    )
    loaded = runtime.asset_loader(path)
    if (
        loaded.manifest_sha256 != record["sha256"]
        or str(loaded.recipe_path) != record["recipe_path"]
        or loaded.recipe_sha256 != record["recipe_sha256"]
    ):
        raise CLIValidationError("safety asset manifest provenance drift")
    _external_file(
        record["recipe_path"],
        label="safety asset recipe",
        expected_sha256=record["recipe_sha256"],
    )
    return loaded


def _validate_tool_record(
    value: object,
    *,
    expected_tool: str,
    runtime: CLIRuntime,
) -> LoadedToolPin:
    record = _strict_payload(
        value,
        name=f"{expected_tool} tool record",
        keys=frozenset({"tool", "path", "sha256", "version", "version_args", "pin_file_path", "pin_file_sha256"}),
    )
    if record["tool"] != expected_tool:
        raise CLIValidationError(f"{expected_tool} tool record has the wrong tool name")
    _external_file(record["path"], label=f"{expected_tool} tool", expected_sha256=record["sha256"])
    pin_path = _external_file(
        record["pin_file_path"],
        label=f"{expected_tool} tool pin",
        expected_sha256=record["pin_file_sha256"],
    )
    loaded = runtime.tool_pin_loader(
        pin_path,
        expected_tool=expected_tool,
        runner=runtime.command_runner,
        timeout=300.0,
    )
    if (
        str(loaded.pin.path) != record["path"]
        or loaded.pin.sha256 != record["sha256"]
        or loaded.pin.version != record["version"]
        or list(loaded.pin.version_args) != record["version_args"]
        or loaded.pin_file_sha256 != record["pin_file_sha256"]
    ):
        raise CLIValidationError(f"{expected_tool} tool provenance drift")
    return loaded


def _read_fasta_sequence_mapping(path: Path, *, label: str) -> dict[str, str]:
    """Read an ASCII FASTA into a unique ID-to-sequence mapping."""
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError) as error:
        raise CLIValidationError(f"cannot read {label}: {error}") from error
    records: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        if line.startswith(">"):
            sequence_id = line[1:].split(None, 1)[0]
            if not sequence_id or sequence_id in records:
                raise CLIValidationError(f"{label} contains an empty or duplicate FASTA ID")
            records[sequence_id] = []
            current = sequence_id
        elif line.strip():
            if current is None:
                raise CLIValidationError(f"{label} contains sequence before its first header")
            records[current].append("".join(line.split()).upper())
    if any(not sequence for sequence in records.values()):
        raise CLIValidationError(f"{label} contains an empty FASTA record")
    return {sequence_id: "".join(parts) for sequence_id, parts in records.items()}


def _validate_orf_provenance(
    value: object,
    *,
    root: Path,
    record: FastaRecord,
    record_id: str,
    input_index: int,
    circular: bool,
    expected_identity: Mapping[str, object],
    measured: bool,
    orf_replayer: Callable[..., ReplayedORFEvidence],
) -> ValidatedORFProvenance:
    wrapper = _strict_payload(
        value,
        name=f"{record_id} ORF provenance wrapper",
        keys=frozenset(
            {
                "artifact",
                "preparation_state",
                "generation_identity_sha256",
                "query_inventory_sha256",
            }
        ),
    )
    prefix = f"records/{input_index:06d}-{record_id}/"
    path = _validate_owned_artifact(
        wrapper["artifact"],
        root=root,
        label=f"{record_id} ORF provenance",
        required=measured,
        prefix=prefix,
    )
    if path is None:
        if measured:
            raise CLIValidationError("measured adapter evidence lacks ORF provenance")
        if any(wrapper[key] is not None for key in ("generation_identity_sha256", "query_inventory_sha256")):
            raise CLIValidationError("missing ORF provenance carries unverified identities")
        return ValidatedORFProvenance(artifacts=None, query_index={})
    payload = _strict_payload(
        _load_strict_json(path, label=f"{record_id} ORF provenance"),
        name=f"{record_id} ORF provenance",
        keys=frozenset(
            {
                "schema_version",
                "preparation_state",
                "reason_codes",
                "generation_identity",
                "generation_identity_sha256",
                "fallback_policy",
                "fallback_policy_sha256",
                "artifacts",
                "queries",
                "query_inventory_sha256",
            }
        ),
    )
    _validate_json_value(payload, label=f"{record_id} ORF provenance")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise CLIValidationError("ORF provenance schema version mismatch")
    try:
        preparation_state = SafetyState(payload["preparation_state"])
    except (TypeError, ValueError) as error:
        raise CLIValidationError("ORF preparation state is invalid") from error
    _string_list(payload["reason_codes"], label="ORF preparation reason_codes")
    if wrapper["preparation_state"] != preparation_state.value:
        raise CLIValidationError("ORF preparation wrapper state drift")
    if measured and preparation_state is not SafetyState.PASS:
        raise CLIValidationError("measured adapter evidence lacks successful ORF preparation")
    identity = payload["generation_identity"]
    if identity != expected_identity:
        raise CLIValidationError("ORF predictor identity drift")
    identity_sha256 = _canonical_mapping_sha256(identity)
    if (
        payload["generation_identity_sha256"] != identity_sha256
        or wrapper["generation_identity_sha256"] != identity_sha256
    ):
        raise CLIValidationError("ORF predictor identity digest drift")
    if (
        payload["fallback_policy"] != _ORF_FALLBACK_DESCRIPTOR
        or payload["fallback_policy_sha256"] != _ORF_FALLBACK_SHA256
    ):
        raise CLIValidationError("ORF fallback policy drift")
    artifacts_value = payload["artifacts"]
    if not isinstance(artifacts_value, Mapping):
        raise CLIValidationError("ORF artifacts must be a mapping")
    expected_roles = {"genomes_fna", "proteins_faa", "proteins_fna", "proteins_gff", "all_queries_faa"}
    if preparation_state is SafetyState.PASS and set(artifacts_value) != expected_roles:
        raise CLIValidationError("ORF provenance must bind exactly five artifacts")
    if preparation_state is not SafetyState.PASS and artifacts_value != {}:
        raise CLIValidationError("failed ORF preparation must not claim artifacts")
    record_root = root / "records" / f"{input_index:06d}-{record_id}"
    artifact_paths: dict[str, Path] = {}
    for role, artifact_value in artifacts_value.items():
        artifact = _strict_payload(
            artifact_value,
            name=f"{record_id} ORF artifact {role}",
            keys=frozenset({"path", "sha256", "size_bytes"}),
        )
        if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] < 0:
            raise CLIValidationError("ORF artifact size must be a nonnegative integer")
        artifact_path = _owned_path(record_root, artifact["path"], label=f"{record_id} ORF artifact {role}")
        if _sha256_regular_file(artifact_path, label=f"{record_id} ORF artifact {role}") != _require_digest(
            artifact["sha256"], label=f"{record_id} ORF artifact {role} SHA-256"
        ):
            raise CLIValidationError("ORF artifact digest drift")
        if artifact_path.stat().st_size != artifact["size_bytes"]:
            raise CLIValidationError("ORF artifact size drift")
        artifact_paths[role] = artifact_path
    queries = payload["queries"]
    if not isinstance(queries, list):
        raise CLIValidationError("ORF query inventory must be a list")
    if _canonical_mapping_sha256({"queries": queries}) != payload["query_inventory_sha256"]:
        raise CLIValidationError("ORF query inventory digest drift")
    if wrapper["query_inventory_sha256"] != payload["query_inventory_sha256"]:
        raise CLIValidationError("ORF query inventory wrapper drift")
    replayed_queries: dict[str, ORFQueryRecord] = {}
    if preparation_state is SafetyState.PASS:
        try:
            replayed = orf_replayer(record, circular=circular)
        except (CLIValidationError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise CLIValidationError(f"ORF predictor replay failed: {error}") from error
        if set(replayed.artifact_bytes) != expected_roles:
            raise CLIValidationError("ORF predictor replay did not produce exactly five artifacts")
        for role in expected_roles:
            if artifact_paths[role].read_bytes() != replayed.artifact_bytes[role]:
                raise CLIValidationError(f"ORF artifact {role} differs from Task 3 genome replay")
        expected_queries = [_orf_query_provenance_record(query) for query in replayed.query_records]
        if queries != expected_queries:
            raise CLIValidationError("ORF query inventory differs from Task 3 genome replay")
        replayed_queries = {query.query_id: query for query in replayed.query_records}
        if len(replayed_queries) != len(replayed.query_records):
            raise CLIValidationError("ORF predictor replay produced duplicate query IDs")
    query_index: dict[str, Mapping[str, object]] = {}
    for query_value in queries:
        query = _strict_payload(
            query_value,
            name=f"{record_id} ORF query",
            keys=frozenset(
                {
                    "query_id",
                    "sequence_id",
                    "start",
                    "end",
                    "strand",
                    "frame",
                    "evidence_path",
                    "nucleotide_length",
                    "nucleotide_sha256",
                    "protein_length",
                    "protein_sha256",
                }
            ),
        )
        query_id = query["query_id"]
        if not isinstance(query_id, str) or not query_id or query_id in query_index:
            raise CLIValidationError("ORF query IDs must be non-empty and unique")
        if query["sequence_id"] != record_id:
            raise CLIValidationError("ORF query crosses its record boundary")
        for key in ("start", "end", "frame", "nucleotide_length", "protein_length"):
            if type(query[key]) is not int:
                raise CLIValidationError("ORF query coordinates and lengths must be integers")
        if (
            query["start"] < 1
            or query["end"] < query["start"]
            or query["strand"] not in {"+", "-"}
            or query["frame"] not in {-3, -2, -1, 1, 2, 3}
            or query["nucleotide_length"] < 1
            or query["protein_length"] < 1
        ):
            raise CLIValidationError("ORF query coordinates/lengths are invalid")
        if query["evidence_path"] not in {"pyrodigal-gv", "six-frame-fallback"}:
            raise CLIValidationError("ORF query evidence path is invalid")
        _require_digest(query["nucleotide_sha256"], label="ORF query nucleotide SHA-256")
        _require_digest(query["protein_sha256"], label="ORF query protein SHA-256")
        query_index[query_id] = query
    if preparation_state is SafetyState.PASS:
        all_queries = _read_fasta_sequence_mapping(artifact_paths["all_queries_faa"], label="all ORF queries")
        if set(all_queries) != set(query_index):
            raise CLIValidationError("ORF query inventory differs from all_queries.faa")
        for query_id, protein in all_queries.items():
            query = query_index[query_id]
            if (
                len(protein) != query["protein_length"]
                or hashlib.sha256(protein.encode()).hexdigest() != query["protein_sha256"]
            ):
                raise CLIValidationError("ORF query protein provenance drift")
        primary_proteins = _read_fasta_sequence_mapping(artifact_paths["proteins_faa"], label="primary ORF proteins")
        primary_nucleotides = _read_fasta_sequence_mapping(
            artifact_paths["proteins_fna"], label="primary ORF nucleotides"
        )
        if set(primary_proteins) != set(primary_nucleotides):
            raise CLIValidationError("primary ORF protein/nucleotide inventories differ")
        if not set(primary_proteins).issubset(query_index):
            raise CLIValidationError("primary ORF inventory contains unknown queries")
        query_records: list[ORFQueryRecord] = []
        for query_id, query in query_index.items():
            protein = all_queries[query_id]
            replayed_query = replayed_queries[query_id]
            nucleotide = replayed_query.nucleotide
            if query_id in primary_proteins and (
                primary_proteins[query_id] != protein
                or query["evidence_path"] != "pyrodigal-gv"
                or len(nucleotide) != query["nucleotide_length"]
                or hashlib.sha256(nucleotide.encode()).hexdigest() != query["nucleotide_sha256"]
            ):
                raise CLIValidationError("primary ORF query provenance drift")
            query_records.append(
                ORFQueryRecord(
                    query_id=query_id,
                    sequence_id=record_id,
                    start=int(query["start"]),
                    end=int(query["end"]),
                    strand=str(query["strand"]),
                    frame=int(query["frame"]),
                    nucleotide=nucleotide,
                    protein=protein,
                    evidence_path=str(query["evidence_path"]),
                )
            )
        artifacts = ORFArtifacts(
            genomes_fna=artifact_paths["genomes_fna"],
            proteins_faa=artifact_paths["proteins_faa"],
            proteins_fna=artifact_paths["proteins_fna"],
            proteins_gff=artifact_paths["proteins_gff"],
            all_queries_faa=artifact_paths["all_queries_faa"],
            query_records=tuple(query_records),
        )
    else:
        artifacts = None
    return ValidatedORFProvenance(artifacts=artifacts, query_index=query_index)


def _validate_finding(
    value: object,
    *,
    record_id: str,
    safety_class: str,
    expected_policy: tuple[str, str],
    provenance: Mapping[str, object] | None = None,
) -> NormalizedSafetyFinding:
    if not isinstance(value, Mapping):
        raise CLIValidationError("normalized finding must be a mapping")
    _validate_json_value(value, label="normalized finding")
    try:
        finding = NormalizedSafetyFinding.from_dict(value)
    except (TypeError, ValueError) as error:
        raise CLIValidationError(f"normalized finding schema mismatch: {error}") from error
    if finding.sequence_id != record_id or finding.safety_class != safety_class:
        raise CLIValidationError("normalized finding crosses its record or class boundary")
    if (finding.threshold_policy, finding.threshold_policy_sha256) != expected_policy:
        raise CLIValidationError("normalized finding policy identity mismatch")
    for label, digest in (
        ("source", finding.source_sha256),
        ("tool", finding.tool_sha256),
        ("threshold policy", finding.threshold_policy_sha256),
    ):
        _require_digest(digest, label=f"normalized finding {label}")
    if provenance is not None:
        detector_by_accession = provenance.get("detector_by_accession", {})
        if not isinstance(detector_by_accession, Mapping):
            raise CLIValidationError("normalized finding detector provenance is malformed")
        profile_by_accession = provenance.get("profile_by_accession", {})
        if not isinstance(profile_by_accession, Mapping):
            raise CLIValidationError("normalized finding profile provenance is malformed")
        expected_values = {
            "detector": detector_by_accession.get(finding.accession, provenance["detector"]),
            **{
                key: provenance[key]
                for key in (
                    "source_path",
                    "source_sha256",
                    "tool_version",
                    "database_version",
                    "tool_path",
                    "tool_sha256",
                )
            },
        }
        observed_values = {
            "detector": finding.detector,
            "source_path": finding.source_path,
            "source_sha256": finding.source_sha256,
            "tool_version": finding.tool_version,
            "database_version": finding.database_version,
            "tool_path": finding.tool_path,
            "tool_sha256": finding.tool_sha256,
        }
        if observed_values != expected_values:
            if (
                finding.tool_path != provenance["tool_path"]
                or finding.tool_sha256 != provenance["tool_sha256"]
                or finding.tool_version != provenance["tool_version"]
            ):
                raise CLIValidationError("normalized finding tool provenance mismatch")
            raise CLIValidationError("normalized finding source provenance mismatch")
        expected_method = provenance.get("evidence_method")
        if expected_method is not None and finding.evidence_method != expected_method:
            raise CLIValidationError("normalized finding evidence method mismatch")
        expected_profile = profile_by_accession.get(finding.accession, provenance.get("profile"))
        if expected_profile == "ACCESSION":
            if finding.profile != finding.accession:
                raise CLIValidationError("normalized finding profile identity mismatch")
        elif finding.profile != expected_profile:
            raise CLIValidationError("normalized finding profile identity mismatch")
        policy_descriptor = provenance["policy_descriptor"]
        if safety_class == "amr" or _VIRULENCE_REASON in finding.reason_codes:
            expected_thresholds: Mapping[str, object] = {}
        elif finding.state is SafetyState.FAIL:
            expected_thresholds = policy_descriptor["high"]
        elif finding.detector == "diamond-curated-toxin-domain":
            expected_thresholds = policy_descriptor["curated_domain_review"]
        elif "TOXIN_FRAGMENT_REVIEW_HOMOLOGY" in finding.reason_codes:
            expected_thresholds = policy_descriptor["fragment_review"]
        else:
            expected_thresholds = policy_descriptor["review"]
        if dict(finding.thresholds) != dict(expected_thresholds):
            raise CLIValidationError("normalized finding threshold band provenance mismatch")
    return finding


def _validate_class_results(
    value: object,
    *,
    record_id: str,
    applicability: Mapping[str, bool],
    query_index: Mapping[str, Mapping[str, object]] | None = None,
    sequence_length: int | None = None,
    sequence: str | None = None,
    circular: bool = False,
    finding_provenance: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[tuple[SafetyClassResult, ...], set[str]]:
    if not isinstance(value, list) or len(value) != 3:
        raise CLIValidationError("class_results must contain exactly three safety classes")
    results: list[SafetyClassResult] = []
    finding_ids: set[str] = set()
    semantic_findings: set[tuple[object, ...]] = set()
    for expected_class, item in zip(("amr", "toxin", "lysogeny"), value, strict=True):
        row = _strict_payload(
            item,
            name=f"{record_id}.{expected_class} class result",
            keys=frozenset({"safety_class", "state", "required", "findings", "reason_codes"}),
        )
        if row["safety_class"] != expected_class or row["required"] is not applicability[expected_class]:
            raise CLIValidationError("class result applicability or ordering mismatch")
        try:
            state = SafetyState(row["state"])
        except (TypeError, ValueError) as error:
            raise CLIValidationError("class result has an invalid state") from error
        reasons = _string_list(row["reason_codes"], label="class result reason_codes")
        findings_value = row["findings"]
        if not isinstance(findings_value, list):
            raise CLIValidationError("class result findings must be a list")
        findings: list[NormalizedSafetyFinding] = []
        for finding_value in findings_value:
            expected_policy = _ADAPTER_POLICIES[expected_class]
            if (
                expected_class == "toxin"
                and isinstance(finding_value, Mapping)
                and _VIRULENCE_REASON in finding_value.get("reason_codes", ())
            ):
                expected_policy = _ADAPTER_POLICIES["amr"]
            provenance_key = (
                "amr"
                if expected_class == "toxin"
                and isinstance(finding_value, Mapping)
                and _VIRULENCE_REASON in finding_value.get("reason_codes", ())
                else expected_class
            )
            finding = _validate_finding(
                finding_value,
                record_id=record_id,
                safety_class=expected_class,
                expected_policy=expected_policy,
                provenance=(None if finding_provenance is None else finding_provenance[provenance_key]),
            )
            if not finding.finding_id or finding.finding_id in finding_ids:
                raise CLIValidationError("missing, duplicate, or empty normalized finding ID")
            finding_ids.add(finding.finding_id)
            if finding.state is SafetyState.PASS:
                raise CLIValidationError("normalized findings may not claim PASS")
            if (
                finding.start < 1
                or finding.end < finding.start
                or finding.frame not in {-3, -2, -1, 1, 2, 3}
                or (finding.strand == "+" and finding.frame < 0)
                or (finding.strand == "-" and finding.frame > 0)
            ):
                raise CLIValidationError("normalized finding coordinates/frame are invalid")
            if sequence_length is not None and (
                finding.start > sequence_length
                or (not circular and finding.end > sequence_length)
                or finding.end - finding.start + 1 > sequence_length
            ):
                raise CLIValidationError("normalized finding coordinates leave the scanned genome")
            if finding.evidence_path == safety_adapters._AMRFINDER_NUCLEOTIDE_EVIDENCE_PATH:
                if (
                    finding.detector != "amrfinder-plus"
                    or finding.evidence_method not in safety_adapters._AMRFINDER_NUCLEOTIDE_METHODS
                    or sequence is None
                    or sequence_length != len(sequence)
                    or query_index is None
                    or finding.query_id in query_index
                ):
                    raise CLIValidationError("normalized AMRFinder nucleotide evidence provenance mismatch")
                expected_query_id = safety_adapters._amrfinder_nucleotide_query_id(
                    sequence_id=record_id,
                    sequence=sequence,
                    start=finding.start,
                    end=finding.end,
                    strand=finding.strand,
                )
                expected_frame = safety_adapters._amrfinder_nucleotide_frame(
                    start=finding.start,
                    end=finding.end,
                    strand=finding.strand,
                    sequence_length=len(sequence),
                )
                expected_finding_id = f"{finding.safety_class}:{expected_query_id}:{finding.accession}"
                if (
                    finding.query_id != expected_query_id
                    or finding.frame != expected_frame
                    or finding.finding_id != expected_finding_id
                ):
                    raise CLIValidationError("normalized AMRFinder nucleotide evidence provenance mismatch")
            elif query_index is not None:
                query = query_index.get(finding.query_id)
                if query is None:
                    raise CLIValidationError("normalized finding references an unknown ORF query")
                if (
                    finding.start != query["start"]
                    or finding.end != query["end"]
                    or finding.strand != query["strand"]
                    or finding.frame != query["frame"]
                    or finding.evidence_path != query["evidence_path"]
                ):
                    raise CLIValidationError("normalized finding does not match ORF query provenance")
            semantic_key = (
                finding.safety_class,
                finding.detector,
                finding.accession,
                finding.query_id,
                finding.sequence_id,
                finding.start,
                finding.end,
                finding.strand,
                finding.frame,
                finding.evidence_path,
                finding.evidence_method,
                finding.source_sha256,
                finding.tool_sha256,
                finding.threshold_policy_sha256,
            )
            if semantic_key in semantic_findings:
                raise CLIValidationError("semantically duplicate normalized finding")
            semantic_findings.add(semantic_key)
            findings.append(finding)
        if findings:
            finding_state = (
                SafetyState.FAIL
                if any(finding.state is SafetyState.FAIL for finding in findings)
                else SafetyState.INDETERMINATE
            )
            if state is not finding_state:
                raise CLIValidationError("class result violates normalized finding precedence")
        results.append(
            SafetyClassResult(
                safety_class=expected_class,
                state=state,
                required=applicability[expected_class],
                findings=tuple(findings),
                reason_codes=tuple(reasons),
            )
        )
    return tuple(results), finding_ids


def _task3_normalized_output_bytes(raw_path: Path, *, safety_class: str) -> bytes:
    columns = {
        "toxin": safety_adapters._DIAMOND_COLUMNS,
        "lysogeny": safety_adapters._PHROGS_COLUMNS,
    }.get(safety_class)
    if columns is None:
        raise CLIValidationError(f"unsupported normalized output class: {safety_class}")
    with tempfile.TemporaryDirectory(prefix="evo2-phage-normalization-replay-") as temporary:
        normalized = Path(temporary) / "normalized.tsv"
        try:
            safety_adapters._write_normalized_header(normalized, columns, raw_path)
            return normalized.read_bytes()
        except (OSError, UnicodeError) as error:
            raise CLIValidationError(f"cannot replay {safety_class} raw output normalization: {error}") from error


def _validate_adapter_attempts(
    value: object,
    *,
    root: Path,
    record_id: str,
    input_index: int,
    class_results: tuple[SafetyClassResult, ...],
    expected_tool_paths: Mapping[str, str],
    expected_commands: Mapping[str, Sequence[str]] | None = None,
    shared_executions: Mapping[str, Mapping[str, object]] | None = None,
    finding_provenance: Mapping[str, Mapping[str, object]] | None = None,
    orf_artifacts: ORFArtifacts | None = None,
    asset_manifest: Mapping[str, object] | None = None,
    diamond_pin: ToolPin | None = None,
    mmseqs_pin: ToolPin | None = None,
    host_domain: HostDomain | None = None,
    strict_lysis: bool = False,
    adapter_replayer: Callable[..., AdapterResult] | None = None,
    validated_phrogs_assets=None,
) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise CLIValidationError("adapter_attempts must contain exactly three safety classes")
    prefix = f"records/{input_index:06d}-{record_id}/"
    class_by_name = {result.safety_class: result for result in class_results}
    supplemental_ids: set[str] = set()
    for expected_class, item in zip(("amr", "toxin", "lysogeny"), value, strict=True):
        base_keys = frozenset(
            {
                "safety_class",
                "execution_status",
                "state",
                "reason_codes",
                "policy_id",
                "policy_sha256",
                "command_cwd",
                "command",
                "normalized_output",
                "raw_command_output",
                "primary_findings",
                "supplemental_findings",
            }
        )
        item_keys = frozenset(item) if isinstance(item, Mapping) else frozenset()
        attempt_keys = base_keys | ({"shared_execution_id"} if "shared_execution_id" in item_keys else set())
        attempt = _strict_payload(
            item,
            name=f"{record_id}.{expected_class} adapter attempt",
            keys=frozenset(attempt_keys),
        )
        if attempt["safety_class"] != expected_class:
            raise CLIValidationError("adapter attempt ordering or class mismatch")
        if attempt["execution_status"] not in {"NOT_STARTED", "FAILED", "COMPLETED_AND_PARSED"}:
            raise CLIValidationError("adapter execution_status is invalid")
        if attempt["command_cwd"] != "@OUTPUT_ROOT":
            raise CLIValidationError("adapter command cwd is not the canonical output root")
        try:
            state = SafetyState(attempt["state"])
        except (TypeError, ValueError) as error:
            raise CLIValidationError("adapter attempt has an invalid state") from error
        _string_list(attempt["reason_codes"], label="adapter attempt reason_codes")
        if (attempt["policy_id"], attempt["policy_sha256"]) != _ADAPTER_POLICIES[expected_class]:
            raise CLIValidationError("adapter attempt policy identity mismatch")
        command = _string_list(attempt["command"], label="adapter command", unique=False)
        completed = attempt["execution_status"] == "COMPLETED_AND_PARSED"
        measured = state in {SafetyState.PASS, SafetyState.FAIL}
        if measured and not completed:
            raise CLIValidationError("measured adapter result lacks completed and parsed execution")
        if attempt["execution_status"] == "NOT_STARTED" and command:
            raise CLIValidationError("not-started adapter attempt claims a command")
        if attempt["execution_status"] != "NOT_STARTED" and not command:
            raise CLIValidationError("started adapter attempt lacks its command")
        shared_execution_id = attempt.get("shared_execution_id")
        shared_execution = None
        if shared_execution_id is not None:
            if (
                not isinstance(shared_execution_id, str)
                or shared_executions is None
                or not isinstance(shared_executions.get(shared_execution_id), Mapping)
            ):
                raise CLIValidationError("adapter shared execution identity is invalid")
            shared_execution = shared_executions[shared_execution_id]
            if (
                expected_class not in {"amr", "toxin"}
                or shared_execution.get("safety_class") != expected_class
                or record_id not in shared_execution.get("record_ids", ())
                or input_index not in shared_execution.get("record_indices", ())
                or command != list(shared_execution.get("command", ()))
            ):
                raise CLIValidationError("adapter does not match its shared execution")
        elif expected_commands is not None and command and command != list(expected_commands[expected_class]):
            raise CLIValidationError("adapter command does not match the exact Task 3 replay command")
        expected_tool_path = expected_tool_paths.get(expected_class)
        if command and expected_tool_path is not None and command[0] != expected_tool_path:
            raise CLIValidationError("adapter command tool provenance mismatch")
        normalized_path = _validate_owned_artifact(
            attempt["normalized_output"],
            root=root,
            label=f"{record_id}.{expected_class} normalized output",
            required=completed,
            prefix=prefix,
        )
        raw_path = _validate_owned_artifact(
            attempt["raw_command_output"],
            root=root,
            label=f"{record_id}.{expected_class} raw command output",
            required=completed,
            prefix=prefix,
        )
        if completed:
            if expected_class == "amr" and raw_path != normalized_path:
                raise CLIValidationError("AMRFinder raw and normalized output must be the same unambiguous file")
            if expected_class in {"toxin", "lysogeny"} and raw_path == normalized_path:
                raise CLIValidationError("homology adapter raw and normalized outputs must be distinct")
            if expected_class in {"toxin", "lysogeny"} and (
                raw_path is None
                or normalized_path is None
                or normalized_path.read_bytes()
                != _task3_normalized_output_bytes(raw_path, safety_class=expected_class)
            ):
                raise CLIValidationError(
                    f"{expected_class} normalized output is not the exact Task 3 header-plus-raw output"
                )
        primary_value = attempt["primary_findings"]
        if not isinstance(primary_value, list):
            raise CLIValidationError("primary_findings must be a list")
        primary_findings = [
            _validate_finding(
                finding_value,
                record_id=record_id,
                safety_class=expected_class,
                expected_policy=_ADAPTER_POLICIES[expected_class],
                provenance=(None if finding_provenance is None else finding_provenance[expected_class]),
            )
            for finding_value in primary_value
        ]
        supplemental = attempt["supplemental_findings"]
        if not isinstance(supplemental, list):
            raise CLIValidationError("supplemental_findings must be a list")
        if expected_class != "amr" and supplemental:
            raise CLIValidationError("supplemental virulence findings may only come from AMRFinderPlus")
        for finding_value in supplemental:
            finding = _validate_finding(
                finding_value,
                record_id=record_id,
                safety_class="toxin",
                expected_policy=_ADAPTER_POLICIES["amr"],
                provenance=None if finding_provenance is None else finding_provenance["amr"],
            )
            if _VIRULENCE_REASON not in finding.reason_codes:
                raise CLIValidationError("AMRFinder supplemental finding is not normalized virulence evidence")
            if finding.finding_id is not None:
                if finding.finding_id in supplemental_ids:
                    raise CLIValidationError("duplicate supplemental finding ID")
                supplemental_ids.add(finding.finding_id)
        class_result = class_by_name[expected_class]
        expected_primary = [
            finding
            for finding in class_result.findings
            if not (expected_class == "toxin" and _VIRULENCE_REASON in finding.reason_codes)
        ]
        if primary_findings != expected_primary:
            raise CLIValidationError("adapter primary findings differ from the class result")
        if expected_class != "toxin" and (
            state is not class_result.state or attempt["reason_codes"] != list(class_result.reason_codes)
        ):
            raise CLIValidationError("adapter and class result state mismatch")
        if completed and adapter_replayer is not None:
            if (
                normalized_path is None
                or orf_artifacts is None
                or asset_manifest is None
                or diamond_pin is None
                or mmseqs_pin is None
                or host_domain is None
            ):
                raise CLIValidationError("completed adapter result lacks parser replay prerequisites")
            try:
                replayed = adapter_replayer(
                    expected_class,
                    normalized_output=normalized_path,
                    artifacts=orf_artifacts,
                    asset_manifest=asset_manifest,
                    diamond_pin=diamond_pin,
                    mmseqs_pin=mmseqs_pin,
                    host_domain=host_domain,
                    strict_lysis=strict_lysis,
                    required=class_result.required,
                    _validated_phrogs_assets=validated_phrogs_assets,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise CLIValidationError(f"{expected_class} parser replay failed: {error}") from error
            if (
                (replayed.policy_id, replayed.policy_sha256) != (attempt["policy_id"], attempt["policy_sha256"])
                or replayed.class_result.state is not state
                or list(replayed.class_result.reason_codes) != attempt["reason_codes"]
                or [finding.to_dict() for finding in replayed.class_result.findings] != primary_value
                or [finding.to_dict() for finding in replayed.supplemental_findings] != supplemental
                or replayed.raw_output_path != str(normalized_path)
                or replayed.raw_output_sha256 != attempt["normalized_output"]["sha256"]
            ):
                raise CLIValidationError(f"{expected_class} parser replay differs from the declared adapter result")
    toxin = class_by_name["toxin"]
    if supplemental_ids:
        toxin_ids = {finding.finding_id for finding in toxin.findings if finding.finding_id is not None}
        if not supplemental_ids.issubset(toxin_ids) or toxin.state is SafetyState.PASS:
            raise CLIValidationError("supplemental virulence evidence was not merged into the toxin class")
        toxin_attempt = value[1]
        expected_toxin_state = (
            SafetyState.FAIL if toxin_attempt["state"] == SafetyState.FAIL.value else SafetyState.INDETERMINATE
        )
        if toxin.state is not expected_toxin_state:
            raise CLIValidationError("adapter and class result state mismatch")
    elif SafetyState(value[1]["state"]) is not toxin.state:
        raise CLIValidationError("adapter and class result state mismatch")


def _asset_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CLIValidationError(f"{label} must be a non-empty string")
    return value


def _finding_provenance_contracts(
    *,
    assets: LoadedSafetyAssetManifest,
    diamond: LoadedToolPin,
    mmseqs: LoadedToolPin,
) -> dict[str, dict[str, object]]:
    amr = assets.manifest.get("amrfinder_plus")
    toxin = assets.manifest.get("toxin_reference")
    phrogs = assets.manifest.get("phrogs_v4")
    if not isinstance(amr, Mapping) or not isinstance(toxin, Mapping) or not isinstance(phrogs, Mapping):
        raise CLIValidationError("safety asset manifest sections are incomplete")
    toxin_files = toxin.get("files")
    toxin_database = None if not isinstance(toxin_files, Mapping) else toxin_files.get("diamond_database")
    phrogs_database = phrogs.get("profile_database")
    if not isinstance(toxin_database, Mapping) or not isinstance(phrogs_database, Mapping):
        raise CLIValidationError("finding database provenance is incomplete")
    phrogs_search_database = phrogs.get("search_database")
    if phrogs_search_database is not None and not isinstance(phrogs_search_database, Mapping):
        raise CLIValidationError("PHROGs safety search database provenance is invalid")
    phrogs_source_database = phrogs_search_database or phrogs_database
    phrogs_provenance = phrogs_database.get("provenance")
    if isinstance(phrogs_provenance, Mapping):
        phrogs_version = (
            f"{_asset_string(phrogs_provenance.get('dataset_release'), label='PHROGs dataset release')} / "
            f"{_asset_string(phrogs_provenance.get('release'), label='PHROGs profile release')}"
        )
    else:
        phrogs_version = "UNAVAILABLE"
    return {
        "amr": {
            "detector": "amrfinder-plus",
            "source_path": _asset_string(amr.get("database_path"), label="AMRFinder database path"),
            "source_sha256": _require_digest(amr.get("database_sha256"), label="AMRFinder database SHA-256"),
            "tool_version": _asset_string(amr.get("amrfinder_version"), label="AMRFinder version"),
            "database_version": _asset_string(amr.get("database_version"), label="AMRFinder database version"),
            "tool_path": _asset_string(amr.get("binary_path"), label="AMRFinder binary path"),
            "tool_sha256": _require_digest(amr.get("binary_sha256"), label="AMRFinder binary SHA-256"),
            "evidence_method": None,
            "profile": None,
            "policy_descriptor": _ADAPTER_POLICY_DESCRIPTORS["amr"],
        },
        "toxin": {
            "detector": "diamond-reviewed-toxin",
            "detector_by_accession": {
                target.accession: "diamond-curated-toxin-domain"
                for target in safety_adapters._curated_toxin_hazards(toxin).values()
            },
            "profile_by_accession": {
                target.accession: target.accession for target in safety_adapters._curated_toxin_hazards(toxin).values()
            },
            "source_path": _asset_string(toxin_database.get("path"), label="toxin database path"),
            "source_sha256": _require_digest(toxin_database.get("sha256"), label="toxin database SHA-256"),
            "tool_version": diamond.pin.version,
            "database_version": safety_adapters._toxin_database_version(toxin),
            "tool_path": str(diamond.pin.path),
            "tool_sha256": diamond.pin.sha256,
            "evidence_method": "diamond-blastp",
            "profile": None,
            "policy_descriptor": _ADAPTER_POLICY_DESCRIPTORS["toxin"],
        },
        "lysogeny": {
            "detector": "mmseqs-phrogs-v4",
            "source_path": _asset_string(phrogs_source_database.get("path"), label="PHROGs profile path"),
            "source_sha256": _require_digest(phrogs_source_database.get("sha256"), label="PHROGs profile SHA-256"),
            "tool_version": mmseqs.pin.version,
            "database_version": phrogs_version,
            "tool_path": str(mmseqs.pin.path),
            "tool_sha256": mmseqs.pin.sha256,
            "evidence_method": "mmseqs-profile-search",
            "profile": "ACCESSION",
            "policy_descriptor": _ADAPTER_POLICY_DESCRIPTORS["lysogeny"],
        },
    }


def _expected_adapter_commands(
    *,
    record_id: str,
    input_index: int,
    assets: LoadedSafetyAssetManifest,
    diamond: LoadedToolPin,
    mmseqs: LoadedToolPin,
    threads: int,
    phrogs_threads: int | None = None,
) -> dict[str, list[str]]:
    """Rebuild the exact Task 3 argv using stable @OUTPUT_ROOT paths."""
    prefix = Path("@OUTPUT_ROOT") / "records" / f"{input_index:06d}-{record_id}"
    amr_section = _strict_payload(
        assets.manifest["amrfinder_plus"],
        name="AMRFinder asset section",
        keys=frozenset(assets.manifest["amrfinder_plus"]),
    )
    toxin_section = assets.manifest["toxin_reference"]
    phrogs_section = assets.manifest["phrogs_v4"]
    if not isinstance(toxin_section, Mapping) or not isinstance(phrogs_section, Mapping):
        raise CLIValidationError("safety asset manifest sections are incomplete")
    toxin_files = toxin_section.get("files")
    profile_database = phrogs_section.get("search_database") or phrogs_section.get("profile_database")
    if not isinstance(toxin_files, Mapping) or not isinstance(toxin_files.get("diamond_database"), Mapping):
        raise CLIValidationError("toxin database provenance is incomplete")
    if not isinstance(profile_database, Mapping):
        raise CLIValidationError("PHROGs profile database provenance is incomplete")
    amrfinder = _asset_string(amr_section.get("binary_path"), label="AMRFinder binary path")
    amr_database = _asset_string(amr_section.get("database_path"), label="AMRFinder database path")
    blastx = _asset_string(amr_section.get("blastx_path"), label="AMRFinder BLASTX path")
    hmmsearch = _asset_string(amr_section.get("hmmsearch_path"), label="AMRFinder HMM search path")
    toxin_database = _asset_string(
        toxin_files["diamond_database"].get("path"),
        label="toxin DIAMOND database path",
    )
    phrogs_database = _asset_string(
        profile_database.get("path"),
        label="PHROGs profile database path",
    )
    return {
        "amr": build_amrfinder_command(
            amrfinder=Path(amrfinder),
            genomes_fna=prefix / "genomes.fna",
            proteins_faa=prefix / "proteins.faa",
            proteins_gff=prefix / "proteins.gff",
            database_dir=Path(amr_database),
            blast_bin_dir=Path(blastx).parent,
            hmmer_bin_dir=Path(hmmsearch).parent,
            threads=threads,
            output_tsv=prefix / "amrfinder.tsv",
        ),
        "toxin": build_diamond_command(
            diamond=diamond.pin.path,
            queries_faa=prefix / "all_queries.faa",
            database=Path(toxin_database),
            output_tsv=prefix / "toxin_diamond.raw.tsv",
            threads=threads,
        ),
        "lysogeny": build_phrogs_command(
            mmseqs=mmseqs.pin.path,
            profile_database=Path(phrogs_database),
            proteins_faa=prefix / "proteins.faa",
            output_tsv=prefix / "phrogs.raw.tsv",
            temporary_dir=prefix / "tmp",
            threads=threads if phrogs_threads is None else phrogs_threads,
        ),
    }


def _validate_scan_manifest_mapping(
    manifest: Mapping[str, object],
    *,
    root: Path,
    runtime: CLIRuntime,
) -> SafetyState:
    legacy_top = frozenset(
        {
            "schema_version",
            "manifest_type",
            "created_at",
            "completed_at",
            "cli_identity",
            "policy",
            "adapter_policies",
            "safety_asset_manifest",
            "input_fasta",
            "host_evidence_input",
            "resolved_profile",
            "runtime_parameters",
            "tools",
            "environment",
            "records",
            "aggregate",
            "derivatives",
            "claim_boundary",
        }
    )
    actual_top = frozenset(manifest) if isinstance(manifest, Mapping) else frozenset()
    supported_tops = {legacy_top, legacy_top | {"shared_executions"}}
    if actual_top not in supported_tops:
        raise CLIValidationError("scan manifest top-level keys do not match a supported schema")
    strict = _strict_payload(manifest, name="scan manifest", keys=actual_top)
    _validate_json_value(strict, label="scan manifest")
    if (
        type(strict["schema_version"]) is not int
        or strict["schema_version"] != MANIFEST_SCHEMA_VERSION
        or strict["manifest_type"] != "sequence_safety_scan"
    ):
        raise CLIValidationError("unsupported scan manifest type or schema version")
    created = _manifest_timestamp(strict["created_at"], label="created_at")
    completed = _manifest_timestamp(strict["completed_at"], label="completed_at")
    if completed < created:
        raise CLIValidationError("scan manifest completed_at precedes created_at")
    _validate_cli_identity(strict["cli_identity"])
    _validate_policy_record(strict["policy"])
    _validate_adapter_policy_records(strict["adapter_policies"])
    assets = _validate_asset_manifest_record(strict["safety_asset_manifest"], runtime=runtime)
    phrogs_section = assets.manifest.get("phrogs_v4")
    if not isinstance(phrogs_section, Mapping):
        raise CLIValidationError("validated safety assets lack the PHROGs section")
    try:
        validated_phrogs_assets = runtime.phrogs_asset_validator(phrogs_section)
    except (AssetProvenanceError, KeyError, OSError, TypeError, ValueError) as error:
        raise CLIValidationError("PHROGs assets failed transaction validation") from error

    input_record = _strict_payload(
        strict["input_fasta"],
        name="input_fasta",
        keys=frozenset({"path", "sha256", "count"}),
    )
    input_path = _external_file(
        input_record["path"],
        label="input FASTA",
        expected_sha256=input_record["sha256"],
    )
    if type(input_record["count"]) is not int or input_record["count"] < 1:
        raise CLIValidationError("input FASTA count must be a positive integer")
    input_records = parse_fasta_records(input_path)
    if len(input_records) != input_record["count"]:
        raise CLIValidationError("input FASTA count drift")

    evidence_input = _strict_payload(
        strict["host_evidence_input"],
        name="host_evidence_input",
        keys=frozenset({"kind", "canonical_sha256", "value"}),
    )
    if evidence_input["kind"] != "inline_json" or not isinstance(evidence_input["value"], Mapping):
        raise CLIValidationError("host evidence input schema mismatch")
    canonical_evidence = json.dumps(
        evidence_input["value"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if hashlib.sha256(canonical_evidence).hexdigest() != evidence_input["canonical_sha256"]:
        raise CLIValidationError("host evidence digest drift")
    evidence = _parse_host_evidence_json(json.dumps(evidence_input["value"], allow_nan=False))
    if not evaluate_host_evidence(evidence).allowed:
        raise CLIValidationError("host evidence is no longer eligible")
    resolved = _strict_payload(
        strict["resolved_profile"],
        name="resolved_profile",
        keys=frozenset({"host_domain", "strict_lysis", "applicability"}),
    )
    if type(resolved["strict_lysis"]) is not bool:
        raise CLIValidationError("resolved strict_lysis must be a boolean")
    try:
        host_domain = HostDomain(resolved["host_domain"])
    except (TypeError, ValueError) as error:
        raise CLIValidationError("resolved host profile is invalid") from error
    _normalize_requested_host_domain(host_domain, evidence)
    applicability = _applicability(host_domain, strict_lysis=resolved["strict_lysis"])
    if resolved["applicability"] != applicability:
        raise CLIValidationError("class applicability drift")
    runtime_parameters = strict["runtime_parameters"]
    if not isinstance(runtime_parameters, Mapping):
        raise CLIValidationError("runtime_parameters must be a mapping")
    legacy_runtime_keys = frozenset({"threads", "timeout_seconds", "circular", "minimum_fallback_amino_acids"})
    parallel_runtime_keys = legacy_runtime_keys | frozenset(
        {"record_workers", "maximum_concurrent_tool_threads", "available_cpu_slots"}
    )
    batch_runtime_keys = legacy_runtime_keys | frozenset(
        {
            "batch_size",
            "batch_workers",
            "phrogs_threads",
            "phrogs_workers",
            "maximum_concurrent_tool_threads",
            "available_cpu_slots",
        }
    )
    if frozenset(runtime_parameters) not in {legacy_runtime_keys, parallel_runtime_keys, batch_runtime_keys}:
        raise CLIValidationError("runtime_parameters keys do not match a supported schema")
    if type(runtime_parameters["threads"]) is not int or runtime_parameters["threads"] < 1:
        raise CLIValidationError("runtime threads must be a positive integer")
    if (
        type(runtime_parameters["timeout_seconds"]) not in {int, float}
        or not math.isfinite(runtime_parameters["timeout_seconds"])
        or runtime_parameters["timeout_seconds"] <= 0
    ):
        raise CLIValidationError("runtime timeout must be positive and finite")
    if (
        type(runtime_parameters["circular"]) is not bool
        or type(runtime_parameters["minimum_fallback_amino_acids"]) is not int
        or runtime_parameters["minimum_fallback_amino_acids"] != 8
    ):
        raise CLIValidationError("runtime sequence geometry/fallback contract mismatch")
    if "record_workers" in runtime_parameters:
        if (
            type(runtime_parameters["record_workers"]) is not int
            or runtime_parameters["record_workers"] < 1
            or type(runtime_parameters["maximum_concurrent_tool_threads"]) is not int
            or runtime_parameters["maximum_concurrent_tool_threads"]
            != runtime_parameters["record_workers"] * runtime_parameters["threads"]
            or type(runtime_parameters["available_cpu_slots"]) is not int
            or runtime_parameters["available_cpu_slots"] < runtime_parameters["maximum_concurrent_tool_threads"]
        ):
            raise CLIValidationError("runtime record-worker resource contract mismatch")
    if "batch_size" in runtime_parameters:
        batch_size = runtime_parameters["batch_size"]
        batch_workers = runtime_parameters["batch_workers"]
        phrogs_threads = runtime_parameters["phrogs_threads"]
        phrogs_workers = runtime_parameters["phrogs_workers"]
        available_slots = runtime_parameters["available_cpu_slots"]
        maximum_threads = runtime_parameters["maximum_concurrent_tool_threads"]
        if (
            type(batch_size) is not int
            or batch_size < 2
            or type(batch_workers) is not int
            or batch_workers < 1
            or batch_workers > math.ceil(len(input_records) / batch_size)
            or type(phrogs_threads) is not int
            or phrogs_threads < 1
            or type(phrogs_workers) is not int
            or phrogs_workers < 1
            or phrogs_workers > batch_size
            or type(available_slots) is not int
            or type(maximum_threads) is not int
            or maximum_threads
            != max(batch_workers * runtime_parameters["threads"], batch_workers * phrogs_workers * phrogs_threads)
            or available_slots < maximum_threads
        ):
            raise CLIValidationError("runtime batch resource contract mismatch")

    tools = _strict_payload(
        strict["tools"],
        name="tools",
        keys=frozenset({"diamond", "mmseqs", "orf_predictor"}),
    )
    diamond_tool = _validate_tool_record(tools["diamond"], expected_tool="diamond", runtime=runtime)
    mmseqs_tool = _validate_tool_record(tools["mmseqs"], expected_tool="mmseqs", runtime=runtime)
    finding_provenance = _finding_provenance_contracts(
        assets=assets,
        diamond=diamond_tool,
        mmseqs=mmseqs_tool,
    )
    expected_orf_identity = dict(runtime.orf_identity_collector())
    if tools["orf_predictor"] != {
        "identity": expected_orf_identity,
        "identity_sha256": _canonical_mapping_sha256(expected_orf_identity),
    }:
        raise CLIValidationError("ORF predictor provenance boundary mismatch")
    if not isinstance(strict["environment"], Mapping):
        raise CLIValidationError("environment provenance must be a mapping")
    shared_executions = _validate_shared_executions(
        strict.get("shared_executions", []),
        root=root,
        input_records=input_records,
        assets=assets,
        diamond=diamond_tool,
        threads=runtime_parameters["threads"],
        batch_size=runtime_parameters.get("batch_size", 1),
    )
    if "batch_size" not in runtime_parameters and shared_executions:
        raise CLIValidationError("legacy runtime cannot claim shared executions")

    rows = strict["records"]
    if not isinstance(rows, list) or len(rows) != len(input_records):
        raise CLIValidationError("record inventory does not match input FASTA")
    seen_ids: set[str] = set()
    record_states: list[SafetyState] = []
    record_reasons: list[str] = []
    record_orf_artifacts: dict[str, ORFArtifacts] = {}
    record_indices: dict[str, int] = {}
    for input_index, (row_value, fasta_record) in enumerate(zip(rows, input_records, strict=True)):
        row = _strict_payload(
            row_value,
            name=f"record[{input_index}]",
            keys=frozenset(
                {
                    "record_id",
                    "input_index",
                    "sequence_sha256",
                    "original_record_sha256",
                    "sequence_length",
                    "circular",
                    "host_evidence",
                    "host_evidence_sha256",
                    "resolved_host_profile",
                    "strict_lysis",
                    "applicability",
                    "orf_provenance",
                    "state",
                    "reason_codes",
                    "class_results",
                    "adapter_attempts",
                }
            ),
        )
        if type(row["input_index"]) is not int:
            raise CLIValidationError("record input_index must be an integer")
        if row["record_id"] != fasta_record.sequence_id or row["input_index"] != input_index:
            raise CLIValidationError("record ID/order drift")
        if row["record_id"] in seen_ids:
            raise CLIValidationError("duplicate record ID in manifest")
        seen_ids.add(row["record_id"])
        if row["sequence_sha256"] != hashlib.sha256(fasta_record.normalized_sequence.encode()).hexdigest():
            raise CLIValidationError("normalized sequence digest drift")
        if row["original_record_sha256"] != hashlib.sha256(fasta_record.original_bytes).hexdigest():
            raise CLIValidationError("original FASTA record digest drift")
        if (
            type(row["sequence_length"]) is not int
            or type(row["circular"]) is not bool
            or row["sequence_length"] != len(fasta_record.normalized_sequence)
            or row["circular"] is not runtime_parameters["circular"]
        ):
            raise CLIValidationError("record length or circularity provenance mismatch")
        if (
            row["host_evidence"] != evidence_input["value"]
            or row["resolved_host_profile"] != host_domain.value
            or row["strict_lysis"] is not resolved["strict_lysis"]
            or row["applicability"] != applicability
        ):
            raise CLIValidationError("per-record host provenance mismatch")
        evidence_sha256 = hashlib.sha256(
            json.dumps(row["host_evidence"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if row["host_evidence_sha256"] != evidence_sha256:
            raise CLIValidationError("per-record host evidence digest drift")
        attempts_value = row["adapter_attempts"]
        measured = isinstance(attempts_value, list) and any(
            isinstance(attempt, Mapping) and attempt.get("state") in {SafetyState.PASS.value, SafetyState.FAIL.value}
            for attempt in attempts_value
        )
        orf_provenance = _validate_orf_provenance(
            row["orf_provenance"],
            root=root,
            record=fasta_record,
            record_id=fasta_record.sequence_id,
            input_index=input_index,
            circular=runtime_parameters["circular"],
            expected_identity=expected_orf_identity,
            measured=measured,
            orf_replayer=runtime.orf_replayer,
        )
        record_indices[fasta_record.sequence_id] = input_index
        if orf_provenance.artifacts is not None:
            record_orf_artifacts[fasta_record.sequence_id] = orf_provenance.artifacts
        class_results, _ = _validate_class_results(
            row["class_results"],
            record_id=fasta_record.sequence_id,
            applicability=applicability,
            query_index=orf_provenance.query_index,
            sequence_length=len(fasta_record.normalized_sequence),
            sequence=fasta_record.normalized_sequence,
            circular=runtime_parameters["circular"],
            finding_provenance=finding_provenance,
        )
        state = aggregate_safety_state(class_results)
        try:
            declared_state = SafetyState(row["state"])
        except (TypeError, ValueError) as error:
            raise CLIValidationError("record has an invalid aggregate state") from error
        if declared_state is not state:
            raise CLIValidationError("record aggregate state drift")
        reasons = list(dict.fromkeys(reason for result in class_results for reason in result.reason_codes))
        if row["reason_codes"] != reasons:
            raise CLIValidationError("record aggregate reason-code drift")
        _validate_adapter_attempts(
            row["adapter_attempts"],
            root=root,
            record_id=fasta_record.sequence_id,
            input_index=input_index,
            class_results=class_results,
            expected_tool_paths={
                **(
                    {"amr": str(assets.manifest["amrfinder_plus"]["binary_path"])}
                    if isinstance(assets.manifest.get("amrfinder_plus"), Mapping)
                    and isinstance(assets.manifest["amrfinder_plus"].get("binary_path"), str)
                    else {}
                ),
                "toxin": str(diamond_tool.pin.path),
                "lysogeny": str(mmseqs_tool.pin.path),
            },
            expected_commands=_expected_adapter_commands(
                record_id=fasta_record.sequence_id,
                input_index=input_index,
                assets=assets,
                diamond=diamond_tool,
                mmseqs=mmseqs_tool,
                threads=runtime_parameters["threads"],
                phrogs_threads=runtime_parameters.get("phrogs_threads"),
            ),
            shared_executions=shared_executions,
            finding_provenance=finding_provenance,
            orf_artifacts=orf_provenance.artifacts,
            asset_manifest=assets.manifest,
            diamond_pin=diamond_tool.pin,
            mmseqs_pin=mmseqs_tool.pin,
            host_domain=host_domain,
            strict_lysis=resolved["strict_lysis"],
            adapter_replayer=runtime.adapter_replayer,
            validated_phrogs_assets=validated_phrogs_assets,
        )
        record_states.append(state)
        record_reasons.extend(reasons)

    _validate_shared_execution_record_bindings(
        shared_executions,
        root=root,
        record_artifacts=record_orf_artifacts,
        record_index=record_indices,
    )
    try:
        runtime.phrogs_asset_revalidator(phrogs_section, validated_phrogs_assets)
    except (AssetProvenanceError, KeyError, OSError, TypeError, ValueError) as error:
        raise CLIValidationError("PHROGs assets changed during manifest replay") from error

    if SafetyState.FAIL in record_states:
        batch_state = SafetyState.FAIL
    elif SafetyState.INDETERMINATE in record_states:
        batch_state = SafetyState.INDETERMINATE
    else:
        batch_state = SafetyState.PASS
    aggregate = _strict_payload(
        strict["aggregate"],
        name="aggregate",
        keys=frozenset({"state", "reason_codes", "counts"}),
    )
    counts = aggregate["counts"]
    if not isinstance(counts, Mapping) or set(counts) != {state.value for state in SafetyState}:
        raise CLIValidationError("aggregate counts schema mismatch")
    if any(type(value) is not int for value in counts.values()):
        raise CLIValidationError("aggregate counts must be integers")
    expected_counts = {state.value: record_states.count(state) for state in SafetyState}
    if counts != expected_counts or aggregate["state"] != batch_state.value:
        raise CLIValidationError("batch aggregate state/count drift")
    if aggregate["reason_codes"] != list(dict.fromkeys(record_reasons)):
        raise CLIValidationError("batch aggregate reason-code drift")
    if strict["derivatives"] != {}:
        raise CLIValidationError("scan manifests must not claim derivative FASTAs")
    if strict["claim_boundary"] != _CLAIM_BOUNDARY:
        raise CLIValidationError("claim boundary drift")
    return batch_state


def _validate_diagnostic_manifest_mapping(
    manifest: Mapping[str, object],
    *,
    runtime: CLIRuntime,
) -> SafetyState:
    """Validate an explicitly non-filterable missing-prerequisite diagnostic."""
    strict = _strict_payload(
        manifest,
        name="diagnostic manifest",
        keys=frozenset(
            {
                "schema_version",
                "manifest_type",
                "created_at",
                "completed_at",
                "cli_identity",
                "policy",
                "input_fasta",
                "safety_asset_manifest_input",
                "host_evidence_input",
                "resolved_profile",
                "missing_prerequisites",
                "records",
                "aggregate",
                "claim_boundary",
            }
        ),
    )
    _validate_json_value(strict, label="diagnostic manifest")
    if type(strict["schema_version"]) is not int or strict["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise CLIValidationError("unsupported diagnostic manifest schema version")
    if strict["manifest_type"] != "sequence_safety_diagnostic":
        raise CLIValidationError("unsupported diagnostic manifest type")
    created = _manifest_timestamp(strict["created_at"], label="created_at")
    completed = _manifest_timestamp(strict["completed_at"], label="completed_at")
    if completed < created:
        raise CLIValidationError("diagnostic completed_at precedes created_at")
    _validate_cli_identity(strict["cli_identity"])
    _validate_policy_record(strict["policy"])
    input_record = _strict_payload(
        strict["input_fasta"],
        name="diagnostic input_fasta",
        keys=frozenset({"path", "sha256", "count"}),
    )
    input_path = _external_file(
        input_record["path"],
        label="diagnostic input FASTA",
        expected_sha256=input_record["sha256"],
    )
    if type(input_record["count"]) is not int or input_record["count"] < 1:
        raise CLIValidationError("diagnostic input count must be a positive integer")
    input_records = parse_fasta_records(input_path)
    if len(input_records) != input_record["count"]:
        raise CLIValidationError("diagnostic input count drift")
    asset_input = _strict_payload(
        strict["safety_asset_manifest_input"],
        name="diagnostic safety asset input",
        keys=frozenset({"path", "sha256", "validation_status"}),
    )
    _external_file(
        asset_input["path"],
        label="diagnostic safety asset input",
        expected_sha256=asset_input["sha256"],
    )
    if asset_input["validation_status"] != "NOT_VALIDATED_DUE_TO_MISSING_TOOL_PINS":
        raise CLIValidationError("diagnostic safety asset validation boundary drift")
    evidence_input = _strict_payload(
        strict["host_evidence_input"],
        name="diagnostic host evidence",
        keys=frozenset({"kind", "canonical_sha256", "value"}),
    )
    if evidence_input["kind"] != "inline_json" or not isinstance(evidence_input["value"], Mapping):
        raise CLIValidationError("diagnostic host evidence schema mismatch")
    canonical_evidence = json.dumps(
        evidence_input["value"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if hashlib.sha256(canonical_evidence).hexdigest() != evidence_input["canonical_sha256"]:
        raise CLIValidationError("diagnostic host evidence digest drift")
    evidence = _parse_host_evidence_json(json.dumps(evidence_input["value"], allow_nan=False))
    if not evaluate_host_evidence(evidence).allowed:
        raise CLIValidationError("diagnostic host evidence is no longer eligible")
    resolved = _strict_payload(
        strict["resolved_profile"],
        name="diagnostic resolved profile",
        keys=frozenset({"host_domain", "strict_lysis"}),
    )
    if type(resolved["strict_lysis"]) is not bool:
        raise CLIValidationError("diagnostic strict_lysis must be a boolean")
    try:
        host_domain = HostDomain(resolved["host_domain"])
    except (TypeError, ValueError) as error:
        raise CLIValidationError("diagnostic host profile is invalid") from error
    _normalize_requested_host_domain(host_domain, evidence)
    prerequisites = _string_list(
        strict["missing_prerequisites"],
        label="diagnostic missing_prerequisites",
    )
    allowed_prerequisites = {
        "MISSING_TRUSTED_DIAMOND_TOOL_PIN",
        "MISSING_TRUSTED_MMSEQS_TOOL_PIN",
    }
    if not prerequisites or set(prerequisites) - allowed_prerequisites:
        raise CLIValidationError("diagnostic missing prerequisites are invalid")
    rows = strict["records"]
    if not isinstance(rows, list) or len(rows) != len(input_records):
        raise CLIValidationError("diagnostic record inventory drift")
    for index, (row_value, fasta_record) in enumerate(zip(rows, input_records, strict=True)):
        row = _strict_payload(
            row_value,
            name=f"diagnostic record[{index}]",
            keys=frozenset({"record_id", "input_index", "state", "reason_codes"}),
        )
        if (
            row["record_id"] != fasta_record.sequence_id
            or type(row["input_index"]) is not int
            or row["input_index"] != index
            or row["state"] != SafetyState.INDETERMINATE.value
            or row["reason_codes"] != prerequisites
        ):
            raise CLIValidationError("diagnostic record state/order drift")
    aggregate = _strict_payload(
        strict["aggregate"],
        name="diagnostic aggregate",
        keys=frozenset({"state", "reason_codes", "counts"}),
    )
    expected_counts = {
        SafetyState.PASS.value: 0,
        SafetyState.FAIL.value: 0,
        SafetyState.INDETERMINATE.value: len(input_records),
    }
    if (
        aggregate["state"] != SafetyState.INDETERMINATE.value
        or aggregate["reason_codes"] != prerequisites
        or aggregate["counts"] != expected_counts
        or any(type(value) is not int for value in aggregate["counts"].values())
    ):
        raise CLIValidationError("diagnostic aggregate drift")
    if strict["claim_boundary"] != _CLAIM_BOUNDARY:
        raise CLIValidationError("diagnostic claim boundary drift")
    return SafetyState.INDETERMINATE


def _validate_filter_manifest_mapping(
    manifest: Mapping[str, object],
    *,
    root: Path,
    runtime: CLIRuntime,
) -> SafetyState:
    legacy_top = frozenset(
        {
            "schema_version",
            "manifest_type",
            "created_at",
            "completed_at",
            "cli_identity",
            "source_scan_manifest",
            "policy",
            "adapter_policies",
            "safety_asset_manifest",
            "input_fasta",
            "host_evidence_input",
            "resolved_profile",
            "runtime_parameters",
            "tools",
            "environment",
            "records",
            "aggregate",
            "derivatives",
            "eligibility",
            "claim_boundary",
        }
    )
    actual_top = frozenset(manifest) if isinstance(manifest, Mapping) else frozenset()
    if actual_top not in {legacy_top, legacy_top | {"shared_executions"}}:
        raise CLIValidationError("filter manifest top-level keys do not match a supported schema")
    strict = _strict_payload(manifest, name="filter manifest", keys=actual_top)
    _validate_json_value(strict, label="filter manifest")
    if (
        type(strict["schema_version"]) is not int
        or strict["schema_version"] != MANIFEST_SCHEMA_VERSION
        or strict["manifest_type"] != "sequence_safety_filter"
    ):
        raise CLIValidationError("unsupported filter manifest type or schema version")
    created = _manifest_timestamp(strict["created_at"], label="created_at")
    completed = _manifest_timestamp(strict["completed_at"], label="completed_at")
    if completed < created:
        raise CLIValidationError("filter manifest completed_at precedes created_at")
    _validate_cli_identity(strict["cli_identity"])
    source_record = _strict_payload(
        strict["source_scan_manifest"],
        name="source_scan_manifest",
        keys=frozenset({"path", "sha256"}),
    )
    source_path = _external_file(
        source_record["path"],
        label="source scan manifest",
        expected_sha256=source_record["sha256"],
    )
    source = validate_manifest_file(
        source_path,
        runtime=runtime,
        expected_type="sequence_safety_scan",
    )
    for key in (
        "policy",
        "adapter_policies",
        "safety_asset_manifest",
        "input_fasta",
        "host_evidence_input",
        "resolved_profile",
        "runtime_parameters",
        "tools",
        "records",
        "aggregate",
        "claim_boundary",
    ):
        if strict[key] != source[key]:
            raise CLIValidationError(f"filter manifest {key} drift from source scan")
    if strict.get("shared_executions") != source.get("shared_executions"):
        raise CLIValidationError("filter manifest shared execution drift from source scan")
    if not isinstance(strict["environment"], Mapping):
        raise CLIValidationError("filter environment provenance must be a mapping")
    expected_eligibility = {
        "sft": "PASS_ONLY",
        "positive_rl_references": "PASS_ONLY",
        "calibration_winners": "PASS_ONLY",
        "final_promotion": "PASS_ONLY",
    }
    if strict["eligibility"] != expected_eligibility:
        raise CLIValidationError("filter eligibility contract drift")

    input_path = Path(source["input_fasta"]["path"])
    input_records = parse_fasta_records(input_path)
    state_by_id = {row["record_id"]: SafetyState(row["state"]) for row in source["records"]}
    expected_payloads = partition_fasta_records(input_records, state_by_id)
    derivatives = _strict_payload(
        strict["derivatives"],
        name="derivatives",
        keys=frozenset({"pass", "fail", "indeterminate"}),
    )
    all_ids: list[str] = []
    for state, key, filename in (
        (SafetyState.PASS, "pass", "pass.fna"),
        (SafetyState.FAIL, "fail", "fail.fna"),
        (SafetyState.INDETERMINATE, "indeterminate", "indeterminate.fna"),
    ):
        derivative = _strict_payload(
            derivatives[key],
            name=f"{key} derivative",
            keys=frozenset({"role", "path", "sha256", "count", "record_ids", "owned"}),
        )
        expected_ids = [record.sequence_id for record in input_records if state_by_id[record.sequence_id] is state]
        if type(derivative["count"]) is not int:
            raise CLIValidationError("derivative count must be an integer")
        if (
            derivative["role"] != state.value
            or derivative["path"] != filename
            or derivative["owned"] is not True
            or derivative["record_ids"] != expected_ids
            or derivative["count"] != len(expected_ids)
        ):
            raise CLIValidationError(f"{key} derivative partition metadata drift")
        path = _validate_owned_artifact(
            {"path": derivative["path"], "sha256": derivative["sha256"], "owned": derivative["owned"]},
            root=root,
            label=f"{key} derivative",
            required=True,
        )
        if path is None or path.read_bytes() != expected_payloads[state]:
            raise CLIValidationError(f"{key} derivative bytes drift from the source FASTA")
        all_ids.extend(expected_ids)
    expected_all_ids = [record.sequence_id for record in input_records]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(expected_all_ids):
        raise CLIValidationError("derivative partitions are not disjoint and complete")
    return SafetyState(source["aggregate"]["state"])


def validate_manifest_file(
    path: Path,
    *,
    runtime: CLIRuntime | None = None,
    expected_type: str | None = None,
) -> Mapping[str, object]:
    """Strictly revalidate every external identity and owned output in a scan/filter manifest."""
    selected_runtime = CLIRuntime() if runtime is None else runtime
    manifest_path = Path(path).absolute()
    _reject_symlink_components(manifest_path, label="manifest")
    payload = _load_strict_json(manifest_path, label="manifest")
    if not isinstance(payload, Mapping):
        raise CLIValidationError("manifest must be a mapping")
    manifest_type = payload.get("manifest_type")
    if expected_type is not None and manifest_type != expected_type:
        raise CLIValidationError(f"expected {expected_type} manifest")
    if manifest_type == "sequence_safety_scan":
        _validate_scan_manifest_mapping(payload, root=manifest_path.parent, runtime=selected_runtime)
    elif manifest_type == "sequence_safety_filter":
        _validate_filter_manifest_mapping(payload, root=manifest_path.parent, runtime=selected_runtime)
    elif manifest_type == "sequence_safety_diagnostic":
        _validate_diagnostic_manifest_mapping(payload, runtime=selected_runtime)
    else:
        raise CLIValidationError("unsupported manifest type")
    return payload


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _state_exit_code(state: SafetyState) -> int:
    if state is SafetyState.FAIL:
        return 2
    if state is SafetyState.INDETERMINATE:
        return 3
    return 0


def _publish_filter_generation(args: argparse.Namespace, runtime: CLIRuntime) -> SafetyState:
    output_dir = Path(args.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise CLIValidationError("output directory must not already exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    published = False
    try:
        created_at = _timestamp(runtime.clock())
        scan_path = Path(args.scan_manifest).absolute()
        initial_scan_sha256 = _sha256_regular_file(scan_path, label="source scan manifest")
        source = validate_manifest_file(
            scan_path,
            runtime=runtime,
            expected_type="sequence_safety_scan",
        )
        input_path = Path(args.input_fasta).absolute()
        if str(input_path) != source["input_fasta"]["path"]:
            raise CLIValidationError("filter input FASTA path differs from the trusted scan input")
        initial_input_sha256 = _sha256_regular_file(input_path, label="filter input FASTA")
        if initial_input_sha256 != source["input_fasta"]["sha256"]:
            raise CLIValidationError("filter input FASTA digest drift")
        input_records = parse_fasta_records(input_path)
        state_by_id = {row["record_id"]: SafetyState(row["state"]) for row in source["records"]}
        partitions = partition_fasta_records(input_records, state_by_id)
        derivative_records: dict[str, object] = {}
        for state, key, filename in (
            (SafetyState.PASS, "pass", "pass.fna"),
            (SafetyState.FAIL, "fail", "fail.fna"),
            (SafetyState.INDETERMINATE, "indeterminate", "indeterminate.fna"),
        ):
            derivative_path = staging / filename
            _write_bytes_fsync(derivative_path, partitions[state])
            record_ids = [record.sequence_id for record in input_records if state_by_id[record.sequence_id] is state]
            derivative_records[key] = {
                "role": state.value,
                "path": filename,
                "sha256": _sha256_regular_file(derivative_path, label=f"{key} derivative"),
                "count": len(record_ids),
                "record_ids": record_ids,
                "owned": True,
            }
        environment = dict(runtime.environment_collector())
        if _sha256_regular_file(scan_path, label="source scan manifest") != initial_scan_sha256:
            raise CLIValidationError("source scan manifest changed during filtering")
        if _sha256_regular_file(input_path, label="filter input FASTA") != initial_input_sha256:
            raise CLIValidationError("input FASTA changed during filtering")
        completed_at = _timestamp(runtime.clock())
        source_path = Path(__file__).resolve()
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "manifest_type": "sequence_safety_filter",
            "created_at": created_at,
            "completed_at": completed_at,
            "cli_identity": {
                "name": CLI_ID,
                "version": CLI_VERSION,
                "entry_point": "bionemo.evo2_phage_gen.sequence_safety_cli:main",
                "source_path": str(source_path),
                "source_sha256": _sha256_file(source_path),
            },
            "source_scan_manifest": {
                "path": str(scan_path),
                "sha256": initial_scan_sha256,
            },
            "policy": source["policy"],
            "adapter_policies": source["adapter_policies"],
            "safety_asset_manifest": source["safety_asset_manifest"],
            "input_fasta": source["input_fasta"],
            "host_evidence_input": source["host_evidence_input"],
            "resolved_profile": source["resolved_profile"],
            "runtime_parameters": source["runtime_parameters"],
            "tools": source["tools"],
            "environment": environment,
            "records": source["records"],
            **({"shared_executions": source["shared_executions"]} if "shared_executions" in source else {}),
            "aggregate": source["aggregate"],
            "derivatives": derivative_records,
            "eligibility": {
                "sft": "PASS_ONLY",
                "positive_rl_references": "PASS_ONLY",
                "calibration_winners": "PASS_ONLY",
                "final_promotion": "PASS_ONLY",
            },
            "claim_boundary": _CLAIM_BOUNDARY,
        }
        _write_json_fsync(staging / "manifest.json", manifest)
        validate_manifest_file(
            staging / "manifest.json",
            runtime=runtime,
            expected_type="sequence_safety_filter",
        )
        staging.chmod(0o755)
        _fsync_directory(staging)
        runtime.replace(staging, output_dir)
        published = True
        try:
            _fsync_directory(output_dir.parent)
        except OSError:
            pass
        return SafetyState(source["aggregate"]["state"])
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _publish_scan_generation(args: argparse.Namespace, runtime: CLIRuntime) -> tuple[dict[str, object], SafetyState]:
    if type(args.threads) is not int or args.threads < 1:
        raise CLIValidationError("threads must be a positive integer")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise CLIValidationError("timeout must be positive and finite")
    output_dir = Path(args.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise CLIValidationError("output directory must not already exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    published = False
    try:
        created_at = _timestamp(runtime.clock())
        policy_path = Path(args.policy).absolute()
        if policy_path.is_symlink() or not policy_path.is_file():
            raise CLIValidationError("policy must be a non-symlink regular file")
        initial_policy_sha256 = _sha256_regular_file(policy_path, label="policy")
        _load_strict_yaml(policy_path, label="policy")
        policy = load_phage_safety_policy(policy_path)
        evidence = _parse_host_evidence_json(args.host_evidence_json)
        evidence_decision = evaluate_host_evidence(evidence)
        if not evidence_decision.allowed:
            raise CLIValidationError("host evidence is not eligible for sequence design")
        host_domain = _normalize_requested_host_domain(HostDomain(args.host_domain), evidence)
        input_fasta_path = Path(args.input_fasta).absolute()
        initial_input_sha256 = _sha256_regular_file(input_fasta_path, label="input FASTA")
        records = parse_fasta_records(input_fasta_path)
        available_cpu_slots = _available_cpu_slots()
        if type(args.batch_size) is not int or args.batch_size < 1:
            raise CLIValidationError("batch size must be a positive integer")
        batch_mode = args.batch_size > 1
        if batch_mode:
            for label, value in (
                ("batch workers", args.batch_workers),
                ("PHROGs threads", args.phrogs_threads),
                ("PHROGs workers", args.phrogs_workers),
            ):
                if type(value) is not int or value < 1:
                    raise CLIValidationError(f"{label} must be a positive integer")
            if args.record_workers != 1:
                raise CLIValidationError("--record-workers is mutually exclusive with --batch-size > 1")
            batch_workers = min(args.batch_workers, math.ceil(len(records) / args.batch_size))
            if args.phrogs_workers > args.batch_size:
                raise CLIValidationError("PHROGs workers may not exceed the batch size")
            maximum_concurrent_tool_threads = max(
                batch_workers * args.threads,
                batch_workers * args.phrogs_workers * args.phrogs_threads,
            )
            if maximum_concurrent_tool_threads > available_cpu_slots:
                raise CLIValidationError(
                    "batch topology exceeds the available CPU allocation: "
                    f"requires {maximum_concurrent_tool_threads}, available {available_cpu_slots}"
                )
            record_workers = 1
        else:
            record_workers = _resolve_record_workers(
                requested=args.record_workers,
                record_count=len(records),
                tool_threads=args.threads,
                cpu_slots=available_cpu_slots,
            )
            batch_workers = 1
            maximum_concurrent_tool_threads = record_workers * args.threads
        assets = runtime.asset_loader(Path(args.asset_manifest).absolute())
        diamond = runtime.tool_pin_loader(
            Path(args.diamond_tool_pin).absolute(),
            expected_tool="diamond",
            runner=runtime.command_runner,
            timeout=args.timeout,
        )
        mmseqs = runtime.tool_pin_loader(
            Path(args.mmseqs_tool_pin).absolute(),
            expected_tool="mmseqs",
            runner=runtime.command_runner,
            timeout=args.timeout,
        )
        work_root = staging / "records"
        orf_identity = dict(runtime.orf_identity_collector())

        shared_executions: tuple[BatchAdapterExecution, ...] = ()
        if batch_mode:
            batched = runtime.batch_scanner(
                records,
                work_root=work_root,
                shared_root=staging / "shared-executions",
                asset_manifest=assets.manifest,
                diamond_pin=diamond.pin,
                mmseqs_pin=mmseqs.pin,
                host_domain=host_domain,
                strict_lysis=args.strict_lysis,
                threads=args.threads,
                batch_size=args.batch_size,
                batch_workers=batch_workers,
                phrogs_threads=args.phrogs_threads,
                phrogs_workers=args.phrogs_workers,
                timeout=args.timeout,
                circular=not args.linear,
                orf_generation_identity=orf_identity,
                runner=runtime.command_runner,
                phrogs_asset_validator=runtime.phrogs_asset_validator,
                phrogs_asset_revalidator=runtime.phrogs_asset_revalidator,
            )
            batch = batched.batch
            shared_executions = batched.shared_executions
        else:

            def scan_one(record: FastaRecord, input_index: int) -> Mapping[str, AdapterResult]:
                adapters = runtime.record_scanner(
                    record,
                    input_index,
                    work_root=work_root,
                    asset_manifest=assets.manifest,
                    diamond_pin=diamond.pin,
                    mmseqs_pin=mmseqs.pin,
                    host_domain=host_domain,
                    strict_lysis=args.strict_lysis,
                    threads=args.threads,
                    timeout=args.timeout,
                    circular=not args.linear,
                    orf_generation_identity=orf_identity,
                    runner=runtime.command_runner,
                )
                record_root = work_root / f"{input_index:06d}-{record.sequence_id}"
                return _trusted_adapter_bundle(adapters, record_root=record_root)

            batch = scan_records(
                records,
                scanner=scan_one,
                host_domain=host_domain,
                strict_lysis=args.strict_lysis,
                max_workers=record_workers,
            )
        _fsync_owned_regular_files(work_root)
        if shared_executions:
            _fsync_owned_regular_files(staging / "shared-executions")
        serialized_records = [
            _serialize_scanned_record(
                scanned,
                records[scanned.input_index],
                root=staging,
                evidence=evidence,
                host_domain=host_domain,
                strict_lysis=args.strict_lysis,
                circular=not args.linear,
            )
            for scanned in batch.records
        ]
        counts = {state.value: 0 for state in SafetyState}
        for record in batch.records:
            counts[record.result.state.value] += 1
        if _sha256_regular_file(policy_path, label="policy") != initial_policy_sha256:
            raise CLIValidationError("policy changed during scan")
        if _sha256_regular_file(input_fasta_path, label="input FASTA") != initial_input_sha256:
            raise CLIValidationError("input FASTA changed during scan")
        if _sha256_regular_file(assets.manifest_path, label="safety asset manifest") != assets.manifest_sha256:
            raise CLIValidationError("safety asset manifest changed during scan")
        completed_at = _timestamp(runtime.clock())
        source_path = Path(__file__).resolve()
        serialized_shared_executions = [
            _serialize_shared_execution(
                execution,
                root=staging,
                record_indices={record.sequence_id: index for index, record in enumerate(records)},
            )
            for execution in shared_executions
        ]
        runtime_parameters = {
            "threads": args.threads,
            "timeout_seconds": args.timeout,
            "circular": not args.linear,
            "minimum_fallback_amino_acids": 8,
            "maximum_concurrent_tool_threads": maximum_concurrent_tool_threads,
            "available_cpu_slots": available_cpu_slots,
            **(
                {
                    "batch_size": args.batch_size,
                    "batch_workers": batch_workers,
                    "phrogs_threads": args.phrogs_threads,
                    "phrogs_workers": args.phrogs_workers,
                }
                if batch_mode
                else {"record_workers": record_workers}
            ),
        }
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "manifest_type": "sequence_safety_scan",
            "created_at": created_at,
            "completed_at": completed_at,
            "cli_identity": {
                "name": CLI_ID,
                "version": CLI_VERSION,
                "entry_point": "bionemo.evo2_phage_gen.sequence_safety_cli:main",
                "source_path": str(source_path),
                "source_sha256": _sha256_file(source_path),
            },
            "policy": {
                "path": str(policy_path),
                "raw_sha256": initial_policy_sha256,
                "policy_id": policy.policy_id,
                "canonical_sha256": policy.sha256,
            },
            "adapter_policies": [
                {
                    "safety_class": name,
                    "policy_id": identity[0],
                    "policy_sha256": identity[1],
                    "descriptor": _ADAPTER_POLICY_DESCRIPTORS[name],
                }
                for name, identity in _ADAPTER_POLICIES.items()
            ],
            "safety_asset_manifest": {
                "path": str(assets.manifest_path),
                "sha256": assets.manifest_sha256,
                "recipe_path": str(assets.recipe_path),
                "recipe_sha256": assets.recipe_sha256,
            },
            "input_fasta": {
                "path": str(input_fasta_path),
                "sha256": initial_input_sha256,
                "count": len(records),
            },
            "host_evidence_input": {
                "kind": "inline_json",
                "canonical_sha256": hashlib.sha256(
                    json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                ).hexdigest(),
                "value": evidence.to_dict(),
            },
            "resolved_profile": {
                "host_domain": host_domain.value,
                "strict_lysis": bool(args.strict_lysis),
                "applicability": _applicability(host_domain, strict_lysis=args.strict_lysis),
            },
            "runtime_parameters": runtime_parameters,
            "tools": {
                "diamond": _tool_manifest_record(diamond),
                "mmseqs": _tool_manifest_record(mmseqs),
                "orf_predictor": {
                    "identity": orf_identity,
                    "identity_sha256": _canonical_mapping_sha256(orf_identity),
                },
            },
            "environment": dict(runtime.environment_collector()),
            "records": serialized_records,
            **({"shared_executions": serialized_shared_executions} if batch_mode else {}),
            "aggregate": {
                "state": batch.state.value,
                "reason_codes": list(
                    dict.fromkeys(reason for record in serialized_records for reason in record["reason_codes"])
                ),
                "counts": counts,
            },
            "derivatives": {},
            "claim_boundary": _CLAIM_BOUNDARY,
        }
        _write_json_fsync(staging / "manifest.json", manifest)
        validate_manifest_file(
            staging / "manifest.json",
            runtime=runtime,
            expected_type="sequence_safety_scan",
        )
        for directory in sorted((path for path in staging.rglob("*") if path.is_dir()), reverse=True):
            _fsync_directory(directory)
        staging.chmod(0o755)
        _fsync_directory(staging)
        runtime.replace(staging, output_dir)
        published = True
        try:
            _fsync_directory(output_dir.parent)
        except OSError:
            pass
        return manifest, batch.state
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


@dataclass(frozen=True)
class DesignScopeValidation:
    """Typed objective and evidence plus their independent scope decisions."""

    objective: DesignObjective
    host_evidence: HostEvidence
    objective_decision: ScopeDecision
    host_evidence_decision: ScopeDecision

    @property
    def allowed(self) -> bool:
        """Return whether both typed scope checks permit the request."""
        return self.state is SafetyState.PASS

    @property
    def state(self) -> SafetyState:
        """Map eukaryotic replication objectives to FAIL and evidence errors to INDETERMINATE."""
        if (
            not self.objective_decision.allowed
            and "EUKARYOTIC_REPLICATION_OBJECTIVE" in self.objective_decision.reason_codes
        ):
            return SafetyState.FAIL
        if not self.objective_decision.allowed or not self.host_evidence_decision.allowed:
            return SafetyState.INDETERMINATE
        return SafetyState.PASS

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Return stable de-duplicated reasons from objective and host-evidence checks."""
        return tuple(dict.fromkeys((*self.objective_decision.reason_codes, *self.host_evidence_decision.reason_codes)))

    def to_dict(self) -> dict[str, object]:
        """Serialize the typed request and both independent decisions."""
        return {
            "allowed": self.allowed,
            "state": self.state.value,
            "reason_codes": list(self.reason_codes),
            "objective": self.objective.to_dict(),
            "host_evidence": self.host_evidence.to_dict(),
            "objective_decision": self.objective_decision.to_dict(),
            "host_evidence_decision": self.host_evidence_decision.to_dict(),
        }


def _strict_payload(value: object, *, name: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CLIValidationError(f"{name} must be a mapping")
    actual = frozenset(value)
    if actual != keys:
        raise CLIValidationError(
            f"{name} keys do not match schema; unknown={sorted(actual - keys)}, missing={sorted(keys - actual)}"
        )
    return value


def _host_domains(value: object, *, name: str) -> frozenset[HostDomain]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise CLIValidationError(f"{name} must be a non-empty string list")
    try:
        domains = tuple(HostDomain(item) for item in value)
    except ValueError as error:
        raise CLIValidationError(f"{name} contains an unsupported host domain") from error
    if len(domains) != len(set(domains)):
        raise CLIValidationError(f"{name} contains duplicate host domains")
    return frozenset(domains)


def validate_design_scope_payload(
    *,
    objective: Mapping[str, object],
    host_evidence: Mapping[str, object],
) -> DesignScopeValidation:
    """Coerce a strict JSON payload into Task 1's typed objective/evidence API."""
    objective_payload = _strict_payload(
        objective,
        name="objective",
        keys=frozenset({"kind", "direction", "replication_host_domains", "endpoint"}),
    )
    evidence_payload = _strict_payload(
        host_evidence,
        name="host_evidence",
        keys=frozenset({"source", "source_version", "replication_host_domains", "confirmed", "metadata"}),
    )
    try:
        typed_objective = DesignObjective(
            kind=ObjectiveKind(objective_payload["kind"]),
            direction=ObjectiveDirection(objective_payload["direction"]),
            replication_host_domains=_host_domains(
                objective_payload["replication_host_domains"],
                name="objective.replication_host_domains",
            ),
            endpoint=ObjectiveEndpoint(objective_payload["endpoint"]),
        )
    except (TypeError, ValueError) as error:
        raise CLIValidationError(f"invalid typed objective: {error}") from error
    source = evidence_payload["source"]
    source_version = evidence_payload["source_version"]
    confirmed = evidence_payload["confirmed"]
    metadata = evidence_payload["metadata"]
    if not isinstance(source, str) or not source:
        raise CLIValidationError("host_evidence.source must be a non-empty string")
    if source_version is not None and (not isinstance(source_version, str) or not source_version):
        raise CLIValidationError("host_evidence.source_version must be a non-empty string or null")
    if type(confirmed) is not bool:
        raise CLIValidationError("host_evidence.confirmed must be a boolean")
    if not isinstance(metadata, Mapping):
        raise CLIValidationError("host_evidence.metadata must be a mapping")
    try:
        typed_evidence = HostEvidence(
            source=source,
            source_version=source_version,
            replication_host_domains=_host_domains(
                evidence_payload["replication_host_domains"],
                name="host_evidence.replication_host_domains",
            ),
            confirmed=confirmed,
            metadata=metadata,
        )
    except (TypeError, ValueError) as error:
        raise CLIValidationError(f"invalid typed host evidence: {error}") from error
    return DesignScopeValidation(
        objective=typed_objective,
        host_evidence=typed_evidence,
        objective_decision=validate_design_scope(typed_objective),
        host_evidence_decision=evaluate_host_evidence(typed_evidence),
    )


def _applicability(host_domain: HostDomain, *, strict_lysis: bool) -> dict[str, bool]:
    if host_domain in {HostDomain.BACTERIA, HostDomain.BACTERIA_AND_ARCHAEA}:
        return {"amr": True, "toxin": True, "lysogeny": True}
    if host_domain is HostDomain.ARCHAEA:
        return {"amr": True, "toxin": True, "lysogeny": strict_lysis}
    raise CLIValidationError(f"unsupported sequence-safety host profile: {host_domain.value}")


def _indeterminate_class(safety_class: str, *, required: bool, reason: str) -> SafetyClassResult:
    return SafetyClassResult(
        safety_class=safety_class,
        state=SafetyState.INDETERMINATE,
        required=required,
        reason_codes=(reason,),
    )


def aggregate_adapter_results(
    adapters: Mapping[str, AdapterResult],
    *,
    host_domain: HostDomain,
    strict_lysis: bool = False,
) -> GenomeSafetyResult:
    """Aggregate exactly one adapter per applicable class, including supplemental toxin evidence."""
    applicability = _applicability(HostDomain(host_domain), strict_lysis=strict_lysis)
    unknown = sorted(set(adapters) - set(applicability))
    if unknown:
        raise CLIValidationError(f"unknown adapter safety classes: {unknown}")
    class_results: dict[str, SafetyClassResult] = {}
    for safety_class, required in applicability.items():
        adapter = adapters.get(safety_class)
        if adapter is None:
            class_results[safety_class] = _indeterminate_class(
                safety_class,
                required=required,
                reason="REQUIRED_ADAPTER_RESULT_MISSING" if required else "INFORMATIONAL_ADAPTER_RESULT_MISSING",
            )
            continue
        if adapter.class_result.safety_class != safety_class:
            class_results[safety_class] = _indeterminate_class(
                safety_class,
                required=required,
                reason="ADAPTER_SAFETY_CLASS_MISMATCH",
            )
            continue
        class_results[safety_class] = replace(adapter.class_result, required=required)

    amr = adapters.get("amr")
    supplemental = (
        ()
        if amr is None
        else tuple(
            finding
            for finding in amr.supplemental_findings
            if finding.safety_class == "toxin" and _VIRULENCE_REASON in finding.reason_codes
        )
    )
    if supplemental:
        toxin = class_results["toxin"]
        toxin_state = SafetyState.FAIL if toxin.state is SafetyState.FAIL else SafetyState.INDETERMINATE
        supplemental_reasons = tuple(code for finding in supplemental for code in finding.reason_codes)
        class_results["toxin"] = replace(
            toxin,
            state=toxin_state,
            findings=(*toxin.findings, *supplemental),
            reason_codes=tuple(dict.fromkeys((*toxin.reason_codes, *supplemental_reasons))),
        )
    ordered = tuple(class_results[name] for name in ("amr", "toxin", "lysogeny"))
    return GenomeSafetyResult.from_class_results(ordered)


def _parse_host_evidence_json(value: str) -> HostEvidence:
    if not isinstance(value, str):
        raise CLIValidationError("host evidence must be valid JSON")
    payload = _load_strict_json_text(value, label="host evidence")
    evidence = _strict_payload(
        payload,
        name="host_evidence",
        keys=frozenset({"source", "source_version", "replication_host_domains", "confirmed", "metadata"}),
    )
    source = evidence["source"]
    version = evidence["source_version"]
    confirmed = evidence["confirmed"]
    metadata = evidence["metadata"]
    if not isinstance(source, str) or not source:
        raise CLIValidationError("host_evidence.source must be a non-empty string")
    if version is not None and (not isinstance(version, str) or not version):
        raise CLIValidationError("host_evidence.source_version must be a non-empty string or null")
    if type(confirmed) is not bool or not isinstance(metadata, Mapping):
        raise CLIValidationError("host evidence has invalid confirmed or metadata fields")
    _validate_json_value(metadata, label="host_evidence.metadata")
    return HostEvidence(
        source=source,
        source_version=version,
        replication_host_domains=_host_domains(
            evidence["replication_host_domains"], name="host_evidence.replication_host_domains"
        ),
        confirmed=confirmed,
        metadata=metadata,
    )


def _fasta_ids(path: Path) -> list[str]:
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise CLIValidationError(f"cannot read input FASTA: {error}") from error
    identifiers: list[str] = []
    current_has_sequence = False
    for line in lines:
        if line.startswith(b">"):
            if identifiers and not current_has_sequence:
                raise CLIValidationError("FASTA record has no sequence")
            header = line[1:].strip()
            if not header:
                raise CLIValidationError("FASTA header is empty")
            try:
                sequence_id = header.split(None, 1)[0].decode("ascii")
            except UnicodeDecodeError as error:
                raise CLIValidationError("FASTA identifiers must be ASCII") from error
            if sequence_id in identifiers:
                raise CLIValidationError(f"duplicate FASTA record ID: {sequence_id}")
            identifiers.append(sequence_id)
            current_has_sequence = False
        elif line.strip():
            if not identifiers:
                raise CLIValidationError("FASTA sequence bytes precede the first header")
            current_has_sequence = True
    if not identifiers:
        raise CLIValidationError("input FASTA batch is empty")
    if not current_has_sequence:
        raise CLIValidationError("FASTA record has no sequence")
    return identifiers


def _publish_missing_pin_diagnostic(args: argparse.Namespace, runtime: CLIRuntime) -> int | None:
    """Publish an INDETERMINATE diagnostic that is valid but never filter-eligible."""
    reasons: list[str] = []
    if args.diamond_tool_pin is None:
        reasons.append("MISSING_TRUSTED_DIAMOND_TOOL_PIN")
    if args.mmseqs_tool_pin is None:
        reasons.append("MISSING_TRUSTED_MMSEQS_TOOL_PIN")
    if not reasons:
        return None
    output_dir = Path(args.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise CLIValidationError("output directory must not already exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    published = False
    try:
        created_at = _timestamp(runtime.clock())
        policy_path = Path(args.policy).absolute()
        _load_strict_yaml(policy_path, label="policy")
        policy = load_phage_safety_policy(policy_path)
        input_path = Path(args.input_fasta).absolute()
        input_records = parse_fasta_records(input_path)
        asset_path = Path(args.asset_manifest).absolute()
        evidence = _parse_host_evidence_json(args.host_evidence_json)
        if not evaluate_host_evidence(evidence).allowed:
            raise CLIValidationError("host evidence is not eligible for sequence design")
        host_domain = _normalize_requested_host_domain(HostDomain(args.host_domain), evidence)
        source_path = Path(__file__).resolve()
        evidence_mapping = evidence.to_dict()
        manifest: dict[str, object] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "manifest_type": "sequence_safety_diagnostic",
            "created_at": created_at,
            "completed_at": _timestamp(runtime.clock()),
            "cli_identity": {
                "name": CLI_ID,
                "version": CLI_VERSION,
                "entry_point": "bionemo.evo2_phage_gen.sequence_safety_cli:main",
                "source_path": str(source_path),
                "source_sha256": _sha256_file(source_path),
            },
            "policy": {
                "path": str(policy_path),
                "raw_sha256": _sha256_regular_file(policy_path, label="policy"),
                "policy_id": policy.policy_id,
                "canonical_sha256": policy.sha256,
            },
            "input_fasta": {
                "path": str(input_path),
                "sha256": _sha256_regular_file(input_path, label="input FASTA"),
                "count": len(input_records),
            },
            "safety_asset_manifest_input": {
                "path": str(asset_path),
                "sha256": _sha256_regular_file(asset_path, label="safety asset manifest input"),
                "validation_status": "NOT_VALIDATED_DUE_TO_MISSING_TOOL_PINS",
            },
            "host_evidence_input": {
                "kind": "inline_json",
                "canonical_sha256": hashlib.sha256(
                    json.dumps(evidence_mapping, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                ).hexdigest(),
                "value": evidence_mapping,
            },
            "resolved_profile": {
                "host_domain": host_domain.value,
                "strict_lysis": bool(args.strict_lysis),
            },
            "missing_prerequisites": reasons,
            "records": [
                {
                    "record_id": record.sequence_id,
                    "input_index": index,
                    "state": SafetyState.INDETERMINATE.value,
                    "reason_codes": reasons,
                }
                for index, record in enumerate(input_records)
            ],
            "aggregate": {
                "state": SafetyState.INDETERMINATE.value,
                "reason_codes": reasons,
                "counts": {
                    SafetyState.PASS.value: 0,
                    SafetyState.FAIL.value: 0,
                    SafetyState.INDETERMINATE.value: len(input_records),
                },
            },
            "claim_boundary": _CLAIM_BOUNDARY,
        }
        _write_json_fsync(staging / "manifest.json", manifest)
        validate_manifest_file(
            staging / "manifest.json",
            runtime=runtime,
            expected_type="sequence_safety_diagnostic",
        )
        staging.chmod(0o755)
        _fsync_directory(staging)
        runtime.replace(staging, output_dir)
        published = True
        try:
            _fsync_directory(output_dir.parent)
        except OSError:
            pass
        return 3
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _run_scan(args: argparse.Namespace, runtime: CLIRuntime | None = None) -> int:
    selected_runtime = CLIRuntime() if runtime is None else runtime
    if args.diamond_tool_pin is None or args.mmseqs_tool_pin is None:
        return int(_publish_missing_pin_diagnostic(args, selected_runtime))
    _, state = _publish_scan_generation(args, selected_runtime)
    return _state_exit_code(state)


def _run_filter_fasta(args: argparse.Namespace, runtime: CLIRuntime | None = None) -> int:
    selected_runtime = CLIRuntime() if runtime is None else runtime
    return _state_exit_code(_publish_filter_generation(args, selected_runtime))


def _run_validate_manifest(args: argparse.Namespace, runtime: CLIRuntime | None = None) -> int:
    selected_runtime = CLIRuntime() if runtime is None else runtime
    manifest = validate_manifest_file(args.manifest, runtime=selected_runtime)
    return _state_exit_code(SafetyState(manifest["aggregate"]["state"]))


def _run_validate_scope(args: argparse.Namespace, runtime: CLIRuntime | None = None) -> int:
    try:
        payload = _load_strict_json(args.input, label="scope request")
        strict = _strict_payload(payload, name="scope_request", keys=frozenset({"objective", "host_evidence"}))
        result = validate_design_scope_payload(
            objective=strict["objective"],
            host_evidence=strict["host_evidence"],
        )
    except (OSError, TypeError, ValueError) as error:
        raise CLIValidationError(str(error)) from error
    serialized = json.dumps(result.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized)
    if result.state is SafetyState.FAIL:
        return 2
    if result.state is SafetyState.INDETERMINATE:
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argv-aware parser without reserving exit two for usage errors."""
    parser = _ArgumentParser(prog="evo2_phage_sequence_safety")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="scan every FASTA record with pinned safety adapters")
    scan.add_argument("--input-fasta", type=Path, required=True)
    scan.add_argument("--output-dir", type=Path, required=True)
    scan.add_argument("--policy", type=Path, required=True)
    scan.add_argument("--asset-manifest", type=Path, required=True)
    scan.add_argument("--host-domain", choices=tuple(domain.value for domain in HostDomain), required=True)
    scan.add_argument("--host-evidence-json", required=True)
    scan.add_argument("--diamond-tool-pin", type=Path)
    scan.add_argument("--mmseqs-tool-pin", type=Path)
    scan.add_argument("--strict-lysis", action="store_true")
    scan.add_argument("--linear", action="store_true", help="treat every input genome as linear")
    scan.add_argument("--threads", type=int, default=1)
    scan.add_argument(
        "--record-workers",
        type=int,
        default=1,
        help="concurrent records; workers x --threads must fit the available CPU allocation",
    )
    scan.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="records per shared AMRFinder/DIAMOND command; values >1 enable verified batching",
    )
    scan.add_argument(
        "--batch-workers",
        type=int,
        default=1,
        help="concurrent batches when --batch-size is greater than one",
    )
    scan.add_argument(
        "--phrogs-threads",
        type=int,
        default=1,
        help="threads per independent per-record PHROGs search in batch mode",
    )
    scan.add_argument(
        "--phrogs-workers",
        type=int,
        default=1,
        help="concurrent independent PHROGs searches inside each batch",
    )
    scan.add_argument("--timeout", type=float, default=300.0)
    scan.set_defaults(handler=_run_scan)

    filter_fasta = subparsers.add_parser("filter-fasta", help="partition FASTA records from a trusted manifest")
    filter_fasta.add_argument("--input-fasta", type=Path, required=True)
    filter_fasta.add_argument("--scan-manifest", type=Path, required=True)
    filter_fasta.add_argument("--output-dir", type=Path, required=True)
    filter_fasta.set_defaults(handler=_run_filter_fasta)

    validate_manifest = subparsers.add_parser("validate-manifest", help="revalidate a scanner or filter manifest")
    validate_manifest.add_argument("--manifest", type=Path, required=True)
    validate_manifest.set_defaults(handler=_run_validate_manifest)

    validate_scope = subparsers.add_parser(
        "validate-design-scope", help="validate a typed objective and host evidence"
    )
    validate_scope.add_argument("--input", type=Path, required=True)
    validate_scope.add_argument("--output", type=Path)
    validate_scope.set_defaults(handler=_run_validate_scope)
    return parser


def main(argv: Sequence[str] | None = None, *, runtime: CLIRuntime | None = None) -> int:
    """Run the CLI and map validation/infrastructure errors to exit status three."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        handler = getattr(args, "handler", None)
        if handler is None:
            raise CLIValidationError(f"subcommand {args.command!r} is not implemented")
        return int(handler(args, runtime))
    except (CLIValidationError, OSError, RuntimeError, subprocess.SubprocessError, TypeError, ValueError) as error:
        parser._print_message(f"{parser.prog}: error: {error}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
