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

"""Record-preserving batch inputs and output splitters that block PASS on unresolved evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import unquote

from bionemo.evo2_phage_gen.sequence_safety import SafetyState
from bionemo.evo2_phage_gen.sequence_safety_adapters import (
    _AMRFINDER_COLUMNS,
    _AMRFINDER_POLICY_ID,
    _AMRFINDER_POLICY_SHA256,
    _DIAMOND_COLUMNS,
    TOXIN_HOMOLOGY_POLICY_V2,
    AdapterResult,
    AssetProvenanceError,
    ORFArtifacts,
    ORFQueryRecord,
    ToolPin,
    _indeterminate_adapter_result,
    _parse_amrfinder_database_version,
    _parse_amrfinder_output_validated,
    _parse_toxin_diamond_output_validated,
    _sha256_file,
    _validate_amrfinder_manifest_section,
    _validate_toxin_assets,
    _write_normalized_header,
    build_amrfinder_command,
    build_diamond_command,
    validate_tool_pin,
)


_RECORD_ID = re.compile(r"[A-Za-z0-9_.-]+")
_AMRFINDER_SPLIT_POLICY = {
    "policy_id": "amrfinder-record-split-v1",
    "header": list(_AMRFINDER_COLUMNS),
    "primary_owner": "exact Contig id",
    "fallback_owner": "exact Protein id query inventory when Contig id is unavailable",
    "nucleotide_only_owner": "exact Contig id when Protein id is NA",
    "row_order": "batch encounter order within each record",
    "failure": "reject unknown, conflicting, malformed, or duplicate rows",
}
AMRFINDER_SPLIT_POLICY_ID = str(_AMRFINDER_SPLIT_POLICY["policy_id"])
AMRFINDER_SPLIT_POLICY_SHA256 = hashlib.sha256(
    json.dumps(_AMRFINDER_SPLIT_POLICY, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
_DIAMOND_SPLIT_POLICY = {
    "policy_id": "diamond-query-owner-split-v1",
    "columns": list(_DIAMOND_COLUMNS),
    "owner": "exact qseqid in authenticated all-query ORF inventory",
    "row_order": "batch encounter order within each record",
    "failure": "reject unknown or malformed query rows",
}
DIAMOND_SPLIT_POLICY_ID = str(_DIAMOND_SPLIT_POLICY["policy_id"])
DIAMOND_SPLIT_POLICY_SHA256 = hashlib.sha256(
    json.dumps(_DIAMOND_SPLIT_POLICY, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class BatchSafetyError(RuntimeError):
    """A batch cannot be proven equivalent to its independent record inputs."""


@dataclass(frozen=True)
class BatchedORFInputs:
    """Exact combined inputs plus immutable record/query ownership."""

    artifacts: ORFArtifacts
    record_ids: tuple[str, ...]
    query_owners: tuple[tuple[str, str], ...]

    def query_owner_mapping(self) -> dict[str, str]:
        """Return a fresh query-to-record map after construction-time uniqueness checks."""
        return dict(self.query_owners)


@dataclass(frozen=True)
class BatchAdapterExecution:
    """One real shared tool execution and its independently parsed record results."""

    batch_id: str
    safety_class: str
    record_ids: tuple[str, ...]
    inputs: BatchedORFInputs
    command: tuple[str, ...]
    raw_output_path: Path | None
    raw_output_sha256: str | None
    split_policy_id: str
    split_policy_sha256: str
    record_results: tuple[tuple[str, AdapterResult], ...]
    execution_status: str = "COMPLETED_AND_PARSED"

    def __post_init__(self) -> None:
        """Reject ambiguous execution, input, split, or per-record identities."""
        if _RECORD_ID.fullmatch(self.batch_id) is None:
            raise ValueError("batch execution ID is invalid")
        if tuple(record_id for record_id, _ in self.record_results) != self.record_ids:
            raise ValueError("batch result order differs from its record inventory")
        if (self.raw_output_path is None) != (self.raw_output_sha256 is None):
            raise ValueError("batch raw-output path and digest must be recorded together")
        if self.execution_status not in {"NOT_STARTED", "FAILED", "COMPLETED_AND_PARSED"}:
            raise ValueError("batch execution status is invalid")
        if self.execution_status == "NOT_STARTED" and (self.command or self.raw_output_path is not None):
            raise ValueError("not-started batch execution cannot claim a command or output")
        if self.execution_status == "FAILED" and not self.command:
            raise ValueError("failed batch execution must record its attempted command")
        if self.execution_status == "COMPLETED_AND_PARSED" and (not self.command or self.raw_output_path is None):
            raise ValueError("completed batch execution must record its command and raw output")


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise BatchSafetyError(f"{label} is not a non-symlink regular file")
    try:
        return selected.read_bytes()
    except OSError as error:
        raise BatchSafetyError(f"cannot read {label}: {error}") from error


def _parse_exact_fasta(path: Path, *, label: str) -> tuple[tuple[str, str], ...]:
    payload = _read_regular_bytes(path, label=label)
    try:
        text = payload.decode("ascii")
    except UnicodeError as error:
        raise BatchSafetyError(f"{label} is not ASCII FASTA") from error
    if not text or not text.endswith("\n") or "\r" in text:
        raise BatchSafetyError(f"{label} must be newline-terminated canonical FASTA")
    records: list[tuple[str, str]] = []
    selected_id: str | None = None
    sequence_parts: list[str] = []
    for line in text[:-1].split("\n"):
        if line.startswith(">"):
            if selected_id is not None:
                if not sequence_parts:
                    raise BatchSafetyError(f"{label} contains an empty FASTA record")
                records.append((selected_id, "".join(sequence_parts)))
            selected_id = line[1:].split(maxsplit=1)[0]
            if not selected_id or _RECORD_ID.fullmatch(selected_id) is None:
                raise BatchSafetyError(f"{label} contains an invalid FASTA identifier")
            sequence_parts = []
        else:
            if selected_id is None or not line or any(character.isspace() for character in line):
                raise BatchSafetyError(f"{label} contains malformed FASTA sequence bytes")
            sequence_parts.append(line)
    if selected_id is None or not sequence_parts:
        raise BatchSafetyError(f"{label} contains no complete FASTA record")
    records.append((selected_id, "".join(sequence_parts)))
    identifiers = [identifier for identifier, _ in records]
    if len(identifiers) != len(set(identifiers)):
        raise BatchSafetyError(f"{label} contains duplicate FASTA identifiers")
    return tuple(records)


def _validate_gff(
    path: Path,
    *,
    record_id: str,
    sequence_length: int,
    primary_query_ids: tuple[str, ...],
) -> tuple[str, ...]:
    payload = _read_regular_bytes(path, label=f"{record_id} GFF")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise BatchSafetyError(f"{record_id} GFF is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise BatchSafetyError(f"{record_id} GFF is not canonical newline-terminated text")
    lines = tuple(text[:-1].split("\n"))
    if not lines or lines[0] != "##gff-version 3" or lines.count("##gff-version 3") != 1:
        raise BatchSafetyError(f"{record_id} GFF must contain one leading version directive")
    expected_region = f"##sequence-region {record_id} 1 {sequence_length}"
    regions = tuple(line for line in lines if line.startswith("##sequence-region "))
    if regions != (expected_region,):
        raise BatchSafetyError(f"{record_id} GFF sequence-region identity is invalid")
    observed_feature_ids: list[str] = []
    for line in lines[1:]:
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 9 or fields[0] != record_id:
            raise BatchSafetyError(f"{record_id} GFF feature identity is invalid")
        attributes = dict(item.split("=", 1) for item in fields[8].split(";") if item and "=" in item)
        feature_id = unquote(attributes.get("ID", ""))
        if not feature_id or feature_id in observed_feature_ids:
            raise BatchSafetyError(f"{record_id} GFF feature ID is missing or duplicated")
        observed_feature_ids.append(feature_id)
    if tuple(observed_feature_ids) != primary_query_ids:
        raise BatchSafetyError(f"{record_id} GFF feature inventory differs from primary ORFs")
    return lines[1:]


def _require_file_matches_records(
    path: Path,
    expected: Iterable[tuple[str, str]],
    *,
    label: str,
) -> bytes:
    expected_tuple = tuple(expected)
    if _parse_exact_fasta(path, label=label) != expected_tuple:
        raise BatchSafetyError(f"{label} differs from the ORF query inventory")
    return _read_regular_bytes(path, label=label)


def materialize_batched_orf_inputs(
    record_artifacts: Sequence[tuple[str, ORFArtifacts]],
    output_dir: Path,
) -> BatchedORFInputs:
    """Combine independent record inputs without changing their bytes, order, or identities."""
    selected = tuple(record_artifacts)
    if not selected:
        raise BatchSafetyError("a safety batch must contain at least one record")
    record_ids = tuple(record_id for record_id, _ in selected)
    if any(_RECORD_ID.fullmatch(record_id) is None for record_id in record_ids):
        raise BatchSafetyError("batch record identity is invalid")
    if len(record_ids) != len(set(record_ids)):
        raise BatchSafetyError("batch record identity is duplicated")
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        raise BatchSafetyError(f"batch input directory already exists: {root}")
    root.mkdir(parents=True)

    combined_payloads: dict[str, bytearray] = {
        "genomes_fna": bytearray(),
        "proteins_faa": bytearray(),
        "proteins_fna": bytearray(),
        "all_queries_faa": bytearray(),
    }
    combined_gff_lines = ["##gff-version 3"]
    all_queries: list[ORFQueryRecord] = []
    query_owners: list[tuple[str, str]] = []
    seen_queries: set[str] = set()
    try:
        for record_id, artifacts in selected:
            queries = tuple(artifacts.query_records)
            if not queries or any(query.sequence_id != record_id for query in queries):
                raise BatchSafetyError(f"{record_id} ORF record identity differs from its batch owner")
            query_ids = tuple(query.query_id for query in queries)
            if len(query_ids) != len(set(query_ids)) or seen_queries.intersection(query_ids):
                raise BatchSafetyError("batch query identity is duplicated")
            seen_queries.update(query_ids)
            primary = tuple(query for query in queries if query.evidence_path == "pyrodigal-gv")
            genome_records = _parse_exact_fasta(artifacts.genomes_fna, label=f"{record_id} genomes FASTA")
            if len(genome_records) != 1 or genome_records[0][0] != record_id:
                raise BatchSafetyError(f"{record_id} genome FASTA record identity is invalid")
            sequence_length = len(genome_records[0][1])
            combined_payloads["genomes_fna"].extend(
                _read_regular_bytes(artifacts.genomes_fna, label=f"{record_id} genomes FASTA")
            )
            combined_payloads["proteins_faa"].extend(
                _require_file_matches_records(
                    artifacts.proteins_faa,
                    ((query.query_id, query.protein) for query in primary),
                    label=f"{record_id} primary protein FASTA",
                )
            )
            combined_payloads["proteins_fna"].extend(
                _require_file_matches_records(
                    artifacts.proteins_fna,
                    ((query.query_id, query.nucleotide) for query in primary),
                    label=f"{record_id} primary nucleotide FASTA",
                )
            )
            combined_payloads["all_queries_faa"].extend(
                _require_file_matches_records(
                    artifacts.all_queries_faa,
                    ((query.query_id, query.protein) for query in queries),
                    label=f"{record_id} all-query protein FASTA",
                )
            )
            combined_gff_lines.extend(
                _validate_gff(
                    artifacts.proteins_gff,
                    record_id=record_id,
                    sequence_length=sequence_length,
                    primary_query_ids=tuple(query.query_id for query in primary),
                )
            )
            all_queries.extend(queries)
            query_owners.extend((query.query_id, record_id) for query in queries)
        paths = {
            role: root
            / {
                "genomes_fna": "genomes.fna",
                "proteins_faa": "proteins.faa",
                "proteins_fna": "proteins.fna",
                "all_queries_faa": "all_queries.faa",
            }[role]
            for role in combined_payloads
        }
        for role, payload in combined_payloads.items():
            paths[role].write_bytes(bytes(payload))
        proteins_gff = root / "proteins.gff"
        proteins_gff.write_text("\n".join(combined_gff_lines) + "\n")
    except Exception:
        for child in root.iterdir():
            child.unlink(missing_ok=True)
        root.rmdir()
        raise
    artifacts = ORFArtifacts(
        genomes_fna=paths["genomes_fna"],
        proteins_faa=paths["proteins_faa"],
        proteins_fna=paths["proteins_fna"],
        proteins_gff=proteins_gff,
        all_queries_faa=paths["all_queries_faa"],
        query_records=tuple(all_queries),
    )
    return BatchedORFInputs(
        artifacts=artifacts,
        record_ids=record_ids,
        query_owners=tuple(query_owners),
    )


def _new_split_root(output_root: Path) -> Path:
    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise BatchSafetyError(f"batch split directory already exists: {root}")
    root.mkdir(parents=True)
    return root


def _write_record_splits(
    root: Path,
    *,
    record_ids: Sequence[str],
    filename: str,
    prefixes: Mapping[str, Sequence[str]],
    rows: Mapping[str, Sequence[str]],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    try:
        for record_id in record_ids:
            record_root = root / record_id
            record_root.mkdir()
            path = record_root / filename
            output_lines = [*prefixes[record_id], *rows[record_id]]
            path.write_text("" if not output_lines else "\n".join(output_lines) + "\n")
            paths[record_id] = path
    except Exception:
        for child in sorted(root.rglob("*"), reverse=True):
            child.rmdir() if child.is_dir() else child.unlink(missing_ok=True)
        root.rmdir()
        raise
    return paths


def split_amrfinder_batch_output(
    raw_output: Path,
    *,
    batched: BatchedORFInputs,
    output_root: Path,
) -> dict[str, Path]:
    """Split one AMRFinder table using exact contig/query ownership, failing on ambiguity."""
    payload = _read_regular_bytes(raw_output, label="AMRFinder batch output")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise BatchSafetyError("AMRFinder batch output is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise BatchSafetyError("AMRFinder batch output is not canonical newline-terminated text")
    lines = text[:-1].split("\n")
    prefixes: list[str] = []
    header_index: int | None = None
    for index, line in enumerate(lines):
        if not line.strip() or line.startswith("#"):
            prefixes.append(line)
            continue
        header_index = index
        break
    expected_header = "\t".join(_AMRFINDER_COLUMNS)
    if header_index is None or lines[header_index] != expected_header:
        raise BatchSafetyError("AMRFinder batch output header is invalid")
    if any(not line.strip() or line.startswith("#") for line in lines[header_index + 1 :]):
        raise BatchSafetyError("AMRFinder batch output contains comments or blanks after its header")
    query_owners = batched.query_owner_mapping()
    record_ids = frozenset(batched.record_ids)
    rows: dict[str, list[str]] = {record_id: [] for record_id in batched.record_ids}
    seen_rows: set[tuple[str, ...]] = set()
    for line in lines[header_index + 1 :]:
        fields = tuple(line.split("\t"))
        if len(fields) != len(_AMRFINDER_COLUMNS) or fields in seen_rows:
            raise BatchSafetyError("AMRFinder batch output contains malformed or duplicate rows")
        seen_rows.add(fields)
        row = dict(zip(_AMRFINDER_COLUMNS, fields, strict=True))
        contig_owner = row["Contig id"] if row["Contig id"] in record_ids else None
        query_owner = None if row["Protein id"] == "NA" else query_owners.get(row["Protein id"])
        if contig_owner is not None and query_owner is not None and contig_owner != query_owner:
            raise BatchSafetyError("AMRFinder row has conflicting contig and protein owners")
        owner = contig_owner or query_owner
        if owner is None or (row["Protein id"] == "NA" and contig_owner is None):
            raise BatchSafetyError("AMRFinder row has no exact record owner")
        rows[owner].append(line)
    root = _new_split_root(output_root)
    shared_prefix = (*prefixes, expected_header)
    return _write_record_splits(
        root,
        record_ids=batched.record_ids,
        filename="amrfinder.tsv",
        prefixes=dict.fromkeys(batched.record_ids, shared_prefix),
        rows=rows,
    )


def split_diamond_batch_output(
    raw_output: Path,
    *,
    batched: BatchedORFInputs,
    output_root: Path,
) -> dict[str, Path]:
    """Split headerless DIAMOND output by exact query ownership without reordering rows."""
    payload = _read_regular_bytes(raw_output, label="DIAMOND batch output")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise BatchSafetyError("DIAMOND batch output is not UTF-8") from error
    if text and (not text.endswith("\n") or "\r" in text):
        raise BatchSafetyError("DIAMOND batch output is not canonical newline-terminated text")
    query_owners = batched.query_owner_mapping()
    rows: dict[str, list[str]] = {record_id: [] for record_id in batched.record_ids}
    for line in [] if not text else text[:-1].split("\n"):
        fields = line.split("\t")
        if len(fields) != len(_DIAMOND_COLUMNS):
            raise BatchSafetyError("DIAMOND batch output row schema is invalid")
        owner = query_owners.get(fields[0])
        if owner is None:
            raise BatchSafetyError(f"unknown DIAMOND query in batch output: {fields[0]}")
        rows[owner].append(line)
    root = _new_split_root(output_root)
    return _write_record_splits(
        root,
        record_ids=batched.record_ids,
        filename="toxin_diamond.raw.tsv",
        prefixes=dict.fromkeys(batched.record_ids, ()),
        rows=rows,
    )


def _validated_record_output_roots(
    record_ids: Sequence[str], record_output_roots: Mapping[str, Path]
) -> dict[str, Path]:
    if set(record_output_roots) != set(record_ids):
        raise BatchSafetyError("record output roots do not exactly match the batch inventory")
    validated: dict[str, Path] = {}
    for record_id in record_ids:
        root = Path(record_output_roots[record_id])
        if root.is_symlink() or not root.is_dir():
            raise BatchSafetyError(f"record output root is not an existing regular directory: {root}")
        validated[record_id] = root
    return validated


def _amrfinder_indeterminate_results(
    record_ids: Sequence[str],
    *,
    required: bool,
    reason_code: str,
    command: tuple[str, ...] = (),
    raw_output_path: Path | None = None,
) -> tuple[tuple[str, AdapterResult], ...]:
    results: list[tuple[str, AdapterResult]] = []
    for record_id in record_ids:
        base = _indeterminate_adapter_result(
            "amr",
            required=required,
            reason_code=reason_code,
            command=command,
            raw_output_path=raw_output_path,
        )
        results.append(
            (
                record_id,
                AdapterResult(
                    class_result=base.class_result,
                    supplemental_findings=base.supplemental_findings,
                    command=base.command,
                    raw_output_path=base.raw_output_path,
                    raw_output_sha256=base.raw_output_sha256,
                    policy_id=_AMRFINDER_POLICY_ID,
                    policy_sha256=_AMRFINDER_POLICY_SHA256,
                ),
            )
        )
    return tuple(results)


def _amrfinder_execution(
    *,
    work_dir: Path,
    batched: BatchedORFInputs,
    command: tuple[str, ...],
    raw_output: Path | None,
    results: tuple[tuple[str, AdapterResult], ...],
) -> BatchAdapterExecution:
    execution_status = (
        "NOT_STARTED"
        if not command
        else (
            "FAILED"
            if any(result.class_result.state is SafetyState.INDETERMINATE for _, result in results)
            else "COMPLETED_AND_PARSED"
        )
    )
    command_output_path = str((raw_output or work_dir / "amrfinder.raw.tsv").absolute())
    bound_results = tuple(
        (
            record_id,
            replace(
                result,
                shared_execution_id=work_dir.name,
                raw_output_path=result.raw_output_path if execution_status == "COMPLETED_AND_PARSED" else None,
                raw_output_sha256=result.raw_output_sha256 if execution_status == "COMPLETED_AND_PARSED" else None,
                command_output_path=command_output_path,
            ),
        )
        for record_id, result in results
    )
    return BatchAdapterExecution(
        batch_id=work_dir.name,
        safety_class="amr",
        record_ids=batched.record_ids,
        inputs=batched,
        command=command,
        raw_output_path=raw_output,
        raw_output_sha256=None if raw_output is None else _sha256_file(raw_output),
        split_policy_id=AMRFINDER_SPLIT_POLICY_ID,
        split_policy_sha256=AMRFINDER_SPLIT_POLICY_SHA256,
        record_results=bound_results,
        execution_status=execution_status,
    )


def run_amrfinder_batch(
    record_artifacts: Sequence[tuple[str, ORFArtifacts]],
    *,
    manifest_section: Mapping[str, object],
    work_dir: Path,
    record_output_roots: Mapping[str, Path],
    threads: int = 1,
    required: bool = True,
    runner=subprocess.run,
    timeout: float = 300.0,
) -> BatchAdapterExecution:
    """Run one combined AMRFinder invocation and parse exact per-record split files."""
    if type(threads) is not int or threads < 1:
        raise BatchSafetyError("AMRFinder batch threads must be a positive integer")
    selected = tuple(record_artifacts)
    execution_root = Path(work_dir)
    if execution_root.exists() or execution_root.is_symlink():
        raise BatchSafetyError(f"batch execution directory already exists: {execution_root}")
    execution_root.mkdir(parents=True)
    batched = materialize_batched_orf_inputs(selected, execution_root / "inputs")
    output_roots = _validated_record_output_roots(batched.record_ids, record_output_roots)
    try:
        tool_pin, database_path, database_version, blast_bin_dir, hmmer_bin_dir = _validate_amrfinder_manifest_section(
            manifest_section
        )
        validate_tool_pin(tool_pin, runner=runner, timeout=timeout)
        database_completed = runner(
            [str(tool_pin.path), "--database", str(database_path), "--database_version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if _parse_amrfinder_database_version(database_completed.stdout) != database_version:
            raise BatchSafetyError("AMRFinder database version changed before batch execution")
    except (
        BatchSafetyError,
        AssetProvenanceError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ):
        return _amrfinder_execution(
            work_dir=execution_root,
            batched=batched,
            command=(),
            raw_output=None,
            results=_amrfinder_indeterminate_results(
                batched.record_ids,
                required=required,
                reason_code="AMRFINDER_ASSET_PROVENANCE_MISMATCH",
            ),
        )
    raw_output = execution_root / "amrfinder.raw.tsv"
    command = tuple(
        build_amrfinder_command(
            amrfinder=tool_pin.path,
            genomes_fna=batched.artifacts.genomes_fna,
            proteins_faa=batched.artifacts.proteins_faa,
            proteins_gff=batched.artifacts.proteins_gff,
            database_dir=database_path,
            blast_bin_dir=blast_bin_dir,
            hmmer_bin_dir=hmmer_bin_dir,
            threads=threads,
            output_tsv=raw_output,
        )
    )
    try:
        runner(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return _amrfinder_execution(
            work_dir=execution_root,
            batched=batched,
            command=command,
            raw_output=raw_output if raw_output.is_file() else None,
            results=_amrfinder_indeterminate_results(
                batched.record_ids,
                required=required,
                reason_code="AMRFINDER_EXECUTION_TIMEOUT",
                command=command,
                raw_output_path=raw_output,
            ),
        )
    except (subprocess.CalledProcessError, OSError):
        return _amrfinder_execution(
            work_dir=execution_root,
            batched=batched,
            command=command,
            raw_output=raw_output if raw_output.is_file() else None,
            results=_amrfinder_indeterminate_results(
                batched.record_ids,
                required=required,
                reason_code="AMRFINDER_EXECUTION_FAILED",
                command=command,
                raw_output_path=raw_output,
            ),
        )
    if not raw_output.is_file():
        return _amrfinder_execution(
            work_dir=execution_root,
            batched=batched,
            command=command,
            raw_output=None,
            results=_amrfinder_indeterminate_results(
                batched.record_ids,
                required=required,
                reason_code="AMRFINDER_OUTPUT_MISSING",
                command=command,
                raw_output_path=raw_output,
            ),
        )
    promoted_paths: list[Path] = []
    try:
        split_paths = split_amrfinder_batch_output(
            raw_output,
            batched=batched,
            output_root=execution_root / "split",
        )
        artifacts_by_id = dict(selected)
        destinations: dict[str, Path] = {}
        parsed_by_id: dict[str, AdapterResult] = {}
        for record_id in batched.record_ids:
            destination = output_roots[record_id] / "amrfinder.tsv"
            if destination.exists() or destination.is_symlink():
                raise BatchSafetyError(f"record AMRFinder output already exists: {destination}")
            destinations[record_id] = destination
            parsed_by_id[record_id] = _parse_amrfinder_output_validated(
                split_paths[record_id],
                artifacts=artifacts_by_id[record_id],
                manifest_section=manifest_section,
                required=required,
            )
        record_results: list[tuple[str, AdapterResult]] = []
        for record_id in batched.record_ids:
            destination = destinations[record_id]
            split_paths[record_id].replace(destination)
            promoted_paths.append(destination)
            parsed = parsed_by_id[record_id]
            record_results.append(
                (
                    record_id,
                    AdapterResult(
                        class_result=parsed.class_result,
                        supplemental_findings=parsed.supplemental_findings,
                        command=command,
                        raw_output_path=str(destination),
                        raw_output_sha256=_sha256_file(destination),
                        policy_id=parsed.policy_id,
                        policy_sha256=parsed.policy_sha256,
                        shared_execution_id=execution_root.name,
                        command_output_path=str(raw_output.absolute()),
                    ),
                )
            )
        for record_id in batched.record_ids:
            (execution_root / "split" / record_id).rmdir()
        (execution_root / "split").rmdir()
    except (BatchSafetyError, KeyError, OSError, TypeError, ValueError):
        for promoted_path in promoted_paths:
            promoted_path.unlink(missing_ok=True)
        return _amrfinder_execution(
            work_dir=execution_root,
            batched=batched,
            command=command,
            raw_output=raw_output,
            results=_amrfinder_indeterminate_results(
                batched.record_ids,
                required=required,
                reason_code="AMRFINDER_BATCH_SPLIT_MISMATCH",
                command=command,
                raw_output_path=raw_output,
            ),
        )
    return _amrfinder_execution(
        work_dir=execution_root,
        batched=batched,
        command=command,
        raw_output=raw_output,
        results=tuple(record_results),
    )


def _toxin_indeterminate_results(
    record_ids: Sequence[str],
    *,
    required: bool,
    reason_code: str,
    command: tuple[str, ...] = (),
    raw_output_path: Path | None = None,
) -> tuple[tuple[str, AdapterResult], ...]:
    results: list[tuple[str, AdapterResult]] = []
    for record_id in record_ids:
        base = _indeterminate_adapter_result(
            "toxin",
            required=required,
            reason_code=reason_code,
            command=command,
            raw_output_path=raw_output_path,
        )
        results.append(
            (
                record_id,
                AdapterResult(
                    class_result=base.class_result,
                    command=base.command,
                    raw_output_path=base.raw_output_path,
                    policy_id=TOXIN_HOMOLOGY_POLICY_V2.policy_id,
                    policy_sha256=TOXIN_HOMOLOGY_POLICY_V2.sha256,
                ),
            )
        )
    return tuple(results)


def _toxin_execution(
    *,
    work_dir: Path,
    batched: BatchedORFInputs,
    command: tuple[str, ...],
    raw_output: Path | None,
    results: tuple[tuple[str, AdapterResult], ...],
) -> BatchAdapterExecution:
    execution_status = (
        "NOT_STARTED"
        if not command
        else (
            "FAILED"
            if any(result.class_result.state is SafetyState.INDETERMINATE for _, result in results)
            else "COMPLETED_AND_PARSED"
        )
    )
    command_output_path = str((raw_output or work_dir / "toxin_diamond.raw.tsv").absolute())
    bound_results = tuple(
        (
            record_id,
            replace(
                result,
                shared_execution_id=work_dir.name,
                command_output_path=command_output_path,
                raw_output_path=result.raw_output_path if execution_status == "COMPLETED_AND_PARSED" else None,
                raw_output_sha256=result.raw_output_sha256 if execution_status == "COMPLETED_AND_PARSED" else None,
            ),
        )
        for record_id, result in results
    )
    return BatchAdapterExecution(
        batch_id=work_dir.name,
        safety_class="toxin",
        record_ids=batched.record_ids,
        inputs=batched,
        command=command,
        raw_output_path=raw_output,
        raw_output_sha256=None if raw_output is None else _sha256_file(raw_output),
        split_policy_id=DIAMOND_SPLIT_POLICY_ID,
        split_policy_sha256=DIAMOND_SPLIT_POLICY_SHA256,
        record_results=bound_results,
        execution_status=execution_status,
    )


def run_toxin_diamond_batch(
    record_artifacts: Sequence[tuple[str, ORFArtifacts]],
    *,
    manifest_section: Mapping[str, object],
    tool_pin: ToolPin,
    work_dir: Path,
    record_output_roots: Mapping[str, Path],
    threads: int = 1,
    required: bool = True,
    runner=subprocess.run,
    timeout: float = 300.0,
) -> BatchAdapterExecution:
    """Run one DIAMOND search over exact combined queries, then parse each record independently."""
    if type(threads) is not int or threads < 1:
        raise BatchSafetyError("DIAMOND batch threads must be a positive integer")
    selected = tuple(record_artifacts)
    execution_root = Path(work_dir)
    if execution_root.exists() or execution_root.is_symlink():
        raise BatchSafetyError(f"batch execution directory already exists: {execution_root}")
    execution_root.mkdir(parents=True)
    batched = materialize_batched_orf_inputs(selected, execution_root / "inputs")
    output_roots = _validated_record_output_roots(batched.record_ids, record_output_roots)
    try:
        database_path, _, _ = _validate_toxin_assets(manifest_section)
        validate_tool_pin(tool_pin, runner=runner, timeout=timeout)
    except (AssetProvenanceError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        return _toxin_execution(
            work_dir=execution_root,
            batched=batched,
            command=(),
            raw_output=None,
            results=_toxin_indeterminate_results(
                batched.record_ids,
                required=required,
                reason_code="TOXIN_ASSET_PROVENANCE_MISMATCH",
            ),
        )
    raw_output = execution_root / "toxin_diamond.raw.tsv"
    command = tuple(
        build_diamond_command(
            diamond=tool_pin.path,
            queries_faa=batched.artifacts.all_queries_faa,
            database=database_path,
            output_tsv=raw_output,
            threads=threads,
        )
    )
    try:
        runner(list(command), check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _toxin_execution(
            work_dir=execution_root,
            batched=batched,
            command=command,
            raw_output=raw_output if raw_output.is_file() else None,
            results=_toxin_indeterminate_results(
                batched.record_ids,
                required=required,
                reason_code="TOXIN_DIAMOND_EXECUTION_TIMEOUT",
                command=command,
                raw_output_path=raw_output,
            ),
        )
    except (subprocess.CalledProcessError, OSError):
        return _toxin_execution(
            work_dir=execution_root,
            batched=batched,
            command=command,
            raw_output=raw_output if raw_output.is_file() else None,
            results=_toxin_indeterminate_results(
                batched.record_ids,
                required=required,
                reason_code="TOXIN_DIAMOND_EXECUTION_FAILED",
                command=command,
                raw_output_path=raw_output,
            ),
        )
    if not raw_output.is_file():
        return _toxin_execution(
            work_dir=execution_root,
            batched=batched,
            command=command,
            raw_output=None,
            results=_toxin_indeterminate_results(
                batched.record_ids,
                required=required,
                reason_code="TOXIN_DIAMOND_OUTPUT_MISSING",
                command=command,
                raw_output_path=raw_output,
            ),
        )
    promoted_paths: list[Path] = []
    try:
        split_paths = split_diamond_batch_output(
            raw_output,
            batched=batched,
            output_root=execution_root / "split",
        )
        artifacts_by_id = dict(selected)
        destinations: dict[str, tuple[Path, Path]] = {}
        normalized_split_paths: dict[str, Path] = {}
        parsed_by_id: dict[str, AdapterResult] = {}
        for record_id in batched.record_ids:
            raw_destination = output_roots[record_id] / "toxin_diamond.raw.tsv"
            normalized_destination = output_roots[record_id] / "toxin_diamond.tsv"
            if any(path.exists() or path.is_symlink() for path in (raw_destination, normalized_destination)):
                raise BatchSafetyError(f"record toxin output already exists for {record_id}")
            destinations[record_id] = (raw_destination, normalized_destination)
            normalized_split_path = split_paths[record_id].with_name("toxin_diamond.tsv")
            normalized_split_paths[record_id] = normalized_split_path
            _write_normalized_header(normalized_split_path, _DIAMOND_COLUMNS, split_paths[record_id])
            parsed_by_id[record_id] = _parse_toxin_diamond_output_validated(
                normalized_split_path,
                artifacts=artifacts_by_id[record_id],
                manifest_section=manifest_section,
                tool_pin=tool_pin,
                required=required,
                policy=TOXIN_HOMOLOGY_POLICY_V2,
            )
        record_results: list[tuple[str, AdapterResult]] = []
        for record_id in batched.record_ids:
            raw_destination, normalized_destination = destinations[record_id]
            split_paths[record_id].replace(raw_destination)
            promoted_paths.append(raw_destination)
            normalized_split_paths[record_id].replace(normalized_destination)
            promoted_paths.append(normalized_destination)
            parsed = parsed_by_id[record_id]
            record_results.append(
                (
                    record_id,
                    AdapterResult(
                        class_result=parsed.class_result,
                        supplemental_findings=parsed.supplemental_findings,
                        command=command,
                        raw_output_path=str(normalized_destination),
                        raw_output_sha256=_sha256_file(normalized_destination),
                        policy_id=parsed.policy_id,
                        policy_sha256=parsed.policy_sha256,
                        shared_execution_id=execution_root.name,
                        command_output_path=str(raw_output.absolute()),
                    ),
                )
            )
        for record_id in batched.record_ids:
            (execution_root / "split" / record_id).rmdir()
        (execution_root / "split").rmdir()
    except (BatchSafetyError, KeyError, OSError, TypeError, ValueError):
        for promoted_path in promoted_paths:
            promoted_path.unlink(missing_ok=True)
        return _toxin_execution(
            work_dir=execution_root,
            batched=batched,
            command=command,
            raw_output=raw_output,
            results=_toxin_indeterminate_results(
                batched.record_ids,
                required=required,
                reason_code="TOXIN_DIAMOND_BATCH_SPLIT_MISMATCH",
                command=command,
                raw_output_path=raw_output,
            ),
        )
    return _toxin_execution(
        work_dir=execution_root,
        batched=batched,
        command=command,
        raw_output=raw_output,
        results=tuple(record_results),
    )
