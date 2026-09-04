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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Dependency-light evidence helpers for the final PhiX174 rollout."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


_IUPAC_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")
_IUPAC_SYMBOLS = frozenset("ACGTRYSWKMBDHVN")
_UNAMBIGUOUS_DNA = frozenset("ACGT")
_ARC_COUNT_FILE_KEYS = (
    "nucleotide_filter_counts_file_save_location",
    "orf_filter_counts_file_save_location",
    "homology_filter_counts_file_save_location",
    "diversification_filter_counts_file_save_location",
    "synteny_filter_counts_file_save_location",
)
_ARC_WATERFALL_COLUMNS = (
    "count_initial_before_nucleotide_metrics",
    "count_nt_filter",
    "count_genome_len_filter",
    "count_gc_filter",
    "count_nt_homopolymer_filter",
    "count_initial_before_orf_metrics",
    "count_orf_count_filter",
    "count_orf_len_filter",
    "count_coding_density_filter",
    "count_aa_homopolymer_len_filter",
    "count_initial_before_homology_metrics",
    "count_protein_database_hit_count_filter",
    "count_training_data_sequence_identity_filter",
    "count_checkv_quality_filter",
    "count_seq_ident_to_reference_genome_filter",
    "count_genetic_architecture_score_filter",
    "count_tropism_protein_sequence_identity_filter",
    "count_initial_before_diversification_metrics",
    "count_mmseqs_reference_genome_sequence_identity_remove_filter",
    "count_genetic_architecture_score_remove_filter",
    "count_average_protein_sequence_identity_filter",
    "count_required_genes_filter",
    "count_syntenic_gene_count_filter",
)
_WORKFLOW_ORDER = (
    "raw_generation",
    "exact_circular_reverse_complement_deduplication",
    "safety_and_target_hard_qc",
    "post_qc_mmseqs_99pct_clustering",
    "ranking",
)


def read_fasta(path: Path, *, allow_empty: bool = False) -> list[tuple[str, str]]:
    """Read complete FASTA records while preserving source order."""
    path = Path(path)
    records: list[tuple[str, str]] = []
    record_id: str | None = None
    sequence_parts: list[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if record_id is not None:
                records.append((record_id, "".join(sequence_parts).upper()))
            description = line[1:].strip()
            record_id = description.split(maxsplit=1)[0] if description else ""
            sequence_parts = []
        else:
            if record_id is None:
                raise ValueError(f"sequence precedes FASTA header at {path}:{line_number}")
            sequence_parts.append(line)
    if record_id is not None:
        records.append((record_id, "".join(sequence_parts).upper()))
    if not records:
        if allow_empty:
            return []
        raise ValueError(f"no complete FASTA records found: {path}")
    if any(not record_id or not sequence for record_id, sequence in records):
        raise ValueError(f"incomplete FASTA record in {path}")
    identifiers = [record_id for record_id, _ in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate FASTA identifiers in {path}")
    return records


def _wrap_sequence(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[index : index + width] for index in range(0, len(sequence), width))


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">{record_id}\n{_wrap_sequence(sequence)}\n" for record_id, sequence in records))
    return path


def _least_rotation(sequence: str) -> str:
    if not sequence:
        return ""
    doubled = sequence + sequence
    first, second, offset = 0, 1, 0
    length = len(sequence)
    while first < length and second < length and offset < length:
        left, right = doubled[first + offset], doubled[second + offset]
        if left == right:
            offset += 1
            continue
        if left > right:
            first = first + offset + 1
            if first == second:
                first += 1
        else:
            second = second + offset + 1
            if first == second:
                second += 1
        offset = 0
    start = min(first, second)
    return doubled[start : start + length]


def canonical_circular_sequence(sequence: str) -> str:
    """Return a rotation- and strand-invariant representation of circular DNA."""
    sequence = sequence.upper()
    unsupported = sorted(set(sequence) - _IUPAC_SYMBOLS)
    if unsupported:
        raise ValueError(f"unsupported IUPAC symbols: {''.join(unsupported)}")
    reverse_complement = sequence.translate(_IUPAC_COMPLEMENT)[::-1]
    return min(_least_rotation(sequence), _least_rotation(reverse_complement))


def deduplicate_fasta(
    source_fasta: Path,
    representative_fasta: Path,
    mapping_csv: Path,
    report_json: Path,
) -> Path:
    """Keep the first exact/circular/reverse-complement representative in generation order."""
    records = read_fasta(source_fasta)
    exact_representative: dict[str, str] = {}
    circular_representative: dict[str, str] = {}
    representatives: list[tuple[str, str]] = []
    mapping_rows: list[dict[str, Any]] = []
    for index, (record_id, sequence) in enumerate(records):
        representative_id = exact_representative.get(sequence)
        duplicate_reason = "exact" if representative_id is not None else ""
        canonical = canonical_circular_sequence(sequence) if set(sequence) <= _UNAMBIGUOUS_DNA else None
        if representative_id is None and canonical is not None:
            representative_id = circular_representative.get(canonical)
            if representative_id is not None:
                duplicate_reason = "circular_or_reverse_complement"
        if representative_id is None:
            representative_id = record_id
            representatives.append((record_id, sequence))
            if canonical is not None:
                circular_representative[canonical] = record_id
        exact_representative[sequence] = representative_id
        mapping_rows.append(
            {
                "raw_index": index,
                "record_id": record_id,
                "representative_id": representative_id,
                "is_representative": str(record_id == representative_id).lower(),
                "duplicate_reason": duplicate_reason,
                "length_nt": len(sequence),
            }
        )

    _write_fasta(representative_fasta, representatives)
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    with mapping_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(mapping_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(mapping_rows)
    reasons = Counter(row["duplicate_reason"] for row in mapping_rows if row["duplicate_reason"])
    report = {
        "schema_version": 1,
        "state": "succeeded",
        "order": "generation order; first biological occurrence is the representative",
        "canonicalization": "exact, then least circular rotation across forward and reverse-complement DNA",
        "counts": {
            "raw_records": len(records),
            "representative_records": len(representatives),
            "exact_duplicates_removed": reasons["exact"],
            "circular_or_reverse_complement_duplicates_removed": reasons["circular_or_reverse_complement"],
        },
        "artifacts": {
            "source_fasta": str(Path(source_fasta).resolve()),
            "representative_fasta": str(Path(representative_fasta).resolve()),
            "mapping_csv": str(Path(mapping_csv).resolve()),
        },
        "note": "Non-ACGT records are eligible for exact deduplication only and remain subject to hard QC.",
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    return representative_fasta


def _single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one count row in {path}, found {len(rows)}")
    return rows[0]


def _integer_count(value: str, *, field: str, path: Path) -> int:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"non-numeric count {field}={value!r} in {path}") from error
    if not math.isfinite(parsed) or parsed < 0 or not parsed.is_integer():
        raise ValueError(f"invalid count {field}={value!r} in {path}")
    return int(parsed)


def summarize_arc_screen(
    config_path: Path,
    input_fasta: Path,
    output_json: Path,
    *,
    expected_filter7: bool,
) -> Path:
    """Validate one Arc branch and write its concise hard-QC waterfall."""
    import yaml

    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Arc config is not a mapping: {config_path}")
    if bool(config.get("genetic_architecture_remove_filter")) is not expected_filter7:
        raise ValueError("Arc filter-7 setting differs from the requested branch")
    if bool(config.get("mmseqs_clustering_filter")):
        raise ValueError("Arc internal MMseqs clustering must be disabled before hard QC")
    results_dir = Path(str(config["results_save_dir"]))
    if not results_dir.is_absolute():
        results_dir = (config_path.parent / results_dir).resolve()
    count_values: dict[str, int] = {}
    count_files: list[str] = []
    for config_key in _ARC_COUNT_FILE_KEYS:
        filename = config.get(config_key)
        if not filename:
            raise ValueError(f"Arc config lacks {config_key}")
        count_path = results_dir / str(filename)
        if not count_path.is_file():
            raise FileNotFoundError(count_path)
        count_files.append(str(count_path.resolve()))
        for field, raw_value in _single_csv_row(count_path).items():
            if not field.startswith("count_") or raw_value in (None, ""):
                continue
            value = _integer_count(raw_value, field=field, path=count_path)
            previous = count_values.get(field)
            if previous is not None and previous != value:
                raise ValueError(f"Arc count {field} disagrees across stage files: {previous} != {value}")
            count_values[field] = value

    input_count = len(read_fasta(input_fasta, allow_empty=True))
    initial_count = count_values.get("count_initial_before_nucleotide_metrics")
    if initial_count is not None and initial_count != input_count:
        raise ValueError(f"Arc initial count {initial_count} differs from input FASTA count {input_count}")
    terminal_name = config.get("synteny_filter_seqs_fasta_file_save_location")
    if not terminal_name:
        raise ValueError("Arc config lacks synteny_filter_seqs_fasta_file_save_location")
    terminal_fasta = results_dir / str(terminal_name)
    if not terminal_fasta.is_file():
        raise FileNotFoundError(terminal_fasta)
    final_count = len(read_fasta(terminal_fasta, allow_empty=True))
    waterfall: list[dict[str, int | str]] = []
    previous_count = input_count
    for field in _ARC_WATERFALL_COLUMNS:
        if field not in count_values:
            continue
        count = count_values[field]
        if count > previous_count:
            raise ValueError(f"non-monotonic Arc waterfall at {field}: {count} > {previous_count}")
        waterfall.append({"stage": field, "count": count})
        previous_count = count
    if waterfall and waterfall[-1]["count"] != final_count:
        raise ValueError("Arc terminal FASTA count differs from the last available waterfall count")

    report = {
        "schema_version": 1,
        "state": "succeeded",
        "branch": "diagnostic-filter7-on" if expected_filter7 else "target-filter7-off",
        "architecture_removal_filter_7": expected_filter7,
        "arc_internal_mmseqs_clustering": False,
        "input_representatives": input_count,
        "final_pass_count": final_count,
        "final_pass_rate": final_count / input_count if input_count else 0.0,
        "waterfall": waterfall,
        "artifacts": {
            "config": str(config_path.resolve()),
            "count_files": count_files,
            "terminal_fasta": str(terminal_fasta.resolve()),
        },
        "limitations": "This branch is computational hard-QC evidence, not proof of bootability or viability.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    return output_json


def _safety_states(safety_manifest: Path, expected_ids: set[str]) -> tuple[dict[str, str], Counter[str]]:
    payload = json.loads(Path(safety_manifest).read_text())
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("safety manifest lacks records")
    states: dict[str, str] = {}
    for row in records:
        record_id = str(row["record_id"])
        state = str(row["state"]).upper()
        if record_id in states:
            raise ValueError(f"duplicate safety record: {record_id}")
        if state not in {"PASS", "FAIL", "INDETERMINATE"}:
            raise ValueError(f"unknown safety state for {record_id}: {state}")
        states[record_id] = state
    if set(states) != expected_ids:
        missing = sorted(expected_ids - set(states))
        extra = sorted(set(states) - expected_ids)
        raise ValueError(f"safety/input mismatch: missing={missing}, extra={extra}")
    return states, Counter(states.values())


def _reconcile_safety_input(
    representatives: list[tuple[str, str]],
    safety_manifest: Path,
    safety_input_fasta: Path | None,
) -> tuple[dict[str, str], Counter[str], set[str]]:
    representative_by_id = dict(representatives)
    safety_input = representatives if safety_input_fasta is None else read_fasta(safety_input_fasta, allow_empty=True)
    safety_input_by_id = dict(safety_input)
    unknown = sorted(set(safety_input_by_id) - set(representative_by_id))
    sequence_mismatches = sorted(
        record_id
        for record_id, sequence in safety_input
        if record_id in representative_by_id and representative_by_id[record_id] != sequence
    )
    if unknown or sequence_mismatches:
        raise ValueError(
            f"safety input/representative mismatch: unknown={unknown}, sequence_mismatches={sequence_mismatches}"
        )
    states, counts = _safety_states(safety_manifest, set(safety_input_by_id))
    excluded = set(representative_by_id) - set(safety_input_by_id)
    return states, counts, excluded


def _canonical_representatives(
    representatives: list[tuple[str, str]],
    *,
    allow_unsupported_ids: set[str],
) -> dict[str, str | None]:
    canonical_by_id: dict[str, str | None] = {}
    for record_id, sequence in representatives:
        try:
            canonical_by_id[record_id] = canonical_circular_sequence(sequence)
        except ValueError:
            if record_id not in allow_unsupported_ids:
                raise
            canonical_by_id[record_id] = None
    return canonical_by_id


def select_hard_qc_passers(
    representative_fasta: Path,
    safety_manifest: Path,
    target_fasta: Path,
    output_fasta: Path,
    report_json: Path,
    *,
    safety_input_fasta: Path | None = None,
) -> Path:
    """Intersect safety-PASS representatives with the target Arc hard-QC branch."""
    representatives = read_fasta(representative_fasta, allow_empty=True)
    safety_by_id, state_counts, pre_safety_excluded = _reconcile_safety_input(
        representatives,
        safety_manifest,
        safety_input_fasta,
    )
    target_sequences = {
        canonical_circular_sequence(sequence) for _, sequence in read_fasta(target_fasta, allow_empty=True)
    }
    representative_canonical = _canonical_representatives(
        representatives,
        allow_unsupported_ids=pre_safety_excluded,
    )
    target_pass_ids = {
        record_id for record_id, canonical in representative_canonical.items() if canonical in target_sequences
    }
    hard_qc_records = [
        (record_id, sequence)
        for record_id, sequence in representatives
        if safety_by_id.get(record_id) == "PASS" and record_id in target_pass_ids
    ]
    _write_fasta(output_fasta, hard_qc_records)
    report = {
        "schema_version": 1,
        "state": "succeeded",
        "input_representatives": len(representatives),
        "safety_input_representatives": len(representatives) - len(pre_safety_excluded),
        "pre_safety_qc_excluded_representatives": len(pre_safety_excluded),
        "safety_states": {state: state_counts[state] for state in ("PASS", "FAIL", "INDETERMINATE")},
        "target_profile_pass": len(target_pass_ids),
        "hard_qc_pass": len(hard_qc_records),
        "definition": "safety PASS intersected with the target-profile Arc terminal pass set",
        "artifacts": {
            "representative_fasta": str(Path(representative_fasta).resolve()),
            "safety_input_fasta": str(Path(safety_input_fasta or representative_fasta).resolve()),
            "safety_manifest": str(Path(safety_manifest).resolve()),
            "target_fasta": str(Path(target_fasta).resolve()),
            "hard_qc_fasta": str(Path(output_fasta).resolve()),
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    return output_fasta


def cluster_post_qc_fasta(
    input_fasta: Path,
    representative_fasta: Path,
    memberships_csv: Path,
    report_json: Path,
    *,
    work_dir: Path,
    mmseqs_bin: str | Path = "mmseqs",
    threads: int = 16,
) -> Path:
    """Cluster hard-QC passers at the pinned final-order MMseqs contract."""
    if threads < 1:
        raise ValueError("threads must be positive")
    records = read_fasta(input_fasta, allow_empty=True)
    identifiers = [record_id for record_id, _ in records]
    if any(set(sequence) - _UNAMBIGUOUS_DNA for _, sequence in records):
        raise ValueError("post-QC clustering requires unambiguous DNA")
    mmseqs = str(mmseqs_bin)
    version = subprocess.run(
        [mmseqs, "version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    database = work_dir / "sequences"
    clusters = work_dir / "clusters"
    temporary = work_dir / "tmp"
    cluster_tsv = work_dir / "clusters.tsv"
    commands = [
        [mmseqs, "createdb", str(Path(input_fasta)), str(database)],
        [
            mmseqs,
            "cluster",
            str(database),
            str(clusters),
            str(temporary),
            "--min-seq-id",
            "0.99",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--cluster-mode",
            "0",
            "--threads",
            str(threads),
        ],
        [mmseqs, "createtsv", str(database), str(database), str(clusters), str(cluster_tsv)],
    ]
    memberships: list[tuple[str, str]] = []
    if records:
        for command in commands:
            subprocess.run(command, check=True)
        for line_number, line in enumerate(cluster_tsv.read_text().splitlines(), start=1):
            fields = line.split("\t")
            if len(fields) != 2 or not all(fields):
                raise ValueError(f"malformed MMseqs membership at {cluster_tsv}:{line_number}")
            memberships.append((fields[0], fields[1]))
    member_ids = [member_id for _, member_id in memberships]
    if len(member_ids) != len(set(member_ids)) or set(member_ids) != set(identifiers):
        raise ValueError("MMseqs memberships must cover every hard-QC candidate exactly once")
    representative_ids = {representative_id for representative_id, _ in memberships}
    if not representative_ids <= set(identifiers):
        raise ValueError("MMseqs returned an unknown cluster representative")
    if not representative_ids <= set(member_ids):
        raise ValueError("MMseqs cluster representative is not a cluster member")
    source_order = {record_id: index for index, record_id in enumerate(identifiers)}
    memberships.sort(key=lambda row: (source_order[row[0]], source_order[row[1]]))
    representative_records = [record for record in records if record[0] in representative_ids]
    _write_fasta(representative_fasta, representative_records)
    memberships_csv.parent.mkdir(parents=True, exist_ok=True)
    with memberships_csv.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("representative_id", "member_id"))
        writer.writerows(memberships)
    report = {
        "schema_version": 1,
        "state": "succeeded",
        "order": "all safety and target hard QC before nucleotide clustering",
        "counts": {
            "hard_qc_passers": len(records),
            "clusters": len(representative_records),
            "duplicates_removed": len(records) - len(representative_records),
        },
        "mmseqs": {
            "version": version,
            "min_sequence_identity": 0.99,
            "coverage": 0.8,
            "coverage_mode": 0,
            "cluster_mode": 0,
            "threads": threads,
        },
        "commands": commands,
        "artifacts": {
            "input_fasta": str(Path(input_fasta).resolve()),
            "representative_fasta": str(Path(representative_fasta).resolve()),
            "memberships_csv": str(Path(memberships_csv).resolve()),
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    return representative_fasta


def _load_mapping(path: Path, raw_ids: list[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {str(row["record_id"]): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(raw_ids):
        raise ValueError("deduplication mapping must contain every raw identifier exactly once")
    representatives = [
        record_id for record_id in raw_ids if by_id[record_id].get("is_representative", "").lower() == "true"
    ]
    representative_set = set(representatives)
    if any(row["representative_id"] not in representative_set for row in rows):
        raise ValueError("deduplication mapping refers to a non-representative record")
    return by_id, representatives


def _load_likelihoods(path: Path, raw_ids: list[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    score_ids = [str(row["record_id"]) for row in rows]
    if len(score_ids) != len(set(score_ids)) or set(score_ids) != set(raw_ids):
        raise ValueError("likelihood table must contain every raw identifier exactly once")
    for row in rows:
        for field in ("total_log_probability", "mean_log_probability_per_nucleotide"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"non-finite {field} for {row['record_id']}")
    return sorted(rows, key=lambda row: (-float(row["mean_log_probability_per_nucleotide"]), row["record_id"]))


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered) * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered, strict=True)) / denominator


def _spearman(lengths: list[int], scores: list[float]) -> tuple[float | None, float | None]:
    if len(set(lengths)) < 2 or len(set(scores)) < 2:
        return None, None
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return _pearson(_average_ranks([float(value) for value in lengths]), _average_ranks(scores)), None
    result = spearmanr(lengths, scores)
    return float(result.statistic), float(result.pvalue)


def _load_optional_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    if payload.get("state") != "succeeded":
        raise ValueError(f"nonterminal evidence report: {path}")
    return payload


def _safety_provenance(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    return {
        "manifest": str(Path(path).resolve()),
        "schema_version": payload.get("schema_version"),
        "manifest_type": payload.get("manifest_type"),
        "policy": payload.get("policy"),
        "asset_state": payload.get("asset_state"),
        "resolved_profile": payload.get("resolved_profile"),
        "tools": payload.get("tools"),
        "databases": payload.get("databases"),
        "execution": payload.get("execution"),
    }


def finalize_rollout_report(
    generated_fasta: Path,
    deduplication_mapping_csv: Path,
    safety_manifest: Path,
    target_fasta: Path,
    diagnostic_fasta: Path,
    likelihood_csv: Path,
    cluster_representative_fasta: Path,
    cluster_memberships_csv: Path,
    output_json: Path,
    accepted_fasta: Path,
    summary_path: Path,
    *,
    model_checkpoint: str,
    rl_checkpoint: str,
    sampling_selection: Path | None = None,
    deduplication_report: Path | None = None,
    hard_qc_report: Path | None = None,
    target_report: Path | None = None,
    diagnostic_report: Path | None = None,
    clustering_report: Path | None = None,
    run_log: Path | None = None,
    safety_input_fasta: Path | None = None,
) -> Path:
    """Reconcile raw, representative, hard-QC, clustering, and ranking evidence."""
    raw_records = read_fasta(generated_fasta)
    raw_ids = [record_id for record_id, _ in raw_records]
    sequence_by_id = dict(raw_records)
    generation_order = {record_id: index for index, record_id in enumerate(raw_ids)}
    mapping_by_id, representative_ids = _load_mapping(deduplication_mapping_csv, raw_ids)
    representative_records = [(record_id, sequence_by_id[record_id]) for record_id in representative_ids]
    safety_by_id, safety_counts, pre_safety_excluded = _reconcile_safety_input(
        representative_records,
        safety_manifest,
        safety_input_fasta,
    )
    safety_state_by_representative = {
        record_id: safety_by_id.get(record_id, "NOT_SCREENED_PRE_SAFETY_QC") for record_id in representative_ids
    }
    target_sequences = {
        canonical_circular_sequence(sequence) for _, sequence in read_fasta(target_fasta, allow_empty=True)
    }
    diagnostic_sequences = {
        canonical_circular_sequence(sequence) for _, sequence in read_fasta(diagnostic_fasta, allow_empty=True)
    }
    representative_canonical = _canonical_representatives(
        representative_records,
        allow_unsupported_ids=pre_safety_excluded,
    )
    target_pass_ids = {
        record_id for record_id, canonical in representative_canonical.items() if canonical in target_sequences
    }
    diagnostic_pass_ids = {
        record_id for record_id, canonical in representative_canonical.items() if canonical in diagnostic_sequences
    }
    hard_qc_ids = {record_id for record_id in target_pass_ids if safety_by_id.get(record_id) == "PASS"}

    with Path(cluster_memberships_csv).open(newline="") as handle:
        membership_rows = list(csv.DictReader(handle))
    if any({"representative_id", "member_id"} - set(row) for row in membership_rows):
        raise ValueError("cluster membership table lacks required columns")
    member_ids = [str(row["member_id"]) for row in membership_rows]
    if len(member_ids) != len(set(member_ids)) or set(member_ids) != hard_qc_ids:
        raise ValueError("cluster memberships must cover every hard-QC representative exactly once")
    cluster_by_member = {str(row["member_id"]): str(row["representative_id"]) for row in membership_rows}
    cluster_representatives = [
        record_id for record_id, _ in read_fasta(cluster_representative_fasta, allow_empty=True)
    ]
    if set(cluster_representatives) != set(cluster_by_member.values()):
        raise ValueError("cluster representative FASTA and membership table disagree")

    score_rows = _load_likelihoods(likelihood_csv, raw_ids)
    score_by_id = {str(row["record_id"]): row for row in score_rows}
    lengths = [len(sequence) for _, sequence in raw_records]
    scores = [float(score_by_id[record_id]["mean_log_probability_per_nucleotide"]) for record_id in raw_ids]
    rho, p_value = _spearman(lengths, scores)
    informative_scores = len(set(scores)) > 1
    strong_length_association = rho is not None and abs(rho) >= 0.5
    apply_likelihood_order = informative_scores and not strong_length_association
    if apply_likelihood_order:
        accepted_ids = [
            str(row["record_id"]) for row in score_rows if str(row["record_id"]) in set(cluster_representatives)
        ]
    else:
        accepted_ids = sorted(cluster_representatives, key=generation_order.__getitem__)
    accepted_rank_by_id = {record_id: index for index, record_id in enumerate(accepted_ids, start=1)}
    likelihood_rank_by_id = {str(row["record_id"]): index for index, row in enumerate(score_rows, start=1)}

    report_rows: list[dict[str, Any]] = []
    for record_id in sorted(raw_ids, key=likelihood_rank_by_id.__getitem__):
        mapping = mapping_by_id[record_id]
        representative_id = mapping["representative_id"]
        is_representative = record_id == representative_id
        score = score_by_id[record_id]
        report_rows.append(
            {
                "likelihood_rank": likelihood_rank_by_id[record_id],
                "accepted_rank": accepted_rank_by_id.get(record_id),
                "record_id": record_id,
                "representative_id": representative_id,
                "is_biological_representative": is_representative,
                "duplicate_reason": mapping.get("duplicate_reason") or None,
                "sequence": sequence_by_id[record_id],
                "length_nt": len(sequence_by_id[record_id]),
                "scored_nucleotides": int(score["scored_nucleotides"]),
                "total_log_probability": float(score["total_log_probability"]),
                "mean_log_probability_per_nucleotide": float(score["mean_log_probability_per_nucleotide"]),
                "safety_state": (
                    safety_state_by_representative[record_id] if is_representative else "NOT_EVALUATED_DUPLICATE"
                ),
                "representative_safety_state": safety_state_by_representative[representative_id],
                "target_profile_pass": record_id in target_pass_ids if is_representative else None,
                "diagnostic_filter7_pass": record_id in diagnostic_pass_ids if is_representative else None,
                "hard_qc_pass": record_id in hard_qc_ids if is_representative else None,
                "post_qc_cluster_representative_id": cluster_by_member.get(record_id),
                "accepted": record_id in accepted_rank_by_id,
            }
        )

    _write_fasta(accepted_fasta, [(record_id, sequence_by_id[record_id]) for record_id in accepted_ids])
    duplicate_count = len(raw_ids) - len(representative_ids)
    sampling_payload: dict[str, Any] | None = None
    if sampling_selection is not None:
        import yaml

        sampling_payload = yaml.safe_load(Path(sampling_selection).read_text())
        if not isinstance(sampling_payload, dict):
            raise ValueError("sampling selection must be a YAML mapping")
    evidence = {
        "deduplication": _load_optional_report(deduplication_report),
        "hard_qc": _load_optional_report(hard_qc_report),
        "target_profile": _load_optional_report(target_report),
        "diagnostic_filter7": _load_optional_report(diagnostic_report),
        "post_qc_clustering": _load_optional_report(clustering_report),
    }
    evidence = {name: value for name, value in evidence.items() if value is not None}
    counts = {
        "raw_generated": len(raw_ids),
        "raw_likelihood_scored": len(score_rows),
        "biological_representatives": len(representative_ids),
        "duplicates_removed": duplicate_count,
        "safety_input_representatives": len(representative_ids) - len(pre_safety_excluded),
        "pre_safety_qc_excluded_representatives": len(pre_safety_excluded),
        "safety_pass_representatives": safety_counts["PASS"],
        "safety_fail_representatives": safety_counts["FAIL"],
        "safety_indeterminate_representatives": safety_counts["INDETERMINATE"],
        "target_profile_pass_representatives": len(target_pass_ids),
        "diagnostic_filter7_pass_representatives": len(diagnostic_pass_ids),
        "hard_qc_pass_representatives": len(hard_qc_ids),
        "post_qc_99pct_clusters": len(cluster_representatives),
        "accepted_cluster_representatives": len(accepted_ids),
    }
    report = {
        "schema_version": 2,
        "state": "succeeded",
        "scope": "computational PhiX174 whole-genome rollout; no wet-lab viability claim",
        "workflow_order": list(_WORKFLOW_ORDER),
        "checkpoints": {"selected_rl": rl_checkpoint, "selected_pre_rl_sft_for_likelihood": model_checkpoint},
        "sampling_selection": sampling_payload,
        "artifacts": {
            "raw_generated_fasta": str(Path(generated_fasta).resolve()),
            "deduplication_mapping": str(Path(deduplication_mapping_csv).resolve()),
            "raw_sft_likelihoods": str(Path(likelihood_csv).resolve()),
            "safety_input_fasta": (
                str(Path(safety_input_fasta).resolve()) if safety_input_fasta is not None else None
            ),
            "target_profile_terminal_fasta": str(Path(target_fasta).resolve()),
            "filter7_diagnostic_terminal_fasta": str(Path(diagnostic_fasta).resolve()),
            "post_qc_cluster_memberships": str(Path(cluster_memberships_csv).resolve()),
            "accepted_cluster_representatives": str(Path(accepted_fasta).resolve()),
        },
        "counts": counts,
        "ranking": {
            "score": "mean_log_probability_per_nucleotide",
            "conditioning_prefix": "+~",
            "applied_to_accepted_candidate_order": apply_likelihood_order,
            "residual_length_association": {
                "method": "Spearman correlation across all raw generated designs",
                "n": len(raw_ids),
                "spearman_rho": rho,
                "p_value": p_value,
                "strong_correlation_threshold_abs_rho": 0.5,
                "strong_correlation": strong_length_association,
            },
            "interpretation": "Within-protocol prioritization only; not a bootability probability or threshold.",
        },
        "sequence_safety_provenance": _safety_provenance(safety_manifest),
        "evidence": evidence,
        "run_log": str(Path(run_log).resolve()) if run_log is not None else None,
        "limitations": [
            "Computational screening does not establish bootability, host range, therapeutic safety, or efficacy.",
            "FAIL and INDETERMINATE safety results are distinct; only safety PASS can enter hard-QC clustering.",
            "The filter-7 branch is diagnostic and does not replace the target profile.",
        ],
        "records": report_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "# PhiX174 run summary\n\n"
        f"- Raw generated and SFT-likelihood scored: {counts['raw_generated']}\n"
        f"- Biological representatives after exact/circular/RC deduplication: {counts['biological_representatives']}\n"
        f"- Representatives submitted to safety / excluded by pre-safety QC: "
        f"{counts['safety_input_representatives']} / {counts['pre_safety_qc_excluded_representatives']}\n"
        f"- Safety states (PASS / FAIL / INDETERMINATE): {counts['safety_pass_representatives']} / "
        f"{counts['safety_fail_representatives']} / {counts['safety_indeterminate_representatives']}\n"
        f"- Safety-PASS target hard-QC representatives: {counts['hard_qc_pass_representatives']}\n"
        f"- Post-QC 99%-identity clusters and accepted representatives: {counts['post_qc_99pct_clusters']}\n\n"
        "Order: raw generation → biological deduplication → safety and hard QC → post-QC clustering → ranking.\n\n"
        "Likelihood is a within-protocol ranking signal, not a universal bootability threshold. "
        "Computational candidates are not evidence of wet-lab viability or safety.\n"
    )
    return output_json
