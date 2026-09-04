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

"""Measure target/SFT copy risk for a calibration sweep."""

from __future__ import annotations

import argparse
import subprocess
import uuid
from pathlib import Path

import pandas as pd

from bionemo.evo2_phage_gen.calibration_scoring import load_generation_records
from bionemo.evo2_phage_gen.qc import save_fasta


SEARCH_COLUMNS = ("query", "target", "pident", "qcov", "tcov", "alnlen", "qlen", "tlen", "evalue")
IUPAC_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")
IUPAC_SYMBOLS = frozenset("ACGTRYSWKMBDHVN")


def _least_rotation(sequence: str) -> str:
    sequence = sequence.upper()
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
    """Return the rotation- and strand-invariant canonical form of circular DNA."""
    sequence = sequence.upper()
    unsupported = sorted(set(sequence) - IUPAC_SYMBOLS)
    if unsupported:
        raise ValueError(f"unsupported IUPAC symbols: {''.join(unsupported)}")
    reverse_complement = sequence.translate(IUPAC_COMPLEMENT)[::-1]
    return min(_least_rotation(sequence), _least_rotation(reverse_complement))


def _read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    record_id: str | None = None
    parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if record_id is not None:
                records.append((record_id, "".join(parts).upper()))
            record_id = line[1:].strip()
            parts = []
        elif line.strip():
            if record_id is None:
                raise ValueError(f"sequence precedes FASTA header in {path}")
            parts.append(line.strip())
    if record_id is not None:
        records.append((record_id, "".join(parts).upper()))
    if not records or any(not record_id or not sequence for record_id, sequence in records):
        raise ValueError(f"no FASTA sequences in {path}")
    return records


def _read_fasta_sequences(path: Path) -> list[str]:
    return [sequence for _, sequence in _read_fasta_records(path)]


def normalize_prompted_fasta(source: Path, output: Path) -> Path:
    """Write nucleotide payloads, stripping a two-character model control token if present."""
    records = []
    for record_id, sequence in _read_fasta_records(source):
        payload = sequence[2:] if sequence.startswith("+") else sequence
        invalid = sorted(set(payload) - set("ACGT"))
        if not payload or invalid:
            symbols = "".join(invalid) or "<empty>"
            raise ValueError(f"non-DNA payload for {record_id}: {symbols}")
        records.append((record_id, payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f">{record_id}\n{sequence}\n" for record_id, sequence in records))
    return output


def _load_sweep(generation_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((generation_root / "jsonl").glob("prefix*_temp*.jsonl")):
        cell = path.stem
        frame = load_generation_records(path)
        frame["cell"] = cell
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no completed sweep JSONL files under {generation_root / 'jsonl'}")
    sweep = pd.concat(rows, ignore_index=True)
    if sweep["id_prompt"].astype(str).duplicated().any():
        raise ValueError("duplicate IDs across calibration cells")
    return sweep


def _run_search(
    mmseqs_bin: Path,
    query_fasta: Path,
    target_fasta: Path,
    output_m8: Path,
    tmp_dir: Path,
    threads: int,
    log_path: Path,
) -> None:
    command = [
        str(mmseqs_bin),
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(output_m8),
        str(tmp_dir),
        "--search-type",
        "3",
        "--threads",
        str(threads),
        "-s",
        "7.5",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--format-output",
        ",".join(SEARCH_COLUMNS),
    ]
    with log_path.open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def _top_hits(path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        hits = pd.DataFrame(columns=SEARCH_COLUMNS)
    else:
        hits = pd.read_csv(path, sep="\t", header=None, names=SEARCH_COLUMNS)
    for column in ("pident", "qcov", "tcov", "alnlen", "qlen", "tlen", "evalue"):
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    hits = hits.sort_values(["query", "pident", "qcov", "alnlen"], ascending=[True, False, False, False])
    hits = hits.drop_duplicates("query", keep="first")
    return hits.rename(
        columns={
            "query": "id_prompt",
            **{column: f"{prefix}_{column}" for column in SEARCH_COLUMNS if column not in {"query"}},
        }
    )


def measure_novelty(
    *,
    generation_root: Path,
    reference_fasta: Path,
    sft_fasta: Path,
    tool_bin_dir: Path,
    work_dir: Path,
    output_csv: Path,
    threads: int,
) -> pd.DataFrame:
    """Measure exact and near-copy novelty against target and SFT references."""
    sweep = _load_sweep(generation_root)
    work_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = work_dir / f"attempt_{uuid.uuid4().hex}"
    attempt_dir.mkdir()
    query_fasta = attempt_dir / "sweep.fasta"
    save_fasta(sweep[["id_prompt", "sequence"]], query_fasta)
    sft_payload_fasta = normalize_prompted_fasta(
        sft_fasta,
        attempt_dir / "sft-payload.fasta",
    )
    reference_payload_fasta = normalize_prompted_fasta(
        reference_fasta,
        attempt_dir / "reference-payload.fasta",
    )
    mmseqs_bin = (tool_bin_dir / "mmseqs").resolve()

    target_m8 = attempt_dir / "target.m8"
    sft_m8 = attempt_dir / "sft.m8"
    _run_search(
        mmseqs_bin,
        query_fasta,
        reference_payload_fasta,
        target_m8,
        attempt_dir / "target-tmp",
        threads,
        attempt_dir / "target-search.log",
    )
    _run_search(
        mmseqs_bin,
        query_fasta,
        sft_payload_fasta,
        sft_m8,
        attempt_dir / "sft-tmp",
        threads,
        attempt_dir / "sft-search.log",
    )

    target_hashes = {
        canonical_circular_sequence(sequence) for sequence in _read_fasta_sequences(reference_payload_fasta)
    }
    sft_hashes = {canonical_circular_sequence(sequence) for sequence in _read_fasta_sequences(sft_payload_fasta)}
    canonical = sweep["sequence"].map(canonical_circular_sequence)
    metrics = sweep[["id_prompt", "cell"]].copy()
    metrics["exact_target_circular_or_revcomp"] = canonical.isin(target_hashes).astype(float)
    metrics["exact_sft_circular_or_revcomp"] = canonical.isin(sft_hashes).astype(float)
    metrics = metrics.merge(_top_hits(target_m8, "target"), on="id_prompt", how="left")
    metrics = metrics.merge(_top_hits(sft_m8, "sft"), on="id_prompt", how="left")
    metrics["target_near_copy_98_9pct"] = (metrics["target_pident"].fillna(0.0) >= 98.9).astype(float)
    metrics["sft_near_copy_98_9pct"] = (metrics["sft_pident"].fillna(0.0) >= 98.9).astype(float)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)
    return metrics


def summarize_novelty(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact- and near-copy rates for each calibration cell."""
    rows = []
    for cell, group in metrics.groupby("cell", sort=True):
        rows.append(
            {
                "cell": cell,
                "exact_target_copy_rate": float(group["exact_target_circular_or_revcomp"].mean()),
                "exact_sft_copy_rate": float(group["exact_sft_circular_or_revcomp"].mean()),
                "target_near_copy_rate": float(group["target_near_copy_98_9pct"].mean()),
                "sft_near_copy_rate": float(group["sft_near_copy_98_9pct"].mean()),
                "target_pident_mean": float(pd.to_numeric(group["target_pident"], errors="coerce").fillna(0).mean()),
                "sft_pident_mean": float(pd.to_numeric(group["sft_pident"], errors="coerce").fillna(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def validate_novelty_file(path: Path, expected_records: int) -> None:
    """Validate the novelty record count and unique prompt identifiers."""
    metrics = pd.read_csv(path)
    if len(metrics) != expected_records:
        raise ValueError(f"{path}: expected {expected_records} records, found {len(metrics)}")
    if "id_prompt" not in metrics:
        raise ValueError(f"{path}: missing id_prompt column")
    if metrics["id_prompt"].astype(str).duplicated().any():
        raise ValueError(f"{path}: duplicate id_prompt values")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure")
    measure.add_argument("--generation-root", type=Path, required=True)
    measure.add_argument("--reference-fasta", type=Path, required=True)
    measure.add_argument("--sft-fasta", type=Path, required=True)
    measure.add_argument("--tool-bin-dir", type=Path, required=True)
    measure.add_argument("--work-dir", type=Path, required=True)
    measure.add_argument("--output-csv", type=Path, required=True)
    measure.add_argument("--threads", type=int, default=32)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--metrics-csv", type=Path, required=True)
    validate.add_argument("--expected-records", type=int, required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--metrics-csv", type=Path, required=True)
    summarize.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the novelty measurement, validation, or summarization command."""
    args = _parse_args()
    if args.command == "measure":
        measure_novelty(
            generation_root=args.generation_root,
            reference_fasta=args.reference_fasta,
            sft_fasta=args.sft_fasta,
            tool_bin_dir=args.tool_bin_dir,
            work_dir=args.work_dir,
            output_csv=args.output_csv,
            threads=max(1, args.threads),
        )
        print(args.output_csv)
        return
    if args.command == "validate":
        validate_novelty_file(args.metrics_csv, args.expected_records)
        print(args.metrics_csv)
        return
    summary = summarize_novelty(pd.read_csv(args.metrics_csv))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()
