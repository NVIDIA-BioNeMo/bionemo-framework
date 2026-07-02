#!/usr/bin/env python
"""Score paper-style Evo2 phage HPO generation outputs.

This script is intentionally resumable and file-oriented so the README can point
to one command for reproducing the fast, dependency-light scoring pass.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO

from bionemo.evo2_phage_gen.qc import save_fasta, trim_at_first_eos
from bionemo.evo2_phage_gen.reward import score_nucleotide_metrics


DNA_ALPHABET = frozenset("ACGTacgt")
CELL_RE = re.compile(r"phix174_prompt(?P<prompt_len>\d+)_temp(?P<temperature>[0-9.]+)$")


def _sequence_before_eos(sequence: Any) -> str:
    return trim_at_first_eos(str(sequence).replace("\n", "").strip())


def _prompt_nucleotides(prompt: Any) -> str:
    return "".join(char for char in _sequence_before_eos(prompt) if char in DNA_ALPHABET)


def _parse_cell(stem: str) -> tuple[int | None, float | None]:
    match = CELL_RE.match(stem)
    if not match:
        return None, None
    return int(match.group("prompt_len")), float(match.group("temperature"))


def _load_generation_records(jsonl_path: Path, target_records: int | None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load unique generation records, optionally capped to the manifest target count."""
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    json_records = 0
    malformed_records = 0
    duplicate_records = 0

    with jsonl_path.open(errors="ignore") as handle:
        for line_idx, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_records += 1
                continue
            json_records += 1
            record_id = str(record.get("id") or f"{jsonl_path.stem}_{line_idx:06d}")
            if record_id in seen_ids:
                duplicate_records += 1
                continue
            seen_ids.add(record_id)

            sequence = (
                _prompt_nucleotides(record.get("prompt", ""))
                + _sequence_before_eos(record.get("completion", ""))
            ).upper()
            rows.append({"id_prompt": record_id, "sequence": sequence})
            if target_records is not None and len(rows) >= target_records:
                break

    stats = {
        "json_records_seen": json_records,
        "unique_records_scored": len(rows),
        "duplicate_records_skipped": duplicate_records,
        "malformed_records_skipped": malformed_records,
    }
    return pd.DataFrame(rows, columns=["id_prompt", "sequence"]), stats


def _fasta_record_count(fasta_path: Path) -> int | None:
    if not fasta_path.exists():
        return None
    return sum(1 for _ in SeqIO.parse(str(fasta_path), "fasta"))


def _summarize_cell(cell: str, prompt_len: int | None, temperature: float | None, scored_df: pd.DataFrame) -> dict[str, Any]:
    records = len(scored_df)
    length_pass = scored_df["genome_length"].between(4000, 6000)
    gc_pass = scored_df["gc_content"].between(30.0, 65.0)
    homopolymer_pass = scored_df["max_nt_homopolymer_length"] <= 10
    return {
        "cell": cell,
        "prompt_len": prompt_len,
        "temperature": temperature,
        "records": records,
        "valid_nt_chars_pass": int(scored_df["valid_nt_chars"].sum()),
        "length_pass": int(length_pass.sum()),
        "gc_pass": int(gc_pass.sum()),
        "homopolymer_pass": int(homopolymer_pass.sum()),
        "nucleotide_pass": int(scored_df["reward_nucleotide_pass"].sum()),
        "valid_nt_chars_rate": float(scored_df["valid_nt_chars"].mean()) if records else 0.0,
        "length_pass_rate": float(length_pass.mean()) if records else 0.0,
        "gc_pass_rate": float(gc_pass.mean()) if records else 0.0,
        "homopolymer_pass_rate": float(homopolymer_pass.mean()) if records else 0.0,
        "nucleotide_pass_rate": float(scored_df["reward_nucleotide_pass"].mean()) if records else 0.0,
        "median_length": float(scored_df["genome_length"].median()) if records else 0.0,
        "median_gc": float(scored_df["gc_content"].median()) if records else 0.0,
        "median_max_homopolymer": float(scored_df["max_nt_homopolymer_length"].median()) if records else 0.0,
        "p90_max_homopolymer": float(scored_df["max_nt_homopolymer_length"].quantile(0.9)) if records else 0.0,
    }


def _write_markdown_report(summary_df: pd.DataFrame, output_path: Path, target_records: int | None) -> None:
    total_records = int(summary_df["records"].sum()) if not summary_df.empty else 0
    total_pass = int(summary_df["nucleotide_pass"].sum()) if not summary_df.empty else 0
    overall_rate = total_pass / total_records if total_records else 0.0
    best = summary_df.sort_values("nucleotide_pass_rate", ascending=False).head(8)

    lines = [
        "# HPO Nucleotide Scoring Summary",
        "",
        f"- Scored records: {total_records}",
        f"- Target records per cell: {target_records if target_records is not None else 'all unique records'}",
        f"- Nucleotide pass: {total_pass} / {total_records} = {overall_rate:.4f}",
        "- Filters: valid A/C/G/T, 4000-6000 bp, 30-65% GC, max homopolymer <= 10",
        "",
        "## Best Cells",
        "",
        best[
            [
                "prompt_len",
                "temperature",
                "records",
                "nucleotide_pass",
                "nucleotide_pass_rate",
                "valid_nt_chars_rate",
                "gc_pass_rate",
                "homopolymer_pass_rate",
                "median_max_homopolymer",
            ]
        ].to_markdown(index=False),
        "",
        "## Paper Comparison Notes",
        "",
        "- The paper reports the useful region around prompt lengths 4-9 nt and temperatures 0.7-0.9.",
        "- The paper's final reported post-diversification retention for Evo2 SFT generations is 17.2%.",
        "- This fast summary is nucleotide-only; full Arc-suite pass rates require the companion full-Arc script.",
        "",
    ]
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score paper-style HPO generation JSONL outputs")
    parser.add_argument("--run-root", type=Path, required=True, help="Generation run root containing jsonl/")
    parser.add_argument("--jsonl-dir", type=Path, default=None, help="Override JSONL directory")
    parser.add_argument("--fasta-dir", type=Path, default=None, help="Output FASTA directory")
    parser.add_argument("--score-dir", type=Path, default=None, help="Output score directory")
    parser.add_argument("--glob", type=str, default="phix174_prompt*_temp*.jsonl")
    parser.add_argument("--target-records", type=int, default=1000, help="Unique records to score per cell; <=0 means all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root
    jsonl_dir = args.jsonl_dir or run_root / "jsonl"
    fasta_dir = args.fasta_dir or run_root / "fasta"
    score_dir = args.score_dir or run_root / "scores"
    target_records = None if args.target_records <= 0 else args.target_records

    fasta_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    jsonl_paths = sorted(jsonl_dir.glob(args.glob))
    if not jsonl_paths:
        raise FileNotFoundError(f"No JSONL files matched {args.glob!r} under {jsonl_dir}")

    suffix = f"manifest{target_records}" if target_records is not None else "all_unique"
    for jsonl_path in jsonl_paths:
        cell = jsonl_path.stem
        fasta_path = fasta_dir / f"{cell}.{suffix}.fasta"
        score_path = score_dir / f"{cell}.{suffix}_scores.csv"
        prompt_len, temperature = _parse_cell(cell)

        if score_path.exists() and not args.overwrite:
            scored_df = pd.read_csv(score_path)
            fasta_record_count = _fasta_record_count(fasta_path)
            if fasta_record_count != len(scored_df):
                save_fasta(scored_df[["id_prompt", "sequence"]], fasta_path)
            stats = {
                "json_records_seen": -1,
                "unique_records_scored": len(scored_df),
                "duplicate_records_skipped": -1,
                "malformed_records_skipped": -1,
            }
        else:
            sequences_df, stats = _load_generation_records(jsonl_path, target_records)
            save_fasta(sequences_df, fasta_path)
            scored_df = score_nucleotide_metrics(sequences_df)
            scored_df.to_csv(score_path, index=False)

        row = _summarize_cell(cell, prompt_len, temperature, scored_df)
        row.update(stats)
        row["jsonl"] = str(jsonl_path)
        row["fasta"] = str(fasta_path)
        row["scores_csv"] = str(score_path)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(["temperature", "prompt_len", "cell"])
    summary_path = score_dir / f"hpo_nucleotide_summary_{suffix}.csv"
    report_path = score_dir / f"hpo_nucleotide_summary_{suffix}.md"
    summary_df.to_csv(summary_path, index=False)
    _write_markdown_report(summary_df, report_path, target_records)

    print(f"summary_csv: {summary_path}")
    print(f"summary_md: {report_path}")
    print(
        summary_df[
            [
                "prompt_len",
                "temperature",
                "records",
                "nucleotide_pass",
                "nucleotide_pass_rate",
                "valid_nt_chars_rate",
                "gc_pass_rate",
                "homopolymer_pass_rate",
                "median_max_homopolymer",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
