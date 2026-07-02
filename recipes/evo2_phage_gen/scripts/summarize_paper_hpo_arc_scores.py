#!/usr/bin/env python
"""Summarize per-cell full Arc filtering outputs for paper-style HPO runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CELL_RE = re.compile(r"phix174_prompt(?P<prompt_len>\d+)_temp(?P<temperature>[0-9]+(?:\.[0-9]+)?)")
COUNT_FILES = (
    "qc6_synteny_filter_counts.csv",
    "qc5_diversification_filter_counts.csv",
    "qc4_homology_filter_counts.csv",
    "qc3_orf_filter_counts.csv",
    "qc2_nt_filter_counts.csv",
)
COUNT_STAGE_COLUMNS = (
    "count_syntenic_gene_count_filter",
    "count_required_genes_filter",
    "count_average_protein_sequence_identity_filter",
    "count_genetic_architecture_score_remove_filter",
    "count_mmseqs_clustering_filter",
    "count_tropism_protein_sequence_identity_filter",
    "count_genetic_architecture_score_filter",
    "count_checkv_quality_filter",
    "count_protein_database_hit_count_filter",
    "count_aa_homopolymer_len_filter",
    "count_coding_density_filter",
    "count_orf_len_filter",
    "count_orf_count_filter",
    "count_nt_homopolymer_filter",
    "count_gc_filter",
    "count_genome_len_filter",
    "count_nt_filter",
    "count_initial_before_nucleotide_metrics",
)


def _parse_cell(cell: str) -> tuple[int | None, float | None]:
    match = CELL_RE.search(cell)
    if not match:
        return None, None
    return int(match.group("prompt_len")), float(match.group("temperature"))


def _read_counts(cell_dir: Path) -> dict[str, float | int | str]:
    for filename in COUNT_FILES:
        path = cell_dir / filename
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                row = df.iloc[-1].to_dict()
                row["counts_file"] = str(path)
                row["latest_stage"] = filename.removesuffix("_filter_counts.csv")
                return row
    return {"counts_file": "", "latest_stage": ""}


def _final_count(row: pd.Series) -> int:
    for column in COUNT_STAGE_COLUMNS:
        if column in row and pd.notna(row[column]):
            return int(row[column])
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full Arc HPO pass rates")
    parser.add_argument("--arc-root", type=Path, required=True, help="Directory containing one Arc output dir per cell")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for cell_dir in sorted(path for path in args.arc_root.iterdir() if path.is_dir()):
        prompt_len, temperature = _parse_cell(cell_dir.name)
        row = {
            "cell": cell_dir.name,
            "prompt_len": prompt_len,
            "temperature": temperature,
            "arc_dir": str(cell_dir),
        }
        row.update(_read_counts(cell_dir))
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise FileNotFoundError(f"No Arc cell directories found under {args.arc_root}")

    initial_col = "count_initial_before_nucleotide_metrics"
    summary["final_count"] = summary.apply(_final_count, axis=1)
    if initial_col in summary:
        initial = pd.to_numeric(summary[initial_col], errors="coerce").fillna(0)
        summary["final_pass_rate"] = summary["final_count"] / initial.where(initial > 0, 1)
    else:
        summary["final_pass_rate"] = 0.0

    summary = summary.sort_values(["temperature", "prompt_len", "cell"])
    output_csv = args.output_csv or args.arc_root / "hpo_full_arc_summary.csv"
    output_md = args.output_md or args.arc_root / "hpo_full_arc_summary.md"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    best = summary.sort_values("final_pass_rate", ascending=False).head(10)
    if initial_col in summary:
        total_initial = int(pd.to_numeric(summary[initial_col], errors="coerce").fillna(0).sum())
        best_columns = ["prompt_len", "temperature", initial_col, "final_count", "final_pass_rate"]
    else:
        total_initial = 0
        best_columns = ["prompt_len", "temperature", "final_count", "final_pass_rate"]

    lines = [
        "# HPO Full Arc Summary",
        "",
        f"- Cells summarized: {len(summary)}",
        f"- Total initial records: {total_initial}",
        f"- Total final candidates: {int(summary['final_count'].sum())}",
        "",
        "## Best Cells",
        "",
        best[best_columns].to_markdown(index=False),
        "",
        "## Paper Comparison Notes",
        "",
        "- Compare `final_pass_rate` after diversification/synteny to the paper's reported 17.2% post-diversification retention for Evo2 SFT generations.",
        "- Compare intermediate count columns to locate the first major mismatch with the paper pipeline.",
        "",
    ]
    output_md.write_text("\n".join(lines))

    print(f"summary_csv: {output_csv}")
    print(f"summary_md: {output_md}")
    print(best[["prompt_len", "temperature", "final_count", "final_pass_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
