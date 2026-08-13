#!/usr/bin/env python

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

"""Summarize paper Sheet 1 candidate metrics from a TSV export.

Sheet 1 is useful for checking final candidate metric ranges, but it is not a
full filter-funnel table unless the complete workbook/export is available.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    """A numeric Sheet 1 metric and the local/paper-style acceptance range."""

    column: str
    threshold: str
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    notes: str = ""


METRIC_SPECS = (
    MetricSpec("genome_length", "4000-6000 bp", 4000, 6000),
    MetricSpec("gc_content", "30-65%", 30.0, 65.0),
    MetricSpec("max_nt_homopolymer_length", "<=10 nt", None, 10),
    MetricSpec("prodigal_orf_count", "reported metric; local ORF filter is nonrestrictive", 0, None),
    MetricSpec("prodigal_coding_density", "reported metric; local ORF filter is nonrestrictive", 0, None),
    MetricSpec("max_prodigal_aa_homopolymer_len", "reported metric; local ORF filter is nonrestrictive", 0, None),
    MetricSpec("protein_database_hit_count", ">=7 PHROG/protein hits", 7, None),
    MetricSpec("reference_genome_percent_identity", "reported metric; not in current full-Arc run config", None, None),
    MetricSpec("architecture_similarity_score", "homology keep: 0-10; diversification removes 0.9-1.1", 0, 10),
    MetricSpec("tropism_protein_mmseqs_percent_identity", "60-100%", 60.0, 100.0),
    MetricSpec("average_protein_percent_identity", "0-95%", 0, 95.0),
    MetricSpec("num_syntenic_genes", "10-12 syntenic genes in local synteny config", 10, 12),
    MetricSpec("total_num_genes", "10-12 total genes in local synteny config", 10, 12),
)

CATEGORICAL_COLUMNS = (
    "checkv_quality",
    "tropism_protein_mmseqs_target",
    "non_syntenic_annotations",
)


def _validate_tsv(path: Path, expected_rows: int | None) -> dict[str, Any]:
    raw_lines = path.read_text(errors="replace").splitlines()
    if not raw_lines:
        raise ValueError(f"{path} is empty")

    expected_fields = raw_lines[0].count("\t") + 1
    malformed_rows = []
    for line_no, line in enumerate(raw_lines[1:], start=2):
        field_count = line.count("\t") + 1
        if field_count != expected_fields:
            first_field = line.split("\t", maxsplit=1)[0]
            malformed_rows.append(f"line {line_no}: {field_count} fields, id={first_field}")

    data_rows = len(raw_lines) - 1
    return {
        "expected_fields": expected_fields,
        "data_rows": data_rows,
        "expected_rows": expected_rows,
        "missing_expected_rows": max(expected_rows - data_rows, 0) if expected_rows is not None else None,
        "malformed_rows": malformed_rows,
    }


def _passes_threshold(values: pd.Series, spec: MetricSpec) -> pd.Series:
    mask = pd.Series(True, index=values.index)
    if spec.lower is not None:
        mask &= values >= spec.lower if spec.lower_inclusive else values > spec.lower
    if spec.upper is not None:
        mask &= values <= spec.upper if spec.upper_inclusive else values < spec.upper
    if spec.lower is None and spec.upper is None:
        mask &= False
    return mask


def _summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in METRIC_SPECS:
        if spec.column not in df:
            rows.append(
                {
                    "metric": spec.column,
                    "available_count": 0,
                    "missing_count": len(df),
                    "threshold": spec.threshold,
                    "available_pass_count": pd.NA,
                    "available_pass_rate": pd.NA,
                    "notes": "column missing",
                }
            )
            continue

        values = pd.to_numeric(df[spec.column], errors="coerce")
        observed = values.dropna()
        pass_mask = _passes_threshold(values, spec) if spec.lower is not None or spec.upper is not None else None
        observed_pass_mask = pass_mask[values.notna()] if pass_mask is not None else None
        rows.append(
            {
                "metric": spec.column,
                "available_count": int(observed.count()),
                "missing_count": int(values.isna().sum()),
                "min": float(observed.min()) if not observed.empty else pd.NA,
                "q25": float(observed.quantile(0.25)) if not observed.empty else pd.NA,
                "median": float(observed.median()) if not observed.empty else pd.NA,
                "q75": float(observed.quantile(0.75)) if not observed.empty else pd.NA,
                "max": float(observed.max()) if not observed.empty else pd.NA,
                "mean": float(observed.mean()) if not observed.empty else pd.NA,
                "threshold": spec.threshold,
                "available_pass_count": int(observed_pass_mask.sum()) if observed_pass_mask is not None else pd.NA,
                "available_pass_rate": (
                    float(observed_pass_mask.mean())
                    if observed_pass_mask is not None and len(observed_pass_mask)
                    else pd.NA
                ),
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def _summarize_categories(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in CATEGORICAL_COLUMNS:
        if column not in df:
            continue
        counts = df[column].fillna("<missing>").astype(str).value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({"column": column, "value": value, "count": int(count), "rate": float(count / len(df))})
    return pd.DataFrame(rows, columns=["column", "value", "count", "rate"])


def _write_markdown(
    output_path: Path,
    validation: dict[str, Any],
    metric_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    sheet_path: Path,
) -> None:
    malformed_rows = validation["malformed_rows"]
    expected_rows = validation["expected_rows"] if validation["expected_rows"] is not None else "not specified"
    missing_rows = validation["missing_expected_rows"]

    lines = [
        "# Paper Sheet 1 Candidate Summary",
        "",
        f"- Input TSV: `{sheet_path}`",
        f"- Parsed data rows: {validation['data_rows']}",
        f"- Expected data rows: {expected_rows}",
        f"- Expected fields per row: {validation['expected_fields']}",
    ]
    if missing_rows:
        lines.append(f"- Warning: export appears partial; missing about {missing_rows} expected rows.")
    if malformed_rows:
        lines.append(f"- Warning: {len(malformed_rows)} row(s) have fewer/more fields than the header.")
    lines.extend(
        [
            "",
            "Sheet 1 examples are treated as final/selected candidate metrics, not as a complete filter-funnel denominator.",
            "",
            "## Numeric Metrics",
            "",
            metric_summary.to_markdown(index=False),
        ]
    )
    if not categorical_summary.empty:
        lines.extend(
            [
                "",
                "## Categorical Metrics",
                "",
                categorical_summary.head(40).to_markdown(index=False),
            ]
        )
    if malformed_rows:
        lines.extend(["", "## Malformed Rows", ""])
        lines.extend(f"- {row}" for row in malformed_rows[:20])
    lines.append("")
    output_path.write_text("\n".join(lines))


def main() -> None:
    """Summarize candidate metrics from a Sheet 1 TSV export."""
    parser = argparse.ArgumentParser(description="Summarize paper Sheet 1 candidate metrics")
    parser.add_argument("--sheet1-tsv", type=Path, default=Path("dist/sheet_1.tsv"))
    parser.add_argument("--expected-rows", type=int, default=303, help="Expected Sheet 1 data rows; <=0 disables")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    expected_rows = None if args.expected_rows <= 0 else args.expected_rows
    validation = _validate_tsv(args.sheet1_tsv, expected_rows)
    df = pd.read_csv(args.sheet1_tsv, sep="\t")

    output_dir = args.output_dir or args.sheet1_tsv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_summary = _summarize_metrics(df)
    categorical_summary = _summarize_categories(df)

    metric_csv = output_dir / "paper_sheet1_metric_summary.csv"
    categorical_csv = output_dir / "paper_sheet1_categorical_summary.csv"
    report_md = output_dir / "paper_sheet1_candidate_summary.md"
    metric_summary.to_csv(metric_csv, index=False)
    categorical_summary.to_csv(categorical_csv, index=False)
    _write_markdown(report_md, validation, metric_summary, categorical_summary, args.sheet1_tsv)

    print(f"metric_csv: {metric_csv}")
    print(f"categorical_csv: {categorical_csv}")
    print(f"report_md: {report_md}")
    print(metric_summary[["metric", "available_count", "min", "median", "max", "threshold"]].to_string(index=False))


if __name__ == "__main__":
    main()
