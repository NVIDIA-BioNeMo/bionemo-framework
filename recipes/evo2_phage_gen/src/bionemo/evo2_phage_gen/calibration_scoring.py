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

"""Score selected-SFT sampling cells with the exact online RL objective stack."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from bionemo.evo2_phage_gen.design_scope import HostDomain, HostEvidence
from bionemo.evo2_phage_gen.qc import NucleotideQCConfig, prompt_nucleotides, trim_at_first_eos
from bionemo.evo2_phage_gen.reward import (
    ExternalQCRewardConfig,
    MMseqsClusterDiversityConfig,
    RewardWeights,
    SequenceSafetyRewardConfig,
    score_nucleotide_metrics,
)


CELL_RE = re.compile(
    r"(?:(?P<anchor>[A-Za-z0-9][A-Za-z0-9_-]*)_)?"
    r"prefix(?P<prefix>\d+)_temp(?P<temperature>\d+(?:\.\d+)?)$"
)
EXTERNAL_OBJECTIVES = {
    "protein_hit_count": "protein_database_hit_count",
    "tropism": "tropism",
    "required_genes": "required_genes",
    "synteny": "synteny",
    "average_protein_identity": "average_protein_identity",
}
SAFETY_OBJECTIVES = ("amr", "toxin", "lysogeny")
EXPLICIT_SAFETY_INAPPLICABILITY = {
    "toxin": ("NOT_RUN", frozenset({"TOXIN_NO_PROTEIN_QUERIES"})),
    "lysogeny": ("NOT_RUN", frozenset({"PHROGS_NO_PREDICTED_GENES"})),
}
REWARD_COLUMNS = (
    "reward_valid_nt_chars",
    "reward_genome_length",
    "reward_gc_content",
    "reward_nt_homopolymer",
    "reward_dustmask_end",
    "reward_nucleotide_pass",
    "reward_external_protein_hit_count",
    "reward_external_tropism",
    "reward_external_required_genes",
    "reward_external_synteny",
    "reward_external_average_protein_identity",
    "reward_mmseqs_cluster_diversity",
    "reward_safety_amr",
    "reward_safety_toxin",
    "reward_safety_lysogeny",
)
BIOLOGY_COLUMNS = (
    "protein_database_hit_count",
    "tropism_protein_mmseqs_percent_identity",
    "required_genes_matched_count",
    "required_genes_total",
    "num_syntenic_genes",
    "total_num_genes",
    "average_protein_percent_identity",
    "average_protein_identity_gene_count",
    "mmseqs_cluster_is_singleton",
)


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _reason_code_set(value: object) -> frozenset[str] | None:
    """Parse serialized safety reasons without forgiving malformed telemetry."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, list | tuple) or any(
        not isinstance(reason, str)
        or not reason
        or len(reason) > 128
        or not reason.isascii()
        or not reason.replace("_", "").isalnum()
        for reason in value
    ):
        return None
    return frozenset(value)


def safety_objective_interpretability(scored: pd.DataFrame) -> pd.DataFrame:
    """Classify safety evidence as measured or explicitly biologically inapplicable."""
    interpretable: dict[str, pd.Series] = {}
    for safety_class in SAFETY_OBJECTIVES:
        prefix = f"safety_{safety_class}"
        availability = _numeric_column(scored, f"{prefix}_measurement_available")
        statuses = (
            scored[f"{prefix}_execution_status"]
            if f"{prefix}_execution_status" in scored
            else pd.Series(None, index=scored.index, dtype=object)
        )
        reasons = (
            scored[f"{prefix}_reason_codes"].map(_reason_code_set)
            if f"{prefix}_reason_codes" in scored
            else pd.Series(None, index=scored.index, dtype=object)
        )
        healthy_measurement = availability.eq(1.0) & statuses.eq("COMPLETED_AND_PARSED") & reasons.notna()
        explicit_inapplicability = pd.Series(False, index=scored.index, dtype=bool)
        expected = EXPLICIT_SAFETY_INAPPLICABILITY.get(safety_class)
        if expected is not None:
            expected_status, expected_reasons = expected
            explicit_inapplicability = (
                availability.eq(0.0) & statuses.eq(expected_status) & reasons.eq(expected_reasons)
            )
        interpretable[safety_class] = healthy_measurement | explicit_inapplicability
    return pd.DataFrame(interpretable, index=scored.index)


def load_generation_records(path: Path) -> pd.DataFrame:
    """Reconstruct marker-free genomes from one strict generation JSONL."""
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        record_id = str(record.get("id") or f"{path.stem}_{line_number:06d}")
        if record_id in seen:
            raise ValueError(f"duplicate generation ID: {record_id}")
        seen.add(record_id)
        prompt = prompt_nucleotides(trim_at_first_eos(str(record.get("prompt", ""))))
        completion = trim_at_first_eos(str(record.get("completion", "")).replace("\n", "").strip())
        rows.append({"id_prompt": record_id, "sequence": (prompt + completion).upper()})
    if not rows:
        raise ValueError(f"no generation records in {path}")
    return pd.DataFrame(rows)


def summarize_cell(cell: str, scored: pd.DataFrame) -> dict[str, float | int | str | bool | None]:
    """Summarize reward, hard-pass, and support without conflating zeros with missingness."""
    match = CELL_RE.fullmatch(cell)
    external_environment_ok = bool(
        len(scored)
        and "external_qc_tool_succeeded" in scored
        and (pd.to_numeric(scored["external_qc_tool_succeeded"], errors="coerce").fillna(0.0) == 1.0).all()
    )
    safety_interpretable = safety_objective_interpretability(scored)
    safety_evidence_interpretable = bool(len(scored) and safety_interpretable.all().all())
    row: dict[str, float | int | str | bool | None] = {
        "cell": cell,
        "prompt_anchor": match.group("anchor") if match and match.group("anchor") else "origin",
        "prefix_length": int(match.group("prefix")) if match else -1,
        "temperature": float(match.group("temperature")) if match else float("nan"),
        "records": len(scored),
        "metric_environment_ok": external_environment_ok and safety_evidence_interpretable,
    }
    for objective, support_prefix in EXTERNAL_OBJECTIVES.items():
        reward_column = f"reward_external_{objective}"
        support_column = f"{support_prefix}_measurement_available"
        row[f"{objective}_reward_mean"] = float(_numeric_column(scored, reward_column).mean())
        row[f"{objective}_support_rate"] = float(_numeric_column(scored, support_column).mean())
    external_support_columns = [f"{prefix}_measurement_available" for prefix in EXTERNAL_OBJECTIVES.values()]
    external_support = pd.concat(
        [_numeric_column(scored, column) for column in external_support_columns], axis=1
    ).fillna(0.0)
    row["all_external_measurements_available_rate"] = float(external_support.min(axis=1).mean())
    for column in REWARD_COLUMNS:
        row[f"{column}_mean"] = float(_numeric_column(scored, column).mean())
    for column in BIOLOGY_COLUMNS:
        row[f"{column}_mean"] = float(_numeric_column(scored, column).mean())
    cluster_count = _numeric_column(scored, "mmseqs_cluster_num_clusters").max()
    row["mmseqs_cluster_num_clusters"] = int(cluster_count) if pd.notna(cluster_count) else None
    for column in (
        "reward_nucleotide_pass",
        "reward_binary_core_pass",
        "reward_binary_core_cluster_deduplicated_pass",
        "reward_binary_full_qc_pass",
        "reward_binary_full_qc_cluster_deduplicated_pass",
        "safety_gate_pass",
    ):
        row[f"{column}_rate"] = float(_numeric_column(scored, column).mean())
    row["aggregate_reward_mean"] = float(_numeric_column(scored, "reward").mean())
    return row


def validate_score_file(path: Path, expected_records: int) -> None:
    """Validate the record count and identifiers in a score CSV."""
    scored = pd.read_csv(path)
    if len(scored) != expected_records:
        raise ValueError(f"{path}: expected {expected_records} records, found {len(scored)}")
    if "id_prompt" not in scored:
        raise ValueError(f"{path}: missing id_prompt column")
    if scored["id_prompt"].astype(str).duplicated().any():
        raise ValueError(f"{path}: duplicate id_prompt values")


def score_cell(
    *,
    generation_jsonl: Path,
    output_csv: Path,
    arc_config: Path,
    pipeline_script: Path,
    work_dir: Path,
    tool_bin_dir: Path,
    threads: int,
    sequence_safety: SequenceSafetyRewardConfig,
) -> pd.DataFrame:
    """Run one cell through the same shaped objectives used by RL."""
    arc = yaml.safe_load(arc_config.read_text())
    sequences = load_generation_records(generation_jsonl)
    external = ExternalQCRewardConfig(
        enabled=True,
        config_path=arc_config,
        pipeline_script=pipeline_script,
        work_dir=work_dir,
        tool_bin_dir=tool_bin_dir,
        fail_on_error=True,
        enable_protein_hit_count=bool(arc.get("protein_database_hit_count_filter")),
        enable_tropism=bool(arc.get("tropism_protein_sequence_identity_filter")),
        enable_synteny=bool(arc.get("syntenic_gene_count_filter")),
        synteny_mode="full",
        enable_average_protein_identity=bool(arc.get("average_protein_sequence_identity_filter")),
        enable_required_genes=bool(arc.get("required_genes_filter")),
        required_genes_evidence_target=float(arc.get("required_genes_evidence_target", 10.0)),
        protein_match_min_reciprocal_coverage=float(arc.get("protein_match_min_reciprocal_coverage", 0.75)),
        tropism_match_min_reciprocal_coverage=float(arc.get("tropism_match_min_reciprocal_coverage", 0.95)),
        lovis4u_parallel_jobs=max(1, threads),
        lovis4u_chunk_size=max(1, threads),
        lovis4u_collect_pdfs=False,
    )
    scored = score_nucleotide_metrics(
        sequences,
        config=NucleotideQCConfig(
            genome_length_min=5306,
            genome_length_max=5493,
            genome_length_reward_lower_zero=5305,
            genome_length_reward_lower_full=5359,
            genome_length_reward_upper_full=5391,
            genome_length_reward_upper_zero=5494,
            dustmask_filter=True,
            dustmasker_bin=str((tool_bin_dir / "dustmasker").resolve()),
        ),
        weights=RewardWeights(
            valid_nt_chars=1,
            genome_length=1,
            gc_content=1,
            nt_homopolymer=1,
            dustmask_end=1,
            nucleotide_pass=1,
            protein_hit_count=1,
            tropism=1,
            required_genes=1,
            synteny=1,
            average_protein_identity=1,
            mmseqs_cluster_diversity=1,
        ),
        external_qc=external,
        mmseqs_cluster_diversity=MMseqsClusterDiversityConfig(
            enabled=True,
            mmseqs_bin=str((tool_bin_dir / "mmseqs").resolve()),
            work_dir=work_dir.parent / "cluster-diversity" / generation_jsonl.stem,
            threads=max(1, threads),
        ),
        sequence_safety=sequence_safety,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    return scored


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score-cell")
    score.add_argument("--generation-jsonl", type=Path, required=True)
    score.add_argument("--output-csv", type=Path, required=True)
    score.add_argument("--arc-config", type=Path, required=True)
    score.add_argument("--pipeline-script", type=Path, required=True)
    score.add_argument("--work-dir", type=Path, required=True)
    score.add_argument("--tool-bin-dir", type=Path, required=True)
    score.add_argument("--threads", type=int, default=2)
    score.add_argument("--safety-asset-manifest", type=Path, required=True)
    score.add_argument("--safety-policy", type=Path, required=True)
    score.add_argument("--safety-host-domain", type=HostDomain, required=True)
    score.add_argument("--safety-host-evidence-json", required=True)
    score.add_argument("--safety-timeout-seconds", type=float, default=1800.0)
    score.add_argument("--safety-batch-size", type=int, default=64)
    score.add_argument("--safety-threads", type=int, default=2)
    score.add_argument("--safety-orf-workers", type=int, default=2)
    score.add_argument("--safety-phrogs-threads", type=int, default=2)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--score-dir", type=Path, required=True)
    summarize.add_argument("--output-csv", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--score-csv", type=Path, required=True)
    validate.add_argument("--expected-records", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the selected sampling-calibration scoring command."""
    args = _parse_args()
    if args.command == "score-cell":
        evidence_payload = json.loads(args.safety_host_evidence_json)
        host_evidence = HostEvidence(
            source=evidence_payload["source"],
            source_version=evidence_payload.get("source_version"),
            replication_host_domains=frozenset(
                HostDomain(domain) for domain in evidence_payload["replication_host_domains"]
            ),
            confirmed=evidence_payload["confirmed"],
            metadata=evidence_payload.get("metadata", {}),
        )
        sequence_safety = SequenceSafetyRewardConfig(
            host_domain=args.safety_host_domain,
            host_evidence=host_evidence,
            asset_manifest_path=args.safety_asset_manifest,
            diamond_bin=args.tool_bin_dir / "diamond",
            mmseqs_bin=args.tool_bin_dir / "mmseqs",
            policy_path=args.safety_policy,
            work_dir=args.work_dir.parent / "sequence-safety" / args.generation_jsonl.stem,
            enabled=True,
            strict_lysis=False,
            circular=True,
            threads=args.safety_threads,
            batch_size=args.safety_batch_size,
            orf_workers=args.safety_orf_workers,
            phrogs_threads=args.safety_phrogs_threads,
            timeout_seconds=args.safety_timeout_seconds,
        )
        scored = score_cell(
            generation_jsonl=args.generation_jsonl,
            output_csv=args.output_csv,
            arc_config=args.arc_config,
            pipeline_script=args.pipeline_script,
            work_dir=args.work_dir,
            tool_bin_dir=args.tool_bin_dir,
            threads=args.threads,
            sequence_safety=sequence_safety,
        )
        print(json.dumps(summarize_cell(args.generation_jsonl.stem, scored), sort_keys=True))
        return
    if args.command == "validate":
        validate_score_file(args.score_csv, args.expected_records)
        print(args.score_csv)
        return
    rows = [
        summarize_cell(path.name.removesuffix(".scores.csv"), pd.read_csv(path))
        for path in sorted(args.score_dir.glob("*.scores.csv"))
    ]
    if not rows:
        raise FileNotFoundError(f"no score CSVs under {args.score_dir}")
    output = pd.DataFrame(rows).sort_values(["temperature", "prefix_length", "prompt_anchor"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()
