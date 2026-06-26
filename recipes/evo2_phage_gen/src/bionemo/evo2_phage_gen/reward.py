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

"""Pluggable reward functions for Evo2 phage design RL."""

import argparse
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from bionemo.evo2_phage_gen.qc import NucleotideQCConfig, add_nucleotide_metrics, load_fasta_records, save_fasta


RECIPE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = RECIPE_ROOT.parents[1]
ARC_PATH_KEYS = (
    "reference_genome_fasta",
    "genetic_architecture_reference_genome",
    "reference_tropism_protein",
    "mmseqs_db_protein_database",
    "training_data_genomes_fasta",
    "mmseqs_db_tropism_protein",
    "genetic_architecture_visualization_script",
    "protein_annotation_file",
    "reference_genome_gff_file_save_location",
)


@dataclass(frozen=True)
class RewardWeights:
    """Weights for phage-design reward components."""

    valid_nt_chars: float = 1.0
    genome_length: float = 1.0
    gc_content: float = 1.0
    nt_homopolymer: float = 1.0
    orf: float = 0.0
    coding_density: float = 0.0
    protein_hit_count: float = 0.0
    tropism: float = 0.0
    genetic_architecture: float = 0.0
    checkv: float = 0.0
    synteny: float = 0.0
    mmseqs_clustering: float = 0.0
    diversity: float = 0.0


@dataclass(frozen=True)
class ExternalQCRewardConfig:
    """Configuration for Arc external-QC reward components."""

    enabled: bool = False
    config_path: Path = Path("configs/arc_genome_design_filtering_local.yaml")
    pipeline_script: Path = Path("data/arc_pipeline_patched/genome_design_filtering_pipeline.py")
    work_dir: Path = Path("data/checkpoints/phage_grpo_external_qc")
    checkv_db_path: Path = Path("data/external/checkv/checkv-db-v1.5")
    keep_artifacts: bool = False
    enable_orf: bool = True
    enable_coding_density: bool = True
    enable_protein_hit_count: bool = True
    enable_tropism: bool = True
    enable_genetic_architecture: bool = True
    enable_checkv: bool = False
    enable_synteny: bool = False
    enable_mmseqs_clustering: bool = False
    enable_diversity: bool = False
    diversity_quality_threshold: float = 0.6
    diversity_kmer_size: int = 8


def _recipe_path(path: str | Path) -> Path:
    """Resolve recipe-relative paths while preserving absolute paths."""
    path = Path(path)
    return path if path.is_absolute() else RECIPE_ROOT / path


def _repo_path(path: str | Path) -> Path:
    """Resolve repo-root-relative config paths while preserving absolute paths."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _external_qc_env(external_qc: ExternalQCRewardConfig) -> dict[str, str]:
    """Build the environment for Arc external-QC subprocesses."""
    env = os.environ.copy()
    if external_qc.enable_checkv:
        checkv_db_path = _recipe_path(external_qc.checkv_db_path)
        if not checkv_db_path.exists():
            raise FileNotFoundError(
                f"CheckV database not found: {checkv_db_path}. "
                "Run `evo2_phage_prepare_external_assets` without `--skip-checkv` first."
            )
        env["CHECKVDB"] = str(checkv_db_path)
    return env


def _interval_score(value: float, lower: float, upper: float) -> float:
    """Return 1 inside an interval and a smooth bounded penalty outside it."""
    if lower <= value <= upper:
        return 1.0
    distance = lower - value if value < lower else value - upper
    width = max(upper - lower, 1.0)
    return max(0.0, 1.0 - distance / width)


def score_nucleotide_metrics(
    sequences_df: pd.DataFrame,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
    external_qc: ExternalQCRewardConfig | None = None,
) -> pd.DataFrame:
    """Score sequences with nucleotide QC and optional Arc external-QC components."""
    df = add_nucleotide_metrics(sequences_df)
    df["reward_valid_nt_chars"] = df["valid_nt_chars"].astype(float)
    df["reward_genome_length"] = df["genome_length"].map(
        lambda value: _interval_score(value, config.genome_length_min, config.genome_length_max)
    )
    df["reward_gc_content"] = df["gc_content"].map(
        lambda value: _interval_score(value, config.gc_content_min, config.gc_content_max)
    )
    df["reward_nt_homopolymer"] = df["max_nt_homopolymer_length"].map(
        lambda value: _interval_score(value, config.homopolymer_min, config.homopolymer_max)
    )

    if external_qc and external_qc.enabled:
        df = add_external_qc_rewards(df, external_qc)

    reward_terms = [
        (weights.valid_nt_chars, "reward_valid_nt_chars"),
        (weights.genome_length, "reward_genome_length"),
        (weights.gc_content, "reward_gc_content"),
        (weights.nt_homopolymer, "reward_nt_homopolymer"),
        (weights.orf, "reward_external_orf"),
        (weights.coding_density, "reward_external_coding_density"),
        (weights.protein_hit_count, "reward_external_protein_hit_count"),
        (weights.tropism, "reward_external_tropism"),
        (weights.genetic_architecture, "reward_external_genetic_architecture"),
        (weights.checkv, "reward_external_checkv"),
        (weights.synteny, "reward_external_synteny"),
        (weights.mmseqs_clustering, "reward_external_mmseqs_clustering"),
        (weights.diversity, "reward_diversity"),
    ]
    active_terms = [(weight, column) for weight, column in reward_terms if weight > 0.0]
    weighted_sum = sum(weight * df[column] for weight, column in active_terms if column in df)
    total_weight = sum(weight for weight, column in active_terms if column in df)
    if total_weight == 0.0:
        raise ValueError("At least one available reward weight must be positive.")
    df["reward"] = weighted_sum / total_weight
    return df


def _write_external_qc_config(
    base_config_path: Path,
    run_dir: Path,
    input_fasta: Path,
    external_qc: ExternalQCRewardConfig,
) -> Path:
    """Write an Arc pipeline config for one RL reward batch."""
    config = yaml.safe_load(base_config_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = run_dir / "arc_external_qc.yaml"
    config["results_save_dir"] = str(run_dir)
    config["current_config_file"] = str(run_config_path)
    config["evo_gen_seqs_fasta_file_save_location"] = str(input_fasta)
    config["overwrite_sequence_ids"] = True
    for key in ARC_PATH_KEYS:
        if config.get(key):
            config[key] = str(_repo_path(config[key]))

    orf_enabled = external_qc.enable_orf or external_qc.enable_coding_density
    homology_enabled = (
        external_qc.enable_protein_hit_count
        or external_qc.enable_tropism
        or external_qc.enable_genetic_architecture
        or external_qc.enable_checkv
        or external_qc.enable_synteny
        or external_qc.enable_mmseqs_clustering
    )

    config["orf_filtering"] = bool(orf_enabled)
    config["prodigal_based_filters"] = bool(orf_enabled)
    config["orf_count_filter"] = bool(external_qc.enable_orf)
    config["orf_lengths_filter"] = bool(external_qc.enable_orf)
    config["coding_density_filter"] = bool(external_qc.enable_coding_density)
    config["aminoacid_homopolymer_length_filter"] = bool(external_qc.enable_orf)

    config["homology_filtering"] = bool(homology_enabled)
    config["use_orf_filtered_df"] = bool(orf_enabled)
    config["use_nucleotide_filtered_df_instead"] = not bool(orf_enabled)
    config["protein_database_hit_count_filter"] = bool(external_qc.enable_protein_hit_count)
    config["genetic_architecture_filter"] = bool(external_qc.enable_genetic_architecture)
    config["tropism_protein_sequence_identity_filter"] = bool(external_qc.enable_tropism)
    config["checkv_filter"] = bool(external_qc.enable_checkv)

    config["diversification_filtering"] = bool(external_qc.enable_mmseqs_clustering)
    config["use_homology_filtered_df"] = True
    config["use_orf_filtered_df_instead"] = False
    config["use_nucleotide_filtered_df_instead_2"] = False
    config["mmseqs_clustering_filter"] = bool(external_qc.enable_mmseqs_clustering)
    config["genetic_architecture_remove_filter"] = False
    config["genetic_architecture_visualization_and_synteny_filtering"] = False

    run_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return run_config_path


def _sequence_ids_from_csv(path: Path) -> set[str]:
    """Read Arc output IDs from a staged CSV file."""
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "id_prompt" not in df:
        return set()
    return set(df["id_prompt"].astype(str))


def _genome_ids_from_orf_hits(hits_df: pd.DataFrame) -> pd.Series:
    """Map Arc ORF-level MMseqs query IDs back to genome IDs."""
    return hits_df["id_prompt"].astype(str).str.split("_").str[:-1].str.join("_")


def _as_arc_pass_mask(scored_df: pd.DataFrame, pass_ids: set[str]) -> pd.Series:
    """Return a mask for Arc UMI IDs while preserving original IDs in output."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    return scored_df[id_column].astype(str).isin(pass_ids)


def _orf_order(orf_id: str) -> int:
    """Extract a stable ORF order from Arc/Orfipy IDs."""
    match = re.search(r"ORF\.(\d+)", str(orf_id))
    return int(match.group(1)) if match else 0


def _normalize_phrog_target(value: str) -> str:
    """Normalize PHROGs target identifiers for annotation joins."""
    value = str(value)
    match = re.search(r"phrog[_-]?(\d+)", value, flags=re.IGNORECASE)
    return match.group(1) if match else value


def _load_phrog_annotations(annotation_file: str | Path) -> pd.DataFrame:
    """Load PHROGs annotations with normalized join keys."""
    annotations_path = _repo_path(annotation_file)
    if not annotations_path.exists():
        return pd.DataFrame(columns=["phrog_number", "annot", "category"])
    annotations = pd.read_csv(annotations_path, sep="\t")
    if "phrog" in annotations:
        annotations["phrog_number"] = annotations["phrog"].map(_normalize_phrog_target)
    elif "hit_label" in annotations:
        annotations["phrog_number"] = annotations["hit_label"].map(_normalize_phrog_target)
    else:
        return pd.DataFrame(columns=["phrog_number", "annot", "category"])
    for column in ["annot", "category"]:
        if column not in annotations:
            annotations[column] = ""
    return annotations[["phrog_number", "annot", "category"]]


def _required_gene_score(products: list[str], required_products: list[str]) -> float:
    """Score how many required gene annotations are present by substring match."""
    if not required_products:
        return 1.0
    normalized_products = [str(product).lower() for product in products]
    hits = 0
    for required in required_products:
        required_lower = str(required).lower()
        if any(required_lower in product for product in normalized_products):
            hits += 1
    return hits / len(required_products)


def _ordered_required_gene_score(products: list[str], required_products: list[str]) -> float:
    """Use LCS over required-gene labels as a synteny-aligned order proxy."""
    if not required_products:
        return 1.0
    product_labels: list[str] = []
    for product in products:
        product_lower = str(product).lower()
        matches = [required for required in required_products if str(required).lower() in product_lower]
        if matches:
            product_labels.append(matches[0])
    if not product_labels:
        return 0.0

    required_labels = [str(required) for required in required_products]
    dp = [[0] * (len(required_labels) + 1) for _ in range(len(product_labels) + 1)]
    for i, product in enumerate(product_labels, start=1):
        for j, required in enumerate(required_labels, start=1):
            if product == required:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1] / len(required_labels)


def _add_checkv_rewards(scored_df: pd.DataFrame, run_dir: Path, config: dict) -> pd.DataFrame:
    """Add CheckV quality rewards from Arc output."""
    quality_path = run_dir / config["checkv_results_dir_save_location"] / "quality_summary.tsv"
    if not quality_path.exists():
        return scored_df
    quality_df = pd.read_csv(quality_path, sep="\t")
    if not {"contig_id", "checkv_quality"}.issubset(quality_df.columns):
        return scored_df
    allowed = set(config.get("checkv_quality_range", []))
    quality_by_id = quality_df.set_index("contig_id")["checkv_quality"]
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    scored_df["checkv_quality"] = scored_df[id_column].map(quality_by_id).fillna("")
    scored_df["reward_external_checkv"] = scored_df["checkv_quality"].isin(allowed).astype(float)
    return scored_df


def _add_synteny_proxy_rewards(scored_df: pd.DataFrame, run_dir: Path, config: dict) -> pd.DataFrame:
    """Add a synteny-correlated score from PHROGs/ORF artifacts used by Arc synteny."""
    phrogs_hits_path = run_dir / config["mmseqs_protein_database_results_dir_save_location"] / "mmseqs2_hits.csv"
    if not phrogs_hits_path.exists():
        return scored_df
    hits_df = pd.read_csv(phrogs_hits_path)
    if not {"id_prompt", "protein_database_mmseqs_target"}.issubset(hits_df.columns):
        return scored_df

    hits_df = hits_df.copy()
    hits_df["arc_qc_id"] = _genome_ids_from_orf_hits(hits_df)
    hits_df["orf_order"] = hits_df["id_prompt"].map(_orf_order)
    hits_df["phrog_number"] = hits_df["protein_database_mmseqs_target"].map(_normalize_phrog_target)
    annotations = _load_phrog_annotations(config.get("protein_annotation_file", ""))
    hits_df = hits_df.merge(annotations, on="phrog_number", how="left")
    hits_df["annot"] = hits_df["annot"].fillna("")
    hits_df["category"] = hits_df["category"].fillna("")

    required_products = [str(product) for product in config.get("required_genes_list", [])]
    total_gene_range = config.get("total_gene_count_range", [10, 12])
    rows = []
    for arc_qc_id, group in hits_df.sort_values(["arc_qc_id", "orf_order"]).groupby("arc_qc_id"):
        products = group["annot"].astype(str).tolist()
        total_gene_count = int(group["id_prompt"].nunique())
        rows.append(
            {
                "arc_qc_id": arc_qc_id,
                "synteny_required_gene_score": _required_gene_score(products, required_products),
                "synteny_order_score": _ordered_required_gene_score(products, required_products),
                "synteny_total_gene_score": _interval_score(
                    total_gene_count,
                    float(total_gene_range[0]),
                    float(total_gene_range[1]),
                ),
                "synteny_proxy_gene_count": total_gene_count,
            }
        )

    if rows:
        synteny_df = pd.DataFrame(rows).set_index("arc_qc_id")
        id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
        for column in [
            "synteny_required_gene_score",
            "synteny_order_score",
            "synteny_total_gene_score",
            "synteny_proxy_gene_count",
        ]:
            scored_df[column] = scored_df[id_column].map(synteny_df[column]).fillna(0.0)
        scored_df["reward_external_synteny"] = (
            scored_df["synteny_required_gene_score"]
            + scored_df["synteny_order_score"]
            + scored_df["synteny_total_gene_score"]
        ) / 3.0

    synteny_csv = run_dir / config.get("synteny_filter_seqs_csv_file_save_location", "")
    if synteny_csv.exists():
        scored_df["reward_external_synteny"] = _as_arc_pass_mask(scored_df, _sequence_ids_from_csv(synteny_csv)).astype(float)
    return scored_df


def _add_mmseqs_clustering_rewards(scored_df: pd.DataFrame, run_dir: Path, config: dict) -> pd.DataFrame:
    """Reward sequences that remain as MMseqs clustering representatives."""
    clustered_csv = run_dir / config.get("diversification_filter_seqs_csv_file_save_location", "")
    if clustered_csv.exists():
        scored_df["reward_external_mmseqs_clustering"] = _as_arc_pass_mask(
            scored_df, _sequence_ids_from_csv(clustered_csv)
        ).astype(float)
    return scored_df


def _kmer_set(sequence: str, k: int) -> set[str]:
    """Return the set of sequence k-mers."""
    sequence = str(sequence).upper()
    if len(sequence) < k:
        return {sequence} if sequence else set()
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def _zscore(values: pd.Series) -> pd.Series:
    """Z-score values with a stable zero fallback."""
    std = float(values.std(ddof=0))
    if std == 0.0:
        return pd.Series(0.0, index=values.index)
    return (values - float(values.mean())) / std


def _add_diversity_rewards(scored_df: pd.DataFrame, external_qc: ExternalQCRewardConfig) -> pd.DataFrame:
    """Add gated batch k-mer diversity reward, normalized like a GRPO group bonus."""
    if len(scored_df) <= 1:
        scored_df["diversity_raw"] = 0.0
        scored_df["reward_diversity"] = 0.0
        return scored_df

    kmer_sets = scored_df["sequence"].map(lambda sequence: _kmer_set(sequence, external_qc.diversity_kmer_size))
    raw_scores = []
    for i, kmers_i in enumerate(kmer_sets):
        distances = []
        for j, kmers_j in enumerate(kmer_sets):
            if i == j:
                continue
            union = kmers_i | kmers_j
            jaccard = len(kmers_i & kmers_j) / len(union) if union else 1.0
            distances.append(1.0 - jaccard)
        raw_scores.append(sum(distances) / len(distances) if distances else 0.0)

    scored_df["diversity_raw"] = raw_scores
    normalized = _zscore(pd.Series(raw_scores, index=scored_df.index)).clip(-2.0, 2.0)
    quality_columns = [
        "reward_valid_nt_chars",
        "reward_genome_length",
        "reward_gc_content",
        "reward_nt_homopolymer",
        "reward_external_orf",
        "reward_external_protein_hit_count",
        "reward_external_tropism",
        "reward_external_genetic_architecture",
        "reward_external_checkv",
        "reward_external_synteny",
    ]
    available_quality_columns = [column for column in quality_columns if column in scored_df]
    quality = scored_df[available_quality_columns].mean(axis=1) if available_quality_columns else pd.Series(1.0, index=scored_df.index)
    normalized = normalized.where(quality >= external_qc.diversity_quality_threshold, 0.0)
    scored_df["reward_diversity"] = normalized
    return scored_df


def _add_mmseqs_hit_rewards(scored_df: pd.DataFrame, run_dir: Path, config: dict) -> pd.DataFrame:
    """Add protein-hit-count and tropism rewards from Arc MMseqs outputs."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    phrogs_hits_path = run_dir / config["mmseqs_protein_database_results_dir_save_location"] / "mmseqs2_hits.csv"
    if phrogs_hits_path.exists():
        hits_df = pd.read_csv(phrogs_hits_path)
        if "id_prompt" in hits_df:
            genome_counts = _genome_ids_from_orf_hits(hits_df).value_counts()
            min_hits = int(config.get("protein_database_hit_count", 7))
            scored_df["protein_database_hit_count"] = scored_df[id_column].map(genome_counts).fillna(0).astype(int)
            scored_df["reward_external_protein_hit_count"] = (
                scored_df["protein_database_hit_count"] >= min_hits
            ).astype(float)

    tropism_hits_path = run_dir / config["mmseqs_tropism_protein_results_dir_save_location"] / "mmseqs2_hits.csv"
    if tropism_hits_path.exists():
        hits_df = pd.read_csv(tropism_hits_path)
        if {"id_prompt", "tropism_protein_mmseqs_percent_identity"}.issubset(hits_df.columns):
            hits_df = hits_df.copy()
            hits_df["genome_id"] = _genome_ids_from_orf_hits(hits_df)
            hits_df["tropism_protein_mmseqs_percent_identity"] = pd.to_numeric(
                hits_df["tropism_protein_mmseqs_percent_identity"], errors="coerce"
            ).fillna(0.0)
            best_pident = hits_df.groupby("genome_id")["tropism_protein_mmseqs_percent_identity"].max()
            lower, upper = config.get("tropism_protein_sequence_identity_range", [60, 100])
            scored_df["tropism_protein_mmseqs_percent_identity"] = (
                scored_df[id_column].map(best_pident).fillna(0.0)
            )
            scored_df["reward_external_tropism"] = scored_df[
                "tropism_protein_mmseqs_percent_identity"
            ].between(float(lower), float(upper)).astype(float)
    return scored_df


def add_external_qc_rewards(
    scored_df: pd.DataFrame,
    external_qc: ExternalQCRewardConfig,
) -> pd.DataFrame:
    """Run Arc external QC on a batch and add binary staged reward columns."""
    base_config_path = _recipe_path(external_qc.config_path)
    pipeline_script = _recipe_path(external_qc.pipeline_script)
    work_dir = _recipe_path(external_qc.work_dir)
    if not pipeline_script.exists():
        raise FileNotFoundError(f"Arc pipeline script not found: {pipeline_script}")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Arc external-QC config not found: {base_config_path}")

    run_dir = work_dir / f"batch_{uuid.uuid4().hex}"
    input_fasta = run_dir / "input_sequences.fasta"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = scored_df.copy()
    for column in [
        "reward_external_orf",
        "reward_external_coding_density",
        "reward_external_protein_hit_count",
        "reward_external_tropism",
        "reward_external_genetic_architecture",
        "reward_external_checkv",
        "reward_external_synteny",
        "reward_external_mmseqs_clustering",
        "reward_diversity",
    ]:
        df[column] = 0.0

    try:
        df["arc_qc_id"] = [f"umi{i + 1}" for i in range(len(df))]
        save_fasta(df.rename(columns={"id_prompt": "original_id_prompt", "arc_qc_id": "id_prompt"})[["id_prompt", "sequence"]], input_fasta)
        run_config_path = _write_external_qc_config(base_config_path, run_dir, input_fasta, external_qc)
        subprocess.run(
            ["python", str(pipeline_script), str(run_config_path)],
            check=True,
            cwd=str(pipeline_script.parent),
            env=_external_qc_env(external_qc),
        )
        config = yaml.safe_load(run_config_path.read_text())

        orf_csv = run_dir / config["orf_filter_seqs_csv_file_save_location"]
        orf_pass_ids = _sequence_ids_from_csv(orf_csv)
        if external_qc.enable_orf:
            df["reward_external_orf"] = df["arc_qc_id"].astype(str).isin(orf_pass_ids).astype(float)
        if external_qc.enable_coding_density:
            df["reward_external_coding_density"] = df["arc_qc_id"].astype(str).isin(orf_pass_ids).astype(float)

        homology_csv = run_dir / config["homology_filter_seqs_csv_file_save_location"]
        homology_df = pd.read_csv(homology_csv) if homology_csv.exists() else pd.DataFrame()
        if external_qc.enable_genetic_architecture and not homology_df.empty:
            lower, upper = config.get("genetic_architecture_score_range", [0, 10])
            score_column = "genetic_architecture_score"
            if score_column in homology_df:
                arch_pass_ids = set(
                    homology_df[
                        pd.to_numeric(homology_df[score_column], errors="coerce").between(float(lower), float(upper))
                    ]["id_prompt"].astype(str)
                )
                df["reward_external_genetic_architecture"] = (
                    df["arc_qc_id"].astype(str).isin(arch_pass_ids).astype(float)
                )

        df = _add_mmseqs_hit_rewards(df, run_dir, config)
        if external_qc.enable_checkv:
            df = _add_checkv_rewards(df, run_dir, config)
        if external_qc.enable_synteny:
            df = _add_synteny_proxy_rewards(df, run_dir, config)
        if external_qc.enable_mmseqs_clustering:
            df = _add_mmseqs_clustering_rewards(df, run_dir, config)
        if external_qc.enable_diversity:
            df = _add_diversity_rewards(df, external_qc)
    finally:
        if not external_qc.keep_artifacts:
            shutil.rmtree(run_dir, ignore_errors=True)
    return df


def score_fasta(
    input_fasta: Path,
    output_csv: Path,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
) -> Path:
    """Score a FASTA file and write per-sequence reward diagnostics."""
    sequences_df = load_fasta_records(input_fasta)
    scored_df = score_nucleotide_metrics(sequences_df, config=config, weights=weights)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    """CLI entry point for scoring FASTA files with the online reward."""
    parser = argparse.ArgumentParser(description="Score Evo2 phage FASTA sequences with online-safe reward components")
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--genome-length-min", type=int, default=4000)
    parser.add_argument("--genome-length-max", type=int, default=6000)
    parser.add_argument("--gc-content-min", type=float, default=30.0)
    parser.add_argument("--gc-content-max", type=float, default=65.0)
    parser.add_argument("--homopolymer-max", type=int, default=10)
    args = parser.parse_args()

    output = score_fasta(
        input_fasta=args.input_fasta,
        output_csv=args.output_csv,
        config=NucleotideQCConfig(
            genome_length_min=args.genome_length_min,
            genome_length_max=args.genome_length_max,
            gc_content_min=args.gc_content_min,
            gc_content_max=args.gc_content_max,
            homopolymer_max=args.homopolymer_max,
        ),
    )
    print(f"reward_csv: {output}")


if __name__ == "__main__":
    main()
