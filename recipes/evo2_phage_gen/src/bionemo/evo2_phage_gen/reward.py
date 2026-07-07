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
    dustmask_end: float = 0.0
    nucleotide_pass: float = 0.0
    orf: float = 0.0
    coding_density: float = 0.0
    protein_hit_count: float = 0.0
    tropism: float = 0.0
    synteny: float = 0.0
    average_protein_identity: float = 0.0
    required_genes: float = 0.0
    mmseqs_cluster_diversity: float = 0.0


@dataclass(frozen=True)
class RewardComponent:
    """A swappable 0-1 reward component used by the aggregate RL score."""

    name: str
    weight_attr: str
    score_column: str
    required_for_binary_pass: bool = True


REWARD_COMPONENTS: tuple[RewardComponent, ...] = (
    RewardComponent("valid_nt_chars", "valid_nt_chars", "reward_valid_nt_chars"),
    RewardComponent("genome_length", "genome_length", "reward_genome_length"),
    RewardComponent("gc_content", "gc_content", "reward_gc_content"),
    RewardComponent("nt_homopolymer", "nt_homopolymer", "reward_nt_homopolymer"),
    RewardComponent("dustmask_end", "dustmask_end", "reward_dustmask_end"),
    RewardComponent("nucleotide_pass", "nucleotide_pass", "reward_nucleotide_pass"),
    RewardComponent("protein_hit_count", "protein_hit_count", "reward_external_protein_hit_count"),
    RewardComponent("tropism", "tropism", "reward_external_tropism"),
    RewardComponent("synteny", "synteny", "reward_external_synteny", required_for_binary_pass=False),
    RewardComponent(
        "average_protein_identity",
        "average_protein_identity",
        "reward_external_average_protein_identity",
        required_for_binary_pass=False,
    ),
    RewardComponent(
        "required_genes",
        "required_genes",
        "reward_external_required_genes",
        required_for_binary_pass=False,
    ),
    RewardComponent(
        "mmseqs_cluster_diversity",
        "mmseqs_cluster_diversity",
        "reward_mmseqs_cluster_diversity",
        required_for_binary_pass=False,
    ),
)


@dataclass(frozen=True)
class ExternalQCRewardConfig:
    """Configuration for Arc external-QC reward components."""

    enabled: bool = False
    config_path: Path = Path("configs/arc_genome_design_filtering_local.yaml")
    pipeline_script: Path = Path("data/arc_pipeline_patched/genome_design_filtering_pipeline.py")
    work_dir: Path = Path("data/checkpoints/phage_grpo_external_qc")
    keep_artifacts: bool = False
    enable_orf: bool = False
    enable_coding_density: bool = False
    enable_protein_hit_count: bool = True
    enable_tropism: bool = True
    enable_synteny: bool = False
    synteny_mode: str = "proxy"
    enable_average_protein_identity: bool = False
    enable_required_genes: bool = False
    required_genes_evidence_target: float = 9.0
    lovis4u_parallel_jobs: int | None = 12
    lovis4u_chunk_size: int | None = None
    lovis4u_collect_pdfs: bool = False


@dataclass(frozen=True)
class MMseqsClusterDiversityConfig:
    """Configuration for batch-local MMseqs cluster-diversity rewards."""

    enabled: bool = False
    mmseqs_bin: str = "mmseqs"
    work_dir: Path = Path("data/checkpoints/phage_grpo_mmseqs_cluster_diversity")
    keep_artifacts: bool = False
    min_seq_id: float = 0.99
    coverage: float = 0.0
    cov_mode: int = 0
    seq_id_mode: int = 0
    cluster_mode: int = 0
    threads: int | None = None


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
    return os.environ.copy()


def _resolve_executable_path(executable: str) -> str:
    """Resolve recipe-relative executable paths while leaving PATH lookups alone."""
    path = Path(executable)
    if path.is_absolute() or len(path.parts) > 1:
        return str(_recipe_path(path))
    return executable


def _basic_feasibility_mask(scored_df: pd.DataFrame, config: NucleotideQCConfig) -> pd.Series:
    """Return the nucleotide feasibility gate used before expensive diversity scoring."""
    return (
        scored_df["valid_nt_chars"].astype(bool)
        & scored_df["genome_length"].between(config.genome_length_min, config.genome_length_max)
        & scored_df["gc_content"].between(config.gc_content_min, config.gc_content_max)
        & (scored_df["max_nt_homopolymer_length"] <= config.homopolymer_max)
    )


def _mmseqs_cluster_command(
    config: MMseqsClusterDiversityConfig,
    input_fasta: Path,
    result_prefix: Path,
    tmp_dir: Path,
) -> list[str]:
    """Build the pinned MMseqs easy-cluster command for batch diversity rewards."""
    command = [
        _resolve_executable_path(config.mmseqs_bin),
        "easy-cluster",
        str(input_fasta),
        str(result_prefix),
        str(tmp_dir),
        "--min-seq-id",
        f"{float(config.min_seq_id):.6g}",
        "-c",
        f"{float(config.coverage):.6g}",
        "--cov-mode",
        str(int(config.cov_mode)),
        "--seq-id-mode",
        str(int(config.seq_id_mode)),
        "--cluster-mode",
        str(int(config.cluster_mode)),
    ]
    if config.threads is not None:
        command.extend(["--threads", str(int(config.threads))])
    return command


def _parse_mmseqs_cluster_tsv(cluster_tsv: Path) -> dict[str, set[str]]:
    """Read an MMseqs cluster TSV into representative-to-member sets."""
    clusters: dict[str, set[str]] = {}
    with cluster_tsv.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            representative, member = parts[0], parts[1]
            clusters.setdefault(representative, set()).add(member)
    return clusters


def _cluster_valid_sequence_group(
    group_df: pd.DataFrame,
    run_dir: Path,
    group_index: int,
    config: MMseqsClusterDiversityConfig,
) -> tuple[dict[object, tuple[str, int, float]], int, int]:
    """Cluster one prompt group and return row-index rewards plus cluster counts."""
    if group_df.empty:
        return {}, 0, 0
    if len(group_df) == 1:
        row_index = group_df.index[0]
        return {row_index: (f"group{group_index}:seq_0", 1, 1.0)}, 1, 1

    group_dir = run_dir / f"prompt_group_{group_index:04d}"
    group_dir.mkdir(parents=True, exist_ok=True)
    input_fasta = group_dir / "input_sequences.fasta"
    result_prefix = group_dir / "clusters"
    tmp_dir = group_dir / "tmp"
    sequence_ids = [f"seq_{position}" for position in range(len(group_df))]
    row_by_sequence_id = dict(zip(sequence_ids, group_df.index.tolist(), strict=True))
    fasta_df = pd.DataFrame(
        {
            "id_prompt": sequence_ids,
            "sequence": group_df["sequence"].astype(str).tolist(),
        }
    )
    save_fasta(fasta_df, input_fasta)

    subprocess.run(_mmseqs_cluster_command(config, input_fasta, result_prefix, tmp_dir), check=True)
    cluster_tsv = Path(f"{result_prefix}_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs cluster TSV not found: {cluster_tsv}")

    clusters = _parse_mmseqs_cluster_tsv(cluster_tsv)
    rewards_by_row: dict[object, tuple[str, int, float]] = {}
    valid_cluster_count = 0
    for representative, members in clusters.items():
        known_members = sorted(member for member in members if member in row_by_sequence_id)
        cluster_size = len(known_members)
        if cluster_size == 0:
            continue
        valid_cluster_count += 1
        cluster_id = f"group{group_index}:{representative}"
        reward = 1.0 / float(cluster_size)
        for member in known_members:
            rewards_by_row[row_by_sequence_id[member]] = (cluster_id, cluster_size, reward)

    missing_members = set(sequence_ids) - {
        member for members in clusters.values() for member in members if member in row_by_sequence_id
    }
    for member in missing_members:
        row_index = row_by_sequence_id[member]
        rewards_by_row[row_index] = ("", 0, 0.0)
    return rewards_by_row, valid_cluster_count, len(missing_members)


def add_mmseqs_cluster_diversity_rewards(
    scored_df: pd.DataFrame,
    config: NucleotideQCConfig,
    mmseqs_config: MMseqsClusterDiversityConfig,
) -> pd.DataFrame:
    """Add ``1 / cluster_size`` rewards from batch-local MMseqs clustering."""
    df = scored_df.copy()
    df["reward_mmseqs_cluster_diversity"] = 0.0
    df["mmseqs_cluster_id"] = ""
    df["mmseqs_cluster_size"] = 0
    df["mmseqs_cluster_is_singleton"] = 0.0
    df["mmseqs_cluster_valid_for_clustering"] = _basic_feasibility_mask(df, config).astype(float)
    df["mmseqs_cluster_missing_from_output"] = 0.0
    if not mmseqs_config.enabled:
        return df

    valid_df = df[df["mmseqs_cluster_valid_for_clustering"].astype(bool)]
    if valid_df.empty:
        return df

    work_dir = _recipe_path(mmseqs_config.work_dir)
    run_dir = work_dir / f"batch_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        prompt_groups = (
            valid_df["prompt_group"] if "prompt_group" in valid_df else pd.Series("__all__", index=valid_df.index)
        )
        total_clusters = 0
        total_missing = 0
        for group_index, (_prompt_group, group_df) in enumerate(valid_df.groupby(prompt_groups, sort=False)):
            rewards_by_row, num_clusters, num_missing = _cluster_valid_sequence_group(
                group_df,
                run_dir,
                group_index,
                mmseqs_config,
            )
            total_clusters += num_clusters
            total_missing += num_missing
            for row_index, (cluster_id, cluster_size, reward) in rewards_by_row.items():
                df.loc[row_index, "mmseqs_cluster_id"] = cluster_id
                df.loc[row_index, "mmseqs_cluster_size"] = int(cluster_size)
                df.loc[row_index, "reward_mmseqs_cluster_diversity"] = float(reward)
                df.loc[row_index, "mmseqs_cluster_is_singleton"] = 1.0 if cluster_size == 1 else 0.0
        df["mmseqs_cluster_num_clusters"] = total_clusters
        df["mmseqs_cluster_num_missing_from_output"] = total_missing
        missing_output_mask = df["mmseqs_cluster_valid_for_clustering"].astype(bool) & (
            df["mmseqs_cluster_size"].astype(int) == 0
        )
        df.loc[missing_output_mask, "mmseqs_cluster_missing_from_output"] = 1.0
    finally:
        if not mmseqs_config.keep_artifacts:
            shutil.rmtree(run_dir, ignore_errors=True)
    return df


def _interval_score(value: float, lower: float, upper: float) -> float:
    """Return 1 inside an interval and a smooth bounded penalty outside it."""
    if lower <= value <= upper:
        return 1.0
    distance = lower - value if value < lower else value - upper
    width = max(upper - lower, 1.0)
    return max(0.0, 1.0 - distance / width)


def _upper_bound_ratio_score(value: float, upper: float) -> float:
    """Return a dense score for upper-bound-only metrics such as homopolymer length."""
    if value <= upper:
        return 1.0
    if value <= 0.0:
        return 0.0
    return max(0.0, min(1.0, upper / value))


def _lower_bound_ratio_score(value: float, lower: float) -> float:
    """Return a dense capped score for lower-bound thresholds."""
    if value >= lower:
        return 1.0
    if lower <= 0.0:
        return 0.0
    return max(0.0, min(1.0, value / lower))


def _bounded_range_score(value: float, lower: float, upper: float) -> float:
    """Return 1 inside a target range and bounded partial credit outside it."""
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return _lower_bound_ratio_score(value, lower)
    return _upper_bound_ratio_score(value, upper)


def _bounded_percent_range_score(value: float, lower: float, upper: float) -> float:
    """Return 1 in a percent range and linearly decay to 0 at 0 or 100 outside it."""
    value = max(0.0, min(100.0, float(value)))
    lower = max(0.0, min(100.0, float(lower)))
    upper = max(0.0, min(100.0, float(upper)))
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return 0.0 if lower <= 0.0 else max(0.0, min(1.0, value / lower))
    return 0.0 if upper >= 100.0 else max(0.0, min(1.0, (100.0 - value) / (100.0 - upper)))


def _spike_identity_score(identity: float | None, measured_hit: bool, threshold: float = 60.0) -> float:
    """Plateau spike/tropism reward at the paper identity threshold."""
    if not measured_hit:
        return 0.0
    identity = max(0.0, float(identity or 0.0))
    if identity >= threshold:
        return 1.0
    if threshold <= 0.0:
        return 0.0
    return max(0.0, min(1.0, identity / threshold))


def _aai_novelty_score(aai: float) -> float:
    """Reward AAI novelty up to 95%, then keep high-similarity genomes fractional."""
    aai = max(0.0, min(100.0, float(aai)))
    if aai <= 95.0:
        return 1.0
    return max(0.25, (100.0 - aai) / 5.0)


def _aai_evidence_score(num_aai_entries: float) -> float:
    """Require enough measured proteins before trusting AAI novelty."""
    return max(0.0, min(1.0, float(num_aai_entries) / 10.0))


ARC_VALID_SYNTENY_PAIRS: frozenset[tuple[int, int]] = frozenset({(10, 10), (10, 11), (10, 12), (11, 12), (12, 12)})


def _distance_to_interval(value: float, lower: float, upper: float) -> float:
    """Return zero inside an interval, otherwise distance to the nearest endpoint."""
    if lower <= value <= upper:
        return 0.0
    return lower - value if value < lower else value - upper


def _synteny_distance_score(syntenic_genes: float, total_genes: float) -> tuple[float, float, float, float]:
    """Score closeness to Arc-valid syntenic/total gene-count pairs."""
    if syntenic_genes > total_genes:
        return 0.0, 0.0, 0.0, 0.0

    total_distance = _distance_to_interval(float(total_genes), 10.0, 12.0)
    total_score = 1.0 / (1.0 + total_distance)
    pair_distance = min(
        abs(float(syntenic_genes) - valid_syntenic) + abs(float(total_genes) - valid_total)
        for valid_syntenic, valid_total in ARC_VALID_SYNTENY_PAIRS
    )
    pair_score = 1.0 / (1.0 + pair_distance)
    synteny_score = total_score * pair_score
    return synteny_score, total_score, pair_score, pair_distance


def _active_reward_components(weights: RewardWeights, scored_df: pd.DataFrame) -> list[tuple[float, RewardComponent]]:
    """Return weighted reward components whose score columns are available."""
    active_components = []
    for component in REWARD_COMPONENTS:
        weight = float(getattr(weights, component.weight_attr))
        if weight > 0.0 and component.score_column in scored_df:
            active_components.append((weight, component))
    return active_components


def _aggregate_reward(scored_df: pd.DataFrame, weights: RewardWeights) -> pd.DataFrame:
    """Aggregate available 0-1 component scores into the scalar RL reward."""
    active_components = _active_reward_components(weights, scored_df)
    if not active_components:
        raise ValueError("At least one available reward weight must be positive.")

    weighted_sum = 0.0
    total_weight = 0.0
    for weight, component in active_components:
        scored_df[component.score_column] = scored_df[component.score_column].astype(float).clip(0.0, 1.0)
        weighted_sum = weighted_sum + weight * scored_df[component.score_column]
        total_weight += weight

    scored_df["reward"] = weighted_sum / total_weight
    scored_df["reward_active_components"] = ",".join(component.name for _, component in active_components)
    scored_df["reward_total_weight"] = total_weight
    binary_pass = binary_core_pass_mask(scored_df, weights)
    scored_df["reward_binary_core_pass"] = binary_pass.astype(float)
    scored_df["reward_binary_core_cluster_deduplicated_pass"] = binary_cluster_deduplicated_pass_mask(
        scored_df,
        binary_pass,
    ).astype(float)
    full_qc_pass = binary_full_qc_pass_mask(scored_df, binary_pass)
    if full_qc_pass is not None:
        scored_df["reward_binary_full_qc_pass"] = full_qc_pass.astype(float)
        scored_df["reward_binary_full_qc_cluster_deduplicated_pass"] = binary_cluster_deduplicated_pass_mask(
            scored_df,
            full_qc_pass,
        ).astype(float)
    return scored_df


def binary_core_pass_mask(scored_df: pd.DataFrame, weights: RewardWeights) -> pd.Series:
    """Return the lab-facing binary pass mask for active non-diversity criteria."""
    active_components = [
        component
        for _, component in _active_reward_components(weights, scored_df)
        if component.required_for_binary_pass
    ]
    if not active_components:
        return pd.Series(False, index=scored_df.index)
    pass_mask = pd.Series(True, index=scored_df.index)
    for component in active_components:
        pass_mask &= scored_df[component.score_column].astype(float) >= 1.0
    return pass_mask


def binary_cluster_deduplicated_pass_mask(scored_df: pd.DataFrame, pass_mask: pd.Series) -> pd.Series:
    """Return one passing representative per MMseqs cluster when cluster data is available."""
    pass_mask = pass_mask.astype(bool)
    deduplicated = pd.Series(False, index=scored_df.index)
    if not {"mmseqs_cluster_id", "mmseqs_cluster_size"}.issubset(scored_df.columns):
        deduplicated.loc[pass_mask] = True
        return deduplicated

    cluster_sizes = pd.to_numeric(scored_df["mmseqs_cluster_size"], errors="coerce").fillna(0).astype(int)
    cluster_ids = scored_df["mmseqs_cluster_id"].astype(str)
    clustered_pass = pass_mask & (cluster_sizes > 0) & (cluster_ids != "")
    for _cluster_id, cluster_df in scored_df.loc[clustered_pass].groupby(cluster_ids[clustered_pass], sort=False):
        deduplicated.loc[cluster_df.index[0]] = True

    if "mmseqs_cluster_valid_for_clustering" in scored_df:
        valid_for_clustering = (
            pd.to_numeric(scored_df["mmseqs_cluster_valid_for_clustering"], errors="coerce").fillna(0.0) > 0.0
        )
        nonclusterable_pass = pass_mask & ~valid_for_clustering
    else:
        nonclusterable_pass = pass_mask & ~clustered_pass
    deduplicated.loc[nonclusterable_pass] = True
    return deduplicated


def binary_full_qc_pass_mask(scored_df: pd.DataFrame, binary_pass: pd.Series) -> pd.Series | None:
    """Return binary pass plus available full Arc QC gates such as synteny, AAI, and required genes."""
    full_qc_pass_columns = [
        "reward_external_synteny_pass",
        "reward_external_average_protein_identity_pass",
        "reward_external_required_genes_pass",
    ]
    active_pass_columns = [column for column in full_qc_pass_columns if column in scored_df]
    if not active_pass_columns:
        return None

    pass_mask = binary_pass.astype(bool).copy()
    for column in active_pass_columns:
        pass_mask &= pd.to_numeric(scored_df[column], errors="coerce").fillna(0.0) >= 1.0
    return pass_mask


def score_nucleotide_metrics(
    sequences_df: pd.DataFrame,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
    external_qc: ExternalQCRewardConfig | None = None,
    mmseqs_cluster_diversity: MMseqsClusterDiversityConfig | None = None,
) -> pd.DataFrame:
    """Score sequences with nucleotide QC, optional external QC, and optional batch diversity."""
    df = add_nucleotide_metrics(sequences_df, config=config)
    df["reward_valid_nt_chars"] = df["valid_nt_chars"].astype(float)
    df["reward_genome_length"] = df["genome_length"].map(
        lambda value: _interval_score(value, config.genome_length_min, config.genome_length_max)
    )
    df["reward_gc_content"] = df["gc_content"].map(
        lambda value: _interval_score(value, config.gc_content_min, config.gc_content_max)
    )
    df["reward_nt_homopolymer"] = df["max_nt_homopolymer_length"].map(
        lambda value: _upper_bound_ratio_score(value, config.homopolymer_max)
    )
    df["reward_dustmask_end"] = df["dustmask_max_end_masked_fraction"].map(
        lambda value: _upper_bound_ratio_score(value, config.dustmask_max_end_fraction)
    )
    dustmask_end_pass = (
        df["dustmask_end_pass"].astype(bool) if config.dustmask_filter else pd.Series(True, index=df.index)
    )
    df["reward_nucleotide_pass"] = (
        df["valid_nt_chars"]
        & df["genome_length"].between(config.genome_length_min, config.genome_length_max)
        & df["gc_content"].between(config.gc_content_min, config.gc_content_max)
        & (df["max_nt_homopolymer_length"] <= config.homopolymer_max)
        & dustmask_end_pass
    ).astype(float)

    if external_qc and external_qc.enabled:
        df = add_external_qc_rewards(df, external_qc)
    if mmseqs_cluster_diversity and mmseqs_cluster_diversity.enabled:
        df = add_mmseqs_cluster_diversity_rewards(df, config, mmseqs_cluster_diversity)

    return _aggregate_reward(df, weights)


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

    synteny_mode = str(external_qc.synteny_mode).lower()
    if synteny_mode not in {"proxy", "full"}:
        raise ValueError(f"Unsupported synteny_mode={external_qc.synteny_mode!r}; expected 'proxy' or 'full'.")
    full_synteny_enabled = bool(external_qc.enable_synteny and synteny_mode == "full")
    paper_synteny_stage_enabled = bool(
        full_synteny_enabled or external_qc.enable_average_protein_identity or external_qc.enable_required_genes
    )

    orf_enabled = external_qc.enable_orf or external_qc.enable_coding_density
    homology_enabled = (
        external_qc.enable_protein_hit_count
        or external_qc.enable_tropism
        or external_qc.enable_synteny
        or external_qc.enable_average_protein_identity
        or external_qc.enable_required_genes
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
    config["protein_database_hit_count_filter"] = bool(
        external_qc.enable_protein_hit_count or paper_synteny_stage_enabled
    )
    config["training_data_sequence_identity_filter"] = False
    config["genetic_architecture_filter"] = False
    config["tropism_protein_sequence_identity_filter"] = bool(external_qc.enable_tropism)
    config["checkv_filter"] = False

    config["diversification_filtering"] = False
    config["use_homology_filtered_df"] = True
    config["use_orf_filtered_df_instead"] = False
    config["use_nucleotide_filtered_df_instead_2"] = False
    config["mmseqs_clustering_filter"] = False
    config["mmseqs_reference_genome_sequence_identity_remove_filter"] = False
    config["genetic_architecture_remove_filter"] = False
    config["genetic_architecture_visualization_and_synteny_filtering"] = paper_synteny_stage_enabled
    config["average_protein_sequence_identity_filter"] = bool(external_qc.enable_average_protein_identity)
    config["required_genes_filter"] = bool(external_qc.enable_required_genes)
    config["syntenic_gene_count_filter"] = full_synteny_enabled
    if external_qc.lovis4u_parallel_jobs is not None:
        parallel_jobs = max(1, int(external_qc.lovis4u_parallel_jobs))
        config["lovis4u_parallel_jobs"] = parallel_jobs
        config["n_parallel_jobs"] = parallel_jobs
    if external_qc.lovis4u_chunk_size is not None:
        chunk_size = max(1, int(external_qc.lovis4u_chunk_size))
    elif external_qc.lovis4u_parallel_jobs is not None:
        chunk_size = max(1, int(external_qc.lovis4u_parallel_jobs))
    else:
        chunk_size = int(config.get("chunk_size", 10))
    config["lovis4u_chunk_size"] = chunk_size
    config["chunk_size"] = chunk_size
    config["lovis4u_collect_pdfs"] = bool(external_qc.lovis4u_collect_pdfs)
    if paper_synteny_stage_enabled:
        if not bool(config.get("allow_gff_product_order_synteny_fallback", False)):
            config["reference_genome_gff_file_save_location"] = None
        config.setdefault(
            "average_protein_sequence_identity_metrics_file_save_location",
            "qc6_average_protein_sequence_identity_metrics.csv",
        )
        config.setdefault("required_genes_metrics_file_save_location", "qc6_required_genes_metrics.csv")
        config.setdefault("synteny_metrics_file_save_location", "qc6_synteny_filter_metrics.csv")
        config["required_genes_evidence_target"] = float(external_qc.required_genes_evidence_target)

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
                "synteny_total_gene_score": _bounded_range_score(
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
            * scored_df["synteny_order_score"]
            * scored_df["synteny_total_gene_score"]
        )

    synteny_csv = run_dir / config.get("synteny_filter_seqs_csv_file_save_location", "")
    if synteny_csv.exists():
        pass_mask = _as_arc_pass_mask(scored_df, _sequence_ids_from_csv(synteny_csv))
        scored_df["reward_external_synteny_pass"] = pass_mask.astype(float)
        scored_df.loc[pass_mask, "reward_external_synteny"] = 1.0
    return scored_df


def _add_full_synteny_rewards(scored_df: pd.DataFrame, run_dir: Path, config: dict) -> pd.DataFrame:
    """Add continuous synteny rewards from Arc/LoVis4u syntenic-gene artifacts."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    metrics_path = run_dir / config.get("synteny_metrics_file_save_location", "qc6_synteny_filter_metrics.csv")
    if not metrics_path.exists():
        metrics_path = run_dir / config.get("synteny_filter_seqs_csv_file_save_location", "")

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        if {"id_prompt", "num_syntenic_genes", "total_num_genes"}.issubset(metrics_df.columns):
            metrics_df = metrics_df.copy()
            metrics_df["num_syntenic_genes"] = pd.to_numeric(metrics_df["num_syntenic_genes"], errors="coerce").fillna(
                0.0
            )
            metrics_df["total_num_genes"] = pd.to_numeric(metrics_df["total_num_genes"], errors="coerce").fillna(0.0)
            metrics_by_id = metrics_df.set_index(metrics_df["id_prompt"].astype(str))
            scored_df["num_syntenic_genes"] = (
                scored_df[id_column].astype(str).map(metrics_by_id["num_syntenic_genes"]).fillna(0.0)
            )
            scored_df["total_num_genes"] = (
                scored_df[id_column].astype(str).map(metrics_by_id["total_num_genes"]).fillna(0.0)
            )

            scores = [
                _synteny_distance_score(float(num_syntenic), float(total_genes))
                for num_syntenic, total_genes in zip(
                    scored_df["num_syntenic_genes"],
                    scored_df["total_num_genes"],
                    strict=False,
                )
            ]
            scored_df["reward_external_synteny"] = [score for score, _, _, _ in scores]
            scored_df["synteny_total_gene_score"] = [total_score for _, total_score, _, _ in scores]
            scored_df["synteny_pair_score"] = [pair_score for _, _, pair_score, _ in scores]
            scored_df["synteny_pair_distance"] = [pair_distance for _, _, _, pair_distance in scores]
            scored_df["syntenic_gene_count_score"] = scored_df["synteny_pair_score"]

    synteny_csv = run_dir / config.get("synteny_filter_seqs_csv_file_save_location", "")
    if synteny_csv.exists():
        pass_mask = _as_arc_pass_mask(scored_df, _sequence_ids_from_csv(synteny_csv))
        scored_df["reward_external_synteny_pass"] = pass_mask.astype(float)
    return scored_df


def _add_average_protein_identity_rewards(
    scored_df: pd.DataFrame,
    run_dir: Path,
    config: dict,
) -> pd.DataFrame:
    """Add continuous rewards for Arc's average protein percent-identity filter."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    metrics_path = run_dir / config.get(
        "average_protein_sequence_identity_metrics_file_save_location",
        "qc6_average_protein_sequence_identity_metrics.csv",
    )
    if not metrics_path.exists():
        metrics_path = run_dir / config.get("synteny_filter_seqs_csv_file_save_location", "")
    if not metrics_path.exists():
        return scored_df

    metrics_df = pd.read_csv(metrics_path)
    if not {"id_prompt", "average_protein_percent_identity"}.issubset(metrics_df.columns):
        return scored_df

    metrics_df = metrics_df.copy()
    metrics_df["average_protein_percent_identity"] = pd.to_numeric(
        metrics_df["average_protein_percent_identity"], errors="coerce"
    ).fillna(0.0)
    evidence_column = "average_protein_identity_gene_count"
    if evidence_column not in metrics_df:
        evidence_column = "total_num_genes" if "total_num_genes" in metrics_df else ""
    if evidence_column:
        metrics_df[evidence_column] = pd.to_numeric(metrics_df[evidence_column], errors="coerce").fillna(0.0)
    metrics_by_id = metrics_df.set_index(metrics_df["id_prompt"].astype(str))
    mapped_identity = scored_df[id_column].astype(str).map(metrics_by_id["average_protein_percent_identity"])
    has_identity_metric = mapped_identity.notna()
    scored_df["average_protein_percent_identity"] = mapped_identity.fillna(0.0)
    mapped_evidence = (
        scored_df[id_column].astype(str).map(metrics_by_id[evidence_column])
        if evidence_column
        else pd.Series(0.0, index=scored_df.index)
    )
    scored_df["average_protein_identity_gene_count"] = mapped_evidence.fillna(0.0)

    lower, upper = config.get("average_protein_sequence_identity_range", [0, 95])
    novelty_scores = mapped_identity.map(lambda value: _aai_novelty_score(float(value)) if pd.notna(value) else 0.0)
    evidence_scores = mapped_evidence.map(lambda value: _aai_evidence_score(float(value)) if pd.notna(value) else 0.0)
    scored_df["average_protein_identity_raw_score"] = novelty_scores
    scored_df["average_protein_identity_novelty_score"] = novelty_scores
    scored_df["average_protein_identity_evidence_score"] = evidence_scores
    scored_df["reward_external_average_protein_identity"] = (novelty_scores * evidence_scores).where(
        has_identity_metric,
        0.0,
    )
    scored_df["reward_external_average_protein_identity_pass"] = (
        has_identity_metric & (mapped_evidence > 0) & mapped_identity.between(float(lower), float(upper))
    ).astype(float)
    return scored_df


def _add_required_gene_rewards(
    scored_df: pd.DataFrame,
    run_dir: Path,
    config: dict,
    evidence_target: float = 9.0,
) -> pd.DataFrame:
    """Add continuous rewards for Arc's required-gene annotation filter."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    metrics_path = run_dir / config.get("required_genes_metrics_file_save_location", "qc6_required_genes_metrics.csv")
    if not metrics_path.exists():
        metrics_path = run_dir / config.get("synteny_filter_seqs_csv_file_save_location", "")
    if not metrics_path.exists():
        return scored_df

    metrics_df = pd.read_csv(metrics_path)
    required_columns = {"id_prompt", "required_genes_matched_count", "required_genes_total_count"}
    if not required_columns.issubset(metrics_df.columns):
        return scored_df

    metrics_df = metrics_df.copy()
    for column in ["required_genes_matched_count", "required_genes_total_count"]:
        metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce").fillna(0.0)
    metrics_by_id = metrics_df.set_index(metrics_df["id_prompt"].astype(str))
    mapped_matched = scored_df[id_column].astype(str).map(metrics_by_id["required_genes_matched_count"])
    mapped_total = scored_df[id_column].astype(str).map(metrics_by_id["required_genes_total_count"])
    has_required_gene_metric = mapped_matched.notna() & mapped_total.notna()
    scored_df["required_genes_matched_count"] = mapped_matched.fillna(0.0)
    scored_df["required_genes_total_count"] = mapped_total.fillna(0.0)
    scored_df["required_genes_raw_score"] = [
        0.0 if (not has_metric or total <= 0) else max(0.0, min(1.0, matched / total))
        for matched, total, has_metric in zip(
            scored_df["required_genes_matched_count"],
            scored_df["required_genes_total_count"],
            has_required_gene_metric,
            strict=False,
        )
    ]
    scored_df["required_genes_evidence_score"] = (
        scored_df["required_genes_total_count"]
        .map(lambda total: max(0.0, min(1.0, float(total) / max(float(evidence_target), 1.0))))
        .where(has_required_gene_metric & (scored_df["required_genes_total_count"] > 0), 0.0)
    )
    scored_df["reward_external_required_genes"] = (
        scored_df["required_genes_raw_score"] * scored_df["required_genes_evidence_score"]
    )
    scored_df["reward_external_required_genes_pass"] = (
        has_required_gene_metric
        & (scored_df["required_genes_total_count"] > 0)
        & (scored_df["required_genes_matched_count"] >= scored_df["required_genes_total_count"])
    ).astype(float)
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
            scored_df["reward_external_protein_hit_count"] = scored_df["protein_database_hit_count"].map(
                lambda value: _lower_bound_ratio_score(float(value), float(min_hits))
            )

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
            lower, _upper = config.get("tropism_protein_sequence_identity_range", [60, 100])
            mapped_identity = scored_df[id_column].map(best_pident)
            measured_hit = mapped_identity.notna()
            scored_df["tropism_protein_mmseqs_percent_identity"] = mapped_identity.fillna(0.0)
            scored_df["tropism_protein_measured_hit"] = measured_hit.astype(float)
            scored_df["reward_external_tropism"] = [
                _spike_identity_score(identity, has_hit, float(lower))
                for identity, has_hit in zip(
                    scored_df["tropism_protein_mmseqs_percent_identity"],
                    measured_hit,
                    strict=False,
                )
            ]
            scored_df["reward_external_tropism_pass"] = (
                measured_hit & (scored_df["tropism_protein_mmseqs_percent_identity"] >= float(lower))
            ).astype(float)
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
        "reward_external_synteny",
        "reward_external_average_protein_identity",
        "reward_external_required_genes",
    ]:
        df[column] = 0.0
    if external_qc.enable_synteny:
        df["reward_external_synteny_pass"] = 0.0
    if external_qc.enable_average_protein_identity:
        df["reward_external_average_protein_identity_pass"] = 0.0
    if external_qc.enable_required_genes:
        df["reward_external_required_genes_pass"] = 0.0

    try:
        df["arc_qc_id"] = [f"umi{i + 1}" for i in range(len(df))]
        save_fasta(
            df.rename(columns={"id_prompt": "original_id_prompt", "arc_qc_id": "id_prompt"})[
                ["id_prompt", "sequence"]
            ],
            input_fasta,
        )
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

        df = _add_mmseqs_hit_rewards(df, run_dir, config)
        if external_qc.enable_synteny:
            if str(external_qc.synteny_mode).lower() == "full":
                df = _add_full_synteny_rewards(df, run_dir, config)
            else:
                df = _add_synteny_proxy_rewards(df, run_dir, config)
        if external_qc.enable_average_protein_identity:
            df = _add_average_protein_identity_rewards(
                df,
                run_dir,
                config,
            )
        if external_qc.enable_required_genes:
            df = _add_required_gene_rewards(
                df,
                run_dir,
                config,
                external_qc.required_genes_evidence_target,
            )
    finally:
        if not external_qc.keep_artifacts:
            shutil.rmtree(run_dir, ignore_errors=True)
    return df


def score_fasta(
    input_fasta: Path,
    output_csv: Path,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
    mmseqs_cluster_diversity: MMseqsClusterDiversityConfig | None = None,
) -> Path:
    """Score a FASTA file and write per-sequence reward diagnostics."""
    sequences_df = load_fasta_records(input_fasta)
    scored_df = score_nucleotide_metrics(
        sequences_df,
        config=config,
        weights=weights,
        mmseqs_cluster_diversity=mmseqs_cluster_diversity,
    )
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
    parser.add_argument("--dustmask-filter", action="store_true")
    parser.add_argument("--dustmasker-bin", default="dustmasker")
    parser.add_argument("--dustmask-use-fallback", action="store_true")
    parser.add_argument("--dustmask-window", type=int, default=64)
    parser.add_argument("--dustmask-level", type=float, default=20.0)
    parser.add_argument("--dustmask-end-window", type=int, default=200)
    parser.add_argument("--dustmask-max-end-fraction", type=float, default=0.9)
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
            dustmask_filter=args.dustmask_filter,
            dustmasker_bin=args.dustmasker_bin,
            dustmask_use_external=not args.dustmask_use_fallback,
            dustmask_window=args.dustmask_window,
            dustmask_level=args.dustmask_level,
            dustmask_end_window=args.dustmask_end_window,
            dustmask_max_end_fraction=args.dustmask_max_end_fraction,
        ),
    )
    print(f"reward_csv: {output}")


if __name__ == "__main__":
    main()
