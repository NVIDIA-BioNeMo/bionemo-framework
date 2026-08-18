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
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import pandas as pd
import yaml

from bionemo.evo2_phage_gen import sequence_safety_cli
from bionemo.evo2_phage_gen.design_scope import HostDomain, HostEvidence
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
TIMING_COLUMN_PREFIX = "timing/phage_qc/"
SEQUENCE_SAFETY_CLASSES = ("amr", "toxin", "lysogeny")
SAFETY_REVIEW_CREDIT = 0.25


def sequence_safety_reward_fields(
    *,
    class_states: dict[str, str],
    required_by_class: dict[str, bool],
    review_eligible_by_class: dict[str, bool] | None = None,
) -> dict[str, object]:
    """Derive the strict acceptance gate and independent class rewards from measured states."""
    expected_classes = set(SEQUENCE_SAFETY_CLASSES)
    if set(class_states) != expected_classes or set(required_by_class) != expected_classes:
        raise ValueError("sequence-safety reward inputs must define every safety class exactly once")
    if not all(state in {"PASS", "FAIL", "INDETERMINATE"} for state in class_states.values()):
        raise ValueError("sequence-safety reward input contains an invalid state")
    if not all(type(required) is bool for required in required_by_class.values()):
        raise ValueError("sequence-safety reward applicability must be boolean")
    review_eligible = (
        dict.fromkeys(SEQUENCE_SAFETY_CLASSES, False) if review_eligible_by_class is None else review_eligible_by_class
    )
    if set(review_eligible) != expected_classes or not all(type(value) is bool for value in review_eligible.values()):
        raise ValueError("sequence-safety review eligibility must define every class as boolean")
    if any(
        review_eligible[name] and (not required_by_class[name] or class_states[name] != "INDETERMINATE")
        for name in SEQUENCE_SAFETY_CLASSES
    ):
        raise ValueError("sequence-safety review credit requires a required INDETERMINATE class")

    required_states = [class_states[name] for name in SEQUENCE_SAFETY_CLASSES if required_by_class[name]]
    if not required_states:
        gate_state = "INDETERMINATE"
    elif "FAIL" in required_states:
        gate_state = "FAIL"
    elif "INDETERMINATE" in required_states:
        gate_state = "INDETERMINATE"
    else:
        gate_state = "PASS"
    gate_pass = float(gate_state == "PASS")
    telemetry: dict[str, object] = {
        "safety_gate_state": gate_state,
        "safety_gate_pass": gate_pass,
        "reward_safety_penalty": 1.0 - gate_pass,
    }
    telemetry.update(
        {
            f"reward_safety_{name}": (
                1.0
                if not required_by_class[name] or class_states[name] == "PASS"
                else SAFETY_REVIEW_CREDIT
                if review_eligible[name]
                else 0.0
            )
            for name in SEQUENCE_SAFETY_CLASSES
        }
    )
    return telemetry


def _attach_timing_columns(scored_df: pd.DataFrame, timings: dict[str, float]) -> pd.DataFrame:
    """Attach batch-level timing values as per-row columns for rollout metric aggregation."""
    for name, value in timings.items():
        scored_df[f"{TIMING_COLUMN_PREFIX}{name}"] = float(value)
    return scored_df


def _record_elapsed(timings: dict[str, float], name: str, start: float) -> None:
    """Record an elapsed perf-counter interval in seconds."""
    timings[name] = time.perf_counter() - start


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
    weight_attr: str | None
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
    RewardComponent("safety_amr", None, "reward_safety_amr"),
    RewardComponent("safety_toxin", None, "reward_safety_toxin"),
    RewardComponent("safety_lysogeny", None, "reward_safety_lysogeny"),
)


@dataclass(frozen=True)
class ExternalQCRewardConfig:
    """Configuration for Arc external-QC reward components."""

    enabled: bool = False
    config_path: Path = Path("configs/arc_genome_design_filtering_local.yaml")
    pipeline_script: Path = Path("data/arc_pipeline_patched/genome_design_filtering_pipeline.py")
    work_dir: Path = Path("data/checkpoints/phage_grpo_external_qc")
    tool_bin_dir: Path | None = None
    keep_artifacts: bool = False
    fail_on_error: bool = True
    timeout_seconds: float | None = 1800.0
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
    lovis4u_mmseqs_threads: int | None = None
    lovis4u_metrics_only: bool = False
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
    verbosity: int = 0


@dataclass(frozen=True)
class SequenceSafetyRewardConfig:
    """Configuration for AMR, toxin, and lysogeny sequence-safety rewards."""

    host_domain: HostDomain
    host_evidence: HostEvidence
    asset_manifest_path: Path
    diamond_bin: Path
    mmseqs_bin: Path
    policy_path: Path = Path("configs/phage_safety_policy.yaml")
    work_dir: Path = Path("data/checkpoints/phage_sequence_safety_reward")
    enabled: bool = True
    strict_lysis: bool = False
    circular: bool = True
    threads: int = 1
    batch_size: int = 1
    orf_workers: int = 1
    phrogs_threads: int = 1
    timeout_seconds: float = 300.0


def _sequence_safety_config_is_valid(config: object) -> bool:
    """Validate runtime types and bounds before any safety configuration can influence eligibility."""
    if type(config) is not SequenceSafetyRewardConfig:
        return False
    if type(config.host_domain) is not HostDomain or config.host_domain not in {
        HostDomain.BACTERIA,
        HostDomain.ARCHAEA,
        HostDomain.BACTERIA_AND_ARCHAEA,
    }:
        return False
    if type(config.host_evidence) is not HostEvidence:
        return False
    evidence = config.host_evidence
    if (
        type(evidence.source) is not str
        or (evidence.source_version is not None and type(evidence.source_version) is not str)
        or type(evidence.confirmed) is not bool
        or any(type(domain) is not HostDomain for domain in evidence.replication_host_domains)
    ):
        return False
    path_values = (
        config.asset_manifest_path,
        config.diamond_bin,
        config.mmseqs_bin,
        config.policy_path,
        config.work_dir,
    )
    if not all(isinstance(path, Path) for path in path_values):
        return False
    if any(type(value) is not bool for value in (config.enabled, config.strict_lysis, config.circular)):
        return False
    if any(
        type(value) is not int or value < 1
        for value in (config.threads, config.batch_size, config.orf_workers, config.phrogs_threads)
    ):
        return False
    if (
        not isinstance(config.timeout_seconds, Real)
        or isinstance(config.timeout_seconds, bool)
        or not math.isfinite(float(config.timeout_seconds))
        or config.timeout_seconds <= 0
    ):
        return False
    try:
        json.dumps(config.host_evidence.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


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
    if external_qc.tool_bin_dir is not None:
        tool_bin_dir = _recipe_path(external_qc.tool_bin_dir).resolve()
        env["PATH"] = os.pathsep.join((str(tool_bin_dir), env.get("PATH", "")))
    return env


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
    """Build the configured MMseqs easy-cluster command for batch diversity rewards."""
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
        "-v",
        str(int(config.verbosity)),
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
        return {row_index: (f"group{group_index}:seq_0", 1, 1.0)}, 1, 0

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
        if component.weight_attr is None:
            continue
        weight = float(getattr(weights, component.weight_attr))
        if weight > 0.0 and component.score_column in scored_df:
            active_components.append((weight, component))
    return active_components


def _exact_safety_gate_pass_mask(scored_df: pd.DataFrame) -> pd.Series:
    """Accept only a real numeric scalar equal to one, never bools or numeric strings."""
    values = scored_df.get("safety_gate_pass", pd.Series(0.0, index=scored_df.index))
    return values.map(lambda value: isinstance(value, Real) and not isinstance(value, bool) and value == 1.0)


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

    scored_df["reward_historical"] = weighted_sum / total_weight
    safety_gate_pass = _exact_safety_gate_pass_mask(scored_df)
    scored_df["reward_safety_penalty"] = (~safety_gate_pass).astype(float)
    scored_df["reward"] = (scored_df["reward_historical"] - scored_df["reward_safety_penalty"]).clip(0.0, 1.0)
    scored_df["reward_active_components"] = ",".join(component.name for _, component in active_components)
    scored_df["reward_total_weight"] = total_weight
    historical_binary_pass = _historical_binary_core_pass_mask(scored_df, weights)
    scored_df["reward_binary_historical_core_pass"] = historical_binary_pass.astype(float)
    scored_df["reward_binary_historical_core_cluster_deduplicated_pass"] = binary_cluster_deduplicated_pass_mask(
        scored_df,
        historical_binary_pass,
    ).astype(float)
    binary_pass = binary_core_pass_mask(scored_df, weights)
    scored_df["reward_binary_core_pass"] = binary_pass.astype(float)
    scored_df["reward_binary_core_cluster_deduplicated_pass"] = binary_cluster_deduplicated_pass_mask(
        scored_df,
        binary_pass,
    ).astype(float)
    historical_full_qc_pass = _historical_binary_full_qc_pass_mask(scored_df, historical_binary_pass)
    if historical_full_qc_pass is not None:
        scored_df["reward_binary_historical_full_qc_pass"] = historical_full_qc_pass.astype(float)
        scored_df["reward_binary_historical_full_qc_cluster_deduplicated_pass"] = (
            binary_cluster_deduplicated_pass_mask(
                scored_df,
                historical_full_qc_pass,
            ).astype(float)
        )
    full_qc_pass = binary_full_qc_pass_mask(scored_df, binary_pass)
    if full_qc_pass is not None:
        scored_df["reward_binary_full_qc_pass"] = full_qc_pass.astype(float)
        scored_df["reward_binary_full_qc_cluster_deduplicated_pass"] = binary_cluster_deduplicated_pass_mask(
            scored_df,
            full_qc_pass,
        ).astype(float)
    return scored_df


def _historical_binary_core_pass_mask(scored_df: pd.DataFrame, weights: RewardWeights) -> pd.Series:
    """Return the pre-safety binary pass mask for paper comparability."""
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


def binary_core_pass_mask(scored_df: pd.DataFrame, weights: RewardWeights) -> pd.Series:
    """Return the safety-qualified lab-facing binary pass mask."""
    historical_pass = _historical_binary_core_pass_mask(scored_df, weights)
    return historical_pass & _exact_safety_gate_pass_mask(scored_df)


def binary_cluster_deduplicated_pass_mask(scored_df: pd.DataFrame, pass_mask: pd.Series) -> pd.Series:
    """Return one passing representative per MMseqs cluster when cluster data is available."""
    if len(pass_mask) != len(scored_df):
        raise ValueError("binary pass mask length does not match scored rows")
    positions = pd.RangeIndex(len(scored_df))
    positional_pass = pd.Series(pass_mask.astype(bool).to_numpy(), index=positions)
    deduplicated = pd.Series(False, index=positions)
    if not {"mmseqs_cluster_id", "mmseqs_cluster_size"}.issubset(scored_df.columns):
        deduplicated.loc[positional_pass] = True
        return pd.Series(deduplicated.to_numpy(), index=scored_df.index)

    cluster_sizes = pd.Series(
        pd.to_numeric(scored_df["mmseqs_cluster_size"], errors="coerce").fillna(0).astype(int).to_numpy(),
        index=positions,
    )
    cluster_ids = pd.Series(scored_df["mmseqs_cluster_id"].astype(str).to_numpy(), index=positions)
    clustered_pass = positional_pass & (cluster_sizes > 0) & (cluster_ids != "")
    cluster_rows = pd.DataFrame({"cluster_id": cluster_ids}).loc[clustered_pass]
    for _cluster_id, cluster_df in cluster_rows.groupby("cluster_id", sort=False):
        deduplicated.iloc[cluster_df.index[0]] = True

    if "mmseqs_cluster_valid_for_clustering" in scored_df:
        valid_for_clustering = pd.Series(
            (
                pd.to_numeric(scored_df["mmseqs_cluster_valid_for_clustering"], errors="coerce").fillna(0.0) > 0.0
            ).to_numpy(),
            index=positions,
        )
        nonclusterable_pass = positional_pass & ~valid_for_clustering
    else:
        nonclusterable_pass = positional_pass & ~clustered_pass
    deduplicated.loc[nonclusterable_pass] = True
    return pd.Series(deduplicated.to_numpy(), index=scored_df.index)


def _historical_binary_full_qc_pass_mask(
    scored_df: pd.DataFrame,
    binary_pass: pd.Series,
) -> pd.Series | None:
    """Return the pre-safety core pass plus available full Arc QC gates."""
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


def binary_full_qc_pass_mask(scored_df: pd.DataFrame, binary_pass: pd.Series) -> pd.Series | None:
    """Return safety-qualified binary pass plus available full Arc QC gates."""
    pass_mask = _historical_binary_full_qc_pass_mask(scored_df, binary_pass)
    if pass_mask is None:
        return None
    return pass_mask & _exact_safety_gate_pass_mask(scored_df)


def _sequence_safety_required_by_class(config: SequenceSafetyRewardConfig) -> dict[str, bool]:
    """Record Task 4 applicability when a row has no usable scan manifest."""
    lysogeny_required = config.host_domain is not HostDomain.ARCHAEA or config.strict_lysis
    return {"amr": True, "toxin": True, "lysogeny": lysogeny_required}


def _add_unavailable_sequence_safety_rewards(
    scored_df: pd.DataFrame,
    *,
    reason_code: str,
    required_by_class: dict[str, bool] | None = None,
    strict_lysis: bool = False,
) -> pd.DataFrame:
    """Record zero reward and an explicit reason when required safety evidence is unavailable."""
    required_classes = dict.fromkeys(SEQUENCE_SAFETY_CLASSES, True) if required_by_class is None else required_by_class
    reasons_json = json.dumps([reason_code], separators=(",", ":"))
    defaults: dict[str, object] = {
        "safety_gate_state": "INDETERMINATE",
        "safety_gate_pass": 0.0,
        "safety_gate_reason_codes": reasons_json,
        "safety_environment_healthy": 0.0,
        "safety_gate_measurement_available": 0.0,
        "safety_required_class_count": sum(required_classes.values()),
        "safety_required_class_pass_count": 0,
        "safety_scan_record_id": "",
        "safety_scan_input_index": -1,
        "safety_policy_id": "",
        "safety_asset_state_path": "",
        "safety_scan_manifest_path": "",
        "safety_resolved_profile": "",
        "safety_amrfinder_version": "",
        "safety_diamond_version": "",
        "safety_mmseqs_version": "",
        "safety_strict_lysis": strict_lysis,
    }
    for safety_class in SEQUENCE_SAFETY_CLASSES:
        required = required_classes[safety_class]
        prefix = f"safety_{safety_class}"
        defaults.update(
            {
                f"{prefix}_state": "INDETERMINATE",
                f"{prefix}_required": float(required),
                f"{prefix}_reason_codes": reasons_json,
                f"{prefix}_finding_count": 0,
                f"{prefix}_measurement_available": 0.0,
                f"{prefix}_execution_status": "NOT_STARTED",
                f"{prefix}_policy_id": "",
                f"reward_safety_{safety_class}": float(not required),
            }
        )
    original_index = scored_df.index
    base = scored_df.drop(columns=[column for column in defaults if column in scored_df]).reset_index(drop=True)
    telemetry = pd.DataFrame({column: [value] * len(base) for column, value in defaults.items()}, index=base.index)
    combined = pd.concat([base, telemetry], axis=1).copy()
    combined.index = original_index
    return combined


def _sequence_is_scannable(sequence: object) -> bool:
    return isinstance(sequence, str) and re.fullmatch(r"[ACGTNacgtn]+", sequence) is not None


def _set_row_values(scored_df: pd.DataFrame, position: int, values: dict[str, object]) -> None:
    for column, value in values.items():
        if column not in scored_df:
            scored_df[column] = ""
        scored_df.iloc[position, scored_df.columns.get_loc(column)] = value


def _set_unavailable_reason(scored_df: pd.DataFrame, positions: list[int], reason_code: str) -> None:
    reasons = json.dumps([reason_code], separators=(",", ":"))
    for position in positions:
        values = {"safety_gate_reason_codes": reasons}
        values.update({f"safety_{name}_reason_codes": reasons for name in SEQUENCE_SAFETY_CLASSES})
        _set_row_values(scored_df, position, values)


def _json_reason_codes(value: object) -> str:
    if not isinstance(value, list) or not all(isinstance(reason, str) for reason in value):
        raise ValueError("sequence-safety reason codes must be a string list")
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _manifest_safety_row(
    record: object,
    *,
    expected_input_index: int,
    expected_record_id: str,
    manifest: dict[str, object],
    manifest_path: Path,
) -> dict[str, object]:
    """Convert one validated scan record into compact RL telemetry."""
    if not isinstance(record, dict):
        raise ValueError("sequence-safety record must be a mapping")
    if record.get("input_index") != expected_input_index or record.get("record_id") != expected_record_id:
        raise ValueError("sequence-safety record mapping changed")
    class_results = record.get("class_results")
    attempts = record.get("adapter_attempts")
    if not isinstance(class_results, list) or not isinstance(attempts, list):
        raise ValueError("sequence-safety class telemetry is missing")
    class_by_name = {item.get("safety_class"): item for item in class_results if isinstance(item, dict)}
    attempt_by_name = {item.get("safety_class"): item for item in attempts if isinstance(item, dict)}
    if set(class_by_name) != set(SEQUENCE_SAFETY_CLASSES) or set(attempt_by_name) != set(SEQUENCE_SAFETY_CLASSES):
        raise ValueError("sequence-safety class inventory changed")

    policy = manifest.get("policy")
    assets = manifest.get("asset_state")
    profile = manifest.get("resolved_profile")
    tools = manifest.get("tools")
    if not all(isinstance(value, dict) for value in (policy, assets, profile, tools)):
        raise ValueError("sequence-safety runtime state is incomplete")
    values: dict[str, object] = {
        "safety_scan_record_id": expected_record_id,
        "safety_scan_input_index": expected_input_index,
        "safety_gate_reason_codes": _json_reason_codes(record.get("reason_codes")),
        "safety_policy_id": policy.get("policy_id"),
        "safety_asset_state_path": assets.get("path"),
        "safety_scan_manifest_path": str(manifest_path),
        "safety_resolved_profile": profile.get("host_domain"),
        "safety_strict_lysis": profile.get("strict_lysis"),
        "safety_amrfinder_version": tools.get("amrfinder", {}).get("version", ""),
        "safety_diamond_version": tools.get("diamond", {}).get("version", ""),
        "safety_mmseqs_version": tools.get("mmseqs", {}).get("version", ""),
    }
    required_count = 0
    required_pass_count = 0
    required_measurements: list[bool] = []
    states: dict[str, str] = {}
    required_by_class: dict[str, bool] = {}
    review_eligible: dict[str, bool] = {}
    reasons: list[str] = []
    for safety_class in SEQUENCE_SAFETY_CLASSES:
        result = class_by_name[safety_class]
        attempt = attempt_by_name[safety_class]
        state = result.get("state")
        required = result.get("required")
        findings = result.get("findings")
        status = attempt.get("execution_status")
        class_reasons = result.get("reason_codes")
        if (
            state not in {"PASS", "FAIL", "INDETERMINATE"}
            or type(required) is not bool
            or not isinstance(findings, list)
            or not isinstance(status, str)
        ):
            raise ValueError("sequence-safety class telemetry is invalid")
        reasons.extend(class_reasons)
        measured = status == "COMPLETED_AND_PARSED"
        if required:
            required_count += 1
            required_pass_count += int(state == "PASS")
            required_measurements.append(measured)
        states[safety_class] = state
        required_by_class[safety_class] = required
        review_eligible[safety_class] = bool(required and state == "INDETERMINATE" and measured and findings)
        prefix = f"safety_{safety_class}"
        values.update(
            {
                f"{prefix}_state": state,
                f"{prefix}_required": float(required),
                f"{prefix}_reason_codes": _json_reason_codes(class_reasons),
                f"{prefix}_finding_count": len(findings),
                f"{prefix}_measurement_available": float(measured),
                f"{prefix}_execution_status": status,
                f"{prefix}_policy_id": attempt.get("policy_id", ""),
            }
        )
    safety_fields = sequence_safety_reward_fields(
        class_states=states,
        required_by_class=required_by_class,
        review_eligible_by_class=review_eligible,
    )
    if record.get("state") != safety_fields["safety_gate_state"] or record.get("reason_codes") != list(
        dict.fromkeys(reasons)
    ):
        raise ValueError("sequence-safety record aggregate is inconsistent")
    healthy = bool(required_measurements) and all(required_measurements)
    if safety_fields["safety_gate_state"] == "PASS" and not healthy:
        raise ValueError("sequence-safety PASS lacks completed required measurements")
    values.update(safety_fields)
    values["safety_environment_healthy"] = float(healthy)
    values["safety_gate_measurement_available"] = float(healthy)
    values["safety_required_class_count"] = required_count
    values["safety_required_class_pass_count"] = required_pass_count
    return values


def _sequence_safety_scan_argv(
    *,
    input_fasta: Path,
    output_dir: Path,
    config: SequenceSafetyRewardConfig,
) -> list[str]:
    argv = [
        "scan",
        "--input-fasta",
        str(input_fasta),
        "--output-dir",
        str(output_dir),
        "--policy",
        str(_recipe_path(config.policy_path)),
        "--asset-manifest",
        str(_recipe_path(config.asset_manifest_path)),
        "--host-domain",
        config.host_domain.value,
        "--host-evidence-json",
        json.dumps(config.host_evidence.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False),
        "--diamond-bin",
        str(_recipe_path(config.diamond_bin)),
        "--mmseqs-bin",
        str(_recipe_path(config.mmseqs_bin)),
        "--threads",
        str(config.threads),
        "--batch-size",
        str(config.batch_size),
        "--orf-workers",
        str(config.orf_workers),
        "--phrogs-threads",
        str(config.phrogs_threads),
        "--timeout",
        str(config.timeout_seconds),
    ]
    if config.strict_lysis:
        argv.append("--strict-lysis")
    if not config.circular:
        argv.append("--linear")
    return argv


def add_sequence_safety_rewards(
    scored_df: pd.DataFrame,
    config: SequenceSafetyRewardConfig,
) -> pd.DataFrame:
    """Run Task 4 and map its validated per-record results back to the batch."""
    if not _sequence_safety_config_is_valid(config):
        return _add_unavailable_sequence_safety_rewards(
            scored_df.copy(),
            reason_code="SEQUENCE_SAFETY_CONFIG_INVALID",
        )
    result = _add_unavailable_sequence_safety_rewards(
        scored_df.copy(),
        reason_code="SEQUENCE_SAFETY_RECORD_UNSCANNABLE",
        required_by_class=_sequence_safety_required_by_class(config),
        strict_lysis=config.strict_lysis,
    )
    valid_positions = [
        position for position, sequence in enumerate(result["sequence"].tolist()) if _sequence_is_scannable(sequence)
    ]
    if not valid_positions:
        return result
    _set_unavailable_reason(result, valid_positions, "SEQUENCE_SAFETY_SCAN_UNAVAILABLE")
    try:
        work_dir = _recipe_path(config.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = work_dir / f"batch_{uuid.uuid4().hex}"
        run_dir.mkdir()
        input_fasta = run_dir / "input.fasta"
        output_dir = run_dir / "scan"
        record_ids = [f"safety_record_{position:06d}" for position in valid_positions]
        scan_df = pd.DataFrame(
            {
                "id_prompt": record_ids,
                "sequence": [result.iloc[position]["sequence"] for position in valid_positions],
            }
        )
        save_fasta(scan_df, input_fasta)
        exit_code = sequence_safety_cli.main(
            _sequence_safety_scan_argv(
                input_fasta=input_fasta,
                output_dir=output_dir,
                config=config,
            )
        )
        if exit_code not in {0, 2, 3}:
            raise RuntimeError(f"sequence-safety scanner returned unsupported exit code {exit_code}")
        manifest_path = (output_dir / "manifest.json").absolute()
        try:
            manifest = sequence_safety_cli.validate_manifest_file(
                manifest_path,
                expected_type="sequence_safety_scan",
            )
            if not isinstance(manifest, dict) or manifest.get("manifest_type") != "sequence_safety_scan":
                raise ValueError("sequence-safety scan did not produce the expected result manifest")
            records = manifest.get("records")
            if not isinstance(records, list) or len(records) != len(valid_positions):
                raise ValueError("sequence-safety result count does not match the input batch")
            mapped_rows = [
                _manifest_safety_row(
                    record,
                    expected_input_index=scan_index,
                    expected_record_id=record_id,
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
                for scan_index, (record_id, record) in enumerate(zip(record_ids, records, strict=True))
            ]
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            _set_unavailable_reason(result, valid_positions, "SEQUENCE_SAFETY_MANIFEST_REJECTED")
            return result
        for position, values in zip(valid_positions, mapped_rows, strict=True):
            _set_row_values(result, position, values)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return result
    return result


def score_nucleotide_metrics(
    sequences_df: pd.DataFrame,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
    external_qc: ExternalQCRewardConfig | None = None,
    mmseqs_cluster_diversity: MMseqsClusterDiversityConfig | None = None,
    sequence_safety: SequenceSafetyRewardConfig | None = None,
) -> pd.DataFrame:
    """Score sequences with nucleotide QC, optional external QC, and optional batch diversity."""
    timings: dict[str, float] = {"reward/begin_unix_s": time.time()}
    reward_start = time.perf_counter()

    phase_start = time.perf_counter()
    df = add_nucleotide_metrics(sequences_df, config=config)
    _record_elapsed(timings, "reward/nucleotide_qc_s", phase_start)

    phase_start = time.perf_counter()
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
    _record_elapsed(timings, "reward/nucleotide_reward_scores_s", phase_start)

    if external_qc and external_qc.enabled:
        phase_start = time.perf_counter()
        df = add_external_qc_rewards(df, external_qc)
        timings.setdefault("reward/external_qc/total_s", time.perf_counter() - phase_start)
    if mmseqs_cluster_diversity and mmseqs_cluster_diversity.enabled:
        phase_start = time.perf_counter()
        df = add_mmseqs_cluster_diversity_rewards(df, config, mmseqs_cluster_diversity)
        _record_elapsed(timings, "reward/mmseqs_cluster_diversity_s", phase_start)

    if sequence_safety is None:
        df = _add_unavailable_sequence_safety_rewards(df, reason_code="SEQUENCE_SAFETY_CONFIG_MISSING")
    elif not _sequence_safety_config_is_valid(sequence_safety):
        df = _add_unavailable_sequence_safety_rewards(df, reason_code="SEQUENCE_SAFETY_CONFIG_INVALID")
    elif not sequence_safety.enabled:
        df = _add_unavailable_sequence_safety_rewards(
            df,
            reason_code="SEQUENCE_SAFETY_DISABLED",
            required_by_class=_sequence_safety_required_by_class(sequence_safety),
            strict_lysis=sequence_safety.strict_lysis,
        )
    else:
        phase_start = time.perf_counter()
        df = add_sequence_safety_rewards(df, sequence_safety)
        _record_elapsed(timings, "reward/sequence_safety_s", phase_start)

    phase_start = time.perf_counter()
    df = _aggregate_reward(df, weights)
    _record_elapsed(timings, "reward/aggregate_s", phase_start)
    timings["reward/end_unix_s"] = time.time()
    timings["reward/total_s"] = time.perf_counter() - reward_start
    return _attach_timing_columns(df, timings)


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
    config["online_measurement_mode"] = True
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
    if external_qc.lovis4u_mmseqs_threads is not None:
        config["lovis4u_mmseqs_threads"] = max(1, int(external_qc.lovis4u_mmseqs_threads))
    config["lovis4u_metrics_only"] = bool(external_qc.lovis4u_metrics_only)
    config["lovis4u_collect_pdfs"] = bool(external_qc.lovis4u_collect_pdfs)
    if paper_synteny_stage_enabled:
        config["use_reference_genome"] = full_synteny_enabled
        if not full_synteny_enabled and not bool(config.get("allow_gff_product_order_synteny_fallback", False)):
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


def _genome_ids_from_orf_ids(orf_ids: pd.Series) -> pd.Series:
    """Map Arc ORF IDs back to genome IDs."""
    return orf_ids.astype(str).str.split("_").str[:-1].str.join("_")


def _fasta_header_ids(path: Path) -> list[str]:
    """Read FASTA record IDs without loading sequence payloads."""
    if not path.exists():
        return []
    with path.open() as handle:
        return [line[1:].strip().split()[0] for line in handle if line.startswith(">")]


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


def _canonical_function(value: object) -> str:
    """Normalize PHROGs annotation text for supplementary unique-function metrics."""
    function = str(value).strip().lower()
    if function in {"", "nan", "none", "unknown", "unknown gene", "hypothetical protein"}:
        return ""
    return re.sub(r"\s+", " ", function)


def _add_predicted_orf_counts(scored_df: pd.DataFrame, run_dir: Path, config: dict) -> pd.DataFrame:
    """Add predicted ORF counts from Arc's ORF FASTA when available."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    proteins_path = run_dir / config.get("orfipy_proteins_file_save_location", "qc4_orfipy_proteins.fasta")
    orf_ids = pd.Series(_fasta_header_ids(proteins_path), dtype="object")
    if orf_ids.empty:
        return scored_df
    predicted_counts = _genome_ids_from_orf_ids(orf_ids).value_counts()
    scored_df["predicted_orf_count"] = scored_df[id_column].map(predicted_counts).fillna(0).astype(int)
    return scored_df


def _add_phrogs_hit_metrics(scored_df: pd.DataFrame, hits_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add supplementary PHROGs metrics without changing Arc-compatible hit counts."""
    id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
    hits_df = hits_df.copy()
    hits_df["arc_qc_id"] = _genome_ids_from_orf_hits(hits_df)
    hits_df["phrog_number"] = hits_df["protein_database_mmseqs_target"].map(_normalize_phrog_target)
    annotations = _load_phrog_annotations(config.get("protein_annotation_file", ""))
    hits_df = hits_df.merge(annotations, on="phrog_number", how="left")
    hits_df["canonical_function"] = hits_df["annot"].map(_canonical_function)

    rows = []
    for arc_qc_id, group in hits_df.groupby("arc_qc_id"):
        rows.append(
            {
                "arc_qc_id": arc_qc_id,
                "phrogs_hit_orf_count": int(group["id_prompt"].nunique()),
                "phrogs_annotated_orf_count": int(group.loc[group["canonical_function"] != "", "id_prompt"].nunique()),
                "unique_phrog_family_count": int(group["phrog_number"].nunique()),
                "unique_canonical_function_count": int(
                    group.loc[group["canonical_function"] != "", "canonical_function"].nunique()
                ),
            }
        )

    if not rows:
        return scored_df
    metrics_df = pd.DataFrame(rows).set_index("arc_qc_id")
    for column in [
        "phrogs_hit_orf_count",
        "phrogs_annotated_orf_count",
        "unique_phrog_family_count",
        "unique_canonical_function_count",
    ]:
        scored_df[column] = scored_df[id_column].map(metrics_df[column]).fillna(0).astype(int)
    if "predicted_orf_count" in scored_df:
        scored_df["phrogs_hit_fraction"] = [
            float(hit_count) / float(predicted_count) if float(predicted_count) > 0 else 0.0
            for hit_count, predicted_count in zip(
                scored_df["phrogs_hit_orf_count"],
                scored_df["predicted_orf_count"],
                strict=False,
            )
        ]
    return scored_df


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
    phrogs_dir = config.get("mmseqs_protein_database_results_dir_save_location")
    if not phrogs_dir:
        return scored_df
    phrogs_hits_path = run_dir / phrogs_dir / "mmseqs2_hits.csv"
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
        # Proxy mode only sees ORFs with PHROGs hits, not the true total ORF count.
        hit_gene_count = int(group["id_prompt"].nunique())
        rows.append(
            {
                "arc_qc_id": arc_qc_id,
                "synteny_required_gene_score": _required_gene_score(products, required_products),
                "synteny_order_score": _ordered_required_gene_score(products, required_products),
                "synteny_total_gene_score": _bounded_range_score(
                    hit_gene_count,
                    float(total_gene_range[0]),
                    float(total_gene_range[1]),
                ),
                "synteny_proxy_hit_gene_count": hit_gene_count,
                "synteny_proxy_gene_count": hit_gene_count,
            }
        )

    if rows:
        synteny_df = pd.DataFrame(rows).set_index("arc_qc_id")
        id_column = "arc_qc_id" if "arc_qc_id" in scored_df else "id_prompt"
        for column in [
            "synteny_required_gene_score",
            "synteny_order_score",
            "synteny_total_gene_score",
            "synteny_proxy_hit_gene_count",
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
    scored_df["synteny_stage_reached"] = 0.0
    scored_df["synteny_measurement_available"] = 0.0
    scored_df["synteny_missing_artifact"] = 0.0
    measured = pd.Series(False, index=scored_df.index)
    metrics_path = run_dir / config.get("synteny_metrics_file_save_location", "qc6_synteny_filter_metrics.csv")
    if not metrics_path.exists():
        metrics_path = run_dir / config.get("synteny_filter_seqs_csv_file_save_location", "")

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        if {"id_prompt", "num_syntenic_genes", "total_num_genes"}.issubset(metrics_df.columns):
            metrics_df = metrics_df.copy()
            metrics_df["num_syntenic_genes"] = pd.to_numeric(metrics_df["num_syntenic_genes"], errors="coerce")
            metrics_df["total_num_genes"] = pd.to_numeric(metrics_df["total_num_genes"], errors="coerce")
            if "missing_synteny_output" not in metrics_df:
                metrics_df["missing_synteny_output"] = False
            metrics_df["missing_synteny_output"] = metrics_df["missing_synteny_output"].astype(bool)
            metrics_by_id = metrics_df.set_index(metrics_df["id_prompt"].astype(str))
            row_ids = scored_df[id_column].astype(str)
            stage_reached = row_ids.isin(metrics_by_id.index)
            missing_artifact = row_ids.map(metrics_by_id["missing_synteny_output"]).eq(True)
            scored_df["num_syntenic_genes"] = row_ids.map(metrics_by_id["num_syntenic_genes"])
            scored_df["total_num_genes"] = row_ids.map(metrics_by_id["total_num_genes"])
            measured = (
                stage_reached
                & ~missing_artifact
                & scored_df["num_syntenic_genes"].notna()
                & scored_df["total_num_genes"].notna()
            )
            scored_df["synteny_stage_reached"] = stage_reached.astype(float)
            scored_df["synteny_measurement_available"] = measured.astype(float)
            scored_df["synteny_missing_artifact"] = missing_artifact.astype(float)

            scores = [
                _synteny_distance_score(float(num_syntenic), float(total_genes))
                if is_measured
                else (0.0, pd.NA, pd.NA, pd.NA)
                for num_syntenic, total_genes, is_measured in zip(
                    scored_df["num_syntenic_genes"],
                    scored_df["total_num_genes"],
                    measured,
                    strict=False,
                )
            ]
            scored_df["reward_external_synteny"] = [score for score, _, _, _ in scores]
            scored_df["synteny_total_gene_score"] = [total_score for _, total_score, _, _ in scores]
            scored_df["synteny_pair_score"] = [pair_score for _, _, pair_score, _ in scores]
            scored_df["synteny_pair_distance"] = [pair_distance for _, _, _, pair_distance in scores]
            scored_df["syntenic_gene_count_score"] = scored_df["synteny_pair_score"]

    synteny_csv = run_dir / config.get("synteny_filter_seqs_csv_file_save_location", "")
    if config.get("online_measurement_mode", False):
        pass_mask = pd.Series(
            [
                bool(is_measured) and (int(num_syntenic), int(total_genes)) in ARC_VALID_SYNTENY_PAIRS
                for num_syntenic, total_genes, is_measured in zip(
                    scored_df.get("num_syntenic_genes", pd.Series(0, index=scored_df.index)),
                    scored_df.get("total_num_genes", pd.Series(0, index=scored_df.index)),
                    measured,
                    strict=False,
                )
            ],
            index=scored_df.index,
        )
        scored_df["reward_external_synteny_pass"] = pass_mask.astype(float)
    elif synteny_csv.exists():
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
    scored_df["average_protein_identity_stage_reached"] = 0.0
    scored_df["average_protein_identity_measurement_available"] = 0.0
    scored_df["average_protein_identity_missing_artifact"] = 0.0
    metrics_path = run_dir / config.get(
        "average_protein_sequence_identity_metrics_file_save_location",
        "qc6_average_protein_sequence_identity_metrics.csv",
    )
    if not metrics_path.exists():
        metrics_path = run_dir / config.get("synteny_filter_seqs_csv_file_save_location", "")
    if not metrics_path.exists():
        scored_df["average_protein_identity_missing_artifact"] = 1.0
        return scored_df

    metrics_df = pd.read_csv(metrics_path)
    if not {"id_prompt", "average_protein_percent_identity"}.issubset(metrics_df.columns):
        scored_df["average_protein_identity_missing_artifact"] = 1.0
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
    scored_df["average_protein_identity_stage_reached"] = (
        scored_df[id_column].astype(str).isin(metrics_by_id.index).astype(float)
    )
    scored_df["average_protein_identity_measurement_available"] = has_identity_metric.astype(float)
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
    scored_df["required_genes_stage_reached"] = 0.0
    scored_df["required_genes_measurement_available"] = 0.0
    scored_df["required_genes_missing_artifact"] = 0.0
    metrics_path = run_dir / config.get("required_genes_metrics_file_save_location", "qc6_required_genes_metrics.csv")
    if not metrics_path.exists():
        metrics_path = run_dir / config.get("synteny_filter_seqs_csv_file_save_location", "")
    if not metrics_path.exists():
        scored_df["required_genes_missing_artifact"] = 1.0
        return scored_df

    metrics_df = pd.read_csv(metrics_path)
    required_columns = {"id_prompt", "required_genes_matched_count", "required_genes_total_count"}
    if not required_columns.issubset(metrics_df.columns):
        scored_df["required_genes_missing_artifact"] = 1.0
        return scored_df

    metrics_df = metrics_df.copy()
    for column in ["required_genes_matched_count", "required_genes_total_count"]:
        metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce").fillna(0.0)
    metrics_by_id = metrics_df.set_index(metrics_df["id_prompt"].astype(str))
    mapped_matched = scored_df[id_column].astype(str).map(metrics_by_id["required_genes_matched_count"])
    mapped_total = scored_df[id_column].astype(str).map(metrics_by_id["required_genes_total_count"])
    has_required_gene_metric = mapped_matched.notna() & mapped_total.notna()
    scored_df["required_genes_stage_reached"] = (
        scored_df[id_column].astype(str).isin(metrics_by_id.index).astype(float)
    )
    scored_df["required_genes_measurement_available"] = has_required_gene_metric.astype(float)
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
    scored_df = _add_predicted_orf_counts(scored_df, run_dir, config)
    phrogs_dir = config.get("mmseqs_protein_database_results_dir_save_location")
    phrogs_hits_path = run_dir / phrogs_dir / "mmseqs2_hits.csv" if phrogs_dir else None
    scored_df["protein_database_hit_count_stage_reached"] = 0.0
    scored_df["protein_database_hit_count_measurement_available"] = 0.0
    scored_df["protein_database_hit_count_missing_artifact"] = 0.0
    scored_df["protein_database_hit_count_hit_present"] = 0.0
    if phrogs_hits_path and phrogs_hits_path.exists():
        scored_df["protein_database_hit_count_stage_reached"] = 1.0
        scored_df["protein_database_hit_count_measurement_available"] = 1.0
        hits_df = pd.read_csv(phrogs_hits_path)
        if {"id_prompt", "protein_database_mmseqs_target"}.issubset(hits_df.columns):
            scored_df = _add_phrogs_hit_metrics(scored_df, hits_df, config)
            genome_counts = _genome_ids_from_orf_hits(hits_df).value_counts()
            min_hits = int(config.get("protein_database_hit_count", 7))
            scored_df["protein_database_hit_count"] = scored_df[id_column].map(genome_counts).fillna(0).astype(int)
            scored_df["protein_database_hit_count_hit_present"] = (scored_df["protein_database_hit_count"] > 0).astype(
                float
            )
            scored_df["reward_external_protein_hit_count"] = scored_df["protein_database_hit_count"].map(
                lambda value: _lower_bound_ratio_score(float(value), float(min_hits))
            )
            scored_df["reward_external_protein_hit_count_pass"] = (
                scored_df["protein_database_hit_count"] >= min_hits
            ).astype(float)
    elif phrogs_hits_path:
        scored_df["protein_database_hit_count_missing_artifact"] = 1.0

    tropism_dir = config.get("mmseqs_tropism_protein_results_dir_save_location")
    tropism_hits_path = run_dir / tropism_dir / "mmseqs2_hits.csv" if tropism_dir else None
    scored_df["tropism_stage_reached"] = 0.0
    scored_df["tropism_measurement_available"] = 0.0
    scored_df["tropism_missing_artifact"] = 0.0
    scored_df["tropism_hit_present"] = 0.0
    if tropism_hits_path and tropism_hits_path.exists():
        scored_df["tropism_stage_reached"] = 1.0
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
            scored_df["tropism_measurement_available"] = 1.0
            scored_df["tropism_hit_present"] = measured_hit.astype(float)
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
    elif tropism_hits_path:
        scored_df["tropism_missing_artifact"] = 1.0
    return scored_df


def add_external_qc_rewards(
    scored_df: pd.DataFrame,
    external_qc: ExternalQCRewardConfig,
) -> pd.DataFrame:
    """Run Arc external QC on a batch and add binary staged reward columns."""
    timings: dict[str, float] = {"reward/external_qc/begin_unix_s": time.time()}
    external_start = time.perf_counter()

    def finish_timing(df: pd.DataFrame) -> pd.DataFrame:
        timings["reward/external_qc/end_unix_s"] = time.time()
        timings["reward/external_qc/total_s"] = time.perf_counter() - external_start
        return _attach_timing_columns(df, timings)

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
    df["external_qc_tool_succeeded"] = 0.0
    df["external_qc_measurement_available"] = 0.0

    external_qc_failed = False
    try:
        phase_start = time.perf_counter()
        df["arc_qc_id"] = [f"umi{i + 1}" for i in range(len(df))]
        save_fasta(
            df.rename(columns={"id_prompt": "original_id_prompt", "arc_qc_id": "id_prompt"})[
                ["id_prompt", "sequence"]
            ],
            input_fasta,
        )
        run_config_path = _write_external_qc_config(base_config_path, run_dir, input_fasta, external_qc)
        _record_elapsed(timings, "reward/external_qc/prepare_inputs_s", phase_start)
        try:
            timings["reward/external_qc/subprocess_begin_unix_s"] = time.time()
            phase_start = time.perf_counter()
            try:
                subprocess.run(
                    [sys.executable, str(pipeline_script), str(run_config_path)],
                    check=True,
                    cwd=str(pipeline_script.parent),
                    env=_external_qc_env(external_qc),
                    timeout=external_qc.timeout_seconds,
                )
            finally:
                _record_elapsed(timings, "reward/external_qc/subprocess_s", phase_start)
                timings["reward/external_qc/subprocess_end_unix_s"] = time.time()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            external_qc_failed = True
            message = (
                f"Arc external QC failed for {run_dir}; failed artifacts were retained and "
                "external reward columns remain at 0.0"
            )
            if external_qc.fail_on_error:
                raise RuntimeError(message) from exc
            warnings.warn(f"{message}: {exc}", RuntimeWarning, stacklevel=2)
            return finish_timing(df)
        df["external_qc_tool_succeeded"] = 1.0
        df["external_qc_measurement_available"] = 1.0

        phase_start = time.perf_counter()
        config = yaml.safe_load(run_config_path.read_text())
        orf_csv_name = config.get("orf_filter_seqs_csv_file_save_location")
        orf_pass_ids = _sequence_ids_from_csv(run_dir / orf_csv_name) if orf_csv_name else set()
        if external_qc.enable_orf:
            df["reward_external_orf"] = df["arc_qc_id"].astype(str).isin(orf_pass_ids).astype(float)
        if external_qc.enable_coding_density:
            df["reward_external_coding_density"] = df["arc_qc_id"].astype(str).isin(orf_pass_ids).astype(float)
        _record_elapsed(timings, "reward/external_qc/parse_orf_s", phase_start)

        phase_start = time.perf_counter()
        df = _add_mmseqs_hit_rewards(df, run_dir, config)
        _record_elapsed(timings, "reward/external_qc/parse_protein_hit_count_tropism_s", phase_start)
        if external_qc.enable_synteny:
            phase_start = time.perf_counter()
            if str(external_qc.synteny_mode).lower() == "full":
                df = _add_full_synteny_rewards(df, run_dir, config)
            else:
                df = _add_synteny_proxy_rewards(df, run_dir, config)
            _record_elapsed(timings, "reward/external_qc/parse_synteny_s", phase_start)
        if external_qc.enable_average_protein_identity:
            phase_start = time.perf_counter()
            df = _add_average_protein_identity_rewards(
                df,
                run_dir,
                config,
            )
            _record_elapsed(timings, "reward/external_qc/parse_average_protein_identity_s", phase_start)
        if external_qc.enable_required_genes:
            phase_start = time.perf_counter()
            df = _add_required_gene_rewards(
                df,
                run_dir,
                config,
                external_qc.required_genes_evidence_target,
            )
            _record_elapsed(timings, "reward/external_qc/parse_required_genes_s", phase_start)
    finally:
        if not external_qc.keep_artifacts and not external_qc_failed:
            shutil.rmtree(run_dir, ignore_errors=True)
    return finish_timing(df)


def score_fasta(
    input_fasta: Path,
    output_csv: Path,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
    mmseqs_cluster_diversity: MMseqsClusterDiversityConfig | None = None,
    sequence_safety: SequenceSafetyRewardConfig | None = None,
) -> Path:
    """Score a FASTA file and write per-sequence reward diagnostics."""
    sequences_df = load_fasta_records(input_fasta)
    scored_df = score_nucleotide_metrics(
        sequences_df,
        config=config,
        weights=weights,
        mmseqs_cluster_diversity=mmseqs_cluster_diversity,
        sequence_safety=sequence_safety,
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
