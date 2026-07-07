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

"""NeMo-RL environment wrapper for online phage sequence rewards."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from bionemo.evo2_phage_gen.qc import NucleotideQCConfig
from bionemo.evo2_phage_gen.reward import (
    REWARD_COMPONENTS,
    ExternalQCRewardConfig,
    MMseqsClusterDiversityConfig,
    RewardWeights,
    binary_cluster_deduplicated_pass_mask,
    binary_core_pass_mask,
    binary_full_qc_pass_mask,
    score_nucleotide_metrics,
)


@dataclass(frozen=True)
class GDPOObjective:
    """One positional GDPO reward objective derived from reward dataframe columns."""

    name: str
    columns: tuple[str, ...]
    reducer: str = "mean"


DEFAULT_GDPO_OBJECTIVES: tuple[GDPOObjective, ...] = (
    GDPOObjective(
        name="feasibility",
        columns=(
            "reward_valid_nt_chars",
            "reward_genome_length",
            "reward_gc_content",
            "reward_nt_homopolymer",
            "reward_dustmask_end",
            "reward_nucleotide_pass",
        ),
    ),
    GDPOObjective(
        name="function",
        columns=(
            "reward_external_protein_hit_count",
            "reward_external_tropism",
            "reward_external_required_genes",
        ),
    ),
    GDPOObjective(name="architecture", columns=("reward_external_synteny",)),
    GDPOObjective(
        name="novelty",
        columns=(
            "reward_external_average_protein_identity",
            "reward_mmseqs_cluster_diversity",
        ),
    ),
)


def _coerce_gdpo_objectives(raw_objectives: Any) -> tuple[GDPOObjective, ...]:
    """Parse GDPO objective config into a stable positional objective list."""
    if not raw_objectives:
        return DEFAULT_GDPO_OBJECTIVES

    objectives: list[GDPOObjective] = []
    for raw in raw_objectives:
        if not isinstance(raw, dict):
            raise TypeError("Each gdpo_objectives entry must be a mapping with name and columns.")
        name = str(raw.get("name", "")).strip()
        columns = tuple(str(column) for column in raw.get("columns", ()) if str(column).strip())
        reducer = str(raw.get("reducer", "mean")).strip().lower()
        if not name or not columns:
            raise ValueError("Each gdpo_objectives entry must define a non-empty name and columns list.")
        objectives.append(GDPOObjective(name=name, columns=columns, reducer=reducer))
    return tuple(objectives)


def gdpo_objective_scores_from_scored(
    scored: pd.DataFrame,
    objectives: tuple[GDPOObjective, ...],
) -> pd.DataFrame:
    """Build a positional GDPO reward matrix from scored reward columns."""
    objective_scores = pd.DataFrame(index=scored.index)
    for objective in objectives:
        missing_columns = [column for column in objective.columns if column not in scored]
        if missing_columns:
            raise ValueError(
                f"GDPO objective {objective.name!r} missing reward column(s): {', '.join(missing_columns)}"
            )
        values = scored[list(objective.columns)].astype(float).clip(0.0, 1.0)
        if objective.reducer == "mean":
            objective_scores[objective.name] = values.mean(axis=1)
        elif objective.reducer == "product":
            objective_scores[objective.name] = values.prod(axis=1)
        elif objective.reducer == "min":
            objective_scores[objective.name] = values.min(axis=1)
        else:
            raise ValueError(
                f"Unsupported GDPO reducer {objective.reducer!r} for objective {objective.name!r}; "
                "expected 'mean', 'product', or 'min'."
            )
    return objective_scores


try:  # pragma: no cover - exercised only when NeMo-RL is installed.
    import ray
    import torch
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
except ModuleNotFoundError as exc:  # pragma: no cover
    _NEMO_RL_IMPORT_ERROR = exc
else:  # pragma: no cover
    _NEMO_RL_IMPORT_ERROR = None


def extract_assistant_sequence(message_log: list[dict[str, Any]]) -> str:
    """Concatenate assistant messages into the generated DNA sequence."""
    return "".join(str(message["content"]) for message in message_log if message.get("role") == "assistant")


def _prompt_nucleotides(message_log: list[dict[str, Any]]) -> str:
    """Extract DNA bases from user prompts while dropping fine-tuning soft tokens."""
    prompt = "".join(str(message["content"]) for message in message_log if message.get("role") == "user")
    return "".join(char for char in prompt.upper() if char in {"A", "C", "G", "T"})


def extract_scored_sequence(message_log: list[dict[str, Any]]) -> str:
    """Build the sequence scored by QC: nucleotide prompt prefix plus raw assistant text."""
    return _prompt_nucleotides(message_log) + extract_assistant_sequence(message_log)


def score_message_logs(
    message_log_batch: list[list[dict[str, Any]]],
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
    external_qc: ExternalQCRewardConfig | None = None,
    mmseqs_cluster_diversity: MMseqsClusterDiversityConfig | None = None,
) -> pd.DataFrame:
    """Score a NeMo-RL message-log batch with the dependency-light phage reward."""
    sequences_df = pd.DataFrame(
        {
            "id_prompt": [str(i) for i in range(len(message_log_batch))],
            "prompt_nt_length": [len(_prompt_nucleotides(message_log)) for message_log in message_log_batch],
            "prompt_group": [_prompt_nucleotides(message_log) for message_log in message_log_batch],
            "sequence": [extract_scored_sequence(message_log) for message_log in message_log_batch],
        }
    )
    return score_nucleotide_metrics(
        sequences_df,
        config=config,
        weights=weights,
        external_qc=external_qc,
        mmseqs_cluster_diversity=mmseqs_cluster_diversity,
    )


def phage_qc_metrics_from_scored(scored: pd.DataFrame, weights: RewardWeights) -> dict[str, float | int]:
    """Summarize per-sequence phage QC scores into scalar logger metrics."""
    if scored.empty:
        return {"num_sequences": 0}

    metrics: dict[str, float | int] = {"num_sequences": len(scored)}
    for component in REWARD_COMPONENTS:
        if component.score_column in scored:
            metrics[f"{component.name}_score_mean"] = float(scored[component.score_column].astype(float).mean())
            metrics[f"{component.name}_pass_rate"] = float(
                (scored[component.score_column].astype(float) >= 1.0).mean()
            )

    for column in [
        "prompt_nt_length",
        "genome_length",
        "gc_content",
        "max_nt_homopolymer_length",
        "dustmask_masked_bases",
        "dustmask_masked_fraction",
        "dustmask_left_end_masked_bases",
        "dustmask_left_end_masked_fraction",
        "dustmask_right_end_masked_bases",
        "dustmask_right_end_masked_fraction",
        "dustmask_max_end_masked_fraction",
        "reward_dustmask_end",
        "protein_database_hit_count",
        "tropism_protein_mmseqs_percent_identity",
        "tropism_protein_measured_hit",
        "num_syntenic_genes",
        "total_num_genes",
        "syntenic_gene_count_score",
        "synteny_pair_score",
        "synteny_pair_distance",
        "synteny_total_gene_score",
        "average_protein_percent_identity",
        "average_protein_identity_gene_count",
        "average_protein_identity_raw_score",
        "average_protein_identity_novelty_score",
        "average_protein_identity_evidence_score",
        "required_genes_matched_count",
        "required_genes_total_count",
        "required_genes_raw_score",
        "required_genes_evidence_score",
        "reward_mmseqs_cluster_diversity",
        "mmseqs_cluster_size",
        "mmseqs_cluster_is_singleton",
        "mmseqs_cluster_valid_for_clustering",
        "mmseqs_cluster_missing_from_output",
        "reward",
        "reward_binary_core_pass",
        "reward_binary_core_cluster_deduplicated_pass",
        "reward_binary_full_qc_pass",
        "reward_binary_full_qc_cluster_deduplicated_pass",
    ]:
        if column in scored:
            values = pd.to_numeric(scored[column], errors="coerce").fillna(0.0)
            metrics[f"{column}_mean"] = float(values.mean())
            if column == "prompt_nt_length":
                metrics[f"{column}_min"] = float(values.min())
                metrics[f"{column}_max"] = float(values.max())

    if "mmseqs_cluster_size" in scored:
        cluster_sizes = pd.to_numeric(scored["mmseqs_cluster_size"], errors="coerce").fillna(0).astype(int)
        valid_cluster_mask = cluster_sizes > 0
        valid_cluster_count = int(valid_cluster_mask.sum())
        batch_size = len(scored)
        if valid_cluster_count > 0:
            cluster_rows = scored.loc[
                valid_cluster_mask,
                ["mmseqs_cluster_id", "mmseqs_cluster_size"],
            ].drop_duplicates()
            num_clusters = int(cluster_rows["mmseqs_cluster_id"].astype(str).nunique())
            metrics["mmseqs_cluster_num_clusters"] = num_clusters
            metrics["mmseqs_cluster_clusters_per_sequence"] = float(num_clusters / max(batch_size, 1))
            metrics["mmseqs_cluster_clusters_per_valid_sequence"] = float(num_clusters / valid_cluster_count)
            metrics["mmseqs_cluster_singleton_fraction"] = float((cluster_sizes[valid_cluster_mask] == 1).mean())
            metrics["mmseqs_cluster_largest_cluster_fraction"] = float(cluster_sizes.max() / valid_cluster_count)
            for size, count in cluster_rows["mmseqs_cluster_size"].astype(int).value_counts().sort_index().items():
                metrics[f"mmseqs_cluster_size_histogram/size_{int(size)}"] = int(count)
        else:
            metrics["mmseqs_cluster_num_clusters"] = 0
            metrics["mmseqs_cluster_clusters_per_sequence"] = 0.0
            metrics["mmseqs_cluster_clusters_per_valid_sequence"] = 0.0
            metrics["mmseqs_cluster_singleton_fraction"] = 0.0
            metrics["mmseqs_cluster_largest_cluster_fraction"] = 0.0

    binary_pass = binary_core_pass_mask(scored, weights)
    cluster_deduplicated_pass = binary_cluster_deduplicated_pass_mask(scored, binary_pass)
    pass_count = int(binary_pass.sum())
    cluster_deduplicated_count = int(cluster_deduplicated_pass.sum())
    metrics["binary_core_pass_count"] = pass_count
    metrics["binary_core_pass_rate"] = float(binary_pass.astype(float).mean())
    metrics["binary_core_pass_cluster_deduplicated_count"] = cluster_deduplicated_count
    metrics["binary_core_pass_cluster_deduplicated_rate"] = float(cluster_deduplicated_pass.astype(float).mean())
    metrics["binary_core_pass_cluster_duplicate_count"] = max(0, pass_count - cluster_deduplicated_count)
    metrics["binary_core_pass_cluster_deduplication_fraction"] = (
        0.0 if pass_count == 0 else float(cluster_deduplicated_count / pass_count)
    )

    full_qc_pass = binary_full_qc_pass_mask(scored, binary_pass)
    if full_qc_pass is not None:
        full_qc_cluster_deduplicated_pass = binary_cluster_deduplicated_pass_mask(scored, full_qc_pass)
        full_qc_pass_count = int(full_qc_pass.sum())
        full_qc_cluster_deduplicated_count = int(full_qc_cluster_deduplicated_pass.sum())
        metrics["binary_full_qc_pass_count"] = full_qc_pass_count
        metrics["binary_full_qc_pass_rate"] = float(full_qc_pass.astype(float).mean())
        metrics["binary_full_qc_pass_cluster_deduplicated_count"] = full_qc_cluster_deduplicated_count
        metrics["binary_full_qc_pass_cluster_deduplicated_rate"] = float(
            full_qc_cluster_deduplicated_pass.astype(float).mean()
        )
        metrics["binary_full_qc_pass_cluster_duplicate_count"] = max(
            0,
            full_qc_pass_count - full_qc_cluster_deduplicated_count,
        )
        metrics["binary_full_qc_pass_cluster_deduplication_fraction"] = (
            0.0 if full_qc_pass_count == 0 else float(full_qc_cluster_deduplicated_count / full_qc_pass_count)
        )
    return metrics


if _NEMO_RL_IMPORT_ERROR is None:  # pragma: no cover

    @ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)
    class PhageQCEnvironment(EnvironmentInterface[dict[str, Any]]):
        """Single-turn NeMo-RL environment for phage sequence QC reward."""

        def __init__(self, cfg: dict[str, Any]):
            """Create a phage QC environment from NeMo-RL environment config."""
            self.cfg = cfg
            self.config = NucleotideQCConfig(
                genome_length_min=int(cfg.get("genome_length_min", 4000)),
                genome_length_max=int(cfg.get("genome_length_max", 6000)),
                gc_content_min=float(cfg.get("gc_content_min", 30.0)),
                gc_content_max=float(cfg.get("gc_content_max", 65.0)),
                homopolymer_max=int(cfg.get("homopolymer_max", 10)),
                dustmask_filter=bool(cfg.get("dustmask_filter", False)),
                dustmasker_bin=str(cfg.get("dustmasker_bin", "dustmasker")),
                dustmask_use_external=bool(cfg.get("dustmask_use_external", True)),
                dustmask_window=int(cfg.get("dustmask_window", 64)),
                dustmask_level=float(cfg.get("dustmask_level", 20.0)),
                dustmask_end_window=int(cfg.get("dustmask_end_window", 200)),
                dustmask_max_end_fraction=float(cfg.get("dustmask_max_end_fraction", 0.9)),
            )
            self.weights = RewardWeights(
                valid_nt_chars=float(cfg.get("weight_valid_nt_chars", 1.0)),
                genome_length=float(cfg.get("weight_genome_length", 1.0)),
                gc_content=float(cfg.get("weight_gc_content", 1.0)),
                nt_homopolymer=float(cfg.get("weight_nt_homopolymer", 1.0)),
                dustmask_end=float(cfg.get("weight_dustmask_end", 0.0)),
                nucleotide_pass=float(cfg.get("weight_nucleotide_pass", 0.0)),
                orf=float(cfg.get("weight_orf", 0.0)),
                coding_density=float(cfg.get("weight_coding_density", 0.0)),
                protein_hit_count=float(cfg.get("weight_protein_hit_count", 0.0)),
                tropism=float(cfg.get("weight_tropism", 0.0)),
                synteny=float(cfg.get("weight_synteny", 0.0)),
                average_protein_identity=float(cfg.get("weight_average_protein_identity", 0.0)),
                required_genes=float(cfg.get("weight_required_genes", 0.0)),
                mmseqs_cluster_diversity=float(cfg.get("weight_mmseqs_cluster_diversity", 0.0)),
            )
            reward_output_mode = str(cfg.get("reward_output_mode", "scalar")).strip().lower()
            if reward_output_mode == "grpo":
                reward_output_mode = "scalar"
            if reward_output_mode not in {"scalar", "gdpo"}:
                raise ValueError("reward_output_mode must be 'scalar', 'grpo', or 'gdpo'.")
            self.reward_output_mode = reward_output_mode
            self.gdpo_objectives = _coerce_gdpo_objectives(cfg.get("gdpo_objectives"))
            external_qc_cfg = cfg.get("external_qc", {}) or {}
            self.external_qc = ExternalQCRewardConfig(
                enabled=bool(external_qc_cfg.get("enabled", False)),
                config_path=external_qc_cfg.get("config_path", "configs/arc_genome_design_filtering_local.yaml"),
                pipeline_script=external_qc_cfg.get(
                    "pipeline_script", "data/arc_pipeline_patched/genome_design_filtering_pipeline.py"
                ),
                work_dir=external_qc_cfg.get("work_dir", "data/checkpoints/phage_grpo_external_qc"),
                keep_artifacts=bool(external_qc_cfg.get("keep_artifacts", False)),
                enable_orf=bool(external_qc_cfg.get("enable_orf", False)),
                enable_coding_density=bool(external_qc_cfg.get("enable_coding_density", False)),
                enable_protein_hit_count=bool(external_qc_cfg.get("enable_protein_hit_count", True)),
                enable_tropism=bool(external_qc_cfg.get("enable_tropism", True)),
                enable_synteny=bool(external_qc_cfg.get("enable_synteny", False)),
                synteny_mode=str(external_qc_cfg.get("synteny_mode", "proxy")),
                enable_average_protein_identity=bool(external_qc_cfg.get("enable_average_protein_identity", False)),
                enable_required_genes=bool(external_qc_cfg.get("enable_required_genes", False)),
                required_genes_evidence_target=float(external_qc_cfg.get("required_genes_evidence_target", 9.0)),
                lovis4u_parallel_jobs=external_qc_cfg.get("lovis4u_parallel_jobs", 12),
                lovis4u_chunk_size=external_qc_cfg.get("lovis4u_chunk_size"),
                lovis4u_collect_pdfs=bool(external_qc_cfg.get("lovis4u_collect_pdfs", False)),
            )
            mmseqs_cfg = cfg.get("mmseqs_cluster_diversity", {}) or {}
            self.mmseqs_cluster_diversity = MMseqsClusterDiversityConfig(
                enabled=bool(mmseqs_cfg.get("enabled", False)),
                mmseqs_bin=str(mmseqs_cfg.get("mmseqs_bin", "mmseqs")),
                work_dir=mmseqs_cfg.get("work_dir", "data/checkpoints/phage_grpo_mmseqs_cluster_diversity"),
                keep_artifacts=bool(mmseqs_cfg.get("keep_artifacts", False)),
                min_seq_id=float(mmseqs_cfg.get("min_seq_id", 0.99)),
                coverage=float(mmseqs_cfg.get("coverage", 0.0)),
                cov_mode=int(mmseqs_cfg.get("cov_mode", 0)),
                seq_id_mode=int(mmseqs_cfg.get("seq_id_mode", 0)),
                cluster_mode=int(mmseqs_cfg.get("cluster_mode", 0)),
                threads=mmseqs_cfg.get("threads"),
            )

        def step(
            self,
            message_log_batch: list[list[dict[str, Any]]],
            metadata: list[dict[str, Any]],
        ) -> EnvironmentReturn[dict[str, Any]]:
            """Score generated assistant sequences and terminate each rollout."""
            scored = score_message_logs(
                message_log_batch,
                config=self.config,
                weights=self.weights,
                external_qc=self.external_qc,
                mmseqs_cluster_diversity=self.mmseqs_cluster_diversity,
            )
            self._last_scored = scored
            if self.reward_output_mode == "gdpo":
                objective_scores = gdpo_objective_scores_from_scored(scored, self.gdpo_objectives)
                self._last_gdpo_objective_scores = objective_scores
                rewards = torch.tensor(objective_scores.to_numpy(dtype=float), dtype=torch.float32).cpu()
            else:
                self._last_gdpo_objective_scores = pd.DataFrame()
                rewards = torch.tensor(scored["reward"].tolist(), dtype=torch.float32).cpu()
            observations = [
                {"role": "environment", "content": f"phage_qc_reward={reward:.6f}"}
                for reward in scored["reward"].tolist()
            ]
            return EnvironmentReturn(
                observations=observations,
                metadata=metadata,
                next_stop_strings=[None] * len(message_log_batch),
                rewards=rewards,
                terminateds=torch.ones(rewards.shape[0], dtype=torch.bool).cpu(),
                answers=scored["sequence"].tolist(),
            )

        def global_post_process_and_metrics(
            self, batch: BatchedDataDict
        ) -> tuple[BatchedDataDict, dict[str, float | int]]:
            """Report rollout-level reward metrics."""
            reward_tensor = batch["rewards"] if "rewards" in batch else batch["total_reward"]
            rewards = reward_tensor if reward_tensor.ndim == 1 else reward_tensor.float().mean(dim=1)
            scored = getattr(self, "_last_scored", pd.DataFrame())
            phage_metrics = phage_qc_metrics_from_scored(scored, self.weights) if not scored.empty else {}
            binary_pass_rate = float(phage_metrics.get("binary_core_pass_rate", 0.0))
            cluster_deduplicated_pass_rate = float(
                phage_metrics.get("binary_core_pass_cluster_deduplicated_rate", binary_pass_rate)
            )
            metrics = {
                "mean_reward": rewards.float().mean().item(),
                "pass_rate": cluster_deduplicated_pass_rate,
                "binary_core_pass_rate": binary_pass_rate,
                "binary_core_pass_cluster_deduplicated_rate": cluster_deduplicated_pass_rate,
                "dense_reward_ge_1_rate": (rewards >= 1.0).float().mean().item(),
                "num_sequences": int(rewards.shape[0]),
            }
            objective_scores = getattr(self, "_last_gdpo_objective_scores", pd.DataFrame())
            if not objective_scores.empty:
                metrics["gdpo/num_objectives"] = int(objective_scores.shape[1])
                for objective_name in objective_scores.columns:
                    metrics[f"gdpo/{objective_name}_mean"] = float(
                        objective_scores[objective_name].astype(float).mean()
                    )
            if phage_metrics:
                for key, value in phage_metrics.items():
                    metrics[f"phage_qc/{key}"] = value
            return batch, metrics

else:

    class PhageQCEnvironment:  # pragma: no cover
        """Placeholder that explains how to enable the NeMo-RL integration."""

        def __init__(self, *_args, **_kwargs):
            """Raise a clear error when NeMo-RL is not installed."""
            raise ModuleNotFoundError(
                "PhageQCEnvironment requires NeMo-RL. Install the recipe environment with nemo-rl available."
            ) from _NEMO_RL_IMPORT_ERROR
