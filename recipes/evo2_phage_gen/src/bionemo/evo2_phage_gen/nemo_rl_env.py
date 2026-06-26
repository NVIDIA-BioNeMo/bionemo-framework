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

from typing import Any

import pandas as pd

from bionemo.evo2_phage_gen.qc import NucleotideQCConfig
from bionemo.evo2_phage_gen.reward import ExternalQCRewardConfig, RewardWeights, score_nucleotide_metrics


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
) -> pd.DataFrame:
    """Score a NeMo-RL message-log batch with the dependency-light phage reward."""
    sequences_df = pd.DataFrame(
        {
            "id_prompt": [str(i) for i in range(len(message_log_batch))],
            "sequence": [extract_scored_sequence(message_log) for message_log in message_log_batch],
        }
    )
    return score_nucleotide_metrics(sequences_df, config=config, weights=weights, external_qc=external_qc)


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
            )
            self.weights = RewardWeights(
                valid_nt_chars=float(cfg.get("weight_valid_nt_chars", 1.0)),
                genome_length=float(cfg.get("weight_genome_length", 1.0)),
                gc_content=float(cfg.get("weight_gc_content", 1.0)),
                nt_homopolymer=float(cfg.get("weight_nt_homopolymer", 1.0)),
                orf=float(cfg.get("weight_orf", 0.0)),
                coding_density=float(cfg.get("weight_coding_density", 0.0)),
                protein_hit_count=float(cfg.get("weight_protein_hit_count", 0.0)),
                tropism=float(cfg.get("weight_tropism", 0.0)),
                genetic_architecture=float(cfg.get("weight_genetic_architecture", 0.0)),
                checkv=float(cfg.get("weight_checkv", 0.0)),
                synteny=float(cfg.get("weight_synteny", 0.0)),
                mmseqs_clustering=float(cfg.get("weight_mmseqs_clustering", 0.0)),
                diversity=float(cfg.get("weight_diversity", 0.0)),
            )
            external_qc_cfg = cfg.get("external_qc", {}) or {}
            self.external_qc = ExternalQCRewardConfig(
                enabled=bool(external_qc_cfg.get("enabled", False)),
                config_path=external_qc_cfg.get(
                    "config_path", "configs/arc_genome_design_filtering_local.yaml"
                ),
                pipeline_script=external_qc_cfg.get(
                    "pipeline_script", "data/arc_pipeline_patched/genome_design_filtering_pipeline.py"
                ),
                work_dir=external_qc_cfg.get("work_dir", "data/checkpoints/phage_grpo_external_qc"),
                checkv_db_path=external_qc_cfg.get("checkv_db_path", "data/external/checkv/checkv-db-v1.5"),
                keep_artifacts=bool(external_qc_cfg.get("keep_artifacts", False)),
                enable_orf=bool(external_qc_cfg.get("enable_orf", True)),
                enable_coding_density=bool(external_qc_cfg.get("enable_coding_density", True)),
                enable_protein_hit_count=bool(external_qc_cfg.get("enable_protein_hit_count", True)),
                enable_tropism=bool(external_qc_cfg.get("enable_tropism", True)),
                enable_genetic_architecture=bool(external_qc_cfg.get("enable_genetic_architecture", True)),
                enable_checkv=bool(external_qc_cfg.get("enable_checkv", False)),
                enable_synteny=bool(external_qc_cfg.get("enable_synteny", False)),
                enable_mmseqs_clustering=bool(external_qc_cfg.get("enable_mmseqs_clustering", False)),
                enable_diversity=bool(external_qc_cfg.get("enable_diversity", False)),
                diversity_quality_threshold=float(external_qc_cfg.get("diversity_quality_threshold", 0.6)),
                diversity_kmer_size=int(external_qc_cfg.get("diversity_kmer_size", 8)),
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
            )
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
                terminateds=torch.ones_like(rewards).cpu(),
                answers=scored["sequence"].tolist(),
            )

        def global_post_process_and_metrics(
            self, batch: BatchedDataDict
        ) -> tuple[BatchedDataDict, dict[str, float | int]]:
            """Report rollout-level reward metrics."""
            rewards = batch["rewards"] if batch["rewards"].ndim == 1 else batch["rewards"][:, 0]
            metrics = {
                "mean_reward": rewards.float().mean().item(),
                "pass_rate": (rewards >= 1.0).float().mean().item(),
                "num_sequences": int(rewards.shape[0]),
            }
            return batch, metrics

else:

    class PhageQCEnvironment:  # pragma: no cover
        """Placeholder that explains how to enable the NeMo-RL integration."""

        def __init__(self, *_args, **_kwargs):
            """Raise a clear error when NeMo-RL is not installed."""
            raise ModuleNotFoundError(
                "PhageQCEnvironment requires NeMo-RL. Install the recipe environment with nemo-rl available."
            ) from _NEMO_RL_IMPORT_ERROR
