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
from bionemo.evo2_phage_gen.reward import RewardWeights, score_nucleotide_metrics


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


def score_message_logs(
    message_log_batch: list[list[dict[str, Any]]],
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
) -> pd.DataFrame:
    """Score a NeMo-RL message-log batch with the dependency-light phage reward."""
    sequences_df = pd.DataFrame(
        {
            "id_prompt": [str(i) for i in range(len(message_log_batch))],
            "sequence": [extract_assistant_sequence(message_log) for message_log in message_log_batch],
        }
    )
    return score_nucleotide_metrics(sequences_df, config=config, weights=weights)


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
            )

        def step(
            self,
            message_log_batch: list[list[dict[str, Any]]],
            metadata: list[dict[str, Any]],
        ) -> EnvironmentReturn[dict[str, Any]]:
            """Score generated assistant sequences and terminate each rollout."""
            scored = score_message_logs(message_log_batch, config=self.config, weights=self.weights)
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
