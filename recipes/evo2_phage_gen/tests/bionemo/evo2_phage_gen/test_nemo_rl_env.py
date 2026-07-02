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

"""Tests for ``bionemo.evo2_phage_gen.nemo_rl_env`` helpers."""

import pandas as pd
import pytest
import torch

import bionemo.evo2_phage_gen.nemo_rl_env as nemo_rl_env
from bionemo.evo2_phage_gen.nemo_rl_env import (
    extract_assistant_sequence,
    extract_scored_sequence,
    phage_qc_metrics_from_scored,
    score_message_logs,
)
from bionemo.evo2_phage_gen.reward import RewardWeights


def test_extract_assistant_sequence_concatenates_assistant_messages():
    """Only assistant messages should contribute to generated DNA."""
    message_log = [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "ACGT"},
        {"role": "environment", "content": "ignored"},
        {"role": "assistant", "content": "TGCA"},
    ]

    assert extract_assistant_sequence(message_log) == "ACGTTGCA"


def test_extract_scored_sequence_keeps_prompt_dna_and_drops_soft_tokens():
    """QC should include the nucleotide prompt but not fine-tuning soft tokens."""
    message_log = [
        {"role": "user", "content": "+~GAGT"},
        {"role": "assistant", "content": "ACGT"},
    ]

    assert extract_scored_sequence(message_log) == "GAGTACGT"


def test_score_message_logs_uses_phage_reward():
    """A passing assistant sequence should receive reward 1."""
    scored = score_message_logs([[{"role": "user", "content": "+~GAGT"}, {"role": "assistant", "content": "ACGT" * 1000}]])

    assert scored["reward"].tolist() == [1.0]
    assert scored["prompt_nt_length"].tolist() == [4]


def test_phage_qc_metrics_from_scored_flattens_reward_components():
    """Scalar QC metrics should be suitable for TensorBoard and W&B logging."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0],
            "reward_external_tropism": [1.0, 0.5],
            "reward_external_training_data_identity": [1.0, 0.0],
            "reward_external_synteny": [0.25, 0.75],
            "reward_external_average_protein_identity": [1.0, 0.5],
            "reward_external_required_genes": [1.0, 0.0],
            "reward_external_reference_genome_identity": [1.0, 0.5],
            "prompt_nt_length": [10, 10],
            "genome_length": [5000, 3900],
            "tropism_protein_mmseqs_percent_identity": [75.0, 30.0],
            "training_data_mmseqs_percent_identity": [0.0, 100.0],
            "average_protein_percent_identity": [80.0, 97.5],
            "required_genes_matched_count": [9, 4],
            "required_genes_total_count": [9, 9],
            "reference_genome_mmseqs_percent_identity": [80.0, 99.45],
            "reward": [0.8, 0.4],
        }
    )

    metrics = phage_qc_metrics_from_scored(
        scored,
        RewardWeights(
            valid_nt_chars=1.0,
            tropism=1.0,
            training_data_identity=1.0,
            synteny=1.0,
            average_protein_identity=1.0,
            required_genes=1.0,
            reference_genome_identity=1.0,
        ),
    )

    assert metrics["num_sequences"] == 2
    assert metrics["valid_nt_chars_score_mean"] == 1.0
    assert metrics["tropism_score_mean"] == 0.75
    assert metrics["tropism_pass_rate"] == 0.5
    assert metrics["training_data_identity_score_mean"] == 0.5
    assert metrics["synteny_score_mean"] == 0.5
    assert metrics["average_protein_identity_score_mean"] == 0.75
    assert metrics["required_genes_pass_rate"] == 0.5
    assert metrics["reference_genome_identity_score_mean"] == 0.75
    assert metrics["prompt_nt_length_mean"] == 10.0
    assert metrics["prompt_nt_length_min"] == 10.0
    assert metrics["prompt_nt_length_max"] == 10.0
    assert metrics["genome_length_mean"] == 4450.0
    assert metrics["tropism_protein_mmseqs_percent_identity_mean"] == 52.5
    assert metrics["training_data_mmseqs_percent_identity_mean"] == 50.0
    assert metrics["average_protein_percent_identity_mean"] == 88.75
    assert metrics["required_genes_matched_count_mean"] == 6.5
    assert metrics["reference_genome_mmseqs_percent_identity_mean"] == 89.725


def test_global_post_process_metrics_accepts_rollout_total_reward(monkeypatch):
    """Rollout batches expose total_reward rather than per-turn rewards."""
    if getattr(nemo_rl_env, "_NEMO_RL_IMPORT_ERROR", None) is not None:
        pytest.skip("NeMo-RL is unavailable")

    env_cls = nemo_rl_env.PhageQCEnvironment.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env.weights = RewardWeights(valid_nt_chars=1.0)
    env._last_scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 0.0],
            "reward": [1.0, 0.0],
            "reward_binary_pass": [1.0, 0.0],
        }
    )

    batch = {"total_reward": torch.tensor([1.0, 0.0])}
    returned_batch, metrics = env_cls.global_post_process_and_metrics(env, batch)

    assert returned_batch is batch
    assert metrics["mean_reward"] == 0.5
    assert metrics["phage_qc/valid_nt_chars_score_mean"] == 0.5
