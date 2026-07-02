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
    _scored_records,
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
    scored = score_message_logs(
        [[{"role": "user", "content": "+~GAGT"}, {"role": "assistant", "content": "ACGT" * 1000}]]
    )

    assert scored["reward"].tolist() == [1.0]
    assert scored["prompt_nt_length"].tolist() == [4]


def test_phage_qc_metrics_from_scored_flattens_reward_components():
    """Scalar QC metrics should be suitable for TensorBoard and W&B logging."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0],
            "reward_external_tropism": [1.0, 0.5],
            "reward_external_synteny": [0.25, 0.75],
            "reward_external_average_protein_identity": [1.0, 0.5],
            "reward_external_required_genes": [1.0, 0.0],
            "prompt_nt_length": [10, 10],
            "genome_length": [5000, 3900],
            "tropism_stage_reached": [1.0, 1.0],
            "tropism_measurement_available": [1.0, 1.0],
            "tropism_missing_artifact": [0.0, 0.0],
            "reward_external_tropism_pass": [1.0, 0.0],
            "tropism_protein_mmseqs_percent_identity": [75.0, 30.0],
            "tropism_protein_measured_hit": [1.0, 1.0],
            "synteny_stage_reached": [1.0, 1.0],
            "synteny_measurement_available": [1.0, 0.0],
            "synteny_missing_artifact": [0.0, 1.0],
            "reward_external_synteny_pass": [0.0, 0.0],
            "synteny_pair_score": [0.25, 0.75],
            "synteny_pair_distance": [3.0, 1.0],
            "average_protein_percent_identity": [80.0, 97.5],
            "average_protein_identity_gene_count": [10, 9],
            "average_protein_identity_evidence_score": [1.0, 0.9],
            "required_genes_matched_count": [9, 4],
            "required_genes_total_count": [9, 9],
            "required_genes_evidence_score": [1.0, 1.0],
            "reward": [0.8, 0.4],
        }
    )

    metrics = phage_qc_metrics_from_scored(
        scored,
        RewardWeights(
            valid_nt_chars=1.0,
            tropism=1.0,
            synteny=1.0,
            average_protein_identity=1.0,
            required_genes=1.0,
        ),
    )

    assert metrics["num_sequences"] == 2
    assert metrics["valid_nt_chars_score_mean"] == 1.0
    assert metrics["tropism_score_mean"] == 0.75
    assert metrics["tropism_pass_rate"] == 0.5
    assert metrics["tropism_stage_reached_rate"] == 1.0
    assert metrics["tropism_measurement_available_rate"] == 1.0
    assert metrics["tropism_n_measured"] == 2
    assert metrics["tropism_conditional_score_mean"] == 0.75
    assert metrics["tropism_conditional_pass_rate"] == 0.5
    assert metrics["synteny_score_mean"] == 0.5
    assert metrics["synteny_weighted_contribution_mean"] == 0.1
    assert metrics["average_protein_identity_score_mean"] == 0.75
    assert metrics["required_genes_pass_rate"] == 0.5
    assert metrics["prompt_nt_length_mean"] == 10.0
    assert metrics["prompt_nt_length_min"] == 10.0
    assert metrics["prompt_nt_length_max"] == 10.0
    assert metrics["genome_length_mean"] == 4450.0
    assert metrics["tropism_protein_mmseqs_percent_identity_mean"] == 52.5
    assert metrics["tropism_protein_measured_hit_mean"] == 1.0
    assert metrics["synteny_pair_score_mean"] == 0.5
    assert metrics["synteny_pair_distance_mean"] == 2.0
    assert metrics["synteny_stage_reached_rate"] == 1.0
    assert metrics["synteny_measurement_available_rate"] == 0.5
    assert metrics["synteny_n_measured"] == 1
    assert metrics["synteny_missing_artifact_count"] == 1
    assert metrics["synteny_conditional_score_mean"] == 0.25
    assert metrics["synteny_conditional_pass_rate"] == 0.0
    assert metrics["average_protein_percent_identity_mean"] == 88.75
    assert metrics["average_protein_identity_gene_count_mean"] == 9.5
    assert metrics["average_protein_identity_evidence_score_mean"] == 0.95
    assert metrics["required_genes_matched_count_mean"] == 6.5
    assert metrics["required_genes_evidence_score_mean"] == 1.0


def test_phage_qc_metrics_groups_training_metrics_by_prompt_prefix_length():
    """Prompt-length groups let W&B compare each prefix only with matching prefixes."""
    scored = pd.DataFrame(
        {
            "prompt_nt_length": [4, 4, 10],
            "reward_valid_nt_chars": [1.0, 0.0, 1.0],
            "reward": [1.0, 0.0, 0.5],
        }
    )

    metrics = phage_qc_metrics_from_scored(scored, RewardWeights(valid_nt_chars=1.0))

    assert metrics["by_prompt_nt_length/4/num_sequences"] == 2
    assert metrics["by_prompt_nt_length/4/reward_mean"] == 0.5
    assert metrics["by_prompt_nt_length/4/valid_nt_chars_score_mean"] == 0.5
    assert metrics["by_prompt_nt_length/4/core_qc_pass_rate"] == 0.5
    assert metrics["by_prompt_nt_length/10/num_sequences"] == 1
    assert metrics["by_prompt_nt_length/10/reward_mean"] == 0.5
    assert metrics["by_prompt_nt_length/10/core_qc_pass_rate"] == 1.0


def test_scored_records_exclude_full_sequence_from_rollout_metadata():
    """Rollout metadata should carry scalar scores/status, not full generated sequences."""
    scored = pd.DataFrame(
        {
            "sequence": ["A" * 6000],
            "id_prompt": ["seq1"],
            "reward": [0.5],
            "synteny_measurement_available": [1.0],
            "missing_status": ["unavailable"],
        }
    )

    records = _scored_records(scored)

    assert records == [{"reward": 0.5, "synteny_measurement_available": 1.0}]


def test_global_post_process_metrics_accepts_rollout_total_reward():
    """Rollout batches expose total_reward rather than per-turn rewards."""
    if getattr(nemo_rl_env, "_NEMO_RL_IMPORT_ERROR", None) is not None:
        pytest.skip("NeMo-RL is unavailable")

    env_cls = nemo_rl_env.PhageQCEnvironment.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env.weights = RewardWeights(valid_nt_chars=1.0)
    batch = {
        "total_reward": torch.tensor([1.0, 0.0]),
        "extra_env_info": [
            {"_phage_qc_scored": {"reward_valid_nt_chars": 1.0, "reward": 1.0, "reward_binary_pass": 1.0}},
            {"_phage_qc_scored": {"reward_valid_nt_chars": 0.0, "reward": 0.0, "reward_binary_pass": 0.0}},
        ],
    }
    returned_batch, metrics = env_cls.global_post_process_and_metrics(env, batch)

    assert returned_batch is batch
    assert metrics["mean_reward"] == 0.5
    assert metrics["phage_qc/valid_nt_chars_score_mean"] == 0.5


def test_global_post_process_metrics_falls_back_to_last_scored_for_old_rollout_metadata():
    """The actor-local fallback keeps compatibility with unpatched NeMo-RL rollout batches."""
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

    _returned_batch, metrics = env_cls.global_post_process_and_metrics(
        env,
        {"total_reward": torch.tensor([1.0, 0.0])},
    )

    assert metrics["phage_qc/valid_nt_chars_score_mean"] == 0.5
