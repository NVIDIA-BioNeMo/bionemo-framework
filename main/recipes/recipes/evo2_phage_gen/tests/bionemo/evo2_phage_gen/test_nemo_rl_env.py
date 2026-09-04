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

import copy
import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

import bionemo.evo2_phage_gen.nemo_rl_env as nemo_rl_env
from bionemo.evo2_phage_gen.design_scope import HostDomain, HostEvidence
from bionemo.evo2_phage_gen.nemo_rl_env import (
    TIMING_METRIC_MARKER_PREFIX,
    GDPOObjective,
    _scored_records,
    extract_assistant_sequence,
    extract_scored_sequence,
    gdpo_objective_scores_from_scored,
    phage_qc_metrics_from_scored,
    score_message_logs,
)
from bionemo.evo2_phage_gen.reward import TIMING_COLUMN_PREFIX, RewardWeights, SequenceSafetyRewardConfig


def _sequence_safety_mapping(tmp_path: Path) -> dict[str, object]:
    """Return one strictly typed prokaryotic sequence-safety config mapping."""
    return {
        "enabled": True,
        "host_domain": "BACTERIA",
        "host_evidence": {
            "source": "test-curation",
            "source_version": "v1",
            "replication_host_domains": ["BACTERIA"],
            "confirmed": True,
            "metadata": {"record_id": "NC_001422.1"},
        },
        "asset_manifest_path": str(tmp_path / "asset_manifest.yaml"),
        "diamond_bin": str(tmp_path / "diamond"),
        "mmseqs_bin": str(tmp_path / "mmseqs"),
        "policy_path": str(tmp_path / "policy.yaml"),
        "work_dir": str(tmp_path / "work"),
        "strict_lysis": False,
        "circular": True,
        "threads": 2,
        "batch_size": 5,
        "orf_workers": 3,
        "phrogs_threads": 7,
        "timeout_seconds": 30.0,
    }


def _disabled_sequence_safety_config(tmp_path: Path) -> SequenceSafetyRewardConfig:
    """Build a valid disabled config for dependency-free INDETERMINATE scoring."""
    return SequenceSafetyRewardConfig(
        host_domain=HostDomain.BACTERIA,
        host_evidence=HostEvidence(
            source="test-curation",
            source_version="v1",
            replication_host_domains=frozenset({HostDomain.BACTERIA}),
            confirmed=True,
            metadata={"record_id": "NC_001422.1"},
        ),
        asset_manifest_path=tmp_path / "asset_manifest.yaml",
        diamond_bin=tmp_path / "diamond",
        mmseqs_bin=tmp_path / "mmseqs",
        policy_path=tmp_path / "policy.yaml",
        work_dir=tmp_path / "work",
        enabled=False,
    )


def _new_step_environment(
    *,
    reward_output_mode: str,
    gdpo_objectives: tuple[GDPOObjective, ...],
) -> tuple[type, object]:
    """Construct the minimally initialized Ray actor used by step tests."""
    if getattr(nemo_rl_env, "_NEMO_RL_IMPORT_ERROR", None) is not None:
        pytest.skip("NeMo-RL is unavailable")

    env_cls = nemo_rl_env.PhageQCEnvironment.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env.config = object()
    env.weights = RewardWeights(valid_nt_chars=1.0)
    env.external_qc = object()
    env.mmseqs_cluster_diversity = object()
    env.sequence_safety = object()
    env.reward_output_mode = reward_output_mode
    env.gdpo_objectives = gdpo_objectives
    return env_cls, env


def test_extract_assistant_sequence_concatenates_assistant_messages():
    """Only assistant messages should contribute to generated DNA."""
    message_log = [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "ACGT"},
        {"role": "environment", "content": "ignored"},
        {"role": "assistant", "content": "TGCA"},
    ]

    assert extract_assistant_sequence(message_log) == "ACGTTGCA"


def test_extract_scored_sequence_keeps_prompt_dna_and_trims_terminal_eos():
    """QC should drop prompt soft tokens and the generated terminal action."""
    message_log = [
        {"role": "user", "content": "+~GAGT"},
        {"role": "assistant", "content": "ACGT<EOS>NOT_DNA"},
    ]

    assert extract_scored_sequence(message_log) == "GAGTACGT"


def test_score_message_logs_sends_only_pre_eos_dna_to_qc(monkeypatch):
    """Terminal EOS remains an RL action but must not enter biological scoring."""
    captured = {}

    def _capture_sequences(sequences_df, **_kwargs):
        captured["sequences"] = sequences_df.copy()
        return sequences_df

    monkeypatch.setattr(nemo_rl_env, "score_nucleotide_metrics", _capture_sequences)

    scored = score_message_logs(
        [[{"role": "user", "content": "+~GAGT"}, {"role": "assistant", "content": "ACGT<EOD>junk"}]]
    )

    assert captured["sequences"]["sequence"].tolist() == ["GAGTACGT"]
    assert scored["sequence"].tolist() == ["GAGTACGT"]


def test_score_message_logs_without_safety_config_returns_zero_reward():
    """A historically passing sequence remains ineligible when safety is not configured."""
    scored = score_message_logs(
        [[{"role": "user", "content": "+~GAGT"}, {"role": "assistant", "content": "ACGT" * 1000}]]
    )

    assert scored["reward_historical"].tolist() == [1.0]
    assert scored["reward"].tolist() == [0.0]
    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_gate_reason_codes"].tolist() == ['["SEQUENCE_SAFETY_CONFIG_MISSING"]']
    assert scored["prompt_nt_length"].tolist() == [4]


def test_score_message_logs_forwards_safety_config_and_retains_historical_reward(tmp_path: Path):
    """GRPO scoring must retain historical telemetry while an unavailable safety gate zeros reward."""
    safety_config = _disabled_sequence_safety_config(tmp_path)

    scored = score_message_logs(
        [[{"role": "user", "content": "+~GAGT"}, {"role": "assistant", "content": "ACGT" * 1000}]],
        sequence_safety=safety_config,
    )

    assert scored["reward_historical"].tolist() == [1.0]
    assert scored["reward"].tolist() == [0.0]
    assert scored["safety_gate_state"].tolist() == ["INDETERMINATE"]
    assert scored["safety_gate_reason_codes"].tolist() == ['["SEQUENCE_SAFETY_DISABLED"]']


def test_sequence_safety_mapping_is_parsed_without_bool_or_host_scope_coercion(tmp_path: Path):
    """Environment config parsing must produce typed Task 6 safety settings."""
    raw = _sequence_safety_mapping(tmp_path)

    parsed = nemo_rl_env._coerce_sequence_safety_config(raw)

    assert type(parsed) is SequenceSafetyRewardConfig
    assert parsed.host_domain is HostDomain.BACTERIA
    assert parsed.host_evidence.to_dict() == raw["host_evidence"]
    assert parsed.enabled is True
    assert parsed.strict_lysis is False
    assert parsed.circular is True
    assert parsed.batch_size == 5
    assert parsed.orf_workers == 3
    assert parsed.phrogs_threads == 7
    assert parsed.asset_manifest_path == tmp_path / "asset_manifest.yaml"

    additive = copy.deepcopy(raw)
    additive["notes"] = "recorded by a newer config writer"
    additive["host_evidence"]["curator"] = "lab notebook"
    assert nemo_rl_env._coerce_sequence_safety_config(additive) == parsed

    string_bool = copy.deepcopy(raw)
    string_bool["enabled"] = "false"
    with pytest.raises(TypeError, match="enabled.*boolean"):
        nemo_rl_env._coerce_sequence_safety_config(string_bool)

    unconfirmed = copy.deepcopy(raw)
    unconfirmed["host_evidence"]["confirmed"] = "true"
    with pytest.raises(TypeError, match="confirmed.*boolean"):
        nemo_rl_env._coerce_sequence_safety_config(unconfirmed)

    eukaryotic_evidence = copy.deepcopy(raw)
    eukaryotic_evidence["host_evidence"]["replication_host_domains"] = ["EUKARYOTA"]
    with pytest.raises(ValueError, match="prokaryotic"):
        nemo_rl_env._coerce_sequence_safety_config(eukaryotic_evidence)


def test_environment_requires_enabled_sequence_safety(tmp_path: Path):
    """A training actor must fail fast instead of running with a disabled or absent mandatory gate."""
    if getattr(nemo_rl_env, "_NEMO_RL_IMPORT_ERROR", None) is not None:
        pytest.skip("NeMo-RL is unavailable")

    env_cls = nemo_rl_env.PhageQCEnvironment.__ray_metadata__.modified_class
    with pytest.raises(ValueError, match="sequence_safety.*required"):
        env_cls({})

    disabled = _sequence_safety_mapping(tmp_path)
    disabled["enabled"] = False
    with pytest.raises(ValueError, match="sequence_safety.*enabled"):
        env_cls({"sequence_safety": disabled})


def test_gdpo_objective_scores_reduce_named_columns_positionally():
    """GDPO helper should build a stable [B, K] objective table without adding aggregate reward."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 0.0],
            "reward_genome_length": [1.0, 0.5],
            "reward_external_tropism": [0.25, 0.75],
            "safety_gate_state": ["PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0],
        }
    )

    objectives = (
        GDPOObjective("feasibility", ("reward_valid_nt_chars", "reward_genome_length"), "mean"),
        GDPOObjective("function", ("reward_external_tropism",), "mean"),
    )
    objective_scores = gdpo_objective_scores_from_scored(scored, objectives)

    assert list(objective_scores.columns) == ["feasibility", "function"]
    assert objective_scores.to_numpy().tolist() == [[1.0, 0.25], [0.25, 0.75]]


def test_default_gdpo_objectives_expose_three_independent_safety_signals():
    """Omitting custom objectives must not silently drop the three mandatory safety signals."""
    objectives = nemo_rl_env._coerce_gdpo_objectives(None)

    safety_objectives = objectives[-3:]
    assert [objective.name for objective in safety_objectives] == ["safety_amr", "safety_toxin", "safety_lysogeny"]
    assert [objective.columns for objective in safety_objectives] == [
        ("reward_safety_amr",),
        ("reward_safety_toxin",),
        ("reward_safety_lysogeny",),
    ]
    assert all(objective.requires_safety_eligibility is False for objective in safety_objectives)
    assert all(objective.requires_safety_eligibility is True for objective in objectives[:-3])


@pytest.mark.parametrize("invalid", ["false", 0, 1, None])
def test_gdpo_objective_parser_rejects_non_boolean_safety_eligibility(invalid: object):
    """Truthy strings and integer lookalikes must not bypass objective eligibility masking."""
    raw = [
        {
            "name": "biological",
            "columns": ["reward_valid_nt_chars"],
            "requires_safety_eligibility": invalid,
        }
    ]

    with pytest.raises(TypeError, match="requires_safety_eligibility.*boolean"):
        nemo_rl_env._coerce_gdpo_objectives(raw)


def test_gdpo_masks_only_optimization_values_with_exact_safety_eligibility():
    """Biology requires reconciled PASS telemetry while valid class objectives stay exposed."""
    scored = pd.DataFrame(
        {
            "reward_biological": [0.75] * 9,
            "reward_safety_amr": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "safety_amr_state": [
                "PASS",
                "PASS",
                "FAIL",
                "INDETERMINATE",
                "PASS",
                "PASS",
                "PASS",
                "FAIL",
                "INDETERMINATE",
            ],
            "safety_amr_required": [1.0] * 9,
            "safety_gate_pass": pd.Series([1.0, 1, 1.0, 1.0, True, "1", "1.0", 0.0, None], dtype=object),
            "average_protein_percent_identity": [80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0],
            "safety_gate_state": [
                "PASS",
                "PASS",
                "FAIL",
                "INDETERMINATE",
                "PASS",
                "PASS",
                "PASS",
                "PASS",
                "PASS",
            ],
        }
    )
    original = scored.copy(deep=True)
    objectives = (
        GDPOObjective("biological", ("reward_biological",)),
        GDPOObjective(
            "safety_amr",
            ("reward_safety_amr",),
            requires_safety_eligibility=False,
        ),
    )

    objective_scores = gdpo_objective_scores_from_scored(scored, objectives)

    assert objective_scores["biological"].tolist() == [0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert objective_scores["safety_amr"].tolist() == scored["reward_safety_amr"].tolist()
    pd.testing.assert_frame_equal(scored, original)


def test_gdpo_ineligible_nan_biological_objective_is_exactly_zero():
    """Masking must not let NaN survive multiplication into an ineligible optimization reward."""
    scored = pd.DataFrame(
        {"reward_biological": [float("nan")], "safety_gate_state": ["FAIL"], "safety_gate_pass": [0.0]}
    )

    objective_scores = gdpo_objective_scores_from_scored(
        scored,
        (GDPOObjective("biological", ("reward_biological",)),),
    )

    assert objective_scores["biological"].tolist() == [0.0]
    assert pd.isna(scored.loc[0, "reward_biological"])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), True, "1.0", 1 + 0j])
def test_gdpo_objectives_reject_nonfinite_or_nonexact_values(invalid: object):
    """No biological or safety objective may emit coerced, NaN, or infinite values."""
    scored = pd.DataFrame(
        {
            "reward_biological": [invalid],
            "reward_safety_amr": [invalid],
            "safety_gate_state": ["PASS"],
            "safety_gate_pass": [1.0],
            "safety_amr_state": ["PASS"],
            "safety_amr_required": [1.0],
        }
    )
    original = scored.copy(deep=True)

    objective_scores = gdpo_objective_scores_from_scored(
        scored,
        (
            GDPOObjective("biological", ("reward_biological",)),
            GDPOObjective("safety_amr", ("reward_safety_amr",), requires_safety_eligibility=False),
        ),
    )

    assert objective_scores.to_numpy().tolist() == [[0.0, 0.0]]
    pd.testing.assert_frame_equal(scored, original)


def test_gdpo_safety_objective_requires_reconciled_class_state_reward_and_required_flag():
    """Safety credit requires matching class state, applicability, and numeric reward fields."""
    scored = pd.DataFrame(
        {
            "reward_safety_lysogeny": [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "safety_lysogeny_state": ["PASS", "FAIL", "FAIL", "INDETERMINATE", "FAIL", None],
            "safety_lysogeny_required": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        }
    )
    objective = GDPOObjective(
        "safety_lysogeny",
        ("reward_safety_lysogeny",),
        requires_safety_eligibility=False,
    )

    scores = gdpo_objective_scores_from_scored(scored, (objective,))
    missing_support = gdpo_objective_scores_from_scored(
        pd.DataFrame({"reward_safety_lysogeny": [1.0]}),
        (objective,),
    )

    assert scores["safety_lysogeny"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert missing_support["safety_lysogeny"].tolist() == [0.0]


def test_gdpo_safety_objective_accepts_partial_credit_only_for_a_measured_review_finding():
    objective = GDPOObjective(
        "safety_toxin",
        ("reward_safety_toxin",),
        requires_safety_eligibility=False,
    )
    scored = pd.DataFrame(
        {
            "reward_safety_toxin": [0.25, 0.25, 0.25, 0.0],
            "safety_toxin_state": ["INDETERMINATE"] * 4,
            "safety_toxin_required": [1.0] * 4,
            "safety_toxin_measurement_available": [1.0, 0.0, 1.0, 0.0],
            "safety_toxin_finding_count": [1, 1, 0, 0],
        }
    )

    scores = gdpo_objective_scores_from_scored(scored, (objective,))

    assert scores["safety_toxin"].tolist() == [0.25, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("reducer", ["mean", "product", "min"])
def test_gdpo_reducer_zeros_entire_objective_when_any_component_is_invalid(reducer: str):
    """A reducer cannot hide one invalid component by skipping or averaging it."""
    scored = pd.DataFrame(
        {
            "reward_component_a": [1.0],
            "reward_component_b": [float("nan")],
            "safety_gate_state": ["PASS"],
            "safety_gate_pass": [1.0],
        }
    )

    scores = gdpo_objective_scores_from_scored(
        scored,
        (GDPOObjective("biological", ("reward_component_a", "reward_component_b"), reducer),),
    )

    assert scores["biological"].tolist() == [0.0]


def test_gdpo_objective_scores_reject_missing_columns():
    """Config mistakes should fail before NeMo-RL receives a malformed reward tensor."""
    scored = pd.DataFrame({"reward_valid_nt_chars": [1.0]})

    with pytest.raises(ValueError, match="missing reward column"):
        gdpo_objective_scores_from_scored(
            scored,
            (GDPOObjective("novelty", ("reward_mmseqs_cluster_diversity",), "mean"),),
        )


def test_empty_gdpo_objective_scores_keep_configured_shape():
    """An empty assembled rollout still has a stable zero-row positional objective schema."""
    objectives = (
        GDPOObjective("biological", ("reward_biological",)),
        GDPOObjective("safety_amr", ("reward_safety_amr",), requires_safety_eligibility=False),
    )

    scores = gdpo_objective_scores_from_scored(pd.DataFrame(), objectives)

    assert scores.shape == (0, 2)
    assert list(scores.columns) == ["biological", "safety_amr"]

    indexed_empty_scores = gdpo_objective_scores_from_scored(pd.DataFrame(index=range(2)), objectives)
    assert indexed_empty_scores.shape == (2, 2)
    assert indexed_empty_scores.to_numpy().tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_phage_qc_metrics_from_scored_flattens_reward_components():
    """Scalar QC metrics should be suitable for TensorBoard and W&B logging."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0],
            "reward_external_tropism": [1.0, 0.5],
            "reward_external_tropism_pass": [1.0, 0.0],
            "reward_dustmask_end": [1.0, 0.5],
            "reward_external_synteny": [0.25, 0.75],
            "reward_external_synteny_pass": [1.0, 0.0],
            "reward_external_average_protein_identity": [1.0, 0.5],
            "reward_external_average_protein_identity_pass": [1.0, 0.0],
            "reward_external_required_genes": [1.0, 0.0],
            "reward_external_required_genes_pass": [1.0, 0.0],
            "prompt_nt_length": [10, 10],
            "genome_length": [5000, 3900],
            "tropism_stage_reached": [1.0, 1.0],
            "tropism_measurement_available": [1.0, 1.0],
            "tropism_missing_artifact": [0.0, 0.0],
            "tropism_protein_mmseqs_percent_identity": [75.0, 30.0],
            "tropism_protein_measured_hit": [1.0, 1.0],
            "synteny_stage_reached": [1.0, 1.0],
            "synteny_measurement_available": [1.0, 0.0],
            "synteny_missing_artifact": [0.0, 1.0],
            "synteny_pair_score": [0.25, 0.75],
            "synteny_pair_distance": [3.0, 1.0],
            "average_protein_percent_identity": [80.0, 97.5],
            "average_protein_identity_gene_count": [10, 9],
            "average_protein_identity_evidence_score": [1.0, 0.9],
            "required_genes_matched_count": [9, 4],
            "required_genes_total_count": [9, 9],
            "required_genes_evidence_score": [1.0, 1.0],
            "reward_mmseqs_cluster_diversity": [1.0, 0.5],
            "mmseqs_cluster_id": ["group0:seq_0", "group0:seq_1"],
            "mmseqs_cluster_size": [1, 2],
            "mmseqs_cluster_is_singleton": [1.0, 0.0],
            "mmseqs_cluster_valid_for_clustering": [1.0, 1.0],
            "safety_gate_state": ["PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0],
            "reward": [0.8, 0.4],
        }
    )

    metrics = phage_qc_metrics_from_scored(
        scored,
        RewardWeights(
            valid_nt_chars=1.0,
            tropism=1.0,
            dustmask_end=1.0,
            synteny=1.0,
            average_protein_identity=1.0,
            required_genes=1.0,
            mmseqs_cluster_diversity=1.0,
        ),
    )

    assert metrics["num_sequences"] == 2
    assert metrics["valid_nt_chars_score_mean"] == 1.0
    assert metrics["tropism_score_mean"] == 0.75
    assert metrics["dustmask_end_score_mean"] == 0.75
    assert metrics["tropism_pass_rate"] == 0.5
    assert metrics["tropism_stage_reached_rate"] == 1.0
    assert metrics["tropism_measurement_available_rate"] == 1.0
    assert metrics["tropism_n_measured"] == 2
    assert metrics["tropism_conditional_score_mean"] == 0.75
    assert metrics["tropism_conditional_pass_rate"] == 0.5
    assert metrics["synteny_score_mean"] == 0.5
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
    assert metrics["synteny_conditional_pass_rate"] == 1.0
    assert metrics["average_protein_percent_identity_mean"] == 88.75
    assert metrics["average_protein_identity_gene_count_mean"] == 9.5
    assert metrics["average_protein_identity_evidence_score_mean"] == 0.95
    assert metrics["required_genes_matched_count_mean"] == 6.5
    assert metrics["required_genes_evidence_score_mean"] == 1.0
    assert metrics["mmseqs_cluster_diversity_score_mean"] == 0.75
    assert metrics["mmseqs_cluster_num_clusters"] == 2
    assert metrics["mmseqs_cluster_clusters_per_sequence"] == 1.0
    assert metrics["mmseqs_cluster_singleton_fraction"] == 0.5
    assert metrics["mmseqs_cluster_largest_cluster_fraction"] == 1.0
    assert metrics["mmseqs_cluster_size_histogram/size_1"] == 1
    assert metrics["mmseqs_cluster_size_histogram/size_2"] == 1
    assert metrics["binary_core_pass_count"] == 1
    assert metrics["binary_core_pass_rate"] == 0.5
    assert metrics["binary_full_qc_pass_count"] == 1
    assert metrics["binary_full_qc_pass_rate"] == 0.5
    assert metrics["binary_full_qc_pass_cluster_deduplicated_count"] == 1
    assert metrics["binary_full_qc_pass_cluster_deduplicated_rate"] == 0.5


def test_empty_phage_qc_metrics_publish_required_checkpoint_zero():
    """Checkpoint selection must receive an explicit safe zero for an empty batch."""
    metrics = phage_qc_metrics_from_scored(pd.DataFrame(), RewardWeights(valid_nt_chars=1.0))

    assert metrics == {
        "num_sequences": 0,
        "binary_safety_qualified_full_qc_cluster_deduplicated_rate": 0.0,
    }


def test_phage_qc_metrics_interprets_mmseqs_cluster_sizes_with_full_batch_denominator():
    """Cluster scalar metrics should reflect cluster rows while keeping batch-size denominators explicit."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0, 1.0, 0.0],
            "reward_mmseqs_cluster_diversity": [0.5, 0.5, 1.0, 0.0],
            "mmseqs_cluster_id": ["group0:seq_0", "group0:seq_0", "group0:seq_2", ""],
            "mmseqs_cluster_size": [2, 2, 1, 0],
            "mmseqs_cluster_is_singleton": [0.0, 0.0, 1.0, 0.0],
            "mmseqs_cluster_valid_for_clustering": [1.0, 1.0, 1.0, 0.0],
            "mmseqs_cluster_missing_from_output": [0.0, 0.0, 0.0, 0.0],
            "safety_gate_state": ["PASS", "PASS", "PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0, 1.0, 1.0],
            "reward": [0.5, 0.5, 1.0, 0.0],
        }
    )

    metrics = phage_qc_metrics_from_scored(
        scored,
        RewardWeights(
            valid_nt_chars=1.0,
            mmseqs_cluster_diversity=1.0,
        ),
    )

    assert metrics["num_sequences"] == 4
    assert metrics["mmseqs_cluster_diversity_score_mean"] == 0.5
    assert metrics["mmseqs_cluster_size_mean"] == 1.25
    assert metrics["mmseqs_cluster_valid_for_clustering_mean"] == 0.75
    assert metrics["mmseqs_cluster_num_clusters"] == 2
    assert metrics["mmseqs_cluster_clusters_per_sequence"] == 0.5
    assert metrics["mmseqs_cluster_singleton_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["mmseqs_cluster_largest_cluster_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["mmseqs_cluster_size_histogram/size_1"] == 1
    assert metrics["mmseqs_cluster_size_histogram/size_2"] == 1


def test_phage_qc_metrics_report_safety_states_rewards_and_qualified_full_qc():
    """Safety observability must distinguish eligibility from historical model quality."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0, 1.0, 1.0],
            "reward_external_synteny_pass": [1.0, 1.0, 1.0, 1.0],
            "reward_external_average_protein_identity_pass": [1.0, 1.0, 1.0, 1.0],
            "reward_external_required_genes_pass": [1.0, 1.0, 1.0, 1.0],
            "safety_gate_state": ["PASS", "FAIL", "INDETERMINATE", "PASS"],
            "safety_gate_pass": [1.0, 1.0, 0.0, 0.0],
            "safety_amr_state": ["PASS", "FAIL", "INDETERMINATE", "PASS"],
            "safety_toxin_state": ["PASS", "PASS", "INDETERMINATE", "FAIL"],
            "safety_lysogeny_state": ["PASS", "INDETERMINATE", "PASS", "PASS"],
            "reward_historical": [0.8, 0.9, 0.7, 0.6],
            "reward": [0.8, 0.0, 0.0, 0.6],
            "mmseqs_cluster_id": ["group0:pass", "group0:fail", "group0:indet", "group0:pass"],
            "mmseqs_cluster_size": [2, 1, 1, 2],
            "mmseqs_cluster_valid_for_clustering": [1.0, 1.0, 1.0, 1.0],
        }
    )

    metrics = phage_qc_metrics_from_scored(scored, RewardWeights(valid_nt_chars=1.0))

    assert metrics["safety_gate_state_count/PASS"] == 2
    assert metrics["safety_gate_state_count/FAIL"] == 1
    assert metrics["safety_gate_state_count/INDETERMINATE"] == 1
    assert metrics["safety_gate_pass_rate"] == 0.25
    assert metrics["safety_gate_indeterminate_rate"] == 0.25
    assert metrics["safety_amr_pass_rate"] == 0.5
    assert metrics["safety_amr_indeterminate_rate"] == 0.25
    assert metrics["safety_toxin_pass_rate"] == 0.5
    assert metrics["safety_toxin_indeterminate_rate"] == 0.25
    assert metrics["safety_lysogeny_pass_rate"] == 0.75
    assert metrics["safety_lysogeny_indeterminate_rate"] == 0.25
    assert metrics["reward_historical_mean"] == pytest.approx(0.75)
    assert metrics["reward_safety_qualified_mean"] == pytest.approx(0.2)
    assert metrics["binary_safety_qualified_full_qc_cluster_deduplicated_rate"] == 0.25


def test_phage_qc_metrics_report_zero_acceptance_when_aggregate_state_is_missing():
    """A numeric gate bit without its exact PASS state cannot qualify a checkpoint metric."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0],
            "reward_external_synteny_pass": [1.0],
            "reward_external_average_protein_identity_pass": [1.0],
            "reward_external_required_genes_pass": [1.0],
            "safety_gate_pass": [1.0],
            "reward": [1.0],
            "mmseqs_cluster_id": ["group0:seq_0"],
            "mmseqs_cluster_size": [1],
        }
    )

    metrics = phage_qc_metrics_from_scored(scored, RewardWeights(valid_nt_chars=1.0))

    assert metrics["safety_gate_pass_rate"] == 0.0
    assert metrics["reward_safety_qualified_mean"] == 0.0
    assert metrics["binary_safety_qualified_full_qc_cluster_deduplicated_rate"] == 0.0


def test_phage_qc_qualified_reward_mean_rejects_out_of_range_values():
    """Qualified reward reporting uses the same bounded scalar range as optimization."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0, 1.0],
            "safety_gate_state": ["PASS", "PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0, 1.0],
            "reward": [-0.5, 1.5, 0.6],
        }
    )

    metrics = phage_qc_metrics_from_scored(scored, RewardWeights(valid_nt_chars=1.0))

    assert metrics["reward_safety_qualified_mean"] == pytest.approx(0.2)


@pytest.mark.parametrize("state", ["FAIL", "INDETERMINATE"])
def test_grpo_step_uses_final_reward_and_preserves_safety_evidence(monkeypatch, state: str):
    """An ineligible rollout returns zero while retaining historical and safety support telemetry."""
    env_cls, env = _new_step_environment(reward_output_mode="scalar", gdpo_objectives=())

    def fake_score_message_logs(
        message_log_batch,
        *,
        config,
        weights,
        external_qc,
        mmseqs_cluster_diversity,
        sequence_safety,
    ):
        assert sequence_safety is env.sequence_safety
        assert len(message_log_batch) == 1
        return pd.DataFrame(
            {
                "sequence": ["ACGT"],
                "reward": [0.0],
                "reward_historical": [0.75],
                "safety_gate_state": [state],
                "safety_gate_pass": [0.0],
                "safety_gate_reason_codes": ['["AMR_FINDING"]'],
                "safety_environment_healthy": [1.0],
                "safety_gate_measurement_available": [1.0],
                "safety_amr_state": [state],
                "safety_amr_reason_codes": ['["AMR_FINDING"]'],
                "safety_amr_execution_status": ["COMPLETED_AND_PARSED"],
                "safety_amr_finding_count": [1],
                "safety_amr_policy_id": ["ema-phage-v1"],
            }
        )

    monkeypatch.setattr(nemo_rl_env, "score_message_logs", fake_score_message_logs)

    result = env_cls.step(env, [[{"role": "assistant", "content": "ACGT"}]], [{}])

    assert result.rewards.tolist() == [0.0]
    scored_metadata = result.metadata[0]["_phage_qc_scored"]
    assert scored_metadata["reward_historical"] == 0.75
    assert scored_metadata["safety_gate_state"] == state
    assert scored_metadata["safety_gate_reason_codes"] == '["AMR_FINDING"]'
    assert scored_metadata["safety_amr_state"] == state
    assert scored_metadata["safety_amr_reason_codes"] == '["AMR_FINDING"]'
    assert scored_metadata["safety_amr_execution_status"] == "COMPLETED_AND_PARSED"
    assert scored_metadata["safety_amr_finding_count"] == 1
    assert scored_metadata["safety_amr_policy_id"] == "ema-phage-v1"


@pytest.mark.parametrize(
    ("state", "gate_value", "raw_reward", "expected_reward"),
    [
        ("PASS", 1.0, 0.75, 0.75),
        ("PASS", 1, 0.75, 0.75),
        ("FAIL", 1.0, 0.75, 0.0),
        ("INDETERMINATE", 1.0, 0.75, 0.0),
        ("pass", 1.0, 0.75, 0.0),
        ("PASS", True, 0.75, 0.0),
        ("PASS", "1", 0.75, 0.0),
        ("PASS", 1.0, -0.25, 0.0),
        ("PASS", 1.0, 1.25, 0.0),
        ("PASS", 1.0, float("nan"), 0.0),
        ("PASS", 1.0, float("inf"), 0.0),
    ],
)
def test_scalar_step_regates_raw_reward_without_mutating_finite_diagnostics(
    monkeypatch,
    state: str,
    gate_value: object,
    raw_reward: float,
    expected_reward: float,
):
    """Scalar optimization output requires exact eligibility and a finite raw reward."""
    env_cls, env = _new_step_environment(reward_output_mode="scalar", gdpo_objectives=())

    def fake_score_message_logs(*_args, **_kwargs):
        return pd.DataFrame(
            {
                "sequence": ["ACGT"],
                "reward": [raw_reward],
                "reward_historical": [0.9],
                "safety_gate_state": [state],
                "safety_gate_pass": [gate_value],
            }
        )

    monkeypatch.setattr(nemo_rl_env, "score_message_logs", fake_score_message_logs)

    result = env_cls.step(env, [[{"role": "assistant", "content": "ACGT"}]], [{}])

    assert result.rewards.tolist() == [expected_reward]
    assert result.observations == [{"role": "environment", "content": f"phage_qc_reward={expected_reward:.6f}"}]
    scored_metadata = result.metadata[0]["_phage_qc_scored"]
    assert scored_metadata["reward_historical"] == 0.9
    assert scored_metadata["safety_gate_state"] == state
    if math.isfinite(raw_reward):
        assert scored_metadata["reward"] == raw_reward
    else:
        assert "reward" not in scored_metadata


@pytest.mark.parametrize(
    ("state", "gate_value", "raw_reward", "expected_observation", "expected_objective"),
    [
        ("PASS", 1.0, 0.6, 0.6, 0.75),
        ("FAIL", 1.0, 0.6, 0.0, 0.0),
        ("INDETERMINATE", 1.0, float("inf"), 0.0, 0.0),
        ("PASS", 1.0, float("inf"), 0.0, 0.75),
        ("PASS", 1.0, "malformed", 0.0, 0.75),
        ("PASS", 1.0, 1.5, 0.0, 0.75),
    ],
)
def test_gdpo_step_observation_uses_reconciled_bounded_scalar_reward(
    monkeypatch,
    state: str,
    gate_value: object,
    raw_reward: object,
    expected_observation: float,
    expected_objective: float,
):
    """GDPO observations never expose or format an untrusted raw scalar reward."""
    env_cls, env = _new_step_environment(
        reward_output_mode="gdpo",
        gdpo_objectives=(GDPOObjective("biological", ("reward_biological",)),),
    )

    def fake_score_message_logs(*_args, **_kwargs):
        return pd.DataFrame(
            {
                "sequence": ["ACGT"],
                "reward": [raw_reward],
                "reward_historical": [0.8],
                "reward_biological": [0.75],
                "safety_gate_state": [state],
                "safety_gate_pass": [gate_value],
            }
        )

    monkeypatch.setattr(nemo_rl_env, "score_message_logs", fake_score_message_logs)

    result = env_cls.step(env, [[{"role": "assistant", "content": "ACGT"}]], [{}])

    assert result.rewards.tolist() == [[expected_objective]]
    assert result.observations == [{"role": "environment", "content": f"phage_qc_reward={expected_observation:.6f}"}]
    assert result.metadata[0]["_phage_qc_scored"]["reward_historical"] == 0.8


def test_phage_qc_metrics_marks_timing_metrics_for_nemorl_timing_logger():
    """Timing columns should be returned with a marker for NeMo-RL timing/train routing."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 0.0],
            "safety_gate_state": ["PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0],
            "reward": [1.0, 0.0],
            f"{TIMING_COLUMN_PREFIX}reward/total_s": [2.0, 4.0],
        }
    )

    metrics = phage_qc_metrics_from_scored(scored, RewardWeights(valid_nt_chars=1.0))

    timing_key = f"{TIMING_METRIC_MARKER_PREFIX}phage_qc/reward/total_s"
    assert metrics[timing_key] == 3.0
    assert "timing/phage_qc/reward/total_s" not in metrics


def test_phage_qc_metrics_deduplicates_binary_passes_by_mmseqs_cluster():
    """Collapsed passing clusters should count once in the headline pass metric."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0, 1.0, 0.0],
            "mmseqs_cluster_id": ["group0:seq_0", "group0:seq_0", "group0:seq_2", ""],
            "mmseqs_cluster_size": [2, 2, 1, 0],
            "mmseqs_cluster_valid_for_clustering": [1.0, 1.0, 1.0, 0.0],
            "safety_gate_state": ["PASS", "PASS", "PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0, 1.0, 1.0],
            "reward": [1.0, 1.0, 1.0, 0.0],
        }
    )

    metrics = phage_qc_metrics_from_scored(scored, RewardWeights(valid_nt_chars=1.0))

    assert metrics["binary_core_pass_count"] == 3
    assert metrics["binary_core_pass_rate"] == 0.75
    assert metrics["binary_core_pass_cluster_deduplicated_count"] == 2
    assert metrics["binary_core_pass_cluster_deduplicated_rate"] == 0.5
    assert "binary_core_pass_cluster_duplicate_count" not in metrics
    assert "binary_core_pass_cluster_deduplication_fraction" not in metrics


def test_phage_qc_metrics_deduplicates_full_qc_passes_by_mmseqs_cluster():
    """Full-QC paper gates should have their own cluster-deduplicated headline metrics."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0, 1.0, 1.0],
            "reward_external_synteny_pass": [1.0, 1.0, 0.0, 1.0],
            "reward_external_average_protein_identity_pass": [1.0, 1.0, 1.0, 1.0],
            "reward_external_required_genes_pass": [1.0, 1.0, 1.0, 1.0],
            "mmseqs_cluster_id": ["group0:seq_0", "group0:seq_0", "group0:seq_2", ""],
            "mmseqs_cluster_size": [2, 2, 1, 0],
            "mmseqs_cluster_valid_for_clustering": [1.0, 1.0, 1.0, 0.0],
            "safety_gate_state": ["PASS", "PASS", "PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0, 1.0, 1.0],
            "reward": [1.0, 1.0, 1.0, 1.0],
        }
    )

    metrics = phage_qc_metrics_from_scored(scored, RewardWeights(valid_nt_chars=1.0))

    assert metrics["binary_core_pass_count"] == 4
    assert metrics["binary_core_pass_cluster_deduplicated_count"] == 3
    assert metrics["binary_full_qc_pass_count"] == 3
    assert metrics["binary_full_qc_pass_cluster_deduplicated_count"] == 2
    assert "binary_full_qc_pass_cluster_duplicate_count" not in metrics
    assert "binary_full_qc_pass_cluster_deduplication_fraction" not in metrics


def test_phage_qc_metrics_tolerates_legacy_metadata_without_mmseqs_cluster_id():
    """Partially preserved MMseqs metadata should not crash metric aggregation."""
    scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0],
            "mmseqs_cluster_size": [2, 2],
            "safety_gate_state": ["PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0],
            "reward": [1.0, 1.0],
        }
    )

    metrics = phage_qc_metrics_from_scored(scored, RewardWeights(valid_nt_chars=1.0))

    assert metrics["binary_core_pass_rate"] == 1.0
    assert metrics["binary_core_pass_cluster_deduplicated_rate"] == 1.0
    assert "mmseqs_cluster_num_clusters" not in metrics


def test_phage_qc_metrics_groups_training_metrics_by_prompt_prefix_length():
    """Prompt-length groups let W&B compare each prefix only with matching prefixes."""
    scored = pd.DataFrame(
        {
            "prompt_nt_length": [4, 4, 10],
            "reward_valid_nt_chars": [1.0, 0.0, 1.0],
            "safety_gate_state": ["PASS", "PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0, 1.0],
            "reward": [1.0, 0.0, 0.5],
        }
    )

    metrics = phage_qc_metrics_from_scored(scored, RewardWeights(valid_nt_chars=1.0))

    assert metrics["by_prompt_nt_length/4/num_sequences"] == 2
    assert metrics["by_prompt_nt_length/4/reward_mean"] == 0.5
    assert metrics["by_prompt_nt_length/4/valid_nt_chars_score_mean"] == 0.5
    assert metrics["by_prompt_nt_length/4/binary_core_pass_rate"] == 0.5
    assert metrics["by_prompt_nt_length/10/num_sequences"] == 1
    assert metrics["by_prompt_nt_length/10/reward_mean"] == 0.5
    assert metrics["by_prompt_nt_length/10/binary_core_pass_rate"] == 1.0


def test_scored_records_exclude_full_sequence_from_rollout_metadata():
    """Rollout metadata should carry scalar scores/status, not full generated sequences."""
    scored = pd.DataFrame(
        {
            "sequence": ["A" * 6000],
            "id_prompt": ["seq1"],
            "reward": [0.5],
            "reward_biological": [0.75],
            "reward_safety_amr": [1.0],
            "synteny_measurement_available": [1.0],
            "missing_status": ["unavailable"],
            "mmseqs_cluster_id": ["group0:seq_0"],
            "safety_gate_state": ["PASS"],
            "safety_gate_reason_codes": ['["SAFETY_OK"]'],
            "safety_amr_state": ["PASS"],
            "safety_amr_reason_codes": ["[]"],
            "safety_amr_execution_status": ["COMPLETED_AND_PARSED"],
            "safety_amr_policy_id": ["ema-phage-v1"],
            "safety_amr_finding_count": [0],
            "safety_nested_payload": [{"state": "PASS"}],
            "safety_list_payload": [["PASS"]],
            "safety_unbounded_payload": ["x" * 4097],
            "safety_invalid_unicode": ["\ud800"],
            "reward_nonfinite": [float("inf")],
            "reward_nan": [float("nan")],
            "reward_complex": [1 + 2j],
        }
    )

    records = _scored_records(scored)

    assert records == [
        {
            "reward": 0.5,
            "reward_biological": 0.75,
            "reward_safety_amr": 1.0,
            "synteny_measurement_available": 1.0,
            "mmseqs_cluster_id": "group0:seq_0",
            "safety_gate_state": "PASS",
            "safety_gate_reason_codes": '["SAFETY_OK"]',
            "safety_amr_state": "PASS",
            "safety_amr_reason_codes": "[]",
            "safety_amr_execution_status": "COMPLETED_AND_PARSED",
            "safety_amr_policy_id": "ema-phage-v1",
            "safety_amr_finding_count": 0,
        }
    ]
    json.dumps(records, allow_nan=False)


def test_global_post_process_metrics_leave_task_namespace_to_nemo_rl():
    """The environment hook returns bare keys because NeMo-RL adds the task namespace."""
    if getattr(nemo_rl_env, "_NEMO_RL_IMPORT_ERROR", None) is not None:
        pytest.skip("NeMo-RL is unavailable")

    env_cls = nemo_rl_env.PhageQCEnvironment.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env.weights = RewardWeights(valid_nt_chars=1.0)
    env.reward_output_mode = "gdpo"
    env.gdpo_objectives = (
        GDPOObjective("valid_nt_chars", ("reward_valid_nt_chars",)),
        GDPOObjective("protein_hit_count", ("reward_external_protein_hit_count",)),
    )
    env._last_gdpo_objective_scores = pd.DataFrame({"valid_nt_chars": [0.99], "protein_hit_count": [0.99]})
    batch = {
        "total_reward": torch.tensor([1.0, 0.0]),
        "extra_env_info": [
            {
                "_phage_qc_scored": {
                    "reward_valid_nt_chars": 1.0,
                    "reward_external_protein_hit_count": 0.25,
                    "safety_gate_state": "PASS",
                    "safety_gate_pass": 1.0,
                    "reward": 1.0,
                    f"{TIMING_COLUMN_PREFIX}reward/total_s": 2.0,
                }
            },
            {
                "_phage_qc_scored": {
                    "reward_valid_nt_chars": 0.0,
                    "reward_external_protein_hit_count": 0.75,
                    "safety_gate_state": "PASS",
                    "safety_gate_pass": 1.0,
                    "reward": 0.0,
                    f"{TIMING_COLUMN_PREFIX}reward/total_s": 4.0,
                }
            },
        ],
    }
    returned_batch, metrics = env_cls.global_post_process_and_metrics(env, batch)

    assert returned_batch is batch
    assert metrics["mean_reward"] == 0.5
    assert metrics["pass_rate"] == 0.5
    assert metrics["valid_nt_chars_score_mean"] == 0.5
    assert metrics["binary_core_pass_rate"] == 0.5
    assert "phage_qc/valid_nt_chars_score_mean" not in metrics
    assert metrics["gdpo/valid_nt_chars_mean"] == 0.5
    assert metrics["gdpo/valid_nt_chars_std"] == pytest.approx(0.5)
    assert metrics["gdpo/valid_nt_chars_min"] == 0.0
    assert metrics["gdpo/valid_nt_chars_max"] == 1.0
    assert metrics["gdpo/valid_nt_chars_nonzero_rate"] == 0.5
    assert metrics["gdpo/protein_hit_count_std"] == pytest.approx(0.25)
    assert metrics["gdpo/protein_hit_count_nonzero_rate"] == 1.0
    assert metrics[f"{TIMING_METRIC_MARKER_PREFIX}phage_qc/reward/total_s"] == 3.0
    assert "phage_qc/__timing__/phage_qc/reward/total_s" not in metrics


def test_global_post_process_metrics_handles_empty_gdpo_batch_without_actor_cache():
    """An empty assembled batch emits finite zero metrics and cannot reuse stale actor rows."""
    if getattr(nemo_rl_env, "_NEMO_RL_IMPORT_ERROR", None) is not None:
        pytest.skip("NeMo-RL is unavailable")

    env_cls = nemo_rl_env.PhageQCEnvironment.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env.weights = RewardWeights(valid_nt_chars=1.0)
    env.reward_output_mode = "gdpo"
    env.gdpo_objectives = (
        GDPOObjective("biological", ("reward_biological",)),
        GDPOObjective("safety_amr", ("reward_safety_amr",), requires_safety_eligibility=False),
    )
    env._last_scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0],
            "safety_gate_state": ["PASS"],
            "safety_gate_pass": [1.0],
            "reward": [1.0],
        }
    )
    env._last_gdpo_objective_scores = pd.DataFrame({"stale": [1.0]})

    _returned_batch, metrics = env_cls.global_post_process_and_metrics(
        env,
        {"total_reward": torch.empty((0, 2)), "extra_env_info": []},
    )

    assert metrics["mean_reward"] == 0.0
    assert metrics["pass_rate"] == 0.0
    assert metrics["dense_reward_ge_1_rate"] == 0.0
    assert metrics["num_sequences"] == 0
    assert metrics["binary_safety_qualified_full_qc_cluster_deduplicated_rate"] == 0.0
    assert metrics["gdpo/num_objectives"] == 2
    assert not any(key.startswith("gdpo/stale") for key in metrics)


def test_global_post_process_metrics_report_zero_acceptance_without_rollout_metadata():
    """Missing assembled metadata cannot be certified from an actor-local cache."""
    if getattr(nemo_rl_env, "_NEMO_RL_IMPORT_ERROR", None) is not None:
        pytest.skip("NeMo-RL is unavailable")

    env_cls = nemo_rl_env.PhageQCEnvironment.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env.weights = RewardWeights(valid_nt_chars=1.0)
    env._last_scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 0.0],
            "safety_gate_state": ["PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0],
            "reward": [1.0, 0.0],
        }
    )

    _returned_batch, metrics = env_cls.global_post_process_and_metrics(
        env,
        {"total_reward": torch.tensor([1.0, 0.0])},
    )

    assert metrics["pass_rate"] == 0.0
    assert metrics["num_sequences"] == 0
    assert metrics["binary_safety_qualified_full_qc_cluster_deduplicated_rate"] == 0.0
    assert "phage_qc/valid_nt_chars_score_mean" not in metrics


def test_global_post_process_metrics_does_not_fill_optional_fields_from_actor_cache():
    """Assembled rollout rows remain authoritative when optional cluster identity is absent."""
    if getattr(nemo_rl_env, "_NEMO_RL_IMPORT_ERROR", None) is not None:
        pytest.skip("NeMo-RL is unavailable")

    env_cls = nemo_rl_env.PhageQCEnvironment.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env.weights = RewardWeights(valid_nt_chars=1.0)
    env._last_scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0],
            "mmseqs_cluster_id": ["group0:seq_0", "group0:seq_0"],
            "mmseqs_cluster_size": [2, 2],
            "safety_gate_state": ["PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0],
            "reward": [1.0, 1.0],
        }
    )
    batch = {
        "total_reward": torch.tensor([1.0, 1.0]),
        "extra_env_info": [
            {
                "_phage_qc_scored": {
                    "reward_valid_nt_chars": 1.0,
                    "mmseqs_cluster_size": 2,
                    "safety_gate_state": "PASS",
                    "safety_gate_pass": 1.0,
                    "reward": 1.0,
                }
            },
            {
                "_phage_qc_scored": {
                    "reward_valid_nt_chars": 1.0,
                    "mmseqs_cluster_size": 2,
                    "safety_gate_state": "PASS",
                    "safety_gate_pass": 1.0,
                    "reward": 1.0,
                }
            },
        ],
    }

    _returned_batch, metrics = env_cls.global_post_process_and_metrics(env, batch)

    assert "mmseqs_cluster_num_clusters" not in metrics
    assert metrics["binary_core_pass_cluster_deduplicated_count"] == 2
    assert metrics["pass_rate"] == 1.0


@pytest.mark.parametrize(
    "extra_env_info",
    [
        [
            {
                "_phage_qc_scored": {
                    "reward_valid_nt_chars": 1.0,
                    "reward_external_synteny_pass": 1.0,
                    "reward_external_average_protein_identity_pass": 1.0,
                    "reward_external_required_genes_pass": 1.0,
                    "safety_gate_state": "PASS",
                    "safety_gate_pass": 1.0,
                    "reward": 1.0,
                }
            },
            None,
        ],
        [None, None],
        [
            {
                "_phage_qc_scored": {
                    "reward_valid_nt_chars": 1.0,
                    "reward_external_synteny_pass": 1.0,
                    "reward_external_average_protein_identity_pass": 1.0,
                    "reward_external_required_genes_pass": 1.0,
                    "safety_gate_state": "PASS",
                    "safety_gate_pass": 1.0,
                    "reward": 1.0,
                }
            }
        ],
    ],
)
def test_global_post_process_metrics_rejects_partial_or_misaligned_metadata(
    extra_env_info: list[dict[str, object] | None],
):
    """Dropping or compacting any rollout position invalidates batch-level certification."""
    if getattr(nemo_rl_env, "_NEMO_RL_IMPORT_ERROR", None) is not None:
        pytest.skip("NeMo-RL is unavailable")

    env_cls = nemo_rl_env.PhageQCEnvironment.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env.weights = RewardWeights(valid_nt_chars=1.0)
    env.reward_output_mode = "scalar"
    env._last_scored = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0, 1.0],
            "safety_gate_state": ["PASS", "PASS"],
            "safety_gate_pass": [1.0, 1.0],
            "reward": [1.0, 1.0],
        }
    )

    _returned_batch, metrics = env_cls.global_post_process_and_metrics(
        env,
        {"total_reward": torch.tensor([1.0, 1.0]), "extra_env_info": extra_env_info},
    )

    assert metrics["pass_rate"] == 0.0
    assert metrics["num_sequences"] == 0
    assert metrics["binary_safety_qualified_full_qc_cluster_deduplicated_rate"] == 0.0


@pytest.mark.parametrize("scored_row", [{}, {"reward": 1.0}, {"safety_gate_state": "PASS", "reward": 1.0}])
def test_global_gdpo_metrics_reject_correct_count_but_incomplete_scored_rows(
    scored_row: dict[str, object],
):
    """Correct row count cannot certify missing aggregate or configured objective telemetry."""
    if getattr(nemo_rl_env, "_NEMO_RL_IMPORT_ERROR", None) is not None:
        pytest.skip("NeMo-RL is unavailable")

    env_cls = nemo_rl_env.PhageQCEnvironment.__ray_metadata__.modified_class
    env = object.__new__(env_cls)
    env.weights = RewardWeights(valid_nt_chars=1.0)
    env.reward_output_mode = "gdpo"
    env.gdpo_objectives = (GDPOObjective("biological", ("reward_biological",)),)

    _returned_batch, metrics = env_cls.global_post_process_and_metrics(
        env,
        {
            "total_reward": torch.ones((2, 1)),
            "extra_env_info": [
                {"_phage_qc_scored": dict(scored_row)},
                {"_phage_qc_scored": dict(scored_row)},
            ],
        },
    )

    assert metrics["pass_rate"] == 0.0
    assert metrics["num_sequences"] == 0
    assert metrics["binary_safety_qualified_full_qc_cluster_deduplicated_rate"] == 0.0
    assert metrics["gdpo/num_objectives"] == 1
    assert not any(key.startswith("gdpo/biological_") for key in metrics)
