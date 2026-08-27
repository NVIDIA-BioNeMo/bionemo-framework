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

from __future__ import annotations

import pandas as pd
import pytest

from bionemo.evo2_phage_gen.calibration_selection import build_selection_table, summarize_setting


def _with_healthy_safety(frame: pd.DataFrame) -> pd.DataFrame:
    for safety_class in ("amr", "toxin", "lysogeny"):
        frame[f"safety_{safety_class}_measurement_available"] = 1.0
        frame[f"safety_{safety_class}_execution_status"] = "COMPLETED_AND_PARSED"
        frame[f"safety_{safety_class}_reason_codes"] = "[]"
    return frame


def test_summarize_setting_clusters_only_rows_in_that_setting(tmp_path):
    path = tmp_path / "prefix4_temp1.0.scores.csv"
    scored = pd.DataFrame(
        {
            "reward": [0.8, 0.6],
            "reward_external_protein_hit_count": [1.0, 0.5],
            "reward_external_tropism": [1.0, 0.0],
            "reward_external_required_genes": [0.5, 0.5],
            "reward_external_synteny": [0.7, 0.1],
            "reward_external_average_protein_identity": [1.0, 1.0],
            "reward_binary_full_qc_pass": [1.0, 0.0],
            "reward_binary_full_qc_cluster_deduplicated_pass": [1.0, 0.0],
            "external_qc_tool_succeeded": [1.0, 1.0],
            "protein_database_hit_count_measurement_available": [1.0, 1.0],
            "tropism_measurement_available": [1.0, 1.0],
            "required_genes_measurement_available": [1.0, 1.0],
            "synteny_measurement_available": [1.0, 1.0],
            "average_protein_identity_measurement_available": [1.0, 1.0],
            "mmseqs_cluster_num_clusters": [1, 1],
            "mmseqs_cluster_valid_for_clustering": [1.0, 1.0],
            "mmseqs_cluster_is_singleton": [0.0, 0.0],
        }
    )
    _with_healthy_safety(scored).to_csv(path, index=False)

    summary = summarize_setting(path, bootstrap_replicates=100)

    assert summary["within_setting_99pct_cluster_count"] == 1
    assert summary["within_setting_clusterable_count"] == 2
    assert summary["within_setting_99pct_distinct_rate"] == 0.5
    assert summary["target_signal_mean"] == pytest.approx(7 / 12)


def test_summarize_setting_marks_header_only_scores_ineligible(tmp_path):
    path = tmp_path / "prefix0_temp0.7.scores.csv"
    pd.DataFrame(columns=["reward", "mmseqs_cluster_num_clusters"]).to_csv(path, index=False)

    summary = summarize_setting(path, bootstrap_replicates=100)

    assert summary["records"] == 0
    assert summary["metric_environment_ok"] is False
    assert summary["within_setting_99pct_cluster_count"] == 0


def test_summarize_setting_rejects_unexplained_missing_safety_evidence(tmp_path):
    path = tmp_path / "prefix4_temp1.0.scores.csv"
    pd.DataFrame(
        {
            "external_qc_tool_succeeded": [1.0],
            "protein_database_hit_count_measurement_available": [1.0],
            "tropism_measurement_available": [1.0],
            "required_genes_measurement_available": [1.0],
            "synteny_measurement_available": [1.0],
            "average_protein_identity_measurement_available": [1.0],
            "safety_amr_measurement_available": [1.0],
            "safety_amr_execution_status": ["COMPLETED_AND_PARSED"],
            "safety_amr_reason_codes": ["[]"],
            "safety_toxin_measurement_available": [0.0],
            "safety_toxin_execution_status": ["NOT_STARTED"],
            "safety_toxin_reason_codes": ['["TOXIN_SCORER_NOT_STARTED"]'],
            "safety_lysogeny_measurement_available": [1.0],
            "safety_lysogeny_execution_status": ["COMPLETED_AND_PARSED"],
            "safety_lysogeny_reason_codes": ["[]"],
        }
    ).to_csv(path, index=False)

    summary = summarize_setting(path, bootstrap_replicates=10)

    assert summary["metric_environment_ok"] is False


def test_build_selection_table_records_configurable_comparability_margin(tmp_path):
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    frame = pd.DataFrame(
        {
            "reward": [0.8],
            "reward_external_protein_hit_count": [0.8],
            "reward_external_tropism": [0.8],
            "reward_external_required_genes": [0.8],
            "external_qc_tool_succeeded": [1.0],
            "protein_database_hit_count_measurement_available": [1.0],
            "tropism_measurement_available": [1.0],
            "required_genes_measurement_available": [1.0],
            "synteny_measurement_available": [1.0],
            "average_protein_identity_measurement_available": [1.0],
        }
    )
    _with_healthy_safety(frame).to_csv(score_dir / "prefix4_temp1.0.scores.csv", index=False)

    table = build_selection_table(score_dir, bootstrap_replicates=100, comparability_margin=0.02)

    assert table["comparability_margin"].tolist() == [0.02]
    with pytest.raises(ValueError, match="comparability_margin"):
        build_selection_table(score_dir, bootstrap_replicates=100, comparability_margin=-0.01)
