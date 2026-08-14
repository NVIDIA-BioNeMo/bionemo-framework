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

import json

import pandas as pd
import pytest

from bionemo.evo2_phage_gen.calibration_scoring import (
    load_generation_records,
    summarize_cell,
    validate_score_file,
)


def test_load_generation_records_reconstructs_marker_free_genome(tmp_path):
    path = tmp_path / "prefix4_temp1.0.jsonl"
    path.write_text(json.dumps({"id": "a", "prompt": "+~GAGT", "completion": "ACGT"}) + "\n")

    records = load_generation_records(path)

    assert records.to_dict("records") == [{"id_prompt": "a", "sequence": "GAGTACGT"}]


def test_load_generation_records_uses_fallback_for_null_ids(tmp_path):
    path = tmp_path / "null-ids.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": None, "prompt": "+~AC", "completion": completion}) for completion in ("GT", "TG"))
        + "\n"
    )

    records = load_generation_records(path)

    assert records.to_dict("records") == [
        {"id_prompt": "null-ids_000000", "sequence": "ACGT"},
        {"id_prompt": "null-ids_000001", "sequence": "ACTG"},
    ]


def test_summarize_cell_separates_measured_zero_from_missing_support():
    scored = pd.DataFrame(
        {
            "reward_nucleotide_pass": [1.0, 1.0],
            "reward_external_protein_hit_count": [0.5, 0.0],
            "reward_external_tropism": [0.0, 0.0],
            "reward_external_required_genes": [0.2, 0.0],
            "reward_external_synteny": [0.1, 0.0],
            "reward_external_average_protein_identity": [0.8, 0.0],
            "reward_binary_full_qc_pass": [0.0, 0.0],
            "reward_binary_full_qc_cluster_deduplicated_pass": [0.0, 0.0],
            "external_qc_tool_succeeded": [1.0, 1.0],
            "protein_database_hit_count_measurement_available": [1.0, 1.0],
            "tropism_measurement_available": [1.0, 1.0],
            "required_genes_measurement_available": [1.0, 0.0],
            "synteny_measurement_available": [1.0, 0.0],
            "average_protein_identity_measurement_available": [1.0, 0.0],
            "mmseqs_cluster_num_clusters": [2, 2],
        }
    )

    summary = summarize_cell("prefix4_temp1.0", scored)

    assert summary["tropism_reward_mean"] == 0.0
    assert summary["tropism_support_rate"] == 1.0
    assert summary["required_genes_support_rate"] == 0.5
    assert summary["all_external_measurements_available_rate"] == 0.5
    assert summary["mmseqs_cluster_num_clusters"] == 2
    assert summary["metric_environment_ok"] is True


def test_validate_score_file_requires_complete_unique_ids(tmp_path):
    path = tmp_path / "scores.csv"
    pd.DataFrame({"id_prompt": ["a", "a"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        validate_score_file(path, expected_records=2)


def test_summarize_cell_preserves_missing_metrics_and_empty_cluster_count():
    summary = summarize_cell("prefix0_temp1.0", pd.DataFrame(index=[]))

    assert pd.isna(summary["reward_valid_nt_chars_mean"])
    assert summary["mmseqs_cluster_num_clusters"] is None
