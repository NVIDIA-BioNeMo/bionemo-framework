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

from bionemo.evo2_phage_gen import calibration_scoring
from bionemo.evo2_phage_gen.calibration_scoring import (
    load_generation_records,
    summarize_cell,
    validate_score_file,
)
from bionemo.evo2_phage_gen.design_scope import HostDomain, HostEvidence
from bionemo.evo2_phage_gen.reward import SequenceSafetyRewardConfig


def _sequence_safety_config(tmp_path) -> SequenceSafetyRewardConfig:
    return SequenceSafetyRewardConfig(
        host_domain=HostDomain.BACTERIA,
        host_evidence=HostEvidence(
            source="test",
            source_version="v1",
            replication_host_domains=frozenset({HostDomain.BACTERIA}),
            confirmed=True,
            metadata={"accession": "NC_001422.1"},
        ),
        asset_manifest_path=tmp_path / "asset-manifest.yaml",
        diamond_bin=tmp_path / "bin" / "diamond",
        mmseqs_bin=tmp_path / "bin" / "mmseqs",
        policy_path=tmp_path / "policy.yaml",
        work_dir=tmp_path / "safety-work",
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


def test_score_cell_uses_the_phix_capsid_length_envelope(tmp_path, monkeypatch):
    generation = tmp_path / "prefix16_temp1.0.jsonl"
    generation.write_text(json.dumps({"id": "a", "prompt": "+~AC", "completion": "GT"}) + "\n")
    arc_config = tmp_path / "arc.yaml"
    arc_config.write_text("{}\n")
    captured = {}

    safety_config = _sequence_safety_config(tmp_path)

    def fake_score(sequences, *, config, sequence_safety, **_kwargs):
        captured["config"] = config
        captured["sequence_safety"] = sequence_safety
        return sequences.assign(reward_genome_length=1.0)

    monkeypatch.setattr(calibration_scoring, "score_nucleotide_metrics", fake_score)

    calibration_scoring.score_cell(
        generation_jsonl=generation,
        output_csv=tmp_path / "scores.csv",
        arc_config=arc_config,
        pipeline_script=tmp_path / "pipeline.py",
        work_dir=tmp_path / "work",
        tool_bin_dir=tmp_path / "bin",
        threads=1,
        sequence_safety=safety_config,
    )

    config = captured["config"]
    assert captured["sequence_safety"] is safety_config
    assert (config.genome_length_min, config.genome_length_max) == (5306, 5493)
    assert (
        config.genome_length_reward_lower_zero,
        config.genome_length_reward_lower_full,
        config.genome_length_reward_upper_full,
        config.genome_length_reward_upper_zero,
    ) == (3000, 5359, 5391, 5426)


def test_summarize_cell_separates_measured_zero_from_missing_support():
    scored = pd.DataFrame(
        {
            "reward_nucleotide_pass": [1.0, 1.0],
            "reward_external_protein_hit_count": [0.5, 0.0],
            "reward_external_tropism": [0.0, 0.0],
            "reward_external_required_genes": [0.2, 0.0],
            "reward_external_synteny": [0.1, 0.0],
            "reward_gene_a_origin": [0.3, 0.0],
            "reward_external_average_protein_identity": [0.8, 0.0],
            "reward_binary_full_qc_pass": [0.0, 0.0],
            "reward_binary_full_qc_cluster_deduplicated_pass": [0.0, 0.0],
            "external_qc_tool_succeeded": [1.0, 1.0],
            "protein_database_hit_count_measurement_available": [1.0, 1.0],
            "tropism_measurement_available": [1.0, 1.0],
            "required_genes_measurement_available": [1.0, 0.0],
            "synteny_measurement_available": [1.0, 0.0],
            "smooth_reference_measurement_available": [1.0, 0.0],
            "average_protein_identity_measurement_available": [1.0, 0.0],
            "reward_safety_amr": [1.0, 1.0],
            "reward_safety_toxin": [1.0, 0.0],
            "reward_safety_lysogeny": [1.0, 0.25],
            "safety_amr_measurement_available": [1.0, 1.0],
            "safety_amr_execution_status": ["COMPLETED_AND_PARSED", "COMPLETED_AND_PARSED"],
            "safety_amr_reason_codes": ["[]", "[]"],
            "safety_toxin_measurement_available": [1.0, 0.0],
            "safety_toxin_execution_status": ["COMPLETED_AND_PARSED", "NOT_STARTED"],
            "safety_toxin_reason_codes": ["[]", '["TOXIN_SCORER_NOT_STARTED"]'],
            "safety_lysogeny_measurement_available": [1.0, 0.0],
            "safety_lysogeny_execution_status": ["COMPLETED_AND_PARSED", "NOT_STARTED"],
            "safety_lysogeny_reason_codes": ["[]", '["PHROGS_SCORER_NOT_STARTED"]'],
            "safety_environment_healthy": [1.0, 0.0],
            "safety_gate_pass": [1.0, 0.0],
            "mmseqs_cluster_num_clusters": [2, 2],
        }
    )

    summary = summarize_cell("prefix4_temp1.0", scored)

    assert summary["tropism_reward_mean"] == 0.0
    assert summary["tropism_support_rate"] == 0.5
    assert summary["synteny_support_rate"] == 0.5
    assert summary["gene_a_origin_reward_mean"] == 0.15
    assert summary["gene_a_origin_support_rate"] == 0.5
    assert summary["reward_gene_a_origin_mean"] == 0.15
    assert summary["required_genes_support_rate"] == 0.5
    assert summary["all_external_measurements_available_rate"] == 0.5
    assert summary["reward_safety_amr_mean"] == 1.0
    assert summary["reward_safety_toxin_mean"] == 0.5
    assert summary["reward_safety_lysogeny_mean"] == 0.625
    assert summary["mmseqs_cluster_num_clusters"] == 2
    assert summary["metric_environment_ok"] is False


def test_summarize_cell_accepts_explicit_safety_inapplicability():
    scored = pd.DataFrame(
        {
            "external_qc_tool_succeeded": [1.0],
            "protein_database_hit_count_measurement_available": [1.0],
            "tropism_measurement_available": [1.0],
            "required_genes_measurement_available": [1.0],
            "synteny_measurement_available": [1.0],
            "smooth_reference_measurement_available": [1.0],
            "average_protein_identity_measurement_available": [1.0],
            "safety_amr_measurement_available": [1.0],
            "safety_amr_execution_status": ["COMPLETED_AND_PARSED"],
            "safety_amr_reason_codes": ["[]"],
            "safety_toxin_measurement_available": [0.0],
            "safety_toxin_execution_status": ["NOT_RUN"],
            "safety_toxin_reason_codes": ['["TOXIN_NO_PROTEIN_QUERIES"]'],
            "safety_lysogeny_measurement_available": [0.0],
            "safety_lysogeny_execution_status": ["NOT_RUN"],
            "safety_lysogeny_reason_codes": ['["PHROGS_NO_PREDICTED_GENES"]'],
            "safety_environment_healthy": [0.0],
        }
    )

    summary = summarize_cell("prefix0_temp0.7", scored)

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
