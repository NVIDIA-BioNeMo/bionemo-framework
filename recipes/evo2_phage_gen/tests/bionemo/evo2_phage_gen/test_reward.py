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

"""Tests for ``bionemo.evo2_phage_gen.reward``."""

from pathlib import Path

import pandas as pd
import yaml

from bionemo.evo2_phage_gen.reward import (
    ExternalQCRewardConfig,
    RewardWeights,
    _add_diversity_rewards,
    score_fasta,
    score_nucleotide_metrics,
)


def test_score_nucleotide_metrics_rewards_passing_sequence():
    """A sequence satisfying all online nucleotide filters should receive reward 1."""
    df = pd.DataFrame({"id_prompt": ["pass"], "sequence": ["ACGT" * 1000]})

    scored = score_nucleotide_metrics(df)

    assert scored.loc[0, "reward"] == 1.0
    assert scored.loc[0, "reward_valid_nt_chars"] == 1.0


def test_score_nucleotide_metrics_penalizes_invalid_sequence():
    """Invalid characters and out-of-range metrics should reduce the reward."""
    df = pd.DataFrame({"id_prompt": ["bad"], "sequence": ["NNNN"]})

    scored = score_nucleotide_metrics(df)

    assert scored.loc[0, "reward"] < 1.0
    assert scored.loc[0, "reward_valid_nt_chars"] == 0.0


def test_score_fasta_writes_reward_csv(tmp_path):
    """The reward CLI backing function should write per-sequence diagnostics."""
    input_fasta = tmp_path / "input.fasta"
    output_csv = tmp_path / "rewards.csv"
    input_fasta.write_text(">seq1\n" + "ACGT" * 1000 + "\n")

    score_fasta(input_fasta, output_csv)

    scored = pd.read_csv(output_csv)
    assert scored["reward"].tolist() == [1.0]


def test_score_nucleotide_metrics_can_fold_in_external_qc_rewards(tmp_path, monkeypatch):
    """The external Arc wrapper should map staged outputs back to per-sequence rewards."""
    annotation_file = tmp_path / "phrog_annot_v4.tsv"
    annotation_file.write_text("phrog\tannot\tcategory\nphrog_1\tterminase\tpackaging\nphrog_2\tendolysin\tlysis\n")
    base_config = {
        "results_save_dir": "unused",
        "current_config_file": "unused",
        "evo_gen_seqs_fasta_file_save_location": "unused",
        "overwrite_sequence_ids": False,
        "orf_filter_seqs_csv_file_save_location": "qc3_orf_filter_seqs.csv",
        "homology_filter_seqs_csv_file_save_location": "qc4_homology_filter_seqs.csv",
        "mmseqs_protein_database_results_dir_save_location": "qc4_mmseqs_results_protein_database",
        "mmseqs_tropism_protein_results_dir_save_location": "qc4_mmseqs_results_tropism_protein",
        "checkv_results_dir_save_location": "qc4_checkv_results",
        "diversification_filter_seqs_csv_file_save_location": "qc5_diversification_filter_seqs.csv",
        "synteny_filter_seqs_csv_file_save_location": "qc6_synteny_filter_seqs.csv",
        "protein_database_hit_count": 2,
        "protein_annotation_file": str(annotation_file),
        "required_genes_list": ["terminase", "endolysin"],
        "total_gene_count_range": [2, 2],
        "genetic_architecture_score_range": [0, 10],
        "tropism_protein_sequence_identity_range": [60, 100],
        "checkv_quality_range": ["Complete"],
    }
    config_path = tmp_path / "arc_config.yaml"
    config_path.write_text(yaml.safe_dump(base_config))
    pipeline_script = tmp_path / "genome_design_filtering_pipeline.py"
    pipeline_script.write_text("print('mock pipeline')\n")
    checkv_db = tmp_path / "checkv-db-v1.5"
    checkv_db.mkdir()

    def fake_run(args, check, cwd, env):
        assert env["CHECKVDB"] == str(checkv_db)
        run_config_path = Path(args[-1])
        run_config = yaml.safe_load(run_config_path.read_text())
        run_dir = Path(run_config["results_save_dir"])
        pd.DataFrame({"id_prompt": ["umi1"], "sequence": ["ACGT"]}).to_csv(
            run_dir / "qc3_orf_filter_seqs.csv", index=False
        )
        pd.DataFrame(
            {
                "id_prompt": ["umi1"],
                "sequence": ["ACGT"],
                "genetic_architecture_score": [1.0],
            }
        ).to_csv(run_dir / "qc4_homology_filter_seqs.csv", index=False)
        phrogs_dir = run_dir / "qc4_mmseqs_results_protein_database"
        phrogs_dir.mkdir()
        pd.DataFrame(
            {
                "id_prompt": ["umi1_ORF.1", "umi1_ORF.2"],
                "sequence": ["M", "M"],
                "protein_database_mmseqs_target": ["phrog_1", "phrog_2"],
                "protein_database_mmseqs_e_value": [1e-5, 1e-6],
                "protein_database_mmseqs_percent_identity": [80.0, 75.0],
            }
        ).to_csv(phrogs_dir / "mmseqs2_hits.csv", index=False)
        tropism_dir = run_dir / "qc4_mmseqs_results_tropism_protein"
        tropism_dir.mkdir()
        pd.DataFrame(
            {
                "id_prompt": ["umi1_ORF.1"],
                "sequence": ["M"],
                "tropism_protein_mmseqs_target": ["G"],
                "tropism_protein_mmseqs_e_value": [1e-5],
                "tropism_protein_mmseqs_percent_identity": [90.0],
            }
        ).to_csv(tropism_dir / "mmseqs2_hits.csv", index=False)
        checkv_dir = run_dir / "qc4_checkv_results"
        checkv_dir.mkdir()
        pd.DataFrame({"contig_id": ["umi1"], "checkv_quality": ["Complete"]}).to_csv(
            checkv_dir / "quality_summary.tsv", sep="\t", index=False
        )
        pd.DataFrame({"id_prompt": ["umi1"], "sequence": ["ACGT"]}).to_csv(
            run_dir / "qc5_diversification_filter_seqs.csv", index=False
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    df = pd.DataFrame({"id_prompt": ["seq0", "seq1"], "sequence": ["ACGT" * 1000, "ACGT" * 1000]})
    scored = score_nucleotide_metrics(
        df,
        weights=RewardWeights(
            valid_nt_chars=0,
            genome_length=0,
            gc_content=0,
            nt_homopolymer=0,
            orf=1,
            coding_density=1,
            protein_hit_count=1,
            tropism=1,
            genetic_architecture=1,
            checkv=1,
            synteny=1,
            mmseqs_clustering=1,
        ),
        external_qc=ExternalQCRewardConfig(
            enabled=True,
            config_path=config_path,
            pipeline_script=pipeline_script,
            work_dir=tmp_path / "work",
            checkv_db_path=checkv_db,
            keep_artifacts=True,
            enable_checkv=True,
            enable_synteny=True,
            enable_mmseqs_clustering=True,
        ),
    )

    assert scored.loc[0, "reward"] == 1.0
    assert scored.loc[1, "reward"] == 0.0
    assert scored.loc[0, "reward_external_checkv"] == 1.0
    assert scored.loc[0, "reward_external_synteny"] == 1.0
    assert scored.loc[0, "reward_external_mmseqs_clustering"] == 1.0


def test_diversity_reward_is_zscored_and_quality_gated():
    """Diversity should be normalized within the batch and suppressed for low-quality samples."""
    df = pd.DataFrame(
        {
            "id_prompt": ["near_a", "near_b", "far"],
            "sequence": ["A" * 20, "A" * 19 + "C", "CGTACGTACGTACGTACGTA"],
            "reward_valid_nt_chars": [1.0, 1.0, 0.0],
            "reward_genome_length": [1.0, 1.0, 0.0],
            "reward_gc_content": [1.0, 1.0, 0.0],
            "reward_nt_homopolymer": [1.0, 1.0, 0.0],
        }
    )

    scored = _add_diversity_rewards(
        df,
        ExternalQCRewardConfig(enabled=True, enable_diversity=True, diversity_quality_threshold=0.5, diversity_kmer_size=4),
    )

    assert scored.loc[2, "reward_diversity"] == 0.0
    assert round(float(scored["reward_diversity"].mean()), 6) <= 0.5
    assert scored["reward_diversity"].abs().max() <= 2.0
