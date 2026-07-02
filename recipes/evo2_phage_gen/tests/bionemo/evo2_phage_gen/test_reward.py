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
    REWARD_COMPONENTS,
    RewardWeights,
    _add_average_protein_identity_rewards,
    _add_checkv_rewards,
    _add_diversity_rewards,
    _add_full_synteny_rewards,
    _add_genetic_architecture_rewards,
    _add_mmseqs_clustering_rewards,
    _add_percent_identity_rewards,
    _add_required_gene_rewards,
    _aggregate_reward,
    _apply_reward_floor,
    _bounded_percent_range_score,
    _bounded_range_score,
    _lower_bound_ratio_score,
    _scale_score_above_random_baseline,
    _upper_bound_ratio_score,
    _write_external_qc_config,
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


def test_score_nucleotide_metrics_homopolymer_reward_stays_dense():
    """Oversized homopolymers should retain an optimization signal instead of saturating at zero."""
    df = pd.DataFrame(
        {
            "id_prompt": ["short_run", "long_run"],
            "sequence": ["A" * 20 + "CGT" * 1500, "A" * 2000 + "CGT" * 1000],
        }
    )

    scored = score_nucleotide_metrics(df)

    assert 0.0 < scored.loc[1, "reward_nt_homopolymer"] < scored.loc[0, "reward_nt_homopolymer"] < 1.0


def test_score_nucleotide_metrics_can_weight_nucleotide_pass_bonus():
    """A pass-gated term should increase pressure on satisfying all online nucleotide filters."""
    df = pd.DataFrame(
        {
            "id_prompt": ["pass", "long_homopolymer"],
            "sequence": ["ACGT" * 1000, "A" * 200 + "CGT" * 1267],
        }
    )

    scored = score_nucleotide_metrics(
        df,
        weights=RewardWeights(
            valid_nt_chars=0.0,
            genome_length=0.0,
            gc_content=0.0,
            nt_homopolymer=0.0,
            nucleotide_pass=1.0,
        ),
    )

    assert scored.loc[0, "reward_nucleotide_pass"] == 1.0
    assert scored.loc[1, "reward_nucleotide_pass"] == 0.0
    assert scored["reward"].tolist() == [1.0, 0.0]


def test_reward_components_are_registered_and_clipped_to_unit_interval():
    """The aggregate RL score should be easy to reweight and stay in [0, 1]."""
    component_names = {component.name for component in REWARD_COMPONENTS}
    assert {"valid_nt_chars", "genome_length", "gc_content", "tropism", "diversity"}.issubset(component_names)

    df = pd.DataFrame(
        {
            "reward_valid_nt_chars": [2.0, -1.0],
            "reward_gc_content": [0.5, 0.25],
        }
    )
    scored = _aggregate_reward(
        df,
        RewardWeights(valid_nt_chars=1.0, genome_length=0.0, gc_content=1.0, nt_homopolymer=0.0),
    )

    assert scored["reward_valid_nt_chars"].tolist() == [1.0, 0.0]
    assert scored["reward"].tolist() == [0.75, 0.125]
    assert scored["reward_binary_pass"].tolist() == [0.0, 0.0]
    assert scored["reward_active_components"].tolist() == ["valid_nt_chars,gc_content"] * 2


def test_threshold_reward_helpers_plateau_at_pass_criteria():
    """Continuous threshold scores should not prefer over-matching acceptable criteria."""
    assert _lower_bound_ratio_score(7, 7) == 1.0
    assert _lower_bound_ratio_score(8, 7) == 1.0
    assert _lower_bound_ratio_score(20, 7) == 1.0
    assert _lower_bound_ratio_score(3.5, 7) == 0.5

    assert _upper_bound_ratio_score(10, 10) == 1.0
    assert _upper_bound_ratio_score(8, 10) == 1.0
    assert _upper_bound_ratio_score(20, 10) == 0.5

    assert _bounded_range_score(7, 7, 9) == 1.0
    assert _bounded_range_score(8, 7, 9) == 1.0
    assert _bounded_range_score(9, 7, 9) == 1.0
    assert _bounded_range_score(3.5, 7, 9) == 0.5
    assert _bounded_range_score(18, 7, 9) == 0.5

    assert _bounded_percent_range_score(0, 0, 95) == 1.0
    assert _bounded_percent_range_score(95, 0, 95) == 1.0
    assert _bounded_percent_range_score(97.5, 0, 95) == 0.5
    assert _bounded_percent_range_score(100, 0, 95) == 0.0


def test_random_baseline_scaling_maps_random_to_zero_and_pass_to_one():
    """RL components can normalize a random-like raw score down to zero."""
    assert _scale_score_above_random_baseline(1.0, 0.5) == 1.0
    assert _scale_score_above_random_baseline(0.5, 0.5) == 0.0
    assert _scale_score_above_random_baseline(0.75, 0.5) == 0.5
    assert _apply_reward_floor(0.1, 0.75) == 0.75
    assert _apply_reward_floor(0.9, 0.75) == 0.9


def test_soft_preference_components_do_not_gate_binary_pass():
    """Global pass should not reject known-viable-like designs for soft novelty preferences."""
    df = pd.DataFrame(
        {
            "reward_valid_nt_chars": [1.0],
            "reward_external_synteny": [0.0],
            "reward_external_average_protein_identity": [0.0],
        }
    )

    scored = _aggregate_reward(
        df,
        RewardWeights(valid_nt_chars=1.0, synteny=1.0, average_protein_identity=1.0),
    )

    assert scored["reward_binary_pass"].tolist() == [1.0]


def test_score_fasta_writes_reward_csv(tmp_path):
    """The reward CLI backing function should write per-sequence diagnostics."""
    input_fasta = tmp_path / "input.fasta"
    output_csv = tmp_path / "rewards.csv"
    input_fasta.write_text(">seq1\n" + "ACGT" * 1000 + "\n")

    score_fasta(input_fasta, output_csv)

    scored = pd.read_csv(output_csv)
    assert scored["reward"].tolist() == [1.0]


def test_external_qc_config_enables_paper_ready_validation_filters(tmp_path):
    """AAI and required-gene rewards should make Arc run the final paper-stage filters."""
    base_config = {
        "results_save_dir": "unused",
        "current_config_file": "unused",
        "evo_gen_seqs_fasta_file_save_location": "unused",
        "reference_genome_fasta": "",
        "genetic_architecture_reference_genome": "",
        "reference_tropism_protein": "",
        "mmseqs_db_protein_database": "",
        "training_data_genomes_fasta": "",
        "mmseqs_db_tropism_protein": "",
        "genetic_architecture_visualization_script": "",
        "protein_annotation_file": "",
        "reference_genome_gff_file_save_location": "",
    }
    base_config_path = tmp_path / "arc_config.yaml"
    base_config_path.write_text(yaml.safe_dump(base_config))
    input_fasta = tmp_path / "input.fasta"
    input_fasta.write_text(">umi1\nACGT\n")

    run_config_path = _write_external_qc_config(
        base_config_path,
        tmp_path / "run",
        input_fasta,
        ExternalQCRewardConfig(
            enable_synteny=True,
            synteny_mode="full",
            enable_average_protein_identity=True,
            enable_required_genes=True,
            enable_training_data_identity=True,
            enable_reference_genome_identity=True,
        ),
    )

    run_config = yaml.safe_load(run_config_path.read_text())
    assert run_config["training_data_sequence_identity_filter"] is True
    assert run_config["training_data_sequence_identity_metrics_file_save_location"] == (
        "qc4_training_data_sequence_identity_metrics.csv"
    )
    assert run_config["diversification_filtering"] is True
    assert run_config["mmseqs_reference_genome_sequence_identity_remove_filter"] is True
    assert run_config["reference_genome_sequence_identity_metrics_file_save_location"] == (
        "qc5_reference_genome_sequence_identity_metrics.csv"
    )
    assert run_config["genetic_architecture_visualization_and_synteny_filtering"] is True
    assert run_config["syntenic_gene_count_filter"] is True
    assert run_config["average_protein_sequence_identity_filter"] is True
    assert run_config["required_genes_filter"] is True
    assert run_config["reference_genome_gff_file_save_location"] is None


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
                "id_prompt": ["umi1", "umi2"],
                "sequence": ["ACGT", "ACGT"],
                "genetic_architecture_score": [1.0, 20.0],
            }
        ).to_csv(run_dir / "qc4_homology_filter_seqs.csv", index=False)
        phrogs_dir = run_dir / "qc4_mmseqs_results_protein_database"
        phrogs_dir.mkdir()
        pd.DataFrame(
            {
                "id_prompt": ["umi1_ORF.1", "umi1_ORF.2", "umi2_ORF.1"],
                "sequence": ["M", "M", "M"],
                "protein_database_mmseqs_target": ["phrog_1", "phrog_2", "phrog_1"],
                "protein_database_mmseqs_e_value": [1e-5, 1e-6, 1e-4],
                "protein_database_mmseqs_percent_identity": [80.0, 75.0, 70.0],
            }
        ).to_csv(phrogs_dir / "mmseqs2_hits.csv", index=False)
        tropism_dir = run_dir / "qc4_mmseqs_results_tropism_protein"
        tropism_dir.mkdir()
        pd.DataFrame(
            {
                "id_prompt": ["umi1_ORF.1", "umi2_ORF.1"],
                "sequence": ["M", "M"],
                "tropism_protein_mmseqs_target": ["G", "G"],
                "tropism_protein_mmseqs_e_value": [1e-5, 1e-4],
                "tropism_protein_mmseqs_percent_identity": [90.0, 30.0],
            }
        ).to_csv(tropism_dir / "mmseqs2_hits.csv", index=False)
        checkv_dir = run_dir / "qc4_checkv_results"
        checkv_dir.mkdir()
        pd.DataFrame(
            {"contig_id": ["umi1", "umi2"], "checkv_quality": ["Complete", "Medium-quality"]}
        ).to_csv(
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
    assert 0.0 < scored.loc[1, "reward"] < 1.0
    assert scored.loc[0, "reward_external_checkv"] == 1.0
    assert scored.loc[0, "reward_external_synteny"] == 1.0
    assert scored.loc[0, "reward_external_mmseqs_clustering"] == 1.0
    assert scored.loc[1, "reward_external_protein_hit_count"] == 0.5
    assert scored.loc[1, "reward_external_tropism"] == 0.5
    assert scored.loc[1, "reward_external_genetic_architecture"] == 0.5
    assert scored.loc[1, "reward_external_checkv"] == 0.35
    assert round(scored.loc[1, "reward_external_synteny"], 6) == 0.125


def test_mmseqs_clustering_reward_plateaus_for_representatives(tmp_path):
    """Cluster representatives should all get full credit regardless of cluster size."""
    run_dir = tmp_path / "arc_run"
    clusters_dir = run_dir / "qc5_mmseqs_results_clustering" / "mmseqs_results"
    clusters_dir.mkdir(parents=True)
    (clusters_dir / "clusters.tsv").write_text(
        "umi1\tumi1\n"
        "umi2\tumi2\n"
        "umi2\tumi3\n"
        "umi4\tumi4\n"
        "umi4\tumi5\n"
        "umi4\tumi6\n"
    )
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi2", "umi4"],
            "reward_external_mmseqs_clustering": [0.0, 0.0, 0.0],
        }
    )

    scored = _add_mmseqs_clustering_rewards(
        df,
        run_dir,
        {"mmseqs_clustering_results_dir_save_location": "qc5_mmseqs_results_clustering"},
    )

    assert scored["reward_external_mmseqs_clustering_pass"].tolist() == [1.0, 1.0, 1.0]
    assert scored["reward_external_mmseqs_clustering"].tolist() == [1.0, 1.0, 1.0]


def test_mmseqs_clustering_reward_gives_partial_credit_to_non_representatives(tmp_path):
    """Non-representative cluster members can keep a capped optimization signal."""
    run_dir = tmp_path / "arc_run"
    clusters_dir = run_dir / "qc5_mmseqs_results_clustering" / "mmseqs_results"
    clusters_dir.mkdir(parents=True)
    (clusters_dir / "clusters.tsv").write_text(
        "umi2\tumi2\n"
        "umi2\tumi3\n"
    )
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi2", "umi3"],
            "reward_external_mmseqs_clustering": [0.0, 0.0],
        }
    )

    scored = _add_mmseqs_clustering_rewards(
        df,
        run_dir,
        {"mmseqs_clustering_results_dir_save_location": "qc5_mmseqs_results_clustering"},
    )

    assert scored["reward_external_mmseqs_clustering_pass"].tolist() == [1.0, 0.0]
    assert scored["reward_external_mmseqs_clustering"].tolist() == [1.0, 0.5]


def test_mmseqs_clustering_reward_fails_closed_for_missing_cluster_members(tmp_path):
    """Rows absent from clustering output should not get singleton representative credit."""
    run_dir = tmp_path / "arc_run"
    clusters_dir = run_dir / "qc5_mmseqs_results_clustering" / "mmseqs_results"
    clusters_dir.mkdir(parents=True)
    (clusters_dir / "clusters.tsv").write_text("umi1\tumi1\n")
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi_missing"],
            "reward_external_mmseqs_clustering": [0.0, 0.0],
        }
    )

    scored = _add_mmseqs_clustering_rewards(
        df,
        run_dir,
        {"mmseqs_clustering_results_dir_save_location": "qc5_mmseqs_results_clustering"},
    )

    assert scored["reward_external_mmseqs_clustering_pass"].tolist() == [1.0, 0.0]
    assert scored["reward_external_mmseqs_clustering"].tolist() == [1.0, 0.0]


def test_full_synteny_reward_uses_arc_syntenic_gene_metrics(tmp_path):
    """Full synteny mode should score exact Arc syntenic/total gene artifacts."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["umi1", "umi2", "umi3"],
            "num_syntenic_genes": [2, 1, 11],
            "total_num_genes": [7, 7, 11],
        }
    ).to_csv(run_dir / "qc6_synteny_filter_metrics.csv", index=False)
    pd.DataFrame({"id_prompt": ["umi1"]}).to_csv(run_dir / "qc6_synteny_filter_seqs.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi2", "umi3"],
            "reward_external_synteny": [0.0, 0.0, 0.0],
        }
    )

    scored = _add_full_synteny_rewards(
        df,
        run_dir,
        {
            "synteny_metrics_file_save_location": "qc6_synteny_filter_metrics.csv",
            "synteny_filter_seqs_csv_file_save_location": "qc6_synteny_filter_seqs.csv",
            "syntenic_gene_count_range": [2, 14],
            "total_gene_count_range": [7, 14],
            "syntenic_total_gene_count_remove": [[11, 11]],
            "synteny_removed_pair_score_floor": 0.75,
        },
    )

    assert scored["num_syntenic_genes"].tolist() == [2, 1, 11]
    assert scored["total_num_genes"].tolist() == [7, 7, 11]
    assert scored["reward_external_synteny_pass"].tolist() == [1.0, 0.0, 0.0]
    assert scored["reward_external_synteny"].tolist() == [1.0, 0.5, 0.75]


def test_average_protein_identity_reward_uses_prefilter_metrics(tmp_path):
    """Average protein identity should plateau in range and punish over-similarity."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["umi1", "umi2", "umi3"],
            "average_protein_percent_identity": [0.0, 95.0, 97.5],
        }
    ).to_csv(run_dir / "qc6_average_protein_sequence_identity_metrics.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi2", "umi3"],
            "reward_external_average_protein_identity": [0.0, 0.0, 0.0],
        }
    )

    scored = _add_average_protein_identity_rewards(
        df,
        run_dir,
        {
            "average_protein_sequence_identity_metrics_file_save_location": (
                "qc6_average_protein_sequence_identity_metrics.csv"
            ),
            "average_protein_sequence_identity_range": [0, 95],
        },
        random_baseline=0.0,
        reward_floor=0.0,
    )

    assert scored["reward_external_average_protein_identity_pass"].tolist() == [1.0, 1.0, 0.0]
    assert scored["reward_external_average_protein_identity"].tolist() == [1.0, 1.0, 0.5]


def test_average_protein_identity_reward_floor_keeps_viable_like_high_aai_soft(tmp_path):
    """High AAI should remain a preference penalty, not a near-hard rejection."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["umi1"],
            "average_protein_percent_identity": [99.8],
        }
    ).to_csv(run_dir / "qc6_average_protein_sequence_identity_metrics.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1"],
            "reward_external_average_protein_identity": [0.0],
        }
    )

    scored = _add_average_protein_identity_rewards(
        df,
        run_dir,
        {
            "average_protein_sequence_identity_metrics_file_save_location": (
                "qc6_average_protein_sequence_identity_metrics.csv"
            ),
            "average_protein_sequence_identity_range": [0, 95],
        },
        random_baseline=0.0,
        reward_floor=0.75,
    )

    assert scored.loc[0, "average_protein_identity_raw_score"] < 0.1
    assert scored.loc[0, "reward_external_average_protein_identity"] == 0.75


def test_average_protein_identity_reward_floor_requires_measured_metric(tmp_path):
    """Missing AAI rows should fail closed rather than receiving the soft reward floor."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["umi1"],
            "average_protein_percent_identity": [99.8],
        }
    ).to_csv(run_dir / "qc6_average_protein_sequence_identity_metrics.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi_missing"],
            "reward_external_average_protein_identity": [0.0, 0.0],
        }
    )

    scored = _add_average_protein_identity_rewards(
        df,
        run_dir,
        {
            "average_protein_sequence_identity_metrics_file_save_location": (
                "qc6_average_protein_sequence_identity_metrics.csv"
            ),
            "average_protein_sequence_identity_range": [0, 95],
        },
        random_baseline=0.0,
        reward_floor=0.75,
    )

    assert scored["reward_external_average_protein_identity"].tolist() == [0.75, 0.0]
    assert scored["reward_external_average_protein_identity_pass"].tolist() == [0.0, 0.0]


def test_sequence_identity_rewards_use_paper_keep_range(tmp_path):
    """Training/reference identity rewards should pass no-hit/low-similarity and penalize near-copies."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["umi1", "umi2", "umi3"],
            "training_data_mmseqs_percent_identity": [0.0, 98.9, 99.45],
        }
    ).to_csv(run_dir / "qc4_training_data_sequence_identity_metrics.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi2", "umi3", "umi_missing"],
            "reward_external_training_data_identity": [0.0, 0.0, 0.0, 0.0],
        }
    )

    scored = _add_percent_identity_rewards(
        df,
        run_dir,
        {
            "training_data_sequence_identity_metrics_file_save_location": (
                "qc4_training_data_sequence_identity_metrics.csv"
            ),
            "training_data_sequence_identity_range": [0, 98.9],
        },
        metric_file_key="training_data_sequence_identity_metrics_file_save_location",
        default_metric_file="qc4_training_data_sequence_identity_metrics.csv",
        identity_column="training_data_mmseqs_percent_identity",
        range_key="training_data_sequence_identity_range",
        default_range=[0, 98.9],
        score_column="reward_external_training_data_identity",
        raw_score_column="training_data_identity_raw_score",
        pass_column="reward_external_training_data_identity_pass",
        random_baseline=0.0,
    )

    assert scored["reward_external_training_data_identity_pass"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert scored.loc[0, "reward_external_training_data_identity"] == 1.0
    assert scored.loc[1, "reward_external_training_data_identity"] == 1.0
    assert round(scored.loc[2, "reward_external_training_data_identity"], 6) == 0.5
    assert scored.loc[3, "reward_external_training_data_identity"] == 0.0


def test_genetic_architecture_reward_requires_measured_metric():
    """Rows missing from Arc's architecture output should not pass via the [0, 10] range lower bound."""
    homology_df = pd.DataFrame(
        {
            "id_prompt": ["umi1", "umi2"],
            "genetic_architecture_score": [0.0, 20.0],
        }
    )
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi2", "umi_missing"],
            "reward_external_genetic_architecture": [0.0, 0.0, 0.0],
        }
    )

    scored = _add_genetic_architecture_rewards(
        df,
        homology_df,
        {"genetic_architecture_score_range": [0, 10]},
    )

    assert scored["reward_external_genetic_architecture"].tolist() == [1.0, 0.5, 0.0]


def test_required_gene_reward_scales_above_random_baseline(tmp_path):
    """Required-gene reward should give no RL credit at a configured random baseline."""
    run_dir = tmp_path / "arc_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "id_prompt": ["umi1", "umi2", "umi3"],
            "required_genes_matched_count": [9, 6, 4],
            "required_genes_total_count": [9, 9, 9],
        }
    ).to_csv(run_dir / "qc6_required_genes_metrics.csv", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi2", "umi3"],
            "reward_external_required_genes": [0.0, 0.0, 0.0],
        }
    )

    scored = _add_required_gene_rewards(
        df,
        run_dir,
        {"required_genes_metrics_file_save_location": "qc6_required_genes_metrics.csv"},
        random_baseline=0.5,
    )

    assert scored["reward_external_required_genes_pass"].tolist() == [1.0, 0.0, 0.0]
    assert round(scored.loc[1, "reward_external_required_genes"], 6) == round(1 / 3, 6)
    assert scored.loc[0, "reward_external_required_genes"] == 1.0
    assert scored.loc[2, "reward_external_required_genes"] == 0.0


def test_checkv_reward_plateaus_for_all_configured_acceptable_labels(tmp_path):
    """Any CheckV label accepted by config should get full credit."""
    run_dir = tmp_path / "arc_run"
    quality_dir = run_dir / "qc4_checkv_results"
    quality_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "contig_id": ["umi1", "umi2", "umi3", "umi4"],
            "checkv_quality": ["Low-quality", "Medium-quality", "High-quality", "Complete"],
        }
    ).to_csv(quality_dir / "quality_summary.tsv", sep="\t", index=False)
    df = pd.DataFrame(
        {
            "arc_qc_id": ["umi1", "umi2", "umi3", "umi4"],
            "reward_external_checkv": [0.0, 0.0, 0.0, 0.0],
        }
    )

    scored = _add_checkv_rewards(
        df,
        run_dir,
        {
            "checkv_results_dir_save_location": "qc4_checkv_results",
            "checkv_quality_range": ["Low-quality", "Medium-quality", "High-quality", "Complete"],
        },
    )

    assert scored["reward_external_checkv_pass"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert scored["reward_external_checkv"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_diversity_reward_is_unit_scaled_and_quality_gated():
    """Diversity should be a 0-1 component and suppressed for low-quality samples."""
    df = pd.DataFrame(
        {
            "id_prompt": ["near_a", "near_b", "far_good", "far_low_quality"],
            "sequence": ["A" * 20, "A" * 19 + "C", "CGTACGTACGTACGTACGTA", "TGCATGCATGCATGCATGCA"],
            "reward_valid_nt_chars": [1.0, 1.0, 1.0, 0.0],
            "reward_genome_length": [1.0, 1.0, 1.0, 0.0],
            "reward_gc_content": [1.0, 1.0, 1.0, 0.0],
            "reward_nt_homopolymer": [1.0, 1.0, 1.0, 0.0],
        }
    )

    scored = _add_diversity_rewards(
        df,
        ExternalQCRewardConfig(enabled=True, enable_diversity=True, diversity_quality_threshold=0.5, diversity_kmer_size=4),
    )

    assert scored.loc[3, "reward_diversity"] == 0.0
    assert scored["reward_diversity"].between(0.0, 1.0).all()
    assert scored["reward_diversity"].max() > 0.0
