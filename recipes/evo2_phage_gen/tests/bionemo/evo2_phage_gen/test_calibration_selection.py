from __future__ import annotations

import pandas as pd

from bionemo.evo2_phage_gen.calibration_selection import summarize_setting


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
    scored.to_csv(path, index=False)

    summary = summarize_setting(path, bootstrap_replicates=100)

    assert summary["within_setting_99pct_cluster_count"] == 1
    assert summary["within_setting_clusterable_count"] == 2
    assert summary["within_setting_99pct_distinct_rate"] == 0.5
    assert summary["target_signal_mean"] == 7 / 12
