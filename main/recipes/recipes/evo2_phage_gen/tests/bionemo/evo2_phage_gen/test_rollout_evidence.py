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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Focused tests for final-rollout evidence ordering and reconciliation."""

import csv
import json

import pytest

from bionemo.evo2_phage_gen.rollout_evidence import (
    cluster_post_qc_fasta,
    deduplicate_fasta,
    finalize_rollout_report,
    select_hard_qc_passers,
    summarize_arc_screen,
)


def test_deduplication_preserves_first_representative(tmp_path):
    raw = tmp_path / "raw.fasta"
    raw.write_text(">first\nAACG\n>exact\nAACG\n>rotation\nACGA\n>reverse-complement\nCGTT\n>unique\nGGTT\n")

    deduplicate_fasta(
        raw,
        tmp_path / "representatives.fasta",
        tmp_path / "mapping.csv",
        tmp_path / "deduplication.json",
    )

    assert (tmp_path / "representatives.fasta").read_text() == ">first\nAACG\n>unique\nGGTT\n"
    with (tmp_path / "mapping.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["representative_id"] for row in rows] == ["first", "first", "first", "first", "unique"]
    assert [row["duplicate_reason"] for row in rows] == [
        "",
        "exact",
        "circular_or_reverse_complement",
        "circular_or_reverse_complement",
        "",
    ]
    report = json.loads((tmp_path / "deduplication.json").read_text())
    assert report["counts"] == {
        "raw_records": 5,
        "representative_records": 2,
        "exact_duplicates_removed": 1,
        "circular_or_reverse_complement_duplicates_removed": 2,
    }


def test_hard_qc_requires_safety_and_target_pass(tmp_path):
    representatives = tmp_path / "representatives.fasta"
    representatives.write_text(">pass\nAACG\n>unsafe\nGGTT\n>indeterminate\nTTGC\n")
    safety = tmp_path / "safety.json"
    safety.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_type": "sequence_safety_scan",
                "policy": {"policy_id": "test-policy"},
                "tools": {"mmseqs": {"version": "test-mmseqs"}},
                "databases": {"phrogs": {"version": "test-phrogs"}},
                "records": [
                    {"record_id": "pass", "state": "PASS"},
                    {"record_id": "unsafe", "state": "FAIL"},
                    {"record_id": "indeterminate", "state": "INDETERMINATE"},
                ],
            }
        )
    )
    target = tmp_path / "target.fasta"
    target.write_text(">renamed-pass\nACGA\n>renamed-unsafe\nGGTT\n")

    select_hard_qc_passers(
        representatives,
        safety,
        target,
        tmp_path / "hard-qc.fasta",
        tmp_path / "hard-qc.json",
    )

    assert (tmp_path / "hard-qc.fasta").read_text() == ">pass\nAACG\n"
    report = json.loads((tmp_path / "hard-qc.json").read_text())
    assert report["safety_states"] == {"PASS": 1, "FAIL": 1, "INDETERMINATE": 1}
    assert report["target_profile_pass"] == 2
    assert report["hard_qc_pass"] == 1


def test_hard_qc_preserves_declared_pre_safety_qc_exclusions(tmp_path):
    representatives = tmp_path / "representatives.fasta"
    representatives.write_text(">screened\nAACG\n>invalid-before-safety\nAA~N\n")
    safety_input = tmp_path / "safety-input.fasta"
    safety_input.write_text(">screened\nAACG\n")
    safety = tmp_path / "safety.json"
    safety.write_text(json.dumps({"records": [{"record_id": "screened", "state": "PASS"}]}))
    target = tmp_path / "target.fasta"
    target.write_text(">target-screened\nAACG\n")

    select_hard_qc_passers(
        representatives,
        safety,
        target,
        tmp_path / "hard-qc.fasta",
        tmp_path / "hard-qc.json",
        safety_input_fasta=safety_input,
    )

    assert (tmp_path / "hard-qc.fasta").read_text() == ">screened\nAACG\n"
    report = json.loads((tmp_path / "hard-qc.json").read_text())
    assert report["input_representatives"] == 2
    assert report["safety_input_representatives"] == 1
    assert report["pre_safety_qc_excluded_representatives"] == 1
    assert report["safety_states"] == {"PASS": 1, "FAIL": 0, "INDETERMINATE": 0}
    assert report["target_profile_pass"] == 1


def test_hard_qc_still_rejects_missing_manifest_rows_for_safety_input(tmp_path):
    representatives = tmp_path / "representatives.fasta"
    representatives.write_text(">recorded\nAACG\n>missing\nGGTT\n>excluded\nAANN\n")
    safety_input = tmp_path / "safety-input.fasta"
    safety_input.write_text(">recorded\nAACG\n>missing\nGGTT\n")
    safety = tmp_path / "safety.json"
    safety.write_text(json.dumps({"records": [{"record_id": "recorded", "state": "PASS"}]}))
    target = tmp_path / "target.fasta"
    target.write_text(">recorded\nAACG\n")

    with pytest.raises(ValueError, match=r"safety/input mismatch: missing=\['missing'\], extra=\[\]"):
        select_hard_qc_passers(
            representatives,
            safety,
            target,
            tmp_path / "hard-qc.fasta",
            tmp_path / "hard-qc.json",
            safety_input_fasta=safety_input,
        )


def test_hard_qc_does_not_hide_malformed_sequences_submitted_to_safety(tmp_path):
    representatives = tmp_path / "representatives.fasta"
    representatives.write_text(">malformed\nAA~N\n")
    safety_input = tmp_path / "safety-input.fasta"
    safety_input.write_text(">malformed\nAA~N\n")
    safety = tmp_path / "safety.json"
    safety.write_text(json.dumps({"records": [{"record_id": "malformed", "state": "PASS"}]}))
    target = tmp_path / "target.fasta"
    target.write_text(">other\nAACG\n")

    with pytest.raises(ValueError, match=r"unsupported IUPAC symbols: ~"):
        select_hard_qc_passers(
            representatives,
            safety,
            target,
            tmp_path / "hard-qc.fasta",
            tmp_path / "hard-qc.json",
            safety_input_fasta=safety_input,
        )


def test_post_qc_clustering_pins_contract(tmp_path):
    candidates = tmp_path / "hard-qc.fasta"
    candidates.write_text(">first\nAACG\n>near\nAACT\n>other\nGGTT\n")
    fake_mmseqs = tmp_path / "mmseqs"
    fake_mmseqs.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "if sys.argv[1] == 'version':\n"
        "    print('fake-mmseqs 1.0')\n"
        "elif sys.argv[1] == 'createtsv':\n"
        "    pathlib.Path(sys.argv[-1]).write_text('first\\tfirst\\nfirst\\tnear\\nother\\tother\\n')\n"
    )
    fake_mmseqs.chmod(0o755)

    cluster_post_qc_fasta(
        candidates,
        tmp_path / "cluster-representatives.fasta",
        tmp_path / "memberships.csv",
        tmp_path / "clustering.json",
        work_dir=tmp_path / "work",
        mmseqs_bin=fake_mmseqs,
        threads=7,
    )

    assert (tmp_path / "cluster-representatives.fasta").read_text() == ">first\nAACG\n>other\nGGTT\n"
    report = json.loads((tmp_path / "clustering.json").read_text())
    assert report["counts"] == {"hard_qc_passers": 3, "clusters": 2, "duplicates_removed": 1}
    assert report["mmseqs"] == {
        "version": "fake-mmseqs 1.0",
        "min_sequence_identity": 0.99,
        "coverage": 0.8,
        "coverage_mode": 0,
        "cluster_mode": 0,
        "threads": 7,
    }
    cluster_command = next(command for command in report["commands"] if command[1] == "cluster")
    assert cluster_command[cluster_command.index("--min-seq-id") + 1] == "0.99"
    assert cluster_command[cluster_command.index("-c") + 1] == "0.8"


def test_arc_summary_omits_internal_clustering(tmp_path):
    representatives = tmp_path / "representatives.fasta"
    representatives.write_text(">a\nAACG\n>b\nGGTT\n>c\nTTGC\n")
    arc = tmp_path / "arc"
    arc.mkdir()
    (arc / "qc2_nt_filter_counts.csv").write_text(
        "count_initial_before_nucleotide_metrics,count_nt_filter,count_genome_len_filter\n3,3,2\n"
    )
    (arc / "qc3_orf_filter_counts.csv").write_text("count_orf_count_filter\n2\n")
    (arc / "qc4_homology_filter_counts.csv").write_text("count_protein_database_hit_count_filter\n2\n")
    (arc / "qc5_diversification_filter_counts.csv").write_text("count_genetic_architecture_score_remove_filter\n2\n")
    (arc / "qc6_synteny_filter_counts.csv").write_text(
        "count_required_genes_filter,count_syntenic_gene_count_filter\n2,1\n"
    )
    (arc / "qc6_synteny_filter_seqs.fasta").write_text(">a\nAACG\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "results_save_dir": str(arc),
                "mmseqs_clustering_filter": False,
                "genetic_architecture_remove_filter": False,
                "nucleotide_filter_counts_file_save_location": "qc2_nt_filter_counts.csv",
                "orf_filter_counts_file_save_location": "qc3_orf_filter_counts.csv",
                "homology_filter_counts_file_save_location": "qc4_homology_filter_counts.csv",
                "diversification_filter_counts_file_save_location": "qc5_diversification_filter_counts.csv",
                "synteny_filter_counts_file_save_location": "qc6_synteny_filter_counts.csv",
                "synteny_filter_seqs_fasta_file_save_location": "qc6_synteny_filter_seqs.fasta",
            }
        )
    )

    summarize_arc_screen(
        config,
        representatives,
        tmp_path / "target-screening.json",
        expected_filter7=False,
    )

    report = json.loads((tmp_path / "target-screening.json").read_text())
    assert report["input_representatives"] == 3
    assert report["final_pass_count"] == 1
    assert not report["arc_internal_mmseqs_clustering"]
    assert report["waterfall"][-1] == {"stage": "count_syntenic_gene_count_filter", "count": 1}


def test_final_report_reconciles_raw_and_representative_denominators(tmp_path):
    raw = tmp_path / "raw.fasta"
    raw.write_text(">a\nAACG\n>a-rotation\nACGA\n>b\nGG~T\n>c\nTTGC\n")
    mapping = tmp_path / "mapping.csv"
    mapping.write_text(
        "raw_index,record_id,representative_id,is_representative,duplicate_reason,length_nt\n"
        "0,a,a,true,,4\n"
        "1,a-rotation,a,false,circular_or_reverse_complement,4\n"
        "2,b,b,true,,4\n"
        "3,c,c,true,,4\n"
    )
    safety = tmp_path / "safety.json"
    safety.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_type": "sequence_safety_scan",
                "policy": {"policy_id": "test-policy"},
                "tools": {"mmseqs": {"version": "test-mmseqs"}},
                "databases": {"phrogs": {"version": "test-phrogs"}},
                "records": [
                    {"record_id": "a", "state": "PASS"},
                    {"record_id": "c", "state": "PASS"},
                ],
            }
        )
    )
    safety_input = tmp_path / "safety-input.fasta"
    safety_input.write_text(">a\nAACG\n>c\nTTGC\n")
    target = tmp_path / "target.fasta"
    target.write_text(">target-a\nACGA\n>target-c\nTTGC\n")
    diagnostic = tmp_path / "diagnostic.fasta"
    diagnostic.write_text(">diagnostic-c\nTTGC\n")
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "likelihood_rank,record_id,length_nt,scored_nucleotides,total_log_probability,"
        "mean_log_probability_per_nucleotide\n"
        "1,c,4,4,-0.4,-0.1\n"
        "2,a-rotation,4,4,-0.8,-0.2\n"
        "3,a,4,4,-1.2,-0.3\n"
        "4,b,4,4,-1.6,-0.4\n"
    )
    cluster_representatives = tmp_path / "cluster-representatives.fasta"
    cluster_representatives.write_text(">a\nAACG\n")
    memberships = tmp_path / "memberships.csv"
    memberships.write_text("representative_id,member_id\na,a\na,c\n")
    selection = tmp_path / "sampling-selection.yaml"
    selection.write_text("temperature: 1.0\nprompt_lengths: [16, 24]\n")

    finalize_rollout_report(
        raw,
        mapping,
        safety,
        target,
        diagnostic,
        scores,
        cluster_representatives,
        memberships,
        tmp_path / "final-designs.json",
        tmp_path / "accepted.fasta",
        tmp_path / "SUMMARY.md",
        model_checkpoint="sft/iter_0005200",
        rl_checkpoint="rl/step_400",
        sampling_selection=selection,
        safety_input_fasta=safety_input,
    )

    payload = json.loads((tmp_path / "final-designs.json").read_text())
    assert payload["workflow_order"] == [
        "raw_generation",
        "exact_circular_reverse_complement_deduplication",
        "safety_and_target_hard_qc",
        "post_qc_mmseqs_99pct_clustering",
        "ranking",
    ]
    assert payload["counts"] == {
        "raw_generated": 4,
        "raw_likelihood_scored": 4,
        "biological_representatives": 3,
        "duplicates_removed": 1,
        "safety_input_representatives": 2,
        "pre_safety_qc_excluded_representatives": 1,
        "safety_pass_representatives": 2,
        "safety_fail_representatives": 0,
        "safety_indeterminate_representatives": 0,
        "target_profile_pass_representatives": 2,
        "diagnostic_filter7_pass_representatives": 1,
        "hard_qc_pass_representatives": 2,
        "post_qc_99pct_clusters": 1,
        "accepted_cluster_representatives": 1,
    }
    duplicate = next(row for row in payload["records"] if row["record_id"] == "a-rotation")
    assert duplicate["representative_id"] == "a"
    assert duplicate["safety_state"] == "NOT_EVALUATED_DUPLICATE"
    assert not duplicate["accepted"]
    excluded = next(row for row in payload["records"] if row["record_id"] == "b")
    assert excluded["safety_state"] == "NOT_SCREENED_PRE_SAFETY_QC"
    assert excluded["representative_safety_state"] == "NOT_SCREENED_PRE_SAFETY_QC"
    assert not excluded["target_profile_pass"]
    assert not excluded["hard_qc_pass"]
    assert payload["sampling_selection"] == {"temperature": 1.0, "prompt_lengths": [16, 24]}
    assert payload["sequence_safety_provenance"]["tools"] == {"mmseqs": {"version": "test-mmseqs"}}
    assert payload["sequence_safety_provenance"]["databases"] == {"phrogs": {"version": "test-phrogs"}}
    assert (tmp_path / "accepted.fasta").read_text() == ">a\nAACG\n"
