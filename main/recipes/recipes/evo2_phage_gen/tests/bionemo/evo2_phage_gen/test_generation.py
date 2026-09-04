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

"""Tests for ``bionemo.evo2_phage_gen.generation``."""

import csv
import json

import pytest
import torch

from bionemo.evo2_phage_gen.generation import (
    collect_sft_likelihoods,
    ensure_paper_useful_rl_prompt_files,
    finalize_ranked_rollout,
    infer_jsonl_to_fasta,
    phix174_prompts,
    write_inference_prompt_shards,
    write_prompt_sweep_jsonl,
    write_rl_prompt_bank,
    write_sft_likelihood_fasta,
)


def test_phix174_prompts_use_reference_prefixes():
    """Prompt strings should be paper-style soft-token prefixes of the PhiX174 consensus start."""
    assert phix174_prompts(prompt_lengths=[4, 5, 9]) == {
        4: "+~GAGT",
        5: "+~GAGTT",
        9: "+~GAGTTTTAT",
    }


def test_phix174_prompts_allow_marker_only_calibration_control():
    """A zero-nucleotide calibration prompt should retain the learned marker."""
    assert phix174_prompts(prompt_lengths=[0]) == {0: "+~"}


def test_ensure_paper_useful_rl_prompt_files_materializes_openai_jsonl(tmp_path):
    """Paper-useful RL prompt files are deterministic PhiX174-start prompt artifacts."""
    paths = ensure_paper_useful_rl_prompt_files(tmp_path)

    train_records = [json.loads(line) for line in paths["train"].read_text().splitlines()]
    validation_records = [json.loads(line) for line in paths["validation"].read_text().splitlines()]
    train_prompts = [record["messages"][0]["content"].removeprefix("+~") for record in train_records]
    validation_prompts = [record["messages"][0]["content"].removeprefix("+~") for record in validation_records]

    assert [len(prompt) for prompt in train_prompts] == [4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 11]
    assert paths["validation"].name == "phage_prompts_paper_useful_rl_validation_prompt10_96.jsonl"
    assert len(validation_prompts) == 96
    assert set(validation_prompts) == {"GAGTTTTATC"}
    assert train_records[0]["messages"] == [
        {"role": "user", "content": "+~GAGT"},
        {"role": "assistant", "content": ""},
    ]


def test_write_rl_prompt_bank_supports_alternating_and_grouped_mixtures(tmp_path):
    alternating = write_rl_prompt_bank(
        tmp_path / "train.jsonl",
        prompt_lengths=[16, 24],
        repeats_per_length=2,
        id_prefix="train",
    )
    grouped = write_rl_prompt_bank(
        tmp_path / "validation.jsonl",
        prompt_lengths=[16, 24],
        repeats_per_length=2,
        id_prefix="validation",
        grouped=True,
    )

    alternating_records = [json.loads(line) for line in alternating.read_text().splitlines()]
    grouped_records = [json.loads(line) for line in grouped.read_text().splitlines()]
    assert [len(row["messages"][0]["content"]) - 2 for row in alternating_records] == [16, 24, 16, 24]
    assert [len(row["messages"][0]["content"]) - 2 for row in grouped_records] == [16, 16, 24, 24]
    assert len({row["id"] for row in alternating_records + grouped_records}) == 8


def test_write_rl_prompt_bank_supports_exact_nondivisible_record_count(tmp_path):
    path = write_rl_prompt_bank(
        tmp_path / "train.jsonl",
        prompt_lengths=[4, 6, 8],
        num_records=5,
        id_prefix="train",
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [len(row["messages"][0]["content"]) - 2 for row in records] == [4, 6, 8, 4, 6]
    assert len({row["id"] for row in records}) == 5


def test_write_prompt_sweep_jsonl_repeats_prompts(tmp_path):
    """Prompt JSONL files should be ready for ``infer_evo2 --prompt-file``."""
    paths = write_prompt_sweep_jsonl(tmp_path, prompt_lengths=[4, 6], num_prompts=2, id_prefix="test")

    assert [path.name for path in paths] == ["test_prompt4_2.jsonl", "test_prompt6_2.jsonl"]
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert records == [
        {"id": "test_prompt4_0000", "prompt": "+~GAGT"},
        {"id": "test_prompt4_0001", "prompt": "+~GAGT"},
    ]


def test_write_inference_prompt_shards_interleaves_lengths_with_exact_total(tmp_path):
    inputs = write_prompt_sweep_jsonl(
        tmp_path / "source",
        prompt_lengths=[4, 6, 8],
        num_prompts=4,
        id_prefix="test",
    )

    shards = write_inference_prompt_shards(
        inputs,
        tmp_path / "shards",
        num_records=10,
        num_shards=2,
    )

    records_by_shard = [[json.loads(line) for line in path.read_text().splitlines()] for path in shards]
    assert [len(records) for records in records_by_shard] == [5, 5]
    assert [len(record["prompt"]) - 2 for records in records_by_shard for record in records] == [
        4,
        6,
        8,
        4,
        6,
        8,
        4,
        6,
        8,
        4,
    ]
    assert len({record["id"] for records in records_by_shard for record in records}) == 10


def test_infer_jsonl_to_fasta_prepends_prompt_and_trims_eos(tmp_path):
    """FASTA reconstruction should combine prompt and completion before QC."""
    input_jsonl = tmp_path / "prompt4_temp0.7.jsonl"
    input_jsonl.write_text(
        json.dumps({"id": "seq1", "prompt": "+~GAGT", "completion": "ACGT STOP"})
        + "\n"
        + json.dumps({"id": "seq2", "prompt": "+~GAGT", "completion": "tgca<EOS>ignored"})
        + "\n"
    )
    output_fasta = tmp_path / "generated.fasta"

    infer_jsonl_to_fasta([input_jsonl], output_fasta)

    assert output_fasta.read_text() == (">seq1|prompt4_temp0.7\nGAGTACGT\n>seq2|prompt4_temp0.7\nGAGTTGCA\n")


def test_sft_likelihood_outputs_rank_every_design_by_mean_nucleotide_score(tmp_path):
    generated = tmp_path / "generated.fasta"
    generated.write_text(">high\nACGT\n>accepted-a\nTGCA\n>accepted-b\nAAAA\n")
    scoring_fasta = write_sft_likelihood_fasta(generated, tmp_path / "scoring.fasta")
    assert scoring_fasta.read_text() == ">high\n+~ACGT\n>accepted-a\n+~TGCA\n>accepted-b\n+~AAAA\n"

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "seq_idx_map.json").write_text(json.dumps({"accepted-a": 0, "accepted-b": 1, "high": 2}))
    torch.save(
        {
            "seq_idx": torch.tensor([0, 2]),
            "log_probs_seqs": torch.tensor(
                [
                    [-9.0, -0.2, -0.2, -0.2, -0.2],
                    [-9.0, -0.1, -0.1, -0.1, -0.1],
                ]
            ),
            "loss_mask": torch.ones((2, 5), dtype=torch.bool),
        },
        predictions / "predictions__rank_0__dp_rank_0.pt",
    )
    torch.save(
        {
            "seq_idx": torch.tensor([1]),
            "log_probs_seqs": torch.tensor([[-9.0, -0.3, -0.3, -0.3, -0.3]]),
            "loss_mask": torch.ones((1, 5), dtype=torch.bool),
        },
        predictions / "predictions__rank_1__dp_rank_1.pt",
    )

    score_csv = collect_sft_likelihoods(predictions, generated, tmp_path / "scores.csv")
    with score_csv.open() as handle:
        score_rows = list(csv.DictReader(handle))
    assert [row["record_id"] for row in score_rows] == ["high", "accepted-a", "accepted-b"]
    assert [int(row["likelihood_rank"]) for row in score_rows] == [1, 2, 3]
    assert [float(row["mean_log_probability_per_nucleotide"]) for row in score_rows] == pytest.approx(
        [-0.1, -0.2, -0.3]
    )
    assert [float(row["total_log_probability"]) for row in score_rows] == pytest.approx([-0.4, -0.8, -1.2])

    safety = tmp_path / "safety.json"
    safety.write_text(
        json.dumps(
            {
                "records": [
                    {"record_id": "high", "state": "FAIL"},
                    {"record_id": "accepted-a", "state": "PASS"},
                    {"record_id": "accepted-b", "state": "PASS"},
                ]
            }
        )
    )
    target = tmp_path / "target.fasta"
    target.write_text(">renamed-a\nTGCA\n>renamed-b\nAAAA\n")
    report = tmp_path / "final-designs.json"
    accepted = tmp_path / "accepted.fasta"
    summary = tmp_path / "SUMMARY.md"

    finalize_ranked_rollout(
        generated,
        safety,
        target,
        score_csv,
        report,
        accepted,
        summary,
        model_checkpoint="selected-sft/iter_0005600",
    )

    payload = json.loads(report.read_text())
    assert payload["counts"] == {
        "generated": 3,
        "likelihood_scored": 3,
        "safety_pass": 2,
        "target_profile_pass": 2,
        "accepted": 2,
    }
    assert payload["ranking"]["primary_score"] == "mean_log_probability_per_nucleotide"
    assert [row["record_id"] for row in payload["records"]] == ["high", "accepted-a", "accepted-b"]
    assert [row["accepted_rank"] for row in payload["records"]] == [None, 1, 2]
    assert accepted.read_text() == ">accepted-a\nTGCA\n>accepted-b\nAAAA\n"
    assert "within-protocol ranking signal" in summary.read_text()


def test_final_rollout_does_not_apply_likelihood_order_when_length_confounded(tmp_path):
    generated = tmp_path / "generated.fasta"
    generated.write_text(">short\nA\n>medium\nAC\n>long\nACG\n")
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "likelihood_rank,record_id,length_nt,scored_nucleotides,total_log_probability,mean_log_probability_per_nucleotide\n"
        "1,long,3,3,-0.3,-0.1\n"
        "2,medium,2,2,-0.4,-0.2\n"
        "3,short,1,1,-0.3,-0.3\n"
    )
    safety = tmp_path / "safety.json"
    safety.write_text(
        json.dumps(
            {"records": [{"record_id": record_id, "state": "PASS"} for record_id in ("short", "medium", "long")]}
        )
    )
    target = tmp_path / "target.fasta"
    target.write_text(generated.read_text())
    report = tmp_path / "final-designs.json"
    accepted = tmp_path / "accepted.fasta"
    summary = tmp_path / "SUMMARY.md"

    finalize_ranked_rollout(
        generated,
        safety,
        target,
        scores,
        report,
        accepted,
        summary,
        model_checkpoint="selected-sft",
    )

    payload = json.loads(report.read_text())
    diagnostic = payload["ranking"]["residual_length_association"]
    assert diagnostic["spearman_rho"] == pytest.approx(1.0)
    assert diagnostic["strong_correlation_threshold_abs_rho"] == 0.5
    assert not payload["ranking"]["applied_to_accepted_candidate_order"]
    assert accepted.read_text() == ">short\nA\n>medium\nAC\n>long\nACG\n"
    assert "not applied" in summary.read_text()


def test_final_rollout_reports_uninformative_constant_scores_without_nan(tmp_path):
    generated = tmp_path / "generated.fasta"
    generated.write_text(">short\nA\n>long\nACG\n")
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "likelihood_rank,record_id,length_nt,scored_nucleotides,total_log_probability,mean_log_probability_per_nucleotide\n"
        "1,short,1,1,-0.2,-0.2\n"
        "2,long,3,3,-0.6,-0.2\n"
    )
    safety = tmp_path / "safety.json"
    safety.write_text(
        json.dumps({"records": [{"record_id": record_id, "state": "PASS"} for record_id in ("short", "long")]})
    )
    target = tmp_path / "target.fasta"
    target.write_text(generated.read_text())
    report = tmp_path / "report" / "final-designs.json"

    finalize_ranked_rollout(
        generated,
        safety,
        target,
        scores,
        report,
        tmp_path / "report" / "accepted.fasta",
        tmp_path / "report" / "SUMMARY.md",
        model_checkpoint="selected-sft",
    )

    payload = json.loads(report.read_text(), parse_constant=lambda value: pytest.fail(f"unexpected {value}"))
    diagnostic = payload["ranking"]["residual_length_association"]
    assert diagnostic["spearman_rho"] is None
    assert diagnostic["p_value"] is None
    assert not diagnostic["strong_correlation"]
    assert not payload["ranking"]["applied_to_accepted_candidate_order"]
    assert (tmp_path / "report" / "accepted.fasta").read_text() == ">short\nA\n>long\nACG\n"


def test_final_rollout_allows_zero_target_passes(tmp_path):
    generated = tmp_path / "generated.fasta"
    generated.write_text(">candidate\nACGT\n")
    safety = tmp_path / "safety.json"
    safety.write_text(json.dumps({"records": [{"record_id": "candidate", "state": "PASS"}]}))
    target = tmp_path / "target.fasta"
    target.write_text("")
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "likelihood_rank,record_id,length_nt,scored_nucleotides,total_log_probability,"
        "mean_log_probability_per_nucleotide\n"
        "1,candidate,4,4,-0.4,-0.1\n"
    )
    report = tmp_path / "final-designs.json"
    accepted = tmp_path / "accepted.fasta"

    finalize_ranked_rollout(
        generated,
        safety,
        target,
        scores,
        report,
        accepted,
        tmp_path / "SUMMARY.md",
        model_checkpoint="selected-sft",
    )

    assert json.loads(report.read_text())["counts"]["target_profile_pass"] == 0
    assert json.loads(report.read_text())["counts"]["accepted"] == 0
    assert accepted.read_text() == ""
