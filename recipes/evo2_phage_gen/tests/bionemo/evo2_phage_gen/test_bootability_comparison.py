import csv
import json
from pathlib import Path

import pytest
import torch

from bionemo.evo2_phage_gen.bootability_comparison import (
    analyze_model_score_distributions,
    analyze_phix174_similar_controls,
    annotate_natural_sft_membership,
    collect_predict_scores,
    compare_order_audit_scores,
    paired_stratified_auc_bootstrap,
    prepare_bootability_cohort,
    prepare_natural_positive_cohort,
    prepare_order_audit_cohort,
    prepare_prompt_counterfactual_cohort,
    summarize_sequence_logprobs,
)


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in records))


def test_prepare_bootability_cohort_writes_model_native_prompts_and_manifest(tmp_path):
    viable = tmp_path / "viable.fna"
    nonviable = tmp_path / "nonviable.fna"
    output = tmp_path / "cohort"
    _write_fasta(viable, [("works one", "acgt"), ("works two", "TGCA")])
    _write_fasta(nonviable, [("fails one", "AAAA")])

    manifest = prepare_bootability_cohort(viable, nonviable, output, marker="+~")

    rows = list(csv.DictReader(manifest.open()))
    assert [(row["sequence_id"], row["label"], row["original_id"]) for row in rows] == [
        ("viable_0000", "1", "works one"),
        ("viable_0001", "1", "works two"),
        ("nonviable_0000", "0", "fails one"),
    ]
    assert (output / "cohort_raw.fna").read_text() == (
        ">viable_0000\nACGT\n>viable_0001\nTGCA\n>nonviable_0000\nAAAA\n"
    )
    assert (output / "cohort_native.fna").read_text() == (
        ">viable_0000\n+~ACGT\n>viable_0001\n+~TGCA\n>nonviable_0000\n+~AAAA\n"
    )
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_prepare_bootability_cohort_rejects_non_dna(tmp_path):
    viable = tmp_path / "viable.fna"
    nonviable = tmp_path / "nonviable.fna"
    _write_fasta(viable, [("works", "ACNT")])
    _write_fasta(nonviable, [("fails", "AAAA")])

    with pytest.raises(ValueError, match="non-ACGT"):
        prepare_bootability_cohort(viable, nonviable, tmp_path / "cohort")


def test_prepare_order_audit_cohort_renames_and_interleaves_labels(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sequence_id,label,original_id,length,sha256\n"
        "viable_0000,1,works,4,aaa\n"
        "nonviable_0000,0,fails-a,4,bbb\n"
        "nonviable_0001,0,fails-b,4,ccc\n"
    )
    raw = tmp_path / "raw.fna"
    native = tmp_path / "native.fna"
    _write_fasta(raw, [("viable_0000", "AAAA"), ("nonviable_0000", "CCCC"), ("nonviable_0001", "GGGG")])
    _write_fasta(
        native,
        [("viable_0000", "+~AAAA"), ("nonviable_0000", "+~CCCC"), ("nonviable_0001", "+~GGGG")],
    )

    output = prepare_order_audit_cohort(manifest, raw, native, tmp_path / "audit", seed=174)

    rows = list(csv.DictReader(output.open()))
    assert [row["sequence_id"] for row in rows] == ["orderaudit_0000", "orderaudit_0001", "orderaudit_0002"]
    assert [row["label"] for row in rows] == ["0", "1", "0"]
    assert {row["source_sequence_id"] for row in rows} == {
        "viable_0000",
        "nonviable_0000",
        "nonviable_0001",
    }
    assert [line for line in (tmp_path / "audit" / "cohort_raw.fna").read_text().splitlines() if line.startswith(">")] == [
        ">orderaudit_0000",
        ">orderaudit_0001",
        ">orderaudit_0002",
    ]


def test_prepare_natural_positive_cohort_is_deterministic_and_preserves_native_prompts(tmp_path):
    source = tmp_path / "natural.fna"
    _write_fasta(
        source,
        [
            ("one", "+~AAAA"),
            ("two", "+$CCCC"),
            ("three", "+#GGGG"),
            ("four", "+^TTTT"),
        ],
    )

    first = prepare_natural_positive_cohort(source, tmp_path / "first", sample_count=3, seed=174)
    second = prepare_natural_positive_cohort(source, tmp_path / "second", sample_count=3, seed=174)

    first_rows = list(csv.DictReader(first.open()))
    second_rows = list(csv.DictReader(second.open()))
    assert first_rows == second_rows
    assert len(first_rows) == 3
    assert {row["prompt"] for row in first_rows} <= {"+~", "+$", "+#", "+^"}
    assert "+" not in (tmp_path / "first" / "natural_positive_raw.fna").read_text().splitlines()[1]
    assert "+" in (tmp_path / "first" / "natural_positive_native.fna").read_text().splitlines()[1]
    fixed_lines = (tmp_path / "first" / "natural_positive_fixed_plus_tilde.fna").read_text().splitlines()
    assert all(fixed_lines[index].startswith("+~") for index in range(1, len(fixed_lines), 2))


def test_annotate_natural_sft_membership_joins_payload_hash_and_preserves_order(tmp_path):
    natural_manifest = tmp_path / "natural.csv"
    natural_manifest.write_text(
        "sequence_id,sha256,length\n"
        "natural_00000,aaa,4000\n"
        "natural_00001,bbb,5000\n"
    )
    split_records = tmp_path / "split.jsonl"
    split_records.write_text(
        '{"payload_sha256":"bbb","split":"validation"}\n'
        '{"payload_sha256":"aaa","split":"train"}\n'
    )

    output = annotate_natural_sft_membership(
        natural_manifest,
        split_records,
        tmp_path / "annotated.csv",
    )

    rows = list(csv.DictReader(output.open()))
    assert [(row["sequence_id"], row["sft_membership"]) for row in rows] == [
        ("natural_00000", "train"),
        ("natural_00001", "held-out-validation"),
    ]


def test_prepare_prompt_counterfactual_cohort_reuses_sequences_across_prompts(tmp_path):
    source = tmp_path / "natural.fna"
    _write_fasta(source, [("one", "+~AAAA"), ("two", "+$CCCCCC"), ("three", "+#GGGGGGGG")])
    natural_dir = tmp_path / "natural"
    natural_manifest = prepare_natural_positive_cohort(source, natural_dir, sample_count=3, seed=174)

    output_manifest = prepare_prompt_counterfactual_cohort(
        natural_manifest,
        natural_dir / "natural_positive_raw.fna",
        tmp_path / "counterfactual",
        sample_count=2,
        prompts=("+$", "+~"),
    )

    rows = list(csv.DictReader(output_manifest.open()))
    assert len(rows) == 4
    assert {row["prompt"] for row in rows} == {"+$", "+~"}
    assert sorted(row["base_sequence_id"] for row in rows).count(rows[0]["base_sequence_id"]) == 2
    assert len({row["sha256"] for row in rows}) == 2


def test_paired_stratified_auc_bootstrap_preserves_pairing_and_is_deterministic():
    labels = [1, 1, 1, 0, 0, 0]
    model_scores = {
        "best": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
        "reversed": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    }

    first = paired_stratified_auc_bootstrap(labels, model_scores, replicates=200, seed=174)
    second = paired_stratified_auc_bootstrap(labels, model_scores, replicates=200, seed=174)

    assert first == second
    assert first["models"]["best"]["auc"] == 1.0
    assert first["models"]["reversed"]["auc"] == 0.0
    assert first["paired_auc_differences"]["best-minus-reversed"]["estimate"] == 1.0
    assert first["paired_auc_differences"]["best-minus-reversed"]["ci_low"] == 1.0


@pytest.mark.parametrize(
    ("per_token", "prefix_length", "sequence_length", "expected"),
    [
        ([-1.0, -2.0, -3.0], 0, 4, (-6.0, -2.0, 3)),
        ([-99.0, -1.0, -2.0, -3.0, -4.0], 2, 4, (-10.0, -2.5, 4)),
    ],
)
def test_summarize_sequence_logprobs_excludes_prompt_targets(
    per_token, prefix_length, sequence_length, expected
):
    assert summarize_sequence_logprobs(
        per_token,
        prefix_length=prefix_length,
        sequence_length=sequence_length,
    ) == expected


def test_collect_predict_scores_joins_dp_ranks_and_preserves_manifest_order(tmp_path):
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "seq_idx_map.json").write_text(json.dumps({"second": 0, "first": 1}))
    torch.save(
        {
            "seq_idx": torch.tensor([0]),
            "log_probs_seqs": torch.tensor([[-8.0, -1.0, -2.0, -3.0, 0.0]]),
            "loss_mask": torch.tensor([[True, True, True, True, False]]),
        },
        prediction_dir / "predictions__rank_0__dp_rank_0.pt",
    )
    torch.save(
        {
            "seq_idx": torch.tensor([1]),
            "log_probs_seqs": torch.tensor([[-9.0, -1.0, -2.0, -3.0, -4.0]]),
            "loss_mask": torch.tensor([[True, True, True, True, True]]),
        },
        prediction_dir / "predictions__rank_1__dp_rank_1.pt",
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sequence_id,label,original_id,prompt,length,sha256\n"
        "first,1,original first,+~,4,abc\n"
        "second,0,original second,+$,3,def\n"
    )

    output = collect_predict_scores(
        prediction_dir,
        manifest,
        tmp_path / "scores.csv",
        model="selected-sft",
        protocol="sft-native-prompt",
        prefix_length=2,
    )

    rows = list(csv.DictReader(output.open()))
    assert [row["sequence_id"] for row in rows] == ["first", "second"]
    assert [float(row["joint_log_likelihood"]) for row in rows] == [-10.0, -6.0]
    assert [float(row["mean_log_likelihood"]) for row in rows] == [-2.5, -2.0]
    assert [int(row["scored_tokens"]) for row in rows] == [4, 3]
    assert [row["prompt"] for row in rows] == ["+~", "+$"]


def test_analyze_model_score_distributions_reports_length_and_prompt_confounding():
    natural = [
        {
            "length": str(length),
            "mean_log_likelihood": str(score),
            "joint_log_likelihood": str(score * (length - 1)),
            "prompt": prompt,
        }
        for length, score, prompt in [
            (2000, -1.0, "+$"),
            (3000, -0.9, "+$"),
            (4000, -0.8, "+~"),
            (5000, -0.7, "+~"),
        ]
    ]
    bootability = [
        {"label": "1", "length": "4500", "mean_log_likelihood": "-0.75", "joint_log_likelihood": "-3374"},
        {"label": "1", "length": "4600", "mean_log_likelihood": "-0.72", "joint_log_likelihood": "-3311"},
        {"label": "0", "length": "4500", "mean_log_likelihood": "-1.8", "joint_log_likelihood": "-8098"},
        {"label": "0", "length": "4600", "mean_log_likelihood": "-1.6", "joint_log_likelihood": "-7358"},
    ]
    counterfactual = [
        {"base_sequence_id": base, "prompt": prompt, "mean_log_likelihood": str(score)}
        for base, plus_tilde, plus_dollar in [("a", -0.7, -1.0), ("b", -0.8, -1.2)]
        for prompt, score in [("+~", plus_tilde), ("+$", plus_dollar)]
    ]

    result = analyze_model_score_distributions(bootability, natural, counterfactual)

    assert result["bootability"]["auc_mean"] == 1.0
    assert result["bootability"]["average_precision_mean"] == 1.0
    assert result["natural"]["length_range"] == [2000, 5000]
    assert len(result["natural"]["length_bins"]) == 4
    assert result["natural_percentiles"]["viable"]["median"] > result["natural_percentiles"]["nonviable"]["median"]
    assert result["natural_percentiles"]["viable"]["fraction_central_90"] == 1.0
    assert result["natural_percentiles"]["nonviable"]["fraction_central_90"] == 0.0
    assert result["prompt_counterfactual"]["+$-minus-+~"]["median"] == pytest.approx(-0.35)


def test_analyze_phix174_similar_controls_reports_task_slice_and_robust_thresholds():
    bootability = [
        {"label": "1", "mean_log_likelihood": str(score)}
        for score in (-0.4, -0.3, -0.2, -0.1)
    ] + [
        {"label": "0", "mean_log_likelihood": str(score)}
        for score in (-1.0, -0.8, -0.6, -0.2)
    ]
    natural = [
        {
            "prompt": prompt,
            "length": str(length),
            "mean_log_likelihood": str(score),
            "original_id": original_id,
        }
        for prompt, length, score, original_id in [
            ("+~", 3999, -0.9, "outside low, complete genome"),
            ("+~", 4000, -0.4, "AB1 cultured phage, complete genome"),
            ("+~", 5000, -0.3, "MAG: inferred phage, complete genome"),
            ("+~", 6000, -0.2, "IMGVR_UViG_1"),
            ("+~", 6001, -0.1, "outside high, complete genome"),
            ("+$", 5000, -0.05, "wrong prompt, complete genome"),
        ]
    ]

    result = analyze_phix174_similar_controls(
        bootability,
        natural,
        prompt="+~",
        length_range=(4000, 6000),
    )

    assert result["cohorts"]["viable"]["n"] == 4
    assert result["cohorts"]["nonviable"]["n"] == 4
    assert result["cohorts"]["phix174_similar"]["n"] == 5
    assert result["cohorts"]["phix174_similar_length_matched"]["n"] == 3
    assert result["cohorts"]["phix174_similar_named_complete"]["n"] == 1
    assert result["length_range"] == [4000, 6000]
    assert result["viable_reference"]["median"] == pytest.approx(-0.25)
    assert result["viable_reference"]["mad"] == pytest.approx(0.1)
    separation = result["maximum_balanced_separation"]
    assert separation["hard_margin_exists"] is False
    assert separation["threshold"] == pytest.approx(-0.5)
    assert separation["sensitivity"] == 1.0
    assert separation["specificity"] == 0.75
    assert separation["pass"]["phix174_similar_length_matched"]["passed"] == 3
    threshold = result["thresholds"]["viable_median_minus_1_robust_sd"]
    assert threshold["threshold"] == pytest.approx(-0.39826)
    assert threshold["pass"]["viable"]["passed"] == 3
    assert threshold["pass"]["nonviable"]["passed"] == 1
    assert threshold["pass"]["phix174_similar_length_matched"]["passed"] == 2
    assert threshold["pass"]["phix174_similar_length_matched"]["fraction"] == pytest.approx(2 / 3)


def test_compare_order_audit_scores_quantifies_effect_on_auc_and_labels():
    original = [
        {"sequence_id": "p1", "label": "1", "mean_log_likelihood": "0.9"},
        {"sequence_id": "p2", "label": "1", "mean_log_likelihood": "0.8"},
        {"sequence_id": "n1", "label": "0", "mean_log_likelihood": "0.3"},
        {"sequence_id": "n2", "label": "0", "mean_log_likelihood": "0.2"},
    ]
    reordered = [
        {"source_sequence_id": "n2", "label": "0", "mean_log_likelihood": "0.19"},
        {"source_sequence_id": "p1", "label": "1", "mean_log_likelihood": "0.89"},
        {"source_sequence_id": "n1", "label": "0", "mean_log_likelihood": "0.31"},
        {"source_sequence_id": "p2", "label": "1", "mean_log_likelihood": "0.81"},
    ]

    result = compare_order_audit_scores(original, reordered)

    assert result["n"] == 4
    assert result["auc_original"] == 1.0
    assert result["auc_reordered"] == 1.0
    assert result["auc_change"] == 0.0
    assert result["paired_delta"]["max_abs"] == pytest.approx(0.01)
    assert result["paired_delta_by_label"]["viable"]["n"] == 2
