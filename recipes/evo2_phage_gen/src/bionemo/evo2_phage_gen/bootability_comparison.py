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

"""Utilities for the one-off PhiX174 likelihood-scorer comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    record_id: str | None = None
    chunks: list[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if record_id is not None:
                records.append((record_id, "".join(chunks).upper()))
            record_id = line[1:].strip()
            chunks = []
            if not record_id:
                raise ValueError(f"empty FASTA identifier at {path}:{line_number}")
        elif record_id is None:
            raise ValueError(f"sequence precedes FASTA identifier at {path}:{line_number}")
        else:
            chunks.append(line)
    if record_id is not None:
        records.append((record_id, "".join(chunks).upper()))
    if not records:
        raise ValueError(f"no FASTA records found: {path}")
    return records


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{path}.partial")
    partial.write_text(text)
    os.replace(partial, path)


def prepare_bootability_cohort(
    viable_fasta: Path,
    nonviable_fasta: Path,
    output_dir: Path,
    *,
    marker: str = "+~",
) -> Path:
    """Normalize the 302-design cohort and emit raw/native-prompt FASTAs plus a manifest."""
    labeled_records: list[tuple[int, str, str, str]] = []
    seen_original_ids: set[str] = set()
    for label_name, label, path in (
        ("viable", 1, viable_fasta),
        ("nonviable", 0, nonviable_fasta),
    ):
        for index, (original_id, sequence) in enumerate(_read_fasta(path)):
            if original_id in seen_original_ids:
                raise ValueError(f"duplicate FASTA identifier: {original_id}")
            seen_original_ids.add(original_id)
            invalid = sorted(set(sequence) - set("ACGT"))
            if invalid:
                raise ValueError(f"non-ACGT symbols for {original_id}: {''.join(invalid)}")
            if not sequence:
                raise ValueError(f"empty sequence for {original_id}")
            labeled_records.append((label, f"{label_name}_{index:04d}", original_id, sequence))

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_text = "".join(f">{sequence_id}\n{sequence}\n" for _, sequence_id, _, sequence in labeled_records)
    native_text = "".join(f">{sequence_id}\n{marker}{sequence}\n" for _, sequence_id, _, sequence in labeled_records)
    _atomic_write(output_dir / "cohort_raw.fna", raw_text)
    _atomic_write(output_dir / "cohort_native.fna", native_text)

    manifest_path = output_dir / "cohort_manifest.csv"
    partial_manifest = Path(f"{manifest_path}.partial")
    with partial_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sequence_id", "label", "original_id", "length", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        for label, sequence_id, original_id, sequence in labeled_records:
            writer.writerow(
                {
                    "sequence_id": sequence_id,
                    "label": label,
                    "original_id": original_id,
                    "length": len(sequence),
                    "sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                }
            )
    os.replace(partial_manifest, manifest_path)
    return manifest_path


def prepare_order_audit_cohort(
    manifest_path: Path,
    raw_fasta: Path,
    native_fasta: Path,
    output_dir: Path,
    *,
    seed: int = 174,
) -> Path:
    """Rename and interleave a labeled cohort so sorted dataset order cannot preserve label blocks."""
    with manifest_path.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not {"sequence_id", "label"} <= set(fieldnames):
        raise ValueError("manifest must contain sequence_id and label")
    if {row["label"] for row in rows} != {"0", "1"}:
        raise ValueError("manifest must contain labels 0 and 1")
    raw_sequences = dict(_read_fasta(raw_fasta))
    native_sequences = dict(_read_fasta(native_fasta))
    expected_ids = {row["sequence_id"] for row in rows}
    if set(raw_sequences) != expected_ids or set(native_sequences) != expected_ids:
        raise ValueError("manifest and order-audit FASTAs must contain the same sequence IDs")

    by_label = {
        label: sorted(
            (row for row in rows if row["label"] == label),
            key=lambda row: hashlib.sha256(f"{seed}\0{row['sequence_id']}".encode()).hexdigest(),
        )
        for label in ("0", "1")
    }
    positive_positions = set(np.rint(np.linspace(0, len(rows) - 1, len(by_label["1"]) + 2)[1:-1]).astype(int).tolist())
    if len(positive_positions) != len(by_label["1"]):
        raise ValueError("could not assign distinct interleaved positions")
    negative_iter = iter(by_label["0"])
    positive_iter = iter(by_label["1"])
    ordered = [
        next(positive_iter) if index in positive_positions else next(negative_iter) for index in range(len(rows))
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    raw_chunks: list[str] = []
    native_chunks: list[str] = []
    for index, row in enumerate(ordered):
        source_id = row["sequence_id"]
        sequence_id = f"orderaudit_{index:04d}"
        output_rows.append(row | {"sequence_id": sequence_id, "source_sequence_id": source_id})
        raw_chunks.append(f">{sequence_id}\n{raw_sequences[source_id]}\n")
        native_chunks.append(f">{sequence_id}\n{native_sequences[source_id]}\n")
    _atomic_write(output_dir / "cohort_raw.fna", "".join(raw_chunks))
    _atomic_write(output_dir / "cohort_native.fna", "".join(native_chunks))

    output_manifest = output_dir / "cohort_manifest.csv"
    partial = Path(f"{output_manifest}.partial")
    output_fields = (*fieldnames, "source_sequence_id")
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    os.replace(partial, output_manifest)
    return output_manifest


def prepare_natural_positive_cohort(
    source_fasta: Path,
    output_dir: Path,
    *,
    sample_count: int = 10_000,
    seed: int = 174,
) -> Path:
    """Select deterministic natural-phage positive controls and preserve native prompts."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    candidates: dict[str, tuple[str, str, str]] = {}
    for original_id, prompted_sequence in _read_fasta(source_fasta):
        if len(prompted_sequence) < 3 or prompted_sequence[0] != "+" or prompted_sequence[1] not in "$#^~!":
            raise ValueError(f"invalid native prompt for {original_id}: {prompted_sequence[:2]!r}")
        prompt = prompted_sequence[:2]
        sequence = prompted_sequence[2:]
        invalid = sorted(set(sequence) - set("ACGT"))
        if invalid:
            raise ValueError(f"non-ACGT symbols for {original_id}: {''.join(invalid)}")
        sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
        candidates.setdefault(sequence_hash, (original_id, prompt, sequence))
    if len(candidates) < sample_count:
        raise ValueError(f"requested {sample_count} controls from only {len(candidates)} unique genomes")

    ranked = sorted(
        candidates.items(),
        key=lambda item: hashlib.sha256(f"{seed}\0{item[0]}\0{item[1][0]}".encode()).hexdigest(),
    )[:sample_count]
    records = [
        (f"natural_{index:05d}", original_id, prompt, sequence, sequence_hash)
        for index, (sequence_hash, (original_id, prompt, sequence)) in enumerate(ranked)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output_dir / "natural_positive_raw.fna",
        "".join(f">{sequence_id}\n{sequence}\n" for sequence_id, _, _, sequence, _ in records),
    )
    _atomic_write(
        output_dir / "natural_positive_native.fna",
        "".join(f">{sequence_id}\n{prompt}{sequence}\n" for sequence_id, _, prompt, sequence, _ in records),
    )
    _atomic_write(
        output_dir / "natural_positive_fixed_plus_tilde.fna",
        "".join(f">{sequence_id}\n+~{sequence}\n" for sequence_id, _, _, sequence, _ in records),
    )

    manifest_path = output_dir / "natural_positive_manifest.csv"
    partial_manifest = Path(f"{manifest_path}.partial")
    with partial_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sequence_id", "label", "control_type", "original_id", "prompt", "length", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        for sequence_id, original_id, prompt, sequence, sequence_hash in records:
            writer.writerow(
                {
                    "sequence_id": sequence_id,
                    "label": 1,
                    "control_type": "natural-positive",
                    "original_id": original_id,
                    "prompt": prompt,
                    "length": len(sequence),
                    "sha256": sequence_hash,
                }
            )
    os.replace(partial_manifest, manifest_path)
    return manifest_path


def annotate_natural_sft_membership(
    natural_manifest: Path,
    split_records_jsonl: Path,
    output_manifest: Path,
) -> Path:
    """Join natural controls to the selected SFT split by exact payload hash."""
    split_by_hash: dict[str, str] = {}
    for line in split_records_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sequence_hash = record["payload_sha256"]
        membership = "train" if record["split"] == "train" else f"held-out-{record['split']}"
        previous = split_by_hash.setdefault(sequence_hash, membership)
        if previous != membership:
            raise ValueError(f"payload hash occurs in multiple SFT splits: {sequence_hash}")

    with natural_manifest.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    if "sha256" not in fieldnames:
        raise ValueError("natural manifest must contain sha256")
    missing = [row["sequence_id"] for row in rows if row["sha256"] not in split_by_hash]
    if missing:
        raise ValueError(f"natural controls absent from SFT split manifest: {missing[:5]}")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{output_manifest}.partial")
    output_fields = fieldnames + (() if "sft_membership" in fieldnames else ("sft_membership",))
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row | {"sft_membership": split_by_hash[row["sha256"]]})
    os.replace(partial, output_manifest)
    return output_manifest


def prepare_prompt_counterfactual_cohort(
    natural_manifest: Path,
    natural_raw_fasta: Path,
    output_dir: Path,
    *,
    sample_count: int = 100,
    prompts: Sequence[str] = ("+!", "+$", "+#", "+^", "+~"),
) -> Path:
    """Repeat a length-spanning natural subset under each candidate control token."""
    if sample_count <= 1:
        raise ValueError("sample_count must exceed one")
    if not prompts or any(len(prompt) != 2 or prompt[0] != "+" for prompt in prompts):
        raise ValueError("prompts must be nonempty two-character control tokens")
    sequences = dict(_read_fasta(natural_raw_fasta))
    rows = list(csv.DictReader(natural_manifest.open()))
    if sample_count > len(rows):
        raise ValueError(f"requested {sample_count} controls from only {len(rows)} manifest rows")
    rows.sort(key=lambda row: (int(row["length"]), row["sha256"], row["sequence_id"]))
    selected_indices = np.rint(np.linspace(0, len(rows) - 1, sample_count)).astype(int)
    if len(set(selected_indices.tolist())) != sample_count:
        raise ValueError("length-spanning selection produced duplicate indices")

    output_rows: list[dict[str, str | int]] = []
    fasta_chunks: list[str] = []
    for subset_index, row_index in enumerate(selected_indices):
        row = rows[int(row_index)]
        sequence = sequences[row["sequence_id"]]
        if len(sequence) != int(row["length"]):
            raise ValueError(f"manifest/FASTA length mismatch for {row['sequence_id']}")
        for prompt_index, prompt in enumerate(prompts):
            sequence_id = f"counterfactual_{subset_index:03d}_{prompt_index}"
            fasta_chunks.append(f">{sequence_id}\n{prompt}{sequence}\n")
            output_rows.append(
                {
                    "sequence_id": sequence_id,
                    "label": 1,
                    "control_type": "prompt-counterfactual",
                    "base_sequence_id": row["sequence_id"],
                    "original_id": row["original_id"],
                    "original_prompt": row["prompt"],
                    "prompt": prompt,
                    "length": row["length"],
                    "sha256": row["sha256"],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / "prompt_counterfactual.fna", "".join(fasta_chunks))
    manifest_path = output_dir / "prompt_counterfactual_manifest.csv"
    partial_manifest = Path(f"{manifest_path}.partial")
    with partial_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    os.replace(partial_manifest, manifest_path)
    return manifest_path


def summarize_sequence_logprobs(
    per_token: Sequence[float],
    *,
    prefix_length: int,
    sequence_length: int,
) -> tuple[float, float, int]:
    """Aggregate only nucleotide targets, excluding any model-native prompt targets."""
    if prefix_length < 0 or sequence_length <= 0:
        raise ValueError("prefix_length must be non-negative and sequence_length must be positive")
    start = max(prefix_length - 1, 0)
    scored_tokens = sequence_length if prefix_length else sequence_length - 1
    values = np.asarray(per_token, dtype=float)[start : start + scored_tokens]
    if len(values) != scored_tokens or not np.isfinite(values).all():
        raise ValueError("per-token log-probs do not cover the expected finite nucleotide targets")
    return float(values.sum()), float(values.mean()), int(scored_tokens)


def collect_predict_scores(
    prediction_dir: Path,
    manifest_path: Path,
    output_csv: Path,
    *,
    model: str,
    protocol: str,
    prefix_length: int,
) -> Path:
    """Join DP-rank outputs and emit one validated score row per manifest sequence."""
    import torch

    index_map = {
        name: int(index) for name, index in json.loads((prediction_dir / "seq_idx_map.json").read_text()).items()
    }
    id_by_index = {index: name for name, index in index_map.items()}
    if len(id_by_index) != len(index_map):
        raise ValueError("seq_idx_map contains duplicate indices")

    predictions: dict[str, tuple[list[float], list[bool]]] = {}
    prediction_files = sorted(prediction_dir.glob("predictions__rank_*__dp_rank_*.pt"))
    if not prediction_files:
        raise ValueError(f"no epoch prediction files found: {prediction_dir}")
    for prediction_file in prediction_files:
        payload = torch.load(prediction_file, map_location="cpu", weights_only=True)
        required = {"seq_idx", "log_probs_seqs", "loss_mask"}
        if not required <= payload.keys():
            raise ValueError(f"missing per-token prediction fields in {prediction_file}")
        for sequence_index, log_probs, loss_mask in zip(
            payload["seq_idx"], payload["log_probs_seqs"], payload["loss_mask"], strict=True
        ):
            index = int(sequence_index.item())
            if index not in id_by_index:
                raise ValueError(f"unknown sequence index {index} in {prediction_file}")
            sequence_id = id_by_index[index]
            if sequence_id in predictions:
                raise ValueError(f"duplicate prediction for {sequence_id}")
            predictions[sequence_id] = (
                log_probs.detach().cpu().tolist(),
                loss_mask.detach().cpu().tolist(),
            )

    with manifest_path.open() as handle:
        manifest_reader = csv.DictReader(handle)
        manifest_fields = tuple(manifest_reader.fieldnames or ())
        manifest_rows = list(manifest_reader)
    required_manifest_fields = {"sequence_id", "label", "original_id", "length"}
    if not required_manifest_fields <= set(manifest_fields):
        raise ValueError(f"manifest must contain {sorted(required_manifest_fields)}")
    manifest_ids = [row["sequence_id"] for row in manifest_rows]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ValueError("manifest contains duplicate sequence identifiers")
    if set(predictions) != set(manifest_ids):
        missing = sorted(set(manifest_ids) - set(predictions))
        extra = sorted(set(predictions) - set(manifest_ids))
        raise ValueError(f"prediction/manifest mismatch: missing={missing}, extra={extra}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{output_csv}.partial")
    score_fields = (
        "model",
        "protocol",
        "joint_log_likelihood",
        "mean_log_likelihood",
        "scored_tokens",
    )
    fieldnames = manifest_fields + tuple(field for field in score_fields if field not in manifest_fields)
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in manifest_rows:
            sequence_length = int(row["length"])
            log_probs, loss_mask = predictions[row["sequence_id"]]
            expected_valid = sequence_length + prefix_length - 1
            mask = np.asarray(loss_mask, dtype=bool)
            if (
                expected_valid <= 0
                or len(mask) < expected_valid
                or not mask[:expected_valid].all()
                or mask[expected_valid:].any()
            ):
                raise ValueError(f"unexpected loss mask for {row['sequence_id']}")
            joint, mean, scored_tokens = summarize_sequence_logprobs(
                log_probs,
                prefix_length=prefix_length,
                sequence_length=sequence_length,
            )
            writer.writerow(
                row
                | {
                    "length": sequence_length,
                    "model": model,
                    "protocol": protocol,
                    "joint_log_likelihood": joint,
                    "mean_log_likelihood": mean,
                    "scored_tokens": scored_tokens,
                }
            )
    os.replace(partial, output_csv)
    return output_csv


def paired_stratified_auc_bootstrap(
    labels: Sequence[int],
    model_scores: dict[str, Sequence[float]],
    *,
    replicates: int = 10_000,
    seed: int = 174,
    confidence: float = 0.95,
) -> dict:
    """Return paired AUROCs and stratified bootstrap intervals for comparable scorers."""
    labels_array = np.asarray(labels, dtype=int)
    if set(labels_array.tolist()) != {0, 1}:
        raise ValueError("labels must contain both 0 and 1")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    score_arrays = {name: np.asarray(values, dtype=float) for name, values in model_scores.items()}
    if len(score_arrays) < 2:
        raise ValueError("at least two model score vectors are required")
    if any(values.shape != labels_array.shape for values in score_arrays.values()):
        raise ValueError("every score vector must match labels")
    if any(not np.isfinite(values).all() for values in score_arrays.values()):
        raise ValueError("scores must be finite")

    positive = np.flatnonzero(labels_array == 1)
    negative = np.flatnonzero(labels_array == 0)
    rng = np.random.default_rng(seed)
    bootstrap_aucs = {name: np.empty(replicates, dtype=float) for name in score_arrays}
    for replicate in range(replicates):
        indices = np.concatenate(
            (
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            )
        )
        replicate_labels = labels_array[indices]
        for name, values in score_arrays.items():
            bootstrap_aucs[name][replicate] = roc_auc_score(replicate_labels, values[indices])

    alpha = (1.0 - confidence) / 2.0
    quantiles = (100 * alpha, 100 * (1.0 - alpha))
    models: dict[str, dict[str, float]] = {}
    for name, values in score_arrays.items():
        ci_low, ci_high = np.percentile(bootstrap_aucs[name], quantiles)
        models[name] = {
            "auc": float(roc_auc_score(labels_array, values)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }

    paired: dict[str, dict[str, float]] = {}
    for first, second in itertools.combinations(score_arrays, 2):
        differences = bootstrap_aucs[first] - bootstrap_aucs[second]
        ci_low, ci_high = np.percentile(differences, quantiles)
        paired[f"{first}-minus-{second}"] = {
            "estimate": models[first]["auc"] - models[second]["auc"],
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "models": models,
        "paired_auc_differences": paired,
    }


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("distribution values must be nonempty and finite")
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def analyze_model_score_distributions(
    bootability_rows: Sequence[dict[str, str]],
    natural_rows: Sequence[dict[str, str]],
    counterfactual_rows: Sequence[dict[str, str]] = (),
) -> dict:
    """Summarize one scorer on labeled designs, natural controls, and prompt counterfactuals."""
    if not bootability_rows or not natural_rows:
        raise ValueError("bootability and natural rows must be nonempty")

    labels = np.asarray([int(row["label"]) for row in bootability_rows], dtype=int)
    mean_scores = np.asarray([float(row["mean_log_likelihood"]) for row in bootability_rows])
    joint_scores = np.asarray([float(row["joint_log_likelihood"]) for row in bootability_rows])
    natural_scores = np.asarray([float(row["mean_log_likelihood"]) for row in natural_rows])
    natural_lengths = np.asarray([int(row["length"]) for row in natural_rows], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("bootability rows must contain labels 0 and 1")
    if not all(np.isfinite(values).all() for values in (mean_scores, joint_scores, natural_scores)):
        raise ValueError("scores must be finite")
    if (natural_lengths <= 0).any():
        raise ValueError("natural sequence lengths must be positive")

    viable = labels == 1
    nonviable = labels == 0
    length_bins = []
    for bin_start in sorted(set((natural_lengths // 1000 * 1000).tolist())):
        selected = natural_lengths // 1000 * 1000 == bin_start
        length_bins.append(
            {
                "start": int(bin_start),
                "end": int(bin_start + 999),
                "length_min": int(natural_lengths[selected].min()),
                "length_max": int(natural_lengths[selected].max()),
                **_distribution_summary(natural_scores[selected]),
            }
        )

    centered_lengths = natural_lengths - natural_lengths.mean()
    centered_scores = natural_scores - natural_scores.mean()
    denominator = float(np.dot(centered_lengths, centered_lengths))
    slope_per_kb = float(np.dot(centered_lengths, centered_scores) / denominator * 1000.0) if denominator else 0.0
    fitted = natural_scores.mean() + slope_per_kb / 1000.0 * centered_lengths
    total_variation = float(np.dot(centered_scores, centered_scores))
    r_squared = 1.0 - float(np.square(natural_scores - fitted).sum()) / total_variation if total_variation else 0.0
    length_ranks = _average_ranks(natural_lengths.astype(float))
    score_ranks = _average_ranks(natural_scores)
    spearman = float(np.corrcoef(length_ranks, score_ranks)[0, 1]) if len(natural_scores) > 1 else 0.0

    natural_sorted = np.sort(natural_scores)
    natural_percentiles = np.searchsorted(natural_sorted, mean_scores, side="right") / len(natural_sorted)

    def percentile_summary(values: np.ndarray) -> dict[str, float | int]:
        return _distribution_summary(values) | {
            "fraction_central_90": float(((values >= 0.05) & (values <= 0.95)).mean())
        }

    natural_section: dict[str, object] = {
        "length_range": [int(natural_lengths.min()), int(natural_lengths.max())],
        "overall": _distribution_summary(natural_scores),
        "length_bins": length_bins,
        "length_spearman": spearman,
        "length_linear_slope_per_kb": slope_per_kb,
        "length_linear_r_squared": r_squared,
    }
    for field, output_name in (("prompt", "prompt_strata"), ("sft_membership", "sft_membership_strata")):
        groups = sorted({row.get(field, "") for row in natural_rows if row.get(field, "")})
        if groups:
            natural_section[output_name] = {
                group: _distribution_summary(
                    [float(row["mean_log_likelihood"]) for row in natural_rows if row.get(field, "") == group]
                )
                for group in groups
            }

    prompt_effects: dict[str, dict[str, float | int]] = {}
    if counterfactual_rows:
        by_sequence: dict[str, dict[str, float]] = {}
        for row in counterfactual_rows:
            by_sequence.setdefault(row["base_sequence_id"], {})[row["prompt"]] = float(row["mean_log_likelihood"])
        prompts = sorted({row["prompt"] for row in counterfactual_rows} - {"+~"})
        for prompt in prompts:
            deltas = [
                scores[prompt] - scores["+~"] for scores in by_sequence.values() if prompt in scores and "+~" in scores
            ]
            if deltas:
                prompt_effects[f"{prompt}-minus-+~"] = _distribution_summary(deltas)

    return {
        "score": "mean_log_likelihood",
        "bootability": {
            "n": len(labels),
            "auc_mean": float(roc_auc_score(labels, mean_scores)),
            "auc_joint": float(roc_auc_score(labels, joint_scores)),
            "average_precision_mean": float(average_precision_score(labels, mean_scores)),
            "average_precision_joint": float(average_precision_score(labels, joint_scores)),
            "viable": _distribution_summary(mean_scores[viable]),
            "nonviable": _distribution_summary(mean_scores[nonviable]),
        },
        "natural": natural_section,
        "natural_percentiles": {
            "viable": percentile_summary(natural_percentiles[viable]),
            "nonviable": percentile_summary(natural_percentiles[nonviable]),
        },
        "prompt_counterfactual": prompt_effects,
    }


def analyze_phix174_similar_controls(
    bootability_rows: Sequence[dict[str, str]],
    natural_rows: Sequence[dict[str, str]],
    *,
    prompt: str = "+~",
    length_range: tuple[int, int] = (4000, 6000),
) -> dict:
    """Compare PhiX174-similar natural controls with labeled-design score thresholds."""
    if not bootability_rows or not natural_rows:
        raise ValueError("bootability and natural rows must be nonempty")
    if len(prompt) != 2 or not prompt.startswith("+"):
        raise ValueError("prompt must be a two-character control token")
    length_min, length_max = length_range
    if length_min <= 0 or length_min > length_max:
        raise ValueError("length_range must be positive and ordered")

    labels = np.asarray([int(row["label"]) for row in bootability_rows], dtype=int)
    design_scores = np.asarray(
        [float(row["mean_log_likelihood"]) for row in bootability_rows],
        dtype=float,
    )
    if set(labels.tolist()) != {0, 1} or not np.isfinite(design_scores).all():
        raise ValueError("bootability rows must contain finite scores and labels 0 and 1")

    similar_rows = [row for row in natural_rows if row.get("prompt") == prompt]
    matched_rows = [row for row in similar_rows if length_min <= int(row["length"]) <= length_max]
    named_complete_rows = [
        row
        for row in matched_rows
        if "complete genome" in row.get("original_id", "").lower()
        and "mag:" not in row.get("original_id", "").lower()
        and "unverified:" not in row.get("original_id", "").lower()
    ]
    if not similar_rows or not matched_rows:
        raise ValueError("natural rows contain no requested PhiX174-similar cohort")

    cohorts = {
        "viable": design_scores[labels == 1],
        "nonviable": design_scores[labels == 0],
        "phix174_similar": np.asarray(
            [float(row["mean_log_likelihood"]) for row in similar_rows],
            dtype=float,
        ),
        "phix174_similar_length_matched": np.asarray(
            [float(row["mean_log_likelihood"]) for row in matched_rows],
            dtype=float,
        ),
    }
    if named_complete_rows:
        cohorts["phix174_similar_named_complete"] = np.asarray(
            [float(row["mean_log_likelihood"]) for row in named_complete_rows],
            dtype=float,
        )
    if any(not values.size or not np.isfinite(values).all() for values in cohorts.values()):
        raise ValueError("comparison cohorts must contain finite scores")

    viable = cohorts["viable"]
    viable_median = float(np.median(viable))
    viable_mad = float(np.median(np.abs(viable - viable_median)))
    robust_sd = 1.4826 * viable_mad
    threshold_values = {
        "viable_q05": float(np.quantile(viable, 0.05)),
        **{
            f"viable_median_minus_{multiple}_robust_sd": viable_median - multiple * robust_sd for multiple in (1, 2, 3)
        },
    }

    def pass_summary(values: np.ndarray, threshold: float) -> dict[str, float | int]:
        passed = int((values >= threshold).sum())
        n = len(values)
        fraction = passed / n
        z = 1.959963984540054
        denominator = 1.0 + z * z / n
        center = (fraction + z * z / (2 * n)) / denominator
        half_width = z * np.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / denominator
        return {
            "n": n,
            "passed": passed,
            "fraction": float(fraction),
            "wilson95_low": float(max(0.0, center - half_width)),
            "wilson95_high": float(min(1.0, center + half_width)),
        }

    unique_scores = np.unique(design_scores)
    candidate_thresholds = (unique_scores[:-1] + unique_scores[1:]) / 2 if len(unique_scores) > 1 else unique_scores
    candidates = []
    for threshold in candidate_thresholds:
        predicted_viable = design_scores >= threshold
        sensitivity = float(predicted_viable[labels == 1].mean())
        specificity = float((~predicted_viable[labels == 0]).mean())
        candidates.append(
            {
                "threshold": float(threshold),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": (sensitivity + specificity) / 2,
                "youden_j": sensitivity + specificity - 1,
                "margin_to_nearest_observation": float(np.min(np.abs(design_scores - threshold))),
            }
        )
    separation = max(
        candidates,
        key=lambda candidate: (
            candidate["balanced_accuracy"],
            candidate["margin_to_nearest_observation"],
            min(candidate["sensitivity"], candidate["specificity"]),
            candidate["specificity"],
        ),
    )
    separation_threshold = separation["threshold"]
    worst_viable = float(viable.min())
    best_nonviable = float(cohorts["nonviable"].max())
    separation |= {
        "rule": "maximize balanced accuracy; break ties by largest score margin",
        "hard_margin_exists": worst_viable > best_nonviable,
        "worst_viable": worst_viable,
        "best_nonviable": best_nonviable,
        "overlap": {
            "nonviable_at_or_above_worst_viable": int((cohorts["nonviable"] >= worst_viable).sum()),
            "viable_at_or_below_best_nonviable": int((viable <= best_nonviable).sum()),
        },
        "pass": {cohort_name: pass_summary(values, separation_threshold) for cohort_name, values in cohorts.items()},
    }

    return {
        "score": "mean_log_likelihood",
        "higher_is_better": True,
        "source_prompt": prompt,
        "length_range": [length_min, length_max],
        "cohorts": {name: _distribution_summary(values) for name, values in cohorts.items()},
        "viable_reference": {
            "median": viable_median,
            "mad": viable_mad,
            "robust_sd": robust_sd,
        },
        "maximum_balanced_separation": separation,
        "thresholds": {
            name: {
                "threshold": threshold,
                "pass": {cohort_name: pass_summary(values, threshold) for cohort_name, values in cohorts.items()},
            }
            for name, threshold in threshold_values.items()
        },
    }


def compare_order_audit_scores(
    original_rows: Sequence[dict[str, str]],
    reordered_rows: Sequence[dict[str, str]],
) -> dict:
    """Measure whether changed loader/batch/rank order materially changes scorer discrimination."""
    original = {row["sequence_id"]: (int(row["label"]), float(row["mean_log_likelihood"])) for row in original_rows}
    reordered = {
        row["source_sequence_id"]: (int(row["label"]), float(row["mean_log_likelihood"])) for row in reordered_rows
    }
    if len(original) != len(original_rows) or len(reordered) != len(reordered_rows):
        raise ValueError("order-audit rows must have unique sequence identifiers")
    if set(original) != set(reordered):
        raise ValueError("original and reordered cohorts contain different sequences")

    labels: list[int] = []
    original_scores: list[float] = []
    reordered_scores: list[float] = []
    for sequence_id, (label, score) in original.items():
        reordered_label, reordered_score = reordered[sequence_id]
        if reordered_label != label:
            raise ValueError(f"label mismatch for {sequence_id}")
        labels.append(label)
        original_scores.append(score)
        reordered_scores.append(reordered_score)
    label_array = np.asarray(labels, dtype=int)
    original_array = np.asarray(original_scores, dtype=float)
    reordered_array = np.asarray(reordered_scores, dtype=float)
    if set(label_array.tolist()) != {0, 1} or not np.isfinite(np.concatenate((original_array, reordered_array))).all():
        raise ValueError("order-audit labels and scores are invalid")

    deltas = reordered_array - original_array

    def delta_summary(values: np.ndarray) -> dict[str, float | int]:
        return _distribution_summary(values) | {"max_abs": float(np.abs(values).max())}

    auc_original = float(roc_auc_score(label_array, original_array))
    auc_reordered = float(roc_auc_score(label_array, reordered_array))
    pearson = float(np.corrcoef(original_array, reordered_array)[0, 1])
    spearman = float(np.corrcoef(_average_ranks(original_array), _average_ranks(reordered_array))[0, 1])
    return {
        "n": len(label_array),
        "positive": int(label_array.sum()),
        "negative": int((label_array == 0).sum()),
        "auc_original": auc_original,
        "auc_reordered": auc_reordered,
        "auc_change": auc_reordered - auc_original,
        "score_pearson": pearson,
        "score_spearman": spearman,
        "paired_delta": delta_summary(deltas),
        "paired_delta_by_label": {
            "viable": delta_summary(deltas[label_array == 1]),
            "nonviable": delta_summary(deltas[label_array == 0]),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Normalize labeled FASTAs for scorer inference")
    prepare.add_argument("--viable-fasta", type=Path, required=True)
    prepare.add_argument("--nonviable-fasta", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--marker", default="+~")

    prepare_order = subparsers.add_parser("prepare-order-audit", help="Interleave and rename a labeled cohort")
    prepare_order.add_argument("--manifest", type=Path, required=True)
    prepare_order.add_argument("--raw-fasta", type=Path, required=True)
    prepare_order.add_argument("--native-fasta", type=Path, required=True)
    prepare_order.add_argument("--output-dir", type=Path, required=True)
    prepare_order.add_argument("--seed", type=int, default=174)

    prepare_natural = subparsers.add_parser("prepare-natural", help="Select natural-phage positive controls")
    prepare_natural.add_argument("--source-fasta", type=Path, required=True)
    prepare_natural.add_argument("--output-dir", type=Path, required=True)
    prepare_natural.add_argument("--sample-count", type=int, default=10_000)
    prepare_natural.add_argument("--seed", type=int, default=174)

    prepare_counterfactual = subparsers.add_parser(
        "prepare-counterfactual", help="Build a same-sequence control-token cohort"
    )
    prepare_counterfactual.add_argument("--natural-manifest", type=Path, required=True)
    prepare_counterfactual.add_argument("--natural-raw-fasta", type=Path, required=True)
    prepare_counterfactual.add_argument("--output-dir", type=Path, required=True)
    prepare_counterfactual.add_argument("--sample-count", type=int, default=100)
    prepare_counterfactual.add_argument("--prompts", nargs="+", default=["+!", "+$", "+#", "+^", "+~"])

    collect = subparsers.add_parser("collect", help="Collect per-token DP prediction outputs")
    collect.add_argument("--prediction-dir", type=Path, required=True)
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument("--output-csv", type=Path, required=True)
    collect.add_argument("--model", required=True)
    collect.add_argument("--protocol", required=True)
    collect.add_argument("--prefix-length", type=int, required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Compare score columns with paired AUROC bootstrap")
    bootstrap.add_argument("--scores-csv", type=Path, required=True)
    bootstrap.add_argument("--models", nargs="+", required=True)
    bootstrap.add_argument("--output-json", type=Path, required=True)
    bootstrap.add_argument("--replicates", type=int, default=10_000)
    bootstrap.add_argument("--seed", type=int, default=174)

    compare_order = subparsers.add_parser("compare-order", help="Compare original and reordered scores")
    compare_order.add_argument("--original-csv", type=Path, required=True)
    compare_order.add_argument("--reordered-csv", type=Path, required=True)
    compare_order.add_argument("--output-json", type=Path, required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one scorer's labeled and natural cohorts")
    analyze.add_argument("--bootability-csv", type=Path, required=True)
    analyze.add_argument("--natural-csv", type=Path, required=True)
    analyze.add_argument("--counterfactual-csv", type=Path)
    analyze.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    """Run the selected bootability-comparison subcommand."""
    args = _build_parser().parse_args()
    if args.command == "prepare":
        manifest = prepare_bootability_cohort(
            args.viable_fasta,
            args.nonviable_fasta,
            args.output_dir,
            marker=args.marker,
        )
        print(manifest)
        return

    if args.command == "prepare-order-audit":
        manifest = prepare_order_audit_cohort(
            args.manifest,
            args.raw_fasta,
            args.native_fasta,
            args.output_dir,
            seed=args.seed,
        )
        print(manifest)
        return

    if args.command == "prepare-counterfactual":
        manifest = prepare_prompt_counterfactual_cohort(
            args.natural_manifest,
            args.natural_raw_fasta,
            args.output_dir,
            sample_count=args.sample_count,
            prompts=args.prompts,
        )
        print(manifest)
        return

    if args.command == "prepare-natural":
        manifest = prepare_natural_positive_cohort(
            args.source_fasta,
            args.output_dir,
            sample_count=args.sample_count,
            seed=args.seed,
        )
        print(manifest)
        return

    if args.command == "collect":
        output = collect_predict_scores(
            args.prediction_dir,
            args.manifest,
            args.output_csv,
            model=args.model,
            protocol=args.protocol,
            prefix_length=args.prefix_length,
        )
        print(output)
        return

    if args.command == "compare-order":
        original_rows = list(csv.DictReader(args.original_csv.open()))
        reordered_rows = list(csv.DictReader(args.reordered_csv.open()))
        result = compare_order_audit_scores(original_rows, reordered_rows)
        _atomic_write(args.output_json, json.dumps(result, indent=2) + "\n")
        print(args.output_json)
        return

    if args.command == "analyze":
        bootability_rows = list(csv.DictReader(args.bootability_csv.open()))
        natural_rows = list(csv.DictReader(args.natural_csv.open()))
        counterfactual_rows = list(csv.DictReader(args.counterfactual_csv.open())) if args.counterfactual_csv else []
        result = analyze_model_score_distributions(
            bootability_rows,
            natural_rows,
            counterfactual_rows,
        )
        _atomic_write(args.output_json, json.dumps(result, indent=2) + "\n")
        print(args.output_json)
        return

    rows = list(csv.DictReader(args.scores_csv.open()))
    result = paired_stratified_auc_bootstrap(
        [int(row["label"]) for row in rows],
        {model: [float(row[model]) for row in rows] for model in args.models},
        replicates=args.replicates,
        seed=args.seed,
    )
    _atomic_write(args.output_json, json.dumps(result, indent=2) + "\n")
    print(args.output_json)


if __name__ == "__main__":
    main()
