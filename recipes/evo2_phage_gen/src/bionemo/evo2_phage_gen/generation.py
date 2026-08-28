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

"""Utilities for paper-style Evo2 Microviridae generation replication."""

import argparse
import csv
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

from Bio import SeqIO

from bionemo.evo2_phage_gen.qc import prompt_nucleotides as _prompt_nucleotides
from bionemo.evo2_phage_gen.qc import trim_at_first_eos


PHIX174_REFERENCE_START = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAA"
DEFAULT_PROMPT_LENGTHS = tuple(range(1, 12))
DEFAULT_PROMPT_PREFIX = "+~"
PAPER_USEFUL_RL_PROMPT_LENGTHS = (4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 11)
PAPER_USEFUL_RL_VALIDATION_PROMPT_LENGTH = 10
PAPER_USEFUL_RL_VALIDATION_RECORDS = 96
RECIPE_ROOT = Path(__file__).resolve().parents[3]


def phix174_prompts(
    reference_start: str = PHIX174_REFERENCE_START,
    prompt_lengths: Sequence[int] = DEFAULT_PROMPT_LENGTHS,
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
) -> dict[int, str]:
    """Return paper-style PhiX174-start prompts keyed by nucleotide prompt length."""
    reference_start = reference_start.strip().upper()
    prompts: dict[int, str] = {}
    for prompt_len in prompt_lengths:
        if prompt_len < 0:
            raise ValueError(f"Prompt length must be non-negative, got {prompt_len}")
        if prompt_len > len(reference_start):
            raise ValueError(f"Prompt length {prompt_len} exceeds reference length {len(reference_start)}")
        prompts[int(prompt_len)] = f"{prompt_prefix}{reference_start[:prompt_len]}"
    return prompts


def _openai_prompt_record(prompt: str) -> dict[str, list[dict[str, str]]]:
    """Return the OpenAI-style prompt record consumed by the phage RL processor."""
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]}


def write_openai_prompt_jsonl(path: Path, prompts: Sequence[str]) -> Path:
    """Write OpenAI-style user-message prompts to one JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for prompt in prompts:
            handle.write(json.dumps(_openai_prompt_record(prompt)) + "\n")
    return path


def write_rl_prompt_bank(
    path: Path,
    *,
    prompt_lengths: Sequence[int],
    repeats_per_length: int | None = None,
    num_records: int | None = None,
    reference_start: str = PHIX174_REFERENCE_START,
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
    id_prefix: str = "rl-prompt",
    grouped: bool = False,
) -> Path:
    """Write an interleaved, near-equal prompt mixture for NeMo-RL."""
    if (repeats_per_length is None) == (num_records is None):
        raise ValueError("exactly one of repeats_per_length and num_records must be provided")
    if not prompt_lengths:
        raise ValueError("prompt_lengths must not be empty")
    if repeats_per_length is not None and repeats_per_length <= 0:
        raise ValueError("repeats_per_length must be positive")
    if num_records is not None and num_records <= 0:
        raise ValueError("num_records must be positive")
    prompts_by_length = phix174_prompts(reference_start, prompt_lengths, prompt_prefix=prompt_prefix)
    if repeats_per_length is not None:
        if grouped:
            order = [length for length in prompt_lengths for _ in range(repeats_per_length)]
        else:
            order = [length for _ in range(repeats_per_length) for length in prompt_lengths]
    elif grouped:
        base_count, extra_count = divmod(num_records, len(prompt_lengths))
        order = [
            length for index, length in enumerate(prompt_lengths) for _ in range(base_count + (index < extra_count))
        ]
    else:
        order = [prompt_lengths[index % len(prompt_lengths)] for index in range(num_records)]
    counters = dict.fromkeys(prompt_lengths, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for length in order:
            index = counters[length]
            counters[length] += 1
            record = {
                "id": f"{id_prefix}-p{length}-{index:04d}",
                **_openai_prompt_record(prompts_by_length[length]),
            }
            handle.write(json.dumps(record) + "\n")
    return path


def write_inference_prompt_shards(
    jsonl_paths: Sequence[Path],
    output_dir: Path,
    *,
    num_records: int,
    num_shards: int,
) -> list[Path]:
    """Interleave prompt strata and split an exact-size inference bank into balanced shards."""
    if not jsonl_paths:
        raise ValueError("at least one input JSONL path is required")
    if num_records <= 0:
        raise ValueError("num_records must be positive")
    if num_shards <= 0 or num_shards > num_records:
        raise ValueError("num_shards must be between one and num_records")

    groups = [
        [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] for path in jsonl_paths
    ]
    if any(not group for group in groups):
        raise ValueError("every input JSONL path must contain at least one prompt")

    records: list[dict] = []
    for record_index in range(max(len(group) for group in groups)):
        for group in groups:
            if record_index < len(group):
                records.append(group[record_index])
                if len(records) == num_records:
                    break
        if len(records) == num_records:
            break
    if len(records) != num_records:
        raise ValueError(f"requested {num_records} records but the inputs contain only {len(records)}")
    record_ids = [record.get("id") for record in records]
    if any(record_id is None for record_id in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("input prompt IDs must be present and unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for shard_index in range(num_shards):
        start = shard_index * num_records // num_shards
        end = (shard_index + 1) * num_records // num_shards
        output_path = output_dir / f"dp{shard_index}.jsonl"
        output_path.write_text("".join(json.dumps(record) + "\n" for record in records[start:end]))
        paths.append(output_path)
    return paths


def ensure_paper_useful_rl_prompt_files(
    data_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Materialize deterministic paper-useful RL prompt JSONL files when absent."""
    data_dir = data_dir or RECIPE_ROOT / "data"
    prompts_by_length = phix174_prompts(prompt_lengths=sorted(set(PAPER_USEFUL_RL_PROMPT_LENGTHS)))
    train_path = data_dir / "phage_prompts_paper_useful_rl.jsonl"
    validation_path = data_dir / "phage_prompts_paper_useful_rl_validation_prompt10_96.jsonl"

    if overwrite or not train_path.exists():
        train_prompts = [prompts_by_length[prompt_len] for prompt_len in PAPER_USEFUL_RL_PROMPT_LENGTHS]
        write_openai_prompt_jsonl(train_path, train_prompts)
    if overwrite or not validation_path.exists():
        validation_prompt = prompts_by_length[PAPER_USEFUL_RL_VALIDATION_PROMPT_LENGTH]
        write_openai_prompt_jsonl(
            validation_path,
            [validation_prompt] * PAPER_USEFUL_RL_VALIDATION_RECORDS,
        )
    return {"train": train_path, "validation": validation_path}


def write_prompt_sweep_jsonl(
    output_dir: Path,
    *,
    reference_start: str = PHIX174_REFERENCE_START,
    prompt_lengths: Sequence[int] = DEFAULT_PROMPT_LENGTHS,
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
    num_prompts: int = 1000,
    id_prefix: str = "phix174",
) -> list[Path]:
    """Write repeated prompt JSONL files for ``infer_evo2 --prompt-file``."""
    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be positive, got {num_prompts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for prompt_len, prompt in phix174_prompts(reference_start, prompt_lengths, prompt_prefix=prompt_prefix).items():
        output_path = output_dir / f"{id_prefix}_prompt{prompt_len}_{num_prompts}.jsonl"
        with output_path.open("w") as f:
            for idx in range(num_prompts):
                record = {"id": f"{id_prefix}_prompt{prompt_len}_{idx:04d}", "prompt": prompt}
                f.write(json.dumps(record) + "\n")
        written_paths.append(output_path)
    return written_paths


def _sequence_before_eos(sequence: str) -> str:
    """Return generated sequence content before textual EOS markers."""
    return trim_at_first_eos(str(sequence).replace("\n", "").strip())


def _wrap_fasta_sequence(sequence: str, width: int = 80) -> str:
    """Wrap a FASTA sequence to a fixed line width."""
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def infer_jsonl_to_fasta(
    jsonl_paths: Iterable[Path],
    output_fasta: Path,
    *,
    include_source_stem: bool = True,
) -> Path:
    """Convert ``infer_evo2`` JSONL records to FASTA by prepending prompt nucleotides to completion."""
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with output_fasta.open("w") as fasta:
        for jsonl_path in sorted(Path(path) for path in jsonl_paths):
            with jsonl_path.open() as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record_id = str(record.get("id") or f"{jsonl_path.stem}_{line_idx:06d}")
                    header = f"{record_id}|{jsonl_path.stem}" if include_source_stem else record_id
                    prompt = _prompt_nucleotides(_sequence_before_eos(record.get("prompt", "")))
                    completion = _sequence_before_eos(record.get("completion", ""))
                    sequence = f"{prompt}{completion}".upper()
                    fasta.write(f">{header}\n{_wrap_fasta_sequence(sequence)}\n")
    return output_fasta


def _fasta_records(path: Path, *, allow_empty: bool = False) -> list[tuple[str, str]]:
    records = [(record.id, str(record.seq).upper()) for record in SeqIO.parse(path, "fasta")]
    if not records:
        if allow_empty:
            return []
        raise ValueError(f"no complete FASTA records found: {path}")
    if any(not record_id or not sequence for record_id, sequence in records):
        raise ValueError(f"no complete FASTA records found: {path}")
    ids = [record_id for record_id, _ in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate FASTA identifiers in {path}")
    return records


def write_sft_likelihood_fasta(
    source_fasta: Path,
    output_fasta: Path,
    *,
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
) -> Path:
    """Prepare generated genomes for conditional likelihood scoring by the SFT model."""
    if not prompt_prefix:
        raise ValueError("prompt_prefix must not be empty")
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    output_fasta.write_text(
        "".join(
            f">{record_id}\n{_wrap_fasta_sequence(prompt_prefix + sequence)}\n"
            for record_id, sequence in _fasta_records(source_fasta)
        )
    )
    return output_fasta


def collect_sft_likelihoods(
    prediction_dir: Path,
    source_fasta: Path,
    output_csv: Path,
    *,
    prefix_length: int = len(DEFAULT_PROMPT_PREFIX),
) -> Path:
    """Collect DP prediction shards and rank every genome by mean nucleotide log probability."""
    import torch

    if prefix_length < 1:
        raise ValueError("prefix_length must be positive for an SFT-conditioned score")
    records = _fasta_records(source_fasta)
    sequence_by_id = dict(records)
    index_map = {
        name: int(index) for name, index in json.loads((prediction_dir / "seq_idx_map.json").read_text()).items()
    }
    id_by_index = {index: name for name, index in index_map.items()}
    if len(id_by_index) != len(index_map):
        raise ValueError("seq_idx_map contains duplicate indices")

    predictions: dict[str, tuple[list[float], list[bool]]] = {}
    prediction_files = sorted(prediction_dir.rglob("predictions__rank_*__dp_rank_*.pt"))
    if not prediction_files:
        raise ValueError(f"no prediction files found: {prediction_dir}")
    for prediction_file in prediction_files:
        payload = torch.load(prediction_file, map_location="cpu", weights_only=True)
        required = {"seq_idx", "log_probs_seqs", "loss_mask"}
        if not required <= payload.keys():
            raise ValueError(f"missing per-token fields in {prediction_file}")
        for sequence_index, log_probs, loss_mask in zip(
            payload["seq_idx"], payload["log_probs_seqs"], payload["loss_mask"], strict=True
        ):
            index = int(sequence_index.item())
            if index not in id_by_index:
                raise ValueError(f"unknown sequence index {index} in {prediction_file}")
            record_id = id_by_index[index]
            if record_id in predictions:
                raise ValueError(f"duplicate prediction for {record_id}")
            predictions[record_id] = (
                log_probs.detach().cpu().tolist(),
                loss_mask.detach().cpu().tolist(),
            )

    expected_ids = set(sequence_by_id)
    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))
        extra = sorted(set(predictions) - expected_ids)
        raise ValueError(f"prediction/FASTA mismatch: missing={missing}, extra={extra}")

    rows = []
    for record_id, sequence in records:
        log_probs, loss_mask = predictions[record_id]
        expected_valid = len(sequence) + prefix_length - 1
        if (
            expected_valid <= 0
            or len(loss_mask) < expected_valid
            or not all(loss_mask[:expected_valid])
            or any(loss_mask[expected_valid:])
        ):
            raise ValueError(f"unexpected loss mask for {record_id}")
        start = prefix_length - 1
        nucleotide_log_probs = [float(value) for value in log_probs[start : start + len(sequence)]]
        if len(nucleotide_log_probs) != len(sequence) or not all(
            math.isfinite(value) for value in nucleotide_log_probs
        ):
            raise ValueError(f"incomplete nucleotide log probabilities for {record_id}")
        total = sum(nucleotide_log_probs)
        rows.append(
            {
                "record_id": record_id,
                "length_nt": len(sequence),
                "scored_nucleotides": len(nucleotide_log_probs),
                "total_log_probability": total,
                "mean_log_probability_per_nucleotide": total / len(nucleotide_log_probs),
            }
        )

    rows.sort(key=lambda row: (-row["mean_log_probability_per_nucleotide"], row["record_id"]))
    for rank, row in enumerate(rows, start=1):
        row["likelihood_rank"] = rank
    fieldnames = (
        "likelihood_rank",
        "record_id",
        "length_nt",
        "scored_nucleotides",
        "total_log_probability",
        "mean_log_probability_per_nucleotide",
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def finalize_ranked_rollout(
    generated_fasta: Path,
    safety_manifest: Path,
    target_fasta: Path,
    likelihood_csv: Path,
    output_json: Path,
    accepted_fasta: Path,
    summary_path: Path,
    *,
    model_checkpoint: str,
) -> Path:
    """Join likelihood and QC evidence, then rank accepted candidates without making a viability claim."""
    from scipy.stats import spearmanr

    from bionemo.evo2_phage_gen.calibration_novelty import canonical_circular_sequence

    generated = _fasta_records(generated_fasta)
    sequence_by_id = dict(generated)
    generated_order = {record_id: index for index, (record_id, _) in enumerate(generated)}
    safety_payload = json.loads(safety_manifest.read_text())
    safety_by_id = {str(row["record_id"]): str(row["state"]) for row in safety_payload["records"]}
    target_sequences = {
        canonical_circular_sequence(sequence) for _, sequence in _fasta_records(target_fasta, allow_empty=True)
    }
    with likelihood_csv.open() as handle:
        score_rows = list(csv.DictReader(handle))
    score_ids = [row["record_id"] for row in score_rows]
    if len(set(score_ids)) != len(score_ids) or set(score_ids) != set(sequence_by_id):
        raise ValueError("likelihood table must contain each generated FASTA identifier exactly once")
    score_rows.sort(key=lambda row: (-float(row["mean_log_probability_per_nucleotide"]), row["record_id"]))

    lengths = [len(sequence) for _, sequence in generated]
    score_by_id = {row["record_id"]: float(row["mean_log_probability_per_nucleotide"]) for row in score_rows}
    scores_in_generation_order = [score_by_id[record_id] for record_id, _ in generated]
    informative_scores = len(set(scores_in_generation_order)) > 1
    if len(set(lengths)) > 1 and informative_scores:
        correlation = spearmanr(lengths, scores_in_generation_order)
        spearman_rho = float(correlation.statistic)
        spearman_pvalue = float(correlation.pvalue)
    else:
        spearman_rho = None
        spearman_pvalue = None
    strong_length_association = spearman_rho is not None and abs(spearman_rho) >= 0.5
    apply_likelihood_order = informative_scores and not strong_length_association
    candidate_order = (
        score_rows if apply_likelihood_order else sorted(score_rows, key=lambda row: generated_order[row["record_id"]])
    )

    accepted_ids: list[str] = []
    accepted_rank_by_id: dict[str, int] = {}
    duplicate_of_by_id: dict[str, str] = {}
    accepted_canonical: dict[str, str] = {}
    for score in candidate_order:
        record_id = score["record_id"]
        canonical = canonical_circular_sequence(sequence_by_id[record_id])
        duplicate_of = accepted_canonical.get(canonical)
        if duplicate_of is not None:
            duplicate_of_by_id[record_id] = duplicate_of
        if safety_by_id.get(record_id) == "PASS" and canonical in target_sequences and duplicate_of is None:
            accepted_ids.append(record_id)
            accepted_rank_by_id[record_id] = len(accepted_ids)
            accepted_canonical[canonical] = record_id

    report_rows = []
    for likelihood_rank, score in enumerate(score_rows, start=1):
        record_id = score["record_id"]
        sequence = sequence_by_id[record_id]
        canonical = canonical_circular_sequence(sequence)
        report_rows.append(
            {
                "likelihood_rank": likelihood_rank,
                "accepted_rank": accepted_rank_by_id.get(record_id),
                "record_id": record_id,
                "sequence": sequence,
                "length_nt": len(sequence),
                "scored_nucleotides": int(score["scored_nucleotides"]),
                "total_log_probability": float(score["total_log_probability"]),
                "mean_log_probability_per_nucleotide": float(score["mean_log_probability_per_nucleotide"]),
                "safety_state": safety_by_id.get(record_id, "NOT_EVALUATED"),
                "target_profile_pass": canonical in target_sequences,
                "accepted": record_id in accepted_rank_by_id,
                "duplicate_of_higher_priority_candidate": duplicate_of_by_id.get(record_id),
            }
        )

    accepted_fasta.parent.mkdir(parents=True, exist_ok=True)
    accepted_fasta.write_text(
        "".join(f">{record_id}\n{_wrap_fasta_sequence(sequence_by_id[record_id])}\n" for record_id in accepted_ids)
    )
    if not informative_scores:
        length_interpretation = "All normalized scores are equal, so they provide no ordering signal and length association is not estimable."
    elif spearman_rho is None:
        length_interpretation = "All designs have the same length, so residual length association is not estimable."
    elif strong_length_association:
        length_interpretation = (
            "Strong residual length association detected; do not use SFT likelihood to order accepted candidates."
        )
    else:
        length_interpretation = "No strong residual length association detected at the stated threshold."
    report = {
        "schema_version": 1,
        "ranking": {
            "model_checkpoint": model_checkpoint,
            "conditioning_prefix": DEFAULT_PROMPT_PREFIX,
            "primary_score": "mean_log_probability_per_nucleotide",
            "order": "descending; higher (less negative) is ranked first",
            "applied_to_accepted_candidate_order": apply_likelihood_order,
            "total_score_field": "total_log_probability",
            "residual_length_association": {
                "method": "Spearman correlation across all generated designs",
                "n": len(generated),
                "score_varies": informative_scores,
                "spearman_rho": spearman_rho,
                "p_value": spearman_pvalue,
                "strong_correlation_threshold_abs_rho": 0.5,
                "strong_correlation": strong_length_association,
                "interpretation": length_interpretation,
            },
            "interpretation": (
                "Within-protocol ranking signal only; not a universal bootability threshold or proof of viability."
            ),
            "evidence": (
                "Black et al. (2026), Quantifying evolutionary novelty and design efficiency in "
                "generative genome design, https://doi.org/10.64898/2026.06.12.731871"
            ),
        },
        "counts": {
            "generated": len(generated),
            "likelihood_scored": len(report_rows),
            "safety_pass": sum(row["safety_state"] == "PASS" for row in report_rows),
            "target_profile_pass": sum(row["target_profile_pass"] for row in report_rows),
            "accepted": len(accepted_ids),
        },
        "records": report_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if apply_likelihood_order:
        ordering_summary = "SFT likelihood ordering applied after the residual length check."
    elif not informative_scores:
        ordering_summary = (
            "SFT likelihood ordering not applied because the normalized scores do not discriminate designs."
        )
    else:
        ordering_summary = (
            "SFT likelihood ordering not applied because the normalized score remained strongly length-correlated."
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "# PhiX174 run summary\n\n"
        f"- Generated and SFT-likelihood scored: {len(report_rows)}\n"
        f"- Safety-PASS and target-profile candidates: {len(accepted_ids)}\n"
        "- Score: mean nucleotide log probability from the selected pre-RL SFT checkpoint\n"
        f"- Length-bias check: {length_interpretation} {ordering_summary}\n\n"
        "Likelihood is a within-protocol ranking signal, not a universal bootability threshold. "
        "Computational candidates are not evidence of wet-lab viability or safety.\n"
    )
    return output_json


def _resolve_jsonl_inputs(input_jsonl: Sequence[Path], input_dir: Path | None, glob_pattern: str) -> list[Path]:
    """Resolve explicit JSONL files plus optional directory glob inputs."""
    paths = [Path(path) for path in input_jsonl]
    if input_dir is not None:
        paths.extend(sorted(Path(input_dir).glob(glob_pattern)))
    if not paths:
        raise ValueError("Provide at least one --input-jsonl or --input-dir")
    return paths


def main() -> None:
    """CLI entry point for generation replication helpers."""
    parser = argparse.ArgumentParser(description="Evo2 phage generation replication helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rl_prompt_parser = subparsers.add_parser(
        "prepare-rl-prompts",
        help="Materialize deterministic paper-useful RL prompt JSONL files",
    )
    rl_prompt_parser.add_argument("--data-dir", type=Path, default=RECIPE_ROOT / "data")
    rl_prompt_parser.add_argument("--overwrite", action="store_true")

    rl_bank_parser = subparsers.add_parser(
        "write-rl-prompts",
        help="Write an interleaved near-equal PhiX174 prompt bank for NeMo-RL",
    )
    rl_bank_parser.add_argument("--output", type=Path, required=True)
    rl_bank_parser.add_argument("--prompt-lengths", type=int, nargs="+", required=True)
    rl_bank_size = rl_bank_parser.add_mutually_exclusive_group(required=True)
    rl_bank_size.add_argument("--repeats-per-length", type=int)
    rl_bank_size.add_argument("--num-records", type=int)
    rl_bank_parser.add_argument("--reference-start", type=str, default=PHIX174_REFERENCE_START)
    rl_bank_parser.add_argument("--prompt-prefix", type=str, default=DEFAULT_PROMPT_PREFIX)
    rl_bank_parser.add_argument("--id-prefix", type=str, default="rl-prompt")
    rl_bank_parser.add_argument("--grouped", action="store_true")

    prompt_parser = subparsers.add_parser("write-prompts", help="Write PhiX174 prompt sweep JSONL files")
    prompt_parser.add_argument("--output-dir", type=Path, required=True)
    prompt_parser.add_argument("--reference-start", type=str, default=PHIX174_REFERENCE_START)
    prompt_parser.add_argument("--prompt-lengths", type=int, nargs="+", default=list(DEFAULT_PROMPT_LENGTHS))
    prompt_parser.add_argument("--prompt-prefix", type=str, default=DEFAULT_PROMPT_PREFIX)
    prompt_parser.add_argument("--num-prompts", type=int, default=1000)
    prompt_parser.add_argument("--id-prefix", type=str, default="phix174")

    shard_parser = subparsers.add_parser(
        "write-inference-shards",
        help="Interleave prompt-length files into exact-size inference shards",
    )
    shard_parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    shard_parser.add_argument("--output-dir", type=Path, required=True)
    shard_parser.add_argument("--num-records", type=int, required=True)
    shard_parser.add_argument("--num-shards", type=int, required=True)

    fasta_parser = subparsers.add_parser("jsonl-to-fasta", help="Convert infer_evo2 JSONL outputs to FASTA")
    fasta_parser.add_argument("--input-jsonl", type=Path, nargs="*", default=[])
    fasta_parser.add_argument("--input-dir", type=Path, default=None)
    fasta_parser.add_argument("--glob", type=str, default="*.jsonl")
    fasta_parser.add_argument("--output-fasta", type=Path, required=True)
    fasta_parser.add_argument("--no-source-stem", action="store_true")

    likelihood_fasta_parser = subparsers.add_parser(
        "prepare-sft-likelihood",
        help="Add the SFT control prefix before whole-genome likelihood scoring",
    )
    likelihood_fasta_parser.add_argument("--input-fasta", type=Path, required=True)
    likelihood_fasta_parser.add_argument("--output-fasta", type=Path, required=True)
    likelihood_fasta_parser.add_argument("--prompt-prefix", default=DEFAULT_PROMPT_PREFIX)

    likelihood_parser = subparsers.add_parser(
        "collect-sft-likelihood",
        help="Collect per-token prediction shards and rank every generated genome",
    )
    likelihood_parser.add_argument("--prediction-dir", type=Path, required=True)
    likelihood_parser.add_argument("--source-fasta", type=Path, required=True)
    likelihood_parser.add_argument("--output-csv", type=Path, required=True)
    likelihood_parser.add_argument("--prefix-length", type=int, default=len(DEFAULT_PROMPT_PREFIX))

    deduplication_parser = subparsers.add_parser(
        "deduplicate-fasta",
        help="Remove exact, circular, and reverse-complement biological duplicates",
    )
    deduplication_parser.add_argument("--input-fasta", type=Path, required=True)
    deduplication_parser.add_argument("--output-fasta", type=Path, required=True)
    deduplication_parser.add_argument("--mapping-csv", type=Path, required=True)
    deduplication_parser.add_argument("--report", type=Path, required=True)

    arc_summary_parser = subparsers.add_parser(
        "summarize-arc-screen",
        help="Validate an Arc hard-QC branch and record its waterfall",
    )
    arc_summary_parser.add_argument("--config", type=Path, required=True)
    arc_summary_parser.add_argument("--input-fasta", type=Path, required=True)
    arc_summary_parser.add_argument("--output-json", type=Path, required=True)
    arc_summary_parser.add_argument("--expected-filter7", choices=("true", "false"), required=True)

    hard_qc_parser = subparsers.add_parser(
        "select-hard-qc-passers",
        help="Intersect safety-PASS representatives with target-profile hard-QC passers",
    )
    hard_qc_parser.add_argument("--representative-fasta", type=Path, required=True)
    hard_qc_parser.add_argument("--safety-input-fasta", type=Path)
    hard_qc_parser.add_argument("--safety-manifest", type=Path, required=True)
    hard_qc_parser.add_argument("--target-fasta", type=Path, required=True)
    hard_qc_parser.add_argument("--output-fasta", type=Path, required=True)
    hard_qc_parser.add_argument("--report", type=Path, required=True)

    clustering_parser = subparsers.add_parser(
        "cluster-post-qc",
        help="Cluster hard-QC passers with the pinned final-order MMseqs contract",
    )
    clustering_parser.add_argument("--input-fasta", type=Path, required=True)
    clustering_parser.add_argument("--output-fasta", type=Path, required=True)
    clustering_parser.add_argument("--memberships-csv", type=Path, required=True)
    clustering_parser.add_argument("--report", type=Path, required=True)
    clustering_parser.add_argument("--work-dir", type=Path, required=True)
    clustering_parser.add_argument("--mmseqs-bin", default="mmseqs")
    clustering_parser.add_argument("--threads", type=int, default=16)

    final_parser = subparsers.add_parser(
        "finalize-rollout",
        help="Join likelihood and final QC evidence into the ranked rollout report",
    )
    final_parser.add_argument("--generated-fasta", type=Path, required=True)
    final_parser.add_argument("--safety-input-fasta", type=Path)
    final_parser.add_argument("--safety-manifest", type=Path, required=True)
    final_parser.add_argument("--target-fasta", type=Path, required=True)
    final_parser.add_argument("--likelihood-csv", type=Path, required=True)
    final_parser.add_argument("--output-json", type=Path, required=True)
    final_parser.add_argument("--accepted-fasta", type=Path, required=True)
    final_parser.add_argument("--summary", type=Path, required=True)
    final_parser.add_argument("--model-checkpoint", required=True)
    final_parser.add_argument("--rl-checkpoint")
    final_parser.add_argument("--deduplication-mapping", type=Path)
    final_parser.add_argument("--diagnostic-fasta", type=Path)
    final_parser.add_argument("--cluster-representatives-fasta", type=Path)
    final_parser.add_argument("--cluster-memberships", type=Path)
    final_parser.add_argument("--sampling-selection", type=Path)
    final_parser.add_argument("--deduplication-report", type=Path)
    final_parser.add_argument("--hard-qc-report", type=Path)
    final_parser.add_argument("--target-report", type=Path)
    final_parser.add_argument("--diagnostic-report", type=Path)
    final_parser.add_argument("--clustering-report", type=Path)
    final_parser.add_argument("--run-log", type=Path)

    args = parser.parse_args()
    if args.command == "prepare-rl-prompts":
        for path in ensure_paper_useful_rl_prompt_files(args.data_dir, overwrite=args.overwrite).values():
            print(path)
    elif args.command == "write-rl-prompts":
        print(
            write_rl_prompt_bank(
                args.output,
                prompt_lengths=args.prompt_lengths,
                repeats_per_length=args.repeats_per_length,
                num_records=args.num_records,
                reference_start=args.reference_start,
                prompt_prefix=args.prompt_prefix,
                id_prefix=args.id_prefix,
                grouped=args.grouped,
            )
        )
    elif args.command == "write-prompts":
        for path in write_prompt_sweep_jsonl(
            args.output_dir,
            reference_start=args.reference_start,
            prompt_lengths=args.prompt_lengths,
            prompt_prefix=args.prompt_prefix,
            num_prompts=args.num_prompts,
            id_prefix=args.id_prefix,
        ):
            print(path)
    elif args.command == "write-inference-shards":
        for path in write_inference_prompt_shards(
            args.input_jsonl,
            args.output_dir,
            num_records=args.num_records,
            num_shards=args.num_shards,
        ):
            print(path)
    elif args.command == "jsonl-to-fasta":
        paths = _resolve_jsonl_inputs(args.input_jsonl, args.input_dir, args.glob)
        print(infer_jsonl_to_fasta(paths, args.output_fasta, include_source_stem=not args.no_source_stem))
    elif args.command == "prepare-sft-likelihood":
        print(
            write_sft_likelihood_fasta(
                args.input_fasta,
                args.output_fasta,
                prompt_prefix=args.prompt_prefix,
            )
        )
    elif args.command == "collect-sft-likelihood":
        print(
            collect_sft_likelihoods(
                args.prediction_dir,
                args.source_fasta,
                args.output_csv,
                prefix_length=args.prefix_length,
            )
        )
    elif args.command == "deduplicate-fasta":
        from bionemo.evo2_phage_gen.rollout_evidence import deduplicate_fasta

        print(deduplicate_fasta(args.input_fasta, args.output_fasta, args.mapping_csv, args.report))
    elif args.command == "summarize-arc-screen":
        from bionemo.evo2_phage_gen.rollout_evidence import summarize_arc_screen

        print(
            summarize_arc_screen(
                args.config,
                args.input_fasta,
                args.output_json,
                expected_filter7=args.expected_filter7 == "true",
            )
        )
    elif args.command == "select-hard-qc-passers":
        from bionemo.evo2_phage_gen.rollout_evidence import select_hard_qc_passers

        print(
            select_hard_qc_passers(
                args.representative_fasta,
                args.safety_manifest,
                args.target_fasta,
                args.output_fasta,
                args.report,
                safety_input_fasta=args.safety_input_fasta,
            )
        )
    elif args.command == "cluster-post-qc":
        from bionemo.evo2_phage_gen.rollout_evidence import cluster_post_qc_fasta

        print(
            cluster_post_qc_fasta(
                args.input_fasta,
                args.output_fasta,
                args.memberships_csv,
                args.report,
                work_dir=args.work_dir,
                mmseqs_bin=args.mmseqs_bin,
                threads=args.threads,
            )
        )
    elif args.command == "finalize-rollout":
        if args.deduplication_mapping is None:
            print(
                finalize_ranked_rollout(
                    args.generated_fasta,
                    args.safety_manifest,
                    args.target_fasta,
                    args.likelihood_csv,
                    args.output_json,
                    args.accepted_fasta,
                    args.summary,
                    model_checkpoint=args.model_checkpoint,
                )
            )
        else:
            from bionemo.evo2_phage_gen.rollout_evidence import finalize_rollout_report

            required = {
                "--rl-checkpoint": args.rl_checkpoint,
                "--diagnostic-fasta": args.diagnostic_fasta,
                "--cluster-representatives-fasta": args.cluster_representatives_fasta,
                "--cluster-memberships": args.cluster_memberships,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                parser.error(f"enhanced finalization requires: {', '.join(missing)}")
            print(
                finalize_rollout_report(
                    args.generated_fasta,
                    args.deduplication_mapping,
                    args.safety_manifest,
                    args.target_fasta,
                    args.diagnostic_fasta,
                    args.likelihood_csv,
                    args.cluster_representatives_fasta,
                    args.cluster_memberships,
                    args.output_json,
                    args.accepted_fasta,
                    args.summary,
                    model_checkpoint=args.model_checkpoint,
                    rl_checkpoint=args.rl_checkpoint,
                    sampling_selection=args.sampling_selection,
                    deduplication_report=args.deduplication_report,
                    hard_qc_report=args.hard_qc_report,
                    target_report=args.target_report,
                    diagnostic_report=args.diagnostic_report,
                    clustering_report=args.clustering_report,
                    run_log=args.run_log,
                    safety_input_fasta=args.safety_input_fasta,
                )
            )


if __name__ == "__main__":
    main()
