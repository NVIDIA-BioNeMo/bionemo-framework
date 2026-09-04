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

"""Dependency-light phage genome QC filters.

This module mirrors the first nucleotide-filtering stage of Arc's
``genome_design_filtering_pipeline.py`` and is intentionally small enough to
use online as an RL reward component.
"""

import argparse
import csv
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


DNA_ALPHABET = frozenset("ACGTacgt")
EOS_TEXT_MARKERS = ("<EOS>", "<EOD>", "<STOP>", "EOS", "EOD", "STOP")
RECIPE_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class NucleotideQCConfig:
    """Thresholds for the paper's dependency-light nucleotide QC stage."""

    genome_length_min: int = 4000
    genome_length_max: int = 6000
    gc_content_min: float = 30.0
    gc_content_max: float = 65.0
    homopolymer_min: int = 0
    homopolymer_max: int = 10
    dustmask_filter: bool = False
    dustmasker_bin: str = "dustmasker"
    dustmask_use_external: bool = True
    dustmasker_timeout_s: float = 300.0
    dustmask_window: int = 64
    dustmask_level: float = 20.0
    dustmask_end_window: int = 200
    dustmask_max_end_fraction: float = 0.9


@dataclass(frozen=True)
class DustmaskMetrics:
    """Summary of DUST-style low-complexity masking for one sequence."""

    masked_bases: int
    masked_fraction: float
    left_end_masked_bases: int
    left_end_masked_fraction: float
    right_end_masked_bases: int
    right_end_masked_fraction: float
    max_end_masked_fraction: float
    end_pass: bool


def trim_at_first_eos(sequence: str) -> str:
    """Keep sequence content before textual EOS/EOD/STOP markers."""
    sequence = str(sequence)
    stop_index = len(sequence)
    for idx, char in enumerate(sequence):
        if char.isspace():
            stop_index = min(stop_index, idx)
            break
    upper_sequence = sequence.upper()
    for marker in EOS_TEXT_MARKERS:
        marker_index = upper_sequence.find(marker)
        if marker_index != -1:
            stop_index = min(stop_index, marker_index)
    return sequence[:stop_index]


def prompt_nucleotides(prompt: str) -> str:
    """Return only DNA bases from a prompt that may contain control tokens."""
    return "".join(char for char in str(prompt) if char in DNA_ALPHABET)


def clean_sequence(sequence: str, keep_only_up_to_first_eos: bool = True) -> str:
    """Normalize a generated sequence for nucleotide QC."""
    sequence = str(sequence).replace("\n", "").strip()
    if keep_only_up_to_first_eos:
        sequence = trim_at_first_eos(sequence)
    return sequence


def load_fasta_records(fasta_path: Path, keep_only_up_to_first_eos: bool = True) -> pd.DataFrame:
    """Load FASTA records into a dataframe with Arc-compatible columns."""
    rows = [
        {
            "id_prompt": record.id,
            "description": record.description,
            "sequence": clean_sequence(str(record.seq), keep_only_up_to_first_eos=keep_only_up_to_first_eos),
        }
        for record in SeqIO.parse(str(fasta_path), "fasta")
    ]
    return pd.DataFrame(rows, columns=["id_prompt", "description", "sequence"])


def has_valid_nt_chars(sequence: str) -> bool:
    """Return true when the sequence contains only A/C/G/T characters."""
    return all(char in DNA_ALPHABET for char in sequence)


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC percentage for a nucleotide sequence."""
    if not sequence:
        return 0.0
    seq = sequence.upper()
    return 100.0 * (seq.count("G") + seq.count("C")) / len(seq)


def _dust_window_score(window: str) -> float:
    """Return a DUST-style triplet repetition score for one window."""
    triplet_count = len(window) - 2
    if triplet_count <= 0:
        return 0.0
    counts: dict[str, int] = {}
    for idx in range(triplet_count):
        triplet = window[idx : idx + 3]
        if any(char not in DNA_ALPHABET for char in triplet):
            return 0.0
        counts[triplet.upper()] = counts.get(triplet.upper(), 0) + 1
    pair_sum = sum(count * (count - 1) // 2 for count in counts.values())
    return 10.0 * pair_sum / triplet_count


def dustmask_low_complexity_mask(
    sequence: str,
    *,
    window: int = 64,
    level: float = 20.0,
) -> list[bool]:
    """Return a DUST-style low-complexity mask using triplet repetition scores."""
    seq = str(sequence).upper()
    seq_len = len(seq)
    if seq_len < 3:
        return [False] * seq_len
    window = max(3, min(int(window), seq_len))
    mask = [False] * seq_len
    triplet_count = window - 2
    counts: dict[str, int] = {}
    pair_sum = 0
    invalid_bases = sum(char not in DNA_ALPHABET for char in seq[:window])

    def add_triplet(triplet: str) -> None:
        nonlocal pair_sum
        if any(char not in DNA_ALPHABET for char in triplet):
            return
        count = counts.get(triplet, 0)
        pair_sum += count
        counts[triplet] = count + 1

    def remove_triplet(triplet: str) -> None:
        nonlocal pair_sum
        if any(char not in DNA_ALPHABET for char in triplet):
            return
        count = counts[triplet]
        pair_sum -= count - 1
        if count == 1:
            del counts[triplet]
        else:
            counts[triplet] = count - 1

    for triplet_start in range(triplet_count):
        add_triplet(seq[triplet_start : triplet_start + 3])

    threshold = float(level)
    for start in range(0, seq_len - window + 1):
        if start:
            outgoing_base = seq[start - 1]
            incoming_base = seq[start + window - 1]
            invalid_bases -= outgoing_base not in DNA_ALPHABET
            invalid_bases += incoming_base not in DNA_ALPHABET
            remove_triplet(seq[start - 1 : start + 2])
            add_triplet(seq[start + window - 3 : start + window])
        score = 0.0 if invalid_bases else 10.0 * pair_sum / triplet_count
        if score >= threshold:
            mask[start : start + window] = [True] * window
    return mask


def calculate_dustmask_metrics(
    sequence: str,
    *,
    window: int = 64,
    level: float = 20.0,
    end_window: int = 200,
    max_end_fraction: float = 0.9,
) -> DustmaskMetrics:
    """Calculate DUST-style low-complexity metrics, emphasizing sequence ends."""
    mask = dustmask_low_complexity_mask(sequence, window=window, level=level)
    return _dustmask_metrics_from_mask(
        mask,
        end_window=end_window,
        max_end_fraction=max_end_fraction,
    )


def _dustmask_metrics_from_mask(
    mask: list[bool],
    *,
    end_window: int = 200,
    max_end_fraction: float = 0.9,
) -> DustmaskMetrics:
    """Summarize an already-computed low-complexity mask."""
    seq_len = len(mask)
    if seq_len == 0:
        return DustmaskMetrics(0, 0.0, 0, 0.0, 0, 0.0, 0.0, True)
    end_len = max(1, min(int(end_window), seq_len))
    masked_bases = int(sum(mask))
    left_end_masked_bases = int(sum(mask[:end_len]))
    right_end_masked_bases = int(sum(mask[-end_len:]))
    left_end_fraction = left_end_masked_bases / end_len
    right_end_fraction = right_end_masked_bases / end_len
    max_end_masked_fraction = max(left_end_fraction, right_end_fraction)
    return DustmaskMetrics(
        masked_bases=masked_bases,
        masked_fraction=masked_bases / seq_len,
        left_end_masked_bases=left_end_masked_bases,
        left_end_masked_fraction=left_end_fraction,
        right_end_masked_bases=right_end_masked_bases,
        right_end_masked_fraction=right_end_fraction,
        max_end_masked_fraction=max_end_masked_fraction,
        end_pass=max_end_masked_fraction <= float(max_end_fraction),
    )


def _write_dustmasker_input(sequences_df: pd.DataFrame, output_path: Path) -> list[int]:
    """Write dustmasker input FASTA with stable synthetic IDs and return sequence lengths."""
    lengths: list[int] = []
    with output_path.open("w") as handle:
        for idx, sequence in enumerate(sequences_df["sequence"].astype(str)):
            safe_sequence = _ascii_safe_sequence(sequence).upper()
            lengths.append(len(safe_sequence))
            handle.write(f">seq_{idx}\n")
            for start in range(0, len(safe_sequence), 80):
                handle.write(safe_sequence[start : start + 80] + "\n")
    return lengths


def _parse_dustmasker_interval_output(interval_path: Path, sequence_lengths: list[int]) -> list[list[bool]]:
    """Parse dustmasker interval output into per-sequence boolean masks."""
    intervals_by_index: list[list[tuple[int, int]]] = [[] for _ in sequence_lengths]
    current_index: int | None = None
    if not interval_path.exists():
        return [[False] * seq_len for seq_len in sequence_lengths]

    for raw_line in interval_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        seq_match = re.search(r"seq_(\d+)", line)
        if line.startswith(">"):
            current_index = int(seq_match.group(1)) if seq_match else None
            continue
        seq_index = int(seq_match.group(1)) if seq_match else current_index
        if seq_index is None or seq_index >= len(sequence_lengths):
            continue
        coordinate_text = re.sub(r"seq_\d+", "", line).strip()
        interval_match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", coordinate_text)
        if interval_match is None:
            raise ValueError(f"invalid dustmasker interval for seq_{seq_index}: {coordinate_text!r}")
        start, end = (int(value) for value in interval_match.groups())
        sequence_length = sequence_lengths[seq_index]
        if start < 0 or end < start or end >= sequence_length:
            raise ValueError(
                f"invalid dustmasker interval for seq_{seq_index}: {start} - {end} outside 0 - {sequence_length - 1}"
            )
        intervals_by_index[seq_index].append((start, end))

    masks = [[False] * seq_len for seq_len in sequence_lengths]
    for seq_index, intervals in enumerate(intervals_by_index):
        seq_len = sequence_lengths[seq_index]
        if seq_len <= 0:
            continue
        for start, end in intervals:
            masks[seq_index][start : end + 1] = [True] * (end - start + 1)
    return masks


def _resolve_recipe_tool_path(executable: str) -> str:
    """Resolve recipe-relative tool paths while preserving PATH lookups."""
    path = Path(executable)
    if path.is_absolute() or len(path.parts) <= 1:
        return str(path)
    recipe_path = RECIPE_ROOT / path
    return str(recipe_path if recipe_path.exists() else path)


def _run_dustmasker(
    command: list[str],
    *,
    check: bool,
    capture_output: bool,
    text: bool,
    timeout: float,
) -> None:
    """Run dustmasker with a finite timeout and consistent errors."""
    try:
        subprocess.run(
            command,
            check=check,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"dustmasker execution failed: {error}") from error


def calculate_dustmasker_metrics(
    sequences_df: pd.DataFrame,
    config: NucleotideQCConfig,
) -> list[DustmaskMetrics]:
    """Run NCBI dustmasker once for a dataframe and return per-sequence metrics."""
    with tempfile.TemporaryDirectory(prefix="evo2_phage_dustmasker_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        input_fasta = tmp_dir / "input.fasta"
        interval_output = tmp_dir / "dustmasker.interval"
        sequence_lengths = _write_dustmasker_input(sequences_df, input_fasta)
        _run_dustmasker(
            [
                _resolve_recipe_tool_path(config.dustmasker_bin),
                "-in",
                str(input_fasta),
                "-out",
                str(interval_output),
                "-outfmt",
                "interval",
                "-window",
                str(int(config.dustmask_window)),
                "-level",
                f"{float(config.dustmask_level):g}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=float(config.dustmasker_timeout_s),
        )
        masks = _parse_dustmasker_interval_output(interval_output, sequence_lengths)
    return [
        _dustmask_metrics_from_mask(
            mask,
            end_window=config.dustmask_end_window,
            max_end_fraction=config.dustmask_max_end_fraction,
        )
        for mask in masks
    ]


def calculate_nt_homopolymer_len(sequence: str) -> int:
    """Return the longest A/C/G/T homopolymer length."""
    sequence = sequence.upper()
    longest = 0
    for nucleotide in "ACGT":
        matches = re.findall(f"{nucleotide}+", sequence)
        if matches:
            longest = max(longest, *(len(match) for match in matches))
    return longest


def add_nucleotide_metrics(
    sequences_df: pd.DataFrame,
    config: NucleotideQCConfig = NucleotideQCConfig(),
) -> pd.DataFrame:
    """Add nucleotide QC metric columns without filtering rows."""
    df = sequences_df.copy()
    df["valid_nt_chars"] = df["sequence"].map(has_valid_nt_chars)
    df["genome_length"] = df["sequence"].map(len)
    df["gc_content"] = df["sequence"].map(calculate_gc_content)
    df["max_nt_homopolymer_length"] = df["sequence"].map(calculate_nt_homopolymer_len)
    if not config.dustmask_filter:
        dust_metrics = [DustmaskMetrics(0, 0.0, 0, 0.0, 0, 0.0, 0.0, True) for _ in df["sequence"]]
    elif config.dustmask_use_external:
        dust_metrics = calculate_dustmasker_metrics(df, config)
    else:
        dust_metrics = [
            calculate_dustmask_metrics(
                sequence,
                window=config.dustmask_window,
                level=config.dustmask_level,
                end_window=config.dustmask_end_window,
                max_end_fraction=config.dustmask_max_end_fraction,
            )
            for sequence in df["sequence"]
        ]
    df["dustmask_masked_bases"] = [metrics.masked_bases for metrics in dust_metrics]
    df["dustmask_masked_fraction"] = [metrics.masked_fraction for metrics in dust_metrics]
    df["dustmask_left_end_masked_bases"] = [metrics.left_end_masked_bases for metrics in dust_metrics]
    df["dustmask_left_end_masked_fraction"] = [metrics.left_end_masked_fraction for metrics in dust_metrics]
    df["dustmask_right_end_masked_bases"] = [metrics.right_end_masked_bases for metrics in dust_metrics]
    df["dustmask_right_end_masked_fraction"] = [metrics.right_end_masked_fraction for metrics in dust_metrics]
    df["dustmask_max_end_masked_fraction"] = [metrics.max_end_masked_fraction for metrics in dust_metrics]
    df["dustmask_end_pass"] = [metrics.end_pass for metrics in dust_metrics]
    return df


def apply_nucleotide_qc(
    sequences_df: pd.DataFrame,
    config: NucleotideQCConfig = NucleotideQCConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply staged nucleotide QC and return ``(metrics_df, counts_df)``."""
    metrics_df = add_nucleotide_metrics(sequences_df, config=config)
    stage_masks = {
        "qc1_initial": pd.Series(True, index=metrics_df.index),
        "valid_nt_chars": metrics_df["valid_nt_chars"],
        "genome_length": metrics_df["genome_length"].between(config.genome_length_min, config.genome_length_max),
        "gc_content": metrics_df["gc_content"].between(config.gc_content_min, config.gc_content_max),
        "nt_homopolymer": metrics_df["max_nt_homopolymer_length"].between(
            config.homopolymer_min, config.homopolymer_max
        ),
    }
    if config.dustmask_filter:
        stage_masks["dustmask_end"] = metrics_df["dustmask_end_pass"]

    current = pd.Series(True, index=metrics_df.index)
    count_rows = []
    for stage, mask in stage_masks.items():
        current &= mask
        count_rows.append({"stage": stage, "num_sequences": int(current.sum())})

    filtered_df = metrics_df[current].reset_index(drop=True)
    counts_df = pd.DataFrame(count_rows)
    return filtered_df, counts_df


def save_fasta(sequences_df: pd.DataFrame, output_path: Path) -> None:
    """Write dataframe rows with ``id_prompt`` and ``sequence`` columns to FASTA."""
    records = [
        SeqRecord(Seq(_ascii_safe_sequence(row.sequence)), id=str(row.id_prompt), description="")
        for row in sequences_df[["id_prompt", "sequence"]].itertuples(index=False)
    ]
    SeqIO.write(records, str(output_path), "fasta")


def _ascii_safe_sequence(sequence: str) -> str:
    """Replace FASTA-unsafe/generated tokens so QC rejects them instead of corrupting records."""
    return "".join(char if char in DNA_ALPHABET else "N" for char in str(sequence))


def run_nucleotide_qc(
    input_fasta: Path,
    output_dir: Path,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    keep_only_up_to_first_eos: bool = True,
) -> dict[str, Path]:
    """Run the dependency-light nucleotide QC stage and write Arc-style outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_df = load_fasta_records(input_fasta, keep_only_up_to_first_eos=keep_only_up_to_first_eos)
    filtered_df, counts_df = apply_nucleotide_qc(initial_df, config=config)

    outputs = {
        "initial_csv": output_dir / "qc1_initial_seqs.csv",
        "initial_fasta": output_dir / "qc1_initial_seqs.fasta",
        "nucleotide_counts_csv": output_dir / "qc2_nt_filter_counts.csv",
        "nucleotide_csv": output_dir / "qc2_nt_filter_seqs.csv",
        "nucleotide_fasta": output_dir / "qc2_nt_filter_seqs.fasta",
    }
    initial_df.to_csv(outputs["initial_csv"], index=False, quoting=csv.QUOTE_MINIMAL)
    save_fasta(initial_df, outputs["initial_fasta"])
    counts_df.to_csv(outputs["nucleotide_counts_csv"], index=False)
    filtered_df.to_csv(outputs["nucleotide_csv"], index=False, quoting=csv.QUOTE_MINIMAL)
    save_fasta(filtered_df, outputs["nucleotide_fasta"])
    return outputs


def main() -> None:
    """CLI entry point for dependency-light nucleotide QC."""
    parser = argparse.ArgumentParser(description="Run Evo2 phage nucleotide QC filters")
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--genome-length-min", type=int, default=4000)
    parser.add_argument("--genome-length-max", type=int, default=6000)
    parser.add_argument("--gc-content-min", type=float, default=30.0)
    parser.add_argument("--gc-content-max", type=float, default=65.0)
    parser.add_argument("--homopolymer-max", type=int, default=10)
    parser.add_argument("--dustmask-filter", action="store_true")
    parser.add_argument("--dustmasker-bin", default="dustmasker")
    parser.add_argument(
        "--dustmask-use-fallback",
        action="store_true",
        help="Use the internal DUST-style scorer instead of the NCBI dustmasker binary.",
    )
    parser.add_argument("--dustmask-window", type=int, default=64)
    parser.add_argument("--dustmask-level", type=float, default=20.0)
    parser.add_argument("--dustmask-end-window", type=int, default=200)
    parser.add_argument("--dustmask-max-end-fraction", type=float, default=0.9)
    parser.add_argument("--keep-all-eos-segments", action="store_true")
    args = parser.parse_args()

    outputs = run_nucleotide_qc(
        input_fasta=args.input_fasta,
        output_dir=args.output_dir,
        config=NucleotideQCConfig(
            genome_length_min=args.genome_length_min,
            genome_length_max=args.genome_length_max,
            gc_content_min=args.gc_content_min,
            gc_content_max=args.gc_content_max,
            homopolymer_max=args.homopolymer_max,
            dustmask_filter=args.dustmask_filter,
            dustmasker_bin=args.dustmasker_bin,
            dustmask_use_external=not args.dustmask_use_fallback,
            dustmask_window=args.dustmask_window,
            dustmask_level=args.dustmask_level,
            dustmask_end_window=args.dustmask_end_window,
            dustmask_max_end_fraction=args.dustmask_max_end_fraction,
        ),
        keep_only_up_to_first_eos=not args.keep_all_eos_segments,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
