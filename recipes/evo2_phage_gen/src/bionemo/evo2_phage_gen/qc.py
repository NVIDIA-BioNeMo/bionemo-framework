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
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


DNA_ALPHABET = frozenset("ACGTacgt")
EOS_TEXT_MARKERS = ("<EOS>", "<EOD>", "<STOP>", "EOS", "EOD", "STOP")


@dataclass(frozen=True)
class NucleotideQCConfig:
    """Thresholds for the paper's dependency-light nucleotide QC stage."""

    genome_length_min: int = 4000
    genome_length_max: int = 6000
    gc_content_min: float = 30.0
    gc_content_max: float = 65.0
    homopolymer_min: int = 0
    homopolymer_max: int = 10


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


def calculate_nt_homopolymer_len(sequence: str) -> int:
    """Return the longest A/C/G/T homopolymer length."""
    sequence = sequence.upper()
    longest = 0
    for nucleotide in "ACGT":
        matches = re.findall(f"{nucleotide}+", sequence)
        if matches:
            longest = max(longest, *(len(match) for match in matches))
    return longest


def add_nucleotide_metrics(sequences_df: pd.DataFrame) -> pd.DataFrame:
    """Add nucleotide QC metric columns without filtering rows."""
    df = sequences_df.copy()
    df["valid_nt_chars"] = df["sequence"].map(has_valid_nt_chars)
    df["genome_length"] = df["sequence"].map(len)
    df["gc_content"] = df["sequence"].map(calculate_gc_content)
    df["max_nt_homopolymer_length"] = df["sequence"].map(calculate_nt_homopolymer_len)
    return df


def apply_nucleotide_qc(
    sequences_df: pd.DataFrame,
    config: NucleotideQCConfig = NucleotideQCConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply staged nucleotide QC and return ``(metrics_df, counts_df)``."""
    metrics_df = add_nucleotide_metrics(sequences_df)
    stage_masks = {
        "qc1_initial": pd.Series(True, index=metrics_df.index),
        "valid_nt_chars": metrics_df["valid_nt_chars"],
        "genome_length": metrics_df["genome_length"].between(config.genome_length_min, config.genome_length_max),
        "gc_content": metrics_df["gc_content"].between(config.gc_content_min, config.gc_content_max),
        "nt_homopolymer": metrics_df["max_nt_homopolymer_length"].between(
            config.homopolymer_min, config.homopolymer_max
        ),
    }

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
        ),
        keep_only_up_to_first_eos=not args.keep_all_eos_segments,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
