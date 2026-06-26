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

"""Pluggable reward functions for Evo2 phage design RL."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bionemo.evo2_phage_gen.qc import NucleotideQCConfig, add_nucleotide_metrics, load_fasta_records


@dataclass(frozen=True)
class RewardWeights:
    """Weights for dependency-light online phage-design reward components."""

    valid_nt_chars: float = 1.0
    genome_length: float = 1.0
    gc_content: float = 1.0
    nt_homopolymer: float = 1.0


def _interval_score(value: float, lower: float, upper: float) -> float:
    """Return 1 inside an interval and a smooth bounded penalty outside it."""
    if lower <= value <= upper:
        return 1.0
    distance = lower - value if value < lower else value - upper
    width = max(upper - lower, 1.0)
    return max(0.0, 1.0 - distance / width)


def score_nucleotide_metrics(
    sequences_df: pd.DataFrame,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
) -> pd.DataFrame:
    """Score sequences with online-safe nucleotide QC reward components."""
    df = add_nucleotide_metrics(sequences_df)
    df["reward_valid_nt_chars"] = df["valid_nt_chars"].astype(float)
    df["reward_genome_length"] = df["genome_length"].map(
        lambda value: _interval_score(value, config.genome_length_min, config.genome_length_max)
    )
    df["reward_gc_content"] = df["gc_content"].map(
        lambda value: _interval_score(value, config.gc_content_min, config.gc_content_max)
    )
    df["reward_nt_homopolymer"] = df["max_nt_homopolymer_length"].map(
        lambda value: _interval_score(value, config.homopolymer_min, config.homopolymer_max)
    )

    weighted_sum = (
        weights.valid_nt_chars * df["reward_valid_nt_chars"]
        + weights.genome_length * df["reward_genome_length"]
        + weights.gc_content * df["reward_gc_content"]
        + weights.nt_homopolymer * df["reward_nt_homopolymer"]
    )
    total_weight = weights.valid_nt_chars + weights.genome_length + weights.gc_content + weights.nt_homopolymer
    df["reward"] = weighted_sum / total_weight
    return df


def score_fasta(
    input_fasta: Path,
    output_csv: Path,
    config: NucleotideQCConfig = NucleotideQCConfig(),
    weights: RewardWeights = RewardWeights(),
) -> Path:
    """Score a FASTA file and write per-sequence reward diagnostics."""
    sequences_df = load_fasta_records(input_fasta)
    scored_df = score_nucleotide_metrics(sequences_df, config=config, weights=weights)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    """CLI entry point for scoring FASTA files with the online reward."""
    parser = argparse.ArgumentParser(description="Score Evo2 phage FASTA sequences with online-safe reward components")
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--genome-length-min", type=int, default=4000)
    parser.add_argument("--genome-length-max", type=int, default=6000)
    parser.add_argument("--gc-content-min", type=float, default=30.0)
    parser.add_argument("--gc-content-max", type=float, default=65.0)
    parser.add_argument("--homopolymer-max", type=int, default=10)
    args = parser.parse_args()

    output = score_fasta(
        input_fasta=args.input_fasta,
        output_csv=args.output_csv,
        config=NucleotideQCConfig(
            genome_length_min=args.genome_length_min,
            genome_length_max=args.genome_length_max,
            gc_content_min=args.gc_content_min,
            gc_content_max=args.gc_content_max,
            homopolymer_max=args.homopolymer_max,
        ),
    )
    print(f"reward_csv: {output}")


if __name__ == "__main__":
    main()
