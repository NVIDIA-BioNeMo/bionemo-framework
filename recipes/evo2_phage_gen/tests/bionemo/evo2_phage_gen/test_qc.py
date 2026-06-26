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

"""Tests for ``bionemo.evo2_phage_gen.qc``."""

import pandas as pd

from bionemo.evo2_phage_gen.qc import (
    NucleotideQCConfig,
    apply_nucleotide_qc,
    calculate_gc_content,
    calculate_nt_homopolymer_len,
    has_valid_nt_chars,
    run_nucleotide_qc,
)


def test_nucleotide_metrics_match_arc_filter_semantics():
    """Basic nucleotide metrics should match the Arc pipeline definitions."""
    assert has_valid_nt_chars("ACGTacgt")
    assert not has_valid_nt_chars("ACGTN")
    assert calculate_gc_content("ACGT") == 50.0
    assert calculate_nt_homopolymer_len("AAACCGTTTT") == 4


def test_apply_nucleotide_qc_tracks_staged_counts():
    """Staged counts should decrease as each nucleotide QC filter is applied."""
    df = pd.DataFrame(
        {
            "id_prompt": ["pass", "bad_nt", "short", "high_gc", "homopolymer"],
            "sequence": ["ACGT" * 1000, "ACGTN" * 1000, "ACGT", "G" * 4000, "ACGT" * 1000 + "A" * 11],
        }
    )

    filtered, counts = apply_nucleotide_qc(
        df,
        NucleotideQCConfig(
            genome_length_min=4000,
            genome_length_max=6000,
            gc_content_min=30.0,
            gc_content_max=65.0,
            homopolymer_max=10,
        ),
    )

    assert filtered["id_prompt"].tolist() == ["pass"]
    assert counts["stage"].tolist() == [
        "qc1_initial",
        "valid_nt_chars",
        "genome_length",
        "gc_content",
        "nt_homopolymer",
    ]
    assert counts["num_sequences"].tolist() == [5, 4, 3, 2, 1]


def test_run_nucleotide_qc_writes_arc_style_outputs(tmp_path):
    """The CLI backing function should emit staged CSV and FASTA files."""
    input_fasta = tmp_path / "input.fasta"
    input_fasta.write_text(">seq1\n" + "ACGT" * 1000 + "\n>seq2\nACGTN\n")

    outputs = run_nucleotide_qc(input_fasta, tmp_path / "qc")

    assert outputs["initial_csv"].exists()
    assert outputs["initial_fasta"].exists()
    assert outputs["nucleotide_counts_csv"].exists()
    assert outputs["nucleotide_csv"].exists()
    assert outputs["nucleotide_fasta"].exists()
