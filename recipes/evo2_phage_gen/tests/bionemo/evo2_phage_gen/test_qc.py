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

import random
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from bionemo.evo2_phage_gen.qc import (
    NucleotideQCConfig,
    _parse_dustmasker_interval_output,
    add_nucleotide_metrics,
    apply_nucleotide_qc,
    calculate_dustmask_metrics,
    calculate_gc_content,
    calculate_nt_homopolymer_len,
    has_valid_nt_chars,
    run_nucleotide_qc,
    save_fasta,
    trim_at_first_eos,
)


def _deterministic_dna(length: int) -> str:
    """Build a reproducible DNA sequence without long simple terminal runs."""
    rng = random.Random(7)
    return "".join(rng.choice("ACGT") for _ in range(length))


def test_nucleotide_metrics_match_arc_filter_semantics():
    """Basic nucleotide metrics should match the Arc pipeline definitions."""
    assert has_valid_nt_chars("ACGTacgt")
    assert not has_valid_nt_chars("ACGTN")
    assert calculate_gc_content("ACGT") == 50.0
    assert calculate_nt_homopolymer_len("AAACCGTTTT") == 4


def test_trim_at_first_eos_handles_literal_markers():
    """Decoded EOS text should not be treated as biological sequence."""
    assert trim_at_first_eos("ACGT STOP ignored") == "ACGT"
    assert trim_at_first_eos("ACGT<EOS>ignored") == "ACGT"
    assert trim_at_first_eos("ACGTEODignored") == "ACGT"
    assert trim_at_first_eos("ACGT") == "ACGT"


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


def test_save_fasta_replaces_non_ascii_generated_tokens(tmp_path):
    """Generated Unicode tokens should fail QC as invalid bases rather than crashing FASTA output."""
    output_fasta = tmp_path / "unicode.fasta"
    df = pd.DataFrame({"id_prompt": ["seq1"], "sequence": ["ACGT\u013bACGT"]})

    save_fasta(df, output_fasta)

    assert output_fasta.read_text() == ">seq1\nACGTNACGT\n"


def test_dustmask_fallback_flags_low_complexity_sequence_ends():
    """The fallback DUST-style scorer should catch simple generated tails."""
    sequence = _deterministic_dna(400) + "A" * 160

    metrics = calculate_dustmask_metrics(
        sequence,
        window=64,
        level=20.0,
        end_window=200,
        max_end_fraction=0.5,
    )

    assert metrics.right_end_masked_fraction > 0.5
    assert metrics.max_end_masked_fraction > 0.5
    assert not metrics.end_pass


def test_dustmask_fallback_default_matches_qc_config_threshold(monkeypatch):
    """The public helper and config-driven path must share the 0.9 end threshold."""
    monkeypatch.setattr(
        "bionemo.evo2_phage_gen.qc.dustmask_low_complexity_mask",
        lambda _sequence, **_kwargs: [False] * 20 + [True] * 80,
    )

    metrics = calculate_dustmask_metrics("A" * 100, end_window=100)

    assert metrics.max_end_masked_fraction == 0.8
    assert metrics.end_pass


def test_add_nucleotide_metrics_uses_external_dustmasker_interval_output(monkeypatch):
    """When enabled, nucleotide metrics should call NCBI dustmasker once per batch."""
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        output_path = args[args.index("-out") + 1]
        Path(output_path).write_text(">seq_0\n0 - 79\n>seq_1\n320 - 399\n")

    monkeypatch.setattr("bionemo.evo2_phage_gen.qc.subprocess.run", fake_run)
    df = pd.DataFrame(
        {
            "id_prompt": ["left", "right"],
            "sequence": [_deterministic_dna(400), _deterministic_dna(400)],
        }
    )

    scored = add_nucleotide_metrics(
        df,
        NucleotideQCConfig(
            dustmask_filter=True,
            dustmasker_bin="fake-dustmasker",
            dustmask_level=20.5,
            dustmasker_timeout_s=17.5,
            dustmask_use_external=True,
            dustmask_end_window=100,
            dustmask_max_end_fraction=0.9,
        ),
    )

    assert len(calls) == 1
    assert calls[0][0][0] == "fake-dustmasker"
    assert calls[0][0][calls[0][0].index("-outfmt") + 1] == "interval"
    assert calls[0][0][calls[0][0].index("-level") + 1] == "20.5"
    assert calls[0][1]["timeout"] == 17.5
    assert scored["dustmask_left_end_masked_fraction"].tolist() == [0.8, 0.0]
    assert scored["dustmask_right_end_masked_fraction"].tolist() == [0.0, 0.8]
    assert scored["dustmask_end_pass"].tolist() == [True, True]


def test_dustmasker_accepts_zero_based_inclusive_full_length_interval(tmp_path):
    """A fully masked sequence is a valid QC result, not a parser failure."""
    interval_path = tmp_path / "dustmasker.interval"
    interval_path.write_text(">seq_0\n0 - 5999\n")

    masks = _parse_dustmasker_interval_output(interval_path, [6000])

    assert len(masks[0]) == 6000
    assert all(masks[0])


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("dustmasker not found"),
        subprocess.CalledProcessError(2, ["dustmasker"]),
        subprocess.TimeoutExpired(["dustmasker"], 300),
    ],
    ids=["missing", "nonzero", "timeout"],
)
def test_external_dustmasker_failures_are_bounded_and_wrapped(monkeypatch, error):
    """External dustmasker failures should use consistent runtime errors."""

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("bionemo.evo2_phage_gen.qc.subprocess.run", fail)
    frame = pd.DataFrame({"id_prompt": ["candidate"], "sequence": ["ACGT" * 100]})

    with pytest.raises(RuntimeError, match="dustmasker execution failed"):
        add_nucleotide_metrics(
            frame,
            NucleotideQCConfig(dustmask_filter=True, dustmask_use_external=True),
        )


@pytest.mark.parametrize(
    "interval",
    ["0 - 4", "3 - 2", "4 - 4", "-1 - 2", "0 - -1", "not-an-interval"],
)
def test_dustmasker_intervals_reject_invalid_zero_based_coordinates(tmp_path, interval):
    """Malformed or out-of-range dustmasker coordinates should be rejected."""
    interval_path = tmp_path / "dustmasker.interval"
    interval_path.write_text(f">seq_0\n{interval}\n")

    with pytest.raises(ValueError, match="invalid dustmasker interval"):
        _parse_dustmasker_interval_output(interval_path, [4])
