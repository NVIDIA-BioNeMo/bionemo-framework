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

from __future__ import annotations

import pandas as pd
import pytest

import bionemo.evo2_phage_gen.calibration_novelty as novelty
from bionemo.evo2_phage_gen.calibration_novelty import (
    SEARCH_COLUMNS,
    _top_hits,
    canonical_circular_sequence,
    measure_novelty,
    normalize_prompted_fasta,
    summarize_novelty,
    validate_novelty_file,
)


def test_canonical_circular_sequence_handles_rotation_and_reverse_complement():
    sequence = "ACGTTT"
    rotated = "TTTACG"
    reverse_complement = "AAACGT"

    assert canonical_circular_sequence(sequence) == canonical_circular_sequence(rotated)
    assert canonical_circular_sequence(sequence) == canonical_circular_sequence(reverse_complement)


def test_canonical_circular_sequence_complements_iupac_symbols():
    sequence = "ACGTRYSWKMBDHVN"
    reverse_complement = "NBDHVKMWSRYACGT"

    assert canonical_circular_sequence(sequence) == canonical_circular_sequence(reverse_complement)
    with pytest.raises(ValueError, match="unsupported IUPAC"):
        canonical_circular_sequence("ACGTZ")


def test_normalize_prompted_fasta_strips_control_tokens_and_rejects_non_dna(tmp_path):
    source = tmp_path / "prompted.fna"
    source.write_text(">prompted\n+~ACGT\n>raw\nTGCA\n")
    output = tmp_path / "payload.fna"

    normalize_prompted_fasta(source, output)

    assert output.read_text() == ">prompted\nACGT\n>raw\nTGCA\n"

    invalid = tmp_path / "invalid.fna"
    invalid.write_text(">bad\n+~ACNT\n")
    with pytest.raises(ValueError, match="non-DNA"):
        normalize_prompted_fasta(invalid, tmp_path / "unused.fna")


def test_measure_novelty_normalizes_target_reference_before_search_and_keying(tmp_path, monkeypatch):
    reference = tmp_path / "reference.fna"
    reference.write_text(">reference\n+~ACGT\n")
    sft = tmp_path / "sft.fna"
    sft.write_text(">sft\n+~ACGT\n")
    monkeypatch.setattr(
        novelty,
        "_load_sweep",
        lambda _root: pd.DataFrame({"id_prompt": ["generated"], "cell": ["cell"], "sequence": ["ACGT"]}),
    )
    searched_references = []

    def fake_search(_binary, _query, search_reference, output, *_args):
        searched_references.append(search_reference)
        output.touch()

    monkeypatch.setattr(novelty, "_run_search", fake_search)

    metrics = measure_novelty(
        generation_root=tmp_path / "generation",
        reference_fasta=reference,
        sft_fasta=sft,
        tool_bin_dir=tmp_path,
        work_dir=tmp_path / "work",
        output_csv=tmp_path / "metrics.csv",
        threads=1,
    )

    assert searched_references[0].name == "reference-payload.fasta"
    assert searched_references[0].read_text() == ">reference\nACGT\n"
    assert metrics.loc[0, "exact_target_circular_or_revcomp"] == 1.0


def test_summarize_novelty_reports_copy_rates():
    metrics = pd.DataFrame(
        {
            "cell": ["prefix0_temp1.0", "prefix0_temp1.0"],
            "exact_target_circular_or_revcomp": [1.0, 0.0],
            "exact_sft_circular_or_revcomp": [1.0, 0.0],
            "target_near_copy_98_9pct": [1.0, 0.0],
            "sft_near_copy_98_9pct": [1.0, 1.0],
            "target_pident": [100.0, 80.0],
            "sft_pident": [100.0, 99.0],
        }
    )

    summary = summarize_novelty(metrics).iloc[0]

    assert summary["exact_target_copy_rate"] == 0.5
    assert summary["sft_near_copy_rate"] == 1.0


@pytest.mark.parametrize("create_file", [False, True], ids=["missing", "empty"])
def test_top_hits_preserves_prefixed_schema_without_hits(tmp_path, create_file):
    path = tmp_path / "hits.m8"
    if create_file:
        path.touch()

    hits = _top_hits(path, "target")

    assert hits.empty
    assert list(hits.columns) == [
        "id_prompt",
        *(f"target_{column}" for column in SEARCH_COLUMNS if column != "query"),
    ]


def test_validate_novelty_file_requires_id_prompt(tmp_path):
    path = tmp_path / "novelty.csv"
    pd.DataFrame({"cell": ["prefix0_temp1.0"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing id_prompt column"):
        validate_novelty_file(path, expected_records=1)
