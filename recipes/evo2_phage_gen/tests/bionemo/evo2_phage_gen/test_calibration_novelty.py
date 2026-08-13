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

from bionemo.evo2_phage_gen.calibration_novelty import (
    SEARCH_COLUMNS,
    _top_hits,
    canonical_circular_sequence,
    normalize_prompted_fasta,
    summarize_novelty,
)


def test_canonical_circular_sequence_handles_rotation_and_reverse_complement():
    sequence = "ACGTTT"
    rotated = "TTTACG"
    reverse_complement = "AAACGT"

    assert canonical_circular_sequence(sequence) == canonical_circular_sequence(rotated)
    assert canonical_circular_sequence(sequence) == canonical_circular_sequence(reverse_complement)


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
