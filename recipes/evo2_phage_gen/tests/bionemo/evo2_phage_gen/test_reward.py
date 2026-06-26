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

"""Tests for ``bionemo.evo2_phage_gen.reward``."""

import pandas as pd

from bionemo.evo2_phage_gen.reward import score_fasta, score_nucleotide_metrics


def test_score_nucleotide_metrics_rewards_passing_sequence():
    """A sequence satisfying all online nucleotide filters should receive reward 1."""
    df = pd.DataFrame({"id_prompt": ["pass"], "sequence": ["ACGT" * 1000]})

    scored = score_nucleotide_metrics(df)

    assert scored.loc[0, "reward"] == 1.0
    assert scored.loc[0, "reward_valid_nt_chars"] == 1.0


def test_score_nucleotide_metrics_penalizes_invalid_sequence():
    """Invalid characters and out-of-range metrics should reduce the reward."""
    df = pd.DataFrame({"id_prompt": ["bad"], "sequence": ["NNNN"]})

    scored = score_nucleotide_metrics(df)

    assert scored.loc[0, "reward"] < 1.0
    assert scored.loc[0, "reward_valid_nt_chars"] == 0.0


def test_score_fasta_writes_reward_csv(tmp_path):
    """The reward CLI backing function should write per-sequence diagnostics."""
    input_fasta = tmp_path / "input.fasta"
    output_csv = tmp_path / "rewards.csv"
    input_fasta.write_text(">seq1\n" + "ACGT" * 1000 + "\n")

    score_fasta(input_fasta, output_csv)

    scored = pd.read_csv(output_csv)
    assert scored["reward"].tolist() == [1.0]
