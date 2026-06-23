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

"""CPU tests for the per-token labelers (pure masks, no model)."""

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from labelers import LABELERS, SeqContext


def _ctx(dna, tag_len=0):
    text = "X" * tag_len + dna
    return SeqContext(text=text, tag_len=tag_len, dna=dna, kingdom="prok", hidden_norm=np.zeros(tag_len + len(dna)))


def test_consensus_motifs_fire_at_match_positions():
    """The table-driven motifs mark the right positions (ATG/stop = start, TATA = span)."""
    ctx = _ctx("ATGTAACGT")  # ATG @0 ; TAA (stop) @3
    assert list(LABELERS["motif_ATG"](ctx).nonzero()[0]) == [0]
    assert list(LABELERS["motif_stop"](ctx).nonzero()[0]) == [3]
    assert list(LABELERS["motif_TATA"](_ctx("TATAAA")).nonzero()[0]) == [0, 1, 2, 3, 4, 5]  # spans the match


def test_base_labelers_fire_per_nucleotide():
    """base_A/C/G/T each fire exactly on their nucleotide."""
    ctx = _ctx("ACGTAA")
    assert list(LABELERS["base_A"](ctx).nonzero()[0]) == [0, 4, 5]
    assert list(LABELERS["base_G"](ctx).nonzero()[0]) == [2]


def test_tag_prefix_is_unlabeled():
    """Sequence-derived labels are False over the leading phylo-tag tokens."""
    ctx = _ctx("ATG", tag_len=2)  # tokens: [tag, tag, A, T, G]
    m = LABELERS["motif_ATG"](ctx)
    assert len(m) == 5 and not m[:2].any() and m[2]  # ATG starts at DNA pos 0 -> token 2
