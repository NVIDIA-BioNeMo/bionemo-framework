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

import numpy as np
from evo2_sae.eval.probing.labelers import LABELERS, SeqContext


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


def test_gc_window_labelers():
    """gc_high/low fire on GC-rich / AT-rich windows (mean GC over the ±10 window)."""
    assert LABELERS["gc_high_window"](_ctx("G" * 30)).all()  # all-GC -> high everywhere
    assert not LABELERS["gc_high_window"](_ctx("A" * 30)).any()
    assert LABELERS["gc_low_window"](_ctx("A" * 30)).all()  # all-AT -> low everywhere


def test_homopolymer_window():
    """homopolymer_window marks runs of >=5 identical bases (and nothing shorter)."""
    m = LABELERS["homopolymer_window"](_ctx("TTAAAAACC"))  # AAAAA (5) at positions 2..6
    assert list(m.nonzero()[0]) == [2, 3, 4, 5, 6]


def test_dinuc_repeat_window():
    """dinuc_repeat_window marks alternating dinucleotide repeats (>=3 reps, span >=6)."""
    assert LABELERS["dinuc_repeat_window"](_ctx("ATATATAT")).all()  # (AT)x4
    assert not LABELERS["dinuc_repeat_window"](_ctx("ATAT")).any()  # (AT)x2 -> too short


def test_consensus_offsets_kozak_and_splice():
    """The consensus labelers mark the biologically meaningful offset within the match."""
    assert list(LABELERS["kozak_atg"](_ctx("AAAATGG")).nonzero()[0]) == [3]  # [AG]..ATGG -> the ATG
    assert list(LABELERS["splice_donor"](_ctx("CCGTAAGTCC")).nonzero()[0]) == [2]  # GT[AG]AGT -> the GT
    assert list(LABELERS["splice_acceptor"](_ctx("TTTTTTAG")).nonzero()[0]) == [6, 7]  # ...AG -> the AG


def test_frame_and_start_forward_and_reverse():
    """The CDS frame helper anchors frame 0 at the strand-correct start (the reverse-strand
    off-by-one): start = low coord on +strand, high coord on -strand, counting back."""
    from evo2_sae.eval.probing.labelers import _frame_and_start

    frame, start = _frame_and_start(10, 19, strand=1)  # 9 bp = 3 codons, forward
    assert start == 10 and list(frame) == [0, 1, 2, 0, 1, 2, 0, 1, 2]

    frame, start = _frame_and_start(10, 19, strand=-1)  # reverse: start at the high coord
    assert start == 18 and list(frame) == [2, 1, 0, 2, 1, 0, 2, 1, 0]
