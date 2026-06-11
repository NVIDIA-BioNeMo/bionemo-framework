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

"""Extensible per-token biological labelers for SAE feature probing.

Each labeler maps a `SeqContext` (one tokenized sequence) to a per-token boolean
mask of length `T`. The per-feature AUROC probe (`probe_features.py`) asks, for
every label and every SAE feature, how well the feature's activation separates
positive from negative tokens.

Adding a feature is just writing a function and decorating it:

    @labeler("my_concept")
    def _my(ctx):
        return some_bool_array_len_T

`complex=True` flags labelers that are proxies or need real external annotation
(e.g. true gene models) and should be refined later — they're the natural home
for the "more complicated features" we want to add at the end.

Conventions
-----------
* Tokens 0..tag_len-1 are the phylogenetic-tag prefix; sequence-derived motif /
  positional labels are False there (use `_dna_mask`). Sequence-level labels
  (`is_prok`) and norm-based labels (`is_sink_token`) may mark tag tokens.
* Byte-level Evo2 tokenization is 1 char = 1 token, so token i in the DNA region
  corresponds to base `ctx.dna[i - tag_len]`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


# name -> fn(ctx) -> np.ndarray[bool] of length T
LABELERS: dict[str, callable] = {}
# labelers that are proxies / need real annotations (documented, refine later)
COMPLEX_LABELERS: set[str] = set()

# Sink-token norm threshold (residual L2). Set by the driver from the data
# (Evo2 7B layer-26 sinks sit ~1638 vs a ~21 median, so this cleanly isolates them).
SINK_NORM_THRESHOLD: float = 100.0


def labeler(name: str, complex: bool = False):
    """Register a per-token labeler under `name`."""

    def deco(fn):
        LABELERS[name] = fn
        if complex:
            COMPLEX_LABELERS.add(name)
        return fn

    return deco


@dataclass
class SeqContext:
    """Everything a labeler needs about one tokenized sequence."""

    text: str  # tag + dna (1 char == 1 token)
    tag_len: int  # number of leading phylo-tag tokens
    dna: str  # the DNA region (uppercase ACGTN), len == T - tag_len
    kingdom: str  # 'prok' | 'euk'
    hidden_norm: np.ndarray  # [T] residual L2 norm per token
    # Gene-structure annotation over DNA positions (filled by a gene caller; None if absent).
    cds_mask: Optional[np.ndarray] = None  # bool[len(dna)] — within a predicted CDS (either strand)
    cds_frame: Optional[np.ndarray] = None  # int8[len(dna)] — codon position 0/1/2 within CDS, -1 if not
    gene_starts: Optional[np.ndarray] = None  # bool[len(dna)] — predicted translation start positions

    @property
    def T(self) -> int:  # noqa: D102
        return self.tag_len + len(self.dna)


def _dna_mask(ctx: SeqContext, dna_bool: np.ndarray) -> np.ndarray:
    """Lift a per-DNA-position bool array to a per-token mask (tag tokens False)."""
    out = np.zeros(ctx.T, dtype=bool)
    out[ctx.tag_len : ctx.tag_len + len(dna_bool)] = dna_bool
    return out


def _bytes(dna: str) -> np.ndarray:
    return np.frombuffer(dna.encode("ascii", "replace"), dtype=np.uint8)


# --------------------------------------------------------------------- positional
@labeler("first_100bp")
def _first(ctx):
    d = np.zeros(len(ctx.dna), bool)
    d[:100] = True
    return _dna_mask(ctx, d)


@labeler("last_100bp")
def _last(ctx):
    d = np.zeros(len(ctx.dna), bool)
    if len(d):
        d[-100:] = True
    return _dna_mask(ctx, d)


# --------------------------------------------------------------------- composition
def _gc_window(dna: str, radius: int = 10) -> np.ndarray:
    arr = _bytes(dna)
    gc = ((arr == ord("G")) | (arr == ord("C"))).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(gc)])
    n = len(gc)
    idx = np.arange(n)
    lo = np.maximum(0, idx - radius)
    hi = np.minimum(n, idx + radius + 1)
    return (csum[hi] - csum[lo]) / np.maximum(1, hi - lo)


@labeler("gc_high_window")
def _gch(ctx):
    return _dna_mask(ctx, _gc_window(ctx.dna) >= 0.60)


@labeler("gc_low_window")
def _gcl(ctx):
    return _dna_mask(ctx, _gc_window(ctx.dna) <= 0.30)


@labeler("homopolymer_window")
def _homo(ctx, k: int = 5):
    d, n = ctx.dna, len(ctx.dna)
    out = np.zeros(n, bool)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and d[j + 1] == d[i]:
            j += 1
        if j - i + 1 >= k:
            out[i : j + 1] = True
        i = j + 1
    return _dna_mask(ctx, out)


@labeler("dinuc_repeat_window")
def _dinuc(ctx, min_reps: int = 3):
    d, n = ctx.dna, len(ctx.dna)
    out = np.zeros(n, bool)
    i = 0
    while i < n - 1:
        if d[i] != d[i + 1]:
            j = i
            while j + 2 < n and d[j + 2] == d[j]:
                j += 1
            span = j + 2 - i
            if span >= 2 * min_reps:
                out[i : j + 2] = True
            i = max(j + 1, i + 1)
        else:
            i += 1
    return _dna_mask(ctx, out)


# --------------------------------------------------------------------- motifs
def _starts(dna: str, pattern: str) -> np.ndarray:
    out = np.zeros(len(dna), bool)
    for m in re.finditer(pattern, dna):
        out[m.start()] = True
    return out


def _spans(dna: str, pattern: str) -> np.ndarray:
    out = np.zeros(len(dna), bool)
    for m in re.finditer(pattern, dna):
        out[m.start() : m.end()] = True
    return out


@labeler("motif_ATG")
def _atg(ctx):
    return _dna_mask(ctx, _starts(ctx.dna, r"ATG"))


@labeler("motif_stop")
def _stop(ctx):
    return _dna_mask(ctx, _starts(ctx.dna, r"TAA|TAG|TGA"))


@labeler("motif_TATA")
def _tata(ctx):
    return _dna_mask(ctx, _spans(ctx.dna, r"TATA[AT]A"))


@labeler("motif_RBS_SD")
def _rbs(ctx):
    # Shine-Dalgarno ribosome-binding site
    return _dna_mask(ctx, _spans(ctx.dna, r"AGGAGG"))


# --------------------------------------------------- complex / consensus (refine later)
@labeler("kozak_atg", complex=True)
def _kozak(ctx):
    # Kozak: (A/G)xxATGG — mark the ATG start (match start + 3)
    out = np.zeros(len(ctx.dna), bool)
    for m in re.finditer(r"[AG]..ATGG", ctx.dna):
        out[m.start() + 3] = True
    return _dna_mask(ctx, out)


@labeler("splice_donor", complex=True)
def _sd(ctx):
    # 5' donor consensus GT(A/G)AGT — mark the GT
    return _dna_mask(ctx, _starts(ctx.dna, r"GT[AG]AG"))


@labeler("splice_acceptor", complex=True)
def _sa(ctx):
    # 3' acceptor: polypyrimidine tract then AG — mark the AG
    out = np.zeros(len(ctx.dna), bool)
    for m in re.finditer(r"[CT]{6}[ACGT]?AG", ctx.dna):
        out[m.end() - 2 : m.end()] = True
    return _dna_mask(ctx, out)


# --------------------------------------------------------------- sequence / norm level
@labeler("is_prok")
def _prok(ctx):
    return np.full(ctx.T, ctx.kingdom == "prok", dtype=bool)


@labeler("is_sink_token", complex=True)
def _sink(ctx):
    return ctx.hidden_norm > SINK_NORM_THRESHOLD


# --------------------------------------------- gene structure (real annotation, prok)
# These read a CDS annotation attached to the context by a gene caller (see
# predict_cds, prokaryotes only). They are no-ops when the annotation is absent.
@labeler("cds_coding", complex=True)
def _cds(ctx):
    if ctx.cds_mask is None:
        return np.zeros(ctx.T, bool)
    return _dna_mask(ctx, ctx.cds_mask)


@labeler("cds_start", complex=True)
def _cds_start(ctx):
    if ctx.gene_starts is None:
        return np.zeros(ctx.T, bool)
    return _dna_mask(ctx, ctx.gene_starts)


@labeler("cds_frame_1", complex=True)
def _cds_f1(ctx):
    # codon position 1 within a REAL predicted CDS (not the frame-0-from-start proxy)
    if ctx.cds_frame is None:
        return np.zeros(ctx.T, bool)
    return _dna_mask(ctx, ctx.cds_frame == 0)


@labeler("cds_frame_3", complex=True)
def _cds_f3(ctx):
    if ctx.cds_frame is None:
        return np.zeros(ctx.T, bool)
    return _dna_mask(ctx, ctx.cds_frame == 2)


_GENE_FINDER = None

# Standard genetic code (NCBI translation table 1), codons in TCAG x TCAG x TCAG order.
_BASES = "TCAG"
_AA1 = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {
    a + b + c: _AA1[i] for i, (a, b, c) in enumerate((x, y, z) for x in _BASES for y in _BASES for z in _BASES)
}
CODON_LIST = sorted(CODON_TABLE)  # 64 codons
CODON_TO_IDX = {c: i for i, c in enumerate(CODON_LIST)}
AA_LIST = sorted(set(CODON_TABLE.values()))  # 20 aa + '*' (stop)
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}
_COMP = str.maketrans("ACGTN", "TGCAN")


def _revcomp(s):
    return s.translate(_COMP)[::-1]


def predict_codons(dna: str):
    """In-frame codon + amino-acid identity at strand-correct codon anchors (prok genes).

    Returns (codon_id[N], aa_id[N]) over forward DNA coordinates; the anchor is the
    first translated base of each codon (low coord on +strand, high coord on -strand),
    other positions are -1.  codon_id in 0..63 (CODON_LIST), aa_id in 0..20 (AA_LIST).
    """
    global _GENE_FINDER
    n = len(dna)
    codon_id = np.full(n, -1, dtype=np.int16)
    aa_id = np.full(n, -1, dtype=np.int8)
    if n < 60:
        return codon_id, aa_id
    if _GENE_FINDER is None:
        import pyrodigal

        _GENE_FINDER = pyrodigal.GeneFinder(meta=True)
    for g in _GENE_FINDER.find_genes(dna.encode("ascii", "replace")):
        b, e = max(0, g.begin - 1), min(n, g.end)
        sub = dna[b:e]
        coding = sub if g.strand == 1 else _revcomp(sub)
        for i in range(len(coding) // 3):
            cod = coding[3 * i : 3 * i + 3]
            j = CODON_TO_IDX.get(cod)
            if j is None:
                continue
            p = b + 3 * i if g.strand == 1 else (e - 1 - 3 * i)
            if 0 <= p < n:
                codon_id[p] = j
                aa_id[p] = AA_TO_IDX[CODON_TABLE[cod]]
    return codon_id, aa_id


def predict_cds(dna: str):
    """Prokaryotic gene calling via pyrodigal (meta mode) on a single DNA chunk.

    Returns (cds_mask, cds_frame, gene_starts) over forward DNA coordinates:
      cds_mask[i]    True if position i lies within any predicted CDS (either strand)
      cds_frame[i]   codon position 0/1/2 relative to that gene's start (strand-aware), else -1
      gene_starts[i] True at predicted translation starts
    """
    global _GENE_FINDER
    n = len(dna)
    cds_mask = np.zeros(n, dtype=bool)
    cds_frame = np.full(n, -1, dtype=np.int8)
    gene_starts = np.zeros(n, dtype=bool)
    if n < 60:
        return cds_mask, cds_frame, gene_starts
    if _GENE_FINDER is None:
        import pyrodigal

        _GENE_FINDER = pyrodigal.GeneFinder(meta=True)
    for g in _GENE_FINDER.find_genes(dna.encode("ascii", "replace")):
        b, e = g.begin - 1, g.end  # 0-based half-open, forward coords
        b, e = max(0, b), min(n, e)
        if e <= b:
            continue
        cds_mask[b:e] = True
        idx = np.arange(b, e)
        if g.strand == 1:
            gene_starts[b] = True
            cds_frame[b:e] = (idx - b) % 3
        else:  # reverse strand: start codon sits at the (forward) end
            gene_starts[e - 1] = True
            cds_frame[b:e] = ((e - 1) - idx) % 3
    return cds_mask, cds_frame, gene_starts


# Default label set for the probe (order preserved in outputs).
DEFAULT_LABELS = list(LABELERS.keys())
