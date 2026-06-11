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

r"""Generic interval-track loader for the "user-supplied annotated dataset" eval.

The user hands in an annotated dataset: a FASTA of sequences + one or more annotation
tracks (BED or GFF) naming intervals — RefSeq genes/exons, Rfam ncRNA, JASPAR TFBS,
ENCODE cCREs, etc. Each interval is one annotation **instance**. This module tiles the
sequences into windows and produces, per concept, a per-token boolean mask + per-token
**global** instance IDs (stable across the windows an interval spans) — exactly the
inputs `sae.eval.probing.domain_f1` (recall-per-instance) and `auroc_all` (per-feature)
consume. No model here; the SAE-encode step lives in the probe CLI (`probe.py domain-eval`).

This is the generic sibling of `euk_windows.py` (which decomposes RefSeq gene models into
exon/intron/cds). Both feed the same shared scorers.
"""

from __future__ import annotations

import gzip
from collections import defaultdict


def read_fasta_dict(path: str) -> dict[str, str]:
    """Read a (multi-record) FASTA into ``{seq_id: sequence}`` (``.gz`` transparent).

    ``seq_id`` is the first whitespace-delimited token of the header, so it matches the
    chrom/seqid column of BED/GFF tracks.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    seqs: dict[str, str] = {}
    name, parts = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(parts)
                header = line[1:].strip().split()
                name, parts = (header[0] if header else f"seq_{len(seqs)}"), []
            else:
                parts.append(line)
    if name is not None:
        seqs[name] = "".join(parts)
    return seqs


def _parse_bed(path):
    """Yield (chrom, start0, end0) from a BED file (0-based, half-open — used as-is)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            if len(f) < 3:
                continue
            yield f[0], int(f[1]), int(f[2])


def _parse_gff(path, feature_type=None):
    """Yield (seqid, start0, end0) from GFF/GTF (1-based inclusive -> 0-based half-open)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            if feature_type and f[2] != feature_type:
                continue
            yield f[0], int(f[3]) - 1, int(f[4])


def load_track(path: str, feature_type: str | None = None, fmt: str | None = None) -> dict[str, list[tuple[int, int]]]:
    """Load one annotation track into ``{seqid: [(start0, end0), ...]}`` (0-based half-open).

    Args:
        path: BED or GFF/GTF file (``.gz`` ok).
        feature_type: GFF only — keep just this column-3 type (e.g. ``"exon"``, ``"ncRNA"``).
        fmt: ``"bed"`` / ``"gff"``; inferred from the extension when omitted.

    Returns:
        Intervals grouped by sequence id, each sorted. Every interval is one instance.
    """
    fmt = fmt or ("gff" if str(path).endswith((".gff", ".gff3", ".gtf", ".gff.gz", ".gff3.gz")) else "bed")
    rows = _parse_gff(path, feature_type) if fmt == "gff" else _parse_bed(path)
    by_seq: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chrom, s, e in rows:
        if e > s:
            by_seq[chrom].append((s, e))
    return {k: sorted(v) for k, v in by_seq.items()}


def label_windows(seqs, tracks, seq_len=1024, stride=None, max_tokens=None, min_n_frac=0.5):
    """Tile sequences into windows, labeling each position per concept (mask + global instance id).

    Args:
        seqs: ``{seqid: dna_str}``.
        tracks: ``{concept: {seqid: [(start0, end0), ...]}}`` (e.g. from `load_track`).
        seq_len: window length in bp.
        stride: step between windows (defaults to non-overlapping = seq_len).
        max_tokens: stop once this many positions are emitted (None = all).
        min_n_frac: skip windows whose ``N`` fraction exceeds this.

    Returns:
        (windows, stats). Each window is ``{"dna": str, "labels": {concept: bool[L]},
        "instances": {concept: int32[L]}}``. Each interval gets one global id, stable across
        the windows it spans, so `domain_f1`'s recall-per-instance counts a split interval once.
    """
    import numpy as np

    stride = stride or seq_len
    concepts = list(tracks.keys())
    # assign a global instance id to every interval, per concept
    concept_iv: dict[str, dict[str, list[tuple[int, int, int]]]] = {}
    n_inst: dict[str, int] = {}
    for concept in concepts:
        gid = 0
        cc: dict[str, list[tuple[int, int, int]]] = {}
        for seqid, ivs in tracks[concept].items():
            cc[seqid] = [(s, e, (gid := gid + 1) - 1) for (s, e) in ivs]
        concept_iv[concept] = cc
        n_inst[concept] = gid

    windows, tot = [], 0
    for seqid, dna in seqs.items():
        dna = dna.upper()
        N = len(dna)
        for w0 in range(0, max(1, N - seq_len + 1), stride):
            w1 = min(N, w0 + seq_len)
            sub = dna[w0:w1]
            L = w1 - w0
            if L < 60 or sub.count("N") > min_n_frac * L:
                continue
            labels = {c: np.zeros(L, bool) for c in concepts}
            inst = {c: np.full(L, -1, np.int32) for c in concepts}
            for c in concepts:
                for s, e, gid in concept_iv[c].get(seqid, []):
                    if e <= w0 or s >= w1:
                        continue
                    labels[c][max(s, w0) - w0 : min(e, w1) - w0] = True
                    inst[c][max(s, w0) - w0 : min(e, w1) - w0] = gid
            windows.append({"dna": sub, "labels": labels, "instances": inst})
            tot += L
            if max_tokens and tot >= max_tokens:
                return windows, {"tokens": tot, "n_inst": n_inst, "concepts": concepts}
    return windows, {"tokens": tot, "n_inst": n_inst, "concepts": concepts}
