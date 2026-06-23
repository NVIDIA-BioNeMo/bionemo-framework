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

r"""Build instance-level exon/intron/CDS-labeled windows from a genome FASTA + GFF3.

Eukaryotic gene-structure annotation for SAE feature probing. Unlike the
sequence-derived labelers, these labels come from real gene models, and crucially
carry *instance IDs* (which exon / which intron / which gene each position belongs
to) so domain-adjusted F1 can compute recall PER ANNOTATION INSTANCE (a feature
"recalls" an exon if it fires anywhere inside it), not per position.

For each protein-coding gene we take a representative transcript (longest by total
exon length), tile its span ± flank into windows, and label every position:
  exon / intron / cds / utr / intergenic   (+ per-position instance IDs for
  exon, intron, gene)

`python euk_windows.py --fasta chr21.fa --gff chr21.gff3 --dry-run` prints
coverage stats without building sequences.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import numpy as np


def _attrs(s):
    return dict(kv.split("=", 1) for kv in s.strip().split(";") if "=" in kv)


def parse_gff(gff_path):
    """Return {gene_id: {strand, tx: {tx_id: {'exon': [(s,e)], 'cds': [(s,e)]}}}} (protein_coding)."""
    gene_strand, gene_biotype = {}, {}
    tx_gene, tx_biotype = {}, {}
    tx_exon = defaultdict(list)
    tx_cds = defaultdict(list)
    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            typ, s, e, strand, attr = f[2], int(f[3]), int(f[4]), f[6], f[8]
            a = _attrs(attr)
            if typ == "gene":
                gid = a.get("ID", "").replace("gene:", "")
                gene_strand[gid] = strand
                gene_biotype[gid] = a.get("biotype", "")
            elif typ in ("mRNA", "transcript"):
                tid = a.get("ID", "").replace("transcript:", "")
                tx_gene[tid] = a.get("Parent", "").replace("gene:", "")
                tx_biotype[tid] = a.get("biotype", "")
            elif typ == "exon":
                for tid in a.get("Parent", "").replace("transcript:", "").split(","):
                    if tid:
                        tx_exon[tid].append((s, e))
            elif typ == "CDS":
                for tid in a.get("Parent", "").replace("transcript:", "").split(","):
                    if tid:
                        tx_cds[tid].append((s, e))
    genes = {}
    for tid, gid in tx_gene.items():
        if gene_biotype.get(gid) != "protein_coding" or tx_biotype.get(tid) != "protein_coding":
            continue
        if not tx_exon.get(tid):
            continue
        genes.setdefault(gid, {"strand": gene_strand.get(gid, "+"), "tx": {}})
        genes[gid]["tx"][tid] = {"exon": sorted(tx_exon[tid]), "cds": sorted(tx_cds.get(tid, []))}
    return genes


def representative_tx(gene):
    """Longest transcript by total exon length."""
    best, best_len = None, -1
    for tid, t in gene["tx"].items():
        ln = sum(e - s + 1 for s, e in t["exon"])
        if ln > best_len:
            best, best_len = tid, ln
    return best, gene["tx"][best]


def _label_window(chrom, w0, w1, gm, N):
    """Label a window [w0,w1) using one gene model's intervals (central-gene approx)."""
    L = w1 - w0
    pos = np.arange(w0, w1)
    lab = {k: np.zeros(L, bool) for k in ("exon", "intron", "cds", "utr", "intergenic")}
    inst = {k: np.full(L, -1, np.int32) for k in ("exon", "intron", "gene")}
    g_start, g_end = gm["span"]
    in_tx = (pos >= g_start - 1) & (pos < g_end)
    lab["intergenic"][~in_tx] = True
    inst["gene"][in_tx] = gm["gi"]
    for (s, e), iid in zip(gm["exons"], gm["exon_ids"]):
        m = (pos >= s - 1) & (pos < e)
        lab["exon"][m] = True
        inst["exon"][m] = iid
    for (s, e), iid in zip(gm["introns"], gm["intron_ids"]):
        m = (pos >= s - 1) & (pos < e)
        lab["intron"][m] = True
        inst["intron"][m] = iid
    for s, e in gm["cds"]:
        lab["cds"][(pos >= s - 1) & (pos < e)] = True
    lab["utr"] = lab["exon"] & ~lab["cds"]
    return {"dna": chrom[w0:w1], "labels": lab, "instances": inst}


def build_windows(  # noqa: D103
    fasta, gff, seq_len=1024, max_tokens=300_000, flank=300, seed=0, intergenic_frac=0.12, dry_run=False
):
    seqs = []
    with open(fasta) as fh:
        for line in fh:
            if not line.startswith(">"):
                seqs.append(line.strip())
    chrom = "".join(seqs).upper()
    N = len(chrom)
    genes = parse_gff(gff)

    exon_id, intron_id, gene_id = {}, {}, {}
    stats = defaultdict(int)
    gene_models, gene_spans = [], []
    for gid, gene in genes.items():
        tid, tx = representative_tx(gene)
        exons, cds = tx["exon"], tx["cds"]
        if not exons:
            continue
        g_start, g_end = exons[0][0], exons[-1][1]
        introns = [
            (exons[i][1] + 1, exons[i + 1][0] - 1)
            for i in range(len(exons) - 1)
            if exons[i + 1][0] - 1 >= exons[i][1] + 1
        ]
        gi = gene_id.setdefault(gid, len(gene_id))
        eids = [exon_id.setdefault((tid, i), len(exon_id)) for i in range(len(exons))]
        iids = [intron_id.setdefault((tid, i), len(intron_id)) for i in range(len(introns))]
        gene_models.append(
            {
                "exons": exons,
                "introns": introns,
                "cds": cds,
                "gi": gi,
                "exon_ids": eids,
                "intron_ids": iids,
                "span": (g_start, g_end),
            }
        )
        gene_spans.append((g_start, g_end))
        stats["genes"] += 1
        stats["exons"] += len(exons)
        stats["introns"] += len(introns)
        stats["exon_bp"] += sum(e - s + 1 for s, e in exons)
        stats["intron_bp"] += sum(e - s + 1 for s, e in introns)
        stats["cds_bp"] += sum(e - s + 1 for s, e in cds)
    if dry_run:
        return [], dict(stats), 0, N

    rng = random.Random(seed)
    # exon-centered windows sampled across ALL genes' exons (diverse + exon/intron balanced)
    exon_refs = [(gi, ei) for gi, gm in enumerate(gene_models) for ei in range(len(gm["exons"]))]
    rng.shuffle(exon_refs)
    windows, tot = [], 0
    budget_genic = int(max_tokens * (1 - intergenic_frac))
    for gi, ei in exon_refs:
        if tot >= budget_genic:
            break
        gm = gene_models[gi]
        s, e = gm["exons"][ei]
        center = (s - 1 + e) // 2
        w0 = max(0, center - seq_len // 2)
        w1 = min(N, w0 + seq_len)
        if w1 - w0 < 60:
            continue
        win = _label_window(chrom, w0, w1, gm, N)
        if win["dna"].count("N") > 0.5 * len(win["dna"]):
            continue
        windows.append(win)
        tot += w1 - w0
    # intergenic windows: random spots clear of any gene span (+flank)
    spans = sorted(gene_spans)
    tries = 0
    while tot < max_tokens and tries < 20000:
        tries += 1
        w0 = rng.randint(0, N - seq_len)
        w1 = w0 + seq_len
        if any(not (w1 < gs - flank or w0 > ge + flank) for gs, ge in spans):
            continue
        dna = chrom[w0:w1]
        if dna.count("N") > 0.5 * seq_len:
            continue
        lab = {k: np.zeros(seq_len, bool) for k in ("exon", "intron", "cds", "utr", "intergenic")}
        lab["intergenic"][:] = True
        inst = {k: np.full(seq_len, -1, np.int32) for k in ("exon", "intron", "gene")}
        windows.append({"dna": dna, "labels": lab, "instances": inst})
        tot += seq_len
    return windows, dict(stats), tot, N


def main():  # noqa: D103
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=300_000)
    ap.add_argument("--flank", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    windows, stats, tot, N = build_windows(
        args.fasta, args.gff, args.seq_len, args.max_tokens, args.flank, dry_run=args.dry_run
    )
    print(f"chromosome length: {N:,} bp")
    print(f"protein-coding genes used: {stats.get('genes', 0):,}")
    print(f"exons: {stats.get('exons', 0):,}  introns: {stats.get('introns', 0):,}")
    if args.dry_run:
        print(
            f"exon bp: {stats.get('exon_bp', 0):,}  intron bp: {stats.get('intron_bp', 0):,}  cds bp: {stats.get('cds_bp', 0):,}"
        )
        return
    print(f"windows built: {len(windows):,}  total tokens: {tot:,}")
    # coverage over built windows
    cov = defaultdict(int)
    ninst = {k: set() for k in ("exon", "intron", "gene")}
    for w in windows:
        for k, m in w["labels"].items():
            cov[k] += int(m.sum())
        for k in ninst:
            ids = w["instances"][k]
            ninst[k].update(int(x) for x in np.unique(ids) if x >= 0)
    print("per-position coverage (of built windows):")
    for k in ("exon", "intron", "cds", "utr", "intergenic"):
        print(f"   {k:11s} {cov[k]:>9,} ({100 * cov[k] / max(1, tot):5.1f}%)")
    print(f"instances: exons={len(ninst['exon']):,}  introns={len(ninst['intron']):,}  genes={len(ninst['gene']):,}")


if __name__ == "__main__":
    main()
