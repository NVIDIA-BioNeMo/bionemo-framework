# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Generate a synthetic genes.tsv for the gene-UMAP precompute pipeline.

500 records with realistic-shaped values:
- gene_symbol: ROOT123-style two-letter prefix + 3 digits
- species: drawn from a small fixed list with plausible frequencies
- sequence: random DNA of length uniform in [2000, 5000]

This is a stand-in for a curated 500-gene list. It exists so the
precompute and frontend can be wired up end-to-end before a real
catalog is delivered. Replace with the real file once available.
"""

import argparse
import random
from pathlib import Path


SPECIES = [
    ("Homo sapiens", 0.40),
    ("Mus musculus", 0.20),
    ("Escherichia coli", 0.15),
    ("Saccharomyces cerevisiae", 0.10),
    ("Drosophila melanogaster", 0.07),
    ("Arabidopsis thaliana", 0.05),
    ("Caenorhabditis elegans", 0.03),
]

# Two-letter gene prefixes loosely sampled from real biology.
PREFIXES = [
    "BRCA", "TP", "EGFR", "MYC", "RAS", "PTEN", "RB", "APC", "ATM",
    "MLH", "CDK", "CCND", "AKT", "PIK3", "NRAS", "KRAS", "HRAS",
    "VEGF", "FGF", "PDGF", "INS", "GLUT", "HSP", "HOX", "FOX",
    "GATA", "PAX", "SOX", "WNT", "NOTCH", "TGFB", "BMP", "ACTB",
    "TUBB", "RPS", "RPL", "EIF", "DNMT", "HDAC", "TET", "EZH",
    "P53", "MDM", "BAX", "BCL", "CASP", "FAS", "TNF", "IL", "IFN",
    "MHC", "HLA", "TLR", "NFKB", "STAT", "JAK", "SRC", "ERK",
]


def _random_dna(rng: random.Random, length: int) -> str:
    """Random uniform A/C/G/T string."""
    return "".join(rng.choices("ACGT", k=length))


def _weighted_choice(rng: random.Random, items):
    """Pick one (item, weight) from a weighted list."""
    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    for item, w in items:
        acc += w
        if r <= acc:
            return item
    return items[-1][0]


def main():
    """Write a 500-row genes.tsv with the columns the precompute expects."""
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("/data/interp/evo2/scratch/fake_genes.tsv"))
    p.add_argument("--n-genes", type=int, default=500)
    p.add_argument("--min-length", type=int, default=2000)
    p.add_argument("--max-length", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    used = set()
    rows = []
    while len(rows) < args.n_genes:
        prefix = rng.choice(PREFIXES)
        num = rng.randint(1, 30)
        symbol = f"{prefix}{num}"
        # Avoid duplicates so downstream code can use symbol as a primary key.
        if symbol in used:
            continue
        used.add(symbol)

        species = _weighted_choice(rng, SPECIES)
        length = rng.randint(args.min_length, args.max_length)
        sequence = _random_dna(rng, length)
        rows.append((symbol, species, sequence))

    with open(args.output, "w") as f:
        f.write("gene_symbol\tspecies\tsequence\n")
        for sym, sp, seq in rows:
            f.write(f"{sym}\t{sp}\t{seq}\n")

    print(f"wrote {len(rows)} genes -> {args.output}")
    print("species distribution:")
    from collections import Counter

    counts = Counter(r[1] for r in rows)
    for sp, n in counts.most_common():
        print(f"  {sp}: {n}")


if __name__ == "__main__":
    main()
