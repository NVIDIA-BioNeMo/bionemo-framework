# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compose a prokaryotic + eukaryotic FASTA from OpenGenome2 subsets.

Two sources only:
- Prokaryotic: filtered_metagenomes_pt1 (truncated to --metagenome-window bp/contig)
- Eukaryotic:  eukaryotic_genic_windows (~5kb euk genic regions)

Output headers are renumbered as `>seq_{i} {source}` to satisfy predict_evo2's
unique-id check (the source files share NCBI-style accession headers across
records).
"""

import argparse
import gzip
import subprocess
import sys
from pathlib import Path


def _open_text(path: Path):
    """Open a .fasta or .fasta.gz file in text mode."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _iter_records(fh):
    """Yield (header, seq_lines) from a FASTA file handle."""
    header = None
    lines = []
    for line in fh:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                yield header, lines
            header = line
            lines = []
        elif line:
            lines.append(line)
    if header is not None:
        yield header, lines


def _take_n(fh, n):
    """Take the first n records from fh."""
    for i, (h, ls) in enumerate(_iter_records(fh)):
        if i >= n:
            return
        yield h, ls


def _take_n_truncated(fh, n, max_bp):
    """Take the first n records, truncating each sequence to <= max_bp bases."""
    for i, (h, ls) in enumerate(_iter_records(fh)):
        if i >= n:
            return
        seq = "".join(ls)[:max_bp]
        lines = [seq[j : j + 80] for j in range(0, len(seq), 80)]
        yield h, lines


def main():
    """Compose a prok+euk mixed FASTA with unique seq_{i} headers."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/data/interp/evo2/OpenGenome2/fasta"),
        help="Root dir holding the OpenGenome2 subset directories.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/data/interp/evo2/scratch/mixed_25M_prokeuk.fasta"),
    )
    ap.add_argument("--n-metagenome", type=int, default=1000, help="Metagenome contigs (prok).")
    ap.add_argument("--metagenome-window", type=int, default=50_000, help="Max bp per metagenome contig (truncate).")
    ap.add_argument("--n-euk-windows", type=int, default=10_000, help="Eukaryotic_genic_windows records (euk).")
    args = ap.parse_args()

    metagenome_file = args.root / "metagenomes" / "filtered_metagenomes_pt1.fasta.gz"
    euk_parts = sorted((args.root / "eukaryotic_genic_windows").glob("*.fasta.gz.*"))

    if not metagenome_file.exists():
        print(f"ERROR: missing metagenome source at {metagenome_file}", file=sys.stderr)
        sys.exit(1)
    if not euk_parts:
        print(f"ERROR: no eukaryotic_genic_windows parts under {args.root}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counter = 0
    bp_by_source: dict[str, int] = {}

    def _emit(out, header, lines, source):
        """Write a record with a globally unique header and tally bp."""
        nonlocal counter
        out.write(f">seq_{counter} {source}\n")
        counter += 1
        for line in lines:
            out.write(line + "\n")
        bp_by_source[source] = bp_by_source.get(source, 0) + sum(len(line) for line in lines)

    with open(args.output, "w") as out:
        # 1. Prokaryotic: metagenome contigs, truncated to --metagenome-window each.
        print(f"adding {args.n_metagenome} metagenome contigs (truncated to {args.metagenome_window} bp each)...")
        with _open_text(metagenome_file) as fh:
            for h, ls in _take_n_truncated(fh, args.n_metagenome, args.metagenome_window):
                _emit(out, h, ls, "prok_metagenomes")

        # 2. Eukaryotic: read split parts as one stream.
        print(f"adding {args.n_euk_windows} eukaryotic_genic_windows...")
        cat_parts = subprocess.Popen(
            ["bash", "-c", f"cat {' '.join(str(p) for p in euk_parts)} | zcat"],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            for h, ls in _take_n(cat_parts.stdout, args.n_euk_windows):
                _emit(out, h, ls, "euk_genic_windows")
        finally:
            cat_parts.stdout.close()
            cat_parts.terminate()

    total_bp = sum(bp_by_source.values())
    print(f"\nwrote {counter} sequences, {total_bp:,} total bp -> {args.output}")
    print("by source (bp):")
    for src, bp in bp_by_source.items():
        pct = 100 * bp / total_bp if total_bp else 0
        print(f"  {src:<22} {bp:>12,} bp  ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
