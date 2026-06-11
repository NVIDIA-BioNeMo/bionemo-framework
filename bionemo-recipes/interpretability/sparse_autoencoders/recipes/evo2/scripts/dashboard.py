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

"""Generate the Feature-atlas dashboard data from an SAE + a FASTA corpus.

Runs the corpus through the same ``Evo2SAE`` engine the server uses, then writes the
three parquet files ``launch_dashboard.py --data-dir`` stages into ``public/``:

    features_atlas.parquet   1 row / feature: feature_id, x, y (UMAP), label,
                             activation_freq, max_activation  -> the scatter points
    feature_metadata.parquet 1 row / feature: feature_id, label, activation_freq,
                             max_activation                   -> the catalog
    feature_examples.parquet N rows / feature: top-activating sequences with the
                             per-base activation track        -> the example cards

The heavy lifting is reused, not reimplemented: ``encode_batch`` (engine) for the
activations and ``sae.analysis.compute_feature_umap`` for the 2-D layout.

Memory is bounded by a two-pass scheme (mirrors the codonfm generator): pass 1 keeps
only the per-(sequence, feature) max to pick top examples; pass 2 re-encodes just the
sequences that won, to pull their per-base tracks.

Example:
    python scripts/dashboard.py \
        --evo2-ckpt-dir $EVO2_CKPT_DIR --sae-ckpt-path $SAE_CKPT_PATH \
        --feature-annotations $FEATURE_ANNOTATIONS --layer 26 \
        --fasta corpus.fa --output-dir dashboard_data
    # then: python scripts/launch_dashboard.py --data-dir dashboard_data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def parse_args():
    """Parse engine config (flags or env) + corpus / output / UMAP options."""
    p = argparse.ArgumentParser(description="Generate Feature-atlas dashboard parquets from an SAE + FASTA")
    # Engine config — same env defaults as the inference CLI.
    p.add_argument("--evo2-ckpt-dir", default=os.environ.get("EVO2_CKPT_DIR"))
    p.add_argument("--sae-ckpt-path", default=os.environ.get("SAE_CKPT_PATH"))
    p.add_argument("--feature-annotations", default=os.environ.get("FEATURE_ANNOTATIONS"))
    p.add_argument("--layer", type=int, default=int(os.environ.get("EMBEDDING_LAYER", "26")))
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    p.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "8192")))
    # Corpus + output.
    p.add_argument("--fasta", required=True, help="FASTA corpus to characterize features over")
    p.add_argument("--output-dir", required=True, help="Directory to write the 3 parquets into")
    p.add_argument("--organism", default="None (raw DNA)", help="Phylo-tag preset to prepend (default: raw DNA)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-examples", type=int, default=6, help="Top examples per feature")
    p.add_argument(
        "--max-example-bp",
        type=int,
        default=0,
        help="Trim each stored example to this many bp around its peak (0 = keep full sequence)",
    )
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    return p.parse_args()


def _pass1_max_acts(eng, seqs, tag, tag_len, batch_size):
    """Pass 1: per-(sequence, feature) max activation, keeping only [n_seq, n_features].

    Encodes in chunks and immediately reduces each sequence's per-base codes to a max
    over positions (dropping the per-base tensors) so memory stays bounded by the result.
    """
    n_seq, n_features = len(seqs), eng.n_features
    max_acts = torch.zeros(n_seq, n_features, dtype=torch.float32)
    for start in range(0, n_seq, batch_size):
        chunk = seqs[start : start + batch_size]
        codes_list = eng.encode_batch([tag + s for s in chunk], batch_size=batch_size)
        for j, codes in enumerate(codes_list):
            region = codes[tag_len:] if codes.shape[0] > tag_len else codes
            if region.shape[0]:
                max_acts[start + j] = region.max(dim=0).values
        print(f"  pass 1: {min(start + batch_size, n_seq)}/{n_seq} sequences", end="\r")
    print()
    return max_acts


def _pass2_examples(eng, seqs, ids, tag, tag_len, top_idx, labels, max_example_bp, batch_size):
    """Pass 2: re-encode only the winning sequences, build the feature_examples rows.

    ``top_idx`` is [n_examples, n_features] sequence indices from topk over pass-1 maxima.
    Each (feature, rank) row gets the DNA substring + aligned per-base activation track,
    optionally trimmed to a window around the per-feature peak.
    """
    n_examples, n_features = top_idx.shape
    # Which sequences need re-encoding, and for which features.
    needed: dict[int, set[int]] = {}
    for f in range(n_features):
        for r in range(n_examples):
            needed.setdefault(int(top_idx[r, f]), set()).add(f)

    # Re-encode the needed sequences -> {seq_idx: codes[region_len, n_features]} (region excludes tag).
    codes_by_seq: dict[int, torch.Tensor] = {}
    need_ids = sorted(needed)
    for start in range(0, len(need_ids), batch_size):
        batch_ids = need_ids[start : start + batch_size]
        codes_list = eng.encode_batch([tag + seqs[i] for i in batch_ids], batch_size=batch_size)
        for i, codes in zip(batch_ids, codes_list):
            codes_by_seq[i] = codes[tag_len:] if codes.shape[0] > tag_len else codes
        print(f"  pass 2: {min(start + batch_size, len(need_ids))}/{len(need_ids)} sequences", end="\r")
    print()

    rows = []
    for f in range(n_features):
        for rank in range(n_examples):
            seq_idx = int(top_idx[rank, f])
            codes = codes_by_seq.get(seq_idx)
            if codes is None or codes.shape[0] == 0:
                continue
            track = codes[:, f]
            dna = seqs[seq_idx][: codes.shape[0]]  # align to encoded region length
            lo, hi = 0, codes.shape[0]
            if max_example_bp and codes.shape[0] > max_example_bp:
                peak = int(track.argmax())
                lo = max(0, peak - max_example_bp // 2)
                hi = min(codes.shape[0], lo + max_example_bp)
                lo = max(0, hi - max_example_bp)
            acts = [round(float(v), 4) for v in track[lo:hi].tolist()]
            rows.append(
                {
                    "feature_id": f,
                    "example_rank": rank,
                    "sequence_id": ids[seq_idx],
                    "start": lo,
                    "end": hi,
                    "sequence": dna[lo:hi],
                    "activations": acts,
                    "max_activation": max(acts) if acts else 0.0,
                    "best_annotation": labels.get(f, ""),
                }
            )
    rows.sort(key=lambda r: (r["feature_id"], r["example_rank"]))
    return rows


def _write_parquets(out_dir, n_features, freq, peak, geom, labels, example_rows):
    """Write the 3 parquets in the exact schema the dashboard's DuckDB queries read."""
    import math

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    fids = list(range(n_features))
    lbls = [labels.get(f, f"Feature {f}") for f in fids]
    freq_l = [float(freq[f]) for f in fids]
    peak_l = [float(peak[f]) for f in fids]

    # features_atlas.parquet — scatter points (UMAP x/y) + stats.
    atlas = pa.table(
        {
            "feature_id": pa.array(fids, type=pa.int32()),
            "x": pa.array([float(v) for v in geom.umap_x], type=pa.float32()),
            "y": pa.array([float(v) for v in geom.umap_y], type=pa.float32()),
            "label": pa.array(lbls),
            "activation_freq": pa.array(freq_l, type=pa.float32()),
            "max_activation": pa.array(peak_l, type=pa.float32()),
            "log_frequency": pa.array([math.log10(v) if v > 0 else -10.0 for v in freq_l], type=pa.float32()),
        }
    )
    pq.write_table(atlas, out_dir / "features_atlas.parquet", compression="snappy")

    # feature_metadata.parquet — the catalog.
    meta = pa.table(
        {
            "feature_id": pa.array(fids, type=pa.int32()),
            "label": pa.array(lbls),
            "activation_freq": pa.array(freq_l, type=pa.float32()),
            "max_activation": pa.array(peak_l, type=pa.float32()),
        }
    )
    pq.write_table(meta, out_dir / "feature_metadata.parquet", compression="snappy")

    # feature_examples.parquet — top sequences + per-base tracks (sorted by feature_id).
    examples = pa.table(
        {
            "feature_id": pa.array([r["feature_id"] for r in example_rows], type=pa.int32()),
            "example_rank": pa.array([r["example_rank"] for r in example_rows], type=pa.int8()),
            "sequence_id": pa.array([r["sequence_id"] for r in example_rows]),
            "start": pa.array([r["start"] for r in example_rows], type=pa.int32()),
            "end": pa.array([r["end"] for r in example_rows], type=pa.int32()),
            "sequence": pa.array([r["sequence"] for r in example_rows]),
            "activations": pa.array([r["activations"] for r in example_rows], type=pa.list_(pa.float32())),
            "max_activation": pa.array([r["max_activation"] for r in example_rows], type=pa.float32()),
            "best_annotation": pa.array([r["best_annotation"] for r in example_rows]),
        }
    )
    pq.write_table(examples, out_dir / "feature_examples.parquet", row_group_size=600, compression="snappy")


def main():
    """Build the engine, run the corpus, and write the 3 dashboard parquets."""
    args = parse_args()  # before heavy imports so --help works without the model stack

    from evo2_sae.core import Evo2SAE, clean_dna
    from evo2_sae.fasta import read_fasta
    from sae.analysis import compute_feature_umap

    out_dir = Path(args.output_dir)

    eng = Evo2SAE(
        evo2_ckpt_dir=args.evo2_ckpt_dir,
        sae_ckpt_path=args.sae_ckpt_path,
        layer=args.layer,
        device=args.device,
        max_seq_len=args.max_seq_len,
        feature_annotations=args.feature_annotations,
    ).load()

    ids, seqs = [], []
    for sid, seq in read_fasta(args.fasta):
        ids.append(sid)
        seqs.append(clean_dna(seq))
    if not seqs:
        raise SystemExit(f"No sequences read from {args.fasta}")
    tag = eng.resolve_tag(args.organism, None) or ""
    # tag_len in *encoded* positions; clamp so a too-long tag never drops the whole sequence.
    tag_len = len(tag) if tag else 0
    print(f"[dashboard] {len(seqs)} sequences, {eng.n_features} features, organism={args.organism!r}")

    # Pass 1: per-feature stats from per-(sequence, feature) maxima.
    max_acts = _pass1_max_acts(eng, seqs, tag, tag_len, args.batch_size)
    freq = (max_acts > 0).float().mean(dim=0)  # fraction of sequences in which the feature fires
    peak = max_acts.max(dim=0).values

    # Top examples per feature -> [n_examples, n_features].
    k = min(args.n_examples, len(seqs))
    top_idx = torch.topk(max_acts, k=k, dim=0).indices

    # Pass 2: per-base tracks for the winning sequences.
    example_rows = _pass2_examples(
        eng, seqs, ids, tag, tag_len, top_idx, eng.labels, args.max_example_bp, args.batch_size
    )

    # UMAP layout from the SAE decoder directions (reused from the shared sae package).
    print("[dashboard] computing UMAP layout...")
    geom = compute_feature_umap(
        eng.sae,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        compute_clusters=False,
    )

    _write_parquets(out_dir, eng.n_features, freq, peak, geom, eng.labels, example_rows)
    print(
        f"[dashboard] wrote {eng.n_features} features + {len(example_rows)} examples -> {out_dir}\n"
        f"            launch with: python scripts/launch_dashboard.py --data-dir {out_dir}"
    )


if __name__ == "__main__":
    main()
