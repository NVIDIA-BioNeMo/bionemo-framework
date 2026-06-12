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

"""Generate the Feature-atlas dashboard parquets via two subcommands, split by cost.

    atlas    features_atlas.parquet + feature_metadata.parquet
             Stats from a RANDOM SAMPLE of cached layer activations (an extract.py store)
             run through the SAE; UMAP x/y from the SAE decoder. Loads ONLY the SAE — no
             Evo2 7B, no megatron. This is the same activation store the SAE trained on.

    examples feature_examples.parquet
             Top-activating sequences + per-base tracks. Needs sequence-aligned activations
             (which the anonymous token-level cache can't give), so this one loads the full
             Evo2SAE engine (7B -> SAE) over a SMALL --examples-fasta.

Why the split: the expensive 7B forward pass already ran once (extract.py, reused for SAE
training). The atlas only needs feature firing-rates + decoder geometry, so it samples that
cache through the SAE — cheap and representative. Only the example cards inherently need a
fresh forward pass, and that's a small, bounded job.

Feature *labels* are produced elsewhere — by the feature-probing / label-producer pipeline
(PR #1630), read from --feature-annotations and joined into `label`; unlabeled -> "Feature N".

Example:
    # atlas — point at the SAE's training activation store (no 7B):
    python scripts/dashboard.py atlas --sae-ckpt-path $SAE_CKPT_PATH \
        --feature-annotations $FEATURE_ANNOTATIONS \
        --activations-dir /path/to/activation_store --output-dir dashboard_data

    # examples — small corpus through the engine (loads the 7B):
    python scripts/dashboard.py examples --evo2-ckpt-dir $EVO2_CKPT_DIR \
        --sae-ckpt-path $SAE_CKPT_PATH --feature-annotations $FEATURE_ANNOTATIONS \
        --examples-fasta small_corpus.fa --output-dir dashboard_data

    python scripts/launch_dashboard.py --data-dir dashboard_data   # then view
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch


def _add_sae_args(p):
    """Args common to both modes: the SAE, labels, layer, device, output, UMAP knobs."""
    p.add_argument("--sae-ckpt-path", default=os.environ.get("SAE_CKPT_PATH"))
    p.add_argument("--feature-annotations", default=os.environ.get("FEATURE_ANNOTATIONS"))
    p.add_argument("--layer", type=int, default=int(os.environ.get("EMBEDDING_LAYER", "26")))
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    p.add_argument("--output-dir", required=True, help="Directory to write the parquet(s) into")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)


def parse_args():
    """Parse the `atlas` / `examples` subcommand and its options."""
    ap = argparse.ArgumentParser(description="Generate Feature-atlas dashboard parquets (atlas | examples)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa_ = sub.add_parser("atlas", help="features_atlas + feature_metadata from cached activations (no 7B)")
    _add_sae_args(pa_)
    pa_.add_argument("--activations-dir", required=True, help="extract.py activation store (shard_*.parquet)")
    pa_.add_argument("--sample-tokens", type=int, default=2_000_000, help="random tokens to sample from the store")
    pa_.add_argument("--batch-size", type=int, default=8192, help="SAE-encode batch (rows)")
    pa_.add_argument("--seed", type=int, default=0)
    pa_.add_argument(
        "--layout",
        choices=["auto", "umap", "pca", "tsne"],
        default="auto",
        help="2-D layout: auto = umap if importable else pca (umap needs NumPy<=2.3; pca/tsne are numba-free)",
    )

    pe = sub.add_parser("examples", help="feature_examples from a small FASTA (loads the 7B)")
    _add_sae_args(pe)
    pe.add_argument("--evo2-ckpt-dir", default=os.environ.get("EVO2_CKPT_DIR"))
    pe.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "8192")))
    pe.add_argument("--examples-fasta", required=True, help="SMALL representative FASTA (a few hundred seqs)")
    pe.add_argument("--max-sequences", type=int, default=1000, help="Cap sequences read (keep it small)")
    pe.add_argument("--organism", default="None (raw DNA)", help="Phylo-tag preset to prepend")
    pe.add_argument("--batch-size", type=int, default=4)
    pe.add_argument("--n-examples", type=int, default=6, help="Top examples per feature")
    pe.add_argument("--max-example-bp", type=int, default=256, help="Window each example to N bp around its peak")
    return ap.parse_args()


# --------------------------------------------------------------------------- shared
def _load_sae_only(args):
    """Load just the SAE (+ labels) by reusing the engine's loaders — no 7B / megatron.

    ``Evo2SAE.__init__`` only records config; ``_load_sae``/``_load_feature_meta`` touch the
    SAE checkpoint and annotation parquet but never ``bionemo.evo2``, so this stays light.
    """
    from evo2_sae.core import Evo2SAE

    eng = Evo2SAE(
        evo2_ckpt_dir="",
        sae_ckpt_path=args.sae_ckpt_path,
        layer=args.layer,
        device=args.device,
        feature_annotations=args.feature_annotations,
    )
    sae, n_features = eng._load_sae()
    labels, _ = eng._load_feature_meta()
    return sae, n_features, labels


def _write_label_columns(n_features, labels):
    """(feature_ids, label_list) with the #1630 labels joined in, 'Feature N' otherwise."""
    fids = list(range(n_features))
    return fids, [labels.get(f, f"Feature {f}") for f in fids]


# --------------------------------------------------------------------------- atlas mode
def _iter_sampled_activations(shards, sample_tokens, batch_size):
    """Yield ``[<=batch_size, hidden_dim]`` CPU tensors sampled from random activation shards."""
    import numpy as np
    import pyarrow.parquet as pq

    seen = 0
    for path in shards:
        if seen >= sample_tokens:
            return
        tbl = pq.read_table(path)
        dims = [c for c in tbl.column_names if c.startswith("dim_")]
        arr = np.stack([tbl.column(c).to_numpy(zero_copy_only=False) for c in dims], axis=1)
        for i in range(0, arr.shape[0], batch_size):
            if seen >= sample_tokens:
                return
            chunk = arr[i : i + batch_size]
            seen += chunk.shape[0]
            yield torch.from_numpy(chunk).float()


def _compute_layout(sae, args):
    """2-D feature layout from the SAE decoder directions -> (x, y, method_used).

    `--layout auto` uses UMAP when importable (codonfm-style, best clusters) and otherwise
    falls back to PCA — so the atlas runs single-env: UMAP needs numba (NumPy <= 2.3), while
    PCA/TSNE are numba-free and run in the megatron venv (NumPy 2.5). UMAP reads the decoder
    itself; PCA/TSNE operate on the L2-normalized decoder columns.
    """
    import numpy as np

    method = args.layout
    if method == "auto":
        try:
            import umap  # noqa: F401

            method = "umap"
        except Exception:
            method = "pca"
            print("[atlas] umap-learn unavailable (NumPy/numba) — falling back to --layout pca")

    if method == "umap":
        from sae.analysis import compute_feature_umap

        geom = compute_feature_umap(
            sae, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, compute_clusters=False
        )
        return np.asarray(geom.umap_x), np.asarray(geom.umap_y), "umap"

    # Numba-free paths operate on L2-normalized decoder columns (one per feature).
    w = sae.decoder.weight.detach().cpu().float().numpy().T  # [n_features, dim]
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
    if method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        reduced = PCA(n_components=min(50, w.shape[1]), random_state=args.seed).fit_transform(w)
        xy = TSNE(n_components=2, init="pca", random_state=args.seed).fit_transform(reduced)
        return xy[:, 0], xy[:, 1], "tsne"

    # pca (default fallback) — top-2 principal directions via torch.
    _, _, v = torch.pca_lowrank(torch.from_numpy(w), q=2)
    xy = (torch.from_numpy(w) @ v).numpy()
    return xy[:, 0], xy[:, 1], "pca"


def run_atlas(args):
    """features_atlas + feature_metadata from a random sample of the cached activation store."""
    import math

    import pyarrow as pa
    import pyarrow.parquet as pq

    sae, n_features, labels = _load_sae_only(args)
    shards = sorted(Path(args.activations_dir).glob("shard_*.parquet"))
    if not shards:
        raise SystemExit(
            f"No activation store in {args.activations_dir!r}. Generate one with extract.py "
            f"(the same activations your SAE trained on):\n"
            f"  torchrun --nproc_per_node 8 scripts/extract.py --ckpt-dir $EVO2_CKPT_DIR "
            f"--embedding-layer {args.layer} --fasta corpus.fa --activation-store-dir {args.activations_dir}"
        )
    random.Random(args.seed).shuffle(shards)

    # Streaming, vectorized firing-rate + peak over a random token sample (no 7B). We hand-roll
    # these instead of sae.analysis.compute_feature_stats: we need only freq/max, while that
    # helper also builds per-feature top-example heaps over anonymous token indices we can't use.
    device = args.device
    sae.eval().to(device)
    fire = torch.zeros(n_features, device=device)
    peak = torch.zeros(n_features, device=device)
    total = 0
    with torch.no_grad():
        for batch in _iter_sampled_activations(shards, args.sample_tokens, args.batch_size):
            codes = sae.encode(batch.to(device))
            fire += (codes > 0).sum(dim=0).float()
            peak = torch.maximum(peak, codes.max(dim=0).values)
            total += codes.shape[0]
            print(f"  atlas: sampled {total:,}/{args.sample_tokens:,} tokens", end="\r")
    print()
    if total == 0:
        raise SystemExit("Sampled 0 tokens — is the activation store empty?")
    freq = (fire / total).cpu()
    peak = peak.cpu()

    x, y, used = _compute_layout(sae, args)
    print(f"[atlas] layout: {used}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fids, lbls = _write_label_columns(n_features, labels)
    freq_l = [float(v) for v in freq.tolist()]
    peak_l = [float(v) for v in peak.tolist()]
    cols = {
        "feature_id": pa.array(fids, type=pa.int32()),
        "label": pa.array(lbls),
        "activation_freq": pa.array(freq_l, type=pa.float32()),
        "max_activation": pa.array(peak_l, type=pa.float32()),
    }
    atlas = pa.table(
        {
            **cols,
            "x": pa.array([float(v) for v in x], type=pa.float32()),
            "y": pa.array([float(v) for v in y], type=pa.float32()),
            "log_frequency": pa.array([math.log10(v) if v > 0 else -10.0 for v in freq_l], type=pa.float32()),
        }
    )
    pq.write_table(atlas, out / "features_atlas.parquet", compression="snappy")
    pq.write_table(pa.table(cols), out / "feature_metadata.parquet", compression="snappy")
    live = int((peak > 0).sum())
    print(
        f"[atlas] wrote features_atlas + feature_metadata ({n_features} features, {live} live, {total:,} tokens) -> {out}"
    )


# --------------------------------------------------------------------------- examples mode
def _pass1_max_acts(eng, seqs, tag, tag_len, batch_size):
    """Per-(sequence, feature) max activation -> [n_seq, n_features], reducing each batch eagerly."""
    n_seq, n_features = len(seqs), eng.n_features
    max_acts = torch.zeros(n_seq, n_features, dtype=torch.float32)
    for start in range(0, n_seq, batch_size):
        chunk = seqs[start : start + batch_size]
        codes_list = eng.encode_batch([tag + s for s in chunk], batch_size=batch_size)
        for j, codes in enumerate(codes_list):
            region = codes[tag_len:] if codes.shape[0] > tag_len else codes
            if region.shape[0]:
                max_acts[start + j] = region.max(dim=0).values
        print(f"  examples pass 1: {min(start + batch_size, n_seq)}/{n_seq} sequences", end="\r")
    print()
    return max_acts


def _pass2_examples(eng, seqs, ids, tag, tag_len, top_idx, peak, labels, max_example_bp, batch_size):
    """Re-encode only the winning sequences and pull each example's windowed per-base track.

    Memory-bounded: per (seq, feature) we extract just that feature's column and immediately
    materialize a short windowed list (``.tolist()`` detaches it), so the full ``[S, n_features]``
    code tensor is freed each batch — never accumulated. Dead features (peak == 0) are skipped.
    """
    n_examples, n_features = top_idx.shape
    alive = [f for f in range(n_features) if peak[f] > 0]
    needed: dict[int, set[int]] = {}
    for f in alive:
        for r in range(n_examples):
            needed.setdefault(int(top_idx[r, f]), set()).add(f)

    win: dict[tuple, tuple] = {}  # (seq_idx, feat) -> (lo, hi, [activations])
    need_ids = sorted(needed)
    for start in range(0, len(need_ids), batch_size):
        batch_ids = need_ids[start : start + batch_size]
        codes_list = eng.encode_batch([tag + seqs[i] for i in batch_ids], batch_size=batch_size)
        for i, codes in zip(batch_ids, codes_list):
            region = codes[tag_len:] if codes.shape[0] > tag_len else codes
            for f in needed[i]:
                track = region[:, f]
                lo, hi = 0, track.shape[0]
                if max_example_bp and hi > max_example_bp:
                    pk = int(track.argmax())
                    lo = max(0, pk - max_example_bp // 2)
                    hi = min(track.shape[0], lo + max_example_bp)
                    lo = max(0, hi - max_example_bp)
                win[(i, f)] = (lo, hi, [round(float(v), 4) for v in track[lo:hi].tolist()])
        print(f"  examples pass 2: re-encoded {min(start + batch_size, len(need_ids))}/{len(need_ids)} seqs", end="\r")
    print()

    rows = []
    for f in alive:
        for rank in range(n_examples):
            si = int(top_idx[rank, f])
            w = win.get((si, f))
            if w is None:
                continue
            lo, hi, acts = w
            if not acts or max(acts) <= 0:
                continue
            rows.append(
                {
                    "feature_id": f,
                    "example_rank": rank,
                    "sequence_id": ids[si],
                    "start": lo,
                    "end": hi,
                    "sequence": seqs[si][lo:hi],
                    "activations": acts,
                    "max_activation": max(acts),
                    "best_annotation": labels.get(f, ""),
                }
            )
    rows.sort(key=lambda r: (r["feature_id"], r["example_rank"]))
    return rows


def run_examples(args):
    """feature_examples from a small corpus run through the full Evo2SAE engine (loads the 7B)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from evo2_sae.core import Evo2SAE, clean_dna
    from evo2_sae.fasta import read_fasta

    eng = Evo2SAE(
        evo2_ckpt_dir=args.evo2_ckpt_dir,
        sae_ckpt_path=args.sae_ckpt_path,
        layer=args.layer,
        device=args.device,
        max_seq_len=args.max_seq_len,
        feature_annotations=args.feature_annotations,
    ).load()

    ids, seqs = [], []
    for sid, seq in read_fasta(args.examples_fasta):
        if len(seqs) >= args.max_sequences:
            break
        ids.append(sid)
        seqs.append(clean_dna(seq))
    if not seqs:
        raise SystemExit(f"No sequences read from {args.examples_fasta}")
    tag = eng.resolve_tag(args.organism, None) or ""
    tag_len = len(tag) if tag else 0
    print(f"[examples] {len(seqs)} sequences, {eng.n_features} features, organism={args.organism!r}")

    max_acts = _pass1_max_acts(eng, seqs, tag, tag_len, args.batch_size)
    peak = max_acts.max(dim=0).values
    k = min(args.n_examples, len(seqs))
    top_idx = torch.topk(max_acts, k=k, dim=0).indices
    rows = _pass2_examples(
        eng, seqs, ids, tag, tag_len, top_idx, peak, eng.labels, args.max_example_bp, args.batch_size
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tbl = pa.table(
        {
            "feature_id": pa.array([r["feature_id"] for r in rows], type=pa.int32()),
            "example_rank": pa.array([r["example_rank"] for r in rows], type=pa.int8()),
            "sequence_id": pa.array([r["sequence_id"] for r in rows]),
            "start": pa.array([r["start"] for r in rows], type=pa.int32()),
            "end": pa.array([r["end"] for r in rows], type=pa.int32()),
            "sequence": pa.array([r["sequence"] for r in rows]),
            "activations": pa.array([r["activations"] for r in rows], type=pa.list_(pa.float32())),
            "max_activation": pa.array([r["max_activation"] for r in rows], type=pa.float32()),
            "best_annotation": pa.array([r["best_annotation"] for r in rows]),
        }
    )
    pq.write_table(tbl, out / "feature_examples.parquet", row_group_size=600, compression="snappy")
    print(f"[examples] wrote {len(rows)} example rows for {int((peak > 0).sum())} live features -> {out}")


def main():
    """Dispatch to the atlas / examples subcommand."""
    args = parse_args()  # before heavy imports so --help works without the model stack
    if args.cmd == "atlas":
        run_atlas(args)
    else:
        run_examples(args)


if __name__ == "__main__":
    main()
