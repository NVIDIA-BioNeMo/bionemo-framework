# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Evo2 SAE inference CLI — one engine, three modes:

    serve   : start the FastAPI server (one sequence at a time, interactive)
    encode  : annotate ONE sequence -> top features (stdout JSON)
    batch   : run a FASTA of MANY sequences -> parquet of per-sequence top features

All three build the same `Evo2SAE` engine; config comes from flags or env
(EVO2_CKPT_DIR / SAE_CKPT_PATH / FEATURE_ANNOTATIONS / EMBEDDING_LAYER)."""

from __future__ import annotations

import argparse
import gzip
import json
import os


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--evo2-ckpt-dir", default=os.environ.get("EVO2_CKPT_DIR", "/data/interp/evo2/checkpoints/evo2_1b_base_mbridge"))
    p.add_argument("--sae-ckpt-path", default=os.environ.get("SAE_CKPT_PATH", "/data/interp/evo2/sae/v2_diverse/layer19_C13_nofilter/checkpoints/checkpoint_final.pt"))
    p.add_argument("--feature-annotations", default=os.environ.get("FEATURE_ANNOTATIONS", "/data/interp/evo2/sae_eval/dashboard_data/l19_C13_nofilter/feature_metadata.parquet"))
    p.add_argument("--layer", type=int, default=int(os.environ.get("EMBEDDING_LAYER", "19")))
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    p.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "8192")))


def _engine(args):
    from .core import Evo2SAE

    return Evo2SAE(
        evo2_ckpt_dir=args.evo2_ckpt_dir, sae_ckpt_path=args.sae_ckpt_path, layer=args.layer,
        device=args.device, max_seq_len=args.max_seq_len, feature_annotations=args.feature_annotations,
    )


def _read_fasta(path: str):
    seqs, ids = [], []
    name, parts = None, []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs.append("".join(parts))
                    ids.append(name)
                name, parts = line[1:].split()[0] if len(line) > 1 else f"seq_{len(ids)}", []
            else:
                parts.append(line)
    if name is not None:
        seqs.append("".join(parts))
        ids.append(name)
    return ids, seqs


def main():
    ap = argparse.ArgumentParser(description="Evo2 SAE inference (serve | encode | batch)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("serve", help="start the FastAPI inference server")
    _add_common(ps)
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")))

    pe = sub.add_parser("encode", help="annotate ONE sequence -> top features (JSON)")
    _add_common(pe)
    pe.add_argument("--sequence", required=True)
    pe.add_argument("--organism", default="None (raw DNA)")
    pe.add_argument("--top-k", type=int, default=8)

    pb = sub.add_parser("batch", help="MANY sequences (FASTA) -> parquet of per-sequence top features")
    _add_common(pb)
    pb.add_argument("--fasta", required=True)
    pb.add_argument("--out", required=True)
    pb.add_argument("--top-k", type=int, default=16)
    pb.add_argument("--batch-size", type=int, default=8)

    args = ap.parse_args()

    if args.cmd == "serve":
        import uvicorn

        from .server import build_app

        uvicorn.run(build_app(_engine(args)), host=args.host, port=args.port, log_level="info")
        return

    from .core import clean_dna

    eng = _engine(args).load()

    if args.cmd == "encode":
        tag = eng.resolve_tag(args.organism, None) or ""
        dna = clean_dna(args.sequence)
        codes = eng.encode(tag + dna)
        tag_len = len(tag) if codes.shape[0] >= len(tag) else 0
        per = codes[tag_len:].max(dim=0).values if codes.shape[0] > tag_len else codes.max(dim=0).values
        top = per.topk(min(args.top_k, per.numel())).indices.tolist()
        feats = [{"feature_id": int(i), "label": eng.labels.get(int(i)), "max_activation": round(float(per[i]), 4)} for i in top if per[i].item() > 0]
        print(json.dumps({"sequence": dna, "organism": args.organism, "bases": len(dna), "top_features": feats}, indent=2))

    elif args.cmd == "batch":
        import pandas as pd

        ids, seqs = _read_fasta(args.fasta)
        print(f"[batch] {len(seqs)} sequences from {args.fasta}; encoding (batch_size={args.batch_size})…")
        codes_list = eng.encode_batch(seqs, batch_size=args.batch_size)
        rows = []
        for sid, codes in zip(ids, codes_list):
            if codes.shape[0] == 0:
                continue
            per = codes.max(dim=0).values
            top = per.topk(min(args.top_k, per.numel())).indices.tolist()
            for rank, fi in enumerate(top):
                v = float(per[fi].item())
                if v <= 0:
                    continue
                rows.append({"sequence_id": sid, "bp": int(codes.shape[0]), "rank": rank,
                             "feature_id": int(fi), "label": eng.labels.get(int(fi)), "max_activation": round(v, 4)})
        df = pd.DataFrame(rows)
        df.to_parquet(args.out, index=False)
        print(f"[batch] wrote {len(df)} rows for {len(seqs)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
