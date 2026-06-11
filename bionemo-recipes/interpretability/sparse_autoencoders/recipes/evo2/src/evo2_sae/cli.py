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

"""Evo2 SAE inference CLI — one engine, three modes.

    serve   : start the FastAPI server (one sequence at a time, interactive)
    encode  : annotate ONE sequence -> top features (stdout JSON)
    batch   : run a FASTA of MANY sequences -> parquet of per-sequence top features

All three build the same `Evo2SAE` engine; config comes from flags or env
(EVO2_CKPT_DIR / SAE_CKPT_PATH / FEATURE_ANNOTATIONS / EMBEDDING_LAYER).
"""

from __future__ import annotations

import argparse
import json
import os


def _add_common(p: argparse.ArgumentParser) -> None:
    """Register the shared inference arguments (checkpoints, layer, device) on a parser.

    Defaults come from env vars (``EVO2_CKPT_DIR``, ``SAE_CKPT_PATH``, ``FEATURE_ANNOTATIONS``,
    ``EMBEDDING_LAYER``, ``DEVICE``, ``MAX_SEQ_LEN``); pass the flags to override. No hardcoded
    paths — the checkpoints must be supplied via flag or env.

    Args:
        p: The argparse parser (or subparser) to add the shared arguments to.

    Returns:
        None. Mutates ``p`` in place.
    """
    p.add_argument("--evo2-ckpt-dir", default=os.environ.get("EVO2_CKPT_DIR"))
    p.add_argument("--sae-ckpt-path", default=os.environ.get("SAE_CKPT_PATH"))
    p.add_argument("--feature-annotations", default=os.environ.get("FEATURE_ANNOTATIONS"))
    p.add_argument("--layer", type=int, default=int(os.environ.get("EMBEDDING_LAYER", "26")))
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    p.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "8192")))


def _engine(args):
    """Construct an Evo2SAE engine from parsed CLI args.

    Args:
        args: Parsed argparse namespace with ``evo2_ckpt_dir``, ``sae_ckpt_path``, ``layer``,
            ``device``, ``max_seq_len``, ``feature_annotations``.

    Returns:
        An (unloaded) ``Evo2SAE`` instance — call ``.load()`` before use.
    """
    from .core import Evo2SAE

    return Evo2SAE(
        evo2_ckpt_dir=args.evo2_ckpt_dir,
        sae_ckpt_path=args.sae_ckpt_path,
        layer=args.layer,
        device=args.device,
        max_seq_len=args.max_seq_len,
        feature_annotations=args.feature_annotations,
    )


def main():
    """Parse args and dispatch to the serve / encode / batch subcommand."""
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
        feats = eng.top_features(codes, tag_len=tag_len, k=args.top_k)
        print(
            json.dumps(
                {"sequence": dna, "organism": args.organism, "bases": len(dna), "top_features": feats}, indent=2
            )
        )

    elif args.cmd == "batch":
        import pandas as pd

        from .fasta import read_fasta

        ids, seqs = [], []
        for sid, seq in read_fasta(args.fasta):
            ids.append(sid)
            seqs.append(seq)
        print(f"[batch] {len(seqs)} sequences from {args.fasta}; encoding (batch_size={args.batch_size})…")
        codes_list = eng.encode_batch(seqs, batch_size=args.batch_size)
        rows = []
        for sid, codes in zip(ids, codes_list):
            for rank, ft in enumerate(eng.top_features(codes, k=args.top_k)):
                rows.append({"sequence_id": sid, "bp": int(codes.shape[0]), "rank": rank, **ft})
        df = pd.DataFrame(rows)
        df.to_parquet(args.out, index=False)
        print(f"[batch] wrote {len(df)} rows for {len(seqs)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
