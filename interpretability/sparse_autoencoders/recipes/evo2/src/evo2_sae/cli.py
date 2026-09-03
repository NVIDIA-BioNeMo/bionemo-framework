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

"""Evo2 SAE inference CLI — one engine, four modes.

    serve   : start the FastAPI server (one sequence at a time, interactive)
    encode  : annotate ONE sequence -> top features (stdout JSON)
    batch   : run a FASTA of MANY sequences -> parquet of per-sequence top features
    generate: generate DNA, optionally steering SAE features (stdout JSON)

They all build the same `Evo2SAE` engine; config comes from flags or env
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
    # int() the env defaults explicitly: argparse's type= only coerces values passed on the command
    # line, never the default — so an env-sourced (or absent) value would otherwise stay a str.
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
    ap = argparse.ArgumentParser(description="Evo2 SAE inference (serve | encode | annotate | batch | generate)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("serve", help="start the FastAPI inference server")
    _add_common(ps)
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "8001"))
    )  # int: uvicorn.run needs an int port

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

    pg = sub.add_parser("generate", help="generate DNA, optionally steering SAE features")
    _add_common(pg)
    pg.add_argument("--prompt", default="", help="DNA to seed; steering applies to the continuation")
    pg.add_argument("--organism", default="None (raw DNA)")
    pg.add_argument(
        "--clamp",
        action="append",
        default=[],
        metavar="FEATURE_ID[:STRENGTH]",
        help="clamp a feature on the continuation; repeatable (e.g. --clamp 29244:300). "
        "Find feature ids with `encode`.",
    )
    pg.add_argument("--n-tokens", type=int, default=120)
    pg.add_argument("--temperature", type=float, default=1.0)
    pg.add_argument("--top-k", type=int, default=0)
    pg.add_argument("--compare-baseline", action="store_true", help="also generate unsteered, for comparison")

    pa = sub.add_parser(
        "annotate",
        help="per-base annotation (each feature's activation at every base): ONE --sequence -> JSON "
        "(like /api/annotate), or a --fasta -> parquet",
    )
    _add_common(pa)
    pa.add_argument("--sequence", help="ONE sequence -> JSON on stdout")
    pa.add_argument("--fasta", help="MANY sequences -> parquet (needs --out)")
    pa.add_argument("--out", help="output parquet path (required with --fasta)")
    pa.add_argument("--organism", default="None (raw DNA)")
    pa.add_argument(
        "--top-k", type=int, default=16, help="annotate the top-k features by peak (ignored if --feature-ids)"
    )
    pa.add_argument("--feature-ids", help="comma-separated feature ids to annotate instead of top-k")
    pa.add_argument(
        "--long",
        action="store_true",
        help="batch parquet: one row per (sequence, feature, base) instead of a per-feature activations list column",
    )
    pa.add_argument("--batch-size", type=int, default=8)

    args = ap.parse_args()

    if args.cmd == "serve":
        import uvicorn

        from .server import build_app

        uvicorn.run(build_app(_engine(args)), host=args.host, port=args.port, log_level="info")
        return

    from . import core

    eng = _engine(args).load()

    if args.cmd == "encode":
        try:
            dna, _tag, codes, tag_len = core.annotate(eng, args.sequence, args.organism)
        except ValueError as e:
            raise SystemExit(str(e))
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

    elif args.cmd == "generate":
        try:
            out = eng.generate(
                prompt=args.prompt,
                organism=args.organism,
                features=args.clamp,  # raw "ID[:STRENGTH]" strings; core.parse_clamp_spec normalizes
                n_tokens=args.n_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                compare_baseline=args.compare_baseline,
            )
        except ValueError as e:
            raise SystemExit(str(e))
        result = {
            "prompt": out["prompt"],
            "organism": out["organism"],
            "steered": out["steered"],
            "features": out["features"],
            "sequence": out["generation"]["sequence"],
        }
        if out.get("baseline"):
            result["baseline_sequence"] = out["baseline"]["sequence"]
        print(json.dumps(result, indent=2))

    elif args.cmd == "annotate":
        if bool(args.sequence) == bool(args.fasta):
            raise SystemExit("annotate: pass exactly one of --sequence or --fasta")

        def _pick(codes, tag_len):
            """Features to annotate: explicit --feature-ids, else top-k by peak."""
            if args.feature_ids:
                chosen = [int(i) for i in args.feature_ids.split(",") if i.strip()]
                bad = sorted({i for i in chosen if not (0 <= i < eng.n_features)})
                if bad:
                    raise SystemExit(f"feature_id(s) {bad} out of range [0, {eng.n_features})")
                return [{"feature_id": fid, "label": eng.label_for(fid)} for fid in chosen]
            return eng.top_features(codes, tag_len=tag_len, k=args.top_k)

        def _track(codes, tag_len, fid):
            """Per-base activation of feature `fid` over the DNA region (tag stripped)."""
            region = codes[tag_len:] if codes.shape[0] > tag_len else codes
            return [round(float(v), 4) for v in region[:, fid].tolist()]

        if args.sequence:  # ONE sequence -> JSON with per-base tracks (mirrors /api/annotate)
            try:
                dna, _tag, codes, tag_len = core.annotate(eng, args.sequence, args.organism)
            except ValueError as e:
                raise SystemExit(str(e))
            feats = []
            for f in _pick(codes, tag_len):
                track = _track(codes, tag_len, f["feature_id"])
                feats.append(
                    {
                        "feature_id": f["feature_id"],
                        "label": f.get("label"),
                        "max_activation": round(max(track), 4) if track else 0.0,
                        "activations": track,
                    }
                )
            print(
                json.dumps(
                    {
                        "sequence": dna,
                        "organism": args.organism,
                        "tag_len": tag_len,
                        "bases": len(dna),
                        "layer": eng.layer,
                        "features": feats,
                    },
                    indent=2,
                )
            )
        else:  # FASTA -> parquet (per-base tracks; raw/untagged, like `batch`)
            if not args.out:
                raise SystemExit("annotate --fasta requires --out")
            import pandas as pd

            from .fasta import read_fasta

            ids, seqs = [], []
            for sid, seq in read_fasta(args.fasta):
                ids.append(sid)
                seqs.append(seq)
            print(f"[annotate] {len(seqs)} sequences from {args.fasta}; encoding (batch_size={args.batch_size})…")
            codes_list = eng.encode_batch(seqs, batch_size=args.batch_size)  # raw: no organism tag
            rows = []
            for sid, codes in zip(ids, codes_list):
                for f in _pick(codes, 0):
                    fid = f["feature_id"]
                    track = _track(codes, 0, fid)
                    if args.long:
                        for pos, val in enumerate(track):
                            rows.append(
                                {
                                    "sequence_id": sid,
                                    "position": pos,
                                    "feature_id": fid,
                                    "label": f.get("label"),
                                    "activation": val,
                                }
                            )
                    else:
                        rows.append(
                            {
                                "sequence_id": sid,
                                "bp": int(codes.shape[0]),
                                "feature_id": fid,
                                "label": f.get("label"),
                                "max_activation": round(max(track), 4) if track else 0.0,
                                "activations": track,
                            }
                        )
            df = pd.DataFrame(rows)
            df.to_parquet(args.out, index=False)
            shape = "long" if args.long else "per-feature"
            print(f"[annotate] wrote {len(df)} rows for {len(seqs)} sequences -> {args.out} ({shape})")


if __name__ == "__main__":
    main()
