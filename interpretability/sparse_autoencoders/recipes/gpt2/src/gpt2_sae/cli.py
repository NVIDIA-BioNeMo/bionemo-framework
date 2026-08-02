"""`gpt2-sae` terminal CLI: annotate text, steer a generation, or serve the dashboard.

    gpt2-sae serve [PORT]                              # UI + API (default port 8749)
    gpt2-sae annotate "the cat sat on the mat" -k 8    # top-k features per token
    gpt2-sae generate "I think the best thing to do is" -f 20174:45 -f 21634:40 -n 24

Each `annotate`/`generate` loads the model + SAE once (~a few seconds on GPU, longer on CPU).
"""

from __future__ import annotations

import argparse
import json


def _engine():
    from gpt2_sae.server import engine

    engine.load()
    return engine


def _cmd_annotate(args):
    eng = _engine()
    res = eng.annotate(args.text, mode="topk", k=args.k, feature_ids=None)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    print("tokens:", res["bases"])
    for f in res["features"]:
        print(f"  #{f['feature_id']:<6} max {f['max_activation']:7.2f}  {f['label']}")


def _cmd_generate(args):
    eng = _engine()
    feats = []
    for spec in args.feature or []:
        fid, _, strength = spec.partition(":")
        feats.append({"feature_id": int(fid), "strength": float(strength or 40)})
    res = eng.generate(
        args.prompt,
        feats,
        n_tokens=args.n,
        temperature=args.temperature,
        top_k=args.top_k,
        compare_baseline=args.baseline,
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return
    print("STEERED :", res["generation"]["sequence"])
    if res.get("baseline"):
        print("BASELINE:", res["baseline"]["sequence"])


def _cmd_serve(args):
    import sys

    from gpt2_sae import server

    sys.argv = [sys.argv[0]] + ([str(args.port)] if args.port else [])
    server.main()


def main():
    """Parse args and dispatch to the annotate / generate / serve subcommand."""
    p = argparse.ArgumentParser(
        prog="gpt2-sae", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("annotate", help="top-k SAE features per token of a text")
    a.add_argument("text")
    a.add_argument("-k", type=int, default=8)
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=_cmd_annotate)

    g = sub.add_parser("generate", help="generate while clamping features (feat:strength)")
    g.add_argument("prompt")
    g.add_argument(
        "-f",
        "--feature",
        action="append",
        metavar="ID:STRENGTH",
        help="clamp feature ID to STRENGTH (repeatable); e.g. -f 20174:45",
    )
    g.add_argument("-n", type=int, default=24, help="tokens to generate")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--baseline", action="store_true", help="also show the unsteered generation")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=_cmd_generate)

    s = sub.add_parser("serve", help="serve the dashboard + API")
    s.add_argument("port", nargs="?", type=int, default=None)
    s.set_defaults(func=_cmd_serve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
