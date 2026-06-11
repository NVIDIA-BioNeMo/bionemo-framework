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

"""GPU-free mock of the Evo2-SAE server for dashboard dev, e2e tests, and reviewer demos.

Serves canned /health, /features, /annotate, /generate, /gene_embed responses with the SAME
shapes as `evo2_sae.server` (the live backend), so the dashboard runs fully without a model:

    python feature_explorer/mock_backend.py          # serves :8001, same as `launch_inference.sh serve`

Then point the dashboard at it (the Vite /api proxy already targets :8001). Deterministic
(seeded) so e2e snapshots are stable. Not a substitute for the real server — only the response
contract is real, the numbers are fake.
"""

import argparse
import base64
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


N_FEATURES = 64
LAYER = 26
ORGANISM_TAGS = {"None (raw DNA)": "", "Human": "|d__Eukaryota;...|", "E. coli": "|d__Bacteria;...|"}
LABELS = {0: "motif_ATG", 1: "is_euk_genic", 2: "first_100bp", 5: "GC_rich", 9: "splice_donor"}
_RNG = np.random.default_rng(0)

app = FastAPI(title="Evo2 SAE mock backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnnotateRequest(BaseModel):
    """Mirror of the server's /annotate request body."""

    sequence: str
    organism: str = "None (raw DNA)"
    tag: str | None = None
    mode: str = "topk"
    k: int = 8
    feature_ids: list[int] | None = None
    feature_id: int | None = None


class GenerateRequest(BaseModel):
    """Mirror of the server's /generate request body."""

    prompt: str = ""
    organism: str = "None (raw DNA)"
    tag: str | None = None
    features: list[dict] = []
    n_tokens: int = 120
    temperature: float = 1.0
    top_k: int = 0
    compare_baseline: bool = False


class GeneEmbedRequest(BaseModel):
    """Mirror of the server's /gene_embed request body."""

    genes: list[dict]
    organism: str = "None (raw DNA)"
    tag: str | None = None
    min_firing: int = 10


def _clean(seq: str) -> str:
    return "".join(c for c in (seq or "").upper() if c in "ACGTN")


@app.get("/health")
def health():
    """Canned /health — engine metadata."""
    return {
        "ready": True,
        "layer": LAYER,
        "n_features": N_FEATURES,
        "n_labels": len(LABELS),
        "sae_path": "mock.pt",
        "organisms": list(ORGANISM_TAGS),
        "organism_tags": ORGANISM_TAGS,
        "device": "cpu (mock)",
    }


@app.get("/features")
def features():
    """Canned /features — per-feature labels + natural peaks."""
    return [
        {"id": f, "label": LABELS.get(f), "natural_peak": round(float(_RNG.uniform(1, 5)), 3)}
        for f in range(N_FEATURES)
    ]


@app.post("/annotate")
def annotate(req: AnnotateRequest):
    """Canned /annotate — fake per-base feature activations for the pasted sequence."""
    dna = _clean(req.sequence)
    n = len(dna)
    chosen = (
        (req.feature_ids or ([req.feature_id] if req.feature_id is not None else []))
        if req.mode == "pick"
        else [0, 1, 2, 5]
    )
    feats = []
    for fid in chosen[: req.k]:
        acts = np.clip(_RNG.normal(0.5, 0.4, size=n), 0, None)
        feats.append(
            {
                "feature_id": int(fid),
                "label": LABELS.get(int(fid)),
                "max_activation": round(float(acts.max()) if n else 0.0, 4),
                "activations": [round(float(v), 4) for v in acts],
            }
        )
    return {
        "sequence": dna,
        "organism": req.organism,
        "tag": "",
        "tag_len": 0,
        "bases": list(dna),
        "n_tokens": n,
        "layer": LAYER,
        "features": feats,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    """Canned /generate — random DNA + fake per-feature activation tracks."""
    dna = "".join(_RNG.choice(list("ACGT"), size=req.n_tokens))
    fids = [int(f["feature_id"]) for f in req.features]
    acts = {fid: [round(float(v), 4) for v in np.clip(_RNG.normal(1, 0.5, len(dna)), 0, None)] for fid in fids}
    resp = {
        "prompt": _clean(req.prompt),
        "organism": req.organism,
        "n_tokens": req.n_tokens,
        "features": [
            {
                "id": int(f["feature_id"]),
                "label": LABELS.get(int(f["feature_id"])),
                "strength": float(f.get("strength", 1.0)),
            }
            for f in req.features
        ],
        "steered": bool(req.features),
        "generation": {"sequence": dna, "activations": acts},
        "baseline": None,
    }
    if req.compare_baseline and req.features:
        bdna = "".join(_RNG.choice(list("ACGT"), size=req.n_tokens))
        resp["baseline"] = {
            "sequence": bdna,
            "activations": {
                fid: [round(float(v), 4) for v in np.clip(_RNG.normal(0.5, 0.4, len(bdna)), 0, None)] for fid in fids
            },
        }
    return resp


@app.post("/gene_embed")
def gene_embed(req: GeneEmbedRequest):
    """Canned /gene_embed — fake per-gene feature matrix (base64) for client-side UMAP."""
    genes = [g for g in req.genes if len(_clean(str(g.get("sequence", "")))) >= 3][:1000]
    ng = max(len(genes), 1)
    gmean = np.clip(_RNG.normal(0.3, 0.5, size=(ng, N_FEATURES)), 0, None).astype(np.float32)
    gmax = (gmean + np.clip(_RNG.normal(0.5, 0.3, size=(ng, N_FEATURES)), 0, None)).astype(np.float32)
    n_firing = (gmax > 0).sum(0)
    stats = sorted(
        (
            {
                "feature_id": int(f),
                "n_firing": int(n_firing[f]),
                "mean_act_when_firing": round(
                    float(gmean[:, f][gmean[:, f] > 0].mean()) if (gmean[:, f] > 0).any() else 0.0, 4
                ),
                "max_act": round(float(gmax[:, f].max()), 4),
                "label": LABELS.get(int(f)),
            }
            for f in np.nonzero(n_firing >= req.min_firing)[0]
        ),
        key=lambda s: -s["n_firing"],
    )
    meta = [
        {
            "gene_symbol": g.get("symbol") or g.get("gene_symbol") or f"gene{i}",
            "label": g.get("label"),
            "species": g.get("species"),
        }
        for i, g in enumerate(genes)
    ] or [{"gene_symbol": "gene0", "label": None, "species": None}]
    return {
        "G_b64": base64.b64encode(gmean.tobytes()).decode(),
        "Gmax_b64": base64.b64encode(gmax.tobytes()).decode(),
        "nf": N_FEATURES,
        "ng": ng,
        "meta": meta,
        "stats": stats,
    }


def write_demo_atlas(public_dir: Path) -> None:
    """Write small *fake* atlas parquets into public_dir so the Feature-atlas tab works offline.

    Demo data only (seeded, fabricated) — the real atlas is produced from your SAE elsewhere.
    Matches the schema the dashboard reads: features_atlas (per-feature x/y + stats),
    feature_metadata (per-feature labels/stats), feature_examples (top examples per feature).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    public_dir.mkdir(parents=True, exist_ok=True)
    n = 200
    rng = np.random.default_rng(1)
    fids = list(range(n))
    labels = [LABELS.get(f) for f in fids]
    freq = [round(float(v), 3) for v in rng.uniform(0.01, 0.5, n)]
    maxact = [round(float(v), 3) for v in rng.uniform(1, 8, n)]
    x = [round(float(v), 3) for v in rng.normal(0, 5, n)]
    y = [round(float(v), 3) for v in rng.normal(0, 5, n)]
    pq.write_table(
        pa.table(
            {"feature_id": fids, "x": x, "y": y, "label": labels, "activation_freq": freq, "max_activation": maxact}
        ),
        public_dir / "features_atlas.parquet",
    )
    pq.write_table(
        pa.table({"feature_id": fids, "label": labels, "activation_freq": freq, "max_activation": maxact}),
        public_dir / "feature_metadata.parquet",
    )
    ex = {
        k: []
        for k in (
            "feature_id",
            "example_rank",
            "sequence",
            "activations",
            "max_activation",
            "best_annotation",
            "sequence_id",
            "start",
            "end",
        )
    }
    for f in fids:
        for r in range(3):
            start = int(rng.integers(0, 1_000_000))
            ex["feature_id"].append(f)
            ex["example_rank"].append(r)
            ex["sequence"].append("".join(rng.choice(list("ACGT"), 40)))
            ex["activations"].append([round(float(v), 3) for v in rng.uniform(0, maxact[f], 40)])
            ex["max_activation"].append(round(maxact[f] * float(rng.uniform(0.5, 1)), 3))
            ex["best_annotation"].append(labels[f] or "intergenic")
            ex["sequence_id"].append(f"demo_chr{f % 5}")
            ex["start"].append(start)
            ex["end"].append(start + 40)
    pq.write_table(pa.table(ex), public_dir / "feature_examples.parquet")
    print(f"wrote demo atlas parquets ({n} features) -> {public_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mock Evo2-SAE backend (no GPU)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument(
        "--demo-data",
        action="store_true",
        help="also write fake atlas parquets to ./public so the Feature-atlas tab works offline",
    )
    args = ap.parse_args()
    if args.demo_data:
        write_demo_atlas(Path(__file__).resolve().parent / "public")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
