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

"""FastAPI server over the Evo2SAE engine — the live backend the viz talks to.

Endpoints: /health, /features, /annotate (per-base activations for a pasted
sequence), /generate (autoregressive generation + optional SAE-feature clamp).
This is a thin layer; all model work lives in `core.Evo2SAE`.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core import Evo2SAE, clean_dna


logger = logging.getLogger("evo2_sae_infer.server")


class AnnotateRequest(BaseModel):
    """Request body for /annotate (top-k feature scan or an explicit feature pick)."""

    sequence: str
    organism: str = "None (raw DNA)"
    tag: Optional[str] = None
    mode: str = "topk"  # "topk" | "pick"
    k: int = 8
    feature_ids: Optional[list[int]] = None
    feature_id: Optional[int] = None


class FeatureClamp(BaseModel):
    """A single SAE-feature steering clamp (feature id + target strength)."""

    feature_id: int
    strength: float = 1.0


class GenerateRequest(BaseModel):
    """Request body for /generate (autoregressive generation + optional SAE-feature clamps)."""

    prompt: str = ""
    organism: str = "None (raw DNA)"
    tag: Optional[str] = None
    features: list[FeatureClamp] = []
    n_tokens: int = 120
    temperature: float = 1.0
    top_k: int = 0
    compare_baseline: bool = False


class GeneEmbedRequest(BaseModel):
    """Request body for /gene_embed (embed many sequences into per-feature vectors for UMAP)."""

    genes: list[dict]  # [{symbol, sequence, label?, species?}, ...]
    organism: str = "None (raw DNA)"
    tag: Optional[str] = None
    min_firing: int = 10  # feature_stats keeps features firing in >= this many sequences


def build_app(engine: Evo2SAE) -> FastAPI:
    """Build the FastAPI app; the engine is loaded once in the lifespan handler."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            engine.load()
            logger.info("engine ready")
        except Exception:
            logger.exception("engine startup failed — /health stays not-ready")
        yield

    app = FastAPI(title="Evo2 SAE inference", lifespan=lifespan)
    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")  # comma-separated; "*" by default (local backend)
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"], allow_headers=["*"])

    def _require_ready():
        if not engine.ready:
            raise HTTPException(503, "Backend not ready")

    @app.get("/health")
    def health():
        return {
            "ready": bool(engine.ready),
            "layer": engine.layer,
            "n_features": engine.n_features,
            "n_labels": len(engine.labels),
            "sae_path": engine.sae_ckpt_path,
            "organisms": list(engine.organism_tags.keys()),
            "organism_tags": engine.organism_tags,
            "device": engine.device,
        }

    @app.get("/features")
    def features():
        _require_ready()
        rows = [
            {"id": int(f), "label": lab, "natural_peak": engine.peaks.get(int(f))} for f, lab in engine.labels.items()
        ]
        rows.sort(key=lambda r: r["id"])
        return rows

    @app.post("/annotate")
    def annotate(req: AnnotateRequest):
        _require_ready()
        dna = clean_dna(req.sequence)
        if not dna:
            raise HTTPException(400, "No valid nucleotides in sequence")
        tag = engine.resolve_tag(req.organism, req.tag)
        if tag is None:
            raise HTTPException(400, f"Unknown organism '{req.organism}' and no custom tag")
        full = tag + dna
        tag_len = len(tag)
        codes = engine.encode(full)  # [S, n_features], lock held inside
        if codes.shape[0] < tag_len:
            tag_len = 0
        if req.mode not in ("pick", "topk"):
            raise HTTPException(400, f"Invalid mode {req.mode!r}: must be 'pick' or 'topk'")
        if req.mode == "pick":
            ids = req.feature_ids or ([req.feature_id] if req.feature_id is not None else [])
            if not ids:
                raise HTTPException(400, "mode='pick' requires feature_ids")
            chosen = [int(i) for i in ids]
        else:
            k = max(1, min(int(req.k), 64))
            chosen = [ft["feature_id"] for ft in engine.top_features(codes, tag_len=tag_len, k=k)]
        feats = []
        for fid in chosen:
            col = codes[:, fid]
            feats.append(
                {
                    "feature_id": fid,
                    "label": engine.labels.get(fid),
                    "max_activation": float(col[tag_len:].max().item())
                    if codes.shape[0] > tag_len
                    else float(col.max().item()),
                    "activations": [round(float(v), 4) for v in col.tolist()],
                }
            )
        return {
            "sequence": dna,
            "organism": req.organism,
            "tag": tag,
            "tag_len": tag_len,
            "bases": list(full),
            "n_tokens": codes.shape[0],
            "layer": engine.layer,
            "features": feats,
        }

    @app.post("/generate")
    def generate(req: GenerateRequest):
        _require_ready()
        try:
            return engine.generate(
                prompt=req.prompt,
                organism=req.organism,
                tag=req.tag,
                features=[f.model_dump() for f in req.features],
                n_tokens=req.n_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                compare_baseline=req.compare_baseline,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/gene_embed")
    def gene_embed(req: GeneEmbedRequest):
        """Embed sequences for the Sequence-UMAP tab.

        Each sequence -> Evo2 layer-L -> SAE -> pool over the DNA region into a per-feature
        vector. One encode per sequence yields both mean- and max-pooled vectors (base64
        float32 [n x n_features]) so the client can toggle pooling without re-running the model;
        UMAP runs client-side. Also returns per-sequence metadata + feature stats.
        """
        if not engine.ready:
            raise HTTPException(503, "Backend not ready")
        import base64

        import numpy as np

        tag = engine.resolve_tag(req.organism, req.tag)
        if tag is None:
            raise HTTPException(400, f"Unknown organism '{req.organism}' and no custom tag")
        tag_len = len(tag)
        seqs, meta = [], []
        for g in req.genes[:1000]:
            dna = clean_dna(str(g.get("sequence", "")))
            if len(dna) < 3:
                continue
            seqs.append(tag + dna)
            meta.append(
                {
                    "gene_symbol": g.get("symbol") or g.get("gene_symbol") or f"gene{len(meta)}",
                    "label": g.get("label"),
                    "species": g.get("species"),
                }
            )
        if not seqs:
            raise HTTPException(400, "No valid gene sequences")

        rows_mean, rows_max, meta_out = [], [], []
        for codes, m in zip(engine.encode_batch(seqs), meta):  # codes: [S, n_features]
            tl = tag_len if codes.shape[0] > tag_len else 0
            seg = codes[tl:]  # DNA region only (drop the phylo-tag tokens)
            if seg.shape[0] == 0:
                continue
            rows_mean.append(seg.mean(dim=0).numpy().astype(np.float32))
            rows_max.append(seg.max(dim=0).values.numpy().astype(np.float32))
            meta_out.append(m)
        if not rows_mean:
            raise HTTPException(400, "No valid gene sequences")

        gmean = np.stack(rows_mean).astype(np.float32)  # [n_genes, n_features]
        gmax = np.stack(rows_max).astype(np.float32)
        n_firing = (gmax > 0).sum(0)  # TopK/ReLU codes >= 0 -> firing set is pooling-invariant
        stats = []
        for fid in np.nonzero(n_firing >= req.min_firing)[0]:
            fid = int(fid)
            col = gmean[:, fid]
            stats.append(
                {
                    "feature_id": fid,
                    "n_firing": int(n_firing[fid]),
                    "mean_act_when_firing": float(col[col > 0].mean()) if (col > 0).any() else 0.0,
                    "max_act": float(gmax[:, fid].max()),
                    "label": engine.labels.get(fid),
                }
            )
        stats.sort(key=lambda s: -s["n_firing"])
        return {
            "G_b64": base64.b64encode(gmean.tobytes()).decode(),
            "Gmax_b64": base64.b64encode(gmax.tobytes()).decode(),
            "n_features": int(gmean.shape[1]),
            "n_genes": int(gmean.shape[0]),
            "genes": meta_out,
            "feature_stats": stats,
        }

    return app
