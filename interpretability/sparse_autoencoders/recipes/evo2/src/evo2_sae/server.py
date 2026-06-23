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

Endpoints (under /api): /api/health, /api/features, /api/annotate (per-base activations for a
pasted sequence), /api/generate (autoregressive generation + optional SAE-feature clamp). The
/api prefix lets a prebuilt frontend be served from "/" on the same origin (single-container
deploy); when no frontend is configured the server is API-only. This is a thin layer; all model
work lives in `core.Evo2SAE`.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import anyio
from fastapi import APIRouter, FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import core
from .core import Evo2SAE


logger = logging.getLogger("evo2_sae_infer.server")


def _resolve_static_dir(static_dir: Optional[str]) -> Optional[str]:
    """Directory of a prebuilt frontend to serve at ``/``, or None for an API-only server.

    Explicit ``static_dir`` arg wins, else the ``DASHBOARD_DIST`` env var. Returns None unless the
    path is an existing directory — so a server with no built frontend (dev, or an image built
    without one) just serves the API and ``/`` 404s, instead of crashing. This layer is generic:
    it serves whatever static dir it's pointed at and knows nothing about the dashboard (the
    dashboard recipe supplies the dir via DASHBOARD_DIST / the Docker build).
    """
    cand = static_dir or os.getenv("DASHBOARD_DIST")
    return cand if (cand and Path(cand).is_dir()) else None


class AnnotateRequest(BaseModel):
    """Request body for /annotate (top-k feature scan or an explicit feature pick)."""

    sequence: str
    organism: str = "None (raw DNA)"
    tag: Optional[str] = None
    mode: str = "topk"  # "topk" | "pick"
    k: int = 8
    feature_ids: Optional[list[int]] = None


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


def build_app(engine: Evo2SAE, static_dir: Optional[str] = None) -> FastAPI:
    """Build the FastAPI app; the engine is loaded once in the lifespan handler.

    API routes live under ``/api`` (so the dashboard and the API can be served from one origin:
    the frontend always calls ``/api/*``, in dev via the Vite proxy and in production from the
    same server). If a built frontend is found (``static_dir`` / ``DASHBOARD_DIST``), it is mounted
    at ``/`` so a single container serves both the UI and the API; otherwise the server is API-only.
    """
    # One GPU (the engine serializes model calls with a lock), so cap how many sync requests run
    # at once: excess requests wait for a worker instead of piling up dozens of parked threads.
    # NOTE: generation is bounded only by the context window now, so a single /generate can run
    # long — under concurrent load requests queue behind it. Sync endpoints run in Starlette's
    # AnyIO threadpool (default 40); shrink it. Tune MAX_CONCURRENCY.
    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "8"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        anyio.to_thread.current_default_thread_limiter().total_tokens = max_concurrency
        try:
            engine.load()
            logger.info("engine ready")
        except Exception:
            logger.exception("engine startup failed — /health stays not-ready")
        yield

    app = FastAPI(title="Evo2 SAE inference", lifespan=lifespan)

    # No CORS middleware: the dashboard always reaches the backend same-origin (Vite proxies
    # /api -> :8001), so cross-origin is never used. CORS is browser-only and not an access
    # control anyway — scripts ignore it; SSO + the limits below are what gate this endpoint.

    # Reject oversized request bodies up front (a multi-MB sequence would be read into memory
    # before per-field validation could reject it). Default 16 MiB; override with MAX_BODY_BYTES.
    # NOTE: advisory — this trusts the Content-Length header, so a chunked request (no length) or a
    # lying header bypasses it; it guards well-behaved clients, not a hard cap. Fine behind SSO;
    # real enforcement would count streamed bytes.
    max_body = int(os.getenv("MAX_BODY_BYTES", str(16 * 1024 * 1024)))

    @app.middleware("http")
    async def _limit_body(request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None and cl.isdigit() and int(cl) > max_body:
            return JSONResponse({"detail": f"request body too large (> {max_body} bytes)"}, status_code=413)
        return await call_next(request)

    def _require_ready():
        if not engine.ready:
            raise HTTPException(503, "Backend not ready")

    # All endpoints under /api (mounted below) so the SPA at "/" never collides with them.
    api = APIRouter()

    @api.get("/health")
    def health(response: Response):
        if not engine.ready:
            response.status_code = 503  # readiness probes shed this pod until load finishes (body still informative)
        return {
            "ready": bool(engine.ready),
            "layer": engine.layer,
            "n_features": engine.n_features,
            "n_labels": len(engine.labels),
            "organisms": list(engine.organism_tags.keys()),
            "organism_tags": engine.organism_tags,
            "device": engine.device,
            "max_seq_len": engine.max_seq_len,  # context budget — UI caps generation length to this
        }

    @api.get("/features")
    def features():
        _require_ready()
        rows = [
            {"id": int(f), "label": lab, "natural_peak": engine.peaks.get(int(f))} for f, lab in engine.labels.items()
        ]
        rows.sort(key=lambda r: r["id"])
        return rows

    @api.post("/annotate")
    def annotate(req: AnnotateRequest):
        _require_ready()
        try:
            dna, tag, codes, tag_len = core.annotate(engine, req.sequence, req.organism, req.tag)
        except ValueError as e:
            raise HTTPException(413 if "too long" in str(e) else 400, str(e))
        full = tag + dna
        if req.mode not in ("pick", "topk"):
            raise HTTPException(400, f"Invalid mode {req.mode!r}: must be 'pick' or 'topk'")
        if req.mode == "pick":
            if not req.feature_ids:
                raise HTTPException(400, "mode='pick' requires feature_ids")
            chosen = [int(i) for i in req.feature_ids]
            # Pick ids are user-supplied; an out-of-range id would IndexError (500) and a negative
            # one would silently index the wrong feature via torch negative-indexing. Reject -> 400.
            bad = sorted({i for i in chosen if not (0 <= i < engine.n_features)})
            if bad:
                raise HTTPException(400, f"feature_id(s) {bad} out of range [0, {engine.n_features})")
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

    @api.post("/generate")
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
            raise HTTPException(413 if "too long" in str(e) else 400, str(e))

    # Per-request sequence cap for /gene_embed (one encode each on the single GPU). Surfaced in the
    # response (max_genes) and enforced by reporting overflow rather than silently dropping it.
    MAX_GENES = 1000

    @api.post("/gene_embed")
    def gene_embed(req: GeneEmbedRequest):
        """Embed sequences for the Sequence-UMAP tab.

        Each sequence -> Evo2 layer-L -> SAE -> pool over the DNA region into a per-feature
        vector. One encode per sequence yields both mean- and max-pooled vectors (base64
        float32 [n x n_firing_features]) so the client can toggle pooling without re-running the
        model; UMAP runs client-side. Also returns per-sequence metadata + feature stats.
        """
        _require_ready()
        tag = engine.resolve_tag(req.organism, req.tag)
        if tag is None:
            raise HTTPException(400, f"Unknown organism '{req.organism}' and no custom tag")
        n_received = len(req.genes)
        n_over_cap = max(0, n_received - MAX_GENES)  # sequences past the per-request cap
        seqs, meta = [], []
        n_too_short = n_too_long = 0
        for g in req.genes[:MAX_GENES]:
            dna = core.clean_dna(str(g.get("sequence", "")))
            if len(dna) < 3:
                n_too_short += 1
                continue
            full = tag + dna
            # Don't silently truncate an over-length sequence into a misleading vector (which is what
            # tokenize would do) — skip it and report it, matching /annotate's reject-don't-truncate.
            if len(full) > engine.max_seq_len:
                n_too_long += 1
                continue
            seqs.append(full)
            meta.append(
                {
                    "gene_symbol": g.get("symbol") or g.get("gene_symbol") or f"gene{len(meta)}",
                    "label": g.get("label"),
                    "species": g.get("species"),
                }
            )
        # pooling + stats + base64 packing live in Evo2SAE.embed_bundle, shared with the offline
        # dashboard.py precompute so the static bundle is the same shape as this response.
        bundle = engine.embed_bundle(seqs, len(tag), meta, min_firing=req.min_firing) if seqs else None
        if bundle is None:
            raise HTTPException(
                400,
                f"No embeddable sequences (received {n_received}: {n_too_short} too short, "
                f"{n_too_long} over the {engine.max_seq_len}-base context limit, "
                f"{n_over_cap} past the {MAX_GENES}-sequence cap)",
            )
        # Report everything we excluded so the UI can warn the user, rather than silently returning a
        # UMAP of fewer sequences than they submitted (over-cap, too-short, and over-length are dropped).
        bundle["n_received"] = n_received
        bundle["n_skipped_short"] = n_too_short
        bundle["n_skipped_too_long"] = n_too_long
        bundle["n_dropped_over_cap"] = n_over_cap
        bundle["max_seq_len"] = engine.max_seq_len
        bundle["max_genes"] = MAX_GENES
        return bundle

    app.include_router(api, prefix="/api")

    # Serve a prebuilt frontend at "/" when present, so one container serves UI + API. Mounted
    # AFTER the API router, so /api/* always resolves to the API; unknown /api/* paths 404 here
    # (StaticFiles only serves index.html for "/", not as a SPA catch-all — the UI is tabs at "/").
    resolved = _resolve_static_dir(static_dir)
    if resolved:
        app.mount("/", StaticFiles(directory=resolved, html=True), name="dashboard")
        logger.info("serving dashboard from %s", resolved)
    else:
        logger.info("no frontend mounted (API-only)")

    return app
