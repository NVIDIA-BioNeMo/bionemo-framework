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
    async def annotate(req: AnnotateRequest, request: Request):
        # Single-flight like /generate and /gene_embed: on the single GPU, annotate also has to take
        # the core GPU lock (encode -> encode_batch), so a concurrent annotate during a long generate
        # would silently queue behind it for minutes. Route it through the same guard so it fast-rejects
        # with 409 "Engine busy" instead — one consistent signal across every GPU endpoint.
        _require_ready()

        def work(cancel):
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

        return await _run_cancellable(request, work)

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
