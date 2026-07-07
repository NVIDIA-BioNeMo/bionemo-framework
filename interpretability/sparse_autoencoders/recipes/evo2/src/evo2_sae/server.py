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

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import anyio
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import core
from .core import Evo2SAE


logger = logging.getLogger("evo2_sae_infer.server")

# Exit code the /restart endpoint uses to ask the launch supervisor for a fresh worker (distinct from a
# crash so launch_inference.sh can respawn immediately without counting it against the crash budget).
EXIT_RESTART = 42


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


_engine_busy = threading.Lock()
_BUSY_MSG = "Engine busy — it runs one request at a time on the single GPU. Try again in a moment."


async def _run_cancellable(request: Request, work):
    """Single-flight guard: one heavy model call at a time.

    Reject a concurrent request with 409 rather than silently queueing it behind a long-running one (a
    reloaded tab, an impatient re-click) — which is what let a stale request appear to "stick" in the
    engine. Then run it via the cancellable path.
    """
    if not _engine_busy.acquire(blocking=False):
        raise HTTPException(409, _BUSY_MSG)
    try:
        return await _run_cancellable_inner(request, work)
    finally:
        _engine_busy.release()


async def _run_cancellable_inner(request: Request, work):
    """Run blocking ``work(cancel)`` in the threadpool; signal it to stop if the client disconnects.

    ``work`` receives a ``cancel`` predicate (``() -> bool``) to poll at safe checkpoints and abort
    (raise ``core.RequestAborted``) when it returns True. Threads can't be force-killed, so cancel is
    cooperative: a watcher task flips a flag when the HTTP connection drops, and the worker notices at
    its next checkpoint — freeing the single serialized GPU instead of finishing output nobody will
    read. The work runs in the same AnyIO threadpool (honoring MAX_CONCURRENCY) that sync endpoints use.

    The work's return value / exception is captured inside the task group and surfaced *after* it
    exits, so a real error (ValueError, RequestAborted) reaches the caller unchanged — an exception
    left to escape an anyio task group would be repackaged into an ExceptionGroup the endpoint's
    ``except ValueError`` / ``except RequestAborted`` clauses wouldn't match.
    """
    cancelled = threading.Event()
    box: dict = {}

    async def _watch_disconnect():
        # Poll the connection while the work runs; a dropped client trips the cancel flag. A polling
        # failure must never falsely cancel real work or crash the request, so swallow and stop.
        while not cancelled.is_set():
            try:
                dropped = await request.is_disconnected()
            except Exception:
                return
            if dropped:
                cancelled.set()
                return
            await anyio.sleep(0.5)

    async with anyio.create_task_group() as tg:
        tg.start_soon(_watch_disconnect)
        try:
            box["result"] = await anyio.to_thread.run_sync(lambda: work(cancelled.is_set))
        except BaseException as exc:  # capture (incl. RequestAborted) here to re-raise it unwrapped below
            box["error"] = exc
        finally:
            cancelled.set()  # stop the worker at its next checkpoint (if still running) ...
            tg.cancel_scope.cancel()  # ... and end the watcher task

    if "error" in box:
        raise box["error"]
    return box["result"]


KEEPALIVE_SECS = float(os.getenv("KEEPALIVE_SECS", "10"))


async def _stream_cancellable(request: Request, work):
    """Single-flight guard + a keepalive stream so an idle-timeout proxy/tunnel can't kill a long request.

    ``/generate`` and ``/gene_embed`` can run for minutes with NO interim bytes (the model call isn't
    incremental), and an intermediate hop — a Brev/ingress reverse proxy, an SSH/Teleport tunnel — may
    drop a connection that's idle for ~60-120s, no matter how generous our client/Vite timeouts are.
    So while the blocking ``work(cancel)`` runs in the threadpool we stream whitespace heartbeats;
    leading whitespace keeps the body valid JSON, so the client parses the final object with
    ``response.json()`` unchanged. A server-side error is framed as ``{"__error__": {status, detail}}``
    in the body (the HTTP status is already 200 once streaming has begun) and the client re-raises it.
    409 (busy) is still a real status — raised here before any streaming starts.
    """
    if not _engine_busy.acquire(blocking=False):
        raise HTTPException(409, _BUSY_MSG)

    async def _gen():
        cancelled = threading.Event()
        worker = None
        try:
            worker = asyncio.ensure_future(anyio.to_thread.run_sync(lambda: work(cancelled.is_set)))
            # Heartbeat until the model call finishes (or errors); poll disconnect between beats so a
            # dropped client still cancels the GPU work at its next checkpoint.
            while True:
                try:
                    await asyncio.wait_for(asyncio.shield(worker), timeout=KEEPALIVE_SECS)
                    break  # worker finished — result/exception retrieved below
                except asyncio.TimeoutError:
                    yield b" "  # heartbeat: keeps the connection non-idle through any proxy/tunnel
                    if await request.is_disconnected():
                        cancelled.set()
                except Exception:
                    break  # worker raised; surfaced via worker.result() below
            try:
                result = worker.result()
            except core.RequestAborted:
                logger.info("client disconnected mid-request — aborted, GPU freed")
                return
            except ValueError as e:
                yield json.dumps({"__error__": {"status": 413 if "too long" in str(e) else 400, "detail": str(e)}}).encode()
                return
            except Exception as e:  # surface a clean message, not a mid-stream 500 traceback
                logger.exception("streamed work failed")
                yield json.dumps({"__error__": {"status": 500, "detail": str(e)}}).encode()
                return
            yield json.dumps(result).encode()
        finally:
            cancelled.set()  # if the client dropped mid-stream, stop the worker at its next checkpoint
            if worker is not None and not worker.done():
                worker.add_done_callback(lambda t: t.cancelled() or t.exception())  # swallow "never retrieved"
            _engine_busy.release()

    return StreamingResponse(_gen(), media_type="application/json")


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
            "restart_enabled": os.getenv("ALLOW_ENGINE_RESTART") == "1",  # UI shows a restart button iff true
        }

    @api.post("/restart")
    async def restart():
        """Hard cancel: exit this worker so the launch supervisor respawns a fresh one.

        Reloads the model in ~30-45s. It's the only way to free the GPU from an in-flight generation,
        since INF.generate can't be interrupted mid-call. Gated by ALLOW_ENGINE_RESTART (set by
        launch_inference.sh for `serve`) so it does nothing under a plain run with no supervisor.
        """
        if os.getenv("ALLOW_ENGINE_RESTART") != "1":
            raise HTTPException(403, "Engine restart is not enabled on this server.")
        logger.warning("engine restart requested via /api/restart — exiting worker (code %d) for respawn", EXIT_RESTART)
        # Exit just after the 202 flushes; the supervisor's respawn loop reloads the model.
        threading.Timer(0.3, lambda: os._exit(EXIT_RESTART)).start()
        return JSONResponse({"restarting": True}, status_code=202)

    @api.get("/features")
    def features():
        _require_ready()
        rows = []
        for f, lab in engine.labels.items():
            fid = int(f)
            ex = engine.feature_extra.get(fid, {})
            rows.append(
                {
                    "id": fid,
                    "label": lab,
                    "natural_peak": engine.peaks.get(fid),
                    # Curated per-feature metadata (present when the annotation file carries it): served
                    # here keyed to the feature, and surfaced by the UI (⚡ badge for steerable, the
                    # description as a tooltip). Descriptive — it doesn't change encode/generate.
                    "steerable": bool(ex.get("steerable")),
                    "description": ex.get("description"),
                    "auroc": ex.get("auroc"),
                }
            )
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
    async def generate(req: GenerateRequest, request: Request):
        _require_ready()
        # async + threadpool (not a plain sync endpoint) so we can watch for a client disconnect and
        # cancel: /generate can run for minutes, and the client-side timeout only abandons the fetch —
        # the GPU keeps churning unless we stop it. See _run_cancellable / core.generate(cancel=...).
        return await _stream_cancellable(
            request,
            lambda cancel: engine.generate(
                prompt=req.prompt,
                organism=req.organism,
                tag=req.tag,
                features=[f.model_dump() for f in req.features],
                n_tokens=req.n_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                compare_baseline=req.compare_baseline,
                cancel=cancel,
            ),
        )

    # Per-request sequence cap for /gene_embed (one encode each on the single GPU). Surfaced in the
    # response (max_genes) and enforced by reporting overflow rather than silently dropping it.
    MAX_GENES = 1000

    @api.post("/gene_embed")
    async def gene_embed(req: GeneEmbedRequest, request: Request):
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
        n_too_short = n_clamped = 0
        for g in req.genes[:MAX_GENES]:
            dna = core.clean_dna(str(g.get("sequence", "")))
            if len(dna) < 3:
                n_too_short += 1
                continue
            full = tag + dna
            # Clamp an over-length sequence to the context window (tag + leading DNA) and embed it
            # anyway, reporting how many were clamped. Unlike /annotate and /generate (which reject),
            # the UMAP tab favors keeping every point on the map over exactness — a vector from the
            # first max_seq_len bases beats dropping the sequence entirely.
            if len(full) > engine.max_seq_len:
                full = full[: engine.max_seq_len]
                n_clamped += 1
            seqs.append(full)
            meta.append(
                {
                    "gene_symbol": g.get("symbol") or g.get("gene_symbol") or f"gene{len(meta)}",
                    "label": g.get("label"),
                    "species": g.get("species"),
                }
            )
        # pooling + stats + base64 packing live in Evo2SAE.embed_bundle, shared with the offline
        # dashboard.py precompute so the static bundle is the same shape as this response. This is the
        # heavy part (one encode per sequence, up to MAX_GENES on the single GPU) and the longest call
        # the server serves, so run it cancellably: if the client gives up, stop between micro-batches.
        # Assemble the full bundle inside the streamed work so the heavy encode + the reporting fields
        # ship together; a "nothing embeddable" case raises ValueError -> framed 400 by _stream_cancellable.
        def _build(cancel):
            bundle = engine.embed_bundle(seqs, len(tag), meta, min_firing=req.min_firing, cancel=cancel)
            if bundle is None:
                raise ValueError(
                    f"No embeddable sequences (received {n_received}: {n_too_short} too short, "
                    f"{n_over_cap} past the {MAX_GENES}-sequence cap)"
                )
            # Report what we changed so the UI can warn the user, rather than silently returning a UMAP
            # that differs from what they submitted: over-cap/too-short are dropped; over-length clamped.
            bundle["n_received"] = n_received
            bundle["n_skipped_short"] = n_too_short
            bundle["n_clamped"] = n_clamped
            bundle["n_dropped_over_cap"] = n_over_cap
            bundle["max_seq_len"] = engine.max_seq_len
            bundle["max_genes"] = MAX_GENES
            return bundle

        # Long call (one encode per sequence on the single GPU) with no interim bytes -> stream keepalives.
        return await _stream_cancellable(request, _build)

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
