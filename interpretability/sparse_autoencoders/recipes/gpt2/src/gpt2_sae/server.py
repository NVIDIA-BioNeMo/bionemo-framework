"""Interactive GPT-2 SAE backend: TransformerLens GPT-2 + Bloom SAE (blocks.7.hook_resid_pre).

Serves the feature-explorer dashboard (static) plus a small JSON API — /annotate (per-token feature
activations), /generate (feature-clamped steering), /gene_embed (pooled per-text SAE vectors) — from a
single process. The model + SAE run on GPU if available, else CPU (GPT-2 small is CPU-friendly).
"""

import json
import os
import threading
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from safetensors.torch import load_file
from transformer_lens import HookedTransformer


REPO, HP, LAYER = "jbloom/GPT2-Small-SAEs-Reformatted", "blocks.7.hook_resid_pre", 7
# The built dashboard + bundled data (parquets, labels) live in feature_explorer/dist/. The server
# mounts this dir at "/" and reads the atlas/labels from it, so one process serves UI + API + data.
# Override with GPT2_SAE_DASHBOARD to point at a dist/ built elsewhere.
SERVE_DIR = Path(
    os.environ.get("GPT2_SAE_DASHBOARD", Path(__file__).resolve().parents[2] / "feature_explorer" / "dist")
)
USER_LABELS = SERVE_DIR / "user_labels.json"
# Neuronpedia auto-interp explanations for this exact SAE (gpt2-small/7-res-jb, index-aligned
# to our feature ids). {feature_id -> short semantic description}. Makes ~all 24576 features
# searchable by concept in the steering/inspector pickers. See scripts/fetch_neuronpedia.py.
NP_LABELS = SERVE_DIR / "neuronpedia_labels.json"
MAX_CLAMP = 200.0


class Engine:
    """Owns the GPT-2 model, the SAE weights, feature labels, and the steering hook."""

    def __init__(self):
        """Set up empty state; call load() to bring up the model + SAE."""
        self.ready = False
        self.layer = LAYER
        self.max_seq_len = 1024
        self.organism_tags = {"None (raw text)": ""}
        self.labels = {}
        self.user_labels = {}
        self.np_labels = {}
        self._lock = threading.Lock()

    def load(self):
        """Load GPT-2 (TransformerLens) + the Bloom SAE + bundled labels/peaks (one-time)."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HookedTransformer.from_pretrained("gpt2", device=self.device)
        self.model.eval()
        w = load_file(hf_hub_download(REPO, f"{HP}/sae_weights.safetensors"))
        self.W_enc, self.W_dec, self.b_enc, self.b_dec = (
            w[k].float().to(self.device) for k in ("W_enc", "W_dec", "b_enc", "b_dec")
        )
        self.n_features = self.W_enc.shape[1]
        atlas = pq.read_table(SERVE_DIR / "features_atlas.parquet").to_pydict()
        self.peaks = {int(f): float(m) for f, m in zip(atlas["feature_id"], atlas["max_activation"])}
        if NP_LABELS.exists():
            self.np_labels = {int(k): v for k, v in json.loads(NP_LABELS.read_text()).items()}
        if USER_LABELS.exists():
            self.user_labels = {int(k): v for k, v in json.loads(USER_LABELS.read_text()).items()}
        self.ready = True

    def _sae_encode(self, x):
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    @torch.no_grad()
    def encode(self, text):
        """Run the model, take layer-7 residuals, SAE-encode -> (str tokens, codes[T, n_features])."""
        _, cache = self.model.run_with_cache(text, names_filter=HP)
        x = cache[HP][0].float()
        return self.model.to_str_tokens(text), self._sae_encode(x).cpu()

    def label_for(self, fid):
        """Display label: user star-label > Neuronpedia label > 'Feature N'."""
        return self.user_labels.get(int(fid)) or self.np_labels.get(int(fid)) or f"Feature {int(fid)}"

    def user_labels_snapshot(self):
        """user_labels as {str id: label} for the JSON API."""
        return {str(k): v for k, v in self.user_labels.items()}

    def set_user_label(self, fid, label):
        """Persist (or clear) a user rename for a feature; returns the new label."""
        fid = int(fid)
        if (label or "").strip():
            self.user_labels[fid] = label.strip()
        else:
            self.user_labels.pop(fid, None)
        USER_LABELS.write_text(json.dumps({str(k): v for k, v in self.user_labels.items()}))
        return self.user_labels.get(fid)

    def top_features(self, codes, k, tag_len=0):
        """Feature ids with the highest peak activation over the text (BOS excluded)."""
        m = (codes[tag_len:] if codes.shape[0] > tag_len else codes).max(0).values
        return [int(i) for i in torch.topk(m, min(k, self.n_features)).indices.tolist()]

    def annotate(self, sequence, mode, k, feature_ids):
        """Per-token SAE activations for a text: top-k features or a picked set."""
        with self._lock:
            text = (sequence or "").strip()
            if not text:
                raise ValueError("empty input")
            labels, codes = self.encode(text)
            tag_len = 1 if codes.shape[0] > 1 else 0
            if mode == "pick":
                chosen = [int(i) for i in (feature_ids or [])]
            else:
                chosen = self.top_features(codes, max(1, min(int(k), 64)), tag_len)
            feats = []
            for fid in chosen:
                col = codes[:, fid]
                mx = float(col[tag_len:].max().item()) if codes.shape[0] > tag_len else float(col.max().item())
                feats.append(
                    {
                        "feature_id": fid,
                        "label": self.label_for(fid),
                        "max_activation": mx,
                        "activations": [round(float(v), 4) for v in col.tolist()],
                    }
                )
            return {
                "sequence": text,
                "organism": "None (raw text)",
                "tag": "",
                "tag_len": tag_len,
                "bases": labels,
                "n_tokens": codes.shape[0],
                "layer": self.layer,
                "features": feats,
            }

    @torch.no_grad()
    def feature_tracks(self, text, fids):
        """Per-token activation series for specific feature ids."""
        if not text or not fids:
            return {str(int(f)): [] for f in fids}
        _, codes = self.encode(text)
        return {str(int(f)): [round(float(v), 4) for v in codes[:, int(f)].tolist()] for f in fids}

    @torch.no_grad()
    def generate(self, prompt, features, n_tokens, temperature, top_k, compare_baseline):
        """Generate from a prompt, optionally clamping features on the continuation; returns steered (+ baseline)."""
        with self._lock:
            prompt = prompt or ""
            clamps = {
                int(f["feature_id"]): max(-MAX_CLAMP, min(MAX_CLAMP, float(f.get("strength", 1.0))))
                for f in (features or [])
            }
            fids = list(clamps.keys())
            n_tokens = max(1, min(int(n_tokens), 256))

            def clamp_fn(resid, hook):  # resid: [B, S, 768]  (TL always gives [B,S,H])
                codes = self._sae_encode(resid)
                new = codes.clone()
                for fid, s in clamps.items():
                    new[..., fid] = s
                return resid + (new - codes) @ self.W_dec  # delta trick (b_dec cancels)

            def run(steer):
                torch.manual_seed(0)  # same sampling -> difference is purely the steering
                hooks = [(HP, clamp_fn)] if (steer and clamps) else []
                with self.model.hooks(fwd_hooks=hooks):
                    return self.model.generate(
                        prompt,
                        max_new_tokens=n_tokens,
                        do_sample=(temperature > 0),
                        temperature=(temperature or 1.0),
                        top_k=(top_k or None),
                        stop_at_eos=False,
                        prepend_bos=True,
                        return_type="str",
                        verbose=False,
                    )

            main = run(True)
            resp = {
                "prompt": prompt,
                "organism": "None (raw text)",
                "tag": "",
                "tag_len": 0,
                "n_tokens": n_tokens,
                "features": [
                    {"feature_id": fid, "label": self.label_for(fid), "strength": clamps[fid]} for fid in fids
                ],
                "steered": bool(clamps),
                "generation": {"sequence": main, "activations": self.feature_tracks(main, fids)},
                "baseline": None,
            }
            if compare_baseline:
                base = run(False)
                resp["baseline"] = {"sequence": base, "activations": self.feature_tracks(base, fids)}
            return resp

    @torch.no_grad()
    def embed_bundle(self, items, min_firing=10, max_genes=1000):
        """Mean/max-pool each text into a per-feature SAE vector for the UMAP tab."""
        import base64

        import numpy as np

        with self._lock:
            items = (items or [])[:max_genes]
            rows_mean, rows_max, meta = [], [], []
            n_short = 0
            for it in items:
                text = (it.get("sequence") or "").strip()
                if not text:
                    n_short += 1
                    continue
                _, codes = self.encode(text)
                seg = codes[1:] if codes.shape[0] > 1 else codes  # drop BOS
                if seg.shape[0] < 2:
                    n_short += 1
                    continue
                rows_mean.append(seg.mean(0).numpy().astype(np.float32))
                rows_max.append(seg.max(0).values.numpy().astype(np.float32))
                meta.append(
                    {
                        "symbol": it.get("symbol") or text[:24],
                        "label": it.get("label"),
                        "n_tokens": int(codes.shape[0]),
                    }
                )
            if not rows_mean:
                return None, n_short
            gmean = np.stack(rows_mean)
            gmax = np.stack(rows_max)
            n_firing = (gmax > 0).sum(0)
            fire_ids = np.nonzero(n_firing >= min_firing)[0]
            if len(fire_ids) == 0:
                fire_ids = np.nonzero(n_firing >= 1)[0]  # relax for tiny batches
            stats = []
            for fid in fire_ids:
                fid = int(fid)
                col = gmean[:, fid]
                stats.append(
                    {
                        "feature_id": fid,
                        "n_firing": int(n_firing[fid]),
                        "mean_act_when_firing": float(col[col > 0].mean()) if (col > 0).any() else 0.0,
                        "max_act": float(gmax[:, fid].max()),
                        "label": self.label_for(fid),
                    }
                )
            stats.sort(key=lambda s: -s["n_firing"])
            gmean = gmean[:, fire_ids]
            gmax = gmax[:, fire_ids]
            return {
                "G_b64": base64.b64encode(gmean.tobytes()).decode(),
                "Gmax_b64": base64.b64encode(gmax.tobytes()).decode(),
                "n_features": int(gmean.shape[1]),
                "n_genes": int(gmean.shape[0]),
                "feature_ids": [int(f) for f in fire_ids],
                "genes": meta,
                "feature_stats": stats,
            }, n_short


engine = Engine()


class AnnotateRequest(BaseModel):
    """Request body for the /annotate endpoint (top-k feature scan or an explicit feature pick)."""

    sequence: str
    organism: str = "None (raw text)"
    tag: Optional[str] = None
    mode: str = "topk"
    k: int = 8
    feature_ids: Optional[list[int]] = None


class FeatureClamp(BaseModel):
    """One feature id + the strength to clamp it to during steering."""

    feature_id: int
    strength: float = 1.0


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint (prompt + optional feature clamps)."""

    prompt: str = ""
    organism: str = "None (raw text)"
    tag: Optional[str] = None
    features: list[FeatureClamp] = []
    n_tokens: int = 120
    temperature: float = 1.0
    top_k: int = 0
    compare_baseline: bool = False


class Gene(BaseModel):
    """One text item (symbol + sequence) to embed for the UMAP tab."""

    symbol: str = ""
    sequence: str
    label: Optional[str] = None
    species: Optional[str] = None


class GeneEmbedRequest(BaseModel):
    """Request body for the /gene_embed endpoint (a batch of texts to pool)."""

    genes: list[Gene] = []
    organism: str = "None (raw text)"
    tag: Optional[str] = None
    min_firing: int = 10


class LabelRequest(BaseModel):
    """Request body for the /label endpoint (rename or clear a feature label)."""

    feature_id: int
    label: str = ""


app = FastAPI()


@app.get("/api/health")
def health(response: Response):
    """Model + SAE readiness and dimensions."""
    if not engine.ready:
        response.status_code = 503
    return {
        "ready": bool(engine.ready),
        "layer": engine.layer,
        "n_features": getattr(engine, "n_features", 0),
        "n_labels": 0,
        "organisms": list(engine.organism_tags.keys()),
        "organism_tags": engine.organism_tags,
        "device": getattr(engine, "device", "cpu"),
        "max_seq_len": engine.max_seq_len,
        "restart_enabled": False,
    }


@app.get("/api/features")
def features():
    """Full labeled feature catalog for the pickers."""
    # Full labeled catalog so any concept is searchable in the pickers: every feature with a
    # Neuronpedia or user label. Starred (user-labeled) first, then by natural peak, then id.
    ids = set(engine.np_labels) | set(engine.user_labels)

    # ⭐ verified-steerable set first, then other user labels, then everything by natural peak.
    def rank(f):
        starred = str(engine.label_for(f)).startswith("⭐")
        return (0 if starred else (1 if f in engine.user_labels else 2), -engine.peaks.get(int(f), 0.0), int(f))

    ordered = sorted(ids, key=rank)
    return [
        {
            "id": int(f),
            "label": engine.label_for(f),
            "natural_peak": engine.peaks.get(int(f)),
            "steerable": True,
            "description": engine.np_labels.get(int(f)),
            "auroc": None,
        }
        for f in ordered
    ]


@app.get("/api/labels")
def labels():
    """Current user star-labels as {str id: label}."""
    return engine.user_labels_snapshot()


@app.post("/api/label")
def set_label(req: LabelRequest):
    """Rename (or clear) a feature label; returns the new label."""
    if not (0 <= req.feature_id < engine.n_features):
        raise HTTPException(400, "feature_id out of range")
    return {"feature_id": req.feature_id, "label": engine.set_user_label(req.feature_id, req.label)}


@app.post("/api/annotate")
def annotate(req: AnnotateRequest):
    """Per-token SAE activations for a text (top-k or a picked feature set)."""
    if not engine.ready:
        raise HTTPException(503, "Backend not ready")
    if req.mode not in ("topk", "pick"):
        raise HTTPException(400, "mode must be 'topk' or 'pick'")
    if req.mode == "pick" and not req.feature_ids:
        raise HTTPException(400, "mode='pick' requires feature_ids")
    try:
        return engine.annotate(req.sequence, req.mode, req.k, req.feature_ids)
    except ValueError as e:
        raise HTTPException(413 if "too long" in str(e) else 400, str(e))


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Generate from a prompt with optional feature clamps; returns steered (+ baseline)."""
    if not engine.ready:
        raise HTTPException(503, "Backend not ready")
    bad = [f.feature_id for f in req.features if not (0 <= f.feature_id < engine.n_features)]
    if bad:
        raise HTTPException(400, f"feature_id(s) {bad} out of range")
    return engine.generate(
        req.prompt,
        [f.model_dump() for f in req.features],
        req.n_tokens,
        req.temperature,
        req.top_k,
        req.compare_baseline,
    )


@app.post("/api/gene_embed")
def gene_embed(req: GeneEmbedRequest):
    """Pool a batch of texts into per-feature SAE vectors for the UMAP tab."""
    if not engine.ready:
        raise HTTPException(503, "Backend not ready")
    items = [g.model_dump() for g in req.genes]
    n_received = len(items)
    bundle, n_short = engine.embed_bundle(items, min_firing=req.min_firing)
    if bundle is None:
        raise HTTPException(400, "Nothing embeddable (all items too short)")
    bundle.update(
        {
            "n_received": n_received,
            "n_skipped_short": n_short,
            "n_clamped": 0,
            "n_dropped_over_cap": max(0, n_received - 1000),
            "max_seq_len": engine.max_seq_len,
            "max_genes": 1000,
        }
    )
    return bundle


@app.middleware("http")
async def _no_cache(request, call_next):
    resp = await call_next(request)
    # Data files (parquets/labels) are refreshed by rebuilds; never let a stale browser copy shadow them.
    if request.url.path.endswith((".parquet", ".json")):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


if SERVE_DIR.exists():
    app.mount("/", StaticFiles(directory=str(SERVE_DIR), html=True), name="dash")


def main():
    """`gpt2-sae-serve [PORT]` — load the model + SAE once, then serve UI + API."""
    import sys

    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8749))
    print("loading engine ...", flush=True)
    engine.load()
    print(f"engine ready: {engine.n_features} features, layer {engine.layer}, device {engine.device}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
