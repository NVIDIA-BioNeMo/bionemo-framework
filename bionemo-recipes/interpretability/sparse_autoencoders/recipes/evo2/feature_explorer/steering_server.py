# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Live per-base SAE activation backend for the Evo2 Feature Explorer.

This is the ONLY data path the annotator front-end uses — there is no mockup
fallback. It loads a base Evo2 model and a trained SAE once at startup, then
serves per-base SAE feature activations for arbitrary pasted DNA sequences.

Pipeline per request:
    DNA string ──(optional phylo tag)──> Evo2 forward (post_process=False)
        ──> layer-`L` hidden states [S, H] ──> SAE.encode ──> codes [S, n_features]
        ──> pick top-K features by max activation (or a chosen feature)
        ──> attach human labels from the feature-annotation file
        ──> JSON {bases, tag_len, features:[{id,label,max,activations[]}]}

The heavy Evo2 machinery (Megatron model load, tokenizer, forward step) is
reused verbatim from `bionemo.evo2.run.predict` so tokenisation matches the
activation-extraction pipeline the SAE was trained on.

Run (from the evo2_megatron recipe environment, single GPU):

    EVO2_CKPT_DIR=/data/interp/evo2/checkpoints/evo2_1b_base_mbridge \
    SAE_CKPT_PATH=/data/interp/evo2/sae/v2_diverse/layer19_C13_nofilter/checkpoints/checkpoint_final.pt \
    FEATURE_ANNOTATIONS=/data/interp/evo2/sae_eval/dashboard_data/l19_C13_nofilter/feature_metadata.parquet \
    EMBEDDING_LAYER=19 \
    python steering_server.py

The front-end (Vite dev server) calls http://localhost:8001.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Disable Inductor CUDA graphs before torch initializes inductor — graph capture
# conflicts with our residual-stream forward hook (which replaces the layer's
# output tensor) and with re-feeding a growing sequence each decode step.
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("steering_server")

# -----------------------------------------------------------------------------
# Configuration (all overridable via environment — see module docstring)
# -----------------------------------------------------------------------------
EVO2_CKPT_DIR = os.environ.get("EVO2_CKPT_DIR", "/data/interp/evo2/checkpoints/evo2_1b_base_mbridge")
SAE_CKPT_PATH = os.environ.get(
    "SAE_CKPT_PATH",
    "/data/interp/evo2/sae/v2_diverse/layer19_C13_nofilter/checkpoints/checkpoint_final.pt",
)
FEATURE_ANNOTATIONS = os.environ.get(
    "FEATURE_ANNOTATIONS",
    "/data/interp/evo2/sae_eval/dashboard_data/l19_C13_nofilter/feature_metadata.parquet",
)
EMBEDDING_LAYER = int(os.environ.get("EMBEDDING_LAYER", "19"))
PORT = int(os.environ.get("PORT", "8001"))
DEVICE = os.environ.get("DEVICE", "cuda")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "8192"))

# The `sae` package lives a few levels up under sparse_autoencoders/sae/src.
_SAE_SRC = os.environ.get(
    "SAE_SRC",
    str(Path(__file__).resolve().parents[3] / "sae" / "src"),
)
if _SAE_SRC not in sys.path:
    sys.path.insert(0, _SAE_SRC)

# Phylogenetic-tag prefixes per organism. Evo2 was trained with lineage tags in
# front of each sequence; prepending the right tag conditions the model. These
# are intentionally editable — set EVO2_ORGANISM_TAGS to a JSON map to override.
# "tag_len" tokens are stripped by the front-end in "DNA only" mode.
_DEFAULT_ORGANISM_TAGS = {
    "None (raw DNA)": "",
    "Human": "|d__Eukaryota;p__Chordata;c__Mammalia;o__Primates;f__Hominidae;g__Homo;s__Homo sapiens|",
    "E. coli": "|d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli|",
    "S. cerevisiae": "|d__Eukaryota;p__Ascomycota;c__Saccharomycetes;o__Saccharomycetales;f__Saccharomycetaceae;g__Saccharomyces;s__Saccharomyces cerevisiae|",
}
ORGANISM_TAGS = json.loads(os.environ["EVO2_ORGANISM_TAGS"]) if "EVO2_ORGANISM_TAGS" in os.environ else _DEFAULT_ORGANISM_TAGS


def _resolve_tag(organism: str, tag: Optional[str]) -> Optional[str]:
    """Resolve the phylo prefix: an explicit custom `tag` wins; otherwise look up
    the organism preset. Returns None if neither resolves (caller errors)."""
    if tag is not None:
        return tag
    return ORGANISM_TAGS.get(organism)

_VALID_BASES = re.compile(r"[^ACGTN]")

# Holds the loaded artefacts; populated in the lifespan handler.
_STATE: dict = {"ready": False}


# -----------------------------------------------------------------------------
# Model / SAE loading
# -----------------------------------------------------------------------------
def _init_single_process_distributed() -> None:
    """Set the env vars Megatron's distributed init expects for a 1-GPU server."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")


def _load_evo2():
    """Load the base Evo2 model truncated to EMBEDDING_LAYER with post_process=False.

    Mirrors steps 1-5 of `bionemo.evo2.run.predict.predict`, but stops before the
    FASTA dataset so we keep a persistent model for interactive single-seq forwards.
    Returns (model_module, tokenizer).
    """
    from bionemo.evo2.run import predict as P

    resolved = P.resolve_checkpoint_path(Path(EVO2_CKPT_DIR))
    run_config = P.read_run_config(P.get_checkpoint_run_config_filename(str(resolved)))
    model_provider = P.instantiate(run_config["model"])

    # Single-GPU, no model parallelism.
    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.context_parallel_size = 1
    model_provider.sequence_parallel = False

    # Mixed precision: honour the checkpoint's recipe, else bf16.
    mp_value = run_config.get("mixed_precision")
    if isinstance(mp_value, str):
        mp_config = P.get_mixed_precision_config(mp_value)
    elif mp_value is not None:
        mp_config = P.instantiate(mp_value)
    else:
        mp_config = P.get_mixed_precision_config("bf16_mixed")
    mp_config.finalize()
    mp_config.setup(model_provider)

    # Tokenizer.
    tok_dir = resolved / "tokenizer"
    tokenizer = P._HuggingFaceTokenizer(tok_dir) if tok_dir.exists() else P._HuggingFaceTokenizer(P.DEFAULT_HF_TOKENIZER_MODEL_PATH)
    model_provider.vocab_size = tokenizer.vocab_size
    model_provider.should_pad_vocab = True

    # Truncate to the embedding layer and emit hidden states instead of logits.
    original_num_layers = model_provider.num_layers
    layer = EMBEDDING_LAYER
    target = original_num_layers + layer + 1 if layer < 0 else layer + 1
    if target <= 0 or target > original_num_layers:
        raise ValueError(f"EMBEDDING_LAYER={layer} invalid for {original_num_layers}-layer model")
    model_provider.num_layers = target
    model_provider.post_process = False
    if getattr(model_provider, "hybrid_override_pattern", None) and len(model_provider.hybrid_override_pattern) > target:
        model_provider.hybrid_override_pattern = model_provider.hybrid_override_pattern[:target]
    if target == 1 and getattr(model_provider, "remove_activation_post_first_layer", False):
        model_provider.remove_activation_post_first_layer = False

    rng_config = P.instantiate(run_config["rng"]) if run_config.get("rng") else P.RNGConfig(seed=1234)
    dist_config = P.instantiate(run_config["dist"]) if run_config.get("dist") else P.DistributedInitConfig()
    P.initialize_inference_distributed(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        micro_batch_size=1,
        global_batch_size=1,
        rng_config=rng_config,
        dist_config=dist_config,
    )

    model_provider.finalize()
    model = model_provider.provide_distributed_model(
        ddp_config=None,
        wrap_with_ddp=False,
        data_parallel_random_init=False,
        bf16=mp_config.bf16,
        fp16=mp_config.fp16,
        mixed_precision_wrapper=P.Float16Module if (mp_config.bf16 or mp_config.fp16) else None,
    )
    for m in model:
        m.eval()
    P._load_model_weights_from_checkpoint(checkpoint_path=str(resolved), model=model, dist_ckpt_strictness="ignore_all")
    logger.info("Evo2 loaded (layer %d of %d, post_process=False)", layer, original_num_layers)
    return model[0], tokenizer


def _load_sae():
    """Load a trained SAE checkpoint and return (sae_module, n_features)."""
    ckpt = torch.load(SAE_CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_config"])
    state = ckpt["model_state_dict"]
    # DDP checkpoints prefix params with "module."
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    # We deliberately target a plain TopK SAE (no BatchTopK dependency). ReLU SAEs
    # are also supported; both load from the same {pre_bias, latent_bias,
    # encoder.weight, decoder.weight} state dict.
    arch = ""
    train_cfg = ckpt.get("config")
    if isinstance(train_cfg, dict):
        arch = str(train_cfg.get("architecture", train_cfg.get("arch", ""))).lower()
    hint = (arch + " " + SAE_CKPT_PATH).lower()

    from sae.architectures import ReLUSAE, TopKSAE  # noqa: E402

    cls = ReLUSAE if "relu" in hint else TopKSAE
    sae = cls(**cfg)
    sae.load_state_dict(state)
    sae.eval()
    sae.to(DEVICE)
    logger.info("SAE loaded: %s, input_dim=%d, n_features=%d", cls.__name__, cfg["input_dim"], cfg["hidden_dim"])
    return sae, int(cfg["hidden_dim"])


def _load_annotations() -> dict[int, str]:
    """Load feature_id -> label from the feature-annotation file (parquet/tsv/csv/json)."""
    path = Path(FEATURE_ANNOTATIONS)
    if not path.exists():
        logger.warning("Feature-annotation file %s not found — features will be unlabeled", path)
        return {}
    labels: dict[int, str] = {}
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        tbl = pq.read_table(path, columns=None).to_pydict()
        ids = tbl.get("feature_id", [])
        names = tbl.get("label", tbl.get("annotation", [None] * len(ids)))
        labels = {int(i): (str(n) if n is not None else None) for i, n in zip(ids, names)}
    elif suffix == ".json":
        raw = json.loads(path.read_text())
        items = raw.items() if isinstance(raw, dict) else ((r["feature_id"], r.get("label")) for r in raw)
        for k, v in items:
            labels[int(k)] = (v.get("label") if isinstance(v, dict) else v)
    else:  # tsv / csv
        import csv

        delim = "\t" if suffix in (".tsv", ".txt") else ","
        with path.open() as f:
            for row in csv.DictReader(f, delimiter=delim):
                labels[int(row["feature_id"])] = row.get("label") or row.get("annotation")
    labels = {k: v for k, v in labels.items() if v}
    logger.info("Loaded %d feature labels from %s", len(labels), path)
    return labels


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load everything once at startup."""
    try:
        _init_single_process_distributed()
        model, tokenizer = _load_evo2()
        sae, n_features = _load_sae()
        _STATE.update(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            n_features=n_features,
            labels=_load_annotations(),
            peaks=_load_natural_peaks(),
            gen_model=None,  # full model for /generate, loaded lazily on first request
            ready=True,
        )
        logger.info("Backend ready on port %d", PORT)
    except Exception:  # surface load failures loudly; /health stays not-ready
        logger.exception("Startup failed — backend will report not-ready")
    yield


app = FastAPI(title="Evo2 SAE Feature Explorer — live backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def _clean_sequence(seq: str) -> str:
    """Uppercase, strip everything that isn't a nucleotide."""
    return _VALID_BASES.sub("", seq.upper())


@torch.no_grad()
def _encode(text: str) -> torch.Tensor:
    """Tokenise `text` (1 token/char, no BOS — matches the extraction pipeline) and
    return SAE codes of shape [seq_len, n_features] on CPU."""
    tokenizer = _STATE["tokenizer"]
    model = _STATE["model"]
    sae = _STATE["sae"]

    ids = tokenizer.tokenize(text) if hasattr(tokenizer, "tokenize") else tokenizer.text_to_ids(text)
    ids = ids[:MAX_SEQ_LEN]
    tokens = torch.tensor([ids], dtype=torch.long, device=DEVICE)  # [1, S]
    batch = {
        "tokens": tokens,
        "position_ids": torch.arange(len(ids), dtype=torch.long, device=DEVICE).unsqueeze(0),
        "loss_mask": torch.ones_like(tokens),
        "seq_idx": torch.zeros(1, dtype=torch.long, device=DEVICE),
    }

    from bionemo.evo2.run import predict as P

    result = P._predict_step(model=model, batch=batch, output_embeddings=True)
    hidden = result["hidden_embeddings"][0].to(DEVICE).float()  # [S, H]
    codes = sae.encode(hidden)  # [S, n_features]
    return codes.detach().cpu()


class AnnotateRequest(BaseModel):
    sequence: str
    organism: str = "None (raw DNA)"
    tag: Optional[str] = None  # custom phylo prefix; overrides `organism` when set
    mode: str = "topk"  # "topk" | "pick"
    k: int = 8
    feature_id: Optional[int] = None
    feature_ids: Optional[list[int]] = None  # pick mode: one or more features (by-name picker)


@app.get("/health")
def health():
    return {
        "ready": bool(_STATE.get("ready")),
        "layer": EMBEDDING_LAYER,
        "sae_path": SAE_CKPT_PATH,
        "n_features": _STATE.get("n_features"),
        "n_labels": len(_STATE.get("labels", {})),
        "organisms": list(ORGANISM_TAGS.keys()),
        "organism_tags": ORGANISM_TAGS,  # name -> phylo prefix, so the UI can prefill an editable tag
        "device": DEVICE,
    }


@app.post("/annotate")
def annotate(req: AnnotateRequest):
    if not _STATE.get("ready"):
        raise HTTPException(status_code=503, detail="Backend not ready (model/SAE still loading or failed to load)")

    dna = _clean_sequence(req.sequence)
    if not dna:
        raise HTTPException(status_code=400, detail="No valid nucleotides in sequence")
    tag = _resolve_tag(req.organism, req.tag)
    if tag is None:
        raise HTTPException(status_code=400, detail=f"Unknown organism '{req.organism}' and no custom tag provided")
    full = tag + dna
    tag_len = len(tag)  # byte tokenizer => 1 token per char
    with _GEN_LOCK:  # serialize GPU access — the Megatron model isn't thread-safe
        codes = _encode(full)  # [S, n_features]
    if codes.shape[0] < tag_len:
        tag_len = 0  # truncated; don't over-strip

    # Feature ranking is done over the DNA region only (after the phylo tag).
    dna_codes = codes[tag_len:]
    labels = _STATE["labels"]

    if req.mode == "pick":
        ids = req.feature_ids if req.feature_ids else ([req.feature_id] if req.feature_id is not None else [])
        if not ids:
            raise HTTPException(status_code=400, detail="mode='pick' requires feature_ids")
        chosen = [int(i) for i in ids]
    else:
        per_feat_max = dna_codes.max(dim=0).values  # [n_features]
        k = max(1, min(int(req.k), 64))
        top = torch.topk(per_feat_max, k=min(k, per_feat_max.numel())).indices.tolist()
        # Only keep features that actually fire somewhere in the DNA region.
        chosen = [int(i) for i in top if per_feat_max[i].item() > 0.0]

    features = []
    for fid in chosen:
        col = codes[:, fid]
        features.append(
            {
                "feature_id": fid,
                "label": labels.get(fid),
                "max_activation": float(col[tag_len:].max().item()) if codes.shape[0] > tag_len else float(col.max().item()),
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
        "layer": EMBEDDING_LAYER,
        "features": features,
    }


# -----------------------------------------------------------------------------
# Generative steering — autoregressive generation with an SAE-feature clamp
# -----------------------------------------------------------------------------
import threading  # noqa: E402

_GEN_LOCK = threading.Lock()


def _load_natural_peaks() -> dict[int, float]:
    """feature_id -> max_activation ("natural peak") from the annotation parquet."""
    path = Path(FEATURE_ANNOTATIONS)
    if not path.exists() or path.suffix.lower() != ".parquet":
        return {}
    import pyarrow.parquet as pq

    tbl = pq.read_table(path).to_pydict()
    ids = tbl.get("feature_id", [])
    peaks = tbl.get("max_activation", [None] * len(ids))
    return {int(i): float(p) for i, p in zip(ids, peaks) if p is not None}


def _ensure_gen_model():
    """Lazily load the FULL Evo2 model (all layers + LM head) for generation."""
    if _STATE.get("gen_model") is not None:
        return _STATE["gen_model"]
    from bionemo.evo2.run import predict as P

    resolved = P.resolve_checkpoint_path(Path(EVO2_CKPT_DIR))
    run_config = P.read_run_config(P.get_checkpoint_run_config_filename(str(resolved)))
    model_provider = P.instantiate(run_config["model"])
    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.context_parallel_size = 1
    model_provider.sequence_parallel = False
    mp_value = run_config.get("mixed_precision")
    if isinstance(mp_value, str):
        mp_config = P.get_mixed_precision_config(mp_value)
    elif mp_value is not None:
        mp_config = P.instantiate(mp_value)
    else:
        mp_config = P.get_mixed_precision_config("bf16_mixed")
    mp_config.finalize()
    mp_config.setup(model_provider)
    model_provider.vocab_size = _STATE["tokenizer"].vocab_size
    model_provider.should_pad_vocab = True
    model_provider.post_process = True  # keep the LM head -> logits for sampling
    # No-cache generation uses plain forwards (no decode-time CUDA-graph capture,
    # same as the annotate path), so just keep graph capture off. Do NOT set
    # cuda_graph_impl — it must stay a valid enum (None fails config validation).
    if hasattr(model_provider, "enable_cuda_graph"):
        model_provider.enable_cuda_graph = False
    model_provider.finalize()
    model = model_provider.provide_distributed_model(
        ddp_config=None,
        wrap_with_ddp=False,
        data_parallel_random_init=False,
        bf16=mp_config.bf16,
        fp16=mp_config.fp16,
        mixed_precision_wrapper=P.Float16Module if (mp_config.bf16 or mp_config.fp16) else None,
    )
    for m in model:
        m.eval()
    P._load_model_weights_from_checkpoint(
        checkpoint_path=str(resolved), model=model, dist_ckpt_strictness="ignore_all"
    )
    _STATE["gen_model"] = model[0]
    logger.info("Full Evo2 (generation) loaded — post_process=True")
    return _STATE["gen_model"]


def _clamp_hook(specs, pre_bias, prompt_len):
    """Additive multi-feature clamp on the layer-19 residual stream, applied to
    the GENERATED continuation only (positions >= prompt_len; the prompt is left
    untouched).

    Canonical additive feature clamp, summed over the requested features:

        h  <-  h  +  Σ_f ( t_f - a_f(h) ) · d_f

      a_f(h) = relu((h - pre_bias)·W_enc[f] + b[f])   current activation
      d_f    = SAE decoder column for feature f       (its own scale)
      t_f    = target activation (strength_f × natural peak_f)

    `specs` is a list of (enc_f [H], b_f float, dec_f [H], target float). The edit
    is additive relative to the base hidden state h (steering), not a replacement.
    Generation re-runs a full forward each step (no cache), so each forward sees
    the whole current sequence as [S, B, H].
    """

    def hook(_module, _inp, output):
        hs = output[0] if isinstance(output, tuple) else output  # [S, B, H]
        x = hs.float()
        xc = x - pre_bias
        add = torch.zeros_like(x)
        for enc_f, b_f, dec_f, target in specs:
            a = torch.relu(torch.matmul(xc, enc_f) + b_f)  # [S, B]
            add = add + (target - a).unsqueeze(-1) * dec_f
        gen = (torch.arange(hs.shape[0], device=hs.device) >= prompt_len).view(-1, 1, 1)
        new = (x + add * gen.to(x.dtype)).to(hs.dtype)
        if isinstance(output, tuple):
            return (new, *output[1:])
        return new

    return hook


@torch.no_grad()
def _autoregress(model, prompt_ids, n_tokens, temperature, top_k, allowed_ids, hook_layer=None, hook_fn=None):
    """Cache-free autoregressive sampling: full forward each step, sample the last
    position, append. No KV/SSM cache (so the steering hook is safe — see /generate).
    Sampling is restricted to `allowed_ids` (the 4 nucleotides). Returns the
    generated token ids only (prompt excluded).
    """
    cur = list(prompt_ids)
    allowed = torch.tensor(allowed_ids, dtype=torch.long, device=DEVICE)
    handle = hook_layer.register_forward_hook(hook_fn) if (hook_layer is not None and hook_fn is not None) else None
    temp = float(temperature)
    k = int(top_k)
    try:
        for _ in range(int(n_tokens)):
            x = torch.tensor([cur], dtype=torch.long, device=DEVICE)
            logits = model(input_ids=x, position_ids=None, attention_mask=None, labels=None, runtime_gather_output=True)
            row = logits[0, -1, :].float()
            nl = torch.full_like(row, float("-inf"))
            nl[allowed] = row[allowed]  # restrict to nucleotides
            if temp > 0:
                nl = nl / temp
            if k > 0:
                kth = torch.topk(nl, min(k, len(allowed_ids)))[0][..., -1, None]
                nl = torch.where(nl < kth, torch.full_like(nl, float("-inf")), nl)
            if temp > 0 and k != 1:
                nxt = int(torch.multinomial(torch.softmax(nl, dim=-1), 1).item())
            else:
                nxt = int(torch.argmax(nl).item())
            cur.append(nxt)
    finally:
        if handle is not None:
            handle.remove()
    return cur[len(prompt_ids):]


def _detok(ids: list[int]) -> str:
    """Token ids -> cleaned DNA string."""
    tok = _STATE["tokenizer"]
    for meth in ("detokenize", "ids_to_text"):
        if hasattr(tok, meth):
            try:
                return _clean_sequence(getattr(tok, meth)(ids))
            except Exception:
                pass
    return _clean_sequence("".join(chr(i) for i in ids if 0 <= i < 256))


@torch.no_grad()
def _feature_track(dna: str, feature_id: int) -> list[float]:
    """Per-base activation of one feature on `dna` (annotate model + SAE)."""
    if not dna:
        return []
    codes = _encode(dna)  # [S, n_features], no phylo tag
    return [round(float(v), 4) for v in codes[:, feature_id].tolist()]


@torch.no_grad()
def _feature_tracks(dna: str, fids: list[int]) -> dict:
    """Per-base activation of several features on `dna`, encoded once. {fid: [..]}."""
    if not dna:
        return {int(f): [] for f in fids}
    codes = _encode(dna)  # [S, n_features]
    return {int(f): [round(float(v), 4) for v in codes[:, int(f)].tolist()] for f in fids}


class FeatureClamp(BaseModel):
    feature_id: int
    strength: float = 1.0  # target activation = strength × the feature's natural peak


class GenerateRequest(BaseModel):
    prompt: str = ""
    organism: str = "None (raw DNA)"
    tag: Optional[str] = None  # custom phylo prefix; overrides `organism` when set
    features: list[FeatureClamp] = []  # zero or more features to clamp; [] = plain generation
    n_tokens: int = 120
    temperature: float = 1.0
    top_k: int = 0
    compare_baseline: bool = False  # also return an unsteered sample for side-by-side


@app.get("/features")
def features():
    """Labeled-feature catalog for the steering picker: id, label, natural peak."""
    if not _STATE.get("ready"):
        raise HTTPException(status_code=503, detail="Backend not ready")
    labels = _STATE["labels"]
    peaks = _STATE.get("peaks", {})
    rows = [{"id": int(fid), "label": lab, "natural_peak": peaks.get(int(fid))} for fid, lab in labels.items()]
    rows.sort(key=lambda r: r["id"])
    return rows


@app.post("/generate")
def generate(req: GenerateRequest):
    """Autoregressively generate DNA, unsteered vs with an SAE-feature clamp."""
    if not _STATE.get("ready"):
        raise HTTPException(status_code=503, detail="Backend not ready")
    from megatron.core.utils import unwrap_model

    sae = _STATE["sae"]
    tok = _STATE["tokenizer"]
    clamps = req.features or []  # may be empty -> plain (unsteered) generation
    for c in clamps:
        if not (0 <= int(c.feature_id) < _STATE["n_features"]):
            raise HTTPException(status_code=400, detail=f"feature_id {c.feature_id} out of range [0,{_STATE['n_features']})")

    tag = _resolve_tag(req.organism, req.tag)
    if tag is None:
        raise HTTPException(status_code=400, detail=f"Unknown organism '{req.organism}' and no custom tag provided")
    dna = _clean_sequence(req.prompt)
    ids = (tok.tokenize(tag + dna) if hasattr(tok, "tokenize") else tok.text_to_ids(tag + dna))[:MAX_SEQ_LEN]
    if not ids:
        raise HTTPException(status_code=400, detail="Provide a prompt or pick an organism (need >=1 token to seed)")
    n_tokens = max(1, min(int(req.n_tokens), 400))
    # Restrict sampling to the four nucleotide tokens so generations are clean DNA
    # and stay aligned with the per-base activation re-encode.
    allowed = sorted({(tok.tokenize(b) if hasattr(tok, "tokenize") else tok.text_to_ids(b))[0] for b in "ACGT"})
    fids = [int(c.feature_id) for c in clamps]

    with _GEN_LOCK:
        gen_model = _ensure_gen_model()
        layers = unwrap_model(gen_model).decoder.layers
        if not (0 <= EMBEDDING_LAYER < len(layers)):
            raise HTTPException(status_code=500, detail=f"layer {EMBEDDING_LAYER} invalid for {len(layers)}-layer model")
        hook_layer = layers[EMBEDDING_LAYER]
        pre_bias = sae.pre_bias.detach().float().to(DEVICE)

        specs, feat_meta = [], []
        for c in clamps:
            f = int(c.feature_id)
            enc_f = sae.encoder.weight[f].detach().float().to(DEVICE)  # [H]
            b_f = float(sae.latent_bias[f].detach())
            dec_f = sae.decoder.weight[:, f].detach().float().to(DEVICE)  # [H]
            target = float(c.strength)  # clamp this feature's activation directly to `strength`
            specs.append((enc_f, b_f, dec_f, target))
            feat_meta.append({"id": f, "label": _STATE["labels"].get(f), "strength": float(c.strength)})
        hook_fn = _clamp_hook(specs, pre_bias, len(ids)) if specs else None  # additive, continuation-only

        # Main generation: steered if any features were given, else a plain sample.
        main_ids = _autoregress(
            gen_model, ids, n_tokens, req.temperature, req.top_k, allowed,
            hook_layer=(hook_layer if hook_fn else None), hook_fn=hook_fn,
        )
        # Baseline (unsteered) only when explicitly requested AND we actually clamped.
        base_ids = None
        if req.compare_baseline and specs:
            base_ids = _autoregress(gen_model, ids, n_tokens, req.temperature, req.top_k, allowed)

    main_dna = _detok(main_ids)
    resp = {
        "prompt": dna,
        "organism": req.organism,
        "tag": tag,
        "tag_len": len(tag),
        "n_tokens": n_tokens,
        "features": feat_meta,
        "steered": bool(specs),
        "generation": {"sequence": main_dna, "activations": _feature_tracks(main_dna, fids)},
        "baseline": None,
    }
    if base_ids is not None:
        base_dna = _detok(base_ids)
        resp["baseline"] = {"sequence": base_dna, "activations": _feature_tracks(base_dna, fids)}
    return resp


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
