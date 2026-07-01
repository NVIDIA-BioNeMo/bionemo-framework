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

"""Evo2 + SAE inference core — one importable engine for live and batch use.

`Evo2SAE` loads a base Evo2 model and a trained SAE once, then exposes:

    encode(dna)            -> codes [S, n_features]            # ONE sequence (interactive)
    encode_batch(seqs)     -> list of codes [S_i, n_features]  # MANY sequences (batched on GPU)
    feature_tracks(dna, f) -> {feature_id: [per-base activation]}
    generate(...)          -> autoregressive DNA generation with optional additive
                              SAE-feature clamping on the generated continuation

It has NO web dependency: the FastAPI server (`server.py`) and the batch CLI
(`cli.py`) are thin wrappers over this class, and the viz backend imports it too.

The heavy Evo2 machinery is reused from the recipe: a SINGLE inference engine
(`infer.setup_inference_engine`, run eager with `cuda_graph_impl="none"` so the
residual-stream steering hook applies) serves both paths. Encode/highlight runs a normal
full-sequence forward on the engine's model and reads layer `layer` off a forward hook;
generation uses the same model via `infer.generate`. This module only adds the SAE layer:
encode, feature labels, and the decode-only feature-clamp hook.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import threading
from pathlib import Path
from typing import Optional

import torch


logger = logging.getLogger("evo2_sae_infer")

# Disable Inductor CUDA graphs before torch initializes inductor — graph capture
# conflicts with the residual-stream forward hook (which replaces the layer output)
# and with re-feeding a growing sequence each decode step.
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

# Make the local `sae` package importable (sparse_autoencoders/sae/src).
_SAE_SRC = os.environ.get("SAE_SRC", str(Path(__file__).resolve().parents[4] / "sae" / "src"))
if _SAE_SRC not in sys.path:
    sys.path.insert(0, _SAE_SRC)

_VALID_BASES = re.compile(r"[^ACGTN]")

# Steering clamp targets are absolute SAE-code values; this SAE's features peak ~100-300.
# Cap the magnitude so an extreme target can't blow the logits to NaN (which device-asserts
# and wedges the process). Generous headroom for amplification, bounded against runaway.
MAX_CLAMP_STRENGTH = float(os.environ.get("MAX_CLAMP_STRENGTH", "300"))

# Phylogenetic-tag prefixes per organism (Evo2 was trained with lineage tags).
DEFAULT_ORGANISM_TAGS = {
    "None (raw DNA)": "",
    "Human": "|d__Eukaryota;p__Chordata;c__Mammalia;o__Primates;f__Hominidae;g__Homo;s__Homo sapiens|",
    "E. coli": "|d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli|",
    "S. cerevisiae": "|d__Eukaryota;p__Ascomycota;c__Saccharomycetes;o__Saccharomycetales;f__Saccharomycetaceae;g__Saccharomyces;s__Saccharomyces cerevisiae|",
}


def clean_dna(seq: str) -> str:
    """Uppercase and strip everything that isn't a nucleotide."""
    return _VALID_BASES.sub("", (seq or "").upper())


def annotate(engine, sequence: str, organism: str = "None (raw DNA)", tag: Optional[str] = None):
    """Shared encode path for the CLI ``encode`` and the server ``/annotate`` (topk).

    Cleans the sequence, resolves the phylo tag, encodes once, and computes the tag length
    (the leading tag tokens to skip, ignored if it would drop the whole sequence). Returns
    ``(dna, resolved_tag, codes, tag_len)``; callers add their own top-k / per-base presentation.

    Raises ``ValueError`` on an empty/non-DNA sequence or an unknown organism — the server maps
    these to HTTP 400, the CLI to a clean exit. Takes the engine as an argument (rather than
    living on ``Evo2SAE``) so it composes with any object exposing ``resolve_tag``/``encode``.
    """
    dna = clean_dna(sequence)
    if not dna:
        raise ValueError("No valid nucleotides in sequence")
    resolved_tag = engine.resolve_tag(organism, tag)
    if resolved_tag is None:
        raise ValueError(f"Unknown organism '{organism}' and no custom tag")
    full = resolved_tag + dna
    # Reject over-length input rather than letting encode() silently truncate to max_seq_len —
    # the per-base `activations` would then be shorter than `bases` and the viz would misalign.
    if len(full) > engine.max_seq_len:
        raise ValueError(f"sequence too long: {len(full)} > max_seq_len ({engine.max_seq_len})")
    codes = engine.encode(full)
    if codes.shape[0] != len(full):  # belt-and-suspenders: encode must be 1:1 with the input here
        raise ValueError("encoded length != input length (tokenizer truncated)")
    tag_len = len(resolved_tag) if codes.shape[0] >= len(resolved_tag) else 0
    return dna, resolved_tag, codes, tag_len


def parse_clamp_spec(spec) -> dict:
    """Normalize one feature-clamp spec into ``{"feature_id": int, "strength": float}``.

    Accepts a ``"FEATURE_ID[:STRENGTH]"`` string (CLI ``--clamp``; strength defaults to 1.0,
    e.g. ``29244:300`` or ``29244``), or a mapping with ``feature_id``/``strength`` (server
    ``FeatureClamp``). Single source of truth so the CLI and the API parse clamps identically.
    Magnitude/finiteness/range are enforced later by ``_sanitize_steering``.

    Raises ``ValueError`` on a malformed string (caller decides whether that's a 400 or an exit).
    """
    if isinstance(spec, str):
        fid, sep, strength = spec.partition(":")
        try:
            return {"feature_id": int(fid), "strength": float(strength) if (sep and strength) else 1.0}
        except ValueError:
            raise ValueError(f"invalid clamp {spec!r}: expected FEATURE_ID[:STRENGTH] with numeric values")
    return {"feature_id": int(spec["feature_id"]), "strength": float(spec.get("strength", 1.0))}


def _is_unrecoverable_cuda(e: Exception) -> bool:
    """True for CUDA errors that poison the process context (device-side assert / sticky CUDA error).

    These don't clear on the next request — the worker must be recycled — so the caller marks the
    engine not-ready rather than retrying into a stream of 500s.
    """
    s = str(e).lower()
    return isinstance(e, RuntimeError) and ("device-side assert" in s or "cuda error" in s)


def _sanitize_steering(features, n_features, temperature, top_k):
    """Validate/normalize steering inputs (pure, no GPU); raise on bad input.

    Each guard prevents a CUDA device-side assert that would corrupt the context and wedge
    the server until restart:

      * feature id outside [0, n_features) indexes off the SAE codes -> ValueError (server -> 400);
      * |strength| beyond MAX_CLAMP_STRENGTH blows the logits to inf/NaN -> capped;
      * temperature <= 0 makes the recipe's sampler divide logits by temperature (NaN under
        multinomial) -> coerce to greedy top-1, which is deterministic and skips that path;
      * negative top_k is an invalid sampler argument -> coerce to 0 (no top-k filtering).

    Returns ``(clamps: dict[int, float], fids: list[int], temperature: float, top_k: int)``.
    """
    bad = sorted({int(f["feature_id"]) for f in features if not (0 <= int(f["feature_id"]) < n_features)})
    if bad:
        raise ValueError(f"feature_id(s) {bad} out of range [0, {n_features})")
    clamps = {}
    for f in features:
        s = float(f.get("strength", 1.0))
        if not math.isfinite(s):  # NaN/±inf would blow the logits to NaN -> neutralize explicitly
            s = 0.0
        clamps[int(f["feature_id"])] = max(-MAX_CLAMP_STRENGTH, min(MAX_CLAMP_STRENGTH, s))
    temperature = float(temperature)
    top_k = max(0, int(top_k))  # negative top_k is an invalid sampler arg -> 0 (no filtering)
    if temperature <= 0:  # greedy — avoid the sampler's logits/temperature division (NaN)
        top_k = max(top_k, 1)
    return clamps, list(clamps), temperature, top_k


def _init_single_process_distributed() -> None:
    """Set the env vars Megatron's distributed init expects for a 1-GPU process."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")


class Evo2SAE:
    """Persistent Evo2 + SAE inference engine (single-sequence and batched)."""

    def __init__(
        self,
        evo2_ckpt_dir: str,
        sae_ckpt_path: str,
        layer: int,
        device: str = "cuda",
        max_seq_len: int = 8192,
        feature_annotations: Optional[str] = None,
        organism_tags: Optional[dict] = None,
    ):
        """Record config; call .load() to actually load the model + SAE onto the GPU."""
        self.evo2_ckpt_dir = evo2_ckpt_dir
        self.sae_ckpt_path = sae_ckpt_path
        self.layer = int(layer)
        self.device = device
        self.max_seq_len = int(max_seq_len)
        self.feature_annotations = feature_annotations
        self.organism_tags = dict(organism_tags) if organism_tags else dict(DEFAULT_ORGANISM_TAGS)

        self.model = None  # the engine's mcore model (full): encode forward + steering hook target
        self.gen_components = None  # recipe inference engine (the single model), built in load()
        self.tokenizer = None
        self.sae = None
        self.n_features = None
        self.labels: dict[int, str] = {}
        self.peaks: dict[int, float] = {}
        self._lock = threading.Lock()  # serialize GPU access (Megatron isn't thread-safe)
        self.ready = False

    # ------------------------------------------------------------------ loading
    def load(self) -> "Evo2SAE":
        """Load the single Evo2 inference engine + SAE + feature labels (one-time, ~1 min).

        One model serves both paths: encode/highlight runs a full-sequence forward and reads
        layer ``layer`` off a hook (see ``_forward_hidden``); generate drives the same model's
        dynamic-decode engine with a steering hook on the same layer.
        """
        from megatron.core.utils import unwrap_model

        _init_single_process_distributed()
        comp = self._ensure_engine()  # the one engine; reused by generate()
        self.model = unwrap_model(comp.model)  # mcore model: encode forward + steering-hook target
        self.tokenizer = comp.tokenizer
        self.sae, self.n_features = self._load_sae()
        # Fail loudly now if the SAE doesn't fit the model, instead of a cryptic matmul error on the
        # first encode. We can only catch a dim mismatch (wrong SAE/model); a wrong layer with the
        # same hidden size is undetectable here (the SAE checkpoint records no training layer).
        self._check_dim(self._sae_input_dim, self._model_hidden_size(), self.layer)
        self.labels, self.peaks = self._load_feature_meta()
        self.ready = True
        logger.info("Evo2SAE ready: layer=%d n_features=%d n_labels=%d", self.layer, self.n_features, len(self.labels))
        return self

    def _ensure_engine(self):
        """Lazily build the recipe's inference engine (eager/hookable) for generation.

        cuda_graph_impl="none" keeps decode eager so the residual-stream steering hook
        takes effect (a CUDA-graph-captured decode would replay frozen ops and ignore it).
        """
        if self.gen_components is None:
            from bionemo.evo2.run import infer as INF

            # Defensive: if anything earlier initialized the global num-microbatches calculator,
            # setup_inference_engine re-inits it and asserts unless we tear the singleton down first.
            try:
                from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator

                destroy_num_microbatches_calculator()
            except Exception:
                pass

            self.gen_components = INF.setup_inference_engine(
                Path(self.evo2_ckpt_dir), max_seq_length=self.max_seq_len, cuda_graph_impl="none"
            )
        return self.gen_components

    def _model_hidden_size(self):
        """The model's hidden size at this layer (== encode's feature dim).

        Config first (cheap); else a 1-token forward (ground truth); ``None`` if neither works
        (then the dim check is skipped, never blocking load).
        """
        try:
            from megatron.core.utils import unwrap_model

            cfg = getattr(unwrap_model(self.model), "config", None)
            if cfg is not None and getattr(cfg, "hidden_size", None):
                return int(cfg.hidden_size)
        except Exception:
            pass
        try:
            probe = self._forward_hidden([self.tokenize("A")])
            if probe and probe[0].numel():
                return int(probe[0].shape[-1])
        except Exception:
            pass
        return None

    @staticmethod
    def _check_dim(sae_input_dim, hidden, layer):
        """Raise if the SAE input_dim != the model hidden size (skip if hidden is unknown)."""
        if hidden is not None and hidden != sae_input_dim:
            raise ValueError(
                f"SAE input_dim={sae_input_dim} does not match the Evo2 hidden size={hidden} at "
                f"layer {layer} — wrong SAE/model pairing (check --sae-ckpt-path / --layer)."
            )

    def _load_sae(self):
        ckpt = torch.load(self.sae_ckpt_path, map_location="cpu", weights_only=False)
        cfg = dict(ckpt["model_config"])
        state = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        from sae.architectures import TopKSAE

        sae = TopKSAE(**cfg)
        sae.load_state_dict(state)
        sae.eval().to(self.device)
        self._sae_input_dim = int(cfg["input_dim"])  # must equal the model hidden size (checked in load)
        logger.info("SAE loaded: TopKSAE input_dim=%d n_features=%d", cfg["input_dim"], cfg["hidden_dim"])
        return sae, int(cfg["hidden_dim"])

    def _load_feature_meta(self):
        """feature_id -> (label, natural peak) from the annotation **parquet** (produced by probe.py annotate)."""
        labels: dict[int, str] = {}
        peaks: dict[int, float] = {}
        if not self.feature_annotations:
            return labels, peaks
        path = Path(self.feature_annotations)
        if not path.exists():
            logger.warning("Feature annotations %s not found — features unlabeled", path)
            return labels, peaks
        if path.suffix.lower() != ".parquet":
            logger.warning("Feature annotations %s: only .parquet is supported — features unlabeled", path)
            return labels, peaks
        import pyarrow.parquet as pq

        tbl = pq.read_table(path).to_pydict()
        ids = tbl.get("feature_id", [])
        names = tbl.get("label", tbl.get("annotation", [None] * len(ids)))
        pk = tbl.get("max_activation", [None] * len(ids))
        for i, n, p in zip(ids, names, pk):
            if n is not None:
                labels[int(i)] = str(n)
            if p is not None:
                peaks[int(i)] = float(p)
        logger.info("Loaded %d labels from %s", len(labels), path)
        return labels, peaks

    # ------------------------------------------------------------------ tokenize
    def tokenize(self, text: str) -> list[int]:
        """Tokenize text to token ids, truncated to max_seq_len."""
        tok = self.tokenizer
        ids = tok.tokenize(text) if hasattr(tok, "tokenize") else tok.text_to_ids(text)
        return ids[: self.max_seq_len]

    def resolve_tag(self, organism: str, tag: Optional[str]) -> Optional[str]:
        """Explicit custom `tag` wins; else look up the organism preset."""
        if tag is not None:
            return tag
        return self.organism_tags.get(organism)

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(self, dna: str) -> torch.Tensor:
        """ONE sequence -> SAE codes [seq_len, n_features] on CPU. No phylo tag."""
        return self.encode_batch([dna])[0]

    @torch.no_grad()
    def encode_batch(self, seqs: list[str], batch_size: int = 8) -> list[torch.Tensor]:
        """MANY sequences -> list of SAE codes [S_i, n_features], batched on the GPU.

        Sequences are padded to the longest in each micro-batch; padding is masked
        out before SAE-encoding so each result has the true per-base length.

        Work is length-bucketed (processed in token-length order) so each micro-batch holds
        similar-length sequences and wastes little padding on mixed-length inputs; results are
        written back by original index, so the returned order matches the input order.
        """
        out: list[torch.Tensor] = [None] * len(seqs)  # type: ignore
        order = [(i, self.tokenize(s)) for i, s in enumerate(seqs)]
        order.sort(key=lambda it: len(it[1]))  # length-bucket to minimize padding (out[orig_i] un-sorts)
        with self._lock:
            for start in range(0, len(order), batch_size):
                chunk = order[start : start + batch_size]
                id_lists = [ids for _, ids in chunk]
                hiddens = self._forward_hidden(id_lists)  # list of [S_i, H]
                for (orig_i, ids), h in zip(chunk, hiddens):
                    out[orig_i] = (
                        self.sae.encode(h.to(self.device)).detach().cpu()
                        if h.shape[0] > 0
                        else torch.empty(0, self.n_features)
                    )
        return out

    @torch.no_grad()
    def _forward_hidden(self, id_lists: list[list[int]]) -> list[torch.Tensor]:
        """Run the engine's model on a (padded) batch of token-id lists.

        Returns the unpadded layer-`layer` hidden states [S_i, H] per sequence.

        The single engine model is ``post_process=True`` (it produces logits for generation), so
        we can't ask ``_predict_step`` for embeddings. Instead we drive a normal full-sequence
        forward and read layer ``layer`` off a forward hook — the same module + ``[S, B, H]``
        layout the steering ``clamp_hook`` uses, so encode and steer see identical activations.
        """
        from bionemo.evo2.run import predict as P

        lens = [len(ids) for ids in id_lists]
        maxlen = max(lens) if lens else 0
        if maxlen == 0:
            return [torch.empty(0, 0) for _ in id_lists]
        b = len(id_lists)
        tokens = torch.zeros(b, maxlen, dtype=torch.long, device=self.device)
        loss_mask = torch.zeros(b, maxlen, dtype=torch.long, device=self.device)
        for i, ids in enumerate(id_lists):
            if ids:
                tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                loss_mask[i, : len(ids)] = 1
        batch = {
            "tokens": tokens,
            "position_ids": torch.arange(maxlen, dtype=torch.long, device=self.device).unsqueeze(0).expand(b, -1),
            "loss_mask": loss_mask,
            "seq_idx": torch.arange(b, dtype=torch.long, device=self.device),
        }
        # Capture layer-`layer` output via a forward hook (engine model has no embedding-only mode).
        captured: dict[str, torch.Tensor] = {}

        def _capture(_module, _inputs, output):
            captured["h"] = output[0] if isinstance(output, tuple) else output  # [S, B, H]

        handle = self.model.decoder.layers[self.layer].register_forward_hook(_capture)
        # Evo2 runs bf16; TransformerEngine asserts param/input dtypes match unless inside an
        # autocast region. predict()'s loop relies on this too, so wrap the direct _predict_step.
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        try:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                P._predict_step(model=self.model, batch=batch, output_embeddings=False)
        finally:
            handle.remove()
        hidden = captured["h"].transpose(0, 1).contiguous()  # [S, B, H] -> [B, S, H]
        return [hidden[i, : lens[i]].float() for i in range(b)]

    def feature_tracks(self, dna: str, fids: list[int]) -> dict:
        """Per-base activation of several features on `dna`. {fid: [..]} (encoded once)."""
        if not dna:
            return {int(f): [] for f in fids}
        codes = self.encode(dna)
        return {int(f): [round(float(v), 4) for v in codes[:, int(f)].tolist()] for f in fids}

    def top_features(self, codes: torch.Tensor, tag_len: int = 0, k: int = 8) -> list[dict]:
        """Top-k features by per-base max activation over the DNA region (excluding the tag).

        `codes` is [S, n_features] from `encode`/`encode_batch`; `tag_len` skips the leading
        phylo-tag tokens (ignored if it would drop the whole sequence). Returns the strictly
        positive features as [{feature_id, label, max_activation}], used by the CLI and server.
        """
        if codes.shape[0] == 0:
            return []
        region = codes[tag_len:] if codes.shape[0] > tag_len else codes
        per = region.max(dim=0).values
        idx = per.topk(min(int(k), per.numel())).indices.tolist()
        return [
            {"feature_id": int(i), "label": self.labels.get(int(i)), "max_activation": round(float(per[i]), 4)}
            for i in idx
            if per[i].item() > 0
        ]

    # ------------------------------------------------------------------ generate
    def generate(
        self,
        prompt="",
        organism="None (raw DNA)",
        tag=None,
        features=None,
        n_tokens=120,
        temperature=1.0,
        top_k=0,
        compare_baseline=False,
    ) -> dict:
        """Autoregressively generate DNA, optionally clamping features on the continuation.

        `features` = list of {"feature_id": int, "strength": float} (or []). Generation runs
        through the recipe's inference engine (`infer.generate`, eager so the hook applies);
        steering is a decode-only forward hook on layer `layer`. Returns
        {generation:{sequence,activations}, baseline:..|None, features, steered}.
        """
        from bionemo.evo2.run import infer as INF

        # Accept CLI "ID[:STRENGTH]" strings and server FeatureClamp dicts identically, then
        # validate/cap below via _sanitize_steering.
        features = [parse_clamp_spec(f) for f in (features or [])]
        resolved_tag = self.resolve_tag(organism, tag)
        if resolved_tag is None:
            raise ValueError(f"Unknown organism '{organism}' and no custom tag")
        dna = clean_dna(prompt)
        full_prompt = resolved_tag + dna
        if not full_prompt:
            raise ValueError("Provide a prompt or pick an organism (need >=1 token to seed)")
        # Reject an over-context prompt rather than silently truncating it in tokenize() (parity
        # with annotate; the server maps "too long" -> 413). Raise MAX_SEQ_LEN to allow longer.
        if len(full_prompt) > self.max_seq_len:
            raise ValueError(f"prompt too long: {len(full_prompt)} > max_seq_len ({self.max_seq_len})")
        # Cap to the engine's configured context budget (prompt + generation must fit max_seq_len),
        # not an arbitrary constant. Raise MAX_SEQ_LEN at launch to generate longer — the 7B is the
        # long-context (1M) model, so it's memory-bound, not architecture-bound (OOD past training len).
        budget = max(1, self.max_seq_len - len(self.tokenize(full_prompt)))
        n_tokens = max(1, min(int(n_tokens), budget))
        # Validate/normalize steering inputs — out-of-range ids, extreme clamps, and temperature
        # 0 each trigger CUDA device-side asserts that wedge the server (see _sanitize_steering).
        clamps, fids, temperature, top_k = _sanitize_steering(features, self.n_features, temperature, top_k)

        try:
            with self._lock:
                comp = self._ensure_engine()
                hook_layer = self.model.decoder.layers[self.layer]  # same module encode reads; steer here
                from evo2_sae.steering import clamp_hook

                feat_meta = [{"id": fid, "label": self.labels.get(fid), "strength": s} for fid, s in clamps.items()]

                def _run(steer: bool) -> str:
                    handle = (
                        hook_layer.register_forward_hook(clamp_hook(self.sae, clamps, decode_only=True))
                        if (steer and clamps)
                        else None
                    )
                    try:
                        out = INF.generate(
                            comp, [full_prompt], max_new_tokens=n_tokens, temperature=temperature, top_k=top_k
                        )
                        return clean_dna(INF._unwrap_result(out[0]).generated_text)
                    finally:
                        if handle is not None:
                            handle.remove()

                main_dna = _run(steer=True)
                base_dna = _run(steer=False) if (compare_baseline and clamps) else None
        except Exception as e:
            # PURELY DEFENSIVE: the known client-reachable CUDA-assert triggers are all neutralized
            # earlier by _sanitize_steering (out-of-range id, magnitude cap, non-finite strength,
            # temperature<=0, negative top_k), so a wedge here implies a genuine hardware/driver
            # fault, NOT a crafted request — it is not client-inducible (so exit+restart is not a
            # remote DoS vector). If that ever stops holding, a new trigger must be added to
            # _sanitize_steering, not handled by leaning on this path.
            #
            # When it does happen, the device-side assert poisons the CUDA context for the whole
            # process — unrecoverable in-process (re-init won't clear it), so restart is the only
            # cure. Mark not-ready so /health flips to 503; and if EXIT_ON_CUDA_WEDGE is set
            # (launch_inference.sh sets it for `serve`, which runs under a restart loop), exit the
            # worker so ANY restart-on-exit supervisor (the launch loop / docker --restart /
            # systemd / k8s) respawns it — host-independent recovery, no readiness probe required.
            # Default (unset): fail-closed at 503 (safe for library/CLI/test use — must not exit).
            if _is_unrecoverable_cuda(e):
                self.ready = False
                logger.exception("unrecoverable CUDA error in generate() — marking engine not-ready")
                if os.environ.get("EXIT_ON_CUDA_WEDGE") == "1":
                    logger.critical("EXIT_ON_CUDA_WEDGE=1 — exiting the worker for the supervisor to restart")
                    os._exit(1)
            raise

        resp = {
            "prompt": dna,
            "organism": organism,
            "tag": resolved_tag,
            "tag_len": len(resolved_tag),
            "n_tokens": n_tokens,
            "features": feat_meta,
            "steered": bool(clamps),
            "generation": {"sequence": main_dna, "activations": self.feature_tracks(main_dna, fids)},
            "baseline": None,
        }
        if base_dna is not None:
            resp["baseline"] = {"sequence": base_dna, "activations": self.feature_tracks(base_dna, fids)}
        return resp
