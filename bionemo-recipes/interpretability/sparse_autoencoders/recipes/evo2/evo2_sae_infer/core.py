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

The heavy Evo2 machinery is reused from the recipe: model loading via
`predict.load_model_to_layer` and generation via `infer.setup_inference_engine` /
`infer.generate` (run eager, `cuda_graph_impl="none"`, so the residual-stream steering
hook applies). This module only adds the SAE layer: encode, feature labels, and the
decode-only feature-clamp hook.
"""

from __future__ import annotations

import logging
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
_SAE_SRC = os.environ.get("SAE_SRC", str(Path(__file__).resolve().parents[3] / "sae" / "src"))
if _SAE_SRC not in sys.path:
    sys.path.insert(0, _SAE_SRC)

_VALID_BASES = re.compile(r"[^ACGTN]")

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

        self.model = None  # truncated model (post_process=False) for activations
        self.gen_components = None  # recipe inference engine (full model) for generation, lazy
        self.tokenizer = None
        self.sae = None
        self.n_features = None
        self.labels: dict[int, str] = {}
        self.peaks: dict[int, float] = {}
        self._lock = threading.Lock()  # serialize GPU access (Megatron isn't thread-safe)
        self.ready = False

    # ------------------------------------------------------------------ loading
    def load(self) -> "Evo2SAE":
        """Load the truncated Evo2 model + SAE + feature labels (one-time, ~1 min)."""
        from bionemo.evo2.run import predict as P

        _init_single_process_distributed()
        self.model, self.tokenizer = P.load_model_to_layer(self.evo2_ckpt_dir, self.layer, full=False)
        self.sae, self.n_features = self._load_sae()
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

            self.gen_components = INF.setup_inference_engine(
                Path(self.evo2_ckpt_dir), max_seq_length=self.max_seq_len, cuda_graph_impl="none"
            )
        return self.gen_components

    def _load_sae(self):
        ckpt = torch.load(self.sae_ckpt_path, map_location="cpu", weights_only=False)
        cfg = dict(ckpt["model_config"])
        state = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        train_cfg = ckpt.get("config")
        arch = (
            str(train_cfg.get("architecture", train_cfg.get("arch", ""))).lower()
            if isinstance(train_cfg, dict)
            else ""
        )
        hint = (arch + " " + self.sae_ckpt_path).lower()
        from sae.architectures import ReLUSAE, TopKSAE

        cls = ReLUSAE if "relu" in hint else TopKSAE
        sae = cls(**cfg)
        sae.load_state_dict(state)
        sae.eval().to(self.device)
        logger.info("SAE loaded: %s input_dim=%d n_features=%d", cls.__name__, cfg["input_dim"], cfg["hidden_dim"])
        return sae, int(cfg["hidden_dim"])

    def _load_feature_meta(self):
        """feature_id -> (label, natural peak) from the annotation parquet/tsv/csv/json."""
        labels: dict[int, str] = {}
        peaks: dict[int, float] = {}
        if not self.feature_annotations:
            return labels, peaks
        path = Path(self.feature_annotations)
        if not path.exists():
            logger.warning("Feature annotations %s not found — features unlabeled", path)
            return labels, peaks
        if path.suffix.lower() == ".parquet":
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
        ids = self.tokenize(dna)
        if not ids:
            return torch.empty(0, self.n_features)
        with self._lock:
            hidden = self._forward_hidden([ids])[0]  # [S, H]
            return self.sae.encode(hidden.to(self.device)).detach().cpu()

    @torch.no_grad()
    def encode_batch(self, seqs: list[str], batch_size: int = 8) -> list[torch.Tensor]:
        """MANY sequences -> list of SAE codes [S_i, n_features], batched on the GPU.

        Sequences are padded to the longest in each micro-batch; padding is masked
        out before SAE-encoding so each result has the true per-base length.
        """
        out: list[torch.Tensor] = [None] * len(seqs)  # type: ignore
        order = [(i, self.tokenize(s)) for i, s in enumerate(seqs)]
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
        """Run the truncated model on a (padded) batch of token-id lists.

        Returns the unpadded layer-`layer` hidden states [S_i, H] per sequence.
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
        result = P._predict_step(model=self.model, batch=batch, output_embeddings=True)
        hidden = result["hidden_embeddings"]  # [B, S, H]
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
    def _clamp_hook(self, specs, pre_bias):
        """Forward hook that clamps SAE features on the residual during DECODE steps only.

        A decode step processes a single new token (sequence dim == 1); the prompt prefill
        (sequence dim > 1) is left untouched, giving continuation-only steering through
        `infer.generate`:  h <- h + Σ_f (t_f - a_f(h)) · d_f
        `specs` = list of (enc_f [H], b_f float, dec_f [H], target float).
        """

        def hook(_module, _inp, output):
            hs = output[0] if isinstance(output, tuple) else output  # [S, B, H]
            if hs.shape[0] != 1:  # prefill (whole prompt) — leave untouched
                return output
            x = hs.float()
            xc = x - pre_bias
            add = torch.zeros_like(x)
            for enc_f, b_f, dec_f, target in specs:
                a = torch.relu(torch.matmul(xc, enc_f) + b_f)
                add = add + (target - a).unsqueeze(-1) * dec_f
            new = (x + add).to(hs.dtype)
            return (new, *output[1:]) if isinstance(output, tuple) else new

        return hook

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
        from megatron.core.utils import unwrap_model

        from bionemo.evo2.run import infer as INF

        features = features or []
        resolved_tag = self.resolve_tag(organism, tag)
        if resolved_tag is None:
            raise ValueError(f"Unknown organism '{organism}' and no custom tag")
        dna = clean_dna(prompt)
        full_prompt = resolved_tag + dna
        if not full_prompt:
            raise ValueError("Provide a prompt or pick an organism (need >=1 token to seed)")
        n_tokens = max(1, min(int(n_tokens), 400))
        fids = [int(f["feature_id"]) for f in features]

        with self._lock:
            comp = self._ensure_engine()
            hook_layer = unwrap_model(comp.model).decoder.layers[self.layer]
            pre_bias = self.sae.pre_bias.detach().float().to(self.device)
            specs, feat_meta = [], []
            for f in features:
                fid = int(f["feature_id"])
                specs.append(
                    (
                        self.sae.encoder.weight[fid].detach().float().to(self.device),
                        float(self.sae.latent_bias[fid].detach()),
                        self.sae.decoder.weight[:, fid].detach().float().to(self.device),
                        float(f.get("strength", 1.0)),
                    )
                )
                feat_meta.append({"id": fid, "label": self.labels.get(fid), "strength": float(f.get("strength", 1.0))})

            def _run(steer: bool) -> str:
                handle = (
                    hook_layer.register_forward_hook(self._clamp_hook(specs, pre_bias)) if (steer and specs) else None
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
            base_dna = _run(steer=False) if (compare_baseline and specs) else None

        resp = {
            "prompt": dna,
            "organism": organism,
            "tag": resolved_tag,
            "tag_len": len(resolved_tag),
            "n_tokens": n_tokens,
            "features": feat_meta,
            "steered": bool(specs),
            "generation": {"sequence": main_dna, "activations": self.feature_tracks(main_dna, fids)},
            "baseline": None,
        }
        if base_dna is not None:
            resp["baseline"] = {"sequence": base_dna, "activations": self.feature_tracks(base_dna, fids)}
        return resp
