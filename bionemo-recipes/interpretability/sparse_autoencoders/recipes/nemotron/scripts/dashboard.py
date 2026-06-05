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

r"""Generate dashboard data from a trained Nemotron SAE.

Loads a trained SAE checkpoint + Nemotron-3-Nano, runs FineWeb text through both,
and exports the parquet/JSON files consumed by the ``atlas/`` text dashboard:

    features_atlas.parquet   -- one row per feature: UMAP coords, label, stats
    feature_metadata.parquet -- one row per feature: activation_freq, max_activation
    feature_examples.parquet -- one row per (feature, example): token-level activations
    vocab_logits.json        -- top promoted/suppressed tokens per (live) feature

This is the text analogue of ``codonfm/scripts/dashboard.py``: instead of per-codon
activations over coding sequences, it exports per-token activations over natural
language, so the dashboard highlights tokenized text instead of codon triplets.

Run on GPUs only -- it loads the 30B Nemotron model (model-parallel via
``device_map="auto"``; do NOT use torchrun).

Usage:
    python scripts/dashboard.py \
        --checkpoint outputs/stream_run/checkpoints/checkpoint_step_20000.pt \
        --layer 39 --num-texts 1000 --max-length 256 \
        --output-dir outputs/stream_run/dashboard

Then serve it:
    python scripts/launch_dashboard.py --data-dir outputs/stream_run/dashboard
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import torch
from nemotron_sae.data import load_fineweb
from nemotron_sae.models import NemotronModel
from sae.analysis import compute_feature_stats, compute_feature_umap, save_feature_atlas
from sae.architectures import ReLUSAE, TopKSAE
from sae.utils import get_device, set_seed
from tqdm import tqdm


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Generate Nemotron SAE dashboard data")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a single-file SAE checkpoint (.pt) OR a tensor-parallel sharded "
        "checkpoint directory (containing meta.json + shard_*.pt).",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=NemotronModel.DEFAULT_MODEL,
        help="Nemotron model name or local path (must match the extraction model)",
    )
    p.add_argument("--layer", type=int, default=39, help="Layer the SAE was trained on (must match extraction)")
    p.add_argument("--num-texts", type=int, default=1000, help="Number of FineWeb texts to run")
    p.add_argument("--max-length", type=int, default=256, help="Max tokens per text")
    p.add_argument("--batch-size", type=int, default=4, help="Texts per Nemotron forward pass")
    p.add_argument("--n-examples", type=int, default=6, help="Top examples per feature")
    p.add_argument("--logit-top-k", type=int, default=12, help="Top promoted/suppressed tokens per feature")
    p.add_argument("--output-dir", type=str, default="./outputs/dashboard")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_sae_from_checkpoint(checkpoint_path: str) -> torch.nn.Module:
    """Rebuild an SAE from a Trainer checkpoint (handles DDP ``module.`` prefix).

    Accepts either a single-file checkpoint (``.pt``) or a tensor-parallel sharded
    checkpoint directory (``meta.json`` + ``shard_*.pt``), which is merged into a dense
    ``TopKSAE``. Mirrors ``scripts/eval.py`` so the dashboard loads the same SAE the eval does.
    """
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir() and (ckpt_path / "meta.json").exists():
        from sae.parallel import load_and_merge

        print(f"Loading sharded (tensor-parallel) checkpoint from {ckpt_path} and merging...")
        return load_and_merge(str(ckpt_path))

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    input_dim = ckpt.get("input_dim")
    hidden_dim = ckpt.get("hidden_dim")
    if input_dim is None or hidden_dim is None:
        w = state_dict["encoder.weight"]
        hidden_dim = hidden_dim or w.shape[0]
        input_dim = input_dim or w.shape[1]

    mc = ckpt.get("model_config", {})
    if "top_k" in mc:
        sae = TopKSAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            top_k=mc["top_k"],
            normalize_input=mc.get("normalize_input", False),
            auxk=mc.get("auxk"),
            auxk_coef=mc.get("auxk_coef", 1 / 32),
            dead_tokens_threshold=mc.get("dead_tokens_threshold", 10_000_000),
        )
        print(f"Loaded TopKSAE: {input_dim} -> {hidden_dim:,} latents (top-{mc['top_k']})")
    else:
        sae = ReLUSAE(input_dim=input_dim, hidden_dim=hidden_dim, l1_coeff=mc.get("l1_coeff", 1e-2))
        print(f"Loaded ReLUSAE: {input_dim} -> {hidden_dim:,} latents")

    sae.load_state_dict(state_dict)
    return sae


def extract_text_activations(
    nemotron: NemotronModel,
    texts: List[str],
    max_length: int,
    batch_size: int,
) -> Tuple[List[List[str]], List[torch.Tensor]]:
    """Run Nemotron over texts, returning per-text token strings and activations.

    Returns:
        token_strs: list (per text) of token-string lists, length L_i
        embs:       list (per text) of CPU float32 tensors [L_i, hidden_dim]

    Pad tokens are dropped, so token_strs[i] aligns 1:1 with embs[i] rows.
    """
    tokenizer = nemotron.tokenizer
    token_strs: List[List[str]] = []
    embs: List[torch.Tensor] = []

    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Extracting activations"):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(nemotron.model.device)
        attention_mask = enc["attention_mask"].to(nemotron.model.device)

        hidden, mask = nemotron.forward_features(input_ids, attention_mask)  # [B, L, D], [B, L]
        hidden = hidden.float().cpu()
        mask = mask.bool().cpu()
        ids_cpu = input_ids.cpu()

        for b in range(hidden.shape[0]):
            keep = mask[b]
            valid_ids = ids_cpu[b][keep].tolist()
            if not valid_ids:
                continue
            token_strs.append([tokenizer.decode([t]) for t in valid_ids])
            embs.append(hidden[b][keep].contiguous())

        del hidden, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return token_strs, embs


def get_unembedding(nemotron: NemotronModel) -> Tuple[torch.Tensor, List[str]]:
    """Return (unembedding [vocab, hidden], vocab token strings).

    Uses the LM head (tied embeddings fall back to input embeddings). The decoder
    directions live in the residual stream at ``layer``; projecting them through the
    final unembedding is the standard "logit lens" approximation, matching how the
    codon dashboard projects through the Encodon LM head.
    """
    head = nemotron.model.get_output_embeddings()
    if head is not None and getattr(head, "weight", None) is not None:
        W = head.weight.detach()
    else:
        W = nemotron.model.get_input_embeddings().weight.detach()  # tied
    W = W.float()
    vocab_size = W.shape[0]
    tokenizer = nemotron.tokenizer
    vocab = [tokenizer.decode([i]) for i in range(vocab_size)]
    return W, vocab


def compute_token_logits(
    sae: torch.nn.Module,
    unembedding: torch.Tensor,
    vocab: List[str],
    live_ids: List[int],
    top_k: int,
    device: str,
    feature_chunk: int = 512,
) -> dict:
    """Memory-bounded top promoted/suppressed tokens per feature.

    The full (vocab x n_features) logit matrix is far too large for an LM vocab, so
    we project chunks of decoder columns through the (mean-centered) unembedding and
    keep only the top-k tokens each side. Returns {str(feature_id): {top_positive,
    top_negative}} for the requested ``live_ids`` only.
    """
    W_dec = sae.decoder.weight.detach().to(device)  # [hidden, n_features]
    U = unembedding.to(device)  # [vocab, hidden]
    # Mean-center across the vocab so values reflect feature-specific effects, not
    # the model's shared baseline bias toward common tokens.
    U = U - U.mean(dim=0, keepdim=True)

    live_ids = sorted(live_ids)
    result = {}
    for start in tqdm(range(0, len(live_ids), feature_chunk), desc="  Token logits"):
        chunk_ids = live_ids[start : start + feature_chunk]
        cols = W_dec[:, chunk_ids]  # [hidden, c]
        effects = U @ cols  # [vocab, c]
        pos_vals, pos_idx = torch.topk(effects, top_k, dim=0)
        neg_vals, neg_idx = torch.topk(-effects, top_k, dim=0)
        pos_vals, pos_idx = pos_vals.cpu(), pos_idx.cpu()
        neg_idx = neg_idx.cpu()
        neg_real = (-neg_vals).cpu()
        for j, fid in enumerate(chunk_ids):
            result[str(fid)] = {
                "top_positive": [[vocab[pos_idx[r, j].item()], round(pos_vals[r, j].item(), 3)] for r in range(top_k)],
                "top_negative": [[vocab[neg_idx[r, j].item()], round(neg_real[r, j].item(), 3)] for r in range(top_k)],
            }
        del effects, cols
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def export_text_features_parquet(
    sae: torch.nn.Module,
    token_strs: List[List[str]],
    embs: List[torch.Tensor],
    output_dir: Path,
    n_examples: int,
    device: str,
) -> List[int]:
    """Export per-token feature activations for the text dashboard.

    Two-pass algorithm (mirrors the codon dashboard):
        Pass 1: max activation per (text, feature)
        Pass 2: per-token activations for each feature's top examples only

    Returns the list of live feature ids (activation_freq > 0).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    sae = sae.eval().to(device)
    n_features = sae.hidden_dim
    n_texts = len(embs)

    # ── Pass 1: max activation per (text, feature) ───────────────────────
    print("  Pass 1: max activations per text...")
    max_acts = torch.zeros(n_texts, n_features)
    for i in tqdm(range(n_texts), desc="  Max activations"):
        with torch.no_grad():
            codes = sae.encode(embs[i].to(device))  # [L, n_features]
        max_acts[i] = codes.max(dim=0).values.cpu()
        del codes

    # Live features = fired on at least one token of one text.
    activation_freq = (max_acts > 0).float().mean(dim=0)
    live_ids = torch.nonzero(activation_freq > 0, as_tuple=False).squeeze(1).tolist()
    print(f"  {len(live_ids):,} / {n_features:,} features are live")

    # Top examples per (live) feature.
    top_indices = torch.topk(max_acts, k=min(n_examples, n_texts), dim=0).indices  # [n_examples, n_features]

    # Reverse index: which texts need re-encoding, for which features.
    needed = {}
    for feat_idx in live_ids:
        for rank in range(top_indices.shape[0]):
            t = int(top_indices[rank, feat_idx].item())
            if max_acts[t, feat_idx].item() <= 0:
                continue
            needed.setdefault(t, set()).add(feat_idx)

    # ── Pass 2: per-token activations for needed (text, feature) pairs ───
    print(f"  Pass 2: per-token activations ({len(needed)} texts)...")
    example_acts = {}
    for t in tqdm(sorted(needed.keys()), desc="  Per-token activations"):
        with torch.no_grad():
            codes = sae.encode(embs[t].to(device)).cpu()  # [L, n_features]
        for feat_idx in needed[t]:
            example_acts[(t, feat_idx)] = codes[:, feat_idx].numpy().tolist()
        del codes

    # ── feature_metadata.parquet ─────────────────────────────────────────
    print("  Writing feature_metadata.parquet...")
    meta_table = pa.table(
        {
            "feature_id": pa.array(list(range(n_features)), type=pa.int32()),
            "description": pa.array([f"Feature {i}" for i in range(n_features)]),
            "activation_freq": pa.array(activation_freq.tolist(), type=pa.float32()),
            "max_activation": pa.array(max_acts.max(dim=0).values.tolist(), type=pa.float32()),
        }
    )
    pq.write_table(meta_table, output_dir / "feature_metadata.parquet", compression="snappy")

    # ── feature_examples.parquet ─────────────────────────────────────────
    print("  Writing feature_examples.parquet...")
    rows = []
    for feat_idx in live_ids:
        for rank in range(top_indices.shape[0]):
            t = int(top_indices[rank, feat_idx].item())
            key = (t, feat_idx)
            if key not in example_acts:
                continue
            acts_list = example_acts[key]
            rows.append(
                {
                    "feature_id": feat_idx,
                    "example_rank": rank,
                    "text_id": t,
                    "tokens": token_strs[t],
                    "activations": acts_list,
                    "max_activation": max(acts_list) if acts_list else 0.0,
                }
            )

    rows.sort(key=lambda r: (r["feature_id"], r["example_rank"]))
    examples_table = pa.table(
        {
            "feature_id": pa.array([r["feature_id"] for r in rows], type=pa.int32()),
            "example_rank": pa.array([r["example_rank"] for r in rows], type=pa.int8()),
            "text_id": pa.array([r["text_id"] for r in rows], type=pa.int32()),
            "tokens": pa.array([r["tokens"] for r in rows], type=pa.list_(pa.utf8())),
            "activations": pa.array([r["activations"] for r in rows], type=pa.list_(pa.float32())),
            "max_activation": pa.array([r["max_activation"] for r in rows], type=pa.float32()),
        }
    )
    pq.write_table(
        examples_table,
        output_dir / "feature_examples.parquet",
        row_group_size=max(1, n_examples * 100),
        compression="snappy",
    )
    print(f"  Wrote {n_features:,} features, {len(rows):,} examples")

    return live_ids


def main():  # noqa: D103
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SAE
    sae = load_sae_from_checkpoint(args.checkpoint)

    # 2. Load Nemotron (model-parallel across visible GPUs via device_map="auto")
    print(f"\nLoading {args.model_name} (layer {args.layer})...")
    nemotron = NemotronModel(model_name=args.model_name, layer=args.layer, max_length=args.max_length)

    # 3. Load text corpus
    texts = load_fineweb(max_samples=args.num_texts, subset="sample-10BT")
    print(f"Loaded {len(texts)} texts for dashboard")

    # 4. Extract per-text token activations
    print("\nExtracting per-text activations...")
    token_strs, embs = extract_text_activations(nemotron, texts, args.max_length, args.batch_size)
    total_tokens = sum(e.shape[0] for e in embs)
    print(f"  {total_tokens:,} tokens across {len(embs)} texts, dim={embs[0].shape[1]}")

    # 5. Feature statistics (over the flattened token activations)
    print("\n[1/4] Computing feature statistics...")
    t0 = time.time()
    flat = torch.cat(embs, dim=0)
    stats, _ = compute_feature_stats(sae, flat, device=device)
    del flat
    print(f"       Done in {time.time() - t0:.1f}s")

    # 6. UMAP from decoder weights
    print("[2/4] Computing UMAP from decoder weights...")
    t0 = time.time()
    geometry = compute_feature_umap(
        sae,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.seed,
        compute_clusters=True,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
    )
    print(f"       Done in {time.time() - t0:.1f}s")

    # 7. Feature atlas
    print("[3/4] Saving feature atlas...")
    t0 = time.time()
    atlas_path = output_dir / "features_atlas.parquet"
    save_feature_atlas(stats, geometry, atlas_path)
    print(f"       Saved to {atlas_path} in {time.time() - t0:.1f}s")

    # 8. Per-token examples + metadata
    print("[4/4] Exporting token examples...")
    t0 = time.time()
    live_ids = export_text_features_parquet(
        sae=sae,
        token_strs=token_strs,
        embs=embs,
        output_dir=output_dir,
        n_examples=args.n_examples,
        device=device,
    )
    print(f"       Done in {time.time() - t0:.1f}s")

    # 9. Decoder logits (top promoted/suppressed tokens) for live features
    print("\nComputing decoder logits (vocab_logits.json)...")
    t0 = time.time()
    unembedding, vocab = get_unembedding(nemotron)
    print(f"  vocab size: {len(vocab):,}")
    # Free the 30B model before the logit projection.
    del nemotron
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logits = compute_token_logits(sae, unembedding, vocab, live_ids, top_k=args.logit_top_k, device=device)
    with open(output_dir / "vocab_logits.json", "w") as f:
        json.dump(logits, f)
    print(f"       Wrote vocab_logits.json ({len(logits):,} features) in {time.time() - t0:.1f}s")

    print(f"\nDashboard data saved to: {output_dir}")
    print(f"  Atlas:    {atlas_path}")
    print(f"  Metadata: {output_dir / 'feature_metadata.parquet'}")
    print(f"  Examples: {output_dir / 'feature_examples.parquet'}")
    print(f"  Logits:   {output_dir / 'vocab_logits.json'}")
    print("\nServe with: python scripts/launch_dashboard.py --data-dir " + str(output_dir))


if __name__ == "__main__":
    main()
