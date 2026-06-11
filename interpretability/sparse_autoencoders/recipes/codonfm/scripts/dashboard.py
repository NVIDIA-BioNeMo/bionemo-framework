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

"""Generate dashboard data from a trained CodonFM SAE.

Loads the SAE checkpoint + Encodon model, runs sequences through both,
and exports feature statistics + per-sequence activation examples
to parquet files for the interactive dashboard.

    python scripts/dashboard.py \
        --checkpoint ./outputs/merged_1b/checkpoints/checkpoint_final.pt \
        --model-path /path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
        --layer -2 --top-k 32 \
        --csv-path /path/to/Primates.csv \
        --output-dir ./outputs/merged_1b/dashboard
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm


# Use codonfm_ptl_te recipe (has TransformerEngine support)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
_CODONFM_TE_DIR = _REPO_ROOT / "recipes" / "codonfm_ptl_te"
sys.path.insert(0, str(_CODONFM_TE_DIR))

from codonfm_sae.data import read_codon_csv  # noqa: E402
from sae.analysis import compute_feature_stats, compute_feature_umap, save_feature_atlas  # noqa: E402
from sae.architectures import TopKSAE  # noqa: E402
from sae.utils import get_device, set_seed  # noqa: E402
from src.data.preprocess.codon_sequence import process_item  # noqa: E402
from src.inference.encodon import EncodonInference  # noqa: E402


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Generate CodonFM SAE dashboard data")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to SAE checkpoint .pt file")
    p.add_argument("--top-k", type=int, default=None, help="Override top-k (default: read from checkpoint)")
    p.add_argument("--model-path", type=str, required=True, help="Path to Encodon checkpoint (.safetensors)")
    p.add_argument("--layer", type=int, default=-2)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--csv-path", type=str, required=True)
    p.add_argument("--seq-column", type=str, default=None)
    p.add_argument("--num-sequences", type=int, default=2000)
    p.add_argument("--n-examples", type=int, default=6, help="Top examples per feature")
    p.add_argument("--output-dir", type=str, default="./outputs/dashboard")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=20)
    p.add_argument(
        "--score-column", type=str, default=None, help="Model score column for variant analysis (auto-detect if None)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_sae_from_checkpoint(checkpoint_path: str, top_k_override: int | None = None) -> TopKSAE:  # noqa: D103
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

    model_config = ckpt.get("model_config", {})
    normalize_input = model_config.get("normalize_input", False)

    # Use checkpoint's top_k by default, allow CLI override
    top_k = top_k_override or model_config.get("top_k")
    if top_k is None:
        raise ValueError("top_k not found in checkpoint model_config. Pass --top-k explicitly.")
    if top_k_override and model_config.get("top_k") and top_k_override != model_config["top_k"]:
        print(f"  WARNING: overriding checkpoint top_k={model_config['top_k']} with --top-k={top_k_override}")

    sae = TopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        top_k=top_k,
        normalize_input=normalize_input,
    )
    sae.load_state_dict(state_dict)
    print(f"Loaded SAE: {input_dim} -> {hidden_dim:,} latents (top-{top_k})")
    return sae


def extract_activations_3d(
    inference,
    sequences: List[str],
    layer: int,
    context_length: int = 2048,
    batch_size: int = 8,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 3D activations (n_sequences, max_seq_len, hidden_dim) + masks.

    Returns padded activations and masks with CLS/SEP excluded.
    """
    all_embeddings = []
    all_masks = []

    n_batches = (len(sequences) + batch_size - 1) // batch_size
    iterator = tqdm(range(0, len(sequences), batch_size), total=n_batches, desc="Extracting activations")

    with torch.no_grad():
        for i in iterator:
            batch_seqs = sequences[i : i + batch_size]
            items = [process_item(s, context_length=context_length, tokenizer=inference.tokenizer) for s in batch_seqs]

            batch = {
                "input_ids": torch.tensor(np.stack([it["input_ids"] for it in items])).to(device),
                "attention_mask": torch.tensor(np.stack([it["attention_mask"] for it in items])).to(device),
            }

            out = inference.model(batch, return_hidden_states=True)
            hidden = out.all_hidden_states[layer].float().cpu()  # [B, L, D]
            attn_mask = batch["attention_mask"].cpu()

            # Build mask excluding CLS (pos 0) and SEP (last real pos)
            keep = attn_mask.clone()
            keep[:, 0] = 0
            lengths = attn_mask.sum(dim=1)
            for b in range(keep.shape[0]):
                sep = int(lengths[b].item()) - 1
                if sep > 0:
                    keep[b, sep] = 0

            all_embeddings.append(hidden)
            all_masks.append(keep)

            del out, batch
            torch.cuda.empty_cache()

    # Pad to same seq_len across batches
    max_len = max(e.shape[1] for e in all_embeddings)

    padded_emb = []
    padded_masks = []
    for emb, msk in zip(all_embeddings, all_masks):
        B, L, D = emb.shape
        if L < max_len:
            emb = torch.cat([emb, torch.zeros(B, max_len - L, D)], dim=1)
            msk = torch.cat([msk, torch.zeros(B, max_len - L, dtype=msk.dtype)], dim=1)
        padded_emb.append(emb)
        padded_masks.append(msk)

    return torch.cat(padded_emb, dim=0), torch.cat(padded_masks, dim=0)


def export_codon_features_parquet(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    sequences: List[str],
    sequence_ids: List[str],
    masks: torch.Tensor,
    output_dir: Path,
    n_examples: int = 6,
    device: str = "cuda",
    records: list | None = None,
    variant_delta_map: dict | None = None,
    precomputed_max_acts: torch.Tensor | None = None,
):
    """Export per-codon feature activations for dashboard.

    Two-pass algorithm:
        Pass 1: compute max activation per (sequence, feature)
        Pass 2: extract per-codon activations for top examples only
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_sequences = activations.shape[0]
    n_features = sae.hidden_dim

    sae = sae.eval().to(device)

    # Valid lengths per sequence (excluding CLS/SEP/padding)
    valid_lens = masks.sum(dim=1).long()

    # Pass 1: max activation per (sequence, feature) — reuse if precomputed
    if precomputed_max_acts is not None:
        print("  Reusing precomputed max activations...")
        max_acts = precomputed_max_acts
    else:
        print("  Pass 1: Computing max activations per sequence...")
        max_acts = torch.zeros(n_sequences, n_features)
        for i in tqdm(range(n_sequences), desc="  Max activations"):
            vl = int(valid_lens[i].item())
            if vl == 0:
                continue
            emb = activations[i, :vl, :].to(device)
            with torch.no_grad():
                _, codes = sae(emb)
            max_acts[i] = codes.max(dim=0).values.cpu()

    # Find top examples per feature
    print("  Finding top examples per feature...")
    top_indices = torch.topk(max_acts, k=min(n_examples, n_sequences), dim=0).indices  # (n_examples, n_features)

    # Build reverse index: which sequences need re-encoding
    needed_sequences = {}
    for feat_idx in range(n_features):
        for rank in range(top_indices.shape[0]):
            seq_idx = int(top_indices[rank, feat_idx].item())
            if seq_idx not in needed_sequences:
                needed_sequences[seq_idx] = set()
            needed_sequences[seq_idx].add(feat_idx)

    # Pass 2: extract per-codon activations for top examples
    print(f"  Pass 2: Extracting per-codon activations ({len(needed_sequences)} sequences)...")
    example_acts = {}

    for seq_idx in tqdm(sorted(needed_sequences.keys()), desc="  Per-codon activations"):
        vl = int(valid_lens[seq_idx].item())
        if vl == 0:
            continue
        emb = activations[seq_idx, :vl, :].to(device)
        with torch.no_grad():
            _, codes = sae(emb)  # (vl, n_features)
        codes_cpu = codes.cpu()

        for feat_idx in needed_sequences[seq_idx]:
            example_acts[(seq_idx, feat_idx)] = codes_cpu[:, feat_idx].numpy().tolist()

    # Build feature_metadata.parquet
    print("  Writing feature_metadata.parquet...")
    meta_rows = []
    for feat_idx in range(n_features):
        freq = (max_acts[:, feat_idx] > 0).float().mean().item()
        max_val = max_acts[:, feat_idx].max().item()
        meta_rows.append(
            {
                "feature_id": feat_idx,
                "description": f"Feature {feat_idx}",
                "activation_freq": freq,
                "max_activation": max_val,
            }
        )

    meta_table = pa.table(
        {
            "feature_id": pa.array([r["feature_id"] for r in meta_rows], type=pa.int32()),
            "description": pa.array([r["description"] for r in meta_rows]),
            "activation_freq": pa.array([r["activation_freq"] for r in meta_rows], type=pa.float32()),
            "max_activation": pa.array([r["max_activation"] for r in meta_rows], type=pa.float32()),
        }
    )
    pq.write_table(meta_table, output_dir / "feature_metadata.parquet", compression="snappy")

    # Build feature_examples.parquet
    print("  Writing feature_examples.parquet...")
    example_rows = []
    for feat_idx in range(n_features):
        for rank in range(top_indices.shape[0]):
            seq_idx = int(top_indices[rank, feat_idx].item())
            key = (seq_idx, feat_idx)
            if key not in example_acts:
                continue

            # Get the codon sequence (triplets)
            raw_seq = sequences[seq_idx]
            n_codons = len(raw_seq) // 3
            codon_seq = " ".join(raw_seq[i * 3 : (i + 1) * 3] for i in range(n_codons))

            acts_list = example_acts[key]
            seq_id = sequence_ids[seq_idx]

            row = {
                "feature_id": feat_idx,
                "example_rank": rank,
                "protein_id": seq_id,
                "sequence": codon_seq,
                "activations": acts_list,
                "max_activation": max(acts_list) if acts_list else 0.0,
            }

            # Add metadata from record if available
            if records is not None and seq_idx < len(records):
                meta = records[seq_idx].metadata
                row["gene"] = meta.get("gene", "")
                row["is_pathogenic"] = str(meta.get("is_pathogenic", ""))
                row["ref_codon"] = meta.get("ref_codon", "")
                row["alt_codon"] = meta.get("alt_codon", "")
                src = str(meta.get("source", "")).lower()
                row["source"] = "clinvar" if "clinvar" in src else ("cosmic" if "cosmic" in src else "")
                vpo = meta.get("var_pos_offset")
                row["var_pos_offset"] = int(float(vpo)) if vpo is not None else -1

                # Variant delta for this feature
                if variant_delta_map is not None and seq_idx in variant_delta_map:
                    row["variant_delta"] = float(variant_delta_map[seq_idx][feat_idx])
                else:
                    row["variant_delta"] = None

            example_rows.append(row)

    # Sort by feature_id for efficient row-group filtering
    example_rows.sort(key=lambda r: (r["feature_id"], r["example_rank"]))

    table_dict = {
        "feature_id": pa.array([r["feature_id"] for r in example_rows], type=pa.int32()),
        "example_rank": pa.array([r["example_rank"] for r in example_rows], type=pa.int8()),
        "protein_id": pa.array([r["protein_id"] for r in example_rows]),
        "sequence": pa.array([r["sequence"] for r in example_rows]),
        "activations": pa.array([r["activations"] for r in example_rows], type=pa.list_(pa.float32())),
        "max_activation": pa.array([r["max_activation"] for r in example_rows], type=pa.float32()),
    }

    # Add metadata columns if present
    if records is not None and example_rows and "gene" in example_rows[0]:
        table_dict["gene"] = pa.array([r.get("gene", "") for r in example_rows])
        table_dict["is_pathogenic"] = pa.array([r.get("is_pathogenic", "") for r in example_rows])
        table_dict["ref_codon"] = pa.array([r.get("ref_codon", "") for r in example_rows])
        table_dict["alt_codon"] = pa.array([r.get("alt_codon", "") for r in example_rows])
        table_dict["source"] = pa.array([r.get("source", "") for r in example_rows])
        table_dict["var_pos_offset"] = pa.array([r.get("var_pos_offset", -1) for r in example_rows], type=pa.int32())
        table_dict["variant_delta"] = pa.array([r.get("variant_delta") for r in example_rows], type=pa.float32())

    examples_table = pa.table(table_dict)

    row_group_size = n_examples * 100
    pq.write_table(
        examples_table, output_dir / "feature_examples.parquet", row_group_size=row_group_size, compression="snappy"
    )

    print(f"  Wrote {len(meta_rows)} features, {len(example_rows)} examples")


def compute_variant_analysis(
    sae: torch.nn.Module,
    records: list,
    activations: torch.Tensor,
    masks: torch.Tensor,
    device: str = "cuda",
    score_column: str | None = None,
) -> dict:
    """Compute per-feature variant analysis with multi-score, local deltas, and distribution metrics.

    For each feature computes:
      - mean_variant_{col} for each available score column (1b_cdwt, 5b_cdwt, 5b)
      - Global deltas (variant - ref, max over full sequence)
      - Local deltas (variant - ref, max over 3-codon window around variant site)
      - Site deltas (variant - ref, at exact variant position)
      - GC content distribution (mean, std) among activating sequences
      - Trinucleotide context distribution (entropy, dominant fraction) among activating variants
      - Gene distribution (entropy, n_unique, dominant fraction) among activating sequences
    """
    from collections import defaultdict

    SCORE_COLUMNS = ["1b_cdwt", "5b_cdwt", "5b"]
    WINDOW_RADIUS = 3  # codons each side of variant

    n_features = sae.hidden_dim
    valid_lens = masks.sum(dim=1).long()
    n_sequences = activations.shape[0]

    # Pre-read var_pos_offsets for the forward pass
    var_offsets_pre = []
    for r in records:
        vo = r.metadata.get("var_pos_offset")
        try:
            var_offsets_pre.append(int(float(vo)) if vo is not None else -1)
        except (ValueError, TypeError):
            var_offsets_pre.append(-1)

    # ── Pass 1: max activations + site/window activations ────────────
    print("  Computing per-sequence max activations...")
    max_acts = torch.zeros(n_sequences, n_features)
    site_acts = {}  # seq_idx -> [n_features] at var_pos
    window_max_acts = {}  # seq_idx -> [n_features] max over local window

    for i in tqdm(range(n_sequences), desc="  Max activations"):
        vl = int(valid_lens[i].item())
        if vl == 0:
            continue
        emb = activations[i, :vl, :].to(device)
        with torch.no_grad():
            _, codes = sae(emb)
        max_acts[i] = codes.max(dim=0).values.cpu()

        vpo = var_offsets_pre[i]
        if vpo >= 0 and vpo < vl:
            codes_cpu = codes.cpu()
            site_acts[i] = codes_cpu[vpo].numpy()
            w_start = max(0, vpo - WINDOW_RADIUS)
            w_end = min(vl, vpo + WINDOW_RADIUS + 1)
            window_max_acts[i] = codes_cpu[w_start:w_end].max(dim=0).values.numpy()

    max_acts_np = max_acts.numpy()

    # ── Read per-sequence metadata ───────────────────────────────────
    # Multi-score columns
    all_scores = {}  # col_name -> list[float|None]
    for col in SCORE_COLUMNS:
        col_scores = []
        for r in records:
            sc = r.metadata.get(col)
            try:
                col_scores.append(float(sc) if sc is not None else None)
            except (ValueError, TypeError):
                col_scores.append(None)
        if any(s is not None for s in col_scores):
            all_scores[col] = col_scores

    # Auto-detect primary score column
    if score_column is None:
        for candidate in SCORE_COLUMNS:
            if candidate in all_scores:
                score_column = candidate
                break
    if score_column:
        print(f"  Primary score column: {score_column}")
    print(f"  Score columns found: {list(all_scores.keys())}")

    phylop_vals = []
    var_offsets = []
    genes = []
    sources = []
    gc_contents = []
    trinuc_contexts = []

    for r in records:
        m = r.metadata

        pp = m.get("phylop")
        try:
            phylop_vals.append(float(pp) if pp is not None else None)
        except (ValueError, TypeError):
            phylop_vals.append(None)

        vo = m.get("var_pos_offset")
        try:
            var_offsets.append(int(float(vo)) if vo is not None else -1)
        except (ValueError, TypeError):
            var_offsets.append(-1)

        genes.append(m.get("gene", ""))

        src = str(m.get("source", "")).lower()
        if "clinvar" in src:
            sources.append("clinvar")
        elif "cosmic" in src:
            sources.append("cosmic")
        else:
            sources.append("other")

        gc = m.get("gc_content")
        try:
            gc_contents.append(float(gc) if gc is not None else None)
        except (ValueError, TypeError):
            gc_contents.append(None)

        trinuc_contexts.append(str(m.get("trinuc_context", "") or ""))

    # ── Per-feature mean variant score (per score column) ────────────
    mean_variant_scores = {}
    for col, col_scores in all_scores.items():
        score_sum = np.zeros(n_features)
        score_count = np.zeros(n_features)
        for i in range(n_sequences):
            if col_scores[i] is None or var_offsets[i] == -1:
                continue
            active = max_acts_np[i] > 0
            score_sum += active * col_scores[i]
            score_count += active
        col_key = col.replace("_", "")  # 1b_cdwt -> 1bcdwt
        mean_variant_scores[f"mean_variant_{col_key}"] = np.where(
            score_count > 0, score_sum / score_count, np.nan
        ).astype(np.float32)

    # High/low score split (primary score column)
    high_score_fire = np.zeros(n_features)
    low_score_fire = np.zeros(n_features)
    primary_scores = all_scores.get(score_column, [None] * n_sequences)
    valid_primary = [s for s in primary_scores if s is not None]
    median_score = float(np.median(valid_primary)) if valid_primary else 0.0
    if score_column:
        print(f"  Median variant score ({score_column}): {median_score:.4f}")

    for i in range(n_sequences):
        if primary_scores[i] is None or var_offsets[i] == -1:
            continue
        active = max_acts_np[i] > 0
        if primary_scores[i] >= median_score:
            high_score_fire += active
        else:
            low_score_fire += active

    total_scored = high_score_fire + low_score_fire
    high_score_fraction = np.where(total_scored > 0, high_score_fire / total_scored, np.nan)

    # ── Source enrichment (ClinVar fraction) ─────────────────────────
    clinvar_fire = np.zeros(n_features)
    cosmic_fire = np.zeros(n_features)
    for i in range(n_sequences):
        if var_offsets[i] == -1:
            continue
        active = max_acts_np[i] > 0
        if sources[i] == "clinvar":
            clinvar_fire += active
        elif sources[i] == "cosmic":
            cosmic_fire += active

    total_sourced = clinvar_fire + cosmic_fire
    clinvar_fraction = np.where(total_sourced > 0, clinvar_fire / total_sourced, np.nan)

    # ── Mean phyloP ──────────────────────────────────────────────────
    phylop_sum = np.zeros(n_features)
    phylop_count = np.zeros(n_features)
    for i in range(n_sequences):
        if phylop_vals[i] is None:
            continue
        active = max_acts_np[i] > 0
        phylop_sum += active * phylop_vals[i]
        phylop_count += active

    mean_phylop = np.where(phylop_count > 0, phylop_sum / phylop_count, np.nan)

    # ── GC content distribution per feature ──────────────────────────
    # Uses all sequences (gc_content is a whole-sequence property)
    gc_sum = np.zeros(n_features)
    gc_sq_sum = np.zeros(n_features)
    gc_count = np.zeros(n_features)
    for i in range(n_sequences):
        if gc_contents[i] is None:
            continue
        active = max_acts_np[i] > 0
        gc_val = gc_contents[i]
        gc_sum += active * gc_val
        gc_sq_sum += active * gc_val**2
        gc_count += active

    gc_mean = np.where(gc_count > 0, gc_sum / gc_count, np.nan).astype(np.float32)
    gc_var = np.where(gc_count > 1, gc_sq_sum / gc_count - (gc_sum / np.maximum(gc_count, 1)) ** 2, np.nan)
    gc_std = np.where(gc_var >= 0, np.sqrt(np.maximum(gc_var, 0)), np.nan).astype(np.float32)

    # ── Trinuc context distribution per feature ──────────────────────
    # Only variant rows (ref rows have no trinuc_context)
    unique_trinucs = sorted({t for t in trinuc_contexts if t})
    trinuc_to_idx = {t: i for i, t in enumerate(unique_trinucs)}
    n_trinucs = len(unique_trinucs)
    print(f"  {n_trinucs} unique trinucleotide contexts")

    trinuc_entropy = np.full(n_features, np.nan, dtype=np.float32)
    trinuc_dominant_frac = np.full(n_features, np.nan, dtype=np.float32)

    if n_trinucs > 0:
        trinuc_counts = np.zeros((n_features, n_trinucs))
        for i in range(n_sequences):
            if var_offsets[i] == -1 or not trinuc_contexts[i]:
                continue
            tidx = trinuc_to_idx.get(trinuc_contexts[i])
            if tidx is None:
                continue
            active = max_acts_np[i] > 0
            trinuc_counts[:, tidx] += active

        # Vectorized entropy: H = -sum(p * log2(p))
        totals = trinuc_counts.sum(axis=1)
        valid = totals > 0
        probs = np.zeros_like(trinuc_counts)
        probs[valid] = trinuc_counts[valid] / totals[valid, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.where(probs > 0, np.log2(probs), 0.0)
        trinuc_entropy_arr = -np.sum(probs * log_probs, axis=1)
        trinuc_entropy_arr[~valid] = np.nan
        trinuc_entropy = trinuc_entropy_arr.astype(np.float32)
        trinuc_dominant_frac = np.where(totals > 0, trinuc_counts.max(axis=1) / totals, np.nan).astype(np.float32)

    # ── Gene distribution per feature ────────────────────────────────
    # Uses all sequences (every row has a gene)
    unique_genes = sorted({g for g in genes if g})
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    n_genes_total = len(unique_genes)
    print(f"  {n_genes_total} unique genes")

    gene_entropy = np.full(n_features, np.nan, dtype=np.float32)
    gene_n_unique = np.zeros(n_features, dtype=np.int32)
    gene_dominant_frac = np.full(n_features, np.nan, dtype=np.float32)

    if n_genes_total > 0:
        gene_counts = np.zeros((n_features, n_genes_total))
        for i in range(n_sequences):
            if not genes[i]:
                continue
            gidx = gene_to_idx.get(genes[i])
            if gidx is None:
                continue
            active = max_acts_np[i] > 0
            gene_counts[:, gidx] += active

        totals = gene_counts.sum(axis=1)
        valid = totals > 0
        probs = np.zeros_like(gene_counts)
        probs[valid] = gene_counts[valid] / totals[valid, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.where(probs > 0, np.log2(probs), 0.0)
        gene_entropy_arr = -np.sum(probs * log_probs, axis=1)
        gene_entropy_arr[~valid] = np.nan
        gene_entropy = gene_entropy_arr.astype(np.float32)
        gene_n_unique = (gene_counts > 0).sum(axis=1).astype(np.int32)
        gene_dominant_frac = np.where(totals > 0, gene_counts.max(axis=1) / totals, np.nan).astype(np.float32)

    # ── Variant-ref deltas: global, site, and local window ───────────
    gene_groups = defaultdict(lambda: {"ref": None, "variants": []})
    for i, rec in enumerate(records):
        g = genes[i]
        if not g:
            continue
        if var_offsets[i] == -1:
            gene_groups[g]["ref"] = i
        else:
            gene_groups[g]["variants"].append(i)

    print("  Computing site-specific and local deltas...")
    ref_site_cache = {}  # (ref_idx, pos) -> [n_features]
    ref_window_cache = {}  # (ref_idx, pos) -> [n_features] max over window
    all_deltas = []

    for g, group in gene_groups.items():
        ref_idx = group["ref"]
        if ref_idx is None:
            continue
        ref_acts = max_acts_np[ref_idx]

        needed_positions = set()
        for vi in group["variants"]:
            vpo = var_offsets[vi]
            if vi in site_acts and vpo >= 0:
                needed_positions.add(vpo)

        # Single SAE forward pass per ref gene for site + window activations
        if needed_positions:
            vl = int(valid_lens[ref_idx].item())
            if vl > 0:
                emb = activations[ref_idx, :vl, :].to(device)
                with torch.no_grad():
                    _, ref_codes_full = sae(emb)
                ref_codes_cpu = ref_codes_full.cpu()
                for pos in needed_positions:
                    if pos < vl:
                        ref_site_cache[(ref_idx, pos)] = ref_codes_cpu[pos].numpy()
                        w_start = max(0, pos - WINDOW_RADIUS)
                        w_end = min(vl, pos + WINDOW_RADIUS + 1)
                        ref_window_cache[(ref_idx, pos)] = ref_codes_cpu[w_start:w_end].max(dim=0).values.numpy()
                del ref_codes_full, ref_codes_cpu

        for vi in group["variants"]:
            delta = max_acts_np[vi] - ref_acts  # global delta
            vpo = var_offsets[vi]
            site_delta = None
            local_delta = None
            if vi in site_acts and (ref_idx, vpo) in ref_site_cache:
                site_delta = site_acts[vi] - ref_site_cache[(ref_idx, vpo)]
            if vi in window_max_acts and (ref_idx, vpo) in ref_window_cache:
                local_delta = window_max_acts[vi] - ref_window_cache[(ref_idx, vpo)]
            sc = primary_scores[vi] if vi < len(primary_scores) else None
            all_deltas.append((delta, site_delta, local_delta, sc, sources[vi]))

    # Aggregate deltas
    mean_variant_delta = np.zeros(n_features)
    mean_site_delta = np.zeros(n_features)
    mean_local_delta = np.zeros(n_features)
    high_score_delta = np.zeros(n_features)
    low_score_delta = np.zeros(n_features)
    n_all = n_site = n_local = n_high = n_low = 0

    for delta, site_delta, local_delta, sc, src in all_deltas:
        mean_variant_delta += delta
        n_all += 1
        if site_delta is not None:
            mean_site_delta += site_delta
            n_site += 1
        if local_delta is not None:
            mean_local_delta += local_delta
            n_local += 1
        if sc is not None:
            if sc >= median_score:
                high_score_delta += delta
                n_high += 1
            else:
                low_score_delta += delta
                n_low += 1

    if n_all > 0:
        mean_variant_delta /= n_all
    if n_site > 0:
        mean_site_delta /= n_site
    if n_local > 0:
        mean_local_delta /= n_local
    if n_high > 0:
        high_score_delta /= n_high
    if n_low > 0:
        low_score_delta /= n_low

    n_genes_with_ref = sum(1 for g in gene_groups.values() if g["ref"] is not None)
    print(f"  {n_genes_with_ref} genes with ref, {n_all} variant-ref pairs ({n_site} site, {n_local} local window)")
    print(f"  {n_high} high-score, {n_low} low-score, {n_all - n_high - n_low} unscored")
    n_clinvar = sum(1 for s in sources if s == "clinvar")
    n_cosmic = sum(1 for s in sources if s == "cosmic")
    print(f"  {n_clinvar} ClinVar, {n_cosmic} COSMIC sequences")

    # Build per-sequence variant_delta for use in examples
    variant_delta_map = {}
    for g, group in gene_groups.items():
        ref_idx = group["ref"]
        if ref_idx is None:
            continue
        ref_acts = max_acts_np[ref_idx]
        for vi in group["variants"]:
            variant_delta_map[vi] = max_acts_np[vi] - ref_acts

    # ── Assemble extra_columns ───────────────────────────────────────
    extra_columns = {
        "high_score_fraction": high_score_fraction.astype(np.float32),
        "clinvar_fraction": clinvar_fraction.astype(np.float32),
        "mean_phylop": mean_phylop.astype(np.float32),
        "mean_variant_delta": mean_variant_delta.astype(np.float32),
        "mean_site_delta": mean_site_delta.astype(np.float32),
        "mean_local_delta": mean_local_delta.astype(np.float32),
        "high_score_delta": high_score_delta.astype(np.float32),
        "low_score_delta": low_score_delta.astype(np.float32),
        "gc_mean": gc_mean,
        "gc_std": gc_std,
        "trinuc_entropy": trinuc_entropy,
        "trinuc_dominant_frac": trinuc_dominant_frac,
        "gene_entropy": gene_entropy,
        "gene_n_unique": gene_n_unique,
        "gene_dominant_frac": gene_dominant_frac,
    }
    # Add per-score-column mean variant scores
    extra_columns.update(mean_variant_scores)

    return {
        "extra_columns": extra_columns,
        "max_acts": max_acts,
        "variant_delta_map": variant_delta_map,
    }


def main():  # noqa: D103
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SAE
    sae = load_sae_from_checkpoint(args.checkpoint, top_k_override=args.top_k)

    # 2. Load Encodon
    print(f"\nLoading Encodon from {args.model_path}...")
    inference = EncodonInference(
        model_path=args.model_path, task_type="embedding_prediction", use_transformer_engine=True
    )
    inference.configure_model()
    inference.model.to(device).eval()

    # 3. Load sequences
    max_codons = args.context_length - 2
    records = read_codon_csv(
        args.csv_path,
        seq_column=args.seq_column,
        max_sequences=args.num_sequences,
        max_codons=max_codons,
    )
    sequences = [r.sequence for r in records]
    sequence_ids = [r.id for r in records]
    print(f"Loaded {len(sequences)} sequences for dashboard")

    # 4. Extract 3D activations
    print("\nExtracting 3D activations...")
    activations, masks = extract_activations_3d(
        inference,
        sequences,
        args.layer,
        context_length=args.context_length,
        batch_size=args.batch_size,
        device=device,
    )
    activations_flat = activations[masks.bool()]
    print(f"  {activations_flat.shape[0]:,} codons, dim={activations_flat.shape[1]}")

    # 5. Feature statistics
    print("\n[1/4] Computing feature statistics...")
    t0 = time.time()
    stats, _ = compute_feature_stats(sae, activations_flat, device=device)
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

    # 7. Variant analysis (pathogenic enrichment + variant-ref deltas)
    print("[3/5] Computing variant analysis...")
    t0 = time.time()
    variant_results = compute_variant_analysis(
        sae,
        records,
        activations,
        masks,
        device=device,
        score_column=args.score_column,
    )
    print(f"       Done in {time.time() - t0:.1f}s")

    # 8. Feature atlas (with variant analysis columns)
    print("[4/5] Saving feature atlas...")
    t0 = time.time()
    atlas_path = output_dir / "features_atlas.parquet"
    save_feature_atlas(stats, geometry, atlas_path, extra_columns=variant_results["extra_columns"])
    print(f"       Saved to {atlas_path} in {time.time() - t0:.1f}s")

    # 9. Protein/codon examples
    print("[5/5] Exporting codon examples...")
    t0 = time.time()
    export_codon_features_parquet(
        sae=sae,
        activations=activations,
        sequences=sequences,
        sequence_ids=sequence_ids,
        masks=masks,
        output_dir=output_dir,
        n_examples=args.n_examples,
        device=device,
        records=records,
        variant_delta_map=variant_results["variant_delta_map"],
        precomputed_max_acts=variant_results["max_acts"],
    )
    print(f"       Done in {time.time() - t0:.1f}s")

    # Free GPU
    del inference
    torch.cuda.empty_cache()

    print(f"\nDashboard data saved to: {output_dir}")
    print(f"  Atlas:    {atlas_path}")
    print(f"  Metadata: {output_dir / 'feature_metadata.parquet'}")
    print(f"  Examples: {output_dir / 'feature_examples.parquet'}")


if __name__ == "__main__":
    main()
