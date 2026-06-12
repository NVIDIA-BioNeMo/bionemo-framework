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

r"""Step 3: Evaluate SAE and build dashboard data with F1-annotated labels.

Loads a trained SAE checkpoint, runs F1 evaluation against SwissProt annotations,
computes loss recovered, and builds dashboard data (atlas + feature examples)
with biological annotation labels derived from the F1 results.

This is step 3 of the 3-step ESM2 SAE workflow:
    1. step1_15b_extract.py  -- extract activations from ESM2
    2. step2_15b_train.py    -- train SAE on cached activations
    3. step3_15b_eval.py     -- evaluate SAE + build dashboard

IMPORTANT: Run on a single GPU. Do NOT use torchrun.

    python scripts/step3_15b_eval.py \\
        --checkpoint ./outputs/650m/checkpoints/checkpoint_final.pt \\
        --model-name nvidia/esm2_t33_650M_UR50D \\
        --layer 24 --top-k 32 --dtype bf16 \\
        --output-dir ./outputs/650m/eval

Skip specific eval stages:
    python scripts/step3_15b_eval.py \\
        --checkpoint ./outputs/15b/checkpoints/checkpoint_final.pt \\
        --skip-loss-recovered \\
        --output-dir ./outputs/15b/eval
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sae.architectures import TopKSAE
from sae.utils import get_device, set_seed
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    """Parse command-line arguments for SAE evaluation."""
    p = argparse.ArgumentParser(description="Evaluate ESM2 SAE and build dashboard")

    # Checkpoint
    p.add_argument("--checkpoint", type=str, required=True, help="Path to SAE checkpoint .pt file")
    p.add_argument("--top-k", type=int, default=128, help="Top-k (must match training config)")

    # Model
    p.add_argument("--model-name", type=str, default="nvidia/esm2_t48_15B_UR50D")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for forward passes")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Model dtype (must match extraction dtype)"
    )

    # Data
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--num-proteins", type=int, default=2000, help="Proteins for dashboard data")
    p.add_argument("--output-dir", type=str, default="./outputs/eval")

    # F1 eval
    p.add_argument("--f1-max-proteins", type=int, default=1000)
    p.add_argument("--f1-min-positives", type=int, default=20)
    p.add_argument("--f1-threshold", type=float, default=0.5, help="F1 threshold for labeling features")
    p.add_argument(
        "--normalization-n-proteins", type=int, default=2000, help="Proteins for activation_max normalization"
    )

    # Loss recovered
    p.add_argument("--loss-recovered-n-sequences", type=int, default=100)

    # Annotation download
    p.add_argument(
        "--annotation-score",
        type=int,
        default=None,
        help="UniProt annotation score filter (1-5, None=no filter). Default None for max coverage.",
    )

    # Dashboard / UMAP
    p.add_argument("--umap-n-neighbors", type=int, default=50, help="UMAP n_neighbors parameter")
    p.add_argument("--umap-min-dist", type=float, default=0.0, help="UMAP min_dist parameter")
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=20, help="HDBSCAN min_cluster_size parameter")
    p.add_argument("--n-examples", type=int, default=6, help="Top proteins per feature for dashboard")

    # Skip flags
    p.add_argument("--skip-f1", action="store_true", help="Skip F1 evaluation")
    p.add_argument("--skip-loss-recovered", action="store_true", help="Skip loss recovered evaluation")
    p.add_argument("--skip-dashboard", action="store_true", help="Skip dashboard data generation")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ── Vocabulary logit analysis ─────────────────────────────────────────


def compute_vocab_logits(sae, model_name, model_dtype, device="cuda"):
    """Project SAE decoder through the ESM2 LM head to get per-feature token logits.

    Returns dict mapping feature_id -> {top_positive, top_negative} with
    mean-centered logit values (baseline subtracted).
    """
    from transformers import AutoModelForMaskedLM

    print("Loading LM head model for vocab logits...")
    lm_kwargs = {"trust_remote_code": True}
    if model_dtype != torch.float32:
        lm_kwargs["dtype"] = model_dtype
    lm_model = AutoModelForMaskedLM.from_pretrained(model_name, **lm_kwargs).to(device).eval()

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Get the LM head
    lm_head = lm_model.lm_head if hasattr(lm_model, "lm_head") else lm_model.cls

    # Decoder weights: (input_dim, n_features)
    W_dec = sae.decoder.weight.to(device).to(model_dtype)

    with torch.no_grad():
        logits = lm_head(W_dec.T).float()  # (n_features, output_vocab_size)

    # Subtract mean logit vector (baseline) so values reflect
    # feature-specific effects rather than the LM head's global bias.
    mean_logits = logits.mean(dim=0, keepdim=True)
    logits = logits - mean_logits

    # Build vocab list matching the LM head output dimension
    # (ESM2 pads output beyond tokenizer.vocab_size)
    output_vocab_size = logits.shape[1]
    vocab = []
    for i in range(output_vocab_size):
        if i < len(tokenizer):
            vocab.append(tokenizer.decode([i]).strip())
        else:
            vocab.append(f"<pad_{i}>")

    # Special tokens to exclude from top lists
    special_tokens = {"<cls>", "<pad>", "<eos>", "<unk>", "<mask>", "<sep>", "<null_1>"}

    valid_mask = torch.ones(output_vocab_size, dtype=torch.bool)
    for i, tok in enumerate(vocab):
        if tok.lower() in special_tokens or tok.startswith("<"):
            valid_mask[i] = False

    n_features = logits.shape[0]
    results = {}
    for f in range(n_features):
        feat_logits = logits[f].cpu()

        masked_logits = feat_logits.clone()
        masked_logits[~valid_mask] = float("-inf")

        top_pos_idx = masked_logits.topk(10).indices.tolist()

        masked_logits_neg = feat_logits.clone()
        masked_logits_neg[~valid_mask] = float("inf")
        top_neg_idx = masked_logits_neg.topk(10, largest=False).indices.tolist()

        top_positive = [(vocab[i], round(feat_logits[i].item(), 3)) for i in top_pos_idx]
        top_negative = [(vocab[i], round(feat_logits[i].item(), 3)) for i in top_neg_idx]

        results[f] = {
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

    del lm_model
    torch.cuda.empty_cache()

    print(f"  Computed mean-centered vocab logits for {n_features} features")
    return results


def load_sae_from_checkpoint(checkpoint_path: str, top_k: int) -> TopKSAE:
    """Load SAE from a Trainer checkpoint, handling DDP module. prefix."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    # Get dims from checkpoint metadata, or infer from encoder weight shape
    input_dim = ckpt.get("input_dim")
    hidden_dim = ckpt.get("hidden_dim")
    if input_dim is None or hidden_dim is None:
        w = state_dict["encoder.weight"]
        hidden_dim = hidden_dim or w.shape[0]
        input_dim = input_dim or w.shape[1]

    # Restore model config from checkpoint (saved by _get_config)
    model_config = ckpt.get("model_config", {})
    normalize_input = model_config.get("normalize_input", False)

    sae = TopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        top_k=top_k,
        normalize_input=normalize_input,
    )
    sae.load_state_dict(state_dict)

    print(f"Loaded SAE: {input_dim} -> {hidden_dim:,} latents (top-{top_k}, normalize_input={normalize_input})")
    return sae


# ── Activation extraction (matches step1_15b_extract.py exactly) ─────────


def load_esm2_model(model_name: str, dtype: torch.dtype, device: str):
    """Load ESM2 model + tokenizer the same way as step1_15b_extract.py."""
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
        "add_pooling_layer": False,
    }
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model = model.to(device).eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    return model, tokenizer


def _remove_special_tokens_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build mask excluding CLS (position 0) and EOS (last real token per sequence).

    Matches step1_15b_extract.py's special token removal logic exactly.
    """
    keep = attention_mask.clone()
    keep[:, 0] = 0  # Remove CLS
    lengths = attention_mask.sum(dim=1)
    for b in range(keep.shape[0]):
        eos = int(lengths[b].item()) - 1
        if eos > 0:
            keep[b, eos] = 0
    return keep


def extract_activations_3d(
    model: torch.nn.Module,
    tokenizer,
    sequences: List[str],
    layer: int,
    batch_size: int = 1,
    max_length: int = 512,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 3D activations matching step1_15b_extract.py's method.

    Returns:
        Tuple of (activations, masks) where:
        - activations: (n_sequences, max_seq_len, hidden_dim) float32, padded
        - masks: (n_sequences, max_seq_len) with CLS/EOS zeroed out
    """
    input_device = next(model.parameters()).device
    all_embeddings = []
    all_masks = []

    n_batches = (len(sequences) + batch_size - 1) // batch_size
    iterator = range(0, len(sequences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="Extracting activations")

    with torch.no_grad():
        for i in iterator:
            batch_seqs = sequences[i : i + batch_size]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer].float().cpu()  # bf16 -> float32
            mask = _remove_special_tokens_mask(inputs["attention_mask"].cpu())

            all_embeddings.append(hidden)
            all_masks.append(mask)

    # Pad to same seq_len across batches for 3D stacking
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


def extract_activations_flat(
    model: torch.nn.Module,
    tokenizer,
    sequences: List[str],
    layer: int,
    batch_size: int = 1,
    max_length: int = 512,
    show_progress: bool = True,
) -> torch.Tensor:
    """Extract flat activations (no padding) matching step1_15b_extract.py."""
    input_device = next(model.parameters()).device
    all_flat = []

    n_batches = (len(sequences) + batch_size - 1) // batch_size
    iterator = range(0, len(sequences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="Extracting activations")

    with torch.no_grad():
        for i in iterator:
            batch_seqs = sequences[i : i + batch_size]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer].float().cpu()
            mask = _remove_special_tokens_mask(inputs["attention_mask"].cpu())

            flat = hidden[mask.bool()]
            all_flat.append(flat)

    return torch.cat(all_flat, dim=0)


# ── F1 helpers ───────────────────────────────────────────────────────────


def build_f1_labels(val_results, n_features, f1_threshold):
    """Build feature labels from F1 results."""
    best_per_feature = {}
    for r in val_results:
        if r.feature_idx not in best_per_feature or r.f1_domain > best_per_feature[r.feature_idx].f1_domain:
            best_per_feature[r.feature_idx] = r

    labels = []
    feature_stats = {}
    for i in range(n_features):
        if i in best_per_feature and best_per_feature[i].f1_domain >= f1_threshold:
            r = best_per_feature[i]
            ann_short = r.concept.split(":")[-1] if ":" in r.concept else r.concept
            labels.append(f"{ann_short} (F1:{r.f1_domain:.2f})")
            feature_stats[i] = {
                "best_annotation": r.concept,
                "best_f1": float(r.f1_domain),
            }
        else:
            labels.append(f"Feature {i}")

    n_labeled = sum(1 for l in labels if not l.startswith("Feature "))
    print(f"  {n_labeled}/{n_features} features labeled (F1 >= {f1_threshold})")

    # Show all matched annotation categories
    from collections import Counter

    category_counts = Counter()
    for i, stats in feature_stats.items():
        concept = stats["best_annotation"]
        category = concept.split(":")[0] if ":" in concept else concept
        category_counts[category] += 1
    if category_counts:
        print(f"  Annotation categories matched ({len(category_counts)} types):")
        for cat, count in category_counts.most_common():
            print(f"    {cat}: {count} features")

    return labels, feature_stats


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    """Run SAE evaluation pipeline: F1 scores, loss recovered, and dashboard data."""
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    # ── 1. Load SAE ──────────────────────────────────────────────────
    sae = load_sae_from_checkpoint(args.checkpoint, args.top_k)
    n_features = sae.hidden_dim

    # ── 2. Load ESM2 (same as step1_15b_extract.py) ─────────────────
    esm2_model = None
    esm2_tokenizer = None

    def get_esm2():
        nonlocal esm2_model, esm2_tokenizer
        if esm2_model is None:
            print(f"Loading {args.model_name} (layer {args.layer}, {args.dtype})...")
            esm2_model, esm2_tokenizer = load_esm2_model(
                args.model_name,
                model_dtype,
                device,
            )
            print(f"  Layers: {esm2_model.config.num_hidden_layers}, Hidden: {esm2_model.config.hidden_size}")
        return esm2_model, esm2_tokenizer

    # ── 3. F1 Evaluation ─────────────────────────────────────────────
    f1_labels = None
    feature_stats_for_dashboard = None
    val_results = []
    test_results = []
    activation_max = None

    if not args.skip_f1:
        from esm2_sae.data import (
            download_annotated_proteins,
            download_swissprot,
            load_annotations_tsv,
            proteins_to_concept_labels,
            read_fasta,
        )
        from esm2_sae.eval import compute_activation_max, compute_f1_scores

        print("\n" + "=" * 60)
        print("F1 EVALUATION")
        print("=" * 60)

        # Download annotations
        annotations_path = data_dir / "swissprot_annotations.tsv.gz"
        if not annotations_path.exists():
            print("Downloading SwissProt annotations...")
            download_annotated_proteins(
                output_path=annotations_path,
                max_length=args.max_seq_len,
                reviewed_only=True,
                annotation_score=args.annotation_score,
                max_results=args.f1_max_proteins,
            )

        proteins, concept_counts = load_annotations_tsv(
            annotations_path,
            min_positives=args.f1_min_positives,
            max_proteins=args.f1_max_proteins,
            use_domain_ids=True,
        )

        if proteins:
            rng = np.random.RandomState(args.seed)
            indices = rng.permutation(len(proteins))
            mid = len(indices) // 2
            val_proteins = [proteins[i] for i in indices[:mid]]
            test_proteins = [proteins[i] for i in indices[mid:]]

            val_sequences, concept_labels_val = proteins_to_concept_labels(val_proteins)
            test_sequences, concept_labels_test = proteins_to_concept_labels(test_proteins)
            print(
                f"F1 eval: {len(val_proteins)} val + {len(test_proteins)} test proteins, "
                f"{len(concept_counts)} concepts"
            )

            model, tokenizer = get_esm2()
            print("Extracting embeddings for F1 evaluation...")
            val_embeddings_3d, val_masks = extract_activations_3d(
                model,
                tokenizer,
                val_sequences,
                args.layer,
                batch_size=args.batch_size,
                max_length=args.max_seq_len,
            )
            test_embeddings_3d, test_masks = extract_activations_3d(
                model,
                tokenizer,
                test_sequences,
                args.layer,
                batch_size=args.batch_size,
                max_length=args.max_seq_len,
            )

            # Compute activation_max for normalization
            norm_sequences = val_sequences[: min(args.normalization_n_proteins, len(val_sequences))]
            if norm_sequences:
                print(f"Computing activation_max from {len(norm_sequences)} proteins...")
                norm_emb, norm_masks = extract_activations_3d(
                    model,
                    tokenizer,
                    norm_sequences,
                    args.layer,
                    batch_size=args.batch_size,
                    max_length=args.max_seq_len,
                )
                activation_max = compute_activation_max(sae, norm_emb, norm_masks, device=device)
                print(f"  activation_max range: [{activation_max.min():.4f}, {activation_max.max():.4f}]")
                del norm_emb, norm_masks

            # Compute F1 scores
            print("Computing F1 scores (val)...")
            t0 = time.time()
            val_results = compute_f1_scores(
                sae=sae,
                embeddings=val_embeddings_3d,
                concept_labels=concept_labels_val,
                masks=val_masks,
                min_positives=args.f1_min_positives,
                device=device,
                show_progress=True,
                activation_max=activation_max,
            )
            print(f"  Val: {len(val_results)} pairs in {time.time() - t0:.1f}s")

            print("Computing F1 scores (test)...")
            t0 = time.time()
            test_results = compute_f1_scores(
                sae=sae,
                embeddings=test_embeddings_3d,
                concept_labels=concept_labels_test,
                masks=test_masks,
                min_positives=args.f1_min_positives,
                device=device,
                show_progress=True,
                activation_max=activation_max,
            )
            print(f"  Test: {len(test_results)} pairs in {time.time() - t0:.1f}s")

            # Build labels from val results
            print("Building feature labels from F1 results...")
            f1_labels, feature_stats_for_dashboard = build_f1_labels(
                val_results,
                n_features,
                args.f1_threshold,
            )

            # Save F1 summary
            f1_summary = _build_f1_summary(
                val_results,
                test_results,
                args.f1_threshold,
            )

            print("\nF1 Summary:")
            print(f"  Concepts matched:       {f1_summary['n_concepts_matched']}")
            print(f"  Mean F1 (domain, test): {f1_summary['mean_f1_domain_test']:.4f}")
            print(f"  Max F1 (domain, test):  {f1_summary['max_f1_domain_test']:.4f}")
            print(f"  Above {f1_summary['f1_threshold']} (val):    {f1_summary['n_above_threshold_val']}")
            print(f"  Above {f1_summary['f1_threshold']} (both):   {f1_summary['n_pairs_above_threshold_both']}")
            if f1_summary["top_pairs"]:
                print("  Top pairs (test):")
                for p in f1_summary["top_pairs"][:5]:
                    print(f"    Feature {p['feature']:>5d}  F1={p['f1_domain']:.3f}  {p['concept']}")

            f1_path = output_dir / "f1_results.json"
            with open(f1_path, "w") as f:
                json.dump(f1_summary, f, indent=2)
            print(f"Saved F1 results to {f1_path}")

            del val_embeddings_3d, test_embeddings_3d, val_masks, test_masks
        else:
            print("Warning: No annotated proteins found for F1 eval")

    # ── 4. Loss Recovered ────────────────────────────────────────────
    loss_recovered_result = None
    if not args.skip_loss_recovered:
        print("\n" + "=" * 60)
        print("LOSS RECOVERED EVALUATION")
        print("=" * 60)

        try:
            from esm2_sae.data import download_swissprot, read_fasta
            from esm2_sae.eval import evaluate_esm2_loss_recovered
            from transformers import AutoModelForMaskedLM

            # Free the base model to make room for the LM head model
            if esm2_model is not None:
                del esm2_model
                esm2_model = None
                torch.cuda.empty_cache()

            # Load LM head model (must match extraction dtype)
            print(f"Loading {args.model_name} with LM head ({args.dtype})...")
            lm_kwargs = {"trust_remote_code": True}
            if model_dtype != torch.float32:
                lm_kwargs["dtype"] = model_dtype
            esm_lm_model = AutoModelForMaskedLM.from_pretrained(args.model_name, **lm_kwargs).to(device)

            try:
                esm_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
            except (ValueError, ImportError):
                esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

            # Get sequences for loss recovered
            swissprot_path = data_dir / "uniprot_sprot.fasta.gz"
            if not swissprot_path.exists():
                download_swissprot(data_dir)
            records = read_fasta(swissprot_path, max_length=args.max_seq_len)
            lr_sequences = [r.sequence for r in records[: args.loss_recovered_n_sequences]]

            layer_idx = args.layer - 1
            print(f"Computing loss recovered on {len(lr_sequences)} sequences (layer_idx={layer_idx})...")
            loss_recovered_result = evaluate_esm2_loss_recovered(
                sae=sae,
                model=esm_lm_model,
                tokenizer=esm_tokenizer,
                sequences=lr_sequences,
                layer_idx=layer_idx,
                device=device,
            )
            print(f"  Loss recovered: {loss_recovered_result.loss_recovered:.4f}")
            print(f"  CE original: {loss_recovered_result.ce_original:.4f}")
            print(f"  CE SAE: {loss_recovered_result.ce_sae:.4f}")
            print(f"  CE zero: {loss_recovered_result.ce_zero:.4f}")

            # Save result
            lr_path = output_dir / "loss_recovered.json"
            with open(lr_path, "w") as f:
                json.dump(
                    {
                        "loss_recovered": loss_recovered_result.loss_recovered,
                        "ce_original": loss_recovered_result.ce_original,
                        "ce_sae": loss_recovered_result.ce_sae,
                        "ce_zero": loss_recovered_result.ce_zero,
                    },
                    f,
                    indent=2,
                )
            print(f"Saved to {lr_path}")

            del esm_lm_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Loss recovered failed: {e}")

    # ── 5. Dashboard Data ────────────────────────────────────────────
    if not args.skip_dashboard:
        print("\n" + "=" * 60)
        print("BUILDING DASHBOARD DATA")
        print("=" * 60)

        from esm2_sae.data import download_swissprot, read_fasta
        from esm2_sae.data_export import export_protein_features_parquet
        from sae.analysis import compute_feature_stats, compute_feature_umap, save_feature_atlas

        dashboard_dir = output_dir / "dashboard"
        dashboard_dir.mkdir(parents=True, exist_ok=True)

        # Load proteins for dashboard
        swissprot_path = data_dir / "uniprot_sprot.fasta.gz"
        if not swissprot_path.exists():
            download_swissprot(data_dir)
        records = read_fasta(swissprot_path, max_length=args.max_seq_len)
        records = records[: args.num_proteins]
        sequences = [r.sequence for r in records]
        protein_ids = [r.id for r in records]
        print(f"Loaded {len(sequences)} proteins for dashboard")

        model, tokenizer = get_esm2()
        print("Extracting 3D activations for dashboard...")
        activations, masks = extract_activations_3d(
            model,
            tokenizer,
            sequences,
            args.layer,
            batch_size=args.batch_size,
            max_length=args.max_seq_len,
        )
        activations_flat = activations[masks.bool()]
        print(f"  {activations_flat.shape[0]:,} residues, dim={activations_flat.shape[1]}")

        # Step 1: Feature statistics
        print("\n[1/5] Computing feature statistics...")
        t0 = time.time()
        stats, _ = compute_feature_stats(sae, activations_flat, device=device)
        print(f"       Done in {time.time() - t0:.1f}s")

        # Step 2: UMAP from decoder weights
        print("[2/5] Computing UMAP from decoder weights...")
        t0 = time.time()
        geometry = compute_feature_umap(
            sae,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.seed,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        )
        print(f"       Done in {time.time() - t0:.1f}s")

        # Step 3: Save feature atlas with F1 labels
        print("[3/5] Saving feature atlas...")
        t0 = time.time()
        atlas_path = dashboard_dir / "features_atlas.parquet"
        save_feature_atlas(stats, geometry, atlas_path, labels=f1_labels)
        print(f"       Saved to {atlas_path} in {time.time() - t0:.1f}s")

        # Step 4: Export protein examples with F1 annotations
        print("[4/5] Exporting protein examples...")
        t0 = time.time()
        export_protein_features_parquet(
            sae=sae,
            activations=activations,
            sequences=sequences,
            protein_ids=protein_ids,
            output_dir=dashboard_dir,
            masks=masks,
            n_examples=args.n_examples,
            device=device,
            feature_stats=feature_stats_for_dashboard,
        )
        print(f"       Done in {time.time() - t0:.1f}s")

        # Step 5: Compute vocab logits (decoder -> LM head projection)
        print("[5/5] Computing vocab logits...")
        t0 = time.time()
        vocab_logits = compute_vocab_logits(sae, args.model_name, model_dtype, device=device)
        logits_path = dashboard_dir / "vocab_logits.json"
        with open(logits_path, "w") as f:
            json.dump(vocab_logits, f)
        print(f"       Saved to {logits_path} in {time.time() - t0:.1f}s")

        print(f"\nDashboard data saved to: {dashboard_dir}")
        print(f"  Atlas:    {atlas_path}")
        print(f"  Features: {dashboard_dir}/feature_metadata.parquet")
        print(f"  Examples: {dashboard_dir}/feature_examples.parquet")
        print(f"  Logits:   {logits_path}")
        print("\nTo view locally:")
        print(f"  scp -r cluster:{dashboard_dir} ./dashboard")
        print(
            '  python -c "from esm2_sae import launch_protein_dashboard; '
            "proc = launch_protein_dashboard('dashboard/features_atlas.parquet', "
            "features_dir='dashboard'); "
            "input('Press Enter to stop.\\n'); proc.terminate()\""
        )

    # Free GPU
    if esm2_model is not None:
        del esm2_model
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"All results saved to: {output_dir}")
    print("=" * 60)


def _build_f1_summary(val_results, test_results, f1_threshold):
    """Build summary dict from val/test F1 results."""
    test_lookup = {}
    for r in test_results:
        key = (r.feature_idx, r.concept)
        if key not in test_lookup or r.f1_domain > test_lookup[key].f1_domain:
            test_lookup[key] = r

    best_per_concept_val = {}
    for r in val_results:
        if r.concept not in best_per_concept_val or r.f1_domain > best_per_concept_val[r.concept].f1_domain:
            best_per_concept_val[r.concept] = r

    test_matched = []
    for concept, val_r in best_per_concept_val.items():
        key = (val_r.feature_idx, concept)
        if key in test_lookup:
            test_matched.append(test_lookup[key])

    n_above_threshold_val = sum(1 for r in best_per_concept_val.values() if r.f1_domain > f1_threshold)
    n_above_threshold_both = sum(
        1
        for concept, val_r in best_per_concept_val.items()
        if val_r.f1_domain > f1_threshold
        and (val_r.feature_idx, concept) in test_lookup
        and test_lookup[(val_r.feature_idx, concept)].f1_domain > f1_threshold
    )

    test_f1d_vals = [r.f1_domain for r in test_matched] if test_matched else [0.0]
    top_pairs = sorted(test_matched, key=lambda x: x.f1_domain, reverse=True)[:10]

    return {
        "n_pairs_val": len(val_results),
        "n_pairs_test": len(test_results),
        "n_concepts_matched": len(test_matched),
        "mean_f1_domain_test": float(np.mean(test_f1d_vals)),
        "max_f1_domain_test": float(np.max(test_f1d_vals)),
        "n_above_threshold_val": n_above_threshold_val,
        "n_pairs_above_threshold_both": n_above_threshold_both,
        "f1_threshold": f1_threshold,
        "top_pairs": [
            {
                "feature": r.feature_idx,
                "concept": r.concept,
                "f1_domain": r.f1_domain,
                "f1": r.f1,
                "precision": r.precision,
                "recall_domain": r.recall_domain,
            }
            for r in top_pairs
        ],
    }


if __name__ == "__main__":
    main()
