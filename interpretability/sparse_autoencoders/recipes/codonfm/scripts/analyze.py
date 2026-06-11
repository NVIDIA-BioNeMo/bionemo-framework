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

"""Compute interpretability analysis for CodonFM SAE features.

Generates:
  - Vocabulary logit analysis (which codons each feature promotes/suppresses)
  - Codon-level computed annotations (usage bias, CpG, wobble, amino acid identity)
  - Auto-interp LLM-generated feature labels (optional)

Usage:
    python scripts/analyze.py \
        --checkpoint ./outputs/merged_1b/checkpoints/checkpoint_final.pt \
        --model-path /path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
        --layer -2 --top-k 32 \
        --csv-path /path/to/Primates.csv \
        --dashboard-dir ./outputs/merged_1b/dashboard \
        --output-dir ./outputs/merged_1b/analysis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# Use codonfm_ptl_te recipe (has TransformerEngine support)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
_CODONFM_TE_DIR = _REPO_ROOT / "recipes" / "codonfm_ptl_te"
sys.path.insert(0, str(_CODONFM_TE_DIR))

from codonfm_sae.data import read_codon_csv  # noqa: E402
from sae.architectures import TopKSAE  # noqa: E402
from sae.utils import get_device, set_seed  # noqa: E402
from src.data.preprocess.codon_sequence import process_item  # noqa: E402
from src.inference.encodon import EncodonInference  # noqa: E402


# ── Standard codon usage table (human, per 1000 codons) ──────────────
# Source: Kazusa Codon Usage Database, Homo sapiens
HUMAN_CODON_USAGE = {
    "TTT": 17.6,
    "TTC": 20.3,
    "TTA": 7.7,
    "TTG": 12.9,
    "CTT": 13.2,
    "CTC": 19.6,
    "CTA": 7.2,
    "CTG": 39.6,
    "ATT": 16.0,
    "ATC": 20.8,
    "ATA": 7.5,
    "ATG": 22.0,
    "GTT": 11.0,
    "GTC": 14.5,
    "GTA": 7.1,
    "GTG": 28.1,
    "TCT": 15.2,
    "TCC": 17.7,
    "TCA": 12.2,
    "TCG": 4.4,
    "CCT": 17.5,
    "CCC": 19.8,
    "CCA": 16.9,
    "CCG": 6.9,
    "ACT": 13.1,
    "ACC": 18.9,
    "ACA": 15.1,
    "ACG": 6.1,
    "GCT": 18.4,
    "GCC": 27.7,
    "GCA": 15.8,
    "GCG": 7.4,
    "TAT": 12.2,
    "TAC": 15.3,
    "TAA": 1.0,
    "TAG": 0.8,
    "CAT": 10.9,
    "CAC": 15.1,
    "CAA": 12.3,
    "CAG": 34.2,
    "AAT": 17.0,
    "AAC": 19.1,
    "AAA": 24.4,
    "AAG": 31.9,
    "GAT": 21.8,
    "GAC": 25.1,
    "GAA": 29.0,
    "GAG": 39.6,
    "TGT": 10.6,
    "TGC": 12.6,
    "TGA": 1.6,
    "TGG": 13.2,
    "CGT": 4.5,
    "CGC": 10.4,
    "CGA": 6.2,
    "CGG": 11.4,
    "AGT": 12.1,
    "AGC": 19.5,
    "AGA": 12.2,
    "AGG": 12.0,
    "GGT": 10.8,
    "GGC": 22.2,
    "GGA": 16.5,
    "GGG": 16.5,
}

# ── Precomputed codon optimality weights ─────────────────────────────
# CAI weight per codon: w_i = freq(codon) / max_freq(synonymous codons for same AA)
# RSCU per codon: observed_freq / (1/n_synonymous) = freq * n_synonymous / sum(freq for AA)
# tAI weights: human tRNA gene copy numbers (GtRNAdb, hg38)
# Source: Chan & Lowe, GtRNAdb 2.0 (2016)

_HUMAN_TRNA_COPY_NUMBERS = {
    "TTT": 10,
    "TTC": 20,
    "TTA": 6,
    "TTG": 11,
    "CTT": 10,
    "CTC": 20,
    "CTA": 5,
    "CTG": 20,
    "ATT": 15,
    "ATC": 23,
    "ATA": 5,
    "ATG": 23,
    "GTT": 11,
    "GTC": 14,
    "GTA": 5,
    "GTG": 16,
    "TCT": 11,
    "TCC": 17,
    "TCA": 7,
    "TCG": 4,
    "CCT": 10,
    "CCC": 12,
    "CCA": 13,
    "CCG": 5,
    "ACT": 10,
    "ACC": 20,
    "ACA": 10,
    "ACG": 6,
    "GCT": 16,
    "GCC": 34,
    "GCA": 10,
    "GCG": 6,
    "TAT": 10,
    "TAC": 16,
    "TAA": 0,
    "TAG": 0,
    "CAT": 10,
    "CAC": 15,
    "CAA": 10,
    "CAG": 34,
    "AAT": 14,
    "AAC": 20,
    "AAA": 15,
    "AAG": 34,
    "GAT": 17,
    "GAC": 25,
    "GAA": 16,
    "GAG": 40,
    "TGT": 10,
    "TGC": 20,
    "TGA": 0,
    "TGG": 10,
    "CGT": 6,
    "CGC": 15,
    "CGA": 5,
    "CGG": 5,
    "AGT": 8,
    "AGC": 18,
    "AGA": 10,
    "AGG": 8,
    "GGT": 10,
    "GGC": 22,
    "GGA": 10,
    "GGG": 8,
}


def _build_codon_weights():
    """Precompute CAI, RSCU, and tAI weight arrays for all 64 codons."""
    from collections import defaultdict

    # Group codons by amino acid
    aa_codons = defaultdict(list)
    for codon, aa in CODON_TO_AA.items():
        if aa != "*":
            aa_codons[aa].append(codon)

    # CAI weights: w_i = freq(codon) / max_freq(synonymous codons)
    cai_weights = {}
    for aa, codons in aa_codons.items():
        freqs = [HUMAN_CODON_USAGE.get(c, 0.0) for c in codons]
        max_freq = max(freqs) if freqs else 1.0
        for c, f in zip(codons, freqs):
            cai_weights[c] = f / max_freq if max_freq > 0 else 0.0

    # RSCU: observed / expected = freq * n_synonymous / sum(freqs for AA)
    rscu_values = {}
    for aa, codons in aa_codons.items():
        freqs = [HUMAN_CODON_USAGE.get(c, 0.0) for c in codons]
        total = sum(freqs)
        n_syn = len(codons)
        for c, f in zip(codons, freqs):
            rscu_values[c] = (f * n_syn / total) if total > 0 else 1.0

    # tAI weights: normalize by max tRNA copy number per AA family
    tai_weights = {}
    for aa, codons in aa_codons.items():
        copies = [_HUMAN_TRNA_COPY_NUMBERS.get(c, 0) for c in codons]
        max_copy = max(copies) if copies else 1
        for c, cp in zip(codons, copies):
            tai_weights[c] = cp / max_copy if max_copy > 0 else 0.0

    return cai_weights, rscu_values, tai_weights


CODON_TO_AA = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

_CAI_WEIGHTS, _RSCU_VALUES, _TAI_WEIGHTS = _build_codon_weights()


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Analyze CodonFM SAE features")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--top-k", type=int, default=None, help="Override top-k (default: read from checkpoint)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--layer", type=int, default=-2)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--csv-path", type=str, required=True, help="CSV with codon sequences (e.g. Primates.csv)")
    p.add_argument("--num-sequences", type=int, default=None, help="Max sequences to analyze (default: all)")
    p.add_argument(
        "--dashboard-dir", type=str, default=None, help="If provided, updates features_atlas.parquet with labels"
    )
    p.add_argument("--output-dir", type=str, default="./outputs/analysis")
    p.add_argument("--auto-interp", action="store_true", help="Run LLM auto-interpretation")
    p.add_argument(
        "--llm-provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "nim", "nvidia-internal"],
        help="LLM provider for auto-interp (default: anthropic)",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model name (defaults: anthropic=claude-sonnet-4-20250514, openai=gpt-4o, nim=nvidia/llama-3.1-nemotron-70b-instruct, nvidia-internal=aws/anthropic/bedrock-claude-3-7-sonnet-v1)",
    )
    p.add_argument("--max-features", type=int, default=None, help="Limit number of features to analyze (for testing)")
    p.add_argument(
        "--max-auto-interp-features",
        type=int,
        default=None,
        help="Limit auto-interp to top N features by activation frequency (default: all with codon annotations)",
    )
    p.add_argument(
        "--auto-interp-workers", type=int, default=1, help="Number of parallel workers for LLM calls (default: 1)"
    )
    p.add_argument(
        "--gsea-report",
        type=str,
        default=None,
        help="Path to gene_enrichment_report.json — adds GSEA context to auto-interp prompts",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_sae(checkpoint_path: str, top_k_override: int | None = None) -> TopKSAE:  # noqa: D103
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    w = state_dict["encoder.weight"]
    hidden_dim, input_dim = w.shape
    model_config = ckpt.get("model_config", {})
    top_k = top_k_override or model_config.get("top_k")
    if top_k is None:
        raise ValueError("top_k not found in checkpoint. Pass --top-k explicitly.")
    if top_k_override and model_config.get("top_k") and top_k_override != model_config["top_k"]:
        print(f"  WARNING: overriding checkpoint top_k={model_config['top_k']} with --top-k={top_k_override}")
    sae = TopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        top_k=top_k,
        normalize_input=model_config.get("normalize_input", False),
    )
    sae.load_state_dict(state_dict)
    return sae


# ── 1. Vocabulary logit analysis ─────────────────────────────────────


def compute_vocab_logits(sae, inference, device="cuda"):
    """Project SAE decoder through the Encodon LM head to get per-feature codon logits."""
    encodon = inference.model.model
    tokenizer = inference.tokenizer

    # Build vocab list indexed by token ID
    vocab = [tokenizer.decoder.get(i, f"<{i}>") for i in range(tokenizer.vocab_size)]

    # Get the LM head (cls module)
    lm_head = encodon.cls

    # Decoder weights: (input_dim, n_features)
    W_dec = sae.decoder.weight.to(device)

    # Project each feature's decoder column through the LM head
    with torch.no_grad():
        # LM head expects (batch, hidden_dim) and outputs (batch, vocab_size)
        logits = lm_head(W_dec.T)  # (n_features, vocab_size)

    # Subtract mean logit vector (baseline) so that values reflect
    # feature-specific effects rather than the LM head's global bias
    # toward common codons (e.g. GCC).  Without this, most features
    # look identical because the shared prior dominates.
    mean_logits = logits.mean(dim=0, keepdim=True)  # (1, vocab_size)
    logits = logits - mean_logits

    # Build per-feature top promoted/suppressed codons
    n_features = logits.shape[0]
    results = {}
    for f in range(n_features):
        feat_logits = logits[f].cpu()
        top_pos_idx = feat_logits.topk(10).indices.tolist()
        top_neg_idx = feat_logits.topk(10, largest=False).indices.tolist()

        top_positive = [(vocab[i], feat_logits[i].item()) for i in top_pos_idx]
        top_negative = [(vocab[i], feat_logits[i].item()) for i in top_neg_idx]

        # Group top positive by amino acid
        aa_counts = {}
        for codon, val in top_positive:
            aa = CODON_TO_AA.get(codon, "?")
            aa_counts[aa] = aa_counts.get(aa, 0) + 1

        results[f] = {
            "top_positive": top_positive,
            "top_negative": top_negative,
            "top_aa_counts": aa_counts,
        }

    return results


# ── 2. Streaming codon annotations + top-K tracking ──────────────────


def _summarize_codon_annotations(
    n_features,
    aa_counts,
    rare_counts,
    common_counts,
    cpg_counts,
    non_cpg_counts,
    wobble_gc_counts,
    wobble_at_counts,
    first30_counts,
    rest_counts,
    cai_log_sum=None,
    tai_log_sum=None,
    rscu_sum=None,
    optimality_count=None,
):
    """Summarize accumulated annotation counts into per-feature dicts."""
    all_aas = sorted(set(CODON_TO_AA.values()))

    results = {}
    for f in range(n_features):
        annotations = {}

        # Best amino acid
        total_fires = int(aa_counts[:, f].sum())
        if total_fires > 0:
            best_aa_idx = int(aa_counts[:, f].argmax())
            best_aa_count = int(aa_counts[best_aa_idx, f])
            if best_aa_count / total_fires > 0.3:
                annotations["amino_acid"] = {
                    "aa": all_aas[best_aa_idx],
                    "fraction": best_aa_count / total_fires,
                }

        # Rare vs common codons
        n_rare = int(rare_counts[f])
        n_common = int(common_counts[f])
        if n_rare + n_common > 10:
            rare_frac = n_rare / (n_rare + n_common)
            if rare_frac > 0.6:
                annotations["codon_usage"] = {"bias": "rare", "fraction": rare_frac}
            elif rare_frac < 0.2:
                annotations["codon_usage"] = {"bias": "common", "fraction": 1 - rare_frac}

        # CpG
        n_cpg = int(cpg_counts[f])
        n_non = int(non_cpg_counts[f])
        if n_cpg + n_non > 10:
            cpg_frac = n_cpg / (n_cpg + n_non)
            if cpg_frac > 0.3:
                annotations["cpg"] = {"enrichment": cpg_frac}

        # Wobble preference
        n_gc = int(wobble_gc_counts[f])
        n_at = int(wobble_at_counts[f])
        if n_gc + n_at > 10:
            gc_frac = n_gc / (n_gc + n_at)
            if gc_frac > 0.7:
                annotations["wobble"] = {"preference": "GC", "fraction": gc_frac}
            elif gc_frac < 0.3:
                annotations["wobble"] = {"preference": "AT", "fraction": 1 - gc_frac}

        # Position in gene
        n_first = int(first30_counts[f])
        n_rest = int(rest_counts[f])
        if n_first + n_rest > 10:
            first_frac = n_first / (n_first + n_rest)
            expected_frac = 30 / 600
            if first_frac > expected_frac * 3:
                annotations["position"] = {"region": "N-terminal", "enrichment": first_frac / expected_frac}

        # Codon optimality metrics (CAI, tAI, RSCU)
        if optimality_count is not None and optimality_count[f] > 10:
            n_opt = int(optimality_count[f])
            # CAI = geometric mean of weights = exp(mean(log(w)))
            if cai_log_sum is not None:
                cai = float(np.exp(cai_log_sum[f] / n_opt))
                annotations["cai"] = round(cai, 4)
            # tAI = geometric mean of tRNA adaptation weights
            if tai_log_sum is not None:
                tai = float(np.exp(tai_log_sum[f] / n_opt))
                annotations["tai"] = round(tai, 4)
            # RSCU = mean RSCU of active codons (1.0 = no bias)
            if rscu_sum is not None:
                mean_rscu = float(rscu_sum[f] / n_opt)
                annotations["rscu"] = round(mean_rscu, 4)

        if annotations:
            results[f] = annotations

    return results


def stream_annotations_and_topk(
    sae,
    inference,
    sequences,
    layer,
    context_length,
    batch_size,
    device="cuda",
    n_top_examples=5,
):
    """Single-pass streaming: extract activations, run SAE, accumulate codon stats + top-K per feature.

    Never materializes the full [n_sequences, max_len, hidden_dim] tensor.
    Memory usage: O(n_features * K) for top-K tracking + O(n_features * n_aa) for codon counts.

    Returns:
        codon_annotations: dict per feature
        top_acts: np.ndarray [n_features, K] - top activation values per feature
        top_indices: np.ndarray [n_features, K] - corresponding sequence indices
    """
    n_features = sae.hidden_dim
    n_sequences = len(sequences)
    K = n_top_examples

    # Codon annotation accumulators
    all_aas = sorted(set(CODON_TO_AA.values()))
    aa_to_idx = {aa: i for i, aa in enumerate(all_aas)}
    n_aa = len(all_aas)

    aa_counts = np.zeros((n_aa, n_features), dtype=np.int64)
    rare_counts = np.zeros(n_features, dtype=np.int64)
    common_counts = np.zeros(n_features, dtype=np.int64)
    cpg_counts = np.zeros(n_features, dtype=np.int64)
    non_cpg_counts = np.zeros(n_features, dtype=np.int64)
    wobble_gc_counts = np.zeros(n_features, dtype=np.int64)
    wobble_at_counts = np.zeros(n_features, dtype=np.int64)
    first30_counts = np.zeros(n_features, dtype=np.int64)
    rest_counts = np.zeros(n_features, dtype=np.int64)

    # CAI/tAI/RSCU accumulators: weighted sums over active codons
    cai_log_sum = np.zeros(n_features, dtype=np.float64)
    tai_log_sum = np.zeros(n_features, dtype=np.float64)
    rscu_sum = np.zeros(n_features, dtype=np.float64)
    optimality_count = np.zeros(n_features, dtype=np.int64)

    # Top-K tracking per feature (vectorized heap replacement)
    top_acts = np.full((n_features, K), -np.inf, dtype=np.float32)
    top_indices = np.full((n_features, K), -1, dtype=np.int64)

    print(f"  Streaming {n_sequences} sequences (batch_size={batch_size})...")
    n_batches = (n_sequences + batch_size - 1) // batch_size

    for batch_start in tqdm(range(0, n_sequences, batch_size), total=n_batches, desc="  Streaming"):
        batch_seqs = sequences[batch_start : batch_start + batch_size]
        items = [process_item(s, context_length=context_length, tokenizer=inference.tokenizer) for s in batch_seqs]

        batch_input = {
            "input_ids": torch.tensor(np.stack([it["input_ids"] for it in items])).to(device),
            "attention_mask": torch.tensor(np.stack([it["attention_mask"] for it in items])).to(device),
        }

        with torch.no_grad():
            out = inference.model(batch_input, return_hidden_states=True)
        hidden = out.all_hidden_states[layer].float()  # [B, L, D] on GPU
        attn = batch_input["attention_mask"]

        # Build mask excluding CLS/SEP
        keep = attn.clone()
        keep[:, 0] = 0
        lengths = attn.sum(dim=1)
        for b in range(keep.shape[0]):
            sep = int(lengths[b].item()) - 1
            if sep > 0:
                keep[b, sep] = 0

        # Process each sequence in the batch
        for b in range(len(batch_seqs)):
            seq_idx = batch_start + b
            vl = int(keep[b].sum().item())
            if vl == 0:
                continue

            # Match original behavior: take first vl positions
            emb = hidden[b, :vl, :]  # [vl, D] on GPU

            with torch.no_grad():
                _, codes = sae(emb)  # [vl, n_features]

            # ── Top-K tracking (vectorized) ──
            max_per_feat = codes.max(dim=0).values.cpu().numpy()  # [n_features]
            min_vals = top_acts.min(axis=1)
            min_positions = top_acts.argmin(axis=1)
            update_mask = max_per_feat > min_vals
            feat_indices = np.where(update_mask)[0]
            top_acts[feat_indices, min_positions[feat_indices]] = max_per_feat[feat_indices]
            top_indices[feat_indices, min_positions[feat_indices]] = seq_idx

            # ── Codon annotation stats ──
            codes_cpu = codes.cpu().numpy()
            seq = batch_seqs[b]
            codons = [seq[j * 3 : (j + 1) * 3].upper() for j in range(vl)]

            aa_idx = np.array([aa_to_idx.get(CODON_TO_AA.get(c, "?"), 0) for c in codons], dtype=np.int32)
            is_rare = np.array([HUMAN_CODON_USAGE.get(c, 10.0) < 10.0 for c in codons])
            wobble_chars = [c[2] if len(c) == 3 else "?" for c in codons]
            is_wobble_gc = np.array([w in ("G", "C") for w in wobble_chars])
            is_first30 = np.arange(vl) < 30

            is_cpg = np.zeros(vl, dtype=bool)
            for j in range(vl - 1):
                if len(codons[j]) == 3 and len(codons[j + 1]) >= 1:
                    is_cpg[j] = codons[j][2] == "C" and codons[j + 1][0] == "G"

            active = codes_cpu > 0

            for a in range(n_aa):
                pos_mask = aa_idx == a
                if pos_mask.any():
                    aa_counts[a] += active[pos_mask].sum(axis=0)

            rare_counts += active[is_rare].sum(axis=0) if is_rare.any() else 0
            common_counts += active[~is_rare].sum(axis=0) if (~is_rare).any() else 0
            cpg_counts += active[is_cpg].sum(axis=0) if is_cpg.any() else 0
            non_cpg_counts += active[~is_cpg].sum(axis=0) if (~is_cpg).any() else 0
            wobble_gc_counts += active[is_wobble_gc].sum(axis=0) if is_wobble_gc.any() else 0
            wobble_at_counts += active[~is_wobble_gc].sum(axis=0) if (~is_wobble_gc).any() else 0
            first30_counts += active[is_first30].sum(axis=0) if is_first30.any() else 0
            rest_counts += active[~is_first30].sum(axis=0) if (~is_first30).any() else 0

            # CAI/tAI/RSCU: accumulate log-weights for active codons (excluding stop codons)
            cai_w = np.array([_CAI_WEIGHTS.get(c, 0.0) for c in codons], dtype=np.float64)
            tai_w = np.array([_TAI_WEIGHTS.get(c, 0.0) for c in codons], dtype=np.float64)
            rscu_v = np.array([_RSCU_VALUES.get(c, 1.0) for c in codons], dtype=np.float64)
            # Mask out stop codons and codons with zero weight
            non_stop = np.array([CODON_TO_AA.get(c, "*") != "*" for c in codons])
            valid_cai = non_stop & (cai_w > 0)
            valid_tai = non_stop & (tai_w > 0)

            if valid_cai.any():
                log_cai = np.zeros(vl, dtype=np.float64)
                log_cai[valid_cai] = np.log(cai_w[valid_cai])
                # Sum of log(w) for active codons at each position -> per feature
                cai_log_sum += (active[valid_cai] * log_cai[valid_cai, None]).sum(axis=0)

            if valid_tai.any():
                log_tai = np.zeros(vl, dtype=np.float64)
                log_tai[valid_tai] = np.log(tai_w[valid_tai])
                tai_log_sum += (active[valid_tai] * log_tai[valid_tai, None]).sum(axis=0)

            if non_stop.any():
                rscu_sum += (active[non_stop] * rscu_v[non_stop, None]).sum(axis=0)
                optimality_count += active[non_stop].sum(axis=0)

        del out, batch_input, hidden
        torch.cuda.empty_cache()

    # Summarize
    print("  Summarizing annotations...")
    codon_annotations = _summarize_codon_annotations(
        n_features,
        aa_counts,
        rare_counts,
        common_counts,
        cpg_counts,
        non_cpg_counts,
        wobble_gc_counts,
        wobble_at_counts,
        first30_counts,
        rest_counts,
        cai_log_sum,
        tai_log_sum,
        rscu_sum,
        optimality_count,
    )

    return codon_annotations, top_acts, top_indices


# ── 3. Auto-interpretation ───────────────────────────────────────────


def get_llm_client(provider: str, model: str | None = None):
    """Create LLM client based on provider."""
    from sae.autointerp import (
        AnthropicClient,
        NIMClient,
        NVIDIAInternalClient,
        OpenAIClient,
    )

    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-sonnet-4-20250514")
    elif provider == "openai":
        return OpenAIClient(model=model or "gpt-4o")
    elif provider == "nim":
        return NIMClient(model=model or "nvidia/llama-3.1-nemotron-70b-instruct")
    elif provider == "nvidia-internal":
        return NVIDIAInternalClient(model=model or "aws/anthropic/bedrock-claude-3-7-sonnet-v1")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def run_auto_interp(
    sae,
    vocab_logits,
    inference,
    sequences,
    records,
    feature_indices,
    top_indices,
    layer,
    context_length,
    batch_size,
    device="cuda",
    llm_provider="anthropic",
    llm_model=None,
    num_workers=1,
    gsea_context=None,
):
    """Run LLM auto-interpretation using precomputed top-K indices.

    Streams through needed sequences, extracting only the per-feature activation
    columns required. Never caches full [vl, n_features] tensors (which OOM at scale).
    """
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = get_llm_client(llm_provider, llm_model)

    # Build reverse index: seq_idx -> set of feature_ids that need it
    seq_to_features = defaultdict(set)
    for f in feature_indices:
        for k in range(top_indices.shape[1]):
            si = int(top_indices[f, k])
            if si >= 0:
                seq_to_features[si].add(f)

    # Storage: feature_id -> list of (max_act, seq_idx, per_codon_acts_numpy)
    # Only stores the single feature column per sequence (~200 floats), not all 32k
    feature_acts = defaultdict(list)

    unique_indices = sorted(seq_to_features.keys())
    print(f"  Re-extracting {len(unique_indices)} unique sequences for {len(feature_indices)} features...")

    n_batches = (len(unique_indices) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(unique_indices), batch_size), total=n_batches, desc="  Re-extracting"):
        batch_idx = unique_indices[batch_start : batch_start + batch_size]
        batch_seqs = [sequences[i] for i in batch_idx]
        items = [process_item(s, context_length=context_length, tokenizer=inference.tokenizer) for s in batch_seqs]

        batch_input = {
            "input_ids": torch.tensor(np.stack([it["input_ids"] for it in items])).to(device),
            "attention_mask": torch.tensor(np.stack([it["attention_mask"] for it in items])).to(device),
        }

        with torch.no_grad():
            out = inference.model(batch_input, return_hidden_states=True)
        hidden = out.all_hidden_states[layer].float()
        attn = batch_input["attention_mask"]

        keep = attn.clone()
        keep[:, 0] = 0
        lengths = attn.sum(dim=1)
        for b in range(keep.shape[0]):
            sep = int(lengths[b].item()) - 1
            if sep > 0:
                keep[b, sep] = 0

        for b in range(len(batch_idx)):
            si = batch_idx[b]
            vl = int(keep[b].sum().item())
            if vl == 0:
                continue

            emb = hidden[b, :vl, :]
            with torch.no_grad():
                _, codes = sae(emb)  # [vl, n_features] on GPU

            # Extract only the feature columns needed for this sequence
            needed_feats = sorted(seq_to_features[si])
            feat_idx_tensor = torch.tensor(needed_feats, dtype=torch.long, device=codes.device)
            selected = codes[:, feat_idx_tensor].cpu().numpy()  # [vl, len(needed_feats)]

            for col, f in enumerate(needed_feats):
                acts = selected[:, col]  # [vl]
                feature_acts[f].append((float(acts.max()), si, acts))

            del codes

        del out, batch_input, hidden
        torch.cuda.empty_cache()

    # Build per-feature example strings
    print("  Preparing examples for auto-interp...")
    feature_examples = {}
    for f in tqdm(feature_indices, desc="  Collecting examples"):
        entries = sorted(feature_acts.get(f, []), reverse=True, key=lambda x: x[0])[:5]

        examples = []
        for max_act, seq_idx, acts in entries:
            seq = sequences[seq_idx]
            vl = len(acts)
            codons = [seq[j * 3 : (j + 1) * 3] for j in range(vl)]

            # Mark top activating codons
            threshold = np.percentile(acts[acts > 0], 80) if (acts > 0).sum() > 0 else 0
            marked = []
            for j, (codon, act) in enumerate(zip(codons, acts)):
                aa = CODON_TO_AA.get(codon.upper(), "?")
                if act > threshold:
                    marked.append(f"***{codon}({aa})***")
                else:
                    marked.append(f"{codon}({aa})")

            # Build metadata string
            meta_str = ""
            if records is not None and seq_idx < len(records):
                m = records[seq_idx].metadata
                meta_parts = []
                gene = m.get("gene")
                if gene:
                    meta_parts.append(f"Gene: {gene}")
                src = m.get("source")
                if src:
                    meta_parts.append(f"Source: {src}")
                ip = m.get("is_pathogenic")
                if ip and str(ip).lower() not in ("", "unknown"):
                    meta_parts.append(f"Pathogenic: {ip}")
                pp = m.get("phylop")
                if pp is not None:
                    meta_parts.append(f"PhyloP: {pp:.2f}")
                ref = m.get("ref_codon")
                alt = m.get("alt_codon")
                vpo = m.get("var_pos_offset")
                if ref and alt:
                    meta_parts.append(f"Variant: {ref}>{alt} at pos {vpo}")
                for score_col in ["1b_cdwt", "5b_cdwt", "1b", "5b"]:
                    sc = m.get(score_col)
                    if sc is not None:
                        meta_parts.append(f"Model score ({score_col}): {float(sc):.3f}")
                        break
                if meta_parts:
                    meta_str = f" [{', '.join(meta_parts)}]"

            examples.append(f"{' '.join(marked)}{meta_str}")

        feature_examples[f] = examples

    # Build prompts and call LLM in parallel
    print(f"  Running LLM interpretation with {num_workers} workers...")

    def interpret_feature(f):
        logits_info = vocab_logits.get(f, {})
        top_pos = logits_info.get("top_positive", [])[:5]
        top_neg = logits_info.get("top_negative", [])[:5]

        pos_str = ", ".join(f"{tok}({CODON_TO_AA.get(tok, '?')}): {v:.2f}" for tok, v in top_pos)
        neg_str = ", ".join(f"{tok}({CODON_TO_AA.get(tok, '?')}): {v:.2f}" for tok, v in top_neg)

        examples_str = "\n".join(f"  Seq {i + 1}: {ex}" for i, ex in enumerate(feature_examples.get(f, [])))

        # Build GSEA enrichment context if available
        gsea_str = ""
        if gsea_context and f in gsea_context:
            gsea_info = gsea_context[f]
            gsea_lines = []
            for db, entry in gsea_info.items():
                if entry:
                    gsea_lines.append(f"  {db}: {entry['term_name']} (FDR={entry['fdr']:.4f})")
            if gsea_lines:
                gsea_str = "\n\nGene-level GSEA enrichment (genes ranked by activation, tested against annotation databases):\n"
                gsea_str += "\n".join(gsea_lines)

        prompt = f"""Analyze this sparse autoencoder feature from a DNA codon language model (CodonFM) to determine what predicts its activation pattern. Each token is a codon (3 nucleotides encoding one amino acid).

Top promoted codons (decoder logits): {pos_str}
Top suppressed codons: {neg_str}

Top activating sequences (***highlighted*** = high activation codons):
Metadata in brackets may include: gene name, data source (ClinVar/COSMIC), pathogenicity, PhyloP conservation, variant info (ref>alt codon at position), model effect score.
{examples_str}{gsea_str}

Analyze what predicts high vs low activation for this feature. This description should be concise but sufficient to predict activation levels on unseen codon sequences. The feature could be specific to a gene family, a codon usage pattern, a sequence motif, a functional role, a structural domain, etc.

Focus on:
- Which codons and amino acids are associated with high vs low activation, and whether specific synonymous codon choices matter
- Where in the gene sequence activation occurs (N-terminal, C-terminal, or throughout)
- What gene-level functional annotations (from GSEA enrichment if provided) characterize the top-activating genes
- Whether codon usage bias, CpG content, wobble position patterns, or GC content are relevant
- Any variant/clinical metadata patterns (pathogenicity, conservation, mutation impact)

Your description will be used to predict activation on held-out sequences, so only highlight factors relevant for prediction.

Format your response as:
Description: <2-3 sentences starting with "The activation patterns are characterized by:">
Label: <one concise phrase summarizing what this feature detects>
Confidence: <0.00 to 1.00>"""

        try:
            response = client.generate(prompt)
            text = response.text.strip()

            label = None
            description = None
            confidence = 0.0

            for line in text.split("\n"):
                if line.startswith("Label:"):
                    label = line.replace("Label:", "").strip()
                elif line.startswith("Description:"):
                    description = line.replace("Description:", "").strip()
                elif line.startswith("Confidence:"):
                    try:
                        confidence = float(line.replace("Confidence:", "").strip())
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        confidence = 0.0

            if not label:
                label = f"Feature {f}"

            return f, label, confidence, description
        except Exception as e:
            print(f"  Warning: auto-interp failed for feature {f}: {e}")
            return f, f"Feature {f}", 0.0, None

    interpretations = {}
    confidences = {}
    descriptions = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(interpret_feature, f): f for f in feature_indices}
        for future in tqdm(as_completed(futures), total=len(feature_indices), desc="  Auto-interp"):
            f, label, confidence, description = future.result()
            interpretations[f] = label
            confidences[f] = confidence
            if description:
                descriptions[f] = description

    return interpretations, confidences, descriptions


# ── Build summary labels ─────────────────────────────────────────────


def build_feature_labels(
    n_features,
    vocab_logits,
    codon_annotations,
    auto_interp_labels=None,
):
    """Combine all analyses into a single label per feature."""
    labels = {}
    details = {}
    llm_confidences = {}

    for f in range(n_features):
        parts = []

        # Auto-interp label takes priority
        if auto_interp_labels and f in auto_interp_labels:
            label_entry = auto_interp_labels[f]
            if isinstance(label_entry, dict):
                labels[f] = label_entry.get("label", f"Feature {f}")
                llm_confidences[f] = label_entry.get("confidence", 0.0)
            else:
                labels[f] = label_entry
                llm_confidences[f] = 0.0
            details[f] = {
                "label": labels[f],
                "llm_confidence": llm_confidences[f],
                "vocab_logits": vocab_logits.get(f, {}),
                "codon_annotations": codon_annotations.get(f, {}),
            }
            continue

        llm_confidences[f] = 0.0

        ann = codon_annotations.get(f, {})
        if "amino_acid" in ann:
            aa = ann["amino_acid"]["aa"]
            frac = ann["amino_acid"]["fraction"]
            parts.append(f"{aa} ({frac:.0%})")
        if "codon_usage" in ann:
            parts.append(f"{ann['codon_usage']['bias']} codons")
        if "wobble" in ann:
            parts.append(f"wobble {ann['wobble']['preference']}")
        if "cpg" in ann:
            parts.append("CpG enriched")
        if "position" in ann:
            parts.append("N-terminal")

        if parts:
            labels[f] = " | ".join(parts)
        else:
            labels[f] = f"Feature {f}"

        details[f] = {
            "label": labels[f],
            "llm_confidence": llm_confidences[f],
            "vocab_logits": vocab_logits.get(f, {}),
            "codon_annotations": codon_annotations.get(f, {}),
        }

    return labels, details, llm_confidences


# ── Main ─────────────────────────────────────────────────────────────


def main():  # noqa: D103
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    # Load SAE
    sae = load_sae(args.checkpoint, top_k_override=args.top_k).eval().to(device)
    n_features = sae.hidden_dim
    print(f"SAE: {sae.input_dim} -> {n_features} features")

    # Load model
    print(f"\nLoading Encodon from {args.model_path}...")
    inference = EncodonInference(
        model_path=args.model_path, task_type="embedding_prediction", use_transformer_engine=True
    )
    inference.configure_model()
    inference.model.to(device).eval()

    # Load sequences
    max_codons = args.context_length - 2
    records = read_codon_csv(
        args.csv_path,
        max_sequences=args.num_sequences,
        max_codons=max_codons,
    )
    sequences = [r.sequence for r in records]
    print(f"Loaded {len(sequences)} sequences")

    # ── Analysis ─────────────────────────────────────────────────────

    # 1. Vocabulary logits (only uses SAE decoder, no activations needed)
    print("\n[1/3] Vocabulary logit analysis...")
    vocab_logits_file = output_dir / "vocab_logits_checkpoint.json"
    if vocab_logits_file.exists():
        print("  Loading vocab logits from checkpoint...")
        with open(vocab_logits_file) as f:
            vocab_logits = json.load(f)
            vocab_logits = {int(k): v for k, v in vocab_logits.items()}
    else:
        vocab_logits = compute_vocab_logits(sae, inference, device)
        with open(vocab_logits_file, "w") as f:
            json.dump(vocab_logits, f)
    print(f"  Computed logits for {len(vocab_logits)} features")

    # 2. Streaming codon annotations + top-K tracking (single pass, constant memory)
    print("\n[2/3] Streaming codon annotations + top-K tracking...")
    codon_annotations_file = output_dir / "codon_annotations_checkpoint.json"
    topk_file = output_dir / "topk_checkpoint.npz"

    if codon_annotations_file.exists() and topk_file.exists():
        print("  Loading codon annotations from checkpoint...")
        with open(codon_annotations_file) as f:
            codon_annotations = json.load(f)
            codon_annotations = {int(k): v for k, v in codon_annotations.items()}
        topk_data = np.load(topk_file)
        top_acts = topk_data["top_acts"]
        top_indices = topk_data["top_indices"]
        print(f"  {len(codon_annotations)} features with codon annotations")
    else:
        codon_annotations, top_acts, top_indices = stream_annotations_and_topk(
            sae,
            inference,
            sequences,
            layer=args.layer,
            context_length=args.context_length,
            batch_size=args.batch_size,
            device=device,
            n_top_examples=max(args.n_examples if hasattr(args, "n_examples") else 5, 5),
        )
        with open(codon_annotations_file, "w") as f:
            json.dump(codon_annotations, f, default=str)
        np.savez_compressed(topk_file, top_acts=top_acts, top_indices=top_indices)
        print(f"  {len(codon_annotations)} features with codon annotations")

    # 3. Auto-interp (optional)
    auto_interp_labels = {}
    auto_interp_ckpt = output_dir / "auto_interp_checkpoint.json"
    if auto_interp_ckpt.exists():
        print("  Loading auto-interp checkpoint...")
        with open(auto_interp_ckpt) as f:
            ckpt_data = json.load(f)
            for k, v in ckpt_data.items():
                k_int = int(k)
                if isinstance(v, dict):
                    auto_interp_labels[k_int] = v
                else:
                    auto_interp_labels[k_int] = {"label": v, "confidence": 0.0}
        print(f"  Loaded {len(auto_interp_labels)} existing interpretations")

    # Load GSEA context if provided
    gsea_context = None
    if args.gsea_report:
        gsea_report_path = Path(args.gsea_report)
        if gsea_report_path.exists():
            print(f"  Loading GSEA report from {gsea_report_path}...")
            with open(gsea_report_path) as f:
                gsea_data = json.load(f)
            gsea_context = {}
            for fl in gsea_data.get("per_feature", []):
                feat_idx = fl["feature_idx"]
                per_db = {}
                for db, entry in fl.get("best_per_database", {}).items():
                    if entry is not None:
                        per_db[db] = entry
                if fl.get("overall_best"):
                    per_db["overall_best"] = fl["overall_best"]
                if per_db:
                    gsea_context[feat_idx] = per_db
            print(f"  GSEA context loaded for {len(gsea_context)} features")
        else:
            print(f"  WARNING: GSEA report not found at {gsea_report_path}")

    if args.auto_interp:
        print("\n[3/3] Auto-interpretation (LLM)...")
        alive_features = [f for f in range(n_features) if f in codon_annotations]
        alive_features_sorted = sorted(
            alive_features,
            key=lambda f: max([abs(v) for _, v in vocab_logits[f].get("top_positive", [])], default=0),
            reverse=True,
        )
        if args.max_auto_interp_features:
            alive_features_sorted = alive_features_sorted[: args.max_auto_interp_features]

        todo_features = [f for f in alive_features_sorted if f not in auto_interp_labels]

        if todo_features:
            print(f"  Running auto-interp on {len(todo_features)} features ({len(auto_interp_labels)} already done)")
            new_labels, new_confidences, new_descriptions = run_auto_interp(
                sae,
                vocab_logits,
                inference,
                sequences,
                records,
                todo_features,
                top_indices,
                layer=args.layer,
                context_length=args.context_length,
                batch_size=args.batch_size,
                device=device,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                num_workers=args.auto_interp_workers,
                gsea_context=gsea_context,
            )
            for f in new_labels:
                auto_interp_labels[f] = {
                    "label": new_labels[f],
                    "confidence": new_confidences[f],
                    "description": new_descriptions.get(f),
                }
            with open(auto_interp_ckpt, "w") as f:
                json.dump(auto_interp_labels, f, indent=2)
        else:
            print(f"  All {len(auto_interp_labels)} features already interpreted")
    else:
        print("\n[3/3] Skipping auto-interp (use --auto-interp to enable)")

    # Build labels
    print("\nBuilding feature labels...")
    labels, details, llm_confidences = build_feature_labels(
        n_features,
        vocab_logits,
        codon_annotations,
        auto_interp_labels,
    )
    n_labeled = sum(1 for v in labels.values() if not v.startswith("Feature "))
    print(f"  {n_labeled}/{n_features} features labeled")

    # Save results
    print("\nSaving results...")

    with open(output_dir / "feature_analysis.json", "w") as f:
        json.dump(details, f, indent=2, default=str)

    with open(output_dir / "feature_labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    with open(output_dir / "llm_confidences.json", "w") as f:
        json.dump(llm_confidences, f, indent=2)

    logits_export = {}
    for feat_id, data in vocab_logits.items():
        logits_export[str(feat_id)] = {
            "top_positive": [[tok, round(val, 3)] for tok, val in data["top_positive"]],
            "top_negative": [[tok, round(val, 3)] for tok, val in data["top_negative"]],
        }
    with open(output_dir / "vocab_logits.json", "w") as f:
        json.dump(logits_export, f)

    # Update dashboard atlas if requested
    if args.dashboard_dir:
        dashboard_dir = Path(args.dashboard_dir)
        atlas_path = dashboard_dir / "features_atlas.parquet"
        if atlas_path.exists():
            import pyarrow as pa
            import pyarrow.parquet as pq

            print(f"\nUpdating {atlas_path} with labels and confidence scores...")
            table = pq.read_table(atlas_path)
            n = table.num_rows
            label_col = [labels.get(i, f"Feature {i}") for i in range(n)]
            confidence_col = [llm_confidences.get(i, 0.0) for i in range(n)]
            table = table.drop("label") if "label" in table.column_names else table
            table = table.drop("llm_confidence") if "llm_confidence" in table.column_names else table
            table = table.append_column("label", pa.array(label_col))
            table = table.append_column("llm_confidence", pa.array(confidence_col, type=pa.float32()))

            # Add codon optimality columns (CAI, tAI, RSCU)
            for metric in ["cai", "tai", "rscu"]:
                col_name = f"codon_{metric}"
                values = [codon_annotations.get(i, {}).get(metric) for i in range(n)]
                if col_name in table.column_names:
                    table = table.drop(col_name)
                table = table.append_column(col_name, pa.array(values, type=pa.float32()))

            pq.write_table(table, atlas_path, compression="snappy")
            print(f"  Updated {n} feature labels, confidence scores, and optimality metrics in atlas")

    # Copy analysis files to dashboard dir
    if args.dashboard_dir:
        import shutil

        dashboard_dir = Path(args.dashboard_dir)
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["vocab_logits.json", "feature_labels.json", "feature_analysis.json"]:
            src = output_dir / fname
            dst = dashboard_dir / fname
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied {fname} to dashboard dir")

    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
