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

"""Evaluate CodoNFM SAE features against SwissProt annotations via F1 scores.

Uses the codonfm_swissprot dataset (produced by download_codonfm_swissprot.py)
which contains both CDS nucleotide sequences and residue-level SwissProt annotations.

Since each codon maps 1:1 to an amino acid position, SwissProt annotations
transfer directly: annotation at amino acid position i → SAE feature at codon position i.

IMPORTANT: Run on a single GPU. Do NOT use torchrun.

    python scripts/eval_swissprot_f1.py \
        --checkpoint ./outputs/1b_layer16/checkpoints/checkpoint_final.pt \
        --model-path checkpoints/NV-CodonFM-Encodon-TE-Cdwt-1B-v1/model.safetensors \
        --layer 16 \
        --swissprot-tsv ./data/codonfm_swissprot/codonfm_swissprot.tsv.gz \
        --output-dir ./outputs/1b_layer16/swissprot_eval
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# Use codonfm_ptl_te recipe (has TransformerEngine support)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
_CODONFM_TE_DIR = _REPO_ROOT / "recipes" / "codonfm_ptl_te"
sys.path.insert(0, str(_CODONFM_TE_DIR))

from sae.architectures import TopKSAE  # noqa: E402
from sae.utils import get_device, set_seed  # noqa: E402
from src.data.preprocess.codon_sequence import process_item  # noqa: E402
from src.inference.encodon import EncodonInference  # noqa: E402


# ── Annotation parsing (adapted from esm2_sae) ─────────────────────────

FEATURE_COLUMNS = {
    "active_site": "ACT_SITE",
    "binding_site": "BINDING",
    "disulfide_bond": "DISULFID",
    "glycosylation": "CARBOHYD",
    "lipidation": "LIPID",
    "modified_residue": "MOD_RES",
    "signal_peptide": "SIGNAL",
    "transit_peptide": "TRANSIT",
    "helix": "HELIX",
    "turn": "TURN",
    "beta_strand": "STRAND",
    "coiled_coil": "COILED",
    "compositional_bias": "COMPBIAS",
    "domain_[ft]": "DOMAIN",
    "motif": "MOTIF",
    "region": "REGION",
    "zinc_finger": "ZN_FING",
}

AA_LEVEL_PREFIXES = ("ACT_SITE", "MOD_RES", "CARBOHYD", "DISULFID")


@dataclass
class AnnotatedCodonProtein:
    """Protein with CDS sequence and parsed annotations (at codon/AA level)."""

    accession: str
    codon_sequence: str  # DNA CDS
    protein_sequence: str  # amino acid sequence
    annotations: Dict[str, np.ndarray]  # concept -> array of shape (n_codons,)


@dataclass
class F1Result:
    """Single feature-concept pair result."""

    feature_idx: int
    concept: str
    f1: float
    precision: float
    recall: float
    threshold: float
    f1_domain: float = 0.0
    recall_domain: float = 0.0


def parse_position(pos_str: str) -> Optional[Tuple[int, int]]:
    """Parse a UniProt position string to (start, end) tuple (0-indexed)."""
    if not pos_str or "?" in pos_str or ":" in pos_str:
        return None
    pos_str = pos_str.strip()
    if ".." in pos_str:
        parts = pos_str.split("..")
        start = parts[0].strip("<").strip()
        end = parts[1].strip(">").strip()
        try:
            return (int(start) - 1, int(end))
        except ValueError:
            return None
    else:
        pos_str = pos_str.strip("<>").strip()
        try:
            pos = int(pos_str)
            return (pos - 1, pos)
        except ValueError:
            return None


def parse_annotation_field(
    field_value: str,
    seq_length: int,
    domain_counter: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Parse a UniProt annotation field into position-level arrays."""
    if pd.isna(field_value) or not field_value.strip():
        return {}

    results = {}
    parts = field_value.split("; ")

    annotations = []
    for part in parts:
        part = part.strip().rstrip(";")
        if not part:
            continue
        if part.startswith("/"):
            if annotations:
                annotations[-1][1].append(part)
        else:
            annotations.append((part, []))

    for type_pos, qualifiers in annotations:
        tokens = type_pos.split(None, 1)
        if len(tokens) < 2:
            continue

        ann_type = tokens[0]
        pos_str = tokens[1]
        pos = parse_position(pos_str)
        if pos is None:
            continue

        start, end = pos
        if start < 0 or end > seq_length:
            continue

        note = None
        for q in qualifiers:
            note_match = re.search(r'/note="([^"]*)"', q)
            if note_match:
                note = note_match.group(1)
                break

        concept = f"{ann_type}:{note}" if note else ann_type

        if concept not in results:
            results[concept] = np.zeros(seq_length, dtype=np.float32)

        if domain_counter is not None:
            domain_counter[concept] = domain_counter.get(concept, 0) + 1
            results[concept][start:end] = domain_counter[concept]
        else:
            results[concept][start:end] = 1.0

    return results


def load_swissprot_codon_dataset(
    tsv_path: str,
    min_positives: int = 10,
    max_proteins: Optional[int] = None,
) -> Tuple[List[AnnotatedCodonProtein], Dict[str, int]]:
    """Load the codonfm_swissprot TSV with codon sequences + annotations.

    Annotations are indexed at the amino acid / codon level (1:1 mapping).
    """
    tsv_path = Path(tsv_path)
    if str(tsv_path).endswith(".gz"):
        df = pd.read_csv(tsv_path, sep="\t", compression="gzip")
    else:
        df = pd.read_csv(tsv_path, sep="\t")

    if max_proteins:
        df = df.head(max_proteins)

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    proteins = []
    concept_counts = {}
    domain_counter = {}

    for _, row in df.iterrows():
        accession = row.get("accession", row.get("entry", ""))
        protein_seq = row.get("sequence", "")
        codon_seq = row.get("codon_sequence", "")

        if not protein_seq or not codon_seq:
            continue

        # Number of codons = number of amino acids
        n_codons = len(protein_seq)
        all_annotations = {}

        for col, ann_type in FEATURE_COLUMNS.items():
            if col not in df.columns:
                continue
            field_value = row.get(col, "")
            parsed = parse_annotation_field(str(field_value), n_codons, domain_counter)
            for concept, arr in parsed.items():
                all_annotations[concept] = arr
                concept_counts[concept] = concept_counts.get(concept, 0) + int((arr > 0).sum())

        if all_annotations:
            proteins.append(
                AnnotatedCodonProtein(
                    accession=accession,
                    codon_sequence=codon_seq,
                    protein_sequence=protein_seq,
                    annotations=all_annotations,
                )
            )

    # Filter concepts by min_positives
    valid_concepts = {c for c, count in concept_counts.items() if count >= min_positives}
    for protein in proteins:
        protein.annotations = {c: arr for c, arr in protein.annotations.items() if c in valid_concepts}

    filtered_counts = {c: count for c, count in concept_counts.items() if c in valid_concepts}
    print(f"Loaded {len(proteins)} proteins with {len(filtered_counts)} concepts (min_positives={min_positives})")
    return proteins, filtered_counts


# ── Activation extraction ────────────────────────────────────────────────


def extract_activations_3d(
    inference: "EncodonInference",
    sequences: List[str],
    layer: int,
    context_length: int = 2048,
    batch_size: int = 1,
    device: str = "cuda",
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 3D activations from CodoNFM for codon sequences.

    Returns:
        (activations, masks) where:
        - activations: (n_sequences, max_codon_len, hidden_dim) float32, padded
        - masks: (n_sequences, max_codon_len) bool, 1=valid codon position
    """
    all_embeddings = []
    all_masks = []

    n_batches = (len(sequences) + batch_size - 1) // batch_size
    iterator = range(0, len(sequences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="Extracting activations")

    with torch.no_grad():
        for i in iterator:
            batch_seqs = sequences[i : i + batch_size]
            items = [process_item(s, context_length=context_length, tokenizer=inference.tokenizer) for s in batch_seqs]

            batch = {
                "input_ids": torch.tensor(np.stack([it["input_ids"] for it in items])).to(device),
                "attention_mask": torch.tensor(np.stack([it["attention_mask"] for it in items])).to(device),
            }

            out = inference.model(batch, return_hidden_states=True)
            layer_acts = out.all_hidden_states[layer]  # [B, L, D]

            for j, it in enumerate(items):
                seq_len = it["attention_mask"].sum()
                # Strip CLS (pos 0) and SEP (last real pos)
                acts = layer_acts[j, 1 : seq_len - 1, :].float().cpu()  # [n_codons, D]
                n_codons = acts.shape[0]
                mask = torch.ones(n_codons, dtype=torch.bool)
                all_embeddings.append(acts)
                all_masks.append(mask)

            del out, layer_acts, batch

    # Pad to same length for 3D stacking
    max_len = max(e.shape[0] for e in all_embeddings)
    hidden_dim = all_embeddings[0].shape[1]

    padded_emb = []
    padded_masks = []
    for emb, msk in zip(all_embeddings, all_masks):
        L = emb.shape[0]
        if L < max_len:
            emb = torch.cat([emb, torch.zeros(max_len - L, hidden_dim)], dim=0)
            msk = torch.cat([msk, torch.zeros(max_len - L, dtype=torch.bool)])
        padded_emb.append(emb.unsqueeze(0))
        padded_masks.append(msk.unsqueeze(0))

    return torch.cat(padded_emb, dim=0), torch.cat(padded_masks, dim=0)


# ── F1 computation (adapted from esm2_sae/eval/f1.py) ───────────────────


def compute_activation_max(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    masks: torch.Tensor,
    device: str = "cuda",
) -> np.ndarray:
    """Compute per-feature max activation for normalization."""
    sae = sae.eval().to(device)
    n_seqs = embeddings.shape[0]
    running_max = None

    with torch.no_grad():
        for i in range(n_seqs):
            emb = embeddings[i].to(device)
            acts = sae.encode(emb)
            acts_np = acts.cpu().numpy()

            valid = masks[i].numpy().astype(bool)
            seq_len = min(len(valid), acts_np.shape[0])
            acts_np = acts_np[:seq_len][valid[:seq_len]]

            seq_max = acts_np.max(axis=0)
            if running_max is None:
                running_max = seq_max.copy()
            else:
                np.maximum(running_max, seq_max, out=running_max)

    return running_max


def compute_f1_scores(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    concept_labels: List[Dict[str, np.ndarray]],
    masks: torch.Tensor,
    thresholds: Optional[List[float]] = None,
    min_positives: int = 10,
    device: str = "cuda",
    show_progress: bool = True,
    activation_max: Optional[np.ndarray] = None,
) -> List[F1Result]:
    """Compute F1 scores between SAE features and biological concepts."""
    if thresholds is None:
        thresholds = [0.0, 0.15, 0.5, 0.6, 0.8]

    n_seqs, seq_len_max, _hidden_dim = embeddings.shape
    assert len(concept_labels) == n_seqs

    sae = sae.eval().to(device)
    n_features = sae.hidden_dim
    n_thresholds = len(thresholds)
    thresholds_arr = np.array(thresholds)

    if activation_max is None:
        print("Computing per-feature activation max for normalization...")
        activation_max = compute_activation_max(sae, embeddings, masks, device)
    act_max_full = np.where(activation_max > 0, activation_max, 1.0).astype(np.float32)

    # Collect concept metadata
    all_concepts: set = set()
    for seq_concepts in concept_labels:
        all_concepts.update(seq_concepts.keys())

    concept_total_pos: Dict[str, int] = {}
    concept_n_domains: Dict[str, int] = {}
    concept_is_aa: Dict[str, bool] = {}

    for concept in all_concepts:
        total_pos = 0
        domain_ids: set = set()
        is_aa = any(concept.startswith(p) for p in AA_LEVEL_PREFIXES)

        for seq_idx in range(n_seqs):
            if concept not in concept_labels[seq_idx]:
                continue
            labels = concept_labels[seq_idx][concept]
            seq_len = min(len(labels), seq_len_max)
            labels = labels[:seq_len]
            if masks is not None:
                valid = masks[seq_idx].numpy()[:seq_len].astype(bool)
                labels_v = labels[valid]
            else:
                labels_v = labels
            total_pos += int((labels_v > 0).sum())
            if not is_aa:
                domain_ids.update(labels_v[labels_v > 0].tolist())

        if total_pos >= min_positives:
            concept_total_pos[concept] = total_pos
            concept_is_aa[concept] = is_aa
            if not is_aa:
                concept_n_domains[concept] = len(domain_ids)

    valid_concepts = set(concept_total_pos.keys())
    if not valid_concepts:
        return []

    # Initialize accumulators
    tp = {c: np.zeros((n_features, n_thresholds), dtype=np.float64) for c in valid_concepts}
    fp = {c: np.zeros((n_features, n_thresholds), dtype=np.float64) for c in valid_concepts}
    domains_hit = {
        c: np.zeros((n_features, n_thresholds), dtype=np.int64) for c in valid_concepts if not concept_is_aa[c]
    }

    # Stream through sequences
    seq_iter = range(n_seqs)
    if show_progress:
        seq_iter = tqdm(seq_iter, desc="Computing F1 scores")

    with torch.no_grad():
        for seq_idx in seq_iter:
            seq_concepts = concept_labels[seq_idx]
            relevant = [c for c in seq_concepts if c in valid_concepts]
            if not relevant:
                continue

            emb = embeddings[seq_idx].to(device)
            acts_full = sae.encode(emb).cpu().numpy()

            for concept in relevant:
                labels = seq_concepts[concept]
                seq_len = min(len(labels), acts_full.shape[0])
                labels = labels[:seq_len]
                acts = acts_full[:seq_len]

                if masks is not None:
                    valid = masks[seq_idx].numpy()[:seq_len].astype(bool)
                else:
                    valid = np.ones(seq_len, dtype=bool)

                labels_valid = labels[valid]
                labels_bool = labels_valid > 0
                if not labels_bool.any():
                    continue

                acts_valid = acts[valid]
                acts_valid = acts_valid / act_max_full

                is_aa = concept_is_aa[concept]
                domain_masks = None
                if not is_aa and concept in domains_hit:
                    unique_ids = np.unique(labels_valid[labels_valid > 0])
                    domain_masks = [(d_id, labels_valid == d_id) for d_id in unique_ids]

                for t_idx, thresh in enumerate(thresholds):
                    preds = acts_valid > thresh
                    tp_mask = preds & labels_bool[:, None]
                    tp[concept][:, t_idx] += tp_mask.sum(axis=0)
                    fp[concept][:, t_idx] += (preds & ~labels_bool[:, None]).sum(axis=0)

                    if domain_masks is not None:
                        for _d_id, is_d in domain_masks:
                            tp_d = tp_mask[is_d]
                            if tp_d.shape[0] > 0:
                                hit = tp_d.any(axis=0)
                                domains_hit[concept][:, t_idx] += hit.astype(np.int64)

            del acts_full

    # Compute final F1
    results = []
    for concept in valid_concepts:
        total_pos = concept_total_pos[concept]
        is_aa = concept_is_aa[concept]
        n_domains = concept_n_domains.get(concept, 0)

        tp_arr = tp[concept]
        fp_arr = fp[concept]

        precision = np.where(tp_arr + fp_arr > 0, tp_arr / (tp_arr + fp_arr), 0.0)
        recall = tp_arr / total_pos
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

        if is_aa:
            recall_domain = recall
            f1_domain = f1
        elif n_domains > 0:
            recall_domain = domains_hit[concept].astype(np.float64) / n_domains
            f1_domain = np.where(
                precision + recall_domain > 0,
                2 * precision * recall_domain / (precision + recall_domain),
                0.0,
            )
        else:
            recall_domain = np.zeros_like(recall)
            f1_domain = np.zeros_like(f1)

        best_thresh_idx = f1_domain.argmax(axis=1)
        feat_indices = np.arange(n_features)
        best_f1 = f1[feat_indices, best_thresh_idx]
        best_precision = precision[feat_indices, best_thresh_idx]
        best_recall = recall[feat_indices, best_thresh_idx]
        best_thresh = thresholds_arr[best_thresh_idx]
        best_f1_domain = f1_domain[feat_indices, best_thresh_idx]
        best_recall_domain = recall_domain[feat_indices, best_thresh_idx]

        for feat_idx in range(n_features):
            if best_f1_domain[feat_idx] > 0:
                results.append(
                    F1Result(
                        feature_idx=feat_idx,
                        concept=concept,
                        f1=float(best_f1[feat_idx]),
                        precision=float(best_precision[feat_idx]),
                        recall=float(best_recall[feat_idx]),
                        threshold=float(best_thresh[feat_idx]),
                        f1_domain=float(best_f1_domain[feat_idx]),
                        recall_domain=float(best_recall_domain[feat_idx]),
                    )
                )

    return results


# ── Label building ───────────────────────────────────────────────────────


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

    n_labeled = sum(1 for label in labels if not label.startswith("Feature "))
    print(f"  {n_labeled}/{n_features} features labeled (F1 >= {f1_threshold})")
    return labels, feature_stats


def build_f1_summary(val_results, test_results, f1_threshold):
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


# ── SAE loading ──────────────────────────────────────────────────────────


def load_sae_from_checkpoint(checkpoint_path: str, top_k_override: Optional[int] = None) -> TopKSAE:
    """Load SAE from a Trainer checkpoint."""
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

    top_k = top_k_override or model_config.get("top_k")
    if top_k is None:
        raise ValueError("top_k not found in checkpoint. Pass --top-k explicitly.")

    sae = TopKSAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        top_k=top_k,
        normalize_input=normalize_input,
    )
    sae.load_state_dict(state_dict)
    print(f"Loaded SAE: {input_dim} -> {hidden_dim:,} latents (top-{top_k})")
    return sae


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Evaluate CodoNFM SAE against SwissProt annotations (F1)")

    # Checkpoint
    p.add_argument("--checkpoint", type=str, required=True, help="Path to SAE checkpoint .pt file")
    p.add_argument("--top-k", type=int, default=None, help="Override top-k (default: read from checkpoint)")

    # Model
    p.add_argument("--model-path", type=str, required=True, help="Path to Encodon checkpoint")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)

    # Data
    p.add_argument(
        "--swissprot-tsv",
        type=str,
        required=True,
        help="Path to codonfm_swissprot.tsv.gz (from download_codonfm_swissprot.py)",
    )

    # F1 eval
    p.add_argument("--f1-max-proteins", type=int, default=8000)
    p.add_argument("--f1-min-positives", type=int, default=10)
    p.add_argument("--f1-threshold", type=float, default=0.3, help="F1 threshold for labeling features")
    p.add_argument("--normalization-n-proteins", type=int, default=2000)

    # Output
    p.add_argument("--output-dir", type=str, default="./outputs/swissprot_eval")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    """Run CodoNFM SAE F1 evaluation against SwissProt annotations."""
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SAE
    sae = load_sae_from_checkpoint(args.checkpoint, top_k_override=args.top_k)
    n_features = sae.hidden_dim

    # 2. Load SwissProt+CDS dataset
    print("\n" + "=" * 60)
    print("LOADING SWISSPROT CODON DATASET")
    print("=" * 60)
    proteins, concept_counts = load_swissprot_codon_dataset(
        args.swissprot_tsv,
        min_positives=args.f1_min_positives,
        max_proteins=args.f1_max_proteins,
    )

    if not proteins:
        print("ERROR: No annotated proteins found. Run download_codonfm_swissprot.py first.")
        return

    # Val/test split
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(proteins))
    mid = len(indices) // 2
    val_proteins = [proteins[i] for i in indices[:mid]]
    test_proteins = [proteins[i] for i in indices[mid:]]

    val_sequences = [p.codon_sequence for p in val_proteins]
    test_sequences = [p.codon_sequence for p in test_proteins]
    val_labels = [p.annotations for p in val_proteins]
    test_labels = [p.annotations for p in test_proteins]

    print(f"F1 eval: {len(val_proteins)} val + {len(test_proteins)} test proteins, {len(concept_counts)} concepts")

    # 3. Load CodoNFM model
    print(f"\nLoading Encodon from {args.model_path}...")
    inference = EncodonInference(
        model_path=args.model_path, task_type="embedding_prediction", use_transformer_engine=True
    )
    inference.configure_model()
    inference.model.to(device).eval()

    num_layers = len(inference.model.model.layers)
    target_layer = args.layer if args.layer >= 0 else num_layers + args.layer
    print(f"  Layers: {num_layers}, Target layer: {target_layer}")

    # 4. Extract activations
    print("\n" + "=" * 60)
    print("EXTRACTING ACTIVATIONS")
    print("=" * 60)

    print("Extracting val embeddings...")
    val_embeddings, val_masks = extract_activations_3d(
        inference,
        val_sequences,
        args.layer,
        context_length=args.context_length,
        batch_size=args.batch_size,
        device=device,
    )

    print("Extracting test embeddings...")
    test_embeddings, test_masks = extract_activations_3d(
        inference,
        test_sequences,
        args.layer,
        context_length=args.context_length,
        batch_size=args.batch_size,
        device=device,
    )

    # Compute activation_max for normalization
    norm_n = min(args.normalization_n_proteins, val_embeddings.shape[0])
    print(f"Computing activation_max from {norm_n} proteins...")
    activation_max = compute_activation_max(
        sae,
        val_embeddings[:norm_n],
        val_masks[:norm_n],
        device=device,
    )
    print(f"  activation_max range: [{activation_max.min():.4f}, {activation_max.max():.4f}]")

    # 5. Compute F1 scores
    print("\n" + "=" * 60)
    print("F1 EVALUATION")
    print("=" * 60)

    print("Computing F1 scores (val)...")
    t0 = time.time()
    val_results = compute_f1_scores(
        sae=sae,
        embeddings=val_embeddings,
        concept_labels=val_labels,
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
        embeddings=test_embeddings,
        concept_labels=test_labels,
        masks=test_masks,
        min_positives=args.f1_min_positives,
        device=device,
        show_progress=True,
        activation_max=activation_max,
    )
    print(f"  Test: {len(test_results)} pairs in {time.time() - t0:.1f}s")

    # Build labels
    print("Building feature labels...")
    f1_labels, feature_stats = build_f1_labels(val_results, n_features, args.f1_threshold)

    # Build and save summary
    f1_summary = build_f1_summary(val_results, test_results, args.f1_threshold)

    print("\nF1 Summary:")
    print(f"  Concepts matched:       {f1_summary['n_concepts_matched']}")
    print(f"  Mean F1 (domain, test): {f1_summary['mean_f1_domain_test']:.4f}")
    print(f"  Max F1 (domain, test):  {f1_summary['max_f1_domain_test']:.4f}")
    print(f"  Above {f1_summary['f1_threshold']} (val):    {f1_summary['n_above_threshold_val']}")
    print(f"  Above {f1_summary['f1_threshold']} (both):   {f1_summary['n_pairs_above_threshold_both']}")
    if f1_summary["top_pairs"]:
        print("  Top pairs (test):")
        for p in f1_summary["top_pairs"][:10]:
            print(f"    Feature {p['feature']:>5d}  F1={p['f1_domain']:.3f}  {p['concept']}")

    f1_path = output_dir / "f1_results.json"
    with open(f1_path, "w") as f:
        json.dump(f1_summary, f, indent=2)
    print(f"\nSaved F1 results to {f1_path}")

    # Save labels
    labels_path = output_dir / "feature_labels.json"
    with open(labels_path, "w") as f:
        json.dump({"labels": f1_labels, "feature_stats": feature_stats}, f, indent=2)
    print(f"Saved feature labels to {labels_path}")

    del inference, val_embeddings, test_embeddings
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"All results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
