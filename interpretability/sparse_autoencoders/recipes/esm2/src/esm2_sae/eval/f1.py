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

"""F1 Score metrics for SAE feature interpretability.

Measures how well SAE features align with known biological concepts
(e.g., Swiss-Prot annotations like domains, binding sites).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm


# Concepts where recall is computed per-amino-acid (not per-domain).
# Matches InterPLM's per_aa_concepts. Our concept names use UniProt short codes
# (e.g. "ACT_SITE:Proton donor", "MOD_RES:Phosphoserine"), so a prefix check works.
AA_LEVEL_PREFIXES = ("ACT_SITE", "MOD_RES", "CARBOHYD", "DISULFID")


@dataclass
class F1Result:
    """Single feature-concept pair result."""

    feature_idx: int
    concept: str
    f1: float
    precision: float
    recall: float
    threshold: float
    f1_domain: float = 0.0  # F1 using domain recall (primary metric)
    recall_domain: float = 0.0  # Domain-level recall

    def __repr__(self) -> str:
        """Return a string representation of the F1 result."""
        return f"F1Result(f{self.feature_idx}->{self.concept[:30]}, F1={self.f1:.3f}, F1d={self.f1_domain:.3f})"


def compute_activation_max(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> np.ndarray:
    """Compute per-feature max activation across a set of embeddings.

    Used for normalization: run SAE on a large sample (e.g. 50k training proteins)
    to get stable max-activation estimates, then use these to normalize activations
    at eval time (matching InterPLM's methodology).

    Args:
        sae: Trained SAE model.
        embeddings: Shape (n_sequences, seq_len, hidden_dim).
        masks: Optional, shape (n_sequences, seq_len). 1=valid, 0=padding.
        device: Device for SAE inference.

    Returns:
        Array of shape (n_features,) with max activation per feature.
    """
    sae = sae.eval().to(device)
    n_seqs = embeddings.shape[0]
    running_max = None

    with torch.no_grad():
        for i in range(n_seqs):
            emb = embeddings[i].to(device)  # (seq_len, hidden_dim)
            acts = sae.encode(emb)  # (seq_len, n_features)
            acts_np = acts.cpu().numpy()

            if masks is not None:
                valid = masks[i].numpy().astype(bool)
                seq_len = min(len(valid), acts_np.shape[0])
                acts_np = acts_np[:seq_len][valid[:seq_len]]

            seq_max = acts_np.max(axis=0)  # (n_features,)
            if running_max is None:
                running_max = seq_max.copy()
            else:
                np.maximum(running_max, seq_max, out=running_max)

    return running_max


def compute_f1_scores(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    concept_labels: List[Dict[str, np.ndarray]],
    masks: Optional[torch.Tensor] = None,
    thresholds: Optional[List[float]] = None,
    min_positives: int = 10,
    device: str = "cuda",
    show_progress: bool = True,
    activation_max: Optional[np.ndarray] = None,
    feature_chunk_size: int = 512,
) -> List[F1Result]:
    """Compute F1 scores between SAE features and biological concepts.

    Encodes each sequence once and streams TP/FP/domain-hit accumulation
    across features, so runtime scales with n_sequences (not n_features).

    Args:
        sae: Trained SAE model
        embeddings: Shape (n_sequences, seq_len, hidden_dim)
        concept_labels: List of dicts, one per sequence.
                       Each dict maps concept_name -> array of shape (seq_len,).
                       Values can be binary (1.0) or domain-instance IDs (incrementing ints).
        masks: Optional, shape (n_sequences, seq_len). 1=valid, 0=padding
        thresholds: Activation thresholds to test. Default [0.0, 0.15, 0.5, 0.6, 0.8]
        min_positives: Minimum positive positions for a concept to be evaluated
        device: Device for SAE inference
        show_progress: Show progress bars
        activation_max: Per-feature max activation from external normalization sample.
            Shape (n_features,). When provided, used for normalization instead of
            eval-set max. Matches InterPLM's methodology.
        feature_chunk_size: Unused (kept for API compatibility). Encoding is now
            done once per sequence regardless of n_features.

    Returns:
        List of F1Result for all feature-concept pairs with f1_domain > 0

    Example:
        >>> results = compute_f1_scores(sae, embeddings, concept_labels)
        >>> top_10 = sorted(results, key=lambda x: x.f1_domain, reverse=True)[:10]
    """
    if thresholds is None:
        thresholds = [0.0, 0.15, 0.5, 0.6, 0.8]

    n_seqs, seq_len_max, _hidden_dim = embeddings.shape
    assert len(concept_labels) == n_seqs

    sae = sae.eval().to(device)
    n_features = sae.hidden_dim
    n_thresholds = len(thresholds)
    thresholds_arr = np.array(thresholds)

    # Compute normalization (streaming — constant memory)
    if activation_max is None:
        print("Computing per-feature activation max for normalization...")
        activation_max = compute_activation_max(sae, embeddings, masks, device)
    act_max_full = np.where(activation_max > 0, activation_max, 1.0).astype(np.float32)

    # ── Pre-pass: collect concept metadata (labels only, no activations) ──
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
            # Truncate to embedding seq dim (shorter when special tokens removed)
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

    # ── Initialize accumulators: (n_features, n_thresholds) per concept ──
    tp = {c: np.zeros((n_features, n_thresholds), dtype=np.float64) for c in valid_concepts}
    fp = {c: np.zeros((n_features, n_thresholds), dtype=np.float64) for c in valid_concepts}
    domains_hit = {
        c: np.zeros((n_features, n_thresholds), dtype=np.int64) for c in valid_concepts if not concept_is_aa[c]
    }

    # ── Stream through sequences: encode each ONCE ──
    seq_iter = range(n_seqs)
    if show_progress:
        seq_iter = tqdm(seq_iter, desc="Computing F1 scores")

    with torch.no_grad():
        for seq_idx in seq_iter:
            seq_concepts = concept_labels[seq_idx]
            relevant = [c for c in seq_concepts if c in valid_concepts]
            if not relevant:
                continue

            # Encode once
            emb = embeddings[seq_idx].to(device)
            acts_full = sae.encode(emb).cpu().numpy()  # (padded_seq_len, n_features)

            for concept in relevant:
                labels = seq_concepts[concept]
                # Truncate to embedding seq dim (shorter when special tokens removed)
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

                acts_valid = acts[valid]  # (n_valid, n_features)
                acts_valid = acts_valid / act_max_full  # normalize (broadcasts)

                is_aa = concept_is_aa[concept]

                # Pre-compute domain masks for this sequence (only for non-AA concepts)
                domain_masks = None
                if not is_aa and concept in domains_hit:
                    unique_ids = np.unique(labels_valid[labels_valid > 0])
                    domain_masks = [(d_id, labels_valid == d_id) for d_id in unique_ids]

                # Accumulate per threshold
                for t_idx, thresh in enumerate(thresholds):
                    preds = acts_valid > thresh  # (n_valid, n_features)
                    tp_mask = preds & labels_bool[:, None]  # (n_valid, n_features)

                    tp[concept][:, t_idx] += tp_mask.sum(axis=0)
                    fp[concept][:, t_idx] += (preds & ~labels_bool[:, None]).sum(axis=0)

                    # Domain recall: count domains hit (each domain_id is globally unique)
                    if domain_masks is not None:
                        for _d_id, is_d in domain_masks:
                            # Slice to only rows in this domain (typically 5-50 residues)
                            tp_d = tp_mask[is_d]  # (n_d, n_features)
                            if tp_d.shape[0] > 0:
                                hit = tp_d.any(axis=0)  # (n_features,)
                                domains_hit[concept][:, t_idx] += hit.astype(np.int64)

            del acts_full

    # ── Compute final F1 from accumulated stats ──
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

        # Best threshold per feature — select by f1_domain (matches InterPLM)
        best_thresh_idx = f1_domain.argmax(axis=1)  # (n_features,)
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


def print_f1_summary(results: List[F1Result], top_k: int = 20, min_f1: float = 0.3):
    """Print summary of F1 results."""
    filtered = [r for r in results if r.f1_domain >= min_f1]
    filtered.sort(key=lambda x: x.f1_domain, reverse=True)

    print(f"\nF1 Summary: {len(filtered)} pairs with F1_domain >= {min_f1}")
    print("-" * 80)

    for r in filtered[:top_k]:
        print(f"  Feature {r.feature_idx:4d} -> {r.concept[:40]:40s} F1d={r.f1_domain:.3f} F1={r.f1:.3f}")

    if len(filtered) > top_k:
        print(f"  ... and {len(filtered) - top_k} more")
