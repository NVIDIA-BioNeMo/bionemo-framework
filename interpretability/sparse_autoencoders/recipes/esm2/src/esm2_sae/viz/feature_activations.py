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

"""Feature activation statistics and example collection."""

import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class FeatureStats:
    """Global statistics for a single feature."""

    feature_id: int
    activation_frequency: float  # Fraction of residues where feature fires
    mean_activation: float  # Mean activation when active
    max_activation: float  # Global max activation
    n_proteins_active: int  # Number of proteins where feature appeared


@dataclass
class FeatureExample:
    """A single high-activation example for a feature."""

    feature_id: int
    protein_id: str
    residue_idx: int
    activation_value: float
    sequence_window: str  # e.g., 10 residues centered on position
    window_start: int  # Start index of window in full sequence
    highlight_values: List[float]  # Per-residue activations in window


def compute_feature_activations(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    protein_ids: List[str],
    sequences: List[str],
    masks: Optional[torch.Tensor] = None,
    n_top_examples: int = 20,
    window_size: int = 21,
    activation_threshold: float = 0.0,
    device: str = "cpu",
    batch_size: int = 32,
) -> Tuple[List[FeatureStats], List[FeatureExample]]:
    """Compute feature activation statistics and collect top examples.

    Args:
        sae: Trained SAE model
        embeddings: Shape (n_sequences, seq_len, hidden_dim)
        protein_ids: List of protein identifiers
        sequences: List of amino acid sequences
        masks: Optional validity masks, shape (n_sequences, seq_len)
        n_top_examples: Number of top examples to keep per feature
        window_size: Context window size for examples (odd number)
        activation_threshold: Minimum activation to count as "fired"
        device: Compute device
        batch_size: Batch size for inference

    Returns:
        Tuple of (feature_stats, feature_examples)
    """
    sae = sae.eval().to(device)
    n_seqs, seq_len, _hidden_dim = embeddings.shape
    n_features = sae.hidden_dim

    # Accumulators
    total_activations = np.zeros(n_features)
    total_active_count = np.zeros(n_features)
    max_activations = np.zeros(n_features)
    proteins_with_feature = [set() for _ in range(n_features)]

    # Min-heaps for top examples: (activation_value, counter, example_data)
    # Counter is used as tiebreaker since FeatureExample is not comparable
    example_heaps = [[] for _ in range(n_features)]
    heap_counter = 0  # Unique counter for heap ordering

    total_valid_positions = 0
    half_window = window_size // 2

    with torch.no_grad():
        for seq_idx in tqdm(range(n_seqs), desc="Computing activations"):
            emb = embeddings[seq_idx].to(device)  # (seq_len, hidden_dim)
            acts = sae.encode(emb).cpu().numpy()  # (seq_len, n_features)

            protein_id = protein_ids[seq_idx]
            sequence = sequences[seq_idx]

            # Get valid positions
            if masks is not None:
                valid = masks[seq_idx].bool().cpu().numpy()
            else:
                valid = np.ones(seq_len, dtype=bool)

            valid_positions = np.where(valid)[0]
            total_valid_positions += len(valid_positions)

            for pos in valid_positions:
                for feat_idx in range(n_features):
                    act_val = acts[pos, feat_idx]

                    if act_val > activation_threshold:
                        total_activations[feat_idx] += act_val
                        total_active_count[feat_idx] += 1
                        proteins_with_feature[feat_idx].add(protein_id)

                        if act_val > max_activations[feat_idx]:
                            max_activations[feat_idx] = act_val

                        # Track top examples
                        if len(example_heaps[feat_idx]) < n_top_examples:
                            # Extract window
                            start = max(0, pos - half_window)
                            end = min(len(sequence), pos + half_window + 1)
                            window_seq = sequence[start:end]
                            window_acts = acts[start:end, feat_idx].tolist()

                            example = FeatureExample(
                                feature_id=feat_idx,
                                protein_id=protein_id,
                                residue_idx=pos,
                                activation_value=float(act_val),
                                sequence_window=window_seq,
                                window_start=start,
                                highlight_values=window_acts,
                            )
                            heapq.heappush(example_heaps[feat_idx], (act_val, heap_counter, example))
                            heap_counter += 1
                        elif act_val > example_heaps[feat_idx][0][0]:
                            start = max(0, pos - half_window)
                            end = min(len(sequence), pos + half_window + 1)
                            window_seq = sequence[start:end]
                            window_acts = acts[start:end, feat_idx].tolist()

                            example = FeatureExample(
                                feature_id=feat_idx,
                                protein_id=protein_id,
                                residue_idx=pos,
                                activation_value=float(act_val),
                                sequence_window=window_seq,
                                window_start=start,
                                highlight_values=window_acts,
                            )
                            heapq.heapreplace(example_heaps[feat_idx], (act_val, heap_counter, example))
                            heap_counter += 1

    # Compile stats
    feature_stats = []
    for feat_idx in range(n_features):
        freq = total_active_count[feat_idx] / total_valid_positions if total_valid_positions > 0 else 0
        mean_act = (
            total_activations[feat_idx] / total_active_count[feat_idx] if total_active_count[feat_idx] > 0 else 0
        )

        feature_stats.append(
            FeatureStats(
                feature_id=feat_idx,
                activation_frequency=freq,
                mean_activation=mean_act,
                max_activation=max_activations[feat_idx],
                n_proteins_active=len(proteins_with_feature[feat_idx]),
            )
        )

    # Extract examples from heaps
    feature_examples = []
    for feat_idx in range(n_features):
        for _, _, example in example_heaps[feat_idx]:
            feature_examples.append(example)

    return feature_stats, feature_examples
