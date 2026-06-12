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

"""Sample examples for feature interpretation."""

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class FeatureExamples:
    """Examples for a single feature."""

    feature_idx: int
    high_examples: list[dict]  # High activation examples
    low_examples: list[dict]  # Low/zero activation examples


class FeatureSampler:
    """Sample high and low activation examples for each feature.

    Args:
        activations: Tensor of shape [n_samples, hidden_dim] with SAE activations
        data: List of raw data items (sequences, text, etc.) corresponding to activations
        format_fn: Function to format a data item for the prompt
                   Signature: format_fn(data_item, activation_value, active_indices) -> str
        n_high: Number of high activation examples to sample
        n_low: Number of low/zero activation examples to sample

    Example:
        ```python
        def format_protein(seq, activation, indices):
            highlighted = list(seq)
            for i in indices:
                highlighted[i] = f"[{highlighted[i]}]"
            return "".join(highlighted)

        sampler = FeatureSampler(
            activations=sae_activations,
            data=sequences,
            format_fn=format_protein,
        )
        examples = sampler.sample_feature(feature_idx=42)
        ```
    """

    def __init__(
        self,
        activations: torch.Tensor,
        data: list[Any],
        format_fn: Callable[[Any, float, list[int]], str],
        n_high: int = 10,
        n_low: int = 5,
    ):
        """Initialize the sampler with activations, data, and formatting function."""
        self.activations = activations
        self.data = data
        self.format_fn = format_fn
        self.n_high = n_high
        self.n_low = n_low

        assert len(data) == activations.shape[0], (
            f"Data length {len(data)} != activations shape {activations.shape[0]}"
        )

    def sample_feature(self, feature_idx: int) -> FeatureExamples:
        """Sample examples for a single feature."""
        feature_acts = self.activations[:, feature_idx]

        # Get indices sorted by activation (descending)
        sorted_indices = torch.argsort(feature_acts, descending=True)

        # High activation examples (top n_high)
        high_examples = []
        for idx in sorted_indices[: self.n_high]:
            idx = idx.item()
            act_value = feature_acts[idx].item()
            if act_value <= 0:
                break  # No more positive activations

            # Find which positions in this sample activated for this feature
            # This requires per-position activations if available
            active_indices = []  # Placeholder - depends on data structure

            high_examples.append(
                {
                    "data_idx": idx,
                    "activation": act_value,
                    "formatted": self.format_fn(self.data[idx], act_value, active_indices),
                }
            )

        # Low/zero activation examples (random from zero activations)
        zero_mask = feature_acts == 0
        zero_indices = torch.where(zero_mask)[0]

        if len(zero_indices) >= self.n_low:
            perm = torch.randperm(len(zero_indices))[: self.n_low]
            low_indices = zero_indices[perm]
        else:
            low_indices = zero_indices

        low_examples = []
        for idx in low_indices:
            idx = idx.item()
            low_examples.append(
                {
                    "data_idx": idx,
                    "activation": 0.0,
                    "formatted": self.format_fn(self.data[idx], 0.0, []),
                }
            )

        return FeatureExamples(
            feature_idx=feature_idx,
            high_examples=high_examples,
            low_examples=low_examples,
        )

    def sample_features(self, feature_indices: list[int]) -> list[FeatureExamples]:
        """Sample examples for multiple features."""
        return [self.sample_feature(idx) for idx in feature_indices]
