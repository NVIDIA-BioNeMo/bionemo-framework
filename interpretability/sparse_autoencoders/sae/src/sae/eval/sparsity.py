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

"""Sparsity statistics for SAE evaluation."""

from dataclasses import dataclass

import torch


@dataclass
class SparsityMetrics:
    """Container for sparsity statistics."""

    mean_l0: float
    n_features: int
    # Feature utilization: fraction that fired at least once on the eval set.
    # Sample-size dependent — NOT the same as "dead latents" in the literature.
    features_used: int
    features_unused: int
    feature_utilization_pct: float
    n_eval_tokens: int
    # Dead latents (literature definition): inactive for > threshold tokens
    # during training.  Only available for models that track stats_last_nonzero.
    dead_pct: float = -1.0
    n_dead: int = -1
    dead_tokens_threshold: int = -1

    def __repr__(self) -> str:
        """Return string representation of sparsity metrics."""
        return (
            f"SparsityMetrics(L0={self.mean_l0:.2f}, "
            f"utilization={self.feature_utilization_pct:.1f}%, "
            f"dead={self.n_dead}/{self.n_features} ({self.dead_pct:.1f}%))"
        )


def evaluate_sparsity(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    batch_size: int = 1024,
    device: str = "cpu",
) -> SparsityMetrics:
    """Compute sparsity statistics for a trained SAE.

    Args:
        sae: Trained SAE model.
        embeddings: Evaluation embeddings, shape (n_samples, hidden_dim).
        batch_size: Batch size for evaluation.
        device: Device for computation.

    Returns:
        SparsityMetrics with mean L0, feature utilization, and dead latent stats.
    """
    sae = sae.eval().to(device)
    n_samples = embeddings.shape[0]
    hidden_dim = sae.hidden_dim

    all_l0 = []
    feature_fired = torch.zeros(hidden_dim, dtype=torch.bool, device=device)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = embeddings[i : i + batch_size].to(device)
            codes = sae.encode(batch)

            l0 = (codes != 0).float().sum(dim=-1)
            all_l0.append(l0.cpu())

            feature_fired = feature_fired | (codes != 0).any(dim=0)

    mean_l0 = torch.cat(all_l0).mean().item()
    features_used = int(feature_fired.sum().item())
    features_unused = hidden_dim - features_used
    feature_utilization_pct = features_used / hidden_dim * 100

    # Dead latents (literature definition): training-time rolling counter
    dead_pct = -1.0
    n_dead = -1
    dead_tokens_threshold = -1
    if hasattr(sae, "stats_last_nonzero") and hasattr(sae, "dead_tokens_threshold"):
        dead_tokens_threshold = sae.dead_tokens_threshold
        dead_mask = sae.stats_last_nonzero > dead_tokens_threshold
        n_dead = int(dead_mask.sum().item())
        dead_pct = n_dead / hidden_dim * 100

    return SparsityMetrics(
        mean_l0=mean_l0,
        n_features=hidden_dim,
        features_used=features_used,
        features_unused=features_unused,
        feature_utilization_pct=feature_utilization_pct,
        n_eval_tokens=n_samples,
        dead_pct=dead_pct,
        n_dead=n_dead,
        dead_tokens_threshold=dead_tokens_threshold,
    )
