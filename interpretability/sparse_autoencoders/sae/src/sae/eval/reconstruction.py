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

"""Reconstruction quality metrics for SAE evaluation.

Measures how well the SAE reconstructs the original embeddings:
- MSE: Mean squared error between original and reconstructed
- Variance Explained (R²): Proportion of variance captured by reconstruction
"""

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ReconstructionMetrics:
    """Container for reconstruction quality metrics."""

    mse: float
    variance_explained: float
    normalized_mse: float  # MSE / var(original)

    def __repr__(self) -> str:
        """Return string representation of reconstruction metrics."""
        return (
            f"ReconstructionMetrics(mse={self.mse:.6f}, "
            f"var_explained={self.variance_explained:.4f}, "
            f"norm_mse={self.normalized_mse:.6f})"
        )


def compute_mse(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """Compute mean squared error between original and reconstructed.

    Args:
        original: Original embeddings, shape (..., hidden_dim)
        reconstructed: Reconstructed embeddings, same shape

    Returns:
        Mean squared error (scalar)
    """
    return torch.nn.functional.mse_loss(reconstructed, original).item()


def compute_variance_explained(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute variance explained (R²) by the reconstruction.

    R² = 1 - Var(residual) / Var(original)

    A value of 1.0 means perfect reconstruction, 0.0 means the reconstruction
    explains none of the variance (as good as predicting the mean).

    Args:
        original: Original embeddings, shape (n_samples, hidden_dim)
        reconstructed: Reconstructed embeddings, same shape
        eps: Small constant for numerical stability

    Returns:
        Variance explained (R²), typically in [0, 1]
    """
    # Compute variance per dimension, then sum
    total_variance = torch.var(original, dim=0).sum()
    residual = original - reconstructed
    residual_variance = torch.var(residual, dim=0).sum()

    # R² = 1 - residual_var / total_var
    r2 = 1.0 - (residual_variance / (total_variance + eps))
    return r2.item()


def compute_normalized_mse(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute MSE normalized by the variance of the original.

    Normalized MSE = MSE / Var(original)

    This gives a scale-invariant measure of reconstruction quality.
    A value of 0 means perfect reconstruction, 1 means MSE equals variance.

    Args:
        original: Original embeddings
        reconstructed: Reconstructed embeddings
        eps: Small constant for numerical stability

    Returns:
        Normalized MSE
    """
    mse = torch.nn.functional.mse_loss(reconstructed, original)
    variance = torch.var(original)
    return (mse / (variance + eps)).item()


def compute_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> ReconstructionMetrics:
    """Compute all reconstruction quality metrics.

    Args:
        original: Original embeddings, shape (n_samples, hidden_dim)
        reconstructed: Reconstructed embeddings, same shape

    Returns:
        ReconstructionMetrics with mse, variance_explained, normalized_mse
    """
    return ReconstructionMetrics(
        mse=compute_mse(original, reconstructed),
        variance_explained=compute_variance_explained(original, reconstructed),
        normalized_mse=compute_normalized_mse(original, reconstructed),
    )


def evaluate_reconstruction(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    batch_size: int = 1024,
    device: str = "cpu",
) -> ReconstructionMetrics:
    """Evaluate SAE reconstruction quality on a dataset.

    For SAEs with normalize_input=True, metrics are computed in the normalized
    space (what the SAE is actually learning to reconstruct).

    Args:
        sae: Trained SAE model
        embeddings: Embeddings to evaluate, shape (n_samples, hidden_dim)
        batch_size: Batch size for evaluation
        device: Device for computation

    Returns:
        ReconstructionMetrics aggregated over the dataset
    """
    sae = sae.eval().to(device)
    n_samples = embeddings.shape[0]

    all_mse = []
    all_var_total = []
    all_var_residual = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = embeddings[i : i + batch_size].to(device)
            reconstructed, _ = sae(batch)

            # Always compute on original data - standard practice
            target = batch
            recon_centered = reconstructed

            # Accumulate for aggregation
            mse = torch.nn.functional.mse_loss(recon_centered, target, reduction="sum")
            var_total = torch.var(target, dim=0).sum() * batch.shape[0]
            residual = target - recon_centered
            var_residual = torch.var(residual, dim=0).sum() * batch.shape[0]

            all_mse.append(mse.item())
            all_var_total.append(var_total.item())
            all_var_residual.append(var_residual.item())

    # Aggregate
    total_mse = sum(all_mse)
    total_var = sum(all_var_total)
    total_var_residual = sum(all_var_residual)

    n_elements = n_samples * embeddings.shape[1]
    mse = total_mse / n_elements
    variance_explained = 1.0 - (total_var_residual / (total_var + 1e-8))
    normalized_mse = mse / (total_var / n_elements + 1e-8)

    return ReconstructionMetrics(
        mse=mse,
        variance_explained=variance_explained,
        normalized_mse=normalized_mse,
    )


def compute_batch_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Dict[str, float]:
    """Compute reconstruction metrics for a single batch (for use in training loop).

    Args:
        original: Original batch
        reconstructed: Reconstructed batch

    Returns:
        Dict with 'mse', 'variance_explained', 'normalized_mse' keys
    """
    metrics = compute_reconstruction_metrics(original, reconstructed)
    return {
        "mse": metrics.mse,
        "variance_explained": metrics.variance_explained,
        "normalized_mse": metrics.normalized_mse,
    }
