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

"""Post-training SAE evaluation suite.

Runs model-agnostic metrics (reconstruction, sparsity) always.
Optionally runs loss recovered and custom recipe-specific metrics
via callables passed by the recipe.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from .loss_recovered import LossRecoveredResult
from .reconstruction import ReconstructionMetrics, evaluate_reconstruction
from .sparsity import SparsityMetrics, evaluate_sparsity


@dataclass
class EvalResults:
    """Container for post-training evaluation results."""

    reconstruction: ReconstructionMetrics
    sparsity: SparsityMetrics
    loss_recovered: Optional[LossRecoveredResult] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert evaluation results to a serializable dictionary."""
        d = {
            "reconstruction": {
                "mse": self.reconstruction.mse,
                "variance_explained": self.reconstruction.variance_explained,
                "normalized_mse": self.reconstruction.normalized_mse,
            },
            "sparsity": {
                "mean_l0": self.sparsity.mean_l0,
                "n_features": self.sparsity.n_features,
                "feature_utilization_pct": self.sparsity.feature_utilization_pct,
                "features_used": self.sparsity.features_used,
                "features_unused": self.sparsity.features_unused,
                "n_eval_tokens": self.sparsity.n_eval_tokens,
                "dead_pct": self.sparsity.dead_pct,
                "n_dead": self.sparsity.n_dead,
                "dead_tokens_threshold": self.sparsity.dead_tokens_threshold,
            },
        }
        if self.loss_recovered is not None:
            d["loss_recovered"] = {
                "loss_recovered": self.loss_recovered.loss_recovered,
                "ce_original": self.loss_recovered.ce_original,
                "ce_sae": self.loss_recovered.ce_sae,
                "ce_zero": self.loss_recovered.ce_zero,
                "n_tokens": self.loss_recovered.n_tokens,
            }
        for name, value in self.custom.items():
            d[name] = value
        return d

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print("\nReconstruction:")
        print(f"  MSE:                {self.reconstruction.mse:.6f}")
        print(f"  Variance Explained: {self.reconstruction.variance_explained:.4f}")
        print(f"  FVU:                {self.reconstruction.normalized_mse:.6f}")

        print("\nSparsity:")
        print(f"  Mean L0:            {self.sparsity.mean_l0:.2f}")
        print(
            f"  Feature utilization: {self.sparsity.features_used}/{self.sparsity.n_features} ({self.sparsity.feature_utilization_pct:.1f}%) fired on {self.sparsity.n_eval_tokens:,} eval tokens"
        )
        if self.sparsity.dead_pct >= 0:
            print(
                f"  Dead latents:       {self.sparsity.n_dead}/{self.sparsity.n_features} ({self.sparsity.dead_pct:.1f}%) inactive > {self.sparsity.dead_tokens_threshold:,} tokens"
            )

        if self.loss_recovered is not None:
            print("\nLoss Recovered:")
            print(f"  Score:       {self.loss_recovered.loss_recovered:.1%}")
            print(f"  CE Original: {self.loss_recovered.ce_original:.4f}")
            print(f"  CE SAE:      {self.loss_recovered.ce_sae:.4f}")
            print(f"  CE Zero:     {self.loss_recovered.ce_zero:.4f}")
            print(f"  Tokens:      {self.loss_recovered.n_tokens}")

        for name, value in self.custom.items():
            print(f"\n{name}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    elif isinstance(v, (list, dict)) and len(str(v)) > 80:
                        print(f"  {k}: ({len(v)} items)")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"  {value}")

        print("=" * 60)

    def __repr__(self) -> str:
        """Return string representation of evaluation results."""
        parts = [
            f"  reconstruction: {self.reconstruction}",
            f"  sparsity: {self.sparsity}",
        ]
        if self.loss_recovered is not None:
            parts.append(f"  loss_recovered: {self.loss_recovered}")
        if self.custom:
            parts.append(f"  custom: {list(self.custom.keys())}")
        return "EvalResults(\n" + "\n".join(parts) + "\n)"


def evaluate_sae(
    sae: torch.nn.Module,
    embeddings: torch.Tensor,
    batch_size: int = 1024,
    device: str = "cpu",
    loss_recovered_fn: Optional[Callable[[], LossRecoveredResult]] = None,
    custom_metrics: Optional[Dict[str, Callable[[], Any]]] = None,
    reconstruction: Optional[ReconstructionMetrics] = None,
) -> EvalResults:
    """Post-training evaluation suite for a trained SAE.

    Computes sparsity statistics on the provided embeddings.
    Reconstruction metrics can be pre-computed (e.g. from training) or
    will be computed from the embeddings if not provided.
    Optionally computes loss recovered and custom metrics via callables.

    Args:
        sae: Trained SAE model.
        embeddings: Evaluation embeddings, shape (n_samples, hidden_dim).
        batch_size: Batch size for sparsity and (if needed) reconstruction eval.
        device: Device for computation.
        loss_recovered_fn: Callable returning LossRecoveredResult.
            Recipe provides this with model-specific logic captured in closure.
        custom_metrics: Dict of {name: callable} for recipe-specific metrics.
            Each callable should return a dict suitable for JSON serialization.
        reconstruction: Pre-computed reconstruction metrics (e.g. from training).
            If None, reconstruction is computed from embeddings.

    Returns:
        EvalResults with reconstruction, sparsity, and any additional metrics.
    """
    if reconstruction is None:
        print("Computing reconstruction metrics...")
        reconstruction = evaluate_reconstruction(sae, embeddings, batch_size=batch_size, device=device)

    print("Computing sparsity statistics...")
    sparsity = evaluate_sparsity(sae, embeddings, batch_size=batch_size, device=device)

    loss_recovered = None
    if loss_recovered_fn is not None:
        print("Computing loss recovered...")
        try:
            loss_recovered = loss_recovered_fn()
        except Exception as e:
            print(f"Warning: loss recovered failed: {e}")

    custom = {}
    if custom_metrics is not None:
        for name, fn in custom_metrics.items():
            print(f"Computing {name}...")
            try:
                custom[name] = fn()
            except Exception as e:
                print(f"Warning: {name} failed: {e}")

    return EvalResults(
        reconstruction=reconstruction,
        sparsity=sparsity,
        loss_recovered=loss_recovered,
        custom=custom,
    )
