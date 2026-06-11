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

# sae/relu_l1.py
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SparseAutoencoder


class ReLUSAE(SparseAutoencoder):
    """ReLU Sparse Autoencoder with L1 penalty.

    From Section 2.2 of OpenAI paper:
        z = ReLU(W_enc(x - b_pre) + b_enc)
        x̂ = W_dec·z + b_pre
        Loss = ||x - x̂||² + λ||z||₁

    Args:
        input_dim: Dimension of input features
        hidden_dim: Number of latent features (dictionary size)
        l1_coeff: L1 sparsity penalty coefficient
        normalize_decoder: Whether to normalize decoder columns to unit norm
        normalize_input: If True, normalize inputs to unit norm after centering
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        l1_coeff: float = 1e-2,
        normalize_decoder: bool = True,
        normalize_input: bool = True,
    ):
        """Initialize the ReLU SAE with encoder, decoder, and bias parameters."""
        super().__init__(input_dim, hidden_dim)
        self.l1_coeff = l1_coeff
        self._normalize_decoder = normalize_decoder
        self.normalize_input = normalize_input

        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize with tied weights (encoder = decoder.T)."""
        # Initialize decoder, normalize columns
        nn.init.xavier_uniform_(self.decoder.weight)
        if self._normalize_decoder:
            self._normalize_decoder_weights()

        # Tied init: encoder = decoder.T
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)

    def _normalize_decoder_weights(self):
        """Normalize decoder columns to unit norm."""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input through ReLU activation to produce sparse codes."""
        x_centered = x - self.pre_bias
        if self.normalize_input:
            x_centered = F.normalize(x_centered, dim=-1)
        pre_acts = self.encoder(x_centered) + self.latent_bias
        return F.relu(pre_acts)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode sparse codes back to input space."""
        return self.decoder(codes) + self.pre_bias

    def loss(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute reconstruction loss with L1 sparsity penalty."""
        recon, codes = self(x)

        recon_loss = F.mse_loss(recon, x)
        l1_loss = codes.abs().sum(dim=-1).mean()  # Sum over features, mean over batch

        total_loss = recon_loss + self.l1_coeff * l1_loss

        # Eval metrics (computed from already-available recon, no extra forward pass)
        with torch.no_grad():
            total_var = torch.var(x, dim=0).sum()
            residual_var = torch.var(recon - x, dim=0).sum()
            var_explained = 1.0 - (residual_var / (total_var + 1e-8))

        return {
            "total": total_loss,
            "fvu": recon_loss,
            "mse": recon_loss.detach(),  # ReLU SAE loss IS raw MSE
            "l1": l1_loss,
            "sparsity": (codes > 0).float().sum(dim=-1).mean(),
            "variance_explained": var_explained,
        }

    def post_step(self) -> None:
        """Normalize decoder weights after each optimizer step."""
        if self._normalize_decoder:
            self._normalize_decoder_weights()

    def init_pre_bias_from_data(
        self,
        data: torch.Tensor,
        max_iter: int = 100,
        eps: float = 1e-6,
    ) -> None:
        """Initialize pre_bias to the geometric median of the data.

        The geometric median minimizes sum of Euclidean distances to all points,
        making it more robust to outliers than the mean. Uses Weiszfeld's algorithm.

        Args:
            data: Sample of training data [n_samples, input_dim]
            max_iter: Maximum iterations for Weiszfeld algorithm
            eps: Convergence threshold
        """
        with torch.no_grad():
            # Work in float32 on CPU for numerical stability
            data = data.float().cpu()

            # Initialize with mean
            median = data.mean(dim=0)

            # Weiszfeld's algorithm for geometric median
            for _ in range(max_iter):
                diffs = data - median.unsqueeze(0)
                distances = diffs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                weights = 1.0 / distances
                new_median = (data * weights).sum(dim=0) / weights.sum()

                if (new_median - median).norm() < eps:
                    break
                median = new_median

            self.pre_bias.data = median.to(self.pre_bias.device)
