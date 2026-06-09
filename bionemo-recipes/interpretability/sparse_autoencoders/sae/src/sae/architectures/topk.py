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

"""Top-K Sparse Autoencoder implementation with optional AuxK dead latent loss.

Based on: https://cdn.openai.com/papers/sparse-autoencoders.pdf
Reference: https://github.com/openai/sparse_autoencoder
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SparseAutoencoder


class TopKSAE(SparseAutoencoder):
    """Top-K Sparse Autoencoder with pre_bias and optional auxiliary loss for dead latents.

    Architecture (matches OpenAI):
        # If normalize_input=True:
        mu, sigma = mean(x), std(x)       # per-sample statistics
        x_norm = (x - mu) / sigma         # standardize
        x_centered = x_norm - pre_bias
        latents = ReLU(TopK(encoder(x_centered) + latent_bias))
        recon_norm = decoder(latents) + pre_bias
        recon = recon_norm * sigma + mu   # restore original scale

        # If normalize_input=False:
        x_centered = x - pre_bias
        latents = ReLU(TopK(encoder(x_centered) + latent_bias))
        recon = decoder(latents) + pre_bias

    The pre_bias centers the input data, which can help optimization.
    The normalize_input option standardizes inputs (zero mean, unit variance per sample),
    improving training stability while preserving magnitude through denormalization.
    The auxiliary loss (AuxK) helps revive dead latents by having them reconstruct
    what the primary top-k latents missed (the residual error).

    Args:
        input_dim: Dimension of input features
        hidden_dim: Number of latent features (dictionary size)
        top_k: Number of top activations to keep per sample
        normalize_input: If True, standardize inputs and denormalize outputs (default: True)
        auxk: Number of auxiliary latents for dead latent loss (None = disabled)
        auxk_coef: Coefficient for auxiliary loss (default: 1/32)
        dead_tokens_threshold: Tokens of inactivity before latent is considered dead (default 10M per Gao et al.)
        init_encoder_from_decoder: If True, initialize encoder weights as transpose
            of decoder weights. From OpenAI paper: this + AuxK → nearly 0% dead latents.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        top_k: int,
        normalize_input: bool = True,
        auxk: Optional[int] = None,
        auxk_coef: float = 1 / 32,
        dead_tokens_threshold: int = 10_000_000,
        init_encoder_from_decoder: bool = True,
        init_pre_bias: bool = True,
        decoder_impl: str = "dense",
    ):
        """Initialize the Top-K SAE with encoder, decoder, and optional auxiliary loss.

        ``decoder_impl`` selects the decode path: "dense" (default) builds the dense
        [batch, hidden_dim] code tensor and runs a full decoder matmul; "triton"
        decodes directly from the top-k (indices, values) via a sparse kernel
        (O(batch*k*d), no dense code tensor), enabling much larger hidden_dim. Weights
        are identical, so checkpoints are interchangeable between the two.
        """
        super().__init__(input_dim, hidden_dim)
        self.top_k = top_k
        self.init_pre_bias = init_pre_bias
        self.normalize_input = normalize_input
        self.auxk = auxk
        self.auxk_coef = auxk_coef
        self.dead_tokens_threshold = dead_tokens_threshold
        if decoder_impl not in ("dense", "triton"):
            raise ValueError(f"decoder_impl must be 'dense' or 'triton', got {decoder_impl!r}")
        self.decoder_impl = decoder_impl

        # Pre-bias (subtracted from normalized input, added to output before denorm)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        # Encoder (no bias - we use separate latent_bias)
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)

        # Latent bias (added after encoder)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Decoder (no bias - we use pre_bias)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        # Initialize encoder as transpose of decoder (OpenAI paper: reduces dead latents)
        # Must be done after both encoder and decoder are created
        if init_encoder_from_decoder:
            self._init_encoder_from_decoder()

        # Track steps since each latent was last active (for auxk loss)
        self.register_buffer("stats_last_nonzero", torch.zeros(hidden_dim, dtype=torch.long))

    def _get_config(self) -> Dict[str, Any]:
        """Return constructor args for checkpoint serialization."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "top_k": self.top_k,
            "normalize_input": self.normalize_input,
            "auxk": self.auxk,
            "auxk_coef": self.auxk_coef,
            "dead_tokens_threshold": self.dead_tokens_threshold,
        }

    def _init_encoder_from_decoder(self) -> None:
        """Initialize encoder weights as transpose of decoder weights.

        From OpenAI paper: "We identify two important ingredients for preventing
        dead latents: initializing the encoder to the transpose of the decoder,
        and using an auxiliary loss."

        This initialization ensures encoder and decoder start aligned, which
        helps prevent latents from dying early in training.
        """
        with torch.no_grad():
            # decoder.weight is [input_dim, hidden_dim]
            # encoder.weight is [hidden_dim, input_dim]
            # So encoder.weight = decoder.weight.T
            self.encoder.weight.data = self.decoder.weight.data.T.clone()

    def _normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Standardize input to zero mean, unit variance per sample.

        Returns:
            x_norm: Normalized tensor
            info: Dict with 'mu' and 'std' for denormalization
        """
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8  # avoid division by zero
        x_norm = (x - mu) / std
        return x_norm, {"mu": mu, "std": std}

    def _denormalize(self, x: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Restore original scale using stored statistics."""
        return x * info["std"] + info["mu"]

    def encode_pre_act(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute pre-activation latent values (before ReLU and top-k).

        Returns:
            pre_act: Pre-activation values
            info: Normalization info (empty dict if normalize_input=False)
        """
        info = {}
        if self.normalize_input:
            x, info = self._normalize(x)
        x_centered = x - self.pre_bias
        pre_act = F.linear(x_centered, self.encoder.weight, self.latent_bias)
        return pre_act, info

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and apply top-k sparsity.

        Note: This returns only codes (no normalization info).
        Use forward() if you need the full reconstruction.
        """
        pre_act, _ = self.encode_pre_act(x)
        codes = torch.relu(pre_act)

        # Apply top-k
        top_k_vals, top_k_indices = torch.topk(codes, self.top_k, dim=-1)
        codes_sparse = torch.zeros_like(codes).scatter(-1, top_k_indices, top_k_vals)

        return codes_sparse

    def decode(self, codes: torch.Tensor, info: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Decode sparse codes.

        Args:
            codes: Sparse latent codes
            info: Normalization info from encoding (required if normalize_input=True)

        Returns:
            Reconstruction on original scale
        """
        recon = self.decoder(codes) + self.pre_bias
        if self.normalize_input and info is not None:
            recon = self._denormalize(recon, info)
        return recon

    def decode_without_bias(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode sparse codes without adding pre_bias (for aux loss)."""
        return self.decoder(codes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, codes).

        Reconstruction is on the same scale as input x.
        """
        pre_act, info = self.encode_pre_act(x)
        codes_relu = torch.relu(pre_act)

        # Apply top-k
        top_k_vals, top_k_indices = torch.topk(codes_relu, self.top_k, dim=-1)
        codes = torch.zeros_like(codes_relu).scatter(-1, top_k_indices, top_k_vals)

        if self.decoder_impl == "triton":
            recon = self._decode_topk_triton(top_k_vals, top_k_indices, info)
        else:
            recon = self.decode(codes, info)
        return recon, codes

    def _decode_topk_triton(
        self,
        top_k_vals: torch.Tensor,
        top_k_indices: torch.Tensor,
        info: Optional[Dict[str, torch.Tensor]] = None,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """Decode from top-k (values, indices) via the sparse Triton kernel.

        Returns reconstruction with pre_bias added; denormalized to input scale when
        ``denormalize`` (set False to get the normalized-space recon for aux loss).
        """
        from ..kernels import TritonDecoderAutograd

        recon = TritonDecoderAutograd.apply(top_k_indices.contiguous(), top_k_vals.contiguous(), self.decoder.weight)
        recon = recon + self.pre_bias
        if denormalize and self.normalize_input and info is not None:
            recon = self._denormalize(recon, info)
        return recon

    def _update_dead_latent_stats_from_indices(self, top_k_indices: torch.Tensor, n_tokens: int) -> None:
        """Update stats_last_nonzero from top-k indices (no dense [batch, hidden] tensor)."""
        active_mask = torch.zeros_like(self.stats_last_nonzero, dtype=torch.bool)
        active_mask[top_k_indices.reshape(-1)] = True
        self.stats_last_nonzero = torch.where(
            active_mask, torch.zeros_like(self.stats_last_nonzero), self.stats_last_nonzero + n_tokens
        )

    def forward_with_aux(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with auxiliary info for auxk loss computation.

        Returns dict with:
            - recon: reconstruction (on original scale)
            - codes: sparse codes (after top-k)
            - pre_act: pre-activation values (before ReLU/top-k)
            - top_k_indices: indices of selected top-k latents
            - norm_info: normalization statistics (if normalize_input=True)
        """
        pre_act, info = self.encode_pre_act(x)
        codes_relu = torch.relu(pre_act)

        # Apply top-k
        top_k_vals, top_k_indices = torch.topk(codes_relu, self.top_k, dim=-1)
        codes = torch.zeros_like(codes_relu).scatter(-1, top_k_indices, top_k_vals)

        recon = self.decode(codes, info)

        return {
            "recon": recon,
            "codes": codes,
            "pre_act": pre_act,
            "top_k_indices": top_k_indices,
            "norm_info": info,
        }

    def _update_dead_latent_stats(self, codes: torch.Tensor) -> None:
        """Update the stats_last_nonzero counter based on which latents fired.

        For each latent: if it fired in this batch, reset counter to 0, else increment by batch token count.
        """
        # Check which latents were active (any sample in batch had activation > threshold)
        active_mask = (codes.abs() > 1e-3).any(dim=0)  # [hidden_dim]

        # Reset counter for active latents, increment by token count for inactive
        n_tokens = codes.shape[0]
        self.stats_last_nonzero = torch.where(
            active_mask, torch.zeros_like(self.stats_last_nonzero), self.stats_last_nonzero + n_tokens
        )

    def _compute_auxk_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        pre_act: torch.Tensor,
        codes: Optional[torch.Tensor],
        norm_info: Optional[Dict[str, torch.Tensor]] = None,
        recon_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute auxiliary loss for dead latents.

        Dead latents try to reconstruct what the primary latents missed.
        Matches OpenAI implementation.
        """
        # Identify dead latents
        dead_mask = self.stats_last_nonzero > self.dead_tokens_threshold  # [hidden_dim]

        # If no dead latents, return zero loss
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Slice to only dead columns to avoid full-width (batch, hidden_dim) tensors
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        pre_act_dead = pre_act[:, dead_indices]  # [batch, n_dead]

        # Apply ReLU and top-auxk selection among dead latents
        codes_dead = torch.relu(pre_act_dead)

        # Select top auxk (or all dead if fewer than auxk)
        k_aux = min(self.auxk, n_dead)
        auxk_vals, auxk_indices = torch.topk(codes_dead, k_aux, dim=-1)
        codes_aux = torch.zeros_like(codes_dead).scatter(-1, auxk_indices, auxk_vals)

        # Decode auxiliary latents using only dead decoder columns (avoids full-width matmul)
        recon_aux = F.linear(codes_aux, self.decoder.weight[:, dead_indices], self.decoder.bias)

        # Target is the residual (what primary reconstruction missed)
        # Work in normalized space for the aux loss
        if self.normalize_input and norm_info is not None:
            # Normalize x to match the space where encoding happened
            x_norm = (x - norm_info["mu"]) / norm_info["std"]
            # Reuse codes from forward pass instead of re-encoding (or a precomputed
            # normalized recon, e.g. from the sparse/triton decode path).
            if recon_norm is None:
                recon_norm = self.decoder(codes) + self.pre_bias
            residual = x_norm - recon_norm.detach()
        else:
            residual = x - recon.detach() + self.pre_bias.detach()

        # Normalized MSE: MSE / variance of target
        mse = (recon_aux - residual).pow(2).mean(dim=-1)  # [batch]
        target_var = residual.pow(2).mean(dim=-1)  # [batch]

        # Avoid division by zero, use nan_to_num like OpenAI
        normalized_mse = (mse / (target_var + 1e-8)).mean()

        return normalized_mse

    def normalize_decoder(self):
        """Normalize decoder weights to unit norm."""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=1)

    def post_step(self):
        """Normalize decoder after each step."""
        self.normalize_decoder()

    def init_pre_bias_from_data(
        self,
        data: torch.Tensor,
        max_iter: int = 100,
        eps: float = 1e-6,
    ) -> None:
        """Initialize pre_bias to the geometric median of the data.

        The geometric median minimizes sum of Euclidean distances to all points,
        making it more robust to outliers than the mean. Uses Weiszfeld's algorithm.

        Note: When normalize_input=True, this operates on standardized data.

        Args:
            data: Sample of training data [n_samples, input_dim]
            max_iter: Maximum iterations for Weiszfeld algorithm
            eps: Convergence threshold
        """
        with torch.no_grad():
            # Work in float32 on CPU for numerical stability (like OpenAI)
            data = data.float().cpu()

            # If normalizing, compute geometric median on normalized data
            if self.normalize_input:
                mu = data.mean(dim=-1, keepdim=True)
                std = data.std(dim=-1, keepdim=True) + 1e-8
                data = (data - mu) / std

            # Initialize with mean
            median = data.mean(dim=0)

            # Weiszfeld's algorithm for geometric median
            for _ in range(max_iter):
                # Compute distances from current estimate to all points
                diffs = data - median.unsqueeze(0)  # [n_samples, input_dim]
                distances = diffs.norm(dim=1, keepdim=True).clamp(min=1e-8)  # [n_samples, 1]

                # Weighted average (weights = 1/distance)
                weights = 1.0 / distances  # [n_samples, 1]
                new_median = (data * weights).sum(dim=0) / weights.sum()

                # Check convergence
                if (new_median - median).norm() < eps:
                    break
                median = new_median

            self.pre_bias.data = median.to(self.pre_bias.device)

    def loss(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute loss with optional auxiliary loss for dead latents.

        Returns dict with:
            - total: total loss (recon + auxk_coef * aux)
            - reconstruction: MSE reconstruction loss
            - sparsity: L0 sparsity metric
            - aux (if auxk enabled): auxiliary loss value
            - dead_pct (if auxk enabled): percentage of dead latents
        """
        if self.decoder_impl == "triton":
            return self._loss_triton(x)

        # Forward pass with auxiliary info
        info = self.forward_with_aux(x)
        recon = info["recon"]
        codes = info["codes"]
        pre_act = info["pre_act"]
        norm_info = info["norm_info"]

        # Update dead latent stats
        self._update_dead_latent_stats(codes)

        # Primary reconstruction loss (FVU: fraction of variance unexplained)
        # Center by pre_bias (learned per-dim mean) so denominator reflects
        # actual signal variance, consistent with var_exp metric
        mse = (recon - x).pow(2).mean(dim=-1)  # [batch]
        x_centered = x - self.pre_bias
        x_var = x_centered.pow(2).mean(dim=-1)  # [batch]
        recon_loss = (mse / (x_var + 1e-8)).mean()

        # Sparsity metric (for logging)
        l0 = (codes != 0).float().sum(dim=-1).mean()

        # Eval metrics (computed from already-available recon, no extra forward pass)
        with torch.no_grad():
            raw_mse = (recon - x).pow(2).mean()
            total_var = torch.var(x, dim=0).sum()
            residual_var = torch.var(recon - x, dim=0).sum()
            var_explained = 1.0 - (residual_var / (total_var + 1e-8))

        result = {
            "total": recon_loss,
            "fvu": 1.0 - var_explained,
            "sparsity": l0,
            "mse": raw_mse,
            "variance_explained": var_explained,
        }

        # Log dead latent percentage (always, for comparison across runs)
        dead_pct = (self.stats_last_nonzero > self.dead_tokens_threshold).float().mean() * 100
        result["dead_pct"] = dead_pct

        # Auxiliary loss for dead latents
        if self.auxk is not None:
            aux_loss = self._compute_auxk_loss(x, recon, pre_act, codes, norm_info)
            result["total"] = recon_loss + self.auxk_coef * aux_loss
            result["aux"] = aux_loss

        return result

    def _loss_triton(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """loss() using the sparse Triton decoder.

        Numerically equivalent to the dense loss() but never materializes the dense
        [batch, hidden_dim] code tensor or runs the full decoder matmul: it decodes
        from the top-k (values, indices) and derives dead-latent stats / L0 from the
        indices. This is what lets hidden_dim scale to ~1M+.
        """
        pre_act, info = self.encode_pre_act(x)
        codes_relu = torch.relu(pre_act)
        top_k_vals, top_k_indices = torch.topk(codes_relu, self.top_k, dim=-1)

        # Sparse decode in normalized space (pre_bias added); denormalize for the main loss.
        recon_norm = self._decode_topk_triton(top_k_vals, top_k_indices, info, denormalize=False)
        recon = self._denormalize(recon_norm, info) if (self.normalize_input and info) else recon_norm

        # Dead-latent stats from indices (no dense codes tensor).
        self._update_dead_latent_stats_from_indices(top_k_indices, x.shape[0])

        # Primary reconstruction loss (FVU), centered by pre_bias -- matches dense loss().
        mse = (recon - x).pow(2).mean(dim=-1)
        x_var = (x - self.pre_bias).pow(2).mean(dim=-1)
        recon_loss = (mse / (x_var + 1e-8)).mean()

        # For TopK, L0 == count of nonzero top-k values.
        l0 = (top_k_vals != 0).float().sum(dim=-1).mean()

        with torch.no_grad():
            raw_mse = (recon - x).pow(2).mean()
            total_var = torch.var(x, dim=0).sum()
            residual_var = torch.var(recon - x, dim=0).sum()
            var_explained = 1.0 - (residual_var / (total_var + 1e-8))

        result = {
            "total": recon_loss,
            "fvu": 1.0 - var_explained,
            "sparsity": l0,
            "mse": raw_mse,
            "variance_explained": var_explained,
        }
        dead_pct = (self.stats_last_nonzero > self.dead_tokens_threshold).float().mean() * 100
        result["dead_pct"] = dead_pct

        if self.auxk is not None:
            aux_loss = self._compute_auxk_loss(x, recon, pre_act, codes=None, norm_info=info, recon_norm=recon_norm)
            result["total"] = recon_loss + self.auxk_coef * aux_loss
            result["aux"] = aux_loss

        return result
