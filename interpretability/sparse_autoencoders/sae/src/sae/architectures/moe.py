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

"""Mixture-of-Experts Sparse Autoencoder.

Partitions the feature dictionary across N expert sub-SAEs with a lightweight
learned router. Each input selects top-k experts; final reconstruction is the
gating-weighted sum of active experts' reconstructions. Effective dictionary
size = d_sae, split as d_sae // n_experts per expert.

Supports:
  - Configurable per-expert sparsity: TopK or ReLU+L1
  - Optional shared backbone SAE (experts handle the residual)
  - Load balancing loss to prevent expert collapse
  - Router entropy regularization (positive=uniform, negative=sharp)
  - Per-expert diagnostics (utilization, dead experts, recon error)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SparseAutoencoder


# ---------------------------------------------------------------------------
# Virtual decoder wrapper for analysis tool compatibility
# ---------------------------------------------------------------------------


class _VirtualDecoder(nn.Module):
    """Exposes a concatenated ``.weight`` from multiple expert decoders.

    Analysis tools (``compute_feature_umap``, ``compute_feature_logits``)
    access ``sae.decoder.weight`` directly.  This wrapper provides that
    interface by horizontally concatenating expert decoder weight matrices.
    """

    def __init__(
        self,
        expert_decoders: nn.ModuleList,
        shared_decoder: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self._expert_decoders = expert_decoders
        self._shared_decoder = shared_decoder

    @property
    def weight(self) -> torch.Tensor:
        """Shape: ``[d_in, d_sae]`` (+ ``d_shared`` if shared backbone)."""
        parts = [d.weight for d in self._expert_decoders]
        if self._shared_decoder is not None:
            parts.append(self._shared_decoder.weight)
        return torch.cat(parts, dim=1)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return F.linear(codes, self.weight.T)


# ---------------------------------------------------------------------------
# Small helper: build one expert's encoder/decoder pair
# ---------------------------------------------------------------------------


def _init_expert_pair(
    encoder: nn.Linear,
    decoder: nn.Linear,
    init_encoder_from_decoder: bool,
) -> None:
    """Xavier-init decoder, optionally tie encoder = decoder^T, normalize."""
    nn.init.xavier_uniform_(decoder.weight)
    with torch.no_grad():
        decoder.weight.data = F.normalize(decoder.weight.data, dim=0)
        if init_encoder_from_decoder:
            encoder.weight.data = decoder.weight.data.T.clone()


# ---------------------------------------------------------------------------
# MoE SAE
# ---------------------------------------------------------------------------


class MoESAE(SparseAutoencoder):
    """Mixture-of-Experts Sparse Autoencoder.

    Args:
        d_in: Input activation dimension.
        d_sae: Total dictionary size (split evenly across experts).
        n_experts: Number of expert sub-SAEs.
        k_experts: Number of experts active per input.
        expert_mode: Per-expert sparsity — ``"topk"`` or ``"relu"``.
        top_k: Top-k per expert (used when ``expert_mode="topk"``).
        l1_coeff: L1 coefficient per expert (used when ``expert_mode="relu"``).
        normalize_input: Per-sample standardisation before encoding.
        load_balance_coeff: Coefficient for load-balancing auxiliary loss.
        router_entropy_coeff: Router entropy regulariser.
            Positive  → encourages uniform routing.
            Negative  → encourages sharp specialisation.
            Zero (default) → off.
        d_shared: If not None, size of a shared backbone SAE that processes
            input first; experts then handle the residual.
        shared_mode: Sparsity mode for the shared backbone (``"topk"`` or
            ``"relu"``).
        shared_top_k: Top-k for shared backbone (when ``shared_mode="topk"``).
        init_encoder_from_decoder: Initialise each encoder as transpose of
            its decoder (reduces dead latents).
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        n_experts: int = 8,
        k_experts: int = 2,
        expert_mode: str = "topk",
        top_k: int = 32,
        l1_coeff: float = 1e-2,
        normalize_input: bool = True,
        load_balance_coeff: float = 0.01,
        router_entropy_coeff: float = 0.0,
        d_shared: Optional[int] = None,
        shared_mode: str = "topk",
        shared_top_k: int = 32,
        init_encoder_from_decoder: bool = True,
    ):
        """Initialize the MoE SAE with expert sub-SAEs and a learned router."""
        if d_sae % n_experts != 0:
            raise ValueError(f"d_sae ({d_sae}) must be divisible by n_experts ({n_experts})")

        total_hidden = d_sae + (d_shared if d_shared else 0)
        super().__init__(d_in, total_hidden)

        # Store configuration
        self.d_in = d_in
        self.d_sae = d_sae
        self.n_experts = n_experts
        self.k_experts = k_experts
        self.expert_mode = expert_mode
        self.top_k = top_k
        self.l1_coeff = l1_coeff
        self.normalize_input = normalize_input
        self.load_balance_coeff = load_balance_coeff
        self.router_entropy_coeff = router_entropy_coeff
        self.d_shared = d_shared
        self.shared_mode = shared_mode
        self.shared_top_k = shared_top_k
        self.expert_dim = d_sae // n_experts

        # ----- Router -----
        self.router = nn.Linear(d_in, n_experts, bias=False)

        # ----- Expert parameters -----
        self.expert_pre_biases = nn.ParameterList([nn.Parameter(torch.zeros(d_in)) for _ in range(n_experts)])
        self.expert_encoders = nn.ModuleList([nn.Linear(d_in, self.expert_dim, bias=False) for _ in range(n_experts)])
        self.expert_latent_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.expert_dim)) for _ in range(n_experts)]
        )
        self.expert_decoders = nn.ModuleList([nn.Linear(self.expert_dim, d_in, bias=False) for _ in range(n_experts)])

        # Initialise expert encoder/decoder pairs
        for enc, dec in zip(self.expert_encoders, self.expert_decoders):
            _init_expert_pair(enc, dec, init_encoder_from_decoder)

        # ----- Optional shared backbone -----
        self.shared_encoder: Optional[nn.Linear] = None
        self.shared_decoder: Optional[nn.Linear] = None
        self.shared_latent_bias: Optional[nn.Parameter] = None
        self.shared_pre_bias: Optional[nn.Parameter] = None

        if d_shared is not None:
            self.shared_pre_bias = nn.Parameter(torch.zeros(d_in))
            self.shared_encoder = nn.Linear(d_in, d_shared, bias=False)
            self.shared_latent_bias = nn.Parameter(torch.zeros(d_shared))
            self.shared_decoder = nn.Linear(d_shared, d_in, bias=False)
            _init_expert_pair(self.shared_encoder, self.shared_decoder, init_encoder_from_decoder)

        # ----- Virtual decoder for analysis compatibility -----
        self._virtual_decoder = _VirtualDecoder(
            self.expert_decoders,
            shared_decoder=self.shared_decoder,
        )

        # ----- Diagnostic buffers (updated during loss()) -----
        self.register_buffer(
            "expert_utilization",
            torch.zeros(n_experts),
        )
        self.register_buffer(
            "expert_recon_error",
            torch.zeros(n_experts),
        )
        self.register_buffer(
            "expert_update_count",
            torch.zeros(1, dtype=torch.long),
        )
        # EMA smoothing factor for running stats
        self._diag_ema = 0.01

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def decoder(self) -> _VirtualDecoder:  # type: ignore[override]
        """Virtual decoder whose ``.weight`` concatenates all expert decoders."""
        return self._virtual_decoder

    # ------------------------------------------------------------------
    # Normalisation helpers (mirrors TopKSAE pattern)
    # ------------------------------------------------------------------

    def _normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        return (x - mu) / std, {"mu": mu, "std": std}

    def _denormalize(self, x: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return x * info["std"] + info["mu"]

    # ------------------------------------------------------------------
    # Shared backbone helpers
    # ------------------------------------------------------------------

    def _shared_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode through the shared backbone."""
        assert self.shared_encoder is not None
        x_centered = x - self.shared_pre_bias
        pre_act = self.shared_encoder(x_centered) + self.shared_latent_bias

        if self.shared_mode == "topk":
            codes = torch.relu(pre_act)
            topvals, topidx = torch.topk(codes, self.shared_top_k, dim=-1)
            codes = torch.zeros_like(codes).scatter(-1, topidx, topvals)
        else:  # relu
            codes = torch.relu(pre_act)
        return codes

    def _shared_decode(self, codes: torch.Tensor) -> torch.Tensor:
        assert self.shared_decoder is not None
        return self.shared_decoder(codes) + self.shared_pre_bias

    # ------------------------------------------------------------------
    # Per-expert encode / decode
    # ------------------------------------------------------------------

    def _expert_encode(self, x: torch.Tensor, expert_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode *x* through expert *expert_idx*.

        Returns ``(codes, l1_value)`` where *l1_value* is the per-sample
        L1 norm of the codes (used for sparsity loss in relu mode).
        """
        x_centered = x - self.expert_pre_biases[expert_idx]
        pre_act = self.expert_encoders[expert_idx](x_centered) + self.expert_latent_biases[expert_idx]
        codes = torch.relu(pre_act)

        if self.expert_mode == "topk":
            topvals, topidx = torch.topk(codes, self.top_k, dim=-1)
            codes = torch.zeros_like(codes).scatter(-1, topidx, topvals)

        l1 = codes.abs().sum(dim=-1)  # [n_selected]
        return codes, l1

    def _expert_decode(self, codes: torch.Tensor, expert_idx: int) -> torch.Tensor:
        return self.expert_decoders[expert_idx](codes) + self.expert_pre_biases[expert_idx]

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing decisions.

        Returns:
            router_logits: ``[batch, n_experts]``
            router_probs:  ``[batch, n_experts]`` (full softmax)
            top_indices:   ``[batch, k_experts]``
            gate_weights:  ``[batch, k_experts]`` (renormalised probs for
                           selected experts, sum to 1 per sample)
        """
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        top_probs, top_indices = torch.topk(router_probs, self.k_experts, dim=-1)
        gate_weights = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-8)

        return router_logits, router_probs, top_indices, gate_weights

    # ------------------------------------------------------------------
    # Forward variants
    # ------------------------------------------------------------------

    def forward_with_aux(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass returning routing metadata.

        Returns dict with keys:
            recon, codes, router_logits, router_probs, top_indices,
            gate_weights, norm_info, sparsity_loss, per_expert_mse
        """
        batch = x.shape[0]
        norm_info: Dict[str, torch.Tensor] = {}

        if self.normalize_input:
            x_normed, norm_info = self._normalize(x)
        else:
            x_normed = x

        # --- Shared backbone (optional) ---
        shared_codes: Optional[torch.Tensor] = None
        shared_recon: Optional[torch.Tensor] = None
        if self.d_shared is not None:
            shared_codes = self._shared_encode(x_normed)
            shared_recon = self._shared_decode(shared_codes)
            route_input = x_normed - shared_recon.detach()
        else:
            route_input = x_normed

        # --- Routing ---
        router_logits, router_probs, top_indices, gate_weights = self._route(route_input)

        # --- Expert processing ---
        combined_codes = x.new_zeros(batch, self.d_sae)
        combined_recon = x.new_zeros(batch, self.d_in)
        total_sparsity_loss = x.new_zeros(1)
        per_expert_mse = x.new_zeros(self.n_experts)

        for e in range(self.n_experts):
            # mask: which samples selected this expert?
            # top_indices is [batch, k_experts]
            match = top_indices == e  # [batch, k_experts]
            sample_mask = match.any(dim=-1)  # [batch]
            n_selected = sample_mask.sum().item()
            if n_selected == 0:
                continue

            # Gate weight for this expert for selected samples
            # For each selected sample, find which position(s) matched
            # and take the corresponding gate weight.
            # In the common case each expert appears at most once per
            # sample, so we sum across the k_experts dim.
            gate_e = (match.float() * gate_weights).sum(dim=-1)  # [batch]
            gate_e_selected = gate_e[sample_mask]  # [n_selected]

            x_selected = route_input[sample_mask]  # [n_selected, d_in]

            # Encode / decode
            codes_e, l1_e = self._expert_encode(x_selected, e)
            recon_e = self._expert_decode(codes_e, e)

            # Place codes in combined tensor
            offset = e * self.expert_dim
            combined_codes[sample_mask, offset : offset + self.expert_dim] = codes_e

            # Gating-weighted reconstruction
            combined_recon[sample_mask] += gate_e_selected.unsqueeze(-1) * recon_e

            # Accumulate sparsity loss (only meaningful for relu mode)
            total_sparsity_loss = total_sparsity_loss + l1_e.mean()

            # Per-expert reconstruction error (for diagnostics)
            with torch.no_grad():
                per_expert_mse[e] = (recon_e - x_selected).pow(2).mean()

        # --- Combine with shared backbone ---
        if shared_recon is not None:
            combined_recon = shared_recon + combined_recon

        # --- Denormalize ---
        if self.normalize_input:
            combined_recon = self._denormalize(combined_recon, norm_info)

        # --- Assemble full codes vector (shared codes appended at end) ---
        if shared_codes is not None:
            all_codes = torch.cat([combined_codes, shared_codes], dim=-1)
        else:
            all_codes = combined_codes

        return {
            "recon": combined_recon,
            "codes": all_codes,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "top_indices": top_indices,
            "gate_weights": gate_weights,
            "norm_info": norm_info,
            "sparsity_loss": total_sparsity_loss,
            "per_expert_mse": per_expert_mse,
            "shared_codes": shared_codes,
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, codes)``."""
        info = self.forward_with_aux(x)
        return info["recon"], info["codes"]

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to combined codes ``[batch, hidden_dim]``."""
        _, codes = self.forward(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode combined codes via the virtual (concatenated) decoder.

        This is a simple linear projection used for analysis convenience;
        the actual forward pass uses per-expert decode with gating weights.
        """
        return self._virtual_decoder(codes)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute total loss including reconstruction, sparsity, load-balancing, and router entropy."""
        info = self.forward_with_aux(x)
        recon = info["recon"]
        codes = info["codes"]
        router_probs = info["router_probs"]
        top_indices = info["top_indices"]

        # ---- 1. Reconstruction loss (MSE) ----
        recon_loss = F.mse_loss(recon, x)

        # ---- 2. Sparsity loss (per-expert, meaningful for relu mode) ----
        sparsity_loss = info["sparsity_loss"]

        # ---- 3. Load-balancing loss ----
        # f_e = fraction of tokens dispatched to expert e
        dispatch = F.one_hot(top_indices, self.n_experts).float()  # [B, k, E]
        f = dispatch.sum(dim=1).mean(dim=0)  # [E]
        # p_e = mean router probability for expert e
        p = router_probs.mean(dim=0)  # [E]
        load_balance_loss = self.n_experts * (f * p).sum()

        # ---- 4. Router entropy regularisation ----
        if self.router_entropy_coeff != 0.0:
            # H(p) = -sum(p * log(p))
            entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean()
            # positive coeff → loss = -coeff*H → encourages high entropy (uniform)
            # negative coeff → loss = -coeff*H → encourages low entropy (sharp)
            router_entropy_loss = -self.router_entropy_coeff * entropy
        else:
            router_entropy_loss = x.new_zeros(1)
            entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean()

        # ---- Shared backbone sparsity (if present, relu mode) ----
        shared_l1 = x.new_zeros(1)
        if self.d_shared is not None and self.shared_mode == "relu":
            shared_l1 = info["shared_codes"].abs().sum(dim=-1).mean()

        # ---- Total loss ----
        total = recon_loss + self.load_balance_coeff * load_balance_loss
        total = total + router_entropy_loss
        if self.expert_mode == "relu":
            total = total + self.l1_coeff * sparsity_loss
        if self.d_shared is not None and self.shared_mode == "relu":
            total = total + self.l1_coeff * shared_l1

        # ---- Eval metrics (no extra forward pass) ----
        with torch.no_grad():
            l0 = (codes != 0).float().sum(dim=-1).mean()
            raw_mse = (recon - x).pow(2).mean()
            total_var = torch.var(x, dim=0).sum()
            residual_var = torch.var(recon - x, dim=0).sum()
            var_explained = 1.0 - (residual_var / (total_var + 1e-8))

            # Expert utilisation for this batch
            batch_util = f.detach()

        # ---- Update diagnostic buffers ----
        self._update_diagnostics(batch_util, info["per_expert_mse"])

        result: Dict[str, torch.Tensor] = {
            "total": total,
            "fvu": recon_loss,
            "sparsity": l0,
            "mse": raw_mse,
            "variance_explained": var_explained,
            "load_balance": load_balance_loss,
            "router_entropy": entropy.detach(),
            # Scalar summary of expert utilisation (std=0 means perfectly balanced)
            "expert_util_std": batch_util.std(),
        }
        if self.expert_mode == "relu":
            result["l1"] = sparsity_loss
        if self.d_shared is not None and self.shared_mode == "relu":
            result["shared_l1"] = shared_l1

        return result

    # ------------------------------------------------------------------
    # Post-step: normalise all expert (and shared) decoders
    # ------------------------------------------------------------------

    def post_step(self) -> None:
        """Normalize all expert and shared decoder weights to unit norm."""
        with torch.no_grad():
            for dec in self.expert_decoders:
                dec.weight.data = F.normalize(dec.weight.data, dim=0)
            if self.shared_decoder is not None:
                self.shared_decoder.weight.data = F.normalize(self.shared_decoder.weight.data, dim=0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _update_diagnostics(
        self,
        batch_util: torch.Tensor,
        per_expert_mse: torch.Tensor,
    ) -> None:
        """EMA update of running utilisation and per-expert recon error."""
        alpha = self._diag_ema
        self.expert_utilization.mul_(1 - alpha).add_(batch_util, alpha=alpha)
        self.expert_recon_error.mul_(1 - alpha).add_(per_expert_mse.detach(), alpha=alpha)
        self.expert_update_count += 1

    def get_expert_diagnostics(self) -> Dict[str, Any]:
        """Return per-expert diagnostic statistics.

        Keys:
            per_expert_utilization: ``[n_experts]`` running fraction of inputs
                routed to each expert (EMA).
            dead_experts: list of expert indices with utilisation < 1/(10*n_experts).
            per_expert_recon_error: ``[n_experts]`` running average MSE per expert.
        """
        util = self.expert_utilization.detach().cpu()
        threshold = 1.0 / (10.0 * self.n_experts)
        dead = (util < threshold).nonzero(as_tuple=False).flatten().tolist()
        return {
            "per_expert_utilization": util,
            "dead_experts": dead,
            "per_expert_recon_error": self.expert_recon_error.detach().cpu(),
        }

    # ------------------------------------------------------------------
    # Router analysis
    # ------------------------------------------------------------------

    def get_router_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return router softmax probabilities ``[batch, n_experts]``."""
        if self.normalize_input:
            x, _ = self._normalize(x)
        if self.d_shared is not None:
            shared_codes = self._shared_encode(x)
            shared_recon = self._shared_decode(shared_codes)
            x = x - shared_recon
        return F.softmax(self.router(x), dim=-1)

    # ------------------------------------------------------------------
    # Pre-bias initialisation
    # ------------------------------------------------------------------

    def init_pre_bias_from_data(
        self,
        data: torch.Tensor,
        max_iter: int = 100,
        eps: float = 1e-6,
    ) -> None:
        """Initialise every expert's pre_bias to the geometric median.

        Uses Weiszfeld's algorithm (same as TopKSAE).  If
        ``normalize_input=True``, operates on standardised data.
        """
        with torch.no_grad():
            data = data.float().cpu()
            if self.normalize_input:
                mu = data.mean(dim=-1, keepdim=True)
                std = data.std(dim=-1, keepdim=True) + 1e-8
                data = (data - mu) / std

            median = data.mean(dim=0)
            for _ in range(max_iter):
                diffs = data - median.unsqueeze(0)
                distances = diffs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                weights = 1.0 / distances
                new_median = (data * weights).sum(dim=0) / weights.sum()
                if (new_median - median).norm() < eps:
                    break
                median = new_median

            for bias in self.expert_pre_biases:
                bias.data = median.to(bias.device)
            if self.shared_pre_bias is not None:
                self.shared_pre_bias.data = median.to(self.shared_pre_bias.device)

    # ------------------------------------------------------------------
    # Config (for checkpointing / wandb)
    # ------------------------------------------------------------------

    def _get_config(self) -> Dict[str, Any]:
        return {
            "architecture": "MoESAE",
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "n_experts": self.n_experts,
            "k_experts": self.k_experts,
            "expert_mode": self.expert_mode,
            "top_k": self.top_k,
            "l1_coeff": self.l1_coeff,
            "normalize_input": self.normalize_input,
            "load_balance_coeff": self.load_balance_coeff,
            "router_entropy_coeff": self.router_entropy_coeff,
            "d_shared": self.d_shared,
            "shared_mode": self.shared_mode,
            "shared_top_k": self.shared_top_k,
            "expert_dim": self.expert_dim,
        }
