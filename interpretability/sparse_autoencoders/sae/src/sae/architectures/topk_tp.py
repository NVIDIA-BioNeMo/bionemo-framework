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

"""Tensor-parallel (latent-sharded) TopK SAE.

Each rank owns a contiguous block of ``hidden_dim // world_size`` latents: its slice
of the encoder rows, latent bias, and decoder columns. ``pre_bias`` is replicated.
Forward: each rank computes local pre-activations, a sharded global top-k selects the
true global top-k, each rank decodes the selections it owns, and the partial
reconstructions are summed across ranks (all-reduce).

Numerically equivalent to the dense ``TopKSAE`` (verified by parity tests). Kept as a
separate class so the dense ``TopKSAE`` is untouched; small helpers (_normalize) are
duplicated rather than refactored out of it.

Replicated ``pre_bias`` gradient note: ``pre_bias`` contributes via both the encoder
(``x - pre_bias``, sharded) and the decoder (added once). We add ``pre_bias /
world_size`` inside the all-reduced decode path so that, after an all-reduce(SUM) of
the ``pre_bias`` gradient across ranks (done by the TP trainer), the total gradient
equals the dense one exactly: sharded encoder parts sum, and the decoder part (1/P per
rank) sums back to a single full contribution.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels import HAS_TRITON
from ..parallel import all_reduce_sum, autograd_all_reduce_sum, global_topk


class ShardedTopKSAE(nn.Module):
    """Latent-sharded TopK SAE (tensor parallel across `world_size` ranks)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        top_k: int,
        rank: int,
        world_size: int,
        normalize_input: bool = True,
        auxk: Optional[int] = None,
        auxk_coef: float = 1 / 32,
        dead_tokens_threshold: int = 10_000_000,
        decoder_impl: str = "dense",
        group=None,
    ):
        """Args mirror TopKSAE; `hidden_dim` is the GLOBAL latent count (divisible by world_size)."""
        super().__init__()
        if hidden_dim % world_size != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by world_size {world_size}")
        if decoder_impl not in ("dense", "triton"):
            raise ValueError(f"decoder_impl must be 'dense' or 'triton', got {decoder_impl!r}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.rank = rank
        self.world_size = world_size
        self.normalize_input = normalize_input
        self.auxk = auxk
        self.auxk_coef = auxk_coef
        self.dead_tokens_threshold = dead_tokens_threshold
        self.decoder_impl = decoder_impl
        self.group = group
        self.latents_per_rank = hidden_dim // world_size

        L = self.latents_per_rank
        self.W_enc_local = nn.Parameter(torch.empty(L, input_dim))
        self.latent_bias_local = nn.Parameter(torch.zeros(L))
        self.W_dec_local = nn.Parameter(torch.empty(input_dim, L))
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))  # replicated
        nn.init.kaiming_uniform_(self.W_dec_local, a=5**0.5)
        with torch.no_grad():
            self.W_enc_local.copy_(self.W_dec_local.t())  # encoder = decoder.T init

        self.register_buffer("stats_last_nonzero", torch.zeros(L, dtype=torch.long))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Unit-norm each decoder ROW over all latents (matches dense normalize_decoder).

        Dense does F.normalize(weight[d, n], dim=1). Each rank holds [d, L], so the
        per-row norm over the full n latents needs an all-reduce of per-row
        sum-of-squares. Load-bearing for TopK training stability.
        """
        sumsq = (self.W_dec_local**2).sum(dim=1)  # [d], local sum over this rank's L latents
        all_reduce_sum(sumsq, self.group)  # -> per-row sum-of-squares over all n latents
        norm = sumsq.clamp_min(1e-12).sqrt().unsqueeze(1)  # [d, 1]
        self.W_dec_local.data = self.W_dec_local.data / norm

    def post_step(self) -> None:
        """Called by the TP trainer after optimizer.step()."""
        self.normalize_decoder()

    def reduce_replicated_grads(self) -> None:
        """All-reduce(SUM) gradients of replicated params (pre_bias) across the TP group.

        Sharded params are distinct per rank and need no sync. pre_bias is replicated:
        summing its grad combines the per-rank encoder contributions and the 1/P decode
        parts into the exact dense gradient (see class docstring). Call after backward,
        before optimizer.step().
        """
        if self.pre_bias.grad is not None:
            all_reduce_sum(self.pre_bias.grad, self.group)

    @torch.no_grad()
    def load_shard_from_dense(self, dense) -> None:
        """Copy this rank's slice of a dense TopKSAE's weights (for tests / merge)."""
        lo = self.rank * self.latents_per_rank
        hi = lo + self.latents_per_rank
        self.W_enc_local.copy_(dense.encoder.weight[lo:hi, :])
        self.latent_bias_local.copy_(dense.latent_bias[lo:hi])
        self.W_dec_local.copy_(dense.decoder.weight[:, lo:hi])
        self.pre_bias.copy_(dense.pre_bias)

    def _normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        return (x - mu) / std, {"mu": mu, "std": std}

    def _denormalize(self, x: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return x * info["std"] + info["mu"]

    @torch.no_grad()
    def init_pre_bias_from_data(self, data: torch.Tensor, max_iter: int = 100, eps: float = 1e-6) -> None:
        """Initialize the (replicated) pre_bias to the geometric median of the data.

        Identical computation to dense TopKSAE.init_pre_bias_from_data; since pre_bias
        is replicated and every rank sees the same data sample, all ranks compute the
        same value (no communication needed).
        """
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
        self.pre_bias.data = median.to(self.pre_bias.device)

    def encode_pre_act(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Local pre-activations [batch, L]: optional normalize, subtract pre_bias, encoder."""
        info: Dict[str, torch.Tensor] = {}
        if self.normalize_input:
            x, info = self._normalize(x)
        x_centered = x - self.pre_bias
        pre_act_local = F.linear(x_centered, self.W_enc_local, self.latent_bias_local)  # [B, L]
        return pre_act_local, info

    def _decode_local(self, vals: torch.Tensor, local_indices: torch.Tensor) -> torch.Tensor:
        """Decode this rank's owned selections -> partial reconstruction [B, d] (normalized space)."""
        if self.decoder_impl == "triton" and HAS_TRITON and vals.is_cuda:
            from ..kernels import TritonDecoderAutograd

            return TritonDecoderAutograd.apply(local_indices.contiguous(), vals.contiguous(), self.W_dec_local)
        # Dense gather-sum (no scatter -> safe against the index-0 padding of unowned slots):
        # partial[b] = sum_j vals[b,j] * W_dec_local[:, local_indices[b,j]]
        gathered = self.W_dec_local.t()[local_indices]  # [B, k, d]
        return (gathered * vals.unsqueeze(-1)).sum(dim=1)

    def forward(self, x: torch.Tensor):
        """Return (reconstruction [B, d], GlobalTopK). recon is replicated across ranks."""
        pre_act_local, info = self.encode_pre_act(x)
        acts_local = torch.relu(pre_act_local)

        gtk = global_topk(acts_local, self.top_k, self.rank, self.latents_per_rank, self.group)
        # Differentiable values from the local activations (grad flows to the encoder);
        # zero out selections this rank does not own.
        vals = acts_local.gather(1, gtk.local_indices) * gtk.owned_mask.to(acts_local.dtype)

        partial = self._decode_local(vals, gtk.local_indices)
        # See class docstring: pre_bias/world_size makes the replicated-grad sum exact.
        partial = partial + self.pre_bias / self.world_size
        recon = autograd_all_reduce_sum(partial, self.group)  # sum partials -> full recon (normalized space)

        if self.normalize_input and info:
            recon = self._denormalize(recon, info)
        return recon, gtk

    def _update_dead_latent_stats_local(self, gtk, vals: torch.Tensor, n_tokens: int) -> None:
        """Mark local latents that fired (owned selection with value > 1e-3); else age them."""
        active = torch.zeros_like(self.stats_last_nonzero, dtype=torch.bool)
        fired = vals > 1e-3  # [B, k]; vals already zeroed on unowned slots
        active[gtk.local_indices[fired]] = True
        self.stats_last_nonzero = torch.where(
            active, torch.zeros_like(self.stats_last_nonzero), self.stats_last_nonzero + n_tokens
        )

    def loss(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Sharded loss with the same keys/values as the dense TopKSAE.loss().

        recon is replicated across ranks, so recon-derived metrics match dense on
        every rank; dead_pct is reduced globally and auxk uses a sharded global
        top-k over dead latents.
        """
        pre_act_local, info = self.encode_pre_act(x)
        acts_local = torch.relu(pre_act_local)
        gtk = global_topk(acts_local, self.top_k, self.rank, self.latents_per_rank, self.group)
        vals = acts_local.gather(1, gtk.local_indices) * gtk.owned_mask.to(acts_local.dtype)

        partial = self._decode_local(vals, gtk.local_indices) + self.pre_bias / self.world_size
        recon_norm = autograd_all_reduce_sum(partial, self.group)
        recon = self._denormalize(recon_norm, info) if (self.normalize_input and info) else recon_norm

        self._update_dead_latent_stats_local(gtk, vals, x.shape[0])

        mse = (recon - x).pow(2).mean(dim=-1)
        # x_var uses pre_bias in a *replicated* (non-sharded) way, so its gradient would
        # be over-counted x world_size by the all-reduce(SUM) of pre_bias grads. Scale the
        # grad by 1/world_size (value unchanged) so the sum recovers the dense gradient --
        # same principle as the pre_bias/world_size decode term.
        pb = self.pre_bias / self.world_size + (self.pre_bias - self.pre_bias / self.world_size).detach()
        x_var = (x - pb).pow(2).mean(dim=-1)
        recon_loss = (mse / (x_var + 1e-8)).mean()
        l0 = (gtk.global_values != 0).float().sum(dim=-1).mean()

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

        # Global dead fraction across all shards.
        with torch.no_grad():
            local_dead = (self.stats_last_nonzero > self.dead_tokens_threshold).sum().float()
            total_dead = all_reduce_sum(local_dead.clone(), self.group)
            result["dead_pct"] = total_dead / self.hidden_dim * 100

        if self.auxk is not None:
            aux_loss = self._compute_auxk_loss(x, recon, recon_norm, pre_act_local, info)
            result["total"] = recon_loss + self.auxk_coef * aux_loss
            result["aux"] = aux_loss

        return result

    def _compute_auxk_loss(self, x, recon, recon_norm, pre_act_local, info) -> torch.Tensor:
        """Auxiliary dead-latent loss: a sharded global top-auxk over dead latents.

        Mirrors dense TopKSAE._compute_auxk_loss (top-auxk among dead by relu value,
        decode, fit the primary residual in normalized space).
        """
        dead_mask_local = self.stats_last_nonzero > self.dead_tokens_threshold  # [L]
        total_dead = int(all_reduce_sum(dead_mask_local.sum().float().clone(), self.group).item())
        if total_dead == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        k_aux = min(self.auxk, total_dead)
        acts_local = torch.relu(pre_act_local)
        # Only dead latents are selectable; -inf so the global top-k never picks live ones.
        masked = acts_local.masked_fill(~dead_mask_local, float("-inf"))
        gtk_aux = global_topk(masked, k_aux, self.rank, self.latents_per_rank, self.group)
        vals_aux = acts_local.gather(1, gtk_aux.local_indices) * gtk_aux.owned_mask.to(acts_local.dtype)
        recon_aux = autograd_all_reduce_sum(self._decode_local(vals_aux, gtk_aux.local_indices), self.group)

        if self.normalize_input and info:
            x_norm = (x - info["mu"]) / info["std"]
            residual = x_norm - recon_norm.detach()
        else:
            residual = x - recon.detach() + self.pre_bias.detach()

        mse = (recon_aux - residual).pow(2).mean(dim=-1)
        target_var = residual.pow(2).mean(dim=-1)
        return (mse / (target_var + 1e-8)).mean()
