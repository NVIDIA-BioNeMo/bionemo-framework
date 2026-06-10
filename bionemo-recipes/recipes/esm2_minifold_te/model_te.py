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

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from miniformer_te import MiniFormerTE
from te_utils import te_layernorm_nd, te_linear_nd


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask):
        """Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """
        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.
        diff[mask == 0] = 0
        output = self.embedding(diff)
        return output


class SequenceToPairTE(nn.Module):
    """TE version of SequenceToPair."""

    def __init__(
        self,
        sequence_state_dim,
        inner_dim,
        pairwise_state_dim,
        params_dtype=torch.float32,
        component_precision=None,
    ):
        super().__init__()
        self._component_precision = component_precision
        self.layernorm = te.LayerNorm(sequence_state_dim, eps=1e-5, params_dtype=params_dtype)
        self.proj = te.Linear(sequence_state_dim, inner_dim * 2, bias=True, params_dtype=params_dtype)
        self.o_proj = te.Linear(2 * inner_dim, pairwise_state_dim, bias=True, params_dtype=params_dtype)

        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, sequence_state):
        """Forward pass.

        Args:
            sequence_state: B x L x sequence_state_dim

        Returns:
            pairwise_state: B x L x L x pairwise_state_dim
        """
        cp = self._component_precision
        ctx = cp.get_context("seq_proj") if cp else nullcontext()
        with ctx:
            s = te_layernorm_nd(self.layernorm, sequence_state)
            s = te_linear_nd(self.proj, s)
            q, k = s.chunk(2, dim=-1)

            prod = q[:, None, :, :] * k[:, :, None, :]
            diff = q[:, None, :, :] - k[:, :, None, :]

            x = torch.cat([prod, diff], dim=-1)
            x = te_linear_nd(self.o_proj, x)

        return x


class PairToSequenceTE(nn.Module):
    """TE version of PairToSequence."""

    def __init__(self, c_z=128, c_s=1024, c_s_out=1024, params_dtype=torch.float32):
        super().__init__()
        self.s_z_norm = te.LayerNorm(c_z, eps=1e-5, params_dtype=params_dtype)
        self.s_z_fc1 = te.Linear(c_z, c_z, params_dtype=params_dtype)
        self.s_z_fc2 = te.Linear(c_z, c_z, params_dtype=params_dtype)
        self.combiner = te.Linear(2 * c_z + c_s, c_s_out, params_dtype=params_dtype)

    def forward(self, s_z, s_s_in, pair_mask):
        """Forward pass.

        Args:
            s_z: Pair representation (B, L, L, c_z).
            s_s_in: Sequence representation (B, L, c_s).
            pair_mask: Pair mask (B, L, L).

        Returns:
            Sequence representation (B, L, c_s_out).
        """
        # MLP on pair features
        s_z = te_layernorm_nd(self.s_z_norm, s_z)
        s_z = te_linear_nd(self.s_z_fc1, s_z)
        s_z = F.relu(s_z)
        s_z = te_linear_nd(self.s_z_fc2, s_z)

        # Apply mask
        s_z = s_z * pair_mask[..., None]

        # Column average
        norm = pair_mask.sum(dim=2).clamp(min=1)
        s_s_c = s_z.sum(dim=2) / norm[..., None]

        # Row average
        norm = pair_mask.sum(dim=1).clamp(min=1)
        s_s_r = s_z.sum(dim=1) / norm[..., None]

        # Combine with initial s_s
        s_s = te_linear_nd(self.combiner, torch.cat([s_s_c, s_s_r, s_s_in], dim=-1))
        return s_s


class FoldingTrunkTE(nn.Module):
    """TE version of FoldingTrunk."""

    def __init__(
        self,
        c_s,
        c_z,
        bins,
        disto_bins=64,
        num_layers=1,
        params_dtype=torch.float32,
        block_precision=None,
        fp8_recipe=None,
        fp4_recipe=None,
        component_precision=None,
    ):
        super().__init__()
        self._component_precision = component_precision
        self.disto_bins = disto_bins
        self.positional_embedding = RelativePosition(bins, c_z)
        self.seq_to_pair = SequenceToPairTE(
            c_s, c_z // 2, c_z, params_dtype=params_dtype, component_precision=component_precision
        )
        self.projection = te.Linear(c_z * 3, c_z, params_dtype=params_dtype)
        self.recycle = te.Linear(disto_bins, c_z, params_dtype=params_dtype)
        self.miniformer = MiniFormerTE(
            c_z,
            blocks=num_layers,
            params_dtype=params_dtype,
            block_precision=block_precision,
            fp8_recipe=fp8_recipe,
            fp4_recipe=fp4_recipe,
            component_precision=component_precision,
        )
        self.fc_out_1 = te.Linear(c_z, c_z, params_dtype=params_dtype)
        self.fc_out_2 = te.Linear(c_z, disto_bins, params_dtype=params_dtype)

        torch.nn.init.zeros_(self.seq_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.seq_to_pair.o_proj.bias)

    def forward(self, s_s, s_z, mask, num_recycling=0):
        """Forward pass.

        Args:
            s_s: Sequence features (B, L, C).
            s_z: Pair features (B, L, L, C).
            mask: Residue mask (B, L).
            num_recycling: Number of recycling iterations.

        Returns:
            Tuple of (predictions, pair representation).
        """
        # Make pairwise mask
        pair_mask = mask[:, None, :] * mask[:, :, None]

        # Add positional embeddings
        residx = torch.arange(s_s.shape[1], device=s_s.device)
        residx = residx.unsqueeze(0).expand(s_s.shape[0], -1)

        # Concatenate and project
        s_z = torch.cat(
            [
                s_z,
                self.seq_to_pair(s_s),
                self.positional_embedding(residx, mask=pair_mask),
            ],
            dim=-1,
        )
        s_z = te_linear_nd(self.projection, s_z)

        # Set masks to floats
        mask = mask.to(s_z)
        pair_mask = pair_mask.to(s_z)

        # Initialize binned distance
        shape = tuple(s_z.shape[:3]) + (self.disto_bins,)
        dists = torch.zeros(shape, device=s_z.device, dtype=s_z.dtype)

        # Perform folding rounds
        for i in range(num_recycling + 1):
            with torch.set_grad_enabled(self.training and (i == num_recycling)):
                if self.training and (i == num_recycling) and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                # Compute blocks
                s_z_c = s_z + te_linear_nd(self.recycle, dists)
                s_z_c = self.miniformer(s_z_c, pair_mask)

                # Output MLP
                cp = self._component_precision
                dist_ctx = cp.get_context("dist_head") if cp else nullcontext()
                with dist_ctx:
                    fc_out = te_linear_nd(self.fc_out_1, s_z_c + s_z_c.transpose(1, 2))
                    fc_out = F.relu(fc_out)
                    preds = te_linear_nd(self.fc_out_2, fc_out)

                # Compute binned distance for recycling
                dists = preds.detach().argmax(dim=-1)
                dists = nn.functional.one_hot(dists, self.disto_bins).to(s_z)

        return preds, s_z_c
