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

"""Global TopK across latent shards for tensor-parallel SAEs.

Each rank owns a contiguous block of `latents_per_rank` latents. To pick the global
top-k over all `world_size * latents_per_rank` latents without materializing them on
one rank, each rank takes its local top-k, all-gathers the candidates, and takes a
final top-k over the gathered `world_size * k` candidates.

This is exact: the global top-k is always a subset of the union of the per-rank
top-ks (a globally-selected latent on rank r is among r's `k` largest, since it is
larger than every unselected latent), so gathering `k` per rank loses nothing.
"""

from typing import NamedTuple

import torch

from .comms import all_gather_cat


class GlobalTopK(NamedTuple):
    """Result of a sharded global top-k.

    Attributes (shapes [batch, k]):
        global_values: top-k activation values (replicated across ranks).
        global_indices: top-k *global* latent indices (replicated).
        local_indices: per-rank local indices of the selections this rank owns
            (0 where not owned -- pair with `owned_mask`).
        owned_mask: True where the selection belongs to this rank's shard.
    """

    global_values: torch.Tensor
    global_indices: torch.Tensor
    local_indices: torch.Tensor
    owned_mask: torch.Tensor


def global_topk(
    pre_act_local: torch.Tensor,
    k: int,
    rank: int,
    latents_per_rank: int,
    group=None,
) -> GlobalTopK:
    """Top-k over latents sharded across ranks.

    Args:
        pre_act_local: [batch, latents_per_rank] local pre-activations on this rank.
        k: number of global latents to keep per token.
        rank: this rank's index within the TP group.
        latents_per_rank: latents per shard (used to offset/de-offset indices).
        group: TP process group (None = default group).

    Returns:
        GlobalTopK (see its docstring).
    """
    _, local_dim = pre_act_local.shape
    local_k = min(k, local_dim)

    local_vals, local_idx = torch.topk(pre_act_local, local_k, dim=-1)  # [batch, local_k]
    global_idx_cand = local_idx + rank * latents_per_rank

    cand_vals = all_gather_cat(local_vals, group=group, dim=1)  # [batch, world*local_k]
    cand_gidx = all_gather_cat(global_idx_cand, group=group, dim=1)

    global_values, pos = torch.topk(cand_vals, k, dim=-1)  # [batch, k]
    global_indices = cand_gidx.gather(1, pos)  # [batch, k] global latent indices

    lo = rank * latents_per_rank
    owned_mask = (global_indices >= lo) & (global_indices < lo + latents_per_rank)
    local_indices = torch.where(owned_mask, global_indices - lo, torch.zeros_like(global_indices))

    return GlobalTopK(global_values, global_indices, local_indices, owned_mask)


def dense_topk_reference(pre_act_full: torch.Tensor, k: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """Single-tensor oracle: top-k over the full (unsharded) latent dim."""
    vals, idx = torch.topk(pre_act_full, k, dim=-1)
    return vals, idx
