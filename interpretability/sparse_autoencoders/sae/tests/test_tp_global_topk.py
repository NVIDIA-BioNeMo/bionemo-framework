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

"""Tests for sharded global top-k (CPU / gloo).

The sharded global top-k must exactly match a single-process torch.topk over the
full (concatenated) latent dimension, and must partition the selections so each is
owned by exactly one rank.
"""

import pytest
import torch
from sae.parallel import global_topk

from ._dist_utils import run_distributed


B, N, K, SEED = 16, 256, 8, 0


def _full_pre_act():
    # Distinct continuous values (randn) -> no ties, so indices match exactly.
    torch.manual_seed(SEED)
    return torch.randn(B, N)


def _w_global_topk(rank, world):
    full = _full_pre_act()
    latents_per_rank = N // world
    local = full[:, rank * latents_per_rank : (rank + 1) * latents_per_rank].contiguous()
    res = global_topk(local, K, rank, latents_per_rank)
    return (res.global_values, res.global_indices, res.owned_mask, res.local_indices)


@pytest.mark.parametrize("world", [2, 4])
def test_global_topk_matches_dense(world):
    results = run_distributed(_w_global_topk, world)
    full = _full_pre_act()
    dense_vals, dense_idx = torch.topk(full, K, dim=-1)
    latents_per_rank = N // world

    gv0, gidx0, _, _ = results[0]
    torch.testing.assert_close(gv0, dense_vals)
    assert torch.equal(gidx0, dense_idx), (gidx0, dense_idx)

    # Global selection is replicated identically on every rank.
    for r in range(1, world):
        assert torch.equal(results[r][1], gidx0)

    # Ownership partitions the selections: exactly one rank owns each (b, j).
    owned_stack = torch.stack([results[r][2].int() for r in range(world)], dim=0)  # [world, B, K]
    assert torch.equal(owned_stack.sum(0), torch.ones(B, K, dtype=torch.int32))

    # Owned local indices map back to the global indices.
    for r in range(world):
        _, gidx, owned, lidx = results[r]
        recon_global = lidx + r * latents_per_rank
        assert torch.equal(recon_global[owned], gidx[owned])
