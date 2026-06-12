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

"""Parity tests for ShardedTopKSAE.loss() incl. metrics, dead_pct, and auxk (CPU/gloo).

dead_tokens_threshold=0 forces dead latents after the first step so dead_pct and the
auxk path are actually exercised. The sharded loss dict must match the dense one.
"""

import pytest
import torch
from sae.architectures import TopKSAE
from sae.architectures.topk_tp import ShardedTopKSAE

from ._dist_utils import run_distributed


D, N, K, B = 16, 64, 8, 32


def _build_dense(normalize, auxk):
    torch.manual_seed(0)
    dense = TopKSAE(
        input_dim=D,
        hidden_dim=N,
        top_k=K,
        normalize_input=normalize,
        auxk=auxk,
        dead_tokens_threshold=0,
    )
    torch.manual_seed(1)
    x = torch.randn(B, D)
    return dense, x


def _w_loss(rank, world, normalize, auxk):
    dense, x = _build_dense(normalize, auxk)
    sh = ShardedTopKSAE(
        D,
        N,
        K,
        rank,
        world,
        normalize_input=normalize,
        auxk=auxk,
        dead_tokens_threshold=0,
        decoder_impl="dense",
    )
    sh.load_shard_from_dense(dense)
    sh.loss(x)  # prime dead-latent stats
    out = sh.loss(x)
    return {k: v.detach() for k, v in out.items()}


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("normalize,auxk", [(True, None), (False, None), (True, 16)])
def test_sharded_loss_matches_dense(world, normalize, auxk):
    res = run_distributed(_w_loss, world, args=(normalize, auxk))

    dense, x = _build_dense(normalize, auxk)
    dense.loss(x)  # prime
    out_d = dense.loss(x)
    tol = {"atol": 1e-5, "rtol": 1e-5}

    keys = ["total", "fvu", "sparsity", "mse", "variance_explained", "dead_pct"]
    if auxk is not None:
        keys.append("aux")
    for key in keys:
        torch.testing.assert_close(res[0][key], out_d[key], msg=lambda m, k=key: f"{k}: {m}", **tol)
