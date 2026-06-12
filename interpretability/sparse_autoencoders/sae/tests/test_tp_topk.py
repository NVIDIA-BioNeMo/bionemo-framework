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

"""Parity tests: ShardedTopKSAE == dense TopKSAE (CPU / gloo).

Shard a dense TopKSAE's weights across ranks and assert the sharded forward
reconstruction and all parameter gradients match the dense model.
"""

import pytest
import torch
import torch.nn.functional as F
from sae.architectures import TopKSAE
from sae.architectures.topk_tp import ShardedTopKSAE

from ._dist_utils import run_distributed


D, N, K, B = 16, 64, 8, 32


def _build_dense(normalize):
    torch.manual_seed(0)
    dense = TopKSAE(input_dim=D, hidden_dim=N, top_k=K, normalize_input=normalize)
    torch.manual_seed(1)
    x = torch.randn(B, D)
    return dense, x


def _w_parity(rank, world, normalize):
    dense, x = _build_dense(normalize)
    sh = ShardedTopKSAE(D, N, K, rank, world, normalize_input=normalize, decoder_impl="dense")
    sh.load_shard_from_dense(dense)
    recon, _ = sh(x)
    F.mse_loss(recon, x).backward()
    return {
        "recon": recon.detach(),
        "W_enc": sh.W_enc_local.grad.detach(),
        "W_dec": sh.W_dec_local.grad.detach(),
        "lb": sh.latent_bias_local.grad.detach(),
        "pre_bias": sh.pre_bias.grad.detach(),
    }


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("normalize", [True, False])
def test_sharded_matches_dense(world, normalize):
    res = run_distributed(_w_parity, world, args=(normalize,))

    dense, x = _build_dense(normalize)
    recon_d, _ = dense(x)
    F.mse_loss(recon_d, x).backward()
    L = N // world
    tol = {"atol": 1e-5, "rtol": 1e-5}

    # Forward reconstruction (replicated; check rank 0).
    torch.testing.assert_close(res[0]["recon"], recon_d, **tol)

    # Sharded parameter grads == the corresponding dense slices.
    for r in range(world):
        torch.testing.assert_close(res[r]["W_enc"], dense.encoder.weight.grad[r * L : (r + 1) * L, :], **tol)
        torch.testing.assert_close(res[r]["W_dec"], dense.decoder.weight.grad[:, r * L : (r + 1) * L], **tol)
        torch.testing.assert_close(res[r]["lb"], dense.latent_bias.grad[r * L : (r + 1) * L], **tol)

    # Replicated pre_bias grad: sum across ranks == dense (the pre_bias/world_size trick).
    pre_bias_sum = sum(res[r]["pre_bias"] for r in range(world))
    torch.testing.assert_close(pre_bias_sum, dense.pre_bias.grad, **tol)
