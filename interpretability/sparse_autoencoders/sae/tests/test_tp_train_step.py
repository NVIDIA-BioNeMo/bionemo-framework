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

"""B1: one optimizer step on the sharded model matches the dense model (CPU / gloo).

Validates that sharded grads + the replicated-pre_bias all-reduce produce the same
parameter update as training the dense TopKSAE.
"""

import pytest
import torch
from sae.architectures import ShardedTopKSAE, TopKSAE

from ._dist_utils import run_distributed


D, N, K, B = 16, 64, 8, 32


def _build_dense(normalize):
    torch.manual_seed(0)
    dense = TopKSAE(input_dim=D, hidden_dim=N, top_k=K, normalize_input=normalize)
    torch.manual_seed(1)
    x = torch.randn(B, D)
    return dense, x


def _w_step(rank, world, normalize):
    dense, x = _build_dense(normalize)
    sh = ShardedTopKSAE(D, N, K, rank, world, normalize_input=normalize, decoder_impl="dense")
    sh.load_shard_from_dense(dense)
    opt = torch.optim.Adam(sh.parameters(), lr=1e-3)
    sh.loss(x)["total"].backward()
    sh.reduce_replicated_grads()
    opt.step()
    return {
        "W_enc": sh.W_enc_local.detach(),
        "W_dec": sh.W_dec_local.detach(),
        "lb": sh.latent_bias_local.detach(),
        "pre_bias": sh.pre_bias.detach(),
    }


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("normalize", [True, False])
def test_one_step_matches_dense(world, normalize):
    res = run_distributed(_w_step, world, args=(normalize,))

    dense, x = _build_dense(normalize)
    opt = torch.optim.Adam(dense.parameters(), lr=1e-3)
    dense.loss(x)["total"].backward()
    opt.step()

    L = N // world
    tol = {"atol": 1e-5, "rtol": 1e-5}
    for r in range(world):
        torch.testing.assert_close(res[r]["W_enc"], dense.encoder.weight[r * L : (r + 1) * L, :], **tol)
        torch.testing.assert_close(res[r]["W_dec"], dense.decoder.weight[:, r * L : (r + 1) * L], **tol)
        torch.testing.assert_close(res[r]["lb"], dense.latent_bias[r * L : (r + 1) * L], **tol)
        torch.testing.assert_close(res[r]["pre_bias"], dense.pre_bias, **tol)  # replicated
