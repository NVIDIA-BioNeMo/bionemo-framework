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

"""B0: sharded checkpoint save -> merge round-trips to the dense model (single process)."""

import pytest
import torch
from sae.architectures import ShardedTopKSAE, TopKSAE
from sae.parallel import load_and_merge, save_sharded


D, N, K, B = 16, 64, 8, 32


@pytest.mark.parametrize("world", [2, 4])
def test_shard_save_merge_roundtrip(tmp_path, world):
    torch.manual_seed(0)
    dense = TopKSAE(input_dim=D, hidden_dim=N, top_k=K, normalize_input=True, auxk=16)
    out = str(tmp_path)

    for r in range(world):
        sh = ShardedTopKSAE(D, N, K, r, world, normalize_input=True, auxk=16)
        sh.load_shard_from_dense(dense)
        save_sharded(sh, out, rank=r)

    merged = load_and_merge(out)
    torch.testing.assert_close(merged.encoder.weight, dense.encoder.weight)
    torch.testing.assert_close(merged.decoder.weight, dense.decoder.weight)
    torch.testing.assert_close(merged.latent_bias, dense.latent_bias)
    torch.testing.assert_close(merged.pre_bias, dense.pre_bias)

    # Merged dense model reproduces the original dense outputs.
    torch.manual_seed(1)
    x = torch.randn(B, D)
    rd, _ = dense(x)
    rm, _ = merged(x)
    torch.testing.assert_close(rm, rd, atol=1e-6, rtol=1e-6)
