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

"""Sharded init_pre_bias_from_data matches dense (single process; pre_bias is replicated)."""

import pytest
import torch
from sae.architectures import ShardedTopKSAE, TopKSAE


@pytest.mark.parametrize("normalize", [True, False])
def test_init_pre_bias_matches_dense(normalize):
    torch.manual_seed(0)
    data = torch.randn(500, 16)

    dense = TopKSAE(input_dim=16, hidden_dim=64, top_k=8, normalize_input=normalize)
    dense.init_pre_bias_from_data(data)

    sh = ShardedTopKSAE(16, 64, 8, rank=0, world_size=2, normalize_input=normalize)
    sh.init_pre_bias_from_data(data)

    torch.testing.assert_close(sh.pre_bias, dense.pre_bias)
