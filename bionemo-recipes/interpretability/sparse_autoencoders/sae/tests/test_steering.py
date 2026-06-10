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

"""CPU tests for sae.steering: the delta-clamp hook adds exactly decode(clamped) - decode(orig)."""

import torch
from sae.architectures import TopKSAE
from sae.steering import clamp_hook, steer
from torch import nn


def _sae():
    torch.manual_seed(0)
    return TopKSAE(input_dim=8, hidden_dim=16, top_k=4, normalize_input=False)


def test_no_clamp_is_a_noop():
    """An empty clamp map leaves the activation unchanged."""
    sae, m, x = _sae(), nn.Identity(), torch.randn(5, 8)
    with steer(m, sae, {}):
        out = m(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_clamp_adds_decoder_delta():
    """Clamping a feature shifts the activation by exactly decode(clamped) - decode(orig)."""
    sae, m, x = _sae(), nn.Identity(), torch.randn(5, 8)
    with torch.no_grad():
        pre, info = sae.encode_pre_act(x.float())
        codes = torch.relu(pre)
        kv, ki = torch.topk(codes, sae.top_k, dim=-1)
        co = torch.zeros_like(codes).scatter(-1, ki, kv)
        cc = co.clone()
        cc[:, 3] = 5.0
        expected = x + (sae.decode(cc, info) - sae.decode(co, info))
    with steer(m, sae, {3: 5.0}):
        out = m(x)
    assert torch.allclose(out, expected, atol=1e-4)


def test_tuple_output_first_element_steered_rest_preserved():
    """When the module returns a tuple, only the hidden state (elem 0) is steered."""

    class M(nn.Module):
        def forward(self, x):
            return (x, "meta")

    sae, x = _sae(), torch.randn(3, 8)
    m = M()
    handle = m.register_forward_hook(clamp_hook(sae, {0: 2.0}))
    out = m(x)
    handle.remove()
    assert isinstance(out, tuple)
    assert out[1] == "meta"
    assert out[0].shape == x.shape
    assert not torch.allclose(out[0], x)  # the clamp moved it
