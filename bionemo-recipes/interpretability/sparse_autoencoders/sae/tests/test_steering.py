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

"""CPU tests for sae.steering: the delta-clamp adds exactly decode(clamped) - decode(orig)."""

import torch
from sae.architectures import TopKSAE
from sae.steering import clamp_hook, steer
from torch import nn


def _sae():
    torch.manual_seed(0)
    return TopKSAE(input_dim=8, hidden_dim=16, top_k=4, normalize_input=False)


def test_delta_clamp_is_exact_and_cancels_recon():
    """No-op clamp leaves the activation unchanged (recon error cancels); a real clamp shifts
    it by exactly decode(clamped) - decode(orig) — the two halves of the delta-clamp contract."""
    sae, m, x = _sae(), nn.Identity(), torch.randn(5, 8)

    # No-op: decode(orig) != x, but the added delta is 0, so the output is unchanged.
    with steer(m, sae, {}):
        assert torch.allclose(m(x), x, atol=1e-5)

    # Real clamp: output == x + (decode(clamped) - decode(orig)), recon error cancelled.
    with torch.no_grad():
        pre, info = sae.encode_pre_act(x.float())
        codes = torch.relu(pre)
        kv, ki = torch.topk(codes, sae.top_k, dim=-1)
        co = torch.zeros_like(codes).scatter(-1, ki, kv)
        cc = co.clone()
        cc[:, 3] = 5.0
        expected = x + (sae.decode(cc, info) - sae.decode(co, info))
    with steer(m, sae, {3: 5.0}):
        assert torch.allclose(m(x), expected, atol=1e-4)


def test_tuple_output_steers_only_hidden_state():
    """When the hooked module returns a tuple, only element 0 is steered; the rest passes through."""

    class M(nn.Module):
        def forward(self, x):
            return (x, "meta")

    sae, x = _sae(), torch.randn(3, 8)
    m = M()
    handle = m.register_forward_hook(clamp_hook(sae, {0: 2.0}))
    out = m(x)
    handle.remove()
    assert isinstance(out, tuple) and out[1] == "meta"
    assert out[0].shape == x.shape and not torch.allclose(out[0], x)  # clamp moved it


def test_decode_only_skips_prefill():
    """decode_only steers single-token decode steps ([1,B,H]) but leaves multi-token prefill alone."""
    sae, m = _sae(), nn.Identity()
    prefill = torch.randn(5, 2, 8)  # [S=5, B, H] — prompt prefill, must pass through
    decode = torch.randn(1, 2, 8)  # [S=1, B, H] — a single new token, must be steered
    handle = m.register_forward_hook(clamp_hook(sae, {3: 5.0}, decode_only=True))
    out_prefill, out_decode = m(prefill), m(decode)
    handle.remove()
    assert torch.allclose(out_prefill, prefill, atol=1e-5)  # prefill untouched
    assert not torch.allclose(out_decode, decode)  # decode step steered
