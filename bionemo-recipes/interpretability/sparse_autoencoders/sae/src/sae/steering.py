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

"""Causal feature steering for SAEs — clamp features in code-space, inject only the delta.

A forward hook on the layer the SAE was trained on: it re-encodes the layer output through
the SAE, overrides chosen features in code-space, decodes, and adds the **delta** back to the
activation. Because we add ``decode(clamped) - decode(original)`` (not the recon itself), the
SAE's reconstruction error cancels and only the clamped feature's decoder contribution moves
the activation. Model-agnostic: needs only the SAE (``encode_pre_act`` / ``decode`` / ``top_k``)
and the module to hook. Measure the effect (e.g. ΔP of a target token) by running the model
with vs. without the hook.
"""

from contextlib import contextmanager
from typing import Dict

import torch


def clamp_hook(sae, clamps: Dict[int, float]):
    """Build a forward hook that clamps ``{feature_idx: value}`` via the delta method.

    The hook adds ``decode(clamped_codes) - decode(original_codes)`` to the hooked module's
    output, so the SAE reconstruction error cancels. ``value=0`` ablates a feature; a negative
    value reverses its decoder direction. Works whether the module returns a tensor or a tuple
    whose first element is the hidden state.

    Args:
        sae: A trained SAE exposing ``encode_pre_act(x) -> (pre_act, info)``, ``decode(codes, info)``,
            and ``top_k``.
        clamps: Map of feature index -> absolute code value to force at every position.

    Returns:
        A ``register_forward_hook``-compatible ``hook(module, inputs, output)``.
    """
    items = [(int(f), float(v)) for f, v in clamps.items()]

    def hook(module, inputs, output):
        h, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        dtype, shape = h.dtype, h.shape
        h_flat = h.reshape(-1, h.shape[-1]).float()
        with torch.no_grad():
            pre_act, info = sae.encode_pre_act(h_flat)
            codes = torch.relu(pre_act)
            kvals, kidx = torch.topk(codes, sae.top_k, dim=-1)
            codes_orig = torch.zeros_like(codes).scatter(-1, kidx, kvals)
            codes_clamped = codes_orig.clone()
            for f, v in items:
                codes_clamped[:, f] = v
            delta = sae.decode(codes_clamped, info) - sae.decode(codes_orig, info)
            h_out = (h_flat + delta).to(dtype).reshape(shape)
        return (h_out, *rest) if rest is not None else h_out

    return hook


@contextmanager
def steer(module, sae, clamps: Dict[int, float]):
    """Register the clamp hook on ``module`` for the duration of the ``with`` block, then remove it."""
    handle = module.register_forward_hook(clamp_hook(sae, clamps))
    try:
        yield
    finally:
        handle.remove()
