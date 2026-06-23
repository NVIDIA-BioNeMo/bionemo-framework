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
the activation. Model-agnostic: needs only the SAE (``encode`` / ``decode``) and the module to
hook. Measure the effect (e.g. ΔP of a target token) by running the model with vs. without the
hook.
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator

import torch


def clamp_hook(sae: Any, clamps: Dict[int, float], decode_only: bool = False) -> Callable:
    """Build a forward hook that clamps ``{feature_idx: value}`` via the delta method.

    The hook adds ``decode(clamped_codes) - decode(original_codes)`` to the hooked module's
    output, so the SAE reconstruction error cancels. ``value=0`` ablates a feature; a negative
    value reverses its decoder direction. Works whether the module returns a tensor or a tuple
    whose first element is the hidden state.

    Args:
        sae: A trained TopK SAE exposing ``encode(x) -> codes``, ``encode_pre_act(x) -> (_, info)``,
            and ``decode(codes, info)``.
        clamps: Map of feature index -> absolute code value to force at every position.
        decode_only: If True, steer only autoregressive *decode* steps and leave the prompt
            prefill untouched (continuation-only steering). Assumes a ``(sequence, batch, hidden)``
            layout — the convention for Evo2/megatron decoder layers — and applies the clamp only
            when the sequence dimension is 1 (a single new token).

    Returns:
        A ``register_forward_hook``-compatible ``hook(module, inputs, output)``.
    """
    items = [(int(f), float(v)) for f, v in clamps.items()]

    def hook(module, inputs, output):
        h, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        if decode_only and h.shape[0] != 1:  # prefill (seq dim > 1) — leave untouched
            return output
        dtype, shape = h.dtype, h.shape
        h_flat = h.reshape(-1, h.shape[-1]).float()
        with torch.no_grad():
            # Encode through the SAE itself (canonical encode) instead of re-deriving relu+topk on
            # sae.top_k: no hardcoded sparsity, and it can never drift from the model's true
            # encoding. encode_pre_act supplies the normalization info decode needs to map back to
            # the original activation scale.
            _, info = sae.encode_pre_act(h_flat)
            codes_orig = sae.encode(h_flat)
            codes_clamped = codes_orig.clone()
            n_feat = codes_orig.shape[-1]
            for f, v in items:
                if not 0 <= f < n_feat:
                    raise ValueError(f"clamp feature {f} out of range [0, {n_feat})")
                codes_clamped[:, f] = v
            delta = sae.decode(codes_clamped, info) - sae.decode(codes_orig, info)
            h_out = (h_flat + delta).to(dtype).reshape(shape)
        return (h_out, *rest) if rest is not None else h_out

    return hook


@contextmanager
def steer(module: "torch.nn.Module", sae: Any, clamps: Dict[int, float], decode_only: bool = False) -> Iterator[None]:
    """Register the clamp hook on ``module`` for the duration of the ``with`` block, then remove it."""
    handle = module.register_forward_hook(clamp_hook(sae, clamps, decode_only=decode_only))
    try:
        yield
    finally:
        handle.remove()
