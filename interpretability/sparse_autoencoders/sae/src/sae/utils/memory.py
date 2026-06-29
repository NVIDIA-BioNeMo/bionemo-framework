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

from __future__ import annotations


def _fmt_bytes(b: int) -> str:
    # Decimal units to match your "3.36GB" math (1 GB = 1e9 bytes)
    if b < 1_000:
        return f"{b}B"
    if b < 1_000_000:
        return f"{b / 1e3:.2f}KB"
    if b < 1_000_000_000:
        return f"{b / 1e6:.2f}MB"
    if b < 1_000_000_000_000:
        return f"{b / 1e9:.2f}GB"
    return f"{b / 1e12:.2f}TB"


def sae_weight_memory(d: int, expansion: float, precision_bytes: int, bias: bool = True) -> str:
    """Compute weight memory for an SAE.

    W_enc: (n, d)
    W_dec: (d, n)
    (optional) biases: (n + d)
    n = int(expansion * d)
    """
    n = int(expansion * d)
    b = 2 * n * d * precision_bytes
    if bias:
        b += (n + d) * precision_bytes
    return _fmt_bytes(b)


def sae_forward_memory(d: int, expansion: float, precision_bytes: int, batch_size: int, k: int) -> dict:
    """Forward activation memory you typically retain for backward.

    Components:
      x:                (B, d)
      pre_activations:  (B, n)     (encoder linear output before nonlinearity/topk)
      z_topk:           (B, k) values + (B, k) indices (int32)
      recon:            (B, d)

    Returns dict with total + per-component.
    """
    n = int(expansion * d)
    B = batch_size
    P = precision_bytes

    comps = {
        "x (B,d)": B * d * P,
        "pre_activations (B,n)": B * n * P,
        "z_values (B,k)": B * k * P,
        "z_indices (B,k, int32)": B * k * 4,
        "reconstruction (B,d)": B * d * P,
    }
    total = sum(comps.values())
    return {
        "total_bytes": total,
        "total": _fmt_bytes(total),
        "components": {name: _fmt_bytes(sz) for name, sz in comps.items()},
        "n": n,
    }


def sae_backward_memory(
    d: int, expansion: float, precision_bytes: int, batch_size: int, optimizer: str = "adam"
) -> dict:
    """Backward/grad/optimizer memory estimate.

    Includes:
      - activation grads: grad_x (B,d) and grad_pre (B,n)  [dominant]
      - parameter grads: same size as params
      - optimizer states:
          adam: 2 * params  (m and v)
          sgd:  0           (no momentum assumed)
          none: 0

    Assumes biases are included in params.
    """
    n = int(expansion * d)
    B = batch_size
    P = precision_bytes

    # Params (including biases)
    param_bytes = (2 * n * d + (n + d)) * P

    comps = {
        "grad_x (B,d)": B * d * P,
        "grad_pre_activations (B,n)": B * n * P,
        "param_grads": param_bytes,
    }

    opt = optimizer.lower()
    if opt == "adam":
        comps["optimizer_states (adam m+v)"] = 2 * param_bytes
    elif opt in ("sgd", "none"):
        comps["optimizer_states"] = 0
    else:
        raise ValueError('optimizer must be "adam", "sgd", or "none"')

    total = sum(comps.values())
    return {
        "total_bytes": total,
        "total": _fmt_bytes(total),
        "components": {name: _fmt_bytes(sz) for name, sz in comps.items()},
        "n": n,
        "param_bytes": param_bytes,
    }


def sae_total_memory(
    d: int, expansion: float, precision_bytes: int, batch_size: int, k: int, optimizer: str = "adam"
) -> str:
    """Total = weights + forward + backward/optimizer. Prints breakdown."""
    n = int(expansion * d)
    P = precision_bytes

    # weights (include biases)
    weights_bytes = (2 * n * d + (n + d)) * P
    fwd = sae_forward_memory(d, expansion, precision_bytes, batch_size, k)
    bwd = sae_backward_memory(d, expansion, precision_bytes, batch_size, optimizer)

    total_bytes = weights_bytes + fwd["total_bytes"] + bwd["total_bytes"]

    lines = []
    lines.append(f"SAE memory estimate (n={n:,}, d={d:,}, B={batch_size:,}, k={k}, P={P} bytes, opt={optimizer}):")
    lines.append(f"Weights: {_fmt_bytes(weights_bytes)}")
    lines.append(f"Forward: {fwd['total']}")
    for name, mem in fwd["components"].items():
        lines.append(f"  - {name}: {mem}")
    lines.append(f"Backward+opt: {bwd['total']}")
    for name, mem in bwd["components"].items():
        lines.append(f"  - {name}: {mem}")
    lines.append(f"TOTAL: {_fmt_bytes(total_bytes)}")
    return "\n".join(lines)
