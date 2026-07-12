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

"""CPU-only round-trip tests for INT8Linear (L0 sanity).

These require neither a GPU nor a model checkpoint, so they run in normal CI.
They exercise the quantize -> store -> dequantize path of INT8Linear on tiny
random Linear layers and assert that reconstruction is near-lossless and that
the stored buffer is genuinely smaller than BF16.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Import the module under test (recipe layout: ../src)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from compressed_linear import INT8Linear  # noqa: E402


def _reference_linear(out_features: int, in_features: int, bias: bool = True) -> nn.Linear:
    torch.manual_seed(0)
    lin = nn.Linear(in_features, out_features, bias=bias)
    # Small, realistic weight magnitudes (like a trained transformer projection).
    with torch.no_grad():
        lin.weight.mul_(0.03)
    return lin.eval()


@pytest.mark.parametrize("out_f,in_f", [(16, 256), (8, 512), (32, 129)])
def test_int8_near_lossless(out_f, in_f):
    lin = _reference_linear(out_f, in_f)
    q = INT8Linear.from_linear(lin, return_bias=False)
    x = torch.randn(4, in_f)
    y_ref = lin(x)
    y_q = q(x)
    cos = torch.cosine_similarity(y_ref.flatten(), y_q.flatten(), dim=0).item()
    assert cos > 0.999, f"INT8 cosine too low: {cos}"


def test_int8_no_systematic_bias():
    """Per-channel symmetric INT8 should not introduce a directional weight bias."""
    lin = _reference_linear(16, 256)
    w_rec = (q := INT8Linear.from_linear(lin)).weight_int8.float() * q.scale.float()
    mean_signed_err = (w_rec - lin.weight.data).mean().item()
    signal = lin.weight.data.abs().mean().item()
    assert abs(mean_signed_err) < 0.05 * signal, (
        f"INT8 has an unexpected systematic bias of {mean_signed_err:.5f} (signal {signal:.5f})"
    )


def test_int8_memory_footprint_shrinks():
    """The stored int8 weight buffer must be ~half of BF16 (plus a tiny per-row scale)."""
    lin = _reference_linear(64, 512)
    bf16_bytes = lin.weight.numel() * 2  # bf16 = 2 bytes/param
    q = INT8Linear.from_linear(lin)
    int8_bytes = q.weight_int8.numel() * q.weight_int8.element_size()
    scale_bytes = q.scale.numel() * q.scale.element_size()
    assert int8_bytes + scale_bytes < 0.55 * bf16_bytes
