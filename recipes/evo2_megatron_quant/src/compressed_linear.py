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

"""Compressed Linear modules: real low-precision weight storage.

Provides a drop-in replacement for nn.Linear that stores weights in INT8,
achieving real memory savings (~2x on the compressed layers) while keeping
inference quality near-lossless via per-channel (per-output-row) scaling.

    INT8 -> torch.int8, dequant-on-the-fly to the activation dtype
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class INT8Linear(nn.Module):
    """Linear layer with weights stored as int8.

    Dequantizes to the activation dtype on the fly for the matmul.
    Per-channel (per-output-row) symmetric quantization preserves accuracy.
    """

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor,
                 bias: torch.Tensor = None, out_features: int = 0, in_features: int = 0,
                 return_bias: bool = True):
        super().__init__()
        self.out_features = out_features or weight_int8.shape[0]
        self.in_features = in_features or weight_int8.shape[1]
        self.return_bias = return_bias
        self.register_buffer("weight_int8", weight_int8)  # torch.int8
        self.register_buffer("scale", scale)               # per-channel: (out_features, 1)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: int8 -> activation dtype (per-output-row scale broadcasts over columns)
        w = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        out = F.linear(x, w, self.bias)
        return (out, None) if self.return_bias else out

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, dtype=int8")

    @staticmethod
    def from_linear(linear: nn.Linear, scale: torch.Tensor = None, per_channel: bool = True,
                    return_bias: bool = True):
        """Convert nn.Linear to INT8Linear. Quantizes on CPU to save GPU memory."""
        device = linear.weight.data.device
        w = linear.weight.data.cpu().float()
        if scale is None:
            if per_channel:
                amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            else:
                amax = w.abs().amax().unsqueeze(0).unsqueeze(0).clamp(min=1e-12)
            scale = amax / 127.0
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8).to(device)
        scale = scale.to(device)
        bias = linear.bias.data.to(device) if linear.bias is not None else None
        return INT8Linear(w_int8, scale, bias, linear.out_features, linear.in_features,
                          return_bias=return_bias)


# === Factory ===

PRECISION_MAP = {
    "int8": INT8Linear,
}

# Map the ModelOpt INT8 method names to the real-weight-compression precision.
METHOD_TO_PRECISION = {
    "INT8_DEFAULT_CFG": "int8",
    "INT8_SMOOTHQUANT_CFG": "int8",
    "INT8_KV_CFG": "int8",
    "MXINT8_DEFAULT_CFG": "int8",
    "MX_INT8_CFG": "int8",
}


def compress_linear(linear: nn.Linear, precision: str = "int8", **kwargs):
    """Convert nn.Linear to a compressed format.

    Args:
        linear: Original linear layer.
        precision: Currently only "int8" is supported.
        **kwargs: Extra args passed to the compressed linear constructor.
    """
    cls = PRECISION_MAP.get(precision)
    if cls is None:
        raise ValueError(f"Unknown precision: {precision}. Choose from: {list(PRECISION_MAP.keys())}")
    return cls.from_linear(linear, **kwargs)
