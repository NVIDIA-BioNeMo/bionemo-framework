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

"""Load Hugging Face Mixtral and optionally enable dense MXFP8 training GEMMs."""

from __future__ import annotations

import torch
from torch import nn


class MXFP8Linear(nn.Linear):
    """An ``nn.Linear`` whose forward and backward GEMMs use TorchAO MXFP8."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply a dynamically quantized MXFP8 linear operation."""
        from torchao.prototype.moe_training.mxfp8_linear import _to_mxfp8_then_scaled_mm
        from torchao.prototype.mx_formats.config import ScaleCalculationMode
        from torchao.quantization.quantize_.common import KernelPreference

        output = _to_mxfp8_then_scaled_mm(
            input,
            self.weight,
            kernel_preference=KernelPreference.AUTO,
            scale_calculation_mode=ScaleCalculationMode.RCEIL,
            wgrad_with_hp=False,
        )
        if self.bias is not None:
            output = output + self.bias
        return output


def enable_dense_mxfp8(model: nn.Module) -> list[str]:
    """Convert decoder dense linears, leaving the BF16 LM head untouched."""
    converted = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name != "lm_head":
            module.__class__ = MXFP8Linear
            converted.append(name)
    return converted


def from_pretrained(*args, mxfp8_dense: bool = False, **kwargs):
    """Load through AutoModel, then opt into its missing dense MXFP8 equivalent."""
    from nemo_automodel import NeMoAutoModelForCausalLM

    model = NeMoAutoModelForCausalLM.from_pretrained(*args, **kwargs)
    if mxfp8_dense:
        converted = enable_dense_mxfp8(model)
        if not converted:
            raise RuntimeError("MXFP8 was requested, but the model had no eligible dense linear layers")
    return model
