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

import torch
from torch import nn

from .model_factory import MXFP8Linear, enable_dense_mxfp8


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.q_proj = nn.Linear(32, 32, bias=False)
        self.model.o_proj = nn.Linear(32, 32, bias=False)
        self.lm_head = nn.Linear(32, 64, bias=False)


def test_enable_dense_mxfp8_excludes_lm_head():
    model = TinyModel()

    converted = enable_dense_mxfp8(model)

    assert converted == ["model.q_proj", "model.o_proj"]
    assert isinstance(model.model.q_proj, MXFP8Linear)
    assert isinstance(model.model.o_proj, MXFP8Linear)
    assert type(model.lm_head) is nn.Linear
    assert model.model.q_proj.weight.shape == (32, 32)


def test_mxfp8_linear_bias_path(monkeypatch):
    module = nn.Linear(32, 32, bias=True)
    module.__class__ = MXFP8Linear
    call = {}

    def fake_mxfp8_linear(
        input_hp,
        weight_hp,
        kernel_preference,
        scale_calculation_mode,
        wgrad_with_hp,
    ):
        call.update(
            kernel_preference=kernel_preference,
            scale_calculation_mode=scale_calculation_mode,
            wgrad_with_hp=wgrad_with_hp,
        )
        return input_hp @ weight_hp.t()

    monkeypatch.setattr(
        "torchao.prototype.moe_training.mxfp8_linear._to_mxfp8_then_scaled_mm",
        fake_mxfp8_linear,
    )
    inputs = torch.randn(2, 32)

    torch.testing.assert_close(module(inputs), nn.functional.linear(inputs, module.weight, module.bias))

    from torchao.prototype.mx_formats.config import ScaleCalculationMode
    from torchao.quantization.quantize_.common import KernelPreference

    assert call == {
        "kernel_preference": KernelPreference.AUTO,
        "scale_calculation_mode": ScaleCalculationMode.RCEIL,
        "wgrad_with_hp": False,
    }
