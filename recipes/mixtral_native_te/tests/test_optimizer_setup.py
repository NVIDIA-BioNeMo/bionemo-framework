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

from types import MethodType

import torch
from optimizer_setup import HighPrecisionInitValues


def _attach_high_precision_value(param: torch.nn.Parameter, value: torch.Tensor) -> None:
    param._high_precision_init_val = value

    def get(self):
        return getattr(self, "_high_precision_init_val", None)

    def clear(self):
        if hasattr(self, "_high_precision_init_val"):
            del self._high_precision_init_val

    param.get_high_precision_init_val = MethodType(get, param)
    param.clear_high_precision_init_val = MethodType(clear, param)


class _Optimizer:
    def __init__(self):
        self.initialized = []
        self.master_values = {}

    def initialize_state(self, param, store_param_remainders):
        self.initialized.append((param, store_param_remainders))

    def set_scaled_state(self, param, state_name, value):
        assert state_name == "master_param"
        self.master_values[param] = value.clone()


def test_high_precision_values_survive_parameter_replacement():
    model = torch.nn.Linear(2, 2, bias=False)
    expected = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
    _attach_high_precision_value(model.weight, expected)

    values = HighPrecisionInitValues.capture(model)
    assert model.weight.get_high_precision_init_val() is None

    # Model the object replacement performed by FSDP2 wrapping.
    model.weight = torch.nn.Parameter(torch.zeros_like(model.weight))
    optimizer = _Optimizer()
    values.initialize_master_weights(optimizer, model, torch.device("cpu"))

    torch.testing.assert_close(optimizer.master_values[model.weight], expected.float())
    assert optimizer.initialized == [(model.weight, False)]


def test_high_precision_values_created_after_capture_are_used():
    model = torch.nn.Linear(2, 2, bias=False)
    values = HighPrecisionInitValues.capture(model)
    expected = torch.full((2, 2), 3, dtype=torch.bfloat16)
    _attach_high_precision_value(model.weight, expected)
    optimizer = _Optimizer()

    values.initialize_master_weights(optimizer, model, torch.device("cpu"))

    torch.testing.assert_close(optimizer.master_values[model.weight], expected.float())
    assert model.weight.get_high_precision_init_val() is None
