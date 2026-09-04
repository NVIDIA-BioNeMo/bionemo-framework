# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from unittest.mock import MagicMock

import pytest
import torch

from src.utils.grad_norm_callback import GradientNormLogger


class MockLightningModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)
        self.log = MagicMock()

    def forward(self, x):
        return self.layer(x)


@pytest.fixture
def pl_module():
    module = MockLightningModule()
    # add some gradients
    module.layer.weight.grad = torch.ones_like(module.layer.weight)
    module.layer.bias.grad = torch.ones_like(module.layer.bias) * 0.5
    return module


@pytest.fixture
def trainer():
    trainer = MagicMock()
    return trainer


class TestGradientNormLogger:
    def test_on_after_backward_log_step(self, pl_module, trainer):
        log_every_n_steps = 10
        callback = GradientNormLogger(log_every_n_steps=log_every_n_steps)

        trainer.global_step = 20  # a multiple of log_every_n_steps

        callback.on_after_backward(trainer, pl_module)

        # Calculate expected avg_grad_norm
        weight_norm = torch.norm(pl_module.layer.weight.grad).item()
        bias_norm = torch.norm(pl_module.layer.bias.grad).item()
        expected_avg_grad_norm = (weight_norm + bias_norm) / 2

        pl_module.log.assert_called_once_with(
            "avg_grad_norm", expected_avg_grad_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

    def test_on_after_backward_not_log_step(self, pl_module, trainer):
        log_every_n_steps = 10
        callback = GradientNormLogger(log_every_n_steps=log_every_n_steps)

        trainer.global_step = 19  # not a multiple of log_every_n_steps

        callback.on_after_backward(trainer, pl_module)

        pl_module.log.assert_not_called()

    def test_on_after_backward_no_grads(self, pl_module, trainer):
        log_every_n_steps = 10
        callback = GradientNormLogger(log_every_n_steps=log_every_n_steps)

        trainer.global_step = 20

        # Remove grads
        pl_module.layer.weight.grad = None
        pl_module.layer.bias.grad = None

        callback.on_after_backward(trainer, pl_module)

        pl_module.log.assert_not_called()

    def test_on_after_backward_some_grads_none(self, pl_module, trainer):
        log_every_n_steps = 10
        callback = GradientNormLogger(log_every_n_steps=log_every_n_steps)

        trainer.global_step = 20

        # Remove one of the grads
        pl_module.layer.bias.grad = None

        callback.on_after_backward(trainer, pl_module)

        expected_avg_grad_norm = torch.norm(pl_module.layer.weight.grad).item()

        pl_module.log.assert_called_once_with(
            "avg_grad_norm", expected_avg_grad_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )
