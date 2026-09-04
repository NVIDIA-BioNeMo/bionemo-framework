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


import lightning.pytorch as pl
import torch


class GradientNormLogger(pl.Callback):
    """Callback that periodically logs the average gradient norm.

    Logs the running average of parameter gradient norms every
    `log_every_n_steps` training steps using the module's logger.

    Args:
        log_every_n_steps: Frequency (in steps) at which to log gradient norms.
    """

    def __init__(self, log_every_n_steps=10):  # noqa: D107
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        """Compute and log the average gradient norm after backward pass.

        Args:
            trainer: The running `pl.Trainer` instance.
            pl_module: The Lightning module containing parameters and logger.
        """
        if (trainer.global_step) % self.log_every_n_steps == 0:
            grad_norms = []
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    grad_norms.append((name, grad_norm))

            if grad_norms:
                avg_grad_norm = sum(gn[1] for gn in grad_norms) / len(grad_norms)
                pl_module.log(
                    "avg_grad_norm", avg_grad_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True
                )
