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

import time

import torch
from lightning.pytorch.callbacks import Callback


class IntervalStepTimingCallback(Callback):
    """Logs mean wall-clock time per optimizer step over a fixed logging interval.

    Mirrors the semantics of `train/step_time` in the native_te recipe's `PerfLogger`:
    samples `time.perf_counter()` only at log boundaries and divides by
    `log_every_n_steps`, yielding the average optimizer-step wall time over the
    last interval rather than a per-step measurement.
    """

    def __init__(self, log_every_n_steps: int = 10):  # noqa: D107
        self.log_every_n_steps = log_every_n_steps
        self.previous_log_time: float | None = None

    def on_train_start(self, trainer, pl_module):  # noqa: D102
        self.previous_log_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D102
        if (batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        step = trainer.global_step
        if step == 0 or step % self.log_every_n_steps != 0:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        step_time = (now - self.previous_log_time) / self.log_every_n_steps
        self.previous_log_time = now

        pl_module.log(
            "timing_train/step_time",
            step_time,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
