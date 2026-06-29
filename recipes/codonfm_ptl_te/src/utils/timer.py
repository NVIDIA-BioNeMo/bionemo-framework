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


import time

import numpy as np
from lightning.pytorch.callbacks import Callback


class StepTimingCallback(Callback):  # noqa: D101
    def __init__(self, log_every_n_steps: int = 100, mode: str = "train", warmup_steps: int = 40):  # noqa: D107
        self.mode = mode
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self.batch_times_wo_optimizer_step = []
        self.batch_times_with_optimizer_step = []
        self.step_start_time = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record batch start time."""
        if self.mode != "train":
            return
        # Record step start time if this is the first batch of a new step
        if trainer.global_step % trainer.accumulate_grad_batches == 0:
            self.step_start_time = time.perf_counter()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):  # noqa: D102
        if self.mode != "train":
            return
        """Record batch_level time from start batch to before optimizer step"""
        if self.step_start_time is not None and trainer.global_step >= self.warmup_steps:
            batch_time = time.perf_counter() - self.step_start_time
            self.batch_times_wo_optimizer_step.append(batch_time)
            if trainer.global_step % self.log_every_n_steps == 0:
                pl_module.log(
                    "timing_train/seconds_from_batch_start_to_before_optimizer_step",
                    np.mean(self.batch_times_wo_optimizer_step),
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )

                self.batch_times_wo_optimizer_step = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D102
        if self.mode != "train":
            return
        """Log batch-level timing."""
        if self.step_start_time is not None and trainer.global_step >= self.warmup_steps:
            batch_time = time.perf_counter() - self.step_start_time
            self.batch_times_with_optimizer_step.append(batch_time)

            if trainer.global_step % self.log_every_n_steps == 0:
                pl_module.log(
                    "timing_train/seconds_per_batch",
                    np.mean(self.batch_times_with_optimizer_step),
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )

                self.batch_times_with_optimizer_step = []

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):  # noqa: D102
        if self.mode != "predict":
            return
        """Record batch start time."""
        self.step_start_time = time.perf_counter()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D102
        if self.mode != "predict":
            return
        """Log batch-level timing."""
        if self.step_start_time is not None:
            batch_time = time.perf_counter() - self.step_start_time
            self.batch_times_wo_optimizer_step.append(batch_time)

            if batch_idx % self.log_every_n_steps == 0:
                print(np.mean(self.batch_times_wo_optimizer_step))

                self.batch_times_wo_optimizer_step = []
