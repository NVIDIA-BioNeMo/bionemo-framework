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

from src.data.metadata import MetadataFields


class ThroughputLogger(Callback):
    """Logs unpadded tokens per second per GPU to wandb via PyTorch Lightning.

    Works with both THD (packed/variable-length) and BSHD (padded) batch formats.
    In THD mode the input_ids tensor is already stripped of padding, so its numel()
    gives the true token count. In BSHD mode the cumulative-sequence-length metadata
    is absent and the attention mask is used instead.

    Args:
        log_every_n_steps: How often (in global steps) to compute and log the metric.
        warmup_steps: Number of initial steps to skip before collecting measurements.
    """

    def __init__(self, log_every_n_steps: int = 100, warmup_steps: int = 40):  # noqa: D107
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self._step_start_time: float | None = None
        self._tokens_per_second: list[float] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record the wall-clock time at the beginning of each step."""
        self._step_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Compute and periodically log unpadded tokens/s/GPU."""
        if self._step_start_time is None or trainer.global_step < self.warmup_steps:
            return

        step_time = time.perf_counter() - self._step_start_time

        if MetadataFields.CU_SEQ_LENS_Q in batch:
            num_unpadded_tokens = int(batch[MetadataFields.CU_SEQ_LENS_Q][-1].item())
        elif MetadataFields.ATTENTION_MASK in batch:
            num_unpadded_tokens = int(batch[MetadataFields.ATTENTION_MASK].sum().item())
        else:
            num_unpadded_tokens = batch[MetadataFields.INPUT_IDS].numel()

        self._tokens_per_second.append(num_unpadded_tokens / step_time)

        if trainer.global_step % self.log_every_n_steps == 0:
            pl_module.log(
                "throughput/unpadded_tokens_per_second_per_gpu",
                np.mean(self._tokens_per_second),
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self._tokens_per_second = []
