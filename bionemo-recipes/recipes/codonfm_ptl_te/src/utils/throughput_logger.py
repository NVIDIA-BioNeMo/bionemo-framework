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


import logging
import time

import numpy as np
from lightning.pytorch.callbacks import Callback

from src.data.metadata import MetadataFields


log = logging.getLogger(__name__)


class ThroughputLogger(Callback):
    """Logs unpadded tokens per second per GPU to wandb via PyTorch Lightning.

    Works with both THD (packed/variable-length) and BSHD (padded) batch formats.
    In THD mode the input_ids tensor is already stripped of padding, so its numel()
    gives the true token count. In BSHD mode the cumulative-sequence-length metadata
    is absent and the attention mask is used instead.

    Tracks training, validation, and prediction throughput independently.

    Args:
        log_every_n_steps: How often (in global steps) to compute and log the metric.
        warmup_steps: Number of initial steps to skip before collecting measurements.
    """

    def __init__(self, log_every_n_steps: int = 100, warmup_steps: int = 40):  # noqa: D107
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self._train_step_start_time: float | None = None
        self._train_tokens_per_second: list[float] = []
        self._val_step_start_time: float | None = None
        self._val_tokens_per_second: list[float] = []
        self._predict_step_start_time: float | None = None
        self._predict_tokens_per_second: list[float] = []

    def _count_unpadded_tokens(self, batch) -> int:
        """Return the number of unpadded tokens in a batch."""
        if MetadataFields.CU_SEQ_LENS_Q in batch:
            return int(batch[MetadataFields.CU_SEQ_LENS_Q][-1].item())
        elif MetadataFields.ATTENTION_MASK in batch:
            return int(batch[MetadataFields.ATTENTION_MASK].sum().item())
        return batch[MetadataFields.INPUT_IDS].numel()

    # -- Training --

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record the wall-clock time at the beginning of each training step."""
        self._train_step_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Compute and periodically log training unpadded tokens/s/GPU."""
        if self._train_step_start_time is None or trainer.global_step < self.warmup_steps:
            return

        step_time = time.perf_counter() - self._train_step_start_time
        self._train_tokens_per_second.append(self._count_unpadded_tokens(batch) / step_time)

        if trainer.global_step % self.log_every_n_steps == 0:
            pl_module.log(
                "throughput/train_unpadded_tokens_per_second_per_gpu",
                np.mean(self._train_tokens_per_second),
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self._train_tokens_per_second = []

    # -- Validation --

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Record the wall-clock time at the beginning of each validation step."""
        self._val_step_start_time = time.perf_counter()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Compute and log validation unpadded tokens/s/GPU at the end of each val epoch."""
        if self._val_step_start_time is None:
            return

        step_time = time.perf_counter() - self._val_step_start_time
        self._val_tokens_per_second.append(self._count_unpadded_tokens(batch) / step_time)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log aggregated validation throughput at the end of each validation epoch."""
        if not self._val_tokens_per_second:
            return

        pl_module.log(
            "throughput/val_unpadded_tokens_per_second_per_gpu",
            np.mean(self._val_tokens_per_second),
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self._val_tokens_per_second = []

    # -- Prediction --

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Record the wall-clock time at the beginning of each prediction step."""
        self._predict_step_start_time = time.perf_counter()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Accumulate prediction throughput per batch."""
        if self._predict_step_start_time is None:
            return

        step_time = time.perf_counter() - self._predict_step_start_time
        self._predict_tokens_per_second.append(self._count_unpadded_tokens(batch) / step_time)

        if batch_idx % self.log_every_n_steps == 0 and self._predict_tokens_per_second:
            mean_tps = np.mean(self._predict_tokens_per_second)
            log.info("predict batch %d — unpadded tokens/s/GPU: %.1f", batch_idx, mean_tps)
            self._predict_tokens_per_second = []

    def on_predict_epoch_end(self, trainer, pl_module):
        """Log remaining aggregated prediction throughput at the end of the predict epoch."""
        if not self._predict_tokens_per_second:
            return

        mean_tps = np.mean(self._predict_tokens_per_second)
        log.info("predict epoch end — unpadded tokens/s/GPU: %.1f", mean_tps)
        self._predict_tokens_per_second = []
