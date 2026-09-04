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
import sys
import time

import torch
import torchmetrics
import torchmetrics.text
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers.modeling_outputs import MaskedLMOutput

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


class PerfLogger:
    """Class to log performance metrics to stdout and wandb, and print final averaged metrics at the end of training.

    Args:
        dist_config: The distributed configuration.
        args: The arguments.

    Attributes:
        min_loss: The minimum loss seen so far.
    """

    def __init__(self, dist_config: DistributedConfig, args: DictConfig):
        """Initialize the logger."""
        self._dist_config = dist_config
        self._run_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

        self.min_loss = float("inf")

        self.logging_frequency = args.logger.frequency
        # Track whether to collect memory stats (disabled by default for max performance)

        metrics_dict = {
            "train/loss": torchmetrics.MeanMetric(),
            "train/grad_norm": torchmetrics.MeanMetric(),
            "train/learning_rate": torchmetrics.MeanMetric(),
            "train/num_tokens_per_gpu": torchmetrics.MeanMetric(),
            "train/total_num_tokens": torchmetrics.SumMetric(),
            "train/num_unpadded_tokens": torchmetrics.MeanMetric(),
            "train/step_time": torchmetrics.MeanMetric(),
            "train/tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/unpadded_tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/total_unpadded_tokens_per_batch": torchmetrics.SumMetric(),
            "train/perplexity": torchmetrics.text.Perplexity(ignore_index=-100),
            "train/gpu_memory_allocated_max_gb": torchmetrics.MaxMetric(),
            "train/gpu_memory_allocated_mean_gb": torchmetrics.MeanMetric(),
            "val/loss": torchmetrics.MeanMetric(),
            "val/accuracy": torchmetrics.MeanMetric(),
        }

        self.metrics = torchmetrics.MetricCollection(metrics_dict)
        # We move metrics to a GPU device so we can use torch.distributed to aggregate them before logging.
        self.metrics.to(torch.device(f"cuda:{dist_config.local_rank}"))
        self.previous_step_time = time.perf_counter()
        self.train_start_time = None
        self.train_end_time = None

        if self._dist_config.is_main_process():
            # Log the entire args object to wandb for experiment tracking and reproducibility.
            wandb.init(**args.wandb_init_args, config=self._run_config)
            self._progress_bar = tqdm(total=args.num_train_steps, file=sys.stderr, desc="Training")

    def log_train_end_time(self):
        """Log when the train loop for a batch ends."""
        self.train_end_time = time.perf_counter()
        return

    def log_train_start_time(self):
        """Log when the train loop for a batch starts."""
        self.train_start_time = time.perf_counter()
        return

    def log_step(
        self,
        step: int,
        batch: dict[str, torch.Tensor],
        outputs: MaskedLMOutput,
        grad_norm: float,
        lr: float,
        val_loss: float | None = None,
        val_acc: float | None = None,
    ):
        """Log a step to the logger and wandb.

        Args:
            step: The step number.
            batch: The batch of data for the step.
            outputs: The outputs of the step.
            grad_norm: The gradient norm of the step.
            lr: The learning rate of the step.
            val_loss: The validation loss of the step (if calculated)
            val_acc: The validation accuracy of the step (if calculated)
        """
        num_tokens = batch["input_ids"].numel()
        # 1 is the padding token for ESM-2.
        num_unpadded_tokens = batch["input_ids"][batch["input_ids"] != 1].numel()

        self.min_loss = min(self.min_loss, outputs.loss.item())
        step_time = self.train_end_time - self.train_start_time

        self.metrics["train/loss"].update(outputs.loss)
        self.metrics["train/num_tokens_per_gpu"].update(num_tokens)
        self.metrics["train/total_num_tokens"].update(num_tokens)
        self.metrics["train/num_unpadded_tokens"].update(num_unpadded_tokens)
        self.metrics["train/learning_rate"].update(lr)
        self.metrics["train/grad_norm"].update(grad_norm)
        self.metrics["train/step_time"].update(step_time)
        self.metrics["train/tokens_per_second_per_gpu"].update(num_tokens / step_time)
        self.metrics["train/unpadded_tokens_per_second_per_gpu"].update(num_unpadded_tokens / step_time)
        self.metrics["train/total_unpadded_tokens_per_batch"].update(num_unpadded_tokens / self.logging_frequency)

        if val_loss is not None:
            self.metrics["val/loss"].update(val_loss)
            self.metrics["val/accuracy"].update(val_acc)

        # Handle sequence packing for torchmetrics calculation.
        if outputs.logits.dim() < 3:
            outputs.logits = outputs.logits.unsqueeze(0)

        self.metrics["train/perplexity"].update(outputs.logits, batch["labels"])

        if step % self.logging_frequency == 0 and step > 0:
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            self.metrics["train/gpu_memory_allocated_max_gb"].update(memory_allocated)
            self.metrics["train/gpu_memory_allocated_mean_gb"].update(memory_allocated)

            metrics = self.metrics.compute()
            self.metrics.reset()
            metrics["train/global_step"] = torch.tensor(step, dtype=torch.int64)

            if self._dist_config.is_main_process():
                train_metrics = {k: v for k, v in metrics.items() if k.startswith("train/")}
                val_metrics = {k: v for k, v in metrics.items() if k.startswith("val/")}

                wandb.log(train_metrics, step=step)

                if val_loss is not None and len(val_metrics) > 0:
                    wandb.log(val_metrics, step=step)

                self._progress_bar.update(self.logging_frequency)
                self._progress_bar.set_postfix({"loss": outputs.loss.item()})

            if self._dist_config.local_rank == 0:
                logger.info(", ".join([f"{k.split('/')[1]}: {v:.3g}" for k, v in metrics.items()]))

    def finish(self):
        """Finish the logger and close the progress bar."""
        if not self._dist_config.is_main_process():
            return

        wandb.finish()
        self._progress_bar.close()
