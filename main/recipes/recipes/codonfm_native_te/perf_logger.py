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

"""Performance logging utilities for CodonFM training."""

import logging
import time

import nvdlfw_inspect.api as debug_api
import torch
import torchmetrics
import torchmetrics.text
import wandb
from distributed_config import DistributedConfig
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from transformers.modeling_outputs import MaskedLMOutput


logger = logging.getLogger(__name__)

# CodonFM pad token ID
PAD_TOKEN_ID = 3


class PerfLogger:
    """Performance logger for CodonFM training.

    Logs metrics to stdout and wandb, and prints final averaged metrics at the end of training.

    Args:
        dist_config: The distributed configuration.
        args: The Hydra arguments.
    """

    def __init__(self, dist_config: DistributedConfig, args: DictConfig):
        """Initialize the logger."""
        self._dist_config = dist_config
        self._run_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

        self.min_loss = torch.tensor(float("inf"), device=torch.device(f"cuda:{dist_config.local_rank}"))

        self.logging_frequency = args.logger.frequency

        metrics_dict = {
            "train/loss": torchmetrics.MeanMetric(),
            "train/grad_norm": torchmetrics.MeanMetric(),
            "train/learning_rate": torchmetrics.MeanMetric(),
            "train/step_time": torchmetrics.MeanMetric(),
            "train/tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/unpadded_tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/perplexity": torchmetrics.text.Perplexity(ignore_index=-100),
            "train/gpu_memory_allocated_max_gb": torchmetrics.MaxMetric(),
            "train/gpu_memory_allocated_mean_gb": torchmetrics.MeanMetric(),
        }

        self.metrics = torchmetrics.MetricCollection(metrics_dict)
        self.metrics.to(torch.device(f"cuda:{dist_config.local_rank}"))
        self.previous_step_time = time.perf_counter()

        if self._dist_config.is_main_process():
            wandb.init(**args.wandb_init_args, config=self._run_config)
            self._progress_bar = tqdm(total=args.num_train_steps, desc="Training")

        # Whether to step debug_api.step() after each step
        self.quant_stats_config = args.quant_stats_config.enabled

        # Gradient accumulation tracking
        self._device = torch.device(f"cuda:{dist_config.local_rank}")
        self.num_tokens = 0
        self.num_unpadded_tokens = torch.tensor(0, dtype=torch.int64, device=self._device)
        self.running_loss = torch.tensor(0.0, device=self._device)
        self.grad_acc_step_count = 0

    def log_micro_step(self, step: int, batch: dict[str, torch.Tensor], outputs: MaskedLMOutput):
        """Store data on micro step for gradient accumulation metrics.

        Args:
            step: The current optimizer step number.
            batch: The input batch for this micro-step.
            outputs: Model outputs for this micro-step (with unscaled loss).
        """
        assert outputs.loss is not None, "Loss is None"

        with torch.no_grad():
            self.grad_acc_step_count += 1
            self.running_loss += outputs.loss

            if step % self.logging_frequency == 0 and step > 0:
                self.num_tokens += batch["input_ids"].numel()
                num_unpadded_tokens = batch["input_ids"][batch["input_ids"] != PAD_TOKEN_ID].numel()
                self.num_unpadded_tokens += num_unpadded_tokens

                # Update perplexity per micro-batch since it needs logits + labels
                logits = outputs.logits
                if logits.dim() < 3:
                    logits = logits.unsqueeze(0)
                self.metrics["train/perplexity"].update(logits, batch["labels"])

    def log_step(
        self,
        step: int,
        grad_norm: torch.Tensor | DTensor | float,
        lr: float,
    ):
        """Log a training step (called once per optimizer step).

        Args:
            step: Current optimizer step.
            grad_norm: Gradient norm value.
            lr: Current learning rate.
        """
        with torch.no_grad():
            assert self.grad_acc_step_count > 0, (
                f"Gradient accumulation steps ({self.grad_acc_step_count}) must be greater than 0, "
                f"and can be incremented by log_micro_step()."
            )

            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.to_local()

            if self.quant_stats_config:
                debug_api.step()

            # Calculate average loss over all micro steps in the logging window
            avg_loss = self.running_loss / self.grad_acc_step_count
            self.min_loss = torch.minimum(self.min_loss, avg_loss)

            if step % self.logging_frequency == 0 and step > 0:
                elapsed_time, self.previous_step_time = (
                    time.perf_counter() - self.previous_step_time,
                    time.perf_counter(),
                )
                step_time = elapsed_time / self.logging_frequency

                self.metrics["train/loss"].update(avg_loss)
                self.metrics["train/learning_rate"].update(lr)
                self.metrics["train/grad_norm"].update(
                    grad_norm if isinstance(grad_norm, torch.Tensor) else torch.tensor(grad_norm)
                )
                self.metrics["train/step_time"].update(step_time)
                self.metrics["train/tokens_per_second_per_gpu"].update(self.num_tokens / step_time)
                self.metrics["train/unpadded_tokens_per_second_per_gpu"].update(self.num_unpadded_tokens / step_time)

                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                self.metrics["train/gpu_memory_allocated_max_gb"].update(memory_allocated)
                self.metrics["train/gpu_memory_allocated_mean_gb"].update(memory_allocated)

                metrics = self.metrics.compute()
                self.metrics.reset()
                metrics = {
                    k: v.detach().cpu().item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
                    for k, v in metrics.items()
                }
                metrics["train/global_step"] = step

                if self._dist_config.is_main_process():
                    wandb.log(metrics, step=step)
                    self._progress_bar.update(self.logging_frequency)
                    self._progress_bar.set_postfix({"loss": avg_loss.item()})

                if self._dist_config.local_rank == 0:
                    logger.info(", ".join([f"{k.split('/')[1]}: {v:.3g}" for k, v in metrics.items()]))

                # Reset running accumulators for next logging window
                self.running_loss.zero_()
                self.num_tokens = 0
                self.num_unpadded_tokens.zero_()
                self.grad_acc_step_count = 0

    def finish(self):
        """Finish the logger."""
        if self.quant_stats_config:
            debug_api.end_debug()

        if not self._dist_config.is_main_process():
            return
        wandb.finish()
        self._progress_bar.close()
