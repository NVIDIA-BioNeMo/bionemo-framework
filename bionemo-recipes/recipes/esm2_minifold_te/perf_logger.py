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

"""Performance logger for ESM2-MiniFold TE structure prediction training."""

import logging
import time

import nvdlfw_inspect.api as debug_api
import torch
import torchmetrics
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor import DTensor
from tqdm import tqdm

import wandb
from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


class PerfLogger:
    """Logs training metrics (loss, lDDT, grad norm, timing) to stdout and wandb.

    Attributes:
        min_loss: The minimum loss seen so far.
    """

    def __init__(self, dist_config: DistributedConfig, args: DictConfig, quant_logger=None):
        self._dist_config = dist_config
        self._run_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

        self.min_loss = torch.tensor(float("inf"), device=torch.device(f"cuda:{dist_config.local_rank}"))
        self.logging_frequency = args.logger.frequency
        self.quant_stats_enabled = args.quant_stats_config.enabled
        self._quant_logger = quant_logger
        self._log_heatmap = args.quant_stats_config.get("log_heatmap", False)

        metrics_dict = {
            "train/loss": torchmetrics.MeanMetric(),
            "train/disto_loss": torchmetrics.MeanMetric(),
            "train/grad_norm": torchmetrics.MeanMetric(),
            "train/learning_rate": torchmetrics.MeanMetric(),
            "train/step_time": torchmetrics.MeanMetric(),
            "train/gpu_memory_allocated_max_gb": torchmetrics.MaxMetric(),
            "train/distogram_acc": torchmetrics.MeanMetric(),
            "train/contact_precision_8A": torchmetrics.MeanMetric(),
            "train/contact_recall_8A": torchmetrics.MeanMetric(),
            "train/lddt_from_distogram": torchmetrics.MeanMetric(),
            "train/mean_distance_error": torchmetrics.MeanMetric(),
            "train/unpadded_tokens_per_sec": torchmetrics.MeanMetric(),
        }

        self.metrics = torchmetrics.MetricCollection(metrics_dict)
        self.metrics.to(torch.device(f"cuda:{dist_config.local_rank}"))
        self.previous_step_time = time.perf_counter()

        if self._dist_config.is_main_process():
            wandb.init(**args.wandb_init_args, config=self._run_config)
            self._progress_bar = tqdm(total=args.num_train_steps, desc="Training")

    def log_step(
        self,
        step: int,
        loss: torch.Tensor,
        disto_loss: torch.Tensor | None = None,
        grad_norm: torch.Tensor | DTensor | float = 0.0,
        lr: float = 0.0,
        structure_metrics: dict[str, torch.Tensor] | None = None,
        unpadded_tokens: float = 0.0,
    ):
        """Log a training step."""
        with torch.no_grad():
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.to_local()

            if step % self.logging_frequency == 0 and step > 0:
                self.min_loss = torch.minimum(self.min_loss, loss)
                elapsed_time, self.previous_step_time = (
                    time.perf_counter() - self.previous_step_time,
                    time.perf_counter(),
                )
                step_time = elapsed_time / self.logging_frequency

                self.metrics["train/loss"].update(loss)
                if disto_loss is not None:
                    self.metrics["train/disto_loss"].update(disto_loss)
                self.metrics["train/learning_rate"].update(lr)
                self.metrics["train/grad_norm"].update(grad_norm)
                self.metrics["train/step_time"].update(step_time)
                if unpadded_tokens > 0 and step_time > 0:
                    self.metrics["train/unpadded_tokens_per_sec"].update(unpadded_tokens / step_time)

                if structure_metrics is not None:
                    for key, value in structure_metrics.items():
                        metric_key = f"train/{key}"
                        if metric_key in self.metrics:
                            self.metrics[metric_key].update(value)

                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                self.metrics["train/gpu_memory_allocated_max_gb"].update(memory_allocated)

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
                    self._progress_bar.set_postfix({"loss": loss.item()})

                if self._dist_config.local_rank == 0:
                    logger.info(", ".join([f"{k.split('/')[1]}: {v:.3g}" for k, v in metrics.items()]))

                if self.quant_stats_enabled:
                    debug_api.step()
                    if self._log_heatmap and self._quant_logger is not None and self._dist_config.is_main_process():
                        import matplotlib.pyplot as plt

                        fig = self._quant_logger.generate_heatmap()
                        if fig is not None:
                            wandb.log({"quant/gradient_underflow_heatmap": wandb.Image(fig)}, step=step)
                            plt.close(fig)

    def finish(self):
        """Finish the logger."""
        if self.quant_stats_enabled:
            debug_api.end_debug()
        if not self._dist_config.is_main_process():
            return
        wandb.finish()
        self._progress_bar.close()
