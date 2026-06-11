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
import os
import time

import nvtx
import torch


try:
    import nvdlfw_inspect.api as debug_api

    HAS_NVDLFW_INSPECT = True
except ImportError:
    debug_api = None
    HAS_NVDLFW_INSPECT = False
import torchmetrics
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor import DTensor
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

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

        self._device = torch.device(f"cuda:{dist_config.local_rank}")
        self.min_loss = torch.tensor(float("inf"), device=self._device)

        self.logging_frequency = args.logger.frequency

        metrics_dict = {
            "train/loss": torchmetrics.MeanMetric(),
            "train/grad_norm": torchmetrics.MeanMetric(),
            "train/learning_rate": torchmetrics.MeanMetric(),
            "train/step_time": torchmetrics.MeanMetric(),
            "train/tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/unpadded_tokens_per_second_per_gpu": torchmetrics.MeanMetric(),
            "train/total_unpadded_tokens_per_batch": torchmetrics.SumMetric(),
            "train/gpu_memory_allocated_max_gb": torchmetrics.MaxMetric(),
            "train/gpu_memory_allocated_mean_gb": torchmetrics.MeanMetric(),
        }

        self.metrics = torchmetrics.MetricCollection(metrics_dict)
        # We move metrics to a GPU device so we can use torch.distributed to aggregate them before logging.
        self.metrics.to(self._device)
        self.previous_step_time = time.perf_counter()
        self._profiler = None

        if self._dist_config.is_main_process():
            # Log the entire args object to wandb for experiment tracking and reproducibility.
            self._wandb_run = wandb.init(**args.wandb, config=self._run_config)
            self._progress_bar = tqdm(total=args.num_train_steps, desc="Training")

            if args.profiler.enabled:
                self._profiler = NsightProfiler(
                    **args.profiler,
                    wandb_run=self._wandb_run,
                    dist_config=dist_config,
                )

        # Gradient accumulation tracking
        self.num_tokens = 0
        self.num_unpadded_tokens = torch.tensor(0, dtype=torch.int64, device=self._device)
        self.running_loss = torch.tensor(0.0, device=self._device)
        self.grad_acc_step_count = 0

        # Whether to step debug_api.step() after each step
        self.fp8_stats_enabled = args.fp8_stats_config.enabled

    @nvtx.annotate("PerfLogger.log_micro_step", color="pink")
    def log_micro_step(self, step: int, batch: dict[str, torch.Tensor], outputs: CausalLMOutputWithPast):
        """Store data on micro step for gradient accumulation metrics.

        Args:
            step: The step number.
            batch: The batch of data for the micro step.
            outputs: The outputs of the micro step.
        """
        if self._dist_config.local_rank == 0:
            logger.debug("log_micro_step")

        assert outputs.loss is not None, "Loss is None"

        with torch.no_grad():
            self.grad_acc_step_count += 1
            self.running_loss += outputs.loss

            if step % self.logging_frequency == 0 and step > 0:
                self.num_tokens += batch["input_ids"].numel()
                # Use attention_mask to count unpadded tokens (works for both BSHD and THD)
                if "attention_mask" in batch:
                    self.num_unpadded_tokens += batch["attention_mask"].sum()
                else:
                    # Fallback for pure sequence packing with no padding: all tokens are unpadded
                    self.num_unpadded_tokens += batch["input_ids"].numel()

    @nvtx.annotate("PerfLogger.log_step", color="purple")
    def log_step(
        self,
        step: int,
        grad_norm: torch.Tensor | DTensor,
        lr: float,
    ):
        """Log a step to the logger and wandb.

        Args:
            step: The step number.
            grad_norm: The gradient norm of the step.
            lr: The learning rate of the step.
        """
        if self._dist_config.local_rank == 0:
            logger.debug("log_step %s", step)

        with torch.no_grad():
            # Use accumulated metrics from gradient accumulation
            assert self.grad_acc_step_count > 0, (
                f"Gradient accumulation steps ({self.grad_acc_step_count}) must be greater than 0, "
                f"and can be incremented by log_micro_step()."
            )

            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.to_local()

            if self._profiler is not None:
                self._profiler.step(step)

            if self.fp8_stats_enabled and HAS_NVDLFW_INSPECT:
                debug_api.step()

            if step % self.logging_frequency == 0 and step > 0:
                # Calculate average loss over all micro steps in the logging window
                avg_loss = self.running_loss / self.grad_acc_step_count
                self.min_loss = torch.minimum(self.min_loss, avg_loss)

                # Calculate an average step time over all steps in the logging window
                now = time.perf_counter()
                step_time = (now - self.previous_step_time) / self.logging_frequency
                self.previous_step_time = now

                # For some reason, these trigger a CudaStreamSynchronize call, which blocks the dataloader in the next
                # step. We therefore only update these once every logging_frequency steps.
                self.metrics["train/loss"].update(avg_loss)
                self.metrics["train/learning_rate"].update(lr)
                self.metrics["train/grad_norm"].update(grad_norm)
                self.metrics["train/step_time"].update(step_time)
                self.metrics["train/tokens_per_second_per_gpu"].update(self.num_tokens / step_time)
                self.metrics["train/unpadded_tokens_per_second_per_gpu"].update(self.num_unpadded_tokens / step_time)
                self.metrics["train/total_unpadded_tokens_per_batch"].update(self.num_unpadded_tokens)

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

                # Reset running loss and other tracking variables for next window
                self.running_loss.zero_()
                self.num_tokens = 0
                self.num_unpadded_tokens.zero_()
                self.grad_acc_step_count = 0

    def log_validation(self, step: int, val_metrics: dict):
        """Log validation metrics to wandb.

        Args:
            step: The current training step.
            val_metrics: Dictionary with val_loss, val_ppl, val_tokens, val_batches, and optional Megatron-style.
        """
        if self._dist_config.is_main_process():
            metrics = {
                "val/loss": val_metrics["val_loss"],
                "val/ppl": val_metrics["val_ppl"],
                "val/tokens": val_metrics["val_tokens"],
                "val/batches": val_metrics["val_batches"],
            }
            # Add Megatron-style metrics if available
            if "val_loss_megatron" in val_metrics:
                metrics["val/loss_megatron"] = val_metrics["val_loss_megatron"]
            if "val_ppl_megatron" in val_metrics:
                metrics["val/ppl_megatron"] = val_metrics["val_ppl_megatron"]
            wandb.log(metrics, step=step)

    def finish(self):
        """Finish the logger and close the progress bar."""
        if not self._dist_config.is_main_process():
            return

        wandb.finish()
        self._progress_bar.close()

        if self.fp8_stats_enabled and HAS_NVDLFW_INSPECT:
            debug_api.end_debug()


class NsightProfiler:
    """Nsight Systems profiler wrapper for performance analysis.

    Args:
        enabled: Whether profiling is enabled.
        start_step: The step number at which to start profiling.
        end_step: The step number at which to end profiling.
        wandb_run: The wandb run for logging artifacts.
        dist_config: The distributed configuration.
    """

    def __init__(
        self,
        enabled: bool,
        start_step: int,
        end_step: int,
        wandb_run: wandb.Run,
        dist_config: DistributedConfig,
    ):
        """Initialize the Nsight profiler."""
        self._wandb_run = wandb_run
        self._dist_config = dist_config

        self.start_step = start_step
        self.end_step = end_step

        self.current_step = 0
        self.profiling_started = False
        self.profiling_finished = False

        # Check if running under nsys
        self.running_under_nsys = "NSYS_PROFILING_SESSION_ID" in os.environ

        if self.running_under_nsys:
            logger.info("Detected running under nsys - will use CUDA Profiler API for range control")
        else:
            logger.warning(
                "Not running under nsys. Profiling will be skipped. "
                "To enable profiling, run your script with: "
                "nsys profile -o output_trace --trace=cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi "
                "--capture-range-end=stop python train_fsdp2.py profiler.enabled=true"
            )

    def step(self, step_num: int):
        """Record a training step and control profiling based on the schedule.

        Args:
            step_num: The current training step number.
        """
        if not self.running_under_nsys or self.profiling_finished:
            return

        self.current_step = step_num

        # Start profiling at start_step
        if self.current_step == self.start_step and not self.profiling_started:
            self._start_profiling()
        # Stop profiling at end_step
        elif self.current_step == self.end_step and self.profiling_started:
            self._stop_profiling()

    def _start_profiling(self):
        """Start CUDA profiling using the CUDA Profiler API."""
        if self.profiling_started:
            return

        logger.info(f"Starting Nsight profiling at step {self.current_step}")
        try:
            torch.cuda.cudart().cudaProfilerStart()  # type: ignore[attr-defined]
            self.profiling_started = True
        except Exception as e:
            logger.error(f"Failed to start CUDA profiler: {e}")

    def _stop_profiling(self):
        """Stop CUDA profiling using the CUDA Profiler API."""
        if not self.profiling_started or self.profiling_finished:
            return

        logger.info(f"Stopping Nsight profiling at step {self.current_step}")
        try:
            torch.cuda.cudart().cudaProfilerStop()  # type: ignore[attr-defined]
            self.profiling_started = False
            self.profiling_finished = True
        except Exception as e:
            logger.error(f"Failed to stop CUDA profiler: {e}")
