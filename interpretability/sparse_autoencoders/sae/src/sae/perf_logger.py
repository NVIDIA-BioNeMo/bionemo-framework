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
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch


try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class PerfLogger:
    """Performance logger for SAE training.

    Tracks and logs training performance metrics including:
    - Training loss (total and components like reconstruction MSE, sparsity)
    - Step timing and throughput (samples per second)
    - GPU memory usage (peak and mean)
    - Gradient norms and learning rate

    The logger maintains running averages and logs aggregated metrics
    at configurable intervals. Supports both stdout and wandb logging.

    Example:
        >>> perf_logger = PerfLogger(log_interval=10, use_wandb=True)
        >>> for step, batch in enumerate(dataloader):
        ...     outputs = model(batch)
        ...     loss.backward()
        ...     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ...     optimizer.step()
        ...     perf_logger.log_step(
        ...         step=step,
        ...         batch=batch,
        ...         loss_dict={'total': loss, 'mse': mse_loss, 'sparsity': sparsity_loss},
        ...         grad_norm=grad_norm,
        ...         lr=optimizer.param_groups[0]['lr'],
        ...     )
        >>> perf_logger.finish()

    Attributes:
        log_interval: How often to log aggregated metrics (in steps)
        use_wandb: Whether to log to Weights & Biases
        print_logs: Whether to print logs to stdout
    """

    def __init__(
        self,
        log_interval: int = 10,
        use_wandb: bool = True,
        print_logs: bool = True,
        device: str = "cuda",
    ):
        """Initialize the performance logger.

        Args:
            log_interval: Log aggregated metrics every N steps
            use_wandb: Whether to log to wandb (requires active wandb run)
            print_logs: Whether to print logs to stdout
            device: Device being used for training (for GPU memory tracking)
        """
        self.log_interval = log_interval
        self.use_wandb = use_wandb and HAS_WANDB
        if use_wandb and not HAS_WANDB:
            print("Warning: wandb not installed. Install with: pip install wandb")
        self.print_logs = print_logs
        self.device = device

        # Running metric accumulators
        self._metrics: Dict[str, List[float]] = defaultdict(list)

        # Timing
        self._previous_step_time: Optional[float] = None
        self._step_times: List[float] = []

        # Track best/min loss
        self.min_loss: float = float("inf")
        self._train_start_time: float = time.perf_counter()

        # GPU memory tracking
        self._gpu_memory_allocated: List[float] = []
        self._gpu_available = torch.cuda.is_available() and "cuda" in device

    def _get_gpu_memory_gb(self) -> float:
        """Get current GPU memory allocated in GB."""
        if not self._gpu_available:
            return 0.0
        return torch.cuda.memory_allocated() / (1024**3)

    def _get_gpu_memory_max_gb(self) -> float:
        """Get peak GPU memory allocated in GB."""
        if not self._gpu_available:
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024**3)

    def _compute_throughput(self, batch_size: int, step_time: float) -> float:
        """Compute samples per second throughput."""
        if step_time <= 0:
            return 0.0
        return batch_size / step_time

    def log_step(
        self,
        step: int,
        batch: torch.Tensor,
        loss_dict: Optional[Dict[str, Any]] = None,
        grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, float]]:
        """Log metrics for a single training step.

        Args:
            step: Current training step (0-indexed)
            batch: Input batch tensor (used for batch size and throughput)
            loss_dict: Dictionary of loss values (e.g., {'total': 0.5, 'mse': 0.3, 'sparsity': 0.2})
            grad_norm: Gradient norm after clipping (optional)
            lr: Current learning rate (optional)
            extra_metrics: Additional custom metrics to log (optional)

        Returns:
            Dictionary of aggregated metrics if this is a logging step, None otherwise.
        """
        current_time = time.perf_counter()
        batch_size = batch.shape[0] if isinstance(batch, torch.Tensor) else len(batch)

        # Compute step time
        if self._previous_step_time is not None:
            step_time = current_time - self._previous_step_time
            self._step_times.append(step_time)
            throughput = self._compute_throughput(batch_size, step_time)
            self._metrics["train/samples_per_second"].append(throughput)
            self._metrics["train/step_time"].append(step_time)

        self._previous_step_time = current_time

        # Track GPU memory
        if self._gpu_available:
            gpu_mem = self._get_gpu_memory_gb()
            self._gpu_memory_allocated.append(gpu_mem)

        # Track losses
        if loss_dict is not None:
            for key, value in loss_dict.items():
                val = value.item() if torch.is_tensor(value) else value
                if key == "total":
                    self._metrics["train/loss"].append(val)
                    self.min_loss = min(self.min_loss, val)
                else:
                    self._metrics[f"train/{key}"].append(val)

        # Track gradient norm
        if grad_norm is not None:
            val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            self._metrics["train/grad_norm"].append(val)

        # Track learning rate
        if lr is not None:
            self._metrics["train/learning_rate"].append(lr)

        # Track extra metrics
        if extra_metrics is not None:
            for key, value in extra_metrics.items():
                self._metrics[f"train/{key}"].append(value)

        # Log at interval
        if (step + 1) % self.log_interval == 0:
            return self._log_aggregated(step)

        return None

    def _log_aggregated(self, step: int) -> Dict[str, float]:
        """Compute and log aggregated metrics.

        Args:
            step: Current training step

        Returns:
            Dictionary of aggregated metrics
        """
        aggregated: Dict[str, float] = {"train/step": step}

        # Aggregate running metrics (mean)
        for key, values in self._metrics.items():
            if values:
                aggregated[key] = np.mean(values)

        # GPU memory stats
        if self._gpu_available and self._gpu_memory_allocated:
            aggregated["train/gpu_memory_allocated_mean_gb"] = np.mean(self._gpu_memory_allocated)
            aggregated["train/gpu_memory_allocated_max_gb"] = self._get_gpu_memory_max_gb()

        # Add min loss tracker
        aggregated["train/min_loss"] = self.min_loss

        #

        # Log to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.log(aggregated, step=step)

        # Print to stdout
        if self.print_logs:
            self._print_metrics(step, aggregated)

        # Reset accumulators
        self._reset_accumulators()

        return aggregated

    def _print_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Print formatted metrics to stdout."""
        parts = [f"Step {step + 1}"]

        if "train/fvu" in metrics:
            parts.append(f"fvu: {metrics['train/fvu']:.4f}")

        if "train/loss" in metrics:
            parts.append(f"loss: {metrics['train/loss']:.4f}")

        if "train/samples_per_second" in metrics:
            parts.append(f"samples/s: {metrics['train/samples_per_second']:.1f}")

        if "train/step_time" in metrics:
            parts.append(f"step_time: {metrics['train/step_time'] * 1000:.1f}ms")

        if "train/gpu_memory_allocated_max_gb" in metrics:
            parts.append(f"gpu_mem: {metrics['train/gpu_memory_allocated_max_gb']:.2f}GB")

        if "train/sparsity" in metrics:
            parts.append(f"avg_nonzero_act: {metrics['train/sparsity']}")

        if "train/grad_norm" in metrics:
            parts.append(f"grad_norm: {metrics['train/grad_norm']:.4f}")

        if "train/dead_latents" in metrics:
            parts.append(f"dead_latents (%): {metrics['train/dead_latents']:.4f}")

        if "train/reconstruction" in metrics:
            parts.append(f"reconstruction: {metrics['train/reconstruction']:.4f}")

        if "train/l1" in metrics:
            parts.append(f"l1: {metrics['train/l1']:.4f}")

        if "train/variance_explained" in metrics:
            parts.append(f"var_exp: {metrics['train/variance_explained']:.4f}")

        if "train/mse" in metrics:
            parts.append(f"mse: {metrics['train/mse']:.6f}")

        print(" | ".join(parts))

    def _reset_accumulators(self) -> None:
        """Reset metric accumulators after logging."""
        self._metrics.clear()
        self._step_times.clear()
        self._gpu_memory_allocated.clear()

    def reset(self) -> None:
        """Reset all state for a new training run."""
        self._reset_accumulators()
        self._previous_step_time = None
        self.min_loss = float("inf")
        self._train_start_time = time.perf_counter()
        if self._gpu_available:
            torch.cuda.reset_peak_memory_stats()

    def finish(self) -> None:
        """Finalize logging at end of training."""
        elapsed = time.perf_counter() - self._train_start_time
        minutes, seconds = divmod(elapsed, 60)

        if self.print_logs:
            print(f"\nTraining finished in {int(minutes)}m {seconds:.1f}s. Min loss: {self.min_loss:.6f}")
            if self._gpu_available:
                print(f"Peak GPU memory: {self._get_gpu_memory_max_gb():.2f} GB")

        if self.use_wandb and HAS_WANDB and wandb.run is not None:
            wandb.log({"train/total_time_seconds": elapsed})
