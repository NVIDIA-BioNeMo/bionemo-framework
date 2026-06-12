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

"""Training utilities for Sparse Autoencoders.

This module provides a Trainer class that handles all training-related concerns,
separating training logic from the SAE model architecture.
"""

import contextlib
import itertools
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .eval import DeadLatentTracker
from .perf_logger import PerfLogger


try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class TrainingConfig:
    """Configuration for SAE training.

    Attributes:
        lr: Learning rate (base LR, may be scaled if lr_scale_with_latents=True)
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on ('cuda', 'cpu', 'mps')
        log_interval: Print loss every N epochs
        shuffle: Whether to shuffle training data
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory in dataloader
        checkpoint_dir: Directory to save checkpoints (None = no checkpointing)
        checkpoint_steps: Save checkpoint every N steps (None = only at end)
        lr_scale_with_latents: Scale LR by 1/sqrt(hidden_dim/reference_dim) per OpenAI paper
        lr_reference_hidden_dim: Reference hidden_dim for LR scaling (default 2048)
        warmup_steps: Number of steps for linear LR warmup (0 = no warmup)
        grad_accumulation_steps: Number of microsteps to accumulate gradients before an optimizer step (1 = no accumulation)
        max_grad_norm: Max gradient norm for clipping (None = no clipping)
        lr_schedule: LR schedule after warmup ('constant', 'cosine', 'linear')
        lr_min: Minimum LR for decay schedules
        lr_decay_steps: Total steps for LR decay (None = use full training duration)
        max_steps: Stop after this many optimizer steps (None = run all n_epochs).
            When set, epochs loop until the step budget is reached, so it controls
            duration directly (useful for streaming, which has no fixed length).
    """

    lr: float = 3e-4
    n_epochs: int = 10
    batch_size: int = 4096
    device: str = "cuda"
    log_interval: int = 100
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    checkpoint_dir: Optional[str] = None
    checkpoint_steps: Optional[int] = None
    lr_scale_with_latents: bool = False
    lr_reference_hidden_dim: int = 2048
    warmup_steps: int = 0
    grad_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None
    lr_schedule: str = "constant"
    lr_min: float = 0.0
    lr_decay_steps: Optional[int] = None
    max_steps: Optional[int] = None


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging.

    Attributes:
        enabled: Whether to enable wandb logging
        project: W&B project name
        run_name: W&B run name (auto-generated if None)
        group: W&B group name for organizing related runs
        job_type: W&B job type tag
        config: Additional config dict to log
        log_interval: Log to W&B every N batches
    """

    enabled: bool = False
    project: str = "biosae"
    run_name: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    log_interval: int = 10


@dataclass
class ParallelConfig:
    """Configuration for distributed data parallel training.

    Attributes:
        dp_size: Data parallel size (number of GPUs). 1 = single GPU, >1 = DDP.
    """

    dp_size: int = 1


class Trainer:
    """Trainer for Sparse Autoencoders.

    Handles all training-related concerns including:
    - Data loading and batching
    - Optimizer setup and management
    - Training loop execution
    - Loss computation and logging
    - Weights & Biases integration
    - Performance logging with PerfLogger

    Example:
        >>> from biosae import TopKSAE
        >>> from biosae.training import Trainer, TrainingConfig, PerfLogger
        >>>
        >>> sae = TopKSAE(input_dim=768, hidden_dim=4096, top_k=32)
        >>> perf_logger = PerfLogger(log_interval=10, use_wandb=True)
        >>> trainer = Trainer(
        ...     sae,
        ...     TrainingConfig(lr=1e-3, n_epochs=10),
        ...     perf_logger=perf_logger,
        ... )
        >>> final_loss = trainer.fit(embeddings)
        >>> print(f"Final loss: {final_loss:.6f}")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        wandb_config: Optional[WandbConfig] = None,
        perf_logger: Optional[PerfLogger] = None,
        parallel_config: Optional[ParallelConfig] = None,
    ):
        """Initialize the trainer.

        Args:
            model: SAE model to train
            config: Training configuration (uses defaults if None)
            optimizer: Custom optimizer (Adam with config.lr if None)
            loss_fn: Custom loss function (uses model.compute_loss if None)
            wandb_config: Weights & Biases configuration
            perf_logger: Performance logger for detailed metrics (optional)
            parallel_config: Distributed data parallel configuration (optional)
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.wandb_config = wandb_config or WandbConfig()
        self.perf_logger = perf_logger
        self.parallel_config = parallel_config or ParallelConfig()
        self.dead_latent_tracker = DeadLatentTracker(model.hidden_dim)

        # DDP state
        self.rank: int = 0
        self.world_size: int = 1
        self.is_distributed: bool = False
        self._data_sharded: bool = False

        # Validate lr_schedule
        valid_schedules = ("constant", "cosine", "linear")
        if self.config.lr_schedule not in valid_schedules:
            raise ValueError(f"Unknown lr_schedule: {self.config.lr_schedule!r}. Expected one of {valid_schedules}.")

        # Will be set during training
        self.dataloader: Optional[DataLoader] = None
        self.wandb_run = None
        self.global_step: int = 0
        self.current_epoch: int = 0
        self._target_lr: float = config.lr if config else 3e-4
        self._total_decay_steps: int = 0  # computed in fit() once we know total steps

    def _setup_dataloader(self, data: Union[torch.Tensor, DataLoader]) -> DataLoader:
        """Setup dataloader from tensor or existing dataloader."""
        if isinstance(data, torch.Tensor):
            dataset = TensorDataset(data)

            # Use DistributedSampler for DDP (unless data is already sharded per rank)
            sampler = None
            shuffle = self.config.shuffle
            if self.is_distributed and not self._data_sharded:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=self.config.shuffle,
                )
                shuffle = False  # Sampler handles shuffling

            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.is_distributed,
            )
        elif isinstance(data, DataLoader):
            return data
        else:
            raise TypeError("data must be either a Tensor or DataLoader")

    def _compute_effective_lr(self) -> float:
        """Compute effective learning rate, applying scaling if configured.

        Per OpenAI paper: LR scales with 1/sqrt(n) where n is number of latents.
        We scale relative to a reference hidden_dim so that:
            effective_lr = base_lr * sqrt(reference_dim / hidden_dim)
        """
        base_lr = self.config.lr

        if not self.config.lr_scale_with_latents:
            return base_lr

        model = self._get_model()
        hidden_dim = model.hidden_dim
        reference_dim = self.config.lr_reference_hidden_dim

        # Scale: lr ∝ 1/sqrt(hidden_dim)
        # effective_lr = base_lr * sqrt(reference_dim / hidden_dim)
        scale_factor = math.sqrt(reference_dim / hidden_dim)
        effective_lr = base_lr * scale_factor

        self._print_rank0(
            f"LR scaling: base_lr={base_lr:.2e} * sqrt({reference_dim}/{hidden_dim}) = {effective_lr:.2e}"
        )

        return effective_lr

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer if not provided."""
        if self.optimizer is None:
            effective_lr = self._compute_effective_lr()
            self._target_lr = effective_lr
            # Start at 0 if warmup enabled, otherwise at target
            initial_lr = 0.0 if self.config.warmup_steps > 0 else effective_lr
            return torch.optim.Adam(self.model.parameters(), lr=initial_lr)
        self._target_lr = self.config.lr
        return self.optimizer

    def _get_lr(self, step: int) -> float:
        """Compute learning rate with warmup and optional decay schedule.

        The schedule has two phases:
        1. Warmup (steps 0..warmup_steps-1): linear ramp from 0 to target_lr
        2. Decay (steps warmup_steps..warmup_steps+decay_steps): schedule-dependent decay

        Args:
            step: Current global training step.

        Returns:
            Learning rate for this step.
        """
        warmup_steps = self.config.warmup_steps
        target_lr = self._target_lr
        lr_min = self.config.lr_min

        # Phase 1: warmup
        if warmup_steps > 0 and step < warmup_steps:
            return target_lr * (step / warmup_steps)

        # Phase 2: decay (or constant)
        schedule = self.config.lr_schedule
        if schedule == "constant":
            return target_lr

        decay_steps = self._total_decay_steps
        if decay_steps <= 0:
            return target_lr

        # How far through the decay phase we are (0.0 to 1.0, clamped)
        steps_since_warmup = step - warmup_steps
        progress = min(steps_since_warmup / decay_steps, 1.0)

        if schedule == "cosine":
            # Cosine annealing: lr_min + 0.5 * (target - lr_min) * (1 + cos(pi * progress))
            return lr_min + 0.5 * (target_lr - lr_min) * (1.0 + math.cos(math.pi * progress))
        elif schedule == "linear":
            return target_lr + (lr_min - target_lr) * progress
        else:
            raise ValueError(f"Unknown lr_schedule: {schedule!r}. Expected 'constant', 'cosine', or 'linear'.")

    def _update_lr(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        """Update learning rate based on schedule."""
        lr = self._get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _setup_loss_fn(self) -> Callable:
        """Setup loss function if not provided."""
        if self.loss_fn is None:
            return self._get_model().loss
        return self.loss_fn

    def _setup_distributed(self) -> None:
        """Initialize distributed data parallel training."""
        if self.parallel_config.dp_size <= 1:
            return

        # Validate environment
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size != self.parallel_config.dp_size:
            raise RuntimeError(
                f"Config dp_size={self.parallel_config.dp_size} but launched with "
                f"WORLD_SIZE={world_size}. Use: torchrun --nproc_per_node={self.parallel_config.dp_size}"
            )

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_distributed = True

        # Set device for this rank
        torch.cuda.set_device(self.rank)
        self.config.device = f"cuda:{self.rank}"

        # Wrap model with DDP
        self.model = self.model.to(self.config.device)
        self.model = DDP(self.model, device_ids=[self.rank])

        if self.rank == 0:
            print(f"Initialized DDP with {self.world_size} GPUs")

    def _get_model(self) -> nn.Module:
        """Get the underlying model (unwrap DDP if needed)."""
        if self.is_distributed:
            return self.model.module
        return self.model

    def _sync_dead_latent_stats(self) -> None:
        """Synchronize stats_last_nonzero across GPUs for auxk loss."""
        if not self.is_distributed:
            return

        model = self._get_model()
        if hasattr(model, "stats_last_nonzero"):
            # MIN reduction: if ANY GPU saw it fire, mark it active (counter=0)
            dist.all_reduce(model.stats_last_nonzero, op=dist.ReduceOp.MIN)

    def _print_rank0(self, msg: str) -> None:
        """Print only on rank 0."""
        if self.rank == 0:
            print(msg)

    def _setup_wandb(self) -> None:
        """Initialize wandb if configured (only on rank 0)."""
        if not self.wandb_config.enabled:
            return

        # Only initialize wandb on rank 0
        if self.rank != 0:
            return

        model = self._get_model()

        # Build config
        config = {
            "model_class": model.__class__.__name__,
            "lr": self.config.lr,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "device": self.config.device,
            "dp_size": self.parallel_config.dp_size,
            "global_batch_size": self.config.batch_size * self.parallel_config.dp_size,
        }

        # Add model-specific config if available
        if hasattr(model, "input_dim"):
            config["input_dim"] = model.input_dim
        if hasattr(model, "hidden_dim"):
            config["hidden_dim"] = model.hidden_dim
        if hasattr(model, "_get_config"):
            config.update(model._get_config())

        # Add user config
        config.update(self.wandb_config.config)

        if HAS_WANDB:
            self.wandb_run = wandb.init(
                project=self.wandb_config.project,
                name=self.wandb_config.run_name,
                group=self.wandb_config.group,
                job_type=self.wandb_config.job_type,
                config=config,
                settings=wandb.Settings(init_timeout=300),
            )
            print(f"wandb run: {self.wandb_run.url}")
        else:
            self.wandb_run = None
            print("Warning: wandb not installed. Skipping wandb initialization.")

    def _log_wandb(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to wandb if enabled."""
        if HAS_WANDB and self.wandb_run is not None:
            wandb.log(metrics, step=step)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Save a training checkpoint.

        Args:
            path: Path to save checkpoint file.
            optimizer: Optimizer to save state from (optional).
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "config": {
                "lr": self.config.lr,
                "n_epochs": self.config.n_epochs,
                "batch_size": self.config.batch_size,
            },
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Save model architecture info if available
        # Unwrap DDP/FSDP to access the underlying model attributes
        raw_model = getattr(self.model, "module", self.model)
        if hasattr(raw_model, "input_dim"):
            checkpoint["input_dim"] = raw_model.input_dim
        if hasattr(raw_model, "hidden_dim"):
            checkpoint["hidden_dim"] = raw_model.hidden_dim
        if hasattr(raw_model, "_get_config"):
            checkpoint["model_config"] = raw_model._get_config()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint file.
            optimizer: Optimizer to load state into (optional).

        Returns:
            Checkpoint dict with metadata (global_step, epoch, etc.)
        """
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)

        # Handle DDP checkpoints (keys prefixed with 'module.') loaded without DDP wrapper
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict) and not any(
            k.startswith("module.") for k in self.model.state_dict()
        ):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    def fit(
        self,
        data: Union[torch.Tensor, DataLoader],
        max_grad_norm: Optional[float] = None,
        resume_from: Optional[Union[str, Path]] = None,
        data_sharded: bool = False,
        **loss_kwargs,
    ) -> float:
        """Train the SAE model.

        Args:
            data: Training data (tensor or dataloader)
            max_grad_norm: Max gradient norm for clipping (overrides config.max_grad_norm if set)
            resume_from: Path to checkpoint to resume training from (optional)
            data_sharded: If True, data is already sharded per rank — skip DistributedSampler
            **loss_kwargs: Additional arguments passed to loss function

        Returns:
            Final training loss
        """
        # Resolve grad clipping: fit() arg overrides config
        if max_grad_norm is None:
            max_grad_norm = self.config.max_grad_norm
        self._data_sharded = data_sharded

        # Setup distributed training first (before moving model to device)
        self._setup_distributed()

        # Setup (if not distributed, move model to device here)
        if not self.is_distributed:
            self.model.to(self.config.device)
        self.model.train()

        self.dataloader = self._setup_dataloader(data)
        optimizer = self._setup_optimizer()
        loss_fn = self._setup_loss_fn()

        # Resume from checkpoint if provided
        if resume_from is not None:
            self._print_rank0(f"Resuming from checkpoint: {resume_from}")
            self.load_checkpoint(resume_from, optimizer)
            self._print_rank0(f"Resumed at step {self.global_step}, epoch {self.current_epoch}")

        # setup wandb (only on rank 0)
        self._setup_wandb()

        # Reset perf_logger if provided
        if self.perf_logger is not None:
            self.perf_logger.reset()

        model = self._get_model()
        self.dead_latent_tracker = DeadLatentTracker(model.hidden_dim, device=self.config.device)

        # Compute global batch size (accounts for gradient accumulation)
        accum_steps = self.config.grad_accumulation_steps
        global_batch_size = self.config.batch_size * self.parallel_config.dp_size * accum_steps

        # Compute total decay steps for LR schedule
        if self.config.lr_decay_steps is not None:
            self._total_decay_steps = self.config.lr_decay_steps
        elif self.config.lr_schedule != "constant":
            if self.config.max_steps is not None:
                # max_steps gives an exact budget (works for streaming too)
                self._total_decay_steps = max(0, self.config.max_steps - self.config.warmup_steps)
            else:
                # Estimate total optimizer steps from dataloader length
                try:
                    batches_per_epoch = len(self.dataloader)
                    steps_per_epoch = batches_per_epoch // accum_steps
                    total_steps = steps_per_epoch * self.config.n_epochs
                    self._total_decay_steps = max(0, total_steps - self.config.warmup_steps)
                except TypeError:
                    self._total_decay_steps = 0
                    self._print_rank0(
                        "WARNING: Cannot compute decay steps for streaming dataloader. "
                        "Set lr_decay_steps or max_steps explicitly, or use lr_schedule='constant'."
                    )
        else:
            self._total_decay_steps = 0

        remaining_info = ""
        if resume_from is not None:
            remaining_info = f" (resuming from epoch {self.current_epoch})"
        if self.config.max_steps is not None:
            self._print_rank0(f"\nTraining SAE for up to {self.config.max_steps:,} steps{remaining_info}...")
        else:
            self._print_rank0(f"\nTraining SAE for {self.config.n_epochs} epochs{remaining_info}...")
        try:
            self._print_rank0(f"Batches per epoch: ~{len(self.dataloader)}")
        except TypeError:
            self._print_rank0("Batches per epoch: unknown (streaming)")
        self._print_rank0(f"Batch size per GPU: {self.config.batch_size}")
        self._print_rank0(f"Global batch size: {global_batch_size}")
        if accum_steps > 1:
            self._print_rank0(f"Gradient accumulation: {accum_steps} microsteps")
        if self.config.warmup_steps > 0:
            self._print_rank0(f"LR warmup: {self.config.warmup_steps} steps")
        if self.config.lr_schedule != "constant":
            self._print_rank0(
                f"LR schedule: {self.config.lr_schedule} decay over {self._total_decay_steps} steps "
                f"(lr_min={self.config.lr_min:.2e})"
            )
        if max_grad_norm is not None:
            self._print_rank0(f"Gradient clipping: max_norm={max_grad_norm}")

        # If resuming, keep restored global_step and current_epoch; otherwise start fresh
        if resume_from is None:
            self.global_step = 0
            start_epoch = 0
        else:
            start_epoch = self.current_epoch
            self._print_rank0(f"Resuming from epoch {start_epoch}, step {self.global_step}")

        epoch_losses = []
        max_steps = self.config.max_steps
        # When max_steps is set, loop epochs indefinitely until the step budget is
        # reached (so duration is controlled by steps, which also works for streaming).
        epoch_iter = (
            itertools.count(start_epoch) if max_steps is not None else range(start_epoch, self.config.n_epochs)
        )
        reached_max_steps = False

        for epoch in epoch_iter:
            if reached_max_steps:
                break
            self.current_epoch = epoch
            batch_losses = []

            # Set epoch for distributed sampler (ensures different shuffling each epoch)
            if self.is_distributed and hasattr(self.dataloader.sampler, "set_epoch"):
                self.dataloader.sampler.set_epoch(epoch)

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.dataloader):
                # Handle batch from TensorDataset
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.config.device)

                micro_step = batch_idx % accum_steps
                is_accum_step = (micro_step == accum_steps - 1) or (
                    batch_idx == len(self.dataloader) - 1 if hasattr(self.dataloader, "__len__") else False
                )

                # Skip DDP gradient allreduce on non-final accumulation microsteps
                maybe_no_sync = (
                    self.model.no_sync if (self.is_distributed and not is_accum_step) else contextlib.nullcontext
                )
                with maybe_no_sync():
                    # Forward pass
                    loss_dict = loss_fn(batch, **loss_kwargs)
                    loss = loss_dict["total"] / accum_steps

                    # Backward pass (DDP allreduce only fires on the final microstep)
                    loss.backward()

                # Track losses (unscaled for logging)
                batch_losses.append(loss_dict["total"].item())

                if not is_accum_step:
                    continue

                # --- Optimizer step (every accum_steps microsteps) ---

                # Update learning rate (handles warmup)
                self._update_lr(optimizer, self.global_step)

                # Sync dead latent stats across GPUs (for auxk loss)
                self._sync_dead_latent_stats()

                # Gradient clipping and norm computation
                grad_norm = None
                if max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                else:
                    # Compute grad norm without clipping for logging
                    grad_norm = self._compute_grad_norm()

                optimizer.step()
                optimizer.zero_grad()

                # Post-step hook (e.g., normalize decoder)
                if hasattr(model, "post_step"):
                    model.post_step()

                # Log with PerfLogger if provided (only on rank 0)
                if self.perf_logger is not None and self.rank == 0:
                    extra_metrics = {
                        "global_batch_size": global_batch_size,
                    }

                    # Dead latents tracking - prefer SAE's internal counter (what auxk uses)
                    if hasattr(model, "stats_last_nonzero") and hasattr(model, "dead_tokens_threshold"):
                        dead_by_auxk = (model.stats_last_nonzero > model.dead_tokens_threshold).float().mean() * 100
                        extra_metrics["dead_latents"] = dead_by_auxk.item()
                    elif self.dead_latent_tracker:
                        dead_stats = self.dead_latent_tracker.get_stats()
                        extra_metrics["dead_latents"] = dead_stats["dead_pct"]

                    # Reset external dead stats tracker periodically (only used as fallback)
                    if self.global_step % 1000 == 0 and self.dead_latent_tracker:
                        self.dead_latent_tracker.reset()

                    # Call log_step on every step (PerfLogger handles its own logging interval)
                    self.perf_logger.log_step(
                        step=self.global_step,
                        batch=batch,
                        loss_dict=loss_dict,
                        grad_norm=grad_norm,
                        lr=optimizer.param_groups[0]["lr"],
                        extra_metrics=extra_metrics,
                    )
                # Fallback to basic wandb logging if no perf_logger (only on rank 0)
                elif self.wandb_run and self.rank == 0 and (self.global_step % self.wandb_config.log_interval == 0):
                    log_dict = {
                        "train/loss": loss.item() * accum_steps,
                        "train/step": self.global_step,
                        "train/global_batch_size": global_batch_size,
                    }
                    for key, value in loss_dict.items():
                        if key != "total":
                            log_dict[f"train/{key}"] = value.item() if torch.is_tensor(value) else value
                    self._log_wandb(log_dict, self.global_step)

                # Checkpointing (only on rank 0)
                if (
                    self.rank == 0
                    and self.config.checkpoint_dir is not None
                    and self.config.checkpoint_steps is not None
                    and self.global_step > 0
                    and self.global_step % self.config.checkpoint_steps == 0
                ):
                    ckpt_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.global_step}.pt"
                    self.save_checkpoint(ckpt_path, optimizer)
                    self._print_rank0(f"Saved checkpoint: {ckpt_path}")

                self.global_step += 1

                if max_steps is not None and self.global_step >= max_steps:
                    reached_max_steps = True
                    break

            # Epoch complete
            avg_loss = np.mean(batch_losses) if batch_losses else float("nan")
            epoch_losses.append(avg_loss)

            # Print progress (only if no perf_logger, as it handles printing)
            if self.perf_logger is None and (epoch + 1) % self.config.log_interval == 0:
                self._print_rank0(f"Epoch {epoch + 1}/{self.config.n_epochs} | Loss: {avg_loss:.6f}")

        # Finalize perf_logger (rank 0 only, other ranks don't track metrics)
        if self.perf_logger is not None and self.rank == 0:
            self.perf_logger.finish()

        # Save final checkpoint (only on rank 0)
        if self.rank == 0 and self.config.checkpoint_dir is not None:
            ckpt_path = Path(self.config.checkpoint_dir) / "checkpoint_final.pt"
            self.save_checkpoint(ckpt_path, optimizer)
            self._print_rank0(f"Saved final checkpoint: {ckpt_path}")

        # Sync all ranks and tear down process group (prevents NCCL timeout during cleanup)
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        # Training complete
        final_loss = epoch_losses[-1] if epoch_losses else float("nan")
        self._print_rank0(f"Training complete! Final loss: {final_loss:.6f}")

        self.model.eval()
        return final_loss

    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5


def train_sae(
    model: nn.Module,
    data: Union[torch.Tensor, DataLoader],
    lr: float = 3e-4,
    n_epochs: int = 10,
    batch_size: int = 4096,
    device: str = "cuda",
    log_interval: int = 1,
    warmup_steps: int = 0,
    max_grad_norm: Optional[float] = None,
    lr_schedule: str = "constant",
    lr_min: float = 0.0,
    **loss_kwargs,
) -> float:
    """Convenience function to train an SAE model.

    This is a simpler interface than using the Trainer class directly.

    Args:
        model: SAE model to train
        data: Training data (tensor or dataloader)
        lr: Learning rate
        n_epochs: Number of epochs
        batch_size: Batch size
        device: Device to train on
        log_interval: Print loss every N epochs
        warmup_steps: Number of steps for linear LR warmup (0 = no warmup)
        max_grad_norm: Max gradient norm for clipping (None = no clipping)
        lr_schedule: LR schedule after warmup ('constant', 'cosine', 'linear')
        lr_min: Minimum LR for decay schedules
        **loss_kwargs: Additional arguments for loss function

    Returns:
        Final training loss

    Example:
        >>> from biosae import TopKSAE
        >>> from biosae.training import train_sae
        >>>
        >>> sae = TopKSAE(input_dim=768, hidden_dim=4096, top_k=32)
        >>> final_loss = train_sae(sae, embeddings, lr=1e-3, n_epochs=10)
    """
    config = TrainingConfig(
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
        log_interval=log_interval,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        lr_schedule=lr_schedule,
        lr_min=lr_min,
    )

    trainer = Trainer(
        model=model,
        config=config,
    )

    return trainer.fit(data, **loss_kwargs)
