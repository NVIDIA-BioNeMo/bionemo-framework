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

"""Checkpoint utilities for CodonFM training."""

import gc
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import torch
from distributed_config import DistributedConfig
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import async_save as dcp_async_save
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.distributed.checkpoint.stateful import Stateful


logger = logging.getLogger(__name__)
_ckpt_futures: dict = {}


class CheckpointOutput(NamedTuple):
    """Output of checkpoint loading."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    step: int
    epoch: int


# ============================================================================
# Helper functions
# ============================================================================


def get_latest_checkpoint(ckpt_path: str | os.PathLike) -> tuple[Path | None, int]:
    """Get the latest checkpoint path and step number."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return None, 0

    checkpoints = [f for f in ckpt_path.iterdir() if f.name.startswith("step_")]
    if not checkpoints:
        return None, 0

    latest = max(checkpoints, key=lambda x: int(Path(x).stem.split("_")[1]))
    step = int(Path(latest).stem.split("_")[1])
    return latest, step


def should_save_checkpoint(step: int, save_every_n_steps: int) -> bool:
    """Determine if a checkpoint should be saved."""
    return save_every_n_steps > 0 and step % save_every_n_steps == 0 and step > 0


def prune_checkpoints(ckpt_path: str | os.PathLike, max_checkpoints: int) -> None:
    """Prune checkpoints to keep only the latest `max_checkpoints` checkpoints."""
    ckpt_path = Path(ckpt_path)
    checkpoints = [f for f in ckpt_path.iterdir() if f.name.startswith("step_")]
    checkpoints.sort(key=lambda x: int(Path(x).stem.split("_")[1]))
    if len(checkpoints) > max_checkpoints:
        for checkpoint in checkpoints[:-max_checkpoints]:
            logger.info(f"Pruning checkpoint {checkpoint}")
            if checkpoint.is_dir():
                shutil.rmtree(checkpoint)
            else:
                os.remove(checkpoint)


# ============================================================================
# FSDP2 Checkpointing
# ============================================================================


@dataclass
class AppState(Stateful):
    """AppState for FSDP2 checkpoint."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    step: int = 0
    epoch: int = 0
    state_dict_options: StateDictOptions = field(
        default_factory=lambda: StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            strict=False,
        )
    )

    def state_dict(self):
        """Get the state dict."""
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer, options=self.state_dict_options
        )
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: dict):
        """Load the state dict."""
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=self.state_dict_options,
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]


def load_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
    process_group: torch.distributed.ProcessGroup | None = None,
) -> CheckpointOutput:
    """Load FSDP2 checkpoint."""
    checkpoint_path, _ = get_latest_checkpoint(ckpt_path)
    if not checkpoint_path:
        logger.info("No FSDP2 checkpoint found, starting from scratch")
        return CheckpointOutput(model, optimizer, scheduler, 0, 0)

    app_state = AppState(model=model, optimizer=optimizer, scheduler=scheduler)
    state_dict = {"app": app_state}
    dcp_load(state_dict, checkpoint_id=checkpoint_path, process_group=process_group)

    logger.info(f"Loaded distributed FSDP2 checkpoint from step {app_state.step}")
    return CheckpointOutput(model, optimizer, scheduler, app_state.step + 1, app_state.epoch)


def save_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    epoch: int,
    dist_config: DistributedConfig,
    process_group: torch.distributed.ProcessGroup | None = None,
    max_checkpoints: int | None = None,
    async_save: bool = False,
) -> None:
    """Save FSDP2 checkpoint."""
    ckpt_path = Path(ckpt_path)
    checkpoint_path = ckpt_path / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if async_save and "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
        _ckpt_futures["fsdp2"].result()

    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier(group=process_group)

    state_dict = {"app": AppState(model=model, optimizer=optimizer, scheduler=scheduler, step=step, epoch=epoch)}
    ckpt_save_func = dcp_async_save if async_save else dcp_save
    _ckpt_futures["fsdp2"] = ckpt_save_func(state_dict, checkpoint_id=checkpoint_path, process_group=process_group)

    if dist_config.is_main_process():
        logger.info(f"Saved distributed FSDP2 checkpoint to {checkpoint_path}")

    if max_checkpoints is not None and dist_config.is_main_process():
        prune_checkpoints(ckpt_path, max_checkpoints)


def save_final_model_fsdp2(
    model: torch.nn.Module,
    config,
    save_directory: str | os.PathLike,
    dist_config: DistributedConfig,
) -> None:
    """Save final model for FSDP2."""
    # Ensure any outstanding async checkpoint completes before saving the final model.
    if "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
        _ckpt_futures["fsdp2"].result()
        _ckpt_futures["fsdp2"] = None

    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    if not dist_config.is_main_process():
        return

    os.makedirs(save_directory, exist_ok=True)
    save_file(model_state_dict, os.path.join(save_directory, "model.safetensors"))
    config.to_json_file(os.path.join(save_directory, "config.json"))
    logger.info(f"Saved final FSDP2 model to {save_directory}")
