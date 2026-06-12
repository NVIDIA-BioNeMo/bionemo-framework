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
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
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
from torchdata.stateful_dataloader import StatefulDataLoader

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


class LenientLoadPlanner(DefaultLoadPlanner):
    """A load planner that skips keys missing from the checkpoint.

    Handles checkpoints saved without TransformerEngine _extra_state keys
    (FP8 metadata). These keys are registered by newer TE versions even when
    FP8 is disabled, but older checkpoints don't contain them.
    """

    def create_local_plan(self):
        """Create a local load plan, skipping keys missing from the checkpoint."""
        missing_keys = [fqn for fqn in self.state_dict if fqn not in self.metadata.state_dict_metadata]
        if missing_keys:
            logger.warning(
                "Skipping %d keys not found in checkpoint: %s%s",
                len(missing_keys),
                missing_keys[:5],
                "..." if len(missing_keys) > 5 else "",
            )
            for key in missing_keys:
                del self.state_dict[key]
        return super().create_local_plan()


# Tracks in-flight async checkpoint futures keyed by strategy name (e.g. "fsdp2").
# Each entry holds the Future returned by dcp_async_save so we can await it before starting
# the next async save or before shutting down.
_ckpt_futures: dict = {}


class CheckpointOutput(NamedTuple):
    """Output of checkpoint loading."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    dataloader: StatefulDataLoader | None
    step: int
    epoch: int


# ============================================================================
# Helper functions
# ============================================================================


def get_latest_checkpoint(ckpt_path: str | os.PathLike) -> tuple[Path | None, int]:
    """Get the latest checkpoint path and step number.

    Returns:
        Tuple of (checkpoint path, step number).
        If no checkpoint files are found, returns (None, 0).
    """
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
    if save_every_n_steps > 0 and step % save_every_n_steps == 0 and step > 0:
        return True
    return False


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
    """AppState for FSDP2 checkpoint.

    Adapted from https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    step: int = 0
    epoch: int = 0

    def state_dict(self):
        """Get the state dict for the model, optimizer, scheduler, and step."""
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: dict):
        """Load the state dict for the model, optimizer, scheduler, and step."""
        # Use strict=False to handle checkpoints saved without TransformerEngine
        # _extra_state keys (FP8 metadata). These keys are registered by newer TE
        # versions even when FP8 is disabled, and are safe to skip.
        incompatible = set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(strict=False),
        )
        if incompatible and (incompatible.missing_keys or incompatible.unexpected_keys):
            if incompatible.missing_keys:
                logger.warning(f"Missing keys when loading checkpoint: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.warning(f"Unexpected keys when loading checkpoint: {incompatible.unexpected_keys}")
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]


def load_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
    process_group: torch.distributed.ProcessGroup | None = None,
) -> CheckpointOutput:
    """Load FSDP2 checkpoint.

    Args:
        model: The model to load.
        optimizer: The optimizer to load.
        scheduler: The LR scheduler to load.
        ckpt_path: The directory containing checkpoints.
        dist_config: The distributed configuration.
        dataloader: The dataloader to load.
        process_group: The process group to use for checkpointing.
    """
    checkpoint_path, _ = get_latest_checkpoint(ckpt_path)
    if not checkpoint_path:
        logger.info("No FSDP2 checkpoint found, starting from scratch")
        return CheckpointOutput(model, optimizer, scheduler, dataloader, 0, 0)

    app_state = AppState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    state_dict = {"app": app_state}
    dcp_load(state_dict, checkpoint_id=checkpoint_path, process_group=process_group, planner=LenientLoadPlanner())

    if dataloader is not None:
        load_dataloader(
            dataloader=dataloader,
            ckpt_path=checkpoint_path,
            dist_config=dist_config,
        )

    logger.info(f"Loaded distributed FSDP2 checkpoint from step {app_state.step}")

    # Increment the step by one to avoid re-running the previous step.
    return CheckpointOutput(model, optimizer, scheduler, dataloader, app_state.step + 1, app_state.epoch)


def save_checkpoint_fsdp2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    epoch: int,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
    process_group: torch.distributed.ProcessGroup | None = None,
    max_checkpoints: int | None = None,
    async_save: bool = False,
) -> None:
    """Save FSDP2 checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The LR scheduler to save.
        ckpt_path: The directory to save the checkpoint.
        step: The step number to save the checkpoint.
        epoch: The epoch number to save the checkpoint.
        dist_config: The distributed configuration.
        dataloader: The dataloader to save.
        process_group: The process group to use for checkpointing.
        max_checkpoints: The maximum number of checkpoints to keep.
        async_save: Whether to save the checkpoint asynchronously.
    """
    start_time = time.perf_counter()
    ckpt_path = Path(ckpt_path)
    checkpoint_path = ckpt_path / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if dataloader is not None:
        save_dataloader(
            dataloader=dataloader,
            ckpt_path=checkpoint_path,
            dist_config=dist_config,
        )
        logger.info(f"Saved FSDP2 dataloader to {ckpt_path}")

    state_dict = {"app": AppState(model=model, optimizer=optimizer, scheduler=scheduler, step=step, epoch=epoch)}
    if async_save:
        # If we're using asynchronous checkpointing, make sure we only have one checkpoint future at a time.
        if "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
            _ckpt_futures["fsdp2"].result()

        _ckpt_futures["fsdp2"] = dcp_async_save(state_dict, checkpoint_id=checkpoint_path, process_group=process_group)
    else:
        dcp_save(state_dict, checkpoint_id=checkpoint_path, process_group=process_group)

    if max_checkpoints is not None and dist_config.is_main_process():
        prune_checkpoints(ckpt_path, max_checkpoints)

    if dist_config.is_main_process():
        logger.info(
            f"Saved distributed FSDP2 checkpoint to {checkpoint_path} "
            f"in {time.perf_counter() - start_time:.2f} seconds"
        )


def save_final_model_fsdp2(
    model: torch.nn.Module,
    save_directory: str | os.PathLike,
    dist_config: DistributedConfig,
) -> None:
    """Save final model for FSDP2 - gather on all ranks, save on main."""
    # ALL ranks must participate in gathering
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )

    # Only main process saves
    if not dist_config.is_main_process():
        return

    os.makedirs(save_directory, exist_ok=True)

    # Save just the weights using safetensors
    save_file(model_state_dict, os.path.join(save_directory, "model.safetensors"))

    # Save the config
    underlying_model = model.module if hasattr(model, "module") else model
    if hasattr(underlying_model, "config"):
        underlying_model.config.save_pretrained(save_directory)

    logger.info(f"Saved final FSDP2 model to {save_directory} (weights + config only)")


# ============================================================================
# Dataloader Checkpointing
# ============================================================================


def save_dataloader(
    dataloader: StatefulDataLoader | None,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
):
    """Save the dataloader state to a file.

    Args:
        dataloader: The dataloader to save the state of.
        ckpt_path: The path to save the dataloader state to.
        dist_config: The distributed configuration.
    """
    if dataloader is None:
        return

    ckpt_path = Path(ckpt_path)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    dataloader_path = ckpt_path / f"dataloader_rank_{dist_config.rank}.pt"

    dataloader_state = dataloader.state_dict()
    dataloader_state["num_workers"] = dataloader.num_workers
    dataloader_state["num_ranks"] = dist_config.world_size
    torch.save(dataloader_state, dataloader_path)
    if dist_config.is_main_process():
        logger.info(f"Saved dataloader state to {dataloader_path}")


def load_dataloader(
    dataloader: StatefulDataLoader | None,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
) -> StatefulDataLoader | None:
    """Load the dataloader state from a file.

    Args:
        dataloader: The dataloader to load the state of.
        ckpt_path: The path to load the dataloader state from.
        dist_config: The distributed configuration.
    """
    if dataloader is None:
        return dataloader

    dataloader_path = Path(ckpt_path) / f"dataloader_rank_{dist_config.rank}.pt"
    if not dataloader_path.exists():
        logger.warning(
            f"No dataloader checkpoint found for rank {dist_config.rank}, starting dataloader from scratch."
        )
        return dataloader

    dataloader_state = torch.load(dataloader_path, weights_only=True)

    if (
        dataloader.num_workers != dataloader_state["num_workers"]
        or dist_config.world_size != dataloader_state["num_ranks"]
    ):
        logger.warning(
            f"Dataloader num_workers mismatch: {dataloader.num_workers} != {dataloader_state['num_workers']} or "
            f"num_ranks mismatch: {dist_config.world_size} != {dataloader_state['num_ranks']}, "
            "starting dataloader from scratch."
        )
        return dataloader

    dataloader.load_state_dict(dataloader_state)
    if dist_config.is_main_process():
        logger.info(f"Loaded dataloader state from {dataloader_path}")

    return dataloader
