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

import gc
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import torch
import transformers
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
from torchdata.stateful_dataloader import StatefulDataLoader

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)
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
# DDP Checkpointing
# ============================================================================


def load_checkpoint_ddp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
) -> CheckpointOutput:
    """Load DDP checkpoint."""
    checkpoint_path, _ = get_latest_checkpoint(ckpt_path)

    if not checkpoint_path:
        logger.info("No DDP checkpoint found, starting from scratch")
        return CheckpointOutput(model, optimizer, scheduler, dataloader, 0, 0)

    checkpoint = torch.load(
        checkpoint_path / "checkpoint.pt",
        map_location=f"cuda:{dist_config.local_rank}",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    dataloader = load_dataloader(dataloader, checkpoint_path, dist_config)
    step = checkpoint["step"]
    epoch = checkpoint["epoch"]

    if dist_config.is_main_process():
        logger.info(f"Loaded DDP checkpoint from step {step}")

    # Increment the step by one to avoid re-running the previous step.
    return CheckpointOutput(model, optimizer, scheduler, dataloader, step + 1, epoch)


def save_checkpoint_ddp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    epoch: int,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
    max_checkpoints: int | None = None,
) -> None:
    """Saves the Dataloader state and the DDP checkpoint."""
    ckpt_path = Path(ckpt_path)
    checkpoint_path = ckpt_path / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Dataloader checkpointing needs to happen on all ranks, while DDP model checkpointing only needs to happen on the
    # main process.
    save_dataloader(dataloader, checkpoint_path, dist_config)

    if not dist_config.is_main_process():
        return

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        checkpoint_path / "checkpoint.pt",
    )

    logger.info(f"Saved DDP checkpoint to {checkpoint_path}")

    if max_checkpoints is not None and dist_config.is_main_process():
        prune_checkpoints(ckpt_path, max_checkpoints)


def save_final_model_ddp(
    model: torch.nn.Module,
    save_directory: str | os.PathLike,
    dist_config: DistributedConfig,
) -> None:
    """Save final model for DDP - only on main process."""
    if not dist_config.is_main_process():
        return

    # Unwrap model if wrapped
    underlying_model: transformers.PreTrainedModel = model.module if hasattr(model, "module") else model  # type: ignore

    os.makedirs(save_directory, exist_ok=True)
    # If we are saving a PEFT model we also save the base_model config.
    # This allows for an streamlined reload of the PEFT model without having to manually reconstruct the config of
    # the base_model.
    # For example:
    # >>> config = AutoConfig.from_pretrained(<save_directory>)
    # >>> base_model = AutoModelForTokenClassification.from_pretrained(<model.tag>, config=config)
    # >>> peft_model = PeftModel.from_pretrained(base_model, <save_directory>)
    if hasattr(underlying_model, "peft_config"):
        underlying_model.config.save_pretrained(save_directory)
    underlying_model.save_pretrained(save_directory, state_dict=underlying_model.state_dict(), safe_serialization=True)
    logger.info(f"Saved final DDP model to {save_directory}")


# ============================================================================
# mFSDP Checkpointing
# ============================================================================


def load_checkpoint_mfsdp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
) -> CheckpointOutput:
    """Load mFSDP distributed checkpoint.

    Args:
        model: The model to load.
        optimizer: The optimizer to load.
        scheduler: The LR scheduler to load.
        ckpt_path: The directory containing checkpoints.
        dist_config: The distributed configuration.
        dataloader: The dataloader to load.

    Returns:
        Tuple of (model, optimizer, scheduler, step).
    """
    checkpoint_path, step = get_latest_checkpoint(ckpt_path)
    if not checkpoint_path:
        logger.info("No mFSDP checkpoint found, starting from scratch")
        return CheckpointOutput(model, optimizer, scheduler, dataloader, 0, 0)

    ckpt_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metadata": {
            "step": step,  # Initialize with current step from filename
            "epoch": 0,  # Initialize with default epoch
        },
    }
    torch.distributed.checkpoint.load(state_dict=ckpt_state_dict, checkpoint_id=checkpoint_path)

    model.load_state_dict(ckpt_state_dict["model"], strict=False)
    optimizer.load_state_dict(ckpt_state_dict["optimizer"])
    scheduler.load_state_dict(ckpt_state_dict["scheduler"])
    dataloader = load_dataloader(dataloader, checkpoint_path, dist_config)

    step = ckpt_state_dict["metadata"]["step"]
    epoch = ckpt_state_dict["metadata"]["epoch"]

    # Ensure all ranks have completed loading before proceeding
    torch.distributed.barrier()

    logger.info(f"Loaded mFSDP checkpoint from step {step}")

    # Increment the step by one to avoid re-running the previous step.
    return CheckpointOutput(model, optimizer, scheduler, dataloader, step + 1, epoch)


def save_checkpoint_mfsdp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
    epoch: int = 0,
    max_checkpoints: int | None = None,
) -> None:
    """Save mFSDP distributed checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The LR scheduler to save.
        ckpt_path: The directory to save the checkpoint.
        step: The step number to save the checkpoint.
        dist_config: The distributed configuration.
        dataloader: The dataloader to save.
        epoch: The epoch number to save the checkpoint.
        max_checkpoints: The maximum number of checkpoints to keep.
    """
    ckpt_path = Path(ckpt_path)
    checkpoint_path = ckpt_path / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save dataloader state, if provided.
    save_dataloader(dataloader, checkpoint_path, dist_config)

    # Save model, optimizer, scheduler state, and metadata
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metadata": {
            "step": step,
            "epoch": epoch,
        },
    }

    torch.distributed.checkpoint.save(state_dict, checkpoint_id=checkpoint_path)

    if dist_config.is_main_process():
        logger.info(f"Saved mFSDP checkpoint to {checkpoint_path}")

    if max_checkpoints is not None and dist_config.is_main_process():
        prune_checkpoints(ckpt_path, max_checkpoints)


def save_final_model_mfsdp(
    model: torch.nn.Module,
    save_directory: str | os.PathLike,
    dist_config: DistributedConfig,
) -> None:
    """Save final model for mFSDP - requires parameter gathering on all ranks."""
    from megatron_fsdp.uneven_dtensor import gather_uneven_dtensor_to_full_tensor

    if dist_config.is_main_process():
        logger.info("Starting mFSDP parameter gathering...")

    # Parameter gathering must happen on ALL processes
    unsharded_state_dict = {
        # Gather all parameters to CPU, and remove the "module." prefix from the Megatron-FSDP class wrapper.
        k.removeprefix("module."): gather_uneven_dtensor_to_full_tensor(
            v, target_device=torch.device("cpu")
        ).to_local()
        if isinstance(v, torch.distributed.tensor.DTensor)
        else v
        for k, v in model.state_dict().items()
    }

    # Only main process saves the model
    if not dist_config.is_main_process():
        return

    os.makedirs(save_directory, exist_ok=True)
    model.module.save_pretrained(save_directory, state_dict=unsharded_state_dict, safe_serialization=True)
    logger.info(f"Saved final mFSDP model to {save_directory}")


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
    state_dict_options: StateDictOptions = field(
        default_factory=lambda: StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        )
    )

    def state_dict(self):
        """Get the state dict for the model, optimizer, scheduler, and step."""
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
        """Load the state dict for the model, optimizer, scheduler, and step."""
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
    dcp_load(state_dict, checkpoint_id=checkpoint_path, process_group=process_group)

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

    # If we're using asynchronous checkpointing, make sure we only have one checkpoint future at a time.
    if async_save and "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
        _ckpt_futures["fsdp2"].result()

    # Clear GPU cache before checkpointing to free up fragmented memory.
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

    For resuming training with long epochs, we save the dataloader state as part of the checkpoint to allow for resuming
    from the exact same step. Here we save the dataloader state based on global rank. Note, the total number of ranks
    and dataloader num_workers should match for resuming training.

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

    Here we load the dataloader state based on global rank.

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

    dataloader_state = torch.load(dataloader_path)

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
