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
import transformers
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
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor


logger = logging.getLogger(__name__)

# Tracks in-flight async checkpoint futures keyed by strategy name (e.g. "fsdp2").
# Each entry holds the Future returned by dcp_async_save so we can await it before starting
# the next async save or before shutting down.
_ckpt_futures: dict = {}


def get_fsdp2_checkpoint_process_group(
    process_group: torch.distributed.ProcessGroup | None,
    *,
    expert_parallel_size: int,
) -> torch.distributed.ProcessGroup | None:
    """Choose the process group used by DCP for FSDP2 checkpoints.

    When EP is active, the state dict contains DTensor expert weights that are sharded on the
    EP mesh as well as FSDP shards on the DP mesh. DCP must therefore coordinate across the
    full world instead of only the DP subgroup.
    """
    if expert_parallel_size > 1:
        return None
    return process_group


def get_ddp_model_checkpoint_path(ckpt_path: Path, rank: int) -> Path:
    """Return the per-rank model checkpoint path for DDP EP checkpoints."""
    return ckpt_path / f"model_rank_{rank}.pt"


def get_ddp_optimizer_checkpoint_path(ckpt_path: Path, rank: int) -> Path:
    """Return the per-rank optimizer checkpoint path for DDP EP checkpoints."""
    return ckpt_path / f"optimizer_rank_{rank}.pt"


def get_ddp_metadata_checkpoint_path(ckpt_path: Path) -> Path:
    """Return the shared metadata checkpoint path for DDP EP checkpoints."""
    return ckpt_path / "metadata.pt"


def unwrap_checkpoint_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip compile/DDP wrappers until we reach the underlying module."""
    unwrapped = model
    while True:
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod  # type: ignore[attr-defined]
            continue
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module  # type: ignore[attr-defined]
            continue
        return unwrapped


def build_unfiltered_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Build a model state dict without losing EP expert parameters or TE extra state."""
    model_state_dict = torch.nn.Module.state_dict(model)
    for name, param in model.named_parameters():
        if name not in model_state_dict:
            model_state_dict[name] = param.detach().clone()
    for name, buffer in model.named_buffers():
        if name not in model_state_dict:
            model_state_dict[name] = buffer.detach().clone()
    return model_state_dict


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
# DDP Checkpointing
# ============================================================================


def load_checkpoint_ddp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
    dataloader: StatefulDataLoader | None = None,
    weights_only: bool = True,
    expert_parallel_size: int = 1,
) -> CheckpointOutput:
    """Load DDP checkpoint.

    Args:
        model: The model to load.
        optimizer: The optimizer to load.
        scheduler: The LR scheduler to load.
        ckpt_path: The path to the checkpoint.
        dist_config: The distributed configuration.
        dataloader: The dataloader to load.
        weights_only: Whether to load the checkpoint weights only. We have to set this to True when loading FP8
            checkpoints.
        expert_parallel_size: Expert parallelism size. When > 1, loads expert weights with EP-aware state dict handling.
    """
    checkpoint_path, _ = get_latest_checkpoint(ckpt_path)

    if not checkpoint_path:
        logger.info("No DDP checkpoint found, starting from scratch")
        return CheckpointOutput(model, optimizer, scheduler, dataloader, 0, 0)

    if expert_parallel_size > 1:
        underlying_model = unwrap_checkpoint_model(model)
        underlying_model.load_state_dict(
            torch.load(
                get_ddp_model_checkpoint_path(checkpoint_path, dist_config.rank),
                map_location=f"cuda:{dist_config.local_rank}",
                weights_only=weights_only,
            )
        )
        optimizer.load_state_dict(
            torch.load(
                get_ddp_optimizer_checkpoint_path(checkpoint_path, dist_config.rank),
                map_location=f"cuda:{dist_config.local_rank}",
                weights_only=weights_only,
            )
        )
        metadata = torch.load(
            get_ddp_metadata_checkpoint_path(checkpoint_path),
            map_location=f"cuda:{dist_config.local_rank}",
            weights_only=weights_only,
        )
        if metadata["world_size"] != dist_config.world_size:
            raise RuntimeError(
                "DDP EP checkpoints require the same world size when resuming: "
                f"{metadata['world_size']} != {dist_config.world_size}"
            )
        scheduler.load_state_dict(metadata["scheduler"])
        dataloader = load_dataloader(dataloader, checkpoint_path, dist_config)
        step = metadata["step"]
        epoch = metadata["epoch"]
    else:
        checkpoint = torch.load(
            checkpoint_path / "checkpoint.pt",
            map_location=f"cuda:{dist_config.local_rank}",
            weights_only=weights_only,
        )

        underlying_model = unwrap_checkpoint_model(model)
        underlying_model.load_state_dict(checkpoint["model"])
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
    expert_parallel_size: int = 1,
) -> None:
    """Saves the Dataloader state and the DDP checkpoint."""
    ckpt_path = Path(ckpt_path)
    checkpoint_path = ckpt_path / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Dataloader checkpointing needs to happen on all ranks, while DDP model checkpointing only needs to happen on the
    # main process.
    save_dataloader(dataloader, checkpoint_path, dist_config)

    if expert_parallel_size == 1 and not dist_config.is_main_process():
        return

    if expert_parallel_size > 1:
        underlying_model = unwrap_checkpoint_model(model)
        torch.save(
            build_unfiltered_model_state_dict(underlying_model),
            get_ddp_model_checkpoint_path(checkpoint_path, dist_config.rank),
        )
        torch.save(
            optimizer.state_dict(),
            get_ddp_optimizer_checkpoint_path(checkpoint_path, dist_config.rank),
        )
        if dist_config.is_main_process():
            torch.save(
                {
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "world_size": dist_config.world_size,
                },
                get_ddp_metadata_checkpoint_path(checkpoint_path),
            )
            logger.info(f"Saved distributed DDP checkpoint to {checkpoint_path}")
    else:
        underlying_model = unwrap_checkpoint_model(model)

        model_state_dict = build_unfiltered_model_state_dict(underlying_model)

        torch.save(
            {
                # Use nn.Module.state_dict to preserve Mixtral EP expert weights and TE _extra_state.
                "model": model_state_dict,
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
    underlying_model: transformers.PreTrainedModel = unwrap_checkpoint_model(model)  # type: ignore

    os.makedirs(save_directory, exist_ok=True)
    underlying_model.save_pretrained(save_directory, state_dict=underlying_model.state_dict(), safe_serialization=True)
    logger.info(f"Saved final DDP model to {save_directory}")


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
        model_state_dict = {k: v for k, v in model_state_dict.items() if not k.endswith("_extra_state")}
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
            options=StateDictOptions(strict=False),
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
    expert_parallel_size: int = 1,
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
        expert_parallel_size: Expert parallelism size. When > 1, loads expert weights with EP-aware state dict handling.
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
    dcp_load(
        state_dict,
        checkpoint_id=checkpoint_path,
        process_group=get_fsdp2_checkpoint_process_group(process_group, expert_parallel_size=expert_parallel_size),
    )

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
    expert_parallel_size: int = 1,
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
        expert_parallel_size: Expert parallelism size. When > 1, saves expert weights with EP-aware state dict handling.
    """
    start_time = time.perf_counter()
    ckpt_path = Path(ckpt_path)
    checkpoint_path = ckpt_path / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model_params = (p.to_local() if isinstance(p, DTensor) else p for p in model.parameters())
    if async_save and any((isinstance(p, QuantizedTensor) for p in model_params)):
        logger.warning(
            "Async checkpointing is not supported for FP8 models, falling back to synchronous checkpointing."
        )
        async_save = False

    if dataloader is not None:
        save_dataloader(
            dataloader=dataloader,
            ckpt_path=checkpoint_path,
            dist_config=dist_config,
        )
        logger.info(f"Saved FSDP2 dataloader to {ckpt_path}")

    state_dict = {"app": AppState(model=model, optimizer=optimizer, scheduler=scheduler, step=step, epoch=epoch)}
    checkpoint_process_group = get_fsdp2_checkpoint_process_group(
        process_group,
        expert_parallel_size=expert_parallel_size,
    )
    if async_save:
        # If we're using asynchronous checkpointing, make sure we only have one checkpoint future at a time.
        if "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
            _ckpt_futures["fsdp2"].result()

        _ckpt_futures["fsdp2"] = dcp_async_save(
            state_dict,
            checkpoint_id=checkpoint_path,
            process_group=checkpoint_process_group,
        )
    else:
        dcp_save(state_dict, checkpoint_id=checkpoint_path, process_group=checkpoint_process_group)

    if max_checkpoints is not None and dist_config.is_main_process():
        prune_checkpoints(ckpt_path, max_checkpoints)

    if dist_config.is_main_process():
        logger.info(
            f"Saved distributed FSDP2 checkpoint to {checkpoint_path} in {time.perf_counter() - start_time:.2f} seconds"
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
