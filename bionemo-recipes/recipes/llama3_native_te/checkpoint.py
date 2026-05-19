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
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam as _FSDPParam
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

from distributed_config import DistributedConfig


# ---------------------------------------------------------------------------
# Monkey-patch FSDP2's FSDPParam.reset_sharded_param to handle QuantizedTensor.
#
# After checkpoint load, set_state_dict calls copy_() on FSDP-sharded params.
# For QuantizedTensor (MXFP8Tensor), copy_() re-quantizes which can invalidate
# the old untyped_storage, causing data_ptr() to crash. The original code
# (PyTorch _fsdp_param.py) compares storage pointers without guarding against
# QuantizedTensor. This patch wraps the comparison in a try/except so that
# reset_sharded_param can proceed normally (re-recording _sharded_param_data).
# ---------------------------------------------------------------------------


def _patched_reset_sharded_param(self):  # type: ignore[no-untyped-def]
    """reset_sharded_param with QuantizedTensor safety."""
    module_info = self._module_info
    new_param = getattr(module_info.module, module_info.param_name)
    if new_param is not self.sharded_param:
        if torch.__future__.get_swap_module_params_on_conversion():
            raise AssertionError(
                f"Expects swap_tensors to preserve object but got {new_param} instead of {self.sharded_param}"
            )
        self.sharded_param = new_param

    local_tensor = new_param._local_tensor
    if local_tensor.is_meta:
        return

    updated_local_tensor = False
    same_local_tensor = False

    if type(self._sharded_param_data) is torch.Tensor:
        try:
            same_local_tensor = (
                self._sharded_param_data.untyped_storage().data_ptr() > 0
                and self._sharded_param_data.untyped_storage().data_ptr() == local_tensor.untyped_storage().data_ptr()
            )
        except RuntimeError:
            # QuantizedTensor (e.g. MXFP8Tensor) can have invalid storage
            # after copy_() re-quantization. Treat as not-same so that
            # _sharded_param_data gets re-recorded below.
            same_local_tensor = False

    padded_sharded_size = self.padded_sharded_param_size
    shard_dim = self.fsdp_placement.dim
    length = local_tensor.size(shard_dim) if local_tensor.numel() > 0 else 0

    if local_tensor.size() != padded_sharded_size and not same_local_tensor:
        if shard_dim != 0:
            raise AssertionError(f"Shard({shard_dim}) requires even sharding: {local_tensor.size()=}")
        padded_local_tensor = local_tensor.new_zeros(padded_sharded_size)
        padded_local_tensor.narrow(dim=shard_dim, start=0, length=length).copy_(local_tensor)
        local_tensor = padded_local_tensor
        updated_local_tensor = True

    if self.pin_memory and not local_tensor.is_pinned():
        local_tensor = local_tensor.cpu().pin_memory()
        updated_local_tensor = True

    if not same_local_tensor:
        self._sharded_param_data = local_tensor.view(-1)

    if not isinstance(self.sharded_param, DTensor):
        raise AssertionError(f"Expected DTensor, got {type(self.sharded_param)}")

    if updated_local_tensor:
        self.sharded_param._local_tensor = local_tensor.narrow(dim=shard_dim, start=0, length=length)
        if not self.sharded_param._local_tensor.is_contiguous():
            raise AssertionError("Expected sharded_param._local_tensor to be contiguous")

    self._sharding_spec = self.sharded_param._spec


_FSDPParam.reset_sharded_param = _patched_reset_sharded_param


logger = logging.getLogger(__name__)

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
    """
    checkpoint_path, _ = get_latest_checkpoint(ckpt_path)

    if not checkpoint_path:
        logger.info("No DDP checkpoint found, starting from scratch")
        return CheckpointOutput(model, optimizer, scheduler, dataloader, 0, 0)

    checkpoint = torch.load(
        checkpoint_path / "checkpoint.pt",
        map_location=f"cuda:{dist_config.local_rank}",
        weights_only=weights_only,
    )

    model.load_state_dict(checkpoint["model"])
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
