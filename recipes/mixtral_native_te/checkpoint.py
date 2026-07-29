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

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import grouped_dcp
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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam as _FSDPParam
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor


# ---------------------------------------------------------------------------
# Monkey-patch FSDP2's FSDPParam.reset_sharded_param to handle QuantizedTensor.
# ---------------------------------------------------------------------------


def _patched_reset_sharded_param(self):  # type: ignore[no-untyped-def]
    """FSDP2 ``FSDPParam.reset_sharded_param`` made safe for TE ``QuantizedTensor`` locals.

    WHY this monkey-patch exists: upstream FSDP2 refreshes a param's padded local shard by inspecting
    the tensor's ``untyped_storage()`` / data_ptr, assuming ordinary dense storage. TE MXFP8
    ``QuantizedTensor`` locals don't expose storage that way and raise on that inspection, so we guard
    the storage checks and update the DTensor's local tensor directly. It patches a *private* FSDP2
    internal, so it's isolated here and should be deleted once PyTorch/FSDP2 handles QuantizedTensor
    natively. This is a pure workaround, not part of the recipe's conceptual design.
    """
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

_ckpt_futures: dict = {}

_EXPERT_KEY_MARKERS = (".experts_gate_up.", ".experts_down.", "._experts_ffn_op.")
_EXPERTS_SUBDIR = "experts"
_EXPERTS_OPTIM_SUBDIR = "experts_optimizer"


class CheckpointOutput(NamedTuple):
    """Output of checkpoint loading."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    dataloader: StatefulDataLoader | None
    step: int
    epoch: int


def is_expert_key(key: str) -> bool:
    """Return True if a model or optimizer state key belongs to expert weights."""
    return any(marker in key for marker in _EXPERT_KEY_MARKERS)


def load_pretrained_state_dict(path: str | os.PathLike) -> dict[str, torch.Tensor]:
    """Memory-map the canonical TE training state dict produced by ``export_hf_state_dict``.

    This is needed only for ``init_from_pretrained``: training checkpoints use DCP through
    :func:`load_checkpoint`. ``models/mixtral/export.py`` owns the conversion format and emits one
    ``torch.save`` file, so this loader deliberately has no legacy format dispatch.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"init_from_pretrained={path} is not a converted TE state-dict file; create it with "
            "models/mixtral/export.export_hf_state_dict"
        )
    # mmap keeps large checkpoints in reclaimable, node-shared page cache instead of anonymous
    # per-rank RSS. This is required for an 8x7B checkpoint under constrained CPU cgroups.
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def load_pretrained_model(
    model: torch.nn.Module,
    path: str | os.PathLike,
    *,
    ep_rank: int,
) -> None:
    """Load converted pretrained weights into an unsharded TE model on one EP rank.

    Call this before FSDP wrapping. The owned Mixtral model API handles runtime EP localization and
    preserves TE's high-precision pretrained values for later FusedAdam master initialization.
    """
    state_dict = load_pretrained_state_dict(path)
    missing, unexpected = model.load_global_state_dict(state_dict, ep_rank=ep_rank, strict=False)

    # Shared fused views (._experts_ffn_op.) and TE _extra_state buffers are expected-missing.
    real_missing = [key for key in missing if "._experts_ffn_op." not in key and not key.endswith("_extra_state")]
    if real_missing:
        logger.warning("Pretrained load: %d unexpected-missing keys, e.g. %s", len(real_missing), real_missing[:5])
    if unexpected:
        logger.warning("Pretrained load: %d unexpected keys, e.g. %s", len(unexpected), list(unexpected)[:5])
    logger.info("Loaded pretrained weights (missing=%d unexpected=%d)", len(missing), len(unexpected))


def filter_non_expert_model_state(state_dict: dict) -> dict:
    """Keep only non-expert model parameters for FSDP2 DCP."""
    return {
        key: value for key, value in state_dict.items() if not is_expert_key(key) and not key.endswith("_extra_state")
    }


def filter_non_expert_optimizer_state(optim_state_dict: dict) -> dict:
    """Keep only non-expert optimizer entries for FSDP2 DCP."""
    filtered = dict(optim_state_dict)
    state = filtered.get("state")
    if isinstance(state, dict):
        filtered["state"] = {key: value for key, value in state.items() if not is_expert_key(key)}
    return filtered


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


@dataclass
class AppState(Stateful):
    """FSDP2 AppState for non-expert model/optimizer state only."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    step: int = 0
    epoch: int = 0

    def state_dict(self):
        """Get the filtered state dict for non-expert params and optimizer state."""
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": filter_non_expert_model_state(model_state_dict),
            "optim": filter_non_expert_optimizer_state(optimizer_state_dict),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: dict):
        """Load non-expert model/optimizer state from checkpoint."""
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


def _fsdp2_process_group(
    ep_mesh: DeviceMesh,
    dp_process_group: torch.distributed.ProcessGroup | None,
    dist_config: DistributedConfig,
) -> torch.distributed.ProcessGroup | None:
    """Return the process group for non-expert FSDP2 DCP save/load."""
    dp_size = dist_config.world_size // ep_mesh.size()
    if dp_size == 1:
        # Replicated non-expert params: all ranks must coordinate one DCP write.
        return None
    return dp_process_group


def _bootstrap_expert_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Allocate FusedAdam expert optimizer slots before consolidated optimizer load."""
    for name, param in model.named_parameters():
        if not is_expert_key(name):
            continue
        if param in optimizer.state and "master_param" in optimizer.state[param]:
            continue
        optimizer.initialize_state(param, store_param_remainders=False)


def _experts_ckpt_dir(checkpoint_path: Path) -> str:
    return str(checkpoint_path / _EXPERTS_SUBDIR)


def _experts_optimizer_ckpt_dir(checkpoint_path: Path) -> str:
    return str(checkpoint_path / _EXPERTS_OPTIM_SUBDIR)


def _expand_stacked_expert_weights(model_state: dict, model: torch.nn.Module, ep_mesh: DeviceMesh) -> None:
    """Expand consolidated expert `.weight` stacks into per-expert `weight{i}` keys."""
    ep_sd = grouped_dcp.build_ep_sharded_state_dict(model, ep_mesh)
    for key, value in ep_sd.items():
        if not key.endswith(".weight") or not is_expert_key(key):
            continue
        if not isinstance(value, DTensor):
            continue
        full_stack = value.full_tensor().cpu()
        prefix = key[: -len(".weight")]
        module_name, attr = prefix.rsplit(".", maxsplit=1)
        module = model.get_submodule(module_name)
        gl = getattr(module, attr)
        if getattr(gl, "weight0", None) is None:
            model_state[key] = full_stack
            continue
        num_experts = 0
        while hasattr(gl, f"weight{num_experts}"):
            num_experts += 1
        for i in range(num_experts):
            model_state[f"{prefix}.weight{i}"] = full_stack[i]


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
    ep_mesh: DeviceMesh,
    dp_process_group: torch.distributed.ProcessGroup | None = None,
    dataloader: StatefulDataLoader | None = None,
) -> CheckpointOutput:
    """Load FSDP2 non-expert checkpoint plus consolidated expert weights/optimizer state."""
    checkpoint_path, _ = get_latest_checkpoint(ckpt_path)
    if not checkpoint_path:
        logger.info("No checkpoint found, starting from scratch")
        return CheckpointOutput(model, optimizer, scheduler, dataloader, 0, 0)

    app_state = AppState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    state_dict = {"app": app_state}
    fsdp_pg = _fsdp2_process_group(ep_mesh, dp_process_group, dist_config)
    dcp_load(state_dict, checkpoint_id=checkpoint_path, process_group=fsdp_pg)

    grouped_dcp.load_consolidated(model, ep_mesh, _experts_ckpt_dir(checkpoint_path))
    experts_opt_dir = Path(_experts_optimizer_ckpt_dir(checkpoint_path))
    if experts_opt_dir.exists():
        _bootstrap_expert_optimizer_state(model, optimizer)
        grouped_dcp.load_optimizer_consolidated(model, optimizer, ep_mesh, str(experts_opt_dir))

    if dataloader is not None:
        load_dataloader(
            dataloader=dataloader,
            ckpt_path=checkpoint_path,
            dist_config=dist_config,
        )

    logger.info(f"Loaded checkpoint from step {app_state.step}")

    return CheckpointOutput(model, optimizer, scheduler, dataloader, app_state.step + 1, app_state.epoch)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str | os.PathLike,
    step: int,
    epoch: int,
    dist_config: DistributedConfig,
    ep_mesh: DeviceMesh,
    dp_process_group: torch.distributed.ProcessGroup | None = None,
    dataloader: StatefulDataLoader | None = None,
    max_checkpoints: int | None = None,
    async_save: bool = False,
) -> None:
    """Save the checkpoint in two complementary formats and prune old ones.

    WHY two formats: dense/non-expert state is standard FSDP2 ``DTensor`` state that stock
    ``torch.distributed.checkpoint`` handles directly, but the EP-partitioned experts are *not* in a
    form DCP can reshard (see ``grouped_dcp``). So we save non-expert model/optimizer state via the
    normal FSDP2 DCP ``AppState``, and route the expert weights + their optimizer slots through
    ``grouped_dcp`` to produce the global, EP-reshardable ``[num_experts, ...]`` representation. The
    two halves land in the same ``step_{step}`` directory and are recombined on load.
    """
    start_time = time.perf_counter()
    ckpt_path = Path(ckpt_path)
    checkpoint_path = ckpt_path / f"step_{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model_params = (p.to_local() if isinstance(p, DTensor) else p for p in model.parameters())
    if async_save and any(isinstance(p, QuantizedTensor) for p in model_params):
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
        logger.info(f"Saved dataloader to {checkpoint_path}")

    grouped_dcp.save_consolidated(model, ep_mesh, _experts_ckpt_dir(checkpoint_path))
    grouped_dcp.save_optimizer_consolidated(model, optimizer, ep_mesh, _experts_optimizer_ckpt_dir(checkpoint_path))

    state_dict = {"app": AppState(model=model, optimizer=optimizer, scheduler=scheduler, step=step, epoch=epoch)}
    fsdp_pg = _fsdp2_process_group(ep_mesh, dp_process_group, dist_config)
    if async_save:
        if "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
            _ckpt_futures["fsdp2"].result()
        _ckpt_futures["fsdp2"] = dcp_async_save(state_dict, checkpoint_id=checkpoint_path, process_group=fsdp_pg)
    else:
        dcp_save(state_dict, checkpoint_id=checkpoint_path, process_group=fsdp_pg)

    if max_checkpoints is not None and dist_config.is_main_process():
        prune_checkpoints(ckpt_path, max_checkpoints)

    if dist_config.is_main_process():
        logger.info(f"Saved checkpoint to {checkpoint_path} in {time.perf_counter() - start_time:.2f} seconds")


def save_final_model(
    model: torch.nn.Module,
    save_directory: str | os.PathLike,
    dist_config: DistributedConfig,
    ep_mesh: DeviceMesh,
) -> None:
    """Gather non-expert FSDP2 weights and consolidated expert weights into safetensors."""
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )
    model_state_dict = filter_non_expert_model_state(model_state_dict)
    _expand_stacked_expert_weights(model_state_dict, model, ep_mesh)

    if not dist_config.is_main_process():
        return

    os.makedirs(save_directory, exist_ok=True)
    save_file(model_state_dict, os.path.join(save_directory, "model.safetensors"))

    underlying_model = model.module if hasattr(model, "module") else model
    if hasattr(underlying_model, "config"):
        underlying_model.config.save_pretrained(save_directory)

    logger.info(f"Saved final model to {save_directory} (weights + config only)")


def save_dataloader(
    dataloader: StatefulDataLoader | None,
    ckpt_path: str | os.PathLike,
    dist_config: DistributedConfig,
):
    """Save the dataloader state to a file."""
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
    """Load the dataloader state from a file."""
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
