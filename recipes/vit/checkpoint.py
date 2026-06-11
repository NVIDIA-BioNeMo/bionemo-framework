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
from pathlib import Path

import torch
import torch.distributed.checkpoint


_logger = logging.getLogger(__name__)


def load_torch_checkpoint(checkpoint_path, model, megatron_fsdp=False):
    """Load a Torch checkpoint from checkpoint_path into an unsharded model.
    Used for converting existing TIMM or Torch checkpoints into a freshly initialized
    model prior to sharding with Megatron-FSDP.

    If the checkpoint was created from a Megatron-FSDP DCP checkpoint, then setting
    megatron_fsdp=True is required and strips a "module." prefix from the keys.

    Docs: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    # Load model checkpoint. Must load with weights_only=False
    # if you have an optimizer state in your checkpoint.
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    # Remove the "module." prefix from the keys of checkpoints
    # derived from Megatron-FSDP.
    # TODO(@cspades): Remove this when the Megatron-FSDP checkpoint naming is fixed.
    model_checkpoint = {(k.removeprefix("module.") if megatron_fsdp else k): v for k, v in checkpoint["model"].items()}
    # Warn about Megatron-FSDP checkpoints.
    first_key = next(iter(model_checkpoint))
    if first_key.startswith("module.") and not megatron_fsdp:
        _logger.warning(
            f"Checkpoint state dictionary keys ({first_key}) may be prefixed "
            "with 'module.' if converted from a Megatron-FSDP DCP checkpoint."
            "Set megatron_fsdp=True to automatically strip the prefix."
        )
    # Load with strict=False because the checkpoint may have
    # TE-specific keys that are not necessary for inference.
    model.load_state_dict(model_checkpoint, strict=False)


def load_dcp_checkpoint(checkpoint_path, model=None, optimizer=None):
    """Load a Torch DCP checkpoint from checkpoint_path into model and optimizer.

    Docs: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
    """
    # Load model and optimizer checkpoints.
    state_dict = {}
    if model is not None:
        state_dict["model"] = model.state_dict()
    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.distributed.checkpoint.load(state_dict, checkpoint_id=checkpoint_path)
    if model is not None:
        model.load_state_dict(state_dict["model"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])


def load_auto_resume_checkpoint(cfg, model, optimizer):
    """Auto-resume training from the latest checkpoint.

    Checkpoint directories should adhere to the simple format: step_<step_idx>_loss_<loss_value>
    If cfg.training.checkpoint.resume_from_metric is '+' or '-', then the loss_value is utilized
    for determining the optimal checkpoint to resume from. Otherwise, the latest checkpoint by
    modification time is chosen for resumption.

    Args:
        cfg: Hydra config.
        model: Model to load checkpoints into.
        optimizer: Optimizer to load checkpoints into.

    Returns:
        The latest step index to resume from.
    """
    # Auto-Resume: Load latest model and optimizer checkpoints.
    latest_step_idx = 0
    if cfg.training.checkpoint.path and Path(cfg.training.checkpoint.path).exists():
        # Get latest checkpoint sub-directory, which should ONLY contain Torch DCP checkpoint sub-directories.
        subdirs = [x.absolute() for x in Path(cfg.training.checkpoint.path).iterdir() if x.is_dir()]
        if len(subdirs) > 0:
            # We expect a checkpoint named as: step_<step_idx>_loss_<loss_value>.
            # Get the latest step, the directory with the most recent modification time.
            opt_metric_coeff = 1 if cfg.training.checkpoint.resume_from_metric == "+" else -1
            latest_subdir = max(
                subdirs,
                key=lambda x: (
                    opt_metric_coeff * float(x.name.split("_")[3])
                    if cfg.training.checkpoint.resume_from_metric
                    else 0,
                    x.stat().st_mtime,
                ),
            )
            # Track latest step to continue training from.
            latest_step_idx = int(latest_subdir.name.split("_")[1])
            # Load model and optimizer checkpoints.
            load_dcp_checkpoint(latest_subdir, model, optimizer)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                _logger.info(f"Loaded latest model and optimizer checkpoints from: {latest_subdir}")

    # Return the auto-resumed step index for training progression.
    return latest_step_idx


def save_dcp_checkpoint(checkpoint_path, model=None, optimizer=None):
    """Save a Torch DCP checkpoint of the model and optimizer to checkpoint_path.

    Docs: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
    """
    # Save model and optimizer checkpoints.
    state_dict = {}
    if model is not None:
        state_dict["model"] = model.state_dict()
    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.distributed.checkpoint.save(state_dict, checkpoint_id=checkpoint_path)


def save_auto_resumable_checkpoint(cfg, model, optimizer, step_idx, loss_value):
    """Save an auto-resumable checkpoint of the model and optimizer at step_idx.

    Checkpoint directories should adhere to the simple format: step_<step_idx>_loss_<loss_value>.
    This is used for auto-resumption of training.

    Args:
        cfg: Hydra config.
        model: Model to save checkpoints of.
        optimizer: Optimizer to save checkpoints of.
        step_idx: Step index to save checkpoint at.
        loss_value: Loss value to save checkpoint at.
    """

    # Save validated checkpoint.
    if cfg.training.checkpoint.path:
        # Create checkpoint sub-directory.
        ckpt_dir = Path(cfg.training.checkpoint.path) / f"step_{step_idx}_loss_{loss_value:.3f}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Save model and optimizer checkpoints.
        save_dcp_checkpoint(ckpt_dir, model, optimizer)
        # Relax checkpoint permissions, which may be helpful when saving checkpoints in a container owned by root.
        mode = 0o777
        for dirpath, _, filenames in os.walk(ckpt_dir):
            # Change current directory perms.
            os.chmod(dirpath, mode)
            for filename in filenames:
                # Change file perms.
                file_path = Path(dirpath) / filename
                os.chmod(file_path, mode)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            _logger.info(f"Saved validated checkpoint to: {ckpt_dir}")
