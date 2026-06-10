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
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed.checkpoint


def has_checkpoint_files(ckpt_dir):
    """Check if there are any checkpoint files in the directory."""
    if not os.path.exists(ckpt_dir):
        return False
    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.startswith("step_")]
    return len(checkpoint_files) > 0


def get_latest_checkpoint(ckpt_dir):
    """Get the latest checkpoint file in the directory."""
    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.startswith("step_")]
    return max(
        checkpoint_files,
        key=lambda x: int(Path(x).stem.split("_")[1]),
    )


def load_checkpoint(
    use_mfsdp: bool,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    dist_config: Dict[str, Any],
    logger: logging.Logger,
    start_step: int,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """Loads a checkpoint from a given path with robust error handling."""
    if not has_checkpoint_files(ckpt_dir):
        logger.info(
            "No checkpoint files found in the directory. Returning model and optimizer without loading checkpoint."
        )
        return model, optimizer, start_step

    logger.info(f"Resuming from checkpoint: {ckpt_dir}")

    try:
        latest_checkpoint = get_latest_checkpoint(ckpt_dir)
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
        start_step = int(latest_checkpoint.split("_")[1])

        # Validate checkpoint before attempting to load
        if not _validate_checkpoint(checkpoint_path, use_mfsdp, logger):
            logger.warning(f"Checkpoint {checkpoint_path} is corrupted or invalid. Starting fresh training.")
            return model, optimizer, 0

    except (ValueError, IndexError, OSError) as e:
        logger.warning(f"Error finding valid checkpoint: {e}")
        logger.warning("Starting fresh training from step 0")
        return model, optimizer, 0

    try:
        if use_mfsdp:
            ckpt_state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.distributed.checkpoint.load(state_dict=ckpt_state_dict, checkpoint_id=str(checkpoint_path))
            model.load_state_dict(ckpt_state_dict["model"])
            optimizer.load_state_dict(ckpt_state_dict["optimizer"])
            logger.info(f"Successfully loaded mfsdp checkpoint from step {start_step}")
        else:
            # For DDP, load model + optimizer state
            checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{dist_config.local_rank}")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if dist_config.is_main_process():
                logger.info(f"Successfully loaded DDP checkpoint from step {checkpoint.get('step', start_step)}")

    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        logger.warning("Checkpoint loading failed. Starting fresh training from step 0")
        return model, optimizer, 0

    return model, optimizer, start_step


def _validate_checkpoint(checkpoint_path: str, use_mfsdp: bool, logger: logging.Logger) -> bool:
    """Validate that a checkpoint is properly formatted and loadable."""
    try:
        if use_mfsdp:
            # For mfsdp, check that it's a directory with required files
            if not os.path.isdir(checkpoint_path):
                logger.warning(f"mfsdp checkpoint should be a directory, but {checkpoint_path} is not")
                return False

            metadata_path = os.path.join(checkpoint_path, ".metadata")
            if not os.path.isfile(metadata_path):
                logger.warning(f"mfsdp checkpoint missing .metadata file at {metadata_path}")
                return False

            logger.debug(f"mfsdp checkpoint validation passed for {checkpoint_path}")
            return True
        else:
            # For DDP, check that it's a file and loadable
            if not os.path.isfile(checkpoint_path):
                logger.warning(f"DDP checkpoint should be a file, but {checkpoint_path} is not")
                return False

            # Try to load just the keys to validate format
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            required_keys = ["model", "optimizer"]
            for key in required_keys:
                if key not in checkpoint:
                    logger.warning(f"DDP checkpoint missing required key: {key}")
                    return False

            logger.debug(f"DDP checkpoint validation passed for {checkpoint_path}")
            return True

    except Exception as e:
        logger.warning(f"Checkpoint validation failed for {checkpoint_path}: {e}")
        return False


def save_checkpoint(
    use_mfsdp: bool,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str,
    dist_config: Dict[str, Any],
    logger: logging.Logger,
    step: int,
) -> None:
    """Saves a checkpoint to a given path with robust error handling."""
    checkpoint_path = os.path.join(ckpt_dir, f"step_{step}")

    # Clean up any existing corrupted checkpoint with the same name
    if os.path.exists(checkpoint_path):
        try:
            if use_mfsdp and os.path.isfile(checkpoint_path):
                # Remove file that should be a directory
                logger.warning(f"Removing corrupted checkpoint file: {checkpoint_path}")
                os.remove(checkpoint_path)
            elif not use_mfsdp and os.path.isdir(checkpoint_path):
                # Remove directory that should be a file
                logger.warning(f"Removing corrupted checkpoint directory: {checkpoint_path}")
                shutil.rmtree(checkpoint_path)
        except (OSError, PermissionError) as e:
            logger.error(f"Could not clean up existing checkpoint {checkpoint_path}: {e}")
            return

    try:
        if use_mfsdp:
            torch.distributed.checkpoint.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                checkpoint_id=checkpoint_path,
            )
            logger.info(f"Successfully saved mfsdp checkpoint to {checkpoint_path}")
        else:
            # For DDP, save model + optimizer state on main process only
            if dist_config.is_main_process():
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                    checkpoint_path,
                )
                logger.info(f"Successfully saved DDP checkpoint to {checkpoint_path}")

        # Validate the saved checkpoint
        if not _validate_checkpoint(checkpoint_path, use_mfsdp, logger):
            logger.error(f"Saved checkpoint {checkpoint_path} failed validation!")

    except Exception as e:
        logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
        # Clean up any partially created checkpoint
        try:
            if os.path.exists(checkpoint_path):
                if os.path.isfile(checkpoint_path):
                    os.remove(checkpoint_path)
                elif os.path.isdir(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
                logger.info(f"Cleaned up partially created checkpoint: {checkpoint_path}")
        except Exception as cleanup_e:
            logger.error(f"Failed to clean up partial checkpoint: {cleanup_e}")


def _get_underlying_model(model):
    """Get the underlying model, handling both mfsdp and DDP wrapping."""
    if hasattr(model, "module"):
        return model.module
    return model


def save_final_model(model, save_directory, logger, use_mfsdp=False, is_main_process=True):
    """Save the final model in safetensors format.

    For mfsdp, this performs parameter gathering across all processes first.
    For DDP, this simply unwraps the model if needed.

    Args:
        model: The model to save (wrapped or unwrapped)
        save_directory: Directory to save the model
        logger: Logger for status messages
        use_mfsdp: Whether using mfsdp (requires parameter gathering)
        is_main_process: Whether this is the main process (only main process saves)

    Note: For mfsdp, parameter gathering operations run on ALL processes, but only
    the main process saves the model.
    """
    if use_mfsdp:
        # Gather Megatron-FSDP parameters to CPU.
        from megatron_fsdp.uneven_dtensor import gather_uneven_dtensor_to_full_tensor

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
        if not is_main_process:
            return

        underlying_model = model.module
        model_state_dict = unsharded_state_dict
        save_type = "mfsdp"
    else:
        # Only main process needs to do anything for DDP
        if not is_main_process:
            return

        underlying_model = _get_underlying_model(model)
        model_state_dict = underlying_model.state_dict()
        save_type = "DDP"

    logger.info(f"Starting {save_type} model saving...")

    underlying_model.save_pretrained(save_directory, state_dict=model_state_dict, safe_serialization=True)

    logger.info(f"✅ {save_type} save_pretrained succeeded with safe_serialization=True")
