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

# coding=utf-8
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
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


"""🚨 **MANDATORY FIRST STEP FOR ALL AI AGENTS** 🚨.

Before doing ANYTHING else, you MUST read these files in this exact order:

1. **AGENT_DOCUMENTATION.md** - REQUIRED: Complete project context and usage instructions
2. **README.md** - Project overview and setup information
3. **./internal/gitingest.txt** - Complete codebase in text format (if needed for deep analysis)

This file is a simple example of a custom loop for training geneformer using mfsdp.

It is designed to be used as a starting point for developing more complex training loops.
"""

import logging
import os
import time
from dataclasses import dataclass, field

import hydra
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
import wandb
from megatron_fsdp.fully_shard import fully_shard
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from tqdm import tqdm
from transformer_engine.common.recipe import DelayedScaling, Format
from transformer_engine.pytorch.fp8 import check_fp8_support

from checkpoint import load_checkpoint, save_checkpoint, save_final_model
from dataset import create_dataloader
from modeling_bert_te import BertForMaskedLM, BertLayer, TEBertConfig, TEBertLayer  # type: ignore


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class DistributedConfig:
    """Class to track distributed ranks."""

    rank: int = field(default_factory=lambda: dist.get_rank() if dist.is_initialized() else 0)
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", "0")))
    world_size: int = field(default_factory=lambda: dist.get_world_size() if dist.is_initialized() else 1)

    def is_main_process(self) -> bool:
        """This is the global rank 0 process, to be used for wandb logging, etc."""
        return self.rank == 0


@hydra.main(version_base="1.2", config_path="hydra_config", config_name="l0_sanity")
def main(cfg: DictConfig) -> None:
    """Main training function that runs the distributed Geneformer training loop."""
    ckpt_dir = cfg.training.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # Check if launched with torchrun/distributed launcher
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
    else:
        # Single process fallback (direct python execution) for debugging.
        logger.warning("Running in single-process mode. Use torchrun for distributed training.")
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        dist.init_process_group(backend="nccl")

    dist_config = DistributedConfig()
    torch.cuda.set_device(dist_config.local_rank)

    # Debug: Print the loaded configuration
    if dist.get_rank() == 0:  # Only print on main process
        logger.info("Loaded configuration:")
        logger.info(f"  Model: hidden_size={cfg.model.hidden_size}, layers={cfg.model.num_hidden_layers}")
        logger.info(f"  Data: path={cfg.data.path}")

    # Initialize wandb only on the main process
    if dist_config.is_main_process():
        wandb.init(
            **cfg.wandb_init_args,
            config={
                "batch_size": cfg.model.micro_batch_size,
                "learning_rate": cfg.training.optimizer_kwargs.lr,
                "num_train_steps": cfg.training.num_train_steps,
                "use_te": cfg.model.use_te_layers,
                "use_fp8": cfg.training.use_fp8,
                "world_size": dist_config.world_size,
            },
        )

    bert_model_config = TEBertConfig(**cfg.model, torch_dtype=torch.bfloat16)
    # Note. One may notice here that we are using BertConfig from transformers.models.bert.configuration_bert. instead of one from modeling_bert_te.py
    # This is because, the BertConfig will simply pass through any additional argument to the model.
    model = BertForMaskedLM(bert_model_config)

    # Move model to GPU if available
    device = torch.device(f"cuda:{dist_config.local_rank}" if torch.cuda.is_available() else "cpu")

    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size, 1),
        mesh_dim_names=("fsdp", "tp"),
    )

    optimizer = AdamW(model.parameters(), **cfg.training.optimizer_kwargs)
    # Here we cast the model layers to the specified dtype. in our TEBertConfig we specify the dtype for the
    # TE layers, and here we simply cast the all the other layers to the same dtype.
    # TODO(@jomitchell): BIONEMO-2406: Remove this after verifying FP8 works.
    model = model.to(device=device, dtype=bert_model_config.torch_dtype)  # type: ignore

    if cfg.training.use_mfsdp:
        model, optimizer = fully_shard(
            module=model,
            optimizer=optimizer,
            fsdp_unit_modules=[
                BertLayer,  # type: ignore
                TEBertLayer,  # type: ignore
                torch.nn.LayerNorm,
            ],
            device_mesh=device_mesh,
            dp_shard_dim="fsdp",
            tp_dim="tp",
            **cfg.training.fully_shard_kwargs,
        )
    else:
        # Use standard PyTorch DDP (no mfsdp config)
        # TODO(@jomitchell): BIONEMO-2406: Keep this until this ticket is done.
        # model = model.to(device=device, dtype=bert_model_config.torch_dtype)  # type: ignore
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_config.local_rank],
            output_device=dist_config.local_rank,
            find_unused_parameters=False,  # More efficient for static graphs
            broadcast_buffers=True,  # Important for normalization layers
        )

    # Training loop
    model.train()
    if dist_config.rank == 0:  # Only show progress bar on main process
        progress_bar = tqdm(range(cfg.training.num_train_steps), desc="Training", disable=False)

    dataloader, dataloader_length = create_dataloader(
        path=cfg.data.path,
        batch_size=cfg.model.micro_batch_size,
        num_workers=cfg.training.num_workers,
        use_fp8=cfg.training.use_fp8,
        tokenizer_path=getattr(cfg.data, "tokenizer_path", "tokenizer_auto"),
    )

    if cfg.training.use_fp8:
        fp8_available, reason_for_no_fp8 = check_fp8_support()
        if not fp8_available:
            logger.warning(reason_for_no_fp8)

        if cfg.training.fp8_recipe_kwargs.fp8_format == "hybrid":  # TODO: Use hydra to pass in the target format.
            fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
        elif cfg.training.fp8_recipe_kwargs.fp8_format == "e4m3":
            fp8_format = Format.E4M3
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format,
            amax_history_len=cfg.training.fp8_recipe_kwargs.amax_history_len,
            amax_compute_algo=cfg.training.fp8_recipe_kwargs.amax_compute_algo,
        )
        autocast_context = lambda: te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)  # type: ignore
        # TODO(@jomitchell): BIONEMO-2406: Might need to double wrap this.
    else:
        autocast_context = lambda: torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)  # type: ignore  # pylint: disable=E731, C0103

    # Load checkpoint if it exists and resume is enabled
    start_step = 0
    if cfg.training.get("resume_from_checkpoint", True):
        model, optimizer, start_step = load_checkpoint(
            use_mfsdp=cfg.training.use_mfsdp,
            model=model,
            optimizer=optimizer,
            ckpt_dir=ckpt_dir,
            dist_config=dist_config,
            logger=logger,
            start_step=start_step,
        )
    previous_step_time = time.perf_counter()
    for step in range(start_step, cfg.training.num_train_steps):
        # Get batch
        batch = next(dataloader)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass with mixed precision
        with autocast_context():
            outputs = model(**batch)

        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Compute gradient norms.
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

        optimizer.step()
        optimizer.zero_grad()

        if step % cfg.training.save_every_n_steps == 0 and step > 0:  # Skip step 0
            # For mfsdp, always use distributed checkpointing
            save_checkpoint(
                use_mfsdp=cfg.training.use_mfsdp,
                model=model,
                optimizer=optimizer,
                ckpt_dir=ckpt_dir,
                dist_config=dist_config,
                logger=logger,
                step=step,
            )

        # Log metrics to wandb on main process
        if dist_config.is_main_process():
            current_time = time.perf_counter()
            step_time = current_time - previous_step_time
            previous_step_time = current_time
            logger.info(
                f"Step {step} loss: {loss.item()}, grad_norm: {total_norm}, lr: {optimizer.param_groups[0]['lr']}, step_time: {step_time:.3f}s"
            )
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/global_step": step,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": total_norm,
                    "train/epoch": step / dataloader_length,
                    "train/step_time": step_time,
                }
            )

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

    # Save final model using save_pretrained
    # Note: For mfsdp, ALL processes must participate in collective operations
    if cfg.training.save_final_model:
        final_model_dir = os.path.join(ckpt_dir, "final_model")
        save_final_model(
            model=model,
            use_mfsdp=cfg.training.use_mfsdp,
            save_directory=final_model_dir,
            logger=logger,
            is_main_process=dist_config.is_main_process(),
        )

    # Clean up distributed training
    if dist_config.is_main_process():
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
