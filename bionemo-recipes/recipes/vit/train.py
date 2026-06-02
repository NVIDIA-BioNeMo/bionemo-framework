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
import math
import time
from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from megatron_fsdp import fully_shard
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from beans import BeansDataset, infinite_dataloader
from checkpoint import load_auto_resume_checkpoint, save_auto_resumable_checkpoint
from distributed import initialize_distributed
from vit import build_vit_model


_logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="vit_base_patch16_224")
def main(cfg) -> None:
    """Train a ViT model on AI-Lab-Makerere/beans using Megatron-FSDP and TransformerEngine (TE)."""

    # Initialize distributed environment.
    with initialize_distributed(**cfg.distributed) as device_mesh:
        """
        Profiling
        """
        if cfg.profiling.torch_memory_profile:
            # Start Torch memory profiling.
            torch.cuda.memory._record_memory_history(**cfg.profiling.torch_memory_profile_kwargs)
            torch_memory_profiler_snapshot = None

        if cfg.profiling.wandb and torch.distributed.get_rank() == 0:
            # Initialize WandB on main process.
            wandb.init(
                **cfg.profiling.wandb_kwargs,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )

        """
        Model
        """
        model = build_vit_model(cfg, device_mesh, meta_init=cfg.fsdp.init_model_with_meta_device)

        # Create optimizer.
        optimizer = AdamW(model.parameters(), **cfg.optimizer)

        # Initialize Megatron-FSDP.
        model, optimizer = fully_shard(
            # Torch (Root) Module
            model,
            # Torch Optimizer
            optimizer=optimizer,
            # ZeRO Sharding Strategy: None (0) -> Optim (1) -> Grad (2) -> Weights (3)
            zero_dp_strategy=cfg.fsdp.zero_dp_strategy,
            # FSDP "Unit Modules" - The sub-modules of the model that you want to shard!
            fsdp_unit_modules=cfg.fsdp.fsdp_unit_modules,
            # Inter / Outer DP Sharding Strategy: None (0) -> Optim (1) -> Grad (2) -> Weights (3)
            # Note: This adds a second stage of sharding that generalizes DP-Replicate. Think of it
            # like an extra stage of NCCL divide-and-conquer when using all-gather or reduce-scatter.
            # Currently, this does not fully-shard the gradients and weights, only the optimizer state,
            # so the memory will be only marginally better than sharding on only DP-Shard.
            outer_dp_sharding_strategy=cfg.fsdp.outer_dp_sharding_strategy,
            # Megatron-FSDP Device Mesh / Distributed Environment
            device_mesh=device_mesh,
            # Always required to use Megatron-FSDP. What we shard on.
            dp_shard_dim="dp_shard" if cfg.distributed.cp == 1 else "dp_cp_shard",
            # Required if using HSDP. The second / intermediate set of data-parallel process groups.
            dp_outer_dim="dp_outer",
            # Required if using TP, either from TransformerEngine (TP=1) / Megatron or DTensor-based TP.
            tp_dim="tp",
            # Required if using HSDP. Created by flattening everything we shard on, e.g. DP-CP.
            hybrid_fsdp_group=device_mesh["hsdp"].get_group(),
            # Load the model on device in shards to avoid OOM. Requires device("meta")-init for model.
            init_model_with_meta_device=cfg.fsdp.init_model_with_meta_device,
            # Store distributed optimization state in FP32.
            preserve_fp32_weights=cfg.fsdp.preserve_fp32_weights,
            # Sync model parameters and gradients each step. Allows for param and gradient mods after BWD
            # pass, but deactivates compute-communication overlap going into the subsequent training step.
            sync_model_each_microbatch=True,
            # Preprocess state dict for DCP checkpointing. Required for Torch Distributed Checkpoint.
            preproc_state_dict_for_dcp_ckpt=True,
        )

        # Auto-Resume: Load latest model and optimizer checkpoints.
        latest_step_idx = load_auto_resume_checkpoint(cfg, model, optimizer)

        """
        Dataset
        """
        # Training
        beans_train_dataset = BeansDataset(image_size=(cfg.model.vit.img_size, cfg.model.vit.img_size), split="train")
        train_sampler = DistributedSampler(
            beans_train_dataset,
            # Send distinct samples to all DP ranks only!
            num_replicas=device_mesh["dp"].size(),
            rank=device_mesh["dp"].get_local_rank(),
            shuffle=cfg.dataset.train.shuffle,
            seed=cfg.random.seed,
        )
        train_dataloader = DataLoader(
            beans_train_dataset,
            batch_size=cfg.dataset.train.batch_size,
            sampler=train_sampler,
            num_workers=cfg.dataset.num_workers,
            # IMPORTANT: persistent_workers=True is required for Megatron-FSDP and
            # Torch DCP, because CUDA/NCCL and Dataloader kill each others' workers!
            # Alternatively, you can set num_workers=0.
            persistent_workers=(cfg.dataset.num_workers > 0),
        )
        # Validation
        beans_val_dataset = BeansDataset(
            image_size=(cfg.model.vit.img_size, cfg.model.vit.img_size), split="validation"
        )
        val_sampler = DistributedSampler(
            beans_val_dataset,
            # Send distinct samples to all DP ranks only!
            num_replicas=device_mesh["dp"].size(),
            rank=device_mesh["dp"].get_local_rank(),
            shuffle=cfg.dataset.val.shuffle,
            seed=cfg.random.seed,
        )
        val_dataloader = DataLoader(
            beans_val_dataset,
            batch_size=cfg.dataset.val.batch_size,
            sampler=val_sampler,
            num_workers=cfg.dataset.num_workers,
            # IMPORTANT: persistent_workers=True is required for Megatron-FSDP and
            # Torch DCP, because CUDA/NCCL and Dataloader kill each others' workers!
            # Alternatively, you can set num_workers=0.
            persistent_workers=(cfg.dataset.num_workers > 0),
        )

        """
        Training Utilities
        """
        # Loss Function
        loss_fn = torch.nn.CrossEntropyLoss().to(device=torch.device(f"cuda:{torch.cuda.current_device()}"))

        # LR Scheduler
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.steps)

        """
        Training Loop
        """

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            progress_bar = tqdm(
                range(cfg.training.steps), desc="Model Training", disable=False, initial=latest_step_idx
            )

        # Training Loop
        t_start = time.perf_counter()
        dataset_size = len(beans_train_dataset)
        global_batch_size = cfg.dataset.train.batch_size * device_mesh["dp"].size()
        steps_per_epoch = math.ceil(dataset_size / global_batch_size)
        for batch_idx, sample in enumerate(
            # Skip to latest step.
            infinite_dataloader(train_dataloader, train_sampler),
            start=latest_step_idx,
        ):
            # Unpack data.
            input, target = sample
            # Measure data load time.
            data_load_time = time.perf_counter() - t_start

            # Set training mode.
            model.train()
            optimizer.zero_grad()

            # Match model input shape.
            if cfg.model.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # Move input and target to GPU, which is set by torch.cuda.set_device.
            input = input.cuda()
            target = target.cuda()

            # Model Forward Pass
            output = model(input)
            loss = loss_fn(output, target)
            loss_value = loss.detach().item()

            # Model Backward Pass
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

            # Step Optimizer and LR Scheduler
            optimizer.step()
            lr_scheduler.step()

            # Validation
            if batch_idx % cfg.training.val_interval == 0 and batch_idx > 0:
                model.eval()
                with torch.inference_mode():
                    loss_sum = 0
                    batch_count = 0
                    for input, target in val_dataloader:
                        # Forward Pass
                        input = input.cuda()
                        target = target.cuda()
                        output = model(input)
                        loss = loss_fn(output, target)
                        # Reduce loss (for logging ONLY). If not using CP, sufficient to reduce across DP instead of HSDP.
                        torch.distributed.all_reduce(
                            loss,
                            op=torch.distributed.ReduceOp.AVG,
                            group=device_mesh["hsdp"].get_group(),
                        )
                        loss_sum += loss.detach().item()
                        batch_count += 1

                # Normalize summed loss by distributed size and number of batches.
                normalized_loss = loss_sum / batch_count
                if torch.distributed.get_rank() == 0:
                    # Log validation loss.
                    _logger.info(f"Validation Loss: {normalized_loss:.3f}")
                    if cfg.profiling.wandb:
                        wandb.log({"val/loss": normalized_loss})

                # Save validated checkpoint.
                save_auto_resumable_checkpoint(cfg, model, optimizer, batch_idx, normalized_loss)

            # Log metrics to logger and wandb on main process.
            if torch.distributed.get_rank() == 0 and batch_idx % cfg.training.log_interval == 0:
                # Measure step time.
                t_end = time.perf_counter()
                step_time = t_end - t_start
                # Compute average learning rate.
                lrl = [param_group["lr"] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                # Log metrics to STDOUT.
                _logger.info(
                    f"Train: [Epoch {batch_idx * global_batch_size // dataset_size} / Step {(batch_idx % steps_per_epoch) + 1:>4d}/{steps_per_epoch} "
                    f"({100.0 * ((batch_idx % steps_per_epoch) + 1) / steps_per_epoch:>3.0f}%)]  "
                    f"Loss: {loss_value:#.3g}  "
                    f"Time: {step_time:.3f}s ({global_batch_size / step_time:>7.2f} samples/sec)  "
                    f"Memory: {torch.cuda.memory.max_memory_reserved() / 1024**3} GB   "
                    f"LR: {lr:.3e}  "
                    f"Data Load Time: {data_load_time:.3f}s"
                )
                # Log metrics to WandB.
                if cfg.profiling.wandb:
                    wandb.log(
                        {
                            "train/loss": loss_value,
                            "train/global_step": batch_idx,
                            "train/learning_rate": lr,
                            "train/grad_norm": total_norm,
                            "train/epoch": batch_idx * global_batch_size / dataset_size,
                            "train/step_time": step_time,
                        }
                    )

                # Update Torch profiler snapshot.
                if cfg.profiling.torch_memory_profile:
                    torch_memory_profiler_snapshot = torch.cuda.memory._snapshot()

                progress_bar.update(cfg.training.log_interval)

            # Reset timer.
            t_start = time.perf_counter()

            # Terminate if completed training steps.
            if batch_idx >= cfg.training.steps:
                break

        # Dump memory profiler snapshot.
        # TODO(@cspades): Migrate to the new Torch profiler!
        if cfg.profiling.torch_memory_profile:
            from pickle import dump

            with open(
                # Path will only exist when using @hydra.main()!
                Path(HydraConfig.get().runtime.output_dir) / "torch_memory_profiler_snapshot.pickle",
                "wb",
            ) as f:
                dump(torch_memory_profiler_snapshot, f)

        if cfg.profiling.wandb and torch.distributed.get_rank() == 0:
            wandb.finish()


if __name__ == "__main__":
    main()
