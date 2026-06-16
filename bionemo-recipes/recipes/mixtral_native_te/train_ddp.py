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

"""Distributed Data Parallel (DDP) training script for Mixtral with TransformerEngine."""

import gc
import logging
from contextlib import nullcontext
from pathlib import Path

import hydra
import nvdlfw_inspect.api as debug_api
import torch
import transformer_engine.pytorch
from checkpoint import load_checkpoint_ddp, save_checkpoint_ddp, save_final_model_ddp, should_save_checkpoint
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from fp8_debugging import initialize_fp8_debugging
from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM
from omegaconf import DictConfig, OmegaConf
from perf_logger import PerfLogger
from scheduler import get_cosine_annealing_schedule_with_warmup
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from train_fsdp2 import _build_dispatcher, clip_grad_norm_ep_aware
from transformer_engine.common.recipe import Format
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train Mixtral with TE layers using DDP.

    Returns:
        float: The minimum loss value observed during training.
    """
    # --- Distributed Setup ---
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    if args.fp8_stats_config.enabled:
        initialize_fp8_debugging(dist_config, **args.fp8_stats_config, fp8_enabled=args.fp8_config.enabled)

    # --- Model Configuration ---
    ep_size = args.expert_parallel_size
    if dist_config.world_size % ep_size != 0:
        raise ValueError(
            f"world_size ({dist_config.world_size}) must be divisible by expert_parallel_size ({ep_size})"
        )
    if ep_size > 1:
        raise ValueError(
            "DDP stop-go checkpointing with expert_parallel_size > 1 is currently unsupported for this recipe. "
            "Use train_fsdp2.py for EP checkpoint save/resume."
        )
    dp_size = dist_config.world_size // ep_size
    device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, ep_size), mesh_dim_names=("dp", "ep"))

    fp8_recipe = None
    if args.fp8_config.enabled:
        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )

    fp4_recipe = None
    if args.fp4_config.enabled:
        fp4_recipe = hydra.utils.get_class(args.fp4_config.fp4_recipe)(**args.fp4_config.fp4_recipe_kwargs)

    # --- Model Initialization ---
    if args.use_te:
        config = NVMixtralConfig.from_pretrained(
            args.config_name_or_path,
            dtype=torch.bfloat16,
            expert_parallel_size=ep_size,
            **args.config_kwargs,
        )
        dispatcher = _build_dispatcher(args, config)
        with torch.device("meta") if args.use_meta_device else nullcontext():
            model = NVMixtralForCausalLM(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe, dispatcher=dispatcher)
    else:
        config = MixtralConfig.from_pretrained(args.config_name_or_path, dtype=torch.bfloat16, **args.config_kwargs)
        with torch.device("meta") if args.use_meta_device else nullcontext():
            model = MixtralForCausalLM(config)

    logger.info("Initialized Model:\n%s", model)

    # --- Expert Parallelism Setup ---
    if args.use_te and ep_size > 1:
        ep_mesh = device_mesh["ep"]
        ep_group = ep_mesh.get_group()
        model.model.set_ep_groups(ep_group, ep_mesh)

    # --- Distributed Wrapping (DDP) ---
    if args.use_meta_device:
        if args.use_te:
            model.init_empty_weights()
        else:
            model.to_empty(device=device)
            model.apply(model._init_weights)

    if args.fp8_stats_config.enabled:
        debug_api.infer_and_assign_layer_names(model)

    model = model.to(device=device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        device_mesh=device_mesh["dp"],
    )

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(model.parameters(), **OmegaConf.to_container(args.adamw_kwargs, resolve=True))  # type: ignore[arg-type]
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_torch_compile:
        model = torch.compile(model)

    # --- Data Loading ---
    if args.use_sequence_packing:
        train_dataloader, dataset_or_sampler = create_thd_dataloader(dist_config, **args.dataset)
    else:
        train_dataloader, dataset_or_sampler = create_bshd_dataloader(dist_config, **args.dataset)

    # --- Checkpoint Resume ---
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_ddp" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        logger.info("Attempting to load checkpoint from %s", ckpt_path)
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_ddp(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,
            expert_parallel_size=ep_size,
        )
        logger.info("Checkpoint loaded, resuming from step %s, epoch %s", start_step, epoch)
    else:
        logger.info("No checkpoint to load, starting from scratch")
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args, start_step=start_step)

    # --- Training Loop ---
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Starting training loop from step %s to %s", start_step, args.num_train_steps)
    step = start_step
    micro_step = 0
    while step < args.num_train_steps:
        for batch in train_dataloader:
            device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            micro_step += 1

            with model.no_sync() if micro_step % args.grad_acc_steps != 0 else nullcontext():
                with transformer_engine.pytorch.autocast(enabled=args.fp8_config.enabled, recipe=fp8_recipe):
                    outputs = model(**device_batch)

                loss = outputs.loss / args.grad_acc_steps
                loss.backward()

                perf_logger.log_micro_step(step=step, batch=device_batch, outputs=outputs)

            if micro_step % args.grad_acc_steps == 0:
                micro_step = 0

                total_norm = clip_grad_norm_ep_aware(model.parameters(), max_norm=1.0, ep_size=ep_size)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                perf_logger.log_step(step=step, grad_norm=total_norm, lr=optimizer.param_groups[0]["lr"])

                if ckpt_path and should_save_checkpoint(step, args.checkpoint.save_every_n_steps):
                    save_checkpoint_ddp(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ckpt_path=ckpt_path,
                        step=step,
                        epoch=epoch,
                        dist_config=dist_config,
                        dataloader=train_dataloader if args.dataset.use_stateful_dataloader else None,
                        max_checkpoints=args.checkpoint.max_checkpoints,
                        expert_parallel_size=ep_size,
                    )

                step += 1
                if step >= args.num_train_steps:
                    break

        epoch += 1
        dataset_or_sampler.set_epoch(epoch)

    # --- Cleanup ---
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_ddp(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
