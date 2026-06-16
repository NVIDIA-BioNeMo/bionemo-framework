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

"""FSDP2 + Context Parallel training script for CodonFM with TransformerEngine layers."""

import logging
from contextlib import nullcontext
from pathlib import Path

import hydra
import nvdlfw_inspect.api as debug_api
import torch
from checkpoint import load_checkpoint_fsdp2, save_checkpoint_fsdp2, save_final_model_fsdp2, should_save_checkpoint
from dataset import create_cp_dataloader
from distributed_config import DistributedConfig
from modeling_codonfm_te import MODEL_PRESETS, CodonFMConfig, CodonFMForMaskedLM
from omegaconf import DictConfig, OmegaConf
from perf_logger import PerfLogger
from quantization import WandBQuantLogger, initialize_quant_stats_logging, resolve_layer_precision
from scheduler import get_linear_schedule_with_warmup
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.optim import AdamW
from transformer_engine.common.recipe import Format


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity_cp", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train CodonFM with TE layers using FSDP2 + Context Parallelism.

    Returns:
        float: The minimum loss value seen during training.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Initialize distributed configuration
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # Validate that world_size is divisible by cp_size
    if dist_config.world_size % args.cp_size != 0:
        raise ValueError(
            f"world_size ({dist_config.world_size}) must be divisible by cp_size ({args.cp_size}). "
            f"Set cp_size to a divisor of world_size."
        )

    # Calculate DP size (number of data parallel replicas)
    dp_size = dist_config.world_size // args.cp_size

    # Create a device mesh for DP and CP.
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dp_size, args.cp_size),
        mesh_dim_names=("dp", "cp"),
    )

    # Our flattened group must have at least 2 ranks to enable Context Parallelism.
    if dp_size * args.cp_size <= 1:
        cp_dp_mesh = device_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_shard_cp")
    else:
        cp_dp_mesh = device_mesh

    logger.info(
        f"Creating device mesh: world_size={dist_config.world_size}, dp_size={dp_size}, cp_size={args.cp_size}"
    )

    perf_logger = None
    try:
        # Build model config from preset
        preset_overrides = MODEL_PRESETS[args.model_preset]

        # Resolve layer-wise quantization assignments
        num_layers = preset_overrides.get("num_hidden_layers", 12)
        layer_precision = resolve_layer_precision(
            num_layers=num_layers,
            fp8_enabled=args.fp8_config.enabled,
            fp4_enabled=args.fp4_config.enabled,
            fp8_layers=OmegaConf.to_container(args.fp8_layers, resolve=True) if args.fp8_layers is not None else None,
            fp4_layers=OmegaConf.to_container(args.fp4_layers, resolve=True) if args.fp4_layers is not None else None,
        )

        # Initialize quant stats logging if enabled
        if args.quant_stats_config.enabled:
            wandb_logger = None
            if args.quant_stats_config.log_to_wandb and dist_config.is_main_process():
                wandb_logger = WandBQuantLogger()
            initialize_quant_stats_logging(
                quant_stats_file=args.quant_stats_config.quant_stats_file,
                quant_log_dir=args.quant_stats_config.quant_log_dir,
                rank=dist_config.rank,
                layer_precision=layer_precision,
                statistics_logger=wandb_logger,
            )

        # Create quantization recipes
        fp8_recipe = None
        fp4_recipe = None
        if args.fp8_config.enabled:
            fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
                fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
            )
        if args.fp4_config.enabled:
            fp4_recipe = hydra.utils.get_class(args.fp4_config.fp4_recipe)(
                fp4_format=Format[args.fp4_config.fp4_format], **args.fp4_config.fp4_recipe_kwargs
            )

        if args.use_fp32_master_weights:
            raise ValueError("FP32 master weights are not supported with FSDP2+CP. Use train_fsdp2.py instead.")

        # Context Parallelism requires THD Sequence Packing.
        assert args.use_sequence_packing, "Context Parallelism requires THD Sequence Packing."

        config = CodonFMConfig(
            attn_input_format="thd",
            max_position_embeddings=args.dataset.max_seq_length,
            layer_precision=layer_precision,
            **preset_overrides,
        )

        with torch.device("meta") if args.use_meta_device else nullcontext():
            model = CodonFMForMaskedLM(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)

        logger.info("Initialized Model:\n%s", model)

        # Apply FSDP2 sharding with CP-aware mesh
        for layer in model.encoder.layers:
            fully_shard(layer, mesh=cp_dp_mesh)
            # Set CP group for layer if CP is enabled.
            if args.cp_size > 1:
                logger.debug(f"Rank {dist_config.rank}: Setting CP group for layer {layer}")
                layer.set_context_parallel_group(
                    device_mesh["cp"].get_group(),
                    torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
                    torch.cuda.Stream(),
                )
        fully_shard(model, mesh=cp_dp_mesh)

        # Initialize weights from meta device
        if args.use_meta_device:
            model.init_empty_weights()

        # Assign layer names for debug API
        if args.quant_stats_config.enabled:
            debug_api.infer_and_assign_layer_names(model)

        # Create optimizer and scheduler
        optimizer = AdamW(model.parameters(), **OmegaConf.to_container(args.adamw_kwargs, resolve=True))
        scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

        # Create CP dataloader
        dataloader_kwargs = OmegaConf.to_container(args.dataset, resolve=True)
        train_dataloader, sampler = create_cp_dataloader(
            dist_config,
            cp_mesh=device_mesh["cp"],
            **dataloader_kwargs,
        )

        # Resume from checkpoint if available
        ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2_cp" if args.checkpoint.ckpt_dir else None
        if args.checkpoint.resume_from_checkpoint and ckpt_path:
            model, optimizer, scheduler, start_step, epoch = load_checkpoint_fsdp2(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ckpt_path=ckpt_path,
                dist_config=dist_config,
            )
        else:
            start_step = 0
            epoch = 0

        perf_logger = PerfLogger(dist_config, args)

        # Training loop
        step = start_step
        while step < args.num_train_steps:
            for batch in train_dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

                # Forward pass
                outputs = model(**batch)

                # Backward pass
                loss = outputs.loss
                loss.backward()

                # Log micro-batch data
                perf_logger.log_micro_step(step=step, batch=batch, outputs=outputs)

                # Grad clip
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                perf_logger.log_step(
                    step=step,
                    grad_norm=total_norm,
                    lr=optimizer.param_groups[0]["lr"],
                )

                if ckpt_path and should_save_checkpoint(step, args.checkpoint.save_every_n_steps):
                    save_checkpoint_fsdp2(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ckpt_path=ckpt_path,
                        step=step,
                        epoch=epoch,
                        dist_config=dist_config,
                        max_checkpoints=args.checkpoint.max_checkpoints,
                    )

                step += 1
                if step >= args.num_train_steps:
                    break

            # Dataloader exhausted, incrementing epoch
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)

        # Save final model
        if args.checkpoint.save_final_model and ckpt_path:
            save_final_model_fsdp2(
                model=model,
                config=config,
                save_directory=ckpt_path / "final_model",
                dist_config=dist_config,
            )

        return float(perf_logger.min_loss.item())
    finally:
        if perf_logger is not None:
            perf_logger.finish()
        else:
            try:
                debug_api.end_debug()
            except RuntimeError:
                pass
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
