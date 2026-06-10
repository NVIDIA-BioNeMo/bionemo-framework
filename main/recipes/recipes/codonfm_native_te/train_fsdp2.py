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

"""FSDP2 training script for CodonFM with TransformerEngine layers."""

import logging
from contextlib import nullcontext
from pathlib import Path

import hydra
import nvdlfw_inspect.api as debug_api
import torch
from checkpoint import load_checkpoint_fsdp2, save_checkpoint_fsdp2, save_final_model_fsdp2, should_save_checkpoint
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from modeling_codonfm_te import MODEL_PRESETS, CodonFMConfig, CodonFMForMaskedLM
from omegaconf import DictConfig, OmegaConf
from perf_logger import PerfLogger
from quantization import WandBQuantLogger, initialize_quant_stats_logging, resolve_layer_precision
from scheduler import get_linear_schedule_with_warmup
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW
from transformer_engine.common.recipe import Format


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train CodonFM with TE layers using FSDP2.

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

    perf_logger = None
    try:
        # Create device mesh for FSDP
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(dist_config.world_size,),
            mesh_dim_names=("dp",),
        )

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

        config = CodonFMConfig(
            attn_input_format="thd" if args.use_sequence_packing else "bshd",
            max_position_embeddings=args.dataset.max_seq_length,
            layer_precision=layer_precision,
            **preset_overrides,
        )

        with torch.device("meta") if args.use_meta_device else nullcontext():
            model = CodonFMForMaskedLM(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)

        logger.info("Initialized Model:\n%s", model)

        # Apply FSDP2 sharding with optional mixed precision policy
        if args.use_fp32_master_weights:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=False,
            )
        else:
            mp_policy = MixedPrecisionPolicy()
        for layer in model.encoder.layers:
            fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
        fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

        # Initialize weights from meta device
        if args.use_meta_device:
            model.init_empty_weights()

        # Assign layer names for debug API
        if args.quant_stats_config.enabled:
            debug_api.infer_and_assign_layer_names(model)

        # Create optimizer and scheduler
        optimizer = AdamW(model.parameters(), **OmegaConf.to_container(args.adamw_kwargs, resolve=True))
        scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

        # Create dataloader
        dataloader_kwargs = OmegaConf.to_container(args.dataset, resolve=True)
        train_dataloader, sampler = (
            create_thd_dataloader(dist_config, **dataloader_kwargs)
            if args.use_sequence_packing
            else create_bshd_dataloader(dist_config, **dataloader_kwargs)
        )

        # Resume from checkpoint if available
        ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2" if args.checkpoint.ckpt_dir else None
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
        micro_step = 0  # Gradient accumulation step counter
        while step < args.num_train_steps:
            batches_in_epoch = 0
            for batch in train_dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

                micro_step += 1

                # Forward pass
                outputs = model(**batch)

                # Backward pass - scale loss by grad_acc_steps for proper gradient averaging
                loss = outputs.loss / args.grad_acc_steps
                loss.backward()

                # Log micro-batch data for accumulation metrics
                perf_logger.log_micro_step(step=step, batch=batch, outputs=outputs)

                # Optimizer step only after accumulating grad_acc_steps micro-batches
                if micro_step % args.grad_acc_steps == 0:
                    micro_step = 0

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

                batches_in_epoch += 1

            if batches_in_epoch == 0:
                raise RuntimeError(
                    f"Dataloader produced zero batches at epoch {epoch}, step {step}/{args.num_train_steps}. "
                    "This would cause an infinite loop."
                )

            epoch += 1
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
