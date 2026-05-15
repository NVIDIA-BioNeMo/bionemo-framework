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

"""DDP training script for CodonFM with TransformerEngine layers."""

import logging
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import hydra
import nvdlfw_inspect.api as debug_api
import torch
from checkpoint import load_checkpoint_ddp, save_checkpoint_ddp, save_final_model_ddp, should_save_checkpoint
from dataset import create_dataloaders
from distributed_config import DistributedConfig
from modeling_codonfm_te import MODEL_PRESETS, CodonFMConfig, CodonFMForMaskedLM
from omegaconf import DictConfig, OmegaConf
from perf_logger import PerfLogger
from quantization import WandBQuantLogger, initialize_quant_stats_logging, resolve_layer_precision
from scheduler import get_linear_schedule_with_warmup
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from transformer_engine.common.recipe import Format


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_VALID_PRECISIONS = ("fp32", "bf16", "bf16-mixed")


def precision_context(precision: str):
    """Return a fresh autocast context for the given precision mode.

    For `bf16-mixed`, wraps forward in `torch.autocast(cuda, bf16)`. For `fp32` and `bf16`,
    returns a nullcontext — params are already in the target dtype, no autocast needed.
    """
    if precision == "bf16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train CodonFM with TE layers using DDP.

    Returns:
        float: The minimum loss value seen during training.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.precision not in _VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {_VALID_PRECISIONS}, got {args.precision!r}")

    # Initialize distributed configuration
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device, timeout=timedelta(hours=1))
    torch.cuda.set_device(dist_config.local_rank)

    perf_logger = None
    try:
        # Mirrors the FSDP2 device mesh — not strictly required for DDP, but keeps configs symmetric.
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(dist_config.world_size,),
            mesh_dim_names=("ddp",),
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

        # Materialize weights. With meta-device init, init_empty_weights() runs the MAGNETO init
        # and moves params to CUDA; otherwise the model was constructed eagerly on CPU.
        if args.use_meta_device:
            model.init_empty_weights()
        else:
            model = model.to(device)

        if args.precision == "bf16":
            model = model.to(dtype=torch.bfloat16)

        # Assign layer names for debug API
        if args.quant_stats_config.enabled:
            debug_api.infer_and_assign_layer_names(model)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_config.local_rank],
            output_device=dist_config.local_rank,
            device_mesh=device_mesh["ddp"],
        )

        # Create optimizer and scheduler
        optimizer = AdamW(model.parameters(), **OmegaConf.to_container(args.adamw_kwargs, resolve=True))
        scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

        dataloader_kwargs = OmegaConf.to_container(args.dataset, resolve=True)
        use_split_dataset = dataloader_kwargs.pop("use_split_dataset", False)
        split_kwargs = dataloader_kwargs.pop("split_kwargs", None)
        train_dataloader, val_dataloader, sampler = create_dataloaders(
            dist_config,
            use_sequence_packing=args.use_sequence_packing,
            build_validation=args.validation.enabled,
            use_split_dataset=use_split_dataset,
            split_kwargs=split_kwargs,
            **dataloader_kwargs,
        )

        # Resume from checkpoint if available
        ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_ddp" if args.checkpoint.ckpt_dir else None
        if args.checkpoint.resume_from_checkpoint and ckpt_path:
            model, optimizer, scheduler, start_step, epoch = load_checkpoint_ddp(
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

                # Skip DDP grad sync on intermediate accumulation micro-steps; the final
                # micro-step (when we will call optimizer.step) syncs as usual.
                is_accumulation_boundary = micro_step % args.grad_acc_steps == 0
                sync_context = nullcontext() if is_accumulation_boundary else model.no_sync()

                with sync_context:
                    # Forward pass under the precision-specific autocast context.
                    # backward inherits the cached autocast state — no need to wrap it.
                    with precision_context(args.precision):
                        outputs = model(**batch)
                        # Scale loss by grad_acc_steps for proper gradient averaging
                        loss = outputs.loss / args.grad_acc_steps
                    loss.backward()

                # Log micro-batch data for accumulation metrics
                perf_logger.log_micro_step(step=step, batch=batch, outputs=outputs)

                # Optimizer step only after accumulating grad_acc_steps micro-batches
                if is_accumulation_boundary:
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
                        save_checkpoint_ddp(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            ckpt_path=ckpt_path,
                            step=step,
                            epoch=epoch,
                            dist_config=dist_config,
                            max_checkpoints=args.checkpoint.max_checkpoints,
                        )

                    if val_dataloader is not None and step > 0 and step % args.validation.eval_interval == 0:
                        model.eval()
                        val_loss_sum = torch.zeros((), device=device)
                        val_batches_seen = torch.zeros((), device=device)
                        val_iter = iter(val_dataloader)
                        with torch.no_grad():
                            for _ in range(args.validation.num_batches):
                                try:
                                    val_batch = next(val_iter)
                                except StopIteration:
                                    break
                                val_batch = {
                                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()
                                }
                                with precision_context(args.precision):
                                    val_outputs = model(**val_batch)
                                val_loss_sum += val_outputs.loss.detach()
                                val_batches_seen += 1
                        torch.distributed.all_reduce(val_loss_sum)
                        torch.distributed.all_reduce(val_batches_seen)
                        avg_val_loss = (val_loss_sum / val_batches_seen.clamp(min=1)).item()
                        perf_logger.log_validation(step, {"loss": avg_val_loss})
                        model.train()

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
            save_final_model_ddp(
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
