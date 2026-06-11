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
from pathlib import Path

import hydra
import nvdlfw_inspect.api as debug_api
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from transformer_engine.common.recipe import Format
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import EsmForMaskedLM

from checkpoint import load_checkpoint_ddp, save_checkpoint_ddp, save_final_model_ddp, should_save_checkpoint
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM
from perf_logger import PerfLogger
from quantization import initialize_quant_stats_logging, resolve_layer_precision
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train ESM-2 with TE layers using DDP.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    if args.use_fp32_master_weights:
        raise ValueError("FP32 master weights are not supported with DDP. Use train_fsdp2.py instead.")

    # Create a device mesh for DDP. While this isn't strictly necessary, it mirrors the device mesh we create for FSDP2
    # and MFSDP.
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("ddp",))

    # Create an empty ESM-2 model with a masked language model head, e.g. "nvidia/esm2_t6_8M_UR50D".
    if args.use_te:
        config = NVEsmConfig.from_pretrained(args.config_name_or_path, dtype=torch.bfloat16, **args.config_kwargs)

        # Resolve layer-wise quantization assignments and store on config.
        layer_precision = resolve_layer_precision(
            num_layers=config.num_hidden_layers,
            fp8_enabled=args.fp8_config.enabled,
            fp4_enabled=args.fp4_config.enabled,
            fp8_layers=OmegaConf.to_container(args.fp8_layers, resolve=True) if args.fp8_layers is not None else None,
            fp4_layers=OmegaConf.to_container(args.fp4_layers, resolve=True) if args.fp4_layers is not None else None,
        )
        config.layer_precision = layer_precision

        if args.quant_stats_config.enabled:
            initialize_quant_stats_logging(
                quant_stats_file=args.quant_stats_config.quant_stats_file,
                quant_log_dir=args.quant_stats_config.quant_log_dir,
                rank=dist_config.rank,
                layer_precision=layer_precision,
            )

        # Create quantization recipes -- these are only used if FP8/FP4 is enabled in the config.
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

        # If we're using sequence packing with TE layers, we need to pass the `attn_input_format` argument.
        if args.use_sequence_packing:
            config.attn_input_format = "thd"

        # For TE models, pass quantization recipes -- the model handles quantized_model_init and autocast internally
        # via `get_autocast_context()`.
        model = NVEsmForMaskedLM(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
    else:
        config = EsmConfig.from_pretrained(args.config_name_or_path, dtype=torch.bfloat16, **args.config_kwargs)
        model = EsmForMaskedLM(config)

    logger.info("Initialized Model:\n%s", model)

    # The ESM model has a contact head that we don't use in masked language pre-training, so we delete it to
    # avoid errors with unused parameters in DDP.
    base = model.model if hasattr(model, "model") else model.esm
    try:
        del base.contact_head
    except AttributeError:
        pass

    # Create optimizer.
    optimizer = AdamW(model.parameters(), **args.adamw_kwargs)
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_te and args.quant_stats_config.enabled:
        debug_api.infer_and_assign_layer_names(model)

    model = model.to(device=device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        device_mesh=device_mesh["ddp"],
    )

    # If we're using sequence packing, create a THD dataloader, otherwise create a BSHD dataloader.
    train_dataloader, dataset_or_sampler = (
        create_thd_dataloader(dist_config, **args.dataset)
        if args.use_sequence_packing
        else create_bshd_dataloader(dist_config, **args.dataset)
    )

    if args.use_torch_compile:
        # If we're using torch.compile, we need to do this before loading the checkpoint to ensure key consistency.
        model = torch.compile(model)

    # If we're resuming from a checkpoint, load it and set the start step. Otherwise, start from step 0.
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_ddp" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_ddp(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,
        )
    else:
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args)

    # Training loop
    step = start_step
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa PLW2901

            # Forward pass.
            outputs = model(**batch)

            # Backward pass.
            loss = outputs.loss
            loss.backward()

            # Compute and clip gradient norms.
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

            # Step optimizer.
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            perf_logger.log_step(
                step=step,
                batch=batch,
                outputs=outputs,
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
                    dataloader=train_dataloader if args.dataset.use_stateful_dataloader else None,
                    max_checkpoints=args.checkpoint.max_checkpoints,
                )

            step += 1
            if step >= args.num_train_steps:
                break

        # Dataloader exhausted, incrementing epoch
        epoch += 1
        dataset_or_sampler.set_epoch(epoch)

    # Save final model to a .safetensors file.
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_ddp(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    # Clean up distributed training
    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
