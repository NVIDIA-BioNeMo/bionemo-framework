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
import torch
import transformer_engine.pytorch
import transformers
from megatron_fsdp.fully_shard import fully_shard
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from transformer_engine.common.recipe import Format
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import EsmForMaskedLM

from checkpoint import load_checkpoint_mfsdp, save_checkpoint_mfsdp, save_final_model_mfsdp, should_save_checkpoint
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM
from perf_logger import PerfLogger
from quantization import resolve_layer_precision
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train ESM-2 with TE layers using mfsdp.

    Model names are valid ESM-2 model sizes, e.g.:
    - "esm2_t6_8M_UR50D"
    - "esm2_t36_3B_UR50D"
    - "esm2_t48_15B_UR50D"

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # Create a device mesh for FSDP.
    # We have to create a dummy mesh dimension for tensor parallel for things to work correctly with mfsdp.
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size, 1),
        mesh_dim_names=("dp", "tp"),
    )

    if args.use_fp32_master_weights:
        raise ValueError("FP32 master weights are not supported with mFSDP. Use train_fsdp2.py instead.")

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

        model = NVEsmForMaskedLM(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
        fsdp_unit_modules = [
            transformer_engine.pytorch.TransformerLayer,
            transformer_engine.pytorch.LayerNorm,
            transformer_engine.pytorch.LayerNormLinear,
        ]
    else:
        config = EsmConfig.from_pretrained(args.config_name_or_path, dtype=torch.bfloat16, **args.config_kwargs)
        model = EsmForMaskedLM(config)
        fsdp_unit_modules = [
            transformers.models.esm.modeling_esm.EsmLayer,
            transformers.models.esm.modeling_esm.EsmEmbeddings,
        ]

    logger.info("Initialized Model:\n%s", model)

    # Create optimizer. Convert OmegaConf to regular dict to avoid serialization issues (BIONEMO-2873).
    optimizer = AdamW(model.parameters(), **OmegaConf.to_container(args.adamw_kwargs, resolve=True))  # type: ignore

    # Wrap model in megatron-fsdp
    model, optimizer = fully_shard(
        module=model,
        optimizer=optimizer,
        fsdp_unit_modules=fsdp_unit_modules,
        device_mesh=device_mesh,
        dp_shard_dim="dp",
        tp_dim="tp",
        **args.fully_shard_kwargs,
    )

    # This is important; the LR scheduler modifies optimizer.step(), so this needs to get created
    # after the optimizer gets wrapped in FSDP. Here we use a warmup and linear decay scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    # If we're using sequence packing, create a THD dataloader, otherwise create a BSHD dataloader.
    train_dataloader, dataset_or_sampler = (
        create_thd_dataloader(dist_config, **args.dataset)
        if args.use_sequence_packing
        else create_bshd_dataloader(dist_config, **args.dataset)
    )

    if args.use_torch_compile:
        logger.warning(
            "BIONEMO-2977: Using torch.compile with mfsdp is currently not supported. `use_torch_compile` was set to "
            "true, but will be ignored."
        )

    # If we're resuming from a checkpoint, load it and set the start step. Otherwise, start from step 0.
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_mfsdp" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_mfsdp(
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
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            # Forward pass.
            outputs = model(**batch)

            # Backward pass.
            loss = outputs.loss
            loss.backward()

            # Compute and clip gradient norms.
            # This is causing training to hang in 26.01 torch base image for multi-process mFSDP.
            # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

            # Step optimizer.
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            perf_logger.log_step(
                step=step,
                batch=batch,
                outputs=outputs,
                grad_norm=0.0,  # total_norm,
                lr=optimizer.param_groups[0]["lr"],
            )

            if ckpt_path and should_save_checkpoint(step, args.checkpoint.save_every_n_steps):
                save_checkpoint_mfsdp(
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
        save_final_model_mfsdp(
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
