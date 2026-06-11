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
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.optim import AdamW
from transformer_engine.common.recipe import Format

from checkpoint import load_checkpoint_fsdp2, save_checkpoint_fsdp2, save_final_model_fsdp2, should_save_checkpoint
from dataset import create_cp_dataloader
from distributed_config import DistributedConfig
from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM
from perf_logger import PerfLogger
from quantization import resolve_layer_precision
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(config_path="hydra_config", config_name="L0_sanity_cp", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train ESM-2 with TE layers using fsdp2.

    Returns:
        float: The loss value for the final batch.
    """
    # Initialize the distributed configuration, including creating the distributed process group.
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
    # The mesh is organized as [CP_dimension, DDP_dimension] where:
    # - DDP dimension: number of data parallel replicas (world_size // cp_size)
    # - CP dimension: context parallel size
    # Total ranks = cp_size * dp_size = world_size
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

    if args.use_fp32_master_weights:
        raise ValueError("FP32 master weights are not supported with FSDP2+CP. Use train_fsdp2.py instead.")

    # Create an empty ESM-2 model with a masked language model head, e.g. "nvidia/esm2_t6_8M_UR50D".
    config = NVEsmConfig.from_pretrained(
        args.config_name_or_path, token_dropout=False, dtype=torch.bfloat16, **args.config_kwargs
    )
    num_layers = config.num_hidden_layers

    # Resolve layer-wise quantization assignments and store on config.
    layer_precision = resolve_layer_precision(
        num_layers=num_layers,
        fp8_enabled=args.fp8_config.enabled,
        fp4_enabled=args.fp4_config.enabled,
        fp8_layers=OmegaConf.to_container(args.fp8_layers, resolve=True) if args.fp8_layers is not None else None,
        fp4_layers=OmegaConf.to_container(args.fp4_layers, resolve=True) if args.fp4_layers is not None else None,
    )
    config.layer_precision = layer_precision
    # If we're using sequence packing with TE layers, we need to pass the `attn_input_format` argument.
    if args.use_sequence_packing:
        config.attn_input_format = "thd"

    with torch.device("meta") if args.use_meta_device else nullcontext():
        model = NVEsmForMaskedLM(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)

    logger.info("Initialized Model:\n%s", model)

    # TE models use `model.model`, facebook HF models use `model.esm`.
    base = model.model if hasattr(model, "model") else model.esm
    # We call the transformer stack "layers" in our TE models, but it's called "layer" in the original ESM-2 models.
    transformer_stack = base.encoder.layers if hasattr(base.encoder, "layers") else base.encoder.layer
    # Fully shard takes in a DeviceMesh object, which is a 2D mesh of dimensions (CP_dimension, DP_dimension).
    # FSDP2 will shard the model across the DP (dim=1) dimension and then duplicate across the CP (dim=0) dimension.
    for layer in transformer_stack:
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

    # If we're using meta device, we need to move sharded weights to the cuda device and initialize the parameters.
    # Note, this should happen before we create the optimizer.
    if args.use_meta_device:
        model.init_empty_weights()

    # Create optimizer. Convert OmegaConf to regular dict to avoid serialization issues (BIONEMO-2873).
    optimizer = AdamW(model.parameters(), **OmegaConf.to_container(args.adamw_kwargs, resolve=True))  # type: ignore
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    # Context Parallelism requires THD Sequence Packing.
    assert args.use_sequence_packing, "Context Parallelism requires THD Sequence Packing."

    train_dataloader, dataset_or_sampler = create_cp_dataloader(
        dist_config,
        cp_mesh=device_mesh["cp"],
        **args.dataset,
    )

    if args.use_torch_compile:
        # If we're using torch.compile, we need to do this before loading the checkpoint to ensure key consistency.
        model = torch.compile(model)

    # If we're resuming from a checkpoint, load it and set the start step. Otherwise, start from step 0.
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_fsdp2(
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
                save_checkpoint_fsdp2(
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
        if dataset_or_sampler is not None:  # The dataset only exists on rank 0
            dataset_or_sampler.set_epoch(epoch)

    # Save final model to a .safetensors file.
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_fsdp2(
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
