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

"""OpenGenome2 FSDP2 with Context Parallelism training script with TransformerEngine.

Combines Fully Sharded Data Parallel v2 with Context Parallelism (CP), where each sequence is
split across multiple GPUs along the sequence dimension. This is useful for training with very long
sequences that do not fit into a single GPU's memory even with FSDP2 alone. Only supports
TE-accelerated models (NVLlamaForCausalLM).

Supports:
- FP32 master weights with MixedPrecisionPolicy (cast_forward_inputs=False)
- Megatron-style scaled initialization for residual output layers
- Spike-No-More embedding initialization (std=1.0)
- Weight decay grouping (skip bias and 1D params)
- FP8 training with configurable first/last layer BF16 override
- Checkpoint resume with LenientLoadPlanner for missing TE keys

For standard FSDP2 training without context parallelism, use ``train_fsdp2.py`` instead.
"""

import gc
import logging
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import nvtx
import torch


try:
    import nvdlfw_inspect.api as debug_api

    HAS_NVDLFW_INSPECT = True
except ImportError:
    debug_api = None
    HAS_NVDLFW_INSPECT = False
import random

import transformer_engine
import transformer_engine.pytorch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW
from transformer_engine.common.recipe import Format

from checkpoint import (
    _ckpt_futures,
    load_checkpoint_fsdp2,
    save_checkpoint_fsdp2,
    save_final_model_fsdp2,
    should_save_checkpoint,
)
from collator import ContextParallelDataLoaderWrapper, DataCollatorForContextParallel
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from fp8_debugging import initialize_fp8_debugging
from opengenome_modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from optimizer import get_parameter_groups_with_weight_decay
from perf_logger import PerfLogger
from scheduler import get_cosine_annealing_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    For FSDP2/DTensor, ALL ranks must use the SAME seed to ensure weights
    are initialized identically before sharding.

    Args:
        seed: Random seed (same on all ranks).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set seed to {seed} (same on all ranks for FSDP2)")


@hydra.main(config_path="hydra_config", config_name="L0_sanity_cp", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train OpenGenome2 Llama with TE layers using FSDP2 with Context Parallelism.

    Returns:
        float: The minimum loss value observed during training.
    """
    # --- Distributed Setup ---
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # Set random seeds (same seed on ALL ranks for FSDP2/DTensor)
    seed = getattr(args, "seed", 42)
    set_seed(seed)

    # TE Debug feature logging - MUST be done BEFORE FSDP wrapping
    if args.fp8_stats_config.enabled:
        initialize_fp8_debugging(dist_config, **args.fp8_stats_config, fp8_enabled=args.fp8_config.enabled)

    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size // args.cp_size, args.cp_size),
        mesh_dim_names=("dp", "cp"),
    )
    logger.info("Created device mesh: %s", device_mesh)

    # --- Model Configuration ---
    fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
        fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
    )

    # Validate config: meta-device init breaks custom initialization
    if getattr(args, "use_meta_device", False):
        if getattr(args, "use_megatron_scaled_init", False):
            raise ValueError("use_meta_device=true is incompatible with use_megatron_scaled_init=true")
        if getattr(args, "spike_no_more_embedding_init", False):
            raise ValueError("use_meta_device=true is incompatible with spike_no_more_embedding_init=true")

    # Determine dtype for model initialization
    use_fp32_master_weights = getattr(args, "use_fp32_master_weights", False)
    model_dtype = torch.float32 if use_fp32_master_weights else torch.bfloat16

    if use_fp32_master_weights:
        logger.info("FP32 master weights enabled: model init in FP32")

    config_kwargs = OmegaConf.to_container(args.config_kwargs, resolve=True) if args.config_kwargs else {}

    # Handle Spike-No-More embedding initialization (https://arxiv.org/abs/2312.16903)
    if getattr(args, "spike_no_more_embedding_init", False):
        config_kwargs["embedding_init_std"] = 1.0
        config_kwargs["tie_word_embeddings"] = False
        logger.info("Spike-No-More enabled: embedding_init_std=1.0, tie_word_embeddings=False")

    # Handle Megatron-style scaled initialization for residual output layers
    if getattr(args, "use_megatron_scaled_init", False):
        config_kwargs["use_megatron_scaled_init"] = True
        logger.info("Megatron scaled init enabled: proj/fc2 use std/sqrt(2*num_layers)")

    config = NVLlamaConfig.from_pretrained(args.config_name_or_path, dtype=model_dtype, **config_kwargs)

    # Log initialization settings
    std = getattr(config, "initializer_range", 0.02)
    num_layers = getattr(config, "num_hidden_layers", 32)
    use_scaled_init = getattr(args, "use_megatron_scaled_init", False)
    expected_output_std = std / (2.0 * num_layers) ** 0.5 if use_scaled_init else std
    embedding_init_std = getattr(config, "embedding_init_std", None)
    logger.info(
        f"Init config: std={std}, scaled_init={use_scaled_init}, output_std={expected_output_std:.6f}, "
        f"embedding_std={embedding_init_std}"
    )

    # --- Model Initialization ---
    with (
        torch.device("meta") if args.use_meta_device else nullcontext(),
        transformer_engine.pytorch.quantized_model_init(
            recipe=fp8_recipe, **args.fp8_config.quantized_model_init_kwargs
        ),
    ):
        model = NVLlamaForCausalLM(config)

    logger.info("Initialized Model:\n%s", model)

    # --- Distributed Wrapping (FSDP2 + CP) ---
    # Create MixedPrecisionPolicy for FSDP when using FP32 master weights
    mp_policy = None
    if use_fp32_master_weights:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=False,  # if True, will downcast top_embeddings to param dtype (bf16)
        )
        logger.info(
            "MixedPrecisionPolicy: param_dtype=bf16, reduce_dtype=fp32, output_dtype=bf16, cast_forward_inputs=False"
        )

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()

    # Flatten dp+cp mesh into single shard dimension for FSDP
    cp_dp_mesh = device_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_shard_cp")

    # Shard transformer layers with FSDP
    for layer in model.model.layers:
        fully_shard(layer, mesh=cp_dp_mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=cp_dp_mesh, mp_policy=mp_policy)

    # Attach the CP group to the model
    for layer in model.model.layers:
        layer.set_context_parallel_group(
            device_mesh["cp"].get_group(),
            torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
            torch.cuda.Stream(),
        )

    # If using meta device, move sharded weights to cuda and initialize parameters.
    # WARNING: meta-device init breaks Megatron-style scaled init for proj/fc2.
    # Use use_meta_device=false when using use_megatron_scaled_init or spike_no_more_embedding_init.
    if args.use_meta_device:
        model.init_empty_weights()

    # Assign names to layers so debug API can identify them
    if args.fp8_stats_config.enabled and HAS_NVDLFW_INSPECT:
        debug_api.infer_and_assign_layer_names(model)

    # --- Optimizer & Scheduler ---
    adamw_kwargs = OmegaConf.to_container(args.adamw_kwargs, resolve=True)

    use_wd_grouping = getattr(args, "use_weight_decay_grouping", True)
    if use_wd_grouping:
        weight_decay = adamw_kwargs.pop("weight_decay", 0.1)
        skip_embedding_wd = getattr(args, "skip_embedding_weight_decay", False)
        param_groups = get_parameter_groups_with_weight_decay(
            model=model,
            weight_decay=weight_decay,
            skip_embeddings=skip_embedding_wd,
        )
        optimizer = AdamW(param_groups, **adamw_kwargs)  # type: ignore
        logger.info(f"Weight decay grouping enabled: wd={weight_decay}, skip_embeddings={skip_embedding_wd}")
    else:
        optimizer = AdamW(model.parameters(), **adamw_kwargs)  # type: ignore
        logger.info(f"Weight decay grouping disabled: wd={adamw_kwargs.get('weight_decay', 0.1)} for all params")

    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_torch_compile:
        model = torch.compile(model)

    # --- Data Loading ---
    # Auto-set padding divisor for dual chunk CP algorithm
    if args.dataset.get("pad_sequences_to_be_divisible_by", None) is None:
        # The dual chunk algorithm gives each CP rank 2 chunks from each sequence, so we need each sequence to be
        # divisible by cp_mesh.size() * 2.
        logger.info("pad_sequences_to_be_divisible_by is not provided, using cp_mesh.size() * 2")
        OmegaConf.update(args, "dataset.pad_sequences_to_be_divisible_by", device_mesh["cp"].size() * 2)

    # We only create the dataloader on CP rank 0, which is responsible for loading data for all CP ranks.
    # This ensures that the data remains synchronized, even if we're using a non-deterministic data pipeline.
    if device_mesh["cp"].get_local_rank() == 0:
        if args.use_sequence_packing:
            train_dataloader, dataset_or_sampler = create_thd_dataloader(dist_config, **args.dataset)
        else:
            train_dataloader, dataset_or_sampler = create_bshd_dataloader(dist_config, **args.dataset)

        train_dataloader.collate_fn = DataCollatorForContextParallel(
            collator=train_dataloader.collate_fn,
            device_mesh=device_mesh,
            qkv_format=args.config_kwargs.attn_input_format,
            is_causal_lm=True,
        )
    else:
        train_dataloader = None
        dataset_or_sampler = None

    # On all ranks, we create a ContextParallelDataLoaderWrapper that broadcasts the data from cp rank 0.
    train_dataloader = ContextParallelDataLoaderWrapper(train_dataloader, device_mesh["cp"])

    # --- Checkpoint Resume ---
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        logger.info(f"Attempting to load checkpoint from {ckpt_path}")
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,
            process_group=cp_dp_mesh.get_group(),
        )
        logger.info(f"Checkpoint loaded, resuming from step {start_step}, epoch {epoch}")
    else:
        logger.info("No checkpoint to load, starting from scratch")
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args)

    gc.collect()
    torch.cuda.empty_cache()

    # --- Training Loop ---
    logger.info(f"Starting training loop from step {start_step} to {args.num_train_steps}")
    step = start_step
    micro_step = 0

    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            micro_step += 1

            # Forward pass with mixed precision.
            with nvtx.annotate("Forward pass", color="green"):
                with transformer_engine.pytorch.autocast(enabled=args.fp8_config.enabled, recipe=fp8_recipe):
                    outputs = model(**batch)

            # Backward pass - scale loss by grad_acc_steps for proper gradient averaging
            loss = outputs.loss / args.grad_acc_steps

            with nvtx.annotate("Backward pass", color="red"):
                loss.backward()

            perf_logger.log_micro_step(step=step, batch=batch, outputs=outputs)

            # Gradient accumulation - only step optimizer after accumulating gradients
            if micro_step % args.grad_acc_steps == 0:
                micro_step = 0

                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                        dataloader=train_dataloader if args.dataset.use_stateful_dataloader else None,
                        process_group=cp_dp_mesh.get_group(),
                        max_checkpoints=args.checkpoint.max_checkpoints,
                        async_save=args.checkpoint.async_save,
                    )

                step += 1
                if step >= args.num_train_steps:
                    break

        # Dataloader exhausted, incrementing epoch
        epoch += 1
        if dataset_or_sampler is not None:  # The dataset only exists on CP rank 0
            dataset_or_sampler.set_epoch(epoch)

    # --- Cleanup ---
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_fsdp2(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    # Wait for any outstanding async checkpoint saves
    if args.checkpoint.async_save and "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
        _ckpt_futures["fsdp2"].result()

    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
