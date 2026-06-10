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

"""FSDP2 with Context Parallelism training script for Llama 3 with TransformerEngine.

Combines Fully Sharded Data Parallel v2 with Context Parallelism (CP), where each sequence is
split across multiple GPUs along the sequence dimension. This is useful for training with very long
sequences that do not fit into a single GPU's memory even with FSDP2 alone. Only supports
TE-accelerated models (NVLlamaForCausalLM).

For standard FSDP2 training without context parallelism, use ``train_fsdp2.py`` instead.
"""

import gc
import logging
from contextlib import nullcontext
from pathlib import Path

import hydra
import nvdlfw_inspect.api as debug_api
import nvtx
import torch
import transformer_engine.pytorch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.optim import AdamW
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.optimizers import FusedAdam

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
from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from perf_logger import PerfLogger
from quantization import initialize_quant_stats_logging, resolve_layer_precision
from scheduler import get_cosine_annealing_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _init_master_weights_from_high_precision(
    optimizer: FusedAdam, model: torch.nn.Module, device: torch.device
) -> None:
    """Initialize optimizer master weights from high-precision init values.

    When quantized_model_init is used with preserve_high_precision_init_val=True, each FP8 parameter
    stores the original BF16 init values in CPU memory. This function initializes optimizer state
    for all parameters, then overwrites master weights for quantized params with the preserved
    high-precision values instead of dequantized FP8 values.
    """
    count = 0
    for name, param in model.named_parameters():
        optimizer.initialize_state(param, store_param_remainders=False)
        local = param._local_tensor if isinstance(param, DTensor) else param
        if hasattr(local, "get_high_precision_init_val"):
            hp_val = local.get_high_precision_init_val()
            if hp_val is not None:
                optimizer.set_scaled_state(param, "master_param", hp_val.to(device=device, dtype=torch.float32))
                local.clear_high_precision_init_val()
                count += 1
                logger.debug("Seeded master weight for %s from high-precision init val", name)
    if count > 0:
        logger.info("Initialized %d master weight(s) from high-precision init values", count)


@hydra.main(config_path="hydra_config", config_name="L0_sanity_cp", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train Llama3 with TE layers using FSDP2 with Context Parallelism.

    Returns:
        float: The loss value for the final batch.
    """
    # --- Distributed Setup ---
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size // args.cp_size, args.cp_size),
        mesh_dim_names=("dp", "cp"),
    )
    logger.info("Created device mesh: %s", device_mesh)

    # --- Model Configuration ---
    config = NVLlamaConfig.from_pretrained(
        args.config_name_or_path,
        dtype=torch.bfloat16,
        **args.config_kwargs,
    )

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

    if args.fp8_config.quantized_model_init_kwargs.get("enabled", False) and not (
        args.fp8_config.enabled or args.fp4_config.enabled
    ):
        raise ValueError(
            "fp8_config.quantized_model_init_kwargs.enabled=true requires fp8_config.enabled=true or "
            "fp4_config.enabled=true. Enable at least one quantization format to use quantized model initialization."
        )

    # --- Model Initialization ---
    # Optionally use transformer engine to initialize only fp8 versions of weights by setting
    # `fp8_config.quantized_model_init_kwargs.enabled` to `True`, as opposed to using the default where both bfloat16
    # and fp8 versions of weights are kept.
    with (
        torch.device("meta") if args.use_meta_device else nullcontext(),
        transformer_engine.pytorch.quantized_model_init(
            recipe=fp8_recipe, **args.fp8_config.quantized_model_init_kwargs
        ),
    ):
        model = NVLlamaForCausalLM(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)

    logger.info("Initialized Model:\n%s", model)

    # --- Distributed Wrapping (FSDP2 + CP) ---
    cp_dp_mesh = device_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_shard_cp")

    # Shard the transformer layers with FSDP. For Llama3, the transformer stack is in model.model.layers.
    # Each decoder layer should be individually sharded before sharding the full model.
    for layer in model.model.layers:
        fully_shard(layer, mesh=cp_dp_mesh)
    fully_shard(model, mesh=cp_dp_mesh)

    # Attach the CP group to the model.
    for layer in model.model.layers:
        layer.set_context_parallel_group(
            device_mesh["cp"].get_group(),
            torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
            torch.cuda.Stream(),
        )

    # Attach quantization recipes to the model (layer precision is already on config).
    model.model.set_recipes(fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)

    if args.use_meta_device:
        # TE layers require special handling to initialize the weights from the meta device.
        model.init_empty_weights()

    # Assign names to layers so debug API can identify them
    if args.quant_stats_config.enabled:
        debug_api.infer_and_assign_layer_names(model)

    # --- Optimizer & Scheduler ---
    # Convert OmegaConf to regular dict to avoid serialization issues (BIONEMO-2873).
    adamw_kwargs = OmegaConf.to_container(args.adamw_kwargs, resolve=True)
    if args.use_fp32_master_weights:
        # TE FusedAdam maintains FP32 master copies of BF16 params internally.
        # 'fused' kwarg is not used by TE's FusedAdam (it's always fused).
        adamw_kwargs.pop("fused", None)
        optimizer = FusedAdam(model.parameters(), master_weights=True, **adamw_kwargs)  # type: ignore
        logger.info("Using TE FusedAdam with FP32 master weights")
    else:
        optimizer = AdamW(model.parameters(), **adamw_kwargs)  # type: ignore
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_torch_compile:
        # If we're using torch.compile, we need to do this before loading the checkpoint to ensure key consistency.
        model = torch.compile(model)

    # --- Data Loading ---
    # Create the context-aware dataloader.
    if args.dataset.get("pad_sequences_to_be_divisible_by", None) is None:
        # The dual chunk algorithm gives each CP rank 2 chunks from each sequence, so we need each sequence to be
        # divisible by cp_mesh.size() * 2.
        logger.info("pad_sequences_to_be_divisible_by is not provided, using cp_mesh.size() * 2")
        OmegaConf.update(args, "dataset.pad_sequences_to_be_divisible_by", device_mesh["cp"].size() * 2)

    # We only create the dataloader on rank 0, which is responsible for loading data for all CP (and eventually TP)
    # ranks. This ensures that the data remains synchronized, even if we're using a non-deterministic data pipeline.
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
        logger.info("Attempting to load checkpoint from %s", ckpt_path)
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,
            process_group=cp_dp_mesh.get_group(),
        )
        logger.info("Checkpoint loaded, resuming from step %s, epoch %s", start_step, epoch)
    else:
        logger.info("No checkpoint to load, starting from scratch")
        start_step = 0
        epoch = 0

        if args.use_fp32_master_weights and args.fp8_config.quantized_model_init_kwargs.get(
            "preserve_high_precision_init_val", False
        ):
            _init_master_weights_from_high_precision(optimizer, model, device)

    perf_logger = PerfLogger(dist_config, args, start_step=start_step)

    gc.collect()
    torch.cuda.empty_cache()

    # --- Training Loop ---
    logger.info("Starting training loop from step %s to %s", start_step, args.num_train_steps)
    step = start_step
    micro_step = 0  # Gradient accumulation step counter
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            micro_step += 1

            # Forward pass - quantization autocast is handled inside the model via set_recipes().
            with nvtx.annotate("Forward pass", color="green"):
                outputs = model(**batch)

            # Backward pass - scale loss by grad_acc_steps for proper gradient averaging
            loss = outputs.loss / args.grad_acc_steps

            with nvtx.annotate("Backward pass", color="red"):
                loss.backward()

            # Log microbatch step data for accumulation metrics
            perf_logger.log_micro_step(step=step, batch=batch, outputs=outputs)

            # The end of a "full" step (i.e. after possibly multiple gradient accumulation steps).
            if micro_step % args.grad_acc_steps == 0:
                micro_step = 0

                # Compute and clip gradient norms.
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Step optimizer.
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
        if dataset_or_sampler is not None:  # The dataset only exists on rank 0
            dataset_or_sampler.set_epoch(epoch)

    # --- Cleanup ---
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_fsdp2(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    # Make sure we don't have any outstanding checkpoint save futures.
    if args.checkpoint.async_save and "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
        _ckpt_futures["fsdp2"].result()

    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
