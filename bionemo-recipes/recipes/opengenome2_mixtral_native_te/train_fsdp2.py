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

"""OpenGenome2 FSDP2 training script for Mixtral with TransformerEngine."""

import gc
import logging
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import hydra
import numpy as np
import torch


try:
    import nvdlfw_inspect.api as debug_api

    HAS_NVDLFW_INSPECT = True
except ImportError:
    debug_api = None
    HAS_NVDLFW_INSPECT = False
import transformer_engine
import transformer_engine.pytorch
from checkpoint import (
    _ckpt_futures,
    load_checkpoint_fsdp2,
    save_checkpoint_fsdp2,
    save_final_model_fsdp2,
    should_save_checkpoint,
)
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from fp8_debugging import initialize_fp8_debugging
from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM
from omegaconf import DictConfig, OmegaConf
from optimizer import get_parameter_groups_with_weight_decay
from perf_logger import PerfLogger
from scheduler import get_cosine_annealing_schedule_with_warmup
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW
from transformer_engine.common.recipe import Format
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from validation import run_validation


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- BEGIN COPIED FUNCTIONS ---
# _build_dispatcher and clip_grad_norm_ep_aware are copied from:
#   bionemo-recipes/recipes/mixtral_native_te/train_fsdp2.py
# Kept inline for recipe self-containment (KISS over DRY).
# --- END COPIED FUNCTIONS ---


def _build_dispatcher(args: DictConfig, config: NVMixtralConfig):
    """Build the requested token dispatcher for EP runs.

    Returns None for the default alltoall dispatcher (handled natively by TE).
    Returns a FusedTokenRouter when token_dispatcher=fused_deepep and deep_ep is available.
    Falls back to alltoall (returns None) when fused_deepep is requested but unavailable,
    if token_dispatcher_fallback=alltoall is set.
    """
    token_dispatcher = str(getattr(args, "token_dispatcher", "alltoall"))
    fallback_dispatcher = str(getattr(args, "token_dispatcher_fallback", "error"))
    if config.expert_parallel_size == 1:
        return None
    if token_dispatcher == "alltoall":
        return None
    if token_dispatcher != "fused_deepep":
        raise ValueError(f"Unsupported token_dispatcher: {token_dispatcher!r}. Expected 'alltoall' or 'fused_deepep'.")

    try:
        from fused_token_router import FusedTokenRouter

        return FusedTokenRouter(
            num_experts=config.num_local_experts,
            num_local_experts=config.num_local_experts // config.expert_parallel_size,
            hidden_size=config.hidden_size,
            ep_size=config.expert_parallel_size,
        )
    except ImportError as exc:
        if fallback_dispatcher == "alltoall":
            logger.warning("Fused DeepEP dispatcher unavailable (%s). Falling back to AllToAllTokenDispatcher.", exc)
            return None
        raise


def clip_grad_norm_ep_aware(params: Iterable[torch.nn.Parameter], max_norm: float, ep_size: int) -> torch.Tensor:
    """Clip gradient norms, handling expert parallelism (DTensor parameters on different meshes).

    When ep_size > 1, parameters may be DTensors on different device meshes (dp vs ep),
    which prevents torch.nn.utils.clip_grad_norm_ from stacking norms across them.
    This function computes norms per-parameter from local shards and clips accordingly.

    Args:
        params: Model parameters (may include DTensor expert weights).
        max_norm: Maximum gradient norm.
        ep_size: Expert parallelism size. If 1, falls back to standard clip_grad_norm_.

    Returns:
        Total gradient norm (approximate for ep_size > 1).
    """
    if ep_size == 1:
        return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)

    # Compute per-param local norms, handling DTensor by extracting the local shard.
    param_list = list(params)
    norms = []
    for p in param_list:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if hasattr(g, "to_local"):
            g = g.to_local()  # Extract local shard of DTensor gradient
        norms.append(g.float().norm())

    if not norms:
        return torch.tensor(0.0)

    total_norm = torch.stack(norms).norm()
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in param_list:
        if p.grad is not None:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

    return total_norm


# --- END COPIED FUNCTIONS ---


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
    logger.info("Set seed to %s (same on all ranks for FSDP2)", seed)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train OpenGenome2 Mixtral with TE layers using FSDP2.

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

    ep_size = args.expert_parallel_size
    if dist_config.world_size % ep_size != 0:
        raise ValueError(
            f"world_size ({dist_config.world_size}) must be divisible by expert_parallel_size ({ep_size})"
        )
    dp_size = dist_config.world_size // ep_size
    device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, ep_size), mesh_dim_names=("dp", "ep"))

    # --- Model Configuration ---
    # Create quantization recipes -- only used if FP8 is enabled in the config.
    fp8_recipe = None
    if args.fp8_config.enabled:
        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )

    # --- Model Initialization ---
    if args.use_te:
        config_class = NVMixtralConfig
        model_class = NVMixtralForCausalLM
    else:
        config_class = MixtralConfig
        model_class = MixtralForCausalLM

    # Determine dtype for model initialization
    use_fp32_master_weights = getattr(args, "use_fp32_master_weights", False)
    model_dtype = torch.float32 if use_fp32_master_weights else torch.bfloat16

    if use_fp32_master_weights:
        logger.info("FP32 master weights enabled: model init in FP32")

    config_kwargs = OmegaConf.to_container(args.config_kwargs, resolve=True) if args.config_kwargs else {}
    # Pass expert_parallel_size to config so the model initializes with the correct
    # num_local_experts = num_experts // expert_parallel_size per rank.
    if args.use_te:
        config_kwargs["expert_parallel_size"] = ep_size

    config = config_class.from_pretrained(args.config_name_or_path, dtype=model_dtype, **config_kwargs)

    logger.info(
        "Init config: std=%s, num_layers=%s, experts=%s, top_k=%s",
        getattr(config, "initializer_range", 0.02),
        getattr(config, "num_hidden_layers", None),
        getattr(config, "num_local_experts", None),
        getattr(config, "num_experts_per_tok", None),
    )

    dispatcher = _build_dispatcher(args, config) if args.use_te else None
    with (
        torch.device("meta") if args.use_meta_device else nullcontext(),
        transformer_engine.pytorch.quantized_model_init(
            recipe=fp8_recipe, **args.fp8_config.quantized_model_init_kwargs
        ),
    ):
        if dispatcher is not None:
            model = model_class(config, dispatcher=dispatcher)
        else:
            model = model_class(config)

    logger.info("Initialized Model:\n%s", model)

    # --- Expert Parallelism Setup ---
    # Expert parallelism setup — MUST happen before fully_shard()
    # Wraps expert weights as DTensors with Shard(0) on the expert dimension.
    if args.use_te and ep_size > 1:
        ep_mesh = device_mesh["ep"]
        ep_group = ep_mesh.get_group()
        model.model.set_ep_groups(ep_group, ep_mesh)

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

    # --- Distributed Wrapping (FSDP2) ---
    # Shard transformer layers with FSDP
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()
    for layer in model.model.layers:
        fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    if args.use_meta_device and isinstance(model, NVMixtralForCausalLM):
        model.init_empty_weights()
    elif args.use_meta_device and isinstance(model, MixtralForCausalLM):
        model.to_empty(device=device)
        model.apply(model._init_weights)

    # Assign names to layers so debug API can identify them
    if args.fp8_stats_config.enabled and HAS_NVDLFW_INSPECT:
        debug_api.infer_and_assign_layer_names(model)

    # --- Optimizer & Scheduler ---
    # Create optimizer
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
        logger.info("Weight decay grouping enabled: wd=%s, skip_embeddings=%s", weight_decay, skip_embedding_wd)
    else:
        optimizer = AdamW(model.parameters(), **adamw_kwargs)  # type: ignore
        logger.info("Weight decay grouping disabled: wd=%s for all params", adamw_kwargs.get("weight_decay", 0.1))

    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    # --- Data Loading ---
    if args.use_sequence_packing:
        train_dataloader, dataset_or_sampler = create_thd_dataloader(dist_config, **args.dataset)
    else:
        train_dataloader, dataset_or_sampler = create_bshd_dataloader(dist_config, **args.dataset)

    if args.use_torch_compile:
        model = torch.compile(model)

    # --- Checkpoint Resume ---
    # Load checkpoint if resuming
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        logger.info("Attempting to load checkpoint from %s", ckpt_path)
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,  # type: ignore[arg-type]
            process_group=device_mesh.get_group("dp"),
            expert_parallel_size=ep_size,
        )
        logger.info("Checkpoint loaded, resuming from step %s, epoch %s", start_step, epoch)
    else:
        logger.info("No checkpoint to load, starting from scratch")
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args)

    # Setup validation if enabled
    val_config = getattr(args, "validation", None)
    val_enabled = val_config is not None and getattr(val_config, "enabled", False)
    val_dataloader = None

    if val_enabled:
        val_data_path = getattr(val_config, "data_path", None)
        if val_data_path:
            logger.info("Setting up validation dataloader from %s", val_data_path)
            val_dataset_kwargs = OmegaConf.to_container(args.dataset, resolve=True)
            val_dataset_kwargs["load_dataset_kwargs"] = {
                "path": "json",
                "data_files": val_data_path,
                "split": "train",
                "streaming": True,
            }
            val_dataset_kwargs["use_stateful_dataloader"] = False
            val_dataset_kwargs["num_workers"] = 0

            if hasattr(val_config, "micro_batch_size") and val_config.micro_batch_size is not None:
                val_dataset_kwargs["micro_batch_size"] = val_config.micro_batch_size

            if args.use_sequence_packing:
                val_dataloader, _ = create_thd_dataloader(dist_config, **val_dataset_kwargs)
            else:
                val_dataloader, _ = create_bshd_dataloader(dist_config, **val_dataset_kwargs)

            logger.info(
                "Validation enabled: every %s steps, %s batches", val_config.eval_interval, val_config.num_batches
            )
        else:
            logger.warning("Validation enabled but no data_path specified, skipping validation")
            val_enabled = False

    # --- Training Loop ---
    gc.collect()
    torch.cuda.empty_cache()

    # Training loop
    logger.info("Starting training loop from step %s to %s", start_step, args.num_train_steps)
    step = start_step
    micro_step = 0

    if train_dataloader is None:
        raise RuntimeError("Expected train_dataloader to be initialized before training.")

    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            micro_step += 1

            with transformer_engine.pytorch.autocast(enabled=args.fp8_config.enabled, recipe=fp8_recipe):
                outputs = model(**batch)

            loss = outputs.loss / args.grad_acc_steps
            loss.backward()

            perf_logger.log_micro_step(step=step, batch=batch, outputs=outputs)

            # Gradient accumulation - only step optimizer after accumulating gradients
            if micro_step % args.grad_acc_steps == 0:
                micro_step = 0

                total_norm = clip_grad_norm_ep_aware(model.parameters(), max_norm=1.0, ep_size=ep_size)

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
                        process_group=device_mesh.get_group("dp"),
                        expert_parallel_size=ep_size,
                        max_checkpoints=args.checkpoint.max_checkpoints,
                        async_save=args.checkpoint.async_save,
                    )

                # Run validation at specified interval
                if val_enabled and val_dataloader is not None and step > 0 and step % val_config.eval_interval == 0:
                    try:
                        val_metrics = run_validation(
                            model=model,
                            val_dataloader=val_dataloader,
                            num_batches=val_config.num_batches,
                            device=device,
                            dist_config=dist_config,
                        )
                        perf_logger.log_validation(step, val_metrics)
                    except Exception as e:
                        logger.error(f"Validation failed at step {step}: {e}")
                        torch.distributed.barrier()

                step += 1
                if step >= args.num_train_steps:
                    break

        epoch += 1
        dataset_or_sampler.set_epoch(epoch)

    # --- Cleanup ---
    # Save final model
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
