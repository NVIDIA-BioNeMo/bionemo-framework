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

"""Mixtral training with expert parallelism (EP), FSDP2, and MXFP8 — using native PyTorch + TE.

This entrypoint is the recipe's reference implementation of expert parallelism "from scratch": it
builds a 2D ``(dp, ep)`` ``DeviceMesh``, sets the expert-parallel groups, FSDP2-``fully_shard``s the
decoder layers over ``dp``, dispatches tokens across ``ep`` with a hand-written differentiable
all-to-all, and explicitly all-reduces the replicated dense gradients over ``ep`` — deliberately
without Megatron-LM. See ``distributed_setup.build_mesh_and_wrap`` and
``modeling_mixtral_te.NVMixtralSparseMoeBlock`` for the load-bearing details.
"""

import gc
import logging
import os
from contextlib import nullcontext
from pathlib import Path


# Must be set before importing Transformer Engine. NVTE_GROUPED_LINEAR_SINGLE_PARAM=0 makes TE expose
# discrete per-expert weight{i} tensors (required by both the fused GroupedMLP path and the
# consolidated expert checkpointing); NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 enables the Blackwell MXFP8
# fused expert kernel (ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8).
os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

import hydra
import torch
import transformer_engine.pytorch
from checkpoint import (
    _ckpt_futures,
    is_expert_key,
    load_checkpoint,
    save_checkpoint,
    save_final_model,
    should_save_checkpoint,
)
from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig
from distributed_setup import all_reduce_dense_grads_over_ep, build_mesh_and_wrap
from modeling_mixtral_te import (
    NVMixtralConfig,
    NVMixtralForCausalLM,
    _ensure_fused_grouped_mlp_registered,
)
from omegaconf import DictConfig, OmegaConf
from perf_logger import PerfLogger
from scheduler import get_cosine_annealing_schedule_with_warmup
from torch.distributed.tensor import DTensor
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.optimizers import FusedAdam


def _load_pretrained_te_state_dict(path: str | os.PathLike) -> dict:
    """Load a converted pretrained TE state dict from a file or a ``save_pretrained`` directory.

    Accepts either a single serialized state dict (``*.pt``) or a Hugging Face checkpoint directory
    produced by ``export.py`` (``model.safetensors`` or a sharded ``model.safetensors.index.json``).
    """
    path = Path(path)
    if path.is_file():
        # mmap=True keeps the (potentially >CPU-RAM) checkpoint as reclaimable page cache instead of
        # anonymous RSS, and is shared read-only across ranks on a node — required to load large
        # models (e.g. 8x7B ~87GB) under a constrained CPU memory cgroup.
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    from safetensors.torch import load_file

    index = path / "model.safetensors.index.json"
    if index.exists():
        import json

        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        state_dict: dict = {}
        for shard in sorted(set(weight_map.values())):
            state_dict.update(load_file(path / shard, device="cpu"))
        return state_dict

    single = path / "model.safetensors"
    if single.exists():
        return load_file(single, device="cpu")

    raise FileNotFoundError(
        f"init_from_pretrained={path} is neither a state-dict file nor a checkpoint directory "
        "containing model.safetensors / model.safetensors.index.json"
    )


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _clip_grad_norm_mixed(
    model: torch.nn.Module,
    max_norm: float,
    ep_group: torch.distributed.ProcessGroup | None = None,
    dp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Global grad-norm clipping when EP-local expert params are mixed with FSDP2 DTensors.

    A global L2 norm needs the sum of squared grads over *all* parameters, but no single group holds
    all of them: after backward each rank has only a partial squared norm, and the missing piece
    lives in a different process group for the two param classes. So we reduce each class over the
    group that owns its missing shards, then combine:

    - **expert** grads are partitioned over ``ep`` (each rank owns disjoint experts) → sum of squares
      is completed by all-reducing over ``ep_group``;
    - **dense** grads are FSDP2-sharded over ``dp`` (and, after the earlier dense ep all-reduce, are
      already identical across ``ep``) → completed by all-reducing over ``dp_group``.

    ``.to_local()`` extracts the owned shard from FSDP2 DTensor grads before the local norm.
    """
    device = next(model.parameters()).device
    expert_sq_sum = torch.zeros((), device=device, dtype=torch.float32)
    fsdp_sq_sum = torch.zeros((), device=device, dtype=torch.float32)

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        norm = torch.linalg.vector_norm(grad, ord=2).to(torch.float32)
        if is_expert_key(name):
            expert_sq_sum += norm * norm
        else:
            fsdp_sq_sum += norm * norm

    if ep_group is not None:
        torch.distributed.all_reduce(expert_sq_sum, op=torch.distributed.ReduceOp.SUM, group=ep_group)
    if dp_group is not None:
        torch.distributed.all_reduce(fsdp_sq_sum, op=torch.distributed.ReduceOp.SUM, group=dp_group)

    group_norms: list[torch.Tensor] = []
    if fsdp_sq_sum > 0:
        group_norms.append(torch.sqrt(fsdp_sq_sum))
    if expert_sq_sum > 0:
        group_norms.append(torch.sqrt(expert_sq_sum))

    if not group_norms:
        return torch.tensor(0.0, device=device)

    total_norm = torch.linalg.vector_norm(torch.stack(group_norms), ord=2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in model.parameters():
            if param.grad is not None:
                if isinstance(param.grad, DTensor):
                    param.grad = param.grad * clip_coef
                else:
                    param.grad.mul_(clip_coef)
    return total_norm


def _assert_fused_mxfp8_supported(expert_ffn_mode: str, fp8_enabled: bool) -> None:
    """Fail loudly if fused grouped MLP + MXFP8 was requested but is unsupported."""
    if expert_ffn_mode != "fused_grouped_mlp" or not fp8_enabled:
        return

    from transformer_engine.pytorch.ops.fused.forward_grouped_mlp import (
        ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8,
    )

    ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8.is_supported.cache_clear()
    if not ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8.is_supported():
        raise RuntimeError(
            "expert_ffn_mode='fused_grouped_mlp' with MXFP8 (fp8_config.enabled=true) requires "
            "ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8 support (Blackwell sm_100+). "
            "Refusing to fall back to the unfused grouped-linear path."
        )
    _ensure_fused_grouped_mlp_registered()


def _init_master_weights_from_high_precision(
    optimizer: FusedAdam, model: torch.nn.Module, device: torch.device
) -> None:
    """Initialize optimizer master weights from high-precision init values."""
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
    else:
        logger.info(
            "No parameters with high-precision init values found (quantized_model_init may not have been used)"
        )


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train Mixtral with TE layers using FSDP2 and Expert Parallelism.

    Returns:
        float | None: Minimum training loss observed, or None if no steps were logged.
    """
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    dp_size = args.parallelism.dp_size
    ep_size = args.parallelism.ep_size
    if dp_size * ep_size != dist_config.world_size:
        raise ValueError(
            f"parallelism.dp_size ({dp_size}) * parallelism.ep_size ({ep_size}) must equal "
            f"world_size ({dist_config.world_size})"
        )

    _assert_fused_mxfp8_supported(args.expert_ffn_mode, args.fp8_config.enabled)

    config = NVMixtralConfig.from_pretrained(
        args.config_name_or_path,
        dtype=torch.bfloat16,
        expert_ffn_mode=args.expert_ffn_mode,
        expert_parallel_size=ep_size,
        **args.config_kwargs,
    )

    fp8_recipe = None
    if args.fp8_config.enabled:
        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )

    if args.fp8_config.quantized_model_init_kwargs.get("enabled", False) and not args.fp8_config.enabled:
        raise ValueError("fp8_config.quantized_model_init_kwargs.enabled=true requires fp8_config.enabled=true.")

    quantized_model_init_enabled = args.fp8_config.quantized_model_init_kwargs.get("enabled", False)

    with (
        torch.device("meta") if args.use_meta_device else nullcontext(),
        transformer_engine.pytorch.quantized_model_init(
            recipe=fp8_recipe, **args.fp8_config.quantized_model_init_kwargs
        ),
    ):
        model = NVMixtralForCausalLM(config, fp8_recipe=fp8_recipe)

    logger.info("Initialized Model:\n%s", model)

    if not args.use_meta_device:
        model = model.to(device=device, dtype=torch.bfloat16)

    # Optionally initialize from a converted pretrained TE checkpoint (before FSDP wrapping, while the
    # model is unsharded). Loading pretrained bf16 values into the quantized params re-quantizes them
    # (copy_ on MXFP8), and FusedAdam later seeds its fp32 master from the loaded weights.
    if args.get("init_from_pretrained"):
        if args.use_meta_device:
            raise ValueError("init_from_pretrained requires use_meta_device=false")
        logger.info("Loading pretrained TE weights from %s", args.init_from_pretrained)
        pretrained_sd = _load_pretrained_te_state_dict(args.init_from_pretrained)
        missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)
        # Shared fused views (._experts_ffn_op.) and TE _extra_state buffers are expected-missing.
        real_missing = [k for k in missing if "._experts_ffn_op." not in k and not k.endswith("_extra_state")]
        if real_missing:
            logger.warning("Pretrained load: %d unexpected-missing keys, e.g. %s", len(real_missing), real_missing[:5])
        if unexpected:
            logger.warning("Pretrained load: %d unexpected keys, e.g. %s", len(unexpected), list(unexpected)[:5])
        logger.info("Loaded pretrained weights (missing=%d unexpected=%d)", len(missing), len(unexpected))
        del pretrained_sd

    mesh = build_mesh_and_wrap(model, dp_size=dp_size, ep_size=ep_size)

    if args.use_meta_device:
        model.init_empty_weights()

    adamw_kwargs = OmegaConf.to_container(args.adamw_kwargs, resolve=True)
    adamw_kwargs.pop("fused", None)
    # EP-local expert params mixed with FSDP2 DTensors require TE FusedAdam (torch AdamW/fused AdamW fail).
    # Master weights are configured independently of quantized_model_init (see defaults.yaml): this
    # recipe always keeps an fp32-precision master (needed for persistent-MXFP8 params and the
    # consolidated expert-optimizer checkpoint). `optimizer_store_param_remainders` stores it as a
    # (bf16 param + 16-bit remainder) instead of a full fp32 copy — same precision, half the footprint.
    master_weights = args.optimizer_master_weights
    store_param_remainders = args.optimizer_store_param_remainders
    optimizer = FusedAdam(
        model.parameters(),
        master_weights=master_weights,
        store_param_remainders=store_param_remainders,
        **adamw_kwargs,
    )  # type: ignore[arg-type]
    logger.info(
        "Using TE FusedAdam (master_weights=%s, store_param_remainders=%s, quantized_model_init=%s)",
        master_weights,
        store_param_remainders,
        quantized_model_init_enabled,
    )

    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    if args.use_torch_compile:
        model = torch.compile(model)

    if args.use_sequence_packing:
        train_dataloader, dataset_or_sampler = create_thd_dataloader(dist_config, **args.dataset)
    else:
        train_dataloader, dataset_or_sampler = create_bshd_dataloader(dist_config, **args.dataset)

    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2_ep" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        logger.info("Attempting to load checkpoint from %s", ckpt_path)
        model, optimizer, scheduler, _dl, start_step, epoch = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            ep_mesh=mesh["ep"],
            dp_process_group=mesh["dp"].get_group(),
            dataloader=train_dataloader if args.dataset.use_stateful_dataloader else None,
        )
        if _dl is not None:
            train_dataloader = _dl
        logger.info("Checkpoint loaded, resuming from step %s, epoch %s", start_step, epoch)
    else:
        logger.info("No checkpoint to load, starting from scratch")
        start_step = 0
        epoch = 0

        if quantized_model_init_enabled and args.fp8_config.quantized_model_init_kwargs.get(
            "preserve_high_precision_init_val", False
        ):
            _init_master_weights_from_high_precision(optimizer, model, device)

    perf_logger = PerfLogger(dist_config, args, start_step=start_step)

    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Starting training loop from step %s to %s", start_step, args.num_train_steps)
    step = start_step
    micro_step = 0
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            micro_step += 1

            outputs = model(**batch)

            loss = outputs.loss / args.grad_acc_steps
            loss.backward()

            perf_logger.log_micro_step(step=step, batch=batch, outputs=outputs)

            if micro_step % args.grad_acc_steps == 0:
                micro_step = 0

                # Dense params are replicated across ep; average their grads over ep so they stay in
                # sync. This is the whole dense reduction at dp==1 (dense unwrapped) and the ep half
                # at dp>1 (dense FSDP-reduced over dp). No-op when ep_size == 1.
                if ep_size > 1:
                    all_reduce_dense_grads_over_ep(model, mesh["ep"].get_group())

                total_norm = _clip_grad_norm_mixed(
                    model,
                    max_norm=1.0,
                    ep_group=mesh["ep"].get_group(),
                    dp_group=mesh["dp"].get_group(),
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                perf_logger.log_step(
                    step=step,
                    grad_norm=total_norm,
                    lr=optimizer.param_groups[0]["lr"],
                )

                if ckpt_path and should_save_checkpoint(step, args.checkpoint.save_every_n_steps):
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ckpt_path=ckpt_path,
                        step=step,
                        epoch=epoch,
                        dist_config=dist_config,
                        ep_mesh=mesh["ep"],
                        dp_process_group=mesh["dp"].get_group(),
                        dataloader=train_dataloader if args.dataset.use_stateful_dataloader else None,
                        max_checkpoints=args.checkpoint.max_checkpoints,
                        async_save=args.checkpoint.async_save,
                    )

                step += 1
                if step >= args.num_train_steps:
                    break

        epoch += 1
        logger.warning("Dataloader exhausted at step %s, incrementing epoch to %s", step, epoch)
        dataset_or_sampler.set_epoch(epoch)

    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
            ep_mesh=mesh["ep"],
        )

    if args.checkpoint.async_save and "fsdp2" in _ckpt_futures and _ckpt_futures["fsdp2"] is not None:
        _ckpt_futures["fsdp2"].result()

    perf_logger.finish()
    torch.distributed.destroy_process_group()

    min_loss_val = perf_logger.min_loss.item()
    if min_loss_val == float("inf"):
        return None
    return min_loss_val


if __name__ == "__main__":
    main()
