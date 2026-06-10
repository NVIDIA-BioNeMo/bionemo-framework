import argparse
import gc
import json
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
import transformer_engine.pytorch as te
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity, profile

from dataset import create_dataloader
from distributed_config import DistributedConfig
from modeling_esm2_minifold_te import ESM2MiniFoldTE
from quantization import ComponentPrecisionConfig, resolve_layer_precision
from scheduler import get_linear_schedule_with_warmup
from train_fsdp2 import compute_distogram_loss, set_global_seed
from train_fsdp2_adam import (
    _cast_model_floating_params,
    _get_quantized_model_init_ctx,
    _materialize_meta_te_modules,
    _seed_fp32_master_weights,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _resolve_recipes(args):
    block_precision = resolve_layer_precision(
        num_layers=args.model.num_blocks,
        fp8_enabled=args.fp8_config.enabled,
        fp4_enabled=args.fp4_config.enabled,
        fp8_layers=OmegaConf.to_container(args.fp8_layers, resolve=True) if args.fp8_layers is not None else None,
        fp4_layers=OmegaConf.to_container(args.fp4_layers, resolve=True) if args.fp4_layers is not None else None,
    )

    fp8_recipe = None
    fp4_recipe = None
    if args.fp8_config.enabled:
        from transformer_engine.common.recipe import Format

        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )
    if args.fp4_config.enabled:
        from transformer_engine.common.recipe import Format

        fp4_recipe = hydra.utils.get_class(args.fp4_config.fp4_recipe)(
            fp4_format=Format[args.fp4_config.fp4_format], **args.fp4_config.fp4_recipe_kwargs
        )
    component_precision = ComponentPrecisionConfig(**OmegaConf.to_container(args.component_precision, resolve=True))
    return block_precision, fp8_recipe, fp4_recipe, component_precision


def _build_model_common(args, params_dtype):
    block_precision, fp8_recipe, fp4_recipe, component_precision = _resolve_recipes(args)
    use_quantized_model_init = bool(getattr(args, "fp8_model_init", None) and args.fp8_model_init.enabled)
    init_ctx = _get_quantized_model_init_ctx(args, fp8_recipe, fp4_recipe) if use_quantized_model_init else nullcontext()
    with init_ctx:
        model = ESM2MiniFoldTE(
            esm_model_name=args.esm_model_name,
            c_s=args.model.c_s,
            c_z=args.model.c_z,
            num_blocks=args.model.num_blocks,
            no_bins=args.model.no_bins,
            use_structure_module=args.model.use_structure_module,
            params_dtype=params_dtype,
            block_precision=block_precision,
            fp8_recipe=fp8_recipe,
            fp4_recipe=fp4_recipe,
            component_precision=component_precision,
            te_module_device="meta" if use_quantized_model_init else None,
        )
    return model, block_precision


def _param_groups(model, args):
    param_groups = [
        {
            "params": list(model.get_folding_head_params()),
            "lr": args.optimizer.folding_lr,
            "name": "folding_head",
        }
    ]
    if args.model.use_structure_module:
        param_groups.append(
            {
                "params": list(model.get_structure_module_params()),
                "lr": args.optimizer.struct_lr,
                "name": "structure_module",
            }
        )
    return param_groups


def _build_old_variant(args, dist_config: DistributedConfig, device: torch.device):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.optim import AdamW

    model, block_precision = _build_model_common(args, params_dtype=torch.float32)
    model = model.to(device)
    mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("dp",))
    if args.use_fp32_master_weights:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=False,
        )
    else:
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    for block in model.fold.miniformer.blocks:
        fully_shard(block, mesh=mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=mesh["dp"], mp_policy=mp_policy)
    optimizer = AdamW(
        _param_groups(model, args),
        betas=tuple(args.optimizer.betas),
        eps=args.optimizer.eps,
        weight_decay=args.optimizer.weight_decay,
        fused=True,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)
    return model, optimizer, scheduler, block_precision


def _build_adam_variant(args, dist_config: DistributedConfig, device: torch.device):
    from torch.distributed._composable.fsdp import fully_shard

    use_quantized_model_init = bool(getattr(args, "fp8_model_init", None) and args.fp8_model_init.enabled)
    params_dtype = torch.float32 if use_quantized_model_init else torch.bfloat16
    model, block_precision = _build_model_common(args, params_dtype=params_dtype)
    if use_quantized_model_init:
        model.backbone.to(device)
    else:
        model = model.to(device)
        _cast_model_floating_params(model, params_dtype)
    mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("dp",))
    for block in model.fold.miniformer.blocks:
        fully_shard(block, mesh=mesh["dp"])
    fully_shard(model, mesh=mesh["dp"])
    _materialize_meta_te_modules(model, device)
    optimizer = te.optimizers.FusedAdam(
        _param_groups(model, args),
        betas=tuple(args.optimizer.betas),
        eps=args.optimizer.eps,
        weight_decay=args.optimizer.weight_decay,
        master_weights=bool(args.use_fp32_master_weights),
        master_weight_dtype=torch.float32,
    )
    if args.use_fp32_master_weights and getattr(args, "fp8_model_init", None) and args.fp8_model_init.preserve_high_precision_init_val:
        _seed_fp32_master_weights(model, optimizer, device, params_dtype)
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)
    return model, optimizer, scheduler, block_precision


def _forward_backward_step(model, optimizer, scheduler, batch, args, device, variant):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    unpadded_tokens = float(batch["mask"].sum().item()) * torch.distributed.get_world_size()
    r_dict = model(batch, num_recycling=args.model.get("num_recycling", 0))
    disto_loss = compute_distogram_loss(
        preds=r_dict["preds"],
        coords=batch["coords"],
        mask=batch["mask"],
        no_bins=args.model.no_bins,
    )
    total_loss = disto_loss
    if args.model.use_structure_module and "sm" in r_dict:
        from loss import AlphaFoldLoss

        loss_of, _ = AlphaFoldLoss(r_dict, batch.get("batch_of", {}))
        total_loss = 0.8 * disto_loss + 0.2 * loss_of

    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    if variant == "adam":
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad()
    return {
        "loss": float(total_loss.detach().cpu()),
        "disto_loss": float(disto_loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        "unpadded_tokens": unpadded_tokens,
    }


def _write_rank_outputs(output_dir: Path, variant: str, rank: int, profiler_obj) -> dict:
    rank_dir = output_dir / variant / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    table = profiler_obj.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=30)
    (rank_dir / "key_averages.txt").write_text(table)
    return {"table_path": str(rank_dir / "key_averages.txt")}


def main():
    parser = argparse.ArgumentParser(description="Profile old vs adam FSDP2 recipe memory usage.")
    parser.add_argument("--variant", choices=("old", "adam"), required=True)
    parser.add_argument("--config-name", default="run_100_real_3B")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()

    os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
    config_dir = str((Path(__file__).resolve().parent / "hydra_config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        args = compose(config_name=cli.config_name, overrides=cli.overrides)

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    set_global_seed(int(args.seed), dist_config.local_rank)

    builder = _build_old_variant if cli.variant == "old" else _build_adam_variant
    model, optimizer, scheduler, block_precision = builder(args, dist_config, device)
    train_dataloader, sampler = create_dataloader(dist_config, **args.dataset)
    data_iter = iter(train_dataloader)

    for warmup_idx in range(cli.warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(warmup_idx + 1)
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        _forward_backward_step(model, optimizer, scheduler, batch, args, device, cli.variant)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step_metrics = []
    step_times_ms = []
    last_prof = None
    for step_idx in range(cli.measure_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(cli.warmup_steps + step_idx + 1)
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            step_start = time.perf_counter()
            metrics = _forward_backward_step(model, optimizer, scheduler, batch, args, device, cli.variant)
            step_times_ms.append((time.perf_counter() - step_start) * 1000.0)
        step_metrics.append(metrics)
        last_prof = prof

    files = _write_rank_outputs(cli.output_dir, cli.variant, dist_config.rank, last_prof)
    payload = {
        "variant": cli.variant,
        "config_name": cli.config_name,
        "overrides": cli.overrides,
        "rank": dist_config.rank,
        "world_size": dist_config.world_size,
        "block_precision": block_precision,
        "measure_steps": cli.measure_steps,
        "step_metrics": step_metrics[-1],
        "mean_step_time_ms": sum(step_times_ms) / len(step_times_ms),
        "mean_unpadded_tokens_per_sec": sum(
            (m["unpadded_tokens"] / (t / 1000.0)) for m, t in zip(step_metrics, step_times_ms)
        )
        / len(step_times_ms),
        "max_loss": max(m["loss"] for m in step_metrics),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
        **files,
    }

    rank_dir = cli.output_dir / cli.variant / f"rank{dist_config.rank}"
    (rank_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    if dist_config.rank == 0:
        print(json.dumps(payload, indent=2))

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
