import argparse
import json
import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW

from dataset import create_dataloader
from distributed_config import DistributedConfig
from modeling_esm2_minifold_te import ESM2MiniFoldTE
from quantization import ComponentPrecisionConfig, resolve_layer_precision
from scheduler import get_linear_schedule_with_warmup
from train_fsdp2 import compute_distogram_loss, compute_distogram_metrics, set_global_seed


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _cuda_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def _make_model_and_optimizer(args, dist_config: DistributedConfig, device: torch.device):
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
    model = ESM2MiniFoldTE(
        esm_model_name=args.esm_model_name,
        c_s=args.model.c_s,
        c_z=args.model.c_z,
        num_blocks=args.model.num_blocks,
        no_bins=args.model.no_bins,
        use_structure_module=args.model.use_structure_module,
        params_dtype=torch.float32,
        block_precision=block_precision,
        fp8_recipe=fp8_recipe,
        fp4_recipe=fp4_recipe,
        component_precision=component_precision,
    ).to(device)

    device_mesh = init_device_mesh("cuda", mesh_shape=(dist_config.world_size,), mesh_dim_names=("dp",))
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
        fully_shard(block, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    param_groups = [
        {
            "params": list(model.get_folding_head_params()),
            "lr": args.optimizer.folding_lr,
            "name": "folding_head",
        },
    ]
    if args.model.use_structure_module:
        param_groups.append(
            {
                "params": list(model.get_structure_module_params()),
                "lr": args.optimizer.struct_lr,
                "name": "structure_module",
            }
        )

    optimizer = AdamW(
        param_groups,
        betas=tuple(args.optimizer.betas),
        eps=args.optimizer.eps,
        weight_decay=args.optimizer.weight_decay,
        fused=True,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)
    return model, optimizer, scheduler


def _profile_step(model, optimizer, scheduler, batch, args, device: torch.device) -> dict[str, float]:
    timings = {}
    batch_device = {}
    timings["batch_to_device_ms"] = _cuda_ms(
        lambda: batch_device.update({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
    )

    r_dict = {}
    timings["forward_ms"] = _cuda_ms(lambda: r_dict.update(model(batch_device, num_recycling=args.model.get("num_recycling", 0))))

    losses = {}

    def _loss():
        disto_loss = compute_distogram_loss(
            preds=r_dict["preds"],
            coords=batch_device["coords"],
            mask=batch_device["mask"],
            no_bins=args.model.no_bins,
        )
        total_loss = disto_loss
        if args.model.use_structure_module and "sm" in r_dict:
            from loss import AlphaFoldLoss

            loss_of, _ = AlphaFoldLoss(r_dict, batch_device.get("batch_of", {}))
            total_loss = 0.8 * disto_loss + 0.2 * loss_of
        losses["disto_loss"] = disto_loss
        losses["total_loss"] = total_loss

    timings["loss_ms"] = _cuda_ms(_loss)
    timings["backward_ms"] = _cuda_ms(lambda: losses["total_loss"].backward())
    clip_norm = {}
    timings["clip_ms"] = _cuda_ms(lambda: clip_norm.update({"value": torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()}))

    def _optim():
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    timings["optim_ms"] = _cuda_ms(_optim)
    metrics = {}
    timings["metrics_ms"] = _cuda_ms(
        lambda: metrics.update(
            compute_distogram_metrics(
                preds=r_dict["preds"].float(),
                coords=batch_device["coords"],
                mask=batch_device["mask"],
                no_bins=args.model.no_bins,
            )
        )
    )
    timings["total_step_ms"] = sum(
        timings[k]
        for k in ("batch_to_device_ms", "forward_ms", "loss_ms", "backward_ms", "clip_ms", "optim_ms", "metrics_ms")
    )
    timings["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    return timings


def main():
    parser = argparse.ArgumentParser(description="Profile recipe training step breakdown for a chosen tri_impl backend.")
    parser.add_argument("--config-name", default="run_100_real")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
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

    model, optimizer, scheduler = _make_model_and_optimizer(args, dist_config, device)
    train_dataloader, sampler = create_dataloader(dist_config, **args.dataset)
    data_iter = iter(train_dataloader)

    warmup = cli.warmup_steps
    measure = cli.measure_steps
    measured_rows: list[dict[str, float]] = []

    for step_idx in range(warmup + measure):
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(step_idx + 1)
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        torch.cuda.reset_peak_memory_stats()
        row = _profile_step(model, optimizer, scheduler, batch, args, device)
        if step_idx >= warmup:
            measured_rows.append(row)

    summary = _mean_dict(measured_rows)
    payload = {
        "backend": args.component_precision.tri_impl,
        "config_name": cli.config_name,
        "overrides": cli.overrides,
        "warmup_steps": warmup,
        "measure_steps": measure,
        "rank": dist_config.rank,
        "world_size": dist_config.world_size,
        "summary": summary,
        "rows": measured_rows,
    }

    if dist_config.rank == 0:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
