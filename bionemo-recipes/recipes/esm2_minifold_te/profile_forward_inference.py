import argparse
import json
import logging
import os
import time
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from dataset import create_dataloader
from distributed_config import DistributedConfig
from modeling_esm2_minifold_te import ESM2MiniFoldTE
from quantization import ComponentPrecisionConfig, resolve_layer_precision
from torch_compile_diagnostics import add_torch_compile_arguments, maybe_capture_dynamo_diagnostics, maybe_compile
from train_fsdp2 import set_global_seed


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _make_model(args, dist_config: DistributedConfig, device: torch.device):
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
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    for block in model.fold.miniformer.blocks:
        fully_shard(block, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)
    return model


def _move_batch_to_device(batch, device: torch.device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _next_batch(data_iter, sampler, dataloader, *, device: torch.device, step_idx: int):
    try:
        batch = next(data_iter)
    except StopIteration:
        sampler.set_epoch(step_idx + 1)
        data_iter = iter(dataloader)
        batch = next(data_iter)
    return _move_batch_to_device(batch, device), data_iter


def _forward_once(model_runner, batch, *, num_recycling: int):
    with torch.no_grad():
        return model_runner(batch, num_recycling=num_recycling)


def main():
    parser = argparse.ArgumentParser(description="Profile forward-only inference throughput for a chosen tri_impl backend.")
    parser.add_argument("--config-name", default="run_100_real_3B")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    add_torch_compile_arguments(parser)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()

    os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("BIONEMO_ALLOW_CONFIG_ONLY_ESM2_INIT", "1")
    config_dir = str((Path(__file__).resolve().parent / "hydra_config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        args = compose(config_name=cli.config_name, overrides=cli.overrides)

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    set_global_seed(int(args.seed), dist_config.local_rank)

    model = _make_model(args, dist_config, device)
    model.eval()
    model_runner = maybe_compile(model, enabled=cli.use_torch_compile, mode=cli.torch_compile_mode)
    dataloader, sampler = create_dataloader(dist_config, **args.dataset)
    diagnostics = None
    if cli.diagnostics_output is not None:
        diagnostics_iter = iter(dataloader)
        diagnostics_batch, diagnostics_iter = _next_batch(
            diagnostics_iter, sampler, dataloader, device=device, step_idx=0
        )
        diagnostics = maybe_capture_dynamo_diagnostics(
            _forward_once,
            args=(model_runner, diagnostics_batch),
            kwargs={"num_recycling": args.model.get("num_recycling", 0)},
            output_path=cli.diagnostics_output,
            label="profile_forward_inference",
            rank=dist_config.rank,
            world_size=dist_config.world_size,
            extra_metadata={
                "use_torch_compile": cli.use_torch_compile,
                "torch_compile_mode": cli.torch_compile_mode,
                "config_name": cli.config_name,
                "tri_impl": args.component_precision.tri_impl,
                "overrides": list(cli.overrides),
            },
        )

    data_iter = iter(dataloader)

    measured = []
    for step_idx in range(cli.warmup_steps + cli.measure_steps):
        batch, data_iter = _next_batch(data_iter, sampler, dataloader, device=device, step_idx=step_idx)
        unpadded_tokens = float(batch["mask"].sum().item()) * torch.distributed.get_world_size()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _forward_once(model_runner, batch, num_recycling=args.model.get("num_recycling", 0))
        torch.cuda.synchronize()
        step_time_ms = (time.perf_counter() - start) * 1000.0
        row = {
            "step_idx": step_idx,
            "step_time_ms": step_time_ms,
            "unpadded_tokens": unpadded_tokens,
            "unpadded_tokens_per_sec": unpadded_tokens / (step_time_ms / 1000.0),
        }
        if step_idx >= cli.warmup_steps:
            measured.append(row)

    payload = {
        "backend": args.component_precision.tri_impl,
        "config_name": cli.config_name,
        "overrides": cli.overrides,
        "rank": dist_config.rank,
        "world_size": dist_config.world_size,
        "measure_steps": cli.measure_steps,
        "use_torch_compile": cli.use_torch_compile,
        "torch_compile_mode": cli.torch_compile_mode,
        "diagnostics_output": str(cli.diagnostics_output) if cli.diagnostics_output else None,
        "mean_step_time_ms": sum(r["step_time_ms"] for r in measured) / len(measured),
        "mean_unpadded_tokens_per_sec": sum(r["unpadded_tokens_per_sec"] for r in measured) / len(measured),
        "rows": measured,
        "dynamo_diagnostics": diagnostics,
    }

    if dist_config.rank == 0:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
