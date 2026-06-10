#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import OmegaConf

from plain_minifold_infer import (
    LINEAR_PRECISION_BF16,
    LINEAR_PRECISION_FP8,
    SUPPORTED_LINEAR_PRECISION,
    PAIR_PRECISION_BF16,
    PAIR_PRECISION_FP8_EXTREME,
    PAIR_PRECISION_FP8_NATIVE,
    SUPPORTED_PAIR_PRECISION,
    SUPPORTED_TRI_IMPLS,
    FoldingTrunk,
    collect_fp8_linear_storage_stats,
    configure_linear_precision,
    resolve_linear_precision,
    resolve_pair_precision,
    validate_fp8_extreme_configuration,
)
from torch_compile_diagnostics import add_torch_compile_arguments, maybe_capture_dynamo_diagnostics, maybe_compile


DEFAULT_CONFIG = Path(__file__).resolve().parent / "hydra_config" / "inference_single_config.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple single-config non-TE MiniFold folding inference benchmark.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--mbs", type=int)
    parser.add_argument("--tri_impl", choices=SUPPORTED_TRI_IMPLS)
    parser.add_argument("--tri_einsum", choices=["off", "bf16"])
    parser.add_argument("--pair_precision", choices=SUPPORTED_PAIR_PRECISION)
    parser.add_argument("--linear_precision", choices=SUPPORTED_LINEAR_PRECISION)
    parser.add_argument("--fp8_activations", action="store_const", const=True, default=None)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_blocks", type=int)
    parser.add_argument("--c_s", type=int)
    parser.add_argument("--c_z", type=int)
    parser.add_argument("--no_bins", type=int)
    parser.add_argument("--bins", type=int)
    parser.add_argument("--num_recycling", type=int)
    parser.add_argument("--output", default="")
    add_torch_compile_arguments(parser)
    return parser


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    cfg = OmegaConf.load(args.config)
    cfg_pair_precision = OmegaConf.select(cfg, "pair_precision")
    cfg_fp8_activations = bool(OmegaConf.select(cfg, "fp8_activations", default=False))
    merged = {
        "seq_len": cfg.seq_len,
        "mbs": cfg.mbs,
        "tri_impl": cfg.tri_impl,
        "tri_einsum": cfg.tri_einsum,
        "pair_precision": resolve_pair_precision(cfg_pair_precision, cfg_fp8_activations),
        "linear_precision": resolve_linear_precision(OmegaConf.select(cfg, "linear_precision")),
        "fp8_activations": cfg_fp8_activations,
        "warmup": cfg.warmup,
        "iters": cfg.iters,
        "seed": cfg.seed,
        "num_blocks": cfg.num_blocks,
        "c_s": cfg.c_s,
        "c_z": cfg.c_z,
        "no_bins": cfg.no_bins,
        "bins": cfg.bins,
        "num_recycling": cfg.num_recycling,
        "output": cfg.output,
        "config": args.config,
        "use_torch_compile": False,
        "torch_compile_mode": "default",
        "diagnostics_output": None,
    }
    for key in merged:
        value = getattr(args, key, None)
        if value not in (None, ""):
            merged[key] = value
    if getattr(args, "pair_precision", None) is None and getattr(args, "fp8_activations", None):
        merged["pair_precision"] = None
    merged["pair_precision"] = resolve_pair_precision(merged.get("pair_precision"), merged.get("fp8_activations"))
    merged["linear_precision"] = resolve_linear_precision(merged.get("linear_precision"))
    merged["fp8_activations"] = merged["pair_precision"] != PAIR_PRECISION_BF16
    return argparse.Namespace(**merged)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace, device: torch.device) -> FoldingTrunk:
    validate_fp8_extreme_configuration(args.pair_precision, args.linear_precision, args.tri_impl)
    model = FoldingTrunk(
        c_s=args.c_s,
        c_z=args.c_z,
        bins=args.bins,
        disto_bins=args.no_bins,
        num_layers=args.num_blocks,
        tri_impl=args.tri_impl,
        tri_einsum=args.tri_einsum,
        pair_precision=args.pair_precision,
        linear_precision=args.linear_precision,
    ).to(device=device, dtype=torch.bfloat16)
    configure_linear_precision(
        model,
        args.linear_precision,
        include_transition=args.pair_precision in (PAIR_PRECISION_FP8_EXTREME, PAIR_PRECISION_FP8_NATIVE),
    )
    model.eval()
    return model


def generate_inputs(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "s_s": torch.randn(args.mbs, args.seq_len, args.c_s, device=device, dtype=torch.bfloat16),
        "s_z": torch.randn(args.mbs, args.seq_len, args.seq_len, args.c_z, device=device, dtype=torch.bfloat16),
        "mask": torch.ones(args.mbs, args.seq_len, device=device, dtype=torch.bool),
    }


def forward_once(
    model_runner,
    batch: dict[str, torch.Tensor],
    num_recycling: int,
    *,
    use_bf16_autocast: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16_autocast else nullcontext()
    with torch.no_grad(), autocast_ctx:
        return model_runner(batch["s_s"], batch["s_z"], batch["mask"], num_recycling=num_recycling)


def benchmark(
    model: FoldingTrunk,
    model_runner,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    *,
    use_bf16_autocast: bool,
) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.warmup):
        forward_once(model_runner, batch, args.num_recycling, use_bf16_autocast=use_bf16_autocast)
        torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times_ms: list[float] = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        forward_once(model_runner, batch, args.num_recycling, use_bf16_autocast=use_bf16_autocast)
        end.record()
        torch.cuda.synchronize(device)
        times_ms.append(float(start.elapsed_time(end)))

    median_ms = float(statistics.median(times_ms))
    result = {
        "pair_precision_mode": args.pair_precision,
        "linear_precision_mode": args.linear_precision,
        "fp8_weights": args.linear_precision == LINEAR_PRECISION_FP8,
        "median_ms": median_ms,
        "mean_ms": float(statistics.mean(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "proteins_per_sec": args.mbs / (median_ms / 1000.0),
        "unpadded_tokens_per_sec": (args.mbs * args.seq_len) / (median_ms / 1000.0),
        "peak_memory_allocated_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "peak_memory_reserved_gib": float(torch.cuda.max_memory_reserved(device) / (1024**3)),
    }
    stats = model.miniformer.last_pair_precision_stats
    if stats is not None:
        result["fp8_activation_storage_mib"] = stats.quantized_bytes / (1024**2)
        result["fp8_scale_storage_mib"] = stats.scale_bytes / (1024**2)
        result["fp8_quantize_ops"] = stats.quantize_ops
        result["fp8_dequantize_ops"] = stats.dequantize_ops
        result["fp8_tensorwise_repack_ops"] = stats.tensorwise_repack_ops
        result["fp8_tensorwise_repack_payload_mib"] = stats.tensorwise_repack_payload_bytes / (1024**2)
        result["fp8_tensorwise_repack_scale_mib"] = stats.tensorwise_repack_scale_bytes / (1024**2)
        result["fp8_linear_requant_ops"] = stats.linear_requant_ops
        result["fp8_linear_requant_payload_mib"] = stats.linear_requant_payload_bytes / (1024**2)
        result["fp8_linear_requant_scale_mib"] = stats.linear_requant_scale_bytes / (1024**2)
        result["fp8_tri_pack_ops"] = stats.tri_pack_ops
        result["fp8_tri_pack_payload_mib"] = stats.tri_pack_payload_bytes / (1024**2)
        result["fp8_tri_pack_scale_mib"] = stats.tri_pack_scale_bytes / (1024**2)
        result["fp8_tri_output_requant_ops"] = stats.tri_output_requant_ops
        result["fp8_tri_output_requant_payload_mib"] = stats.tri_output_requant_payload_bytes / (1024**2)
        result["fp8_tri_output_requant_scale_mib"] = stats.tri_output_requant_scale_bytes / (1024**2)
        result["fp8_native_linear_fused_ops"] = stats.native_linear_fused_ops
        result["fp8_native_linear_payload_mib"] = stats.native_linear_payload_bytes / (1024**2)
        result["fp8_native_linear_scale_mib"] = stats.native_linear_scale_bytes / (1024**2)
        result["fp8_native_gate_fused_ops"] = stats.native_gate_fused_ops
        result["fp8_native_gate_payload_mib"] = stats.native_gate_payload_bytes / (1024**2)
        result["fp8_native_gate_scale_mib"] = stats.native_gate_scale_bytes / (1024**2)
        result["fp8_native_tri_fused_ops"] = stats.native_tri_fused_ops
        result["fp8_native_tri_payload_mib"] = stats.native_tri_payload_bytes / (1024**2)
        result["fp8_native_tri_scale_mib"] = stats.native_tri_scale_bytes / (1024**2)
        result["pair_boundary_counts"] = stats.boundary_counts
    linear_stats = collect_fp8_linear_storage_stats(model)
    if linear_stats["linear_count"] > 0:
        result["fp8_linear_count"] = linear_stats["linear_count"]
        result["fp8_linear_weight_storage_mib"] = linear_stats["weight_bytes"] / (1024**2)
        result["fp8_linear_scale_storage_mib"] = linear_stats["scale_bytes"] / (1024**2)
        result["fp8_linear_tensorwise_weight_storage_mib"] = linear_stats["tensorwise_weight_bytes"] / (1024**2)
        result["fp8_linear_tensorwise_scale_storage_mib"] = linear_stats["tensorwise_scale_bytes"] / (1024**2)
    return result


def render_markdown(args: argparse.Namespace, result: dict[str, float]) -> str:
    pair_precision = resolve_pair_precision(getattr(args, "pair_precision", None), getattr(args, "fp8_activations", None))
    linear_precision = resolve_linear_precision(getattr(args, "linear_precision", None))
    notes = []
    if pair_precision != PAIR_PRECISION_BF16:
        notes.append(pair_precision)
    if linear_precision != LINEAR_PRECISION_BF16:
        notes.append(f"linear={linear_precision}")
    return "\n".join(
        [
            "| tri_impl | pair precision | linear precision | seq_len | mbs | median (ms) | prot/s | tok/s | peak mem (GiB) | notes |",
            "|----------|----------------|------------------|---------|-----|-------------|--------|-------|----------------|-------|",
            (
                f"| {args.tri_impl} | {pair_precision} | {linear_precision} | {args.seq_len} | {args.mbs} | {result['median_ms']:.2f} | "
                f"{result['proteins_per_sec']:.2f} | {result['unpadded_tokens_per_sec']:.2f} | "
                f"{result['peak_memory_allocated_gib']:.2f} | {', '.join(notes)} |"
            ),
        ]
    )


def main() -> None:
    raw_args = build_arg_parser().parse_args()
    args = load_config(raw_args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference_single_config.py")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    model = build_model(args, device)
    model_runner = maybe_compile(model, enabled=args.use_torch_compile, mode=args.torch_compile_mode)
    batch = generate_inputs(args, device)
    use_bf16_autocast = args.pair_precision not in (PAIR_PRECISION_FP8_EXTREME, PAIR_PRECISION_FP8_NATIVE)

    def _diagnostic_call(s_s: torch.Tensor, s_z: torch.Tensor, mask: torch.Tensor):
        return forward_once(
            model_runner,
            {"s_s": s_s, "s_z": s_z, "mask": mask},
            args.num_recycling,
            use_bf16_autocast=use_bf16_autocast,
        )

    diagnostics = maybe_capture_dynamo_diagnostics(
        _diagnostic_call,
        args=(batch["s_s"], batch["s_z"], batch["mask"]),
        output_path=args.diagnostics_output,
        label="inference_single_config",
        rank=0,
        world_size=1,
        extra_metadata={
            "use_torch_compile": args.use_torch_compile,
            "torch_compile_mode": args.torch_compile_mode,
            "tri_impl": args.tri_impl,
            "pair_precision": args.pair_precision,
            "linear_precision": args.linear_precision,
        },
    )

    result = benchmark(model, model_runner, batch, args, device, use_bf16_autocast=use_bf16_autocast)
    markdown = render_markdown(args, result)
    payload = {
        "config": {
            "config_file": args.config,
            "seq_len": args.seq_len,
            "mbs": args.mbs,
            "tri_impl": args.tri_impl,
            "tri_einsum": args.tri_einsum,
            "pair_precision": args.pair_precision,
            "linear_precision": args.linear_precision,
            "fp8_activations": args.fp8_activations,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "num_blocks": args.num_blocks,
            "c_s": args.c_s,
            "c_z": args.c_z,
            "no_bins": args.no_bins,
            "bins": args.bins,
            "num_recycling": args.num_recycling,
            "implementation": "plain_pytorch_single_config",
            "mode": "mock_backbone_only",
            "use_torch_compile": args.use_torch_compile,
            "torch_compile_mode": args.torch_compile_mode,
            "diagnostics_output": str(args.diagnostics_output) if args.diagnostics_output else None,
        },
        "result": result,
        "dynamo_diagnostics": diagnostics,
        "markdown_summary": markdown,
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        out_path.with_suffix(".md").write_text(markdown + "\n")
        print(markdown)
        print(f"\nJSON: {out_path}")
        print(f"Markdown: {out_path.with_suffix('.md')}")
    else:
        print(json.dumps(payload, indent=2))
        print()
        print(markdown)


if __name__ == "__main__":
    main()
