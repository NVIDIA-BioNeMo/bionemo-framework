import argparse
import time

import torch

from bmm_ext import pack_nvfp4
from bmm_ext.ops import bmm_block_scaled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs MXFP8 vs NVFP4 batched matmul.")
    parser.add_argument("--batch", type=int, default=1, help="Batch dimension.")
    parser.add_argument("--m", type=int, default=10240, help="Left matrix rows.")
    parser.add_argument("--k", type=int, default=10240, help="Reduction dimension.")
    parser.add_argument("--n", type=int, default=10240, help="Right matrix columns.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per kernel.")
    parser.add_argument("--iters", type=int, default=3, help="Timed iterations per kernel.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Source dtype before low-precision conversion.",
    )
    parser.add_argument(
        "--nvfp4-clip",
        type=float,
        default=6.0,
        help="Clamp range used before NVFP4 packing. The NVFP4 distortion reference uses the same clamped inputs.",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 for the FP32 torch.bmm reference. Disabled by default for a stricter FP32 baseline.",
    )
    return parser.parse_args()


def source_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported source dtype {name}")


def sync_timer(fn, warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    return out, elapsed_ms


def gib(num_bytes: int) -> float:
    return num_bytes / float(1024 ** 3)


def tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size()) * int(torch.Size(shape).numel())


def estimate_problem_bytes(batch: int, m: int, k: int, n: int) -> dict[str, int]:
    a = tensor_bytes((batch, m, k), torch.float32)
    b = tensor_bytes((batch, k, n), torch.float32)
    c = tensor_bytes((batch, m, n), torch.float32)
    return {"a": a, "b": b, "c": c, "total_fp32_live": a + b + c}


def distortion_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = (out - ref).abs()
    rel = diff / torch.clamp(ref.abs(), min=1e-6)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(torch.sqrt((diff * diff).mean())),
        "mean_rel": float(rel.mean()),
    }


def print_result(name: str, elapsed_ms: float, out: torch.Tensor, ref: torch.Tensor, k_dim: int) -> None:
    metrics = distortion_metrics(out, ref)
    tflops = 2.0 * ref.shape[0] * ref.shape[1] * ref.shape[2] * k_dim / (elapsed_ms * 1.0e9)
    print(
        f"{name:>6}  time_ms={elapsed_ms:9.3f}  "
        f"tflops={tflops:8.2f}  "
        f"max_abs={metrics['max_abs']:9.4f}  "
        f"mean_abs={metrics['mean_abs']:9.4f}  "
        f"rmse={metrics['rmse']:9.4f}  "
        f"mean_rel={metrics['mean_rel']:9.4f}"
    )


def benchmark_fp32(a_fp32: torch.Tensor, b_fp32: torch.Tensor, warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    return sync_timer(lambda: torch.bmm(a_fp32, b_fp32), warmup=warmup, iters=iters)


def benchmark_mxfp8(
    a_src: torch.Tensor,
    b_src: torch.Tensor,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float]:
    a = a_src.to(torch.float8_e4m3fn)
    b = b_src.to(torch.float8_e4m3fn)
    a_scale = torch.ones((a.shape[0], a.shape[1], a.shape[2] // 32), device=a.device, dtype=torch.float8_e8m0fnu)
    b_scale = torch.ones((b.shape[0], b.shape[1] // 32, b.shape[2]), device=b.device, dtype=torch.float8_e8m0fnu)
    return sync_timer(
        lambda: bmm_block_scaled(
            a,
            b,
            a_scale=a_scale,
            b_scale=b_scale,
            format="mxfp8",
            out_dtype=torch.float32,
            sf_vec_size=32,
        ),
        warmup=warmup,
        iters=iters,
    )


def benchmark_nvfp4(
    a_fp32: torch.Tensor,
    b_fp32: torch.Tensor,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float]:
    a = pack_nvfp4(a_fp32, role="lhs")
    b = pack_nvfp4(b_fp32, role="rhs")
    return sync_timer(
        lambda: bmm_block_scaled(
            a,
            b,
            a_scale=a.scale_inv,
            b_scale=b.scale_inv,
            format="nvfp4",
            out_dtype=torch.float32,
            sf_vec_size=16,
        ),
        warmup=warmup,
        iters=iters,
    )


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    if args.k % 32 != 0:
        raise ValueError("--k must be divisible by 32 so both MXFP8 and NVFP4 are legal.")
    if args.n % 16 != 0:
        raise ValueError("--n must be divisible by 16 for NVFP4 RHS packing.")
    if (args.batch * args.m) % 16 != 0:
        raise ValueError("--batch * --m must be divisible by 16 for NVFP4 LHS packing.")

    src_dtype = source_dtype(args.dtype)
    a_src = torch.randn((args.batch, args.m, args.k), device="cuda", dtype=src_dtype)
    b_src = torch.randn((args.batch, args.k, args.n), device="cuda", dtype=src_dtype)

    a_fp32 = a_src.float()
    b_fp32 = b_src.float()
    a_nvfp4_src = a_fp32.clamp(-args.nvfp4_clip, args.nvfp4_clip)
    b_nvfp4_src = b_fp32.clamp(-args.nvfp4_clip, args.nvfp4_clip)
    mem = estimate_problem_bytes(args.batch, args.m, args.k, args.n)

    print(f"shape=(B={args.batch}, M={args.m}, K={args.k}, N={args.n})  source_dtype={src_dtype}")
    print(f"warmup={args.warmup}  iters={args.iters}  tf32_reference={args.allow_tf32}")
    print(
        "estimated_fp32_bytes="
        f"A:{gib(mem['a']):.2f}GiB  "
        f"B:{gib(mem['b']):.2f}GiB  "
        f"C:{gib(mem['c']):.2f}GiB  "
        f"live_total:{gib(mem['total_fp32_live']):.2f}GiB"
    )

    fp32_out, fp32_ms = benchmark_fp32(a_fp32, b_fp32, warmup=args.warmup, iters=args.iters)
    print_result("FP32", fp32_ms, fp32_out, fp32_out, args.k)

    mxfp8_out, mxfp8_ms = benchmark_mxfp8(a_src, b_src, warmup=args.warmup, iters=args.iters)
    print_result("MXFP8", mxfp8_ms, mxfp8_out, fp32_out, args.k)

    nvfp4_ref, _ = benchmark_fp32(a_nvfp4_src, b_nvfp4_src, warmup=0, iters=1)
    nvfp4_out, nvfp4_ms = benchmark_nvfp4(a_nvfp4_src, b_nvfp4_src, warmup=args.warmup, iters=args.iters)
    print_result("NVFP4", nvfp4_ms, nvfp4_out, nvfp4_ref, args.k)
    print("note: NVFP4 distortion is measured against FP32 torch.bmm on the same clamped inputs used for NVFP4 packing.")


if __name__ == "__main__":
    main()
