from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tri_mul_ext"))

from te_utils import tri_mul_bmm  # noqa: E402
from tri_mul_ext import extension_available, tri_mul_fused  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequence-length sweep for triangular multiplication backends.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--dim-chunk", type=int, default=32, help="Per-branch D after 4-way split.")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[128, 256, 384, 512])
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def bench(fn, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters, out


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)

    print(f"cuda_extension_available={extension_available()}")
    print(f"batch={args.batch} dim_chunk={args.dim_chunk} dtype={dtype}")

    for n in args.sequence_lengths:
        a = torch.randn((args.batch, n, n, args.dim_chunk), device="cuda", dtype=dtype)
        b = torch.randn((args.batch, n, n, args.dim_chunk), device="cuda", dtype=dtype)
        print(f"\nN={n}")
        for k_dim in (2, 1):
            bmm_ms, out_ref = bench(lambda: tri_mul_bmm(a, b, k_dim=k_dim, mode="bf16"), args.warmup, args.iters)
            fused_ms, out = bench(lambda: tri_mul_fused(a, b, k_dim=k_dim, out_dtype=dtype), args.warmup, args.iters)
            max_abs = (out.float() - out_ref.float()).abs().max().item()
            print(
                f"  k_dim={k_dim} "
                f"bmm_ms={bmm_ms:.4f} "
                f"fused_ms={fused_ms:.4f} "
                f"speedup={bmm_ms / fused_ms:.3f} "
                f"max_abs={max_abs:.6f}"
            )


if __name__ == "__main__":
    main()
