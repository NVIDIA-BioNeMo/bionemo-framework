import argparse

import torch

from te_utils import tri_mul_bmm_bdnn, tri_mul_fp8_cublaslt_bdnn, tri_mul_fp8_grouped_bdnn


def _bench_ms(fn, x, *, warmup: int, iters: int, with_backward: bool) -> float:
    for _ in range(warmup):
        y = fn(x)
        if with_backward:
            y.square().mean().backward()
            x.grad = None
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y = fn(x)
        if with_backward:
            y.square().mean().backward()
            x.grad = None
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _run_backend(name: str, x: torch.Tensor, out_dtype: torch.dtype, *, warmup: int, iters: int, with_backward: bool):
    if name == "bmm":
        def fn(t):
            a1, b1, a2, b2 = torch.chunk(t, 4, dim=1)
            x1 = tri_mul_bmm_bdnn(a1, b1, k_dim=2)
            x2 = tri_mul_bmm_bdnn(a2, b2, k_dim=1)
            return torch.cat([x1, x2], dim=-1)
    elif name == "fp8_cublaslt":
        fn = lambda t: tri_mul_fp8_cublaslt_bdnn(t, out_dtype=out_dtype)
    elif name == "fp8_grouped":
        fn = lambda t: tri_mul_fp8_grouped_bdnn(t, out_dtype=out_dtype)
    else:
        raise ValueError(f"Unsupported backend: {name}")

    return _bench_ms(fn, x, warmup=warmup, iters=iters, with_backward=with_backward)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MiniFold tri-mul FP8 backends on packed x_bdnn inputs.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--d-chunk", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["fp8_cublaslt", "fp8_grouped"],
        choices=["bmm", "fp8_cublaslt", "fp8_grouped"],
    )
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.seq_len % 32 != 0 or args.d_chunk % 32 != 0:
        raise SystemExit("--seq-len and --d-chunk must be divisible by 32")

    torch.cuda.set_device(0)
    x = torch.randn(
        (args.batch, 4 * args.d_chunk, args.seq_len, args.seq_len),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=not args.forward_only,
    )

    print(
        {
            "shape": (args.batch, 4 * args.d_chunk, args.seq_len, args.seq_len),
            "with_backward": not args.forward_only,
            "warmup": args.warmup,
            "iters": args.iters,
        }
    )
    for backend in args.backends:
        ms = _run_backend(
            backend,
            x,
            torch.float32,
            warmup=args.warmup,
            iters=args.iters,
            with_backward=not args.forward_only,
        )
        print({backend: ms})


if __name__ == "__main__":
    main()
