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

"""Benchmark: dense vs Triton sparse TopK decoder.

Measures forward+backward latency and peak GPU memory across a sweep of latent
counts, isolating the win the sparse kernel gives as ``n_latents`` grows (the dense
path OOMs first — that OOM is itself the headline result). Two modes:

    --mode kernel  : the decode op alone (TritonDecoderAutograd vs dense reference)
    --mode sae     : a full TopKSAE.loss() fwd+bwd (decoder_impl dense vs triton)

Usage (on a GPU box):
    python -m sae.benchmarks.bench_decoder --impl all --mode kernel
    python -m sae.benchmarks.bench_decoder --impl all --mode sae --batch 4096 --d 2688
    python -m sae.benchmarks.bench_decoder --json out.json
"""

import argparse
import json

import torch

from sae.kernels import HAS_TRITON, TritonDecoderAutograd, reference_decode


# (label, expansion) for d_model=2688 -> n_latents; expansion is informational.
DEFAULT_NS = [21_504, 86_016, 344_064, 688_128, 1_048_576]


def _unique_topk(a, n, k, d, dtype, device):
    scores = torch.rand(a, n, device=device)
    idx = scores.argsort(dim=-1)[:, :k].contiguous().to(torch.int64)
    vals = torch.rand(a, k, device=device, dtype=dtype).contiguous()
    w = torch.randn(d, n, device=device, dtype=dtype)
    return idx, vals, w


def _time_fwd_bwd(fn, iters, warmup):
    """Return (fwd_ms, bwd_ms) medians, or raise the underlying error (e.g. OOM)."""
    fwd, bwd = [], []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        m = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out, bwd_fn = fn()
        m.record()
        bwd_fn(out)
        e.record()
        torch.cuda.synchronize()
        if i >= warmup:
            fwd.append(s.elapsed_time(m))
            bwd.append(m.elapsed_time(e))
    fwd.sort()
    bwd.sort()
    return fwd[len(fwd) // 2], bwd[len(bwd) // 2]


def bench_cell(impl, mode, a, n, k, d, dtype, device, iters, warmup):
    """Benchmark one (impl, n) cell. Returns dict with timings + peak mem, or OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        if mode == "kernel":
            idx, vals, w = _unique_topk(a, n, k, d, dtype, device)

            def fn():
                v = vals.clone().requires_grad_(True)
                ww = w.clone().requires_grad_(True)
                gseed = torch.randn(a, d, device=device, dtype=dtype)
                if impl == "triton":
                    out = TritonDecoderAutograd.apply(idx, v, ww)
                else:
                    out = reference_decode(idx, v, ww)
                return (out * gseed).sum(), lambda loss: loss.backward()
        else:  # sae
            from sae.architectures import TopKSAE

            sae = TopKSAE(input_dim=d, hidden_dim=n, top_k=k, normalize_input=True, decoder_impl=impl).to(device)
            x = torch.randn(a, d, device=device, dtype=dtype)

            def fn():
                sae.zero_grad(set_to_none=True)
                out = sae.loss(x)["total"]
                return out, lambda loss: loss.backward()

        fwd_ms, bwd_ms = _time_fwd_bwd(fn, iters, warmup)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        return {"fwd_ms": round(fwd_ms, 3), "bwd_ms": round(bwd_ms, 3), "peak_gb": round(peak_gb, 2)}
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return {"status": "OOM"}
        raise


def main():
    """Parse args and run the dense-vs-Triton decoder benchmark sweep."""
    p = argparse.ArgumentParser(description="Benchmark dense vs Triton sparse TopK decoder")
    p.add_argument("--impl", choices=["dense", "triton", "all"], default="all")
    p.add_argument("--mode", choices=["kernel", "sae"], default="kernel")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--d", type=int, default=2688)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark.")
    if args.impl in ("triton", "all") and not HAS_TRITON:
        raise SystemExit("Triton not available; install triton or use --impl dense.")

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    impls = ["dense", "triton"] if args.impl == "all" else [args.impl]
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu} | mode={args.mode} | batch={args.batch} d={args.d} k={args.k} dtype={args.dtype}\n")

    results = []
    header = f"{'n_latents':>10} {'exp':>5} " + " ".join(f"{i:>22}" for i in impls)
    print(header)
    print("-" * len(header))
    for n in args.ns:
        row = {"n_latents": n, "expansion": round(n / args.d, 1)}
        cells = []
        for impl in impls:
            r = bench_cell(impl, args.mode, args.batch, n, args.k, args.d, dtype, "cuda", args.iters, args.warmup)
            row[impl] = r
            if "status" in r:
                cells.append(f"{'OOM':>22}")
            else:
                cells.append(f"{r['fwd_ms']:>6.2f}/{r['bwd_ms']:>6.2f}ms {r['peak_gb']:>5.1f}GB")
        results.append(row)
        print(f"{n:>10} {row['expansion']:>5}x " + " ".join(cells))

    print("\n(cells: fwd/bwd ms, peak GB; 'OOM' = ran out of memory)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"gpu": gpu, "args": vars(args), "results": results}, f, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
