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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Compare native BF16, FP8, MXFP8, and NVFP4 projection kernels.

The default dimensions match an Evo2 7B expansion projection. Weights are quantized once, as in
inference. ``*_raw`` measures only GEMM; ``*_quantize`` isolates activation conversion; and
``*_full`` measures activation quantization plus GEMM. Use ``--profile-range`` with Nsight's CUDA
profiler capture range to inspect only warmed kernels without capturing the process environment.
Hopper reports BF16 versus native regular FP8; Blackwell additionally reports native MXFP8 and
NVFP4. The Hopper path does not import or require FlashInfer.
"""

import argparse
import json
from collections.abc import Callable

import torch
from torch.nn import functional


_FP8_MAX = 448.0
_NVFP4_COMBINED_RANGE = 6.0 * _FP8_MAX


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=22528)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--regular-fp8-only",
        action="store_true",
        help="Skip Blackwell block-scaled formats; useful for isolating the Hopper-compatible FP8 path",
    )
    parser.add_argument("--profile-range", action="store_true")
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        raise ValueError("m, n, and k must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    return args


def _time_ms(function: Callable[[], object], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _trace(function: Callable[[], object], *, name: str, iterations: int) -> None:
    torch.cuda.nvtx.range_push(name)
    for _ in range(iterations):
        function()
    # Keep every kernel inside its named range rather than letting asynchronous work spill into
    # the following range. This synchronization is outside all reported event timings.
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


@torch.inference_mode()
def main() -> None:
    """Run the projection comparison and optionally emit an Nsight capture range."""
    args = _parse_args()
    capability = torch.cuda.get_device_capability()
    blackwell_capable = capability[0] >= 10
    block_scaled_enabled = blackwell_capable and not args.regular_fp8_only
    torch.manual_seed(41)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16) / args.k**0.5

    x_inverse_scale = (x.float().abs().amax() / _FP8_MAX).reshape(1)
    weight_inverse_scale = (weight.float().abs().amax() / _FP8_MAX).reshape(1)
    x_fp8 = (x.float() / x_inverse_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    weight_fp8 = (weight.float() / weight_inverse_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)

    def bf16_raw() -> torch.Tensor:
        return functional.linear(x, weight)

    def fp8_raw() -> torch.Tensor:
        return torch._scaled_mm(
            x_fp8,
            weight_fp8.T,
            x_inverse_scale,
            weight_inverse_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    def fp8_quantize() -> tuple[torch.Tensor, torch.Tensor]:
        inverse_scale = (x.float().abs().amax() / _FP8_MAX).reshape(1)
        activation = (x.float() / inverse_scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        return activation, inverse_scale

    def fp8_full() -> torch.Tensor:
        activation, inverse_scale = fp8_quantize()
        return torch._scaled_mm(
            activation,
            weight_fp8.T,
            inverse_scale,
            weight_inverse_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

    functions = {
        "bf16_raw": bf16_raw,
        "fp8_raw": fp8_raw,
        "fp8_quantize": fp8_quantize,
        "fp8_full": fp8_full,
    }
    outputs = {"fp8": fp8_raw().float().flatten()}

    # Hopper exposes native E4M3/E5M2 FP8 Tensor Cores but not SM100 block-scaled
    # MXFP8/NVFP4. Keep this audit useful on Hopper without requiring FlashInfer or
    # attempting Blackwell-only kernels. Regular/global Transformer Engine FP8 is
    # benchmarked end to end by profile_infer.py/profile_predict.py.
    if block_scaled_enabled:
        import flashinfer

        x_mx, x_mx_scales = flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=True, backend="cuda")
        weight_mx, weight_mx_scales = flashinfer.mxfp8_quantize(
            weight,
            is_sf_swizzled_layout=True,
            backend="cuda",
        )
        x_fp4_global_scale = torch.full((1,), _NVFP4_COMBINED_RANGE, device="cuda") / x.float().abs().amax()
        weight_fp4_global_scale = torch.full((1,), _NVFP4_COMBINED_RANGE, device="cuda") / weight.float().abs().amax()
        x_fp4, x_fp4_scales = flashinfer.fp4_quantize(
            x,
            x_fp4_global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=True,
            backend="cuda",
        )
        weight_fp4, weight_fp4_scales = flashinfer.fp4_quantize(
            weight,
            weight_fp4_global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=True,
            backend="cuda",
        )
        fp4_alpha = (x_fp4_global_scale * weight_fp4_global_scale).reciprocal()

        def mxfp8_raw() -> torch.Tensor:
            return flashinfer.mm_mxfp8(
                x_mx,
                weight_mx.T,
                x_mx_scales,
                weight_mx_scales,
                out_dtype=torch.bfloat16,
                backend="cutlass",
            )

        def nvfp4_raw() -> torch.Tensor:
            return flashinfer.mm_fp4(
                x_fp4,
                weight_fp4.T,
                x_fp4_scales,
                weight_fp4_scales.T,
                fp4_alpha,
                out_dtype=torch.bfloat16,
                block_size=16,
                use_8x4_sf_layout=False,
                backend="cutlass",
                use_nvfp4=True,
            )

        def mxfp8_quantize() -> tuple[torch.Tensor, torch.Tensor]:
            return flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=True, backend="cuda")

        def nvfp4_quantize() -> tuple[torch.Tensor, torch.Tensor]:
            return flashinfer.fp4_quantize(
                x,
                x_fp4_global_scale,
                sf_vec_size=16,
                is_sf_swizzled_layout=True,
                backend="cuda",
            )

        def mxfp8_full() -> torch.Tensor:
            activation, scales = mxfp8_quantize()
            return flashinfer.mm_mxfp8(
                activation,
                weight_mx.T,
                scales,
                weight_mx_scales,
                out_dtype=torch.bfloat16,
                backend="cutlass",
            )

        def nvfp4_full() -> torch.Tensor:
            activation, scales = nvfp4_quantize()
            return flashinfer.mm_fp4(
                activation,
                weight_fp4.T,
                scales,
                weight_fp4_scales.T,
                fp4_alpha,
                out_dtype=torch.bfloat16,
                block_size=16,
                use_8x4_sf_layout=False,
                backend="cutlass",
                use_nvfp4=True,
            )

        functions.update(
            {
                "mxfp8_raw": mxfp8_raw,
                "nvfp4_raw": nvfp4_raw,
                "mxfp8_quantize": mxfp8_quantize,
                "nvfp4_quantize": nvfp4_quantize,
                "mxfp8_full": mxfp8_full,
                "nvfp4_full": nvfp4_full,
            }
        )
        outputs.update(
            {
                "mxfp8": mxfp8_raw().float().flatten(),
                "nvfp4": nvfp4_raw().float().flatten(),
            }
        )

    timings = {
        name: _time_ms(function, warmup=args.warmup, iterations=args.iterations)
        for name, function in functions.items()
    }
    reference = bf16_raw().float().flatten()
    timings.update(
        {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "blackwell_block_scaled_capable": blackwell_capable,
            "supported_formats": ["bf16", "fp8", *(("mxfp8", "nvfp4") if block_scaled_enabled else ())],
            "skipped_formats": [] if block_scaled_enabled else ["mxfp8", "nvfp4"],
            **{f"{precision}_raw_speedup": timings["bf16_raw"] / timings[f"{precision}_raw"] for precision in outputs},
            **{
                f"{precision}_full_speedup": timings["bf16_raw"] / timings[f"{precision}_full"]
                for precision in outputs
            },
            **{
                f"{precision}_cosine": float(functional.cosine_similarity(reference, output, dim=0))
                for precision, output in outputs.items()
            },
        }
    )
    print(json.dumps(timings, sort_keys=True), flush=True)

    if args.profile_range:
        torch.cuda.cudart().cudaProfilerStart()
        for name, function in functions.items():
            _trace(function, name=name, iterations=args.iterations)
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
