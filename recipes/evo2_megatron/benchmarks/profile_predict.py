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

"""Benchmark or profile the loaded Evo2 prediction step.

All unrecognized arguments are forwarded to ``bionemo.evo2.run.predict``.  The
first dataloader batch is warmed up and repeated in-place, so timings exclude
checkpoint loading and dataloader startup.  ``--benchmark-profile-range`` wraps
only the measured repetitions in the CUDA profiler API for a small Nsight trace.
"""

import argparse
import json
import sys

import torch

import bionemo.evo2.run.predict as predict_module
from bionemo.evo2.run.low_precision import inference_parameter_storage, inference_precision_kind


def _parse_benchmark_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-iterations", type=int, default=5)
    parser.add_argument("--benchmark-profile-range", action="store_true")
    parser.add_argument("--benchmark-all-batches", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.benchmark_warmup < 0 or args.benchmark_iterations <= 0:
        raise ValueError("benchmark warmup must be non-negative and iterations must be positive")
    sys.argv = [sys.argv[0], *remaining]
    return args


def _aggregate_layout(measurements: list[dict[str, float | int | str]]) -> str:
    """Describe the measured endpoint layout without mislabeling rectangular runs."""
    layouts = {str(item["layout"]) for item in measurements}
    if layouts == {"packed"}:
        return "length-aware-packed"
    if len(layouts) == 1:
        return layouts.pop()
    return "mixed"


def main() -> None:
    """Run prediction while timing repeated copies of the first model step."""
    args = _parse_benchmark_args()
    original_predict_step = predict_module._predict_step
    measured = False
    measurements: list[dict[str, float | int | str]] = []

    def benchmarked_predict_step(*step_args, **step_kwargs):
        nonlocal measured
        if measured and not args.benchmark_all_batches:
            return original_predict_step(*step_args, **step_kwargs)
        measured = True

        batch = step_kwargs.get("batch")
        if batch is None:
            batch = step_args[1]
        model = step_kwargs.get("model")
        if model is None:
            model = step_args[0]
        model_config = getattr(getattr(model, "module", model), "config", None)
        packed = "cu_seqlens" in batch
        physical_tokens = int(batch["tokens"].numel())
        real_tokens = int(batch["loss_mask"].sum().item())

        result = None
        for _ in range(args.benchmark_warmup):
            if result is not None:
                del result
            result = original_predict_step(*step_args, **step_kwargs)
        del result
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        if args.benchmark_profile_range:
            torch.cuda.cudart().cudaProfilerStart()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = None
        for _ in range(args.benchmark_iterations):
            if result is not None:
                del result
            result = original_predict_step(*step_args, **step_kwargs)
        end.record()
        torch.cuda.synchronize()
        if args.benchmark_profile_range:
            torch.cuda.cudart().cudaProfilerStop()

        elapsed_ms = start.elapsed_time(end) / args.benchmark_iterations
        measurement = {
            "elapsed_ms": elapsed_ms,
            "iterations": args.benchmark_iterations,
            "layout": "packed" if packed else "rectangular",
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "physical_tokens": physical_tokens,
            "physical_tokens_per_second": physical_tokens / (elapsed_ms / 1000),
            "precision_kind": inference_precision_kind(model_config),
            "precision_parameter_storage": inference_parameter_storage(model_config),
            "real_tokens": real_tokens,
            "real_tokens_per_second": real_tokens / (elapsed_ms / 1000),
        }
        measurements.append(measurement)
        if (not args.benchmark_all_batches) and (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            print(json.dumps(measurement, sort_keys=True), flush=True)
        return result

    predict_module._predict_step = benchmarked_predict_step
    predict_module.main()
    if (
        args.benchmark_all_batches
        and measurements
        and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
    ):
        elapsed_ms = sum(float(item["elapsed_ms"]) for item in measurements)
        physical_tokens = sum(int(item["physical_tokens"]) for item in measurements)
        real_tokens = sum(int(item["real_tokens"]) for item in measurements)
        print(
            json.dumps(
                {
                    "elapsed_ms": elapsed_ms,
                    "iterations": args.benchmark_iterations,
                    "layout": _aggregate_layout(measurements),
                    "measured_batches": len(measurements),
                    "peak_memory_bytes": max(int(item["peak_memory_bytes"]) for item in measurements),
                    "physical_tokens": physical_tokens,
                    "physical_tokens_per_second": physical_tokens / (elapsed_ms / 1000),
                    "precision_kind": measurements[0]["precision_kind"],
                    "precision_parameter_storage": measurements[0]["precision_parameter_storage"],
                    "real_tokens": real_tokens,
                    "real_tokens_per_second": real_tokens / (elapsed_ms / 1000),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
