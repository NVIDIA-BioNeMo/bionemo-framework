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

"""Benchmark or profile steady-state Evo2 native packed prefill and batched decode.

All unrecognized arguments are forwarded to ``bionemo.evo2.run.infer``.  The
wrapper repeats the first ``generate`` call against the already-loaded model and
dynamic context, so checkpoint loading, Triton JIT, and CUDA-graph capture can be
warmed before measurement.  ``--benchmark-profile-range`` wraps only measured
generations in the CUDA profiler API for a compact, environment-sanitized Nsight
trace.
"""

import argparse
import json
import sys
from typing import Any

import torch

import bionemo.evo2.run.infer as infer_module


def _parse_benchmark_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark-warmup", type=int, default=1)
    parser.add_argument("--benchmark-iterations", type=int, default=3)
    parser.add_argument("--benchmark-profile-range", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.benchmark_warmup < 0 or args.benchmark_iterations <= 0:
        raise ValueError("benchmark warmup must be non-negative and iterations must be positive")
    sys.argv = [sys.argv[0], *remaining]
    return args


def _phase_mean_ms(records: list[dict[str, Any]], phase: str) -> float | None:
    values = [
        1000.0 * float(record[f"{phase}_elapsed_s"]) for record in records if record.get(f"{phase}_performed", False)
    ]
    return sum(values) / len(values) if values else None


def main() -> None:
    """Run native generation with warmup and measured repeated calls."""
    args = _parse_benchmark_args()
    original_generate = infer_module.generate
    measured = False

    def benchmarked_generate(*generate_args, **generate_kwargs):
        nonlocal measured
        if measured:
            return original_generate(*generate_args, **generate_kwargs)
        measured = True

        result = None
        for _ in range(args.benchmark_warmup):
            result = original_generate(*generate_args, **generate_kwargs)
        del result
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        if args.benchmark_profile_range:
            torch.cuda.cudart().cudaProfilerStart()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        phase_records: list[dict[str, Any]] = []
        for _ in range(args.benchmark_iterations):
            result = original_generate(*generate_args, **generate_kwargs)
            if result:
                phase_records.append(dict(getattr(result[0], "timings", None) or {}))
        end.record()
        torch.cuda.synchronize()
        if args.benchmark_profile_range:
            torch.cuda.cudart().cudaProfilerStop()

        assert result is not None
        elapsed_ms = start.elapsed_time(end) / args.benchmark_iterations
        prompt_tokens = sum(len(item.prompt_tokens or []) for item in result)
        completion_tokens = sum(int(item.generated_length or 0) for item in result)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            graph_record = phase_records[-1] if phase_records else {}
            print(
                json.dumps(
                    {
                        "completion_tokens": completion_tokens,
                        "completion_tokens_per_second": completion_tokens / (elapsed_ms / 1000.0),
                        "cuda_graph_manager_count": graph_record.get("cuda_graph_manager_count", 0),
                        "cuda_graph_recorded_count": graph_record.get("cuda_graph_recorded_count", 0),
                        "cuda_graph_replay_verified": graph_record.get("cuda_graph_replay_verified", False),
                        "cuda_graph_runner_count": graph_record.get("cuda_graph_runner_count", 0),
                        "cuda_graph_scope": graph_record.get("cuda_graph_scope", "none"),
                        "decode_elapsed_ms": _phase_mean_ms(phase_records, "decode"),
                        "elapsed_ms": elapsed_ms,
                        "iterations": args.benchmark_iterations,
                        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                        "precision_kind": graph_record.get("precision_kind", "unknown"),
                        "precision_parameter_storage": graph_record.get("precision_parameter_storage", "unknown"),
                        "prefill_elapsed_ms": _phase_mean_ms(phase_records, "prefill"),
                        "prompt_tokens": prompt_tokens,
                        "prompt_tokens_per_second": prompt_tokens
                        / ((_phase_mean_ms(phase_records, "prefill") or elapsed_ms) / 1000.0),
                        "requests": len(result),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return result

    infer_module.generate = benchmarked_generate
    infer_module.main()


if __name__ == "__main__":
    main()
