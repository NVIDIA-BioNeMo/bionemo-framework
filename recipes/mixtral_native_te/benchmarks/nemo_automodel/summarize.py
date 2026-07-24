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

"""Convert NeMo AutoModel benchmark JSON files to a native-recipe-style CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ACTIVE_PARAMETERS = 12_748_587_008
TOKENS_PER_GLOBAL_STEP = 8 * 4096
WORLD_SIZE = 8
BF16_PEAK_PFLOPS = 2.25
MXFP8_PEAK_PFLOPS = 4.5


def summarize(paths: list[Path], output) -> None:
    """Write one native-style throughput row for each AutoModel JSON result."""
    writer = csv.writer(output)
    writer.writerow(
        [
            "layout",
            "dp",
            "tp",
            "precision",
            "tokens_per_s_per_gpu",
            "pflops_per_gpu",
            "mfu_pct",
            "step_time_s",
            "mem_gb",
            "last_loss",
            "n_steady",
            "elapsed_s",
            "rc",
        ]
    )
    for path in paths:
        result = json.loads(path.read_text())
        layout = path.stem
        topology_precision = layout.removeprefix("mixtral_8x7b_")
        dp_part, tp_part, precision = topology_precision.split("_")
        topology = f"{dp_part}_{tp_part}"
        dp = int(dp_part.removeprefix("dp"))
        tp = int(tp_part.removeprefix("tp"))
        step_time = float(result["avg_iter_time_seconds"])
        tokens_per_second = TOKENS_PER_GLOBAL_STEP / WORLD_SIZE / step_time
        pflops = 6 * ACTIVE_PARAMETERS * tokens_per_second / 1e15
        log_path = path.parent / f"{topology}_{precision}.log"
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        memories = re.findall(r"Max Memory Allocated: ([0-9.]+) GB", log_text)
        losses = re.findall(r"\| loss=([0-9.eE+-]+)", log_text)
        elapsed = sum(
            float(result[key]) for key in ("setup_time_seconds", "warmup_time_seconds", "training_time_seconds")
        )
        writer.writerow(
            [
                topology,
                dp,
                tp,
                precision,
                f"{tokens_per_second:.0f}",
                f"{pflops:.4f}",
                f"{100 * pflops / (MXFP8_PEAK_PFLOPS if precision == 'mxfp8' else BF16_PEAK_PFLOPS):.2f}",
                f"{step_time:.3f}",
                f"{max(map(float, memories)):.2f}" if memories else "",
                f"{float(losses[-1]):.4f}" if losses else "",
                int(result["training_steps"]),
                f"{elapsed:.0f}",
                0,
            ]
        )


def main() -> None:
    """Parse benchmark JSON paths and print their CSV summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+", type=Path)
    args = parser.parse_args()
    summarize(args.json, output=__import__("sys").stdout)


if __name__ == "__main__":
    main()
