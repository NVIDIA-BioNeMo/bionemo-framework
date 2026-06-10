#!/usr/bin/env python3

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

"""Extract BF16 baseline metrics from a WandB run into a JSON file for the FP8 precision agent."""

import json
import math

import wandb


ENTITY = "clara-discovery"
PROJECT = "llama3-metagenome-7b"
RUN_ID = "8mfsb27t"

OUTPUT_FILE = "baseline_bf16.json"


def main():
    """Extract BF16 baseline metrics from a WandB run and save to JSON."""
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")

    baseline = {}
    for row in run.scan_history(keys=["train/loss", "train/unpadded_tokens_per_second_per_gpu", "train/global_step"]):
        step = row.get("train/global_step")
        loss = row.get("train/loss")
        tps = row.get("train/unpadded_tokens_per_second_per_gpu")

        if step is None or loss is None:
            continue

        step = int(step)
        baseline[f"step_{step}"] = {
            "perplexity": math.exp(loss),
            "loss": loss,
            "unpadded_tokens_per_sec": tps,
        }

    # Sort by step number
    baseline = dict(sorted(baseline.items(), key=lambda x: int(x[0].split("_")[1])))

    with open(OUTPUT_FILE, "w") as f:
        json.dump(baseline, f, indent=2)

    steps = sorted(int(k.split("_")[1]) for k in baseline)
    print(f"Extracted {len(baseline)} steps: {steps[0]} .. {steps[-1]}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
