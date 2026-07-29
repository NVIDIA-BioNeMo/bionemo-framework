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

"""Grouped bar chart of Mixtral-8x7B training throughput (PFLOP/s/GPU) from mixtral_8x7b_8xB200.csv.

Usage: python plot_perf.py
Produces mixtral_8x7b_B200_pflops.png next to this script.
"""

# E402: pyplot import must follow matplotlib.use/rcParams. I001: import split is intentional.
# RUF001: multiplication-sign and middle-dot glyphs are intentional in chart labels.
# ruff: noqa: E402, I001, RUF001

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Prefer NVIDIA Sans; fall back cleanly to a bundled sans-serif if it isn't installed.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["NVIDIA Sans", "DejaVu Sans", "Arial", "Helvetica"]

import matplotlib.pyplot as plt

HERE = Path(__file__).parent

DEFAULT_CSV = HERE / "mixtral_8x7b_8xB200.csv"
DEFAULT_OUT = HERE / "mixtral_8x7b_B200_pflops.png"
DEFAULT_TITLE = "Mixtral-8x7B training throughput — 8×B200"
DEFAULT_SUBTITLE = (
    "Pretrained weights, local DCLM parquet, THD packing, token_mb=4096, max_seq=4096. "
    "MFU vs dense B200 peaks (fp8 4.5, bf16 2.25 PFLOP/s)."
)

# PFLOP/s/GPU = 6 · N_active · tokens/s/GPU / 1e15, so tokens/s/GPU is an exact linear rescale of the
# left axis (same bars). N_active = attention + top-2 experts + lm_head for Mixtral-8x7B.
N_ACTIVE = 12_748_587_008
TOKENS_PER_PFLOP = 1e15 / (6 * N_ACTIVE)  # tokens/s/GPU per PFLOP/s/GPU

COLORS = {"fp8": "#76B900", "bf16": "#636363"}
LABELS = {"fp8": "MXFP8", "bf16": "BF16"}
# (dp, ep) -> group label. Order left-to-right.
GROUPS = [
    ((1, 8), "EP-only\n(dp1, ep8)"),
    ((2, 4), "EP+FSDP2\n(dp2, ep4)"),
    ((4, 2), "EP+FSDP2\n(dp4, ep2)"),
    ((8, 1), "FSDP2-only\n(dp8, ep1)"),
]


def load(csv_path: Path):
    """Load PFLOP/s/GPU per (dp, ep, precision) from the results CSV."""
    data = {}  # (dp, ep, precision) -> pflops
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data[(int(row["dp"]), int(row["ep"]), row["precision"])] = float(row["pflops_per_gpu"])
    return data


def main():
    """Render the grouped bar chart of PFLOP/s/GPU per (dp, ep) layout and precision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--subtitle", default=DEFAULT_SUBTITLE)
    parser.add_argument("--fig-width", type=float, default=9.0)
    parser.add_argument("--fig-height", type=float, default=5.4)
    args = parser.parse_args()

    data = load(args.csv)
    x = range(len(GROUPS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), dpi=200)
    # bf16 on the left, MXFP8 on the right within each group.
    for i, prec in enumerate(("bf16", "fp8")):
        offset = (i - 0.5) * width
        vals = [data[(dp, ep, prec)] for (dp, ep), _ in GROUPS]
        bars = ax.bar(
            [xi + offset for xi in x], vals, width, label=LABELS[prec], color=COLORS[prec], edgecolor="white"
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, color="#333333")

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in GROUPS], fontsize=10)
    ax.set_ylabel("PFLOP/s/GPU  (6 · N_active)", fontsize=11)
    ax.set_title(args.title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, max(data.values()) * 1.18)

    # Right axis: tokens/s/GPU (exact linear rescale of the PFLOP/s/GPU left axis).
    secax = ax.secondary_yaxis("right", functions=(lambda p: p * TOKENS_PER_PFLOP, lambda t: t / TOKENS_PER_PFLOP))
    secax.set_ylabel("tokens/s/GPU", fontsize=11)
    ax.legend(title="precision", frameon=False, fontsize=10, title_fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.5, -0.02, args.subtitle, ha="center", fontsize=8, color="#666666")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
