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
Produces mixtral_8x7b_pflops.png next to this script.
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
DEFAULT_OUT = HERE / "mixtral_8x7b_pflops.png"
DEFAULT_TITLE = "Mixtral-8x7B training throughput — 8×B200"
DEFAULT_SUBTITLE = (
    "Pretrained weights, THD packing, wikitext, token_mb=4096, max_seq=4096. "
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

# Single-group three-bar comparison (HF baseline vs native-TE BF16 vs native-TE MXFP8), 8×B200.
# Values are tokens/s/GPU; the left PFLOP/s/GPU axis is an exact linear rescale.
HF_COMPARISON = [
    ("HF Baseline\n(BF16)", 3933.13, "#BDBDBD"),
    ("Native TE\n(BF16)", 4213.0, "#636363"),
    ("Native TE\n(MXFP8)", 6183.0, "#76B900"),
]
HF_COMPARISON_OUT = HERE / "mixtral_8x7b_B200_hf_comparison.png"
HF_COMPARISON_TITLE = "Mixtral-8x7B training throughput — 8×B200"
HF_COMPARISON_SUBTITLE = (
    "Pretrained weights, THD packing, wikitext, token_mb=4096, max_seq=4096, FSDP2-only (dp8, ep1). "
    "MFU vs dense B200 peaks (fp8 4.5, bf16 2.25 PFLOP/s)."
)

# Single-group three-bar comparison, 8×B300 (dp4, ep2). tokens/s/GPU, steady state.
# HF baseline: real pretrained checkpoint + real text, grouped_mm experts + torch.compile (sdpa).
HF_COMPARISON_B300 = [
    ("HF Baseline\n(BF16)", 964.0, "#BDBDBD"),
    ("Native TE\n(BF16)", 4770.0, "#636363"),
    ("Native TE\n(MXFP8)", 10577.0, "#76B900"),
]
HF_COMPARISON_B300_OUT = HERE / "mixtral_8x7b_B300_hf_comparison.png"
HF_COMPARISON_B300_TITLE = "Mixtral-8x7B training throughput — 8×B300"
HF_COMPARISON_B300_SUBTITLE = (
    "Random init, THD packing, wikitext, token_mb=16384, max_seq=8192, EP+FSDP2 (dp4, ep2). "
    "MFU vs dense B300 peaks (fp8 5.0, bf16 2.5 PFLOP/s)."
)


def plot_hf_comparison(comparison, out: Path, title: str, subtitle: str, fig_width: float, fig_height: float):
    """Render a single-group three-bar chart comparing HF baseline to native-TE BF16 and MXFP8."""
    labels = [label for label, _, _ in comparison]
    tokens = [tok for _, tok, _ in comparison]
    colors = [color for _, _, color in comparison]
    pflops = [tok / TOKENS_PER_PFLOP for tok in tokens]
    x = range(len(HF_COMPARISON))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    bars = ax.bar(list(x), pflops, 0.6, color=colors, edgecolor="white")
    ax.bar_label(
        bars,
        labels=[f"{p:.3f}\n{t:,.0f} tok/s" for p, t in zip(pflops, tokens)],
        padding=3,
        fontsize=10,
        color="#333333",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("PFLOP/s/GPU  (6 · N_active)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, max(pflops) * 1.22)

    secax = ax.secondary_yaxis("right", functions=(lambda p: p * TOKENS_PER_PFLOP, lambda t: t / TOKENS_PER_PFLOP))
    secax.set_ylabel("tokens/s/GPU", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.5, -0.02, subtitle, ha="center", fontsize=8, color="#666666")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


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
    parser.add_argument(
        "--hf-comparison",
        action="store_true",
        help="Render the single-group HF-baseline vs BF16 vs MXFP8 chart for 8×B200.",
    )
    parser.add_argument(
        "--hf-comparison-b300",
        action="store_true",
        help="Render the single-group HF-baseline vs BF16 vs MXFP8 chart for 8×B300 (dp4, ep2).",
    )
    args = parser.parse_args()

    if args.hf_comparison:
        out = args.out if args.out != DEFAULT_OUT else HF_COMPARISON_OUT
        title = args.title if args.title != DEFAULT_TITLE else HF_COMPARISON_TITLE
        subtitle = args.subtitle if args.subtitle != DEFAULT_SUBTITLE else HF_COMPARISON_SUBTITLE
        plot_hf_comparison(HF_COMPARISON, out, title, subtitle, args.fig_width, args.fig_height)
        return

    if args.hf_comparison_b300:
        out = args.out if args.out != DEFAULT_OUT else HF_COMPARISON_B300_OUT
        title = args.title if args.title != DEFAULT_TITLE else HF_COMPARISON_B300_TITLE
        subtitle = args.subtitle if args.subtitle != DEFAULT_SUBTITLE else HF_COMPARISON_B300_SUBTITLE
        plot_hf_comparison(HF_COMPARISON_B300, out, title, subtitle, args.fig_width, args.fig_height)
        return

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
