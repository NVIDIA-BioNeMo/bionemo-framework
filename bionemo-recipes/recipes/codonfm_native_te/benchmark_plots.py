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

"""Benchmark plots for distributed training step times.

Fits Amdahl's-law model t(N) = a + b/N to each configuration,
extrapolates to 32 and 64 nodes, and plots:
  1. Step time vs nodes (measured + extrapolated), per config
  2. Bar chart of total training time (1M steps) per (config, node count)

Run with the project's venv:
    /Users/balvisio/.venv/bin/python benchmark_plots.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


TOTAL_STEPS = 1_000_000
SECONDS_PER_DAY = 86_400

# Measured step times (seconds). Keys = config label.
DATA: dict[str, dict[int, float]] = {
    "1B - BF16 - BSHD": {1: 2.05, 2: 1.06, 4: 0.57, 8: 0.34, 16: 0.21},
    "1B - MXFP8 - THD": {1: 0.71, 2: 0.34, 4: 0.23, 8: 0.23, 16: 0.19},
    "5B - MXFP8 - THD": {1: 1.42, 2: 0.77, 4: 0.46, 8: 0.31, 16: 0.27},
    "10B - THD": {1: 4.18, 2: 2.22, 4: 1.34, 8: 0.75, 16: 0.53},
    "10B - MXFP8 - THD": {1: 2.88, 2: 1.59, 4: 0.90, 8: 0.60, 16: 0.44},
    "10B - BSHD": {1: 13.19, 2: 6.56, 4: 3.29, 8: 1.67, 16: 0.90, 32: 0.57},
}

EXTRAPOLATE_NODES = [32, 64]
ALL_NODES = [1, 2, 4, 8, 16, 32, 64]


def amdahl(n, a, b):  # noqa: D103
    return a + b / n


def power_law(n, a, b):  # noqa: D103
    return a * np.power(n, -b)


def fit_models(nodes: np.ndarray, times: np.ndarray):  # noqa: D103
    (a_amd, b_amd), _ = curve_fit(amdahl, nodes, times, p0=[0.1, times[0]])
    (a_pow, b_pow), _ = curve_fit(power_law, nodes, times, p0=[times[0], 0.8])
    return (a_amd, b_amd), (a_pow, b_pow)


def days(step_time_s: float) -> float:  # noqa: D103
    return step_time_s * TOTAL_STEPS / SECONDS_PER_DAY


def main() -> None:  # noqa: D103
    fits = {}
    print(f"{'Config':<22} {'Amdahl a (floor)':>18} {'Amdahl b':>12}   {'Power a':>10} {'Power b':>10}")
    print("-" * 78)
    for label, points in DATA.items():
        nodes = np.array(sorted(points.keys()), dtype=float)
        times = np.array([points[int(n)] for n in nodes], dtype=float)
        (a_amd, b_amd), (a_pow, b_pow) = fit_models(nodes, times)
        fits[label] = {
            "amdahl": (a_amd, b_amd),
            "power": (a_pow, b_pow),
            "nodes": nodes,
            "times": times,
        }
        print(f"{label:<22} {a_amd:>18.4f} {b_amd:>12.4f}   {a_pow:>10.4f} {b_pow:>10.4f}")

    print()
    print(f"Extrapolated step times (Amdahl fit) and total days for {TOTAL_STEPS:,} steps:")
    header = f"{'Config':<22} " + " ".join(f"{'N=' + str(n):>11}" for n in ALL_NODES)
    print(header)
    print("-" * len(header))
    extrap_table = {}
    for label, f in fits.items():
        a, b = f["amdahl"]
        row = []
        extrap_table[label] = {}
        for n in ALL_NODES:
            if n in DATA[label]:
                t = DATA[label][n]
                tag = ""
            else:
                t = amdahl(n, a, b)
                tag = "*"
            extrap_table[label][n] = t
            row.append(f"{t:>7.3f}s{tag:<3}")
        print(f"{label:<22} " + " ".join(row))
    print("  * = extrapolated")

    print()
    print(f"Total training time (days) for {TOTAL_STEPS:,} steps:")
    print(header)
    print("-" * len(header))
    for label, by_n in extrap_table.items():
        row = " ".join(f"{days(by_n[n]):>10.2f}d" for n in ALL_NODES)
        print(f"{label:<22} " + row)

    # ---------- Plot 1: step time vs nodes (one panel per config) ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    smooth_n = np.linspace(1, 64, 200)

    for ax, (label, f) in zip(axes, fits.items()):
        a, b = f["amdahl"]
        ax.plot(smooth_n, amdahl(smooth_n, a, b), "-", color="C0", label=f"Amdahl: {a:.3f} + {b:.3f}/N")
        ax.plot(f["nodes"], f["times"], "o", color="C0", markersize=8, label="Measured")
        for n_meas, t_meas in zip(f["nodes"], f["times"]):
            ax.annotate(f"{t_meas:.3f}s", (n_meas, t_meas), xytext=(6, 6), textcoords="offset points", fontsize=9)
        extrap_nodes = [n for n in EXTRAPOLATE_NODES if n not in DATA[label]]
        if extrap_nodes:
            extrap_x = np.array(extrap_nodes, dtype=float)
            ax.plot(extrap_x, amdahl(extrap_x, a, b), "s", color="C3", markersize=9, label="Extrapolated (Amdahl)")
            for n in extrap_nodes:
                t = amdahl(n, a, b)
                ax.annotate(f"{t:.3f}s", (n, t), xytext=(6, 6), textcoords="offset points", fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ALL_NODES)
        ax.set_xticklabels(ALL_NODES)
        ax.set_xlabel("# nodes")
        ax.set_ylabel("Step time (s)")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    # Hide unused 6th subplot
    for ax in axes[len(fits) :]:
        ax.axis("off")
    fig.suptitle(
        "Step time vs # nodes — measured + Amdahl extrapolation to 32/64\nHardware: NVIDIA B300 GPUs", fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("step_time_vs_nodes.png", dpi=140)
    print("\nWrote step_time_vs_nodes.png")

    # ---------- Plot 2: bar chart of total days ----------
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    configs = list(extrap_table.keys())
    n_configs = len(configs)
    bar_width = 0.85 / n_configs
    x = np.arange(len(ALL_NODES))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, n_configs))

    for i, label in enumerate(configs):
        days_per_n = [days(extrap_table[label][n]) for n in ALL_NODES]
        offset = (i - (n_configs - 1) / 2) * bar_width
        bars = ax2.bar(x + offset, days_per_n, bar_width, label=label, color=colors[i])
        for bar, d, n in zip(bars, days_per_n, ALL_NODES):
            tag = "*" if n not in DATA[label] else ""
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{d:.1f}{tag}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n} nodes" for n in ALL_NODES])
    ax2.set_ylabel(f"Total training time (days) — {TOTAL_STEPS:,} steps")
    ax2.set_title(
        "Total training time per configuration — Hardware: NVIDIA B300 GPUs\n(* = extrapolated step time, Amdahl fit)"
    )
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("total_days_per_config.png", dpi=140)
    print("Wrote total_days_per_config.png")


if __name__ == "__main__":
    main()
