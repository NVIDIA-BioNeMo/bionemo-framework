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

"""Model-agnostic FP8 log analyzer and heatmap generator.

Automatically detects model architecture from metadata logs and adapts visualization.

Usage:
    python3 analyze_and_create_heatmap.py <log_directory> [output_name_suffix]

Example:
    python3 analyze_and_create_heatmap.py fp8logswithhead
    python3 analyze_and_create_heatmap.py log_bf16head _bf16head
"""

import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Styling
sns.set_style("white")


def parse_layer_metadata(log_dir):
    """Parse model architecture from metadata logs."""
    metadata_file = log_dir / "rank_0" / "nvdlfw_inspect_logs" / "nvdlfw_inspect_globalrank-0.log"

    if not metadata_file.exists():
        logger.warning(f"No metadata file found at {metadata_file}")
        return None

    logger.info("=" * 80)
    logger.info("PARSING MODEL ARCHITECTURE")
    logger.info("=" * 80)
    logger.info(f"Metadata: {metadata_file}")

    layers = []
    pattern = r"Assigned layer name: (.+)$"

    with open(metadata_file, "r") as f:
        for line in f:
            match = re.search(pattern, line.strip())
            if match:
                layers.append(match.group(1))

    logger.info(f"Found {len(layers)} layer names")

    # Analyze structure
    structure = {"encoder_layers": [], "head_layers": [], "embedding_layers": [], "other_layers": []}

    for layer in layers:
        if re.match(r".*\.encoder\.layers\.(\d+)\.", layer):
            match = re.search(r"\.encoder\.layers\.(\d+)\.", layer)
            if match:
                layer_num = int(match.group(1))
                if layer_num not in [enc_layer["num"] for enc_layer in structure["encoder_layers"]]:
                    structure["encoder_layers"].append({"num": layer_num, "name": layer})
        elif "lm_head" in layer or "head" in layer:
            structure["head_layers"].append(layer)
        elif "embedding" in layer:
            structure["embedding_layers"].append(layer)
        else:
            structure["other_layers"].append(layer)

    # Sort encoder layers
    structure["encoder_layers"] = sorted(structure["encoder_layers"], key=lambda x: x["num"])

    logger.info("")
    logger.info("Model Structure:")
    logger.info(f"  Encoder layers: {len(structure['encoder_layers'])}")
    if structure["encoder_layers"]:
        logger.info(
            f"    Range: Layer {structure['encoder_layers'][0]['num']} to {structure['encoder_layers'][-1]['num']}"
        )
    logger.info(f"  Head layers: {len(structure['head_layers'])}")
    for head in structure["head_layers"]:
        logger.info(f"    - {head}")
    logger.info(f"  Embedding layers: {len(structure['embedding_layers'])}")
    logger.info(f"  Other layers: {len(structure['other_layers'])}")

    return structure


def parse_log_file(log_file_path):
    """Parse FP8 statistics log file."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("PARSING LOG FILE")
    logger.info("=" * 80)
    logger.info(f"File: {log_file_path}")

    pattern = (
        r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+-\s+INFO\s+-\s+(.+?)\s+iteration=(\d+)\s+value=([\d.]+)$"
    )

    data = []
    line_count = 0

    with open(log_file_path, "r") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            match = re.match(pattern, line)

            if match:
                metric_name = match.group(1).strip()
                iteration = int(match.group(2))
                value = float(match.group(3))

                data.append({"metric_name": metric_name, "iteration": iteration, "value": value})

            if line_num % 500000 == 0:
                logger.info(f"Processed {line_num:,} lines...")

            line_count = line_num

    logger.info(f"Total lines: {line_count:,}")
    logger.info(f"Metrics extracted: {len(data):,}")

    df = pd.DataFrame(data)

    if len(df) > 0:
        logger.info(f"Iteration range: {df['iteration'].min()} to {df['iteration'].max()}")
        logger.info(f"Unique metrics: {df['metric_name'].nunique()}")

    return df


def auto_detect_components(df, structure=None):
    """Automatically detect which components have gradient underflow data."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("AUTO-DETECTING COMPONENTS")
    logger.info("=" * 80)

    components = []

    # Get all unique metric names
    all_metrics = df["metric_name"].unique()

    # Find gradient underflow metrics
    underflow_metrics = [m for m in all_metrics if "gradient_underflows%" in m]

    logger.info(f"Found {len(underflow_metrics)} gradient underflow metrics")

    # Categorize by component type
    component_patterns = [
        # Encoder layers - various architectures
        (r"\.encoder\.layers\.(\d+)\.", "encoder", lambda m: int(m.group(1))),
        (r"\.layers\.(\d+)\.", "encoder", lambda m: int(m.group(1))),
        (r"\.transformer\.layers\.(\d+)\.", "encoder", lambda m: int(m.group(1))),
        # Head/output layers - various names
        (r"\.lm_head\.dense", "head", lambda m: "Dense"),
        (r"\.lm_head\.decoder", "head", lambda m: "Decoder"),
        (r"\.head\.dense", "head", lambda m: "Dense"),
        (r"\.head\.decoder", "head", lambda m: "Decoder"),
        (r"\.output_layer", "head", lambda m: "Output"),
        # Embedding layers
        (r"\.embeddings?\.", "embedding", lambda m: "Embedding"),
    ]

    detected = defaultdict(list)

    for metric in underflow_metrics:
        matched = False
        for pattern, comp_type, extractor in component_patterns:
            match = re.search(pattern, metric)
            if match:
                identifier = extractor(match)
                detected[comp_type].append({"metric": metric, "identifier": identifier})
                matched = True
                break

        if not matched:
            # Unknown component
            detected["unknown"].append({"metric": metric, "identifier": "Unknown"})

    # Build component list with proper ordering
    position = 1

    # Sort encoder layers numerically
    if "encoder" in detected:
        encoder_layers = sorted({d["identifier"] for d in detected["encoder"]})
        for layer_num in encoder_layers:
            # Get all metrics for this layer
            layer_metrics = [d["metric"] for d in detected["encoder"] if d["identifier"] == layer_num]

            # Prioritize layernorm_qkv over proj (layernorm_qkv typically has more meaningful data)
            metric = None
            for m in layer_metrics:
                if "layernorm_qkv" in m:
                    metric = m
                    break
            if metric is None:
                metric = layer_metrics[0]  # Fallback to first metric

            components.append(
                {
                    "position": position,
                    "type": "encoder",
                    "identifier": layer_num,
                    "label": f"L{layer_num}",
                    "display_label": f"{layer_num}",
                    "metric": metric,
                    "group": "Encoder",
                }
            )
            position += 1

    # Add head layers
    if "head" in detected:
        for d in detected["head"]:
            components.append(
                {
                    "position": position,
                    "type": "head",
                    "identifier": d["identifier"],
                    "label": d["identifier"],
                    "display_label": d["identifier"],
                    "metric": d["metric"],
                    "group": "Head",
                }
            )
            position += 1

    # Add embedding layers
    if "embedding" in detected:
        for d in detected["embedding"]:
            components.append(
                {
                    "position": position,
                    "type": "embedding",
                    "identifier": d["identifier"],
                    "label": d["identifier"],
                    "display_label": d["identifier"],
                    "metric": d["metric"],
                    "group": "Embedding",
                }
            )
            position += 1

    # Add unknown layers
    if "unknown" in detected:
        for i, d in enumerate(detected["unknown"]):
            components.append(
                {
                    "position": position,
                    "type": "unknown",
                    "identifier": f"Unknown_{i}",
                    "label": f"Unknown_{i}",
                    "display_label": f"Unknown_{i}",
                    "metric": d["metric"],
                    "group": "Unknown",
                }
            )
            position += 1

    logger.info("")
    logger.info("Component Summary:")
    for group in ["Encoder", "Head", "Embedding", "Unknown"]:
        group_comps = [c for c in components if c["group"] == group]
        if group_comps:
            logger.info(f"  {group}: {len(group_comps)} components")
            if len(group_comps) <= 5:
                for c in group_comps:
                    logger.info(f"    - {c['label']}")

    return components


def create_heatmap(df, components, output_dir, suffix=""):
    """Create publication-quality heatmap from parsed data."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("CREATING HEATMAP")
    logger.info("=" * 80)

    # Extract data for all components
    all_data = []

    for comp in components:
        data = df[df["metric_name"] == comp["metric"]]
        if len(data) > 0:
            for _, row in data.iterrows():
                all_data.append(
                    {
                        "position": comp["position"],
                        "label": comp["display_label"],
                        "group": comp["group"],
                        "iteration": row["iteration"],
                        "value": row["value"],
                    }
                )

    if len(all_data) == 0:
        logger.error("No gradient underflow data found!")
        return None

    all_df = pd.DataFrame(all_data)
    logger.info(f"Components: {all_df['position'].nunique()}")
    logger.info(f"Data points: {len(all_df):,}")

    # Create pivot table
    pivot_data = all_df.pivot_table(
        values="value", index=["position", "label", "group"], columns="iteration", aggfunc="mean"
    )

    # Sample iterations for visualization
    sample_iterations = pivot_data.columns[:: max(1, len(pivot_data.columns) // 120)]
    pivot_sample = pivot_data[sample_iterations]

    logger.info(f"Heatmap dimensions: {len(pivot_sample.index)} components x {len(sample_iterations)} time points")

    # Create figure
    _fig = plt.figure(figsize=(22, max(12, len(pivot_sample.index) * 0.4)))
    ax = plt.subplot2grid((20, 20), (0, 1), colspan=18, rowspan=18)
    cax = plt.subplot2grid((20, 20), (0, 19), rowspan=18)

    cmap = sns.color_palette("rocket_r", as_cmap=True)
    max_val = min(6, pivot_sample.values.max())  # Cap at 6% for color scale

    im = ax.imshow(pivot_sample.values, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=max_val)

    # Colorbar
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Gradient Underflows %", fontsize=14, fontweight="bold", rotation=270, labelpad=25)
    cbar.ax.tick_params(labelsize=11)

    # Y-axis (components)
    y_positions = np.arange(len(pivot_sample.index))
    y_labels = [label for _, label, _ in pivot_sample.index]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=min(10, max(6, 200 // len(y_labels))))
    ax.set_ylabel("Component", fontsize=14, fontweight="bold")

    # X-axis (iterations)
    x_tick_positions = np.linspace(0, len(sample_iterations) - 1, min(12, len(sample_iterations))).astype(int)
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([f"{int(sample_iterations[i])}" for i in x_tick_positions], fontsize=11, rotation=0)
    ax.set_xlabel("Training Iteration", fontsize=14, fontweight="bold")

    # Title
    groups = list(all_df["group"].unique())
    title = f"FP8 Gradient Underflows: {' + '.join(groups)}"
    ax.set_title(title, fontsize=18, fontweight="bold", pad=25)

    # Add separator lines between groups
    prev_group = None
    for idx, (_, _, group) in enumerate(pivot_sample.index):
        if prev_group is not None and group != prev_group:
            ax.axhline(y=idx - 0.5, color="white", linestyle="-", linewidth=4, alpha=0.9)
        prev_group = group

    # Mark iteration 3000 if it exists
    if all_df["iteration"].max() >= 3000:
        iter_3000_idx = min(range(len(sample_iterations)), key=lambda i: abs(sample_iterations[i] - 3000))
        ax.axvline(x=iter_3000_idx, color="cyan", linestyle="--", linewidth=3, alpha=0.9)
        ax.text(
            iter_3000_idx,
            -3,
            "Iter 3000",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="cyan",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "cyan", "linewidth": 2},
        )

    # Add group labels on the side
    group_positions = {}
    for idx, (_, _, group) in enumerate(pivot_sample.index):
        if group not in group_positions:
            group_positions[group] = []
        group_positions[group].append(idx)

    group_colors = {"Encoder": "#2E86AB", "Head": "#A23B72", "Embedding": "#F77F00", "Unknown": "#666666"}

    for group, positions in group_positions.items():
        mid_pos = (min(positions) + max(positions)) / 2
        color = group_colors.get(group, "#666666")
        bg_color = {"Encoder": "#E3F2FD", "Head": "#FCE4EC", "Embedding": "#FFF3E0", "Unknown": "#F5F5F5"}

        ax.text(
            -len(sample_iterations) * 0.06,
            mid_pos,
            group.upper(),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            rotation=90,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": bg_color.get(group, "#F5F5F5"),
                "edgecolor": color,
                "linewidth": 2.5,
                "alpha": 0.9,
            },
        )

    # Highlight worst components
    worst_comps = all_df.groupby(["position", "label"])["value"].max().sort_values(ascending=False).head(5)
    for (position, label), max_val in worst_comps.items():
        if max_val > 2.0:
            y_idx = [i for i, (pos, _, _) in enumerate(pivot_sample.index) if pos == position]
            if y_idx:
                rect = Rectangle(
                    (-0.5, y_idx[0] - 0.4),
                    len(sample_iterations),
                    0.8,
                    linewidth=2.5,
                    edgecolor="yellow",
                    facecolor="none",
                    linestyle="-",
                    alpha=0.7,
                )
                ax.add_patch(rect)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#FEF5E7", label="< 0.5% (Acceptable)"),
        mpatches.Patch(facecolor="#F8D7A1", label="0.5-1% (Warning)"),
        mpatches.Patch(facecolor="#F1A468", label="1-2% (Concerning)"),
        mpatches.Patch(facecolor="#E67F83", label="2-4% (Critical)"),
        mpatches.Patch(facecolor="#8B0000", label="> 4% (Severe)"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.95, edgecolor="black", fancybox=True
    )

    # Summary statistics
    total_components = len(pivot_sample.index)
    max_underflow = all_df["value"].max()
    mean_underflow = all_df["value"].mean()
    critical_components = len(worst_comps[worst_comps > 2.0])

    summary_text = f"""Components: {total_components}
Max Underflow: {max_underflow:.2f}%
Mean Underflow: {mean_underflow:.2f}%
Critical (>2%): {critical_components}"""

    ax.text(
        0.98,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "white", "edgecolor": "black", "linewidth": 2, "alpha": 0.95},
    )

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"heatmap_highres{suffix}.png"
    plt.savefig(output_file, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info("")
    logger.info(f"✨ Saved heatmap: {output_file}")
    logger.info(f"   Max underflow: {max_underflow:.2f}%")
    logger.info(f"   Critical components (>2%%): {critical_components}")

    return output_file


def main():
    """Main execution."""
    if len(sys.argv) < 2:
        logger.error("Usage: python3 analyze_and_create_heatmap.py <log_directory> [output_suffix]")
        logger.info("")
        logger.info("Example:")
        logger.info("  python3 analyze_and_create_heatmap.py fp8logswithhead")
        logger.info("  python3 analyze_and_create_heatmap.py log_bf16head _bf16head")
        sys.exit(1)

    log_dir = Path(sys.argv[1])
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""

    logger.info("")
    logger.info("=" * 80)
    logger.info("MODEL-AGNOSTIC FP8 LOG ANALYZER & HEATMAP GENERATOR")
    logger.info("=" * 80)
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Output suffix: '{suffix}' (if provided)")
    logger.info("=" * 80)

    # Parse model structure (optional)
    structure = parse_layer_metadata(log_dir)

    # Find log file
    log_file = log_dir / "rank_0" / "nvdlfw_inspect_statistics_logs" / "nvdlfw_inspect_globalrank-0.log"

    if not log_file.exists():
        logger.error(f"Log file not found: {log_file}")
        sys.exit(1)

    # Step 1: Parse logs
    df = parse_log_file(log_file)

    if len(df) == 0:
        logger.error("No data parsed from log file!")
        sys.exit(1)

    # Step 2: Auto-detect components
    components = auto_detect_components(df, structure)

    if len(components) == 0:
        logger.error("No components detected!")
        sys.exit(1)

    # Step 3: Save to CSV
    csv_dir = Path("analysis_output") / "csv_data"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_file = csv_dir / f"rank_0_metrics{suffix}.csv"

    logger.info("")
    logger.info(f"Saving CSV: {csv_file}")
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved {len(df):,} rows")

    # Step 4: Create heatmap
    heatmap_dir = Path("heatmap_visualization")
    heatmap_file = create_heatmap(df, components, heatmap_dir, suffix)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ COMPLETE")
    logger.info("=" * 80)
    logger.info(f"CSV: {csv_file}")
    logger.info(f"Heatmap: {heatmap_file}")
    logger.info("=" * 80)
    logger.info("")


if __name__ == "__main__":
    main()
