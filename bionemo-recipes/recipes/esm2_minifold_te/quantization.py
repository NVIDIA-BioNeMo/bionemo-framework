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

"""Utilities for block-wise quantization configuration (FP8/FP4) for the MiniFold folding head.

Adapted from esm2_native_te/quantization.py. Uses the same API (fp8_layers/fp4_layers)
but applied to MiniFormer blocks instead of transformer layers.
"""

import logging
import re
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager

import matplotlib
import numpy as np
import transformer_engine.pytorch as te
import yaml
from nvdlfw_inspect.logging import BaseLogger


matplotlib.use("Agg")


logger = logging.getLogger(__name__)


def resolve_layer_precision(
    num_layers: int,
    fp8_enabled: bool,
    fp4_enabled: bool,
    fp8_layers: list[int] | None,
    fp4_layers: list[int] | None,
) -> list[str | None]:
    """Resolve block-wise quantization assignments from user config.

    Takes 1-indexed block lists (as specified by the user in YAML config) and returns a per-block
    precision list (0-indexed by position). When a quantization format is enabled but no block list
    is provided, all blocks default to that format. When one format has explicit blocks and the other
    is enabled without a block list, the unspecified format defaults to the remaining (unclaimed) blocks.

    Args:
        num_layers: Total number of MiniFormer blocks in the folding head.
        fp8_enabled: Whether FP8 quantization is enabled.
        fp4_enabled: Whether FP4 quantization is enabled.
        fp8_layers: 1-indexed list of blocks for FP8, or None if not specified.
        fp4_layers: 1-indexed list of blocks for FP4, or None if not specified.

    Returns:
        A list of length ``num_layers`` where each element is ``"fp8"``, ``"fp4"``, or ``None``
        (BF16 fallback), indexed by block position (0-indexed).

    Raises:
        ValueError: If both formats are enabled with no block lists, or if block lists overlap.
    """
    all_layers = set(range(1, num_layers + 1))

    if fp8_enabled and fp4_enabled and fp8_layers is None and fp4_layers is None:
        raise ValueError(
            "Both fp8_config and fp4_config are enabled but neither fp8_layers nor fp4_layers is specified. "
            "When both are enabled, you must explicitly provide layer lists to indicate which blocks use which format."
        )

    # When one format has explicit layers and the other defaults, fill in the remaining layers.
    if fp8_enabled and fp8_layers is None:
        claimed_by_fp4 = set(fp4_layers) if fp4_layers is not None else set()
        fp8_layers = sorted(all_layers - claimed_by_fp4)
        if claimed_by_fp4:
            logger.warning(
                f"fp8_config.enabled=True with no fp8_layers specified, but fp4_layers={sorted(claimed_by_fp4)} "
                f"are already claimed by FP4. Defaulting FP8 to the remaining blocks: {fp8_layers}"
            )
        else:
            logger.info(
                f"fp8_config.enabled=True with no fp8_layers specified, defaulting all {num_layers} blocks to FP8"
            )

    if fp4_enabled and fp4_layers is None:
        claimed_by_fp8 = set(fp8_layers) if fp8_layers is not None else set()
        fp4_layers = sorted(all_layers - claimed_by_fp8)
        if claimed_by_fp8:
            logger.warning(
                f"fp4_config.enabled=True with no fp4_layers specified, but fp8_layers={sorted(claimed_by_fp8)} "
                f"are already claimed by FP8. Defaulting FP4 to the remaining blocks: {fp4_layers}"
            )
        else:
            logger.info(
                f"fp4_config.enabled=True with no fp4_layers specified, defaulting all {num_layers} blocks to FP4"
            )

    # Disable layer lists when corresponding config is not enabled.
    if not fp8_enabled:
        fp8_layers = None
    if not fp4_enabled:
        fp4_layers = None

    # Validate no overlap between FP8 and FP4 layer assignments.
    if fp8_layers is not None and fp4_layers is not None:
        overlap = set(fp8_layers) & set(fp4_layers)
        if overlap:
            raise ValueError(
                f"fp8_layers and fp4_layers cannot have overlapping block numbers. Found overlap: {sorted(overlap)}"
            )

    # Build per-block precision list (0-indexed by position, 1-indexed for lookup).
    fp8_set = set(fp8_layers) if fp8_layers is not None else set()
    fp4_set = set(fp4_layers) if fp4_layers is not None else set()
    return [
        "fp8" if layer_1indexed in fp8_set else "fp4" if layer_1indexed in fp4_set else None
        for layer_1indexed in range(1, num_layers + 1)
    ]


@dataclass
class ComponentPrecisionConfig:
    """Per-component precision overrides within FP8/FP4 blocks.

    When a block runs in FP8/FP4 via te.autocast, these flags control which sub-components
    participate. Components set to True run in the block's precision; components set to False
    are wrapped in te.autocast(enabled=False) to stay in BF16.

    Only meaningful when block-level FP8/FP4 is enabled. When all blocks are BF16,
    these flags have no effect.

    Attributes:
        tri_proj: Triangular update input/output projections (pi, po).
        tri_gate: Triangular update sigmoid gates (gi, go).
        tri_einsum: Triangular multiplication matmuls (reshaped einsum).
            "off" = forced FP32 (default). "bf16" = ambient dtype (recommended).
        tri_impl: Triangular multiplication backend.
            "einsum" = original literal torch.einsum path.
            "bmm" = current PyTorch/cuBLAS batched path.
            "flash_trimul" = Triton fused gate+tri-mul path with backward recompute.
            "fp8_bmm" = custom MXFP8 batched GEMM path with saved FP8 activations.
            "fp8_cublaslt" = raw cuBLASLt MXFP8 batched GEMM backend.
            "fp8_grouped" = TE grouped FP8 GEMM prototype.
            "cublas_xbdnn" = specialized BF16 cuBLAS backend for `(B, 128, N, N)`.
            "fused_fp8" = paired native MXFP8 cuBLASLt backend exposed as a fused recipe path.
        ffn: Transition update FFN layers (fc1, fc2).
        struct_attn: Structure module attention projections (proj, o_proj, g_proj).
        struct_ffn: Structure module transition MLP layers.
        seq_proj: Sequence and pair feature projections (fc_s, fc_z, seq_to_pair).
        dist_head: Distogram output head (fc_out_1, fc_out_2).
        tri_saved_tensors_fp8: Experimentally pack autograd-saved tensors inside
            TriangularUpdateTE to block-scaled FP8.
        tri_gating_chain_fp8: Use the single dequant/requant triangular gating
            chain wrapper instead of per-op FP8 dispatch for sigmoid/mul.
        ffn_saved_tensors_fp8: Experimentally pack autograd-saved tensors inside
            TransitionUpdateTE to block-scaled FP8.
        ffn_relu_native_fp8: Use the native raw-FP8 ReLU path inside
            TransitionUpdateTE instead of the per-op dispatch path.
        ffn_fused_subgraph_fp8: Use a custom autograd wrapper for TransitionUpdateTE
            that recomputes through TE modules in backward and saves quantized state.
        ffn_fused_subgraph_debug_memory: Print coarse CUDA memory checkpoints inside
            the custom TransitionUpdateTE backward path.
        tri_checkpoint_reentrant: Wrap TriangularUpdateTE in reentrant activation
            checkpointing at the MiniFormer block level.
        tri_prefix_checkpoint_reentrant: Wrap only the cheap projection/gating
            prefix of TriangularUpdateTE in reentrant checkpointing.
        ffn_checkpoint_reentrant: Wrap TransitionUpdateTE in reentrant activation
            checkpointing at the MiniFormer block level.
        saved_tensors_fp8_include_fp32: Also pack FP32 saved tensors, not only BF16.
    """

    tri_proj: bool = True
    tri_gate: bool = True
    tri_einsum: str = "off"
    tri_impl: str = "bmm"
    ffn: bool = True
    struct_attn: bool = True
    struct_ffn: bool = True
    seq_proj: bool = True
    dist_head: bool = True
    tri_saved_tensors_fp8: bool = False
    tri_gating_chain_fp8: bool = False
    ffn_saved_tensors_fp8: bool = False
    ffn_relu_native_fp8: bool = False
    ffn_fused_subgraph_fp8: bool = False
    ffn_fused_subgraph_debug_memory: bool = False
    tri_checkpoint_reentrant: bool = False
    tri_prefix_checkpoint_reentrant: bool = False
    ffn_checkpoint_reentrant: bool = False
    saved_tensors_fp8_include_fp32: bool = False

    def __post_init__(self):
        """Normalize tri_einsum for backward compatibility with bool configs."""
        if isinstance(self.tri_einsum, bool):
            self.tri_einsum = "bf16" if self.tri_einsum else "off"
        valid_tri_impls = {
            "einsum",
            "bmm",
            "flash_trimul",
            "fp8_bmm",
            "fp8_cublaslt",
            "fp8_grouped",
            "cublas_xbdnn",
            "fused",
            "fused_fp8",
        }
        if self.tri_impl not in valid_tri_impls:
            raise ValueError(f"tri_impl must be one of {sorted(valid_tri_impls)}, got {self.tri_impl!r}")

    def get_context(self, component: str) -> ContextManager:
        """Return te.autocast(enabled=False) if the component is disabled, else nullcontext."""
        if getattr(self, component, True):
            return nullcontext()
        return te.autocast(enabled=False)


class WandBQuantLogger(BaseLogger):
    """Forward nvdlfw_inspect quant stats to WandB as scalars.

    Each stat is logged under the ``quant/`` prefix so it appears alongside
    training metrics (loss, lDDT, etc.) in a single WandB dashboard.
    """

    def log_scalar(self, name: str, value: float | int, iteration: int, **kwargs):
        """Log a single quant stat to WandB."""
        import wandb

        wandb.log({f"quant/{name}": value})


_MINIFOLD_UNDERFLOW_PATTERN = re.compile(r"blocks\.(\d+)\.(\w+)\.(\w+)_gradient_underflows%")


class BufferedQuantLogger(BaseLogger):
    """Buffer gradient underflow stats in memory and optionally forward all stats to WandB.

    Accumulates gradient_underflows% values keyed by metric name and iteration,
    enabling periodic heatmap generation without post-hoc log parsing.
    """

    def __init__(self):
        self._underflow_buffer: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def log_scalar(self, name: str, value: float | int, iteration: int, **kwargs):
        """Buffer gradient_underflows% for heatmaps. Scalar stats are logged via file logger."""
        if "gradient_underflows%" in name:
            self._underflow_buffer[name].append((iteration, value))

    def generate_heatmap(self):
        """Create a publication-quality heatmap from buffered gradient underflow data.

        Adapted from fp8_analysis/analyze_and_create_heatmap.py with MiniFold-specific
        block/component grouping, severity legend, summary stats, and yellow highlights
        for critical components.

        Returns:
            matplotlib.figure.Figure or None if no data has been buffered.
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Rectangle

        if not self._underflow_buffer:
            return None

        # Parse metric names into (block_num, module, sublayer) tuples
        components = []
        for metric_name in self._underflow_buffer:
            match = _MINIFOLD_UNDERFLOW_PATTERN.search(metric_name)
            if match:
                block = int(match.group(1))
                module = match.group(2)
                sublayer = match.group(3)
                sort_key = (block, module, sublayer)
                label = f"B{block} {sublayer}"
                group = "Triangular" if module == "triangular" else "FFN"
                components.append((sort_key, label, metric_name, group))

        if not components:
            return None

        components.sort(key=lambda x: x[0])

        # Collect all unique iterations and subsample for visualization
        all_iterations = sorted({it for data in self._underflow_buffer.values() for it, _ in data})
        sample_iterations = all_iterations[:: max(1, len(all_iterations) // 120)]

        # Build 2D array
        iter_to_col = {it: i for i, it in enumerate(sample_iterations)}
        matrix = np.full((len(components), len(sample_iterations)), np.nan)
        labels = []
        groups = []

        for row_idx, (_, label, metric_name, group) in enumerate(components):
            labels.append(label)
            groups.append(group)
            for iteration, value in self._underflow_buffer[metric_name]:
                if iteration in iter_to_col:
                    matrix[row_idx, iter_to_col[iteration]] = value

        # Create figure with colorbar
        fig = plt.figure(figsize=(22, max(10, len(components) * 0.4)))
        ax = plt.subplot2grid((20, 20), (0, 1), colspan=18, rowspan=18)
        cax = plt.subplot2grid((20, 20), (0, 19), rowspan=18)

        sns.set_style("white")
        cmap = sns.color_palette("rocket_r", as_cmap=True)
        max_val = min(6.0, float(np.nanmax(matrix))) if not np.all(np.isnan(matrix)) else 6.0

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=max_val)

        # Colorbar
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Gradient Underflows %", fontsize=14, fontweight="bold", rotation=270, labelpad=25)
        cbar.ax.tick_params(labelsize=11)

        # Y-axis
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=min(10, max(6, 200 // max(len(labels), 1))))
        ax.set_ylabel("Component", fontsize=14, fontweight="bold")

        # X-axis
        x_tick_positions = np.linspace(0, len(sample_iterations) - 1, min(12, len(sample_iterations))).astype(int)
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels([f"{int(sample_iterations[i])}" for i in x_tick_positions], fontsize=11)
        ax.set_xlabel("Training Iteration", fontsize=14, fontweight="bold")

        # Title
        unique_groups = list(dict.fromkeys(groups))
        ax.set_title(
            f"FP8 Gradient Underflows: MiniFold {' + '.join(unique_groups)}", fontsize=18, fontweight="bold", pad=25
        )

        # Block separator lines
        prev_block = None
        for idx, (sort_key, _, _, _) in enumerate(components):
            block = sort_key[0]
            if prev_block is not None and block != prev_block:
                ax.axhline(y=idx - 0.5, color="white", linestyle="-", linewidth=4, alpha=0.9)
            prev_block = block

        # Group labels on the side
        group_positions = {}
        for idx, g in enumerate(groups):
            if g not in group_positions:
                group_positions[g] = []
            group_positions[g].append(idx)

        group_colors = {"FFN": "#2E86AB", "Triangular": "#A23B72"}
        group_bg = {"FFN": "#E3F2FD", "Triangular": "#FCE4EC"}

        for group, positions in group_positions.items():
            mid_pos = (min(positions) + max(positions)) / 2
            color = group_colors.get(group, "#666666")
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
                    "facecolor": group_bg.get(group, "#F5F5F5"),
                    "edgecolor": color,
                    "linewidth": 2.5,
                    "alpha": 0.9,
                },
            )

        # Highlight worst components (>2% underflow)
        for row_idx in range(len(components)):
            row_max = float(np.nanmax(matrix[row_idx])) if not np.all(np.isnan(matrix[row_idx])) else 0
            if row_max > 2.0:
                rect = Rectangle(
                    (-0.5, row_idx - 0.4),
                    len(sample_iterations),
                    0.8,
                    linewidth=2.5,
                    edgecolor="yellow",
                    facecolor="none",
                    linestyle="-",
                    alpha=0.7,
                )
                ax.add_patch(rect)

        # Severity legend
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

        # Summary statistics box
        all_values = matrix[~np.isnan(matrix)]
        total_components = len(components)
        max_underflow = float(all_values.max()) if len(all_values) > 0 else 0
        mean_underflow = float(all_values.mean()) if len(all_values) > 0 else 0
        critical = sum(1 for r in range(len(components)) if float(np.nanmax(matrix[r])) > 2.0)

        summary_text = (
            f"Components: {total_components}\n"
            f"Max Underflow: {max_underflow:.2f}%\n"
            f"Mean Underflow: {mean_underflow:.2f}%\n"
            f"Critical (>2%): {critical}"
        )
        ax.text(
            0.98,
            0.98,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 2,
                "alpha": 0.95,
            },
        )

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        return fig


def generate_layer_regex(
    block_numbers: list[int] | None,
    component_precision: ComponentPrecisionConfig | None = None,
) -> str:
    """Generate a regex pattern to match specific MiniFormer block numbers (1-indexed).

    The debug API (nvdlfw_inspect) uses layer names assigned by ``infer_and_assign_layer_names``.
    Block numbers in the user config are 1-indexed, but module names are 0-indexed, so this
    function converts accordingly. Only includes sublayers whose component_precision is enabled.

    Args:
        block_numbers: List of block numbers (1-indexed, as specified in fp8_layers config).
                       If empty or None, returns a pattern that matches nothing.
        component_precision: Per-component precision config. Only sublayers with enabled components
                             are included in the regex. If None, all sublayers are included.

    Returns:
        Regex pattern string for matching those blocks' te.Linear sublayers.
    """
    if not block_numbers:
        return r"fold\.miniformer\.blocks\.DISABLED_NO_BLOCKS_SPECIFIED"

    # Determine which sublayers are actually running in FP8 based on component_precision
    sublayers = []
    if component_precision is None or component_precision.tri_proj:
        sublayers.extend(["pi", "po"])
    if component_precision is None or component_precision.tri_gate:
        sublayers.extend(["gi", "go"])
    if component_precision is None or component_precision.ffn:
        sublayers.extend(["fc1", "fc2"])

    if not sublayers:
        return r"fold\.miniformer\.blocks\.DISABLED_NO_COMPONENTS_ENABLED"

    # Convert 1-indexed (user config) to 0-indexed (module names)
    block_pattern = "|".join(str(n - 1) for n in sorted(block_numbers))
    sublayer_pattern = "|".join(sublayers)
    return rf"fold\.miniformer\.blocks\.({block_pattern})\..*({sublayer_pattern})"


def update_quant_stats_config(
    config_file: str,
    fp4_layers: list[int] | None,
    fp8_layers: list[int] | None,
    component_precision: ComponentPrecisionConfig | None = None,
) -> str:
    """Update the quant stats YAML config with block-specific regex patterns.

    Args:
        config_file: Path to the original YAML config file.
        fp4_layers: List of block numbers for FP4 (1-indexed).
        fp8_layers: List of block numbers for FP8 (1-indexed).

    Returns:
        Path to the updated config file (a temp file).
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if "example_fp4_tensor_stat_collection" in config:
        fp4_regex = generate_layer_regex(fp4_layers, component_precision=component_precision)
        config["example_fp4_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"] = fp4_regex
        if fp4_layers:
            logger.info(f"Updated FP4 block regex to match blocks: {fp4_layers}")
        else:
            logger.info("FP4 blocks empty - regex set to match nothing")

    if "example_fp8_tensor_stat_collection" in config:
        fp8_regex = generate_layer_regex(fp8_layers, component_precision=component_precision)
        config["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"] = fp8_regex
        if fp8_layers:
            logger.info(f"Updated FP8 block regex to match blocks: {fp8_layers}")
        else:
            logger.info("FP8 blocks empty - regex set to match nothing")

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, temp_file, default_flow_style=False)
    temp_file.close()

    config_str = yaml.dump(config, default_flow_style=False)
    logger.info(f"Created updated quant stats config at: {temp_file.name}")
    logger.info(f"Updated quant stats config contents:\n{config_str}")

    return temp_file.name


def initialize_quant_stats_logging(
    quant_stats_file: str,
    quant_log_dir: str,
    rank: int,
    layer_precision: list[str | None],
    statistics_logger: BaseLogger | None = None,
    component_precision: ComponentPrecisionConfig | None = None,
) -> None:
    """Set up quantization stats logging via nvdlfw_inspect.

    Args:
        quant_stats_file: Path to the base quant stats YAML config file.
        quant_log_dir: Base directory for quant stats logs (a rank subdirectory will be created).
        rank: The global rank of this process.
        layer_precision: Per-block precision list (0-indexed by position). Each element is
            ``"fp8"``, ``"fp4"``, or ``None``.
        statistics_logger: Optional custom logger (e.g. :class:`WandBQuantLogger`) that receives
            every ``log_scalar`` call from the debug API.
        component_precision: Per-component precision config. Only sublayers with enabled components
            are included in the stats regex to avoid inspecting layers not running in FP8.
    """
    import nvdlfw_inspect.api as debug_api
    import transformer_engine

    fp8_layers_1indexed = [i + 1 for i, p in enumerate(layer_precision) if p == "fp8"] or None
    fp4_layers_1indexed = [i + 1 for i, p in enumerate(layer_precision) if p == "fp4"] or None
    updated_config = update_quant_stats_config(
        config_file=quant_stats_file,
        fp4_layers=fp4_layers_1indexed,
        fp8_layers=fp8_layers_1indexed,
        component_precision=component_precision,
    )

    rank_log_dir = Path(quant_log_dir) / f"rank_{rank}"
    rank_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logging quant stats to {rank_log_dir}")

    te_features_dir = str(Path(transformer_engine.__file__).parent / "debug" / "features")
    debug_api.initialize(
        config_file=updated_config,
        feature_dirs=[te_features_dir],
        log_dir=rank_log_dir,
        statistics_logger=statistics_logger,
        default_logging_enabled=True,
    )
