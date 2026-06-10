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

"""Utilities for layer-wise quantization configuration (FP8/FP4)."""

import logging
import tempfile
from pathlib import Path

import yaml
from nvdlfw_inspect.logging import BaseLogger


logger = logging.getLogger(__name__)


class WandBQuantLogger(BaseLogger):
    """Forward nvdlfw_inspect quant stats to WandB as scalars.

    Each stat is logged under the ``quant/`` prefix so it appears alongside
    training metrics (loss, perplexity, etc.) in a single WandB dashboard.
    """

    def log_scalar(self, name: str, value: float | int, iteration: int, **kwargs):
        """Log a single quant stat to WandB."""
        import wandb

        wandb.log({f"quant/{name}": value}, step=iteration)


def generate_layer_regex(layer_numbers: list[int] | None) -> str:
    """Generate a regex pattern to match specific layer numbers (1-indexed).

    The debug API (nvdlfw_inspect) uses 1-indexed layer names after ``infer_and_assign_layer_names``.

    Args:
        layer_numbers: List of layer numbers (1-indexed, as shown in debug logs).
                       If empty or None, returns a pattern that matches nothing.

    Returns:
        Regex pattern string for matching those layers' linear sublayers.
    """
    if not layer_numbers:
        return r"model\.model\.encoder\.layers\.DISABLED_NO_LAYERS_SPECIFIED"
    layer_pattern = "|".join(str(n) for n in sorted(layer_numbers))
    return rf"model\.model\.encoder\.layers\.({layer_pattern})\..*(layernorm_qkv|proj|fc1|fc2)"


def update_quant_stats_config(
    config_file: str,
    fp4_layers: list[int] | None,
    fp8_layers: list[int] | None,
) -> str:
    """Update the quant stats YAML config with layer-specific regex patterns.

    Args:
        config_file: Path to the original YAML config file.
        fp4_layers: List of layer numbers for FP4 (1-indexed).
        fp8_layers: List of layer numbers for FP8 (1-indexed).

    Returns:
        Path to the updated config file (a temp file).
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if "example_fp4_tensor_stat_collection" in config:
        # TODO: Remove this block and replace with FP8-style regex update once a TransformerEngine
        # release with LogNvfp4TensorStats support is available. At that point, this becomes:
        #     fp4_regex = generate_layer_regex(fp4_layers)
        #     config["example_fp4_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"] = fp4_regex
        config["example_fp4_tensor_stat_collection"]["enabled"] = False
        if fp4_layers:
            logger.warning(
                "NVFP4 quant stats logging is not yet supported (requires a future TransformerEngine release). "
                f"Disabling FP4 stats collection for layers {fp4_layers}. FP8 stats will still be collected."
            )
        else:
            logger.info("FP4 stats section disabled (no FP4 layers and feature not yet supported)")

    if "example_fp8_tensor_stat_collection" in config:
        fp8_regex = generate_layer_regex(fp8_layers)
        config["example_fp8_tensor_stat_collection"]["layers"]["layer_name_regex_pattern"] = fp8_regex
        if fp8_layers:
            logger.info(f"Updated FP8 layer regex to match layers: {fp8_layers}")
        else:
            logger.info("FP8 layers empty - regex set to match nothing")

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
) -> None:
    """Set up quantization stats logging via nvdlfw_inspect.

    Updates the quant stats YAML config with resolved layer regex patterns, creates
    the per-rank log directory, and initializes the debug API.

    Args:
        quant_stats_file: Path to the base quant stats YAML config file.
        quant_log_dir: Base directory for quant stats logs (a rank subdirectory will be created).
        rank: The global rank of this process.
        layer_precision: Per-layer precision list (0-indexed by position). Each element is
            ``"fp8"``, ``"fp4"``, or ``None``.
        statistics_logger: Optional custom logger (e.g. :class:`WandBQuantLogger`) that receives
            every ``log_scalar`` call from the debug API.  When provided together with
            ``default_logging_enabled=True`` the file logger is kept as well.
    """
    import nvdlfw_inspect.api as debug_api
    import transformer_engine

    # Derive 1-indexed layer lists for the debug API, which uses 1-indexed layer names.
    fp8_layers_1indexed = [i + 1 for i, p in enumerate(layer_precision) if p == "fp8"] or None
    fp4_layers_1indexed = [i + 1 for i, p in enumerate(layer_precision) if p == "fp4"] or None
    updated_config = update_quant_stats_config(
        config_file=quant_stats_file,
        fp4_layers=fp4_layers_1indexed,
        fp8_layers=fp8_layers_1indexed,
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


def resolve_layer_precision(
    num_layers: int,
    fp8_enabled: bool,
    fp4_enabled: bool,
    fp8_layers: list[int] | None,
    fp4_layers: list[int] | None,
) -> list[str | None]:
    """Resolve layer-wise quantization assignments from user config.

    TODO(BIO-326): Remove this and move to directly initializing something in NVEsmConfig.

    Takes 1-indexed layer lists (as specified by the user in YAML config) and returns a per-layer
    precision list (0-indexed by position). When a quantization format is enabled but no layer list
    is provided, all layers default to that format. When one format has explicit layers and the other
    is enabled without a layer list, the unspecified format defaults to the remaining (unclaimed) layers.

    Args:
        num_layers: Total number of transformer layers in the model.
        fp8_enabled: Whether FP8 quantization is enabled.
        fp4_enabled: Whether FP4 quantization is enabled.
        fp8_layers: 1-indexed list of layers for FP8, or None if not specified.
        fp4_layers: 1-indexed list of layers for FP4, or None if not specified.

    Returns:
        A list of length ``num_layers`` where each element is ``"fp8"``, ``"fp4"``, or ``None``
        (BF16 fallback), indexed by layer position (0-indexed).

    Raises:
        ValueError: If both formats are enabled with no layer lists, or if layer lists overlap.
    """
    all_layers = set(range(1, num_layers + 1))

    if fp8_enabled and fp4_enabled and fp8_layers is None and fp4_layers is None:
        raise ValueError(
            "Both fp8_config and fp4_config are enabled but neither fp8_layers nor fp4_layers is specified. "
            "When both are enabled, you must explicitly provide layer lists to indicate which layers use which format."
        )

    # When one format has explicit layers and the other defaults, fill in the remaining layers.
    if fp8_enabled and fp8_layers is None:
        claimed_by_fp4 = set(fp4_layers) if fp4_layers is not None else set()
        fp8_layers = sorted(all_layers - claimed_by_fp4)
        if claimed_by_fp4:
            logger.warning(
                f"fp8_config.enabled=True with no fp8_layers specified, but fp4_layers={sorted(claimed_by_fp4)} "
                f"are already claimed by FP4. Defaulting FP8 to the remaining layers: {fp8_layers}"
            )
        else:
            logger.info(
                f"fp8_config.enabled=True with no fp8_layers specified, defaulting all {num_layers} layers to FP8"
            )

    if fp4_enabled and fp4_layers is None:
        claimed_by_fp8 = set(fp8_layers) if fp8_layers is not None else set()
        fp4_layers = sorted(all_layers - claimed_by_fp8)
        if claimed_by_fp8:
            logger.warning(
                f"fp4_config.enabled=True with no fp4_layers specified, but fp8_layers={sorted(claimed_by_fp8)} "
                f"are already claimed by FP8. Defaulting FP4 to the remaining layers: {fp4_layers}"
            )
        else:
            logger.info(
                f"fp4_config.enabled=True with no fp4_layers specified, defaulting all {num_layers} layers to FP4"
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
                f"fp8_layers and fp4_layers cannot have overlapping layer numbers. Found overlap: {sorted(overlap)}"
            )

    # Build per-layer precision list (0-indexed by position, 1-indexed for lookup).
    fp8_set = set(fp8_layers) if fp8_layers is not None else set()
    fp4_set = set(fp4_layers) if fp4_layers is not None else set()
    return [
        "fp8" if layer_1indexed in fp8_set else "fp4" if layer_1indexed in fp4_set else None
        for layer_1indexed in range(1, num_layers + 1)
    ]
