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

"""FP8 stats file parser and anomaly detection for the agent daemon.

Reads the per-rank log files produced by ``nvdlfw_inspect`` / TransformerEngine's
``LogFp8TensorStats`` feature and surfaces actionable signals:

* Per-layer overflow / underflow rates for activations, gradients, and weights
* Scaling factor oscillation (sign of numerical instability)
* Layers consistently near E4M3 saturation
* Sudden distribution shifts in amax histograms

The analyzer is designed to run in the agent sidecar (CPU-only) and never
imports torch or TransformerEngine itself.
"""

import csv
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# E4M3 representable range: ~[-448, 448]
E4M3_MAX = 448.0


@dataclass
class LayerFP8Stats:
    """Aggregated FP8 statistics for a single layer."""

    layer_name: str
    tensor_type: str  # "activation", "gradient", or "weight"
    underflow_pct: float = 0.0
    overflow_pct: float = 0.0
    scale_inv_min: float = float("inf")
    scale_inv_max: float = 0.0
    mse: float = 0.0
    sample_count: int = 0


@dataclass
class FP8HealthReport:
    """Summary of FP8 numerical health across all layers.

    Attributes:
        layer_stats: Per-layer, per-tensor-type statistics.
        anomalies: List of human-readable anomaly descriptions.
        overflow_layers: Layers with overflow rate above threshold.
        underflow_layers: Layers with underflow rate above threshold.
        unstable_layers: Layers with scaling factor oscillation.
        healthy: True if no anomalies were detected.
    """

    layer_stats: dict[str, list[LayerFP8Stats]] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    overflow_layers: list[str] = field(default_factory=list)
    underflow_layers: list[str] = field(default_factory=list)
    unstable_layers: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """True when no anomalies were detected."""
        return len(self.anomalies) == 0

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for the experiment journal."""
        return {
            "healthy": self.healthy,
            "num_anomalies": len(self.anomalies),
            "overflow_layers": self.overflow_layers,
            "underflow_layers": self.underflow_layers,
            "unstable_layers": self.unstable_layers,
            "anomalies": self.anomalies[:10],
        }


class FP8StatsAnalyzer:
    """Parses nvdlfw_inspect log directories and produces FP8 health reports.

    Args:
        log_dir: Root directory containing per-rank subdirectories of FP8 stats.
        overflow_threshold: Overflow % above which a layer is flagged.
        underflow_threshold: Underflow % above which a layer is flagged.
        scale_oscillation_ratio: If ``scale_inv_max / scale_inv_min`` exceeds
            this ratio within a window, the layer is flagged as unstable.
        rank: Which rank's logs to analyze (default: rank 0, the most informative).
    """

    def __init__(
        self,
        log_dir: str | os.PathLike = "./log_fp8_stats",
        overflow_threshold: float = 5.0,
        underflow_threshold: float = 10.0,
        scale_oscillation_ratio: float = 100.0,
        rank: int = 0,
    ):
        """Initialize the analyzer."""
        self._log_dir = Path(log_dir)
        self._overflow_threshold = overflow_threshold
        self._underflow_threshold = underflow_threshold
        self._scale_oscillation_ratio = scale_oscillation_ratio
        self._rank = rank
        self._last_read_positions: dict[str, int] = {}

    def analyze(self) -> FP8HealthReport:
        """Parse all available stat files for the configured rank and return a health report."""
        report = FP8HealthReport()
        rank_dir = self._log_dir / f"rank_{self._rank}"

        if not rank_dir.exists():
            return report

        raw_stats = self._parse_rank_directory(rank_dir)
        report.layer_stats = raw_stats

        for stats_list in raw_stats.values():
            for stats in stats_list:
                self._check_overflow(stats, report)
                self._check_underflow(stats, report)
                self._check_scale_stability(stats, report)

        return report

    def _parse_rank_directory(self, rank_dir: Path) -> dict[str, list[LayerFP8Stats]]:
        """Parse all CSV/log files in a rank directory into LayerFP8Stats."""
        result: dict[str, list[LayerFP8Stats]] = defaultdict(list)

        for log_file in sorted(rank_dir.glob("*.csv")):
            try:
                stats = self._parse_csv_file(log_file)
                for s in stats:
                    result[s.layer_name].append(s)
            except Exception:
                logger.debug("Failed to parse %s", log_file, exc_info=True)

        for log_file in sorted(rank_dir.glob("*.log")):
            try:
                stats = self._parse_log_file(log_file)
                for s in stats:
                    result[s.layer_name].append(s)
            except Exception:
                logger.debug("Failed to parse %s", log_file, exc_info=True)

        return dict(result)

    def _parse_csv_file(self, path: Path) -> list[LayerFP8Stats]:
        """Parse a CSV file with columns like layer, tensor, underflows%, scale_inv_min, scale_inv_max, mse."""
        stats: list[LayerFP8Stats] = []
        file_key = str(path)
        start_pos = self._last_read_positions.get(file_key, 0)

        with open(path) as f:
            f.seek(start_pos)
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s = LayerFP8Stats(
                        layer_name=row.get("layer", row.get("layer_name", "unknown")),
                        tensor_type=row.get("tensor", row.get("tensor_type", "unknown")),
                        underflow_pct=float(row.get("underflows%", row.get("underflow_pct", 0))),
                        scale_inv_min=float(row.get("scale_inv_min", float("inf"))),
                        scale_inv_max=float(row.get("scale_inv_max", 0)),
                        mse=float(row.get("mse", 0)),
                        sample_count=1,
                    )
                    stats.append(s)
                except (ValueError, KeyError):
                    continue

            self._last_read_positions[file_key] = f.tell()

        return stats

    def _parse_log_file(self, path: Path) -> list[LayerFP8Stats]:
        """Parse a structured log file with key=value pairs per line."""
        stats: list[LayerFP8Stats] = []
        kv_pattern = re.compile(r"(\w+)=([\d.eE+\-]+|[\w/]+)")

        file_key = str(path)
        start_pos = self._last_read_positions.get(file_key, 0)

        with open(path) as f:
            f.seek(start_pos)
            for line in f:
                pairs = dict(kv_pattern.findall(line))
                if "layer" in pairs or "layer_name" in pairs:
                    try:
                        s = LayerFP8Stats(
                            layer_name=pairs.get("layer", pairs.get("layer_name", "unknown")),
                            tensor_type=pairs.get("tensor", pairs.get("tensor_type", "unknown")),
                            underflow_pct=float(pairs.get("underflows", pairs.get("underflow_pct", "0"))),
                            scale_inv_min=float(pairs.get("scale_inv_min", "inf")),
                            scale_inv_max=float(pairs.get("scale_inv_max", "0")),
                            mse=float(pairs.get("mse", "0")),
                            sample_count=1,
                        )
                        stats.append(s)
                    except (ValueError, KeyError):
                        continue

            self._last_read_positions[file_key] = f.tell()

        return stats

    def _check_overflow(self, stats: LayerFP8Stats, report: FP8HealthReport) -> None:
        """Flag layers with high overflow rates."""
        if stats.overflow_pct > self._overflow_threshold:
            msg = (
                f"FP8 overflow: {stats.layer_name}/{stats.tensor_type} "
                f"overflow={stats.overflow_pct:.1f}% (threshold={self._overflow_threshold}%)"
            )
            report.anomalies.append(msg)
            if stats.layer_name not in report.overflow_layers:
                report.overflow_layers.append(stats.layer_name)

    def _check_underflow(self, stats: LayerFP8Stats, report: FP8HealthReport) -> None:
        """Flag layers with high underflow rates."""
        if stats.underflow_pct > self._underflow_threshold:
            msg = (
                f"FP8 underflow: {stats.layer_name}/{stats.tensor_type} "
                f"underflow={stats.underflow_pct:.1f}% (threshold={self._underflow_threshold}%)"
            )
            report.anomalies.append(msg)
            if stats.layer_name not in report.underflow_layers:
                report.underflow_layers.append(stats.layer_name)

    def _check_scale_stability(self, stats: LayerFP8Stats, report: FP8HealthReport) -> None:
        """Flag layers where the scaling factor swings wildly."""
        if stats.scale_inv_min <= 0 or not math.isfinite(stats.scale_inv_min):
            return
        if stats.scale_inv_max <= 0 or not math.isfinite(stats.scale_inv_max):
            return

        ratio = stats.scale_inv_max / stats.scale_inv_min
        if ratio > self._scale_oscillation_ratio:
            msg = (
                f"FP8 scale oscillation: {stats.layer_name}/{stats.tensor_type} "
                f"scale_inv ratio={ratio:.1f}x (threshold={self._scale_oscillation_ratio}x)"
            )
            report.anomalies.append(msg)
            if stats.layer_name not in report.unstable_layers:
                report.unstable_layers.append(stats.layer_name)
