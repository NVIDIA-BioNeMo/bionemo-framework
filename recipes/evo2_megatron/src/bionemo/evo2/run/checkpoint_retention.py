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

"""Metric-aware checkpoint retention for Megatron Bridge training."""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from pathlib import Path
from typing import Any

import torch.distributed as dist
from megatron.bridge.training.callbacks import Callback, CallbackContext


logger = logging.getLogger(__name__)

_ITERATION_DIRECTORY = re.compile(r"iter_(\d+)")
_VALIDATION_METRICS_FILE = "validation_metrics.json"
_CHECKPOINT_METRICS_FILE = "checkpoint_metrics.json"


def _is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _iteration_directories(checkpoint_root: Path) -> dict[int, Path]:
    directories: dict[int, Path] = {}
    if not checkpoint_root.is_dir():
        return directories
    for path in checkpoint_root.iterdir():
        match = _ITERATION_DIRECTORY.fullmatch(path.name)
        if match and path.is_dir():
            directories[int(match.group(1))] = path
    return directories


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, metric in metrics.items():
        try:
            value = float(metric.item() if hasattr(metric, "item") else metric)
        except (TypeError, ValueError, RuntimeError):
            logger.warning("Validation metric %r is not a scalar and will not be recorded", name)
            continue
        if not math.isfinite(value):
            logger.warning("Validation metric %r is not finite and will not be used for retention", name)
            continue
        values[name] = value
    return values


class MetricCheckpointRetention(Callback):
    """Keep the best validation checkpoints together with recent resume points."""

    def __init__(
        self,
        *,
        metric_name: str,
        keep_best_k: int,
        keep_recent_k: int,
        higher_is_better: bool,
        step_tolerance: int = 1,
        strict_metric: bool = False,
    ) -> None:
        """Configure metric matching and retention counts."""
        if not metric_name:
            raise ValueError("metric_name must not be empty")
        if keep_best_k < 1:
            raise ValueError("keep_best_k must be at least 1")
        if keep_recent_k < 1:
            raise ValueError("keep_recent_k must be at least 1")
        if step_tolerance < 0:
            raise ValueError("step_tolerance must not be negative")
        self.metric_name = metric_name
        self.keep_best_k = keep_best_k
        self.keep_recent_k = keep_recent_k
        self.higher_is_better = higher_is_better
        self.step_tolerance = step_tolerance
        self.strict_metric = strict_metric

    @property
    def direction(self) -> str:
        """Return the configured metric direction."""
        return "maximize" if self.higher_is_better else "minimize"

    def _validation_metrics_path(self, checkpoint_root: Path) -> Path:
        return checkpoint_root / _VALIDATION_METRICS_FILE

    def _checkpoint_metrics_path(self, checkpoint_root: Path) -> Path:
        return checkpoint_root / _CHECKPOINT_METRICS_FILE

    def _load_validation_metrics(self, checkpoint_root: Path) -> dict[str, Any]:
        path = self._validation_metrics_path(checkpoint_root)
        if not path.exists():
            return {"metrics_by_validation_step": {}}
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata.get("metrics_by_validation_step"), dict):
            raise ValueError(f"{path} does not contain metrics_by_validation_step")
        return metadata

    def _load_checkpoint_metrics(
        self,
        checkpoint_root: Path,
        current_step: int,
        *,
        new_checkpoint_step: int | None = None,
    ) -> dict[str, Any]:
        path = self._checkpoint_metrics_path(checkpoint_root)
        if path.exists():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            expected = (self.metric_name, self.direction, self.step_tolerance)
            actual = (
                metadata.get("metric_name"),
                metadata.get("direction"),
                metadata.get("step_tolerance"),
            )
            if actual != expected:
                raise ValueError(f"{path} describes {actual}; expected {expected}")
            metadata.setdefault("best_checkpoint", None)
            return metadata

        historical_steps = set(_iteration_directories(checkpoint_root))
        historical_steps.discard(new_checkpoint_step)
        if historical_steps:
            logger.warning(
                "Metric checkpoint tracking started at step %d with %d existing checkpoints; "
                "keeping them as historical unscored checkpoints",
                current_step,
                len(historical_steps),
            )
        return {
            "metric_name": self.metric_name,
            "direction": self.direction,
            "step_tolerance": self.step_tolerance,
            "tracking_started_at_step": current_step,
            "historical_unscored_steps": sorted(historical_steps),
            "observed_checkpoint_steps": [],
            "matches_by_checkpoint_step": {},
            "unmatched_checkpoint_steps": [],
            "best_checkpoint": None,
            "retained_checkpoint_steps": sorted(historical_steps),
        }

    def _write_json(self, path: Path, metadata: dict[str, Any]) -> None:
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _nearest_metric(self, validation_metadata: dict[str, Any], checkpoint_step: int) -> dict[str, Any] | None:
        candidates: list[tuple[int, int, int, float]] = []
        for step_text, metrics in validation_metadata["metrics_by_validation_step"].items():
            if self.metric_name not in metrics:
                continue
            validation_step = int(step_text)
            delta = validation_step - checkpoint_step
            if abs(delta) <= self.step_tolerance:
                candidates.append(
                    (abs(delta), 0 if delta <= 0 else 1, validation_step, float(metrics[self.metric_name]))
                )
        if not candidates:
            return None
        _, _, validation_step, value = min(candidates)
        return {
            "checkpoint_step": checkpoint_step,
            "validation_step": validation_step,
            "step_delta": validation_step - checkpoint_step,
            "value": value,
        }

    def _prune(self, checkpoint_root: Path, metadata: dict[str, Any]) -> None:
        directories = _iteration_directories(checkpoint_root)
        historical = {int(step) for step in metadata["historical_unscored_steps"]}
        matches = metadata["matches_by_checkpoint_step"]
        scored = [
            (int(step), float(match["value"]), abs(int(match["step_delta"])))
            for step, match in matches.items()
            if int(step) in directories
        ]
        # Each item is (checkpoint step, metric value, validation-step distance).
        # Sort by metric, then prefer newer checkpoints on ties, then closer metric matches.
        if self.higher_is_better:
            scored.sort(key=lambda item: (-item[1], -item[0], item[2]))
        else:
            scored.sort(key=lambda item: (item[1], -item[0], item[2]))
        metadata["best_checkpoint"] = f"iter_{scored[0][0]:07d}" if scored else None

        best_steps = {step for step, _, _ in scored[: self.keep_best_k]}
        recent_steps = set(sorted(directories, reverse=True)[: self.keep_recent_k])
        keep = (historical & directories.keys()) | best_steps | recent_steps
        for step, path in directories.items():
            if step not in keep:
                match = matches.get(str(step))
                metric_detail = (
                    f"matched {self.metric_name!r}={float(match['value']):.6g}"
                    if match is not None
                    else f"no matched {self.metric_name!r}"
                )
                logger.info(
                    "Deleting checkpoint step %d at %s: %s and outside best-%d and recent-%d retention",
                    step,
                    path,
                    metric_detail,
                    self.keep_best_k,
                    self.keep_recent_k,
                )
                shutil.rmtree(path)
        metadata["retained_checkpoint_steps"] = sorted(_iteration_directories(checkpoint_root))

    def on_eval_end(self, context: CallbackContext) -> None:
        """Record scalar metrics from a completed validation."""
        if not _is_rank_zero():
            return
        checkpoint_root = Path(context.state.cfg.checkpoint.save)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        current_step = int(context.state.train_state.step)

        scalar_metrics = _scalar_metrics(context.total_loss_dict or {})
        validation_metadata = self._load_validation_metrics(checkpoint_root)
        validation_metadata["metrics_by_validation_step"][str(current_step)] = scalar_metrics
        self._write_json(self._validation_metrics_path(checkpoint_root), validation_metadata)

        checkpoint_metadata = self._load_checkpoint_metrics(checkpoint_root, current_step)
        self._write_json(self._checkpoint_metrics_path(checkpoint_root), checkpoint_metadata)

        if self.strict_metric and self.metric_name not in scalar_metrics:
            raise RuntimeError(
                f"Configured checkpoint metric {self.metric_name!r} was not present; "
                f"available raw validation metrics: {sorted(scalar_metrics)}"
            )

    def on_checkpoint_save(self, context: CallbackContext) -> None:
        """Match and retain checkpoints after a save completes."""
        if not _is_rank_zero():
            return
        checkpoint_root = Path(context.state.cfg.checkpoint.save)
        current_step = int(context.state.train_state.step)
        directories = _iteration_directories(checkpoint_root)
        expected_step = current_step if current_step in directories else None
        if expected_step is None and directories:
            expected_step = max(directories)

        metadata = self._load_checkpoint_metrics(
            checkpoint_root,
            current_step,
            new_checkpoint_step=expected_step,
        )
        historical = {int(step) for step in metadata["historical_unscored_steps"]}
        observed = {int(step) for step in metadata["observed_checkpoint_steps"]}
        candidates = set(directories) - historical - observed
        if current_step in candidates:
            checkpoint_step = current_step
        elif len(candidates) == 1:
            checkpoint_step = candidates.pop()
        else:
            logger.warning(
                "Could not identify one newly saved checkpoint at training step %d; "
                "skipping metric retention for this save",
                current_step,
            )
            metadata["retained_checkpoint_steps"] = sorted(directories)
            self._write_json(self._checkpoint_metrics_path(checkpoint_root), metadata)
            return

        validation_metadata = self._load_validation_metrics(checkpoint_root)
        match = self._nearest_metric(validation_metadata, checkpoint_step)
        if match is None:
            unmatched = {int(step) for step in metadata["unmatched_checkpoint_steps"]}
            unmatched.add(checkpoint_step)
            metadata["unmatched_checkpoint_steps"] = sorted(unmatched)
            message = (
                f"Checkpoint step {checkpoint_step} has no recorded validation metric {self.metric_name!r} "
                f"within ±{self.step_tolerance} steps"
            )
            if self.strict_metric:
                observed.add(checkpoint_step)
                metadata["observed_checkpoint_steps"] = sorted(observed)
                metadata["retained_checkpoint_steps"] = sorted(directories)
                self._write_json(self._checkpoint_metrics_path(checkpoint_root), metadata)
                raise RuntimeError(message)
            logger.warning("%s; falling back to most-recent checkpoint retention", message)
        else:
            previous_values = [
                float(previous_match["value"]) for previous_match in metadata["matches_by_checkpoint_step"].values()
            ]
            value = float(match["value"])
            is_new_best = not previous_values or (
                value >= max(previous_values) if self.higher_is_better else value <= min(previous_values)
            )
            if is_new_best:
                logger.info(
                    "New best %r=%.6g at validation step %d; retaining checkpoint step %d",
                    self.metric_name,
                    value,
                    match["validation_step"],
                    checkpoint_step,
                )
            else:
                logger.info(
                    "Matched checkpoint step %d to validation step %d with %r=%.6g",
                    checkpoint_step,
                    match["validation_step"],
                    self.metric_name,
                    value,
                )
            metadata["matches_by_checkpoint_step"][str(checkpoint_step)] = match

        observed.add(checkpoint_step)
        metadata["observed_checkpoint_steps"] = sorted(observed)
        self._prune(checkpoint_root, metadata)
        self._write_json(self._checkpoint_metrics_path(checkpoint_root), metadata)
