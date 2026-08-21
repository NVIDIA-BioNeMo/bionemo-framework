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

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from bionemo.evo2.run.checkpoint_retention import MetricCheckpointRetention


def _context(checkpoint_root: Path, step: int, loss: float | None = None) -> SimpleNamespace:
    context = SimpleNamespace(
        state=SimpleNamespace(
            train_state=SimpleNamespace(step=step),
            cfg=SimpleNamespace(checkpoint=SimpleNamespace(save=str(checkpoint_root))),
        ),
        total_loss_dict=None,
    )
    if loss is not None:
        context.total_loss_dict = {"lm loss": torch.tensor(loss)}
    return context


def _formatted_log_calls(log_method: Mock) -> str:
    """Render %-style logger calls without depending on process-global logging configuration."""
    return "\n".join(str(call.args[0]) % tuple(call.args[1:]) for call in log_method.call_args_list)


def test_keeps_best_and_latest(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    retention = MetricCheckpointRetention(
        metric_name="lm loss",
        keep_best_k=2,
        keep_recent_k=1,
        higher_is_better=False,
    )

    for step, loss in ((1, 0.8), (2, 0.4), (3, 0.6), (4, 0.9)):
        context = _context(tmp_path, step, loss)
        if step == 2:
            context.total_loss_dict["validation accuracy"] = torch.tensor(0.75)
        retention.on_eval_end(context)
        (tmp_path / f"iter_{step:07d}").mkdir()
        retention.on_checkpoint_save(context)

    assert {path.name for path in tmp_path.glob("iter_*")} == {
        "iter_0000002",
        "iter_0000003",
        "iter_0000004",
    }
    metrics = json.loads((tmp_path / "validation_metrics.json").read_text())
    assert set(metrics["metrics_by_validation_step"]) == {"1", "2", "3", "4"}
    assert metrics["metrics_by_validation_step"]["2"]["lm loss"] == pytest.approx(0.4)
    assert metrics["metrics_by_validation_step"]["2"]["validation accuracy"] == pytest.approx(0.75)
    matches = json.loads((tmp_path / "checkpoint_metrics.json").read_text())
    assert matches["metric_name"] == "lm loss"
    assert matches["direction"] == "minimize"
    assert matches["best_checkpoint"] == "iter_0000002"
    assert "New best 'lm loss'=0.8 at validation step 1" in caplog.text
    assert "New best 'lm loss'=0.4 at validation step 2" in caplog.text
    assert "Deleting checkpoint step 1" in caplog.text
    assert "outside best-2 and recent-1 retention" in caplog.text


@pytest.mark.parametrize("higher_is_better", [False, True])
def test_equal_metric_favors_newer_checkpoint(tmp_path: Path, caplog, higher_is_better: bool) -> None:
    caplog.set_level(logging.INFO)
    retention = MetricCheckpointRetention(
        metric_name="lm loss",
        keep_best_k=1,
        keep_recent_k=1,
        higher_is_better=higher_is_better,
    )

    first_context = _context(tmp_path, 1, 0.4)
    retention.on_eval_end(first_context)
    (tmp_path / "iter_0000001").mkdir()
    retention.on_checkpoint_save(first_context)

    retention.on_eval_end(_context(tmp_path, 2, 0.4))
    (tmp_path / "iter_0000003").mkdir()
    retention.on_checkpoint_save(_context(tmp_path, 3))

    last_value = 0.1 if higher_is_better else 0.9
    last_context = _context(tmp_path, 4, last_value)
    retention.on_eval_end(last_context)
    (tmp_path / "iter_0000004").mkdir()
    retention.on_checkpoint_save(last_context)

    assert {path.name for path in tmp_path.glob("iter_*")} == {
        "iter_0000003",
        "iter_0000004",
    }
    assert "New best 'lm loss'=0.4 at validation step 2; retaining checkpoint step 3" in caplog.text
    matches = json.loads((tmp_path / "checkpoint_metrics.json").read_text())
    assert matches["best_checkpoint"] == "iter_0000003"


def test_matches_nearest_past_metric(tmp_path: Path) -> None:
    retention = MetricCheckpointRetention(
        metric_name="lm loss",
        keep_best_k=1,
        keep_recent_k=1,
        higher_is_better=False,
    )
    with patch("bionemo.evo2.run.checkpoint_retention.logger.warning") as warning:
        for step, loss in ((1, 0.8), (2, 0.4)):
            context = _context(tmp_path, step, loss)
            retention.on_eval_end(context)
            (tmp_path / f"iter_{step:07d}").mkdir()
            retention.on_checkpoint_save(context)

        (tmp_path / "iter_0000003").mkdir()
        retention.on_checkpoint_save(_context(tmp_path, 3))

        matches = json.loads((tmp_path / "checkpoint_metrics.json").read_text())
        assert matches["matches_by_checkpoint_step"]["3"]["validation_step"] == 2
        assert matches["matches_by_checkpoint_step"]["3"]["value"] == pytest.approx(0.4)

        (tmp_path / "iter_0000005").mkdir()
        retention.on_checkpoint_save(_context(tmp_path, 5))

    assert {path.name for path in tmp_path.glob("iter_*")} == {
        "iter_0000003",
        "iter_0000005",
    }
    metrics = json.loads((tmp_path / "validation_metrics.json").read_text())
    assert set(metrics["metrics_by_validation_step"]) == {"1", "2"}
    matches = json.loads((tmp_path / "checkpoint_metrics.json").read_text())
    assert "5" not in matches["matches_by_checkpoint_step"]
    assert "has no recorded validation metric" in _formatted_log_calls(warning)


def test_preserves_historical_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "iter_0000001").mkdir()
    (tmp_path / "iter_0000002").mkdir()
    retention = MetricCheckpointRetention(
        metric_name="lm loss",
        keep_best_k=1,
        keep_recent_k=1,
        higher_is_better=False,
    )

    with patch("bionemo.evo2.run.checkpoint_retention.logger.warning") as warning:
        for step, loss in ((3, 0.5), (4, 0.7), (5, 0.3)):
            context = _context(tmp_path, step, loss)
            retention.on_eval_end(context)
            (tmp_path / f"iter_{step:07d}").mkdir()
            retention.on_checkpoint_save(context)

    assert {path.name for path in tmp_path.glob("iter_*")} == {
        "iter_0000001",
        "iter_0000002",
        "iter_0000005",
    }
    metrics = json.loads((tmp_path / "validation_metrics.json").read_text())
    assert set(metrics["metrics_by_validation_step"]) == {"3", "4", "5"}
    matches = json.loads((tmp_path / "checkpoint_metrics.json").read_text())
    assert matches["tracking_started_at_step"] == 3
    assert matches["historical_unscored_steps"] == [1, 2]
    assert "keeping them as historical unscored checkpoints" in _formatted_log_calls(warning)


def test_prefers_exact_recorded_step(tmp_path: Path) -> None:
    retention = MetricCheckpointRetention(
        metric_name="lm loss",
        keep_best_k=1,
        keep_recent_k=1,
        higher_is_better=False,
        step_tolerance=3,
    )

    for step, loss in ((98, 0.2), (100, 0.4), (102, 0.1)):
        retention.on_eval_end(_context(tmp_path, step, loss))
    (tmp_path / "iter_0000100").mkdir()
    retention.on_checkpoint_save(_context(tmp_path, 102))

    matches = json.loads((tmp_path / "checkpoint_metrics.json").read_text())
    assert matches["matches_by_checkpoint_step"]["100"] == {
        "checkpoint_step": 100,
        "validation_step": 100,
        "step_delta": 0,
        "value": pytest.approx(0.4),
    }


def test_accepts_reported_plus_one_step(tmp_path: Path) -> None:
    retention = MetricCheckpointRetention(
        metric_name="lm loss",
        keep_best_k=1,
        keep_recent_k=1,
        higher_is_better=False,
        step_tolerance=1,
    )

    retention.on_eval_end(_context(tmp_path, 101, 0.4))
    (tmp_path / "iter_0000100").mkdir()
    retention.on_checkpoint_save(_context(tmp_path, 101))

    matches = json.loads((tmp_path / "checkpoint_metrics.json").read_text())
    assert matches["matches_by_checkpoint_step"]["100"] == {
        "checkpoint_step": 100,
        "validation_step": 101,
        "step_delta": 1,
        "value": pytest.approx(0.4),
    }


def test_does_not_rematch_after_save(tmp_path: Path) -> None:
    retention = MetricCheckpointRetention(
        metric_name="lm loss",
        keep_best_k=1,
        keep_recent_k=1,
        higher_is_better=False,
        step_tolerance=3,
    )

    retention.on_eval_end(_context(tmp_path, 98, 0.2))
    (tmp_path / "iter_0000100").mkdir()
    retention.on_checkpoint_save(_context(tmp_path, 100))
    retention.on_eval_end(_context(tmp_path, 100, 0.1))

    matches = json.loads((tmp_path / "checkpoint_metrics.json").read_text())
    assert matches["matches_by_checkpoint_step"]["100"]["validation_step"] == 98


def test_missing_metric_falls_back_to_recent(tmp_path: Path) -> None:
    retention = MetricCheckpointRetention(
        metric_name="lm loss",
        keep_best_k=2,
        keep_recent_k=2,
        higher_is_better=False,
        step_tolerance=0,
    )

    with patch("bionemo.evo2.run.checkpoint_retention.logger.warning") as warning:
        for step in (1, 2, 3):
            context = _context(tmp_path, step)
            context.total_loss_dict = {"validation accuracy": torch.tensor(step / 10)}
            retention.on_eval_end(context)
            (tmp_path / f"iter_{step:07d}").mkdir()
            retention.on_checkpoint_save(context)

    assert {path.name for path in tmp_path.glob("iter_*")} == {
        "iter_0000002",
        "iter_0000003",
    }
    metrics = json.loads((tmp_path / "validation_metrics.json").read_text())
    assert metrics["metrics_by_validation_step"]["1"]["validation accuracy"] == pytest.approx(0.1)
    matches = json.loads((tmp_path / "checkpoint_metrics.json").read_text())
    assert matches["matches_by_checkpoint_step"] == {}
    assert matches["best_checkpoint"] is None
    assert matches["unmatched_checkpoint_steps"] == [1, 2, 3]
    assert "falling back to most-recent checkpoint retention" in _formatted_log_calls(warning)


def test_strict_missing_metric(tmp_path: Path) -> None:
    retention = MetricCheckpointRetention(
        metric_name="lm loss",
        keep_best_k=1,
        keep_recent_k=1,
        higher_is_better=False,
        strict_metric=True,
    )

    context = _context(tmp_path, 1)
    context.total_loss_dict = {"validation accuracy": torch.tensor(0.5)}
    with pytest.raises(RuntimeError, match="available raw validation metrics"):
        retention.on_eval_end(context)

    metrics = json.loads((tmp_path / "validation_metrics.json").read_text())
    assert metrics["metrics_by_validation_step"]["1"] == {"validation accuracy": pytest.approx(0.5)}


def test_strict_unmatched_step(tmp_path: Path) -> None:
    retention = MetricCheckpointRetention(
        metric_name="lm loss",
        keep_best_k=1,
        keep_recent_k=1,
        higher_is_better=False,
        step_tolerance=0,
        strict_metric=True,
    )

    retention.on_eval_end(_context(tmp_path, 1, 0.4))
    (tmp_path / "iter_0000002").mkdir()
    with pytest.raises(RuntimeError, match="no recorded validation metric"):
        retention.on_checkpoint_save(_context(tmp_path, 2))

    assert (tmp_path / "iter_0000002").is_dir()
    matches = json.loads((tmp_path / "checkpoint_metrics.json").read_text())
    assert matches["unmatched_checkpoint_steps"] == [2]
    assert matches["observed_checkpoint_steps"] == [2]
