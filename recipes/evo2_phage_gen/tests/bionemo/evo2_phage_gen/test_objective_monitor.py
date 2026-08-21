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

import argparse
import sys

import pytest

from bionemo.evo2_phage_gen import objective_monitor
from bionemo.evo2_phage_gen.objective_monitor import (
    _positive_int,
    evaluate_objective_history,
    extract_validation_history,
)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_nonpositive_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _positive_int(value)


def _event(
    step: int,
    reward: float,
    support: float,
    *,
    std: float = 0.2,
    denominator: int = 96,
    pass_rate: float | None = None,
) -> dict:
    objective = {
        "reward_mean": reward,
        "reward_std": std,
        "nonzero_rate": 0.8,
        "support_rate": support,
        "eligible_denominator": denominator,
        "missing_rate": 1.0 - support,
    }
    if pass_rate is not None:
        objective["hard_pass_rate"] = pass_rate
    return {
        "step": step,
        "aggregate_reward": 5.0,
        "objectives": {"protein_hit_count": objective},
    }


def test_reward_gain_with_collapsing_support_starts_rebound_window():
    history = [
        _event(10, 0.20, 0.90),
        _event(20, 0.35, 0.70),
        _event(30, 0.55, 0.45),
    ]

    result = evaluate_objective_history(history)

    assert result["decision"] == "continue"
    assert result["reason"] == "signal_pending_confirmation:1/8"
    assert result["latest_complete_step"] == 30
    assert result["objectives"]["protein_hit_count"]["status"] == "warning"
    assert result["objectives"]["protein_hit_count"]["signal_streak"] == 1
    assert "reward_support_divergence" in result["objectives"]["protein_hit_count"]["signals"]


def test_sustained_reward_support_divergence_pauses_after_rebound_window():
    history = [_event(step * 10, 0.10 + 0.08 * step, 0.90 - 0.08 * step) for step in range(10)]

    result = evaluate_objective_history(history)

    assert result["decision"] == "pause_for_diagnosis"
    assert result["objectives"]["protein_hit_count"]["status"] == "suspicious"
    assert result["objectives"]["protein_hit_count"]["signal_streak"] == 8


def test_reward_and_support_improving_together_continues():
    history = [
        _event(10, 0.20, 0.60, pass_rate=0.10),
        _event(20, 0.35, 0.75, pass_rate=0.20),
        _event(30, 0.50, 0.90, pass_rate=0.35),
    ]

    result = evaluate_objective_history(history)

    assert result["decision"] == "continue"
    assert result["objectives"]["protein_hit_count"]["status"] == "healthy"


def test_objective_history_accepts_custom_hard_pass_and_instability_thresholds():
    hard_pass_history = [
        _event(10, 0.0, 1.0, pass_rate=0.50),
        _event(20, 0.1, 1.0, pass_rate=0.48),
        _event(30, 0.2, 1.0, pass_rate=0.46),
    ]
    instability_history = [
        _event(10, 0.0, 1.0),
        _event(20, 0.3, 1.0),
        _event(30, 0.0, 1.0),
    ]

    strict_hard_pass = evaluate_objective_history(hard_pass_history, hard_pass_drop_threshold=0.10)
    sensitive_hard_pass = evaluate_objective_history(hard_pass_history, hard_pass_drop_threshold=0.03)
    strict_instability = evaluate_objective_history(
        instability_history, objective_reward_range_threshold=0.50, minimum_reward_sign_changes=2
    )
    sensitive_instability = evaluate_objective_history(
        instability_history, objective_reward_range_threshold=0.25, minimum_reward_sign_changes=1
    )

    assert "reward_hard_pass_divergence" not in strict_hard_pass["objectives"]["protein_hit_count"]["signals"]
    assert "reward_hard_pass_divergence" in sensitive_hard_pass["objectives"]["protein_hit_count"]["signals"]
    assert "objective_instability" not in strict_instability["objectives"]["protein_hit_count"]["signals"]
    assert "objective_instability" in sensitive_instability["objectives"]["protein_hit_count"]["signals"]


def test_missing_per_objective_metrics_pause_after_three_events():
    history = [
        {"step": step, "aggregate_reward": 5.0, "objectives": {"synteny": {"reward_mean": 0.1}}}
        for step in (10, 20, 30)
    ]

    result = evaluate_objective_history(history)

    assert result["decision"] == "pause_for_diagnosis"
    assert "missing_required_telemetry" in result["objectives"]["synteny"]["signals"]


def test_enabled_objective_with_no_measurements_starts_confirmation_window():
    history = [_event(step, reward=0.0, support=0.0, std=0.0, pass_rate=0.0) for step in (10, 20, 30)]

    result = evaluate_objective_history(history)

    assert result["decision"] == "continue"
    assert result["reason"] == "signal_pending_confirmation:1/8"
    assert result["objectives"]["protein_hit_count"]["status"] == "warning"
    assert "objective_unmeasured" in result["objectives"]["protein_hit_count"]["signals"]


def test_enabled_objective_with_no_measurements_pauses_after_confirmation_window():
    history = [_event(step * 10, reward=0.0, support=0.0, std=0.0, pass_rate=0.0) for step in range(1, 11)]

    result = evaluate_objective_history(history)

    assert result["decision"] == "pause_for_diagnosis"
    assert result["objectives"]["protein_hit_count"]["status"] == "suspicious"
    assert result["objectives"]["protein_hit_count"]["signal_streak"] == 8


def _masking_history(active_counts: list[int]) -> list[dict]:
    history = []
    names = ("a", "b", "c", "d")
    for index, active_count in enumerate(active_counts, start=1):
        active_names = set(names[:active_count])
        objectives = {}
        for name in names:
            objectives[name] = {
                "reward_mean": 0.5,
                "reward_std": 0.2 if name in active_names else 0.0,
                "nonzero_rate": 0.5,
                "support_rate": 1.0,
                "eligible_denominator": 96,
                "missing_rate": 0.0,
            }
        history.append({"step": index * 10, "aggregate_reward": 5.0, "objectives": objectives})
    return history


def test_loss_masking_starts_rebound_window_when_only_one_objective_remains_active():
    history = _masking_history([4, 2, 1])

    result = evaluate_objective_history(history)

    assert result["decision"] == "continue"
    assert result["reason"] == "signal_pending_confirmation:1/8"
    assert "objective_loss_masking" in result["global_signals"]
    assert result["global_signal_streaks"]["objective_loss_masking"] == 1
    assert result["active_objective_counts"] == [4, 2, 1]


def test_sustained_loss_masking_pauses_after_seventy_additional_steps():
    history = _masking_history([4, 4, 1, 1, 1, 1, 1, 1, 1, 1])

    result = evaluate_objective_history(history)

    assert result["decision"] == "pause_for_diagnosis"
    assert result["global_signal_streaks"]["objective_loss_masking"] == 8


def test_loss_activity_rebound_clears_pending_masking_signal():
    history = _masking_history([4, 4, 1, 1, 1, 1, 1, 1, 1, 4])

    result = evaluate_objective_history(history)

    assert result["decision"] == "continue"
    assert result["reason"] == "individual_objectives_and_support_are_stable"
    assert result["global_signals"] == []
    assert result["global_signal_streaks"]["objective_loss_masking"] == 0


def test_extract_validation_history_derives_only_emitted_gdpo_objectives(monkeypatch, tmp_path):
    points = {
        "validation/mean_reward": {10: (1.0, 0.5), 20: (2.0, 0.6)},
        "validation/num_sequences": {10: (1.0, 96.0)},
        "validation/gdpo/tropism_mean": {10: (1.0, 0.25)},
        "validation/gdpo/tropism_std": {10: (1.0, 0.1)},
        "validation/gdpo/tropism_nonzero_rate": {10: (1.0, 0.5)},
        "validation/phage_qc/tropism_measurement_available_rate": {10: (1.0, 0.75)},
        "validation/gdpo/mmseqs_cluster_diversity_mean": {10: (1.0, 0.4)},
        "validation/gdpo/mmseqs_cluster_diversity_std": {10: (1.0, 0.2)},
        "validation/gdpo/mmseqs_cluster_diversity_nonzero_rate": {10: (1.0, 0.6)},
        "validation/phage_qc/mmseqs_cluster_valid_for_clustering_mean": {10: (1.0, 0.6)},
        "validation/phage_qc/mmseqs_cluster_missing_from_output_mean": {10: (1.0, 0.4)},
        "validation/gdpo/gc_content_mean": {10: (1.0, 0.8)},
        "validation/gdpo/gc_content_std": {10: (1.0, 0.05)},
        "validation/gdpo/gc_content_nonzero_rate": {10: (1.0, 1.0)},
    }
    monkeypatch.setattr(objective_monitor, "_load_scalar_points", lambda _root: points)

    history = extract_validation_history(tmp_path)

    assert len(history) == 1
    objectives = history[0]["objectives"]
    assert set(objectives) == {"gc_content", "mmseqs_cluster_diversity", "tropism"}
    assert objectives["tropism"]["support_rate"] == 0.75
    assert objectives["tropism"]["missing_rate"] == 0.25
    assert objectives["mmseqs_cluster_diversity"]["support_rate"] == 0.6
    assert objectives["mmseqs_cluster_diversity"]["missing_from_output_rate"] == 0.4
    assert objectives["mmseqs_cluster_diversity"]["missing_rate"] == 0.4
    assert objectives["gc_content"]["support_rate"] == 1.0
    assert objectives["gc_content"]["missing_rate"] == 0.0


def test_extract_validation_history_uses_newest_task_scoped_namespace(monkeypatch, tmp_path):
    points = {
        "validation/rl-validation/mean_reward": {1: (1.0, 0.1)},
        "validation/rl-validation/num_sequences": {1: (1.0, 96.0)},
        "validation/rl-validation/gdpo/tropism_mean": {1: (1.0, 0.1)},
        "validation/rl-validation/gdpo/tropism_std": {1: (1.0, 0.1)},
        "validation/rl-validation/gdpo/tropism_nonzero_rate": {1: (1.0, 0.1)},
        "validation/phage_qc/mean_reward": {1: (10.0, 0.8)},
        "validation/phage_qc/num_sequences": {1: (10.0, 96.0)},
        "validation/phage_qc/gdpo/tropism_mean": {1: (10.0, 0.7)},
        "validation/phage_qc/gdpo/tropism_std": {1: (10.0, 0.2)},
        "validation/phage_qc/gdpo/tropism_nonzero_rate": {1: (10.0, 0.9)},
        "validation/phage_qc/tropism_measurement_available_rate": {1: (10.0, 0.75)},
        "validation/phage_qc/tropism_pass_rate": {1: (10.0, 0.5)},
    }
    monkeypatch.setattr(objective_monitor, "_load_scalar_points", lambda _root: points)

    history = extract_validation_history(tmp_path)

    assert len(history) == 1
    assert history[0]["aggregate_reward"] == 0.8
    assert history[0]["objectives"]["tropism"]["reward_mean"] == 0.7
    assert history[0]["objectives"]["tropism"]["support_rate"] == 0.75
    assert history[0]["objectives"]["tropism"]["hard_pass_rate"] == 0.5


def test_main_rejects_missing_tensorboard_root_before_writing(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["objective-monitor", "--tensorboard-root", str(tmp_path / "missing"), "--output", str(output)],
    )

    with pytest.raises(SystemExit):
        objective_monitor.main()

    assert not output.exists()
