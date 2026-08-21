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

"""Monitor individual GDPO objectives and pause when their biological evidence is inadequate."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml


DEFAULT_OBJECTIVES = (
    "valid_nt_chars",
    "genome_length",
    "gc_content",
    "nt_homopolymer",
    "dustmask_end",
    "nucleotide_pass",
    "protein_hit_count",
    "tropism",
    "required_genes",
    "synteny",
    "average_protein_identity",
    "mmseqs_cluster_diversity",
)

REQUIRED_FIELDS = (
    "reward_mean",
    "reward_std",
    "nonzero_rate",
    "support_rate",
    "eligible_denominator",
    "missing_rate",
)

EXTERNAL_SUPPORT_PREFIX = {
    "protein_hit_count": "protein_database_hit_count",
    "tropism": "tropism",
    "required_genes": "required_genes",
    "synteny": "synteny",
    "average_protein_identity": "average_protein_identity",
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _change(first: Mapping[str, Any], last: Mapping[str, Any], key: str) -> float:
    return float(last[key]) - float(first[key])


def _objective_window_signals(
    window: Sequence[Mapping[str, Any]],
    *,
    minimum_events: int,
    reward_gain_threshold: float,
    support_drop_threshold: float,
    denominator_drop_fraction: float,
    hard_pass_drop_threshold: float,
    objective_reward_range_threshold: float,
    minimum_reward_sign_changes: int,
) -> tuple[list[str], list[str]]:
    signals: list[str] = []
    missing_fields = sorted(
        {field for row in window for field in REQUIRED_FIELDS if field not in row or not _finite_number(row[field])}
    )
    if len(window) < minimum_events:
        return signals, missing_fields
    if missing_fields:
        return ["missing_required_telemetry"], missing_fields

    if all(float(row["support_rate"]) <= 1e-6 for row in window):
        signals.append("objective_unmeasured")

    first, last = window[0], window[-1]
    reward_gain = _change(first, last, "reward_mean")
    support_drop = -_change(first, last, "support_rate")
    initial_denominator = float(first["eligible_denominator"])
    denominator_drop = (
        0.0
        if initial_denominator <= 0
        else max(0.0, (initial_denominator - float(last["eligible_denominator"])) / initial_denominator)
    )
    if reward_gain >= reward_gain_threshold and (
        support_drop >= support_drop_threshold or denominator_drop >= denominator_drop_fraction
    ):
        signals.append("reward_support_divergence")
    if (
        reward_gain >= reward_gain_threshold
        and _finite_number(first.get("hard_pass_rate"))
        and _finite_number(last.get("hard_pass_rate"))
        and float(last["hard_pass_rate"]) + hard_pass_drop_threshold < float(first["hard_pass_rate"])
    ):
        signals.append("reward_hard_pass_divergence")

    rewards = [float(row["reward_mean"]) for row in window]
    deltas = [right - left for left, right in pairwise(rewards)]
    signs = [1 if delta > 0 else -1 if delta < 0 else 0 for delta in deltas]
    sign_changes = sum(left != 0 and right != 0 and left != right for left, right in pairwise(signs))
    if max(rewards) - min(rewards) >= objective_reward_range_threshold and sign_changes >= minimum_reward_sign_changes:
        signals.append("objective_instability")
    return signals, missing_fields


def evaluate_objective_history(
    events: Sequence[Mapping[str, Any]],
    *,
    minimum_events: int = 3,
    diagnosis_confirmation_events: int = 8,
    reward_gain_threshold: float = 0.15,
    support_drop_threshold: float = 0.15,
    denominator_drop_fraction: float = 0.20,
    hard_pass_drop_threshold: float = 0.05,
    objective_reward_range_threshold: float = 0.50,
    minimum_reward_sign_changes: int = 1,
    activity_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Diagnose objective exploitation from comparable checkpoint-validation events.

    GDPO exposes one combined policy-gradient loss, so an objective's raw score
    variance is its effective contribution-activity proxy: a constant objective
    contributes no centered advantage. Reward movement is always compared with
    support, denominator, missingness, and hard-pass telemetry where available.
    """
    ordered = sorted(events, key=lambda event: int(event["step"]))
    latest_step = int(ordered[-1]["step"]) if ordered else None
    objective_names = sorted({str(name) for event in ordered for name in event.get("objectives", {})})
    findings: dict[str, dict[str, Any]] = {}
    confirmed_suspicious = False
    pending_suspicious = False
    max_signal_streak = 0
    pause_signals = {
        "missing_required_telemetry",
        "objective_unmeasured",
        "reward_support_divergence",
        "reward_hard_pass_divergence",
        "objective_instability",
    }

    for name in objective_names:
        series = [event.get("objectives", {}).get(name, {}) for event in ordered]
        window = series[-minimum_events:]
        signals, missing_fields = _objective_window_signals(
            window,
            minimum_events=minimum_events,
            reward_gain_threshold=reward_gain_threshold,
            support_drop_threshold=support_drop_threshold,
            denominator_drop_fraction=denominator_drop_fraction,
            hard_pass_drop_threshold=hard_pass_drop_threshold,
            objective_reward_range_threshold=objective_reward_range_threshold,
            minimum_reward_sign_changes=minimum_reward_sign_changes,
        )
        signal_streak = 0
        for end in range(minimum_events, len(series) + 1):
            candidate_signals, _ = _objective_window_signals(
                series[end - minimum_events : end],
                minimum_events=minimum_events,
                reward_gain_threshold=reward_gain_threshold,
                support_drop_threshold=support_drop_threshold,
                denominator_drop_fraction=denominator_drop_fraction,
                hard_pass_drop_threshold=hard_pass_drop_threshold,
                objective_reward_range_threshold=objective_reward_range_threshold,
                minimum_reward_sign_changes=minimum_reward_sign_changes,
            )
            signal_streak = signal_streak + 1 if pause_signals.intersection(candidate_signals) else 0
        max_signal_streak = max(max_signal_streak, signal_streak)
        has_signal = bool(pause_signals.intersection(signals))
        immediate_telemetry_failure = "missing_required_telemetry" in signals
        confirmed = immediate_telemetry_failure or signal_streak >= diagnosis_confirmation_events
        if confirmed:
            status = "suspicious"
            confirmed_suspicious = True
        elif has_signal:
            status = "warning"
            pending_suspicious = True
        else:
            status = "healthy"
        latest = series[-1] if series else {}
        findings[name] = {
            "status": status,
            "signals": signals,
            "signal_streak": signal_streak,
            "missing_fields": missing_fields,
            "latest": dict(latest),
            "effective_loss_contribution": (
                "active"
                if _finite_number(latest.get("reward_std")) and float(latest["reward_std"]) > activity_epsilon
                else "inactive_or_unobservable"
            ),
        }

    active_counts = [
        sum(
            _finite_number(values.get("reward_std")) and float(values["reward_std"]) > activity_epsilon
            for values in event.get("objectives", {}).values()
        )
        for event in ordered
    ]
    masking_flags: list[bool] = []
    peak_active = 0
    for active_count in active_counts:
        masking_flags.append(peak_active >= 3 and active_count <= max(1, peak_active // 4))
        peak_active = max(peak_active, active_count)
    masking_streak = 0
    for masked in masking_flags:
        masking_streak = masking_streak + 1 if masked else 0
    max_signal_streak = max(max_signal_streak, masking_streak)
    global_signals = ["objective_loss_masking"] if masking_flags and masking_flags[-1] else []
    if masking_streak >= diagnosis_confirmation_events:
        confirmed_suspicious = True
    elif global_signals:
        pending_suspicious = True

    if len(ordered) < minimum_events:
        decision = "continue"
        reason = f"insufficient_comparable_events:{len(ordered)}/{minimum_events}"
    elif confirmed_suspicious:
        decision = "pause_for_diagnosis"
        reason = "per_objective_cheat_mode_or_instability_signal"
    elif pending_suspicious:
        decision = "continue"
        reason = f"signal_pending_confirmation:{max_signal_streak}/{diagnosis_confirmation_events}"
    else:
        decision = "continue"
        reason = "individual_objectives_and_support_are_stable"

    return {
        "schema_version": 1,
        "decision": decision,
        "reason": reason,
        "latest_complete_step": latest_step,
        "comparable_event_count": len(ordered),
        "diagnosis_confirmation_events": diagnosis_confirmation_events,
        "required_fields": list(REQUIRED_FIELDS),
        "objectives": findings,
        "active_objective_counts": active_counts,
        "global_signals": global_signals,
        "global_signal_streaks": {"objective_loss_masking": masking_streak},
    }


def _load_scalar_points(tensorboard_root: Path) -> dict[str, dict[int, tuple[float, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:  # pragma: no cover - runtime dependency gate
        raise RuntimeError("TensorBoard is required for objective monitoring") from error

    points: dict[str, dict[int, tuple[float, float]]] = {}
    for event_file in sorted(tensorboard_root.rglob("events.out.tfevents*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for scalar in accumulator.Scalars(tag):
                previous = points.setdefault(tag, {}).get(int(scalar.step))
                candidate = (float(scalar.wall_time), float(scalar.value))
                if previous is None or candidate[0] >= previous[0]:
                    points[tag][int(scalar.step)] = candidate
    return points


def _scalar(points: Mapping[str, Mapping[int, tuple[float, float]]], tag: str, step: int) -> float | None:
    record = points.get(tag, {}).get(step)
    return None if record is None else float(record[1])


def _validation_prefix(points: Mapping[str, Mapping[int, tuple[float, float]]]) -> str:
    """Select the newest validation namespace with reward and denominator events."""
    candidates: list[tuple[float, str]] = []
    suffix = "/mean_reward"
    for tag, reward_points in points.items():
        if not tag.startswith("validation/") or not tag.endswith(suffix):
            continue
        prefix = tag[: -len(suffix)]
        denominator_points = points.get(f"{prefix}/num_sequences", {})
        shared_steps = reward_points.keys() & denominator_points.keys()
        if shared_steps:
            latest_wall_time = max(reward_points[step][0] for step in shared_steps)
            candidates.append((latest_wall_time, prefix))
    if candidates:
        return max(candidates)[1]
    return "validation"


def _emitted_objective_names(points: Mapping[str, object], validation_prefix: str = "validation") -> tuple[str, ...]:
    prefix = f"{validation_prefix}/gdpo/"
    suffixes = ("_mean", "_std", "_min", "_max", "_nonzero_rate")
    names = set()
    for tag in points:
        if not tag.startswith(prefix):
            continue
        metric_name = tag[len(prefix) :]
        for suffix in suffixes:
            if metric_name.endswith(suffix):
                names.add(metric_name[: -len(suffix)])
                break
    return tuple(sorted(name for name in names if name))


def _phage_scalar(
    points: Mapping[str, Mapping[int, tuple[float, float]]],
    validation_prefix: str,
    metric_name: str,
    step: int,
) -> float | None:
    """Read flattened or legacy nested phage telemetry from one validation namespace."""
    for tag in (f"{validation_prefix}/{metric_name}", f"{validation_prefix}/phage_qc/{metric_name}"):
        value = _scalar(points, tag, step)
        if value is not None:
            return value
    return None


def _configured_objective_names(path: Path) -> tuple[str, ...]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path}: resolved config must be a mapping")
    env = loaded.get("env")
    phage_qc = env.get("phage_qc") if isinstance(env, Mapping) else None
    objectives = phage_qc.get("gdpo_objectives") if isinstance(phage_qc, Mapping) else None
    if not isinstance(objectives, list) or not objectives:
        raise ValueError(f"{path}: missing resolved env.phage_qc.gdpo_objectives")
    names = tuple(
        str(objective.get("name"))
        for objective in objectives
        if isinstance(objective, Mapping) and isinstance(objective.get("name"), str) and objective["name"]
    )
    if len(names) != len(objectives) or len(set(names)) != len(names):
        raise ValueError(f"{path}: GDPO objective names must be non-empty and unique")
    return names


def extract_validation_history(
    tensorboard_root: Path,
    objective_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize complete validation events from one or more TensorBoard files."""
    points = _load_scalar_points(tensorboard_root)
    validation_prefix = _validation_prefix(points)
    selected_objectives = (
        tuple(objective_names) if objective_names is not None else _emitted_objective_names(points, validation_prefix)
    )
    reward_tag = f"{validation_prefix}/mean_reward"
    steps = sorted(points.get(reward_tag, {}))
    events: list[dict[str, Any]] = []
    for step in steps:
        denominator = _scalar(points, f"{validation_prefix}/num_sequences", step)
        if denominator is None:
            continue
        objectives: dict[str, Any] = {}
        for name in selected_objectives:
            prefix = f"{validation_prefix}/gdpo/{name}"
            values: dict[str, Any] = {
                "reward_mean": _scalar(points, f"{prefix}_mean", step),
                "reward_std": _scalar(points, f"{prefix}_std", step),
                "nonzero_rate": _scalar(points, f"{prefix}_nonzero_rate", step),
                "eligible_denominator": denominator,
                "hard_pass_rate": _phage_scalar(points, validation_prefix, f"{name}_pass_rate", step),
            }
            support_prefix = EXTERNAL_SUPPORT_PREFIX.get(name)
            if support_prefix:
                support = _phage_scalar(
                    points,
                    validation_prefix,
                    f"{support_prefix}_measurement_available_rate",
                    step,
                )
                values["support_rate"] = support
                values["measured_count"] = _phage_scalar(
                    points,
                    validation_prefix,
                    f"{support_prefix}_n_measured",
                    step,
                )
                values["stage_reached_rate"] = _phage_scalar(
                    points,
                    validation_prefix,
                    f"{support_prefix}_stage_reached_rate",
                    step,
                )
                values["missing_artifact_count"] = _phage_scalar(
                    points,
                    validation_prefix,
                    f"{support_prefix}_missing_artifact_count",
                    step,
                )
            elif name == "mmseqs_cluster_diversity":
                support = _phage_scalar(
                    points,
                    validation_prefix,
                    "mmseqs_cluster_valid_for_clustering_mean",
                    step,
                )
                values["support_rate"] = support
                values["missing_from_output_rate"] = _phage_scalar(
                    points,
                    validation_prefix,
                    "mmseqs_cluster_missing_from_output_mean",
                    step,
                )
            else:
                support = 1.0
                values["support_rate"] = support
            values["missing_rate"] = None if support is None else 1.0 - float(support)
            objectives[name] = values
        events.append(
            {
                "step": step,
                "aggregate_reward": _scalar(points, reward_tag, step),
                "objectives": objectives,
            }
        )
    return events


def _positive_int(value: str) -> int:
    """Parse a positive integer for command-line event thresholds."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    """Evaluate objective history extracted from a TensorBoard run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorboard-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Resolved GDPO config used to select emitted objectives")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-output", type=Path)
    parser.add_argument("--minimum-events", type=_positive_int, default=3)
    parser.add_argument("--diagnosis-confirmation-events", type=_positive_int, default=8)
    args = parser.parse_args()

    if not args.tensorboard_root.is_dir():
        parser.error(f"tensorboard root is not a directory: {args.tensorboard_root}")
    try:
        objective_names = _configured_objective_names(args.config) if args.config is not None else None
        history = extract_validation_history(args.tensorboard_root, objective_names)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if not history:
        parser.error(f"no usable validation events under {args.tensorboard_root}")
    report = evaluate_objective_history(
        history,
        minimum_events=args.minimum_events,
        diagnosis_confirmation_events=args.diagnosis_confirmation_events,
    )
    report["tensorboard_root"] = str(args.tensorboard_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.history_output:
        args.history_output.parent.mkdir(parents=True, exist_ok=True)
        args.history_output.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
