"""Fail-closed monitoring for individual GDPO objectives and their biological support."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


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
) -> tuple[list[str], list[str]]:
    signals: list[str] = []
    missing_fields = sorted(
        {
            field
            for row in window
            for field in REQUIRED_FIELDS
            if field not in row or not _finite_number(row[field])
        }
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
        and float(last["hard_pass_rate"]) + 0.05 < float(first["hard_pass_rate"])
    ):
        signals.append("reward_hard_pass_divergence")

    rewards = [float(row["reward_mean"]) for row in window]
    deltas = [right - left for left, right in zip(rewards, rewards[1:])]
    signs = [1 if delta > 0 else -1 if delta < 0 else 0 for delta in deltas]
    sign_changes = sum(
        left != 0 and right != 0 and left != right
        for left, right in zip(signs, signs[1:])
    )
    if max(rewards) - min(rewards) >= 0.50 and sign_changes >= 1:
        signals.append("objective_instability")
    return signals, missing_fields


def evaluate_objective_history(
    events: Sequence[Mapping[str, Any]],
    *,
    minimum_events: int = 3,
    audit_confirmation_events: int = 8,
    reward_gain_threshold: float = 0.15,
    support_drop_threshold: float = 0.15,
    denominator_drop_fraction: float = 0.20,
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
    objective_names = sorted(
        {
            str(name)
            for event in ordered
            for name in event.get("objectives", {})
        }
    )
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
        )
        signal_streak = 0
        for end in range(minimum_events, len(series) + 1):
            candidate_signals, _ = _objective_window_signals(
                series[end - minimum_events : end],
                minimum_events=minimum_events,
                reward_gain_threshold=reward_gain_threshold,
                support_drop_threshold=support_drop_threshold,
                denominator_drop_fraction=denominator_drop_fraction,
            )
            signal_streak = signal_streak + 1 if pause_signals.intersection(candidate_signals) else 0
        max_signal_streak = max(max_signal_streak, signal_streak)
        has_signal = bool(pause_signals.intersection(signals))
        immediate_telemetry_failure = "missing_required_telemetry" in signals
        confirmed = immediate_telemetry_failure or signal_streak >= audit_confirmation_events
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

    active_counts: list[int] = []
    for event in ordered:
        active_counts.append(
            sum(
                _finite_number(values.get("reward_std")) and float(values["reward_std"]) > activity_epsilon
                for values in event.get("objectives", {}).values()
            )
        )
    masking_flags: list[bool] = []
    peak_active = 0
    for active_count in active_counts:
        masking_flags.append(
            peak_active >= 3 and active_count <= max(1, peak_active // 4)
        )
        peak_active = max(peak_active, active_count)
    masking_streak = 0
    for masked in masking_flags:
        masking_streak = masking_streak + 1 if masked else 0
    max_signal_streak = max(max_signal_streak, masking_streak)
    global_signals = ["objective_loss_masking"] if masking_flags and masking_flags[-1] else []
    if masking_streak >= audit_confirmation_events:
        confirmed_suspicious = True
    elif global_signals:
        pending_suspicious = True

    if len(ordered) < minimum_events:
        decision = "continue"
        reason = f"insufficient_comparable_events:{len(ordered)}/{minimum_events}"
    elif confirmed_suspicious:
        decision = "pause_for_audit"
        reason = "per_objective_cheat_mode_or_instability_signal"
    elif pending_suspicious:
        decision = "continue"
        reason = f"audit_signal_pending_confirmation:{max_signal_streak}/{audit_confirmation_events}"
    else:
        decision = "continue"
        reason = "individual_objectives_and_support_are_stable"

    return {
        "schema_version": 1,
        "decision": decision,
        "reason": reason,
        "latest_complete_step": latest_step,
        "comparable_event_count": len(ordered),
        "audit_confirmation_events": audit_confirmation_events,
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


def extract_validation_history(
    tensorboard_root: Path,
    objective_names: Iterable[str] = DEFAULT_OBJECTIVES,
) -> list[dict[str, Any]]:
    """Normalize complete validation events from one or more TensorBoard files."""

    points = _load_scalar_points(tensorboard_root)
    reward_tag = "validation/mean_reward"
    steps = sorted(points.get(reward_tag, {}))
    events: list[dict[str, Any]] = []
    for step in steps:
        denominator = _scalar(points, "validation/num_sequences", step)
        if denominator is None:
            continue
        objectives: dict[str, Any] = {}
        for name in objective_names:
            prefix = f"validation/gdpo/{name}"
            values: dict[str, Any] = {
                "reward_mean": _scalar(points, f"{prefix}_mean", step),
                "reward_std": _scalar(points, f"{prefix}_std", step),
                "nonzero_rate": _scalar(points, f"{prefix}_nonzero_rate", step),
                "eligible_denominator": denominator,
                "hard_pass_rate": _scalar(points, f"validation/phage_qc/{name}_pass_rate", step),
            }
            support_prefix = EXTERNAL_SUPPORT_PREFIX.get(name)
            if support_prefix:
                support = _scalar(
                    points,
                    f"validation/phage_qc/{support_prefix}_measurement_available_rate",
                    step,
                )
                values["support_rate"] = support
                values["measured_count"] = _scalar(
                    points,
                    f"validation/phage_qc/{support_prefix}_n_measured",
                    step,
                )
                values["stage_reached_rate"] = _scalar(
                    points,
                    f"validation/phage_qc/{support_prefix}_stage_reached_rate",
                    step,
                )
                values["missing_artifact_count"] = _scalar(
                    points,
                    f"validation/phage_qc/{support_prefix}_missing_artifact_count",
                    step,
                )
            elif name == "mmseqs_cluster_diversity":
                support = _scalar(
                    points,
                    "validation/phage_qc/mmseqs_cluster_valid_for_clustering_mean",
                    step,
                )
                values["support_rate"] = support
                values["missing_from_output_rate"] = _scalar(
                    points,
                    "validation/phage_qc/mmseqs_cluster_missing_from_output_mean",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorboard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-output", type=Path)
    parser.add_argument("--minimum-events", type=int, default=3)
    parser.add_argument("--audit-confirmation-events", type=int, default=8)
    args = parser.parse_args()

    history = extract_validation_history(args.tensorboard_root)
    report = evaluate_objective_history(
        history,
        minimum_events=args.minimum_events,
        audit_confirmation_events=args.audit_confirmation_events,
    )
    report["tensorboard_root"] = str(args.tensorboard_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.history_output:
        args.history_output.parent.mkdir(parents=True, exist_ok=True)
        args.history_output.write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
