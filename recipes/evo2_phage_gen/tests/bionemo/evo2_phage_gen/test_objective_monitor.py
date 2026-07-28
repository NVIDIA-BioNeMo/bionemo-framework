from __future__ import annotations

from bionemo.evo2_phage_gen.objective_monitor import evaluate_objective_history


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
    assert result["reason"] == "audit_signal_pending_confirmation:1/8"
    assert result["latest_complete_step"] == 30
    assert result["objectives"]["protein_hit_count"]["status"] == "warning"
    assert result["objectives"]["protein_hit_count"]["signal_streak"] == 1
    assert "reward_support_divergence" in result["objectives"]["protein_hit_count"]["signals"]


def test_sustained_reward_support_divergence_pauses_after_rebound_window():
    history = [
        _event(step * 10, 0.10 + 0.08 * step, 0.90 - 0.08 * step)
        for step in range(10)
    ]

    result = evaluate_objective_history(history)

    assert result["decision"] == "pause_for_audit"
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


def test_missing_per_objective_telemetry_fails_closed_after_three_events():
    history = [
        {"step": step, "aggregate_reward": 5.0, "objectives": {"synteny": {"reward_mean": 0.1}}}
        for step in (10, 20, 30)
    ]

    result = evaluate_objective_history(history)

    assert result["decision"] == "pause_for_audit"
    assert "missing_required_telemetry" in result["objectives"]["synteny"]["signals"]


def test_enabled_objective_with_no_measurements_fails_closed_after_three_events():
    history = [
        _event(step, reward=0.0, support=0.0, std=0.0, pass_rate=0.0)
        for step in (10, 20, 30)
    ]

    result = evaluate_objective_history(history)

    assert result["decision"] == "pause_for_audit"
    assert result["objectives"]["protein_hit_count"]["status"] == "suspicious"
    assert "objective_unmeasured" in result["objectives"]["protein_hit_count"]["signals"]


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
    assert result["reason"] == "audit_signal_pending_confirmation:1/8"
    assert "objective_loss_masking" in result["global_signals"]
    assert result["global_signal_streaks"]["objective_loss_masking"] == 1
    assert result["active_objective_counts"] == [4, 2, 1]


def test_sustained_loss_masking_pauses_after_seventy_additional_steps():
    history = _masking_history([4, 4, 1, 1, 1, 1, 1, 1, 1, 1])

    result = evaluate_objective_history(history)

    assert result["decision"] == "pause_for_audit"
    assert result["global_signal_streaks"]["objective_loss_masking"] == 8


def test_loss_activity_rebound_clears_pending_masking_signal():
    history = _masking_history([4, 4, 1, 1, 1, 1, 1, 1, 1, 4])

    result = evaluate_objective_history(history)

    assert result["decision"] == "continue"
    assert result["reason"] == "individual_objectives_and_support_are_stable"
    assert result["global_signals"] == []
    assert result["global_signal_streaks"]["objective_loss_masking"] == 0
