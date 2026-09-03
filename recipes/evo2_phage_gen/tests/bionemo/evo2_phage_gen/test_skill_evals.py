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

"""Local-only discovery and preflight for the recipe's Agent Skills evals."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


RECIPE_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
SKILL_ROOT = RECIPE_ROOT / "skills"
EVAL_RUNNER = SKILL_ROOT / "bionemo-phage-design" / "scripts" / "run_skill_evals.py"
RUNNING_IN_CI = os.getenv("CI", "").strip().lower() not in {"", "0", "false", "no", "off"}
REQUIRED_CASE_FIELDS = {"id", "prompt", "expected_output", "assertions", "expected_skill", "expected_script"}


def test_all_skill_eval_files_have_required_case_fields() -> None:
    """Every checked-in eval file should expose the minimal portable case schema."""
    eval_files = sorted(SKILL_ROOT.glob("*/evals/evals.json"))

    assert eval_files
    for eval_file in eval_files:
        payload = json.loads(eval_file.read_text(encoding="utf-8"))
        assert payload["skill_name"] == eval_file.parents[1].name
        assert isinstance(payload["evals"], list) and payload["evals"]
        for case in payload["evals"]:
            assert REQUIRED_CASE_FIELDS <= case.keys(), f"{eval_file}: {case.get('id', '<missing id>')}"


def test_rl_objective_skill_requires_intermediate_reward_shaping() -> None:
    """Sparse terminal objectives should retain explicit, graded biological stepping stones."""
    skill = (SKILL_ROOT / "bionemo-phage-design-plan-rl-objectives" / "SKILL.md").read_text()

    assert "intermediate rewards" in skill
    assert "essential-gene completeness" in skill
    assert "reasonable synteny" in skill
    assert "host-range or bootability" in skill
    assert "partial credit" in skill
    assert "rather than dominate or substitute" not in skill


def test_rl_skill_scopes_safety_objectives() -> None:
    """Whole-genome RL keeps the three safety objectives while allowing narrow-scope exceptions."""
    skill = (SKILL_ROOT / "bionemo-phage-design-plan-rl-objectives" / "SKILL.md").read_text()

    assert "whole-genome designs" in skill
    assert "custom or adapted runs" in skill
    assert "AMR, toxin, and lysogeny" in skill
    assert "locus or module" in skill


def test_controller_treats_reward_invention_as_core() -> None:
    """Adapted designs should reuse sound reward machinery without treating it as a closed catalog."""
    controller = (SKILL_ROOT / "bionemo-phage-design" / "SKILL.md").read_text()
    planner = (SKILL_ROOT / "bionemo-phage-design-plan-rl-objectives" / "SKILL.md").read_text()

    for skill in (controller, planner):
        assert "core agentic capability" in skill
        assert "Prefer modifying tested rewards" in skill
        assert "beyond a faithful experiment rerun" in skill
        assert "novel reward functions" in skill
    assert "not a closed catalog" in controller
    assert "creatively invent" in planner


def test_rl_objective_skills_require_a_human_score_definition_artifact() -> None:
    """Planning and implementation should leave the resolved score contract in the run record."""
    plan_skill = (SKILL_ROOT / "bionemo-phage-design-plan-rl-objectives" / "SKILL.md").read_text()
    implement_skill = (SKILL_ROOT / "bionemo-phage-design-implement-rl-objectives" / "SKILL.md").read_text()

    for skill in (plan_skill, implement_skill):
        assert "artifacts/RL_SCORE_DEFINITIONS.md" in skill
        assert "zero-credit" in skill
        assert "full-credit" in skill
        assert "biological rationale" in skill
        assert "not a required stage of the fully scripted run" in skill
        assert "Current PhiX174 GDPO score definitions" in skill


def test_rl_operator_skill_documents_native_megatron_checkpoint_saving() -> None:
    """Megatron training and rollout should share the native MBridge checkpoint contract."""
    skill = (SKILL_ROOT / "bionemo-phage-design-operate-nemo-rl" / "SKILL.md").read_text()

    assert "native Megatron-Bridge" in skill
    assert "checkpointing.model_save_format: null" in skill
    assert "policy.dtensor_cfg.enabled: false" in skill


def test_phage_design_skills_default_new_runs_to_evo2_7b_1m() -> None:
    """New phage projects should start from the trained-further long-context 7B family."""
    controller = (SKILL_ROOT / "bionemo-phage-design" / "SKILL.md").read_text()
    sft = (SKILL_ROOT / "bionemo-phage-design-operate-mbridge-sft" / "SKILL.md").read_text()

    for skill in (controller, sft):
        assert "evo2/7b-1m:1.0" in skill
        assert "evo2_7b" in skill
        assert "new phage-design" in skill
        assert "mid-run" in skill


def test_controller_keeps_phix_rerun_orchestration_agent_directed() -> None:
    """The realized launcher is a useful reference, not a mandatory orchestration interface."""
    controller = " ".join((SKILL_ROOT / "bionemo-phage-design" / "SKILL.md").read_text().split())

    assert "reference implementation of the realized DAG" in controller
    assert "may run it directly, adapt or wrap it" in controller
    assert "stage subskills" in controller
    assert "let the task and execution environment determine the orchestration" in controller


def test_adapt_execution_documents_portable_container_path() -> None:
    """GPU workstations should use the tested container and native architecture assets."""
    guidance = (
        SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "references" / "infrastructure-guidance.md"
    ).read_text()

    for marker in (
        "NVIDIA PyTorch container",
        ".devcontainer/",
        "bind-mount",
        "./.ci_build.sh",
        ".ci_test_env.sh",
        "x86_64",
        "aarch64",
        "MMseqs2-GPU",
        "DIAMOND",
        "HMMER",
        "AMRFinderPlus",
        "biotite 0.41.2",
        "biotraj 1.2.2",
        "2026-08-21",
    ):
        assert marker in guidance


def test_rl_skills_keep_cross_stage_contracts() -> None:
    """Objective, prompt, scorer, and monitoring contracts should survive handoffs."""
    plan = (SKILL_ROOT / "bionemo-phage-design-plan-rl-objectives" / "SKILL.md").read_text()
    implement = (SKILL_ROOT / "bionemo-phage-design-implement-rl-objectives" / "SKILL.md").read_text()
    calibrate = (SKILL_ROOT / "bionemo-phage-design-calibrate-rl-sampling" / "SKILL.md").read_text()
    monitor = (
        SKILL_ROOT / "bionemo-phage-design-operate-nemo-rl" / "references" / "monitoring-guidance.md"
    ).read_text()
    operate = (SKILL_ROOT / "bionemo-phage-design-operate-nemo-rl" / "SKILL.md").read_text()

    assert "scientific baseline" in plan
    assert "planned, configured, and emitted" in plan
    assert "continue autonomously" in plan
    assert "last resort" in plan
    assert "adding well-supported shaping objectives is important" in plan
    assert "literature review or partial-run evidence" in plan
    assert "mixed valid and invalid" in implement
    assert "numeric reward columns" in implement
    assert "batch and row order" in implement
    assert "global rollout batch" in calibrate
    assert "regions intended to change" in calibrate
    assert "temperature, then top-k, then top-p" in calibrate
    assert "training rollouts" in monitor
    assert "fixed validation bank" in monitor
    assert "prompt composition" in monitor
    assert "emitted metric keys" in monitor
    assert "not an automatic stop or user wait" in operate
    assert "strongest scientifically defensible portfolio" in operate
    assert "next user update" in operate


def test_sft_operator_requires_checkpoint_boundary_stop_decisions() -> None:
    """SFT supervisors should act on sustained train/validation divergence."""
    operate = (SKILL_ROOT / "bionemo-phage-design-operate-mbridge-sft" / "SKILL.md").read_text()
    guidance = (
        SKILL_ROOT / "bionemo-phage-design-operate-mbridge-sft" / "references" / "training-guidance.md"
    ).read_text()
    execution = (SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "SKILL.md").read_text()

    assert "`max_steps` is a safety ceiling" in operate
    assert "`continue | one_more | stop`" in operate
    assert "three consecutive post-best" in guidance
    assert "approximately 1,000 optimizer steps" in guidance
    assert "one additional validation interval" in guidance
    assert "retention does not authorize continuation" in guidance
    assert "unconditional relaunch" in execution


def test_rl_operator_uses_a_noise_tolerant_continuation_contract() -> None:
    """RL should tolerate score noise while detecting failed measurement and sustained decline."""
    operate = (SKILL_ROOT / "bionemo-phage-design-operate-nemo-rl" / "SKILL.md").read_text()
    guidance = (
        SKILL_ROOT / "bionemo-phage-design-operate-nemo-rl" / "references" / "monitoring-guidance.md"
    ).read_text()

    assert "`max_num_steps` is a safety ceiling" in operate
    assert "`continue | diagnose | stop | restart`" in operate
    assert "Do not apply the SFT rule to RL" in guidance
    assert "approximately ten comparable validation events" in guidance
    assert "about 100 optimizer steps" in guidance
    assert "single sharp drop" in guidance
    assert "low safety scores" in guidance
    assert "failure to execute" in guidance


def test_early_stopping_eval_matrix_covers_sft_and_rl_patterns() -> None:
    """Skill evals should distinguish clean SFT divergence from noisy RL movement."""
    sft = json.loads((SKILL_ROOT / "bionemo-phage-design-operate-mbridge-sft" / "evals" / "evals.json").read_text())
    rl = json.loads((SKILL_ROOT / "bionemo-phage-design-operate-nemo-rl" / "evals" / "evals.json").read_text())
    sft_ids = {case["id"] for case in sft["evals"]}
    rl_ids = {case["id"] for case in rl["evals"]}

    assert {
        "bionemo-phage-design-operate-mbridge-sft-007-smooth-divergence",
        "bionemo-phage-design-operate-mbridge-sft-008-one-noisy-regression",
        "bionemo-phage-design-operate-mbridge-sft-009-short-plateau",
    } <= sft_ids
    assert {
        "bionemo-phage-design-operate-nemo-rl-013-noisy-recovery",
        "bionemo-phage-design-operate-nemo-rl-014-sustained-decline",
        "bionemo-phage-design-operate-nemo-rl-015-measurement-failure",
    } <= rl_ids


def test_rl_planning_preserves_editable_regions_and_strong_controls() -> None:
    """Objective planning should constrain prompt placement and require biological controls."""
    plan = (SKILL_ROOT / "bionemo-phage-design-plan-rl-objectives" / "SKILL.md").read_text()
    objective_guidance = (
        SKILL_ROOT / "bionemo-phage-design-plan-rl-objectives" / "references" / "objective-guidance.md"
    ).read_text()
    implement = (SKILL_ROOT / "bionemo-phage-design-implement-rl-objectives" / "SKILL.md").read_text()
    calibrate = (SKILL_ROOT / "bionemo-phage-design-calibrate-rl-sampling" / "SKILL.md").read_text()

    assert "prompt-exclusion intervals" in plan
    assert "known or expected high-scoring" in objective_guidance
    assert "known or expected low-scoring" in objective_guidance
    assert "biological positive and negative controls" in implement
    assert "prompt bases and fraction" in calibrate
    assert "any intended-to-change bases" in calibrate
    assert "shortest workable prompt" in calibrate
    assert "scale prompt length linearly" in calibrate
    assert "longer prompt" in calibrate and "rationale" in calibrate


def test_rl_prompt_and_control_evals_cover_the_new_contract() -> None:
    """Pressure cases should exercise circular prompts and meaningful scorer controls."""
    plan = json.loads((SKILL_ROOT / "bionemo-phage-design-plan-rl-objectives" / "evals" / "evals.json").read_text())
    implement = json.loads(
        (SKILL_ROOT / "bionemo-phage-design-implement-rl-objectives" / "evals" / "evals.json").read_text()
    )
    calibrate = json.loads(
        (SKILL_ROOT / "bionemo-phage-design-calibrate-rl-sampling" / "evals" / "evals.json").read_text()
    )

    assert "bionemo-phage-design-plan-rl-objectives-016-prompt-and-control-handoff" in {
        case["id"] for case in plan["evals"]
    }
    assert "bionemo-phage-design-implement-rl-objectives-007-biological-controls" in {
        case["id"] for case in implement["evals"]
    }
    assert "bionemo-phage-design-calibrate-rl-sampling-008-circular-editable-regions" in {
        case["id"] for case in calibrate["evals"]
    }


def test_infra_guidance_covers_coherent_memory() -> None:
    """Portable infrastructure advice should include ARM builds and coherent-memory nodes."""
    infrastructure = (
        SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "references" / "infrastructure-guidance.md"
    ).read_text()
    compute = (SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "references" / "compute-guidance.md").read_text()

    for marker in (
        "architecture-specific compiled extensions",
        "runtime UID",
        "coherent-memory",
        "memory-only NUMA",
        "checkpoint page cache",
        "second save",
        "exact restart",
        "image build context",
    ):
        assert marker in infrastructure
    assert "nested subprocess" in compute
    assert "all visible CPUs" in compute


@pytest.mark.skipif(RUNNING_IN_CI, reason="Skill eval planning is intentionally local-only.")
def test_all_skill_evals_are_discovered_and_plannable(tmp_path: Path) -> None:
    """Run every skill eval through the no-model-call planning path."""
    results_dir = tmp_path / "skill-evals"
    completed = subprocess.run(
        [
            sys.executable,
            str(EVAL_RUNNER),
            "--skill-root",
            str(SKILL_ROOT),
            "--repo-root",
            str(REPOSITORY_ROOT),
            "--recipe-root",
            str(RECIPE_ROOT),
            "--dry-run",
            "--all",
            "--results-dir",
            str(results_dir),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads((results_dir / "run-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "planned"
    assert status["case_count"] > 0
    planned = [line for line in completed.stdout.splitlines() if line.endswith(": PLANNED")]
    assert len(planned) == status["case_count"]
