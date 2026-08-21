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
