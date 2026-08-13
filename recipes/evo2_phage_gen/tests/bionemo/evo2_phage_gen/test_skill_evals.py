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
