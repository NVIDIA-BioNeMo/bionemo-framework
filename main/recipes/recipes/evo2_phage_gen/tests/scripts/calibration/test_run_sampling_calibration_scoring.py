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

import os
import subprocess
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = RECIPE_ROOT / "scripts" / "calibration" / "run_sampling_calibration_scoring.sh"


def test_scoring_script_creates_root_before_generation_validation_redirect(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text("#!/usr/bin/env bash\nprintf '{\"validated\": true}\\n'\nexit 42\n")
    python.chmod(0o755)
    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    (generation_root / "SUCCEEDED").touch()
    score_root = tmp_path / "not-created-yet" / "scoring"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "SOURCE_ENV": "0",
        "CALIBRATION_ROOT": str(tmp_path / "calibration"),
        "GENERATION_ROOT": str(generation_root),
        "SCORE_ROOT": str(score_root),
        "ARC_CONFIG": str(tmp_path / "arc.yaml"),
        "PIPELINE_SCRIPT": str(tmp_path / "pipeline.py"),
        "TOOL_BIN_DIR": str(tmp_path / "tools"),
        "REFERENCE_FASTA": str(tmp_path / "reference.fna"),
        "SFT_FASTA": str(tmp_path / "sft.fna"),
    }

    completed = subprocess.run(["bash", str(SCRIPT)], cwd=RECIPE_ROOT, env=env, check=False, timeout=120)

    assert completed.returncode == 42
    assert (score_root / "generation-validation.json").read_text() == '{"validated": true}\n'
    assert not Path(f"{score_root}.generation-validation.json").exists()


def test_scoring_workers_use_dedicated_input_descriptor() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "read -r -u 3 cell_index" in script
    assert 'done 3< "${GENERATION_ROOT}/cells.tsv"' in script
