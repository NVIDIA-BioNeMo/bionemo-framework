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
import os
import shlex
import subprocess
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = RECIPE_ROOT / "examples/phix174_8xh100.sh"


def test_dry_run(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True, timeout=10)
    result_root = tmp_path / "result"
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--result-root", str(result_root)],
        cwd=RECIPE_ROOT,
        env={**os.environ, "API_KEY": "do-not-record", "NVIDIA_API_KEY": "also-do-not-record"},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert (result_root / "stage-plan.txt").read_text().splitlines() == [
        "00 prepare inputs/tools/controls",
        "10 safety-screen and prepare SFT",
        "20 train/select/evaluate SFT",
        "30 calibrate sampling",
        "40 pilot/train/monitor/select GDPO",
        "50 generate, SFT-score, and screen 1,000 genomes",
    ]
    settings = json.loads((result_root / "settings.json").read_text())
    assert settings == {
        "gpu_count": 8,
        "gpu_type": "H100 80GB",
        "whole_genome": True,
        "safety_screen": "current configured databases",
        "final_generation_count": 1000,
    }
    log = (result_root / "RUNLOG.md").read_text()
    for command in (
        "evo2_phage_prepare_external_assets",
        "evo2_phage_sequence_safety",
        "evo2_phage_prepare_sft_split",
        "train_evo2",
        "run_sampling_calibration_scoring.sh",
        "evo2_phage_run_gdpo",
        "predict_evo2",
        "collect-sft-likelihood",
        "genome_design_filtering_pipeline.py",
        "finalize-rollout",
    ):
        assert command in log
    assert "do-not-record" not in log
    assert "monitor: external asset preparation" in log
    assert "--prepare-phrogs-consensus-database" in log
    assert "--download-phrogs-sequence-database" not in log
    mmseqs_dir = "data/external/mmseqs/NC_001422_1_Gprotein"
    mkdir_command = f"command: mkdir -p {mmseqs_dir}"
    createdb_command = "command: mmseqs createdb"
    assert mkdir_command in log
    assert log.index(mkdir_command) < log.index(createdb_command)

    control_commands = [
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: evo2_phage_sequence_safety scan" in line and "reference-controls" in line
    ]
    assert len(control_commands) == 6
    for command in control_commands:
        evidence = json.loads(command[command.index("--host-evidence-json") + 1])
        assert evidence["source"] == "NCBI Nucleotide"
        assert evidence["replication_host_domains"] == ["BACTERIA"]
        assert evidence["confirmed"] is True

    safety_commands = [
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: evo2_phage_sequence_safety " in line
    ]
    assert safety_commands
    assert all("--overwrite" in command for command in safety_commands)
