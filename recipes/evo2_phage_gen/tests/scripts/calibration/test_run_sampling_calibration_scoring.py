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
        "SAFETY_ASSET_MANIFEST": str(tmp_path / "asset-manifest.yaml"),
        "SAFETY_POLICY": str(tmp_path / "safety-policy.yaml"),
        "SAFETY_HOST_DOMAIN": "BACTERIA",
        "SAFETY_HOST_EVIDENCE_JSON": (
            '{"source":"test","source_version":"v1","replication_host_domains":["BACTERIA"],"confirmed":true}'
        ),
    }

    completed = subprocess.run(["bash", str(SCRIPT)], cwd=RECIPE_ROOT, env=env, check=False, timeout=120)

    assert completed.returncode == 42
    assert (score_root / "generation-validation.json").read_text() == '{"validated": true}\n'
    assert not Path(f"{score_root}.generation-validation.json").exists()


def test_scoring_worker_passes_online_safety_context(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "python-calls.jsonl"
    python = fake_bin / "python"
    python.write_text(
        """#!/usr/bin/python3
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as handle:
    handle.write(json.dumps(args) + "\\n")
for option in ("--output-csv", "--metrics-csv"):
    if option in args:
        output = Path(args[args.index(option) + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("id_prompt\\nx\\n")
if "validate-all" in args:
    print('{"validated": true}')
"""
    )
    python.chmod(0o755)
    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    (generation_root / "SUCCEEDED").touch()
    (generation_root / "cells.tsv").write_text(
        "index\tcell\tprefix\ttemperature\tprompt_file\tgeneration_jsonl\tprompt_anchor\tprompt_anchor_start\n"
        f"0\tprefix16_temp1.0\t16\t1.0\tprompt.jsonl\t{generation_root / 'cell.jsonl'}\torigin\t1\n"
    )
    safety_evidence = (
        '{"source":"test","source_version":"v1","replication_host_domains":["BACTERIA"],"confirmed":true}'
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CALLS": str(calls),
        "SOURCE_ENV": "0",
        "CALIBRATION_ROOT": str(tmp_path / "calibration"),
        "GENERATION_ROOT": str(generation_root),
        "ARC_CONFIG": str(tmp_path / "arc.yaml"),
        "PIPELINE_SCRIPT": str(tmp_path / "pipeline.py"),
        "TOOL_BIN_DIR": str(tmp_path / "tools"),
        "REFERENCE_FASTA": str(tmp_path / "reference.fna"),
        "SFT_FASTA": str(tmp_path / "sft.fna"),
        "SAFETY_ASSET_MANIFEST": str(tmp_path / "asset-manifest.yaml"),
        "SAFETY_POLICY": str(tmp_path / "safety-policy.yaml"),
        "SAFETY_HOST_DOMAIN": "BACTERIA",
        "SAFETY_HOST_EVIDENCE_JSON": safety_evidence,
        "EXPECTED_RECORDS": "1",
        "WORKERS": "1",
        "MAX_RETRIES": "0",
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT)], cwd=RECIPE_ROOT, env=env, check=False, capture_output=True, text=True, timeout=120
    )

    assert completed.returncode == 0, completed.stderr
    invocations = [json.loads(line) for line in calls.read_text().splitlines()]
    score = next(args for args in invocations if "score-cell" in args)
    expected = {
        "--safety-asset-manifest": str(tmp_path / "asset-manifest.yaml"),
        "--safety-policy": str(tmp_path / "safety-policy.yaml"),
        "--safety-host-domain": "BACTERIA",
        "--safety-host-evidence-json": safety_evidence,
    }
    for option, value in expected.items():
        assert score[score.index(option) + 1] == value


def test_scoring_workers_use_dedicated_input_descriptor() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "read -r -u 3 cell_index" in script
    assert 'done 3< "${GENERATION_ROOT}/cells.tsv"' in script
