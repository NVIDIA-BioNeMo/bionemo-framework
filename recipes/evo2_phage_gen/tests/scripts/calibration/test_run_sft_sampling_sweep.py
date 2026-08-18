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

import json
import os
import subprocess
import sys
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = RECIPE_ROOT / "scripts/calibration/run_sft_sampling_sweep.sh"


def test_sampling_sweep_dry_run_materializes_marker_only_parallel_plan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    run_root = tmp_path / "sweep"
    env = {
        **os.environ,
        "SOURCE_ENV": "0",
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "DRY_RUN": "1",
        "RECIPE_ROOT": str(RECIPE_ROOT),
        "RUN_ROOT": str(run_root),
        "CKPT_DIR": str(checkpoint),
        "PROMPT_LENGTHS": "0 4",
        "TEMPERATURES": "0.7 1.0",
        "NUM_PROMPTS": "2",
        "GPU_IDS": "0 1",
        "TENSOR_PARALLEL_SIZE": "1",
    }

    subprocess.run(["bash", str(SCRIPT)], check=True, env=env, cwd=RECIPE_ROOT, timeout=120)

    sweep_config = json.loads((run_root / "sweep_config.json").read_text())
    assert sweep_config["topology"] == {"gpu_ids": [0, 1], "tensor_parallel_size": 1, "replicas": 2}
    assert sweep_config["cells"] == [
        "prefix0_temp0.7",
        "prefix4_temp0.7",
        "prefix0_temp1.0",
        "prefix4_temp1.0",
    ]
    marker_only = [
        json.loads(line) for line in (run_root / "prompts/prefix0_temp0.7_2.jsonl").read_text().splitlines()
    ]
    assert marker_only[0] == {"id": "prefix0_temp0.7_0000", "prompt": "+~"}
    assert b"\r" not in (run_root / "cells.tsv").read_bytes()
    assert (run_root / "DRY_RUN_COMPLETE").is_file()


def test_sampling_workers_use_dedicated_input_and_guard_token_budget() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "read -r -u 3 cell_index" in script
    assert 'done 3< "${RUN_ROOT}/cells.tsv"' in script
    assert "if (( max_new_tokens <= 0 )); then" in script
    assert "sampling_calibration print-command" in script
    assert "mapfile -d '' -t inference_command" in script
