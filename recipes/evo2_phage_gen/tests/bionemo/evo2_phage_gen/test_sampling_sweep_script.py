# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import json
import os
import subprocess
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = RECIPE_ROOT / "scripts/run_sft_sampling_sweep.sh"


def test_sampling_sweep_dry_run_materializes_marker_only_parallel_plan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    run_root = tmp_path / "sweep"
    env = {
        **os.environ,
        "SOURCE_ENV": "0",
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

    subprocess.run(["bash", str(SCRIPT)], check=True, env=env, cwd=RECIPE_ROOT)

    contract = json.loads((run_root / "sweep_contract.json").read_text())
    assert contract["topology"] == {"gpu_ids": [0, 1], "tensor_parallel_size": 1, "replicas": 2}
    assert contract["cells"] == [
        "prefix0_temp0.7",
        "prefix4_temp0.7",
        "prefix0_temp1.0",
        "prefix4_temp1.0",
    ]
    marker_only = [
        json.loads(line)
        for line in (run_root / "prompts/prefix0_temp0.7_2.jsonl").read_text().splitlines()
    ]
    assert marker_only[0] == {"id": "prefix0_temp0.7_0000", "prompt": "+~"}
    assert b"\r" not in (run_root / "cells.tsv").read_bytes()
    assert (run_root / "DRY_RUN_COMPLETE").is_file()
