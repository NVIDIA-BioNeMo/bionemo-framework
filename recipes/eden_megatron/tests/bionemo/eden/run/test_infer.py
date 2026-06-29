# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for Eden text generation (inference) using MBridge."""

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from ..utils import find_free_network_port


# Capture environment at import time (consistent with test_predict.py)
PRETEST_ENV = copy.deepcopy(os.environ)


def _read_jsonl_results(output_file: Path) -> list[dict]:
    """Read JSONL output file and return parsed records."""
    records = []
    with open(output_file) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


# =============================================================================
# Eden (Llama) inference tests
# =============================================================================


@pytest.fixture(scope="module")
def mbridge_eden_checkpoint_path(mbridge_eden_checkpoint) -> Path:
    """Module-scoped alias for the session-scoped Eden checkpoint."""
    return mbridge_eden_checkpoint


@pytest.mark.slow
def test_infer_eden_runs(mbridge_eden_checkpoint_path, tmp_path):
    """Test that infer.py runs without errors on an Eden (Llama) mbridge checkpoint."""
    output_file = tmp_path / "eden_output.jsonl"
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        "-m",
        "bionemo.eden.run.infer",
        "--ckpt-dir",
        str(mbridge_eden_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "10",
        "--output-file",
        str(output_file),
        "--temperature",
        "1.0",
        "--top-k",
        "1",
    ]

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )

    assert result.returncode == 0, f"Eden infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    records = _read_jsonl_results(output_file)
    assert len(records) >= 1, f"Expected at least 1 result, got {len(records)}"
    record = records[0]
    assert record["prompt"] == prompt
    assert len(record["completion"]) > 0, "Generated text is empty"


@pytest.mark.slow
def test_infer_eden_deterministic(mbridge_eden_checkpoint_path, tmp_path):
    """Test that Eden inference with greedy decoding is deterministic across runs."""
    output_1 = tmp_path / "eden_det_1.jsonl"
    output_2 = tmp_path / "eden_det_2.jsonl"
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

    for output_file in (output_1, output_2):
        open_port = find_free_network_port()
        cmd = [
            "torchrun",
            "--nproc_per_node",
            "1",
            "--nnodes",
            "1",
            "--master_port",
            str(open_port),
            "-m",
            "bionemo.eden.run.infer",
            "--ckpt-dir",
            str(mbridge_eden_checkpoint_path),
            "--prompt",
            prompt,
            "--max-new-tokens",
            "10",
            "--output-file",
            str(output_file),
            "--temperature",
            "1.0",
            "--top-k",
            "1",
            "--seed",
            "42",
        ]

        env = copy.deepcopy(PRETEST_ENV)
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300, env=env)
        assert result.returncode == 0, f"Eden infer failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    records_1 = _read_jsonl_results(output_1)
    records_2 = _read_jsonl_results(output_2)
    assert len(records_1) >= 1 and len(records_2) >= 1
    gen_1 = records_1[0]["completion"]
    gen_2 = records_2[0]["completion"]
    assert len(gen_1) > 0, "First generation produced empty output"
    assert gen_1 == gen_2, f"Deterministic Eden inference produced different outputs:\nRun 1: {gen_1}\nRun 2: {gen_2}"
