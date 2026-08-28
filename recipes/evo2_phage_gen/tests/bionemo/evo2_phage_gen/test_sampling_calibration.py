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
import subprocess
import sys
from pathlib import Path

import pytest

from bionemo.evo2_phage_gen.sampling_calibration import (
    SweepCell,
    _format_temperature,
    _parse_args,
    build_inference_command,
    build_sweep_cells,
    materialize_sweep,
    partition_gpu_groups,
    validate_cell_output,
    write_cell_prompts,
)


def test_build_sweep_cells_includes_marker_only_and_canonical_temperature() -> None:
    cells = build_sweep_cells(prefix_lengths=[0, 4], temperatures=[0.7, 1.0])

    assert [cell.key for cell in cells] == [
        "prefix0_temp0.7",
        "prefix4_temp0.7",
        "prefix0_temp1.0",
        "prefix4_temp1.0",
    ]


def test_build_sweep_cells_validates_nonempty_positive_grid() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_sweep_cells([], [1.0])
    with pytest.raises(ValueError, match="non-empty"):
        build_sweep_cells([0], [])
    with pytest.raises(ValueError, match="non-negative"):
        build_sweep_cells([-1], [1.0])
    with pytest.raises(ValueError, match="positive"):
        build_sweep_cells([0], [0.0])


def test_format_temperature_preserves_numeric_exponent_form() -> None:
    assert _format_temperature(1e-5) == "1e-05"


def test_partition_gpu_groups_prefers_all_available_replicas() -> None:
    assert partition_gpu_groups(list(range(8)), tensor_parallel_size=1) == [
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
    ]
    assert partition_gpu_groups(list(range(8)), tensor_parallel_size=2) == [(0, 1), (2, 3), (4, 5), (6, 7)]
    with pytest.raises(ValueError, match="divisible"):
        partition_gpu_groups([0, 1, 2], tensor_parallel_size=2)


def test_write_cell_prompts_supports_marker_only_control(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    cell = SweepCell(prefix_length=0, temperature=1.0)

    write_cell_prompts(path, cell=cell, reference_start="GAGT", marker="+~", num_prompts=2)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {"id": "prefix0_temp1.0_0000", "prompt": "+~"},
        {"id": "prefix0_temp1.0_0001", "prompt": "+~"},
    ]


def test_build_inference_command_preserves_total_target_length(tmp_path: Path) -> None:
    command = build_inference_command(
        infer_script=tmp_path / "infer.py",
        checkpoint=tmp_path / "checkpoint",
        prompt_file=tmp_path / "prompts.jsonl",
        output_file=tmp_path / "output.jsonl",
        cell=SweepCell(prefix_length=24, temperature=1.0),
        target_length=6000,
        seed=7,
        tensor_parallel_size=1,
        master_port=29551,
        prompt_batch_size=16,
        max_seq_length=10240,
        top_k=17,
        top_p=0.85,
    )

    assert command[command.index("--max-new-tokens") + 1] == "5976"
    assert command[command.index("--temperature") + 1] == "1.0"
    assert command[command.index("--top-k") + 1] == "17"
    assert command[command.index("--top-p") + 1] == "0.85"
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert command[command.index("--inference-backend") + 1] == "dynamic"
    assert "--ignore-eos" in command
    assert "--strict-generation" in command


def test_build_inference_command_enables_full_scope_regular_hopper_fp8(tmp_path: Path) -> None:
    command = build_inference_command(
        infer_script=tmp_path / "infer.py",
        checkpoint=tmp_path / "checkpoint",
        prompt_file=tmp_path / "prompts.jsonl",
        output_file=tmp_path / "output.jsonl",
        cell=SweepCell(prefix_length=24, temperature=1.0),
        target_length=6000,
        seed=7,
        tensor_parallel_size=1,
        master_port=29551,
        prompt_batch_size=16,
        max_seq_length=10240,
        top_k=17,
        top_p=0.85,
        hopper_fp8=True,
    )

    assert command[command.index("--mixed-precision-recipe") + 1] == "bf16_with_fp8_current_scaling_mixed"
    assert "--fp8-all-layers" in command


def test_print_command_cli_emits_complete_nul_delimited_vector(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bionemo.evo2_phage_gen.sampling_calibration",
            "print-command",
            "--infer-script",
            str(tmp_path / "infer.py"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--prompt-file",
            str(tmp_path / "prompts.jsonl"),
            "--output-file",
            str(tmp_path / "output.jsonl"),
            "--prefix-length",
            "24",
            "--temperature",
            "0.9",
            "--target-length",
            "6000",
            "--seed",
            "7",
            "--tensor-parallel-size",
            "2",
            "--master-port",
            "29680",
            "--prompt-batch-size",
            "16",
            "--max-seq-length",
            "10240",
            "--top-k",
            "4",
            "--top-p",
            "1.0",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    command = [item.decode() for item in completed.stdout.split(b"\0") if item]

    assert command[0] == "torchrun"
    assert command[command.index("--max-new-tokens") + 1] == "5976"
    assert command[command.index("--temperature") + 1] == "0.9"
    assert command[command.index("--output-file") + 1] == str(tmp_path / "output.jsonl")


def test_validate_cell_output_requires_exact_prompt_ids(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "output.jsonl"
    cell = SweepCell(prefix_length=4, temperature=0.9)
    write_cell_prompts(prompts, cell=cell, reference_start="GAGT", marker="+~", num_prompts=2)
    expected = [json.loads(line) for line in prompts.read_text().splitlines()]
    output.write_text(
        "".join(
            json.dumps({"id": record["id"], "prompt": record["prompt"], "completion": "A" * 10}) + "\n"
            for record in expected
        )
    )

    assert validate_cell_output(output, prompts, expected_records=2) == 2

    output.write_text(
        json.dumps({"id": "wrong-1", "prompt": "+~GAGT", "completion": "AAAA"})
        + "\n"
        + json.dumps({"id": "wrong-2", "prompt": "+~GAGT", "completion": "AAAA"})
        + "\n"
    )
    with pytest.raises(ValueError, match="IDs"):
        validate_cell_output(output, prompts, expected_records=2)


def test_materialize_sweep_config_mismatch_does_not_modify_outputs(tmp_path: Path) -> None:
    kwargs = {
        "run_root": tmp_path / "run",
        "checkpoint": tmp_path / "checkpoint",
        "prefix_lengths": [0],
        "temperatures": [0.7],
        "num_prompts": 1,
        "reference_start": "GAGT",
        "marker": "+~",
        "gpu_ids": [0],
        "tensor_parallel_size": 1,
        "target_length": 10,
        "top_k": 4,
        "top_p": 1.0,
        "seed": 7,
        "prompt_batch_size": 1,
        "max_seq_length": 16,
    }
    config = materialize_sweep(**kwargs)
    assert config["inference_precision"] == "bf16"
    before = {
        path.relative_to(kwargs["run_root"]): path.read_bytes()
        for path in kwargs["run_root"].rglob("*")
        if path.is_file()
    }

    with pytest.raises(ValueError, match="configuration differs"):
        materialize_sweep(**{**kwargs, "temperatures": [0.9]})
    with pytest.raises(ValueError, match="configuration differs"):
        materialize_sweep(**{**kwargs, "hopper_fp8": True})

    after = {
        path.relative_to(kwargs["run_root"]): path.read_bytes()
        for path in kwargs["run_root"].rglob("*")
        if path.is_file()
    }
    assert after == before


def test_materialize_cli_requires_explicit_sampling_parameters(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampling-calibration",
            "materialize",
            "--run-root",
            "run",
            "--checkpoint",
            "checkpoint",
            "--prefix-lengths",
            "0",
            "--temperatures",
            "1.0",
            "--num-prompts",
            "1",
            "--gpu-ids",
            "0",
            "--tensor-parallel-size",
            "1",
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()
