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

import fcntl
import io
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = RECIPE_ROOT / "examples/phix174_8xh100.sh"


def test_control_fasta_ids(tmp_path: Path, monkeypatch) -> None:
    marker = 'python - configs/phage_safety_reference_controls.yaml "${root}" "${table}" "${DRY_RUN}" <<\'PY\'\n'
    body = SCRIPT.read_text().split(marker, 1)[1].split("\nPY\n", 1)[0]
    config = tmp_path / "controls.yaml"
    root = tmp_path / "controls"
    table = root / "controls.tsv"
    config.write_text(
        json.dumps(
            {
                "controls": [
                    {
                        "control_id": "whole",
                        "accession": "NC_000001.1",
                        "sequence_interval": None,
                        "sequence_length": 4,
                        "topology": "circular",
                    },
                    {
                        "control_id": "slice",
                        "accession": "NC_000002.1",
                        "sequence_interval": {"start": 3, "end": 6},
                        "sequence_length": 4,
                        "topology": "linear",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b">ncbi\nACGT\n"))
    monkeypatch.setattr(sys, "argv", ["-", str(config), str(root), str(table), "0"])

    exec(compile(body, str(SCRIPT), "exec"), {})

    assert (root / "whole.fasta").read_text() == ">NC_000001.1\nACGT\n"
    assert (root / "slice.fasta").read_text() == ">NC_000002.1_3_6\nACGT\n"


def test_same_result_lock(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    result_root.mkdir()
    with (result_root / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = subprocess.run(
            ["bash", str(SCRIPT), "--dry-run", "--result-root", str(result_root)],
            cwd=RECIPE_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    assert completed.returncode != 0
    assert "already running for this result directory" in completed.stderr


def test_dry_run(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True, timeout=10)
    result_root = tmp_path / "result"
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--result-root", str(result_root)],
        cwd=RECIPE_ROOT,
        env={
            **os.environ,
            "API_KEY": "do-not-record",
            "NVIDIA_API_KEY": "also-do-not-record",
            "NEMO_RL_RAY_NUM_CPUS": "96",
        },
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
    large_scans = [
        command
        for command in safety_commands
        if any(
            part.endswith(
                (
                    "/sft/source-safety/biological.fna",
                    "/rollout/sequence-safety/input-qc/qc2_nt_filter_seqs.fasta",
                )
            )
            for part in command
        )
    ]
    assert len(large_scans) == 2
    for command in large_scans:
        assert command[command.index("--batch-size") + 1] == "128"
        assert command[command.index("--orf-workers") + 1] == "32"
        assert command[command.index("--threads") + 1] == "32"
        assert command[command.index("--phrogs-threads") + 1] == "64"
    assert "RL Ray CPU slots: 96" in log

    sft_command = next(
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: torchrun " in line and "--max-steps 12000" in line
    )
    assert sft_command[sft_command.index("--keep-best-k") + 1] == "3"
    assert sft_command[sft_command.index("--most-recent-k") + 1] == "1"
    assert sft_command[sft_command.index("--checkpoint-metric-name") + 1] == "lm loss"
    assert "--strict-checkpoint-metric" in sft_command
    assert sft_command[sft_command.index("--checkpoint-metric-step-tolerance") + 1] == "1"

    arc_commands = [
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: evo2_phage_prepare_arc_pipeline " in line
    ]
    assert len(arc_commands) == 1
    assert "--overwrite" in arc_commands[0]

    rl_control = next(
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: evo2_phage_check_rl " in line and "--control-fasta" in line
    )
    assert rl_control[rl_control.index("--control-fasta") + 1].endswith("NC_001422.1.fna")
    assert rl_control[rl_control.index("--control-dir") + 1].endswith("/rl/environment-control")
    assert log.index("monitor: RL environment control") < log.index("monitor: one-step GDPO pilot")


def test_substage_resume(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    stage_root = result_root / "stages"
    stage_root.mkdir(parents=True)
    (stage_root / "20-sft.done").touch()
    (stage_root / "50-rollout.done").touch()
    (stage_root / "40-rl.done").touch()

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--resume-from",
            "20",
            "--result-root",
            str(result_root),
        ],
        cwd=RECIPE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    log = (result_root / "RUNLOG.md").read_text()
    assert "substage 20-sft already complete" in log
    assert "monitor: SFT smoke" not in log
    assert "evo2_convert_nemo2_to_mbridge" not in log
    assert "monitor: 12,000-step SFT" not in log
    assert "monitor: held-out SFT evaluation" in log
    assert "substage 40-rl already complete" in log
    assert "monitor: RL environment control" not in log
    assert "substage 50-rollout already complete" in log
    assert "evo2_phage_generation write-prompts" not in log
    assert "--max-new-tokens 5976" not in log
    assert "monitor: selected-SFT likelihood scoring" in log
    assert "monitor: one-step GDPO pilot" not in log
    assert "monitor: 500-step DP8 GDPO" not in log
    assert "evo2_phage_monitor_objectives" in log
    assert "evo2_phage_sequence_safety scan" in log
