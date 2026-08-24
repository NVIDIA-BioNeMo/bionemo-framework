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


def _write_sampling_selection(path: Path) -> str:
    text = """\
temperature: 0.9
top_k: 17
top_p: 0.85
max_new_tokens: 5800
prompt_lengths: [4, 8, 16, 24]
rl_seed: 101
rollout_seed: 7
seed_stride: 11
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


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
    assert "export NCCL_GRAPH_REGISTER=0" in SCRIPT.read_text()
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
    assert "RUN COMPLETE: dry run finished; 6/6 steps planned" in completed.stdout
    assert (result_root / "stage-plan.txt").read_text().splitlines() == [
        "00 prepare inputs/tools/controls",
        "10 safety-screen and prepare SFT",
        "20 train/select/evaluate SFT",
        "30 calibrate sampling",
        "40 prepare SFT checkpoint for RL; pilot/check/train/monitor/select GDPO",
        "50 generate, deduplicate, SFT-score, hard-QC, cluster, and report 1,000 genomes",
    ]
    settings = json.loads((result_root / "settings.json").read_text())
    assert settings == {
        "cpu_count": 96,
        "gpu_count": 8,
        "gpu_type": "not queried (dry run)",
        "sft_tensor_parallel_size": 2,
        "model_variant": "7b-base",
        "base_checkpoint": "evo2/7b-8k:1.0",
        "model_size": "evo2_7b_base",
        "whole_genome": True,
        "safety_screen": "current configured databases",
        "final_generation_count": 1000,
        "wandb_enabled": False,
        "wandb_entity": None,
        "wandb_sft_project": "evo2-phage-design-sft",
        "wandb_rl_project": "evo2-phage-design-gdpo",
        "wandb_sft_run_name": "result-7b-base-sft",
        "wandb_rl_run_name": "result-7b-base-gdpo",
    }
    log = (result_root / "RUNLOG.md").read_text()
    for command in (
        "evo2_phage_prepare_external_assets",
        "evo2_phage_sequence_safety",
        "evo2_phage_prepare_sft_split",
        "train_evo2",
        "run_sampling_calibration_scoring.sh",
        "bionemo.evo2_phage_gen.prepare_sft_checkpoint_for_rl",
        "evo2_phage_run_gdpo",
        "predict_evo2",
        "collect-sft-likelihood",
        "deduplicate-fasta",
        "summarize-arc-screen",
        "select-hard-qc-passers",
        "cluster-post-qc",
        "genome_design_filtering_pipeline.py",
        "finalize-rollout",
    ):
        assert command in log
    safety_input_argument = (
        f"--safety-input-fasta {result_root}/rollout/sequence-safety/input-qc/qc2_nt_filter_seqs.fasta"
    )
    assert log.count(safety_input_argument) == 2
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
    assert "Arc CheckV database: <prepared-checkv-db>" in log
    assert f"Arc screening working directory: {RECIPE_ROOT.parents[1]}" in log
    assert "Arc internal MMseqs clustering disabled" in log

    deduplication = log.index("command: evo2_phage_generation deduplicate-fasta")
    likelihood = log.index("monitor: selected-SFT likelihood scoring")
    safety = log.index("command: evo2_phage_sequence_safety scan", likelihood)
    target = log.index("monitor: Arc target profile")
    diagnostic = log.index("monitor: Arc filter-7 diagnostic")
    clustering = log.index("command: evo2_phage_generation cluster-post-qc")
    reporting = log.index("command: evo2_phage_generation finalize-rollout")
    assert deduplication < likelihood < safety < target < diagnostic < clustering < reporting
    assert "/rollout/deduplication/representatives.fasta" in log

    sft_command = next(
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: torchrun " in line and "--max-steps 12000" in line
    )
    assert sft_command[sft_command.index("--keep-best-k") + 1] == "3"
    assert sft_command[sft_command.index("--model-size") + 1] == "evo2_7b_base"
    assert sft_command[sft_command.index("--most-recent-k") + 1] == "1"
    assert sft_command[sft_command.index("--checkpoint-metric-name") + 1] == "lm loss"
    assert "--strict-checkpoint-metric" in sft_command
    assert sft_command[sft_command.index("--checkpoint-metric-step-tolerance") + 1] == "1"
    assert "--wandb-project" not in sft_command

    heldout_command = next(
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: torchrun " in line and "--experiment-name evo2-heldout" in line
    )
    assert heldout_command[heldout_command.index("--max-steps") + 1] == "0"
    assert heldout_command[heldout_command.index("--decay-steps") + 1] == "1"

    arc_commands = [
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: evo2_phage_prepare_arc_pipeline " in line
    ]
    assert len(arc_commands) == 2
    assert all("--overwrite" in command for command in arc_commands)

    rl_control = next(
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: evo2_phage_check_rl " in line and "--control-fasta" in line
    )
    assert rl_control[rl_control.index("--control-fasta") + 1].endswith("NC_001422_1.fna")
    assert rl_control[rl_control.index("--control-dir") + 1].endswith("/rl/environment-control")
    assert rl_control[rl_control.index("--prompt-data") + 1] == str(result_root / "rl/train.jsonl")
    assert rl_control[rl_control.index("--checkpoint") + 1] == "<rl-sft-checkpoint>"
    preparation = log.index("command: python -m bionemo.evo2_phage_gen.prepare_sft_checkpoint_for_rl")
    assert preparation < log.index("command: evo2_phage_check_rl")
    assert log.index("monitor: RL environment control") < log.index("monitor: one-step GDPO pilot")

    gdpo_commands = [
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: evo2_phage_run_gdpo " in line
    ]
    assert len(gdpo_commands) == 2
    assert all("logger.wandb_enabled=false" in command for command in gdpo_commands)
    gdpo = gdpo_commands[0]
    assert "checkpointing.pretrained_checkpoint.path=<rl-sft-checkpoint>" in gdpo
    assert "policy.model_name=bionemo/evo2_7b_base" in gdpo

    conversion = next(
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: evo2_convert_nemo2_to_mbridge " in line
    )
    assert conversion[conversion.index("--nemo2-ckpt-dir") + 1] == "<downloaded-evo2-7b-8k>"
    assert conversion[conversion.index("--model-size") + 1] == "evo2_7b_base"


def test_wandb_dry_run(tmp_path: Path) -> None:
    result_root = tmp_path / "wandb-result"
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--wandb",
            "--wandb-entity",
            "example-team",
            "--wandb-sft-project",
            "custom-sft",
            "--wandb-rl-project",
            "custom-gdpo",
            "--result-root",
            str(result_root),
        ],
        cwd=RECIPE_ROOT,
        env={**os.environ, "WANDB_API_KEY": "do-not-record"},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    settings = json.loads((result_root / "settings.json").read_text())
    assert settings["wandb_enabled"] is True
    assert settings["wandb_entity"] == "example-team"
    assert settings["wandb_sft_project"] == "custom-sft"
    assert settings["wandb_rl_project"] == "custom-gdpo"
    assert settings["wandb_sft_run_name"] == "wandb-result-7b-base-sft"
    assert settings["wandb_rl_run_name"] == "wandb-result-7b-base-gdpo"

    log = (result_root / "RUNLOG.md").read_text()
    assert "do-not-record" not in log
    commands = [shlex.split(line.partition("command: ")[2]) for line in log.splitlines() if "command: " in line]
    sft_commands = [command for command in commands if "train_evo2" in command]
    full_sft = next(command for command in sft_commands if command[command.index("--max-steps") + 1] == "12000")
    assert full_sft[full_sft.index("--wandb-entity") + 1] == "example-team"
    assert full_sft[full_sft.index("--wandb-project") + 1] == "custom-sft"
    assert full_sft[full_sft.index("--wandb-run-name") + 1] == "wandb-result-7b-base-sft"
    assert all("--wandb-project" not in command for command in sft_commands if command is not full_sft)

    gdpo_commands = [command for command in commands if command[:1] == ["evo2_phage_run_gdpo"]]
    assert len(gdpo_commands) == 2
    pilot = next(command for command in gdpo_commands if "grpo.max_num_steps=1" in command)
    full_gdpo = next(command for command in gdpo_commands if command is not pilot)
    assert "logger.wandb_enabled=false" in pilot
    assert not any(part.startswith("logger.wandb.project=") for part in pilot)
    assert "logger.wandb_enabled=true" in full_gdpo
    assert "logger.wandb.project=custom-gdpo" in full_gdpo
    assert "logger.wandb.name=wandb-result-7b-base-gdpo" in full_gdpo


def test_single_gpu_plan(tmp_path: Path) -> None:
    result_root = tmp_path / "single-gpu"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MODEL_VARIANT", "NUM_CPUS", "NUM_GPUS", "SFT_TENSOR_PARALLEL_SIZE"}
    }
    env["NUM_GPUS"] = "1"
    env["NUM_CPUS"] = "72"
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--result-root", str(result_root)],
        cwd=RECIPE_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    settings = json.loads((result_root / "settings.json").read_text())
    assert settings["gpu_count"] == 1
    assert settings["sft_tensor_parallel_size"] == 1

    log = (result_root / "RUNLOG.md").read_text()
    commands = [shlex.split(line.partition("command: ")[2]) for line in log.splitlines() if "command: " in line]
    sft = next(command for command in commands if command[:1] == ["torchrun"] and "--max-steps" in command)
    assert sft[sft.index("--nproc-per-node") + 1] == "1"
    assert sft[sft.index("--tensor-model-parallel-size") + 1] == "1"
    assert sft[sft.index("--global-batch-size") + 1] == "32"
    assert sft[sft.index("--seq-length") + 1] == "10240"

    rollout = [command for command in commands if command[:1] == ["env"] and "--prompt-file" in command]
    assert len(rollout) == 2
    assert [command[1] for command in rollout] == ["CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=0"]
    assert [Path(command[command.index("--prompt-file") + 1]).name for command in rollout] == [
        "dp0.jsonl",
        "dp1.jsonl",
    ]
    assert [command[command.index("--seed") + 1] for command in rollout] == ["7", "1000010"]
    assert "split the equal prompt mixture (16 24) into 2 deterministic shards over 1 GPU(s)" in log


def test_dry_run_supports_preferred_7b_1m_variant(tmp_path: Path) -> None:
    """A fresh result root can explicitly select the trained-further long-context family."""
    result_root = tmp_path / "result-1m"
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--model-variant",
            "7b-1m",
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
    settings = json.loads((result_root / "settings.json").read_text())
    assert settings["model_variant"] == "7b-1m"
    assert settings["base_checkpoint"] == "evo2/7b-1m:1.0"
    assert settings["model_size"] == "evo2_7b"
    log = (result_root / "RUNLOG.md").read_text()
    conversion = next(
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: evo2_convert_nemo2_to_mbridge " in line
    )
    assert conversion[conversion.index("--nemo2-ckpt-dir") + 1] == "<downloaded-evo2-7b-1m>"
    assert conversion[conversion.index("--model-size") + 1] == "evo2_7b"
    gdpo = next(
        shlex.split(line.partition("command: ")[2])
        for line in log.splitlines()
        if "command: evo2_phage_run_gdpo " in line
    )
    assert "policy.model_name=bionemo/evo2_7b" in gdpo


def test_existing_base_run_rejects_mid_run_switch_to_1m(tmp_path: Path) -> None:
    """Historical result roots without a variant marker are known to be 7B-base runs."""
    result_root = tmp_path / "existing-base"
    (result_root / "state").mkdir(parents=True)
    (result_root / "state" / "selected-sft").write_text("/checkpoint/iter_0005200\n")

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--model-variant",
            "7b-1m",
            "--result-root",
            str(result_root),
        ],
        cwd=RECIPE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2
    assert "recorded model variant is 7b-base" in completed.stderr
    assert "new result root" in completed.stderr


def test_sampling_selection_override(tmp_path: Path) -> None:
    source = tmp_path / "custom-selection.yaml"
    expected = _write_sampling_selection(source)
    result_root = tmp_path / "result"

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--result-root",
            str(result_root),
            "--sampling-selection",
            str(source),
        ],
        cwd=RECIPE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert (result_root / "calibration/sampling-selection.yaml").read_text() == expected
    log = (result_root / "RUNLOG.md").read_text()
    assert "WARNING: copied explicit sampling selection" in log
    assert "using explicit sampling selection" in log
    assert "sampling selection: temperature=0.9, prompt lengths=4 8 16 24, max new tokens=5800" in log
    commands = [shlex.split(line.partition("command: ")[2]) for line in log.splitlines() if "command: " in line]
    prompt_banks = [command for command in commands if command[:2] == ["evo2_phage_generation", "write-rl-prompts"]]
    assert len(prompt_banks) == 2
    assert all(
        command[command.index("--prompt-lengths") + 1 : command.index("--repeats-per-length")]
        == ["4", "8", "16", "24"]
        for command in prompt_banks
    )
    assert prompt_banks[0][prompt_banks[0].index("--repeats-per-length") + 1] == "3"
    assert prompt_banks[1][prompt_banks[1].index("--repeats-per-length") + 1] == "24"

    gdpo = next(command for command in commands if command[:1] == ["evo2_phage_run_gdpo"])
    for override in (
        "policy.generation.max_new_tokens=5800",
        "policy.generation.temperature=0.9",
        "policy.generation.top_k=17",
        "policy.generation.top_p=0.85",
        "policy.generation.mcore_generation_config.generation_adapter_config.seed=101",
        "policy.generation.mcore_generation_config.generation_adapter_config.seed_stride=11",
    ):
        assert override in gdpo

    rollout = [command for command in commands if command[:1] == ["env"] and "--prompt-file" in command]
    assert len(rollout) == 8
    for rank, command in enumerate(rollout):
        assert command[command.index("--max-new-tokens") + 1] == "5800"
        assert command[command.index("--temperature") + 1] == "0.9"
        assert command[command.index("--top-k") + 1] == "17"
        assert command[command.index("--top-p") + 1] == "0.85"
        assert command[command.index("--seed") + 1] == str(7 + rank * 11)
    assert "phix174_prompt4-8-16-24_temp0.9.n1000.fasta" in log


def test_calibrate_only_stops_after_scoring_for_sampling_review(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--calibrate-only",
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
    assert "monitor: calibration generation" in log
    assert "monitor: calibration scoring" in log
    assert "calibration review requested" in log
    assert str(result_root / "calibration/scoring/selection-evidence.csv") in log
    assert "write-rl-prompts" not in log
    assert "evo2_phage_run_gdpo" not in log
    assert "RUN PAUSED after step 4/6 (stage 30: calibrate sampling)" in log


def test_failure_footer_reports_stage_progress(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    stage_root = result_root / "stages"
    stage_root.mkdir(parents=True)
    for stage in ("00", "10", "20"):
        (stage_root / f"{stage}.done").touch()
    selection = result_root / "calibration/sampling-selection.yaml"
    selection.parent.mkdir(parents=True)
    selection.write_text("temperature: 0.9\n")

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--resume-from",
            "30",
            "--result-root",
            str(result_root),
        ],
        cwd=RECIPE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    expected = (
        "RUN FAILED during step 4/6 (stage 30: calibrate sampling); "
        f"3/6 steps complete; exit code {completed.returncode}; see {result_root / 'RUNLOG.md'}"
    )
    assert expected in completed.stderr
    assert expected in (result_root / "RUNLOG.md").read_text()


def test_existing_sampling_selection_is_reused(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    canonical = result_root / "calibration/sampling-selection.yaml"
    _write_sampling_selection(canonical)

    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--result-root", str(result_root)],
        cwd=RECIPE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    log = (result_root / "RUNLOG.md").read_text()
    assert "using existing sampling selection and skipping the fresh-calibration default check" in log
    assert "policy.generation.temperature=0.9" in log


def test_invalid_override_preserves_selection(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    canonical = result_root / "calibration/sampling-selection.yaml"
    expected = _write_sampling_selection(canonical)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("temperature: 0.7\n")

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--result-root",
            str(result_root),
            "--sampling-selection",
            str(invalid),
        ],
        cwd=RECIPE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2
    assert "Invalid sampling selection" in completed.stderr
    assert canonical.read_text() == expected


def test_substage_resume(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    stage_root = result_root / "stages"
    stage_root.mkdir(parents=True)
    (stage_root / "20-sft.done").touch()
    (stage_root / "30-calibration-generation.done").touch()
    (stage_root / "30-calibration-scoring.done").touch()
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
    assert "substage 30-calibration-generation already complete" in log
    assert "monitor: calibration generation" not in log
    assert "substage 30-calibration-scoring already complete" in log
    assert "monitor: calibration scoring" not in log
    assert "verify fresh calibration supports the bundled default" in log
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


def test_stage40_pilot_marker_skips_pilot_but_runs_monitor_and_full_training(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    stage_root = result_root / "stages"
    stage_root.mkdir(parents=True)
    (stage_root / "40-pilot.done").touch()

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--resume-from",
            "40",
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
    assert "substage 40-pilot already complete" in log
    assert "monitor: one-step GDPO pilot" not in log
    assert "evo2_phage_monitor_objectives" in log
    assert "monitor: 500-step DP8 GDPO" in log


def test_stage40_pilot_check_marker_skips_completed_pilot_check(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    stage_root = result_root / "stages"
    stage_root.mkdir(parents=True)
    (stage_root / "40-pilot.done").touch()
    (stage_root / "40-pilot-check.done").touch()

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--resume-from",
            "40",
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
    assert "substage 40-pilot already complete" in log
    assert "substage 40-pilot-check already complete" in log
    assert str(result_root / "rl-pilot/objective-health.json") not in log
    assert "monitor: 500-step DP8 GDPO" in log


def test_stage50_granular_markers_skip_completed_work(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    stage_root = result_root / "stages"
    stage_root.mkdir(parents=True)
    for marker in (
        "20-sft.done",
        "30-calibration-generation.done",
        "30-calibration-scoring.done",
        "40-rl.done",
        "50-rollout.done",
        "50-deduplication.done",
        "50-sft-likelihood.done",
        "50-sequence-safety.done",
        "50-target-profile.done",
        "50-filter7-diagnostic.done",
        "50-final-clustering.done",
        "50-report.done",
    ):
        (stage_root / marker).touch()

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--resume-from",
            "50",
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
    for substage in (
        "50-rollout",
        "50-deduplication",
        "50-sft-likelihood",
        "50-sequence-safety",
        "50-target-profile",
        "50-filter7-diagnostic",
        "50-final-clustering",
        "50-report",
    ):
        assert f"substage {substage} already complete" in log
    assert "command: evo2_phage_generation deduplicate-fasta" not in log
    assert "command: evo2_phage_sequence_safety scan" not in log
    assert "command: evo2_phage_generation cluster-post-qc" not in log
    assert "command: evo2_phage_generation finalize-rollout" not in log


def test_topology_env(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--result-root", str(result_root)],
        cwd=RECIPE_ROOT,
        env={
            **os.environ,
            "NUM_GPUS": "4",
            "NUM_CPUS": "48",
            "SFT_TENSOR_PARALLEL_SIZE": "1",
            "NEMO_RL_RAY_NUM_CPUS": "",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    settings = json.loads((result_root / "settings.json").read_text())
    assert settings["gpu_count"] == 4
    assert settings["cpu_count"] == 48
    assert settings["sft_tensor_parallel_size"] == 1
    log = (result_root / "RUNLOG.md").read_text()
    commands = [shlex.split(line.partition("command: ")[2]) for line in log.splitlines() if "command: " in line]
    distributed = [command for command in commands if command[:1] == ["torchrun"] and "--nproc-per-node" in command]
    assert {command[command.index("--nproc-per-node") + 1] for command in distributed} == {"4"}
    sft = next(command for command in distributed if "--max-steps" in command)
    assert sft[sft.index("--tensor-model-parallel-size") + 1] == "1"
    calibration = next(command for command in commands if "scripts/calibration/run_sft_sampling_sweep.sh" in command)
    assert "GPU_IDS=0 1 2 3" in calibration
    rollout = [
        command[1] for command in commands if command[:1] == ["env"] and command[1].startswith("CUDA_VISIBLE_DEVICES=")
    ]
    assert rollout == [f"CUDA_VISIBLE_DEVICES={rank}" for rank in range(4)]
    assert "cluster.gpus_per_node=4" in log
    assert "RL Ray CPU slots: 48" in log
