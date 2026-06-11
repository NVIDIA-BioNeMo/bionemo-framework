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

"""Unified CodonFM SAE pipeline: extract -> train -> eval.

Usage:
    # Full pipeline for 1B model:
    python run.py model=1b csv_path=path/to/Primates.csv

    # Skip extraction (already cached):
    python run.py model=1b csv_path=path/to/data.csv steps.extract=false

    # Override any param:
    python run.py model=1b csv_path=data.csv train.n_epochs=5 nproc=8 dp_size=8

    # Quick smoke test:
    python run.py model=1b csv_path=data.csv num_sequences=100 train.n_epochs=1 nproc=1 dp_size=1
"""

import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _run(cmd: list, description: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    subprocess.run([str(c) for c in cmd], check=True)


def _torchrun_prefix(nproc: int) -> list:
    if nproc > 1:
        return ["torchrun", f"--nproc_per_node={nproc}"]
    return [sys.executable]


def run_extract(cfg: DictConfig, cache_dir: Path) -> None:  # noqa: D103
    cmd = [
        *_torchrun_prefix(cfg.nproc),
        str(SCRIPTS_DIR / "extract.py"),
        "--csv-path",
        cfg.csv_path,
        "--num-sequences",
        str(cfg.num_sequences),
        "--layer",
        str(cfg.layer),
        "--model-path",
        cfg.model_path,
        "--batch-size",
        str(cfg.batch_size),
        "--context-length",
        str(cfg.context_length),
        "--seed",
        str(cfg.seed),
        "--output",
        str(cache_dir),
    ]
    if cfg.extract.get("shard_size"):
        cmd.extend(["--shard-size", str(cfg.extract.shard_size)])

    _run(cmd, f"STEP 1: Extract activations from {cfg.model_path}")


def run_train(cfg: DictConfig, cache_dir: Path, output_dir: Path) -> None:  # noqa: D103
    checkpoint_dir = output_dir / "checkpoints"
    t = cfg.train

    cmd = [
        *_torchrun_prefix(cfg.nproc),
        str(SCRIPTS_DIR / "train.py"),
        "--cache-dir",
        str(cache_dir),
        "--model-path",
        cfg.model_path,
        "--layer",
        str(cfg.layer),
        "--model-type",
        t.model_type,
        "--expansion-factor",
        str(t.expansion_factor),
        "--top-k",
        str(t.top_k),
        "--lr",
        str(t.lr),
        "--n-epochs",
        str(t.n_epochs),
        "--batch-size",
        str(t.batch_size),
        "--log-interval",
        str(t.log_interval),
        "--dp-size",
        str(cfg.dp_size),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-steps",
        str(t.checkpoint_steps),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(cfg.seed),
        "--num-sequences",
        str(cfg.num_sequences),
    ]

    if t.auxk:
        cmd.extend(["--auxk", str(t.auxk)])
    cmd.extend(["--auxk-coef", str(t.auxk_coef)])
    cmd.extend(["--dead-tokens-threshold", str(t.dead_tokens_threshold)])
    if t.init_pre_bias:
        cmd.append("--init-pre-bias")
    if t.normalize_input:
        cmd.append("--normalize-input")
    if t.get("max_grad_norm"):
        cmd.extend(["--max-grad-norm", str(t.max_grad_norm)])
    if t.get("lr_schedule", "constant") != "constant":
        cmd.extend(["--lr-schedule", str(t.lr_schedule)])
    if t.get("lr_min", 0.0) != 0.0:
        cmd.extend(["--lr-min", str(t.lr_min)])
    if t.get("lr_decay_steps"):
        cmd.extend(["--lr-decay-steps", str(t.lr_decay_steps)])
    if t.get("warmup_steps", 0) > 0:
        cmd.extend(["--warmup-steps", str(t.warmup_steps)])

    if t.wandb_enabled:
        cmd.append("--wandb")
        cmd.extend(["--wandb-project", t.wandb_project])
    else:
        cmd.append("--no-wandb")

    _run(cmd, "STEP 2: Train SAE")


def run_eval(cfg: DictConfig, output_dir: Path) -> None:  # noqa: D103
    checkpoint = output_dir / "checkpoints" / "checkpoint_final.pt"
    eval_dir = output_dir / "eval"

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "eval.py"),
        "--checkpoint",
        str(checkpoint),
        "--top-k",
        str(cfg.train.top_k),
        "--model-path",
        cfg.model_path,
        "--layer",
        str(cfg.layer),
        "--context-length",
        str(cfg.context_length),
        "--batch-size",
        str(cfg.batch_size),
        "--csv-path",
        cfg.eval.csv_path,
        "--num-sequences",
        str(cfg.eval.num_sequences),
        "--output-dir",
        str(eval_dir),
        "--seed",
        str(cfg.seed),
    ]

    _run(cmd, "STEP 3: Evaluate SAE (loss recovered)")


@hydra.main(version_base=None, config_path="run_configs", config_name="config")
def main(cfg: DictConfig) -> None:  # noqa: D103
    os.chdir(hydra.utils.get_original_cwd())

    print(OmegaConf.to_yaml(cfg))

    cache_dir = Path(f".cache/activations/{cfg.run_name}_layer{cfg.layer}")
    output_dir = Path(cfg.output_base) / cfg.run_name

    if cfg.steps.extract:
        run_extract(cfg, cache_dir)

    if cfg.steps.train:
        run_train(cfg, cache_dir, output_dir)

    if cfg.steps.eval:
        run_eval(cfg, output_dir)

    print(f"\n{'=' * 60}")
    print(f"  DONE: {cfg.run_name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
