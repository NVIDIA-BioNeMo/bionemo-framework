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

"""Validate the full-shape GDPO pilot's model-only restart and cache release."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


_CACHE_RELEASE_PATTERN = re.compile(
    r"Evo2 native cache release verified: mode=(?P<mode>\S+) "
    r"tensor_state_allocated=(?P<allocated>true|false) "
    r"allocated_before_mib=(?P<before>\d+) allocated_after_mib=(?P<after>\d+)"
)
_TRAINING_STEP_PATTERN = re.compile(r"={5,}\s+Step\s+\d+/\d+\s+={5,}")


def _load_mapping(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a mapping: {path}")
    return payload


def validate_rl_pilot(
    *,
    checkpoint_root: Path,
    expected_step: int,
    runner_log: Path,
    reload_log: Path,
) -> dict[str, Any]:
    """Validate one complete model-only checkpoint and a no-training reload process."""
    checkpoint_root = checkpoint_root.resolve()
    checkpoint = checkpoint_root / f"step_{expected_step}"
    training_info = _load_mapping(checkpoint / "training_info.json", description="training state")
    if training_info.get("total_steps") != expected_step:
        raise ValueError(
            f"checkpoint total_steps is {training_info.get('total_steps')!r}, expected {expected_step}"
        )

    config = _load_mapping(checkpoint / "config.yaml", description="resolved checkpoint config")
    if config.get("checkpointing", {}).get("save_optimizer") is not False:
        raise ValueError("pilot checkpoint must set checkpointing.save_optimizer=false")
    if not (checkpoint / "train_dataloader.pt").is_file():
        raise ValueError(f"missing saved dataloader state: {checkpoint / 'train_dataloader.pt'}")
    weights = checkpoint / "policy/weights/iter_0000000"
    if not (weights / ".metadata").is_file():
        raise ValueError(f"missing completed Megatron policy checkpoint metadata: {weights / '.metadata'}")
    optimizer = checkpoint / "policy/optimizer"
    if optimizer.exists():
        raise ValueError(f"model-only checkpoint unexpectedly contains optimizer state: {optimizer}")

    if not runner_log.is_file():
        raise ValueError(f"missing pilot runner log: {runner_log}")
    releases = []
    for match in _CACHE_RELEASE_PATTERN.finditer(runner_log.read_text(errors="replace")):
        before, after = int(match.group("before")), int(match.group("after"))
        if match.group("mode") not in {"offload", "recompute"}:
            continue
        if match.group("allocated") != "false" or after >= before:
            raise ValueError("invalid finish_generation cache release evidence in pilot log")
        releases.append(before - after)
    if not releases:
        raise ValueError("no verified finish_generation cache release found in pilot log")

    if not reload_log.is_file():
        raise ValueError(f"missing pilot reload log: {reload_log}")
    reload_text = reload_log.read_text(errors="replace")
    if f"Optimizer state not found at {optimizer}" not in reload_text:
        raise ValueError("reload did not resolve the expected model-only checkpoint")
    if "Optimizer will be freshly initialized." not in reload_text:
        raise ValueError("reload did not report fresh optimizer initialization")
    if "Dataset swap detected" in reload_text:
        raise ValueError("reload rejected the saved dataloader state as a dataset swap")
    if _TRAINING_STEP_PATTERN.search(reload_text) or "Training Results:" in reload_text:
        raise ValueError("reload process restarted training instead of restoring the completed step")

    return {
        "checkpoint": str(checkpoint),
        "restored_total_steps": expected_step,
        "optimizer_reinitialized": True,
        "cache_release_observations": len(releases),
        "maximum_observed_cache_release_mib": max(releases),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--runner-log", type=Path, required=True)
    parser.add_argument("--reload-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        summary = validate_rl_pilot(
            checkpoint_root=args.checkpoint_root,
            expected_step=args.expected_step,
            runner_log=args.runner_log,
            reload_log=args.reload_log,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
