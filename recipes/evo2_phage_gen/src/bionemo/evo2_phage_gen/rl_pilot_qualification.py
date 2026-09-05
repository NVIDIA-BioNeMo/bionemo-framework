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

"""Validate the full-shape GDPO pilot's checkpoint restart and cache release."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml


_CACHE_RELEASE_PATTERN = re.compile(
    r"Evo2 native cache release verified: mode=(?P<mode>\S+) "
    r"tensor_state_allocated=(?P<allocated>true|false) "
    r"allocated_before_mib=(?P<before>\d+) allocated_after_mib=(?P<after>\d+)"
)
_WORKER_CACHE_MEMORY_PATTERN = re.compile(
    r"\[GPU Rank (?P<rank>\d+)\] finish_generation (?P<phase>START|END) \| "
    r"alloc=(?P<allocated>\d+)MiB"
)
_TRAINING_STEP_PATTERN = re.compile(r"={5,}\s+Step\s+\d+/\d+\s+={5,}")
_ANSI_CSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _compact_console_text(value: str) -> str:
    """Remove Rich styling and wrapping that may split long checkpoint paths."""
    return re.sub(r"\s+", "", _ANSI_CSI_PATTERN.sub("", value))


def _load_mapping(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a mapping: {path}")
    return payload


def _observed_cache_releases(runner_text: str, *, cache_mode: str) -> list[int]:
    """Return verified allocation drops from explicit or durable worker telemetry."""
    releases = []
    for match in _CACHE_RELEASE_PATTERN.finditer(runner_text):
        before, after = int(match.group("before")), int(match.group("after"))
        if match.group("mode") != cache_mode:
            continue
        if match.group("allocated") != "false" or after >= before:
            raise ValueError("invalid finish_generation cache release evidence in pilot log")
        releases.append(before - after)
    if releases:
        return releases

    starts_by_rank: dict[int, int] = {}
    for match in _WORKER_CACHE_MEMORY_PATTERN.finditer(runner_text):
        rank = int(match.group("rank"))
        allocated = int(match.group("allocated"))
        if match.group("phase") == "START":
            starts_by_rank[rank] = allocated
            continue
        before = starts_by_rank.pop(rank, None)
        if before is not None and allocated < before:
            releases.append(before - allocated)
    return releases


def validate_rl_pilot(
    *,
    checkpoint_root: Path,
    expected_step: int,
    runner_log: Path,
    reload_log: Path,
) -> dict[str, Any]:
    """Validate one complete checkpoint and a no-training reload process."""
    checkpoint_root = checkpoint_root.resolve()
    checkpoint = checkpoint_root / f"step_{expected_step}"
    training_info = _load_mapping(checkpoint / "training_info.json", description="training state")
    if training_info.get("total_steps") != expected_step:
        raise ValueError(f"checkpoint total_steps is {training_info.get('total_steps')!r}, expected {expected_step}")

    config = _load_mapping(checkpoint / "config.yaml", description="resolved checkpoint config")
    save_optimizer = config.get("checkpointing", {}).get("save_optimizer")
    if not isinstance(save_optimizer, bool):
        raise ValueError("pilot checkpoint must resolve checkpointing.save_optimizer to a boolean")
    cache_mode = (
        config.get("policy", {})
        .get("generation", {})
        .get("mcore_generation_config", {})
        .get("kv_cache_management_mode")
    )
    if cache_mode not in {"offload", "recompute"}:
        raise ValueError("pilot checkpoint must use a releasing native KV-cache mode")
    if not (checkpoint / "train_dataloader.pt").is_file():
        raise ValueError(f"missing saved dataloader state: {checkpoint / 'train_dataloader.pt'}")
    weights = checkpoint / "policy/weights/iter_0000000"
    if not (weights / ".metadata").is_file():
        raise ValueError(f"missing completed Megatron policy checkpoint metadata: {weights / '.metadata'}")
    optimizer = checkpoint / "policy/optimizer"
    common_pt = weights / "common.pt"
    embedded_optimizer = False
    if common_pt.is_file():
        try:
            common_state = torch.load(common_pt, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"could not inspect Megatron common state: {common_pt}: {error}") from error
        embedded_optimizer = isinstance(common_state, dict) and "optimizer" in common_state
    optimizer_state_saved = optimizer.exists() or embedded_optimizer
    if save_optimizer and not optimizer_state_saved:
        raise ValueError("full-state checkpoint has no DTensor or embedded Megatron optimizer state")
    if not save_optimizer and optimizer_state_saved:
        raise ValueError("model-only checkpoint unexpectedly contains optimizer state")

    if not runner_log.is_file():
        raise ValueError(f"missing pilot runner log: {runner_log}")
    releases = _observed_cache_releases(runner_log.read_text(errors="replace"), cache_mode=cache_mode)
    if not releases:
        raise ValueError("no verified finish_generation cache release found in pilot log")

    if not reload_log.is_file():
        raise ValueError(f"missing pilot reload log: {reload_log}")
    reload_text = reload_log.read_text(errors="replace")
    compact_reload_text = _compact_console_text(reload_text)
    fresh_optimizer = _compact_console_text("Optimizer will be freshly initialized.") in compact_reload_text
    if save_optimizer:
        if fresh_optimizer:
            raise ValueError("full-state reload discarded the saved optimizer state")
    else:
        if _compact_console_text(f"Optimizer state not found at {optimizer}") not in compact_reload_text:
            raise ValueError("reload did not resolve the expected model-only checkpoint")
        if not fresh_optimizer:
            raise ValueError("reload did not report fresh optimizer initialization")
    if _compact_console_text(f"successfully loaded checkpoint from {weights}") not in compact_reload_text:
        raise ValueError("reload did not report loading the expected policy weights")
    if _compact_console_text("Dataset swap detected") in compact_reload_text:
        raise ValueError("reload rejected the saved dataloader state as a dataset swap")
    if _TRAINING_STEP_PATTERN.search(reload_text) or "Training Results:" in reload_text:
        raise ValueError("reload process restarted training instead of restoring the completed step")

    return {
        "checkpoint": str(checkpoint),
        "restored_total_steps": expected_step,
        "optimizer_state_saved": optimizer_state_saved,
        "optimizer_reinitialized": fresh_optimizer,
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
    """Validate the configured pilot artifacts and write an atomic summary."""
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
