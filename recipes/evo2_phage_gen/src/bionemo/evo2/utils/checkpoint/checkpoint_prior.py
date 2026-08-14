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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: recipes/evo2_megatron/src/bionemo/evo2/utils/checkpoint/checkpoint_prior.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

"""Analyze ambiguous checkpoint-inversion priors for Evo2 Hyena filters."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

from bionemo.evo2.utils.checkpoint.vortex_to_mbridge import _compute_initial_medium_decay


FILTER_SUFFIXES = ("filter.p", "filter.gamma", "filter.R", "filter.h", "filter.decay")


def _resolve_iter_dir(checkpoint_dir: Path) -> Path:
    """Resolve a checkpoint root or iteration directory to an iteration directory."""
    if (checkpoint_dir / ".metadata").exists():
        return checkpoint_dir
    if (checkpoint_dir / "weights" / ".metadata").exists():
        return checkpoint_dir / "weights"
    if re.match(r"^iter_\d+$", checkpoint_dir.name):
        return checkpoint_dir
    if (latest_file := checkpoint_dir / "latest_checkpointed_iteration.txt").exists():
        iteration = latest_file.read_text().strip()
        return checkpoint_dir / f"iter_{int(iteration):07d}"
    iter_dirs = sorted(checkpoint_dir.glob("iter_*"))
    if not iter_dirs:
        raise FileNotFoundError(f"No iter_* directories in {checkpoint_dir}")
    return iter_dirs[-1]


def load_filter_tensors(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load only Evo2 Hyena filter tensors from a DCP checkpoint."""
    iter_dir = _resolve_iter_dir(checkpoint_dir)
    reader = FileSystemReader(str(iter_dir))
    metadata = reader.read_metadata()

    state_dict: dict[str, torch.Tensor] = {}
    for key, item_meta in metadata.state_dict_metadata.items():
        if isinstance(item_meta, BytesStorageMetadata):
            continue
        if key.endswith(FILTER_SUFFIXES):
            state_dict[key] = torch.empty(item_meta.size, dtype=item_meta.properties.dtype, device="cpu")

    dcp.load(state_dict=state_dict, storage_reader=reader, no_dist=True)
    return state_dict


def _tensor_stats(tensors: list[torch.Tensor]) -> dict[str, Any]:
    """Return compact scalar stats for a list of tensors."""
    if not tensors:
        return {"count": 0}
    flat = torch.cat([tensor.detach().to(torch.float32).reshape(-1) for tensor in tensors])
    quantile_points = torch.tensor([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999], dtype=torch.float32)
    quantiles = torch.quantile(flat, quantile_points)
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "quantiles": {f"{float(q):.3f}": float(v.item()) for q, v in zip(quantile_points, quantiles)},
    }


def _group_by_suffix(filter_tensors: dict[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    """Group filter tensors by final parameter name."""
    grouped: dict[str, list[torch.Tensor]] = {"p": [], "gamma": [], "R": [], "h": [], "decay": []}
    for key, tensor in filter_tensors.items():
        suffix = key.rsplit(".", 1)[-1]
        if suffix in grouped:
            grouped[suffix].append(tensor)
    return grouped


def _gamma_prior_stats(gamma_tensors: list[torch.Tensor], gamma_min: float, gamma_max: float) -> dict[str, Any]:
    """Measure trained gamma values against ``log(U(gamma_min, gamma_max))`` support."""
    if not gamma_tensors:
        return {"count": 0}
    gamma = torch.cat([tensor.detach().to(torch.float32).reshape(-1) for tensor in gamma_tensors])
    log_min = torch.tensor(gamma_min, dtype=torch.float32).log()
    log_max = torch.tensor(gamma_max, dtype=torch.float32).log()
    log_mid = (log_min + log_max) / 2
    below = torch.clamp(log_min - gamma, min=0)
    above = torch.clamp(gamma - log_max, min=0)
    outside = below + above
    return {
        "init_log_min": float(log_min.item()),
        "init_log_max": float(log_max.item()),
        "init_log_mid": float(log_mid.item()),
        "fraction_inside_init_range": float(((gamma >= log_min) & (gamma <= log_max)).to(torch.float32).mean().item()),
        "mean_abs_distance_to_log_mid": float((gamma - log_mid).abs().mean().item()),
        "mean_distance_outside_range": float(outside.mean().item()),
        "max_distance_outside_range": float(outside.max().item()),
    }


def _p_prior_stats(p_tensors: list[torch.Tensor]) -> dict[str, Any]:
    """Measure trained p values against the Evo2 initialization value, -1."""
    if not p_tensors:
        return {"count": 0}
    p = torch.cat([tensor.detach().to(torch.float32).reshape(-1) for tensor in p_tensors])
    return {
        "init_value": -1.0,
        "mean_abs_distance_to_init": float((p + 1.0).abs().mean().item()),
        "median_abs_distance_to_init": float((p + 1.0).abs().median().item()),
    }


def _decay_prior_stats(decay_tensors: list[torch.Tensor], decay_preset: str) -> dict[str, Any]:
    """Measure trained medium-filter decay values against deterministic initialization."""
    if not decay_tensors:
        return {"count": 0}
    per_tensor = []
    for tensor in decay_tensors:
        decay = tensor.detach().to(torch.float32)
        expected = _compute_initial_medium_decay(decay.shape[0], decay.shape[1], decay_preset=decay_preset)
        diff = decay - expected
        per_tensor.append(
            {
                "shape": list(decay.shape),
                "rmse_to_init": float(torch.sqrt(torch.mean(diff.square())).item()),
                "relative_rmse_to_init": float(
                    (torch.sqrt(torch.mean(diff.square())) / torch.sqrt(torch.mean(expected.square()))).item()
                ),
                "max_abs_distance_to_init": float(diff.abs().max().item()),
            }
        )
    return {"per_tensor": per_tensor}


def analyze_filter_priors(
    checkpoint_dir: Path,
    gamma_min: float = 0.01,
    gamma_max: float = 0.1,
    decay_preset: str = "weak",
) -> dict[str, Any]:
    """Analyze real checkpoint filters against inversion priors."""
    if gamma_min <= 0:
        raise ValueError("gamma_min must be positive")
    if gamma_min >= gamma_max:
        raise ValueError("gamma_min must be less than gamma_max")

    filter_tensors = load_filter_tensors(checkpoint_dir)
    grouped = _group_by_suffix(filter_tensors)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "num_filter_tensors": len(filter_tensors),
        "stats": {name: _tensor_stats(tensors) for name, tensors in grouped.items()},
        "priors": {
            "gamma": _gamma_prior_stats(grouped["gamma"], gamma_min=gamma_min, gamma_max=gamma_max),
            "p": _p_prior_stats(grouped["p"]),
            "decay": _decay_prior_stats(grouped["decay"], decay_preset=decay_preset),
        },
    }


def _sanitize_nonfinite_for_json(value: Any) -> tuple[Any, int]:
    """Recursively replace non-finite floats with null-compatible values."""
    if isinstance(value, float):
        return (value, 0) if math.isfinite(value) else (None, 1)
    if isinstance(value, dict):
        sanitized = {}
        nonfinite_count = 0
        for key, item in value.items():
            sanitized_item, item_count = _sanitize_nonfinite_for_json(item)
            sanitized[key] = sanitized_item
            nonfinite_count += item_count
        return sanitized, nonfinite_count
    if isinstance(value, (list, tuple)):
        sanitized_items = []
        nonfinite_count = 0
        for item in value:
            sanitized_item, item_count = _sanitize_nonfinite_for_json(item)
            sanitized_items.append(sanitized_item)
            nonfinite_count += item_count
        return sanitized_items, nonfinite_count
    return value, 0


def main() -> None:
    """CLI entry point for filter-prior analysis."""
    parser = argparse.ArgumentParser(description="Analyze Evo2 Hyena filter values against inversion priors")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="MBridge/Nemo DCP checkpoint directory")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save the JSON report")
    parser.add_argument("--gamma-min", type=float, default=0.01)
    parser.add_argument("--gamma-max", type=float, default=0.1)
    parser.add_argument("--decay-preset", type=str, default="weak", choices=["weak", "normal", "strong"])
    args = parser.parse_args()

    report = analyze_filter_priors(
        args.checkpoint_dir,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        decay_preset=args.decay_preset,
    )
    sanitized_report, nonfinite_value_count = _sanitize_nonfinite_for_json(report)
    sanitized_report["nonfinite_value_count"] = nonfinite_value_count
    text = json.dumps(sanitized_report, indent=2, sort_keys=True, allow_nan=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
