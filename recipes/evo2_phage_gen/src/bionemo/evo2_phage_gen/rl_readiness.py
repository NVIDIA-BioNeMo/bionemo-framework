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

"""Readiness checks for the Evo2 phage NeMo-RL scaffold."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRPO_CONFIG = RECIPE_ROOT / "configs" / "grpo_phage_megatron.yaml"


@dataclass(frozen=True)
class RLReadinessCheck:
    """Single NeMo-RL readiness check result."""

    name: str
    ok: bool
    required: bool
    detail: str


def _nested_get(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Read a nested config value without requiring OmegaConf."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _recipe_path(path_like: str | Path | None) -> Path | None:
    """Resolve relative recipe config paths from the recipe root."""
    if path_like in (None, ""):
        return None
    path = Path(path_like)
    return path if path.is_absolute() else RECIPE_ROOT / path


def _path_check(name: str, path_like: str | Path | None, *, required: bool) -> RLReadinessCheck:
    """Create a path-existence readiness check."""
    path = _recipe_path(path_like)
    if path is None:
        return RLReadinessCheck(name=name, ok=False, required=required, detail="missing config value")
    return RLReadinessCheck(name=name, ok=path.exists(), required=required, detail=str(path))


def _config_relative_path(config_path: Path, path_like: str | Path | None) -> Path | None:
    """Resolve a path the same way NeMo-RL config inheritance resolves it."""
    if path_like in (None, ""):
        return None
    path = Path(path_like)
    return path if path.is_absolute() else config_path.parent / path


def _merge_config_mappings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge one recipe config overlay onto its inherited defaults."""
    merged = dict(base)
    for key, value in override.items():
        inherited = merged.get(key)
        if isinstance(inherited, dict) and isinstance(value, dict):
            merged[key] = _merge_config_mappings(inherited, value)
        else:
            merged[key] = value
    return merged


def _load_config_with_defaults(config_path: Path, *, ancestors: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load the recipe's string-valued defaults chain using NeMo-RL overlay semantics."""
    resolved_path = config_path.resolve()
    if resolved_path in ancestors:
        chain = " -> ".join(str(path) for path in (*ancestors, resolved_path))
        raise ValueError(f"config defaults contain a cycle: {chain}")
    config = yaml.safe_load(resolved_path.read_text())
    if not isinstance(config, dict):
        raise TypeError(f"config is not a mapping: {resolved_path}")
    defaults = config.get("defaults")
    if defaults is None:
        return config
    if not isinstance(defaults, str) or not defaults:
        raise TypeError("recipe config defaults must be a non-empty string")
    defaults_path = _config_relative_path(resolved_path, defaults)
    if defaults_path is None or not defaults_path.exists():
        return config
    inherited = _load_config_with_defaults(defaults_path, ancestors=(*ancestors, resolved_path))
    return _merge_config_mappings(inherited, config)


def _module_check(name: str, module_name: str, *, required: bool) -> RLReadinessCheck:
    """Create a Python import-spec readiness check without importing the module."""
    try:
        ok = importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        ok = False
    detail = module_name if ok else f"{module_name} not importable"
    return RLReadinessCheck(name=name, ok=ok, required=required, detail=detail)


def _target_importable(target: str) -> bool:
    """Return true when a Hydra-style ``_target_`` can be imported."""
    module_name, _, attr_name = target.rpartition(".")
    if not module_name or not attr_name:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = importlib.import_module(module_name)
    except Exception:
        return False
    return hasattr(module, attr_name)


def _iter_target_values(value: Any) -> list[str]:
    """Collect Hydra-style ``_target_`` values from nested YAML data."""
    targets: list[str] = []
    if isinstance(value, dict):
        target = value.get("_target_")
        if isinstance(target, str):
            targets.append(target)
        for nested_value in value.values():
            targets.extend(_iter_target_values(nested_value))
    elif isinstance(value, list):
        for nested_value in value:
            targets.extend(_iter_target_values(nested_value))
    return targets


def _cuda_device_count() -> int | None:
    """Return CUDA device count, or ``None`` when PyTorch is unavailable."""
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return int(torch.cuda.device_count())


def _runtime_checks() -> list[RLReadinessCheck]:
    """Check NeMo-RL runtime imports."""
    nemo_rl_importable = importlib.util.find_spec("nemo_rl") is not None
    return [
        RLReadinessCheck(
            name="nemo_rl_install",
            ok=nemo_rl_importable,
            required=True,
            detail="nemo_rl is installed in the active environment"
            if nemo_rl_importable
            else "nemo_rl not importable",
        ),
        _module_check("nemo_rl", "nemo_rl", required=True),
        _module_check("ray", "ray", required=True),
        _module_check("grpo_algorithm", "nemo_rl.algorithms.grpo", required=True),
    ]


def _resolve_run_config(checkpoint_path: str | Path | None) -> Path | None:
    """Resolve a Megatron Bridge checkpoint root or iteration dir to ``run_config.yaml``."""
    if checkpoint_path in (None, ""):
        return None
    path = _recipe_path(checkpoint_path)
    if path is None:
        return None
    if (path / "run_config.yaml").exists():
        return path / "run_config.yaml"
    latest_path = path / "latest_checkpointed_iteration.txt"
    if latest_path.exists():
        latest_iteration = int(latest_path.read_text().strip())
        run_config = path / f"iter_{latest_iteration:07d}" / "run_config.yaml"
        if run_config.exists():
            return run_config
    iter_run_configs = sorted(path.glob("iter_*/run_config.yaml"))
    return iter_run_configs[-1] if iter_run_configs else None


def _checkpoint_run_config_checks(checkpoint_path: str | Path | None) -> list[RLReadinessCheck]:
    """Check that checkpoint config targets needed by NeMo-RL workers are importable."""
    run_config = _resolve_run_config(checkpoint_path)
    checks = [
        _path_check(
            "checkpoint_run_config",
            run_config,
            required=True,
        )
    ]
    if run_config is None or not run_config.exists():
        return checks

    run_config_data = yaml.safe_load(run_config.read_text())
    targets = sorted(
        {
            target
            for target in _iter_target_values(run_config_data)
            if target.startswith(("bionemo.evo2", "bionemo.common"))
        }
    )
    missing_targets = [target for target in targets if not _target_importable(target)]
    checks.append(
        RLReadinessCheck(
            name="checkpoint_bionemo_targets",
            ok=not missing_targets,
            required=True,
            detail=(
                f"{len(targets)} target(s) importable from {run_config}"
                if not missing_targets
                else "missing import target(s): " + ", ".join(missing_targets)
            ),
        )
    )
    return checks


def _checkpoint_iteration_resolution_check(checkpoint_path: str | Path | None) -> RLReadinessCheck:
    """Accept either a checkpoint-root tracker or a selected Bridge iteration directory."""
    path = _recipe_path(checkpoint_path)
    if path is None:
        return RLReadinessCheck(
            name="checkpoint_latest_iteration",
            ok=False,
            required=True,
            detail="missing checkpoint path",
        )
    if (path / "run_config.yaml").exists():
        return RLReadinessCheck(
            name="checkpoint_latest_iteration",
            ok=True,
            required=True,
            detail=f"specific Megatron Bridge iteration directory: {path}",
        )
    tracker = path / "latest_checkpointed_iteration.txt"
    return RLReadinessCheck(
        name="checkpoint_latest_iteration",
        ok=tracker.exists(),
        required=True,
        detail=str(tracker),
    )


def _config_checks(
    config_path: Path, *, require_evo2_adapter: bool, expected_gpus: int | None
) -> list[RLReadinessCheck]:
    """Check files and settings referenced by the GRPO config."""
    checks = [_path_check("grpo_config", config_path, required=True)]
    if not config_path.exists():
        return checks

    try:
        config = _load_config_with_defaults(config_path)
    except (TypeError, ValueError) as error:
        checks.append(RLReadinessCheck("grpo_config_parse", ok=False, required=True, detail=str(error)))
        return checks

    checks.extend(
        [
            _path_check(
                "config_defaults",
                _config_relative_path(config_path, config.get("defaults")),
                required=True,
            ),
            _path_check(
                "pretrained_checkpoint",
                _nested_get(config, ("checkpointing", "pretrained_checkpoint", "path")),
                required=True,
            ),
            _path_check("tokenizer", _nested_get(config, ("policy", "tokenizer", "name")), required=True),
            _path_check("prompt_data", _nested_get(config, ("data", "train", "data_path")), required=True),
        ]
    )
    checkpoint_path = _nested_get(config, ("checkpointing", "pretrained_checkpoint", "path"))
    checks.extend(
        [
            _checkpoint_iteration_resolution_check(checkpoint_path),
            *_checkpoint_run_config_checks(checkpoint_path),
        ]
    )

    generation_backend = _nested_get(config, ("policy", "generation", "backend"))
    checks.append(
        RLReadinessCheck(
            name="generation_backend",
            ok=generation_backend == "megatron",
            required=True,
            detail=f"policy.generation.backend={generation_backend!r}",
        )
    )
    colocated_generation = _nested_get(config, ("policy", "generation", "colocated", "enabled"))
    checks.append(
        RLReadinessCheck(
            name="megatron_generation_topology",
            ok=generation_backend != "megatron" or colocated_generation is not False,
            required=True,
            detail=(
                "colocated inherited from NeMo-RL defaults; GRPO reuses the training policy for generation"
                if colocated_generation is None
                else f"policy.generation.colocated.enabled={colocated_generation!r}"
            ),
        )
    )
    megatron_enabled = bool(_nested_get(config, ("policy", "megatron_cfg", "enabled"), False))
    checks.append(
        RLReadinessCheck(
            name="megatron_policy",
            ok=megatron_enabled,
            required=True,
            detail=f"policy.megatron_cfg.enabled={megatron_enabled!r}",
        )
    )
    env_name = _nested_get(config, ("data", "train", "env_name"))
    checks.append(
        RLReadinessCheck(
            name="phage_qc_environment_config",
            ok=env_name == "phage_qc" and isinstance(_nested_get(config, ("env", "phage_qc")), dict),
            required=True,
            detail=f"data.train.env_name={env_name!r}",
        )
    )

    configured_gpus = int(_nested_get(config, ("cluster", "gpus_per_node"), 1))
    available_gpus = _cuda_device_count() if expected_gpus is None else expected_gpus
    checks.append(
        RLReadinessCheck(
            name="cuda_gpus",
            ok=available_gpus is not None and available_gpus >= configured_gpus,
            required=True,
            detail=f"available={available_gpus}, required={configured_gpus}",
        )
    )

    model_name = str(_nested_get(config, ("policy", "model_name"), ""))
    needs_adapter = model_name.startswith("bionemo/evo2")
    patch_path = RECIPE_ROOT / "patches" / "nemo-rl-evo2-mbridge-grpo.patch"
    allowlist_prefixes = set(_nested_get(config, ("policy", "megatron_cfg", "target_allowlist_prefixes"), []) or [])
    required_prefixes = {"bionemo.evo2.", "bionemo.common."}
    patch_ok = patch_path.exists()
    allowlist_ok = required_prefixes.issubset(allowlist_prefixes)
    adapter_ok = (not needs_adapter) or (patch_ok and allowlist_ok)
    checks.append(
        RLReadinessCheck(
            name="evo2_policy_adapter",
            ok=adapter_ok,
            required=require_evo2_adapter,
            detail=(
                f"patch={patch_path}, target_allowlist_prefixes={sorted(allowlist_prefixes)}"
                if adapter_ok
                else "NeMo-RL Evo2/MBridge source patch or BioNeMo target allowlist prefixes are missing"
            ),
        )
    )
    return checks


def check_rl_readiness(
    config_path: Path = DEFAULT_GRPO_CONFIG,
    *,
    require_evo2_adapter: bool = True,
    expected_gpus: int | None = None,
) -> list[RLReadinessCheck]:
    """Check whether the phage GRPO scaffold is ready to launch."""
    return [
        *_runtime_checks(),
        *_config_checks(config_path, require_evo2_adapter=require_evo2_adapter, expected_gpus=expected_gpus),
    ]


def _print_text_report(checks: list[RLReadinessCheck]) -> None:
    """Print a concise human-readable readiness report."""
    for check in checks:
        severity = "required" if check.required else "optional"
        status = "ok" if check.ok else "missing"
        print(f"{status:7} {severity:8} {check.name}: {check.detail}")


def main() -> None:
    """CLI entry point for NeMo-RL readiness checks."""
    parser = argparse.ArgumentParser(description="Check prerequisites for the Evo2 phage NeMo-RL GRPO scaffold")
    parser.add_argument("--config", type=Path, default=DEFAULT_GRPO_CONFIG)
    parser.add_argument(
        "--allow-template-gaps",
        action="store_true",
        help="Report the known Evo2 policy-adapter gap as optional instead of failing",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--warn-only", action="store_true", help="Report missing required checks without failing")
    args = parser.parse_args()

    checks = check_rl_readiness(
        args.config,
        require_evo2_adapter=not args.allow_template_gaps,
    )
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        _print_text_report(checks)

    missing_required = [check for check in checks if check.required and not check.ok]
    if missing_required and not args.warn_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
