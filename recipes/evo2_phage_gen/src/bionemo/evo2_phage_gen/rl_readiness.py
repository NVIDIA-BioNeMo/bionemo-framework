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
import copy
import importlib
import importlib.util
import json
import math
import warnings
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import yaml


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRPO_CONFIG = RECIPE_ROOT / "configs" / "grpo_phage_megatron.yaml"
PHAGE_OPENAI_DATASET = "bionemo.evo2_phage_gen.nemo_rl_processors.PhageOpenAIFormatDataset"


@dataclass(frozen=True)
class RLReadinessCheck:
    """Single NeMo-RL readiness check result."""

    name: str
    ok: bool
    required: bool
    detail: str


class RLEnvironmentControlError(ValueError):
    """Report an incomplete or invalid exact-environment control."""

    pass


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


def _phage_dataset_task_namespace_check(config: dict[str, Any]) -> RLReadinessCheck:
    """Require path-independent task names for every configured phage dataset."""
    data_config = config.get("data")
    defaults = data_config.get("default", {}) if isinstance(data_config, dict) else {}
    inspected: list[str] = []
    offenders: list[str] = []
    if isinstance(data_config, dict):
        for split in ("train", "validation"):
            raw_entries = data_config.get(split)
            if raw_entries is None:
                continue
            entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                env_name = entry.get("env_name", defaults.get("env_name"))
                if env_name != "phage_qc":
                    continue
                label = f"data.{split}" if len(entries) == 1 else f"data.{split}[{index}]"
                dataset_name = entry.get("dataset_name", defaults.get("dataset_name"))
                inspected.append(f"{label}.dataset_name={dataset_name!r}")
                if dataset_name != PHAGE_OPENAI_DATASET:
                    offenders.append(f"{label}.dataset_name={dataset_name!r}")

    detail = ", ".join(inspected) if inspected else "no phage_qc train or validation dataset"
    if offenders:
        detail = f"{', '.join(offenders)}; expected {PHAGE_OPENAI_DATASET!r}"
    return RLReadinessCheck(
        name="phage_dataset_task_namespace",
        ok=bool(inspected) and not offenders,
        required=True,
        detail=detail,
    )


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
        detail = module_name if ok else f"{module_name} not importable"
    except Exception as error:
        ok = False
        detail = f"{module_name} import discovery failed: {error}"
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
    try:
        if importlib.util.find_spec("torch") is None:
            return None
        import torch
    except Exception:
        return None

    return int(torch.cuda.device_count())


def _runtime_checks() -> list[RLReadinessCheck]:
    """Check NeMo-RL runtime imports."""
    nemo_rl_check = _module_check("nemo_rl", "nemo_rl", required=True)
    return [
        RLReadinessCheck(
            name="nemo_rl_install",
            ok=nemo_rl_check.ok,
            required=True,
            detail="nemo_rl is installed in the active environment" if nemo_rl_check.ok else nemo_rl_check.detail,
        ),
        nemo_rl_check,
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
        try:
            latest_iteration = int(latest_path.read_text().strip())
        except (OSError, ValueError):
            latest_iteration = None
        if latest_iteration is not None:
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

    try:
        run_config_data = yaml.safe_load(run_config.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        checks.append(
            RLReadinessCheck(
                name="checkpoint_bionemo_targets",
                ok=False,
                required=True,
                detail=f"could not read checkpoint run config {run_config}: {error}",
            )
        )
        return checks
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
    config_path: Path,
    *,
    require_evo2_adapter: bool,
    expected_gpus: int | None,
    checkpoint_override: Path | None = None,
    prompt_data_override: Path | None = None,
    gpus_per_node: int | None = None,
) -> list[RLReadinessCheck]:
    """Check files and settings referenced by the GRPO config."""
    checks = [_path_check("grpo_config", config_path, required=True)]
    if not config_path.exists():
        return checks

    try:
        config = _load_config_with_defaults(config_path)
    except (OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as error:
        checks.append(RLReadinessCheck("grpo_config_parse", ok=False, required=True, detail=str(error)))
        return checks

    checkpoint_path = checkpoint_override or _nested_get(config, ("checkpointing", "pretrained_checkpoint", "path"))
    prompt_data_path = (
        prompt_data_override
        if prompt_data_override is not None
        else _nested_get(config, ("data", "train", "data_path"))
    )
    checks.extend(
        [
            _path_check(
                "config_defaults",
                _config_relative_path(config_path, config.get("defaults")),
                required=True,
            ),
            _path_check("pretrained_checkpoint", checkpoint_path, required=True),
            _path_check("tokenizer", _nested_get(config, ("policy", "tokenizer", "name")), required=True),
            _path_check("prompt_data", prompt_data_path, required=True),
        ]
    )
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
    model_save_format = _nested_get(config, ("checkpointing", "model_save_format"))
    save_consolidated = bool(_nested_get(config, ("checkpointing", "save_consolidated"), False))
    dtensor_enabled = bool(_nested_get(config, ("policy", "dtensor_cfg", "enabled"), False))
    checkpoint_save_contract_ok = model_save_format is None and not save_consolidated and not dtensor_enabled
    checks.append(
        RLReadinessCheck(
            name="checkpoint_save_backend_contract",
            ok=checkpoint_save_contract_ok,
            required=True,
            detail=(
                "native Megatron-Bridge torch_dist checkpoint saving is configured"
                if checkpoint_save_contract_ok
                else (
                    "native Megatron-Bridge checkpoint saving requires "
                    "checkpointing.model_save_format=None, checkpointing.save_consolidated=False, "
                    "and policy.dtensor_cfg.enabled=False; got "
                    f"model_save_format={model_save_format!r}, save_consolidated={save_consolidated!r}, "
                    f"dtensor_cfg.enabled={dtensor_enabled!r}"
                )
            ),
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
    checks.append(_phage_dataset_task_namespace_check(config))

    configured_gpus_value = (
        gpus_per_node if gpus_per_node is not None else _nested_get(config, ("cluster", "gpus_per_node"), 1)
    )
    try:
        configured_gpus = int(configured_gpus_value)
    except (TypeError, ValueError):
        checks.append(
            RLReadinessCheck(
                name="cuda_gpus",
                ok=False,
                required=True,
                detail=f"cluster.gpus_per_node is not an integer: {configured_gpus_value!r}",
            )
        )
    else:
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
    checkpoint_override: Path | None = None,
    prompt_data_override: Path | None = None,
    gpus_per_node: int | None = None,
) -> list[RLReadinessCheck]:
    """Check whether the phage GRPO scaffold is ready to launch."""
    return [
        *_runtime_checks(),
        *_config_checks(
            config_path,
            require_evo2_adapter=require_evo2_adapter,
            expected_gpus=expected_gpus,
            checkpoint_override=checkpoint_override,
            prompt_data_override=prompt_data_override,
            gpus_per_node=gpus_per_node,
        ),
    ]


def _bounded_control_value(value: object, label: str) -> float:
    """Return one finite control value in the reward range."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise RLEnvironmentControlError(f"{label} is not a finite number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RLEnvironmentControlError(f"{label} is not a finite number in [0, 1]")
    return number


def _require_control_value(row: dict[str, Any], column: str, expected: float, label: str) -> None:
    """Require an exact binary support value from the scored control row."""
    value = _bounded_control_value(row.get(column), column)
    if value != expected:
        raise RLEnvironmentControlError(label)


def _validate_control_support(row: dict[str, Any], environment: Any) -> dict[str, bool]:
    """Confirm that each configured nontrivial scorer measured the control."""
    support: dict[str, bool] = {}
    external = environment.external_qc
    if external.enabled:
        _require_control_value(row, "external_qc_tool_succeeded", 1.0, "external QC did not complete")
        _require_control_value(row, "external_qc_measurement_available", 1.0, "external QC was not measured")
        support["external_qc"] = True
        external_components = (
            ("enable_orf", "orf", "reward_external_orf", None),
            ("enable_coding_density", "coding_density", "reward_external_coding_density", None),
            (
                "enable_protein_hit_count",
                "protein_database_hit_count",
                "reward_external_protein_hit_count",
                "protein_database_hit_count_measurement_available",
            ),
            ("enable_tropism", "tropism", "reward_external_tropism", "tropism_measurement_available"),
            ("enable_synteny", "synteny", "reward_external_synteny", "synteny_measurement_available"),
            (
                "enable_average_protein_identity",
                "average_protein_identity",
                "reward_external_average_protein_identity",
                "average_protein_identity_measurement_available",
            ),
            (
                "enable_required_genes",
                "required_genes",
                "reward_external_required_genes",
                "required_genes_measurement_available",
            ),
        )
        for enabled_attr, name, reward_column, measurement_column in external_components:
            if not getattr(external, enabled_attr):
                continue
            _bounded_control_value(row.get(reward_column), reward_column)
            if measurement_column is not None:
                _require_control_value(row, measurement_column, 1.0, f"{name} was not measured")
            support[name] = True

    if environment.mmseqs_cluster_diversity.enabled:
        _bounded_control_value(row.get("reward_mmseqs_cluster_diversity"), "reward_mmseqs_cluster_diversity")
        _require_control_value(
            row,
            "mmseqs_cluster_valid_for_clustering",
            1.0,
            "MMseqs diversity control was not eligible for clustering",
        )
        _require_control_value(
            row,
            "mmseqs_cluster_missing_from_output",
            0.0,
            "MMseqs diversity control was missing from clustering output",
        )
        if not isinstance(row.get("mmseqs_cluster_size"), Real) or row["mmseqs_cluster_size"] < 1:
            raise RLEnvironmentControlError("MMseqs diversity control has no cluster assignment")
        support["mmseqs_cluster_diversity"] = True

    _require_control_value(row, "safety_environment_healthy", 1.0, "sequence safety environment was not healthy")
    _require_control_value(row, "safety_gate_measurement_available", 1.0, "sequence safety was not measured")
    _require_control_value(row, "safety_gate_pass", 1.0, "known viable control did not pass sequence safety")
    if row.get("safety_gate_state") != "PASS":
        raise RLEnvironmentControlError("known viable control did not pass sequence safety")
    for safety_class in ("amr", "toxin", "lysogeny"):
        prefix = f"safety_{safety_class}"
        _bounded_control_value(row.get(f"reward_{prefix}"), f"reward_{prefix}")
        _require_control_value(row, f"{prefix}_measurement_available", 1.0, f"{safety_class} was not measured")
        if row.get(f"{prefix}_execution_status") != "COMPLETED_AND_PARSED":
            raise RLEnvironmentControlError(f"{safety_class} detector did not complete")
        if row.get(f"{prefix}_state") != "PASS":
            raise RLEnvironmentControlError(f"known viable control did not pass {safety_class}")
        support[f"safety_{safety_class}"] = True
    return support


def run_environment_control(config_path: Path, control_fasta: Path, output_dir: Path) -> dict[str, Any]:
    """Run one known viable genome through the exact configured NeMo-RL environment."""
    from bionemo.evo2_phage_gen import nemo_rl_env
    from bionemo.evo2_phage_gen.generation import DEFAULT_PROMPT_PREFIX
    from bionemo.evo2_phage_gen.sequence_safety_cli import parse_fasta_records

    config = _load_config_with_defaults(config_path)
    raw_environment = _nested_get(config, ("env", "phage_qc"))
    if not isinstance(raw_environment, dict):
        raise RLEnvironmentControlError("config has no phage_qc environment mapping")
    environment_config = copy.deepcopy(raw_environment)
    if environment_config.get("reward_output_mode") != "gdpo":
        raise RLEnvironmentControlError("exact environment control requires GDPO reward output")

    records = parse_fasta_records(control_fasta)
    if len(records) != 1:
        raise RLEnvironmentControlError("control FASTA must contain exactly one genome")
    record = records[0]
    prefix_length = 16
    if len(record.sequence) <= prefix_length:
        raise RLEnvironmentControlError("control genome must be longer than the 16-base RL prompt")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for section, subdirectory in (
        ("external_qc", "external-qc"),
        ("mmseqs_cluster_diversity", "mmseqs"),
        ("sequence_safety", "safety"),
    ):
        section_config = environment_config.get(section)
        if not isinstance(section_config, dict):
            if section == "sequence_safety":
                raise RLEnvironmentControlError("sequence_safety configuration is required")
            section_config = {}
            environment_config[section] = section_config
        section_config["work_dir"] = str(output_dir / subdirectory)

    actor = getattr(nemo_rl_env.PhageQCEnvironment, "__ray_metadata__", None)
    environment_class = getattr(actor, "modified_class", None)
    if environment_class is None:
        raise RLEnvironmentControlError("NeMo-RL environment is unavailable in the active runtime")
    environment = environment_class(environment_config)
    messages = [
        [
            {"role": "user", "content": DEFAULT_PROMPT_PREFIX + record.sequence[:prefix_length]},
            {"role": "assistant", "content": record.sequence[prefix_length:]},
        ]
    ]
    environment_result = environment_class.step(environment, messages, [{"record_id": record.sequence_id}])
    if environment_result.answers != [record.sequence]:
        raise RLEnvironmentControlError("exact environment did not reconstruct the complete control genome")
    if len(environment_result.metadata) != 1:
        raise RLEnvironmentControlError("exact environment returned the wrong control result count")
    scored = environment_result.metadata[0].get("_phage_qc_scored")
    if not isinstance(scored, dict):
        raise RLEnvironmentControlError("exact environment did not return component telemetry")

    reward_rows = environment_result.rewards.detach().cpu().tolist()
    if len(reward_rows) != 1 or not isinstance(reward_rows[0], list):
        raise RLEnvironmentControlError("exact environment did not return one GDPO reward vector")
    objective_names = [objective.name for objective in environment.gdpo_objectives]
    if len(reward_rows[0]) != len(objective_names):
        raise RLEnvironmentControlError("GDPO reward vector does not match the configured objectives")
    objective_values = {
        name: _bounded_control_value(value, f"objective {name}")
        for name, value in zip(objective_names, reward_rows[0], strict=True)
    }
    for objective in environment.gdpo_objectives:
        for column in objective.columns:
            _bounded_control_value(scored.get(column), column)

    result: dict[str, Any] = {
        "record_id": record.sequence_id,
        "sequence_length": len(record.sequence),
        "objectives": objective_values,
        "support": _validate_control_support(scored, environment),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


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
    parser.add_argument("--gpus-per-node", type=int, help="Check an effective GPU count that overrides the config")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Check this selected SFT checkpoint instead of the template checkpoint path",
    )
    parser.add_argument(
        "--prompt-data",
        type=Path,
        help="Check this generated RL training prompt bank instead of the template prompt path",
    )
    parser.add_argument(
        "--allow-template-gaps",
        action="store_true",
        help="Report the known Evo2 policy-adapter gap as optional instead of failing",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--warn-only", action="store_true", help="Report missing required checks without failing")
    parser.add_argument("--control-fasta", type=Path, help="Run one known viable genome through the exact environment")
    parser.add_argument("--control-dir", type=Path, help="Write exact-environment control work and result files here")
    args = parser.parse_args()
    if (args.control_fasta is None) != (args.control_dir is None):
        parser.error("--control-fasta and --control-dir must be provided together")

    checks = check_rl_readiness(
        args.config,
        require_evo2_adapter=not args.allow_template_gaps,
        checkpoint_override=args.checkpoint,
        prompt_data_override=args.prompt_data,
        gpus_per_node=args.gpus_per_node,
    )
    missing_required = [check for check in checks if check.required and not check.ok]
    if args.control_fasta is not None and not missing_required:
        try:
            run_environment_control(args.config, args.control_fasta, args.control_dir)
        except Exception as error:
            checks.append(
                RLReadinessCheck(
                    name="rl_environment_control",
                    ok=False,
                    required=True,
                    detail=f"{type(error).__name__}: {error}",
                )
            )
        else:
            checks.append(
                RLReadinessCheck(
                    name="rl_environment_control",
                    ok=True,
                    required=True,
                    detail=str((args.control_dir / "result.json").resolve()),
                )
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
