# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Lossless runtime-state augmentation for legacy MBridge checkpoints."""

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader

from bionemo.evo2.models.evo2_provider import HyenaModelProvider
from bionemo.evo2.utils.checkpoint.mbridge_to_vortex import _compute_inv_freq, load_mbridge_state_dict
from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import package_mbridge_checkpoint


@dataclass(frozen=True)
class MBridgeRuntimeStateAugmentation:
    """An augmented state dictionary and the exact keys added to it."""

    state_dict: OrderedDict[str, Any]
    added_keys: tuple[str, ...]


@dataclass(frozen=True)
class MBridgeCheckpointUpgradeReceipt:
    """Identity and key inventory for a packaged checkpoint upgrade."""

    source_checkpoint: Path
    destination_checkpoint: Path
    source_key_count: int
    destination_key_count: int
    added_keys: tuple[str, ...]
    source_run_config_sha256: str
    destination_run_config_sha256: str


def augment_mbridge_runtime_state(
    source_state_dict: dict[str, Any],
    model_provider: HyenaModelProvider,
) -> MBridgeRuntimeStateAugmentation:
    """Add missing deterministic runtime entries without changing source values."""
    state_dict = OrderedDict(source_state_dict.items())
    added_keys: list[str] = []

    def add_if_missing(key: str, value: Any) -> None:
        if key not in state_dict:
            state_dict[key] = value
            added_keys.append(key)

    rotary_dim = model_provider.hidden_size // model_provider.num_attention_heads
    expected_inv_freq = _compute_inv_freq(rotary_dim, float(model_provider.rotary_base))

    for layer_index, symbol in enumerate(model_provider.hybrid_override_pattern):
        prefix = f"decoder.layers.{layer_index}"
        if symbol == "*":
            for suffix in (
                "self_attention.core_attention._extra_state",
                "self_attention.linear_qkv._extra_state",
                "self_attention.linear_proj._extra_state",
            ):
                add_if_missing(f"{prefix}.{suffix}", torch.empty(0, dtype=torch.uint8))

            rotary_key = f"{prefix}.self_attention.rotary_emb.inv_freq"
            if rotary_key in state_dict:
                existing_inv_freq = state_dict[rotary_key]
                if type(existing_inv_freq) is not torch.Tensor or not torch.equal(
                    existing_inv_freq, expected_inv_freq
                ):
                    raise ValueError(f"{rotary_key} conflicts with provider-derived value")
            else:
                add_if_missing(rotary_key, expected_inv_freq.clone())
        else:
            for suffix in (
                "mixer.dense_projection._extra_state",
                "mixer.dense._extra_state",
            ):
                add_if_missing(f"{prefix}.{suffix}", torch.empty(0, dtype=torch.uint8))

        for suffix in ("mlp.linear_fc1._extra_state", "mlp.linear_fc2._extra_state"):
            add_if_missing(f"{prefix}.{suffix}", torch.empty(0, dtype=torch.uint8))

    add_if_missing("decoder.final_norm._extra_state", torch.empty(0, dtype=torch.uint8))
    add_if_missing("output_layer._extra_state", None)
    return MBridgeRuntimeStateAugmentation(state_dict=state_dict, added_keys=tuple(added_keys))


def _resolve_iteration_dir(checkpoint: Path) -> Path:
    if re.fullmatch(r"iter_\d+", checkpoint.name):
        return checkpoint
    latest_path = checkpoint / "latest_checkpointed_iteration.txt"
    if latest_path.is_file():
        iteration_text = latest_path.read_text(encoding="ascii").strip()
        if not iteration_text.isdecimal():
            raise ValueError(f"invalid latest checkpoint iteration: {iteration_text!r}")
        return (checkpoint / f"iter_{int(iteration_text):07d}").resolve(strict=True)
    iteration_dirs = sorted(checkpoint.glob("iter_*"))
    if not iteration_dirs:
        raise FileNotFoundError(f"no iter_* directory in {checkpoint}")
    return iteration_dirs[-1].resolve(strict=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_metadata_keys(iteration_dir: Path) -> set[str]:
    return set(FileSystemReader(str(iteration_dir)).read_metadata().state_dict_metadata)


def upgrade_mbridge_checkpoint(
    *,
    source_checkpoint: Path,
    destination_checkpoint: Path,
    model_provider: HyenaModelProvider,
    tokenizer_path: Path,
    mixed_precision_recipe: str = "bf16_mixed",
) -> MBridgeCheckpointUpgradeReceipt:
    """Repackage a legacy model checkpoint with current deterministic runtime state."""
    source_checkpoint = Path(source_checkpoint).resolve(strict=True)
    destination_checkpoint = Path(destination_checkpoint).resolve(strict=False)
    if destination_checkpoint.exists():
        raise FileExistsError(f"destination checkpoint already exists: {destination_checkpoint}")

    source_iteration = _resolve_iteration_dir(source_checkpoint)
    source_run_config = source_iteration / "run_config.yaml"
    source_metadata_keys = _checkpoint_metadata_keys(source_iteration)
    source_state_dict = load_mbridge_state_dict(source_checkpoint)
    augmentation = augment_mbridge_runtime_state(source_state_dict, model_provider)
    package_mbridge_checkpoint(
        augmentation.state_dict,
        mbridge_ckpt_dir=destination_checkpoint,
        model_provider=model_provider,
        tokenizer_path=Path(tokenizer_path),
        mixed_precision_recipe=mixed_precision_recipe,
    )
    destination_iteration = destination_checkpoint / "iter_0000001"
    destination_run_config = destination_iteration / "run_config.yaml"
    destination_metadata_keys = _checkpoint_metadata_keys(destination_iteration)
    return MBridgeCheckpointUpgradeReceipt(
        source_checkpoint=source_checkpoint,
        destination_checkpoint=destination_checkpoint,
        source_key_count=len(source_metadata_keys),
        destination_key_count=len(destination_metadata_keys),
        added_keys=tuple(sorted(destination_metadata_keys - source_metadata_keys)),
        source_run_config_sha256=_sha256(source_run_config),
        destination_run_config_sha256=_sha256(destination_run_config),
    )
