# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Preflight parity between an RL MBridge checkpoint and its vLLM export."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata

from bionemo.evo2.vllm.artifact_io import read_json_snapshot
from bionemo.evo2.vllm.export import infer_evo2_config, resolve_iteration_dir
from bionemo.evo2.vllm.infer import load_export_identity, resolve_tokenizer_json


_MODEL_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "hybrid_override_pattern",
    "short_conv_length",
    "hcs_filter_length",
    "hcm_filter_length",
    "hcl_state_size",
    "num_groups_hyena",
    "num_groups_hyena_medium",
    "num_groups_hyena_short",
    "rms_norm_eps",
    "rotary_base",
    "rope_theta",
    "seq_len_interpolation_factor",
    "use_short_conv_bias",
    "hidden_act",
    "gelu_approximate",
    "gated_linear_unit",
    "remove_activation_post_first_layer",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_nbytes(metadata: Any) -> int:
    element_size = torch.empty((), dtype=metadata.properties.dtype).element_size()
    return math.prod(int(dimension) for dimension in metadata.size) * element_size


def _source_tensor_inventory(iteration_dir: Path) -> tuple[dict[str, Any], int]:
    checkpoint_metadata = FileSystemReader(str(iteration_dir)).read_metadata()
    tensors = {
        name: metadata
        for name, metadata in checkpoint_metadata.state_dict_metadata.items()
        if not isinstance(metadata, BytesStorageMetadata)
    }
    if not tensors:
        raise RuntimeError(f"RL checkpoint contains no tensor weights: {iteration_dir}")
    return tensors, sum(_tensor_nbytes(metadata) for metadata in tensors.values())


def validate_rl_inference_load_parity(
    *,
    checkpoint: str | Path,
    export: str | Path,
    rl_tokenizer: str | Path,
) -> dict[str, Any]:
    """Require one RL checkpoint and standalone export to describe the same model input state."""
    iteration_dir = resolve_iteration_dir(checkpoint)
    export_identity = load_export_identity(export)
    export_root = export_identity.root
    manifest_snapshot = read_json_snapshot(export_root / "manifest.json", label="Evo2 export manifest")
    config_snapshot = read_json_snapshot(export_root / "config.json", label="Evo2 export config")
    index_snapshot = read_json_snapshot(
        export_root / "model.safetensors.index.json",
        label="Evo2 safetensors index",
    )
    manifest = manifest_snapshot.value
    export_config = config_snapshot.value
    index = index_snapshot.value
    if type(manifest) is not dict or type(export_config) is not dict or type(index) is not dict:
        raise RuntimeError("Evo2 export identity files must be JSON objects")

    run_config_path = iteration_dir / "run_config.yaml"
    run_config_sha256 = _sha256(run_config_path)
    if manifest.get("run_config_sha256") != run_config_sha256:
        raise RuntimeError("RL checkpoint run config does not match the vLLM export manifest")
    source_iteration = int(iteration_dir.name.removeprefix("iter_"))
    if type(manifest.get("source_iteration")) is not int or manifest["source_iteration"] != source_iteration:
        raise RuntimeError("RL checkpoint iteration does not match the vLLM export manifest")

    inferred_config, model_provider = infer_evo2_config(iteration_dir)
    if manifest.get("model_provider") != model_provider:
        raise RuntimeError("RL checkpoint model provider does not match the vLLM export manifest")
    expected_config = inferred_config.to_dict()
    model_config_fields = {name: expected_config.get(name) for name in _MODEL_CONFIG_FIELDS}
    mismatched_config = {
        name: {"checkpoint": expected, "export": export_config.get(name)}
        for name, expected in model_config_fields.items()
        if export_config.get(name) != expected
    }
    if mismatched_config:
        raise RuntimeError(f"RL checkpoint and vLLM export model config differ: {mismatched_config}")

    source_tensors, source_total_size = _source_tensor_inventory(iteration_dir)
    weight_map = index.get("weight_map")
    index_metadata = index.get("metadata")
    if type(weight_map) is not dict or type(index_metadata) is not dict:
        raise RuntimeError("vLLM safetensors index inventory is malformed")
    if set(weight_map) != set(source_tensors):
        raise RuntimeError("RL checkpoint and vLLM export tensor names differ")
    if type(manifest.get("tensor_count")) is not int or manifest["tensor_count"] != len(source_tensors):
        raise RuntimeError("RL checkpoint and vLLM export tensor counts differ")
    if type(manifest.get("total_size")) is not int or manifest["total_size"] != source_total_size:
        raise RuntimeError("RL checkpoint and vLLM export tensor byte totals differ")
    if type(index_metadata.get("total_size")) is not int or index_metadata["total_size"] != source_total_size:
        raise RuntimeError("vLLM safetensors index total_size differs from the RL checkpoint")

    rl_tokenizer_path = resolve_tokenizer_json(export_root=export_root, tokenizer_json=rl_tokenizer)
    export_tokenizer_path = resolve_tokenizer_json(export_root=export_root, tokenizer_json=None)
    rl_tokenizer_snapshot = read_json_snapshot(rl_tokenizer_path, label="RL tokenizer")
    export_tokenizer_snapshot = read_json_snapshot(export_tokenizer_path, label="export tokenizer")
    if rl_tokenizer_snapshot.value != export_tokenizer_snapshot.value:
        raise RuntimeError("RL and standalone inference tokenizer semantics differ")
    tokenizer_semantic_sha256 = _canonical_json_sha256(rl_tokenizer_snapshot.value)

    return {
        "schema_version": 1,
        "checkpoint_iteration": str(iteration_dir),
        "export_root": str(export_root),
        "source_checkpoint_recorded_by_export": manifest.get("source_checkpoint"),
        "source_iteration": source_iteration,
        "source_run_config_sha256": run_config_sha256,
        "export_manifest_sha256": export_identity.manifest_sha256,
        "export_config_sha256": export_identity.config_sha256,
        "export_index_sha256": export_identity.index_sha256,
        "model_provider": model_provider,
        "model_config_fields": model_config_fields,
        "tensor_count": len(source_tensors),
        "tensor_total_size": source_total_size,
        "rl_tokenizer_path": str(rl_tokenizer_path),
        "rl_tokenizer_sha256": rl_tokenizer_snapshot.sha256,
        "export_tokenizer_path": str(export_tokenizer_path),
        "export_tokenizer_sha256": export_tokenizer_snapshot.sha256,
        "tokenizer_semantic_sha256": tokenizer_semantic_sha256,
    }


__all__ = ["validate_rl_inference_load_parity"]
