# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from bionemo.evo2.vllm.export import export_mbridge_to_vllm
from bionemo.evo2.vllm.load_parity import validate_rl_inference_load_parity


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_checkpoint(tmp_path: Path) -> Path:
    checkpoint_root = tmp_path / "checkpoint"
    iteration_dir = checkpoint_root / "iter_0000007"
    source = {
        "embedding.word_embeddings.weight": torch.arange(32, dtype=torch.bfloat16).reshape(8, 4),
        "decoder.layers.0.mixer.dense_projection.weight": torch.arange(48, dtype=torch.bfloat16).reshape(12, 4),
        "decoder.layers.1.mixer.mixer.filter.h": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "decoder.layers.1.mixer.mixer.filter.decay": torch.full((4, 4), 0.5, dtype=torch.float32),
        "decoder.final_norm.weight": torch.arange(4, dtype=torch.bfloat16),
    }
    dcp.save(state_dict=source, checkpoint_id=str(iteration_dir))
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("7\n", encoding="utf-8")
    (iteration_dir / "run_config.yaml").write_text(
        """model:
  _target_: bionemo.evo2.models.evo2_provider.HyenaTestModelProvider
  hidden_size: 4
  num_layers: 4
  num_attention_heads: 2
  ffn_hidden_size: 6
  vocab_size: 8
  seq_length: 10240
  hybrid_override_pattern: SDH*
  num_groups_hyena_short: 4
  num_groups_hyena_medium: 4
  rotary_base: 10000
  seq_len_interpolation_factor: 128
  layernorm_epsilon: 1.0e-6
  activation_func:
    _call_: false
    _target_: torch._C._nn.gelu
  gated_linear_unit: true
  remove_activation_post_first_layer: true
""",
        encoding="utf-8",
    )
    tokenizer_dir = iteration_dir / "tokenizer"
    tokenizer_dir.mkdir()
    _write_json(tokenizer_dir / "tokenizer.json", {"model": {"type": "WordLevel"}, "version": "1.0"})
    return checkpoint_root


def _export(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = _write_checkpoint(tmp_path)
    export = tmp_path / "export"
    export_mbridge_to_vllm(checkpoint, export, max_shard_size=160)
    return checkpoint, export


def test_load_parity_binds_checkpoint_config_tensor_inventory_and_semantic_tokenizer(tmp_path) -> None:
    checkpoint, export = _export(tmp_path)
    rl_tokenizer = tmp_path / "rl-tokenizer"
    rl_tokenizer.mkdir()
    # Different bytes, identical admitted tokenizer structure.
    (rl_tokenizer / "tokenizer.json").write_text(
        '{ "version": "1.0", "model": { "type": "WordLevel" } }',
        encoding="utf-8",
    )

    evidence = validate_rl_inference_load_parity(checkpoint=checkpoint, export=export, rl_tokenizer=rl_tokenizer)

    _require(evidence["source_iteration"] == 7, "source iteration changed")
    _require(evidence["tensor_count"] == 5, "source/export tensor inventory changed")
    _require(
        evidence["source_run_config_sha256"] == _sha256(checkpoint / "iter_0000007/run_config.yaml"),
        "run config changed",
    )
    _require(
        evidence["rl_tokenizer_sha256"] != evidence["export_tokenizer_sha256"],
        "test tokenizer bytes are not distinct",
    )
    _require(evidence["tokenizer_semantic_sha256"], "semantic tokenizer commitment is missing")
    _require(evidence["model_config_fields"]["seq_len_interpolation_factor"] == 128.0, "RoPE scaling changed")


def test_load_parity_rejects_checkpoint_run_config_drift(tmp_path) -> None:
    checkpoint, export = _export(tmp_path)
    run_config = checkpoint / "iter_0000007/run_config.yaml"
    run_config.write_text(run_config.read_text().replace("ffn_hidden_size: 6", "ffn_hidden_size: 8"), encoding="utf-8")

    with pytest.raises(RuntimeError, match="run config"):
        validate_rl_inference_load_parity(
            checkpoint=checkpoint,
            export=export,
            rl_tokenizer=checkpoint / "iter_0000007/tokenizer",
        )


def test_load_parity_rejects_coherent_export_config_drift(tmp_path) -> None:
    checkpoint, export = _export(tmp_path)
    config_path = export / "config.json"
    config = json.loads(config_path.read_text())
    config["intermediate_size"] = 8
    _write_json(config_path, config)
    manifest_path = export / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_sha256"] = _sha256(config_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="model config"):
        validate_rl_inference_load_parity(
            checkpoint=checkpoint,
            export=export,
            rl_tokenizer=checkpoint / "iter_0000007/tokenizer",
        )


def test_load_parity_rejects_tokenizer_semantic_drift(tmp_path) -> None:
    checkpoint, export = _export(tmp_path)
    rl_tokenizer = tmp_path / "rl-tokenizer"
    rl_tokenizer.mkdir()
    _write_json(rl_tokenizer / "tokenizer.json", {"model": {"type": "BPE"}, "version": "1.0"})

    with pytest.raises(RuntimeError, match="tokenizer semantics"):
        validate_rl_inference_load_parity(
            checkpoint=checkpoint,
            export=export,
            rl_tokenizer=rl_tokenizer,
        )
