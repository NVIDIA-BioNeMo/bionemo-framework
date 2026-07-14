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

import hashlib
import json

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from bionemo.evo2.vllm.export import export_mbridge_to_vllm, infer_evo2_config


def _write_synthetic_checkpoint(tmp_path):
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
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("7\n")
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
  layernorm_epsilon: 1.0e-6
"""
    )
    tokenizer_dir = iteration_dir / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text('{"version": "1.0"}\n')
    return checkpoint_root, source


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_infer_evo2_config_from_checkpoint_run_config(tmp_path):
    checkpoint_root, _ = _write_synthetic_checkpoint(tmp_path)

    config, provider = infer_evo2_config(checkpoint_root)

    assert config.hidden_size == 4
    assert config.num_hidden_layers == 4
    assert config.intermediate_size == 6
    assert config.max_position_embeddings == 10240
    assert config.hybrid_override_pattern == "SDH*"
    assert config.num_groups_hyena_short == 4
    assert config.num_groups_hyena_medium == 4
    assert provider.endswith("HyenaTestModelProvider")


def test_streaming_export_round_trips_shards_config_index_and_manifest(tmp_path):
    checkpoint_root, source = _write_synthetic_checkpoint(tmp_path)
    output_dir = tmp_path / "vllm"
    max_shard_size = 160

    manifest = export_mbridge_to_vllm(
        checkpoint_root,
        output_dir,
        max_shard_size=max_shard_size,
    )

    index_path = output_dir / "model.safetensors.index.json"
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.json"
    index = json.loads(index_path.read_text())
    loaded = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        loaded.update(load_file(output_dir / shard_name))

    assert set(loaded) == set(source)
    for name, expected in source.items():
        torch.testing.assert_close(loaded[name], expected, rtol=0, atol=0)
    assert len(set(index["weight_map"].values())) >= 2
    assert index["metadata"]["total_size"] == sum(tensor.numel() * tensor.element_size() for tensor in source.values())
    assert manifest == json.loads(manifest_path.read_text())
    assert manifest["source_iteration"] == 7
    assert manifest["model_provider"].endswith("HyenaTestModelProvider")
    assert manifest["config_sha256"] == _sha256(config_path)
    assert manifest["index_sha256"] == _sha256(index_path)
    assert manifest["peak_shard_bytes"] <= max_shard_size
    assert manifest["estimated_peak_buffered_bytes"] <= (
        manifest["peak_shard_bytes"] + manifest["largest_tensor_bytes"]
    )
    assert (output_dir / "tokenizer" / "tokenizer.json").read_text() == '{"version": "1.0"}\n'
