# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from unittest.mock import patch

import pytest
import torch

from src.utils.load_checkpoint import download_checkpoint, load_checkpoint


def test_load_pytorch_checkpoint(tmp_path):
    """Test loading a PyTorch .ckpt file."""
    # Create a mock checkpoint
    checkpoint_data = {
        "state_dict": {"layer.weight": torch.randn(10, 5)},
        "hyper_parameters": {"lr": 0.001, "model_name": "test"},
    }

    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(checkpoint_data, checkpoint_path)

    # Load the checkpoint
    loaded = load_checkpoint(str(checkpoint_path))

    assert "state_dict" in loaded
    assert "hyper_parameters" in loaded
    assert loaded["hyper_parameters"]["lr"] == 0.001
    assert loaded["hyper_parameters"]["model_name"] == "test"
    assert loaded["state_dict"]["layer.weight"].shape == (10, 5)


def test_load_pytorch_checkpoint_with_map_location(tmp_path):
    """Test loading a PyTorch checkpoint with custom map_location."""
    checkpoint_data = {"state_dict": {"param": torch.randn(5)}}
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(checkpoint_data, checkpoint_path)

    with patch("torch.load") as mock_torch_load:
        mock_torch_load.return_value = checkpoint_data
        load_checkpoint(str(checkpoint_path), map_location="cuda:0")

        mock_torch_load.assert_called_once_with(str(checkpoint_path), map_location="cuda:0", weights_only=True)


def test_load_safetensors_directory(tmp_path):
    """Test loading from a safetensors directory."""
    # Create directory structure
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    # Create a mock safetensors file
    safetensors_path = checkpoint_dir / "model.safetensors"
    safetensors_path.touch()

    # Create config.json
    config_data = {
        "model_name": "encodon_80m",
        "hidden_size": 768,
        "num_layers": 12,
    }
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    # Mock the safetensors load_file function
    mock_state_dict = {"layer.weight": torch.randn(10, 5)}

    with patch("src.utils.load_checkpoint.load_file") as mock_load_file:
        mock_load_file.return_value = mock_state_dict

        result = load_checkpoint(str(checkpoint_dir))

        assert "state_dict" in result
        assert "hyper_parameters" in result
        assert result["hyper_parameters"]["model_name"] == "encodon_80m"
        assert result["hyper_parameters"]["hidden_size"] == 768
        assert result["hyper_parameters"]["num_layers"] == 12
        mock_load_file.assert_called_once_with(str(safetensors_path))


def test_load_safetensors_directory_no_files(tmp_path):
    """Test that loading from a directory with no safetensors files raises an error."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Expected single .safetensors files"):
        load_checkpoint(str(checkpoint_dir))


def test_load_safetensors_directory_multiple_files(tmp_path):
    """Test that loading from a directory with multiple safetensors files raises an error."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    # Create multiple safetensors files
    (checkpoint_dir / "model1.safetensors").touch()
    (checkpoint_dir / "model2.safetensors").touch()

    with pytest.raises(FileNotFoundError, match="Expected single .safetensors files"):
        load_checkpoint(str(checkpoint_dir))


def test_load_safetensors_directory_missing_config(tmp_path):
    """Test that loading from a directory without config.json raises an error."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    # Create a safetensors file but no config.json
    safetensors_path = checkpoint_dir / "model.safetensors"
    safetensors_path.touch()

    with patch("src.utils.load_checkpoint.load_file") as mock_load_file:
        mock_load_file.return_value = {"layer.weight": torch.randn(10, 5)}

        with pytest.raises(FileNotFoundError):
            load_checkpoint(str(checkpoint_dir))


def test_load_nonexistent_file(tmp_path):
    """Test that loading a nonexistent file raises an error."""
    nonexistent_path = tmp_path / "nonexistent.ckpt"

    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(nonexistent_path))


def test_default_map_location_is_cpu(tmp_path):
    """Test that the default map_location is 'cpu'."""
    checkpoint_data = {"state_dict": {"param": torch.randn(5)}}
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(checkpoint_data, checkpoint_path)

    with patch("torch.load") as mock_torch_load:
        mock_torch_load.return_value = checkpoint_data
        load_checkpoint(str(checkpoint_path))

        # Verify that map_location defaults to "cpu"
        call_args = mock_torch_load.call_args
        assert call_args[1]["map_location"] == "cpu"


def test_download_checkpoint_success(tmp_path):
    """Test successful checkpoint download from HuggingFace Hub."""
    repo_id = "nvidia/NV-CodonFM-Encodon-TE-80M-v1"
    local_dir = str(tmp_path / "downloads")

    result = download_checkpoint(repo_id, local_dir)

    assert result == local_dir


def test_convert_state_dict():
    """Test state dict conversion with keymap."""
    from codonfm_ckpt_te_conversion import convert_state_dict

    # Create a simple keymap
    keymap = {
        "model.embedding.weight": "embeddings.weight",
        "model.layers.*.attention.qkv.weight": "layers.*.self_attn.qkv.weight",
        "model.layers.*.mlp.fc1.weight": "layers.*.mlp.dense_h_to_4h.weight",
    }

    # Create source state dict
    src_state_dict = {
        "model.embedding.weight": torch.randn(1000, 768),
        "model.layers.0.attention.qkv.weight": torch.randn(2304, 768),
        "model.layers.1.attention.qkv.weight": torch.randn(2304, 768),
        "model.layers.0.mlp.fc1.weight": torch.randn(3072, 768),
    }

    result = convert_state_dict(src_state_dict, keymap)

    # Check that keys are converted according to keymap
    assert "embeddings.weight" in result
    assert "layers.0.self_attn.qkv.weight" in result
    assert "layers.1.self_attn.qkv.weight" in result
    assert "layers.0.mlp.dense_h_to_4h.weight" in result

    # Check that original keys are not present
    assert "model.embedding.weight" not in result
    assert "model.layers.0.attention.qkv.weight" not in result

    # Check shapes are preserved
    assert result["embeddings.weight"].shape == (1000, 768)
    assert result["layers.0.self_attn.qkv.weight"].shape == (2304, 768)

    """Test that convert_state_dict preserves unmapped keys."""
    from codonfm_ckpt_te_conversion import convert_state_dict

    keymap = {
        "model.embedding.weight": "embeddings.weight",
    }

    src_state_dict = {
        "model.embedding.weight": torch.randn(1000, 768),
        "model.other.weight": torch.randn(100, 50),  # Not in keymap
    }

    result = convert_state_dict(src_state_dict, keymap)

    # Mapped key should be converted
    assert "embeddings.weight" in result

    # Unmapped key should be preserved
    assert "model.other.weight" in result
    assert result["model.other.weight"].shape == (100, 50)
