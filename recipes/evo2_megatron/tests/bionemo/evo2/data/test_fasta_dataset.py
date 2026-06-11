# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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

"""Tests for SimpleFastaDataset."""

import json
from pathlib import Path

import pytest
import torch
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH
from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset
from bionemo.evo2.data.test_utils.create_fasta_file import create_fasta_file


@pytest.fixture
def tokenizer():
    """Return a HuggingFace tokenizer for testing."""
    return build_tokenizer(
        TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            hf_tokenizer_kwargs={"trust_remote_code": False},
            tokenizer_model=DEFAULT_HF_TOKENIZER_MODEL_PATH,
        )
    )


@pytest.fixture
def fasta_dataset(tmp_path: Path, tokenizer) -> SimpleFastaDataset:
    """Fixture to create a SimpleFastaDataset for testing."""
    test_fasta_file_path = create_fasta_file(tmp_path / "test.fasta", num_sequences=10, sequence_length=100)
    return SimpleFastaDataset(test_fasta_file_path, tokenizer)


def test_simple_fasta_dataset_initialization(fasta_dataset: SimpleFastaDataset) -> None:
    """Test initialization of SimpleFastaDataset."""
    # Check dataset length
    assert len(fasta_dataset) == 10, "Dataset length should match number of sequences"

    # Check seqids
    assert len(fasta_dataset.seqids) == 10, "Seqids should match number of sequences"


def test_simple_fasta_dataset_getitem(fasta_dataset: SimpleFastaDataset) -> None:
    """Test __getitem__ method of SimpleFastaDataset."""
    # Test first item
    item = fasta_dataset[0]

    # Check keys
    expected_keys = {"tokens", "position_ids", "seq_idx", "loss_mask"}
    assert set(item.keys()) == expected_keys, "Item should have correct keys"

    # Check token type
    assert isinstance(item["tokens"], torch.Tensor), "Tokens should be a torch.Tensor"
    assert item["tokens"].dtype == torch.long, "Tokens should be long dtype"

    # Check position_ids
    assert isinstance(item["position_ids"], torch.Tensor), "Position IDs should be a torch.Tensor"
    assert item["position_ids"].dtype == torch.long, "Position IDs should be long dtype"

    # Validate sequence index
    assert isinstance(item["seq_idx"], torch.Tensor), "Seq_idx should be a torch.Tensor"
    assert item["seq_idx"].item() == 0, "First item should have seq_idx 0"

    # Check loss_mask
    assert isinstance(item["loss_mask"], torch.Tensor), "Loss mask should be a torch.Tensor"
    assert item["loss_mask"].dtype == torch.long, "Loss mask should be long dtype"

    # With prepend_bos=True (default), the first token should be masked
    assert item["loss_mask"][0].item() == 0, "First token (BOS) should be masked"

    # Tokens length should be sequence_length + 1 (for BOS)
    # Since we create sequences of length 100, tokens should be 101
    assert len(item["tokens"]) == 101, "Tokens should include BOS token"
    assert len(item["position_ids"]) == 101, "Position IDs should match tokens length"


def test_simple_fasta_dataset_write_idx_map(fasta_dataset: SimpleFastaDataset, tmp_path: Path) -> None:
    """Test write_idx_map method of SimpleFastaDataset."""
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write index map
    fasta_dataset.write_idx_map(output_dir)

    # Check if file was created
    idx_map_file = output_dir / "seq_idx_map.json"
    assert idx_map_file.exists(), "seq_idx_map.json should be created"

    with open(idx_map_file) as f:
        idx_map = json.load(f)

    assert len(idx_map) == 10, "Index map should have an entry for each sequence"
    for idx, seqid in enumerate(fasta_dataset.seqids):
        assert idx_map[seqid] == idx, f"Index for {seqid} should match"


def test_simple_fasta_dataset_no_bos(tmp_path: Path, tokenizer) -> None:
    """Test SimpleFastaDataset without BOS token prepending."""
    test_fasta_file_path = create_fasta_file(tmp_path / "test_no_bos.fasta", num_sequences=5, sequence_length=50)
    dataset = SimpleFastaDataset(test_fasta_file_path, tokenizer, prepend_bos=False)

    item = dataset[0]

    # Without BOS, tokens length should equal sequence length
    assert len(item["tokens"]) == 50, "Tokens should not include BOS token"
    assert len(item["position_ids"]) == 50, "Position IDs should match tokens length"

    # All tokens should be unmasked (loss_mask all 1s)
    assert item["loss_mask"].sum().item() == 50, "All tokens should be unmasked without BOS"


def test_simple_fasta_dataset_variable_lengths(tmp_path: Path, tokenizer) -> None:
    """Test SimpleFastaDataset with variable sequence lengths."""
    sequence_lengths = [50, 100, 150, 200, 75]
    test_fasta_file_path = create_fasta_file(
        tmp_path / "test_variable.fasta", num_sequences=5, sequence_lengths=sequence_lengths
    )
    dataset = SimpleFastaDataset(test_fasta_file_path, tokenizer)

    assert len(dataset) == 5, "Dataset should have 5 sequences"

    # Check each item has the correct length (sequence_length + 1 for BOS)
    for i, expected_len in enumerate(sequence_lengths):
        item = dataset[i]
        assert len(item["tokens"]) == expected_len + 1, f"Sequence {i} should have length {expected_len + 1}"


def test_simple_fasta_dataset_iteration(fasta_dataset: SimpleFastaDataset) -> None:
    """Test that we can iterate through the entire dataset."""
    count = 0
    for i in range(len(fasta_dataset)):
        item = fasta_dataset[i]
        assert item is not None, f"Item {i} should not be None"
        assert "tokens" in item, f"Item {i} should have 'tokens' key"
        count += 1

    assert count == 10, "Should iterate through all 10 items"
