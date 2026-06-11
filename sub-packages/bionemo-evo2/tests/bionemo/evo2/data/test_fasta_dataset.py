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


from pathlib import Path

import pytest
import torch
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset
from bionemo.testing.data.fasta import create_fasta_file


@pytest.fixture
def fasta_dataset(tmp_path: Path) -> None:
    """Fixture to create a SimpleFastaDataset for testing."""
    test_fasta_file_path = create_fasta_file(tmp_path / "test.fasta", num_sequences=10, sequence_length=100)
    tokenizer = get_nmt_tokenizer("byte-level")
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

    import json

    with open(idx_map_file, "r") as f:
        idx_map = json.load(f)

    assert len(idx_map) == 10, "Index map should have an entry for each sequence"
    for idx, seqid in enumerate(fasta_dataset.seqids):
        assert idx_map[seqid] == idx, f"Index for {seqid} should match"
