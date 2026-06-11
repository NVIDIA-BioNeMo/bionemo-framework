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

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pol
import pytest
import torch

from bionemo.evo2.data.sharded_eden_dataloader import (
    ShardedEdenDataModule,
    ShardedEdenDataset,
    extract_sample_id,
    precompute_window_database,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_sequences():
    """Create dummy sample sequences for testing."""
    return [
        ("BCR__ECT-SAMPLE1__CT1-1", "ATCGATCGATCGATCG" * 1000),  # 16000 bases
        ("BCR__ECT-SAMPLE1__CT1-2", "GCTAGCTAGCTAGCTA" * 800),  # 12800 bases
        ("BCR__ECT-SAMPLE2__CT1-1", "TAGCTAGCTAGCTAGC" * 1200),  # 19200 bases
        ("BCR__ECT-SAMPLE2__CT1-2", "CGATCGATCGATCGA" * 600),  # 9600 bases
        ("BCR__ECT-SAMPLE3__CT1-1", "ATCGATCGATCGATCG" * 700),  # 11200 bases
    ]


@pytest.fixture
def sequence_db_dir(temp_dir, sample_sequences):
    """Create sample SQLite databases for testing."""
    db_dir = Path(temp_dir) / "sequence_db_dir"
    db_dir.mkdir(exist_ok=True)

    # Group sequences by sample
    sequences_by_sample = {}
    for seq_id, sequence in sample_sequences:
        sample_id = extract_sample_id(seq_id)
        if sample_id not in sequences_by_sample:
            sequences_by_sample[sample_id] = []
        sequences_by_sample[sample_id].append((seq_id, sequence))

    # Create database for each sample
    for sample_id, sequences in sequences_by_sample.items():
        sample_dir = db_dir / sample_id
        sample_dir.mkdir(exist_ok=True)

        db_path = sample_dir / f"glm_dataset_{sample_id}.sqlite"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE sequences (
                contig_id TEXT PRIMARY KEY,
                nt_sequence TEXT NOT NULL
            )
        """)

        # Insert sequences
        for seq_id, sequence in sequences:
            cursor.execute("INSERT INTO sequences (contig_id, nt_sequence) VALUES (?, ?)", (seq_id, sequence))

        conn.commit()
        conn.close()

    return str(db_dir)


@pytest.fixture
def train_parquet(temp_dir, sample_sequences):
    """Create training split Parquet file."""
    # Use first 3 sequences for training
    train_data = pol.DataFrame(
        {
            "contig_id": [seq[0] for seq in sample_sequences[:3]],
            "length": [len(seq[1]) for seq in sample_sequences[:3]],
        }
    )

    parquet_path = Path(temp_dir) / "train_split.parquet"
    train_data.write_parquet(str(parquet_path))
    return str(parquet_path)


@pytest.fixture
def val_parquet(temp_dir, sample_sequences):
    """Create validation split Parquet file."""
    # Use last 2 sequences for validation
    val_data = pol.DataFrame(
        {
            "contig_id": [seq[0] for seq in sample_sequences[3:]],
            "length": [len(seq[1]) for seq in sample_sequences[3:]],
        }
    )

    parquet_path = Path(temp_dir) / "val_split.parquet"
    val_data.write_parquet(str(parquet_path))
    return str(parquet_path)


@pytest.fixture
def test_parquet(temp_dir, sample_sequences):
    """Create test split Parquet file."""
    # Use middle sequence for testing
    test_data = pol.DataFrame({"contig_id": [sample_sequences[2][0]], "length": [len(sample_sequences[2][1])]})

    parquet_path = Path(temp_dir) / "test_split.parquet"
    test_data.write_parquet(str(parquet_path))
    return str(parquet_path)


@pytest.fixture
def window_dbs(temp_dir, train_parquet, val_parquet, test_parquet):
    """Create window databases for all splits."""
    train_db = Path(temp_dir) / "train_windows.db"
    val_db = Path(temp_dir) / "val_windows.db"
    test_db = Path(temp_dir) / "test_windows.db"

    # Pre-compute window databases
    precompute_window_database(train_parquet, str(train_db), window_size=8192, stride=7992)
    precompute_window_database(val_parquet, str(val_db), window_size=8192, stride=7992)
    precompute_window_database(test_parquet, str(test_db), window_size=8192, stride=7992)

    return {"train": str(train_db), "val": str(val_db), "test": str(test_db)}


def test_extract_sample_id():
    """Test sample ID extraction from sequence IDs."""
    assert extract_sample_id("BCR__ECT-SAMPLE1__CT1-1") == "SAMPLE1"
    assert extract_sample_id("BCR__ECT-SAMPLE2__CT1-2") == "SAMPLE2"
    assert extract_sample_id("BCR__ECT-SAMPLE3__CT1-1") == "SAMPLE3"


def test_precompute_window_database(temp_dir, train_parquet):
    """Test window database pre-computation."""
    output_db = Path(temp_dir) / "test_windows.db"

    precompute_window_database(train_parquet, str(output_db), window_size=8192, stride=7992)

    # Verify database was created
    assert output_db.exists()

    # Check database contents
    conn = sqlite3.connect(str(output_db))
    cursor = conn.cursor()

    # Check metadata
    cursor.execute("SELECT key, value FROM metadata")
    metadata = dict(cursor.fetchall())

    assert metadata["window_size"] == 8192
    assert metadata["stride"] == 7992
    assert "total_windows" in metadata
    assert "distinct_sequences" in metadata

    # Check window mappings
    cursor.execute("SELECT COUNT(*) FROM window_mappings")
    window_count = cursor.fetchone()[0]
    assert window_count > 0

    conn.close()


def test_sharded_eden_dataset_initialization(sequence_db_dir, window_dbs):
    """Test ShardedEdenDataset initialization."""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.bos_id = 1
    mock_tokenizer.eos_id = 2
    mock_tokenizer._sep_id = 3
    mock_tokenizer.pad_id = 0
    mock_tokenizer.text_to_ids.return_value = [10, 11, 12]  # Dummy token IDs

    # Create dataset
    dataset = ShardedEdenDataset(
        tokenizer=mock_tokenizer,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_dbs["train"],
        seq_length=8192,
        create_attention_mask=False,
        stride=7992,
        rc_aug=False,
        use_control_tags=False,
        split="train",
    )

    # Verify dataset properties
    assert dataset.seq_length == 8192
    assert dataset.stride == 7992
    assert dataset.split == "train"
    assert len(dataset) > 0  # Should have some windows

    # Verify database connections
    assert hasattr(dataset, "db_connections")
    assert len(dataset.db_connections) > 0

    # Verify window database connection
    assert hasattr(dataset, "window_db_conn")
    assert dataset.window_db_conn is not None

    # Clean up
    dataset.__del__()


@patch("bionemo.evo2.data.sharded_eden_dataloader.get_nmt_tokenizer")
def test_sharded_eden_datamodule_initialization(mock_get_tokenizer, sequence_db_dir, window_dbs):
    """Test ShardedEdenDataModule initialization."""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.bos_id = 1
    mock_tokenizer.eos_id = 2
    mock_tokenizer._sep_id = 3
    mock_tokenizer.pad_id = 0
    mock_tokenizer.text_to_ids.return_value = [10, 11, 12]
    mock_get_tokenizer.return_value = mock_tokenizer

    # Create data module
    data_module = ShardedEdenDataModule(
        sequence_db_dir=sequence_db_dir,
        train_window_db_path=window_dbs["train"],
        val_window_db_path=window_dbs["val"],
        test_window_db_path=window_dbs["test"],
        seq_length=8192,
        micro_batch_size=1,
        global_batch_size=4,
        num_workers=0,  # Use 0 for testing
        rc_aug=False,
        use_control_tags=False,
    )

    # Verify data module properties
    assert data_module.sequence_db_dir == sequence_db_dir
    assert data_module.train_window_db_path == window_dbs["train"]
    assert data_module.val_window_db_path == window_dbs["val"]
    assert data_module.test_window_db_path == window_dbs["test"]
    assert data_module.seq_length == 8192
    assert data_module.micro_batch_size == 1
    assert data_module.global_batch_size == 4
    assert data_module.num_workers == 0

    # Verify tokenizer was created
    assert data_module.tokenizer is not None

    # Verify data sampler was created
    assert data_module.data_sampler is not None
    assert data_module.data_sampler.seq_len == 8192
    assert data_module.data_sampler.micro_batch_size == 1
    assert data_module.data_sampler.global_batch_size == 4


def test_dataset_getitem(sequence_db_dir, window_dbs):
    """Test dataset item retrieval."""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.bos_id = 1
    mock_tokenizer.eos_id = 2
    mock_tokenizer._sep_id = 3
    mock_tokenizer.pad_id = 0
    mock_tokenizer.text_to_ids.return_value = [10, 11, 12]

    # Create dataset
    dataset = ShardedEdenDataset(
        tokenizer=mock_tokenizer,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_dbs["train"],
        seq_length=8192,
        create_attention_mask=False,
        stride=7992,
        rc_aug=False,
        use_control_tags=False,
        split="train",
    )

    # Get first item
    item = dataset[np.int64(0)]

    # Verify item structure
    assert isinstance(item, dict)
    assert "tokens" in item
    assert "labels" in item
    assert "loss_mask" in item
    assert "position_ids" in item

    # Verify tensor shapes
    assert item["tokens"].shape == (8192,)
    assert item["labels"].shape == (8192,)
    assert item["loss_mask"].shape == (8192,)
    assert item["position_ids"].shape == (8192,)

    # Verify data types
    assert item["tokens"].dtype == torch.int64
    assert item["labels"].dtype == torch.int64
    assert item["loss_mask"].dtype == torch.float32
    assert item["position_ids"].dtype == torch.int64

    # Clean up
    dataset.__del__()


def test_dataset_with_control_tags(sequence_db_dir, window_dbs):
    """Test dataset with control tags enabled."""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.bos_id = 1
    mock_tokenizer.eos_id = 2
    mock_tokenizer._sep_id = 3
    mock_tokenizer.pad_id = 0
    mock_tokenizer.text_to_ids.return_value = [100, 101]  # Control tag IDs

    # Create dataset with control tags
    dataset = ShardedEdenDataset(
        tokenizer=mock_tokenizer,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_dbs["train"],
        seq_length=8192,
        create_attention_mask=False,
        stride=7992,
        rc_aug=False,
        use_control_tags=True,  # Enable control tags
        split="train",
    )

    # Verify control tags were prepared
    assert hasattr(dataset, "ctrl_ids_map")
    assert len(dataset.ctrl_ids_map) > 0

    # Get first item
    item = dataset[np.int64(0)]

    # Verify item contains control tags
    assert "tokens" in item
    assert "labels" in item
    assert "loss_mask" in item

    # Clean up
    dataset.__del__()


def test_dataset_with_attention_mask(sequence_db_dir, window_dbs):
    """Test dataset with attention mask creation."""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.bos_id = 1
    mock_tokenizer.eos_id = 2
    mock_tokenizer._sep_id = 3
    mock_tokenizer.pad_id = 0
    mock_tokenizer.text_to_ids.return_value = [10, 11, 12]

    # Create dataset with attention mask
    dataset = ShardedEdenDataset(
        tokenizer=mock_tokenizer,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_dbs["train"],
        seq_length=8192,
        create_attention_mask=True,  # Enable attention mask
        stride=7992,
        rc_aug=False,
        use_control_tags=False,
        split="train",
    )

    # Verify attention mask was created
    assert hasattr(dataset, "attention_mask")
    assert dataset.attention_mask.shape == (1, 8192, 8192)

    # Get first item
    item = dataset[np.int64(0)]

    # Verify attention mask is included
    assert "attention_mask" in item
    assert item["attention_mask"].shape == (1, 8192, 8192)

    # Clean up
    dataset.__del__()


def test_dataset_reverse_complement(sequence_db_dir, window_dbs):
    """Test reverse complement functionality."""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.bos_id = 1
    mock_tokenizer.eos_id = 2
    mock_tokenizer._sep_id = 3
    mock_tokenizer.pad_id = 0
    mock_tokenizer.text_to_ids.return_value = [10, 11, 12]

    # Create dataset with reverse complement augmentation
    dataset = ShardedEdenDataset(
        tokenizer=mock_tokenizer,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_dbs["train"],
        seq_length=8192,
        create_attention_mask=False,
        stride=7992,
        rc_aug=True,  # Enable reverse complement
        use_control_tags=False,
        split="train",
    )

    # Test reverse complement method
    test_seq = "ATCG"
    rc_seq = dataset.reverse_complement(test_seq)
    assert rc_seq == "CGAT"

    # Test with N bases
    test_seq_with_n = "ATCN"
    rc_seq_with_n = dataset.reverse_complement(test_seq_with_n)
    assert rc_seq_with_n == "NGAT"

    # Clean up
    dataset.__del__()


def test_dataset_collate_fn(sequence_db_dir, window_dbs):
    """Test dataset collate function."""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.bos_id = 1
    mock_tokenizer.eos_id = 2
    mock_tokenizer._sep_id = 3
    mock_tokenizer.pad_id = 0
    mock_tokenizer.text_to_ids.return_value = [10, 11, 12]

    # Create dataset
    dataset = ShardedEdenDataset(
        tokenizer=mock_tokenizer,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_dbs["train"],
        seq_length=8192,
        create_attention_mask=False,
        stride=7992,
        rc_aug=False,
        use_control_tags=False,
        split="train",
    )

    # Create a batch of items
    batch = [dataset[np.int64(0)], dataset[np.int64(1)]] if len(dataset) > 1 else [dataset[np.int64(0)]]

    # Test collate function
    collated = dataset.collate_fn(batch)

    # Verify collated structure
    assert isinstance(collated, dict)
    assert "tokens" in collated
    assert "labels" in collated
    assert "loss_mask" in collated
    assert "position_ids" in collated

    # Verify batch dimension
    assert collated["tokens"].dim() == 2
    assert collated["tokens"].shape[0] == len(batch)

    # Clean up
    dataset.__del__()


def test_window_min_length_threshold(temp_dir, train_parquet):
    """Test window database creation with length threshold."""
    output_db = Path(temp_dir) / "threshold_windows.db"

    # Create database with length threshold
    precompute_window_database(
        train_parquet,
        str(output_db),
        window_size=8192,
        stride=7992,
        window_min_length_threshold=10000,  # Only windows >= 10000 bases
    )

    # Verify database was created
    assert output_db.exists()

    # Check metadata
    conn = sqlite3.connect(str(output_db))
    cursor = conn.cursor()

    cursor.execute("SELECT key, value FROM metadata")
    metadata = dict(cursor.fetchall())

    assert metadata["window_min_length_threshold"] == 10000

    conn.close()


def test_dataset_length_and_iteration(sequence_db_dir, window_dbs):
    """Test dataset length and basic iteration."""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.bos_id = 1
    mock_tokenizer.eos_id = 2
    mock_tokenizer._sep_id = 3
    mock_tokenizer.pad_id = 0
    mock_tokenizer.text_to_ids.return_value = [10, 11, 12]

    # Create dataset
    dataset = ShardedEdenDataset(
        tokenizer=mock_tokenizer,
        sequence_db_dir=sequence_db_dir,
        window_db_path=window_dbs["train"],
        seq_length=8192,
        create_attention_mask=False,
        stride=7992,
        rc_aug=False,
        use_control_tags=False,
        split="train",
    )

    # Test length
    dataset_len = len(dataset)
    assert dataset_len > 0

    # Test iteration (just first few items)
    for i in range(min(3, dataset_len)):
        item = dataset[np.int64(i)]
        assert isinstance(item, dict)
        assert "tokens" in item

    # Clean up
    dataset.__del__()
