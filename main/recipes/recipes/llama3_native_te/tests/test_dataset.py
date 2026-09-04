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

import itertools
import os
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.distributed.device_mesh import init_device_mesh

from collator import ContextParallelDataLoaderWrapper, DataCollatorForContextParallel
from dataset import create_bshd_dataloader, create_thd_dataloader, create_tokenized_dataset
from distributed_config import DistributedConfig


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


@pytest.fixture
def simple_parquet(tmp_path):
    """Create a simple Parquet file with multiple genomic sequences for testing batching."""
    parquet_path = tmp_path / "genomic_sequences.parquet"

    # Create multiple sequences of varying lengths for better batching tests
    sequences = [
        "A" * 1000,
        "T" * 1200,
        "C" * 800,
        "G" * 1500,
        "ATCG" * 300,
    ]

    table = pa.table(
        {
            "text": sequences,
        }
    )

    pq.write_table(table, parquet_path)
    return str(parquet_path)


def test_dataset_loads_and_tokenizes_sequence(tokenizer_path, tmp_path):
    """Test that dataset loads and tokenizes a sequence correctly with exact token verification.

    Uses single sequence so shuffling doesn't affect test (similar to SQLite test approach).
    Pattern: expected_sequence = [nucleotide_id] * seqlen
    """
    # Create a Parquet file with a single T sequence of known length
    parquet_path = tmp_path / "genomic_sequences.parquet"
    sequence = "T" * 10  # Small, predictable sequence
    table = pa.table({"text": [sequence]})
    pq.write_table(table, parquet_path)

    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }

    tokenized_dataset, _ = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=20,  # Large enough to fit the sequence
        stride=10,
        buffer_size=10_000,
    )

    # Only 1 sequence → 1 window → dataset[0] is predictable regardless of shuffle
    sample = tokenized_dataset[0]
    assert "input_ids" in sample

    # Get nucleotides (remove BOS and EOS)
    tokens = sample["input_ids"]
    nucleotides = tokens[1:-1]

    # Verify exact expected sequence
    bos = 2
    eos = 0
    t = 84  # ASCII value of 'T'

    expected_sequence = [t] * 10  # All Ts
    received_sequence = nucleotides

    assert tokens[0] == bos, f"First token should be BOS (2), got {tokens[0]}"
    assert tokens[-1] == eos, f"Last token should be EOS (0), got {tokens[-1]}"
    assert received_sequence == expected_sequence, f"Expected {expected_sequence}, got {received_sequence}"


def test_dataloader_returns_expected_batch(tokenizer_path, tmp_path):
    """Test dataloader returns exact expected batch with known input.

    Creates minimal test data with exactly one sequence to get deterministic output.
    Verifies exact token values match expected hardcoded batch.
    """
    # Create minimal test parquet with exactly 1 sequence
    parquet_path = tmp_path / "single_sequence.parquet"
    sequence = "A" * 5  # 5 As
    table = pa.table({"text": [sequence]})
    pq.write_table(table, parquet_path)

    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }

    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=1,  # Just one sample per batch
        num_workers=0,
        max_seq_length=7,  # Large enough for 5bp sequence
        stride=5,
        uppercase_labels=False,  # Use standard collator for this test
        mask_degenerate_bases=False,  # Use standard collator for this test
    )

    returned_batch = next(iter(dataloader))

    # Hardcode expected batch (1 sequence, deterministic output)
    # seq: 5bp of As -> BOS + 5 As + EOS
    bos = 2
    eos = 0
    a = 65  # ASCII value of 'A'

    expected_input_ids = torch.tensor([[bos, a, a, a, a, a, eos]], dtype=torch.long)
    expected_labels = torch.tensor([[bos, a, a, a, a, a, eos]], dtype=torch.long)  # CLM: labels = input_ids
    expected_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)  # All real tokens

    assert torch.equal(returned_batch["input_ids"], expected_input_ids), (
        f"Expected input_ids {expected_input_ids}, got {returned_batch['input_ids']}"
    )
    assert torch.equal(returned_batch["labels"], expected_labels), (
        f"Expected labels {expected_labels}, got {returned_batch['labels']}"
    )
    assert torch.equal(returned_batch["attention_mask"], expected_attention_mask), (
        f"Expected attention_mask {expected_attention_mask}, got {returned_batch['attention_mask']}"
    )


def test_attention_mask_aligns_with_labels(tokenizer_path, simple_parquet):
    """Test attention_mask correctly identifies real vs padded positions in labels.

    Where attention_mask=1: labels should contain real token IDs (matching input_ids)
    Where attention_mask=0: labels should contain ignore_index value (-100)
    """
    # HuggingFace's DataCollatorForLanguageModeling uses -100 as ignore_index by default
    ignore_pad_token = -100

    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
    }

    # Use a moderate window size to ensure we get padding in batches
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
        uppercase_labels=False,  # Use standard collator for this test
        mask_degenerate_bases=False,  # Use standard collator for this test
    )

    batch = next(iter(dataloader))

    # Check first sequence in batch
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]

    # Where attention_mask=1, labels should equal input_ids (real tokens)
    real_positions = attention_mask == 1
    real_labels = labels[real_positions]
    real_input_ids = input_ids[real_positions]

    # For CLM (Causal Language Modeling), labels should match input_ids at real positions
    assert torch.all(real_labels == real_input_ids), "Labels should match input_ids at real token positions"

    # Verify specific token positions contain expected values
    assert real_labels[0].item() == 2, "First token should be BOS (2)"
    assert real_labels[-1].item() == 0, "Last real token should be EOS (0)"
    # Middle tokens should be nucleotides (A=65, T=84, C=67, G=71)
    if len(real_labels) > 2:
        middle_token = real_labels[1].item()
        assert middle_token in [65, 84, 67, 71], f"Nucleotide tokens should be A/T/C/G, got {middle_token}"

    # Ensure NO real position has the ignore padding value
    assert torch.all(real_labels != ignore_pad_token), "Real tokens should not have IGNORE_PAD_TOKEN"

    # Where attention_mask=0, labels should be IGNORE_PAD_TOKEN (-100)
    padded_positions = attention_mask == 0
    if padded_positions.any():
        padded_labels = labels[padded_positions]
        assert torch.all(padded_labels == ignore_pad_token), (
            f"Padded positions should have IGNORE_PAD_TOKEN (-100), got {padded_labels.unique()}"
        )


def test_windowing_in_dataset_creates_multiple_samples(tokenizer_path, tmp_path):
    """Test that the dataset's windowing creates expected number of samples."""
    # Create a 3kbp sequence
    parquet_path = tmp_path / "genomic_sequences.parquet"
    sequence = "A" * 3000
    table = pa.table({"text": [sequence]})
    pq.write_table(table, parquet_path)

    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }

    tokenized_dataset, _ = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=1000,
        stride=800,  # 800 token overlap, so 200 token step
        buffer_size=10_000,
    )

    # Count samples
    num_samples = len(tokenized_dataset)

    # With 3000bp sequence, max_length=1000, stride=800 (800 overlap, 200 step)
    # Formula: ceil((3000+2 - 1000) / 200) + 1 = ceil(2002/200) + 1 = 11 + 1 = 12 windows
    assert num_samples == 12, f"Expected exactly 12 windows, got {num_samples}"


def test_lazy_tokenization_returns_batch(tokenizer_path, simple_parquet):
    """Test that lazy tokenization works and returns valid batches."""
    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
        "streaming": False,
    }

    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
    )

    # Get a batch
    batch = next(iter(dataloader))

    # Verify batch is not None and has correct structure
    assert batch is not None
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert isinstance(batch["input_ids"], torch.Tensor)

    # With lazy tokenization and windowing, batch size can vary due to on-the-fly window expansion
    # Just verify we get at least one sample (lazy tokenization + windowing makes exact count unpredictable)
    assert batch["input_ids"].shape[0] >= 1, f"Expected at least 1 sample in batch, got {batch['input_ids'].shape[0]}"


@pytest.mark.parametrize("streaming", [False, True])
def test_multiple_sequences_batch_correctly(tokenizer_path, simple_parquet, streaming):
    """Test that multiple sequences batch together correctly in both streaming and non-streaming modes.

    This test catches bugs that only appear with multi-row datasets vs single-row:
    - Batching/collation works with multiple sequences
    - Sequences in batch are different (not duplicated)
    - Padding aligns correctly across multiple sequences
    - All sequences are processed across batches
    - Works in both streaming=True and streaming=False modes
    """
    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
        "streaming": streaming,
    }

    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
        buffer_size=10_000,  # Only used for streaming
    )

    # Get first batch
    batch = next(iter(dataloader))

    # KEY TEST 1: Verify batch contains MULTIPLE sequences (not just 1)
    assert batch["input_ids"].shape[0] == 2, f"Batch should contain 2 sequences, got {batch['input_ids'].shape[0]}"

    # KEY TEST 2: Verify sequences in batch are DIFFERENT (catch duplication bugs)
    seq1 = batch["input_ids"][0]
    seq2 = batch["input_ids"][1]
    assert not torch.equal(seq1, seq2), "Sequences in batch should be different, not duplicates"

    # KEY TEST 3: Verify padding aligns across all tensors in batch
    batch_size, seq_length = batch["input_ids"].shape
    assert batch["attention_mask"].shape == (batch_size, seq_length)
    assert batch["labels"].shape == (batch_size, seq_length)

    # KEY TEST 4: Verify all sequences are processed (multiple batches produced)
    # With 5 sequences from simple_parquet (800-1500bp) and max_seq_length=500,
    # windowing will create ~11+ windows total. With batch_size=2, expect ~5-6 batches.
    # We already consumed 1 batch, so should have at least 4 remaining batches.
    all_batches = list(dataloader)
    total_batches = len(all_batches) + 1  # +1 for first batch already consumed
    assert len(all_batches) >= 4, (
        f"Expected at least 4 remaining batches (5 total), got {len(all_batches)} remaining ({total_batches} total)"
    )

    # KEY TEST 5: Verify subsequent batches also valid (not just first batch)
    if len(all_batches) > 0:
        second_batch = all_batches[0]
        # Check structure is consistent across batches
        assert "input_ids" in second_batch
        assert "attention_mask" in second_batch
        assert "labels" in second_batch
        # Verify it also has multiple sequences (could be different count due to windowing)
        assert second_batch["input_ids"].shape[0] >= 1, (
            f"Second batch should have at least 1 sequence, got {second_batch['input_ids'].shape[0]}"
        )
        # Verify tensors align
        batch_size_2, seq_length_2 = second_batch["input_ids"].shape
        assert second_batch["attention_mask"].shape == (batch_size_2, seq_length_2)
        assert second_batch["labels"].shape == (batch_size_2, seq_length_2)


def test_batching_produces_correct_batch_size(tokenizer_path, tmp_path):
    """Test that batching combines multiple sequences correctly with exact batch counts.

    Creates 5 short sequences (no windowing) with micro_batch_size=2.
    Should produce exactly 3 batches with shapes: [2, 2, 1].
    """
    # Create 5 sequences that won't trigger windowing (all very short)
    parquet_path = tmp_path / "five_sequences.parquet"
    sequences = [
        "A" * 10,  # Seq 1
        "T" * 15,  # Seq 2
        "C" * 12,  # Seq 3
        "G" * 8,  # Seq 4
        "ATCG" * 3,  # Seq 5 (12bp)
    ]
    table = pa.table({"text": sequences})
    pq.write_table(table, parquet_path)

    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }

    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=50,  # Large enough - no windowing
        stride=10,
    )

    # Collect all batches
    batches = list(dataloader)

    # With 5 sequences and batch_size=2, expect exactly 3 batches: [2, 2, 1]
    assert len(batches) == 3, f"Expected exactly 3 batches from 5 sequences, got {len(batches)}"

    # Check each batch has correct shape
    assert batches[0]["input_ids"].shape[0] == 2, "Batch 0 should have 2 sequences"
    assert batches[1]["input_ids"].shape[0] == 2, "Batch 1 should have 2 sequences"
    assert batches[2]["input_ids"].shape[0] == 1, "Batch 2 should have 1 sequence (remainder)"


def test_non_streaming_dataset_produces_correct_batch_size(recipe_path):
    """Test that batching combines multiple sequences correctly with exact batch counts.

    Creates 5 short sequences (no windowing) with micro_batch_size=2.
    Should produce exactly 3 batches with shapes: [2, 2, 1].
    """
    distributed_config = DistributedConfig(rank=0, world_size=1)
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                "dataset.load_dataset_kwargs.streaming=False",
            ],
        )

    dataloader, sampler = create_bshd_dataloader(
        distributed_config=distributed_config,
        **sanity_config.dataset,
    )

    assert isinstance(sampler, torch.utils.data.distributed.DistributedSampler), (
        "Sampler should be a DistributedSampler"
    )

    batches = list(itertools.islice(dataloader, 50))

    for batch in batches:
        assert batch["input_ids"].shape[0] == sanity_config.dataset.micro_batch_size
        assert batch["input_ids"].shape[1] <= sanity_config.dataset.max_seq_length


def test_batching_produces_correct_batch_size_sequence_packing(tokenizer_path, tmp_path):
    """Test that batching combines multiple sequences correctly with exact batch counts"""
    parquet_path = tmp_path / "five_sequences.parquet"
    sequences = ["A"] * 20
    table = pa.table({"text": sequences})
    pq.write_table(table, parquet_path)

    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
        "streaming": True,
    }

    dataloader, _ = create_thd_dataloader(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        token_micro_batch_size=15,
        max_seq_length=15,
        stride=10,
        split_samples_in_token_packing=False,
    )

    batches = list(dataloader)

    assert len(batches) > 0

    for batch in batches:
        # BOS, 'A', 'EOS' * 5
        torch.testing.assert_close(batch["input_ids"].squeeze(0), torch.tensor([[2, 65, 0] * 5]).flatten())


def test_streaming_dataset_removes_columns_correctly(tokenizer_path, tmp_path):
    """Test that streaming datasets properly remove input columns (text, record) during tokenization.

    This is a regression test for the OpenGenome2-specific bug where dataset.column_names is None
    for streaming datasets with inconsistent schemas across shards. This causes remove_columns to fail
    and leaves raw text/record columns in the tokenized dataset.

    OpenGenome2 has inconsistent schemas:
    - Some shards: ["text", "record"]
    - Some shards: ["text"] only
    - Result: dataset.column_names = None (can't determine upfront)

    Note: Regular datasets (like ESM2) don't have this issue because they have consistent schemas.

    Reference: https://github.com/NVIDIA/bionemo-framework/commit/3c0aee6de065ef494389591ca9028e8301dc385a
    """
    # Create a Parquet file with both 'text' (sequence column) and 'record' (metadata)
    parquet_path = tmp_path / "genomic_with_metadata.parquet"
    sequences = ["ATCGATCG" * 10, "GCTAGCTA" * 10]
    records = ["chr1:1000-1080", "chr2:2000-2080"]

    table = pa.table(
        {
            "text": sequences,  # Using 'text' to match OpenGenome2 format
            "record": records,  # Metadata column that should be removed
        }
    )
    pq.write_table(table, parquet_path)

    distributed_config = DistributedConfig(rank=0, world_size=1)

    # Load as streaming dataset
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
        "streaming": True,  # This makes dataset.column_names = None
    }

    tokenized_dataset, _ = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=100,
        stride=10,
        buffer_size=1000,
        text_column="text",  # Specify which column has sequences
    )

    # Get first sample from streaming dataset
    sample = next(iter(tokenized_dataset))

    # Verify that only tokenizer outputs remain (no raw text or record columns)
    expected_keys = {"input_ids", "attention_mask", "token_type_ids", "overflow_to_sample_mapping"}
    actual_keys = set(sample.keys())

    assert actual_keys.issubset(expected_keys), (
        f"Unexpected columns found in tokenized dataset. "
        f"Expected only {expected_keys}, but got {actual_keys}. "
        f"Columns 'text' and 'record' should have been removed."
    )

    # Specifically check that problematic columns are NOT present
    assert "text" not in sample, "Column 'text' should have been removed during tokenization"
    assert "record" not in sample, "Column 'record' should have been removed during tokenization"

    # Verify tokenizer outputs are present and valid
    assert "input_ids" in sample, "input_ids should be present"
    assert isinstance(sample["input_ids"], list), "input_ids should be a list"
    assert len(sample["input_ids"]) > 0, "input_ids should not be empty"


def test_streaming_dataset_handles_missing_record_column(tokenizer_path, tmp_path):
    """Test that remove_columns handles missing 'record' column gracefully (OpenGenome2 workaround).

    OpenGenome2 has inconsistent schemas across shards:
    - Some shards have 'record' column (metadata)
    - Some shards don't have 'record' column

    This test verifies that explicitly listing 'record' in columns_to_remove doesn't
    cause errors when the column is absent. This is part of the OpenGenome2 workaround.

    TODO: Remove this workaround once Arc Institute fixes OpenGenome2 schema consistency.

    Reference: https://github.com/NVIDIA/bionemo-framework/commit/a41f306eda7605552ee736e3291c098f2623828a
    """
    # Create a Parquet file with ONLY 'text' column (no 'record')
    parquet_path = tmp_path / "genomic_no_record.parquet"
    sequences = ["ATCGATCG" * 10, "GCTAGCTA" * 10]

    table = pa.table(
        {
            "text": sequences,  # Only text, no record column
        }
    )
    pq.write_table(table, parquet_path)

    distributed_config = DistributedConfig(rank=0, world_size=1)

    # Load as streaming dataset
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
        "streaming": True,
    }

    # This should NOT raise an error even though 'record' is in columns_to_remove
    tokenized_dataset, _ = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=100,
        stride=10,
        buffer_size=1000,
        text_column="text",
    )

    # Get first sample - should work without errors
    sample = next(iter(tokenized_dataset))

    # Verify only tokenizer outputs are present
    assert "text" not in sample, "Column 'text' should have been removed"
    assert "record" not in sample, "Column 'record' was never present, so shouldn't be in output"
    assert "input_ids" in sample, "input_ids should be present"


def test_dataloader_with_genomic_masking(tokenizer_path, tmp_path):
    """Test that create_bshd_dataloader works with genomic masking enabled.

    Integration test verifying:
    - GenomicDataCollatorForCLM is used when masking flags are set
    - Degenerate bases are masked in labels
    - Batches are produced in correct BSHD format
    """
    # Create test data with degenerate bases
    parquet_path = tmp_path / "genomic_with_degenerate.parquet"
    sequences = ["ACGTN", "GGTAR"]  # Has degenerate N and R
    table = pa.table({"text": sequences})
    pq.write_table(table, parquet_path)

    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }

    # Create dataloader with genomic masking enabled
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=10,
        stride=5,
        mask_degenerate_bases=True,  # Enable degenerate masking
    )

    # Get a batch
    batch = next(iter(dataloader))

    # Verify BSHD format
    assert batch["input_ids"].ndim == 2, "Should be BSHD format [B, S]"
    assert batch["labels"].ndim == 2, "Labels should be BSHD format"

    # Verify degenerate bases (N=78, R=82) are masked
    labels = batch["labels"]
    assert 78 not in labels, "Degenerate N (78) should be masked"
    assert 82 not in labels, "Degenerate R (82) should be masked"

    # Verify valid DNA tokens are present
    valid_dna = [65, 67, 71, 84]  # A, C, G, T
    assert any(tok in labels for tok in valid_dna), "Should have valid DNA tokens"


def test_token_packing_dataloader(tokenizer_path):
    """Test that the token packing dataloader works."""

    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "test_genomic_sequences.parquet",
        "streaming": True,
    }

    distributed_config = DistributedConfig(rank=0, world_size=1)

    dataloader, _ = create_thd_dataloader(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        text_column="sequence",
        micro_batch_size=1,
        max_seq_length=1024,
    )

    batches = list(dataloader)
    assert len(batches) > 1


def test_cp_dataloader(tokenizer_path):
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "dlcm_sanity_dataset.parquet",
        "streaming": True,
    }

    dist_config = DistributedConfig()

    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    device_mesh = init_device_mesh("cuda", mesh_shape=(1, 1), mesh_dim_names=("dp", "cp"))

    cp_mesh = device_mesh["cp"]

    # Create the context-parallel dataloader directly following the pattern in train_fsdp2_cp.py
    if cp_mesh.get_local_rank() == 0:
        train_dataloader, _ = create_thd_dataloader(
            distributed_config=dist_config,
            tokenizer_name_or_path=tokenizer_path,
            load_dataset_kwargs=load_dataset_kwargs,
            text_column="text",
            micro_batch_size=1,
            max_seq_length=1024,
            pad_sequences_to_be_divisible_by=cp_mesh.size() * 2,
        )

        train_dataloader.collate_fn = DataCollatorForContextParallel(
            collator=train_dataloader.collate_fn,
            device_mesh=cp_mesh,
            is_causal_lm=True,
        )
    else:
        train_dataloader = None

    dataloader = ContextParallelDataLoaderWrapper(train_dataloader, cp_mesh)

    batches = list(dataloader)
    assert len(batches) > 1

    for batch in batches:
        assert set(batch.keys()) == {
            "max_length_q",
            "max_length_k",
            "input_ids",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "labels",
            "cu_seq_lens_q_padded",
            "cu_seq_lens_k_padded",
            "pad_between_seqs",
            "shift_labels",
        }

    torch.distributed.destroy_process_group()


@requires_multi_gpu
@pytest.mark.parametrize("dataset_path", ["dlcm_sanity_dataset.parquet", "test_genomic_sequences.parquet"])
def test_cp_dataloader_multi_gpu(recipe_path, dataset_path):
    """Tests that the CP dataloader works correctly with multiple GPUs.

    The `test_genomic_sequences.parquet` dataset is too small to even fill a single batch with the default context
    length of 8192 tokens, so this test ensures that the dataloader fails gracefully when it encounters a StopIteration.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(recipe_path)

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "tests/test_dataset.py",
        "--dataset_path",
        dataset_path,
    ]

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
        cwd=str(recipe_path),
        env=env,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dlcm_sanity_dataset.parquet")
    args = parser.parse_args()

    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    device_mesh = init_device_mesh("cuda", mesh_shape=(1, 2), mesh_dim_names=("dp", "cp"))

    cp_mesh = device_mesh["cp"]

    # Create the context-parallel dataloader directly following the pattern in train_fsdp2_cp.py
    if cp_mesh.get_local_rank() == 0:
        train_dataloader, _ = create_thd_dataloader(
            distributed_config=dist_config,
            tokenizer_name_or_path="nvidia/Llama-3.1-8B-Instruct-FP8",
            micro_batch_size=1,
            text_column="text" if args.dataset_path == "dlcm_sanity_dataset.parquet" else "sequence",
            load_dataset_kwargs={
                "path": "parquet",
                "split": "train",
                "data_files": args.dataset_path,
                "streaming": True,
            },
            num_workers=1,
            pad_sequences_to_be_divisible_by=cp_mesh.size() * 2,
        )

        train_dataloader.collate_fn = DataCollatorForContextParallel(
            collator=train_dataloader.collate_fn,
            device_mesh=cp_mesh,
            is_causal_lm=True,
        )
    else:
        train_dataloader = None

    dataloader = ContextParallelDataLoaderWrapper(train_dataloader, cp_mesh)

    batches = list(itertools.islice(dataloader, 10))

    # With CP size 2, each sequence is split into 2 * cp_world_size = 4 slices.
    # Each rank gets 2 slices (beginning and end), so each rank gets approximately
    # (8 * 1024) / 2 = 4096 tokens per rank
    # Note: Sequences are padded to be divisible by pad_sequences_to_be_divisible_by
    # (which defaults to cp_mesh.size() * 2 = 4 if not provided)
    # The actual token count per rank can vary due to:
    # 1. Sequence packing (variable-length sequences packed up to token_micro_batch_size)
    # 2. Per-sequence padding to be divisible by pad_sequences_to_be_divisible_by
    # 3. CP splitting logic that takes slices from beginning and end
    expected_tokens_per_rank = (8 * 1024) // device_mesh["cp"].size()

    for batch in batches:
        actual_shape = batch["input_ids"].shape[1]
        # Allow for variance due to sequence packing, padding, and CP splitting
        # The actual shape should be close to expected_tokens_per_rank but can vary
        # Allow up to 100 tokens of variance (both above and below) to account for
        # sequence packing and padding effects
        assert actual_shape >= expected_tokens_per_rank - 100, (
            f"Expected at least {expected_tokens_per_rank - 100} tokens, got {actual_shape}"
        )
        assert actual_shape <= expected_tokens_per_rank + 100, (
            f"Expected at most {expected_tokens_per_rank + 100} tokens, got {actual_shape}"
        )
        assert batch["labels"] is None
        assert batch["shift_labels"].shape[1] == actual_shape

    dataloader.close()
    torch.distributed.destroy_process_group()
