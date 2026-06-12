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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from dataset import create_bshd_dataloader, create_thd_dataloader, create_tokenized_dataset
from distributed_config import DistributedConfig


@pytest.fixture
def simple_parquet(tmp_path):
    """Create a simple Parquet file with multiple genomic sequences for testing batching."""
    parquet_path = tmp_path / "genomic_sequences.parquet"

    sequences = [
        "A" * 1000,
        "T" * 1200,
        "C" * 800,
        "G" * 1500,
        "ATCG" * 300,
    ]

    table = pa.table({"text": sequences})
    pq.write_table(table, parquet_path)
    return str(parquet_path)


def test_dataset_loads_and_tokenizes_sequence(tokenizer_path, tmp_path):
    """Test that dataset loads and tokenizes a sequence correctly with exact token verification."""
    parquet_path = tmp_path / "genomic_sequences.parquet"
    sequence = "T" * 10
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
        max_seq_length=20,
        stride=10,
        buffer_size=10_000,
    )

    sample = tokenized_dataset[0]
    assert "input_ids" in sample

    tokens = sample["input_ids"]
    nucleotides = tokens[1:-1]

    bos = 2
    eos = 0
    t = 84  # ASCII value of 'T'

    expected_sequence = [t] * 10
    received_sequence = nucleotides

    assert tokens[0] == bos, f"First token should be BOS (2), got {tokens[0]}"
    assert tokens[-1] == eos, f"Last token should be EOS (0), got {tokens[-1]}"
    assert received_sequence == expected_sequence, f"Expected {expected_sequence}, got {received_sequence}"


def test_dataloader_returns_expected_batch(tokenizer_path, tmp_path):
    """Test dataloader returns exact expected batch with known input."""
    parquet_path = tmp_path / "single_sequence.parquet"
    sequence = "A" * 5
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
        micro_batch_size=1,
        num_workers=0,
        max_seq_length=7,
        stride=5,
        uppercase_labels=False,
        mask_degenerate_bases=False,
    )

    returned_batch = next(iter(dataloader))

    bos = 2
    eos = 0
    a = 65  # ASCII value of 'A'

    expected_input_ids = torch.tensor([[bos, a, a, a, a, a, eos]], dtype=torch.long)
    expected_labels = torch.tensor([[bos, a, a, a, a, a, eos]], dtype=torch.long)
    expected_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)

    assert torch.equal(returned_batch["input_ids"], expected_input_ids)
    assert torch.equal(returned_batch["labels"], expected_labels)
    assert torch.equal(returned_batch["attention_mask"], expected_attention_mask)


def test_attention_mask_aligns_with_labels(tokenizer_path, simple_parquet):
    """Test attention_mask correctly identifies real vs padded positions in labels."""
    ignore_pad_token = -100

    distributed_config = DistributedConfig(rank=0, world_size=1)

    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
    }

    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
        uppercase_labels=False,
        mask_degenerate_bases=False,
    )

    batch = next(iter(dataloader))

    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]

    real_positions = attention_mask == 1
    real_labels = labels[real_positions]
    real_input_ids = input_ids[real_positions]

    assert torch.all(real_labels == real_input_ids), "Labels should match input_ids at real token positions"
    assert real_labels[0].item() == 2, "First token should be BOS (2)"
    assert real_labels[-1].item() == 0, "Last real token should be EOS (0)"

    assert torch.all(real_labels != ignore_pad_token), "Real tokens should not have IGNORE_PAD_TOKEN"

    padded_positions = attention_mask == 0
    if padded_positions.any():
        padded_labels = labels[padded_positions]
        assert torch.all(padded_labels == ignore_pad_token)


def test_windowing_in_dataset_creates_multiple_samples(tokenizer_path, tmp_path):
    """Test that the dataset's windowing creates expected number of samples."""
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
        stride=800,
        buffer_size=10_000,
    )

    num_samples = len(tokenized_dataset)
    assert num_samples == 12, f"Expected exactly 12 windows, got {num_samples}"


@pytest.mark.parametrize("streaming", [False, True])
def test_multiple_sequences_batch_correctly(tokenizer_path, simple_parquet, streaming):
    """Test that multiple sequences batch together correctly."""
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
        buffer_size=10_000,
    )

    batch = next(iter(dataloader))

    assert batch["input_ids"].shape[0] == 2, f"Batch should contain 2 sequences, got {batch['input_ids'].shape[0]}"

    seq1 = batch["input_ids"][0]
    seq2 = batch["input_ids"][1]
    assert not torch.equal(seq1, seq2), "Sequences in batch should be different"

    batch_size, seq_length = batch["input_ids"].shape
    assert batch["attention_mask"].shape == (batch_size, seq_length)
    assert batch["labels"].shape == (batch_size, seq_length)


def test_batching_produces_correct_batch_size(tokenizer_path, tmp_path):
    """Test that batching produces correct batch sizes with remainder."""
    parquet_path = tmp_path / "five_sequences.parquet"
    sequences = ["A" * 10, "T" * 15, "C" * 12, "G" * 8, "ATCG" * 3]
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
        max_seq_length=50,
        stride=10,
    )

    batches = list(dataloader)

    assert len(batches) == 3, f"Expected exactly 3 batches from 5 sequences, got {len(batches)}"
    assert batches[0]["input_ids"].shape[0] == 2
    assert batches[1]["input_ids"].shape[0] == 2
    assert batches[2]["input_ids"].shape[0] == 1


def test_non_streaming_dataset_produces_correct_batch_size(recipe_path):
    """Test that non-streaming dataset produces correct batch sizes."""
    distributed_config = DistributedConfig(rank=0, world_size=1)
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=["dataset.load_dataset_kwargs.streaming=False"],
        )

    dataloader, sampler = create_bshd_dataloader(
        distributed_config=distributed_config,
        **sanity_config.dataset,
    )

    assert isinstance(sampler, torch.utils.data.distributed.DistributedSampler)

    batches = list(itertools.islice(dataloader, 50))

    for batch in batches:
        assert batch["input_ids"].shape[0] == sanity_config.dataset.micro_batch_size
        assert batch["input_ids"].shape[1] <= sanity_config.dataset.max_seq_length


def test_batching_produces_correct_batch_size_sequence_packing(tokenizer_path, tmp_path):
    """Test that sequence packing batching works correctly."""
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
        torch.testing.assert_close(batch["input_ids"].squeeze(0), torch.tensor([[2, 65, 0] * 5]).flatten())


def test_dataloader_with_genomic_masking(tokenizer_path, tmp_path):
    """Test that create_bshd_dataloader works with genomic masking enabled."""
    parquet_path = tmp_path / "genomic_with_degenerate.parquet"
    sequences = ["ACGTN", "GGTAR"]
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
        max_seq_length=10,
        stride=5,
        mask_degenerate_bases=True,
    )

    batch = next(iter(dataloader))

    assert batch["input_ids"].ndim == 2
    assert batch["labels"].ndim == 2

    labels = batch["labels"]
    assert 78 not in labels, "Degenerate N (78) should be masked"
    assert 82 not in labels, "Degenerate R (82) should be masked"

    valid_dna = [65, 67, 71, 84]
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


@pytest.mark.parametrize(
    "sequence",
    [
        "ACGTACGT",
        "A" * 100,
        "TTTCCCGGGAAA",
    ],
)
def test_tokenizer_roundtrip_decode(tokenizer_path, sequence):
    """Test that encode -> decode round-trips correctly (no inserted spaces).

    The tokenizer uses a character-level WordLevel model with a Split pre-tokenizer,
    so each nucleotide becomes a separate token. Without the Fuse decoder, decoding
    inserts spaces between tokens (e.g., "AAA" -> [65,65,65] -> "A A A"). The Fuse
    decoder joins them back without spaces.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    token_ids = tokenizer.encode(sequence, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    assert decoded == sequence, f"Round-trip failed: '{sequence}' -> {token_ids} -> '{decoded}'"
