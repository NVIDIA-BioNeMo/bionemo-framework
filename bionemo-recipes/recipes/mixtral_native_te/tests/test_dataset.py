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

"""Dataset and dataloader tests for the Mixtral native TE recipe."""

import gc
import sys
from pathlib import Path

import datasets
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


sys.path.insert(0, Path(__file__).parent.parent.as_posix())

from dataset import create_bshd_dataloader, create_thd_dataloader
from distributed_config import DistributedConfig


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _create_local_tokenizer(tmp_path: Path) -> str:
    """Create a small local WordLevel tokenizer that does not require HuggingFace Hub."""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "[UNK]": 0,
                "[PAD]": 1,
                "[BOS]": 2,
                "[EOS]": 3,
                "hello": 4,
                "world": 5,
                "token": 6,
                "checkpoint": 7,
                "mixtral": 8,
                "data": 9,
                "test": 10,
                "pack": 11,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    fast_tokenizer.save_pretrained(tokenizer_dir)
    return str(tokenizer_dir)


def _make_tiny_dataset():
    """Return an in-memory HuggingFace dataset with short repeated text."""
    return datasets.Dataset.from_dict({"text": ["hello world token checkpoint " * 50] * 20})


@pytest.fixture
def local_tokenizer(tmp_path):
    return _create_local_tokenizer(tmp_path)


@pytest.fixture
def tiny_parquet(tmp_path):
    """Write the tiny dataset to a parquet file and return its path."""
    ds = _make_tiny_dataset()
    path = tmp_path / "tiny.parquet"
    ds.to_parquet(str(path))
    return str(path)


def test_create_bshd_dataloader_returns_correct_batch_keys(local_tokenizer, tiny_parquet):
    """BSHD dataloader batches must contain input_ids, attention_mask, and labels with matching shapes."""
    dist_config = DistributedConfig(rank=0, world_size=1)
    micro_batch_size = 4
    max_seq_length = 32

    dataloader, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name_or_path=local_tokenizer,
        load_dataset_kwargs={"path": "parquet", "data_files": tiny_parquet, "split": "train"},
        micro_batch_size=micro_batch_size,
        num_workers=0,
        max_seq_length=max_seq_length,
        stride=10,
    )

    batch = next(iter(dataloader))

    assert "input_ids" in batch, "Batch missing input_ids"
    assert "attention_mask" in batch, "Batch missing attention_mask"
    assert "labels" in batch, "Batch missing labels"

    assert batch["input_ids"].shape[0] == micro_batch_size
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape

    _cleanup()


def test_create_thd_dataloader_returns_packed_batch(local_tokenizer, tiny_parquet):
    """THD dataloader batches must contain input_ids, labels, cu_seq_lens_q, and cu_seq_lens_k."""
    dist_config = DistributedConfig(rank=0, world_size=1)
    max_seq_length = 32

    dataloader, _ = create_thd_dataloader(
        distributed_config=dist_config,
        tokenizer_name_or_path=local_tokenizer,
        load_dataset_kwargs={"path": "parquet", "data_files": tiny_parquet, "split": "train", "streaming": True},
        token_micro_batch_size=128,
        num_workers=0,
        max_seq_length=max_seq_length,
        stride=10,
    )

    batch = next(iter(dataloader))

    assert "input_ids" in batch, "Batch missing input_ids"
    assert "labels" in batch, "Batch missing labels"
    assert "cu_seq_lens_q" in batch, "Batch missing cu_seq_lens_q"
    assert "cu_seq_lens_k" in batch, "Batch missing cu_seq_lens_k"

    _cleanup()


def test_bshd_dataloader_sequence_length(local_tokenizer, tiny_parquet):
    """BSHD batches must not exceed max_seq_length in the sequence dimension."""
    dist_config = DistributedConfig(rank=0, world_size=1)
    max_seq_length = 16

    dataloader, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name_or_path=local_tokenizer,
        load_dataset_kwargs={"path": "parquet", "data_files": tiny_parquet, "split": "train"},
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=max_seq_length,
        stride=5,
    )

    for batch in dataloader:
        seq_len = batch["input_ids"].shape[1]
        assert seq_len <= max_seq_length, f"Sequence length {seq_len} exceeds max_seq_length {max_seq_length}"

    _cleanup()


def test_thd_dataloader_token_packing_no_padding(local_tokenizer, tiny_parquet):
    """THD batches should have batch_size=1 because sequences are packed into a single flat tensor."""
    dist_config = DistributedConfig(rank=0, world_size=1)
    max_seq_length = 32

    dataloader, _ = create_thd_dataloader(
        distributed_config=dist_config,
        tokenizer_name_or_path=local_tokenizer,
        load_dataset_kwargs={"path": "parquet", "data_files": tiny_parquet, "split": "train", "streaming": True},
        token_micro_batch_size=128,
        num_workers=0,
        max_seq_length=max_seq_length,
        stride=10,
    )

    for batch in dataloader:
        assert batch["input_ids"].shape[0] == 1, (
            f"THD packed batch should have batch_size=1, got {batch['input_ids'].shape[0]}"
        )

    _cleanup()
