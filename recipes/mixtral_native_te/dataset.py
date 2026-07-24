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

import logging

import datasets
import datasets.distributed
from collator import (
    DataCollatorWithFlattening,
    TokenPackingDataset,
)
from distributed_config import DistributedConfig
from torch.utils.data import DataLoader, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling


logger = logging.getLogger(__name__)


def create_tokenized_dataset(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    max_seq_length: int = 512,
    stride: int = 128,
    buffer_size: int = 5_000,
    text_column: str = "text",
    tokenize_batch_size: int = 100,
):
    """Create a tokenized dataset with windowing for small Mixtral pre-training.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name_or_path: Name or path to the tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        max_seq_length: The maximum length of sequences (window size).
        stride: The stride for windowing (overlap = stride tokens).
        buffer_size: The buffer size for shuffle.
        text_column: Name of the column containing text sequences.
        tokenize_batch_size: The batch size for tokenization.

    Returns:
        Tuple of (tokenized_dataset, tokenizer).
    """
    logger.info(f"Loading dataset with kwargs: {load_dataset_kwargs}")
    dataset = datasets.load_dataset(**load_dataset_kwargs)

    if isinstance(dataset, datasets.IterableDataset):
        if distributed_config.world_size > dataset.num_shards:
            logger.info(f"Sharding dataset with {dataset.num_shards} shards with split_dataset_by_node")
            dataset = datasets.distributed.split_dataset_by_node(
                dataset, rank=distributed_config.rank, world_size=distributed_config.world_size
            )
        else:
            logger.info(f"Sharding dataset with {dataset.num_shards} shards with dataset.shard")
            dataset = dataset.shard(num_shards=distributed_config.world_size, index=distributed_config.rank)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def tokenize_with_windowing(examples):
        """Tokenize text sequences with windowing (one-to-many mapping)."""
        return tokenizer(
            examples[text_column],
            max_length=max_seq_length,
            stride=stride,
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )

    tokenized_dataset = dataset.select_columns(text_column).map(
        tokenize_with_windowing,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=[text_column],
    )

    if isinstance(tokenized_dataset, datasets.IterableDataset):
        tokenized_dataset = tokenized_dataset.shuffle(seed=42, buffer_size=buffer_size)

    if tokenizer.pad_token is None:
        logger.warning(f"Tokenizer does not have a padding token. Setting it to the EOS token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenized_dataset, tokenizer


def create_bshd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int,
    num_workers: int = 1,
    prefetch_factor: int = 4,
    max_seq_length: int = 512,
    stride: int = 128,
    seed: int = 42,
    buffer_size: int = 5_000,
    use_stateful_dataloader: bool = False,
    text_column: str = "text",
    pad_sequences_to_be_divisible_by: int | None = None,
):
    """Create a BSHD dataloader for Mixtral pre-training."""
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_name_or_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=max_seq_length,
        stride=stride,
        buffer_size=buffer_size,
        text_column=text_column,
        tokenize_batch_size=micro_batch_size * prefetch_factor,
    )

    if isinstance(tokenized_dataset, datasets.IterableDataset):
        sampler = None
    else:
        sampler = DistributedSampler(
            tokenized_dataset,
            rank=distributed_config.rank,
            num_replicas=distributed_config.world_size,
            seed=seed,
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_sequences_to_be_divisible_by,
    )

    dataloader_class = StatefulDataLoader if use_stateful_dataloader else DataLoader
    train_dataloader = dataloader_class(
        tokenized_dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_dataloader, tokenized_dataset if sampler is None else sampler


def create_thd_dataloader(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    micro_batch_size: int | None = None,
    token_micro_batch_size: int | None = None,
    num_workers: int = 1,
    prefetch_factor: int = 4,
    max_seq_length: int = 512,
    stride: int = 128,
    buffer_size: int = 5_000,
    tokenize_batch_size: int = 100,
    use_stateful_dataloader: bool = False,
    text_column: str = "text",
    split_samples_in_token_packing: bool = True,
    pad_sequences_to_be_divisible_by: int | None = None,
):
    """Create a dataloader that packs up to the maximum number of tokens per batch."""
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_name_or_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=max_seq_length,
        stride=stride,
        buffer_size=buffer_size,
        text_column=text_column,
        tokenize_batch_size=tokenize_batch_size,
    )

    assert isinstance(tokenized_dataset, datasets.IterableDataset), "THD token packing requires a streaming dataset."
    if token_micro_batch_size is None:
        assert micro_batch_size is not None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        token_micro_batch_size = micro_batch_size * max_seq_length
    else:
        assert micro_batch_size is None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        assert token_micro_batch_size >= max_seq_length, "token_micro_batch_size must be greater than max_seq_length."

    data_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        separator_id=-100,
        pad_sequences_to_be_divisible_by=pad_sequences_to_be_divisible_by,
    )

    dataloader_class = StatefulDataLoader if use_stateful_dataloader else DataLoader
    train_dataloader = dataloader_class(
        TokenPackingDataset(
            tokenized_dataset,
            max_tokens_per_batch=token_micro_batch_size,
            split_samples=split_samples_in_token_packing,
        ),
        batch_size=None,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_dataloader, tokenized_dataset
