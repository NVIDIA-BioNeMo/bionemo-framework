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

"""Dataset and dataloader creation for OpenGenome2 Llama pre-training.

Simplified dataset module that always shuffles after tokenization for best
batch diversity. Supports both windowed and pre-chunked tokenization paths.
"""

import logging

import datasets
import datasets.distributed
from torch.utils.data import DataLoader, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from collator import (
    DataCollatorWithFlattening,
    TokenPackingDataset,
)
from distributed_config import DistributedConfig
from opengenome_collator import GenomicDataCollator


logger = logging.getLogger(__name__)


def create_tokenized_dataset(
    distributed_config: DistributedConfig,
    tokenizer_name_or_path: str,
    load_dataset_kwargs: dict,
    max_seq_length: int | None = 8192,
    stride: int | None = 200,
    buffer_size: int = 50_000,
    text_column: str = "text",
    tokenize_batch_size: int = 100,
):
    """Create a tokenized dataset, optionally with windowing.

    When ``max_seq_length`` and ``stride`` are both provided, long sequences are chunked into
    overlapping windows of ``max_seq_length`` tokens with ``stride`` overlap using the tokenizer's
    ``return_overflowing_tokens`` mechanism.

    When ``stride`` is ``None``, sequences are assumed to be pre-chunked (e.g. from globally-shuffled
    shards) and are tokenized directly with BOS/EOS tokens added. No windowing or truncation is applied.

    Streaming datasets are always shuffled after tokenization for best batch diversity.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name_or_path: Name or path to the nucleotide tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        max_seq_length: The maximum length of sequences (window size). Only used when stride is not None.
        stride: The stride for windowing (overlap = stride tokens). Set to None to disable windowing
            for pre-chunked datasets.
        buffer_size: The buffer size for shuffle operations.
        text_column: Name of the column containing genomic sequences (default: "text").
        tokenize_batch_size: The batch size for tokenization.

    Returns:
        Tuple of (tokenized_dataset, tokenizer).
    """
    use_windowing = stride is not None

    logger.info(f"Loading dataset with kwargs: {load_dataset_kwargs}")
    dataset = datasets.load_dataset(**load_dataset_kwargs)

    if isinstance(dataset, datasets.IterableDataset):
        # Hugging Face's `split_dataset_by_node` is quite sensitive to the total number of shards -- if the number of
        # shards is not perfectly divisible by the world size, it defaults to loading the same shards on all nodes and
        # using strided sampling to avoid loading the same data on all nodes. This can be quite inefficient with large
        # numbers of shards and workers, so we use `dataset.shard` instead.
        if distributed_config.world_size > dataset.num_shards:
            logger.info(f"Sharding dataset with {dataset.num_shards} shards with split_dataset_by_node")
            dataset = datasets.distributed.split_dataset_by_node(
                dataset, rank=distributed_config.rank, world_size=distributed_config.world_size
            )
        else:
            logger.info(f"Sharding dataset with {dataset.num_shards} shards with dataset.shard")
            dataset = dataset.shard(num_shards=distributed_config.world_size, index=distributed_config.rank)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    if use_windowing:

        def tokenize_with_windowing(examples):
            """Tokenize nucleotide sequences with windowing (one-to-many mapping)."""
            result = tokenizer(
                examples[text_column],
                max_length=max_seq_length,
                stride=stride,
                truncation=True,
                return_overflowing_tokens=True,
                add_special_tokens=True,
            )
            return result

        tokenize_fn = tokenize_with_windowing
        logger.info(f"Using windowed tokenization: max_seq_length={max_seq_length}, stride={stride}")
    else:

        def tokenize_direct(examples):
            """Tokenize pre-chunked sequences directly, adding only BOS/EOS tokens."""
            result = tokenizer(
                examples[text_column],
                add_special_tokens=True,
                truncation=False,
            )
            return result

        tokenize_fn = tokenize_direct
        logger.info("Using direct tokenization (pre-chunked dataset, no windowing)")

    tokenized_dataset = dataset.select_columns(text_column).map(
        tokenize_fn,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=[text_column],
    )

    # Always shuffle after tokenization for best batch diversity
    if isinstance(tokenized_dataset, datasets.IterableDataset):
        logger.info(f"Shuffling tokenized windows with buffer_size={buffer_size}")
        tokenized_dataset = tokenized_dataset.shuffle(seed=42, buffer_size=buffer_size)

    # Even in THD mode, we use a base MLM collator that requires a padding token to be set.
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
    max_seq_length: int | None = 8192,
    stride: int | None = 200,
    seed: int = 42,
    buffer_size: int = 50_000,
    use_stateful_dataloader: bool = False,
    text_column: str = "text",
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = True,
    pad_sequences_to_be_divisible_by: int | None = None,
):
    """Create a BSHD dataloader for OpenGenome2 pre-training.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name_or_path: Name or path to the nucleotide tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        micro_batch_size: The batch size per device.
        num_workers: The number of workers to use for the dataloader.
        prefetch_factor: The prefetch factor to use for the dataloader.
        max_seq_length: The maximum length of sequences (window size).
        stride: The stride for windowing (overlap = stride tokens).
        seed: The seed to use for the distributed sampler and data collator.
        buffer_size: The buffer size for shuffle operations.
        use_stateful_dataloader: Whether to use the StatefulDataLoader.
        text_column: Name of the column containing text sequences (default: "text").
        uppercase_labels: Whether to uppercase labels (genomic masking). Default: False.
        mask_degenerate_bases: Whether to mask non-ACGT bases in labels. Default: True.
        pad_sequences_to_be_divisible_by: The number to pad sequences to be divisible by, required for FP8 training.

    Returns:
        A tuple of (dataloader, dataset_or_sampler).
    """
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

    data_collator = GenomicDataCollator(
        base_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_sequences_to_be_divisible_by,
        ),
        uppercase_labels=uppercase_labels,
        mask_degenerate_bases=mask_degenerate_bases,
    )

    dataloader_class = StatefulDataLoader if use_stateful_dataloader else DataLoader
    train_dataloader = dataloader_class(
        tokenized_dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True if not use_stateful_dataloader else False,
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
    max_seq_length: int | None = 8192,
    stride: int | None = 200,
    buffer_size: int = 50_000,
    use_stateful_dataloader: bool = False,
    text_column: str = "text",
    uppercase_labels: bool = False,
    mask_degenerate_bases: bool = True,
    split_samples_in_token_packing: bool = True,
    pad_sequences_to_be_divisible_by: int | None = None,
):
    """Create a dataloader that packs up to the maximum number of tokens per batch.

    Args:
        distributed_config: The distributed configuration.
        tokenizer_name_or_path: Name or path to the nucleotide tokenizer directory.
        load_dataset_kwargs: Keyword arguments to pass to `load_dataset`.
        micro_batch_size: The batch size per device.
        token_micro_batch_size: The maximum number of tokens per batch. If None, the micro_batch_size * max_seq_length
            will be used. Defaults to None.
        num_workers: The number of workers to use for the dataloader.
        prefetch_factor: The prefetch factor to use for the dataloader.
        max_seq_length: The maximum length of sequences (window size).
        stride: The stride for windowing (overlap = stride tokens).
        buffer_size: The buffer size for shuffle operations.
        use_stateful_dataloader: Whether to use the StatefulDataLoader.
        text_column: Name of the column containing genomic sequences (default: "text").
        uppercase_labels: Whether to uppercase labels (genomic masking). Default: False.
        mask_degenerate_bases: Whether to mask non-ACGT bases in labels. Default: True.
        split_samples_in_token_packing: Whether to split samples to form batches with exactly token_micro_batch_size
            tokens. Default: True.
        pad_sequences_to_be_divisible_by: If provided, sequences will be padded to be divisible by this value.

    Returns:
        A tuple of (dataloader, dataset_or_sampler).
    """
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_name_or_path=tokenizer_name_or_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=max_seq_length,
        stride=stride,
        buffer_size=buffer_size,
        text_column=text_column,
    )

    assert isinstance(tokenized_dataset, datasets.IterableDataset), "THD token packing requires a streaming dataset."
    if token_micro_batch_size is None:
        assert micro_batch_size is not None, "Only one of micro_batch_size or token_micro_batch_size can be provided."
        assert max_seq_length is not None, (
            "max_seq_length must be set when using micro_batch_size (needed to compute token_micro_batch_size). "
            "Use token_micro_batch_size directly for pre-chunked datasets."
        )
        token_micro_batch_size = micro_batch_size * max_seq_length
    else:
        assert micro_batch_size is None, "Only one of micro_batch_size or token_micro_batch_size can be provided."

    data_collator = GenomicDataCollator(
        base_collator=DataCollatorWithFlattening(
            collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            pad_sequences_to_be_divisible_by=pad_sequences_to_be_divisible_by,
        ),
        uppercase_labels=uppercase_labels,
        mask_degenerate_bases=mask_degenerate_bases,
    )

    dataloader_class = StatefulDataLoader if use_stateful_dataloader else DataLoader
    train_dataloader = dataloader_class(
        TokenPackingDataset(
            tokenized_dataset,
            max_tokens_per_batch=token_micro_batch_size,
            split_samples=split_samples_in_token_packing,
        ),
        batch_size=None,  # The TokenPackingDataset will handle the batching.
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True if not use_stateful_dataloader else False,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_dataloader, tokenized_dataset
