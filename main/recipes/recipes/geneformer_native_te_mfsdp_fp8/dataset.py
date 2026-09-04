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

import pathlib
from typing import Dict, Iterator, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling


def infinite_dataloader(dataloader: DataLoader, sampler: DistributedSampler) -> Iterator[Dict[str, torch.Tensor]]:
    """Create an infinite dataloader that loops through epochs.

    Args:
        dataloader: The DataLoader to loop through.
        sampler: The DistributedSampler to set epochs for.

    Yields:
        Dict containing batch data.
    """
    epoch = 0
    while True:
        sampler.set_epoch(epoch)  # Update epoch for proper shuffling
        for batch in dataloader:
            yield batch
        epoch += 1  # Increment epoch counter after completing one full pass


def create_dataloader(
    path: pathlib.Path | str,
    batch_size: int,
    num_workers: int,
    use_fp8: bool = False,
    tokenizer_path: str = "tokenizer_auto",
) -> Tuple[Iterator[Dict[str, torch.Tensor]], int]:
    """Create a dataloader for the geneformer dataset.

    Args:
        path: path to the parquet file
        batch_size: batch size
        num_workers: number of workers
        use_fp8: whether to use fp8
        tokenizer_path: path to the tokenizer
    Returns:
        train_iterator: iterator for the training data
        dataloader_length: length of the dataloader
    """
    dataset = load_dataset("parquet", data_files=path, split="train")
    dataset = dataset.remove_columns(["length"])
    # Note: the geneformer data is already tokenized, so we use the tokenizer is simply for padding + masking.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        pad_to_multiple_of=16 if use_fp8 else None,
    )
    train_sampler = DistributedSampler(dataset)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_iterator = infinite_dataloader(train_dataloader, train_sampler)
    return train_iterator, len(train_dataloader)
