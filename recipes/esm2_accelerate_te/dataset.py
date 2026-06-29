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

# Create the dataset -- here, we just use a simple parquet file with some raw protein sequences
# stored in the repo itself to avoid external dependencies.

from datasets import IterableDataset, load_dataset
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling


def create_datasets_and_collator(
    tokenizer_name: str,
    train_load_dataset_kwargs: dict,
    eval_load_dataset_kwargs: dict,
    max_seq_length: int = 1024,
    truncate_eval_dataset: int | None = None,
):
    """Create datasets and a data collator to pass to the huggingface trainer.

    Args:
        tokenizer_name: The name of the tokenizer to pull from the HuggingFace Hub.
        train_load_dataset_kwargs: Keyword arguments to pass to `load_dataset` for the train dataset.
        eval_load_dataset_kwargs: Keyword arguments to pass to `load_dataset` for the eval dataset.
        max_seq_length: The maximum length of the protein sequences.
        truncate_eval_dataset: If not `None`, the eval dataset will be truncated to this number of examples.

    This assumes that the dataset has a "sequence" column that will be tokenized.

    Returns:
        Tuple of (train_dataset, eval_dataset, data_collator).
    """
    train_dataset = load_dataset(**train_load_dataset_kwargs)
    eval_dataset = load_dataset(**eval_load_dataset_kwargs)
    if truncate_eval_dataset is not None:
        if isinstance(eval_dataset, IterableDataset):
            raise ValueError(
                "Cannot truncate an IterableDataset, don't use streaming datasets for eval if you want to truncate."
            )
        eval_dataset = eval_dataset.select(range(truncate_eval_dataset))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        """Tokenize the protein sequences."""
        return tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=max_seq_length,
        )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=max_seq_length,
    )

    return train_dataset, eval_dataset, data_collator
