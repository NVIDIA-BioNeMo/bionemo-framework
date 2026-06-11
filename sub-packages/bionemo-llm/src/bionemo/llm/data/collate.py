# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Sequence, TypeVar

import torch

from bionemo.llm.data import types


logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=dict[str, torch.Tensor])
_warned_once: bool = False


MLM_LOSS_IGNORE_INDEX = -100  # This should match the masked value used in the MLM loss mask.


def padding_collate_fn(
    batch: Sequence[_T],
    padding_values: dict[str, int],
    min_length: int | None = None,
    max_length: int | None = None,
) -> _T:
    """Collate function with padding.

    Args:
        batch: List of samples, each of which is a dictionary of tensors.
        padding_values: A dictionary of padding values for each tensor key.
        min_length: Minimum length of the output batch; tensors will be padded to this length. If not
            provided, no extra padding beyond the max_length will be added.
        max_length: Maximum length of the sequence. If not provided, tensors will be padded to the
            longest sequence in the batch.

    Returns:
        A collated batch with the same dictionary input structure.
    """
    global _warned_once
    keys: set[str] | None = None

    if len(batch) == 0:  # empty batches passed through in DDP inference
        return {}

    for entry in batch:
        # First check that we have sane batches where keys align with each other.
        if keys is None:
            keys = set(entry.keys())
        else:
            if set(entry.keys()) != keys:
                raise ValueError(f"All keys in inputs must match each other. Got: {[sorted(e.keys()) for e in batch]}")
        if entry.keys() != padding_values.keys():
            if not _warned_once:
                extra_keys = {k for k in entry.keys() if k not in padding_values}
                missing_keys = {k for k in padding_values.keys() if k not in entry}
                logger.warning(
                    f"Extra keys in batch that will not be padded: {extra_keys}. Missing keys in batch: {missing_keys}"
                )
                _warned_once = True

    def _pad(tensors, padding_value):
        if max_length is not None:
            tensors = [t[:max_length] for t in tensors]
        batched_tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        if min_length is None:
            return batched_tensors
        return torch.nn.functional.pad(batched_tensors, (0, min_length - batched_tensors.size(1)), value=padding_value)

    return {
        k: _pad([s[k] for s in batch], padding_values[k])
        if k in padding_values
        else torch.stack([s[k] for s in batch])
        for k in batch[0].keys()
    }  # type: ignore[return-value]


def bert_padding_collate_fn(
    batch: Sequence[types.BertSample],
    padding_value: int,
    min_length: int | None = None,
    max_length: int | None = None,
) -> types.BertSample:
    """Padding collate function for BERT dataloaders.

    Args:
        batch (list): List of samples.
        padding_value (int, optional): The tokenizer's pad token ID.
        min_length: Minimum length of the output batch; tensors will be padded to this length. If not
            provided, no extra padding beyond the max_length will be added.
        max_length: Maximum length of the sequence. If not provided, tensors will be padded to the
            longest sequence in the batch.
    """
    padding_values = {
        "text": padding_value,
        "types": 0,
        "attention_mask": False,
        "labels": MLM_LOSS_IGNORE_INDEX,  # This should match the masked value used in the MLM loss mask.
        "loss_mask": False,
        "is_random": 0,
    }
    return padding_collate_fn(
        batch=batch,  # type: ignore[assignment]
        padding_values=padding_values,
        min_length=min_length,
        max_length=max_length,
    )
