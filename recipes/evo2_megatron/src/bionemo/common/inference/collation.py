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

"""Batch collation utilities for inference prediction workflows.

This module is part of bionemo.common and MUST NOT import from megatron,
nemo, or mbridge. It depends only on torch.
"""

from typing import List, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor


ReductionT = TypeVar("ReductionT")


def batch_collator(
    batches: Optional[Union[Tuple[ReductionT, ...], List[ReductionT]]],
    batch_dim: int = 0,
    seq_dim: int = 1,
    batch_dim_key_defaults: Optional[dict[str, int]] = None,
    seq_dim_key_defaults: Optional[dict[str, int]] = None,
    preferred_gpu: int = 0,
) -> Optional[ReductionT]:
    """Collate multiple batches into a single batch by concatenating along the batch dimension.

    This function handles nested structures (dicts, lists, tuples) containing tensors.
    Unlike PyTorch's default_collate, this assumes the batch dimension already exists
    (as when parallelizing across microbatches or DP ranks).

    Args:
        batches: Sequence of batches to collate. Each batch can be a tensor, dict, list, or tuple.
            The structure must be consistent across all batches.
        batch_dim: Dimension along which to concatenate tensors. Default 0.
        seq_dim: Sequence dimension, used for padding to max length. Default 1.
        batch_dim_key_defaults: For dict batches, override batch_dim for specific keys.
            Default: {"token_logits": 1} (legacy compatibility, recommend passing {}).
        seq_dim_key_defaults: For dict batches, override seq_dim for specific keys.
            Default: {"token_logits": 0} (legacy compatibility, recommend passing {}).
        preferred_gpu: If any tensor is on GPU, move all to this device. Default 0.

    Returns:
        Collated batch with same structure as input batches, or None if input contains None.

    Raises:
        ValueError: If batches is empty or contains unsupported types.

    Examples:
        >>> # Collate dict batches
        >>> batch1 = {"logits": torch.randn(2, 10, 512), "mask": torch.ones(2, 10)}
        >>> batch2 = {"logits": torch.randn(3, 10, 512), "mask": torch.ones(3, 10)}
        >>> result = batch_collator([batch1, batch2], batch_dim=0, seq_dim=1,
        ...                         batch_dim_key_defaults={}, seq_dim_key_defaults={})
        >>> result["logits"].shape  # torch.Size([5, 10, 512])

        >>> # Collate with padding (different sequence lengths)
        >>> batch1 = {"tokens": torch.randn(2, 100)}
        >>> batch2 = {"tokens": torch.randn(2, 150)}
        >>> result = batch_collator([batch1, batch2], batch_dim=0, seq_dim=1,
        ...                         batch_dim_key_defaults={}, seq_dim_key_defaults={})
        >>> result["tokens"].shape  # torch.Size([4, 150]) - padded to max length
    """
    if batch_dim_key_defaults is None:
        batch_dim_key_defaults = {"token_logits": 1}
    if seq_dim_key_defaults is None:
        seq_dim_key_defaults = {"token_logits": 0}

    match batches:
        case [None, *_]:
            return None

        case [Tensor(), *_]:
            return _collate_tensors(batches, batch_dim=batch_dim, seq_dim=seq_dim, preferred_gpu=preferred_gpu)

        case [dict(), *_]:
            return {
                key: batch_collator(
                    [batch[key] for batch in batches],
                    batch_dim=batch_dim_key_defaults.get(key, batch_dim),
                    seq_dim=seq_dim_key_defaults.get(key, seq_dim),
                    batch_dim_key_defaults=batch_dim_key_defaults,
                    seq_dim_key_defaults=seq_dim_key_defaults,
                    preferred_gpu=preferred_gpu,
                )
                for key in batches[0]
            }

        case [tuple(), *_]:
            return tuple(
                batch_collator(
                    [batch[i] for batch in batches],
                    batch_dim=batch_dim,
                    seq_dim=seq_dim,
                    batch_dim_key_defaults=batch_dim_key_defaults,
                    seq_dim_key_defaults=seq_dim_key_defaults,
                    preferred_gpu=preferred_gpu,
                )
                for i in range(len(batches[0]))
            )

        case [list(), *_]:
            return [
                batch_collator(
                    [batch[i] for batch in batches],
                    batch_dim=batch_dim,
                    seq_dim=seq_dim,
                    batch_dim_key_defaults=batch_dim_key_defaults,
                    seq_dim_key_defaults=seq_dim_key_defaults,
                    preferred_gpu=preferred_gpu,
                )
                for i in range(len(batches[0]))
            ]

        case []:
            raise ValueError("Cannot collate an empty sequence of batches")
        case _:
            raise ValueError(f"Unsupported batch type: {type(batches[0]) if batches else 'empty'}")


def _collate_tensors(
    tensors: List[Tensor],
    batch_dim: int,
    seq_dim: int,
    preferred_gpu: int,
) -> Tensor:
    """Concatenate tensors along batch dimension, padding sequence dimension if needed.

    Args:
        tensors: List of tensors to concatenate
        batch_dim: Dimension to concatenate along
        seq_dim: Dimension to pad to max length
        preferred_gpu: GPU device to use if any tensor is on GPU

    Returns:
        Concatenated tensor
    """
    if any(t.is_cuda for t in tensors):
        device = torch.device(f"cuda:{preferred_gpu}")
        tensors = [t.to(device) for t in tensors]

    if tensors[0].ndim == 1:
        return torch.cat(tensors, dim=0)

    max_seq_len = max(t.size(seq_dim) for t in tensors)
    padded_tensors = []

    for tensor in tensors:
        pad_amount = max_seq_len - tensor.size(seq_dim)
        if pad_amount > 0:
            pad_spec = [0] * (2 * tensor.ndim)
            pad_spec[2 * (tensor.ndim - 1 - seq_dim) + 1] = pad_amount
            padded_tensor = torch.nn.functional.pad(tensor, tuple(pad_spec))
        else:
            padded_tensor = tensor
        padded_tensors.append(padded_tensor)

    return torch.cat(padded_tensors, dim=batch_dim)
