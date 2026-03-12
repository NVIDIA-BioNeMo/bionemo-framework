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


logger = logging.getLogger(__name__)


class TokenPackingBatchSampler:
    """Batch sampler that groups samples into batches bounded by a total token budget.

    Instead of a fixed number of samples per batch, this sampler accumulates samples until
    the total unpadded token count would exceed ``max_tokens_per_batch``.  This guarantees
    predictable peak memory in THD (packed) attention mode regardless of sequence-length
    distribution.

    The sampler requires the underlying dataset to expose a ``get_sequence_length(idx)``
    method that returns the unpadded token count for a given index **without** loading the
    full sample.

    Args:
        sampler: An iterable of dataset indices (e.g. ``DistributedSampler``).
        dataset: The dataset being sampled.  Must implement ``get_sequence_length(idx)``.
        max_tokens_per_batch: Upper bound on total unpadded tokens per batch.
        drop_last: If ``True`` (default), drop the final incomplete batch.
    """

    def __init__(self, sampler, dataset, max_tokens_per_batch: int, drop_last: bool = True):  # noqa: D107
        self.sampler = sampler
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.drop_last = drop_last

    def __iter__(self):  # noqa: D105
        batch: list[int] = []
        current_tokens = 0
        for idx in self.sampler:
            sample_length = self.dataset.get_sequence_length(idx)
            if sample_length > self.max_tokens_per_batch:
                logger.warning(
                    "Skipping sample %d: length %d exceeds max_tokens_per_batch %d",
                    idx,
                    sample_length,
                    self.max_tokens_per_batch,
                )
                continue

            if current_tokens + sample_length > self.max_tokens_per_batch:
                if batch:
                    yield batch
                batch = [idx]
                current_tokens = sample_length
            elif current_tokens + sample_length == self.max_tokens_per_batch:
                yield [*batch, idx]
                batch = []
                current_tokens = 0
            else:
                batch.append(idx)
                current_tokens += sample_length

        if batch and not self.drop_last:
            yield batch
