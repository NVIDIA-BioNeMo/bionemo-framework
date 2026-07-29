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

"""Streaming DCLM dataset with online, padding-free next-token packing."""

from __future__ import annotations

import glob
from dataclasses import dataclass, replace
from typing import Iterator

import datasets
import torch
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset, get_worker_info


@dataclass
class DCLMPackedIterableDataset(IterableDataset):
    """Stream DCLM and yield fixed-size, shifted next-token prediction packs.

    AutoModel computes cross entropy outside the Hugging Face model, so samples
    must contain already-shifted labels. Each document window is shifted before
    packing, which also prevents targets from crossing document boundaries.
    """

    tokenizer: object
    data_files: str | list[str]
    seq_len: int = 4096
    stride: int = 200
    text_column: str = "text"
    tokenize_batch_size: int = 4
    shuffle_buffer_size: int = 1_000
    seed: int = 42
    split: str = "train"
    batch_size: int = 1
    _num_shards: int = 1
    _shard_index: int = 0

    def __post_init__(self) -> None:
        """Validate packing settings and initialize the PyTorch base class."""
        IterableDataset.__init__(self)
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if self.stride < 0 or self.stride >= self.seq_len:
            raise ValueError("stride must satisfy 0 <= stride < seq_len")
        if self.batch_size != 1:
            raise ValueError("DCLMPackedIterableDataset requires step_scheduler.local_batch_size=1")

    def shard(self, num_shards: int, index: int) -> DCLMPackedIterableDataset:
        """Return the rank-local view requested by AutoModel."""
        if not 0 <= index < num_shards:
            raise ValueError(f"invalid shard index {index} for {num_shards} shards")
        return replace(self, _num_shards=num_shards, _shard_index=index)

    def _resolved_files(self) -> list[str]:
        patterns = [self.data_files] if isinstance(self.data_files, str) else self.data_files
        files = sorted({path for pattern in patterns for path in glob.glob(pattern)})
        if not files:
            raise FileNotFoundError(f"No DCLM parquet files matched: {patterns}")
        return files

    def _stream(self):
        stream = datasets.load_dataset(
            "parquet",
            data_files=self._resolved_files(),
            split=self.split,
            streaming=True,
        )

        worker = get_worker_info()
        worker_count = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        num_shards = self._num_shards * worker_count
        shard_index = self._shard_index * worker_count + worker_id

        if num_shards > 1:
            if num_shards <= stream.num_shards:
                stream = stream.shard(num_shards=num_shards, index=shard_index)
            else:
                stream = split_dataset_by_node(
                    stream,
                    world_size=num_shards,
                    rank=shard_index,
                )
        return stream.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)

    def _shifted_windows(self) -> Iterator[tuple[list[int], list[int]]]:
        rows: list[str] = []
        for example in self._stream():
            rows.append(example[self.text_column])
            if len(rows) < self.tokenize_batch_size:
                continue
            yield from self._tokenize_and_shift(rows)
            rows.clear()
        if rows:
            yield from self._tokenize_and_shift(rows)

    def _tokenize_and_shift(self, texts: list[str]) -> Iterator[tuple[list[int], list[int]]]:
        encoded = self.tokenizer(
            texts,
            max_length=self.seq_len,
            stride=self.stride,
            truncation=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )
        for token_ids in encoded["input_ids"]:
            if len(token_ids) >= 2:
                yield token_ids[:-1], token_ids[1:]

    def __iter__(self):
        """Yield full packs, dropping only the final incomplete pack."""
        input_pack: list[int] = []
        label_pack: list[int] = []
        position_ids: list[int] = []
        sequence_lengths: list[int] = []

        for input_ids, labels in self._shifted_windows():
            offset = 0
            while offset < len(input_ids):
                take = min(self.seq_len - len(input_pack), len(input_ids) - offset)
                input_pack.extend(input_ids[offset : offset + take])
                label_pack.extend(labels[offset : offset + take])
                position_ids.extend(range(take))
                sequence_lengths.append(take)
                offset += take

                if len(input_pack) == self.seq_len:
                    yield {
                        "input_ids": input_pack,
                        "labels": label_pack,
                        "position_ids": position_ids,
                        "seq_lens": sequence_lengths,
                        "seq_lens_padded": sequence_lengths,
                    }
                    input_pack = []
                    label_pack = []
                    position_ids = []
                    sequence_lengths = []


def packed_collater(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Batch position-reset packs without invoking AutoModel's TE THD conversion."""
    if len(batch) != 1:
        raise ValueError("packed_collater requires local_batch_size=1")
    sample = batch[0]
    return {
        key: torch.tensor(sample[key], dtype=torch.long).unsqueeze(0)
        for key in ("input_ids", "labels", "position_ids")
    }
