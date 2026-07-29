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

"""Tests for the NeMo AutoModel benchmark dataset adapter."""

from __future__ import annotations

import torch

from . import dataset as benchmark_dataset
from . import mxfp8


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        del kwargs
        return {"input_ids": [[10, 11, 12], [20, 21, 22, 23]][: len(texts)]}


def test_packs_shifted_samples_without_crossing_boundaries(monkeypatch):
    rows = [{"text": "first"}, {"text": "second"}]
    dataset = benchmark_dataset.DCLMPackedIterableDataset(
        tokenizer=FakeTokenizer(),
        data_files="unused",
        seq_len=4,
        stride=1,
        tokenize_batch_size=2,
    )
    monkeypatch.setattr(dataset, "_stream", lambda: iter(rows))

    pack = next(iter(dataset))

    assert pack["input_ids"] == [10, 11, 20, 21]
    assert pack["labels"] == [11, 12, 21, 22]
    assert pack["position_ids"] == [0, 1, 0, 1]
    assert pack["seq_lens"] == [2, 2]


def test_split_window_resets_position_ids_for_next_pack(monkeypatch):
    class LongTokenizer:
        def __call__(self, texts, **kwargs):
            del texts, kwargs
            return {"input_ids": [[1, 2, 3, 4, 5, 6, 7]]}

    dataset = benchmark_dataset.DCLMPackedIterableDataset(
        tokenizer=LongTokenizer(),
        data_files="unused",
        seq_len=4,
        stride=1,
        tokenize_batch_size=1,
    )
    monkeypatch.setattr(dataset, "_stream", lambda: iter([{"text": "long"}, {"text": "long"}]))

    packs = iter(dataset)
    first = next(packs)
    second = next(packs)

    assert first["input_ids"] == [1, 2, 3, 4]
    assert first["labels"] == [2, 3, 4, 5]
    assert first["position_ids"] == [0, 1, 2, 3]
    assert second["input_ids"] == [5, 6, 1, 2]
    assert second["position_ids"] == [0, 1, 0, 1]
    assert second["seq_lens"] == [2, 2]


def test_collater_preserves_a_two_dimensional_position_tensor():
    batch = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [2, 3, 4, 5],
            "position_ids": [0, 1, 0, 1],
            "seq_lens": [2, 2],
            "seq_lens_padded": [2, 2],
        }
    ]

    result = benchmark_dataset.packed_collater(batch)

    assert result["input_ids"].shape == (1, 4)
    assert result["labels"].shape == (1, 4)
    assert result["position_ids"].shape == (1, 4)


def test_mxfp8_padding_preserves_rows_and_aligns_groups():
    values = torch.arange(15).reshape(5, 3)
    offsets = torch.tensor([2, 5], dtype=torch.int32)
    sentinel_mask = torch.zeros(5, dtype=torch.bool)

    padded, indices, padded_offsets = mxfp8.pad_expert_groups(
        values,
        offsets,
        sentinel_mask,
        alignment=4,
    )

    assert padded_offsets.tolist() == [4, 8]
    assert indices.tolist() == [0, 1, 4, 5, 6]
    assert torch.equal(padded[indices], values)


def test_mxfp8_implementation_is_registered():
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    mxfp8.register_mxfp8_experts()

    assert mxfp8.IMPLEMENTATION_NAME in ALL_EXPERTS_FUNCTIONS
