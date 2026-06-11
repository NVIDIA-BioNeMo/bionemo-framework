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


import pytest
import torch

from bionemo.llm.data.collate import bert_padding_collate_fn, padding_collate_fn


def test_padding_collate_fn():
    sample1 = {
        "my_key": torch.tensor([1, 2, 3]),
    }
    sample2 = {
        "my_key": torch.tensor([4, 5, 6, 7, 8]),
    }
    batch = [sample1, sample2]
    collated_batch = padding_collate_fn(batch, padding_values={"my_key": -1})

    assert torch.all(torch.eq(collated_batch["my_key"], torch.tensor([[1, 2, 3, -1, -1], [4, 5, 6, 7, 8]])))


def test_padding_collate_with_missing_keys_raises(caplog):
    sample1 = {
        "my_key": torch.tensor([1, 2, 3]),
    }
    sample2 = {
        "my_key": torch.tensor([4, 5, 6, 7, 8]),
        "other_key": torch.tensor([1, 2, 3]),
    }
    batch = [sample1, sample2]
    with pytest.raises(ValueError, match="All keys in inputs must match each other."):
        padding_collate_fn(batch, padding_values={"my_key": -1})


def test_padding_collate_with_mismatched_padding_values_warns(caplog):
    sample1 = {
        "my_key": torch.tensor([1, 2, 3]),
        "other_key": torch.tensor([1, 2, 3, 4]),
    }
    sample2 = {
        "my_key": torch.tensor([4, 5, 6, 7, 8]),
        "other_key": torch.tensor([1, 2, 3]),
    }
    batch = [sample1, sample2]

    padding_collate_fn(batch, padding_values={"my_key": -1, "other_key": -1, "missing_key": 3})
    # Call 2x and check that we logged once
    padding_collate_fn(batch, padding_values={"my_key": -1, "other_key": -1, "missing_key": 3})
    log_lines = caplog.text.strip("\n").split("\n")
    assert len(log_lines) == 1, f"Expected one line, got: {log_lines}"
    assert log_lines[0].endswith(
        "Extra keys in batch that will not be padded: set(). Missing keys in batch: {'missing_key'}"
    )
    assert log_lines[0].startswith("WARNING")


def test_bert_padding_collate_fn():
    # Create sample data
    sample1 = {
        "text": torch.tensor([1, 2, 3]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, False]),
        "labels": torch.tensor([7, 8, 9]),
        "loss_mask": torch.tensor([True, False, True]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    sample2 = {
        "text": torch.tensor([10, 11, 12]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, False, True]),
        "labels": torch.tensor([16, 17, 18]),
        "loss_mask": torch.tensor([False, True, False]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    batch = [sample1, sample2]

    # Call the collate_fn
    collated_batch = bert_padding_collate_fn(batch, padding_value=-1)

    # Assert the expected output
    assert torch.all(torch.eq(collated_batch["text"], torch.tensor([[1, 2, 3], [10, 11, 12]])))
    assert torch.all(torch.eq(collated_batch["types"], torch.tensor([[0, 0, 0], [0, 0, 0]])))
    assert torch.all(
        torch.eq(collated_batch["attention_mask"], torch.tensor([[True, True, False], [True, False, True]]))
    )
    assert torch.all(torch.eq(collated_batch["labels"], torch.tensor([[7, 8, 9], [16, 17, 18]])))
    assert torch.all(torch.eq(collated_batch["loss_mask"], torch.tensor([[True, False, True], [False, True, False]])))
    assert torch.all(torch.eq(collated_batch["is_random"], torch.tensor([[0, 0, 0], [0, 0, 0]])))


def test_bert_padding_collate_fn_with_padding():
    # Create sample data
    sample1 = {
        "text": torch.tensor([1, 2, 3]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, False]),
        "labels": torch.tensor([7, 8, 9]),
        "loss_mask": torch.tensor([True, False, True]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    sample2 = {
        "text": torch.tensor([4, 5, 6, 7, 8]),
        "types": torch.zeros((5,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, True, True, True]),
        "labels": torch.tensor([-100, 5, -100, 7, 8]),
        "loss_mask": torch.tensor([False, True, False, True, True]),
        "is_random": torch.zeros((5,), dtype=torch.int64),
    }
    batch = [sample1, sample2]

    # Call the collate_fn
    collated_batch = bert_padding_collate_fn(batch, padding_value=10)

    # Assert the expected output
    assert torch.all(torch.eq(collated_batch["text"], torch.tensor([[1, 2, 3, 10, 10], [4, 5, 6, 7, 8]])))
    assert torch.all(torch.eq(collated_batch["types"], torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])))
    assert torch.all(
        torch.eq(
            collated_batch["attention_mask"],
            torch.tensor([[True, True, False, False, False], [True, True, True, True, True]]),
        )
    )
    assert torch.all(torch.eq(collated_batch["labels"], torch.tensor([[7, 8, 9, -100, -100], [-100, 5, -100, 7, 8]])))
    assert torch.all(
        torch.eq(
            collated_batch["loss_mask"],
            torch.tensor([[True, False, True, False, False], [False, True, False, True, True]]),
        )
    )
    assert torch.all(torch.eq(collated_batch["is_random"], torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])))


def test_bert_padding_collate_fn_with_max_length_truncates():
    # Create sample data
    sample1 = {
        "text": torch.tensor([1, 2, 3]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, False]),
        "labels": torch.tensor([7, 8, 9]),
        "loss_mask": torch.tensor([True, False, True]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    sample2 = {
        "text": torch.tensor([4, 5, 6, 7, 8]),
        "types": torch.zeros((5,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, True, True, True]),
        "labels": torch.tensor([-100, 5, -100, 7, 8]),
        "loss_mask": torch.tensor([False, True, False, True, True]),
        "is_random": torch.zeros((5,), dtype=torch.int64),
    }
    batch = [sample1, sample2]

    # Call the collate_fn
    collated_batch = bert_padding_collate_fn(batch, padding_value=10, max_length=4)

    # Assert the expected output
    assert torch.all(torch.eq(collated_batch["text"], torch.tensor([[1, 2, 3, 10], [4, 5, 6, 7]])))
    assert torch.all(torch.eq(collated_batch["types"], torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])))
    assert torch.all(
        torch.eq(
            collated_batch["attention_mask"], torch.tensor([[True, True, False, False], [True, True, True, True]])
        )
    )
    assert torch.all(torch.eq(collated_batch["labels"], torch.tensor([[7, 8, 9, -100], [-100, 5, -100, 7]])))
    assert torch.all(
        torch.eq(collated_batch["loss_mask"], torch.tensor([[True, False, True, False], [False, True, False, True]]))
    )
    assert torch.all(torch.eq(collated_batch["is_random"], torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])))


def test_bert_padding_collate_fn_with_min_length_pads_extra():
    # Create sample data
    sample1 = {
        "text": torch.tensor([1, 2, 3]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, True, False]),
        "labels": torch.tensor([7, 8, 9]),
        "loss_mask": torch.tensor([True, False, True]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    sample2 = {
        "text": torch.tensor([10, 11, 12]),
        "types": torch.zeros((3,), dtype=torch.int64),
        "attention_mask": torch.tensor([True, False, True]),
        "labels": torch.tensor([16, 17, 18]),
        "loss_mask": torch.tensor([False, True, False]),
        "is_random": torch.zeros((3,), dtype=torch.int64),
    }
    batch = [sample1, sample2]

    # Call the collate_fn
    collated_batch = bert_padding_collate_fn(batch, padding_value=-1, min_length=5)
    assert torch.all(torch.eq(collated_batch["text"], torch.tensor([[1, 2, 3, -1, -1], [10, 11, 12, -1, -1]])))
    for val in collated_batch.values():
        assert val.size(1) == 5
