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

from unittest.mock import MagicMock

import dataset as dataset_module
import datasets
from distributed_config import DistributedConfig


def test_thd_dataloader_forwards_tokenize_batch_size(monkeypatch):
    """Online packing can tune streaming-tokenization latency for long documents."""
    captured = {}
    tokenized_dataset = datasets.IterableDataset.from_generator(
        lambda: iter([{"input_ids": [1, 2], "attention_mask": [1, 1]}])
    )
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    def _create_tokenized_dataset(**kwargs):
        captured.update(kwargs)
        return tokenized_dataset, tokenizer

    monkeypatch.setattr(dataset_module, "create_tokenized_dataset", _create_tokenized_dataset)

    dataset_module.create_thd_dataloader(
        distributed_config=DistributedConfig(),
        tokenizer_name_or_path="unused",
        load_dataset_kwargs={"path": "unused"},
        token_micro_batch_size=8,
        max_seq_length=8,
        tokenize_batch_size=4,
        num_workers=0,
    )

    assert captured["tokenize_batch_size"] == 4
