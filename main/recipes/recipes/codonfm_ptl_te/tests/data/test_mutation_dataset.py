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


from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.metadata import MetadataConstants, MetadataFields
from src.data.mutation_dataset import (
    FastRefAlt,
    MutationDataset,
    collate_fn,
)


@pytest.fixture
def mock_csv_path(tmpdir):
    data = {
        "id": [f"var_{i}" for i in range(10)],
        "ref_seq": ["A" * 300] * 10,
        "alt_seq": ["C" * 300] * 10,
        "ref_codon": ["AAA"] * 10,
        "alt_codon": ["CCC"] * 10,
        "codon_position": [10, 20, 30, 40, 50, 60, 70, 80, 90, 99],
        "label": [0, 1] * 5,
    }
    df = pd.DataFrame(data)
    path = Path(str(tmpdir)) / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestFastRefAlt:
    @pytest.mark.parametrize(
        "context_length, codon_pos, expected_rel_pos",
        [
            (20, 15, 10),  # Centered
            (20, 5, 5),  # Near start
            (20, 95, 15),  # Near end
            (120, 50, 50),  # Context larger than sequence
        ],
    )
    def test_call(self, context_length, codon_pos, expected_rel_pos):
        ref_cds = "A" * 300  # 100 codons

        extractor = FastRefAlt(context_length=context_length)
        ref_seq, rel_pos, ref_codon, alt_codon = extractor(ref_cds, codon_pos, "AAA", "CCC")

        expected_len_codons = min(context_length, len(ref_cds) // MetadataConstants.CODON_LENGTH)
        expected_len = expected_len_codons * MetadataConstants.CODON_LENGTH

        assert len(ref_seq) == expected_len
        assert rel_pos == expected_rel_pos
        assert ref_codon == "AAA"
        assert alt_codon == "CCC"


class TestMutationDataset:
    def test_init_and_split(self, mock_csv_path):
        ds = MutationDataset(
            data_path=mock_csv_path,
            tokenizer=None,
            process_item=MagicMock(),
            train_val_test_ratio=[0.8, 0.1, 0.1],
            seed=42,
        )
        assert len(ds.train_idx) == 8
        assert len(ds.val_idx) == 1
        assert len(ds.test_idx) == 1
        assert len(ds) == 8  # initial split is train

        # Test loading from cache
        ds2 = MutationDataset(
            data_path=mock_csv_path,
            tokenizer=None,
            process_item=MagicMock(),
            train_val_test_ratio=[0.8, 0.1, 0.1],
            seed=42,
        )
        assert np.array_equal(ds.train_idx, ds2.train_idx)

    def test_getitem(self, mock_csv_path):
        process_item_mock = MagicMock(return_value={"input_ids": [1, 2, 3]})
        ds = MutationDataset(
            data_path=mock_csv_path,
            tokenizer=None,
            process_item=process_item_mock,
            extract_seq=True,
            label_col="label",
        )

        item = ds[0]  # get first item of the full dataset

        process_item_mock.assert_called_once()
        assert item[MetadataFields.ID] == "var_0"
        assert item[MetadataFields.LABELS] == 0
        assert "input_ids" in item

    def test_split_management(self, mock_csv_path):
        ds = MutationDataset(
            data_path=mock_csv_path, tokenizer=None, process_item=MagicMock(), train_val_test_ratio=[0.8, 0.1, 0.1]
        )

        train_len = ds.get_train_num_samples()
        val_len = ds.get_val_num_samples()

        assert len(ds) == train_len

        val_ds = ds.get_validation(process_item=None)
        assert len(val_ds) == val_len
        assert val_ds.process_item is None
        # original ds should be unchanged
        assert ds.idxs is ds.train_idx


class TestCollateFn:
    def test_collate_fn(self):
        batch = [
            {MetadataFields.ID: "id1", "data": torch.tensor([1, 2])},
            {MetadataFields.ID: "id2", "data": torch.tensor([3, 4])},
        ]

        collated = collate_fn(batch)

        assert collated[MetadataFields.ID] == ["id1", "id2"]
        assert torch.equal(collated["data"], torch.tensor([[1, 2], [3, 4]]))
