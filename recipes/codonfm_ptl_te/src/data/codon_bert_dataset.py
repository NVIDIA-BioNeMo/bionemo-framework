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


from typing import Callable

import pandas as pd
from torch.utils.data import Dataset

from src.data.metadata import MetadataFields


class CodonBertDataset(Dataset):
    """This dataset expects a CSV file with the following required columns:
    - id: Unique identifier for each sequence
    - value: Target value/label for the sequence
    - ref_seq: Reference DNA/RNA sequence (U nucleotides will be converted to T)

    Optional columns:
    - split: Data split indicator ('train', 'val', 'test')
    """  # noqa: D205, D415

    REQUIRED_COLUMNS = ["id", "value", "ref_seq"]  # noqa: RUF012
    OPTIONAL_COLUMNS = ["split"]  # noqa: RUF012

    def __init__(self, data_path, tokenizer, process_item, split_name="all", **kwargs):
        """Initialize the CodonBertDataset.

        Args:
            data_path (str): Path to the CSV file containing the dataset.
                           Must contain columns: 'id', 'value', 'ref_seq'
            tokenizer: Tokenizer object used to tokenize sequences
            process_item (Callable): Function to process individual sequence items.
                                   Should accept (sequence, tokenizer) and return dict
            split_name (str, optional): Which data split to use. Options:
                                      - 'all': Use entire dataset
                                      - 'train': Use only training split
                                      - 'val': Use only validation split
                                      - 'test': Use only test split
                                      Defaults to 'all'
            **kwargs: Additional keyword arguments (currently unused)

        Raises:
            FileNotFoundError: If data_path does not exist
            KeyError: If required columns are missing from the CSV
        """
        self.data_path = data_path
        self.data = pd.read_csv(data_path)

        # Validate required columns exist
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in self.data.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        self.data["ref_seq"] = self.data["ref_seq"].str.replace("U", "T")
        self.data = self.data.reset_index(drop=True)
        self.tokenizer = tokenizer
        if split_name == "train":
            self.data = self.data[self.data["split"] == "train"]
        elif split_name == "val":
            self.data = self.data[self.data["split"] == "val"]
        elif split_name == "test":
            self.data = self.data[self.data["split"] == "test"]

        self.process_item = process_item

    def __len__(self):  # noqa: D105
        return len(self.data)

    def __getitem__(self, idx):  # noqa: D105
        sequence = self.data.iloc[idx]["ref_seq"]
        items = self.process_item(sequence, tokenizer=self.tokenizer)
        items[MetadataFields.LABELS] = self.data.iloc[idx]["value"]
        items[MetadataFields.ID] = self.data.iloc[idx]["id"]
        return items

    def get_train(self, process_item: Callable = None) -> "CodonBertDataset":  # noqa: D102, RUF013
        process_item = process_item if process_item is not None else self.process_item
        return CodonBertDataset(
            data_path=self.data_path, tokenizer=self.tokenizer, process_item=process_item, split_name="train"
        )

    def get_validation(self, process_item: Callable = None) -> "CodonBertDataset":  # noqa: D102, RUF013
        process_item = process_item if process_item is not None else self.process_item
        return CodonBertDataset(
            data_path=self.data_path, tokenizer=self.tokenizer, process_item=process_item, split_name="val"
        )

    def get_test(self, process_item: Callable = None) -> "CodonBertDataset":  # noqa: D102, RUF013
        process_item = process_item if process_item is not None else self.process_item
        return CodonBertDataset(
            data_path=self.data_path, tokenizer=self.tokenizer, process_item=process_item, split_name="test"
        )
