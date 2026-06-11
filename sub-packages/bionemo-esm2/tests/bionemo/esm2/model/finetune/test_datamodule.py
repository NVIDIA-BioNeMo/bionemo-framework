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


import pandas as pd
import pytest
from torch.utils.data import DataLoader

from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.dataset import InMemoryProteinDataset


@pytest.fixture
def dummy_protein_csv(tmp_path, dummy_protein_sequences):
    """Create a mock protein dataset."""
    csv_file = tmp_path / "protein_dataset.csv"
    # Create a DataFrame
    df = pd.DataFrame(dummy_protein_sequences, columns=["sequences"])

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def dataset(dummy_protein_csv):
    return InMemoryProteinDataset.from_csv(dummy_protein_csv, ignore_labels=True)


@pytest.fixture
def data_module(dataset):
    return ESM2FineTuneDataModule(predict_dataset=dataset)


def test_in_memory_csv_dataset(dataset):
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "text" in sample
    assert "labels" in sample


def test_esm2_fine_tune_data_module_init(data_module):
    assert data_module.train_dataset is None
    assert data_module.valid_dataset is None
    assert data_module.predict_dataset is not None


def test_esm2_fine_tune_data_module_predict_dataloader(data_module):
    predict_dataloader = data_module.predict_dataloader()
    assert isinstance(predict_dataloader, DataLoader)
    batch = next(iter(predict_dataloader))
    assert isinstance(batch, dict)
    assert "text" in batch


def test_esm2_fine_tune_data_module_setup(data_module):
    with pytest.raises(RuntimeError):
        data_module.setup("fit")


def test_esm2_fine_tune_data_module_train_dataloader(data_module):
    with pytest.raises(AttributeError):
        data_module.train_dataloader()


def test_esm2_fine_tune_data_module_val_dataloader(data_module):
    with pytest.raises(AttributeError):
        data_module.val_dataloader()
