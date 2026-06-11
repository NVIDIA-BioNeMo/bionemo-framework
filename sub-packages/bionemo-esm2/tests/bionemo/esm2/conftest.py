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
import torch

from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.testing.data.esm2 import create_mock_parquet_train_val_inputs, create_mock_protein_dataset


@pytest.fixture
def data_to_csv():
    """Create a mock protein dataset."""

    def _data_to_csv(data, path):
        csv_file = path / "protein_dataset.csv"
        # Create a DataFrame
        df = pd.DataFrame(data)
        # if df has 2 columns add column names sequences and labels else resolved
        if df.shape[1] == 1:
            df.columns = ["sequences"]
        elif df.shape[1] == 2:
            df.columns = ["sequences", "labels"]
        elif df.shape[1] == 3:
            df.columns = ["sequences", "labels", "resolved"]
        else:
            raise ValueError(f"Data has {df.shape[1]} columns, expected less than 4")

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file, index=False)
        return csv_file

    return _data_to_csv


@pytest.fixture
def tokenizer():
    """Return the ESM2 tokenizer."""
    return get_tokenizer()


@pytest.fixture
def dummy_protein_dataset(tmp_path):
    """Create a mock protein dataset."""
    return create_mock_protein_dataset(tmp_path)


@pytest.fixture
def dummy_parquet_train_val_inputs(tmp_path):
    """Create a mock protein train and val cluster parquet."""
    return create_mock_parquet_train_val_inputs(tmp_path)


@pytest.fixture
def dummy_data_per_token_classification_ft():
    """Fixture providing dummy data for per-token classification fine-tuning.

    Returns:
        list: A list of dummy data for per-token classification fine-tuning.
    """
    data = [
        (
            "TLILGWSDKLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
            "EEEECCCCCHHHHHHHHHHHHHHHCCCEEEEEECCCHHHHHHHHHCCCCCCCCCEEE",
            "101010101010101010101010101010101010101010101010101010101",
        ),
        ("LYSGDHSTQGARFLRDLAENTGRAEYELLSLF", "CCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC", "10101010101010101010101010101010"),
        (
            "GRFNVWLGGNESKIRQVLKAVKEIGVSPTLFAVYEKN",
            "HHHHHCCCCCHHHHHHHHHHHHHHCCCHHHHHHHHHH",
            "1010101010101010101010101010101010101",
        ),
        (
            "DELTALGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
            "HHHHHHHHHHCCCHHHHHCCCCCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC",
            "10101010101010101010101010101010101010101010101010101",
        ),
        (
            "KLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
            "CHHHHHHHHHHHHHHHCCCEEEEEECCCHHHHHHHHHCCCCCCCCCEEE",
            "1010101010101010101010101010101010101010101010101",
        ),
        (
            "LFGAIGNAISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
            "HHHHHHHHHHHHHCHHHHHHHHHHHHCCCEECCCEEEECCEEEEECC",
            "10101010101010101010101010101010101010101010101",
        ),
        (
            "LGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
            "HHHHHCCCHHHHHCCCCCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC",
            "101010101010101010101010101010101010101010101010",
        ),
        ("LYSGDHSTQGARFLRDLAENTGRAEYELLSLF", "CCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCC", "10101010101010101010101010101010"),
        (
            "ISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
            "HHHHHCHHHHHHHHHHHHCCCEECCCEEEECCEEEEECC",
            "101010101010101010101010101010101010101",
        ),
        (
            "SGSKASSDSQDANQCCTSCEDNAPATSYCVECSEPLCETCVEAHQRVKYTKDHTVRSTGPAKT",
            "CCCCCCCCCCCCCCCCCCCCCCCCCCEEECCCCEEECHHHHHHHHHCCCCCCCCEEECCCCCC",
            "101010101010101010101010101010101010101010101010101010101010101",
        ),
    ]
    return data


@pytest.fixture
def dummy_data_single_value_regression_ft(dummy_data_per_token_classification_ft):
    """Fixture providing dummy data for per-token classification fine-tuning.

    Returns:
        list: A list of dummy data for per-token classification fine-tuning.
    """
    data = [(seq, len(seq) / 100.0) for seq, _, _ in dummy_data_per_token_classification_ft]
    return data


@pytest.fixture
def dummy_data_single_value_classification_ft(dummy_data_per_token_classification_ft):
    """Fixture providing dummy data for per-token classification fine-tuning.

    Returns:
        list: A list of dummy data for per-token classification fine-tuning.
    """
    data = [(seq, f"Class_{label[0]}") for seq, label, _ in dummy_data_per_token_classification_ft]
    return data


@pytest.fixture
def dummy_protein_sequences(dummy_data_per_token_classification_ft):
    """Fixture providing dummy data for per-token classification fine-tuning.

    Returns:
        list: A list of dummy data for per-token classification fine-tuning.
    """
    data = [seq for seq, _, _ in dummy_data_per_token_classification_ft]
    return data


@pytest.fixture
def load_dcp():
    """Fixture to load distributed checkpoints.

    Returns:
        Callable: A function that takes a checkpoint directory path and returns the loaded state dict.
    """
    if not torch.cuda.is_available():
        pytest.skip("Distributed checkpoint loading requires CUDA")

    def _load_dcp(ckpt_dir):
        from pathlib import Path

        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader

        if not isinstance(ckpt_dir, Path):
            ckpt_dir = Path(ckpt_dir)
        fs_reader = FileSystemReader(ckpt_dir)
        metadata = fs_reader.read_metadata()

        # Create tensors directly on GPU
        state_dict = {
            k: torch.empty(tp.size, dtype=tp.properties.dtype, device="cuda")
            for k, tp in metadata.state_dict_metadata.items()
            if type(tp).__name__ == "TensorStorageMetadata"
            and not any(keyword in k for keyword in {"head", "adapter", "optimizer", "output"})
        }

        dcp.load(
            state_dict,
            storage_reader=fs_reader,
        )
        return state_dict

    return _load_dcp
