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


from pathlib import Path

import pytest

from bionemo.testing.data.load import load


@pytest.fixture
def test_directory() -> Path:
    """Gets the path to the original synthetic single cell directory with test data (no feature ids).

    Returns:
        A Path object that is the directory with specified test data.
    """
    return load("scdl/sample") / "scdl_data"


@pytest.fixture
def test_directory_feat_ids() -> Path:
    """Gets the path to the directory with the synthetic single cell data (with the feature ids appended).

    Returns:
        A Path object that is the directory with specified test data.
    """
    return load("scdl/sample_scdl_feature_ids") / "scdl_data_with_feature_ids"


@pytest.fixture
def cellx_small_directory() -> Path:
    """Gets the path to the directory with with cellx small dataset in Single Cell Memmap format.

    Returns:
        A Path object that is the directory with the specified test data.
    """
    return load("single_cell/testdata-20241203") / "cellxgene_2023-12-15_small_processed_scdl"
