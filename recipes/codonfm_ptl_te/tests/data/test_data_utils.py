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


import pathlib

import numpy as np
import polars as pl
import pytest

from src.data.utils import (
    get_group_bits,
    load_train_val_test_indices_by_group,
    load_train_val_test_indices_proportional,
    sample_clusters_by_size,
)


@pytest.fixture
def mock_keys_df():
    # 20 samples, 5 groups
    return pl.DataFrame(
        {"group": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C", "D", "D", "D", "D", "E", "E", "E", "E"]}
    )


class TestLoadTrainValTestIndicesByGroup:
    def test_generation_and_saving(self, mock_keys_df, tmpdir):
        metadata_path = pathlib.Path(tmpdir) / "metadata.json"
        with open(metadata_path, "w") as f:
            f.write("{}")

        train_idx, val_idx, test_idx = load_train_val_test_indices_by_group(
            keys=mock_keys_df,
            metadata_path=metadata_path,
            train_val_test_ratio=[0.6, 0.2, 0.2],  # 3 groups train, 1 val, 1 test
            seed=42,
        )

        # 5 unique groups
        assert len(np.unique(mock_keys_df["group"][train_idx])) == 3
        assert len(np.unique(mock_keys_df["group"][val_idx])) == 1
        assert len(np.unique(mock_keys_df["group"][test_idx])) == 1

        # Check that files are saved
        assert (metadata_path.parent / "train_idx.npy").exists()
        assert (metadata_path.parent / "val_idx.npy").exists()
        assert (metadata_path.parent / "test_idx.npy").exists()

    def test_loading_from_disk(self, tmpdir):
        metadata_path = pathlib.Path(tmpdir) / "metadata.json"
        train_path = metadata_path.parent / "train_idx.npy"
        val_path = metadata_path.parent / "val_idx.npy"
        test_path = metadata_path.parent / "test_idx.npy"

        # Create dummy files
        np.save(train_path, np.array([1, 2]))
        np.save(val_path, np.array([3, 4]))
        np.save(test_path, np.array([5, 6]))

        train_idx, val_idx, test_idx = load_train_val_test_indices_by_group(keys=[], metadata_path=metadata_path)

        assert np.array_equal(train_idx, np.array([1, 2]))
        assert np.array_equal(val_idx, np.array([3, 4]))
        assert np.array_equal(test_idx, np.array([5, 6]))


class TestGetGroupBits:
    def test_valid_groups(self):
        bits = get_group_bits(["Primates", "archaea"])
        assert bits == 1 + 2  # 2^0 + 2^1

    def test_invalid_group(self):
        with pytest.raises(ValueError):
            get_group_bits(["invalid_group"])

    def test_all_groups(self):
        all_groups = [
            "Primates",
            "archaea",
            "bacteria",
            "fungi",
            "invertebrate",
            "plant",
            "protozoa",
            "vertebrate_mammalian",
            "vertebrate_other",
            "viral",
        ]
        for i, group in enumerate(all_groups):
            assert get_group_bits([group]) == 2**i

        assert get_group_bits(all_groups) == sum([2**i for i in range(len(all_groups))])


class TestSampleClustersBySize:
    def test_basic_sampling(self):
        count_mat = np.array(
            [
                [10, 0],  # cluster 0, org 0
                [0, 20],  # cluster 1, org 1
                [5, 5],  # cluster 2, org 0, 1
            ]
        )
        global_keys = np.array([0, 0, 1, 1, 1, 2, 2])  # 2 from c0, 3 from c1, 2 from c2

        indices, clusters = sample_clusters_by_size(count_mat, global_keys, target_size=4, seed=42)

        # The logic is complex, so we check for basic properties
        assert len(indices) > 0
        assert len(clusters) > 0
        assert np.all(np.isin(global_keys[indices], clusters))

    def test_used_clusters(self):
        count_mat = np.array([[10, 0], [0, 20]])
        global_keys = np.array([0, 0, 1, 1])

        indices, clusters = sample_clusters_by_size(count_mat, global_keys, used_clusters=[0], target_size=2)
        # Should only select from cluster 1
        assert clusters == [1]
        assert np.all(indices == np.array([2, 3]))


class TestLoadTrainValTestIndicesProportional:
    def test_loading_from_disk(self, tmpdir):
        metadata_path = pathlib.Path(tmpdir) / "metadata.json"
        train_path = metadata_path.parent / "train_idx.npy"
        val_path = metadata_path.parent / "val_idx.npy"
        test_path = metadata_path.parent / "test_idx.npy"

        np.save(train_path, np.array([1, 2]))
        np.save(val_path, np.array([3, 4]))
        np.save(test_path, np.array([5, 6]))

        train, val, test = load_train_val_test_indices_proportional(metadata_path, [], [])

        assert np.array_equal(train, np.array([1, 2]))
        assert np.array_equal(val, np.array([3, 4]))
        assert np.array_equal(test, np.array([5, 6]))
