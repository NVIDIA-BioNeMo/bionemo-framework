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

import os
from pathlib import Path

import pytest

from bionemo.esm2.model.finetune.flip_preprocess import FLIPPreprocess


def test_flip_preprocess_initialization(tmpdir):
    """Test FLIPPreprocess initialization with default and custom root directory."""
    # Test with default root directory
    flip = FLIPPreprocess()
    assert flip.root_directory == Path("/tmp/FLIP")

    # Test with custom root directory
    flip = FLIPPreprocess(root_directory=tmpdir)
    assert flip.root_directory == Path(tmpdir)


@pytest.mark.skip(reason="Need to fix the test")
def test_prepare_all_datasets(tmpdir):
    """Test prepare_all_datasets method."""
    flip = FLIPPreprocess(root_directory=tmpdir)
    output_dir = os.path.join(tmpdir, "output")

    # Run preprocessing for all datasets
    flip.prepare_all_datasets(output_dir=output_dir)

    # List of expected tasks
    expected_tasks = ["aav", "bind", "conservation", "gb1", "meltome", "sav", "scl", "secondary_structure"]

    # Verify each task directory and its contents
    for task in expected_tasks:
        task_dir = os.path.join(output_dir, task)
        assert os.path.exists(task_dir), f"Directory for task {task} not found"

        # Check train, val, and test directories
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(task_dir, split)
            assert os.path.exists(split_dir), f"{split} directory not found for task {task}"

            # Check CSV file
            csv_file = os.path.join(split_dir, "x000.csv")
            assert os.path.exists(csv_file), f"x000.csv not found in {task}/{split} directory"


@pytest.mark.skip(reason="Need to fix the test")
def test_download_flip_data(tmpdir):
    """Test download_FLIP_data method with slow marker."""
    flip = FLIPPreprocess(root_directory=tmpdir)
    download_dir = os.path.join(tmpdir, "download")

    # Test download for secondary_structure task
    seq_path, labels_path, resolved_path = flip.download_FLIP_data(
        download_dir=download_dir, task_name="secondary_structure"
    )

    # Verify downloaded files
    assert os.path.exists(seq_path), "Sequence file not downloaded"
    assert os.path.exists(labels_path), "Labels file not downloaded"
    assert os.path.exists(resolved_path), "Resolved file not downloaded"

    # Check file contents
    for file_path in [seq_path, labels_path, resolved_path]:
        assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"
