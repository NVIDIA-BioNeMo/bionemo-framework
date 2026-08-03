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

"""File existence and basic content checks."""

from pathlib import Path


def assert_files_exist(base_dir: Path, filenames: list[str]) -> None:
    """Assert that all specified files exist in the base directory."""
    missing = [f for f in filenames if not (base_dir / f).exists()]
    assert not missing, f"Missing expected files: {missing}"


def assert_file_not_empty(filepath: Path) -> None:
    """Assert that a file exists and is not empty."""
    assert filepath.exists(), f"File does not exist: {filepath}"
    assert filepath.stat().st_size > 0, f"File is empty: {filepath}"
