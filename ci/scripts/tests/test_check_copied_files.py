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

import pytest

from ci.scripts.check_copied_files import _sync_copied_tree, _validate_copied_tree


def test_sync_copied_tree_removes_files_deleted_or_moved_in_source(tmp_path):
    """A copied directory must remain an exact recursive mirror after refactors."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "old").mkdir(parents=True)
    destination.mkdir()
    (source / "old" / "module.py").write_text("VALUE = 1\n")
    (destination / "stale.py").write_text("VALUE = 0\n")

    _sync_copied_tree(source, destination)

    assert not (destination / "stale.py").exists()
    assert (destination / "old" / "module.py").exists()

    (source / "new").mkdir()
    (source / "old" / "module.py").rename(source / "new" / "module.py")
    (source / "old").rmdir()

    _sync_copied_tree(source, destination)

    assert not (destination / "old").exists()
    assert (destination / "new" / "module.py").exists()


def test_validate_copied_tree_reports_stale_files(tmp_path):
    """Exact-mirror validation reports destination files absent from the source."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "current.txt").write_text("current\n")
    (destination / "current.txt").write_text("current\n")
    (destination / "stale.txt").write_text("stale\n")

    with pytest.raises(ValueError, match=r"unexpected=\['stale.txt'\]"):
        _validate_copied_tree(source, destination, "source")
