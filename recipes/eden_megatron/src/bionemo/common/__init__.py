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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: recipes/evo2_megatron/src/bionemo/common/__init__.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Shared recipe utilities used by BioNeMo recipes."""

import os
from pathlib import Path
from typing import Sequence

import platformdirs


__all__: Sequence[str] = ("BIONEMO_CACHE_DIR",)


def _get_cache_dir() -> Path:
    if cache_dir := os.getenv("BIONEMO_CACHE_DIR"):
        return Path(cache_dir)

    cache_dir = Path(platformdirs.user_cache_dir(appname="bionemo", appauthor="nvidia"))
    try:
        cache_dir.mkdir(exist_ok=True, parents=True)
    except PermissionError as ex:
        raise PermissionError(
            f"Permission denied creating a cache directory at {cache_dir}. Please set BIONEMO_CACHE_DIR to a directory "
            "you have write access to."
        ) from ex
    return cache_dir


BIONEMO_CACHE_DIR = _get_cache_dir()
