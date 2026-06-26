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

"""Utilities for reproducing the Microviridae SFT stage."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ZENODO_DIR = RECIPE_ROOT / "data" / "external" / "zenodo"
DEFAULT_SFT_PROCESSED = DEFAULT_ZENODO_DIR / "microviridae_sft_training_data_processed.fna"
DEFAULT_SFT_RAW = DEFAULT_ZENODO_DIR / "microviridae_sft_training_data_raw.fna"
DEFAULT_SFT_PROCESSED_URL = (
    "https://zenodo.org/records/17101843/files/microviridae_sft_training_data_processed.fna?download=1"
)
DEFAULT_SFT_RAW_URL = "https://zenodo.org/records/17101843/files/microviridae_sft_training_data_raw.fna?download=1"


def _recipe_relative(path: Path) -> str:
    """Return a path relative to the recipe root when possible."""
    path = Path(path)
    try:
        return path.resolve().relative_to(RECIPE_ROOT).as_posix()
    except ValueError:
        return str(path)


def _download(url: str, output_path: Path, *, overwrite: bool = False) -> Path:
    """Download a file unless it already exists."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(output_path)
    return output_path


def prepare_sft_data(*, include_raw: bool = False, overwrite: bool = False) -> list[Path]:
    """Download Microviridae SFT FASTA files from the Zenodo paper record."""
    paths = [_download(DEFAULT_SFT_PROCESSED_URL, DEFAULT_SFT_PROCESSED, overwrite=overwrite)]
    if include_raw:
        paths.append(_download(DEFAULT_SFT_RAW_URL, DEFAULT_SFT_RAW, overwrite=overwrite))
    return paths


def main() -> None:
    """Download Microviridae SFT FASTA files from Zenodo."""
    parser = argparse.ArgumentParser(description="Download Zenodo Microviridae SFT FASTA files")
    parser.add_argument("--include-raw", action="store_true", help="Also download the raw SFT FASTA")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in prepare_sft_data(include_raw=args.include_raw, overwrite=args.overwrite):
        print(_recipe_relative(path))


if __name__ == "__main__":
    main()
