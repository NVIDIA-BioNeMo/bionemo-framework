#!/usr/bin/env python3

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

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


PORTER6_ZIP_URL = "https://github.com/WafaAlanazi/Porter6/raw/main/SS%20datasets.zip"
SCRIPT_DATA_DIR = Path(__file__).resolve().parent

DATASET_FILES = {
    "dataset_train55k_80%.txt": {
        "output": "porter6_train_dataset_55k.parquet",
        "sha256": "4b1c011d8cea0b892743053eb4234db80344b8d9c90243f19b4637781ce8922b",
    },
    "2024Testset_692.adataset": {
        "output": "porter6_val_dataset_2024_692.parquet",
        "sha256": "b4a1b69f2003a66a62eb106aded784f9938fc734e876458223459fd9a10f1ca2",
    },
}


def parse_input_file(path):
    """Parse a Porter6-formatted secondary-structure file into row dictionaries."""
    records = []

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        pdb_id = lines[i]
        _ = lines[i + 1]  # length line, not strictly needed
        seq_line = lines[i + 2]
        ss_line = lines[i + 3]

        sequence = seq_line.replace(" ", "")
        secondary_structure = ss_line.replace(" ", "")
        secondary_structure = secondary_structure.replace(".", "~")

        records.append(
            {
                "PDB_ID": pdb_id,
                "Sequence": sequence,
                "Secondary_structure": secondary_structure,
            }
        )

        i += 4

    return records


def compute_sha256(file_path):
    """Compute SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Download Porter6 datasets and write train/validation parquet files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_path = tmp_path / "SS_datasets.zip"

        print(f"Downloading Porter6 datasets from: {PORTER6_ZIP_URL}")
        urlretrieve(PORTER6_ZIP_URL, zip_path)

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Flatten extracted files so we can match expected names regardless of zip structure.
        extracted_by_name = {}
        for extracted_file in extract_dir.rglob("*"):
            if extracted_file.is_file():
                extracted_by_name[extracted_file.name] = extracted_file

        for input_name, file_config in DATASET_FILES.items():
            if input_name not in extracted_by_name:
                available = ", ".join(sorted(extracted_by_name))
                raise FileNotFoundError(
                    f"Expected file '{input_name}' was not found in downloaded zip. Available files: {available}"
                )

            source_file = extracted_by_name[input_name]
            working_input_path = tmp_path / input_name
            shutil.copy2(source_file, working_input_path)

            actual_sha256 = compute_sha256(working_input_path)
            expected_sha256 = file_config["sha256"]
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"SHA256 mismatch for '{input_name}': expected {expected_sha256}, got {actual_sha256}"
                )

            records = parse_input_file(working_input_path)
            df = pd.DataFrame(records)

            output_path = SCRIPT_DATA_DIR / file_config["output"]
            df.to_parquet(output_path, index=False)

            print(f"Converted {input_name}: {len(df)} records")
            print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
