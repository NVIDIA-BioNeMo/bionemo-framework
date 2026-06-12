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


import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
import requests


FILES = {
    "CoV_Vaccine_Degradation.csv": {
        "url": "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune/CoV_Vaccine_Degradation.csv",
        "columns": ["Sequence", "Value", "Dataset", "Split"],
    },
    "E.Coli_proteins.csv": {
        "url": "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune/E.Coli_proteins.csv",
        "columns": ["Sequence", "Value", "Dataset", "Split"],
    },
    "Fungal_expression.csv": {
        "url": "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune/Fungal_expression.csv",
        "columns": ["Sequence", "Value", "Dataset", "Split"],
    },
    "MLOS.csv": {
        "url": "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune/MLOS.csv",
        "columns": ["CDS", "Value"],
    },
    "Tc-Riboswitches.csv": {
        "url": "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune/Tc-Riboswitches.csv",
        "columns": ["Sequence", "Value", "Dataset", "Split"],
    },
    "mRFP_Expression.csv": {
        "url": "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune/mRFP_Expression.csv",
        "columns": ["Sequence", "Value", "Dataset", "Split"],
    },
    "mRNA_Stability.csv": {
        "url": "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune/mRNA_Stability.csv",
        "columns": ["Sequence", "Value", "Dataset", "Split"],
    },
}


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description="Download and preprocess CodonBERT benchmark datasets.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/validation/",
        help="Directory to save processed files (default: /data/validation/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(FILES.keys()),
        help="Download only a specific dataset (default: download all datasets)",
    )
    args = parser.parse_args()

    DOWNLOAD_DIR = Path(os.getcwd()) / "downloads"  # noqa: N806
    PROCESSED_DIR = Path(args.output_dir)  # noqa: N806

    # Ensure directories exist
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which files to process
    files_to_process = {args.dataset: FILES[args.dataset]} if args.dataset else FILES

    # - download and validate files
    for filename, file_info in files_to_process.items():
        file_path = DOWNLOAD_DIR / filename
        print(f"Downloading {filename}...")

        response = requests.get(file_info["url"])
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"Validating columns for {filename}...")
        df = pd.read_csv(file_path)
        missing_columns = [col for col in file_info["columns"] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"File {filename} is missing columns: {missing_columns}")

    # - preprocess files using preprocess_validation.py
    for filename in files_to_process.keys():
        input_path = DOWNLOAD_DIR / filename
        output_path = PROCESSED_DIR / filename
        print(f"Preprocessing {filename}...")
        columns = FILES[filename]["columns"]  # - first column is the reference sequence

        subprocess.run(
            [
                "python",
                Path(__file__).parent / "preprocess_validation.py",
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "--ref-seq-col",
                columns[0],
            ],
            check=True,
        )

    if args.dataset:
        print(f"Dataset {args.dataset} downloaded and preprocessed successfully.")
    else:
        print("All files downloaded and preprocessed successfully.")


if __name__ == "__main__":
    main()
