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

"""Download and preprocess the RiboNN translation efficiency dataset.

Extracted verbatim from notebooks/4-EnCodon-Downstream-Task-riboNN.ipynb (section 3).
"""

import os
import urllib.request
from pathlib import Path

import polars as pl


# Configurable dataset path
data_path = "/data/validation/processed/data_with_human_TE_cellline_all_NA_plain.csv"

# Source URL for the TE dataset
te_dataset_url = "https://raw.githubusercontent.com/CenikLab/TE_classic_ML/refs/heads/main/data/data_with_human_TE_cellline_all_NA_plain.csv"

# Ensure parent directory exists
Path(os.path.dirname(data_path)).mkdir(parents=True, exist_ok=True)

# Download if missing
if not os.path.exists(data_path):
    print(f"Downloading TE dataset to {data_path} ...")
    urllib.request.urlretrieve(te_dataset_url, data_path)
    print("Download complete.")
else:
    print(f"Found existing dataset at {data_path}.")


# Slice the transcript sequence into CDS / 5'UTR / 3'UTR using utr5_size and cds_size,
# and add a row index column 'id'.
data = pl.read_csv(data_path, separator="\t")
data = data.with_columns(
    [
        pl.struct(["utr5_size", "cds_size", "tx_sequence"])
        .map_elements(
            lambda row: row["tx_sequence"][row["utr5_size"] : row["utr5_size"] + row["cds_size"]], return_dtype=pl.Utf8
        )
        .alias("cds_sequence"),
        pl.struct(["utr5_size", "tx_sequence"])
        .map_elements(lambda row: row["tx_sequence"][: row["utr5_size"]], return_dtype=pl.Utf8)
        .alias("utr5_sequence"),
        pl.struct(["utr5_size", "cds_size", "tx_sequence"])
        .map_elements(lambda row: row["tx_sequence"][row["utr5_size"] + row["cds_size"] :], return_dtype=pl.Utf8)
        .alias("utr3_sequence"),
    ]
).with_row_index("id")
output_path = data_path[:-4] + ".processed.csv"
data.write_csv(output_path)


# Load processed RiboNN dataset and report basic statistics on the mean_te target.
data_loaded = False
if os.path.exists(output_path):
    try:
        data = pl.read_csv(output_path)
        print(f"✅ Loaded {len(data)} sequences from: {output_path}")
        print(f"Shape: {data.shape}")
        print(f"Key columns: {[col for col in ['id', 'cds_sequence', 'mean_te', 'fold'] if col in data.columns]}")

        data_loaded = True
    except Exception as e:
        print(f"Failed to load {output_path}: {e}")

    # Show basic statistics
    te_stats = data.select(
        [
            pl.col("mean_te").mean().alias("mean"),
            pl.col("mean_te").std().alias("std"),
            pl.col("mean_te").min().alias("min"),
            pl.col("mean_te").max().alias("max"),
        ]
    )
    print("\nTranslation Efficiency stats:")
    print(f"  Mean: {te_stats['mean'][0]:.4f}")
    print(f"  Range: [{te_stats['min'][0]:.4f}, {te_stats['max'][0]:.4f}]")
    data_loaded = True
