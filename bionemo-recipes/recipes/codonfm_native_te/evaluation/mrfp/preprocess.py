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

"""Download and preprocess the mRFP Expression dataset.

Extracted verbatim from notebooks/5-EnCodon-Downstream-Task-mRFP-expression.ipynb (section 3),
with the column-normalization step (normally done by data_scripts/preprocess_validation.py)
inlined so this script is self-contained.
"""

import re
import urllib.request
from pathlib import Path

import polars as pl


SCRIPT_DIR = Path(__file__).parent

MRFP_DATASET_URL = (
    "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/"
    "benchmarks/CodonBERT/data/fine-tune/mRFP_Expression.csv"
)

RAW_REF_SEQ_COL = "Sequence"


def _camel_to_snake(name: str) -> str:
    """Convert a column name to snake_case, matching preprocess_validation.py."""
    name = name.replace(" ", "_")
    return re.sub(r"(?<!^)(?<![A-Z])(?=[A-Z])", "_", name).lower()


def main() -> None:
    """Download the mRFP Expression CSV, normalize columns, and write a parquet handoff."""
    raw_path = SCRIPT_DIR / "mRFP_Expression.csv"

    if not raw_path.exists():
        print(f"Downloading mRFP Expression dataset to {raw_path} ...")
        urllib.request.urlretrieve(MRFP_DATASET_URL, raw_path)
        print("Download complete.")
    else:
        print(f"Found existing dataset at {raw_path}.")

    data = pl.read_csv(raw_path)

    rename_map = {col: _camel_to_snake(col) for col in data.columns}
    rename_map[RAW_REF_SEQ_COL] = "ref_seq"
    data = data.rename(rename_map)

    data = data.with_row_index("id")
    data = data.with_columns(pl.col("id").cast(pl.Utf8))

    before = len(data)
    data = data.filter(pl.col("ref_seq").str.len_chars() % 3 == 0)
    dropped = before - len(data)
    if dropped:
        print(f"Dropped {dropped} rows whose ref_seq length is not divisible by 3.")

    if data.is_empty():
        raise ValueError("Output dataframe is empty after filtering.")
    if data["ref_seq"].null_count() > 0:
        raise ValueError("ref_seq column contains nulls.")

    print(f"Loaded {len(data)} sequences")
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns}")
    print(f"Split counts: {data['split'].value_counts().to_dict(as_series=False)}")

    value_stats = data.select(
        [
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().alias("std"),
            pl.col("value").min().alias("min"),
            pl.col("value").max().alias("max"),
        ]
    )
    print("\nmRFP expression stats:")
    print(f"  Mean: {value_stats['mean'][0]:.4f}")
    print(f"  Range: [{value_stats['min'][0]:.4f}, {value_stats['max'][0]:.4f}]")

    output_path = SCRIPT_DIR / "mrfp_expression.parquet"
    data.select(["id", "ref_seq", "value", "dataset", "split"]).write_parquet(output_path)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
