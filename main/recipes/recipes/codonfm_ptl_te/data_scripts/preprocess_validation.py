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
import re
from pathlib import Path

import numpy as np
import pandas as pd


REV_COMP = str.maketrans("ACGT", "TGCA")


class DNASeqValidationProcessor:  # noqa: D101
    def __init__(self, args):  # noqa: D107
        self.args = args
        self.standard_columns = {
            "variant_id": args.variant_id_col,
            "ref_seq": args.ref_seq_col,
            "alt_seq": args.alt_seq_col,
            "codon_position": args.codon_pos_col,  # - codon position 0-based
            "ref_codon": args.ref_codon_col,
            "alt_codon": args.alt_codon_col,
            "strand": args.strand_col,
            "position": args.position_col,  # - nucleotide position 0-based
            "ref_allele": args.ref_allele_col,
            "alt_allele": args.alt_allele_col,
        }

    def process_dataframe(self, df):  # noqa: D102
        df = self._standardize_columns(df)
        df = self._handle_input_cases(df)
        df = self._add_unique_ids(df)
        return self._validate_output(df)

    def _standardize_columns(self, df):
        """Normalize column names and apply user mappings only for provided columns"""  # noqa: D415
        column_mappings = self._get_column_mappings()
        valid_mappings = {old: new for old, new in column_mappings.items() if old in df.columns and new}
        df = df.rename(columns=valid_mappings)

        def convert_camel_case(col):
            col = col.replace(" ", "_")
            return re.sub(r"(?<!^)(?<![A-Z])(?=[A-Z])", "_", col).lower()

        df.columns = [convert_camel_case(col) for col in df.columns]
        return df

    def _handle_input_cases(self, df):
        """Process different input combinations"""  # noqa: D415
        assert "ref_seq" in df.columns, "Reference sequence column is required."
        if all((col in df.columns) for col in ["ref_allele", "alt_allele", "position"]) and "alt_seq" in df.columns:
            df.drop(columns=["alt_seq"], inplace=True)  # - this will be recalculated
        df = df.apply(self._process_row, axis=1)
        return df

    def _process_row(self, row):
        """Process individual rows based on available data"""  # noqa: D415
        # - extract commonly used values
        ref_seq = row.get("ref_seq", "")
        position = row.get("position", np.nan)

        # - calculate codon position if missing
        if pd.notna(position) and "codon_position" not in row:
            row["codon_position"] = position // 3
            row["codon_position"] = int(row["codon_position"])
            row["position"] = int(row["position"])
        elif "codon_position" in row and pd.notna(position):
            assert position // 3 == row["codon_position"], (
                f"Codon position mismatch: {position} // 3 != {row['codon_position']}"
            )
            row["codon_position"] = int(row["codon_position"])

        if all(pd.notna(row.get(col)) for col in ["ref_seq", "codon_position"]) and "ref_codon" not in row:
            row = self._calculate_codons(row, ref_seq)

        # - handle alt allele/alt seq/strand information
        if "alt_codon" not in row:
            row = self._handle_alt_token(row)

        return row

    def _calculate_codons(self, row, ref_seq):
        """Calculate codon-related fields"""  # noqa: D415
        codon_pos = row["codon_position"]
        start = codon_pos * 3
        end = start + 3

        # - extract reference codon
        row["ref_codon"] = ref_seq[start:end]
        return row

    def _handle_alt_token(self, row):
        """Handle reverse strand sequences"""  # noqa: D415
        if all(col in row.index for col in ["ref_allele", "alt_allele", "position"]):
            pos = row["position"]
            codon_pos = pos // 3
            j = pos % 3

            # - validate codon position
            ref_seq = row["ref_seq"]
            ref_codon = ref_seq[codon_pos * 3 : (codon_pos + 1) * 3]
            expected_ref_allele = ref_seq[pos]
            strand = row.get("strand", "+")
            if strand == "+":
                assert expected_ref_allele == row["ref_allele"], (
                    f"Reference allele mismatch at position {pos}: {expected_ref_allele} != {row['ref_allele']}"
                )
                alt_codon = ref_codon[:j] + row["alt_allele"] + ref_codon[j + 1 :]
            elif strand == "-":
                assert ref_seq[pos] == row["ref_allele"].translate(REV_COMP), (
                    f"Reference allele mismatch at position {pos}: {ref_seq[pos]} != {row['ref_allele'].translate(REV_COMP)}"
                )
                alt_codon = ref_codon[:j] + row["alt_allele"].translate(REV_COMP) + ref_codon[j + 1 :]

            row["ref_codon"] = ref_codon
            row["alt_codon"] = alt_codon

            row["alt_seq"] = ref_seq[: codon_pos * 3] + alt_codon + ref_seq[(codon_pos + 1) * 3 :]
        elif "alt_seq" in row.index and "codon_position" in row.index:
            codon_pos = row["codon_position"]
            alt_seq = row["alt_seq"]
            row["alt_codon"] = alt_seq[codon_pos * 3 : (codon_pos + 1) * 3]
        return row

    def _add_unique_ids(self, df):
        """Add unique identifier column using dataframe index"""  # noqa: D415
        df.insert(0, "id", df.index.astype(str))
        return df

    def _validate_output(self, df):
        """Validate final dataframe structure"""  # noqa: D415
        required_cols = ["id", "ref_seq"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        # - drop any rows with seq not divisible by 3
        df = df[df["ref_seq"].apply(lambda x: len(x) % 3 == 0)]
        assert len(df) > 0, "Output dataframe is empty."
        assert df["ref_seq"].notna().all(), "Reference sequence column contains NaN values."
        if "alt_seq" in df.columns:
            assert df["alt_seq"].notna().all(), "Alternate sequence column contains NaN values."
        if "ref_codon" in df.columns:
            assert np.all(df["ref_codon"].apply(lambda x: (len(x) % 3)) == 0), (
                "Not all ref_codon lengths are divisible by 3."
            )
        if "alt_codon" in df.columns:
            assert np.all(df["alt_codon"].apply(lambda x: (len(x) % 3)) == 0), (
                "Not all alt_codon lengths are divisible by 3."
            )
        if "codon_position" in df.columns:
            assert np.all(df.apply(lambda x: len(x["ref_seq"]) > x["codon_position"] * 3, axis=1)), (
                "Not all codon positions are less than sequence length."
            )
            df["codon_position"] = df["codon_position"].astype(int)
        return df

    def _get_column_mappings(self):
        """Generate column mappings from arguments"""  # noqa: D415
        return {v: k for k, v in self.standard_columns.items() if v != k}


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description="DNA Sequence Processor")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input CSV file")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output CSV file")

    parser.add_argument("--ref-seq-col", required=True, help="Reference sequence column name")
    parser.add_argument("--alt-seq-col", default="alt_seq", help="Alternate sequence column name")
    parser.add_argument("--ref-codon-col", default="ref_codon", help="Reference codon column name")
    parser.add_argument("--alt-codon-col", default="alt_codon", help="Alternate codon column name")
    parser.add_argument("--position-col", default="position", help="Nucleotide position column name (0-based)")
    parser.add_argument("--codon-pos-col", default="codon_position", help="Codon position column name (0-based)")
    parser.add_argument("--ref-allele-col", default="ref_allele", help="Reference allele column name")
    parser.add_argument("--alt-allele-col", default="alt_allele", help="Alternate allele column name")
    parser.add_argument("--strand-col", default="strand", help="Strand information column name")
    parser.add_argument("--variant-id-col", default="variant_id", help="Variant ID column name")

    args = parser.parse_args()

    # - process data
    df = pd.read_csv(args.input)
    processor = DNASeqValidationProcessor(args)
    processed_df = processor.process_dataframe(df)

    # - save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
