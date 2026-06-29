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


"""Process GTF annotation files to extract protein-coding transcripts.

This script processes GENCODE GTF files to extract CDS coordinates and
save them in a structured TSV format.
"""

import argparse
from pathlib import Path

import polars as pl


def process_gtf_files(gtf_files, output_dir=None):
    """Process GTF files to extract protein-coding transcripts.

    Args:
        gtf_files: List of GTF file paths
        output_dir: Directory for output files (defaults to same as input)
    """
    for gtf_file in gtf_files:
        gtf_path = Path(gtf_file)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {gtf_file}...")

        # Read GTF file
        gtf = pl.read_csv(gtf_file, comment_prefix="#", separator="\t", has_header=False)
        gtf.columns = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]

        # Extract metadata from attribute column
        gtf = gtf.with_columns(
            [
                pl.col("attribute").str.extract('gene_id "(.*?)"', 1).alias("gene_id"),
                pl.col("attribute").str.extract('transcript_id "(.*?)"', 1).alias("transcript_id"),
                pl.col("attribute").str.extract('gene_name "(.*?)"', 1).alias("gene_name"),
                pl.col("attribute").str.extract('gene_type "(.*?)"', 1).alias("gene_type"),
                pl.col("attribute").str.extract('transcript_type "(.*?)"', 1).alias("transcript_type"),
                pl.col("attribute").str.extract("exon_number (.*?);", 1).alias("exon_number"),
            ]
        )

        # Filter for protein coding genes
        protein_coding_gtf = gtf.filter((pl.col("gene_type") == "protein_coding") & (pl.col("feature") != "gene"))
        protein_coding_gtf = protein_coding_gtf.with_columns(pl.col("start") - 1)

        # Group by transcript_id and get exon coordinates
        exon_starts = (
            protein_coding_gtf.filter(pl.col("feature") == "CDS")
            .group_by("transcript_id")
            .agg(
                (
                    pl.when(pl.col("strand").first() == "-")
                    .then(
                        pl.when(pl.col("start").count() == 1)
                        .then((pl.col("start").first() - 3).cast(str))
                        .otherwise(
                            pl.concat_str(
                                [
                                    (pl.col("start").sort().first() - 3).cast(str),
                                    pl.col("start").sort().slice(1).cast(str).str.join(","),
                                ],
                                separator=",",
                            )
                        )
                    )
                    .otherwise(pl.col("start").sort().cast(str).str.join(","))
                    + ","
                ).alias("exon_starts"),
                (
                    pl.when(pl.col("strand").first() == "+")
                    .then(
                        pl.when(pl.col("end").count() == 1)
                        .then((pl.col("end").last() + 3).cast(str))
                        .otherwise(
                            pl.concat_str(
                                [
                                    pl.col("end").sort().head(pl.col("end").count() - 1).cast(str).str.join(","),
                                    (pl.col("end").sort().last() + 3).cast(str),
                                ],
                                separator=",",
                            )
                        )
                    )
                    .otherwise(pl.col("end").sort().cast(str).str.join(","))
                    + ","
                ).alias("exon_ends"),
                pl.col("exon_number").last().alias("exon_numbers"),
            )
        )

        # Get CDS boundaries
        cds_starts = (
            protein_coding_gtf.filter(pl.col("feature") == "CDS")
            .group_by("transcript_id")
            .agg(
                pl.when(pl.col("strand").first() == "-")
                .then(pl.col("start").last() - 3)
                .otherwise(pl.col("start").first())
                .alias("cds_starts"),
                pl.when(pl.col("strand").first() == "-")
                .then(pl.col("end").first())
                .otherwise(pl.col("end").last() + 3)
                .alias("cds_ends"),
            )
        )

        # Get transcript boundaries
        tx_starts = (
            protein_coding_gtf.filter(pl.col("feature") == "transcript")
            .group_by("transcript_id")
            .agg(
                pl.col("gene_id").first().alias("gene_id"),
                pl.col("gene_name").first().alias("gene_name"),
                pl.col("chrom").first().alias("chrom"),
                pl.col("strand").first().alias("strand"),
                pl.col("start").min().alias("tx_starts"),
                pl.col("end").max().alias("tx_ends"),
                pl.col("transcript_type").first().alias("transcript_type"),
            )
        )

        # Join all dataframes
        joined_df = tx_starts.join(cds_starts, on="transcript_id", how="inner").join(
            exon_starts, on="transcript_id", how="inner"
        )
        joined_df = joined_df.sort(["chrom", "tx_starts"])
        joined_df = joined_df.select(
            [
                "gene_id",
                "transcript_id",
                "chrom",
                "strand",
                "tx_starts",
                "tx_ends",
                "cds_starts",
                "cds_ends",
                "exon_numbers",
                "exon_starts",
                "exon_ends",
                "gene_name",
            ]
        ).rename(
            {
                "transcript_id": "name",
                "tx_starts": "txStart",
                "tx_ends": "txEnd",
                "cds_starts": "cdsStart",
                "cds_ends": "cdsEnd",
                "exon_starts": "exonStarts",
                "exon_ends": "exonEnds",
            }
        )

        # Save processed TSV
        if output_dir:
            output_file = output_dir / f"{gtf_path.stem}.processed.tsv"
        else:
            output_file = gtf_path.with_suffix("").with_suffix(".processed.tsv")

        joined_df.write_csv(str(output_file), separator="\t")
        print(f"Saved processed annotations to {output_file}")
        print(f"Total transcripts: {len(joined_df)}")

        # Save transcript BED file
        if output_dir:
            bed_file = output_dir / f"{gtf_path.stem}.transcripts.bed"
        else:
            bed_file = gtf_path.with_suffix("").with_suffix(".transcripts.bed")

        protein_coding_gtf.filter(pl.col("feature") == "transcript").select(
            ["chrom", "start", "end", "transcript_id"]
        ).write_csv(str(bed_file), separator="\t", include_header=False)
        print(f"Saved transcript BED to {bed_file}")

        # Save exon BED file
        if output_dir:
            exon_file = output_dir / f"{gtf_path.stem}.exons.bed"
        else:
            exon_file = gtf_path.with_suffix("").with_suffix(".exons.bed")

        protein_coding_gtf.filter(pl.col("feature") == "exon").with_columns(
            (pl.col("transcript_id") + "exon" + pl.col("exon_number")).alias("exon_name")
        ).select(["chrom", "start", "end", "exon_name"]).write_csv(
            str(exon_file), separator="\t", include_header=False
        )
        print(f"Saved exon BED to {exon_file}")
        print()


def main():  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Process GTF annotation files to extract protein-coding transcripts",
    )
    parser.add_argument(
        "--gtf-files", nargs="+", required=True, help="GTF files to process (can specify multiple files)"
    )
    parser.add_argument("--output-dir", help="Output directory (defaults to same directory as input files)")

    args = parser.parse_args()

    # Process GTF files
    print("Processing GTF files")
    process_gtf_files(args.gtf_files, args.output_dir)
    print("Processing complete!")


if __name__ == "__main__":
    main()
