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

"""Download TSV files from the UCSC Table Browser via POST requests.

Usage:
    python download_ucsc_tables.py                        # download all four tables
    python download_ucsc_tables.py --table ncbiRefSeq     # download a single table
    python download_ucsc_tables.py --output-dir /data/ncbi  # custom output directory
"""

import argparse
import os

import requests


UCSC_URL = "https://genome.ucsc.edu/cgi-bin/hgTables"

TABLES = {
    "wgEncodeGencodeCompV32": {
        "hgsid": "3727160771_KywqrMbVutzoVUyr47py53TcxZMg",  # pragma: allowlist secret
        "clade": "mammal",
        "org": "Human",
        "db": "hg38",
        "hgta_group": "allTables",
        "hgta_track": "hg38",
        "hgta_table": "wgEncodeGencodeCompV32",
        "hgta_regionType": "genome",
        "position": "chr7:155,799,529-155,812,871",
        "hgta_outSep": "tab",
        "hgta_doTopSubmit": "Get output",
        "filename": "ucsc_gencodev32_hg38.tsv",
    },
    "ncbiRefSeq": {
        "hgsid": "3727549177_A4TjXykIK1JRVnpjZ0HKtMVnKWw0",  # pragma: allowlist secret
        "clade": "mammal",
        "org": "Human",
        "db": "hg38",
        "hgta_group": "allTables",
        "hgta_track": "hg38",
        "hgta_table": "ncbiRefSeq",
        "hgta_regionType": "genome",
        "position": "chr7:155,799,529-155,812,871",
        "hgta_outSep": "tab",
        "hgta_doTopSubmit": "Get output",
        "subdir": "clinvar_syn",
        "filename": "ucsc_refseq_hg38.tsv",
    },
    "ncbiRefSeqHistorical": {
        "hgsid": "3727803393_8Oali1duOyVJT7DtAateRwtkg7Y0",  # pragma: allowlist secret
        "clade": "mammal",
        "org": "Human",
        "db": "hg38",
        "hgta_group": "allTables",
        "hgta_track": "hg38",
        "hgta_table": "ncbiRefSeqHistorical",
        "hgta_regionType": "genome",
        "position": "chr7:155,799,529-155,812,871",
        "hgta_outSep": "tab",
        "hgta_doTopSubmit": "Get output",
        "subdir": "clinvar_syn",
        "filename": "ucsc_refseq_hist_hg38.tsv",
    },
    "pliByGene": {
        "hgsid": "3727823409_x06fwXO5XFeWrbFjKlSQTfU3I6F3",  # pragma: allowlist secret
        "clade": "mammal",
        "org": "Human",
        "db": "hg38",
        "hgta_group": "varRep",
        "hgta_track": "gnomadPLI",
        "hgta_table": "pliByGene",
        "hgta_regionType": "genome",
        "position": "chr7:155,799,529-155,812,871",
        "hgta_outSep": "tab",
        "hgta_doTopSubmit": "Get output",
        "filename": "ucsc_pliByGene_hg38.tsv",
    },
}


def download_table(table_name: str, output_dir: str, api_key: str) -> str:
    """POST to the UCSC Table Browser and save the result as a TSV."""
    cfg = TABLES[table_name]
    cfg["apiKey"] = api_key
    dest_dir = os.path.join(output_dir, cfg.get("subdir", "")) if cfg.get("subdir") else output_dir
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, cfg["filename"])

    if os.path.exists(dest):
        print(f"  [skip] {dest}")
        return dest

    print(f"  Downloading {table_name} → {dest} ...")

    resp = requests.post(UCSC_URL, timeout=300, data=cfg)
    resp.raise_for_status()

    if "<!-- HGERROR-START -->" in resp.text:
        error_start = resp.text.index("<!-- HGERROR-START -->")
        error_end = (
            resp.text.index("<!-- HGERROR-END -->") if "<!-- HGERROR-END -->" in resp.text else error_start + 500
        )
        raise RuntimeError(f"UCSC returned an error:\n{resp.text[error_start:error_end]}")

    lines = resp.text.splitlines(keepends=True)
    while lines:
        tail = lines[-1].strip()
        if not tail or tail.startswith("---") or "cookie" in tail.lower():
            lines.pop()
        else:
            break

    with open(dest, "w") as f:
        f.writelines(lines)

    print(f"  [done] {dest}  ({len(lines):,} lines)")
    return dest


def main():
    """Download UCSC Table Browser tables as TSV."""
    parser = argparse.ArgumentParser(description="Download UCSC Table Browser tables as TSV")
    parser.add_argument("--table", choices=list(TABLES.keys()), help="Single table to download (default: all)")
    parser.add_argument("--output-dir", default=".", help="Base output directory (default: cwd)")
    parser.add_argument("--api-key", required=True, help="API key for UCSC Table Browser")
    args = parser.parse_args()

    tables = [args.table] if args.table else list(TABLES.keys())

    for t in tables:
        print(f"=== {t} ===")
        download_table(t, args.output_dir, args.api_key)


if __name__ == "__main__":
    main()
