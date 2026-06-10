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

"""Download PDB structures and prepare parquet dataset for ESM2-MiniFold TE training.

Downloads mmCIF files from RCSB PDB, extracts Ca coordinates and sequences,
validates structures, and exports to parquet format.

Usage:
    python data/prepare_pdb_dataset.py
    python data/prepare_pdb_dataset.py --max-structures 50  # limit for testing
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
SCRIPT_DIR = Path(__file__).resolve().parent
PDB_IDS_FILE = SCRIPT_DIR / "pdb_ids.txt"
OUTPUT_PARQUET = SCRIPT_DIR / "pdb_structures.parquet"
OUTPUT_CIF_DIR = SCRIPT_DIR / "cif_files"
OUTPUT_MANIFEST = SCRIPT_DIR / "pdb_manifest.json"

# 3-letter to 1-letter amino acid mapping
AA_3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "MSE": "M",  # Selenomethionine -> Methionine
}

MIN_RESIDUES = 50
MAX_RESIDUES = 300
MIN_CA_COMPLETENESS = 0.9
MAX_RETRIES = 3
DOWNLOAD_DELAY = 0.5  # seconds between RCSB requests


def compute_sha256(file_path):
    """Compute SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pdb_ids(path):
    """Read PDB IDs from a text file (one per line, # comments allowed)."""
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line.upper())
    return ids


def download_cif(pdb_id, output_dir, timeout=30):
    """Download a mmCIF file from RCSB PDB with retry logic.

    Returns path to downloaded file, or None on failure.
    """
    url = RCSB_URL.format(pdb_id=pdb_id)
    output_path = output_dir / f"{pdb_id}.cif"

    if output_path.exists():
        return output_path

    for attempt in range(MAX_RETRIES):
        try:
            urlretrieve(url, output_path)
            return output_path
        except (HTTPError, URLError, TimeoutError) as e:
            wait = 2**attempt
            logger.warning("Download %s attempt %d failed: %s (retry in %ds)", pdb_id, attempt + 1, e, wait)
            time.sleep(wait)

    logger.error("Failed to download %s after %d attempts", pdb_id, MAX_RETRIES)
    return None


def parse_mmcif(cif_path, pdb_id):
    """Parse a mmCIF file and extract sequence + Ca coordinates.

    Returns dict with keys: pdb_id, chain_id, sequence, coords, ca_mask, num_residues.
    Returns None if parsing fails or structure is invalid.
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(cif_path))
    except Exception as e:
        logger.warning("Failed to parse %s: %s", pdb_id, e)
        return None

    model = structure[0]

    # Find first protein chain with enough residues
    for chain in model:
        residues = []
        for res in chain.get_residues():
            # Skip heteroatoms and water
            if res.id[0] != " ":
                continue
            resname = res.get_resname().strip()
            if resname not in AA_3TO1:
                continue
            residues.append(res)

        if len(residues) < MIN_RESIDUES:
            continue
        if len(residues) > MAX_RESIDUES:
            residues = residues[:MAX_RESIDUES]

        # Extract sequence and Ca coordinates
        sequence = []
        coords = []
        ca_mask = []

        for res in residues:
            resname = res.get_resname().strip()
            sequence.append(AA_3TO1[resname])

            if "CA" in res:
                ca = res["CA"].get_vector()
                coords.append([float(ca[0]), float(ca[1]), float(ca[2])])
                ca_mask.append(1)
            else:
                coords.append([0.0, 0.0, 0.0])
                ca_mask.append(0)

        # Validate Ca completeness
        completeness = sum(ca_mask) / len(ca_mask)
        if completeness < MIN_CA_COMPLETENESS:
            logger.warning(
                "Skipping %s chain %s: Ca completeness %.1f%% < %.0f%%",
                pdb_id,
                chain.id,
                completeness * 100,
                MIN_CA_COMPLETENESS * 100,
            )
            continue

        # Validate coordinates are finite
        coords_arr = np.array(coords)
        if not np.all(np.isfinite(coords_arr)):
            logger.warning("Skipping %s chain %s: non-finite coordinates", pdb_id, chain.id)
            continue

        return {
            "pdb_id": pdb_id,
            "chain_id": chain.id,
            "sequence": "".join(sequence),
            "coords": coords,
            "ca_mask": ca_mask,
            "num_residues": len(residues),
        }

    logger.warning("No valid chain found in %s", pdb_id)
    return None


def main():
    """Download PDB structures and create parquet dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare PDB dataset for ESM2-MiniFold TE")
    parser.add_argument("--max-structures", type=int, default=None, help="Limit number of structures (for testing)")
    args = parser.parse_args()

    pdb_ids = read_pdb_ids(PDB_IDS_FILE)
    if args.max_structures:
        pdb_ids = pdb_ids[: args.max_structures]

    logger.info("Processing %d PDB IDs", len(pdb_ids))

    OUTPUT_CIF_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    manifest = {}
    failed = []

    for i, pdb_id in enumerate(pdb_ids):
        if i > 0:
            time.sleep(DOWNLOAD_DELAY)

        # Download
        cif_path = download_cif(pdb_id, OUTPUT_CIF_DIR)
        if cif_path is None:
            failed.append(pdb_id)
            continue

        sha256 = compute_sha256(cif_path)

        # Parse
        record = parse_mmcif(cif_path, pdb_id)
        if record is None:
            failed.append(pdb_id)
            continue

        records.append(record)
        manifest[pdb_id] = {
            "sha256": sha256,
            "chain_id": record["chain_id"],
            "num_residues": record["num_residues"],
            "sequence_length": len(record["sequence"]),
        }

        if (i + 1) % 20 == 0:
            logger.info(
                "Progress: %d/%d downloaded, %d valid, %d failed", i + 1, len(pdb_ids), len(records), len(failed)
            )

    # Write parquet
    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Wrote %d structures to %s", len(df), OUTPUT_PARQUET)

    # Write manifest
    with open(OUTPUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", OUTPUT_MANIFEST)

    # Summary
    if records:
        lengths = [r["num_residues"] for r in records]
        logger.info("Summary: %d valid structures, %d failed", len(records), len(failed))
        logger.info("Residue lengths: min=%d, max=%d, mean=%.0f", min(lengths), max(lengths), np.mean(lengths))

    if failed:
        logger.warning("Failed PDB IDs: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
