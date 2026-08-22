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

"""Download a large PDB dataset via RCSB Search API for ESM2-MiniFold TE training.

Queries RCSB PDB for high-quality X-ray structures, downloads mmCIF files in
parallel, parses Ca coordinates, and exports to parquet. Designed for cluster use.

Usage:
    # Default: ~10k structures, 8 download workers
    python data/prepare_pdb_dataset_large.py

    # Smaller test run
    python data/prepare_pdb_dataset_large.py --max-structures 500

    # Custom output directory (e.g., on a fast scratch disk)
    python data/prepare_pdb_dataset_large.py --output-dir /scratch/$USER/pdb_data

    # Custom resolution and length filters
    python data/prepare_pdb_dataset_large.py --max-resolution 2.0 --min-length 80 --max-length 250
"""

import argparse
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen, urlretrieve

import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

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
    "MSE": "M",
}

MAX_RETRIES = 3
MIN_CA_COMPLETENESS = 0.9


def query_rcsb(max_resolution=2.5, min_length=50, max_length=300, max_results=15000):
    """Query RCSB Search API for high-quality protein structures.

    Filters:
    - X-ray diffraction only
    - Resolution better than max_resolution
    - Polymer entity length between min_length and max_length
    - Protein entity type

    Paginates automatically (RCSB caps at 10,000 rows per request).

    Returns list of PDB IDs (4-letter codes, uppercase).
    """
    PAGE_SIZE = 10000  # RCSB maximum rows per request

    base_query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "range",
                        "value": {"from": min_length, "to": max_length},
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }

    logger.info(
        "Querying RCSB: resolution<=%.1fA, length %d-%d, max %d results",
        max_resolution,
        min_length,
        max_length,
        max_results,
    )

    pdb_ids = []
    start = 0
    total_count = None

    while len(pdb_ids) < max_results:
        rows = min(PAGE_SIZE, max_results - len(pdb_ids))
        query = {
            **base_query,
            "request_options": {
                **base_query["request_options"],
                "paginate": {"start": start, "rows": rows},
            },
        }

        req = Request(
            RCSB_SEARCH_URL,
            data=json.dumps(query).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if total_count is None:
            total_count = data.get("total_count", 0)

        page_ids = [r["identifier"] for r in data.get("result_set", [])]
        if not page_ids:
            break

        pdb_ids.extend(page_ids)
        start += len(page_ids)
        logger.info(
            "RCSB page: fetched %d (total so far: %d, available: %d)", len(page_ids), len(pdb_ids), total_count
        )

        if start >= total_count:
            break

    logger.info("RCSB query complete: %d results (total available: %d)", len(pdb_ids), total_count)
    return pdb_ids


def download_cif(pdb_id, output_dir, timeout=30):
    """Download a single mmCIF file with retry. Returns (pdb_id, path, success)."""
    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    output_path = output_dir / f"{pdb_id}.cif"

    if output_path.exists() and output_path.stat().st_size > 0:
        return pdb_id, output_path, True

    for attempt in range(MAX_RETRIES):
        try:
            urlretrieve(url, output_path)
            return pdb_id, output_path, True
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            wait = 2**attempt
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    return pdb_id, None, False


def compute_sha256(file_path):
    """Compute SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mmcif(cif_path, pdb_id, min_residues=50, max_residues=300):
    """Parse a mmCIF file and extract sequence + Ca coordinates.

    Returns dict or None if parsing fails or structure is invalid.
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(cif_path))
    except Exception as e:
        logger.debug("Failed to parse %s: %s", pdb_id, e)
        return None

    model = structure[0]

    for chain in model:
        residues = []
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            resname = res.get_resname().strip()
            if resname not in AA_3TO1:
                continue
            residues.append(res)

        if len(residues) < min_residues:
            continue
        if len(residues) > max_residues:
            residues = residues[:max_residues]

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

        completeness = sum(ca_mask) / len(ca_mask)
        if completeness < MIN_CA_COMPLETENESS:
            continue

        coords_arr = np.array(coords)
        if not np.all(np.isfinite(coords_arr)):
            continue

        return {
            "pdb_id": pdb_id,
            "chain_id": chain.id,
            "sequence": "".join(sequence),
            "coords": coords,
            "ca_mask": ca_mask,
            "num_residues": len(residues),
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare large PDB dataset for ESM2-MiniFold TE")
    parser.add_argument("--max-structures", type=int, default=10000, help="Maximum structures to download")
    parser.add_argument("--max-resolution", type=float, default=2.5, help="Max X-ray resolution in Angstroms")
    parser.add_argument("--min-length", type=int, default=50, help="Min polymer entity length")
    parser.add_argument("--max-length", type=int, default=300, help="Max polymer entity length")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: data/)")
    parser.add_argument("--download-workers", type=int, default=8, help="Parallel download threads")
    parser.add_argument("--parse-workers", type=int, default=4, help="Parallel parse threads")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir
    cif_dir = output_dir / "cif_files"
    cif_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Query RCSB for PDB IDs
    pdb_ids = query_rcsb(
        max_resolution=args.max_resolution,
        min_length=args.min_length,
        max_length=args.max_length,
        max_results=args.max_structures + 2000,  # Query extra to account for failures
    )
    pdb_ids = pdb_ids[: args.max_structures]
    logger.info("Will process %d PDB IDs", len(pdb_ids))

    # Step 2: Download in parallel
    logger.info("Downloading CIF files with %d workers...", args.download_workers)
    downloaded = {}
    failed_download = []

    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures = {pool.submit(download_cif, pid, cif_dir): pid for pid in pdb_ids}
        done_count = 0
        for future in as_completed(futures):
            pdb_id, path, success = future.result()
            done_count += 1
            if success:
                downloaded[pdb_id] = path
            else:
                failed_download.append(pdb_id)
            if done_count % 500 == 0:
                logger.info(
                    "Download progress: %d/%d done, %d succeeded, %d failed",
                    done_count,
                    len(pdb_ids),
                    len(downloaded),
                    len(failed_download),
                )

    logger.info("Download complete: %d succeeded, %d failed", len(downloaded), len(failed_download))

    # Step 3: Parse structures
    logger.info("Parsing %d CIF files...", len(downloaded))
    records = []
    manifest = {}
    failed_parse = []

    # Parse sequentially (BioPython's C extensions may not be thread-safe)
    for i, (pdb_id, cif_path) in enumerate(downloaded.items()):
        record = parse_mmcif(cif_path, pdb_id, min_residues=args.min_length, max_residues=args.max_length)
        if record is None:
            failed_parse.append(pdb_id)
        else:
            records.append(record)
            manifest[pdb_id] = {
                "sha256": compute_sha256(cif_path),
                "chain_id": record["chain_id"],
                "num_residues": record["num_residues"],
                "sequence_length": len(record["sequence"]),
            }

        if (i + 1) % 500 == 0:
            logger.info(
                "Parse progress: %d/%d processed, %d valid, %d failed",
                i + 1,
                len(downloaded),
                len(records),
                len(failed_parse),
            )

    # Step 4: Write outputs
    output_parquet = output_dir / "pdb_structures.parquet"
    output_manifest = output_dir / "pdb_manifest.json"
    output_ids = output_dir / "pdb_ids.txt"

    df = pd.DataFrame(records)
    df.to_parquet(str(output_parquet), index=False)
    logger.info("Wrote %d structures to %s", len(df), output_parquet)

    with open(output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", output_manifest)

    # Write PDB IDs file for reproducibility
    with open(output_ids, "w") as f:
        f.write("# PDB IDs for ESM2-MiniFold TE training\n")
        f.write(f"# Generated: resolution<={args.max_resolution}A, length {args.min_length}-{args.max_length}\n")
        f.write(f"# Total: {len(records)} valid structures\n")
        for r in records:
            f.write(r["pdb_id"] + "\n")
    logger.info("Wrote PDB IDs to %s", output_ids)

    # Summary
    if records:
        lengths = [r["num_residues"] for r in records]
        logger.info("=== Summary ===")
        logger.info("Valid structures: %d", len(records))
        logger.info("Failed download: %d", len(failed_download))
        logger.info("Failed parse: %d", len(failed_parse))
        logger.info(
            "Residue lengths: min=%d, max=%d, mean=%.0f, median=%.0f",
            min(lengths),
            max(lengths),
            np.mean(lengths),
            np.median(lengths),
        )
        logger.info("Output: %s (%.1f MB)", output_parquet, output_parquet.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
