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

"""Download a high-quality evaluation dataset for ESM2-MiniFold TE.

Queries RCSB PDB for very high resolution structures (≤1.5Å), excludes any
PDB IDs in the training set, downloads mmCIF files, and exports to parquet.

Usage:
    python data/prepare_eval_dataset.py
    python data/prepare_eval_dataset.py --max-structures 50  # quick test
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

SCRIPT_DIR = Path(__file__).resolve().parent

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
MIN_CA_COMPLETENESS = 0.95  # Stricter than training (0.9) for eval quality


def read_training_pdb_ids(path=None):
    """Read training PDB IDs to exclude from eval set."""
    if path is None:
        path = SCRIPT_DIR / "pdb_ids.txt"
    if not path.exists():
        logger.warning("Training PDB IDs file not found: %s", path)
        return set()

    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line.upper())
    return ids


def query_rcsb(max_resolution=1.5, min_length=50, max_length=300, max_results=500):
    """Query RCSB Search API for high-resolution protein structures."""
    query = {
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
            "paginate": {"start": 0, "rows": max_results},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }

    logger.info(
        "Querying RCSB for eval set: resolution<=%.1fA, length %d-%d, max %d",
        max_resolution,
        min_length,
        max_length,
        max_results,
    )

    req = Request(
        RCSB_SEARCH_URL,
        data=json.dumps(query).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    total_count = data.get("total_count", 0)
    pdb_ids = [r["identifier"] for r in data.get("result_set", [])]
    logger.info("RCSB returned %d results (total available: %d)", len(pdb_ids), total_count)
    return pdb_ids


def download_cif(pdb_id, output_dir):
    """Download a single mmCIF file with retry. Returns (pdb_id, path, success)."""
    url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    output_path = output_dir / f"{pdb_id}.cif"

    if output_path.exists() and output_path.stat().st_size > 0:
        return pdb_id, output_path, True

    for attempt in range(MAX_RETRIES):
        try:
            urlretrieve(url, output_path)
            return pdb_id, output_path, True
        except (HTTPError, URLError, TimeoutError, OSError):
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
    """Parse a mmCIF file and extract sequence + Ca coordinates."""
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
    ap = argparse.ArgumentParser(description="Prepare eval dataset for ESM2-MiniFold TE")
    ap.add_argument("--max-structures", type=int, default=150, help="Target number of eval structures")
    ap.add_argument("--max-resolution", type=float, default=1.5, help="Max X-ray resolution (Angstroms)")
    ap.add_argument("--min-length", type=int, default=50, help="Min polymer entity length")
    ap.add_argument("--max-length", type=int, default=300, help="Max polymer entity length")
    ap.add_argument("--output-dir", type=str, default=None, help="Output directory (default: data/)")
    ap.add_argument("--download-workers", type=int, default=8, help="Parallel download threads")
    ap.add_argument("--training-ids-file", type=str, default=None, help="Training PDB IDs to exclude")
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR
    cif_dir = output_dir / "eval_cif_files"
    cif_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load training PDB IDs to exclude
    train_ids_path = Path(args.training_ids_file) if args.training_ids_file else None
    train_ids = read_training_pdb_ids(train_ids_path)
    logger.info("Excluding %d training PDB IDs from eval set", len(train_ids))

    # Step 2: Query RCSB — request extra to account for dedup + failures
    query_count = args.max_structures + len(train_ids) + 200
    pdb_ids = query_rcsb(
        max_resolution=args.max_resolution,
        min_length=args.min_length,
        max_length=args.max_length,
        max_results=query_count,
    )

    # Deduplicate against training set
    pdb_ids = [pid for pid in pdb_ids if pid.upper() not in train_ids]
    logger.info(
        "After dedup: %d candidates (removed %d training overlaps)", len(pdb_ids), query_count - len(pdb_ids) - 200
    )
    pdb_ids = pdb_ids[: args.max_structures + 100]  # Keep some buffer for parse failures

    # Step 3: Download in parallel
    logger.info("Downloading %d CIF files with %d workers...", len(pdb_ids), args.download_workers)
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
            if done_count % 100 == 0:
                logger.info("Download progress: %d/%d done, %d succeeded", done_count, len(pdb_ids), len(downloaded))

    logger.info("Download complete: %d succeeded, %d failed", len(downloaded), len(failed_download))

    # Step 4: Parse structures
    logger.info("Parsing %d CIF files...", len(downloaded))
    records = []
    manifest = {}
    failed_parse = []

    for i, (pdb_id, cif_path) in enumerate(downloaded.items()):
        if len(records) >= args.max_structures:
            break

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

        if (i + 1) % 100 == 0:
            logger.info("Parse progress: %d/%d processed, %d valid", i + 1, len(downloaded), len(records))

    # Step 5: Write outputs
    output_parquet = output_dir / "eval_structures.parquet"
    output_manifest = output_dir / "eval_manifest.json"
    output_ids = output_dir / "eval_pdb_ids.txt"

    df = pd.DataFrame(records)
    df.to_parquet(str(output_parquet), index=False)
    logger.info("Wrote %d eval structures to %s", len(df), output_parquet)

    with open(output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    with open(output_ids, "w") as f:
        f.write("# Eval PDB IDs for ESM2-MiniFold TE\n")
        f.write(f"# resolution<={args.max_resolution}A, Ca completeness>={MIN_CA_COMPLETENESS}\n")
        f.write(f"# Total: {len(records)} structures (excluded {len(train_ids)} training IDs)\n")
        for r in records:
            f.write(r["pdb_id"] + "\n")

    # Summary
    if records:
        lengths = [r["num_residues"] for r in records]
        logger.info("=== Eval Dataset Summary ===")
        logger.info("Valid structures: %d", len(records))
        logger.info("Failed download: %d, failed parse: %d", len(failed_download), len(failed_parse))
        logger.info(
            "Residue lengths: min=%d, max=%d, mean=%.0f, median=%.0f",
            min(lengths),
            max(lengths),
            np.mean(lengths),
            np.median(lengths),
        )


if __name__ == "__main__":
    main()
