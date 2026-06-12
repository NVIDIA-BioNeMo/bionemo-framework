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

"""Download SwissProt proteins with both amino acid annotations and nucleotide CDS sequences.

For each well-annotated SwissProt protein, fetches the EMBL/ENA cross-reference
to get the actual coding DNA sequence (CDS). This enables F1 evaluation of
codon-level models (like CodoNFM) against protein-level SwissProt annotations.

Usage:
    python scripts/download_codonfm_swissprot.py \
        --output-dir ./data/codonfm_swissprot \
        --max-proteins 8000 \
        --max-length 512 \
        --workers 8

Output:
    codonfm_swissprot.tsv.gz  -- TSV with columns:
        accession, protein_sequence, codon_sequence, length, + all annotation columns
    summary.json              -- stats on coverage, failures, etc.
"""

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


# UniProt annotation feature fields (same as ESM2 pipeline)
UNIPROT_FEATURE_FIELDS = [
    "ft_act_site",
    "ft_binding",
    "ft_disulfid",
    "ft_carbohyd",
    "ft_lipid",
    "ft_mod_res",
    "ft_signal",
    "ft_transit",
    "ft_helix",
    "ft_turn",
    "ft_strand",
    "ft_coiled",
    "ft_compbias",
    "ft_domain",
    "ft_motif",
    "ft_region",
    "ft_zn_fing",
]

CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def translate_cds(cds: str) -> str:
    """Translate a CDS nucleotide sequence to amino acids (excluding stop codon)."""
    protein = []
    for i in range(0, len(cds) - 2, 3):
        codon = cds[i : i + 3].upper()
        aa = CODON_TABLE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def fetch_embl_cds_ids(accession: str, session: requests.Session) -> List[Dict]:
    """Fetch EMBL cross-references for a UniProt accession.

    Returns list of dicts with keys: cds_id, molecule_type, etc.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    cds_refs = []
    for xref in data.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "EMBL":
            continue
        properties = {p["key"]: p["value"] for p in xref.get("properties", [])}
        protein_id = properties.get("ProteinId", "")
        mol_type = properties.get("MoleculeType", "")
        status = properties.get("Status", "")
        # Skip entries without a valid protein sequence ID
        if protein_id and protein_id != "-" and mol_type != "Genomic_DNA":
            cds_refs.append(
                {
                    "embl_id": xref.get("id", ""),
                    "protein_id": protein_id,
                    "molecule_type": mol_type,
                    "status": status,
                }
            )
    return cds_refs


def fetch_ena_cds_sequence(cds_protein_id: str, session: requests.Session) -> Optional[str]:
    """Fetch a CDS nucleotide sequence from ENA by its protein ID.

    Tries the ENA CDS FASTA endpoint.
    """
    url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{cds_protein_id}?download=true"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        lines = resp.text.strip().split("\n")
        seq = "".join(line.strip() for line in lines if not line.startswith(">"))
        if seq and len(seq) >= 3:
            return seq.upper()
    except Exception:
        pass
    return None


def fetch_cds_for_protein(
    accession: str,
    protein_sequence: str,
    session: requests.Session,
    max_retries: int = 2,
) -> Optional[str]:
    """Try to find a CDS that translates to match the SwissProt protein sequence.

    Tries each EMBL cross-reference until one matches.
    """
    for attempt in range(max_retries):
        try:
            cds_refs = fetch_embl_cds_ids(accession, session)
            break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None

    for ref in cds_refs:
        cds_protein_id = ref["protein_id"]
        for attempt in range(max_retries):
            try:
                cds_seq = fetch_ena_cds_sequence(cds_protein_id, session)
                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    cds_seq = None

        if cds_seq is None:
            continue

        # Validate: translate and compare to protein sequence
        translated = translate_cds(cds_seq)
        if translated == protein_sequence:
            return cds_seq

        # Try with initial methionine mismatch (some CDS start with alt start codons)
        if len(translated) == len(protein_sequence) and translated[1:] == protein_sequence[1:]:
            return cds_seq

    return None


def download_annotated_proteins_tsv(
    max_length: int = 512,
    annotation_score: int = 5,
    max_results: Optional[int] = None,
) -> str:
    """Download annotated proteins from UniProt as TSV string."""
    query_parts = [
        "(reviewed:true)",
        f"(annotation_score:{annotation_score})",
        f"(length:[1 TO {max_length}])",
    ]
    query = " AND ".join(query_parts)

    fields = ["accession", "sequence", "length"] + UNIPROT_FEATURE_FIELDS

    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "fields": ",".join(fields),
        "format": "tsv",
    }
    if max_results:
        params["size"] = max_results

    print(f"Downloading from UniProt: {query}")
    resp = requests.get(url, params=params, stream=True)
    resp.raise_for_status()

    content = resp.text
    lines = content.strip().split("\n")
    print(f"Downloaded {len(lines) - 1} proteins from UniProt")
    return content


def parse_tsv_rows(tsv_content: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse TSV content into header + list of row dicts."""
    lines = tsv_content.strip().split("\n")
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        fields = line.split("\t")
        row = {}
        for i, col in enumerate(header):
            row[col] = fields[i] if i < len(fields) else ""
        rows.append(row)
    return header, rows


def process_single_protein(args_tuple):
    """Worker function for parallel CDS fetching."""
    accession, protein_seq, session = args_tuple
    cds = fetch_cds_for_protein(accession, protein_seq, session)
    return accession, cds


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Download SwissProt proteins with CDS nucleotide sequences for CodoNFM eval"
    )
    p.add_argument("--output-dir", type=str, default="./data/codonfm_swissprot")
    p.add_argument("--max-proteins", type=int, default=8000, help="Max proteins to download from UniProt")
    p.add_argument("--max-length", type=int, default=512, help="Max protein sequence length")
    p.add_argument("--annotation-score", type=int, default=5, help="Min UniProt annotation score (1-5)")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers for ENA fetching")
    p.add_argument("--rate-limit-delay", type=float, default=0.1, help="Delay between ENA requests (seconds)")
    return p.parse_args()


def main():
    """Download annotated SwissProt proteins and fetch their CDS sequences from ENA."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download annotated proteins from UniProt
    print("=" * 60)
    print("STEP 1: Download annotated proteins from UniProt")
    print("=" * 60)
    tsv_content = download_annotated_proteins_tsv(
        max_length=args.max_length,
        annotation_score=args.annotation_score,
        max_results=args.max_proteins,
    )
    header, rows = parse_tsv_rows(tsv_content)

    # Step 2: Fetch CDS for each protein
    print()
    print("=" * 60)
    print(f"STEP 2: Fetch CDS nucleotide sequences from ENA ({len(rows)} proteins)")
    print("=" * 60)

    session = requests.Session()
    session.headers.update({"User-Agent": "BioNeMo-SAE/1.0 (CodoNFM eval pipeline)"})

    cds_map = {}
    failed = []

    tasks = [(row.get("Entry", row.get("Accession", "")), row.get("Sequence", ""), session) for row in rows]

    # Use thread pool for parallel fetching
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for task in tasks:
            future = executor.submit(process_single_protein, task)
            futures[future] = task[0]  # accession

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching CDS"):
            accession = futures[future]
            try:
                _, cds = future.result()
                if cds:
                    cds_map[accession] = cds
                else:
                    failed.append(accession)
            except Exception as e:
                print(f"  Error for {accession}: {e}")
                failed.append(accession)

    print(f"\nCDS fetch results: {len(cds_map)} succeeded, {len(failed)} failed")

    # Step 3: Write output TSV with codon_sequence column
    print()
    print("=" * 60)
    print("STEP 3: Write output dataset")
    print("=" * 60)

    output_path = output_dir / "codonfm_swissprot.tsv.gz"

    # Build output header: insert codon_sequence after sequence
    out_header = []
    for col in header:
        out_header.append(col)
        if col.lower() == "sequence":
            out_header.append("Codon sequence")

    n_written = 0
    with gzip.open(output_path, "wt") as f:
        f.write("\t".join(out_header) + "\n")
        for row in rows:
            accession = row.get("Entry", row.get("Accession", ""))
            if accession not in cds_map:
                continue

            cds = cds_map[accession]
            out_fields = []
            for col in header:
                out_fields.append(row.get(col, ""))
                if col.lower() == "sequence":
                    out_fields.append(cds)
            f.write("\t".join(out_fields) + "\n")
            n_written += 1

    print(f"Wrote {n_written} proteins to {output_path}")

    # Step 4: Write summary
    summary = {
        "total_uniprot_proteins": len(rows),
        "cds_found": len(cds_map),
        "cds_failed": len(failed),
        "proteins_written": n_written,
        "coverage_pct": round(100 * len(cds_map) / max(len(rows), 1), 1),
        "max_length": args.max_length,
        "annotation_score": args.annotation_score,
        "output_file": str(output_path),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")
    print(f"  Coverage: {summary['coverage_pct']}% ({summary['cds_found']}/{summary['total_uniprot_proteins']})")
    print("\nTo use with CodoNFM eval:")
    print(f"  - Load {output_path}")
    print("  - 'Codon sequence' column has the nucleotide CDS")
    print("  - All annotation columns are identical to the ESM2 SwissProt format")
    print("  - Codon position i maps to amino acid position i (codon = nts 3i..3i+2)")


if __name__ == "__main__":
    main()
