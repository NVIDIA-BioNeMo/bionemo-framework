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

"""Prepare external tools and databases for Arc's phage QC workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import ssl
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import yaml

from bionemo.evo2_phage_gen.arc_pipeline import ARC_EVO2_GIT_URL, ARC_EVO2_REV, _assert_arc_source_revision


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_DIR = RECIPE_ROOT / "data" / "external"
DEFAULT_BIN_DIR = DEFAULT_EXTERNAL_DIR / "bin"
DEFAULT_MMSEQS_GPU_URL = "https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
DEFAULT_BLAST_PLUS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/ncbi-blast-2.17.0+-x64-linux.tar.gz"
)
DEFAULT_PHROGS_ANNOTATION_URL = "https://phrogs.lmge.uca.fr/downloads_from_website/phrog_annot_v4.tsv"
DEFAULT_PHROGS_MMSEQS_URL = "https://phrogs.lmge.uca.fr/downloads_from_website/phrogs_mmseqs_db.tar.gz"
DEFAULT_PHROGS_FASTA_URL = "https://phrogs.lmge.uca.fr/downloads_from_website/FAA_phrog.tar.gz"
DEFAULT_ARC_EVO2_REPO_URL = ARC_EVO2_GIT_URL
DEFAULT_ARC_EVO2_REPO_REV = ARC_EVO2_REV
DEFAULT_DIAMOND_URL = "https://github.com/bbuchfink/diamond/releases/download/v2.1.24/diamond-linux64.tar.gz"
DEFAULT_HMMER_URL = "https://conda.anaconda.org/bioconda/linux-64/hmmer-3.4-hb6cb901_4.tar.bz2"
DEFAULT_SAFETY_DIR = DEFAULT_EXTERNAL_DIR / "safety"
DEFAULT_SAFETY_MANIFEST = DEFAULT_SAFETY_DIR / "asset_manifest.yaml"
DEFAULT_SAFETY_RECIPE_PATH = RECIPE_ROOT / "configs" / "phage_safety_assets.yaml"
DEFAULT_AMRFINDER_RELEASE = "amrfinder_v4.2.7"
DEFAULT_AMRFINDER_URL = "https://github.com/ncbi/amr/releases/download/amrfinder_v4.2.7/amrfinder.tar.gz"
DEFAULT_UNIPROT_TOXIN_QUERY = (
    "reviewed:true AND keyword:KW-0800 AND\n(taxonomy_id:2 OR taxonomy_id:2157 OR taxonomy_id:10239)"
)
DEFAULT_UNIPROT_TOXIN_ANNOTATIONS_URL = "https://rest.uniprot.org/uniprotkb/stream?" + urlencode(
    {
        "query": DEFAULT_UNIPROT_TOXIN_QUERY,
        "format": "tsv",
        "fields": "accession,id,protein_name,gene_names,organism_name,organism_id,cc_function",
    }
)
DEFAULT_UNIPROT_TOXIN_FASTA_URL = "https://rest.uniprot.org/uniprotkb/stream?" + urlencode(
    {"query": DEFAULT_UNIPROT_TOXIN_QUERY, "format": "fasta"}
)
PHROGS_INTEGRATION_EXCISION_CATEGORY = "integration/excision"
PHROGS_HIGH_CONFIDENCE_TERMS = (
    "integrase",
    "excisionase",
    "site-specific recombinase",
    "lysogeny repressor",
)
PHROGS_REVIEW_TERMS = ("recombinase", "repressor", "lysogeny", "integration", "excision")
UNIPROT_CC_BY_4_0_ATTRIBUTION = "UniProt data are available under the CC BY 4.0 license."


@dataclass(frozen=True)
class PreparedAsset:
    """Single prepared external asset."""

    name: str
    path: Path
    detail: str


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    """Return a stable SHA-256 digest for a file or a directory tree."""
    path = Path(path)
    if path.is_file():
        return _sha256_file(path)

    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        digest.update(str(child.relative_to(path)).encode())
        digest.update(b"\0")
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected_sha256: str | None) -> str:
    """Verify an optional declared digest and return the observed digest."""
    observed_sha256 = _sha256_file(path)
    if expected_sha256 is None:
        return observed_sha256
    normalized_expected = expected_sha256.removeprefix("sha256:").lower()
    if observed_sha256 != normalized_expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {normalized_expected}, observed {observed_sha256}")
    return observed_sha256


def _download_with_headers(
    url: str,
    output_path: Path,
    *,
    overwrite: bool = False,
    insecure: bool = False,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, str]]:
    """Download a file and retain response headers needed for provenance."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        _verify_sha256(output_path, expected_sha256)
        return output_path, {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    context = ssl._create_unverified_context() if insecure else None
    with urllib.request.urlopen(url, context=context) as response, tmp_path.open("wb") as output:
        shutil.copyfileobj(response, output)
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    try:
        _verify_sha256(tmp_path, expected_sha256)
    except ValueError:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(output_path)
    return output_path, headers


def _download(
    url: str,
    output_path: Path,
    *,
    overwrite: bool = False,
    insecure: bool = False,
    expected_sha256: str | None = None,
) -> Path:
    """Download ``url`` to ``output_path`` unless it already exists."""
    downloaded_path, _ = _download_with_headers(
        url,
        output_path,
        overwrite=overwrite,
        insecure=insecure,
        expected_sha256=expected_sha256,
    )
    return downloaded_path


def _extract_tar(archive_path: Path, output_dir: Path, *, overwrite: bool = False) -> Path:
    """Extract a tar archive into ``output_dir``."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        return output_dir
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(output_dir, filter="data")
    return output_dir


def _write_executable(path: Path, text: str) -> Path:
    """Write an executable script."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def prepare_pyrodigal_wrapper(bin_dir: Path = DEFAULT_BIN_DIR) -> PreparedAsset:
    """Create a ``prodigal`` wrapper backed by the pyrodigal CLI."""
    wrapper = _write_executable(
        Path(bin_dir) / "prodigal",
        '#!/usr/bin/env bash\nset -euo pipefail\nexec pyrodigal "$@"\n',
    )
    return PreparedAsset("prodigal_wrapper", wrapper, "prodigal-compatible wrapper around pyrodigal")


def prepare_mmseqs_gpu(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    mmseqs_url: str = DEFAULT_MMSEQS_GPU_URL,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Download and expose the official MMseqs2-GPU Linux binary."""
    external_dir = Path(external_dir)
    archive_path = _download(
        mmseqs_url,
        external_dir / "downloads" / Path(mmseqs_url).name,
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    extracted_dir = _extract_tar(archive_path, external_dir / "tools" / "mmseqs2-gpu", overwrite=overwrite)
    mmseqs_candidates = sorted(extracted_dir.glob("**/bin/mmseqs")) + sorted(extracted_dir.glob("**/mmseqs"))
    if not mmseqs_candidates:
        raise FileNotFoundError(f"No mmseqs binary found after extracting {archive_path} to {extracted_dir}")
    mmseqs_bin = mmseqs_candidates[0]
    link_path = Path(bin_dir) / "mmseqs" if bin_dir is not None else external_dir / "bin" / "mmseqs"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(mmseqs_bin.resolve())
    return PreparedAsset("mmseqs2_gpu", link_path, f"downloaded from {mmseqs_url}")


def prepare_dustmasker(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    blast_plus_url: str = DEFAULT_BLAST_PLUS_URL,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Download and expose NCBI BLAST+'s ``dustmasker`` binary."""
    external_dir = Path(external_dir)
    archive_path = _download(
        blast_plus_url,
        external_dir / "downloads" / Path(blast_plus_url).name,
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    extracted_dir = _extract_tar(archive_path, external_dir / "tools" / "ncbi-blast-plus", overwrite=overwrite)
    dustmasker_candidates = sorted(extracted_dir.glob("**/bin/dustmasker")) + sorted(
        extracted_dir.glob("**/dustmasker")
    )
    if not dustmasker_candidates:
        raise FileNotFoundError(f"No dustmasker binary found after extracting {archive_path} to {extracted_dir}")
    dustmasker_bin = dustmasker_candidates[0]
    link_path = Path(bin_dir) / "dustmasker" if bin_dir is not None else external_dir / "bin" / "dustmasker"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(dustmasker_bin.resolve())
    return PreparedAsset("dustmasker", link_path, f"downloaded from {blast_plus_url}")


def configure_lovis4u_mmseqs(mmseqs_bin: Path = DEFAULT_BIN_DIR / "mmseqs") -> PreparedAsset:
    """Point LoVis4u at the MMseqs binary prepared for this recipe."""
    mmseqs_bin = Path(mmseqs_bin)
    if not mmseqs_bin.exists():
        raise FileNotFoundError(f"Cannot configure LoVis4u; mmseqs binary does not exist: {mmseqs_bin}")

    subprocess.run(["lovis4u", "--linux"], check=True)
    subprocess.run(["lovis4u", "-smp", str(mmseqs_bin.resolve())], check=True)
    return PreparedAsset("lovis4u_mmseqs_config", mmseqs_bin, "configured LoVis4u to use recipe MMseqs")


def prepare_diamond(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    diamond_url: str = DEFAULT_DIAMOND_URL,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Download and expose the upstream DIAMOND binary used by CheckV."""
    external_dir = Path(external_dir)
    archive_path = _download(
        diamond_url,
        external_dir / "downloads" / Path(diamond_url).name,
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    extracted_dir = _extract_tar(archive_path, external_dir / "tools" / "diamond", overwrite=overwrite)
    diamond_candidates = sorted(extracted_dir.glob("**/diamond"))
    if not diamond_candidates:
        raise FileNotFoundError(f"No diamond binary found after extracting {archive_path} to {extracted_dir}")
    diamond_bin = diamond_candidates[0]
    link_path = Path(bin_dir) / "diamond" if bin_dir is not None else external_dir / "bin" / "diamond"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(diamond_bin.resolve())
    return PreparedAsset("diamond", link_path, f"downloaded from {diamond_url}")


def prepare_hmmer(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    hmmer_url: str = DEFAULT_HMMER_URL,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Download and expose HMMER's ``hmmsearch`` binary used by CheckV."""
    external_dir = Path(external_dir)
    archive_path = _download(
        hmmer_url,
        external_dir / "downloads" / Path(hmmer_url).name,
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    extracted_dir = _extract_tar(archive_path, external_dir / "tools" / "hmmer", overwrite=overwrite)
    hmmer_candidates = sorted(extracted_dir.glob("**/hmmsearch"))
    if not hmmer_candidates:
        raise FileNotFoundError(f"No hmmsearch binary found after extracting {archive_path} to {extracted_dir}")

    source_bin_dir = hmmer_candidates[0].parent
    target_bin_dir = Path(bin_dir) if bin_dir is not None else external_dir / "bin"
    target_bin_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for executable in source_bin_dir.iterdir():
        if not executable.is_file() or not (executable.stat().st_mode & 0o111):
            continue
        link_path = target_bin_dir / executable.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(executable.resolve())
        written.append(link_path)

    hmmsearch_path = target_bin_dir / "hmmsearch"
    if not hmmsearch_path.exists():
        raise FileNotFoundError(f"Expected hmmsearch link was not created in {target_bin_dir}")
    return PreparedAsset("hmmer", hmmsearch_path, f"downloaded {len(written)} executables from {hmmer_url}")


def _link_executable(source_path: Path, link_path: Path) -> Path:
    """Expose an extracted executable through the selected binary directory."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(source_path.resolve())
    return link_path


def _find_extracted_executable(extracted_dir: Path, executable_name: str) -> Path:
    """Find a named executable in an extracted upstream tool archive."""
    candidates = sorted(path for path in extracted_dir.rglob(executable_name) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No {executable_name} binary found after extracting to {extracted_dir}")
    return candidates[0]


def _read_safety_manifest(manifest_path: Path) -> dict:
    """Read a mutable safety asset manifest, rejecting malformed top-level data."""
    if not manifest_path.exists():
        return {"schema_version": 1}
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest is None:
        return {"schema_version": 1}
    if not isinstance(manifest, dict):
        raise ValueError(f"Safety manifest must be a mapping: {manifest_path}")
    return manifest


def _update_safety_manifest(manifest_path: Path, section: str, values: dict) -> None:
    """Persist one scanner-asset record while preserving prior preparation records."""
    if not DEFAULT_SAFETY_RECIPE_PATH.exists():
        raise FileNotFoundError(f"Safety asset recipe does not exist: {DEFAULT_SAFETY_RECIPE_PATH}")
    manifest = _read_safety_manifest(manifest_path)
    manifest["schema_version"] = 1
    manifest["recipe"] = {
        "path": str(DEFAULT_SAFETY_RECIPE_PATH),
        "sha256": _sha256_file(DEFAULT_SAFETY_RECIPE_PATH),
    }
    manifest[section] = values
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))


def prepare_amrfinder_plus(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    amrfinder_url: str = DEFAULT_AMRFINDER_URL,
    amrfinder_sha256: str | None = None,
    database_dir: Path | None = None,
    manifest_path: Path | None = None,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Prepare the pinned AMRFinderPlus release and a recordable custom database directory."""
    external_dir = Path(external_dir)
    safety_dir = external_dir / "safety"
    archive_path = _download(
        amrfinder_url,
        safety_dir / "downloads" / f"{DEFAULT_AMRFINDER_RELEASE}.tar.gz",
        overwrite=overwrite,
        insecure=insecure_downloads,
        expected_sha256=amrfinder_sha256,
    )
    extracted_dir = _extract_tar(
        archive_path,
        safety_dir / "tools" / DEFAULT_AMRFINDER_RELEASE,
        overwrite=overwrite,
    )
    target_bin_dir = Path(bin_dir) if bin_dir is not None else external_dir / "bin"
    amrfinder_path = _link_executable(
        _find_extracted_executable(extracted_dir, "amrfinder"), target_bin_dir / "amrfinder"
    )
    amrfinder_update_path = _link_executable(
        _find_extracted_executable(extracted_dir, "amrfinder_update"), target_bin_dir / "amrfinder_update"
    )

    requested_database_dir = Path(database_dir) if database_dir is not None else safety_dir / "amrfinder" / "database"
    if overwrite and requested_database_dir.exists():
        shutil.rmtree(requested_database_dir)
    needs_database_update = not requested_database_dir.exists() or not any(requested_database_dir.iterdir())
    if needs_database_update:
        requested_database_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(amrfinder_update_path), "-d", str(requested_database_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

    latest_database_dir = requested_database_dir / "latest"
    pinned_database_dir = (
        latest_database_dir.resolve() if latest_database_dir.exists() else requested_database_dir.resolve()
    )
    if not pinned_database_dir.exists():
        raise FileNotFoundError(f"AMRFinder update did not create a database under {requested_database_dir}")
    amrfinder_version = subprocess.run(
        [str(amrfinder_path), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if "4.2.7" not in amrfinder_version:
        raise RuntimeError(
            f"Expected AMRFinderPlus {DEFAULT_AMRFINDER_RELEASE}, but the prepared binary reports: {amrfinder_version}"
        )
    database_version = subprocess.run(
        [str(amrfinder_path), "--database", str(pinned_database_dir), "--database_version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    selected_manifest_path = Path(manifest_path) if manifest_path is not None else safety_dir / "asset_manifest.yaml"
    _update_safety_manifest(
        selected_manifest_path,
        "amrfinder_plus",
        {
            "release": DEFAULT_AMRFINDER_RELEASE,
            "release_url": amrfinder_url,
            "archive_sha256": _sha256_file(archive_path),
            "binary_path": str(amrfinder_path.resolve()),
            "amrfinder_version": amrfinder_version,
            "database_path": str(pinned_database_dir),
            "database_version": database_version,
            "database_sha256": _sha256_path(pinned_database_dir),
        },
    )
    return PreparedAsset("amrfinder_plus", amrfinder_path, f"{DEFAULT_AMRFINDER_RELEASE}: {database_version}")


def prepare_toxin_reference(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    diamond_bin: Path | None = None,
    annotations_url: str = DEFAULT_UNIPROT_TOXIN_ANNOTATIONS_URL,
    fasta_url: str = DEFAULT_UNIPROT_TOXIN_FASTA_URL,
    manifest_path: Path | None = None,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Build a DIAMOND database from the reviewed UniProt toxin reference snapshot."""
    external_dir = Path(external_dir)
    safety_dir = external_dir / "safety"
    toxin_dir = safety_dir / "toxins"
    annotations_path, annotation_headers = _download_with_headers(
        annotations_url,
        toxin_dir / "reviewed_toxins.tsv",
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    fasta_path, fasta_headers = _download_with_headers(
        fasta_url,
        toxin_dir / "reviewed_toxins.faa",
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    selected_manifest_path = Path(manifest_path) if manifest_path is not None else safety_dir / "asset_manifest.yaml"
    existing_manifest = _read_safety_manifest(selected_manifest_path)
    existing_toxin_reference = existing_manifest.get("toxin_reference", {})
    if not isinstance(existing_toxin_reference, dict):
        existing_toxin_reference = {}
    uniprot_release = (
        annotation_headers.get("x-uniprot-release")
        or fasta_headers.get("x-uniprot-release")
        or existing_toxin_reference.get("uniprot_release")
    )
    if not isinstance(uniprot_release, str) or not uniprot_release:
        raise RuntimeError("UniProt response did not include the required X-UniProt-Release header")

    diamond_database = toxin_dir / "reviewed_toxins.dmnd"
    if overwrite or not diamond_database.exists():
        selected_diamond_bin = Path(diamond_bin) if diamond_bin is not None else external_dir / "bin" / "diamond"
        subprocess.run(
            [str(selected_diamond_bin), "makedb", "--in", str(fasta_path), "--db", str(diamond_database)],
            check=True,
        )
    if not diamond_database.exists():
        raise FileNotFoundError(f"DIAMOND did not create toxin database: {diamond_database}")

    _update_safety_manifest(
        selected_manifest_path,
        "toxin_reference",
        {
            "query": DEFAULT_UNIPROT_TOXIN_QUERY,
            "annotations_url": annotations_url,
            "fasta_url": fasta_url,
            "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "uniprot_release": uniprot_release,
            "license": "CC BY 4.0",
            "attribution": UNIPROT_CC_BY_4_0_ATTRIBUTION,
            "files": {
                "annotations": {
                    "path": str(annotations_path.resolve()),
                    "role": "canonical reviewed toxin accessions and annotations",
                    "sha256": _sha256_file(annotations_path),
                },
                "fasta": {
                    "path": str(fasta_path.resolve()),
                    "role": "reviewed toxin protein sequences",
                    "sha256": _sha256_file(fasta_path),
                },
                "diamond_database": {
                    "path": str(diamond_database.resolve()),
                    "role": "DIAMOND index of reviewed toxin proteins",
                    "sha256": _sha256_file(diamond_database),
                },
            },
        },
    )
    return PreparedAsset("toxin_reference", diamond_database, "reviewed UniProt toxin DIAMOND database")


def prepare_phrogs_safety_metadata(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    manifest_path: Path | None = None,
) -> PreparedAsset:
    """Build a PHROGs v4 integration/excision lookup table for lysogeny evidence."""
    external_dir = Path(external_dir)
    annotation_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    if not annotation_path.exists():
        raise FileNotFoundError(f"PHROGs v4 annotation table is required: {annotation_path}")

    safety_dir = external_dir / "safety"
    lookup_path = safety_dir / "phrogs" / "phrogs_integration_excision_v4.tsv"
    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    with annotation_path.open(newline="") as source, lookup_path.open("w", newline="") as output:
        reader = csv.DictReader(source, delimiter="\t")
        required_columns = {"phrog", "annot", "category"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"PHROGs v4 table must contain {sorted(required_columns)}: {annotation_path}")
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(["phrog", "annot", "category", "confidence", "matched_term"])
        for row in reader:
            category = row["category"].strip()
            if category.casefold() != PHROGS_INTEGRATION_EXCISION_CATEGORY:
                continue
            annotation = row["annot"].strip()
            normalized_annotation = annotation.casefold()
            high_confidence_term = next(
                (term for term in PHROGS_HIGH_CONFIDENCE_TERMS if term in normalized_annotation), None
            )
            review_term = next((term for term in PHROGS_REVIEW_TERMS if term in normalized_annotation), None)
            confidence = "high_confidence" if high_confidence_term is not None else "review"
            matched_term = high_confidence_term or review_term or "integration/excision category"
            writer.writerow([row["phrog"].strip(), annotation, category, confidence, matched_term])

    selected_manifest_path = Path(manifest_path) if manifest_path is not None else safety_dir / "asset_manifest.yaml"
    _update_safety_manifest(
        selected_manifest_path,
        "phrogs_v4",
        {
            "annotation_url": DEFAULT_PHROGS_ANNOTATION_URL,
            "source_path": str(annotation_path.resolve()),
            "source_sha256": _sha256_file(annotation_path),
            "category": PHROGS_INTEGRATION_EXCISION_CATEGORY,
            "high_confidence_terms": list(PHROGS_HIGH_CONFIDENCE_TERMS),
            "review_terms": list(PHROGS_REVIEW_TERMS),
            "lookup_path": str(lookup_path.resolve()),
            "lookup_sha256": _sha256_file(lookup_path),
            "sequence_assets": {
                "mmseqs_profile_database": str((external_dir / "phrogs" / "phrogs_mmseqs_db").resolve()),
                "gpu_sequence_database": str((external_dir / "phrogs" / "phrogs_gpu_seq_db_pad").resolve()),
            },
        },
    )
    return PreparedAsset("phrogs_safety_metadata", lookup_path, "PHROGs v4 integration/excision lookup table")


def prepare_phrogs_annotation(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    annotation_url: str = DEFAULT_PHROGS_ANNOTATION_URL,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Download PHROGs v4 annotation table."""
    annotation_path = _download(
        annotation_url,
        Path(external_dir) / "phrogs" / "phrog_annot_v4.tsv",
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    return PreparedAsset("phrogs_annotation", annotation_path, f"downloaded from {annotation_url}")


def prepare_phrogs_mmseqs_db(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    phrogs_mmseqs_url: str = DEFAULT_PHROGS_MMSEQS_URL,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Download and extract PHROGs MMseqs profile database."""
    external_dir = Path(external_dir)
    archive_path = _download(
        phrogs_mmseqs_url,
        external_dir / "downloads" / Path(phrogs_mmseqs_url).name,
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    extracted_dir = _extract_tar(archive_path, external_dir / "phrogs" / "phrogs_mmseqs_db", overwrite=overwrite)
    return PreparedAsset("phrogs_mmseqs_db", extracted_dir, f"downloaded from {phrogs_mmseqs_url}")


def prepare_phrogs_gpu_sequence_db(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    phrogs_fasta_url: str = DEFAULT_PHROGS_FASTA_URL,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Build a padded GPU-ready MMseqs sequence database from PHROGs FASTA files."""
    external_dir = Path(external_dir)
    phrogs_dir = external_dir / "phrogs"
    archive_path = _download(
        phrogs_fasta_url,
        external_dir / "downloads" / Path(phrogs_fasta_url).name,
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    extracted_dir = _extract_tar(archive_path, phrogs_dir / "FAA_phrog", overwrite=overwrite)
    combined_fasta = phrogs_dir / "FAA_phrog_combined.faa"
    padded_db = phrogs_dir / "phrogs_gpu_seq_db_pad"
    if padded_db.exists() and not overwrite:
        return PreparedAsset("phrogs_gpu_seq_db", padded_db, "existing padded GPU MMseqs DB")

    fasta_paths = sorted(
        path
        for path in extracted_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".faa", ".fa", ".fasta"}
    )
    if not fasta_paths:
        raise FileNotFoundError(f"No FASTA files found after extracting {archive_path} to {extracted_dir}")
    with combined_fasta.open("wb") as output:
        for fasta_path in fasta_paths:
            with fasta_path.open("rb") as source:
                shutil.copyfileobj(source, output)

    seq_db = phrogs_dir / "phrogs_gpu_seq_db"
    mmseqs_bin = external_dir / "bin" / "mmseqs"
    mmseqs_cmd = str(mmseqs_bin) if mmseqs_bin.exists() else "mmseqs"
    subprocess.run([mmseqs_cmd, "createdb", str(combined_fasta), str(seq_db)], check=True)
    subprocess.run([mmseqs_cmd, "makepaddedseqdb", str(seq_db), str(padded_db)], check=True)
    return PreparedAsset("phrogs_gpu_seq_db", padded_db, f"built from {len(fasta_paths)} PHROGs FASTA files")


def prepare_checkv_database(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    overwrite: bool = False,
) -> PreparedAsset:
    """Download CheckV's database with the installed ``checkv`` CLI."""
    external_dir = Path(external_dir)
    checkv_root = external_dir / "checkv"
    existing = sorted(checkv_root.glob("checkv-db-*")) if checkv_root.exists() else []
    if existing and not overwrite:
        return PreparedAsset("checkv_database", existing[-1], "existing CheckV database")
    if overwrite and checkv_root.exists():
        shutil.rmtree(checkv_root)
    checkv_root.mkdir(parents=True, exist_ok=True)
    checkv_bin_dir = Path(bin_dir) if bin_dir is not None else external_dir / "bin"
    env = {**os.environ, "PATH": f"{checkv_bin_dir}:{os.environ.get('PATH', '')}"}
    subprocess.run(["checkv", "download_database", str(checkv_root)], check=True, env=env)
    downloaded = sorted(checkv_root.glob("checkv-db-*"))
    if not downloaded:
        raise FileNotFoundError(f"checkv download_database did not create checkv-db-* under {checkv_root}")
    return PreparedAsset("checkv_database", downloaded[-1], "downloaded with checkv download_database")


def prepare_arc_evo2_checkout(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    repo_url: str = DEFAULT_ARC_EVO2_REPO_URL,
    repo_rev: str = DEFAULT_ARC_EVO2_REPO_REV,
    overwrite: bool = False,
) -> PreparedAsset:
    """Clone Arc's Evo2 repository for phage reference data and QC scripts."""
    checkout_dir = Path(external_dir) / "arc_evo2"
    if checkout_dir.exists() and not overwrite:
        _assert_arc_source_revision(checkout_dir, repo_rev)
        return PreparedAsset("arc_evo2", checkout_dir, f"existing Arc Evo2 checkout at {repo_rev}")
    if checkout_dir.exists() and overwrite:
        shutil.rmtree(checkout_dir)
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", repo_url, str(checkout_dir)], check=True)
    subprocess.run(["git", "-C", str(checkout_dir), "checkout", repo_rev], check=True)
    return PreparedAsset("arc_evo2", checkout_dir, f"cloned from {repo_url}@{repo_rev}")


def prepare_external_assets(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    download_mmseqs: bool = True,
    download_dustmasker: bool = True,
    download_diamond: bool = True,
    download_hmmer: bool = True,
    download_phrogs_annotation: bool = True,
    download_arc_evo2: bool = True,
    download_large_databases: bool = False,
    download_checkv: bool = True,
    configure_lovis4u: bool = True,
    with_safety: bool = False,
    safety_manifest: Path | None = None,
    mmseqs_url: str = DEFAULT_MMSEQS_GPU_URL,
    blast_plus_url: str = DEFAULT_BLAST_PLUS_URL,
    diamond_url: str = DEFAULT_DIAMOND_URL,
    hmmer_url: str = DEFAULT_HMMER_URL,
    phrogs_mmseqs_url: str = DEFAULT_PHROGS_MMSEQS_URL,
    phrogs_fasta_url: str = DEFAULT_PHROGS_FASTA_URL,
    arc_evo2_repo_url: str = DEFAULT_ARC_EVO2_REPO_URL,
    arc_evo2_repo_rev: str = DEFAULT_ARC_EVO2_REPO_REV,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> list[PreparedAsset]:
    """Prepare external assets for Arc's phage QC pipeline."""
    external_dir = Path(external_dir)
    target_bin_dir = Path(bin_dir) if bin_dir is not None else external_dir / "bin"
    assets = [prepare_pyrodigal_wrapper(target_bin_dir)]
    mmseqs_asset = None
    if download_mmseqs:
        mmseqs_asset = prepare_mmseqs_gpu(
            external_dir,
            bin_dir=target_bin_dir,
            mmseqs_url=mmseqs_url,
            overwrite=overwrite,
            insecure_downloads=insecure_downloads,
        )
        assets.append(mmseqs_asset)
    if download_dustmasker:
        assets.append(
            prepare_dustmasker(
                external_dir,
                bin_dir=target_bin_dir,
                blast_plus_url=blast_plus_url,
                overwrite=overwrite,
                insecure_downloads=insecure_downloads,
            )
        )
    if configure_lovis4u and mmseqs_asset is not None:
        assets.append(configure_lovis4u_mmseqs(mmseqs_asset.path))
    if download_diamond:
        assets.append(
            prepare_diamond(
                external_dir,
                bin_dir=target_bin_dir,
                diamond_url=diamond_url,
                overwrite=overwrite,
                insecure_downloads=insecure_downloads,
            )
        )
    if download_hmmer:
        assets.append(
            prepare_hmmer(
                external_dir,
                bin_dir=target_bin_dir,
                hmmer_url=hmmer_url,
                overwrite=overwrite,
                insecure_downloads=insecure_downloads,
            )
        )
    if download_phrogs_annotation:
        assets.append(
            prepare_phrogs_annotation(external_dir, overwrite=overwrite, insecure_downloads=insecure_downloads)
        )
    if download_arc_evo2:
        assets.append(
            prepare_arc_evo2_checkout(
                external_dir,
                repo_url=arc_evo2_repo_url,
                repo_rev=arc_evo2_repo_rev,
                overwrite=overwrite,
            )
        )
    if download_large_databases:
        assets.append(
            prepare_phrogs_mmseqs_db(
                external_dir,
                phrogs_mmseqs_url=phrogs_mmseqs_url,
                overwrite=overwrite,
                insecure_downloads=insecure_downloads,
            )
        )
        assets.append(
            prepare_phrogs_gpu_sequence_db(
                external_dir,
                phrogs_fasta_url=phrogs_fasta_url,
                overwrite=overwrite,
                insecure_downloads=insecure_downloads,
            )
        )
        if download_checkv:
            assets.append(prepare_checkv_database(external_dir, bin_dir=target_bin_dir, overwrite=overwrite))
    if with_safety:
        selected_safety_manifest = (
            Path(safety_manifest) if safety_manifest is not None else external_dir / "safety" / "asset_manifest.yaml"
        )
        assets.append(
            prepare_amrfinder_plus(
                external_dir,
                bin_dir=target_bin_dir,
                manifest_path=selected_safety_manifest,
                overwrite=overwrite,
                insecure_downloads=insecure_downloads,
            )
        )
        assets.append(
            prepare_toxin_reference(
                external_dir,
                diamond_bin=target_bin_dir / "diamond",
                manifest_path=selected_safety_manifest,
                overwrite=overwrite,
                insecure_downloads=insecure_downloads,
            )
        )
        assets.append(prepare_phrogs_safety_metadata(external_dir, manifest_path=selected_safety_manifest))
    return assets


def main() -> None:
    """CLI entry point for preparing external analysis assets."""
    parser = argparse.ArgumentParser(description="Prepare external tools/databases for Evo2 phage QC")
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument(
        "--bin-dir", type=Path, default=None, help="Directory for exposed tool links (default: external-dir/bin)"
    )
    parser.add_argument("--skip-mmseqs", action="store_true", help="Do not download MMseqs2-GPU")
    parser.add_argument("--skip-dustmasker", action="store_true", help="Do not download NCBI BLAST+/dustmasker")
    parser.add_argument("--skip-lovis4u-config", action="store_true", help="Do not configure LoVis4u's MMseqs path")
    parser.add_argument("--skip-diamond", action="store_true", help="Do not download the upstream DIAMOND binary")
    parser.add_argument("--skip-hmmer", action="store_true", help="Do not download HMMER")
    parser.add_argument("--skip-phrogs-annotation", action="store_true", help="Do not download PHROGs annotation TSV")
    parser.add_argument("--skip-arc-evo2", action="store_true", help="Do not clone Arc's Evo2 repository")
    parser.add_argument(
        "--download-large-databases", action="store_true", help="Also download PHROGs MMseqs DB and CheckV DB"
    )
    parser.add_argument("--skip-checkv", action="store_true", help="Do not download/build the CheckV database")
    parser.add_argument(
        "--with-safety",
        action="store_true",
        help="Also prepare pinned AMRFinderPlus, toxin, and PHROGs sequence-safety assets",
    )
    parser.add_argument(
        "--safety-manifest",
        type=Path,
        default=DEFAULT_SAFETY_MANIFEST,
        help="Runtime safety asset manifest (default: data/external/safety/asset_manifest.yaml)",
    )
    parser.add_argument("--mmseqs-url", default=DEFAULT_MMSEQS_GPU_URL)
    parser.add_argument("--blast-plus-url", default=DEFAULT_BLAST_PLUS_URL)
    parser.add_argument("--diamond-url", default=DEFAULT_DIAMOND_URL)
    parser.add_argument("--hmmer-url", default=DEFAULT_HMMER_URL)
    parser.add_argument("--phrogs-mmseqs-url", default=DEFAULT_PHROGS_MMSEQS_URL)
    parser.add_argument("--phrogs-fasta-url", default=DEFAULT_PHROGS_FASTA_URL)
    parser.add_argument("--arc-evo2-repo-url", default=DEFAULT_ARC_EVO2_REPO_URL)
    parser.add_argument("--arc-evo2-repo-rev", default=DEFAULT_ARC_EVO2_REPO_REV)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--insecure-downloads",
        action="store_true",
        help="Disable TLS certificate verification for hosts with expired certificates.",
    )
    args = parser.parse_args()

    assets = prepare_external_assets(
        args.external_dir,
        bin_dir=args.bin_dir,
        download_mmseqs=not args.skip_mmseqs,
        download_dustmasker=not args.skip_dustmasker,
        download_diamond=not args.skip_diamond,
        download_hmmer=not args.skip_hmmer,
        download_phrogs_annotation=not args.skip_phrogs_annotation,
        download_arc_evo2=not args.skip_arc_evo2,
        download_large_databases=args.download_large_databases,
        download_checkv=not args.skip_checkv,
        configure_lovis4u=not args.skip_lovis4u_config,
        with_safety=args.with_safety,
        safety_manifest=args.safety_manifest,
        mmseqs_url=args.mmseqs_url,
        blast_plus_url=args.blast_plus_url,
        diamond_url=args.diamond_url,
        hmmer_url=args.hmmer_url,
        phrogs_mmseqs_url=args.phrogs_mmseqs_url,
        phrogs_fasta_url=args.phrogs_fasta_url,
        arc_evo2_repo_url=args.arc_evo2_repo_url,
        arc_evo2_repo_rev=args.arc_evo2_repo_rev,
        overwrite=args.overwrite,
        insecure_downloads=args.insecure_downloads,
    )
    for asset in assets:
        print(f"{asset.name}: {asset.path} ({asset.detail})")
    print(f"export PATH={Path(args.bin_dir) if args.bin_dir else Path(args.external_dir) / 'bin'}:$PATH")
    checkv_dirs = [] if args.skip_checkv else sorted((Path(args.external_dir) / "checkv").glob("checkv-db-*"))
    if checkv_dirs:
        print(f"export CHECKVDB={checkv_dirs[-1]}")


if __name__ == "__main__":
    main()
