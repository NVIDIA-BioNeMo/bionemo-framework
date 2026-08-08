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
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

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
DEFAULT_PHROGS_SAFETY_PROFILE_URL = "https://zenodo.org/record/17110353/files/pharokka_v1.8.0_databases.tar.gz"
DEFAULT_PHROGS_SAFETY_PROFILE_MD5 = "a63c485241b900a11989bd1821bfbb09"
DEFAULT_PHROGS_SAFETY_PROFILE_SIZE = 656_171_247
DEFAULT_PHROGS_SAFETY_PROFILE_RELEASE = "Pharokka database v1.8.0"
DEFAULT_PHROGS_SAFETY_PROFILE_DOI = "10.5281/zenodo.17110353"
DEFAULT_PHROGS_SAFETY_PROFILE_LICENSE = "CC BY 4.0"
DEFAULT_ARC_EVO2_REPO_URL = ARC_EVO2_GIT_URL
DEFAULT_ARC_EVO2_REPO_REV = ARC_EVO2_REV
DEFAULT_DIAMOND_URL = "https://github.com/bbuchfink/diamond/releases/download/v2.1.24/diamond-linux64.tar.gz"
DEFAULT_HMMER_URL = "https://conda.anaconda.org/bioconda/linux-64/hmmer-3.4-hb6cb901_4.tar.bz2"
DEFAULT_SAFETY_DIR = DEFAULT_EXTERNAL_DIR / "safety"
DEFAULT_SAFETY_MANIFEST = DEFAULT_SAFETY_DIR / "asset_manifest.yaml"
DEFAULT_SAFETY_RECIPE_PATH = RECIPE_ROOT / "configs" / "phage_safety_assets.yaml"
DEFAULT_AMRFINDER_RELEASE = "amrfinder_v4.2.7"
DEFAULT_AMRFINDER_URL = (
    "https://github.com/ncbi/amr/releases/download/amrfinder_v4.2.7/amrfinder_binaries_v4.2.7.tar.gz"
)
DEFAULT_AMRFINDER_SHA256 = "68045a8bccdbe3c5dcdf941bebe2352ed419758a9914c41f48f0bbbd6fbade56"
DEFAULT_PHROGS_ANNOTATION_SHA256 = "502f96101597c21133bcce5711803e0b95e0c162cd4e86425c352549bd95e8c2"
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
PHROGS_INTEGRATION_EXCISION_CATEGORY = "integration and excision"
PHROGS_HIGH_CONFIDENCE_TERMS = (
    "integrase",
    "excisionase",
    "site-specific recombinase",
    "lysogeny repressor",
)
PHROGS_REVIEW_TERMS = ("recombinase", "repressor", "lysogeny", "integration", "excision")
PHROGS_PROFILE_DATABASE_NAME = "phrogs_profile_db"
PHROGS_PROFILE_RELEASE_MARKER = "VERSION_1_8_0"
PHROGS_PROFILE_RELEASE = DEFAULT_PHROGS_SAFETY_PROFILE_RELEASE
PHROGS_PROFILE_DATASET_RELEASE = "PHROGs v4"
PHROGS_PROFILE_MIN_MMSEQS_VERSION = "14"
PHROGS_PROFILE_BUILDER_MMSEQS_VERSION = "18.8cc5c"
PHROGS_PROFILE_SEARCH_ORIENTATION = "phrog_profile_query_vs_orf_target"
PHROGS_PROFILE_SEARCH_SCOPE = "full_phrogs_v4_profile_database"
PHROGS_PROFILE_LOOKUP_JOIN_POLICY = "classify_only_profile_ids_present_in_pinned_lookup"
PHROGS_PROFILE_OUTPUT_FIELDS = (
    "query",
    "target",
    "pident",
    "alnlen",
    "qlen",
    "tlen",
    "qcov",
    "tcov",
    "evalue",
    "bits",
)
PHROGS_PROFILE_OUTPUT_UNITS = {"pident": "percent", "qcov": "fraction", "tcov": "fraction"}
PHROGS_PROFILE_QUERY_ID_PATTERN = r"^phrog_[1-9][0-9]*$"
UNIPROT_CC_BY_4_0_ATTRIBUTION = "UniProt data are available under the CC BY 4.0 license."
AMRFINDER_CITATION = (
    "Feldgarden et al. (2021), AMRFinderPlus and the Reference Gene Catalog facilitate examination "
    "of the genomic links among antimicrobial resistance, stress response, and virulence, "
    "Scientific Reports 11:12728."
)
PHROGS_CITATION = (
    "Terzian et al. (2021), PHROG: families of prokaryotic virus proteins clustered using remote homology, "
    "Nucleic Acids Research 49:D1345-D1353."
)
PHROGS_PROFILE_SOURCE_CITATION = "Pharokka database v1.8.0 (DOI: 10.5281/zenodo.17110353)."
PHROGS_USE_TERMS = (
    "Consult the PHROGs project download page for its published use and download conditions; "
    "this manifest does not assert a license."
)


@dataclass(frozen=True)
class PreparedAsset:
    """Single prepared external asset."""

    name: str
    path: Path
    detail: str


_VERIFIED_PHROGS_PROFILE_AUTHORITY = object()


@dataclass(frozen=True)
class _VerifiedPhrogsProfile:
    """Private capability issued only after clean extraction of the pinned Pharokka archive."""

    archive_path: Path
    observed_archive_sha256: str
    extracted_dir: Path
    profile_database: Path
    database_sha256: str
    tree_sha256: str
    profile_id_inventory: dict[str, int | str]
    _authority: object


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    """Return the MD5 digest for an archive with an upstream MD5 checksum."""
    digest = hashlib.md5(usedforsecurity=False)
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


def _verify_md5(path: Path, expected_md5: str) -> str:
    """Verify a published MD5 checksum before extracting the associated archive."""
    normalized_expected = expected_md5.removeprefix("md5:").lower()
    observed_md5 = _md5_file(path)
    if observed_md5 != normalized_expected:
        raise ValueError(f"MD5 mismatch for {path}: expected {normalized_expected}, observed {observed_md5}")
    return observed_md5


def _verify_file_size(path: Path, expected_size: int) -> int:
    """Verify a source-published archive size before extracting it."""
    observed_size = Path(path).stat().st_size
    if observed_size != expected_size:
        raise ValueError(f"Archive size mismatch for {path}: expected {expected_size}, observed {observed_size}")
    return observed_size


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
    """Download and expose the BLAST+ tools needed by QC and AMRFinder updates."""
    external_dir = Path(external_dir)
    archive_path = _download(
        blast_plus_url,
        external_dir / "downloads" / Path(blast_plus_url).name,
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    extracted_dir = _extract_tar(archive_path, external_dir / "tools" / "ncbi-blast-plus", overwrite=overwrite)
    dustmasker_bin = _find_extracted_executable(extracted_dir, "dustmasker")
    makeblastdb_bin = _find_extracted_executable(extracted_dir, "makeblastdb")
    target_bin_dir = Path(bin_dir) if bin_dir is not None else external_dir / "bin"
    link_path = _link_executable(dustmasker_bin, target_bin_dir / "dustmasker")
    _link_executable(makeblastdb_bin, target_bin_dir / "makeblastdb")
    return PreparedAsset("dustmasker", link_path, f"downloaded BLAST+ tools from {blast_plus_url}")


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
    """Download and expose HMMER search/index tools used by CheckV and AMRFinder."""
    external_dir = Path(external_dir)
    archive_path = _download(
        hmmer_url,
        external_dir / "downloads" / Path(hmmer_url).name,
        overwrite=overwrite,
        insecure=insecure_downloads,
    )
    extracted_dir = _extract_tar(archive_path, external_dir / "tools" / "hmmer", overwrite=overwrite)
    hmmsearch_bin = _find_extracted_executable(extracted_dir, "hmmsearch")
    hmmpress_bin = _find_extracted_executable(extracted_dir, "hmmpress")
    target_bin_dir = Path(bin_dir) if bin_dir is not None else external_dir / "bin"
    target_bin_dir.mkdir(parents=True, exist_ok=True)
    written = []
    source_bin_dirs = {hmmsearch_bin.parent, hmmpress_bin.parent}
    for source_bin_dir in source_bin_dirs:
        for executable in source_bin_dir.iterdir():
            if not executable.is_file() or not (executable.stat().st_mode & 0o111):
                continue
            written.append(_link_executable(executable, target_bin_dir / executable.name))

    hmmsearch_path = target_bin_dir / "hmmsearch"
    hmmpress_path = target_bin_dir / "hmmpress"
    if not hmmsearch_path.exists() or not hmmpress_path.exists():
        raise FileNotFoundError(f"Expected hmmsearch and hmmpress links in {target_bin_dir}")
    return PreparedAsset("hmmer", hmmsearch_path, f"downloaded {len(written)} executables from {hmmer_url}")


def _link_executable(source_path: Path, link_path: Path) -> Path:
    """Expose an extracted executable through the selected binary directory."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(source_path.resolve())
    return link_path


def _copy_executable(source_path: Path, destination_path: Path) -> Path:
    """Copy an executable into an immutable safety generation without retaining a mutable link."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Required executable does not exist: {source_path}")
    if source_path.resolve() == destination_path.resolve():
        return destination_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()
    shutil.copy2(source_path, destination_path)
    return destination_path


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


def _set_safety_manifest_recipe(manifest: dict) -> None:
    """Attach the tracked recipe identity to an in-memory safety manifest."""
    if not DEFAULT_SAFETY_RECIPE_PATH.exists():
        raise FileNotFoundError(f"Safety asset recipe does not exist: {DEFAULT_SAFETY_RECIPE_PATH}")
    manifest["schema_version"] = 1
    manifest["recipe"] = {
        "path": str(DEFAULT_SAFETY_RECIPE_PATH),
        "sha256": _sha256_file(DEFAULT_SAFETY_RECIPE_PATH),
    }


def _write_safety_manifest_atomic(manifest_path: Path, manifest: dict) -> None:
    """Durably replace a manifest with a same-filesystem temporary file."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_manifest = yaml.safe_dump(manifest, sort_keys=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized_manifest)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, manifest_path)
        temporary_path = None
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    # The rename is the publication point. A later directory-sync failure cannot be rolled back
    # safely, so preserve commit semantics rather than reporting a false failed publication.
    try:
        directory_descriptor = os.open(manifest_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError:
        return


def _stage_safety_manifest_section(manifest: dict, section: str, values: dict) -> None:
    """Add one validated preparation record to an in-memory manifest generation."""
    _set_safety_manifest_recipe(manifest)
    manifest[section] = values


def _update_safety_manifest(manifest_path: Path, section: str, values: dict) -> None:
    """Atomically publish one scanner-asset record while preserving prior records."""
    manifest = _read_safety_manifest(manifest_path)
    _stage_safety_manifest_section(manifest, section, values)
    _write_safety_manifest_atomic(manifest_path, manifest)


def _record_safety_manifest_section(
    manifest_path: Path,
    section: str,
    values: dict,
    *,
    manifest: dict | None,
) -> None:
    """Stage a section for an outer transaction or atomically publish a direct call."""
    if manifest is None:
        _update_safety_manifest(manifest_path, section, values)
        return
    _stage_safety_manifest_section(manifest, section, values)


def _resolve_amrfinder_latest_database(requested_database_dir: Path) -> Path:
    """Resolve AMRFinder's documented ``latest`` symlink to a contained populated version directory."""
    requested_database_dir = Path(requested_database_dir)
    latest_database_dir = requested_database_dir / "latest"
    if not latest_database_dir.is_symlink():
        raise FileNotFoundError(f"AMRFinder database latest must be a symbolic link under {requested_database_dir}")
    try:
        pinned_database_dir = latest_database_dir.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"AMRFinder database latest link is broken under {requested_database_dir}") from error
    requested_database_root = requested_database_dir.resolve()
    try:
        pinned_database_dir.relative_to(requested_database_root)
    except ValueError as error:
        raise ValueError(
            f"AMRFinder latest database must remain contained under {requested_database_root}: {pinned_database_dir}"
        ) from error
    if pinned_database_dir.parent != requested_database_root:
        raise ValueError(
            "AMRFinder latest database must resolve to a direct version directory under "
            f"{requested_database_root}: {pinned_database_dir}"
        )
    if not pinned_database_dir.is_dir() or not any(pinned_database_dir.rglob("*")):
        raise FileNotFoundError(f"AMRFinder latest database is empty: {pinned_database_dir}")
    return pinned_database_dir


def prepare_amrfinder_plus(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    amrfinder_url: str = DEFAULT_AMRFINDER_URL,
    amrfinder_sha256: str | None = DEFAULT_AMRFINDER_SHA256,
    database_dir: Path | None = None,
    prerequisite_bin_dir: Path | None = None,
    safety_dir: Path | None = None,
    manifest_path: Path | None = None,
    manifest: dict | None = None,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Prepare a digest-pinned AMRFinderPlus release and resolved custom database."""
    if amrfinder_sha256 is None:
        raise ValueError("AMRFinder archive digest is required before extraction or execution")
    del insecure_downloads
    external_dir = Path(external_dir)
    safety_dir = Path(safety_dir) if safety_dir is not None else external_dir / "safety"
    archive_path = _download(
        amrfinder_url,
        safety_dir / "downloads" / f"{DEFAULT_AMRFINDER_RELEASE}.tar.gz",
        overwrite=overwrite,
        insecure=False,
        expected_sha256=amrfinder_sha256,
    )
    archive_sha256 = _verify_sha256(archive_path, amrfinder_sha256)
    extracted_dir = _extract_tar(
        archive_path,
        safety_dir / "tools" / f"{DEFAULT_AMRFINDER_RELEASE}-{archive_sha256[:16]}",
        # The archive digest authenticates only the archive. Never trust a pre-existing tree
        # named after that digest before executing any extracted AMRFinder component.
        overwrite=True,
    )
    target_bin_dir = Path(bin_dir) if bin_dir is not None else external_dir / "bin"
    amrfinder_path = _link_executable(
        _find_extracted_executable(extracted_dir, "amrfinder"), target_bin_dir / "amrfinder"
    )
    amrfinder_index_path = _link_executable(
        _find_extracted_executable(extracted_dir, "amrfinder_index"), target_bin_dir / "amrfinder_index"
    )
    amrfinder_update_path = _link_executable(
        _find_extracted_executable(extracted_dir, "amrfinder_update"), target_bin_dir / "amrfinder_update"
    )
    source_prerequisite_bin_dir = Path(prerequisite_bin_dir) if prerequisite_bin_dir is not None else target_bin_dir
    makeblastdb_source_path = source_prerequisite_bin_dir / "makeblastdb"
    hmmpress_source_path = source_prerequisite_bin_dir / "hmmpress"
    if not makeblastdb_source_path.exists():
        raise FileNotFoundError(f"AMRFinder update requires makeblastdb in {source_prerequisite_bin_dir}")
    if not hmmpress_source_path.exists():
        raise FileNotFoundError(f"AMRFinder update requires hmmpress in {source_prerequisite_bin_dir}")
    makeblastdb_path = _copy_executable(makeblastdb_source_path, target_bin_dir / "makeblastdb")
    hmmpress_path = _copy_executable(hmmpress_source_path, target_bin_dir / "hmmpress")
    blast_bin_dir = makeblastdb_path.parent.resolve()
    hmmer_bin_dir = hmmpress_path.parent.resolve()

    requested_database_dir = Path(database_dir) if database_dir is not None else safety_dir / "amrfinder" / "database"
    if overwrite and requested_database_dir.exists():
        shutil.rmtree(requested_database_dir)
    if requested_database_dir.exists() and not requested_database_dir.is_dir():
        raise ValueError(f"AMRFinder database root is not a directory: {requested_database_dir}")
    needs_database_update = not requested_database_dir.exists() or not any(requested_database_dir.iterdir())
    if not needs_database_update:
        _resolve_amrfinder_latest_database(requested_database_dir)
    if needs_database_update:
        requested_database_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(amrfinder_update_path),
                "-d",
                str(requested_database_dir),
                "--blast_bin",
                str(blast_bin_dir),
                "--hmmer_bin",
                str(hmmer_bin_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    pinned_database_dir = _resolve_amrfinder_latest_database(requested_database_dir)
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
    if not database_version:
        raise RuntimeError("AMRFinder reported no nonempty database version")
    if database_version != pinned_database_dir.name:
        raise RuntimeError(
            "AMRFinder reported database version does not match its resolved version directory: "
            f"{database_version} != {pinned_database_dir.name}"
        )

    selected_manifest_path = Path(manifest_path) if manifest_path is not None else safety_dir / "asset_manifest.yaml"
    _record_safety_manifest_section(
        selected_manifest_path,
        "amrfinder_plus",
        {
            "release": DEFAULT_AMRFINDER_RELEASE,
            "release_url": amrfinder_url,
            "archive_sha256": archive_sha256,
            "binary_path": str(amrfinder_path.resolve()),
            "binary_sha256": _sha256_file(amrfinder_path),
            "amrfinder_index_path": str(amrfinder_index_path.resolve()),
            "amrfinder_index_sha256": _sha256_file(amrfinder_index_path),
            "amrfinder_update_path": str(amrfinder_update_path.resolve()),
            "amrfinder_update_sha256": _sha256_file(amrfinder_update_path),
            "makeblastdb_path": str(makeblastdb_path.resolve()),
            "makeblastdb_sha256": _sha256_file(makeblastdb_path),
            "hmmpress_path": str(hmmpress_path.resolve()),
            "hmmpress_sha256": _sha256_file(hmmpress_path),
            "amrfinder_version": amrfinder_version,
            "database_path": str(pinned_database_dir),
            "database_version": database_version,
            "database_sha256": _sha256_path(pinned_database_dir),
            "citation": AMRFINDER_CITATION,
        },
        manifest=manifest,
    )
    return PreparedAsset("amrfinder_plus", amrfinder_path, f"{DEFAULT_AMRFINDER_RELEASE}: {database_version}")


def _uniprot_tsv_accessions(annotations_path: Path) -> set[str]:
    """Return the nonempty canonical accessions in a UniProt TSV stream result."""
    with annotations_path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None or "Entry" not in reader.fieldnames:
            raise ValueError(f"UniProt annotations TSV lacks the Entry column: {annotations_path}")
        accessions = {row.get("Entry", "").strip() for row in reader}
    accessions.discard("")
    if not accessions:
        raise ValueError(f"UniProt annotations TSV has an empty accession set: {annotations_path}")
    return accessions


def _uniprot_fasta_accessions(fasta_path: Path) -> set[str]:
    """Return canonical accessions from standard UniProt FASTA headers."""
    accessions = set()
    for line in fasta_path.read_text().splitlines():
        if not line.startswith(">"):
            continue
        identifier = line[1:].split(maxsplit=1)[0]
        parts = identifier.split("|")
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"UniProt FASTA header has no canonical accession: {line}")
        accessions.add(parts[1])
    if not accessions:
        raise ValueError(f"UniProt FASTA has an empty accession set: {fasta_path}")
    return accessions


def _validate_uniprot_accession_parity(annotations_path: Path, fasta_path: Path) -> None:
    """Reject TSV/FASTA pairs that do not describe exactly the same proteins."""
    annotations_accessions = _uniprot_tsv_accessions(annotations_path)
    fasta_accessions = _uniprot_fasta_accessions(fasta_path)
    if annotations_accessions != fasta_accessions:
        raise ValueError("UniProt TSV/FASTA accession sets differ")


def _validate_cached_toxin_file(record: object, path: Path, label: str) -> None:
    """Validate a cached file against the exact path and digest in its prior manifest record."""
    if not isinstance(record, dict):
        raise ValueError(f"Cached toxin manifest lacks a {label} record")
    if record.get("path") != str(path.resolve()):
        raise ValueError(f"Cached toxin {label} path does not match its manifest")
    expected_sha256 = record.get("sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(f"Cached toxin manifest lacks a {label} digest")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"Cached toxin {label} digest does not match its manifest")


def _validate_cached_toxin_reference(
    toxin_reference: object,
    *,
    annotations_url: str,
    fasta_url: str,
    annotations_path: Path,
    fasta_path: Path,
    diamond_database: Path,
) -> tuple[str, str | None, str]:
    """Validate a complete cached UniProt snapshot before reusing it."""
    if not isinstance(toxin_reference, dict):
        raise ValueError("Cached toxin files require a prior complete manifest record")
    expected_provenance = {
        "query": DEFAULT_UNIPROT_TOXIN_QUERY,
        "annotations_url": annotations_url,
        "fasta_url": fasta_url,
        "license": "CC BY 4.0",
        "attribution": UNIPROT_CC_BY_4_0_ATTRIBUTION,
    }
    for field, expected_value in expected_provenance.items():
        if toxin_reference.get(field) != expected_value:
            raise ValueError(f"Cached toxin {field} does not match its requested provenance")
    release = toxin_reference.get("uniprot_release")
    retrieved_at = toxin_reference.get("retrieved_at")
    if not isinstance(release, str) or not release:
        raise ValueError("Cached toxin manifest lacks a UniProt release")
    if not isinstance(retrieved_at, str) or not retrieved_at:
        raise ValueError("Cached toxin manifest lacks its original retrieval time")
    release_date = toxin_reference.get("uniprot_release_date")
    if release_date is not None and (not isinstance(release_date, str) or not release_date):
        raise ValueError("Cached toxin manifest has an invalid UniProt release date")
    files = toxin_reference.get("files")
    if not isinstance(files, dict):
        raise ValueError("Cached toxin manifest lacks file records")
    _validate_cached_toxin_file(files.get("annotations"), annotations_path, "annotations")
    _validate_cached_toxin_file(files.get("fasta"), fasta_path, "FASTA")
    _validate_cached_toxin_file(files.get("diamond_database"), diamond_database, "DIAMOND database")
    return release, release_date, retrieved_at


def _validate_uniprot_release_headers(
    annotation_headers: dict[str, str], fasta_headers: dict[str, str]
) -> tuple[str, str | None]:
    """Require matching release headers and matching dates when UniProt provides dates."""
    annotation_release = annotation_headers.get("x-uniprot-release")
    fasta_release = fasta_headers.get("x-uniprot-release")
    if not annotation_release or not fasta_release or annotation_release != fasta_release:
        raise ValueError("UniProt release headers must be present and match")
    annotation_date = annotation_headers.get("x-uniprot-release-date")
    fasta_date = fasta_headers.get("x-uniprot-release-date")
    if annotation_date or fasta_date:
        if not annotation_date or not fasta_date or annotation_date != fasta_date:
            raise ValueError("UniProt release date headers must be present and match")
    return annotation_release, annotation_date


def prepare_toxin_reference(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    diamond_bin: Path | None = None,
    annotations_url: str = DEFAULT_UNIPROT_TOXIN_ANNOTATIONS_URL,
    fasta_url: str = DEFAULT_UNIPROT_TOXIN_FASTA_URL,
    safety_dir: Path | None = None,
    manifest_path: Path | None = None,
    manifest: dict | None = None,
    existing_manifest: dict | None = None,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Build a coherent, provenance-validated DIAMOND toxin reference snapshot."""
    del insecure_downloads
    if annotations_url != DEFAULT_UNIPROT_TOXIN_ANNOTATIONS_URL or fasta_url != DEFAULT_UNIPROT_TOXIN_FASTA_URL:
        raise ValueError("Custom UniProt override URLs require explicit provenance and are not supported")
    external_dir = Path(external_dir)
    safety_dir = Path(safety_dir) if safety_dir is not None else external_dir / "safety"
    toxin_dir = safety_dir / "toxins"
    selected_manifest_path = Path(manifest_path) if manifest_path is not None else safety_dir / "asset_manifest.yaml"
    prior_manifest = (
        existing_manifest if existing_manifest is not None else _read_safety_manifest(selected_manifest_path)
    )
    prior_toxin_reference = prior_manifest.get("toxin_reference")
    annotations_path, annotation_headers = _download_with_headers(
        annotations_url,
        toxin_dir / "reviewed_toxins.tsv",
        overwrite=overwrite,
        insecure=False,
    )
    fasta_path, fasta_headers = _download_with_headers(
        fasta_url,
        toxin_dir / "reviewed_toxins.faa",
        overwrite=overwrite,
        insecure=False,
    )
    diamond_database = toxin_dir / "reviewed_toxins.dmnd"
    annotations_downloaded = bool(annotation_headers)
    fasta_downloaded = bool(fasta_headers)
    if annotations_downloaded != fasta_downloaded:
        raise ValueError("UniProt TSV and FASTA must be downloaded or reused as one coherent snapshot")
    if annotations_downloaded:
        uniprot_release, uniprot_release_date = _validate_uniprot_release_headers(annotation_headers, fasta_headers)
        retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    else:
        uniprot_release, uniprot_release_date, retrieved_at = _validate_cached_toxin_reference(
            prior_toxin_reference,
            annotations_url=annotations_url,
            fasta_url=fasta_url,
            annotations_path=annotations_path,
            fasta_path=fasta_path,
            diamond_database=diamond_database,
        )
    _validate_uniprot_accession_parity(annotations_path, fasta_path)

    # Fresh TSV/FASTA bytes require a DIAMOND index built from those same bytes, even
    # when a stale filename happens to exist from an earlier snapshot.
    if annotations_downloaded or overwrite or not diamond_database.exists():
        selected_diamond_bin = Path(diamond_bin) if diamond_bin is not None else external_dir / "bin" / "diamond"
        subprocess.run(
            [str(selected_diamond_bin), "makedb", "--in", str(fasta_path), "--db", str(diamond_database)],
            check=True,
        )
    if not diamond_database.exists():
        raise FileNotFoundError(f"DIAMOND did not create toxin database: {diamond_database}")

    toxin_manifest = {
        "query": DEFAULT_UNIPROT_TOXIN_QUERY,
        "annotations_url": annotations_url,
        "fasta_url": fasta_url,
        "retrieved_at": retrieved_at,
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
    }
    if uniprot_release_date is not None:
        toxin_manifest["uniprot_release_date"] = uniprot_release_date
    _record_safety_manifest_section(
        selected_manifest_path,
        "toxin_reference",
        toxin_manifest,
        manifest=manifest,
    )
    return PreparedAsset("toxin_reference", diamond_database, "reviewed UniProt toxin DIAMOND database")


def _complete_phrogs_sequence_database(sequence_database: Path) -> tuple[str, list[Path]]:
    """Validate and digest the complete MMseqs padded sequence/header database prefix."""
    sequence_database = Path(sequence_database)
    if not sequence_database.exists():
        raise FileNotFoundError(f"PHROGs searchable sequence database is required: {sequence_database}")
    if not sequence_database.is_file():
        raise FileNotFoundError(
            f"PHROGs searchable sequence database must be an MMseqs prefix file: {sequence_database}"
        )
    required_paths = (
        sequence_database,
        Path(f"{sequence_database}.index"),
        Path(f"{sequence_database}.dbtype"),
        Path(f"{sequence_database}_h"),
        Path(f"{sequence_database}_h.index"),
        Path(f"{sequence_database}_h.dbtype"),
        Path(f"{sequence_database}.lookup"),
    )
    missing_paths = [path for path in required_paths if not path.is_file() or path.stat().st_size == 0]
    if missing_paths:
        raise FileNotFoundError(
            "PHROGs searchable sequence database is not a complete MMseqs padded database; missing "
            + ", ".join(str(path) for path in missing_paths)
        )
    sidecar_paths = sorted(
        path for path in sequence_database.parent.glob(f"{sequence_database.name}*") if path.is_file()
    )
    digest = hashlib.sha256()
    for sidecar_path in sidecar_paths:
        digest.update(sidecar_path.name.encode())
        digest.update(b"\0")
        with sidecar_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest(), sidecar_paths


def _complete_phrogs_profile_database(profile_database: Path) -> tuple[str, list[Path]]:
    """Validate and digest the pinned Pharokka PHROGs MMseqs profile database."""
    profile_database = Path(profile_database)
    if profile_database.name != PHROGS_PROFILE_DATABASE_NAME:
        raise FileNotFoundError(
            "PHROGs safety profile database must use the official "
            f"{PHROGS_PROFILE_DATABASE_NAME} prefix: {profile_database}"
        )
    if not profile_database.is_file():
        raise FileNotFoundError(f"PHROGs safety profile database is required: {profile_database}")
    required_paths = (
        profile_database,
        Path(f"{profile_database}.index"),
        Path(f"{profile_database}.dbtype"),
        Path(f"{profile_database}_h"),
        Path(f"{profile_database}_h.index"),
        Path(f"{profile_database}_h.dbtype"),
        Path(f"{profile_database}.lookup"),
        Path(f"{profile_database}.source"),
        profile_database.parent / PHROGS_PROFILE_RELEASE_MARKER,
    )
    missing_paths = [path for path in required_paths if not path.is_file() or path.stat().st_size == 0]
    if missing_paths:
        raise FileNotFoundError(
            "PHROGs safety profile database is not a complete MMseqs profile database; missing "
            + ", ".join(str(path) for path in missing_paths)
        )
    sidecar_paths = sorted(
        path for path in profile_database.parent.glob(f"{profile_database.name}*") if path.is_file()
    )
    digest = hashlib.sha256()
    for sidecar_path in sidecar_paths:
        digest.update(sidecar_path.name.encode())
        digest.update(b"\0")
        with sidecar_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    _phrogs_profile_id_inventory(profile_database)
    return digest.hexdigest(), sidecar_paths


def _is_phrogs_profile_identifier(identifier: str) -> bool:
    """Return whether a PHROGs family identifier has the documented ``phrog_N`` form."""
    prefix = "phrog_"
    suffix = identifier.removeprefix(prefix)
    return identifier.startswith(prefix) and suffix.isascii() and suffix.isdecimal() and not suffix.startswith("0")


def _phrogs_profile_ids(profile_database: Path) -> set[str]:
    """Parse the MMseqs lookup and return its canonical unique PHROG query identifiers."""
    lookup_path = Path(f"{profile_database}.lookup")
    profile_ids: set[str] = set()
    internal_keys: set[str] = set()
    with lookup_path.open(encoding="utf-8", newline="") as source:
        for line_number, raw_line in enumerate(source, start=1):
            row = raw_line.rstrip("\r\n").split("\t")
            if len(row) != 3 or not row[0].isdecimal() or not row[2].isdecimal():
                raise ValueError(
                    "PHROGs profile lookup is malformed; expected numeric internal key, canonical "
                    f"PHROG ID, and numeric file/index field at {lookup_path}:{line_number}"
                )
            internal_key, profile_id, _file_index = row
            if not _is_phrogs_profile_identifier(profile_id):
                raise ValueError(
                    "PHROGs profile lookup has a noncanonical PHROG identifier at "
                    f"{lookup_path}:{line_number}: {profile_id}"
                )
            if internal_key in internal_keys:
                raise ValueError(
                    "PHROGs profile lookup has a duplicate internal key at "
                    f"{lookup_path}:{line_number}: {internal_key}"
                )
            if profile_id in profile_ids:
                raise ValueError(
                    "PHROGs profile lookup has a duplicate PHROG identifier at "
                    f"{lookup_path}:{line_number}: {profile_id}"
                )
            internal_keys.add(internal_key)
            profile_ids.add(profile_id)
    if not profile_ids:
        raise ValueError(f"PHROGs profile lookup is empty: {lookup_path}")
    return profile_ids


def _phrogs_profile_id_inventory(profile_database: Path) -> dict[str, int | str]:
    """Return a deterministic manifestable inventory for canonical PHROGs profile query identifiers."""
    profile_ids = _phrogs_profile_ids(profile_database)
    digest = hashlib.sha256()
    for profile_id in sorted(profile_ids):
        digest.update(profile_id.encode())
        digest.update(b"\n")
    return {"count": len(profile_ids), "sha256": digest.hexdigest()}


def _find_phrogs_profile_database(profile_root: Path) -> Path:
    """Locate exactly one complete official PHROGs profile database below an extracted root."""
    profile_root = Path(profile_root)
    if not profile_root.is_dir():
        raise FileNotFoundError(f"PHROGs profile database root is required: {profile_root}")
    candidates = sorted(path for path in profile_root.rglob(PHROGS_PROFILE_DATABASE_NAME) if path.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError(
            "PHROGs profile database root must contain exactly one official "
            f"{PHROGS_PROFILE_DATABASE_NAME} prefix: {profile_root}"
        )
    _complete_phrogs_profile_database(candidates[0])
    return candidates[0]


def _extract_verified_phrogs_safety_profile_archive(archive_path: Path, output_dir: Path) -> _VerifiedPhrogsProfile:
    """Cleanly extract the official, size- and MD5-verified Pharokka profile archive."""
    archive_path = Path(archive_path)
    _verify_file_size(archive_path, DEFAULT_PHROGS_SAFETY_PROFILE_SIZE)
    _verify_md5(archive_path, DEFAULT_PHROGS_SAFETY_PROFILE_MD5)
    extracted_dir = _extract_tar(archive_path, output_dir, overwrite=True)
    profile_database = _find_phrogs_profile_database(extracted_dir)
    database_sha256, _database_files = _complete_phrogs_profile_database(profile_database)
    profile_tree_files = _phrogs_profile_tree_files(profile_database)
    profile_root = profile_database.parent
    return _VerifiedPhrogsProfile(
        archive_path=archive_path,
        observed_archive_sha256=_sha256_file(archive_path),
        extracted_dir=extracted_dir,
        profile_database=profile_database,
        database_sha256=database_sha256,
        tree_sha256=_sha256_file_inventory(profile_root, profile_tree_files),
        profile_id_inventory=_phrogs_profile_id_inventory(profile_database),
        _authority=_VERIFIED_PHROGS_PROFILE_AUTHORITY,
    )


def _require_verified_phrogs_profile(profile: object) -> _VerifiedPhrogsProfile:
    """Return a private archive-extraction capability or fail closed at the public boundary."""
    if not isinstance(profile, _VerifiedPhrogsProfile) or profile._authority is not _VERIFIED_PHROGS_PROFILE_AUTHORITY:
        raise RuntimeError(
            "PHROGs safety metadata requires verified PHROGs profile preparation from the pinned archive"
        )
    return profile


def _phrogs_profile_tree_files(profile_database: Path) -> list[Path]:
    """Return only the complete pinned PHROGs profile inventory, never bundled unrelated databases."""
    profile_database = Path(profile_database)
    _database_sha256, database_files = _complete_phrogs_profile_database(profile_database)
    release_marker = profile_database.parent / PHROGS_PROFILE_RELEASE_MARKER
    return sorted({*database_files, release_marker})


def _sha256_file_inventory(root: Path, files: list[Path]) -> str:
    """Return a stable digest for a selected regular-file inventory relative to ``root``."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(Path(path) for path in files):
        if not path.is_file():
            raise FileNotFoundError(f"Required PHROGs profile inventory file is missing: {path}")
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _snapshot_phrogs_profile_database(profile_database: Path, safety_dir: Path) -> tuple[Path, Path]:
    """Copy only the complete PHROGs profile inventory into an unpublished safety generation."""
    profile_database = Path(profile_database)
    source_root = profile_database.parent
    database_sha256, database_files = _complete_phrogs_profile_database(profile_database)
    source_tree_files = _phrogs_profile_tree_files(profile_database)
    source_tree_sha256 = _sha256_file_inventory(source_root, source_tree_files)
    source_profile_ids = _phrogs_profile_id_inventory(profile_database)

    snapshot_root = Path(safety_dir) / "phrogs" / "profile_database"
    if snapshot_root.exists():
        raise FileExistsError(f"PHROGs safety profile snapshot already exists: {snapshot_root}")
    snapshot_root.mkdir(parents=True)
    for source_path in source_tree_files:
        destination_path = snapshot_root / source_path.relative_to(source_root)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    snapshot_database = snapshot_root / profile_database.relative_to(source_root)
    snapshot_sha256, snapshot_files = _complete_phrogs_profile_database(snapshot_database)
    if snapshot_sha256 != database_sha256:
        raise RuntimeError("PHROGs safety profile snapshot digest does not match its source")
    if [path.relative_to(snapshot_root) for path in snapshot_files] != [
        path.relative_to(source_root) for path in database_files
    ]:
        raise RuntimeError("PHROGs safety profile snapshot sidecar inventory does not match its source")
    snapshot_tree_files = _phrogs_profile_tree_files(snapshot_database)
    if [path.relative_to(snapshot_root) for path in snapshot_tree_files] != [
        path.relative_to(source_root) for path in source_tree_files
    ]:
        raise RuntimeError("PHROGs safety profile snapshot tree inventory does not match its source")
    if _sha256_file_inventory(snapshot_root, snapshot_tree_files) != source_tree_sha256:
        raise RuntimeError("PHROGs safety profile snapshot tree digest does not match its source")
    if _phrogs_profile_id_inventory(snapshot_database) != source_profile_ids:
        raise RuntimeError("PHROGs safety profile snapshot identity inventory does not match its source")
    return snapshot_database, snapshot_root


def _publish_phrogs_safety_profile_archive(
    archive_path: Path,
    external_dir: Path,
    *,
    _verified_profile: object | None = None,
) -> Path:
    """Atomically retain a verified Pharokka archive under its observed content digest."""
    archive_path = Path(archive_path)
    if _verified_profile is None:
        _verify_file_size(archive_path, DEFAULT_PHROGS_SAFETY_PROFILE_SIZE)
        _verify_md5(archive_path, DEFAULT_PHROGS_SAFETY_PROFILE_MD5)
        observed_sha256 = _sha256_file(archive_path)
    else:
        verified_profile = _require_verified_phrogs_profile(_verified_profile)
        if archive_path != verified_profile.archive_path:
            raise RuntimeError("PHROGs verified archive cache source does not match verified preparation")
        observed_sha256 = verified_profile.observed_archive_sha256
    cache_path = Path(external_dir) / "downloads" / "phrogs_safety_profile_archives" / f"{observed_sha256}.tar.gz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        temporary_path = cache_path.parent / f".{cache_path.name}.{uuid4().hex}.tmp"
        try:
            shutil.copyfile(archive_path, temporary_path)
            with temporary_path.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            if _sha256_file(temporary_path) != observed_sha256:
                raise RuntimeError("PHROGs safety profile archive cache digest changed during publication")
            os.replace(temporary_path, cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    if _sha256_file(cache_path) != observed_sha256:
        raise RuntimeError("PHROGs safety profile archive cache digest does not match its source")
    return cache_path


def _cached_phrogs_safety_profile_archive(previous_manifest: dict, external_dir: Path) -> Path:
    """Select a prior manifest's content-addressed archive cache, with a legacy verified-cache fallback."""
    try:
        record = previous_manifest["phrogs_v4"]["profile_database"]["provenance"]["verified_archive"]
        archive_path = Path(record["path"])
        observed_sha256 = str(record["sha256"])
    except (KeyError, TypeError):
        archive_path = Path(external_dir) / "downloads" / Path(DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
        observed_sha256 = ""
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise FileNotFoundError(f"PHROGs profile archive cache is required: {archive_path}")
    actual_sha256 = _sha256_file(archive_path)
    if observed_sha256 and actual_sha256 != observed_sha256:
        raise RuntimeError("PHROGs profile archive cache digest does not match the prior trusted manifest")
    return archive_path


def _publish_phrogs_legacy_profile_database(
    profile_database: Path,
    profile_root: Path,
    external_dir: Path,
) -> Path:
    """Publish an already-validated profile snapshot at the shared legacy PHROGs path."""
    profile_database = Path(profile_database)
    profile_root = Path(profile_root)
    database_sha256, database_files = _complete_phrogs_profile_database(profile_database)
    source_tree_files = _phrogs_profile_tree_files(profile_database)
    source_tree_sha256 = _sha256_file_inventory(profile_root, source_tree_files)
    source_profile_ids = _phrogs_profile_id_inventory(profile_database)
    relative_profile_path = profile_database.relative_to(profile_root)

    legacy_parent = Path(external_dir) / "phrogs"
    legacy_parent.mkdir(parents=True, exist_ok=True)
    legacy_root = legacy_parent / "phrogs_mmseqs_db"
    staging_root = Path(tempfile.mkdtemp(prefix=f".{legacy_root.name}.", dir=legacy_parent))
    backup_root: Path | None = None
    try:
        for source_path in source_tree_files:
            destination_path = staging_root / source_path.relative_to(profile_root)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
        staged_profile = staging_root / relative_profile_path
        staged_sha256, staged_files = _complete_phrogs_profile_database(staged_profile)
        if staged_sha256 != database_sha256:
            raise RuntimeError("Staged legacy PHROGs profile digest does not match its safety snapshot")
        if [path.relative_to(staging_root) for path in staged_files] != [
            path.relative_to(profile_root) for path in database_files
        ]:
            raise RuntimeError("Staged legacy PHROGs profile sidecar inventory does not match its safety snapshot")
        staged_tree_files = _phrogs_profile_tree_files(staged_profile)
        if [path.relative_to(staging_root) for path in staged_tree_files] != [
            path.relative_to(profile_root) for path in source_tree_files
        ]:
            raise RuntimeError("Staged legacy PHROGs profile tree inventory does not match its safety snapshot")
        if _sha256_file_inventory(staging_root, staged_tree_files) != source_tree_sha256:
            raise RuntimeError("Staged legacy PHROGs profile tree digest does not match its safety snapshot")
        if _phrogs_profile_id_inventory(staged_profile) != source_profile_ids:
            raise RuntimeError("Staged legacy PHROGs profile identity inventory does not match its safety snapshot")

        if legacy_root.exists() or legacy_root.is_symlink():
            backup_root = legacy_parent / f".{legacy_root.name}.previous-{uuid4().hex}"
            os.replace(legacy_root, backup_root)
        os.replace(staging_root, legacy_root)
        published_profile = legacy_root / relative_profile_path
        published_sha256, published_files = _complete_phrogs_profile_database(published_profile)
        if published_sha256 != database_sha256:
            raise RuntimeError("Published legacy PHROGs profile digest does not match its safety snapshot")
        if [path.relative_to(legacy_root) for path in published_files] != [
            path.relative_to(profile_root) for path in database_files
        ]:
            raise RuntimeError("Published legacy PHROGs profile sidecar inventory does not match its safety snapshot")
        published_tree_files = _phrogs_profile_tree_files(published_profile)
        if [path.relative_to(legacy_root) for path in published_tree_files] != [
            path.relative_to(profile_root) for path in source_tree_files
        ]:
            raise RuntimeError("Published legacy PHROGs profile tree inventory does not match its safety snapshot")
        if _sha256_file_inventory(legacy_root, published_tree_files) != source_tree_sha256:
            raise RuntimeError("Published legacy PHROGs profile tree digest does not match its safety snapshot")
        if _phrogs_profile_id_inventory(published_profile) != source_profile_ids:
            raise RuntimeError("Published legacy PHROGs profile identity inventory does not match its safety snapshot")
        if backup_root is not None:
            shutil.rmtree(backup_root)
            backup_root = None
        return published_profile
    except Exception:
        if backup_root is not None and backup_root.exists():
            if legacy_root.exists() or legacy_root.is_symlink():
                if legacy_root.is_dir() and not legacy_root.is_symlink():
                    shutil.rmtree(legacy_root)
                else:
                    legacy_root.unlink()
            os.replace(backup_root, legacy_root)
            backup_root = None
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        if backup_root is not None and backup_root.exists():
            shutil.rmtree(backup_root, ignore_errors=True)


def _snapshot_phrogs_safety_assets(
    annotation_path: Path,
    sequence_database: Path,
    safety_dir: Path,
) -> tuple[Path, Path]:
    """Copy a complete PHROGs source/database pair into an unpublished safety generation."""
    annotation_path = Path(annotation_path)
    sequence_database = Path(sequence_database)
    database_sha256, database_files = _complete_phrogs_sequence_database(sequence_database)
    if not annotation_path.is_file() or annotation_path.stat().st_size == 0:
        raise FileNotFoundError(f"PHROGs v4 annotation table is required: {annotation_path}")

    snapshot_dir = Path(safety_dir) / "phrogs" / "snapshot"
    if snapshot_dir.exists():
        raise FileExistsError(f"PHROGs safety snapshot already exists: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True)
    snapshot_annotation_path = snapshot_dir / annotation_path.name
    shutil.copy2(annotation_path, snapshot_annotation_path)
    for database_file in database_files:
        shutil.copy2(database_file, snapshot_dir / database_file.name)

    snapshot_database_path = snapshot_dir / sequence_database.name
    snapshot_database_sha256, _ = _complete_phrogs_sequence_database(snapshot_database_path)
    if snapshot_database_sha256 != database_sha256:
        raise RuntimeError("PHROGs safety snapshot database digest does not match its source")
    if _sha256_file(snapshot_annotation_path) != _sha256_file(annotation_path):
        raise RuntimeError("PHROGs safety snapshot annotation digest does not match its source")
    return snapshot_annotation_path, snapshot_database_path


def _publish_phrogs_legacy_assets(
    annotation_path: Path,
    sequence_database: Path,
    external_dir: Path,
) -> tuple[Path, Path]:
    """Publish a verified immutable PHROGs snapshot at the legacy external path.

    This compatibility copy is deliberately made only after the safety manifest
    points at the immutable generation.  A failed safety generation therefore
    cannot alter a shared PHROGs source or padded database trusted by an older
    manifest.
    """
    annotation_path = Path(annotation_path)
    sequence_database = Path(sequence_database)
    annotation_sha256 = _sha256_file(annotation_path)
    database_sha256, database_files = _complete_phrogs_sequence_database(sequence_database)
    expected_database_names = [path.name for path in database_files]

    legacy_dir = Path(external_dir) / "phrogs"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".phrogs-safety-publish-", dir=legacy_dir))
    try:
        staged_annotation_path = staging_dir / annotation_path.name
        shutil.copy2(annotation_path, staged_annotation_path)
        for database_file in database_files:
            shutil.copy2(database_file, staging_dir / database_file.name)

        staged_database_path = staging_dir / sequence_database.name
        staged_database_sha256, staged_database_files = _complete_phrogs_sequence_database(staged_database_path)
        if _sha256_file(staged_annotation_path) != annotation_sha256:
            raise RuntimeError("Staged legacy PHROGs annotation digest does not match its safety snapshot")
        if staged_database_sha256 != database_sha256:
            raise RuntimeError("Staged legacy PHROGs database digest does not match its safety snapshot")
        if [path.name for path in staged_database_files] != expected_database_names:
            raise RuntimeError("Staged legacy PHROGs database file inventory does not match its safety snapshot")

        for staged_path in (staged_annotation_path, *staged_database_files):
            os.replace(staged_path, legacy_dir / staged_path.name)
        for stale_path in legacy_dir.glob(f"{sequence_database.name}*"):
            if stale_path.name not in expected_database_names and (stale_path.is_file() or stale_path.is_symlink()):
                stale_path.unlink()

        legacy_annotation_path = legacy_dir / annotation_path.name
        legacy_database_path = legacy_dir / sequence_database.name
        if _sha256_file(legacy_annotation_path) != annotation_sha256:
            raise RuntimeError("Published legacy PHROGs annotation digest does not match its safety snapshot")
        legacy_database_sha256, legacy_database_files = _complete_phrogs_sequence_database(legacy_database_path)
        if legacy_database_sha256 != database_sha256:
            raise RuntimeError("Published legacy PHROGs database digest does not match its safety snapshot")
        if [path.name for path in legacy_database_files] != expected_database_names:
            raise RuntimeError("Published legacy PHROGs database file inventory does not match its safety snapshot")
        return legacy_annotation_path, legacy_database_path
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _write_phrogs_lookup(lookup_path: Path, rows: list[list[str]]) -> None:
    """Atomically replace the generated PHROGs lookup only after all rows validate."""
    lookup_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=lookup_path.parent,
            prefix=f".{lookup_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.writer(temporary_file, delimiter="\t", lineterminator="\n")
            writer.writerow(["phrog", "annot", "category", "confidence", "matched_term"])
            writer.writerows(rows)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, lookup_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _prepare_phrogs_safety_metadata(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    manifest_path: Path | None = None,
    manifest: dict | None = None,
    annotation_sha256: str | None = DEFAULT_PHROGS_ANNOTATION_SHA256,
    annotation_path: Path | None = None,
    sequence_database: Path | None = None,
    profile_database: Path | None = None,
    profile_archive_path: Path | None = None,
    profile_source_url: str = DEFAULT_PHROGS_SAFETY_PROFILE_URL,
    profile_archive_observed_sha256: str | None = None,
    profile_retrieved_at: str | None = None,
    profile_release: str = PHROGS_PROFILE_RELEASE,
    profile_archive_published_md5: str = DEFAULT_PHROGS_SAFETY_PROFILE_MD5,
    profile_archive_published_size: int = DEFAULT_PHROGS_SAFETY_PROFILE_SIZE,
    profile_doi: str = DEFAULT_PHROGS_SAFETY_PROFILE_DOI,
    profile_license: str = DEFAULT_PHROGS_SAFETY_PROFILE_LICENSE,
    profile_minimum_mmseqs_version: str = PHROGS_PROFILE_MIN_MMSEQS_VERSION,
    profile_built_with_mmseqs_version: str = PHROGS_PROFILE_BUILDER_MMSEQS_VERSION,
    profile_dataset_release: str = PHROGS_PROFILE_DATASET_RELEASE,
    safety_dir: Path | None = None,
) -> PreparedAsset:
    """Build a PHROGs lookup from inputs already authenticated by safety preparation."""
    if annotation_sha256 is None:
        raise ValueError("PHROGs annotation digest is required before using its metadata")
    external_dir = Path(external_dir)
    selected_annotation_path = (
        Path(annotation_path) if annotation_path is not None else external_dir / "phrogs" / "phrog_annot_v4.tsv"
    )
    if not selected_annotation_path.exists():
        raise FileNotFoundError(f"PHROGs v4 annotation table is required: {selected_annotation_path}")
    source_sha256 = _verify_sha256(selected_annotation_path, annotation_sha256)
    selected_sequence_database = (
        Path(sequence_database) if sequence_database is not None else external_dir / "phrogs" / "phrogs_gpu_seq_db_pad"
    )
    sequence_database_sha256, sequence_database_files = _complete_phrogs_sequence_database(selected_sequence_database)
    if profile_database is None:
        raise FileNotFoundError("PHROGs safety profile_database is required for identity-bearing lysogeny search")
    selected_profile_database = Path(profile_database)
    profile_database_sha256, profile_database_files = _complete_phrogs_profile_database(selected_profile_database)
    profile_root = selected_profile_database.parent
    profile_tree_files = _phrogs_profile_tree_files(selected_profile_database)
    profile_id_inventory = _phrogs_profile_id_inventory(selected_profile_database)
    profile_ids = _phrogs_profile_ids(selected_profile_database)
    if not profile_archive_observed_sha256:
        raise ValueError("PHROGs safety profile archive observed SHA-256 is required for provenance")
    normalized_profile_archive_sha256 = profile_archive_observed_sha256.removeprefix("sha256:").lower()
    if len(normalized_profile_archive_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_profile_archive_sha256
    ):
        raise ValueError("PHROGs safety profile archive observed SHA-256 must be a digest")
    if not profile_retrieved_at:
        raise ValueError("PHROGs safety profile retrieval evidence is required for provenance")
    if profile_source_url != DEFAULT_PHROGS_SAFETY_PROFILE_URL:
        raise ValueError("PHROGs safety profile source URL must be the pinned Pharokka v1.8.0 release")
    if profile_release != PHROGS_PROFILE_RELEASE:
        raise ValueError("PHROGs safety profile release must be the pinned Pharokka v1.8.0 release")
    if profile_archive_published_md5.lower() != DEFAULT_PHROGS_SAFETY_PROFILE_MD5:
        raise ValueError("PHROGs safety profile published MD5 must match the pinned Pharokka release")
    if profile_archive_published_size != DEFAULT_PHROGS_SAFETY_PROFILE_SIZE:
        raise ValueError("PHROGs safety profile published archive size must match the pinned Pharokka release")
    if profile_doi != DEFAULT_PHROGS_SAFETY_PROFILE_DOI:
        raise ValueError("PHROGs safety profile DOI must match the pinned Pharokka release")
    if profile_license != DEFAULT_PHROGS_SAFETY_PROFILE_LICENSE:
        raise ValueError("PHROGs safety profile license must match the pinned Pharokka release")
    if profile_minimum_mmseqs_version != PHROGS_PROFILE_MIN_MMSEQS_VERSION:
        raise ValueError("PHROGs safety profile MMseqs minimum version must match the pinned profile format")
    if profile_built_with_mmseqs_version != PHROGS_PROFILE_BUILDER_MMSEQS_VERSION:
        raise ValueError("PHROGs safety profile MMseqs builder version must match the pinned profile format")
    if profile_dataset_release != PHROGS_PROFILE_DATASET_RELEASE:
        raise ValueError("PHROGs safety profile dataset release must be PHROGs v4")

    lookup_rows = []
    selected_phrogs = set()
    with selected_annotation_path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required_columns = {"phrog", "color", "annot", "category"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"PHROGs v4 table must contain {sorted(required_columns)}: {selected_annotation_path}")
        for row in reader:
            category = row["category"].strip()
            if category.casefold() != PHROGS_INTEGRATION_EXCISION_CATEGORY.casefold():
                continue
            phrog = row["phrog"].strip()
            if not phrog:
                raise ValueError(f"PHROGs v4 table has an empty PHROG identifier: {selected_annotation_path}")
            if not _is_phrogs_profile_identifier(phrog):
                raise ValueError(
                    f"PHROGs v4 table has a noncanonical PHROG identifier that cannot join the profile search: {phrog}"
                )
            if phrog in selected_phrogs:
                raise ValueError(f"PHROGs lookup has duplicate PHROG identifier: {phrog}")
            selected_phrogs.add(phrog)
            annotation = row["annot"].strip()
            normalized_annotation = annotation.casefold()
            high_confidence_term = next(
                (term for term in PHROGS_HIGH_CONFIDENCE_TERMS if term in normalized_annotation), None
            )
            review_term = next((term for term in PHROGS_REVIEW_TERMS if term in normalized_annotation), None)
            confidence = "high_confidence" if high_confidence_term is not None else "review"
            matched_term = high_confidence_term or review_term or "integration and excision category"
            lookup_rows.append([phrog, annotation, category, confidence, matched_term])
    if not lookup_rows:
        raise ValueError("PHROGs integration and excision lookup is empty")
    missing_profile_ids = sorted(selected_phrogs - profile_ids)
    if missing_profile_ids:
        raise ValueError(
            "PHROGs integration/excision lookup contains IDs absent from the verified profile database: "
            + ", ".join(missing_profile_ids)
        )
    if not profile_ids - selected_phrogs:
        raise ValueError(
            "PHROGs safety profile must contain families beyond the pinned safety lookup; "
            "a subset-only profile cannot represent the full PHROGs v4 search scope"
        )
    normalized_declared_sha256 = annotation_sha256.removeprefix("sha256:").lower()
    if normalized_declared_sha256 == DEFAULT_PHROGS_ANNOTATION_SHA256:
        high_confidence_count = sum(row[3] == "high_confidence" for row in lookup_rows)
        review_count = sum(row[3] == "review" for row in lookup_rows)
        if (len(lookup_rows), high_confidence_count, review_count) != (109, 57, 52):
            raise ValueError(
                "Pinned PHROGs v4 confidence lookup violated expected bounds "
                f"(109 total, 57 high, 52 review): {(len(lookup_rows), high_confidence_count, review_count)}"
            )

    safety_dir = Path(safety_dir) if safety_dir is not None else external_dir / "safety"
    lookup_path = safety_dir / "phrogs" / "phrogs_integration_excision_v4.tsv"
    _write_phrogs_lookup(lookup_path, lookup_rows)
    selected_manifest_path = Path(manifest_path) if manifest_path is not None else safety_dir / "asset_manifest.yaml"
    _record_safety_manifest_section(
        selected_manifest_path,
        "phrogs_v4",
        {
            "annotation_url": DEFAULT_PHROGS_ANNOTATION_URL,
            "annotation_sha256": normalized_declared_sha256,
            "source_path": str(selected_annotation_path.resolve()),
            "source_sha256": source_sha256,
            "category": PHROGS_INTEGRATION_EXCISION_CATEGORY,
            "high_confidence_terms": list(PHROGS_HIGH_CONFIDENCE_TERMS),
            "review_terms": list(PHROGS_REVIEW_TERMS),
            "lookup_path": str(lookup_path.resolve()),
            "lookup_sha256": _sha256_file(lookup_path),
            "lookup_counts": {
                "total": len(lookup_rows),
                "high_confidence": sum(row[3] == "high_confidence" for row in lookup_rows),
                "review": sum(row[3] == "review" for row in lookup_rows),
            },
            "sequence_database": {
                "path": str(selected_sequence_database.resolve()),
                "role": "complete PHROGs v4 raw padded sequence database for Arc/QC compatibility only",
                "sha256": sequence_database_sha256,
                "files": [str(path.resolve()) for path in sequence_database_files],
            },
            "profile_database": {
                "path": str(selected_profile_database.resolve()),
                "role": "complete PHROGs v4 MMseqs profile database for identity-bearing lysogeny search",
                "sha256": profile_database_sha256,
                "files": [str(path.resolve()) for path in profile_database_files],
                "extracted_tree": {
                    "path": str(profile_root.resolve()),
                    "sha256": _sha256_file_inventory(profile_root, profile_tree_files),
                    "files": [str(path.resolve()) for path in profile_tree_files],
                },
                "search_orientation": PHROGS_PROFILE_SEARCH_ORIENTATION,
                "search_profile_scope": PHROGS_PROFILE_SEARCH_SCOPE,
                "lookup_join_policy": PHROGS_PROFILE_LOOKUP_JOIN_POLICY,
                "output_fields": list(PHROGS_PROFILE_OUTPUT_FIELDS),
                "units": PHROGS_PROFILE_OUTPUT_UNITS,
                "query_id_pattern": PHROGS_PROFILE_QUERY_ID_PATTERN,
                "query_ids_join_lookup": True,
                "profile_id_inventory": profile_id_inventory,
                "provenance": {
                    "source_url": profile_source_url,
                    "archive_observed_sha256": normalized_profile_archive_sha256,
                    "archive_published_sha256": None,
                    "archive_published_md5": profile_archive_published_md5,
                    "archive_published_size": profile_archive_published_size,
                    "retrieved_at": profile_retrieved_at,
                    "release": profile_release,
                    "dataset_release": profile_dataset_release,
                    "doi": profile_doi,
                    "license": profile_license,
                    "citation": PHROGS_PROFILE_SOURCE_CITATION,
                    "minimum_mmseqs_version": profile_minimum_mmseqs_version,
                    "built_with_mmseqs_version": profile_built_with_mmseqs_version,
                    **(
                        {
                            "verified_archive": {
                                "path": str(Path(profile_archive_path).resolve()),
                                "sha256": normalized_profile_archive_sha256,
                            }
                        }
                        if profile_archive_path is not None
                        else {}
                    ),
                },
            },
            "citation": PHROGS_CITATION,
            "use_terms": PHROGS_USE_TERMS,
        },
        manifest=manifest,
    )
    return PreparedAsset("phrogs_safety_metadata", lookup_path, "PHROGs v4 integration and excision lookup table")


def _prepare_verified_phrogs_safety_metadata(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    _verified_profile: object | None = None,
    **kwargs: object,
) -> PreparedAsset:
    """Build staged PHROGs metadata only from a private verified archive-extraction capability."""
    verified_profile = _require_verified_phrogs_profile(_verified_profile)
    supplied_profile_database = kwargs.pop("profile_database", None)
    selected_profile_database = (
        verified_profile.profile_database if supplied_profile_database is None else Path(supplied_profile_database)
    )
    selected_database_sha256, _selected_database_files = _complete_phrogs_profile_database(selected_profile_database)
    selected_tree_sha256 = _sha256_file_inventory(
        selected_profile_database.parent,
        _phrogs_profile_tree_files(selected_profile_database),
    )
    if (
        selected_database_sha256 != verified_profile.database_sha256
        or selected_tree_sha256 != verified_profile.tree_sha256
        or _phrogs_profile_id_inventory(selected_profile_database) != verified_profile.profile_id_inventory
    ):
        raise RuntimeError("PHROGs safety metadata profile database does not match verified PHROGs preparation")
    supplied_archive_sha256 = kwargs.pop("profile_archive_observed_sha256", None)
    if (
        supplied_archive_sha256 is not None
        and str(supplied_archive_sha256).removeprefix("sha256:").lower() != verified_profile.observed_archive_sha256
    ):
        raise RuntimeError("PHROGs safety metadata archive digest does not match verified PHROGs preparation")
    return _prepare_phrogs_safety_metadata(
        external_dir,
        profile_database=selected_profile_database,
        profile_archive_path=verified_profile.archive_path,
        profile_archive_observed_sha256=verified_profile.observed_archive_sha256,
        **kwargs,
    )


def prepare_phrogs_safety_metadata(*_args: object, **_kwargs: object) -> PreparedAsset:
    """Reject direct PHROGs metadata publication; only the safety orchestrator may publish it."""
    raise RuntimeError(
        "prepare_phrogs_safety_metadata is not a public publishing API; use prepare_external_assets with --with-safety"
    )


def prepare_phrogs_annotation(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    annotation_url: str = DEFAULT_PHROGS_ANNOTATION_URL,
    annotation_sha256: str | None = DEFAULT_PHROGS_ANNOTATION_SHA256,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Download the digest-pinned PHROGs v4 annotation table."""
    if annotation_sha256 is None:
        raise ValueError("PHROGs annotation digest is required before download")
    annotation_path = _download(
        annotation_url,
        Path(external_dir) / "phrogs" / "phrog_annot_v4.tsv",
        overwrite=overwrite,
        insecure=insecure_downloads,
        expected_sha256=annotation_sha256,
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


def prepare_phrogs_safety_profile_db(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    overwrite: bool = False,
) -> PreparedAsset:
    """Prepare the versioned Pharokka PHROGs profile asset required for identity-bearing safety search."""
    external_dir = Path(external_dir)
    archive_path = _download(
        DEFAULT_PHROGS_SAFETY_PROFILE_URL,
        external_dir / "downloads" / Path(DEFAULT_PHROGS_SAFETY_PROFILE_URL).name,
        overwrite=overwrite,
        insecure=False,
    )
    verified_profile = _extract_verified_phrogs_safety_profile_archive(
        archive_path,
        external_dir / "phrogs" / "phrogs_mmseqs_db",
    )
    return PreparedAsset(
        "phrogs_safety_profile_db",
        verified_profile.extracted_dir,
        f"{DEFAULT_PHROGS_SAFETY_PROFILE_RELEASE} from {DEFAULT_PHROGS_SAFETY_PROFILE_URL}",
    )


def prepare_phrogs_gpu_sequence_db(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
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
    selected_bin_dir = Path(bin_dir) if bin_dir is not None else external_dir / "bin"
    mmseqs_bin = selected_bin_dir / "mmseqs"
    mmseqs_cmd = str(mmseqs_bin) if mmseqs_bin.exists() else "mmseqs"
    subprocess.run([mmseqs_cmd, "createdb", str(combined_fasta), str(seq_db)], check=True)
    subprocess.run(
        [mmseqs_cmd, "makepaddedseqdb", str(seq_db), str(padded_db), "--write-lookup", "1"],
        check=True,
    )
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


def _validate_recorded_asset_digest(record: dict, path_field: str, digest_field: str, label: str) -> None:
    """Fail closed when a staged manifest asset path is missing or changed before publication."""
    path_value = record.get(path_field)
    expected_sha256 = record.get(digest_field)
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"Safety manifest {label} lacks required path {path_field}")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise RuntimeError(f"Safety manifest {label} lacks required digest {digest_field}")
    asset_path = Path(path_value)
    if not asset_path.exists():
        raise RuntimeError(f"Safety manifest {label} path no longer exists: {asset_path}")
    observed_sha256 = _sha256_path(asset_path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"Safety manifest {label} digest does not match its staged path: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )


def _validate_phrogs_profile_archive_lineage(profile_database: dict) -> None:
    """Prove a staged profile snapshot came from a clean extraction of its pinned archive cache."""
    provenance = profile_database["provenance"]
    verified_archive = provenance.get("verified_archive")
    archive_path = Path(verified_archive["path"])
    observed_archive_sha256 = provenance["archive_observed_sha256"]
    if _sha256_file(archive_path) != observed_archive_sha256:
        raise RuntimeError("Safety manifest PHROGs verified archive cache digest does not match provenance")
    if verified_archive["sha256"] != observed_archive_sha256:
        raise RuntimeError("Safety manifest PHROGs verified archive cache does not match provenance")
    _verify_file_size(archive_path, DEFAULT_PHROGS_SAFETY_PROFILE_SIZE)
    _verify_md5(archive_path, DEFAULT_PHROGS_SAFETY_PROFILE_MD5)

    profile_path = Path(profile_database["path"])
    extraction_parent = Path(tempfile.mkdtemp(prefix=".phrogs-profile-lineage-", dir=profile_path.parent))
    try:
        verified_profile = _extract_verified_phrogs_safety_profile_archive(
            archive_path,
            extraction_parent / "extracted",
        )
        extracted_tree = profile_database["extracted_tree"]
        if verified_profile.database_sha256 != profile_database["sha256"]:
            raise RuntimeError(
                "Safety manifest PHROGs profile database does not match its clean verified archive extraction"
            )
        if verified_profile.tree_sha256 != extracted_tree["sha256"]:
            raise RuntimeError(
                "Safety manifest PHROGs profile tree does not match its clean verified archive extraction"
            )
        if verified_profile.profile_id_inventory != profile_database["profile_id_inventory"]:
            raise RuntimeError("Safety manifest PHROGs profile IDs do not match its clean verified archive extraction")
    finally:
        shutil.rmtree(extraction_parent, ignore_errors=True)


def _validate_staged_safety_manifest(manifest: dict, *, verify_asset_paths: bool = False) -> None:
    """Ensure all scanner sections are present before publishing a new trusted generation."""
    required_fields = {
        "amrfinder_plus": (
            "archive_sha256",
            "binary_path",
            "binary_sha256",
            "amrfinder_index_path",
            "amrfinder_index_sha256",
            "amrfinder_update_path",
            "amrfinder_update_sha256",
            "makeblastdb_path",
            "makeblastdb_sha256",
            "hmmpress_path",
            "hmmpress_sha256",
            "database_path",
            "database_version",
            "database_sha256",
        ),
        "toxin_reference": (
            "query",
            "annotations_url",
            "fasta_url",
            "retrieved_at",
            "uniprot_release",
            "files",
        ),
        "phrogs_v4": ("source_sha256", "lookup_sha256", "sequence_database", "profile_database"),
    }
    for section, fields in required_fields.items():
        record = manifest.get(section)
        if not isinstance(record, dict):
            raise RuntimeError(f"Safety manifest generation is incomplete: missing {section}")
        for field in fields:
            value = record.get(field)
            if value is None or value == "" or value == {}:
                raise RuntimeError(f"Safety manifest {section} lacks required field {field}")
    toxin_files = manifest["toxin_reference"]["files"]
    if not isinstance(toxin_files, dict):
        raise RuntimeError("Safety manifest toxin_reference lacks required field files")
    for file_role in ("annotations", "fasta", "diamond_database"):
        file_record = toxin_files.get(file_role)
        if not isinstance(file_record, dict) or not file_record.get("path") or not file_record.get("sha256"):
            raise RuntimeError(f"Safety manifest toxin_reference lacks required fields files.{file_role}")
    sequence_database = manifest["phrogs_v4"]["sequence_database"]
    if (
        not isinstance(sequence_database, dict)
        or not sequence_database.get("path")
        or not sequence_database.get("sha256")
    ):
        raise RuntimeError("Safety manifest phrogs_v4 lacks required fields sequence_database.path/sha256")
    profile_database = manifest["phrogs_v4"]["profile_database"]
    if not isinstance(profile_database, dict):
        raise RuntimeError("Safety manifest phrogs_v4 lacks required field profile_database")
    for field in (
        "path",
        "sha256",
        "files",
        "extracted_tree",
        "search_orientation",
        "search_profile_scope",
        "lookup_join_policy",
        "output_fields",
        "units",
        "query_id_pattern",
        "query_ids_join_lookup",
        "profile_id_inventory",
        "provenance",
    ):
        value = profile_database.get(field)
        if value is None or value == "" or value == {} or value == []:
            raise RuntimeError(f"Safety manifest phrogs_v4.profile_database lacks required field {field}")
    if profile_database["search_orientation"] != PHROGS_PROFILE_SEARCH_ORIENTATION:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported search_orientation")
    if profile_database["search_profile_scope"] != PHROGS_PROFILE_SEARCH_SCOPE:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported search_profile_scope")
    if profile_database["lookup_join_policy"] != PHROGS_PROFILE_LOOKUP_JOIN_POLICY:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported lookup_join_policy")
    if profile_database["output_fields"] != list(PHROGS_PROFILE_OUTPUT_FIELDS):
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has unsupported output_fields")
    if profile_database["units"] != PHROGS_PROFILE_OUTPUT_UNITS:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has unsupported output units")
    if profile_database["query_id_pattern"] != PHROGS_PROFILE_QUERY_ID_PATTERN:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported query_id_pattern")
    if profile_database["query_ids_join_lookup"] is not True:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database does not guarantee lookup-joinable query IDs")
    profile_id_inventory = profile_database["profile_id_inventory"]
    if (
        not isinstance(profile_id_inventory, dict)
        or not isinstance(profile_id_inventory.get("count"), int)
        or profile_id_inventory["count"] <= 0
        or not isinstance(profile_id_inventory.get("sha256"), str)
        or len(profile_id_inventory["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in profile_id_inventory["sha256"])
    ):
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an invalid profile_id_inventory")
    provenance = profile_database["provenance"]
    if not isinstance(provenance, dict):
        raise RuntimeError("Safety manifest phrogs_v4.profile_database lacks provenance")
    for field in (
        "source_url",
        "archive_observed_sha256",
        "archive_published_md5",
        "archive_published_size",
        "retrieved_at",
        "release",
        "dataset_release",
        "doi",
        "license",
        "citation",
        "minimum_mmseqs_version",
        "built_with_mmseqs_version",
    ):
        if not provenance.get(field):
            raise RuntimeError(f"Safety manifest phrogs_v4.profile_database provenance lacks {field}")
    if provenance.get("archive_published_sha256") is not None:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database must not invent an archive_published_sha256")
    observed_archive_sha256 = provenance["archive_observed_sha256"]
    if (
        not isinstance(observed_archive_sha256, str)
        or len(observed_archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in observed_archive_sha256)
    ):
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an invalid observed archive SHA-256")
    if provenance["source_url"] != DEFAULT_PHROGS_SAFETY_PROFILE_URL:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unpinned profile source URL")
    if provenance["archive_published_md5"] != DEFAULT_PHROGS_SAFETY_PROFILE_MD5:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unpinned profile archive MD5")
    if provenance["archive_published_size"] != DEFAULT_PHROGS_SAFETY_PROFILE_SIZE:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unpinned profile archive size")
    if provenance["release"] != PHROGS_PROFILE_RELEASE:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile release")
    if provenance["dataset_release"] != PHROGS_PROFILE_DATASET_RELEASE:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported PHROGs dataset release")
    if provenance["doi"] != DEFAULT_PHROGS_SAFETY_PROFILE_DOI:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile DOI")
    if provenance["license"] != DEFAULT_PHROGS_SAFETY_PROFILE_LICENSE:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile license")
    if provenance["citation"] != PHROGS_PROFILE_SOURCE_CITATION:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile citation")
    if provenance["minimum_mmseqs_version"] != PHROGS_PROFILE_MIN_MMSEQS_VERSION:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile MMseqs minimum")
    if provenance["built_with_mmseqs_version"] != PHROGS_PROFILE_BUILDER_MMSEQS_VERSION:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile MMseqs builder")
    verified_archive = provenance.get("verified_archive")
    if (
        not isinstance(verified_archive, dict)
        or set(verified_archive) != {"path", "sha256"}
        or not isinstance(verified_archive["path"], str)
        or not isinstance(verified_archive["sha256"], str)
        or verified_archive["sha256"] != observed_archive_sha256
    ):
        raise RuntimeError("Safety manifest phrogs_v4.profile_database lacks a verified archive cache")

    if not verify_asset_paths:
        return
    amrfinder_record = manifest["amrfinder_plus"]
    for path_field, digest_field, label in (
        ("binary_path", "binary_sha256", "AMRFinder binary"),
        ("amrfinder_index_path", "amrfinder_index_sha256", "AMRFinder index"),
        ("amrfinder_update_path", "amrfinder_update_sha256", "AMRFinder updater"),
        ("makeblastdb_path", "makeblastdb_sha256", "AMRFinder BLAST prerequisite"),
        ("hmmpress_path", "hmmpress_sha256", "AMRFinder HMMER prerequisite"),
        ("database_path", "database_sha256", "AMRFinder database"),
    ):
        _validate_recorded_asset_digest(amrfinder_record, path_field, digest_field, label)
    toxin_record = manifest["toxin_reference"]
    for file_role in ("annotations", "fasta", "diamond_database"):
        _validate_recorded_asset_digest(
            toxin_record["files"][file_role],
            "path",
            "sha256",
            f"UniProt toxin {file_role}",
        )
    _validate_recorded_asset_digest(manifest["phrogs_v4"], "source_path", "source_sha256", "PHROGs source")
    _validate_recorded_asset_digest(manifest["phrogs_v4"], "lookup_path", "lookup_sha256", "PHROGs lookup")
    sequence_database_path = Path(sequence_database["path"])
    observed_sequence_database_sha256, _ = _complete_phrogs_sequence_database(sequence_database_path)
    if observed_sequence_database_sha256 != sequence_database["sha256"]:
        raise RuntimeError(
            "Safety manifest PHROGs sequence database digest does not match its staged path: "
            f"expected {sequence_database['sha256']}, observed {observed_sequence_database_sha256}"
        )
    profile_database_path = Path(profile_database["path"])
    observed_profile_database_sha256, observed_profile_database_files = _complete_phrogs_profile_database(
        profile_database_path
    )
    if observed_profile_database_sha256 != profile_database["sha256"]:
        raise RuntimeError(
            "Safety manifest PHROGs profile database digest does not match its staged path: "
            f"expected {profile_database['sha256']}, observed {observed_profile_database_sha256}"
        )
    expected_profile_files = [str(path.resolve()) for path in observed_profile_database_files]
    if profile_database["files"] != expected_profile_files:
        raise RuntimeError("Safety manifest PHROGs profile database sidecar inventory does not match its staged path")
    extracted_tree = profile_database["extracted_tree"]
    if not isinstance(extracted_tree, dict):
        raise RuntimeError("Safety manifest PHROGs profile database lacks an extracted_tree record")
    profile_root = profile_database_path.parent
    observed_profile_tree_files = _phrogs_profile_tree_files(profile_database_path)
    if extracted_tree.get("path") != str(profile_root.resolve()):
        raise RuntimeError("Safety manifest PHROGs profile extracted tree path does not match its staged path")
    if extracted_tree.get("sha256") != _sha256_file_inventory(profile_root, observed_profile_tree_files):
        raise RuntimeError("Safety manifest PHROGs profile extracted tree digest does not match its staged path")
    if extracted_tree.get("files") != [str(path.resolve()) for path in observed_profile_tree_files]:
        raise RuntimeError("Safety manifest PHROGs profile extracted tree inventory does not match its staged path")
    if profile_id_inventory != _phrogs_profile_id_inventory(profile_database_path):
        raise RuntimeError("Safety manifest PHROGs profile ID inventory does not match its staged path")
    _validate_phrogs_profile_archive_lineage(profile_database)


def _create_safety_generation_dir(external_dir: Path) -> Path:
    """Create a new, unpublished immutable safety asset generation on the manifest filesystem."""
    generation_root = Path(external_dir) / "safety" / "generations"
    generation_root.mkdir(parents=True, exist_ok=True)
    while True:
        generation_dir = generation_root / uuid4().hex
        try:
            generation_dir.mkdir()
        except FileExistsError:
            continue
        return generation_dir


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
            insecure_downloads=False,
        )
        assets.append(mmseqs_asset)
    if download_dustmasker:
        assets.append(
            prepare_dustmasker(
                external_dir,
                bin_dir=target_bin_dir,
                blast_plus_url=blast_plus_url,
                overwrite=overwrite,
                insecure_downloads=False,
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
                insecure_downloads=False,
            )
        )
    if download_hmmer:
        assets.append(
            prepare_hmmer(
                external_dir,
                bin_dir=target_bin_dir,
                hmmer_url=hmmer_url,
                overwrite=overwrite,
                insecure_downloads=False,
            )
        )
    safety_generation_dir = _create_safety_generation_dir(external_dir) if with_safety else None
    prepared_phrogs_annotation: PreparedAsset | None = None
    prepared_phrogs_profile_database: PreparedAsset | None = None
    prepared_phrogs_sequence_database: PreparedAsset | None = None
    safety_manifest_published = False
    try:
        if download_phrogs_annotation:
            phrogs_annotation_dir = safety_generation_dir if safety_generation_dir is not None else external_dir
            prepared_phrogs_annotation = prepare_phrogs_annotation(
                phrogs_annotation_dir,
                overwrite=overwrite,
                insecure_downloads=insecure_downloads,
            )
            assets.append(prepared_phrogs_annotation)
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
            phrogs_profile_database_dir = safety_generation_dir if safety_generation_dir is not None else external_dir
            if safety_generation_dir is not None:
                prepared_phrogs_profile_database = prepare_phrogs_safety_profile_db(
                    phrogs_profile_database_dir,
                    overwrite=overwrite,
                )
            else:
                prepared_phrogs_profile_database = prepare_phrogs_mmseqs_db(
                    phrogs_profile_database_dir,
                    phrogs_mmseqs_url=phrogs_mmseqs_url,
                    overwrite=overwrite,
                    insecure_downloads=False,
                )
            if safety_generation_dir is None:
                assets.append(prepared_phrogs_profile_database)
            phrogs_sequence_database_dir = safety_generation_dir if safety_generation_dir is not None else external_dir
            prepared_phrogs_sequence_database = prepare_phrogs_gpu_sequence_db(
                phrogs_sequence_database_dir,
                bin_dir=target_bin_dir,
                phrogs_fasta_url=phrogs_fasta_url,
                overwrite=overwrite,
                insecure_downloads=False,
            )
            assets.append(prepared_phrogs_sequence_database)
            if download_checkv:
                assets.append(prepare_checkv_database(external_dir, bin_dir=target_bin_dir, overwrite=overwrite))
        if with_safety:
            assert safety_generation_dir is not None
            selected_safety_manifest = (
                Path(safety_manifest)
                if safety_manifest is not None
                else external_dir / "safety" / "asset_manifest.yaml"
            )
            previous_manifest = _read_safety_manifest(selected_safety_manifest)
            selected_annotation_path = (
                prepared_phrogs_annotation.path
                if prepared_phrogs_annotation is not None
                else external_dir / "phrogs" / "phrog_annot_v4.tsv"
            )
            selected_sequence_database = (
                prepared_phrogs_sequence_database.path
                if prepared_phrogs_sequence_database is not None
                else external_dir / "phrogs" / "phrogs_gpu_seq_db_pad"
            )
            snapshot_annotation_path, snapshot_sequence_database = _snapshot_phrogs_safety_assets(
                selected_annotation_path,
                selected_sequence_database,
                safety_generation_dir,
            )
            profile_archive_path = (
                safety_generation_dir / "downloads" / Path(DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
                if prepared_phrogs_profile_database is not None
                else _cached_phrogs_safety_profile_archive(previous_manifest, external_dir)
            )
            profile_retrieved_at = (
                datetime.fromtimestamp(profile_archive_path.stat().st_mtime, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
            verified_profile = _extract_verified_phrogs_safety_profile_archive(
                profile_archive_path,
                safety_generation_dir / "phrogs" / "verified_profile_source",
            )
            cached_profile_archive_path = _publish_phrogs_safety_profile_archive(
                profile_archive_path,
                external_dir,
                _verified_profile=verified_profile,
            )
            if _sha256_file(cached_profile_archive_path) != verified_profile.observed_archive_sha256:
                raise RuntimeError("PHROGs verified archive cache changed after clean extraction")
            if profile_archive_path != cached_profile_archive_path:
                profile_archive_path.unlink()
            verified_profile = _VerifiedPhrogsProfile(
                archive_path=cached_profile_archive_path,
                observed_archive_sha256=verified_profile.observed_archive_sha256,
                extracted_dir=verified_profile.extracted_dir,
                profile_database=verified_profile.profile_database,
                database_sha256=verified_profile.database_sha256,
                tree_sha256=verified_profile.tree_sha256,
                profile_id_inventory=verified_profile.profile_id_inventory,
                _authority=_VERIFIED_PHROGS_PROFILE_AUTHORITY,
            )
            snapshot_profile_database, snapshot_profile_root = _snapshot_phrogs_profile_database(
                verified_profile.profile_database,
                safety_generation_dir,
            )
            shutil.rmtree(verified_profile.extracted_dir)
            if (
                prepared_phrogs_profile_database is not None
                and prepared_phrogs_profile_database.path != verified_profile.extracted_dir
            ):
                shutil.rmtree(prepared_phrogs_profile_database.path, ignore_errors=True)
            if prepared_phrogs_profile_database is not None:
                assets.append(
                    PreparedAsset(
                        "phrogs_safety_profile_db",
                        snapshot_profile_database,
                        "immutable PHROGs safety profile snapshot",
                    )
                )
            staged_manifest: dict = {"schema_version": 1}
            _set_safety_manifest_recipe(staged_manifest)
            generation_bin_dir = safety_generation_dir / "bin"
            assets.append(
                prepare_amrfinder_plus(
                    external_dir,
                    bin_dir=generation_bin_dir,
                    prerequisite_bin_dir=target_bin_dir,
                    database_dir=safety_generation_dir / "amrfinder" / "database",
                    safety_dir=safety_generation_dir,
                    manifest=staged_manifest,
                    overwrite=overwrite,
                    insecure_downloads=False,
                )
            )
            assets.append(
                prepare_toxin_reference(
                    external_dir,
                    diamond_bin=target_bin_dir / "diamond",
                    safety_dir=safety_generation_dir,
                    manifest=staged_manifest,
                    existing_manifest=previous_manifest,
                    overwrite=overwrite,
                    insecure_downloads=False,
                )
            )
            assets.append(
                _prepare_verified_phrogs_safety_metadata(
                    external_dir,
                    safety_dir=safety_generation_dir,
                    manifest=staged_manifest,
                    annotation_path=snapshot_annotation_path,
                    sequence_database=snapshot_sequence_database,
                    profile_database=snapshot_profile_database,
                    profile_retrieved_at=profile_retrieved_at,
                    _verified_profile=verified_profile,
                )
            )
            _validate_staged_safety_manifest(staged_manifest, verify_asset_paths=True)
            _write_safety_manifest_atomic(selected_safety_manifest, staged_manifest)
            safety_manifest_published = True
            if prepared_phrogs_annotation is not None or prepared_phrogs_sequence_database is not None:
                _publish_phrogs_legacy_assets(snapshot_annotation_path, snapshot_sequence_database, external_dir)
            if prepared_phrogs_profile_database is not None:
                _publish_phrogs_legacy_profile_database(
                    snapshot_profile_database,
                    snapshot_profile_root,
                    external_dir,
                )
        return assets
    except Exception:
        if safety_generation_dir is not None and not safety_manifest_published:
            shutil.rmtree(safety_generation_dir, ignore_errors=True)
        raise


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
        default=None,
        help="Runtime safety asset manifest (default: <external-dir>/safety/asset_manifest.yaml)",
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
        help="Disable TLS verification only for the digest-pinned PHROGs annotation TSV.",
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
