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
    "reviewed:true AND keyword:KW-0800 AND\n((keyword:KW-0843 AND NOT keyword:KW-0078) OR taxonomy_id:2759)"
)
DEFAULT_UNIPROT_TOXIN_ANNOTATIONS_URL = "https://rest.uniprot.org/uniprotkb/stream?" + urlencode(
    {
        "query": DEFAULT_UNIPROT_TOXIN_QUERY,
        "format": "tsv",
        "fields": ("accession,id,protein_name,gene_names,organism_name,organism_id,keywordid,lineage_ids,cc_function"),
    }
)
DEFAULT_UNIPROT_TOXIN_FASTA_URL = "https://rest.uniprot.org/uniprotkb/stream?" + urlencode(
    {"query": DEFAULT_UNIPROT_TOXIN_QUERY, "format": "fasta"}
)
DEFAULT_WOPIP1_PROTEIN_URLS = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id=CAQ54400.1&rettype=fasta&retmode=text",
    "https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?id=CAQ54400.1&db=protein&report=fasta&retmode=text",
)
DEFAULT_WOPIP1_PROTEIN_SEQUENCE_SHA256 = "8e8eb5098bd972dadd0c94ccbd0718c3ede5e528ac2517c605ece16e9eb08a73"
DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_INTERVAL = (2571, 2706)
DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_SEQUENCE_SHA256 = "9da486e50032ff2f89b493049419d7fb9f754f8cc935abb9339f56631dd6a8be"
CURATED_TOXIN_HAZARD_SET_ID = "phage-domain-hazards-v1"
TOXIN_REFERENCE_CLASSIFICATION_POLICY = {
    "policy_id": "human-harm-toxin-reference-v1",
    "hard_fail_scope": "reviewed_whole_protein_human_harm_or_human_disease_virulence",
    "fragment_action": "REVIEW",
    "domain_homology_action": "REVIEW",
    "antibacterial_only_action": "NON_GATING",
    "antibacterial_keyword": "KW-0078",
    "independent_eukaryotic_harm_overrides_antibacterial_exclusion": True,
}
PHROGS_INTEGRATION_EXCISION_CATEGORY = "integration and excision"
PHROGS_HIGH_CONFIDENCE_TERMS = (
    "integrase",
    "excisionase",
    "site-specific recombinase",
    "lysogeny repressor",
)
PHROGS_ADDITIONAL_HIGH_CONFIDENCE_ANNOTATIONS = ("anti-repressor", "ci-like repressor")
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


@dataclass(frozen=True)
class PhrogsSafetyRelease:
    """Reviewed identity and reconciliation contract for one supported Pharokka/PHROGs release."""

    annotation_source_urls: tuple[str, ...]
    annotation_archive_filename: str
    annotation_sha256: str
    profile_source_urls: tuple[str, ...]
    profile_archive_filename: str
    archive_published_md5: str
    archive_published_size: int
    release: str
    dataset_release: str
    doi: str
    license: str
    citation: str
    minimum_mmseqs_version: str
    built_with_mmseqs_version: str
    database_name: str
    release_marker: str
    release_marker_empty_sentinel_allowed: bool
    generated_release_marker_content: bytes
    source_lookup_counts: tuple[int, int, int]
    searchable_lookup_counts: tuple[int, int, int]
    allowed_profile_exclusions: frozenset[str]


def _release_urls(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"PHROGs safety recipe {field} must be a nonempty URL list")
    return tuple(value)


def _release_lookup_counts(value: object, *, field: str) -> tuple[int, int, int]:
    if not isinstance(value, dict) or set(value) != {"total", "high_confidence", "review"}:
        raise ValueError(f"PHROGs safety recipe {field} must contain total/high_confidence/review")
    counts = (value["total"], value["high_confidence"], value["review"])
    if any(type(count) is not int or count < 0 for count in counts) or counts[0] != counts[1] + counts[2]:
        raise ValueError(f"PHROGs safety recipe {field} has inconsistent counts")
    return counts


def load_phrogs_safety_release(recipe_path: Path = DEFAULT_SAFETY_RECIPE_PATH) -> PhrogsSafetyRelease:
    """Load one reviewed release contract; callers never follow an implicit upstream ``latest``."""
    recipe_path = Path(recipe_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    if not isinstance(recipe, dict) or recipe.get("schema_version") != 2:
        raise ValueError(f"PHROGs safety recipe must use schema_version 2: {recipe_path}")
    try:
        phrogs = recipe["phrogs_v4"]
        profile = phrogs["profile_database"]
        reconciliation = phrogs["lookup_reconciliation"]
        source_counts = _release_lookup_counts(reconciliation["source_counts"], field="source_counts")
        searchable_counts = _release_lookup_counts(reconciliation["searchable_counts"], field="searchable_counts")
        exclusions = frozenset(reconciliation["allowed_profile_exclusions"])
        release = PhrogsSafetyRelease(
            annotation_source_urls=_release_urls(phrogs["annotation_urls"], field="annotation_urls"),
            annotation_archive_filename=str(phrogs["annotation_archive_filename"]),
            annotation_sha256=str(phrogs["annotation_sha256"]).lower(),
            profile_source_urls=_release_urls(profile["source_urls"], field="profile_database.source_urls"),
            profile_archive_filename=str(profile["archive_filename"]),
            archive_published_md5=str(profile["archive_published_md5"]).lower(),
            archive_published_size=int(profile["archive_published_size"]),
            release=str(profile["release"]),
            dataset_release=str(profile["dataset_release"]),
            doi=str(profile["doi"]),
            license=str(profile["license"]),
            citation=str(profile["citation"]),
            minimum_mmseqs_version=str(profile["minimum_mmseqs_version"]),
            built_with_mmseqs_version=str(profile["built_with_mmseqs_version"]),
            database_name=str(profile["database_name"]),
            release_marker=str(profile["release_marker"]),
            release_marker_empty_sentinel_allowed=profile["release_marker_empty_sentinel_allowed"] is True,
            generated_release_marker_content=str(profile["generated_release_marker_content"]).encode(),
            source_lookup_counts=source_counts,
            searchable_lookup_counts=searchable_counts,
            allowed_profile_exclusions=exclusions,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"PHROGs safety recipe is incomplete or malformed: {recipe_path}") from error
    if (
        len(release.annotation_sha256) != 64
        or any(character not in "0123456789abcdef" for character in release.annotation_sha256)
        or len(release.archive_published_md5) != 32
        or any(character not in "0123456789abcdef" for character in release.archive_published_md5)
        or release.archive_published_size <= 0
    ):
        raise ValueError(f"PHROGs safety recipe contains an invalid digest or size: {recipe_path}")
    for name, value in (
        ("annotation_archive_filename", release.annotation_archive_filename),
        ("profile_archive_filename", release.profile_archive_filename),
        ("database_name", release.database_name),
        ("release_marker", release.release_marker),
    ):
        if not value or Path(value).name != value:
            raise ValueError(f"PHROGs safety recipe {name} must be one safe basename")
    if not release.generated_release_marker_content or not release.generated_release_marker_content.endswith(b"\n"):
        raise ValueError("PHROGs generated release marker must be nonempty canonical text ending in a newline")
    if any(not _is_phrogs_profile_identifier(identifier) for identifier in release.allowed_profile_exclusions):
        raise ValueError("PHROGs allowed profile exclusions must be canonical phrog_N identifiers")
    if (
        len(release.allowed_profile_exclusions)
        != release.source_lookup_counts[0] - release.searchable_lookup_counts[0]
    ):
        raise ValueError("PHROGs lookup counts do not reconcile with the explicit profile exclusions")
    return release


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
    release_marker_origin: str = "archive_supplied_canonical_marker"
    release: PhrogsSafetyRelease | None = None


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
    expected_size: int | None = None,
    expected_md5: str | None = None,
    resume: bool = False,
) -> tuple[Path, dict[str, str]]:
    """Download a file, optionally resuming a partial, and verify before promotion."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        _verify_sha256(output_path, expected_sha256)
        if expected_size is not None:
            _verify_file_size(output_path, expected_size)
        if expected_md5 is not None:
            _verify_md5(output_path, expected_md5)
        return output_path, {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if overwrite:
        tmp_path.unlink(missing_ok=True)
    resume_from = tmp_path.stat().st_size if resume and tmp_path.is_file() else 0
    if expected_size is not None and resume_from > expected_size:
        tmp_path.unlink()
        resume_from = 0
    context = ssl._create_unverified_context() if insecure else None
    headers: dict[str, str] = {}
    if not (expected_size is not None and resume_from == expected_size):
        request: str | urllib.request.Request = url
        if resume_from:
            request = urllib.request.Request(url, headers={"Range": f"bytes={resume_from}-"})
        with urllib.request.urlopen(request, context=context) as response:
            mode = "wb"
            if resume_from:
                status = getattr(response, "status", None)
                content_range = str(response.headers.get("Content-Range", ""))
                if status == 206 and content_range.startswith(f"bytes {resume_from}-"):
                    mode = "ab"
                elif status not in {None, 200}:
                    raise OSError(f"server rejected byte-range resume for {url}: HTTP {status}")
            with tmp_path.open(mode) as output:
                shutil.copyfileobj(response, output)
            headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    try:
        if expected_size is not None:
            _verify_file_size(tmp_path, expected_size)
        if expected_md5 is not None:
            _verify_md5(tmp_path, expected_md5)
        _verify_sha256(tmp_path, expected_sha256)
    except (OSError, ValueError):
        if not resume or expected_size is None or not tmp_path.is_file() or tmp_path.stat().st_size >= expected_size:
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
    expected_size: int | None = None,
    expected_md5: str | None = None,
    resume: bool = False,
) -> Path:
    """Download ``url`` to ``output_path`` unless it already exists."""
    downloaded_path, _ = _download_with_headers(
        url,
        output_path,
        overwrite=overwrite,
        insecure=insecure,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        expected_md5=expected_md5,
        resume=resume,
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
    """Download and expose the BLAST+ tools needed by QC and AMRFinder."""
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
    blastp_bin = _find_extracted_executable(extracted_dir, "blastp")
    blastx_bin = _find_extracted_executable(extracted_dir, "blastx")
    target_bin_dir = Path(bin_dir) if bin_dir is not None else external_dir / "bin"
    link_path = _link_executable(dustmasker_bin, target_bin_dir / "dustmasker")
    _link_executable(makeblastdb_bin, target_bin_dir / "makeblastdb")
    _link_executable(blastp_bin, target_bin_dir / "blastp")
    _link_executable(blastx_bin, target_bin_dir / "blastx")
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
        return {"schema_version": 2}
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest is None:
        return {"schema_version": 2}
    if not isinstance(manifest, dict):
        raise ValueError(f"Safety manifest must be a mapping: {manifest_path}")
    return manifest


def _set_safety_manifest_recipe(manifest: dict) -> None:
    """Attach the tracked recipe identity to an in-memory safety manifest."""
    if not DEFAULT_SAFETY_RECIPE_PATH.exists():
        raise FileNotFoundError(f"Safety asset recipe does not exist: {DEFAULT_SAFETY_RECIPE_PATH}")
    manifest["schema_version"] = 2
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


def _parse_amrfinder_database_version(output: str) -> str:
    """Parse AMRFinder's historical scalar or current multi-line database-version output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("AMRFinder reported no nonempty database version")
    if len(lines) == 1 and ":" not in lines[0]:
        return lines[0]
    version_fields = [line for line in lines if line.startswith("Database version:")]
    if not version_fields:
        raise RuntimeError("AMRFinder database-version status is missing a Database version field")
    if len(version_fields) > 1:
        raise RuntimeError("AMRFinder database-version status has multiple Database version fields")
    database_version = version_fields[0].partition(":")[2].strip()
    if not database_version:
        raise RuntimeError("AMRFinder database-version status has an empty Database version field")
    return database_version


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
    blastp_source_path = source_prerequisite_bin_dir / "blastp"
    blastx_source_path = source_prerequisite_bin_dir / "blastx"
    hmmpress_source_path = source_prerequisite_bin_dir / "hmmpress"
    hmmsearch_source_path = source_prerequisite_bin_dir / "hmmsearch"
    if not makeblastdb_source_path.exists():
        raise FileNotFoundError(f"AMRFinder update requires makeblastdb in {source_prerequisite_bin_dir}")
    if not blastp_source_path.exists():
        raise FileNotFoundError(f"AMRFinder scanning requires blastp in {source_prerequisite_bin_dir}")
    if not blastx_source_path.exists():
        raise FileNotFoundError(f"AMRFinder combined scanning requires blastx in {source_prerequisite_bin_dir}")
    if not hmmpress_source_path.exists():
        raise FileNotFoundError(f"AMRFinder update requires hmmpress in {source_prerequisite_bin_dir}")
    if not hmmsearch_source_path.exists():
        raise FileNotFoundError(f"AMRFinder scanning requires hmmsearch in {source_prerequisite_bin_dir}")
    makeblastdb_path = _copy_executable(makeblastdb_source_path, target_bin_dir / "makeblastdb")
    blastp_path = _copy_executable(blastp_source_path, target_bin_dir / "blastp")
    blastx_path = _copy_executable(blastx_source_path, target_bin_dir / "blastx")
    hmmpress_path = _copy_executable(hmmpress_source_path, target_bin_dir / "hmmpress")
    hmmsearch_path = _copy_executable(hmmsearch_source_path, target_bin_dir / "hmmsearch")
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
    database_version_output = subprocess.run(
        [str(amrfinder_path), "--database", str(pinned_database_dir), "--database_version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    database_version = _parse_amrfinder_database_version(database_version_output)
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
            "blastp_path": str(blastp_path.resolve()),
            "blastp_sha256": _sha256_file(blastp_path),
            "blastx_path": str(blastx_path.resolve()),
            "blastx_sha256": _sha256_file(blastx_path),
            "hmmpress_path": str(hmmpress_path.resolve()),
            "hmmpress_sha256": _sha256_file(hmmpress_path),
            "hmmsearch_path": str(hmmsearch_path.resolve()),
            "hmmsearch_sha256": _sha256_file(hmmsearch_path),
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


def _semicolon_values(value: str) -> set[str]:
    """Return trimmed nonempty values from a UniProt semicolon-delimited field."""
    return {item.strip() for item in value.split(";") if item.strip()}


def _uniprot_lineage_ids(value: str) -> set[str]:
    """Parse lineage IDs with or without UniProt's parenthesized rank labels."""
    if "," in value and ";" in value:
        raise ValueError("UniProt lineage value mixes delimiters")
    delimiter = "," if "," in value else ";"
    values = {item.strip() for item in value.split(delimiter) if item.strip()}
    lineage_ids = set()
    for item in values:
        taxon_id, separator, rank = item.partition(" ")
        if (
            not taxon_id.isascii()
            or not taxon_id.isdecimal()
            or taxon_id == "0"
            or (separator and not (rank.startswith("(") and rank.endswith(")")))
        ):
            raise ValueError(f"Malformed UniProt lineage value: {item}")
        lineage_ids.add(taxon_id)
    return lineage_ids


def _validate_uniprot_toxin_scope(annotations_path: Path) -> None:
    """Require each downloaded protein to match the declarative human-harm scope.

    Antibacterial-only bacteriocins are intentionally non-gating.  A reviewed
    bacterial/viral toxin is retained only with UniProt virulence evidence,
    while eukaryotic toxins remain in scope independently of that exclusion.
    """
    with Path(annotations_path).open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        lineage_aliases = {
            "Taxonomic lineage (IDs)",
            "Taxonomic lineage (Ids)",
            "Taxonomic lineage IDs",
        }
        fieldnames = set(reader.fieldnames or ())
        lineage_columns = fieldnames & lineage_aliases
        if not {"Entry", "Keyword ID"}.issubset(fieldnames) or len(lineage_columns) != 1:
            raise ValueError("UniProt toxin annotations lack keyword or lineage evidence")
        lineage_column = lineage_columns.pop()
        for row in reader:
            accession = row.get("Entry", "").strip()
            keywords = _semicolon_values(row.get("Keyword ID", ""))
            lineage_ids = _uniprot_lineage_ids(row.get(lineage_column, ""))
            in_scope = "KW-0800" in keywords and (
                "2759" in lineage_ids or ("KW-0843" in keywords and "KW-0078" not in keywords)
            )
            if not accession or not in_scope:
                raise ValueError(
                    f"UniProt toxin annotation {accession or '<empty>'} is outside the human-harm toxin scope"
                )


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


def _single_protein_fasta_sequence(path: Path, *, accession: str) -> str:
    """Return one exact protein sequence from a canonical accession-version FASTA."""
    lines = Path(path).read_text().splitlines()
    headers = [line for line in lines if line.startswith(">")]
    if len(headers) != 1 or headers[0][1:].split(maxsplit=1)[0] != accession:
        raise ValueError(f"Curated hazard FASTA must contain exactly {accession}")
    sequence = "".join(line.strip() for line in lines if line and not line.startswith(">"))
    if not sequence or not sequence.isascii() or not sequence.isalpha() or sequence != sequence.upper():
        raise ValueError(f"Curated hazard FASTA {accession} has an invalid protein sequence")
    return sequence


def _validate_wopip1_curated_hazard(path: Path) -> None:
    """Authenticate exact accession/version and normalized protein sequence, independent of FASTA wrapping."""
    sequence = _single_protein_fasta_sequence(path, accession="CAQ54400.1")
    if hashlib.sha256(sequence.encode()).hexdigest() != DEFAULT_WOPIP1_PROTEIN_SEQUENCE_SHA256:
        raise ValueError("Curated hazard CAQ54400.1 normalized sequence digest mismatch")


def _wopip1_latrotoxin_domain_sequence(path: Path) -> str:
    """Derive the exact PF15658-aligned WP0292 segment from its authenticated source protein."""
    sequence = _single_protein_fasta_sequence(path, accession="CAQ54400.1")
    start, end = DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_INTERVAL
    if end > len(sequence):
        raise ValueError("Curated hazard CAQ54400.1 is shorter than its reviewed domain interval")
    domain = sequence[start - 1 : end]
    if hashlib.sha256(domain.encode()).hexdigest() != DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_SEQUENCE_SHA256:
        raise ValueError("Curated PF15658.11 domain sequence digest mismatch")
    return domain


def _prepare_wopip1_curated_hazard(toxin_dir: Path, *, overwrite: bool) -> Path:
    """Fetch exact CAQ54400.1 from authenticated fallback endpoints and verify sequence identity."""
    output = Path(toxin_dir) / "curated_hazards" / "CAQ54400.1.faa"
    if output.exists() and not overwrite:
        _validate_wopip1_curated_hazard(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = output.with_suffix(output.suffix + ".candidate")
    errors: list[str] = []
    for url in DEFAULT_WOPIP1_PROTEIN_URLS:
        try:
            _download(url, candidate, overwrite=True, insecure=False)
            _validate_wopip1_curated_hazard(candidate)
            candidate.replace(output)
            return output
        except (OSError, UnicodeError, ValueError) as error:
            candidate.unlink(missing_ok=True)
            errors.append(f"{url}: {error}")
    raise RuntimeError("Cannot authenticate curated hazard CAQ54400.1 from any source: " + "; ".join(errors))


def _write_toxin_search_fasta(uniprot_fasta: Path, curated_fasta: Path, output: Path) -> bool:
    """Materialize reviewed toxins plus a narrowly reviewed hazardous-domain target."""
    domain = _wopip1_latrotoxin_domain_sequence(curated_fasta)
    domain_record = f">domain|PF15658.11|Latrotoxin_C\n{domain}\n".encode()
    combined = uniprot_fasta.read_bytes().rstrip(b"\r\n") + b"\n" + domain_record
    if output.is_file() and output.read_bytes() == combined:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(combined)
    temporary.replace(output)
    return True


def _curated_toxin_hazard_manifest() -> dict:
    """Return the tracked, exact non-functional-claim metadata for the phage-WO control."""
    return {
        "set_id": CURATED_TOXIN_HAZARD_SET_ID,
        "entries": [
            {
                "accession": "PF15658.11",
                "name": "Latrotoxin_C",
                "action": "REVIEW",
                "source_protein_accession": "CAQ54400.1",
                "source_urls": list(DEFAULT_WOPIP1_PROTEIN_URLS),
                "source_protein_sequence_sha256": DEFAULT_WOPIP1_PROTEIN_SEQUENCE_SHA256,
                "source_interval": {
                    "start": DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_INTERVAL[0],
                    "end": DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_INTERVAL[1],
                },
                "sequence_sha256": DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_SEQUENCE_SHA256,
                "reason_code": "TOXIN_LATROTOXIN_C_DOMAIN_HOMOLOGY_REVIEW",
                "evidence_urls": [
                    "https://doi.org/10.1038/ncomms13155",
                    "https://www.ncbi.nlm.nih.gov/Structure/cdd/pfam15658",
                ],
                "interpretation": (
                    "Latrotoxin C-terminal-domain homology; not evidence of intact or functional venom."
                ),
            }
        ],
    }


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
        "classification_policy": TOXIN_REFERENCE_CLASSIFICATION_POLICY,
        "annotations_url": annotations_url,
        "fasta_url": fasta_url,
        "reference_version": f"UniProt {toxin_reference.get('uniprot_release')} + {CURATED_TOXIN_HAZARD_SET_ID}",
        "curated_hazards": _curated_toxin_hazard_manifest(),
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
    curated_hazard_fasta = annotations_path.parent / "curated_hazards" / "CAQ54400.1.faa"
    search_fasta = annotations_path.parent / "toxin_hazards.faa"
    _validate_cached_toxin_file(files.get("annotations"), annotations_path, "annotations")
    _validate_cached_toxin_file(files.get("fasta"), fasta_path, "FASTA")
    _validate_cached_toxin_file(files.get("curated_hazard_fasta"), curated_hazard_fasta, "curated hazard FASTA")
    _validate_cached_toxin_file(files.get("search_fasta"), search_fasta, "combined search FASTA")
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
    curated_hazard_fasta = _prepare_wopip1_curated_hazard(toxin_dir, overwrite=overwrite)
    search_fasta = toxin_dir / "toxin_hazards.faa"
    search_fasta_changed = _write_toxin_search_fasta(fasta_path, curated_hazard_fasta, search_fasta)
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
    _validate_uniprot_toxin_scope(annotations_path)

    # Fresh TSV/FASTA bytes require a DIAMOND index built from those same bytes, even
    # when a stale filename happens to exist from an earlier snapshot.
    if annotations_downloaded or search_fasta_changed or overwrite or not diamond_database.exists():
        selected_diamond_bin = Path(diamond_bin) if diamond_bin is not None else external_dir / "bin" / "diamond"
        subprocess.run(
            [str(selected_diamond_bin), "makedb", "--in", str(search_fasta), "--db", str(diamond_database)],
            check=True,
        )
    if not diamond_database.exists():
        raise FileNotFoundError(f"DIAMOND did not create toxin database: {diamond_database}")

    toxin_manifest = {
        "query": DEFAULT_UNIPROT_TOXIN_QUERY,
        "classification_policy": TOXIN_REFERENCE_CLASSIFICATION_POLICY,
        "annotations_url": annotations_url,
        "fasta_url": fasta_url,
        "retrieved_at": retrieved_at,
        "uniprot_release": uniprot_release,
        "reference_version": f"UniProt {uniprot_release} + {CURATED_TOXIN_HAZARD_SET_ID}",
        "curated_hazards": _curated_toxin_hazard_manifest(),
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
            "curated_hazard_fasta": {
                "path": str(curated_hazard_fasta.resolve()),
                "role": "exact accession-pinned phage domain-hazard protein sequence",
                "sha256": _sha256_file(curated_hazard_fasta),
            },
            "search_fasta": {
                "path": str(search_fasta.resolve()),
                "role": "derived reviewed toxin plus curated phage domain-hazard search FASTA",
                "sha256": _sha256_file(search_fasta),
            },
            "diamond_database": {
                "path": str(diamond_database.resolve()),
                "role": "DIAMOND index of reviewed toxins and curated phage domain hazards",
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


def _complete_phrogs_profile_database(
    profile_database: Path,
    *,
    require_release_marker: bool = True,
    _release: PhrogsSafetyRelease | None = None,
) -> tuple[str, list[Path]]:
    """Validate and digest the pinned Pharokka PHROGs MMseqs profile database."""
    release = _release if _release is not None else load_phrogs_safety_release()
    profile_database = Path(profile_database)
    if profile_database.name != release.database_name:
        raise FileNotFoundError(
            f"PHROGs safety profile database must use the official {release.database_name} prefix: {profile_database}"
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
    )
    missing_paths = [path for path in required_paths if not path.is_file() or path.stat().st_size == 0]
    if missing_paths:
        raise FileNotFoundError(
            "PHROGs safety profile database is not a complete MMseqs profile database; missing "
            + ", ".join(str(path) for path in missing_paths)
        )
    release_marker = profile_database.parent / release.release_marker
    if require_release_marker:
        if release_marker.is_symlink() or not release_marker.is_file():
            raise FileNotFoundError(
                "PHROGs safety profile database is not a complete MMseqs profile database; "
                f"release marker is missing or invalid: {release_marker}"
            )
        marker_content = release_marker.read_bytes()
        allowed_marker_contents = {release.generated_release_marker_content}
        if release.release_marker_empty_sentinel_allowed:
            allowed_marker_contents.add(b"")
        if marker_content not in allowed_marker_contents:
            raise ValueError(f"PHROGs profile release marker has conflicting content: {release_marker}")
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


def _normalize_phrogs_annotation_identifier(identifier: str) -> str:
    """Normalize only PHROGs' documented positive-decimal source alias to ``phrog_N``."""
    if identifier != identifier.strip():
        raise ValueError(f"PHROGs v4 table has a noncanonical PHROG identifier: {identifier!r}")
    if _is_phrogs_profile_identifier(identifier):
        return identifier
    if identifier.isascii() and identifier.isdecimal() and not identifier.startswith("0"):
        return f"phrog_{identifier}"
    raise ValueError(f"PHROGs v4 table has a noncanonical PHROG identifier: {identifier!r}")


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


def _find_phrogs_profile_database(
    profile_root: Path,
    *,
    require_release_marker: bool = True,
    _release: PhrogsSafetyRelease | None = None,
) -> Path:
    """Locate exactly one complete official PHROGs profile database below an extracted root."""
    release = _release if _release is not None else load_phrogs_safety_release()
    profile_root = Path(profile_root)
    if not profile_root.is_dir():
        raise FileNotFoundError(f"PHROGs profile database root is required: {profile_root}")
    candidates = sorted(path for path in profile_root.rglob(release.database_name) if path.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError(
            "PHROGs profile database root must contain exactly one official "
            f"{release.database_name} prefix: {profile_root}"
        )
    _complete_phrogs_profile_database(
        candidates[0],
        require_release_marker=require_release_marker,
        _release=release,
    )
    return candidates[0]


def _ensure_authenticated_phrogs_release_marker(
    profile_database: Path,
    *,
    _release: PhrogsSafetyRelease | None = None,
) -> str:
    """Validate an archive marker or materialize the canonical marker after archive authentication."""
    release = _release if _release is not None else load_phrogs_safety_release()
    release_marker = Path(profile_database).parent / release.release_marker
    if release_marker.is_symlink():
        raise ValueError(f"PHROGs profile release marker must not be a symlink: {release_marker}")
    if release_marker.exists():
        if not release_marker.is_file():
            raise ValueError(f"PHROGs profile release marker must be a regular file: {release_marker}")
        marker_content = release_marker.read_bytes()
        if marker_content == b"" and release.release_marker_empty_sentinel_allowed:
            return "archive_supplied_empty_sentinel"
        if marker_content == release.generated_release_marker_content:
            return "archive_supplied_canonical_marker"
        raise ValueError(f"PHROGs profile release marker has conflicting content: {release_marker}")
    release_marker.write_bytes(release.generated_release_marker_content)
    return "locally_materialized_after_archive_verification"


def _extract_verified_phrogs_safety_profile_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    _release: PhrogsSafetyRelease | None = None,
) -> _VerifiedPhrogsProfile:
    """Cleanly extract the official, size- and MD5-verified Pharokka profile archive."""
    release = _release if _release is not None else load_phrogs_safety_release()
    archive_path = Path(archive_path)
    _verify_file_size(archive_path, release.archive_published_size)
    _verify_md5(archive_path, release.archive_published_md5)
    extracted_dir = _extract_tar(archive_path, output_dir, overwrite=True)
    profile_database = _find_phrogs_profile_database(
        extracted_dir,
        require_release_marker=False,
        _release=release,
    )
    release_marker_origin = _ensure_authenticated_phrogs_release_marker(
        profile_database,
        _release=release,
    )
    database_sha256, _database_files = _complete_phrogs_profile_database(
        profile_database,
        _release=release,
    )
    profile_tree_files = _phrogs_profile_tree_files(profile_database, _release=release)
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
        release_marker_origin=release_marker_origin,
        release=release,
    )


def _require_verified_phrogs_profile(profile: object) -> _VerifiedPhrogsProfile:
    """Return a verified archive handle or reject an unverified profile directory."""
    if not isinstance(profile, _VerifiedPhrogsProfile) or profile._authority is not _VERIFIED_PHROGS_PROFILE_AUTHORITY:
        raise RuntimeError(
            "PHROGs safety metadata requires verified PHROGs profile preparation from the pinned archive"
        )
    return profile


def _phrogs_profile_tree_files(
    profile_database: Path,
    *,
    _release: PhrogsSafetyRelease | None = None,
) -> list[Path]:
    """Return only the complete pinned PHROGs profile inventory, never bundled unrelated databases."""
    release = _release if _release is not None else load_phrogs_safety_release()
    profile_database = Path(profile_database)
    _database_sha256, database_files = _complete_phrogs_profile_database(
        profile_database,
        _release=release,
    )
    release_marker = profile_database.parent / release.release_marker
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


def _snapshot_phrogs_profile_database(
    profile_database: Path,
    safety_dir: Path,
    *,
    _release: PhrogsSafetyRelease | None = None,
) -> tuple[Path, Path]:
    """Copy only the complete PHROGs profile inventory into an unpublished safety generation."""
    release = _release if _release is not None else load_phrogs_safety_release()
    profile_database = Path(profile_database)
    source_root = profile_database.parent
    database_sha256, database_files = _complete_phrogs_profile_database(
        profile_database,
        _release=release,
    )
    source_tree_files = _phrogs_profile_tree_files(profile_database, _release=release)
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
    snapshot_sha256, snapshot_files = _complete_phrogs_profile_database(
        snapshot_database,
        _release=release,
    )
    if snapshot_sha256 != database_sha256:
        raise RuntimeError("PHROGs safety profile snapshot digest does not match its source")
    if [path.relative_to(snapshot_root) for path in snapshot_files] != [
        path.relative_to(source_root) for path in database_files
    ]:
        raise RuntimeError("PHROGs safety profile snapshot sidecar inventory does not match its source")
    snapshot_tree_files = _phrogs_profile_tree_files(snapshot_database, _release=release)
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
    _release: PhrogsSafetyRelease | None = None,
) -> Path:
    """Atomically retain a verified Pharokka archive under its observed content digest."""
    release = _release if _release is not None else load_phrogs_safety_release()
    archive_path = Path(archive_path)
    if _verified_profile is None:
        _verify_file_size(archive_path, release.archive_published_size)
        _verify_md5(archive_path, release.archive_published_md5)
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


def _cached_phrogs_safety_profile_archive(
    previous_manifest: dict,
    external_dir: Path,
    *,
    _release: PhrogsSafetyRelease | None = None,
) -> Path:
    """Find one archive authenticated by release evidence, including pre-manifest cache entries."""
    release = _release if _release is not None else load_phrogs_safety_release()
    external_dir = Path(external_dir)
    candidates: set[Path] = set()
    try:
        record = previous_manifest["phrogs_v4"]["profile_database"]["provenance"]["verified_archive"]
        candidates.add(Path(record["path"]))
    except (KeyError, TypeError):
        pass
    cache_dir = external_dir / "downloads" / "phrogs_safety_profile_archives"
    if cache_dir.is_dir():
        candidates.update(cache_dir.glob("*.tar.gz"))
    candidates.add(external_dir / "downloads" / release.profile_archive_filename)

    authenticated: list[Path] = []
    for archive_path in sorted(candidates):
        if not archive_path.is_file() or archive_path.stat().st_size == 0:
            continue
        try:
            _verify_file_size(archive_path, release.archive_published_size)
            _verify_md5(archive_path, release.archive_published_md5)
        except (OSError, ValueError):
            continue
        observed_sha256 = _sha256_file(archive_path)
        if archive_path.parent == cache_dir and archive_path.name != f"{observed_sha256}.tar.gz":
            continue
        try:
            manifest_record = previous_manifest["phrogs_v4"]["profile_database"]["provenance"]["verified_archive"]
        except (KeyError, TypeError):
            manifest_record = None
        is_manifest_archive = isinstance(manifest_record, dict) and archive_path == Path(manifest_record["path"])
        if is_manifest_archive and str(manifest_record.get("sha256")) != observed_sha256:
            continue
        authenticated.append(archive_path)
    unique_authenticated = sorted(set(authenticated))
    if len(unique_authenticated) > 1:
        raise RuntimeError("PHROGs profile archive cache is ambiguous: multiple authenticated entries")
    if not unique_authenticated:
        raise FileNotFoundError("No authenticated PHROGs profile archive cache is available")
    return unique_authenticated[0]


def _download_reviewed_phrogs_profile_archive(
    external_dir: Path,
    *,
    overwrite: bool = False,
    _release: PhrogsSafetyRelease | None = None,
) -> tuple[Path, str]:
    """Download one explicitly reviewed source candidate and authenticate its published identity."""
    release = _release if _release is not None else load_phrogs_safety_release()
    archive_path = Path(external_dir) / "downloads" / release.profile_archive_filename
    failures: list[str] = []
    for source_url in release.profile_source_urls:
        try:
            downloaded_path = _download(
                source_url,
                archive_path,
                overwrite=overwrite,
                insecure=False,
                expected_size=release.archive_published_size,
                expected_md5=release.archive_published_md5,
                resume=True,
            )
            _verify_file_size(downloaded_path, release.archive_published_size)
            _verify_md5(downloaded_path, release.archive_published_md5)
            return downloaded_path, source_url
        except (OSError, ValueError) as error:
            failures.append(f"{source_url}: {error}")
            archive_path.unlink(missing_ok=True)
    raise RuntimeError(
        "No reviewed Pharokka/PHROGs profile source produced the declared archive identity: " + "; ".join(failures)
    )


def _download_reviewed_phrogs_annotation(
    external_dir: Path,
    *,
    insecure_downloads: bool = False,
    overwrite: bool = False,
    _release: PhrogsSafetyRelease | None = None,
) -> tuple[Path, str]:
    """Fetch the annotation only from reviewed candidates under its exact SHA-256 contract."""
    release = _release if _release is not None else load_phrogs_safety_release()
    annotation_path = Path(external_dir) / "phrogs" / release.annotation_archive_filename
    failures: list[str] = []
    for source_url in release.annotation_source_urls:
        try:
            downloaded_path = _download(
                source_url,
                annotation_path,
                overwrite=overwrite,
                insecure=insecure_downloads,
                expected_sha256=release.annotation_sha256,
            )
            return downloaded_path, source_url
        except (OSError, ValueError) as error:
            failures.append(f"{source_url}: {error}")
            annotation_path.unlink(missing_ok=True)
            annotation_path.with_suffix(annotation_path.suffix + ".tmp").unlink(missing_ok=True)
    raise RuntimeError("No reviewed PHROGs annotation source produced the declared SHA-256: " + "; ".join(failures))


def _find_authenticated_phrogs_annotation(
    extracted_dir: Path,
    *,
    _release: PhrogsSafetyRelease | None = None,
) -> Path:
    """Locate the digest-pinned annotation table inside an already authenticated archive."""
    release = _release if _release is not None else load_phrogs_safety_release()
    candidates = sorted(Path(extracted_dir).rglob(release.annotation_archive_filename))
    regular_candidates = [
        path
        for path in candidates
        if path.is_file() and not path.is_symlink() and _sha256_file(path) == release.annotation_sha256
    ]
    if len(regular_candidates) != 1:
        raise FileNotFoundError(
            "Authenticated Pharokka archive must contain exactly one digest-matching "
            f"{release.annotation_archive_filename}; found {len(regular_candidates)}"
        )
    return regular_candidates[0]


def _snapshot_phrogs_annotation(
    annotation_path: Path,
    safety_dir: Path,
    *,
    _release: PhrogsSafetyRelease | None = None,
) -> Path:
    """Copy the authenticated PHROGs annotation into an unpublished immutable generation."""
    release = _release if _release is not None else load_phrogs_safety_release()
    annotation_path = Path(annotation_path)
    _verify_sha256(annotation_path, release.annotation_sha256)
    snapshot_path = Path(safety_dir) / "phrogs" / "snapshot" / release.annotation_archive_filename
    snapshot_path.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(annotation_path, snapshot_path)
    _verify_sha256(snapshot_path, release.annotation_sha256)
    return snapshot_path


def _publish_phrogs_legacy_annotation(annotation_path: Path, external_dir: Path) -> Path:
    """Publish only the authenticated annotation compatibility copy after manifest publication."""
    annotation_path = Path(annotation_path)
    destination = Path(external_dir) / "phrogs" / annotation_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    try:
        shutil.copy2(annotation_path, temporary_path)
        expected_sha256 = _sha256_file(annotation_path)
        if _sha256_file(temporary_path) != expected_sha256:
            raise RuntimeError("Staged legacy PHROGs annotation digest does not match its safety snapshot")
        os.replace(temporary_path, destination)
        if _sha256_file(destination) != expected_sha256:
            raise RuntimeError("Published legacy PHROGs annotation digest does not match its safety snapshot")
        return destination
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_phrogs_legacy_profile_database(
    profile_database: Path,
    profile_root: Path,
    external_dir: Path,
    *,
    _release: PhrogsSafetyRelease | None = None,
) -> Path:
    """Publish an already-validated profile snapshot at the shared legacy PHROGs path."""
    release = _release if _release is not None else load_phrogs_safety_release()
    profile_database = Path(profile_database)
    profile_root = Path(profile_root)
    database_sha256, database_files = _complete_phrogs_profile_database(
        profile_database,
        _release=release,
    )
    source_tree_files = _phrogs_profile_tree_files(profile_database, _release=release)
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
        staged_sha256, staged_files = _complete_phrogs_profile_database(
            staged_profile,
            _release=release,
        )
        if staged_sha256 != database_sha256:
            raise RuntimeError("Staged legacy PHROGs profile digest does not match its safety snapshot")
        if [path.relative_to(staging_root) for path in staged_files] != [
            path.relative_to(profile_root) for path in database_files
        ]:
            raise RuntimeError("Staged legacy PHROGs profile sidecar inventory does not match its safety snapshot")
        staged_tree_files = _phrogs_profile_tree_files(staged_profile, _release=release)
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
        published_sha256, published_files = _complete_phrogs_profile_database(
            published_profile,
            _release=release,
        )
        if published_sha256 != database_sha256:
            raise RuntimeError("Published legacy PHROGs profile digest does not match its safety snapshot")
        if [path.relative_to(legacy_root) for path in published_files] != [
            path.relative_to(profile_root) for path in database_files
        ]:
            raise RuntimeError("Published legacy PHROGs profile sidecar inventory does not match its safety snapshot")
        published_tree_files = _phrogs_profile_tree_files(published_profile, _release=release)
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
    annotation_sha256: str | None = None,
    annotation_path: Path | None = None,
    sequence_database: Path | None = None,
    profile_database: Path | None = None,
    profile_archive_path: Path | None = None,
    profile_source_url: str | None = None,
    profile_archive_observed_sha256: str | None = None,
    profile_retrieved_at: str | None = None,
    profile_release: str | None = None,
    profile_archive_published_md5: str | None = None,
    profile_archive_published_size: int | None = None,
    profile_doi: str | None = None,
    profile_license: str | None = None,
    profile_minimum_mmseqs_version: str | None = None,
    profile_built_with_mmseqs_version: str | None = None,
    profile_dataset_release: str | None = None,
    profile_release_marker_origin: str = "validated_release_marker",
    safety_dir: Path | None = None,
    _release: PhrogsSafetyRelease | None = None,
) -> PreparedAsset:
    """Build a PHROGs lookup from inputs already authenticated by safety preparation."""
    del sequence_database
    release = _release if _release is not None else load_phrogs_safety_release()
    selected_annotation_sha256 = annotation_sha256 or release.annotation_sha256
    selected_profile_source_url = profile_source_url or release.profile_source_urls[0]
    selected_profile_release = profile_release or release.release
    selected_archive_md5 = profile_archive_published_md5 or release.archive_published_md5
    selected_archive_size = profile_archive_published_size or release.archive_published_size
    selected_profile_doi = profile_doi or release.doi
    selected_profile_license = profile_license or release.license
    selected_minimum_mmseqs_version = profile_minimum_mmseqs_version or release.minimum_mmseqs_version
    selected_built_with_mmseqs_version = profile_built_with_mmseqs_version or release.built_with_mmseqs_version
    selected_dataset_release = profile_dataset_release or release.dataset_release
    external_dir = Path(external_dir)
    selected_annotation_path = (
        Path(annotation_path) if annotation_path is not None else external_dir / "phrogs" / "phrog_annot_v4.tsv"
    )
    if not selected_annotation_path.exists():
        raise FileNotFoundError(f"PHROGs v4 annotation table is required: {selected_annotation_path}")
    source_sha256 = _verify_sha256(selected_annotation_path, selected_annotation_sha256)
    if profile_database is None:
        raise FileNotFoundError("PHROGs safety profile_database is required for identity-bearing lysogeny search")
    selected_profile_database = Path(profile_database)
    profile_database_sha256, profile_database_files = _complete_phrogs_profile_database(
        selected_profile_database,
        _release=release,
    )
    profile_root = selected_profile_database.parent
    profile_tree_files = _phrogs_profile_tree_files(selected_profile_database, _release=release)
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
    expected_identity = (
        selected_profile_release == release.release
        and selected_archive_md5.lower() == release.archive_published_md5
        and selected_archive_size == release.archive_published_size
        and selected_profile_doi == release.doi
        and selected_profile_license == release.license
        and selected_minimum_mmseqs_version == release.minimum_mmseqs_version
        and selected_built_with_mmseqs_version == release.built_with_mmseqs_version
        and selected_dataset_release == release.dataset_release
        and selected_profile_source_url in release.profile_source_urls
    )
    if not expected_identity:
        raise ValueError("PHROGs safety profile provenance does not match the reviewed release contract")

    source_lookup_rows = []
    selected_phrogs = set()
    with selected_annotation_path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required_columns = {"phrog", "color", "annot", "category"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"PHROGs v4 table must contain {sorted(required_columns)}: {selected_annotation_path}")
        for row in reader:
            category = row["category"].strip()
            annotation = row["annot"].strip()
            normalized_annotation = annotation.casefold()
            is_integration_category = category.casefold() == PHROGS_INTEGRATION_EXCISION_CATEGORY.casefold()
            is_additional_high_confidence = normalized_annotation in PHROGS_ADDITIONAL_HIGH_CONFIDENCE_ANNOTATIONS
            if not is_integration_category and not is_additional_high_confidence:
                continue
            source_phrog = row["phrog"]
            if not source_phrog:
                raise ValueError(f"PHROGs v4 table has an empty PHROG identifier: {selected_annotation_path}")
            phrog = _normalize_phrogs_annotation_identifier(source_phrog)
            if phrog in selected_phrogs:
                raise ValueError(f"PHROGs lookup has duplicate PHROG identifier: {phrog}")
            selected_phrogs.add(phrog)
            high_confidence_term = next(
                (term for term in PHROGS_HIGH_CONFIDENCE_TERMS if term in normalized_annotation), None
            )
            if is_additional_high_confidence:
                high_confidence_term = normalized_annotation
            review_term = next((term for term in PHROGS_REVIEW_TERMS if term in normalized_annotation), None)
            confidence = "high_confidence" if high_confidence_term is not None else "review"
            matched_term = high_confidence_term or review_term or "integration and excision category"
            source_lookup_rows.append([phrog, annotation, category, confidence, matched_term])
    if not source_lookup_rows:
        raise ValueError("PHROGs integration and excision lookup is empty")
    missing_profile_ids = sorted(selected_phrogs - profile_ids)
    normalized_declared_sha256 = selected_annotation_sha256.removeprefix("sha256:").lower()

    def lookup_counts(rows: list[list[str]]) -> tuple[int, int, int]:
        return (
            len(rows),
            sum(row[3] == "high_confidence" for row in rows),
            sum(row[3] == "review" for row in rows),
        )

    source_counts = lookup_counts(source_lookup_rows)
    if normalized_declared_sha256 == release.annotation_sha256:
        if source_counts != release.source_lookup_counts:
            raise ValueError(
                "Reviewed PHROGs source lookup count drift: "
                f"expected {release.source_lookup_counts}, observed {source_counts}"
            )
        if frozenset(missing_profile_ids) != release.allowed_profile_exclusions:
            raise ValueError("Reviewed PHROGs source/profile reconciliation drift: " + ", ".join(missing_profile_ids))
    elif missing_profile_ids:
        raise ValueError(
            "PHROGs integration/excision lookup contains IDs absent from the verified profile database: "
            + ", ".join(missing_profile_ids)
        )
    lookup_rows = [row for row in source_lookup_rows if row[0] in profile_ids]
    searchable_counts = lookup_counts(lookup_rows)
    if (
        normalized_declared_sha256 == release.annotation_sha256
        and searchable_counts != release.searchable_lookup_counts
    ):
        raise ValueError(
            "Reviewed PHROGs searchable lookup count drift: "
            f"expected {release.searchable_lookup_counts}, observed {searchable_counts}"
        )
    if not profile_ids - selected_phrogs:
        raise ValueError(
            "PHROGs safety profile must contain families beyond the pinned safety lookup; "
            "a subset-only profile cannot represent the full PHROGs v4 search scope"
        )

    safety_dir = Path(safety_dir) if safety_dir is not None else external_dir / "safety"
    lookup_path = safety_dir / "phrogs" / "phrogs_integration_excision_v4.tsv"
    _write_phrogs_lookup(lookup_path, lookup_rows)
    selected_manifest_path = Path(manifest_path) if manifest_path is not None else safety_dir / "asset_manifest.yaml"
    _record_safety_manifest_section(
        selected_manifest_path,
        "phrogs_v4",
        {
            "annotation_url": release.annotation_source_urls[0],
            "annotation_source_urls": list(release.annotation_source_urls),
            "annotation_sha256": normalized_declared_sha256,
            "source_path": str(selected_annotation_path.resolve()),
            "source_sha256": source_sha256,
            "category": PHROGS_INTEGRATION_EXCISION_CATEGORY,
            "high_confidence_terms": list(PHROGS_HIGH_CONFIDENCE_TERMS),
            "additional_high_confidence_annotations": list(PHROGS_ADDITIONAL_HIGH_CONFIDENCE_ANNOTATIONS),
            "review_terms": list(PHROGS_REVIEW_TERMS),
            "lookup_path": str(lookup_path.resolve()),
            "lookup_sha256": _sha256_file(lookup_path),
            "source_lookup_counts": {
                "total": source_counts[0],
                "high_confidence": source_counts[1],
                "review": source_counts[2],
            },
            "lookup_counts": {
                "total": searchable_counts[0],
                "high_confidence": searchable_counts[1],
                "review": searchable_counts[2],
            },
            "profile_exclusions": {
                "ids": missing_profile_ids,
                "reason": "absent_from_authenticated_profile_lookup",
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
                "release_marker": {
                    "path": str((profile_root / release.release_marker).resolve()),
                    "sha256": _sha256_file(profile_root / release.release_marker),
                    "origin": profile_release_marker_origin,
                    "empty_sentinel": (profile_root / release.release_marker).stat().st_size == 0,
                },
                "provenance": {
                    "source_url": selected_profile_source_url,
                    "source_urls": list(release.profile_source_urls),
                    "archive_observed_sha256": normalized_profile_archive_sha256,
                    "archive_published_sha256": None,
                    "archive_published_md5": selected_archive_md5,
                    "archive_published_size": selected_archive_size,
                    "retrieved_at": profile_retrieved_at,
                    "release": selected_profile_release,
                    "dataset_release": selected_dataset_release,
                    "doi": selected_profile_doi,
                    "license": selected_profile_license,
                    "citation": release.citation,
                    "minimum_mmseqs_version": selected_minimum_mmseqs_version,
                    "built_with_mmseqs_version": selected_built_with_mmseqs_version,
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
    release = verified_profile.release if verified_profile.release is not None else load_phrogs_safety_release()
    supplied_profile_database = kwargs.pop("profile_database", None)
    selected_profile_database = (
        verified_profile.profile_database if supplied_profile_database is None else Path(supplied_profile_database)
    )
    selected_database_sha256, _selected_database_files = _complete_phrogs_profile_database(
        selected_profile_database,
        _release=release,
    )
    selected_tree_sha256 = _sha256_file_inventory(
        selected_profile_database.parent,
        _phrogs_profile_tree_files(selected_profile_database, _release=release),
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
        profile_release_marker_origin=verified_profile.release_marker_origin,
        _release=release,
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
    _release: PhrogsSafetyRelease | None = None,
) -> PreparedAsset:
    """Prepare the versioned Pharokka PHROGs profile asset required for identity-bearing safety search."""
    release = _release if _release is not None else load_phrogs_safety_release()
    external_dir = Path(external_dir)
    archive_path, source_url = _download_reviewed_phrogs_profile_archive(
        external_dir,
        overwrite=overwrite,
        _release=release,
    )
    verified_profile = _extract_verified_phrogs_safety_profile_archive(
        archive_path,
        external_dir / "phrogs" / "phrogs_mmseqs_db",
        _release=release,
    )
    return PreparedAsset(
        "phrogs_safety_profile_db",
        verified_profile.extracted_dir,
        f"{release.release} from {source_url}",
    )


def prepare_phrogs_gpu_sequence_db(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    phrogs_fasta_url: str = DEFAULT_PHROGS_FASTA_URL,
    phrogs_fasta_sha256: str | None = None,
    overwrite: bool = False,
    insecure_downloads: bool = False,
) -> PreparedAsset:
    """Build a padded GPU-ready MMseqs sequence database from PHROGs FASTA files."""
    if phrogs_fasta_sha256 is None:
        raise ValueError("PHROGs FASTA archive digest is required before download")
    external_dir = Path(external_dir)
    phrogs_dir = external_dir / "phrogs"
    archive_path = _download(
        phrogs_fasta_url,
        external_dir / "downloads" / Path(phrogs_fasta_url).name,
        overwrite=overwrite,
        insecure=insecure_downloads,
        expected_sha256=phrogs_fasta_sha256,
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
    """Reject publication when a staged asset is missing or has changed."""
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


def _validate_phrogs_profile_archive_lineage(
    profile_database: dict,
    *,
    release: PhrogsSafetyRelease | None = None,
) -> None:
    """Prove a staged profile snapshot came from a clean extraction of its pinned archive cache."""
    selected_release = release if release is not None else load_phrogs_safety_release()
    provenance = profile_database["provenance"]
    verified_archive = provenance.get("verified_archive")
    archive_path = Path(verified_archive["path"])
    observed_archive_sha256 = provenance["archive_observed_sha256"]
    if archive_path.is_symlink() or not archive_path.is_file():
        raise RuntimeError(f"Safety manifest PHROGs verified archive cache is missing or invalid: {archive_path}")
    if _sha256_file(archive_path) != observed_archive_sha256:
        raise RuntimeError("Safety manifest PHROGs verified archive cache digest does not match provenance")
    if verified_archive["sha256"] != observed_archive_sha256:
        raise RuntimeError("Safety manifest PHROGs verified archive cache does not match provenance")
    _verify_file_size(archive_path, selected_release.archive_published_size)
    _verify_md5(archive_path, selected_release.archive_published_md5)

    profile_path = Path(profile_database["path"])
    extraction_parent = Path(tempfile.mkdtemp(prefix=".phrogs-profile-lineage-", dir=profile_path.parent))
    try:
        verified_profile = _extract_verified_phrogs_safety_profile_archive(
            archive_path,
            extraction_parent / "extracted",
            _release=selected_release,
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
    if manifest.get("schema_version") != 2:
        raise RuntimeError("Safety manifest must use schema_version 2")
    release = load_phrogs_safety_release()
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
            "blastp_path",
            "blastp_sha256",
            "blastx_path",
            "blastx_sha256",
            "hmmpress_path",
            "hmmpress_sha256",
            "hmmsearch_path",
            "hmmsearch_sha256",
            "database_path",
            "database_version",
            "database_sha256",
        ),
        "toxin_reference": (
            "query",
            "classification_policy",
            "annotations_url",
            "fasta_url",
            "retrieved_at",
            "uniprot_release",
            "reference_version",
            "curated_hazards",
            "files",
        ),
        "phrogs_v4": (
            "annotation_source_urls",
            "source_sha256",
            "lookup_sha256",
            "source_lookup_counts",
            "lookup_counts",
            "profile_exclusions",
            "profile_database",
        ),
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
    expected_reference_version = (
        f"UniProt {manifest['toxin_reference']['uniprot_release']} + {CURATED_TOXIN_HAZARD_SET_ID}"
    )
    if manifest["toxin_reference"]["reference_version"] != expected_reference_version:
        raise RuntimeError("Safety manifest toxin_reference has unsupported reference_version")
    if manifest["toxin_reference"]["classification_policy"] != TOXIN_REFERENCE_CLASSIFICATION_POLICY:
        raise RuntimeError("Safety manifest toxin_reference has unsupported classification_policy")
    if manifest["toxin_reference"]["curated_hazards"] != _curated_toxin_hazard_manifest():
        raise RuntimeError("Safety manifest toxin_reference has unsupported curated_hazards")
    for file_role in ("annotations", "fasta", "curated_hazard_fasta", "search_fasta", "diamond_database"):
        file_record = toxin_files.get(file_role)
        if not isinstance(file_record, dict) or not file_record.get("path") or not file_record.get("sha256"):
            raise RuntimeError(f"Safety manifest toxin_reference lacks required fields files.{file_role}")
    phrogs_record = manifest["phrogs_v4"]
    if phrogs_record["annotation_source_urls"] != list(release.annotation_source_urls):
        raise RuntimeError("Safety manifest phrogs_v4 has unsupported annotation source URLs")
    for field, expected_counts in (
        ("source_lookup_counts", release.source_lookup_counts),
        ("lookup_counts", release.searchable_lookup_counts),
    ):
        counts = phrogs_record[field]
        if (
            not isinstance(counts, dict)
            or (
                counts.get("total"),
                counts.get("high_confidence"),
                counts.get("review"),
            )
            != expected_counts
        ):
            raise RuntimeError(f"Safety manifest phrogs_v4 has unsupported {field}")
    exclusions = phrogs_record["profile_exclusions"]
    if (
        not isinstance(exclusions, dict)
        or exclusions.get("reason") != "absent_from_authenticated_profile_lookup"
        or frozenset(exclusions.get("ids", ())) != release.allowed_profile_exclusions
    ):
        raise RuntimeError("Safety manifest phrogs_v4 has unsupported profile_exclusions")
    profile_database = phrogs_record["profile_database"]
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
        "release_marker",
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
    release_marker = profile_database["release_marker"]
    if not isinstance(release_marker, dict):
        raise RuntimeError("Safety manifest phrogs_v4.profile_database lacks release_marker provenance")
    marker_origin = release_marker.get("origin")
    if marker_origin not in {
        "archive_supplied_empty_sentinel",
        "archive_supplied_canonical_marker",
        "locally_materialized_after_archive_verification",
        "validated_release_marker",
    }:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has unsupported release marker origin")
    if (
        not isinstance(release_marker.get("path"), str)
        or not isinstance(release_marker.get("sha256"), str)
        or len(release_marker["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in release_marker["sha256"])
        or type(release_marker.get("empty_sentinel")) is not bool
    ):
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has invalid release marker evidence")
    provenance = profile_database["provenance"]
    if not isinstance(provenance, dict):
        raise RuntimeError("Safety manifest phrogs_v4.profile_database lacks provenance")
    for field in (
        "source_url",
        "source_urls",
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
    if provenance["source_urls"] != list(release.profile_source_urls):
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has unsupported profile source URLs")
    if provenance["source_url"] not in release.profile_source_urls:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unpinned profile source URL")
    if provenance["archive_published_md5"] != release.archive_published_md5:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unpinned profile archive MD5")
    if provenance["archive_published_size"] != release.archive_published_size:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unpinned profile archive size")
    if provenance["release"] != release.release:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile release")
    if provenance["dataset_release"] != release.dataset_release:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported PHROGs dataset release")
    if provenance["doi"] != release.doi:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile DOI")
    if provenance["license"] != release.license:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile license")
    if provenance["citation"] != release.citation:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile citation")
    if provenance["minimum_mmseqs_version"] != release.minimum_mmseqs_version:
        raise RuntimeError("Safety manifest phrogs_v4.profile_database has an unsupported profile MMseqs minimum")
    if provenance["built_with_mmseqs_version"] != release.built_with_mmseqs_version:
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
        ("blastp_path", "blastp_sha256", "AMRFinder BLASTP runtime"),
        ("blastx_path", "blastx_sha256", "AMRFinder BLASTX runtime"),
        ("hmmpress_path", "hmmpress_sha256", "AMRFinder HMMER prerequisite"),
        ("hmmsearch_path", "hmmsearch_sha256", "AMRFinder HMMSEARCH runtime"),
        ("database_path", "database_sha256", "AMRFinder database"),
    ):
        _validate_recorded_asset_digest(amrfinder_record, path_field, digest_field, label)
    toxin_record = manifest["toxin_reference"]
    for file_role in ("annotations", "fasta", "curated_hazard_fasta", "search_fasta", "diamond_database"):
        _validate_recorded_asset_digest(
            toxin_record["files"][file_role],
            "path",
            "sha256",
            f"UniProt toxin {file_role}",
        )
    _validate_recorded_asset_digest(manifest["phrogs_v4"], "source_path", "source_sha256", "PHROGs source")
    _validate_recorded_asset_digest(manifest["phrogs_v4"], "lookup_path", "lookup_sha256", "PHROGs lookup")
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
    marker_path = Path(release_marker["path"])
    if marker_path != profile_root / release.release_marker:
        raise RuntimeError("Safety manifest PHROGs release marker path does not match its staged profile")
    if marker_path.is_symlink() or not marker_path.is_file():
        raise RuntimeError("Safety manifest PHROGs release marker is not a regular staged file")
    marker_content = marker_path.read_bytes()
    if marker_content not in (b"", release.generated_release_marker_content):
        raise RuntimeError("Safety manifest PHROGs release marker content is unsupported")
    if release_marker["empty_sentinel"] != (marker_content == b""):
        raise RuntimeError("Safety manifest PHROGs release marker empty-sentinel claim is inconsistent")
    if release_marker["sha256"] != _sha256_file(marker_path):
        raise RuntimeError("Safety manifest PHROGs release marker digest does not match its staged path")
    _validate_phrogs_profile_archive_lineage(profile_database, release=release)


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
    phrogs_fasta_sha256: str | None = None,
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
    phrogs_release = load_phrogs_safety_release() if with_safety else None
    selected_safety_manifest = (
        Path(safety_manifest) if safety_manifest is not None else external_dir / "safety" / "asset_manifest.yaml"
    )
    previous_manifest = _read_safety_manifest(selected_safety_manifest) if with_safety else {}
    prepared_phrogs_annotation: PreparedAsset | None = None
    prepared_phrogs_profile_database: PreparedAsset | None = None
    prepared_phrogs_sequence_database: PreparedAsset | None = None
    safety_manifest_published = False
    try:
        if download_phrogs_annotation and not with_safety:
            prepared_phrogs_annotation = prepare_phrogs_annotation(
                external_dir,
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
        if download_large_databases and not with_safety:
            prepared_phrogs_profile_database = prepare_phrogs_mmseqs_db(
                external_dir,
                phrogs_mmseqs_url=phrogs_mmseqs_url,
                overwrite=overwrite,
                insecure_downloads=False,
            )
            assets.append(prepared_phrogs_profile_database)
            if phrogs_fasta_sha256 is not None:
                prepared_phrogs_sequence_database = prepare_phrogs_gpu_sequence_db(
                    external_dir,
                    bin_dir=target_bin_dir,
                    phrogs_fasta_url=phrogs_fasta_url,
                    phrogs_fasta_sha256=phrogs_fasta_sha256,
                    overwrite=overwrite,
                    insecure_downloads=False,
                )
                assets.append(prepared_phrogs_sequence_database)
        if download_large_databases:
            if download_checkv:
                assets.append(prepare_checkv_database(external_dir, bin_dir=target_bin_dir, overwrite=overwrite))
        if with_safety:
            assert safety_generation_dir is not None
            assert phrogs_release is not None
            downloaded_profile_archive = False
            try:
                profile_archive_path = _cached_phrogs_safety_profile_archive(
                    previous_manifest,
                    external_dir,
                    _release=phrogs_release,
                )
            except FileNotFoundError:
                if not download_large_databases:
                    raise FileNotFoundError(
                        "No authenticated PHROGs profile archive cache is available; "
                        "rerun with --download-large-databases to acquire the reviewed release"
                    ) from None
                profile_archive_path, profile_source_url = _download_reviewed_phrogs_profile_archive(
                    external_dir,
                    overwrite=overwrite,
                    _release=phrogs_release,
                )
                downloaded_profile_archive = True
            else:
                previous_provenance = (
                    previous_manifest.get("phrogs_v4", {}).get("profile_database", {}).get("provenance", {})
                )
                previous_source_url = previous_provenance.get("source_url")
                profile_source_url = (
                    previous_source_url
                    if previous_source_url in phrogs_release.profile_source_urls
                    else phrogs_release.profile_source_urls[0]
                )
            profile_retrieved_at = (
                datetime.fromtimestamp(profile_archive_path.stat().st_mtime, timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
            verified_profile = _extract_verified_phrogs_safety_profile_archive(
                profile_archive_path,
                safety_generation_dir / "phrogs" / "verified_profile_source",
                _release=phrogs_release,
            )
            try:
                authenticated_annotation_path = _find_authenticated_phrogs_annotation(
                    verified_profile.extracted_dir,
                    _release=phrogs_release,
                )
            except FileNotFoundError:
                if not download_phrogs_annotation:
                    raise
                authenticated_annotation_path, _annotation_source_url = _download_reviewed_phrogs_annotation(
                    safety_generation_dir,
                    insecure_downloads=insecure_downloads,
                    overwrite=overwrite,
                    _release=phrogs_release,
                )
            snapshot_annotation_path = _snapshot_phrogs_annotation(
                authenticated_annotation_path,
                safety_generation_dir,
                _release=phrogs_release,
            )
            cached_profile_archive_path = _publish_phrogs_safety_profile_archive(
                profile_archive_path,
                external_dir,
                _verified_profile=verified_profile,
                _release=phrogs_release,
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
                release_marker_origin=verified_profile.release_marker_origin,
                release=phrogs_release,
            )
            snapshot_profile_database, snapshot_profile_root = _snapshot_phrogs_profile_database(
                verified_profile.profile_database,
                safety_generation_dir,
                _release=phrogs_release,
            )
            shutil.rmtree(verified_profile.extracted_dir)
            assets.append(
                PreparedAsset(
                    "phrogs_safety_profile_db",
                    snapshot_profile_database,
                    "immutable PHROGs safety profile snapshot",
                )
            )
            staged_manifest: dict = {"schema_version": 2}
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
                    profile_database=snapshot_profile_database,
                    profile_source_url=profile_source_url,
                    profile_retrieved_at=profile_retrieved_at,
                    _verified_profile=verified_profile,
                )
            )
            _validate_staged_safety_manifest(staged_manifest, verify_asset_paths=True)
            _write_safety_manifest_atomic(selected_safety_manifest, staged_manifest)
            safety_manifest_published = True
            if download_phrogs_annotation:
                _publish_phrogs_legacy_annotation(snapshot_annotation_path, external_dir)
            if downloaded_profile_archive:
                _publish_phrogs_legacy_profile_database(
                    snapshot_profile_database,
                    snapshot_profile_root,
                    external_dir,
                    _release=phrogs_release,
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
    parser.add_argument(
        "--phrogs-fasta-sha256",
        default=None,
        help="Declared SHA-256 required before preparing the optional legacy FAA-derived Arc database.",
    )
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
        phrogs_fasta_sha256=args.phrogs_fasta_sha256,
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
