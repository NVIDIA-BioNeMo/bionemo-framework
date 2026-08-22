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

"""Prepare external tools and databases used by phage generation and screening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse

import yaml

from bionemo.evo2_phage_gen.arc_pipeline import ARC_EVO2_GIT_URL, ARC_EVO2_REV, _assert_arc_source_revision


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_DIR = RECIPE_ROOT / "data" / "external"
DEFAULT_BIN_DIR = DEFAULT_EXTERNAL_DIR / "bin"
DEFAULT_MMSEQS_GPU_URL = "https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
BLAST_PLUS_VERSION = "2.17.0"
_BLAST_PLUS_URLS = {
    "x86_64": "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/ncbi-blast-2.17.0+-x64-linux.tar.gz",
    "aarch64": "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/ncbi-blast-2.17.0+-aarch64-linux.tar.gz",
}
DEFAULT_DIAMOND_URL = "https://github.com/bbuchfink/diamond/releases/download/v2.1.24/diamond-linux64.tar.gz"
DEFAULT_HMMER_URL = "https://conda.anaconda.org/bioconda/linux-64/hmmer-3.4-hb6cb901_4.tar.bz2"
DEFAULT_PHAROKKA_DATABASE_URL = (
    "https://zenodo.org/records/21755221/files/pharokka_v1.11.0_databases.tar.gz?download=1"
)
DEFAULT_PHAROKKA_DATABASE_MD5 = "143bb375ddb0b0653e5cb5671f4a7629"
DEFAULT_PHAROKKA_DATABASE_RELEASE = "Pharokka database v1.11.0 / PHROGs v4"
DEFAULT_ARC_EVO2_REPO_URL = ARC_EVO2_GIT_URL
DEFAULT_ARC_EVO2_REPO_REV = ARC_EVO2_REV
DEFAULT_AMRFINDER_RELEASE = "amrfinder_v4.2.7"
DEFAULT_AMRFINDER_URL = (
    "https://github.com/ncbi/amr/releases/download/amrfinder_v4.2.7/amrfinder_binaries_v4.2.7.tar.gz"
)
_TOOL_ARCHIVES: dict[str, dict[str, tuple[str, str | None]]] = {
    "mmseqs2-gpu": {
        "x86_64": (DEFAULT_MMSEQS_GPU_URL, None),
        "aarch64": ("https://mmseqs.com/latest/mmseqs-linux-gpu-arm64.tar.gz", None),
    },
    "diamond": {
        "x86_64": (DEFAULT_DIAMOND_URL, None),
        "aarch64": (
            "https://conda.anaconda.org/bioconda/linux-aarch64/diamond-2.1.24-heed1f17_0.conda",
            "7c095bb36b6f99c494b7ae90757df423",
        ),
    },
    "hmmer": {
        "x86_64": (DEFAULT_HMMER_URL, None),
        "aarch64": (
            "https://conda.anaconda.org/bioconda/linux-aarch64/hmmer-3.4-hfe13ca0_4.tar.bz2",
            "6f0a04231ef9418215c516bd56ae6e68",
        ),
    },
    "amrfinder": {
        "x86_64": (DEFAULT_AMRFINDER_URL, None),
        "aarch64": (
            "https://conda.anaconda.org/bioconda/linux-aarch64/ncbi-amrfinderplus-4.2.7-h9686939_0.conda",
            "592621df3f59ce1b562b962a6190496e",
        ),
    },
}
DEFAULT_UNIPROT_TOXIN_QUERY = (
    "reviewed:true AND keyword:KW-0800 AND ((keyword:KW-0843 AND NOT keyword:KW-0078) OR taxonomy_id:2759)"
)
DEFAULT_UNIPROT_TOXIN_ANNOTATIONS_URL = "https://rest.uniprot.org/uniprotkb/stream?" + urlencode(
    {
        "query": DEFAULT_UNIPROT_TOXIN_QUERY,
        "format": "tsv",
        "fields": "accession,id,protein_name,gene_names,organism_name,organism_id,keywordid,lineage_ids,cc_function",
    }
)
DEFAULT_UNIPROT_TOXIN_FASTA_URL = "https://rest.uniprot.org/uniprotkb/stream?" + urlencode(
    {"query": DEFAULT_UNIPROT_TOXIN_QUERY, "format": "fasta"}
)
DEFAULT_WOPIP1_PROTEIN_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id=CAQ54400.1&rettype=fasta&retmode=text"
)
DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_INTERVAL = (2571, 2706)
PHROGS_INTEGRATION_EXCISION_CATEGORY = "integration and excision"
PHROGS_HIGH_CONFIDENCE_TERMS = (
    "integrase",
    "excisionase",
    "site-specific recombinase",
    "lysogeny repressor",
    "anti-repressor",
    "ci-like repressor",
)
PHROGS_REVIEW_TERMS = ("recombinase", "repressor", "lysogeny", "integration", "excision")
PHROGS_PROFILE_DATABASE_NAME = "phrogs_profile_db"
AMRFINDER_RUNTIME_FILES = (
    "fasta_check",
    "fasta2parts",
    "gff_check",
    "amr_report",
    "fasta_extract",
    "disruption2genesymbol",
    "dna_mutation",
    "mutate",
    "stxtyper",
    "amrfinder",
    "amrfinder_index",
    "amrfinder_update",
    "stx.prot",
)


@dataclass(frozen=True)
class PreparedAsset:
    """One tool or database prepared by the recipe."""

    name: str
    path: Path
    detail: str


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(
    url: str,
    output_path: Path,
    *,
    overwrite: bool = False,
    published_md5: str | None = None,
    timeout: float = 60.0,
    attempts: int = 3,
) -> tuple[Path, dict[str, str]]:
    """Download a URL with bounded retries, partial-file resume, and optional provider checksum."""
    output_path = Path(output_path)
    if output_path.is_file() and not overwrite:
        if published_md5 is not None and _md5(output_path) != published_md5.lower():
            raise ValueError(f"Published checksum does not match cached download: {output_path}")
        return output_path, {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".part")
    if overwrite:
        partial.unlink(missing_ok=True)
    if attempts < 1:
        raise ValueError("Download attempts must be positive")
    if partial.is_file() and published_md5 is not None and _md5(partial) == published_md5.lower():
        partial.replace(output_path)
        return output_path, {}

    headers: dict[str, str] = {}
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        request: str | urllib.request.Request = url
        if offset:
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
        print(f"downloading {output_path.name}: {offset} bytes already present")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                with partial.open("ab" if append else "wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
            break
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as error:
            if attempt == attempts:
                raise
            print(
                f"download interrupted for {output_path.name}; retrying ({attempt}/{attempts}): {error}",
                file=sys.stderr,
            )
            time.sleep(attempt)

    if published_md5 is not None and _md5(partial) != published_md5.lower():
        partial.unlink(missing_ok=True)
        raise ValueError(f"Published checksum does not match download: {url}")
    partial.replace(output_path)
    print(f"download complete: {output_path}")
    return output_path, headers


def _extract_tar(archive_path: Path, output_dir: Path, *, overwrite: bool = False) -> Path:
    output_dir = Path(output_dir)
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        return output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(output_dir, filter="data")
    return output_dir


def _extract_archive(archive_path: Path, output_dir: Path, *, overwrite: bool = False) -> Path:
    """Extract tar archives and the payload layer from modern ``.conda`` packages."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    if archive_path.suffix != ".conda":
        return _extract_tar(archive_path, output_dir, overwrite=overwrite)
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        return output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    import zstandard

    with zipfile.ZipFile(archive_path) as package:
        payloads = [name for name in package.namelist() if name.startswith("pkg-") and name.endswith(".tar.zst")]
        if len(payloads) != 1:
            raise ValueError(f"Expected one payload archive in {archive_path}, found {len(payloads)}")
        with package.open(payloads[0]) as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as payload:
                    payload.extractall(output_dir, filter="data")
    return output_dir


def _find_file(root: Path, name: str) -> Path:
    candidates = sorted(path for path in Path(root).rglob(name) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"Could not find {name} below {root}")
    return candidates[0]


def _expose(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    destination.symlink_to(source.resolve())
    return destination


def _tool_version(path: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            [str(path), *(args or ("--version",))],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return (completed.stdout.strip() or completed.stderr.strip()).splitlines()[0]


def _architecture() -> str:
    machine = platform.machine().lower()
    aliases = {"amd64": "x86_64", "arm64": "aarch64"}
    return aliases.get(machine, machine)


def _select_tool_archive(tool: str, override_url: str | None) -> tuple[str, str | None]:
    """Select a native tool archive and its provider-published MD5 checksum."""
    if override_url is not None:
        return override_url, None
    architecture = _architecture()
    try:
        return _TOOL_ARCHIVES[tool][architecture]
    except KeyError as error:
        raise RuntimeError(f"No {tool} download is configured for {architecture}") from error


def _tool_extract_dir(external_dir: Path, tool: str) -> Path:
    """Keep non-x86 tools separate from stale archives copied from x86 hosts."""
    architecture = _architecture()
    directory = tool if architecture == "x86_64" else f"{tool}-{architecture}"
    return Path(external_dir) / "tools" / directory


def prepare_pyrodigal_wrapper(bin_dir: Path = DEFAULT_BIN_DIR) -> PreparedAsset:
    """Install a Prodigal-compatible wrapper backed by Pyrodigal."""
    path = Path(bin_dir) / "prodigal"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec pyrodigal "$@"\n')
    path.chmod(0o755)
    return PreparedAsset("prodigal", path, "pyrodigal compatibility wrapper")


def prepare_mmseqs_gpu(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    mmseqs_url: str | None = None,
    overwrite: bool = False,
) -> PreparedAsset:
    """Download and unpack the native MMseqs2-GPU build."""
    external_dir = Path(external_dir)
    selected_url, published_md5 = _select_tool_archive("mmseqs2-gpu", mmseqs_url)
    archive, _ = _download(
        selected_url,
        external_dir / "downloads" / Path(selected_url).name,
        overwrite=overwrite,
        published_md5=published_md5,
    )
    extracted = _extract_archive(archive, _tool_extract_dir(external_dir, "mmseqs2-gpu"), overwrite=overwrite)
    source = _find_file(extracted, "mmseqs")
    target = _expose(source, (Path(bin_dir) if bin_dir else external_dir / "bin") / "mmseqs")
    return PreparedAsset("mmseqs2_gpu", target, _tool_version(target, "version"))


def prepare_dustmasker(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    blast_plus_url: str | None = None,
    overwrite: bool = False,
) -> PreparedAsset:
    """Install the BLAST+ dustmasker executable."""
    external_dir = Path(external_dir)
    architecture = _architecture()
    selected_url = blast_plus_url or _BLAST_PLUS_URLS.get(architecture)
    if selected_url is None:
        raise RuntimeError(f"No BLAST+ {BLAST_PLUS_VERSION} download is configured for {architecture}")
    archive, _ = _download(
        selected_url,
        external_dir / "downloads" / Path(selected_url).name,
        overwrite=overwrite,
    )
    extracted = _extract_tar(
        archive,
        external_dir / "tools" / f"ncbi-blast-{BLAST_PLUS_VERSION}-{architecture}",
        overwrite=overwrite,
    )
    target_bin = Path(bin_dir) if bin_dir else external_dir / "bin"
    for name in ("dustmasker", "makeblastdb", "blastn", "blastp", "blastx", "tblastn"):
        _expose(_find_file(extracted, name), target_bin / name)
    return PreparedAsset("dustmasker", target_bin / "dustmasker", f"BLAST+ {BLAST_PLUS_VERSION}")


def configure_lovis4u_mmseqs(mmseqs_bin: Path = DEFAULT_BIN_DIR / "mmseqs") -> PreparedAsset:
    """Point LoVis4u at the selected MMseqs executable."""
    mmseqs_bin = Path(mmseqs_bin)
    if not mmseqs_bin.exists():
        raise FileNotFoundError(f"MMseqs is required before configuring LoVis4u: {mmseqs_bin}")
    subprocess.run(["lovis4u", "--linux"], check=True)
    subprocess.run(["lovis4u", "-smp", str(mmseqs_bin.resolve())], check=True)
    return PreparedAsset("lovis4u_mmseqs", mmseqs_bin, "configured")


def prepare_diamond(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    diamond_url: str | None = None,
    overwrite: bool = False,
) -> PreparedAsset:
    """Download and unpack the native DIAMOND build."""
    external_dir = Path(external_dir)
    selected_url, published_md5 = _select_tool_archive("diamond", diamond_url)
    archive, _ = _download(
        selected_url,
        external_dir / "downloads" / Path(selected_url).name,
        overwrite=overwrite,
        published_md5=published_md5,
    )
    extracted = _extract_archive(archive, _tool_extract_dir(external_dir, "diamond"), overwrite=overwrite)
    target = _expose(
        _find_file(extracted, "diamond"), (Path(bin_dir) if bin_dir else external_dir / "bin") / "diamond"
    )
    return PreparedAsset("diamond", target, _tool_version(target, "version"))


def prepare_hmmer(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    hmmer_url: str | None = None,
    overwrite: bool = False,
) -> PreparedAsset:
    """Download and unpack the native HMMER build."""
    external_dir = Path(external_dir)
    selected_url, published_md5 = _select_tool_archive("hmmer", hmmer_url)
    archive, _ = _download(
        selected_url,
        external_dir / "downloads" / Path(selected_url).name,
        overwrite=overwrite,
        published_md5=published_md5,
    )
    extracted = _extract_archive(archive, _tool_extract_dir(external_dir, "hmmer"), overwrite=overwrite)
    target_bin = Path(bin_dir) if bin_dir else external_dir / "bin"
    for name in ("hmmpress", "hmmsearch"):
        _expose(_find_file(extracted, name), target_bin / name)
    return PreparedAsset("hmmer", target_bin / "hmmsearch", _tool_version(target_bin / "hmmsearch", "-h"))


def _amrfinder_database(root: Path) -> Path:
    latest = root / "latest"
    if latest.exists():
        return latest.resolve()
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"AMRFinder database is empty: {root}")
    return candidates[-1].resolve()


def _amrfinder_database_version(amrfinder: Path, database: Path) -> str:
    completed = subprocess.run(
        [str(amrfinder), "--database", str(database), "--database_version"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("Database version:"):
            return line.partition(":")[2].strip()
    return lines[-1] if lines else database.name


def prepare_amrfinder_plus(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    amrfinder_url: str | None = None,
    amrfinder_release: str = DEFAULT_AMRFINDER_RELEASE,
    database_dir: Path | None = None,
    overwrite: bool = False,
) -> PreparedAsset:
    """Install the native AMRFinderPlus build and its current database."""
    external_dir = Path(external_dir)
    target_bin = Path(bin_dir) if bin_dir else external_dir / "bin"
    selected_url, published_md5 = _select_tool_archive("amrfinder", amrfinder_url)
    archive, _ = _download(
        selected_url,
        external_dir / "downloads" / Path(selected_url).name,
        overwrite=overwrite,
        published_md5=published_md5,
    )
    extracted = _extract_archive(
        archive,
        _tool_extract_dir(external_dir, amrfinder_release),
        overwrite=overwrite,
    )
    for name in AMRFINDER_RUNTIME_FILES:
        _expose(_find_file(extracted, name), target_bin / name)
    for name in ("blastn", "blastp", "blastx", "makeblastdb", "tblastn", "hmmpress", "hmmsearch"):
        path = target_bin / name
        if not path.exists():
            raise FileNotFoundError(f"AMRFinder requires {path}")

    root = Path(database_dir) if database_dir else external_dir / "safety" / "amrfinder"
    if overwrite and root.exists():
        shutil.rmtree(root)
    if not root.exists() or not any(root.iterdir()):
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(target_bin / "amrfinder_update"),
                "-d",
                str(root),
                "--blast_bin",
                str(target_bin),
                "--hmmer_bin",
                str(target_bin),
            ],
            check=True,
        )
    database = _amrfinder_database(root)
    version = _amrfinder_database_version(target_bin / "amrfinder", database)
    state = {
        "tool_version": _tool_version(target_bin / "amrfinder"),
        "database_path": str(database),
        "database_version": version,
        "release_url": selected_url,
    }
    state_path = root / "state.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return PreparedAsset("amrfinder_plus", target_bin / "amrfinder", f"database {version}")


def _fasta_sequence(path: Path) -> str:
    lines = [line.strip() for line in path.read_text().splitlines() if line and not line.startswith(">")]
    sequence = "".join(lines).upper()
    if not sequence:
        raise ValueError(f"FASTA contains no sequence: {path}")
    return sequence


def prepare_toxin_reference(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    diamond_bin: Path | None = None,
    annotations_url: str = DEFAULT_UNIPROT_TOXIN_ANNOTATIONS_URL,
    fasta_url: str = DEFAULT_UNIPROT_TOXIN_FASTA_URL,
    overwrite: bool = False,
) -> PreparedAsset:
    """Build the toxin reference used by the sequence screen."""
    external_dir = Path(external_dir)
    toxin_dir = external_dir / "safety" / "toxins"
    annotations, annotation_headers = _download(
        annotations_url,
        toxin_dir / "reviewed_toxins.tsv",
        overwrite=overwrite,
    )
    fasta, fasta_headers = _download(
        fasta_url,
        toxin_dir / "reviewed_toxins.faa",
        overwrite=overwrite,
    )
    curated, _ = _download(
        DEFAULT_WOPIP1_PROTEIN_URL,
        toxin_dir / "CAQ54400.1.faa",
        overwrite=overwrite,
    )
    start, end = DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_INTERVAL
    sequence = _fasta_sequence(curated)
    if len(sequence) < end:
        raise ValueError("CAQ54400.1 is shorter than the documented Latrotoxin_C interval")
    domain = sequence[start - 1 : end]
    search_fasta = toxin_dir / "toxin_hazards.faa"
    combined = fasta.read_text().rstrip() + f"\n>domain|PF15658.11|Latrotoxin_C\n{domain}\n"
    if overwrite or not search_fasta.exists() or search_fasta.read_text() != combined:
        search_fasta.write_text(combined)
    database = toxin_dir / "toxin_hazards.dmnd"
    if overwrite or not database.exists():
        selected_diamond = Path(diamond_bin) if diamond_bin else external_dir / "bin" / "diamond"
        subprocess.run(
            [str(selected_diamond), "makedb", "--in", str(search_fasta), "--db", str(database)],
            check=True,
        )
    release = annotation_headers.get("x-uniprot-release") or fasta_headers.get("x-uniprot-release") or "not reported"
    state = {
        "release": release,
        "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "query": DEFAULT_UNIPROT_TOXIN_QUERY,
        "annotations_path": str(annotations.resolve()),
        "fasta_path": str(search_fasta.resolve()),
        "diamond_database_path": str(database.resolve()),
        "curated_domain": "CAQ54400.1:2571-2706 (PF15658.11 Latrotoxin_C)",
    }
    (toxin_dir / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return PreparedAsset("toxin_reference", database, f"UniProt {release}")


def _find_pharokka_database(root: Path) -> Path:
    """Find one extracted Pharokka database directory with the PHROGs files we consume."""
    candidates: list[Path] = []
    for annotation in Path(root).rglob("phrog_annot_v4.tsv"):
        candidate = annotation.parent
        required = (
            candidate / PHROGS_PROFILE_DATABASE_NAME,
            candidate / f"{PHROGS_PROFILE_DATABASE_NAME}.dbtype",
            candidate / f"{PHROGS_PROFILE_DATABASE_NAME}.index",
            candidate / f"{PHROGS_PROFILE_DATABASE_NAME}.lookup",
            candidate / f"{PHROGS_PROFILE_DATABASE_NAME}_h",
            candidate / f"{PHROGS_PROFILE_DATABASE_NAME}_h.dbtype",
            candidate / f"{PHROGS_PROFILE_DATABASE_NAME}_h.index",
        )
        if all(path.is_file() for path in required) and any(candidate.glob("VERSION_*")):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one complete Pharokka database below {root}, found {len(candidates)}")
    return candidates[0]


def prepare_pharokka_database(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    database_url: str = DEFAULT_PHAROKKA_DATABASE_URL,
    published_md5: str | None = DEFAULT_PHAROKKA_DATABASE_MD5,
    release: str = DEFAULT_PHAROKKA_DATABASE_RELEASE,
    overwrite: bool = False,
) -> PreparedAsset:
    """Download and validate the Pharokka bundle that supplies the recipe's PHROGs assets."""
    external_dir = Path(external_dir)
    archive_name = Path(urlparse(database_url).path).name
    archive, _ = _download(
        database_url,
        external_dir / "downloads" / archive_name,
        overwrite=overwrite,
        published_md5=published_md5,
    )
    bundle_name = archive_name.removesuffix(".tar.gz")
    extracted = _extract_tar(archive, external_dir / "phrogs" / bundle_name, overwrite=overwrite)
    database = _find_pharokka_database(extracted)
    profile = _find_profile_database(database)
    state = {
        "release": release,
        "source_url": database_url,
        "archive": str(archive.resolve()),
        "database_root": str(database.resolve()),
        "annotation_path": str((database / "phrog_annot_v4.tsv").resolve()),
        "profile_database_path": str(profile.resolve()),
    }
    state_path = external_dir / "phrogs" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return PreparedAsset("pharokka_database", database, release)


def prepare_phrogs_annotation(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    database_root: Path | None = None,
    database_url: str = DEFAULT_PHAROKKA_DATABASE_URL,
    published_md5: str | None = DEFAULT_PHAROKKA_DATABASE_MD5,
    release: str = DEFAULT_PHAROKKA_DATABASE_RELEASE,
    overwrite: bool = False,
) -> PreparedAsset:
    """Expose the PHROGs annotation table from a Pharokka database bundle."""
    external_dir = Path(external_dir)
    if database_root is None:
        database_root = prepare_pharokka_database(
            external_dir,
            database_url=database_url,
            published_md5=published_md5,
            release=release,
            overwrite=overwrite,
        ).path
    source = Path(database_root) / "phrog_annot_v4.tsv"
    if not source.is_file():
        raise FileNotFoundError(f"Pharokka PHROGs annotation table is missing: {source}")
    path = _expose(source, external_dir / "phrogs" / "phrog_annot_v4.tsv")
    return PreparedAsset("phrogs_annotation", path, release)


def _profile_ids(profile_database: Path) -> set[str]:
    lookup = Path(f"{profile_database}.lookup")
    identifiers: set[str] = set()
    for line in lookup.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            identifiers.add(fields[1])
    if not identifiers:
        raise ValueError(f"PHROGs profile lookup is empty: {lookup}")
    return identifiers


def _find_profile_database(root: Path) -> Path:
    candidates = sorted(path for path in root.rglob(PHROGS_PROFILE_DATABASE_NAME) if path.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {PHROGS_PROFILE_DATABASE_NAME} below {root}, found {len(candidates)}")
    required = (candidates[0], Path(f"{candidates[0]}.lookup"), Path(f"{candidates[0]}.dbtype"))
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("PHROGs profile database is incomplete")
    return candidates[0]


def prepare_phrogs_profile_db(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    database_root: Path | None = None,
    database_url: str = DEFAULT_PHAROKKA_DATABASE_URL,
    published_md5: str | None = DEFAULT_PHAROKKA_DATABASE_MD5,
    release: str = DEFAULT_PHAROKKA_DATABASE_RELEASE,
    overwrite: bool = False,
) -> PreparedAsset:
    """Expose the PHROGs profile database from a Pharokka database bundle."""
    external_dir = Path(external_dir)
    if database_root is None:
        database_root = prepare_pharokka_database(
            external_dir,
            database_url=database_url,
            published_md5=published_md5,
            release=release,
            overwrite=overwrite,
        ).path
    database = _find_profile_database(Path(database_root))
    return PreparedAsset("phrogs_profile_database", database, release)


def _phrog_id(value: str) -> str:
    value = value.strip()
    if value.startswith("phrog_") and value[6:].isdigit():
        return value
    if value.isdigit():
        return f"phrog_{value}"
    raise ValueError(f"Unrecognized PHROG identifier: {value!r}")


def prepare_phrogs_lookup(
    annotation_path: Path,
    profile_database: Path,
    output_path: Path,
) -> PreparedAsset:
    """Build the lysogeny lookup for the PHROGs profile database."""
    profile_ids = _profile_ids(profile_database)
    rows: list[list[str]] = []
    seen: set[str] = set()
    with Path(annotation_path).open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None or not {"phrog", "annot", "category"} <= set(reader.fieldnames):
            raise ValueError("PHROGs annotation table lacks phrog, annot, or category")
        for row in reader:
            annotation = row["annot"].strip()
            category = row["category"].strip()
            lowered = annotation.casefold()
            extra = next((term for term in PHROGS_HIGH_CONFIDENCE_TERMS if term in lowered), None)
            if category.casefold() != PHROGS_INTEGRATION_EXCISION_CATEGORY and extra is None:
                continue
            identifier = _phrog_id(row["phrog"])
            if identifier in seen:
                raise ValueError(f"Duplicate PHROGs annotation: {identifier}")
            seen.add(identifier)
            if identifier not in profile_ids:
                continue
            high = next((term for term in PHROGS_HIGH_CONFIDENCE_TERMS if term in lowered), None)
            review = next((term for term in PHROGS_REVIEW_TERMS if term in lowered), None)
            rows.append(
                [
                    identifier,
                    annotation,
                    category,
                    "high_confidence" if high else "review",
                    high or review or PHROGS_INTEGRATION_EXCISION_CATEGORY,
                ]
            )
    if not rows:
        raise ValueError("No lysogeny-related PHROGs families were found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("phrog", "annot", "category", "confidence", "matched_term"))
        writer.writerows(rows)
    return PreparedAsset("phrogs_lysogeny_lookup", output_path, f"{len(rows)} profile families")


def _clear_mmseqs_database(prefix: Path) -> None:
    """Remove an incomplete MMseqs database at one explicit prefix."""
    for path in prefix.parent.glob(f"{prefix.name}*"):
        if path.is_file() or path.is_symlink():
            path.unlink()


def prepare_phrogs_safety_db(
    profile_database: Path,
    safety_lookup: Path,
    output_path: Path,
    *,
    mmseqs_path: Path,
    release: str | None = None,
    runner=subprocess.run,
) -> PreparedAsset:
    """Select the PHROGs profiles used by the lysogeny screen."""
    source = Path(profile_database)
    source_keys: dict[str, str] = {}
    for line in Path(f"{source}.lookup").read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            source_keys[fields[1]] = fields[0]

    with Path(safety_lookup).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected_ids = [row["phrog"] for row in rows]
    if not selected_ids:
        raise ValueError("PHROGs safety lookup contains no profiles")
    missing = sorted(set(selected_ids) - set(source_keys))
    if missing:
        raise ValueError(f"PHROGs safety profiles are missing from the database: {', '.join(missing)}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _clear_mmseqs_database(output)
    key_file = output.parent / f".{output.name}.{os.getpid()}.keys"
    key_file.write_text("".join(f"{source_keys[profile]}\n" for profile in selected_ids))
    try:
        runner(
            [str(mmseqs_path), "createsubdb", str(key_file), str(source), str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        for suffix in (".lookup", ".source", "_h", "_h.dbtype", "_h.index"):
            destination = Path(f"{output}{suffix}")
            destination.unlink(missing_ok=True)
            shutil.copy2(Path(f"{source}{suffix}"), destination)
        required = (output, Path(f"{output}.dbtype"), Path(f"{output}.index"), Path(f"{output}.lookup"))
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("MMseqs did not create a complete PHROGs safety database")
    except Exception:
        _clear_mmseqs_database(output)
        raise
    finally:
        key_file.unlink(missing_ok=True)

    count = len(selected_ids)
    family_count = f"{count} selected {'family' if count == 1 else 'families'}"
    return PreparedAsset("phrogs_safety_database", output, f"{release}; {family_count}" if release else family_count)


def prepare_phrogs_consensus_db(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    profile_database: Path | None = None,
    database_url: str = DEFAULT_PHAROKKA_DATABASE_URL,
    published_md5: str | None = DEFAULT_PHAROKKA_DATABASE_MD5,
    release: str = DEFAULT_PHAROKKA_DATABASE_RELEASE,
    overwrite: bool = False,
) -> PreparedAsset:
    """Derive the Arc-compatible PHROGs consensus database from the Pharokka profiles."""
    external_dir = Path(external_dir)
    if profile_database is None:
        profile_database = prepare_phrogs_profile_db(
            external_dir,
            database_url=database_url,
            published_md5=published_md5,
            release=release,
            overwrite=overwrite,
        ).path
    selected_bin = Path(bin_dir) if bin_dir else external_dir / "bin"
    mmseqs = selected_bin / "mmseqs"
    consensus = external_dir / "phrogs" / "phrogs_consensus_db"
    padded = external_dir / "phrogs" / "phrogs_consensus_db_pad"
    required = (padded, Path(f"{padded}.dbtype"), Path(f"{padded}.lookup"))
    if not overwrite and all(path.is_file() for path in required):
        return PreparedAsset("phrogs_consensus_database", padded, release)

    _clear_mmseqs_database(consensus)
    _clear_mmseqs_database(padded)
    subprocess.run([str(mmseqs), "profile2consensus", str(profile_database), str(consensus)], check=True)
    subprocess.run([str(mmseqs), "makepaddedseqdb", str(consensus), str(padded), "--write-lookup", "1"], check=True)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(f"MMseqs did not create a complete PHROGs consensus database: {padded}")
    return PreparedAsset("phrogs_consensus_database", padded, release)


def prepare_checkv_database(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    bin_dir: Path | None = None,
    overwrite: bool = False,
) -> PreparedAsset:
    """Download the current CheckV database."""
    root = Path(external_dir) / "checkv"
    existing = sorted(root.glob("checkv-db-*")) if root.exists() else []
    if existing and not overwrite:
        return PreparedAsset("checkv_database", existing[-1], "existing")
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    selected_bin = Path(bin_dir) if bin_dir else Path(external_dir) / "bin"
    env = {**os.environ, "PATH": os.pathsep.join((str(selected_bin), os.environ.get("PATH", "")))}
    subprocess.run(["checkv", "download_database", str(root)], check=True, env=env)
    databases = sorted(root.glob("checkv-db-*"))
    if not databases:
        raise FileNotFoundError(f"CheckV did not create a database below {root}")
    return PreparedAsset("checkv_database", databases[-1], "downloaded")


def prepare_arc_evo2_checkout(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    repo_url: str = DEFAULT_ARC_EVO2_REPO_URL,
    repo_rev: str = DEFAULT_ARC_EVO2_REPO_REV,
    overwrite: bool = False,
) -> PreparedAsset:
    """Download the Arc phage-filtering code and data."""
    checkout = Path(external_dir) / "arc_evo2"
    if checkout.exists() and not overwrite:
        _assert_arc_source_revision(checkout, repo_rev)
        return PreparedAsset("arc_evo2", checkout, repo_rev)
    if checkout.exists():
        shutil.rmtree(checkout)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", repo_url, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", repo_rev], check=True)
    return PreparedAsset("arc_evo2", checkout, repo_rev)


def _safety_state(
    external_dir: Path, bin_dir: Path, profile: PreparedAsset, lookup: PreparedAsset
) -> dict[str, object]:
    amr_state = json.loads((external_dir / "safety" / "amrfinder" / "state.json").read_text())
    toxin_state = json.loads((external_dir / "safety" / "toxins" / "state.json").read_text())
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tools": {
            "amrfinder": {"path": str((bin_dir / "amrfinder").absolute()), "version": amr_state["tool_version"]},
            "diamond": {
                "path": str((bin_dir / "diamond").resolve()),
                "version": _tool_version(bin_dir / "diamond", "version"),
            },
            "mmseqs": {
                "path": str((bin_dir / "mmseqs").resolve()),
                "version": _tool_version(bin_dir / "mmseqs", "version"),
            },
        },
        "databases": {
            "amrfinder": {
                "path": amr_state["database_path"],
                "version": amr_state["database_version"],
            },
            "toxins": toxin_state,
            "phrogs": {
                "annotation_path": str((external_dir / "phrogs" / "phrog_annot_v4.tsv").resolve()),
                "lookup_path": str(lookup.path.resolve()),
                "profile_database_path": str(profile.path.resolve()),
                "release": profile.detail,
            },
        },
    }


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
    prepare_phrogs_consensus_database: bool = False,
    download_checkv: bool = True,
    configure_lovis4u: bool = True,
    with_safety: bool = False,
    safety_manifest: Path | None = None,
    mmseqs_url: str | None = None,
    blast_plus_url: str | None = None,
    diamond_url: str | None = None,
    hmmer_url: str | None = None,
    pharokka_database_url: str = DEFAULT_PHAROKKA_DATABASE_URL,
    pharokka_database_md5: str | None = DEFAULT_PHAROKKA_DATABASE_MD5,
    pharokka_database_release: str = DEFAULT_PHAROKKA_DATABASE_RELEASE,
    amrfinder_url: str | None = None,
    amrfinder_release: str = DEFAULT_AMRFINDER_RELEASE,
    arc_evo2_repo_url: str = DEFAULT_ARC_EVO2_REPO_URL,
    arc_evo2_repo_rev: str = DEFAULT_ARC_EVO2_REPO_REV,
    overwrite: bool = False,
) -> list[PreparedAsset]:
    """Prepare the external tools and data requested for this recipe."""
    external_dir = Path(external_dir)
    target_bin = Path(bin_dir) if bin_dir else external_dir / "bin"
    assets: list[PreparedAsset] = [prepare_pyrodigal_wrapper(target_bin)]
    if download_mmseqs:
        assets.append(prepare_mmseqs_gpu(external_dir, bin_dir=target_bin, mmseqs_url=mmseqs_url, overwrite=overwrite))
    if download_dustmasker:
        assets.append(
            prepare_dustmasker(
                external_dir,
                bin_dir=target_bin,
                blast_plus_url=blast_plus_url,
                overwrite=overwrite,
            )
        )
    if download_diamond:
        assets.append(prepare_diamond(external_dir, bin_dir=target_bin, diamond_url=diamond_url, overwrite=overwrite))
    if download_hmmer:
        assets.append(prepare_hmmer(external_dir, bin_dir=target_bin, hmmer_url=hmmer_url, overwrite=overwrite))
    if configure_lovis4u:
        assets.append(configure_lovis4u_mmseqs(target_bin / "mmseqs"))

    annotation: PreparedAsset | None = None
    profile: PreparedAsset | None = None
    needs_pharokka = download_phrogs_annotation or prepare_phrogs_consensus_database or with_safety
    if needs_pharokka:
        pharokka = prepare_pharokka_database(
            external_dir,
            database_url=pharokka_database_url,
            published_md5=pharokka_database_md5,
            release=pharokka_database_release,
            overwrite=overwrite,
        )
        assets.append(pharokka)
        profile = prepare_phrogs_profile_db(
            external_dir,
            database_root=pharokka.path,
            release=pharokka_database_release,
        )
        assets.append(profile)
        if download_phrogs_annotation or with_safety:
            annotation = prepare_phrogs_annotation(
                external_dir,
                database_root=pharokka.path,
                release=pharokka_database_release,
            )
            assets.append(annotation)
        if prepare_phrogs_consensus_database:
            assets.append(
                prepare_phrogs_consensus_db(
                    external_dir,
                    bin_dir=target_bin,
                    profile_database=profile.path,
                    release=pharokka_database_release,
                    overwrite=overwrite,
                )
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
    if download_large_databases and download_checkv:
        assets.append(prepare_checkv_database(external_dir, bin_dir=target_bin, overwrite=overwrite))

    if with_safety:
        if annotation is None or profile is None:
            raise RuntimeError("PHROGs preparation unexpectedly skipped")
        amr = prepare_amrfinder_plus(
            external_dir,
            bin_dir=target_bin,
            amrfinder_url=amrfinder_url,
            amrfinder_release=amrfinder_release,
            overwrite=overwrite,
        )
        toxins = prepare_toxin_reference(
            external_dir,
            diamond_bin=target_bin / "diamond",
            overwrite=overwrite,
        )
        lookup = prepare_phrogs_lookup(
            annotation.path,
            profile.path,
            external_dir / "safety" / "phrogs_lysogeny_lookup.tsv",
        )
        safety_profile = prepare_phrogs_safety_db(
            profile.path,
            lookup.path,
            external_dir / "safety" / "phrogs_profile_db",
            mmseqs_path=target_bin / "mmseqs",
            release=profile.detail,
        )
        assets.extend((amr, toxins, lookup, safety_profile))
        manifest_path = Path(safety_manifest) if safety_manifest else external_dir / "safety" / "asset_manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(_safety_state(external_dir, target_bin, safety_profile, lookup), sort_keys=False)
        )

    return assets


def build_parser() -> argparse.ArgumentParser:
    """Build the external-asset command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--skip-mmseqs", action="store_true")
    parser.add_argument("--skip-dustmasker", action="store_true")
    parser.add_argument("--skip-diamond", action="store_true")
    parser.add_argument("--skip-hmmer", action="store_true")
    parser.add_argument("--skip-phrogs-annotation", action="store_true")
    parser.add_argument("--skip-arc-evo2", action="store_true")
    parser.add_argument("--skip-lovis4u-config", action="store_true")
    parser.add_argument("--skip-checkv", action="store_true")
    parser.add_argument("--download-large-databases", action="store_true")
    parser.add_argument("--prepare-phrogs-consensus-database", action="store_true")
    parser.add_argument("--with-safety", action="store_true")
    parser.add_argument("--safety-manifest", type=Path)
    parser.add_argument("--mmseqs-url")
    parser.add_argument("--blast-plus-url")
    parser.add_argument("--diamond-url")
    parser.add_argument("--hmmer-url")
    parser.add_argument("--pharokka-database-url", default=DEFAULT_PHAROKKA_DATABASE_URL)
    parser.add_argument("--pharokka-database-md5", default=DEFAULT_PHAROKKA_DATABASE_MD5)
    parser.add_argument("--pharokka-database-release", default=DEFAULT_PHAROKKA_DATABASE_RELEASE)
    parser.add_argument("--amrfinder-url")
    parser.add_argument("--amrfinder-release", default=DEFAULT_AMRFINDER_RELEASE)
    parser.add_argument("--arc-evo2-repo-url", default=DEFAULT_ARC_EVO2_REPO_URL)
    parser.add_argument("--arc-evo2-repo-rev", default=DEFAULT_ARC_EVO2_REPO_REV)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Prepare requested assets and print their locations."""
    args = build_parser().parse_args()
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
        prepare_phrogs_consensus_database=args.prepare_phrogs_consensus_database,
        download_checkv=not args.skip_checkv,
        configure_lovis4u=not args.skip_lovis4u_config,
        with_safety=args.with_safety,
        safety_manifest=args.safety_manifest,
        mmseqs_url=args.mmseqs_url,
        blast_plus_url=args.blast_plus_url,
        diamond_url=args.diamond_url,
        hmmer_url=args.hmmer_url,
        pharokka_database_url=args.pharokka_database_url,
        pharokka_database_md5=args.pharokka_database_md5,
        pharokka_database_release=args.pharokka_database_release,
        amrfinder_url=args.amrfinder_url,
        amrfinder_release=args.amrfinder_release,
        arc_evo2_repo_url=args.arc_evo2_repo_url,
        arc_evo2_repo_rev=args.arc_evo2_repo_rev,
        overwrite=args.overwrite,
    )
    for asset in assets:
        print(f"{asset.name}: {asset.path} ({asset.detail})")
    selected_bin = args.bin_dir or args.external_dir / "bin"
    print(f"export PATH={selected_bin}:$PATH")
    checkv = sorted((args.external_dir / "checkv").glob("checkv-db-*"))
    if checkv:
        print(f"export CHECKVDB={checkv[-1]}")


if __name__ == "__main__":
    main()
