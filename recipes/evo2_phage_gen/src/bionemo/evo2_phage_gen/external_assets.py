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
import os
import shutil
import ssl
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class PreparedAsset:
    """Single prepared external asset."""

    name: str
    path: Path
    detail: str


def _download(url: str, output_path: Path, *, overwrite: bool = False, insecure: bool = False) -> Path:
    """Download ``url`` to ``output_path`` unless it already exists."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    context = ssl._create_unverified_context() if insecure else None
    with urllib.request.urlopen(url, context=context) as response, tmp_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    tmp_path.replace(output_path)
    return output_path


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
