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
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_DIR = RECIPE_ROOT / "data" / "external"
DEFAULT_BIN_DIR = DEFAULT_EXTERNAL_DIR / "bin"
DEFAULT_MMSEQS_GPU_URL = "https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
DEFAULT_PHROGS_ANNOTATION_URL = "https://phrogs.lmge.uca.fr/downloads_from_website/phrog_annot_v4.tsv"
DEFAULT_PHROGS_MMSEQS_URL = "https://phrogs.lmge.uca.fr/downloads_from_website/phrogs_mmseqs_db.tar.gz"
DEFAULT_ARC_EVO2_REPO_URL = "https://github.com/ArcInstitute/evo2.git"


@dataclass(frozen=True)
class PreparedAsset:
    """Single prepared external asset."""

    name: str
    path: Path
    detail: str


def _download(url: str, output_path: Path, *, overwrite: bool = False) -> Path:
    """Download ``url`` to ``output_path`` unless it already exists."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp_path)
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
    mmseqs_url: str = DEFAULT_MMSEQS_GPU_URL,
    overwrite: bool = False,
) -> PreparedAsset:
    """Download and expose the official MMseqs2-GPU Linux binary."""
    external_dir = Path(external_dir)
    archive_path = _download(mmseqs_url, external_dir / "downloads" / Path(mmseqs_url).name, overwrite=overwrite)
    extracted_dir = _extract_tar(archive_path, external_dir / "tools" / "mmseqs2-gpu", overwrite=overwrite)
    mmseqs_candidates = sorted(extracted_dir.glob("**/bin/mmseqs")) + sorted(extracted_dir.glob("**/mmseqs"))
    if not mmseqs_candidates:
        raise FileNotFoundError(f"No mmseqs binary found after extracting {archive_path} to {extracted_dir}")
    mmseqs_bin = mmseqs_candidates[0]
    link_path = external_dir / "bin" / "mmseqs"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(mmseqs_bin)
    return PreparedAsset("mmseqs2_gpu", link_path, f"downloaded from {mmseqs_url}")


def prepare_phrogs_annotation(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    annotation_url: str = DEFAULT_PHROGS_ANNOTATION_URL,
    overwrite: bool = False,
) -> PreparedAsset:
    """Download PHROGs v4 annotation table."""
    annotation_path = _download(
        annotation_url,
        Path(external_dir) / "phrogs" / "phrog_annot_v4.tsv",
        overwrite=overwrite,
    )
    return PreparedAsset("phrogs_annotation", annotation_path, f"downloaded from {annotation_url}")


def prepare_phrogs_mmseqs_db(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    phrogs_mmseqs_url: str = DEFAULT_PHROGS_MMSEQS_URL,
    overwrite: bool = False,
) -> PreparedAsset:
    """Download and extract PHROGs MMseqs profile database."""
    external_dir = Path(external_dir)
    archive_path = _download(
        phrogs_mmseqs_url,
        external_dir / "downloads" / Path(phrogs_mmseqs_url).name,
        overwrite=overwrite,
    )
    extracted_dir = _extract_tar(archive_path, external_dir / "phrogs" / "phrogs_mmseqs_db", overwrite=overwrite)
    return PreparedAsset("phrogs_mmseqs_db", extracted_dir, f"downloaded from {phrogs_mmseqs_url}")


def prepare_checkv_database(external_dir: Path = DEFAULT_EXTERNAL_DIR, *, overwrite: bool = False) -> PreparedAsset:
    """Download CheckV's database with the installed ``checkv`` CLI."""
    checkv_root = Path(external_dir) / "checkv"
    existing = sorted(checkv_root.glob("checkv-db-*")) if checkv_root.exists() else []
    if existing and not overwrite:
        return PreparedAsset("checkv_database", existing[-1], "existing CheckV database")
    if overwrite and checkv_root.exists():
        shutil.rmtree(checkv_root)
    checkv_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["checkv", "download_database", str(checkv_root)], check=True)
    downloaded = sorted(checkv_root.glob("checkv-db-*"))
    if not downloaded:
        raise FileNotFoundError(f"checkv download_database did not create checkv-db-* under {checkv_root}")
    return PreparedAsset("checkv_database", downloaded[-1], "downloaded with checkv download_database")


def prepare_arc_evo2_checkout(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    repo_url: str = DEFAULT_ARC_EVO2_REPO_URL,
    overwrite: bool = False,
) -> PreparedAsset:
    """Clone Arc's Evo2 repository for phage reference data and QC scripts."""
    checkout_dir = Path(external_dir) / "arc_evo2"
    if checkout_dir.exists() and not overwrite:
        return PreparedAsset("arc_evo2", checkout_dir, "existing Arc Evo2 checkout")
    if checkout_dir.exists() and overwrite:
        shutil.rmtree(checkout_dir)
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(checkout_dir)], check=True)
    return PreparedAsset("arc_evo2", checkout_dir, f"cloned from {repo_url}")


def prepare_external_assets(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    *,
    download_mmseqs: bool = True,
    download_phrogs_annotation: bool = True,
    download_arc_evo2: bool = True,
    download_large_databases: bool = False,
    mmseqs_url: str = DEFAULT_MMSEQS_GPU_URL,
    phrogs_mmseqs_url: str = DEFAULT_PHROGS_MMSEQS_URL,
    arc_evo2_repo_url: str = DEFAULT_ARC_EVO2_REPO_URL,
    overwrite: bool = False,
) -> list[PreparedAsset]:
    """Prepare external assets for Arc's phage QC pipeline."""
    external_dir = Path(external_dir)
    assets = [prepare_pyrodigal_wrapper(external_dir / "bin")]
    if download_mmseqs:
        assets.append(prepare_mmseqs_gpu(external_dir, mmseqs_url=mmseqs_url, overwrite=overwrite))
    if download_phrogs_annotation:
        assets.append(prepare_phrogs_annotation(external_dir, overwrite=overwrite))
    if download_arc_evo2:
        assets.append(prepare_arc_evo2_checkout(external_dir, repo_url=arc_evo2_repo_url, overwrite=overwrite))
    if download_large_databases:
        assets.append(prepare_phrogs_mmseqs_db(external_dir, phrogs_mmseqs_url=phrogs_mmseqs_url, overwrite=overwrite))
        assets.append(prepare_checkv_database(external_dir, overwrite=overwrite))
    return assets


def main() -> None:
    """CLI entry point for preparing external analysis assets."""
    parser = argparse.ArgumentParser(description="Prepare external tools/databases for Evo2 phage QC")
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--skip-mmseqs", action="store_true", help="Do not download MMseqs2-GPU")
    parser.add_argument("--skip-phrogs-annotation", action="store_true", help="Do not download PHROGs annotation TSV")
    parser.add_argument("--skip-arc-evo2", action="store_true", help="Do not clone Arc's Evo2 repository")
    parser.add_argument(
        "--download-large-databases", action="store_true", help="Also download PHROGs MMseqs DB and CheckV DB"
    )
    parser.add_argument("--mmseqs-url", default=DEFAULT_MMSEQS_GPU_URL)
    parser.add_argument("--phrogs-mmseqs-url", default=DEFAULT_PHROGS_MMSEQS_URL)
    parser.add_argument("--arc-evo2-repo-url", default=DEFAULT_ARC_EVO2_REPO_URL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    assets = prepare_external_assets(
        args.external_dir,
        download_mmseqs=not args.skip_mmseqs,
        download_phrogs_annotation=not args.skip_phrogs_annotation,
        download_arc_evo2=not args.skip_arc_evo2,
        download_large_databases=args.download_large_databases,
        mmseqs_url=args.mmseqs_url,
        phrogs_mmseqs_url=args.phrogs_mmseqs_url,
        arc_evo2_repo_url=args.arc_evo2_repo_url,
        overwrite=args.overwrite,
    )
    for asset in assets:
        print(f"{asset.name}: {asset.path} ({asset.detail})")
    print(f"export PATH={Path(args.external_dir) / 'bin'}:$PATH")
    checkv_dirs = sorted((Path(args.external_dir) / "checkv").glob("checkv-db-*"))
    if checkv_dirs:
        print(f"export CHECKVDB={checkv_dirs[-1]}")


if __name__ == "__main__":
    main()
