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

"""Tests for ``bionemo.evo2_phage_gen.external_assets``."""

import tarfile
from pathlib import Path

from bionemo.evo2_phage_gen.external_assets import (
    prepare_checkv_database,
    prepare_external_assets,
    prepare_mmseqs_gpu,
    prepare_pyrodigal_wrapper,
)


def _write_tarball(tmp_path: Path) -> Path:
    """Create a tiny MMseqs-like tarball with a bin/mmseqs executable."""
    source_root = tmp_path / "archive_src" / "mmseqs" / "bin"
    source_root.mkdir(parents=True)
    mmseqs = source_root / "mmseqs"
    mmseqs.write_text("#!/usr/bin/env bash\n")
    archive_path = tmp_path / "mmseqs-linux-gpu.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(tmp_path / "archive_src" / "mmseqs", arcname="mmseqs")
    return archive_path


def test_prepare_pyrodigal_wrapper_writes_prodigal_executable(tmp_path):
    """The Prodigal compatibility wrapper should delegate to pyrodigal."""
    asset = prepare_pyrodigal_wrapper(tmp_path / "bin")

    assert asset.path.name == "prodigal"
    assert "exec pyrodigal" in asset.path.read_text()
    assert asset.path.stat().st_mode & 0o111


def test_prepare_mmseqs_gpu_extracts_archive_and_links_binary(tmp_path):
    """A local MMseqs tarball should produce external/bin/mmseqs."""
    archive_path = _write_tarball(tmp_path)

    asset = prepare_mmseqs_gpu(tmp_path / "external", mmseqs_url=archive_path.as_uri())

    assert asset.path.name == "mmseqs"
    assert asset.path.is_symlink()
    assert asset.path.exists()


def test_prepare_external_assets_can_skip_network_downloads(tmp_path):
    """Callers should be able to create only local wrappers in dry environments."""
    assets = prepare_external_assets(
        tmp_path / "external",
        download_mmseqs=False,
        download_phrogs_annotation=False,
        download_large_databases=False,
    )

    assert [asset.name for asset in assets] == ["prodigal_wrapper"]


def test_prepare_checkv_database_invokes_checkv_download(tmp_path, monkeypatch):
    """The CheckV helper should call the installed checkv CLI and discover the DB."""
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))
        db_dir = tmp_path / "external" / "checkv" / "checkv-db-v1.5"
        db_dir.mkdir(parents=True)

    monkeypatch.setattr("subprocess.run", fake_run)

    asset = prepare_checkv_database(tmp_path / "external")

    assert calls == [(["checkv", "download_database", str(tmp_path / "external" / "checkv")], True)]
    assert asset.path.name == "checkv-db-v1.5"
