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

import pytest

from bionemo.evo2_phage_gen.arc_pipeline import ARC_EVO2_GIT_URL, ARC_EVO2_REV
from bionemo.evo2_phage_gen.external_assets import (
    DEFAULT_ARC_EVO2_REPO_REV,
    DEFAULT_ARC_EVO2_REPO_URL,
    configure_lovis4u_mmseqs,
    prepare_arc_evo2_checkout,
    prepare_checkv_database,
    prepare_diamond,
    prepare_dustmasker,
    prepare_external_assets,
    prepare_hmmer,
    prepare_mmseqs_gpu,
    prepare_pyrodigal_wrapper,
)


def _write_tarball(tmp_path: Path, executable_name: str = "mmseqs", subdir: str = "mmseqs/bin") -> Path:
    """Create a tiny tool-like tarball with an executable."""
    source_root = tmp_path / "archive_src" / subdir
    source_root.mkdir(parents=True)
    executable = source_root / executable_name
    executable.write_text("#!/usr/bin/env bash\n")
    executable.chmod(0o755)
    archive_path = tmp_path / f"{executable_name}-linux64.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(tmp_path / "archive_src", arcname="archive")
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


def test_prepare_dustmasker_extracts_blast_plus_archive_and_links_binary(tmp_path):
    """A local BLAST+ tarball should produce external/bin/dustmasker."""
    archive_path = _write_tarball(tmp_path, executable_name="dustmasker", subdir="ncbi-blast/bin")

    asset = prepare_dustmasker(tmp_path / "external", blast_plus_url=archive_path.as_uri())

    assert asset.path.name == "dustmasker"
    assert asset.path.is_symlink()
    assert asset.path.exists()


def test_configure_lovis4u_mmseqs_points_lovis4u_at_recipe_binary(tmp_path, monkeypatch):
    """LoVis4u needs an explicit Linux MMseqs path before synteny scoring works."""
    mmseqs_bin = tmp_path / "bin" / "mmseqs"
    mmseqs_bin.parent.mkdir()
    mmseqs_bin.write_text("#!/usr/bin/env bash\n")
    mmseqs_bin.chmod(0o755)
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))

    monkeypatch.setattr("subprocess.run", fake_run)

    asset = configure_lovis4u_mmseqs(mmseqs_bin)

    assert asset.name == "lovis4u_mmseqs_config"
    assert asset.path == mmseqs_bin
    assert calls == [
        (["lovis4u", "--linux"], True),
        (["lovis4u", "-smp", str(mmseqs_bin.resolve())], True),
    ]


def test_prepare_diamond_extracts_archive_and_links_binary(tmp_path):
    """A local DIAMOND tarball should produce external/bin/diamond."""
    archive_path = _write_tarball(tmp_path, executable_name="diamond", subdir="")

    asset = prepare_diamond(tmp_path / "external", diamond_url=archive_path.as_uri())

    assert asset.path.name == "diamond"
    assert asset.path.is_symlink()
    assert asset.path.exists()


def test_prepare_hmmer_extracts_archive_and_links_hmmsearch(tmp_path):
    """A local HMMER tarball should produce bin/hmmsearch."""
    archive_path = _write_tarball(tmp_path, executable_name="hmmsearch", subdir="bin")

    asset = prepare_hmmer(tmp_path / "external", bin_dir=tmp_path / "venv" / "bin", hmmer_url=archive_path.as_uri())

    assert asset.path.name == "hmmsearch"
    assert asset.path.is_symlink()
    assert asset.path.exists()
    assert asset.path.parent == tmp_path / "venv" / "bin"


def test_prepare_external_assets_can_skip_network_downloads(tmp_path):
    """Callers should be able to create only local wrappers in dry environments."""
    assets = prepare_external_assets(
        tmp_path / "external",
        download_mmseqs=False,
        download_dustmasker=False,
        download_diamond=False,
        download_hmmer=False,
        download_phrogs_annotation=False,
        download_arc_evo2=False,
        download_large_databases=False,
        configure_lovis4u=False,
    )

    assert [asset.name for asset in assets] == ["prodigal_wrapper"]


def test_prepare_arc_evo2_checkout_clones_single_pinned_revision(tmp_path, monkeypatch):
    """Arc checkout preparation should use the same pinned revision as the maintained patch."""
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))
        if cmd[:2] == ["git", "clone"]:
            Path(cmd[-1]).mkdir(parents=True)

    monkeypatch.setattr("subprocess.run", fake_run)

    asset = prepare_arc_evo2_checkout(tmp_path / "external")

    checkout_dir = tmp_path / "external" / "arc_evo2"
    assert DEFAULT_ARC_EVO2_REPO_URL == ARC_EVO2_GIT_URL
    assert DEFAULT_ARC_EVO2_REPO_REV == ARC_EVO2_REV
    assert asset.path == checkout_dir
    assert calls == [
        (["git", "clone", "--filter=blob:none", ARC_EVO2_GIT_URL, str(checkout_dir)], True),
        (["git", "-C", str(checkout_dir), "checkout", ARC_EVO2_REV], True),
    ]


def test_prepare_arc_evo2_checkout_rejects_existing_wrong_revision(tmp_path, monkeypatch):
    """Existing Arc checkouts should not silently drift away from the patch revision."""
    checkout_dir = tmp_path / "external" / "arc_evo2"
    checkout_dir.mkdir(parents=True)
    monkeypatch.setattr("bionemo.evo2_phage_gen.arc_pipeline._git_head", lambda path: "wrong-revision")

    with pytest.raises(RuntimeError, match=ARC_EVO2_REV):
        prepare_arc_evo2_checkout(tmp_path / "external")


def test_prepare_external_assets_configures_lovis4u_when_mmseqs_is_prepared(tmp_path, monkeypatch):
    """The default MMseqs setup should also configure LoVis4u for synteny scoring."""
    archive_path = _write_tarball(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))

    monkeypatch.setattr("subprocess.run", fake_run)

    assets = prepare_external_assets(
        tmp_path / "external",
        download_mmseqs=True,
        download_diamond=False,
        download_hmmer=False,
        download_phrogs_annotation=False,
        download_arc_evo2=False,
        download_large_databases=False,
        download_dustmasker=False,
        mmseqs_url=archive_path.as_uri(),
    )

    assert [asset.name for asset in assets] == ["prodigal_wrapper", "mmseqs2_gpu", "lovis4u_mmseqs_config"]
    assert calls == [
        (["lovis4u", "--linux"], True),
        (["lovis4u", "-smp", str(assets[1].path.resolve())], True),
    ]


def test_prepare_external_assets_can_target_venv_bin(tmp_path):
    """Small native tools should be installable into ``.venv/bin`` for CI."""
    diamond_archive = _write_tarball(tmp_path / "diamond", executable_name="diamond", subdir="")
    hmmer_archive = _write_tarball(tmp_path / "hmmer", executable_name="hmmsearch", subdir="bin")
    assets = prepare_external_assets(
        tmp_path / "external",
        bin_dir=tmp_path / ".venv" / "bin",
        download_mmseqs=False,
        download_dustmasker=False,
        download_diamond=True,
        download_hmmer=True,
        download_phrogs_annotation=False,
        download_arc_evo2=False,
        download_large_databases=False,
        diamond_url=diamond_archive.as_uri(),
        hmmer_url=hmmer_archive.as_uri(),
    )

    assert [asset.name for asset in assets] == ["prodigal_wrapper", "diamond", "hmmer"]
    assert (tmp_path / ".venv" / "bin" / "diamond").exists()
    assert (tmp_path / ".venv" / "bin" / "hmmsearch").exists()


def test_prepare_checkv_database_invokes_checkv_download(tmp_path, monkeypatch):
    """The CheckV helper should call the installed checkv CLI and discover the DB."""
    calls = []

    def fake_run(cmd, check, env):
        calls.append((cmd, check, env["PATH"]))
        db_dir = tmp_path / "external" / "checkv" / "checkv-db-v1.5"
        db_dir.mkdir(parents=True)

    monkeypatch.setattr("subprocess.run", fake_run)

    asset = prepare_checkv_database(tmp_path / "external")

    assert len(calls) == 1
    assert calls[0][:2] == (["checkv", "download_database", str(tmp_path / "external" / "checkv")], True)
    assert calls[0][2].startswith(str(tmp_path / "external" / "bin"))
    assert asset.path.name == "checkv-db-v1.5"
