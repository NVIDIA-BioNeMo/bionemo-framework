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

import hashlib
import io
import tarfile
from pathlib import Path

import pytest
import yaml

from bionemo.evo2_phage_gen.arc_pipeline import ARC_EVO2_GIT_URL, ARC_EVO2_REV
from bionemo.evo2_phage_gen.external_assets import (
    DEFAULT_ARC_EVO2_REPO_REV,
    DEFAULT_ARC_EVO2_REPO_URL,
    DEFAULT_UNIPROT_TOXIN_QUERY,
    PreparedAsset,
    configure_lovis4u_mmseqs,
    prepare_amrfinder_plus,
    prepare_arc_evo2_checkout,
    prepare_checkv_database,
    prepare_diamond,
    prepare_dustmasker,
    prepare_external_assets,
    prepare_hmmer,
    prepare_mmseqs_gpu,
    prepare_phrogs_safety_metadata,
    prepare_pyrodigal_wrapper,
    prepare_toxin_reference,
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


def _write_amrfinder_tarball(tmp_path: Path) -> Path:
    """Create a tiny AMRFinderPlus-like archive for local preparation tests."""
    source_root = tmp_path / "amrfinder_archive_src" / "amrfinder" / "bin"
    source_root.mkdir(parents=True)
    for executable_name in ("amrfinder", "amrfinder_update"):
        executable = source_root / executable_name
        executable.write_text("#!/usr/bin/env bash\n")
        executable.chmod(0o755)
    archive_path = tmp_path / "amrfinder.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(tmp_path / "amrfinder_archive_src", arcname="archive")
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


def test_prepare_amrfinder_plus_extracts_pinned_archive_links_binary_and_records_versions(tmp_path, monkeypatch):
    """Safety preparation records AMRFinder and database versions from a local pinned archive."""
    archive_path = _write_amrfinder_tarball(tmp_path)
    calls = []

    def fake_run(cmd, check, capture_output, text):
        calls.append((cmd, check, capture_output, text))
        if cmd[0].endswith("amrfinder_update"):
            (tmp_path / "external" / "safety" / "amrfinder" / "database").mkdir(parents=True, exist_ok=True)
            return type("Completed", (), {"stdout": ""})()
        if cmd[-1] == "--version":
            return type("Completed", (), {"stdout": "AMRFinderPlus version 4.2.7\n"})()
        return type("Completed", (), {"stdout": "2026-01-26.1\n"})()

    monkeypatch.setattr("subprocess.run", fake_run)
    manifest_path = tmp_path / "external" / "safety" / "asset_manifest.yaml"

    asset = prepare_amrfinder_plus(
        tmp_path / "external",
        amrfinder_url=archive_path.as_uri(),
        manifest_path=manifest_path,
    )

    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert asset.path.name == "amrfinder"
    assert asset.path.is_symlink()
    assert asset.path.exists()
    assert (tmp_path / "external" / "bin" / "amrfinder_update").is_symlink()
    assert "amrfinder_version: AMRFinderPlus version 4.2.7" in manifest_path.read_text()
    assert "database_version: 2026-01-26.1" in manifest_path.read_text()
    assert f"archive_sha256: {archive_digest}" in manifest_path.read_text()
    assert calls == [
        (
            [
                str(tmp_path / "external" / "bin" / "amrfinder_update"),
                "-d",
                str(tmp_path / "external" / "safety" / "amrfinder" / "database"),
            ],
            True,
            True,
            True,
        ),
        ([str(asset.path), "--version"], True, True, True),
        (
            [
                str(asset.path),
                "--database",
                str(tmp_path / "external" / "safety" / "amrfinder" / "database"),
                "--database_version",
            ],
            True,
            True,
            True,
        ),
    ]


def test_prepare_amrfinder_plus_refuses_download_with_wrong_declared_digest(tmp_path):
    """A declared release digest must match the downloaded AMRFinder archive."""
    archive_path = _write_amrfinder_tarball(tmp_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_amrfinder_plus(
            tmp_path / "external",
            amrfinder_url=archive_path.as_uri(),
            amrfinder_sha256="0" * 64,
        )


def test_prepare_toxin_reference_records_uniprot_provenance_and_builds_diamond_database(tmp_path, monkeypatch):
    """A reviewed toxin snapshot has release metadata and digests for every generated input."""
    annotation_payload = b"Entry\tProtein names\nP00001\tExample toxin\n"
    fasta_payload = b">sp|P00001|TOX Example toxin\nMPEPTIDE\n"
    calls = []

    class Response(io.BytesIO):
        def __init__(self, payload):
            super().__init__(payload)
            self.headers = {"X-UniProt-Release": "2026_01"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(url, context=None):
        del context
        return Response(annotation_payload if "format=tsv" in url else fasta_payload)

    def fake_run(cmd, check):
        calls.append((cmd, check))
        Path(cmd[-1]).write_text("DIAMOND database\n")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("subprocess.run", fake_run)
    manifest_path = tmp_path / "external" / "safety" / "asset_manifest.yaml"

    asset = prepare_toxin_reference(tmp_path / "external", manifest_path=manifest_path)

    manifest = yaml.safe_load(manifest_path.read_text())
    assert asset.path.name == "reviewed_toxins.dmnd"
    assert asset.path.exists()
    assert manifest["toxin_reference"]["uniprot_release"] == "2026_01"
    assert manifest["toxin_reference"]["query"] == DEFAULT_UNIPROT_TOXIN_QUERY
    assert manifest["toxin_reference"]["license"] == "CC BY 4.0"
    assert (
        manifest["toxin_reference"]["files"]["annotations"]["sha256"] == hashlib.sha256(annotation_payload).hexdigest()
    )
    assert manifest["toxin_reference"]["files"]["fasta"]["sha256"] == hashlib.sha256(fasta_payload).hexdigest()
    assert calls == [
        (
            [
                str(tmp_path / "external" / "bin" / "diamond"),
                "makedb",
                "--in",
                str(tmp_path / "external" / "safety" / "toxins" / "reviewed_toxins.faa"),
                "--db",
                str(tmp_path / "external" / "safety" / "toxins" / "reviewed_toxins.dmnd"),
            ],
            True,
        )
    ]


def test_prepare_phrogs_safety_metadata_splits_integration_excision_hits_by_confidence(tmp_path):
    """PHROGs integration/excision annotations produce a versioned high/review lookup table."""
    annotation_path = tmp_path / "external" / "phrogs" / "phrog_annot_v4.tsv"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        "phrog\tannot\tcategory\n"
        "phrog_1\tIntegrase\tintegration/excision\n"
        "phrog_2\tSite-specific recombinase\tintegration/excision\n"
        "phrog_3\tLysogeny repressor\tintegration/excision\n"
        "phrog_4\tPutative recombinase\tintegration/excision\n"
        "phrog_5\tTail fiber\tstructural\n"
    )
    manifest_path = tmp_path / "external" / "safety" / "asset_manifest.yaml"

    asset = prepare_phrogs_safety_metadata(tmp_path / "external", manifest_path=manifest_path)

    rows = [line.split("\t") for line in asset.path.read_text().splitlines()]
    assert asset.path.name == "phrogs_integration_excision_v4.tsv"
    assert rows[0] == ["phrog", "annot", "category", "confidence", "matched_term"]
    assert rows[1:] == [
        ["phrog_1", "Integrase", "integration/excision", "high_confidence", "integrase"],
        [
            "phrog_2",
            "Site-specific recombinase",
            "integration/excision",
            "high_confidence",
            "site-specific recombinase",
        ],
        ["phrog_3", "Lysogeny repressor", "integration/excision", "high_confidence", "lysogeny repressor"],
        ["phrog_4", "Putative recombinase", "integration/excision", "review", "recombinase"],
    ]
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["phrogs_v4"]["source_sha256"] == hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    assert manifest["phrogs_v4"]["high_confidence_terms"] == [
        "integrase",
        "excisionase",
        "site-specific recombinase",
        "lysogeny repressor",
    ]


def test_phage_safety_assets_recipe_pins_scanner_sources_and_expected_roles():
    """The tracked recipe fixes scanner provenance while runtime digests stay out of Git."""
    recipe_path = Path(__file__).resolve().parents[3] / "configs" / "phage_safety_assets.yaml"

    recipe = yaml.safe_load(recipe_path.read_text())

    assert recipe["amrfinder_plus"]["release"] == "amrfinder_v4.2.7"
    assert recipe["toxin_reference"]["query"] == DEFAULT_UNIPROT_TOXIN_QUERY
    assert recipe["toxin_reference"]["license"] == "CC BY 4.0"
    assert recipe["phrogs_v4"]["annotation_url"].endswith("phrog_annot_v4.tsv")
    assert recipe["phrogs_v4"]["high_confidence_terms"] == [
        "integrase",
        "excisionase",
        "site-specific recombinase",
        "lysogeny repressor",
    ]
    assert set(recipe["expected_file_roles"]) >= {
        "amrfinder_binary",
        "amrfinder_database",
        "toxin_annotations",
        "toxin_fasta",
        "toxin_diamond_database",
        "phrogs_lysogeny_table",
    }


def test_prepare_external_assets_with_safety_prepares_all_safety_assets_after_prerequisites(tmp_path, monkeypatch):
    """The opt-in safety path runs AMR, toxin, then PHROGs preparation after regular assets."""
    calls = []

    def fake_asset(name):
        def prepare(*_args, **_kwargs):
            calls.append(name)
            return PreparedAsset(name, tmp_path / name, name)

        return prepare

    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_pyrodigal_wrapper", fake_asset("prodigal"))
    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_diamond", fake_asset("diamond"))
    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_hmmer", fake_asset("hmmer"))
    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_amrfinder_plus", fake_asset("amrfinder"))
    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_toxin_reference", fake_asset("toxins"))
    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_phrogs_safety_metadata", fake_asset("phrogs"))

    assets = prepare_external_assets(
        tmp_path / "external",
        download_mmseqs=False,
        download_dustmasker=False,
        download_diamond=True,
        download_hmmer=True,
        download_phrogs_annotation=False,
        download_arc_evo2=False,
        configure_lovis4u=False,
        with_safety=True,
    )

    assert [asset.name for asset in assets] == ["prodigal", "diamond", "hmmer", "amrfinder", "toxins", "phrogs"]
    assert calls == ["prodigal", "diamond", "hmmer", "amrfinder", "toxins", "phrogs"]


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
