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

import hashlib
import io
import json
from pathlib import Path
from typing import ClassVar

import pytest

from bionemo.evo2_phage_gen import external_assets as assets
from bionemo.evo2_phage_gen.external_assets import (
    PreparedAsset,
    prepare_external_assets,
    prepare_phrogs_lookup,
    prepare_pyrodigal_wrapper,
)


class _Response(io.BytesIO):
    status = 200
    headers: ClassVar[dict[str, str]] = {"X-Release": "current"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_download_checks_provider_checksum_at_download_boundary(tmp_path: Path, monkeypatch) -> None:
    payload = b"provider archive"
    expected = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda _request: _Response(payload))

    output, headers = assets._download(
        "https://example.test/archive.tar.gz",
        tmp_path / "archive.tar.gz",
        published_md5=expected,
    )
    assert output.read_bytes() == payload
    assert headers["x-release"] == "current"

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda _request: _Response(b"changed"))
    with pytest.raises(ValueError, match="Published checksum"):
        assets._download(
            "https://example.test/archive.tar.gz",
            tmp_path / "changed.tar.gz",
            published_md5=expected,
        )


def test_pyrodigal_wrapper_is_a_normal_executable(tmp_path: Path) -> None:
    prepared = prepare_pyrodigal_wrapper(tmp_path / "bin")
    assert prepared.path.stat().st_mode & 0o111
    assert "exec pyrodigal" in prepared.path.read_text()


def test_phrogs_lookup_uses_available_profiles_and_tolerates_database_growth(tmp_path: Path) -> None:
    database = tmp_path / "phrogs_profile_db"
    database.write_text("db")
    Path(f"{database}.dbtype").write_text("type")
    Path(f"{database}.lookup").write_text("0\tphrog_1\t0\n1\tphrog_99\t0\n")
    annotations = tmp_path / "phrogs.tsv"
    annotations.write_text(
        "phrog\tannot\tcategory\n"
        "1\tintegrase\tintegration and excision\n"
        "2\texcisionase added by a newer release\tintegration and excision\n"
        "99\tcapsid\thead and packaging\n"
    )

    prepared = prepare_phrogs_lookup(annotations, database, tmp_path / "lysogeny.tsv")
    text = prepared.path.read_text()
    assert "phrog_1" in text
    assert "phrog_2" not in text
    assert "phrog_99" not in text


def test_phrogs_lookup_records_review_and_high_confidence_families(tmp_path: Path) -> None:
    database = tmp_path / "phrogs_profile_db"
    database.write_text("db")
    Path(f"{database}.dbtype").write_text("type")
    Path(f"{database}.lookup").write_text("0\tphrog_1\t0\n1\tphrog_2\t0\n")
    annotations = tmp_path / "phrogs.tsv"
    annotations.write_text(
        "phrog\tannot\tcategory\n"
        "1\tintegrase\tintegration and excision\n"
        "2\tunknown protein\tintegration and excision\n"
    )

    output = prepare_phrogs_lookup(annotations, database, tmp_path / "lookup.tsv").path.read_text()
    assert "high_confidence" in output
    assert "review" in output


def test_minimal_asset_preparation_can_skip_network_work(tmp_path: Path) -> None:
    prepared = prepare_external_assets(
        tmp_path,
        download_mmseqs=False,
        download_dustmasker=False,
        download_diamond=False,
        download_hmmer=False,
        download_phrogs_annotation=False,
        download_arc_evo2=False,
        download_large_databases=False,
        download_checkv=False,
        configure_lovis4u=False,
    )
    assert [item.name for item in prepared] == ["prodigal"]


def test_safety_state_records_current_versions_and_paths(tmp_path: Path, monkeypatch) -> None:
    external = tmp_path / "external"
    bin_dir = external / "bin"
    amr_dir = external / "safety" / "amrfinder"
    toxin_dir = external / "safety" / "toxins"
    phrogs_dir = external / "phrogs"
    for directory in (bin_dir, amr_dir, toxin_dir, phrogs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for tool in ("amrfinder", "diamond", "mmseqs"):
        (bin_dir / tool).write_text(tool)
    (amr_dir / "state.json").write_text(
        json.dumps(
            {
                "tool_version": "4.2.7",
                "database_path": str(amr_dir / "2026-08"),
                "database_version": "2026-08",
                "release_url": "https://example.test/amr",
            }
        )
    )
    (toxin_dir / "state.json").write_text(
        json.dumps(
            {
                "release": "UniProt current",
                "diamond_database_path": str(toxin_dir / "toxins.dmnd"),
            }
        )
    )
    profile = phrogs_dir / "profiles"
    lookup = phrogs_dir / "lookup.tsv"
    profile.write_text("db")
    lookup.write_text("lookup")
    monkeypatch.setattr(assets, "_tool_version", lambda path, *args: f"{path.name} current")

    state = assets._safety_state(
        external,
        bin_dir,
        PreparedAsset("profile", profile, "PHROGs current"),
        PreparedAsset("lookup", lookup, "one family"),
    )
    assert state["tools"]["diamond"]["version"] == "diamond current"
    assert state["databases"]["phrogs"]["release"] == "PHROGs current"


def test_parser_allows_current_database_overrides() -> None:
    parsed = assets.build_parser().parse_args(
        [
            "--with-safety",
            "--phrogs-profile-url",
            "https://example.test/new-profiles.tar.gz",
            "--phrogs-profile-md5",
            "provider-value",
            "--phrogs-profile-release",
            "PHROGs newer",
            "--amrfinder-url",
            "https://example.test/amr.tar.gz",
            "--amrfinder-release",
            "amrfinder newer",
        ]
    )
    assert parsed.phrogs_profile_release == "PHROGs newer"
    assert parsed.amrfinder_release == "amrfinder newer"
