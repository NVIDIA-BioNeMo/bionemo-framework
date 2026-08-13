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
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

import bionemo.evo2_phage_gen.external_assets as external_assets
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


def _write_multi_executable_tarball(
    tmp_path: Path,
    executable_names: tuple[str, ...],
    *,
    subdir: str,
    archive_name: str,
) -> Path:
    """Create a tiny archive containing a coherent set of executable tools."""
    source_root = tmp_path / f"{archive_name}_src" / subdir
    source_root.mkdir(parents=True)
    for executable_name in executable_names:
        executable = source_root / executable_name
        executable.write_text("#!/usr/bin/env bash\n")
        executable.chmod(0o755)
    archive_path = tmp_path / archive_name
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(tmp_path / f"{archive_name}_src", arcname="archive")
    return archive_path


def _write_amrfinder_runtime(source_root: Path, *, label: str) -> None:
    """Create the complete sibling runtime expected by AMRFinderPlus 4.2.7."""
    source_root.mkdir(parents=True, exist_ok=True)
    for executable_name in external_assets.AMRFINDER_RUNTIME_EXECUTABLES:
        executable = source_root / executable_name
        executable.write_text(f"#!/usr/bin/env bash\n# {label} {executable_name}\n")
        executable.chmod(0o755)
    for file_name in external_assets.AMRFINDER_RUNTIME_DATA_FILES:
        (source_root / file_name).write_text(f"{label} {file_name}\n")


def _write_amrfinder_prerequisites(bin_dir: Path, *, label: str = "prerequisite") -> dict[str, Path]:
    """Create the complete BLAST+/HMMER executable set invoked by AMRFinder."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in (
        *external_assets.AMRFINDER_BLAST_RUNTIME_EXECUTABLES,
        *external_assets.AMRFINDER_HMMER_RUNTIME_EXECUTABLES,
    ):
        path = bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n# {label} {name}\n")
        path.chmod(0o755)
        paths[name] = path
    return paths


def _write_amrfinder_tarball(tmp_path: Path) -> Path:
    """Create a tiny release-shaped AMRFinderPlus archive for local preparation tests."""
    source_root = tmp_path / "amrfinder_archive_src" / "amrfinder" / "bin"
    _write_amrfinder_runtime(source_root, label="trusted")
    stx_root = source_root / "stx"
    stx_root.mkdir()
    for name in ("stxtyper", "stx.prot"):
        (source_root / name).replace(stx_root / name)
    archive_path = tmp_path / "amrfinder.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(tmp_path / "amrfinder_archive_src", arcname="archive")
    return archive_path


def test_amrfinder_source_identity_excludes_path_bearing_generated_indexes(tmp_path):
    """Dataset identity stays stable when BLAST/HMMER indexes embed random build paths."""
    database_dirs = [tmp_path / "first" / "2026-08-07.1", tmp_path / "second" / "2026-08-07.1"]
    for database_dir in database_dirs:
        database_dir.mkdir(parents=True)
        (database_dir / "version.txt").write_text("2026-08-07.1\n")
        (database_dir / "database_format_version.txt").write_text("2026-01-26.1\n")
        (database_dir / "AMRProt.fa").write_text(">AMR\nMPEPTIDE\n")
        (database_dir / "AMR.LIB").write_text("HMMER3/f\n")
        (database_dir / "fam.tsv").write_text("family\tname\n")
    (database_dirs[0] / "AMRProt.fa.pin").write_bytes(b"/tmp/amrfinder_index.AAAAAA/db")
    (database_dirs[1] / "AMRProt.fa.pin").write_bytes(b"/tmp/amrfinder_index.BBBBBB/db")

    identities = [external_assets._amrfinder_database_source_identity(path) for path in database_dirs]

    assert identities[0] == identities[1]
    assert identities[0][1] == ["AMR.LIB", "AMRProt.fa", "database_format_version.txt", "fam.tsv", "version.txt"]
    assert external_assets._sha256_path(database_dirs[0]) != external_assets._sha256_path(database_dirs[1])


def _write_minimal_amrfinder_database_sources(database_dir: Path) -> None:
    """Materialize the path-independent source subset required by AMRFinder fixtures."""
    database_dir.mkdir(parents=True, exist_ok=True)
    (database_dir / "version.txt").write_text(f"{database_dir.name}\n")
    (database_dir / "database_format_version.txt").write_text("2.0\n")
    (database_dir / "AMRProt.fa").write_text(">AMR\nMPEPTIDE\n")
    (database_dir / "AMR.LIB").write_text("HMMER3/f\n")
    (database_dir / "fam.tsv").write_text("family\tname\n")


def _write_mmseqs_padded_database(sequence_database: Path) -> Path:
    """Write the complete MMseqs sequence/header database set made by ``makepaddedseqdb``."""
    sequence_database.parent.mkdir(parents=True, exist_ok=True)
    files = {
        sequence_database: b"MPEPTIDE\n\0",
        Path(f"{sequence_database}.index"): b"0\t0\t10\n",
        Path(f"{sequence_database}.dbtype"): b"\x00\x00\x00\x00",
        Path(f"{sequence_database}_h"): b"phrog_1\n\0",
        Path(f"{sequence_database}_h.index"): b"0\t0\t9\n",
        Path(f"{sequence_database}_h.dbtype"): b"\x0c\x00\x00\x00",
        Path(f"{sequence_database}.lookup"): b"0\tphrog_1\t0\n",
    }
    for path, content in files.items():
        path.write_bytes(content)
    return sequence_database


def _write_phrogs_profile_database(
    profile_database: Path,
    *,
    profile_ids: tuple[str, ...] = (
        "phrog_1",
        "phrog_2",
        "phrog_3",
        "phrog_4",
        "phrog_5",
        "phrog_6",
    ),
) -> Path:
    """Write a complete Pharokka v1.8.0-style PHROGs MMseqs profile database."""
    profile_database.parent.mkdir(parents=True, exist_ok=True)
    lookup_rows = b"".join(f"{index}\t{profile_id}\t0\n".encode() for index, profile_id in enumerate(profile_ids))
    files = {
        profile_database: b"profile\n\0",
        Path(f"{profile_database}.index"): b"0\t0\t8\n",
        Path(f"{profile_database}.dbtype"): b"\x10\x00\x00\x00",
        Path(f"{profile_database}_h"): b"\n".join(profile_id.encode() for profile_id in profile_ids) + b"\n\0",
        Path(f"{profile_database}_h.index"): b"0\t0\t9\n",
        Path(f"{profile_database}_h.dbtype"): b"\x0c\x00\x00\x00",
        Path(f"{profile_database}.lookup"): lookup_rows,
        Path(f"{profile_database}.source"): b"pharokka-v1.8.0\n",
        profile_database.parent / "VERSION_1_8_0": b"1.8.0\n",
    }
    for path, content in files.items():
        path.write_bytes(content)
    return profile_database


def _profile_metadata_kwargs(profile_database: Path, archive_path: Path) -> dict[str, str]:
    """Return complete local provenance arguments for PHROGs profile metadata tests."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        archive_path.write_bytes(b"PHROGs Pharokka profile archive\n")
    return {
        "profile_source_url": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL,
        "profile_archive_observed_sha256": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SHA256,
        "profile_retrieved_at": "2026-08-08T00:00:00Z",
        "profile_release": external_assets.PHROGS_PROFILE_RELEASE,
    }


def test_curated_phrogs_search_database_is_exactly_derived_from_policy_and_full_profile(tmp_path):
    """A smaller search DB is safe only when its selected IDs are exactly the classified policy set."""
    full = _write_phrogs_profile_database(
        tmp_path / "full" / external_assets.PHROGS_PROFILE_DATABASE_NAME,
        profile_ids=("phrog_1", "phrog_2", "phrog_3", "phrog_4"),
    )
    policy = tmp_path / "phrogs_policy.tsv"
    policy.write_text(
        "phrog\tannot\tcategory\tconfidence\tmatched_term\n"
        "phrog_3\tthird\tintegration and excision\treview\tintegrase\n"
        "phrog_1\tfirst\tintegration and excision\thigh_confidence\trecombinase\n"
    )
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_bytes(b"mmseqs-v15")
    mmseqs.chmod(0o755)
    commands = []

    def runner(command, **kwargs):
        commands.append((command, kwargs))
        assert command[:2] == [str(mmseqs), "createsubdb"]
        key_list, source, destination = map(Path, command[2:])
        assert source == full
        assert key_list.read_text() == "0\n2\n"
        destination.write_bytes(b"derived-profile-bytes\n")
        Path(f"{destination}.dbtype").write_bytes(Path(f"{source}.dbtype").read_bytes())
        Path(f"{destination}.index").write_text("0\t0\t8\n2\t8\t8\n")
        for suffix in ("_h", "_h.dbtype", "_h.index", ".lookup", ".source"):
            Path(f"{destination}{suffix}").symlink_to(Path(f"{source}{suffix}"))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    record = external_assets._prepare_phrogs_safety_search_database(
        profile_database=full,
        safety_lookup=policy,
        output_root=tmp_path / "derived",
        mmseqs_path=mmseqs,
        runner=runner,
    )

    derived = Path(record["path"])
    assert len(commands) == 1
    assert record["selected_profile_id_inventory"]["count"] == 2
    assert Path(record["profile_ids_path"]).read_text() == "phrog_1\nphrog_3\n"
    assert Path(record["numeric_keys_path"]).read_text() == "0\n2\n"
    assert all(not path.is_symlink() and path.is_file() for path in map(Path, record["files"]))
    assert Path(f"{derived}.lookup").read_bytes() == Path(f"{full}.lookup").read_bytes()
    assert Path(f"{derived}_h").read_bytes() == Path(f"{full}_h").read_bytes()
    external_assets._validate_phrogs_safety_search_database(
        record,
        profile_database=full,
        safety_lookup=policy,
        verify_asset_paths=True,
    )

    derived.write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="search database digest"):
        external_assets._validate_phrogs_safety_search_database(
            record,
            profile_database=full,
            safety_lookup=policy,
            verify_asset_paths=True,
        )


def test_curated_phrogs_search_database_rejects_unknown_or_duplicate_policy_ids(tmp_path):
    full = _write_phrogs_profile_database(tmp_path / "full" / external_assets.PHROGS_PROFILE_DATABASE_NAME)
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_bytes(b"mmseqs")
    mmseqs.chmod(0o755)
    policy = tmp_path / "policy.tsv"

    for rows, message in (
        (("phrog_1", "phrog_1"), "duplicate"),
        (("phrog_1", "phrog_999"), "absent"),
    ):
        policy.write_text(
            "phrog\tannot\tcategory\tconfidence\tmatched_term\n"
            + "".join(f"{profile}\tx\tintegration and excision\treview\tx\n" for profile in rows)
        )
        with pytest.raises((RuntimeError, ValueError), match=message):
            external_assets._prepare_phrogs_safety_search_database(
                profile_database=full,
                safety_lookup=policy,
                output_root=tmp_path / f"derived-{message}",
                mmseqs_path=mmseqs,
                runner=lambda *_args, **_kwargs: None,
            )


def _write_phrogs_v4_fixture(annotation_path: Path, *, duplicate: bool = False, empty: bool = False) -> Path:
    """Write a small real-schema PHROGs v4 fixture plus complete raw and profile DBs."""
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["phrog\tcolor\tannot\tcategory"]
    if not empty:
        rows.extend(
            [
                "1\t#000000\tIntegrase\tintegration and excision",
                "2\t#000001\tSite-specific recombinase\tintegration and excision",
                "3\t#000002\tLysogeny repressor\tintegration and excision",
                "4\t#000003\tPutative recombinase\tintegration and excision",
                "5\t#000004\tTail fiber\tstructural",
                "6\t#000005\tAnti-repressor\ttranscription regulation",
            ]
        )
    if duplicate:
        rows.append("phrog_1\t#000005\tIntegrase\tintegration and excision")
    annotation_path.write_text("\n".join(rows) + "\n")
    _write_phrogs_profile_database(annotation_path.parent / "phrogs_mmseqs_db" / "phrogs_profile_db")
    archive_path = (
        annotation_path.parent.parent / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        archive_path.write_bytes(b"PHROGs Pharokka profile archive\n")
    return _write_mmseqs_padded_database(annotation_path.parent / "phrogs_gpu_seq_db_pad")


def _mock_phrogs_search_database_builder(monkeypatch, external_dir: Path) -> None:
    """Run the real derived-database builder around a tiny deterministic createsubdb boundary."""
    original_prepare_search_database = external_assets._prepare_phrogs_safety_search_database
    mmseqs = Path(external_dir) / "bin" / "mmseqs"
    mmseqs.parent.mkdir(parents=True, exist_ok=True)
    mmseqs.write_bytes(b"synthetic-mmseqs-createsubdb")
    mmseqs.chmod(0o755)

    def prepare_search_database(**kwargs):
        def createsubdb(command, **_run_kwargs):
            source = Path(command[-2])
            destination = Path(command[-1])
            keys = Path(command[-3]).read_text().splitlines()
            destination.write_bytes(b"synthetic-curated-profile-database\n")
            Path(f"{destination}.dbtype").write_bytes(Path(f"{source}.dbtype").read_bytes())
            Path(f"{destination}.index").write_text("".join(f"{key}\t{index}\t1\n" for index, key in enumerate(keys)))
            for suffix in (".lookup", ".source", "_h", "_h.dbtype", "_h.index"):
                Path(f"{destination}{suffix}").symlink_to(Path(f"{source}{suffix}"))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        return original_prepare_search_database(**{**kwargs, "mmseqs_path": mmseqs, "runner": createsubdb})

    monkeypatch.setattr(external_assets, "_prepare_phrogs_safety_search_database", prepare_search_database)


def _mock_reviewed_phrogs_archive_sha256(monkeypatch) -> None:
    """Authenticate synthetic Pharokka archives at the reviewed SHA-256 boundary."""
    original_sha256_file = external_assets._sha256_file
    archive_name = Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name

    def sha256_file(path):
        path = Path(path)
        if path.name == archive_name or path.parent.name == "phrogs_safety_profile_archives":
            return external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SHA256
        return original_sha256_file(path)

    monkeypatch.setattr(external_assets, "_sha256_file", sha256_file)


def _mock_verified_phrogs_profile_archive(monkeypatch, external_dir: Path) -> Path:
    """Emulate the verified Pharokka archive boundary for local safety-orchestration tests."""
    archive_path = Path(external_dir) / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
    original_extract_tar = external_assets._extract_tar

    def is_verified_archive(path: Path) -> bool:
        path = Path(path)
        return path == archive_path or path.parent.name == "phrogs_safety_profile_archives"

    def verify_size(path, expected_size):
        assert is_verified_archive(path)
        assert expected_size == external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SIZE
        return expected_size

    def verify_md5(path, expected_md5):
        assert is_verified_archive(path)
        assert expected_md5 == external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5
        return expected_md5

    def extract_verified_archive(path, output_dir, *, overwrite):
        if not is_verified_archive(path):
            return original_extract_tar(path, output_dir, overwrite=overwrite)
        assert overwrite is True
        _write_phrogs_profile_database(Path(output_dir) / "phrogs_profile_db")
        return output_dir

    monkeypatch.setattr(external_assets, "_verify_file_size", verify_size)
    monkeypatch.setattr(external_assets, "_verify_md5", verify_md5)
    monkeypatch.setattr(external_assets, "_extract_tar", extract_verified_archive)
    _mock_reviewed_phrogs_archive_sha256(monkeypatch)
    _mock_phrogs_search_database_builder(monkeypatch, external_dir)
    return archive_path


def _install_curated_toxin_fixture(monkeypatch) -> bytes:
    """Use a compact exact-accession protein while retaining both digest checks."""
    payload = b">CAQ54400.1 WP0292\nMPEPTIDE\n"
    monkeypatch.setattr(
        external_assets,
        "DEFAULT_WOPIP1_PROTEIN_SEQUENCE_SHA256",
        hashlib.sha256(b"MPEPTIDE").hexdigest(),
    )
    monkeypatch.setattr(external_assets, "DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_INTERVAL", (1, 8))
    monkeypatch.setattr(
        external_assets,
        "DEFAULT_WOPIP1_LATROTOXIN_DOMAIN_SEQUENCE_SHA256",
        hashlib.sha256(b"MPEPTIDE").hexdigest(),
    )
    return payload


def _write_cached_toxin_snapshot(external_dir: Path, manifest_path: Path) -> dict:
    """Create a coherent cache/manifest pair for provenance validation tests."""
    toxin_dir = external_dir / "safety" / "toxins"
    toxin_dir.mkdir(parents=True)
    annotations_path = toxin_dir / "reviewed_toxins.tsv"
    fasta_path = toxin_dir / "reviewed_toxins.faa"
    curated_path = toxin_dir / "curated_hazards" / "CAQ54400.1.faa"
    search_path = toxin_dir / "toxin_hazards.faa"
    diamond_path = toxin_dir / "reviewed_toxins.dmnd"
    annotations_path.write_text(
        "Entry\tProtein names\tKeyword ID\tTaxonomic lineage (IDs)\nP00001\tExample toxin\tKW-0800; KW-0843\t1; 2\n"
    )
    fasta_path.write_text(">sp|P00001|TOX Example toxin\nMPEPTIDE\n")
    curated_path.parent.mkdir()
    curated_path.write_text(">CAQ54400.1 WP0292\nMPEPTIDE\n")
    search_path.write_bytes(fasta_path.read_bytes() + b">domain|PF15658.11|Latrotoxin_C\nMPEPTIDE\n")
    diamond_path.write_text("DIAMOND database\n")
    manifest = {
        "schema_version": 2,
        "toxin_reference": {
            "query": external_assets.DEFAULT_UNIPROT_TOXIN_QUERY,
            "classification_policy": external_assets.TOXIN_REFERENCE_CLASSIFICATION_POLICY,
            "annotations_url": external_assets.DEFAULT_UNIPROT_TOXIN_ANNOTATIONS_URL,
            "fasta_url": external_assets.DEFAULT_UNIPROT_TOXIN_FASTA_URL,
            "retrieved_at": "2026-01-01T00:00:00Z",
            "uniprot_release": "2026_01",
            "uniprot_release_date": "2026-01-15",
            "reference_version": "UniProt 2026_01 + phage-domain-hazards-v1",
            "curated_hazards": external_assets._curated_toxin_hazard_manifest(),
            "license": "CC BY 4.0",
            "attribution": external_assets.UNIPROT_CC_BY_4_0_ATTRIBUTION,
            "files": {
                "annotations": {
                    "path": str(annotations_path.resolve()),
                    "sha256": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
                },
                "fasta": {
                    "path": str(fasta_path.resolve()),
                    "sha256": hashlib.sha256(fasta_path.read_bytes()).hexdigest(),
                },
                "curated_hazard_fasta": {
                    "path": str(curated_path.resolve()),
                    "sha256": hashlib.sha256(curated_path.read_bytes()).hexdigest(),
                },
                "search_fasta": {
                    "path": str(search_path.resolve()),
                    "sha256": hashlib.sha256(search_path.read_bytes()).hexdigest(),
                },
                "diamond_database": {
                    "path": str(diamond_path.resolve()),
                    "sha256": hashlib.sha256(diamond_path.read_bytes()).hexdigest(),
                },
            },
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


def _mock_safety_manifest_section(name: str) -> tuple[str, dict]:
    """Return the minimum structural record a transaction must receive from a safety helper."""
    records = {
        "amrfinder": (
            "amrfinder_plus",
            {
                "archive_sha256": "archive",
                "binary_path": "binary",
                "binary_sha256": "binary",
                "amrfinder_index_path": "index",
                "amrfinder_index_sha256": "index",
                "amrfinder_update_path": "updater",
                "amrfinder_update_sha256": "updater",
                "runtime_bundle": {
                    "path": "/amrfinder-runtime",
                    "sha256": "0" * 64,
                    "files": [
                        {
                            "name": file_name,
                            "path": f"/amrfinder-runtime/{file_name}",
                            "sha256": "0" * 64,
                            "executable": file_name in external_assets.AMRFINDER_RUNTIME_EXECUTABLES,
                        }
                        for file_name in sorted(external_assets.AMRFINDER_RUNTIME_FILES)
                    ],
                },
                **{
                    f"{tool_name}_{field}": f"{tool_name}-{field}"
                    for tool_name in (
                        *external_assets.AMRFINDER_BLAST_RUNTIME_EXECUTABLES,
                        *external_assets.AMRFINDER_HMMER_RUNTIME_EXECUTABLES,
                    )
                    for field in ("path", "sha256")
                },
                "database_path": "database",
                "database_version": "version",
                "database_sha256": "database",
                "database_source_sha256": "0" * 64,
                "database_source_files": [
                    "AMR.LIB",
                    "AMRProt.fa",
                    "database_format_version.txt",
                    "fam.tsv",
                    "version.txt",
                ],
            },
        ),
        "toxins": (
            "toxin_reference",
            {
                "query": "query",
                "classification_policy": external_assets.TOXIN_REFERENCE_CLASSIFICATION_POLICY,
                "annotations_url": "annotations",
                "fasta_url": "fasta",
                "retrieved_at": "retrieved",
                "uniprot_release": "release",
                "reference_version": "UniProt release + phage-domain-hazards-v1",
                "curated_hazards": external_assets._curated_toxin_hazard_manifest(),
                "files": {
                    "annotations": {"path": "annotations", "sha256": "annotations"},
                    "fasta": {"path": "fasta", "sha256": "fasta"},
                    "curated_hazard_fasta": {"path": "curated", "sha256": "curated"},
                    "search_fasta": {"path": "search", "sha256": "search"},
                    "diamond_database": {"path": "diamond", "sha256": "diamond"},
                },
            },
        ),
        "phrogs": (
            "phrogs_v4",
            {
                "source_path": "source",
                "source_sha256": "source",
                "lookup_path": "lookup",
                "lookup_sha256": "lookup",
                "sequence_database": {"path": "sequence", "sha256": "sequence"},
            },
        ),
    }
    return records[name]


def _materialize_mock_safety_manifest_section(name: str, safety_dir: Path) -> tuple[str, dict]:
    """Create a complete local manifest record for orchestration-order tests."""
    safety_dir.mkdir(parents=True, exist_ok=True)
    if name == "amrfinder":
        bin_dir = safety_dir / "bin"
        _write_amrfinder_runtime(bin_dir, label="mock")
        prerequisite_paths = _write_amrfinder_prerequisites(bin_dir, label="mock")
        binary_paths = {
            "binary": bin_dir / "amrfinder",
            "index": bin_dir / "amrfinder_index",
            "updater": bin_dir / "amrfinder_update",
        }
        staged_runtime = {name: bin_dir / name for name in external_assets.AMRFINDER_RUNTIME_FILES}
        database_path = safety_dir / "amrfinder" / "database" / "2026-01-26.1"
        _write_minimal_amrfinder_database_sources(database_path)
        database_source_sha256, database_source_files = external_assets._amrfinder_database_source_identity(
            database_path
        )
        return (
            "amrfinder_plus",
            {
                "archive_sha256": "archive",
                "binary_path": str(binary_paths["binary"]),
                "binary_sha256": external_assets._sha256_file(binary_paths["binary"]),
                "amrfinder_index_path": str(binary_paths["index"]),
                "amrfinder_index_sha256": external_assets._sha256_file(binary_paths["index"]),
                "amrfinder_update_path": str(binary_paths["updater"]),
                "amrfinder_update_sha256": external_assets._sha256_file(binary_paths["updater"]),
                "runtime_bundle": external_assets._amrfinder_runtime_bundle_record(staged_runtime),
                **{
                    f"{tool_name}_{field}": value
                    for tool_name, path in prerequisite_paths.items()
                    for field, value in (
                        ("path", str(path)),
                        ("sha256", external_assets._sha256_file(path)),
                    )
                },
                "database_path": str(database_path),
                "database_version": "2026-01-26.1",
                "database_sha256": external_assets._sha256_path(database_path),
                "database_source_sha256": database_source_sha256,
                "database_source_files": database_source_files,
            },
        )
    if name == "toxins":
        toxin_dir = safety_dir / "toxins"
        toxin_dir.mkdir()
        files = {}
        for role, filename in (
            ("annotations", "reviewed_toxins.tsv"),
            ("fasta", "reviewed_toxins.faa"),
            ("curated_hazard_fasta", "CAQ54400.1.faa"),
            ("search_fasta", "toxin_hazards.faa"),
            ("diamond_database", "reviewed_toxins.dmnd"),
        ):
            path = toxin_dir / filename
            path.write_text(f"{role}\n")
            files[role] = {"path": str(path), "sha256": external_assets._sha256_file(path)}
        return (
            "toxin_reference",
            {
                "query": "query",
                "classification_policy": external_assets.TOXIN_REFERENCE_CLASSIFICATION_POLICY,
                "annotations_url": "annotations",
                "fasta_url": "fasta",
                "retrieved_at": "retrieved",
                "uniprot_release": "release",
                "reference_version": "UniProt release + phage-domain-hazards-v1",
                "curated_hazards": external_assets._curated_toxin_hazard_manifest(),
                "files": files,
            },
        )
    if name == "phrogs":
        source_path = safety_dir / "phrog_annot_v4.tsv"
        lookup_path = safety_dir / "phrogs_integration_excision_v4.tsv"
        source_path.write_text("source\n")
        lookup_path.write_text("lookup\n")
        sequence_database = _write_mmseqs_padded_database(safety_dir / "phrogs_gpu_seq_db_pad")
        sequence_database_sha256, _ = external_assets._complete_phrogs_sequence_database(sequence_database)
        profile_database = _write_phrogs_profile_database(safety_dir / "phrogs_profile_db")
        profile_database_sha256, profile_files = external_assets._complete_phrogs_profile_database(profile_database)
        profile_tree_files = external_assets._phrogs_profile_tree_files(profile_database)
        return (
            "phrogs_v4",
            {
                "source_path": str(source_path),
                "source_sha256": external_assets._sha256_file(source_path),
                "lookup_path": str(lookup_path),
                "lookup_sha256": external_assets._sha256_file(lookup_path),
                "sequence_database": {
                    "path": str(sequence_database),
                    "sha256": sequence_database_sha256,
                },
                "profile_database": {
                    "path": str(profile_database),
                    "sha256": profile_database_sha256,
                    "files": [str(path) for path in profile_files],
                    "extracted_tree": {
                        "path": str(profile_database.parent),
                        "sha256": external_assets._sha256_file_inventory(profile_database.parent, profile_tree_files),
                        "files": [str(path) for path in profile_tree_files],
                    },
                    "search_orientation": external_assets.PHROGS_PROFILE_SEARCH_ORIENTATION,
                    "search_profile_scope": external_assets.PHROGS_PROFILE_SEARCH_SCOPE,
                    "lookup_join_policy": external_assets.PHROGS_PROFILE_LOOKUP_JOIN_POLICY,
                    "output_fields": list(external_assets.PHROGS_PROFILE_OUTPUT_FIELDS),
                    "units": external_assets.PHROGS_PROFILE_OUTPUT_UNITS,
                    "query_id_pattern": external_assets.PHROGS_PROFILE_QUERY_ID_PATTERN,
                    "query_ids_join_lookup": True,
                    "profile_id_inventory": external_assets._phrogs_profile_id_inventory(profile_database),
                    "provenance": {
                        "source_url": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL,
                        "archive_observed_sha256": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SHA256,
                        "archive_expected_sha256": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SHA256,
                        "archive_published_sha256": None,
                        "archive_published_md5": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5,
                        "archive_published_size": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SIZE,
                        "retrieved_at": "2026-08-08T00:00:00Z",
                        "release": external_assets.PHROGS_PROFILE_RELEASE,
                        "dataset_release": external_assets.PHROGS_PROFILE_DATASET_RELEASE,
                        "doi": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_DOI,
                        "license": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_LICENSE,
                        "citation": external_assets.PHROGS_PROFILE_SOURCE_CITATION,
                        "minimum_mmseqs_version": external_assets.PHROGS_PROFILE_MIN_MMSEQS_VERSION,
                        "built_with_mmseqs_version": external_assets.PHROGS_PROFILE_BUILDER_MMSEQS_VERSION,
                    },
                },
            },
        )
    raise AssertionError(f"Unknown safety fixture: {name}")


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
    """A usable BLAST+ archive exposes the complete QC and AMRFinder runtime together."""
    archive_path = _write_multi_executable_tarball(
        tmp_path,
        ("dustmasker", "makeblastdb", "blastn", "blastp", "blastx", "tblastn"),
        subdir="ncbi-blast/bin",
        archive_name="blast-plus.tar.gz",
    )

    asset = prepare_dustmasker(
        tmp_path / "external",
        blast_plus_url=archive_path.as_uri(),
        blast_plus_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    )

    assert asset.path.name == "dustmasker"
    assert asset.path.is_symlink()
    assert asset.path.exists()
    assert (tmp_path / "external" / "bin" / "makeblastdb").is_symlink()
    assert (tmp_path / "external" / "bin" / "makeblastdb").exists()


@pytest.mark.parametrize(
    ("machine", "architecture", "archive_sha256", "blastx_sha256"),
    [
        (
            "aarch64",
            "aarch64",
            "cb9ac252a1ac8767d90b0bf0a38486f3cb94f71ef9b6b8d194ded19b30250daf",
            "fe541ee4a2b93b607940c9403bc6ffff7a88b51317c5ff0d14b29a6766eb49c6",
        ),
        (
            "arm64",
            "aarch64",
            "cb9ac252a1ac8767d90b0bf0a38486f3cb94f71ef9b6b8d194ded19b30250daf",
            "fe541ee4a2b93b607940c9403bc6ffff7a88b51317c5ff0d14b29a6766eb49c6",
        ),
        (
            "x86_64",
            "x86_64",
            "3888112d8207831aa47371d93583c601f058f88b5db22dc782438b039a3a411b",
            "0dac09bed17043dfdf93ad71d50e6e177f273d5311897b9e653e45142bb3ef80",
        ),
        (
            "amd64",
            "x86_64",
            "3888112d8207831aa47371d93583c601f058f88b5db22dc782438b039a3a411b",
            "0dac09bed17043dfdf93ad71d50e6e177f273d5311897b9e653e45142bb3ef80",
        ),
    ],
)
def test_blast_plus_release_selects_pinned_official_archive_for_cpu(
    machine, architecture, archive_sha256, blastx_sha256
):
    """Selecting the wrong archive or BLASTX binary would break or silently change cross-CPU scans."""
    release = external_assets.resolve_blast_plus_release(machine)

    assert release.version == "2.17.0"
    assert release.architecture == architecture
    assert release.archive_sha256 == archive_sha256
    assert release.blastx_sha256 == blastx_sha256
    assert release.url.startswith("https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/")


def test_blast_plus_release_rejects_unsupported_cpu():
    """An unknown CPU must fail before downloading or executing an incompatible binary."""
    with pytest.raises(RuntimeError, match="unsupported CPU architecture"):
        external_assets.resolve_blast_plus_release("riscv64")


def test_prepare_dustmasker_requires_digest_and_exposes_full_amrfinder_blast_runtime(tmp_path):
    """A custom archive must be pinned and must expose every BLAST executable AMRFinder uses."""
    archive_path = _write_multi_executable_tarball(
        tmp_path,
        ("dustmasker", "makeblastdb", "blastn", "blastp", "blastx", "tblastn"),
        subdir="ncbi-blast/bin",
        archive_name="blast-plus.tar.gz",
    )
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    asset = prepare_dustmasker(
        tmp_path / "external",
        blast_plus_url=archive_path.as_uri(),
        blast_plus_sha256=archive_sha256,
    )

    assert asset.path.name == "dustmasker"
    for executable in ("dustmasker", "makeblastdb", "blastn", "blastp", "blastx", "tblastn"):
        exposed = tmp_path / "external" / "bin" / executable
        assert exposed.is_symlink()
        assert exposed.exists()


def test_prepare_dustmasker_rejects_unpinned_custom_archive(tmp_path):
    """A caller-selected BLAST archive must never bypass the official release digest pins."""
    archive_path = _write_multi_executable_tarball(
        tmp_path,
        ("dustmasker", "makeblastdb", "blastn", "blastp", "blastx", "tblastn"),
        subdir="ncbi-blast/bin",
        archive_name="blast-plus.tar.gz",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        prepare_dustmasker(tmp_path / "external", blast_plus_url=archive_path.as_uri())


def test_prepare_dustmasker_rejects_blast_archive_without_makeblastdb(tmp_path):
    """AMRFinder's updater must never receive a BLAST directory missing makeblastdb."""
    archive_path = _write_tarball(tmp_path, executable_name="dustmasker", subdir="ncbi-blast/bin")

    with pytest.raises(FileNotFoundError, match="makeblastdb"):
        prepare_dustmasker(
            tmp_path / "external",
            blast_plus_url=archive_path.as_uri(),
            blast_plus_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        )


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
    """A local HMMER tarball should expose the search and index executables."""
    archive_path = _write_multi_executable_tarball(
        tmp_path,
        ("hmmsearch", "hmmpress"),
        subdir="bin",
        archive_name="hmmer.tar.gz",
    )

    asset = prepare_hmmer(tmp_path / "external", bin_dir=tmp_path / "venv" / "bin", hmmer_url=archive_path.as_uri())

    assert asset.path.name == "hmmsearch"
    assert asset.path.is_symlink()
    assert asset.path.exists()
    assert asset.path.parent == tmp_path / "venv" / "bin"
    assert (tmp_path / "venv" / "bin" / "hmmpress").exists()


def test_prepare_hmmer_rejects_archive_without_hmmpress(tmp_path):
    """AMRFinder preparation requires HMMER's profile-indexing executable."""
    archive_path = _write_tarball(tmp_path, executable_name="hmmsearch", subdir="bin")

    with pytest.raises(FileNotFoundError, match="hmmpress"):
        prepare_hmmer(tmp_path / "external", hmmer_url=archive_path.as_uri())


def test_prepare_amrfinder_plus_extracts_pinned_archive_as_self_contained_runtime(tmp_path, monkeypatch):
    """Safety preparation copies a release-shaped runtime into one immutable sibling directory."""
    archive_path = _write_amrfinder_tarball(tmp_path)
    external_dir = tmp_path / "external"
    target_bin_dir = external_dir / "bin"
    _write_amrfinder_prerequisites(target_bin_dir)
    calls = []

    def fake_run(cmd, check, capture_output, text):
        calls.append((cmd, check, capture_output, text))
        if cmd[0].endswith("amrfinder_update"):
            database_dir = external_dir / "safety" / "amrfinder" / "database"
            version_dir = database_dir / "2026-01-26.1"
            _write_minimal_amrfinder_database_sources(version_dir)
            (database_dir / "latest").symlink_to(version_dir.name)
            return type("Completed", (), {"stdout": ""})()
        if cmd[-1] == "--version":
            return type("Completed", (), {"stdout": "AMRFinderPlus version 4.2.7\n"})()
        return type("Completed", (), {"stdout": "2026-01-26.1\n"})()

    monkeypatch.setattr("subprocess.run", fake_run)
    manifest_path = tmp_path / "external" / "safety" / "asset_manifest.yaml"

    asset = prepare_amrfinder_plus(
        external_dir,
        amrfinder_url=archive_path.as_uri(),
        amrfinder_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        manifest_path=manifest_path,
    )

    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert asset.path.name == "amrfinder"
    assert not asset.path.is_symlink()
    assert asset.path.exists()
    runtime_paths = [external_dir / "bin" / name for name in external_assets.AMRFINDER_RUNTIME_FILES]
    assert all(path.is_file() and not path.is_symlink() for path in runtime_paths)
    assert {path.parent.resolve() for path in runtime_paths} == {target_bin_dir.resolve()}
    assert "amrfinder_version: AMRFinderPlus version 4.2.7" in manifest_path.read_text()
    assert "database_version: 2026-01-26.1" in manifest_path.read_text()
    manifest = yaml.safe_load(manifest_path.read_text())
    expected_source_sha256, expected_source_files = external_assets._amrfinder_database_source_identity(
        external_dir / "safety" / "amrfinder" / "database" / "2026-01-26.1"
    )
    assert manifest["amrfinder_plus"]["database_source_sha256"] == expected_source_sha256
    assert manifest["amrfinder_plus"]["database_source_files"] == expected_source_files
    assert f"archive_sha256: {archive_digest}" in manifest_path.read_text()
    assert (
        f"amrfinder_index_sha256: {hashlib.sha256((external_dir / 'bin' / 'amrfinder_index').read_bytes()).hexdigest()}"
        in manifest_path.read_text()
    )
    assert (
        f"amrfinder_update_sha256: {hashlib.sha256((external_dir / 'bin' / 'amrfinder_update').read_bytes()).hexdigest()}"
        in manifest_path.read_text()
    )
    assert (
        f"makeblastdb_sha256: {hashlib.sha256((external_dir / 'bin' / 'makeblastdb').read_bytes()).hexdigest()}"
        in manifest_path.read_text()
    )
    assert (
        f"hmmpress_sha256: {hashlib.sha256((external_dir / 'bin' / 'hmmpress').read_bytes()).hexdigest()}"
        in manifest_path.read_text()
    )
    assert calls == [
        (
            [
                str(external_dir / "bin" / "amrfinder_update"),
                "-d",
                str(external_dir / "safety" / "amrfinder" / "database"),
                "--blast_bin",
                str(target_bin_dir.resolve()),
                "--hmmer_bin",
                str(target_bin_dir.resolve()),
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
                str(external_dir / "safety" / "amrfinder" / "database" / "2026-01-26.1"),
                "--database_version",
            ],
            True,
            True,
            True,
        ),
    ]


def test_prepare_amrfinder_plus_records_operator_asserted_source_build_without_release_archive(tmp_path, monkeypatch):
    """A caller-supplied native build is byte-bound without overstating checkout provenance."""
    external_dir = tmp_path / "external"
    source_bin_dir = tmp_path / "native-amrfinder" / "bin"
    _write_amrfinder_runtime(source_bin_dir, label="native")
    target_bin_dir = external_dir / "bin"
    _write_amrfinder_prerequisites(target_bin_dir)

    def fail_download(*_args, **_kwargs):
        pytest.fail("an operator-supplied source build must not download or execute the x86 release archive")

    version_dir = external_dir / "safety" / "amrfinder" / "database" / "2026-08-04.1"

    def fake_run(cmd, check, capture_output, text):
        if cmd[0].endswith("amrfinder_update"):
            _write_minimal_amrfinder_database_sources(version_dir)
            (version_dir.parent / "latest").symlink_to(version_dir.name)
            return type("Completed", (), {"stdout": ""})()
        if cmd[-1] == "--version":
            return type("Completed", (), {"stdout": "AMRFinderPlus version 4.2.7\n"})()
        return type(
            "Completed",
            (),
            {
                "stdout": (
                    f"Software directory: '{source_bin_dir}/'\n"
                    "Software version: 4.2.7\n"
                    f"Database directory: '{version_dir}'\n"
                    "Database version: 2026-08-04.1\n"
                )
            },
        )()

    monkeypatch.setattr(external_assets, "_download", fail_download)
    monkeypatch.setattr("subprocess.run", fake_run)
    manifest_path = external_dir / "safety" / "asset_manifest.yaml"

    asset = prepare_amrfinder_plus(
        external_dir,
        source_bin_dir=source_bin_dir,
        source_repository="https://github.com/ncbi/amr.git",
        source_revision="76bf8527b3cabc6a08fdc0e20783e594bba162e3",
        manifest_path=manifest_path,
    )

    manifest = yaml.safe_load(manifest_path.read_text())
    record = manifest["amrfinder_plus"]
    assert asset.path.read_bytes() == (source_bin_dir / "amrfinder").read_bytes()
    assert record["acquisition"] == "operator_asserted_source_build"
    assert record["provenance_basis"] == "operator_asserted_repository_revision_and_binary_digest"
    assert record["source_repository"] == "https://github.com/ncbi/amr.git"
    assert record["source_revision"] == "76bf8527b3cabc6a08fdc0e20783e594bba162e3"
    assert record["source_binary_sha256"] == hashlib.sha256((source_bin_dir / "amrfinder").read_bytes()).hexdigest()
    assert record["database_version"] == "2026-08-04.1"
    assert (target_bin_dir / "fasta_check").read_bytes() == (source_bin_dir / "fasta_check").read_bytes()
    assert [item["name"] for item in record["runtime_bundle"]["files"]] == sorted(
        external_assets.AMRFINDER_RUNTIME_FILES
    )
    external_assets._validate_amrfinder_runtime_bundle(record, verify_asset_paths=True)


def test_prepare_amrfinder_plus_refuses_source_build_missing_runtime_companion(tmp_path, monkeypatch):
    """The staged scanner must contain the sibling tools invoked by the AMRFinder executable."""
    source_bin_dir = tmp_path / "native-amrfinder" / "bin"
    source_bin_dir.mkdir(parents=True)
    for executable_name in ("amrfinder", "amrfinder_index", "amrfinder_update"):
        executable = source_bin_dir / executable_name
        executable.write_text("#!/usr/bin/env bash\n")
        executable.chmod(0o755)
    target_bin_dir = tmp_path / "external" / "bin"
    for executable_name in ("makeblastdb", "hmmpress"):
        executable = target_bin_dir / executable_name
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/usr/bin/env bash\n")
        executable.chmod(0o755)

    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("an incomplete AMRFinder runtime reached execution"),
    )

    with pytest.raises(FileNotFoundError, match="fasta_check"):
        prepare_amrfinder_plus(
            tmp_path / "external",
            source_bin_dir=source_bin_dir,
            source_repository="https://github.com/ncbi/amr.git",
            source_revision="76bf8527b3cabc6a08fdc0e20783e594bba162e3",
        )


@pytest.mark.parametrize(
    "output",
    (
        "Database version:\n",
        "Database version: 2026-08-04.1\nDatabase version: 2026-08-05.1\n",
        "Software version: 4.2.7\n",
    ),
)
def test_amrfinder_database_version_parser_rejects_missing_or_ambiguous_version(output):
    """AMRFinder status output cannot authorize an absent or ambiguous database identity."""
    with pytest.raises(RuntimeError, match="database version"):
        external_assets._parse_amrfinder_database_version(output)


@pytest.mark.parametrize(
    ("source_repository", "source_revision"),
    ((None, "76bf8527b3cabc6a08fdc0e20783e594bba162e3"), ("https://github.com/ncbi/amr.git", None)),
)
def test_prepare_amrfinder_plus_rejects_unpinned_source_build(
    tmp_path, monkeypatch, source_repository, source_revision
):
    """Preinstalled binaries never bypass explicit repository and immutable-revision provenance."""
    source_bin_dir = tmp_path / "native-amrfinder" / "bin"
    source_bin_dir.mkdir(parents=True)
    for executable_name in ("amrfinder", "amrfinder_index", "amrfinder_update"):
        executable = source_bin_dir / executable_name
        executable.write_text("#!/usr/bin/env bash\n")
        executable.chmod(0o755)
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("unpinned binaries ran"))

    with pytest.raises(ValueError, match="repository and revision"):
        prepare_amrfinder_plus(
            tmp_path / "external",
            source_bin_dir=source_bin_dir,
            source_repository=source_repository,
            source_revision=source_revision,
        )


def test_prepare_amrfinder_plus_refuses_to_extract_without_a_digest(tmp_path):
    """The AMRFinder executable must never be extracted from an unpinned archive."""
    archive_path = _write_amrfinder_tarball(tmp_path)

    with pytest.raises(ValueError, match="digest"):
        prepare_amrfinder_plus(
            tmp_path / "external",
            amrfinder_url=archive_path.as_uri(),
            amrfinder_sha256=None,
        )


def test_prepare_amrfinder_plus_refuses_download_with_wrong_declared_digest(tmp_path):
    """A declared release digest must match the downloaded AMRFinder archive."""
    archive_path = _write_amrfinder_tarball(tmp_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_amrfinder_plus(
            tmp_path / "external",
            amrfinder_url=archive_path.as_uri(),
            amrfinder_sha256="0" * 64,
        )


def test_prepare_amrfinder_plus_never_reuses_a_tampered_digest_named_extraction(tmp_path, monkeypatch):
    """A digest-named extraction tree must be recreated from the verified archive before use."""
    archive_path = _write_amrfinder_tarball(tmp_path)
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    external_dir = tmp_path / "external"
    target_bin_dir = external_dir / "bin"
    _write_amrfinder_prerequisites(target_bin_dir)

    tampered_bin_dir = (
        external_dir / "safety" / "tools" / f"amrfinder_v4.2.7-{archive_sha256[:16]}" / "attacker" / "bin"
    )
    tampered_bin_dir.mkdir(parents=True)
    for executable_name in ("amrfinder", "amrfinder_index", "amrfinder_update"):
        executable = tampered_bin_dir / executable_name
        executable.write_text(f"#!/usr/bin/env bash\n# tampered {executable_name}\n")
        executable.chmod(0o755)

    def fake_run(cmd, check, capture_output, text):
        if cmd[0].endswith("amrfinder_update"):
            database_dir = external_dir / "safety" / "amrfinder" / "database"
            version_dir = database_dir / "2026-01-26.1"
            _write_minimal_amrfinder_database_sources(version_dir)
            (database_dir / "latest").symlink_to(version_dir.name)
            return type("Completed", (), {"stdout": ""})()
        if cmd[-1] == "--version":
            return type("Completed", (), {"stdout": "AMRFinderPlus version 4.2.7\n"})()
        return type("Completed", (), {"stdout": "2026-01-26.1\n"})()

    monkeypatch.setattr("subprocess.run", fake_run)

    asset = prepare_amrfinder_plus(
        external_dir,
        amrfinder_url=archive_path.as_uri(),
        amrfinder_sha256=archive_sha256,
    )

    assert "attacker" not in str(asset.path.resolve())
    assert "trusted amrfinder" in asset.path.read_text()


def test_prepare_amrfinder_plus_refuses_an_archive_missing_amrfinder_index(tmp_path, monkeypatch):
    """AMRFinder's companion index binary must be verified before any archive content runs."""
    archive_path = _write_amrfinder_tarball(tmp_path)
    source_root = tmp_path / "amrfinder_archive_src" / "amrfinder" / "bin"
    (source_root / "amrfinder_index").unlink()
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(tmp_path / "amrfinder_archive_src", arcname="archive")

    external_dir = tmp_path / "external"
    target_bin_dir = external_dir / "bin"
    _write_amrfinder_prerequisites(target_bin_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("an archive missing amrfinder_index reached execution"),
    )

    with pytest.raises(FileNotFoundError, match="amrfinder_index"):
        prepare_amrfinder_plus(
            external_dir,
            amrfinder_url=archive_path.as_uri(),
            amrfinder_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize(
    "missing_field",
    (
        "amrfinder_index_sha256",
        "runtime_bundle",
        "database_version",
        "database_source_sha256",
        "database_source_files",
    ),
)
def test_validate_staged_safety_manifest_requires_amrfinder_index_and_database_version(missing_field):
    """A trusted safety generation needs all AMRFinder executable and database provenance."""
    amrfinder_section, amrfinder_record = _mock_safety_manifest_section("amrfinder")
    toxin_section, toxin_record = _mock_safety_manifest_section("toxins")
    phrogs_section, phrogs_record = _mock_safety_manifest_section("phrogs")
    amrfinder_record.pop(missing_field)

    with pytest.raises(RuntimeError, match=missing_field):
        external_assets._validate_staged_safety_manifest(
            {
                "schema_version": 3,
                amrfinder_section: amrfinder_record,
                toxin_section: toxin_record,
                phrogs_section: phrogs_record,
            }
        )


def test_validate_amrfinder_source_identity_rejects_manifest_drift(tmp_path):
    """A stable dataset identity cannot be replaced with a caller-authored digest or inventory."""
    section, record = _materialize_mock_safety_manifest_section("amrfinder", tmp_path)
    assert section == "amrfinder_plus"

    record["database_source_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="source digest"):
        external_assets._validate_amrfinder_database_source_identity(record)

    record["database_source_sha256"], record["database_source_files"] = (
        external_assets._amrfinder_database_source_identity(Path(record["database_path"]))
    )
    record["database_source_files"] = [*record["database_source_files"], "forged.tsv"]
    with pytest.raises(RuntimeError, match="source inventory"):
        external_assets._validate_amrfinder_database_source_identity(record)


def test_validate_staged_safety_manifest_accepts_operator_asserted_amrfinder_source_build(tmp_path):
    """The safety transaction records the trust boundary for a caller-supplied native runtime."""
    manifest = {"schema_version": 3}
    for name in ("amrfinder", "toxins", "phrogs"):
        section, record = _materialize_mock_safety_manifest_section(name, tmp_path / name)
        manifest[section] = record
    amrfinder = manifest["amrfinder_plus"]
    amrfinder.pop("archive_sha256")
    amrfinder.update(
        {
            "acquisition": "operator_asserted_source_build",
            "provenance_basis": "operator_asserted_repository_revision_and_binary_digest",
            "source_repository": "https://github.com/ncbi/amr.git",
            "source_revision": "76bf8527b3cabc6a08fdc0e20783e594bba162e3",
            "source_binary_sha256": amrfinder["binary_sha256"],
        }
    )
    manifest["phrogs_v4"]["profile_database"]["provenance"]["verified_archive"] = {
        "path": "archive",
        "sha256": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SHA256,
    }

    external_assets._validate_staged_safety_manifest(manifest)


@pytest.mark.parametrize(
    "missing_field", ("provenance_basis", "source_repository", "source_revision", "source_binary_sha256")
)
def test_validate_staged_safety_manifest_rejects_incomplete_amrfinder_source_provenance(tmp_path, missing_field):
    """Reject an operator-attested AMRFinder record when an asserted lineage field is absent."""
    manifest = {"schema_version": 3}
    for name in ("amrfinder", "toxins", "phrogs"):
        section, record = _materialize_mock_safety_manifest_section(name, tmp_path / name)
        manifest[section] = record
    amrfinder = manifest["amrfinder_plus"]
    amrfinder.pop("archive_sha256")
    amrfinder.update(
        {
            "acquisition": "operator_asserted_source_build",
            "provenance_basis": "operator_asserted_repository_revision_and_binary_digest",
            "source_repository": "https://github.com/ncbi/amr.git",
            "source_revision": "76bf8527b3cabc6a08fdc0e20783e594bba162e3",
            "source_binary_sha256": amrfinder["binary_sha256"],
        }
    )
    amrfinder.pop(missing_field)

    with pytest.raises(RuntimeError, match=missing_field):
        external_assets._validate_staged_safety_manifest(manifest)


def test_validate_recorded_asset_digest_rejects_a_tampered_amrfinder_index(tmp_path):
    """A staged generation must reject publication if its recorded AMRFinder index changes."""
    index_path = tmp_path / "amrfinder_index"
    index_path.write_text("trusted index\n")
    record = {
        "amrfinder_index_path": str(index_path),
        "amrfinder_index_sha256": external_assets._sha256_file(index_path),
    }
    index_path.write_text("tampered index\n")

    with pytest.raises(RuntimeError, match="AMRFinder index digest"):
        external_assets._validate_recorded_asset_digest(
            record,
            "amrfinder_index_path",
            "amrfinder_index_sha256",
            "AMRFinder index",
        )


def test_prepare_amrfinder_plus_rejects_latest_symlink_escaping_requested_database_root(tmp_path, monkeypatch):
    """The documented latest pointer must not resolve outside the requested database root."""
    archive_path = _write_amrfinder_tarball(tmp_path)
    external_dir = tmp_path / "external"
    target_bin_dir = external_dir / "bin"
    _write_amrfinder_prerequisites(target_bin_dir)
    database_dir = tmp_path / "requested-database"
    external_version_dir = tmp_path / "outside" / "2026-01-26.1"
    external_version_dir.mkdir(parents=True)
    (external_version_dir / "AMRProt.fa").write_text(">AMR\nMPEPTIDE\n")
    database_dir.mkdir()
    (database_dir / "latest").symlink_to(external_version_dir)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("an escaping AMRFinder database reached execution"),
    )

    with pytest.raises(ValueError, match="contained"):
        prepare_amrfinder_plus(
            external_dir,
            amrfinder_url=archive_path.as_uri(),
            amrfinder_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            database_dir=database_dir,
        )


def test_prepare_amrfinder_plus_rejects_non_symlink_latest_directory(tmp_path, monkeypatch):
    """A normal latest directory is not the pinned AMRFinder version indirection contract."""
    archive_path = _write_amrfinder_tarball(tmp_path)
    external_dir = tmp_path / "external"
    target_bin_dir = external_dir / "bin"
    _write_amrfinder_prerequisites(target_bin_dir)
    database_dir = tmp_path / "requested-database"
    latest_dir = database_dir / "latest"
    latest_dir.mkdir(parents=True)
    (latest_dir / "AMRProt.fa").write_text(">AMR\nMPEPTIDE\n")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("a non-symlink AMRFinder latest directory reached execution"),
    )

    with pytest.raises(FileNotFoundError, match="symbolic link"):
        prepare_amrfinder_plus(
            external_dir,
            amrfinder_url=archive_path.as_uri(),
            amrfinder_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            database_dir=database_dir,
        )


def test_prepare_amrfinder_plus_rejects_an_empty_database_version(tmp_path, monkeypatch):
    """AMRFinder must report a nonempty version agreeing with the resolved pinned directory."""
    archive_path = _write_amrfinder_tarball(tmp_path)
    external_dir = tmp_path / "external"
    target_bin_dir = external_dir / "bin"
    _write_amrfinder_prerequisites(target_bin_dir)
    database_dir = tmp_path / "requested-database"
    version_dir = database_dir / "2026-01-26.1"
    version_dir.mkdir(parents=True)
    (version_dir / "AMRProt.fa").write_text(">AMR\nMPEPTIDE\n")
    (database_dir / "latest").symlink_to(version_dir.name)

    def fake_run(cmd, check, capture_output, text):
        del check, capture_output, text
        if cmd[-1] == "--version":
            return type("Completed", (), {"stdout": "AMRFinderPlus version 4.2.7\n"})()
        return type("Completed", (), {"stdout": "\n"})()

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="nonempty database version"):
        prepare_amrfinder_plus(
            external_dir,
            amrfinder_url=archive_path.as_uri(),
            amrfinder_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            database_dir=database_dir,
        )


def test_prepare_toxin_reference_records_uniprot_provenance_and_builds_diamond_database(tmp_path, monkeypatch):
    """A reviewed toxin snapshot has release metadata and digests for every generated input."""
    annotation_payload = (
        b"Entry\tProtein names\tKeyword ID\tTaxonomic lineage (IDs)\nP00001\tExample toxin\tKW-0800; KW-0843\t1; 2\n"
    )
    fasta_payload = b">sp|P00001|TOX Example toxin\nMPEPTIDE\n"
    curated_payload = _install_curated_toxin_fixture(monkeypatch)
    calls = []

    class Response(io.BytesIO):
        def __init__(self, payload):
            super().__init__(payload)
            self.headers = {"X-UniProt-Release": "2026_01", "X-UniProt-Release-Date": "2026-01-15"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(url, context=None):
        del context
        if url in external_assets.DEFAULT_WOPIP1_PROTEIN_URLS:
            return Response(curated_payload)
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
    assert manifest["toxin_reference"]["uniprot_release_date"] == "2026-01-15"
    assert manifest["toxin_reference"]["query"] == DEFAULT_UNIPROT_TOXIN_QUERY
    assert manifest["toxin_reference"]["license"] == "CC BY 4.0"
    assert manifest["toxin_reference"]["reference_version"] == "UniProt 2026_01 + phage-domain-hazards-v1"
    assert manifest["toxin_reference"]["curated_hazards"]["entries"][0]["accession"] == "PF15658.11"
    assert (
        manifest["toxin_reference"]["files"]["annotations"]["sha256"] == hashlib.sha256(annotation_payload).hexdigest()
    )
    assert manifest["toxin_reference"]["files"]["fasta"]["sha256"] == hashlib.sha256(fasta_payload).hexdigest()
    curated_record = manifest["toxin_reference"]["files"]["curated_hazard_fasta"]
    assert curated_record["sha256"] == hashlib.sha256(curated_payload).hexdigest()
    search_record = manifest["toxin_reference"]["files"]["search_fasta"]
    assert Path(search_record["path"]).read_bytes() == (fasta_payload + b">domain|PF15658.11|Latrotoxin_C\nMPEPTIDE\n")
    assert calls == [
        (
            [
                str(tmp_path / "external" / "bin" / "diamond"),
                "makedb",
                "--in",
                str(tmp_path / "external" / "safety" / "toxins" / "toxin_hazards.faa"),
                "--db",
                str(tmp_path / "external" / "safety" / "toxins" / "reviewed_toxins.dmnd"),
            ],
            True,
        )
    ]


@pytest.mark.parametrize(
    ("keyword_ids", "lineage_ids", "accepted"),
    [
        ("KW-0800; KW-0843", "2; 1224; 1236", True),
        ("KW-0078; KW-0800", "2; 1239", False),
        ("KW-0078; KW-0800", "1; 2759; 33208", True),
        ("KW-0800", "2; 1224", False),
    ],
)
def test_uniprot_toxin_scope_keeps_human_harm_and_excludes_antibacterial_only_entries(
    tmp_path, keyword_ids, lineage_ids, accepted
):
    """The local snapshot check must enforce host-harm scope independently of protein names."""
    annotations = tmp_path / "toxins.tsv"
    annotations.write_text(f"Entry\tKeyword ID\tTaxonomic lineage (IDs)\nP00001\t{keyword_ids}\t{lineage_ids}\n")

    if accepted:
        external_assets._validate_uniprot_toxin_scope(annotations)
    else:
        with pytest.raises(ValueError, match="outside the human-harm toxin scope"):
            external_assets._validate_uniprot_toxin_scope(annotations)


def test_uniprot_toxin_scope_accepts_live_lineage_header_spelling(tmp_path):
    """UniProt currently spells the return-field label ``Ids`` rather than ``IDs``."""
    annotations = tmp_path / "toxins.tsv"
    annotations.write_text(
        "Entry\tKeyword ID\tTaxonomic lineage (Ids)\nP00001\tKW-0800\t131567 (no rank), 2759 (domain), 33154 (clade)\n"
    )

    external_assets._validate_uniprot_toxin_scope(annotations)


def test_prepare_toxin_reference_rebuilds_diamond_after_a_fresh_uniprot_snapshot(tmp_path, monkeypatch):
    """Fresh TSV/FASTA inputs must never bless a DIAMOND index from an older generation."""
    external_dir = tmp_path / "external"
    stale_database = external_dir / "safety" / "toxins" / "reviewed_toxins.dmnd"
    stale_database.parent.mkdir(parents=True)
    stale_database.write_text("stale DIAMOND database\n")
    annotation_payload = (
        b"Entry\tProtein names\tKeyword ID\tTaxonomic lineage (IDs)\nP00001\tExample toxin\tKW-0800; KW-0843\t1; 2\n"
    )
    fasta_payload = b">sp|P00001|TOX Example toxin\nMPEPTIDE\n"
    curated_payload = _install_curated_toxin_fixture(monkeypatch)
    calls = []

    class Response(io.BytesIO):
        def __init__(self, payload):
            super().__init__(payload)
            self.headers = {"X-UniProt-Release": "2026_01", "X-UniProt-Release-Date": "2026-01-15"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(url, context=None):
        del context
        if url in external_assets.DEFAULT_WOPIP1_PROTEIN_URLS:
            return Response(curated_payload)
        return Response(annotation_payload if "format=tsv" in url else fasta_payload)

    def fake_run(cmd, check):
        calls.append((cmd, check))
        Path(cmd[-1]).write_text("fresh DIAMOND database\n")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("subprocess.run", fake_run)

    asset = prepare_toxin_reference(external_dir)

    assert calls == [
        (
            [
                str(external_dir / "bin" / "diamond"),
                "makedb",
                "--in",
                str(external_dir / "safety" / "toxins" / "toxin_hazards.faa"),
                "--db",
                str(stale_database),
            ],
            True,
        )
    ]
    assert asset.path.read_text() == "fresh DIAMOND database\n"


@pytest.mark.parametrize(
    ("annotation_payload", "fasta_payload", "annotation_release", "fasta_release", "error_match"),
    [
        (
            b"Entry\tProtein names\tKeyword ID\tTaxonomic lineage (IDs)\n"
            b"P00001\tExample toxin\tKW-0800; KW-0843\t1; 2\n",
            b">sp|P00001|TOX Example toxin\nMPEPTIDE\n",
            "2026_01",
            "2026_02",
            "release",
        ),
        (
            b"Entry\tProtein names\tKeyword ID\tTaxonomic lineage (IDs)\n"
            b"P00001\tExample toxin\tKW-0800; KW-0843\t1; 2\n",
            b">sp|P00002|OTHER Different toxin\nMPEPTIDE\n",
            "2026_01",
            "2026_01",
            "accession",
        ),
    ],
)
def test_prepare_toxin_reference_rejects_mixed_uniprot_snapshot(
    tmp_path,
    monkeypatch,
    annotation_payload,
    fasta_payload,
    annotation_release,
    fasta_release,
    error_match,
):
    """TSV and FASTA must be a coherent release with the same accession set."""
    curated_payload = _install_curated_toxin_fixture(monkeypatch)

    class Response(io.BytesIO):
        def __init__(self, payload, release):
            super().__init__(payload)
            self.headers = {"X-UniProt-Release": release, "X-UniProt-Release-Date": "2026-01-15"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(url, context=None):
        del context
        if url in external_assets.DEFAULT_WOPIP1_PROTEIN_URLS:
            return Response(curated_payload, annotation_release)
        if "format=tsv" in url:
            return Response(annotation_payload, annotation_release)
        return Response(fasta_payload, fasta_release)

    def fake_run(cmd, check):
        assert check
        Path(cmd[-1]).write_text("DIAMOND database\n")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(ValueError, match=error_match):
        prepare_toxin_reference(tmp_path / "external")


def test_prepare_toxin_reference_preserves_verified_cache_retrieval_time(tmp_path, monkeypatch):
    """A valid cached snapshot retains its original retrieval timestamp and provenance."""
    external_dir = tmp_path / "external"
    manifest_path = external_dir / "safety" / "asset_manifest.yaml"
    _install_curated_toxin_fixture(monkeypatch)
    old_manifest = _write_cached_toxin_snapshot(external_dir, manifest_path)

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("cache unexpectedly downloaded")
    )
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("cache unexpectedly rebuilt"))

    prepare_toxin_reference(external_dir, manifest_path=manifest_path)

    refreshed_manifest = yaml.safe_load(manifest_path.read_text())
    assert refreshed_manifest["toxin_reference"]["retrieved_at"] == old_manifest["toxin_reference"]["retrieved_at"]
    for field in ("query", "annotations_url", "fasta_url", "uniprot_release", "uniprot_release_date", "license"):
        assert refreshed_manifest["toxin_reference"][field] == old_manifest["toxin_reference"][field]
    for file_role in ("annotations", "fasta", "diamond_database"):
        assert (
            refreshed_manifest["toxin_reference"]["files"][file_role]["sha256"]
            == old_manifest["toxin_reference"]["files"][file_role]["sha256"]
        )


def test_prepare_toxin_reference_rejects_tampered_cached_files(tmp_path, monkeypatch):
    """Cached files must still match their previously recorded digests and accession parity."""
    external_dir = tmp_path / "external"
    manifest_path = external_dir / "safety" / "asset_manifest.yaml"
    _install_curated_toxin_fixture(monkeypatch)
    _write_cached_toxin_snapshot(external_dir, manifest_path)
    (external_dir / "safety" / "toxins" / "reviewed_toxins.faa").write_text(">sp|P00002|OTHER\nMPEPTIDE\n")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("tampered cache downloaded"))

    with pytest.raises(ValueError, match="digest"):
        prepare_toxin_reference(external_dir, manifest_path=manifest_path)


def test_prepare_toxin_reference_rejects_unattributed_override_urls(tmp_path, monkeypatch):
    """Arbitrary override URLs cannot inherit the canonical UniProt query/license metadata."""
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: pytest.fail("override was downloaded"))

    with pytest.raises(ValueError, match="explicit provenance"):
        prepare_toxin_reference(
            tmp_path / "external",
            annotations_url="https://example.invalid/toxins.tsv",
            fasta_url="https://example.invalid/toxins.faa",
        )


def test_prepare_toxin_reference_keeps_tls_verification_when_global_flag_is_set(tmp_path, monkeypatch):
    """The PHROGs-only TLS exception must not leak into toxin transports."""
    annotation_payload = (
        b"Entry\tProtein names\tKeyword ID\tTaxonomic lineage (IDs)\nP00001\tExample toxin\tKW-0800; KW-0843\t1; 2\n"
    )
    fasta_payload = b">sp|P00001|TOX Example toxin\nMPEPTIDE\n"
    curated_payload = _install_curated_toxin_fixture(monkeypatch)

    class Response(io.BytesIO):
        def __init__(self, payload):
            super().__init__(payload)
            self.headers = {"X-UniProt-Release": "2026_01", "X-UniProt-Release-Date": "2026-01-15"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(url, context=None):
        assert context is None
        if url in external_assets.DEFAULT_WOPIP1_PROTEIN_URLS:
            return Response(curated_payload)
        return Response(annotation_payload if "format=tsv" in url else fasta_payload)

    def fake_run(cmd, check):
        assert check
        Path(cmd[-1]).write_text("DIAMOND database\n")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("subprocess.run", fake_run)

    prepare_toxin_reference(tmp_path / "external", insecure_downloads=True)


def test_prepare_phrogs_safety_metadata_splits_real_schema_hits_by_confidence(tmp_path):
    """Real PHROGs v4 columns and category produce a versioned confidence lookup."""
    annotation_path = tmp_path / "external" / "phrogs" / "phrog_annot_v4.tsv"
    sequence_database = _write_phrogs_v4_fixture(annotation_path)
    profile_database = annotation_path.parent / "phrogs_mmseqs_db" / "phrogs_profile_db"
    profile_archive = (
        annotation_path.parent.parent / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
    )
    manifest_path = tmp_path / "external" / "safety" / "asset_manifest.yaml"

    asset = external_assets._prepare_phrogs_safety_metadata(
        tmp_path / "external",
        manifest_path=manifest_path,
        annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        sequence_database=sequence_database,
        profile_database=profile_database,
        **_profile_metadata_kwargs(profile_database, profile_archive),
    )

    rows = [line.split("\t") for line in asset.path.read_text().splitlines()]
    assert asset.path.name == "phrogs_integration_excision_v4.tsv"
    assert rows[0] == ["phrog", "annot", "category", "confidence", "matched_term"]
    assert rows[1:] == [
        ["phrog_1", "Integrase", "integration and excision", "high_confidence", "integrase"],
        [
            "phrog_2",
            "Site-specific recombinase",
            "integration and excision",
            "high_confidence",
            "site-specific recombinase",
        ],
        [
            "phrog_3",
            "Lysogeny repressor",
            "integration and excision",
            "high_confidence",
            "lysogeny repressor",
        ],
        ["phrog_4", "Putative recombinase", "integration and excision", "review", "recombinase"],
        ["phrog_6", "Anti-repressor", "transcription regulation", "high_confidence", "anti-repressor"],
    ]
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["phrogs_v4"]["source_sha256"] == hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    assert manifest["phrogs_v4"]["sequence_database"]["path"] == str(sequence_database.resolve())
    expected_database_sha256, expected_database_files = external_assets._complete_phrogs_sequence_database(
        sequence_database
    )
    assert manifest["phrogs_v4"]["sequence_database"]["sha256"] == expected_database_sha256
    assert manifest["phrogs_v4"]["sequence_database"]["files"] == [
        str(path.resolve()) for path in expected_database_files
    ]
    assert manifest["phrogs_v4"]["high_confidence_terms"] == [
        "integrase",
        "excisionase",
        "site-specific recombinase",
        "lysogeny repressor",
    ]
    assert manifest["phrogs_v4"]["additional_high_confidence_terms"] == [
        "anti-repressor",
        "ci-like repressor",
    ]
    assert manifest["phrogs_v4"]["selection_scope"] == (
        "integration/excision category plus unambiguous lifecycle-regulator annotations"
    )


@pytest.mark.parametrize("identifier", ("0", "01", "phrog_01", "PHROG_1", " 1", "1 ", "-1"))
def test_phrogs_annotation_identifier_normalization_rejects_noncanonical_aliases(identifier):
    """Numeric source IDs map exactly once to lookup IDs without permissive aliases."""
    with pytest.raises(ValueError, match="noncanonical PHROG identifier"):
        external_assets._normalize_phrogs_annotation_identifier(identifier)


def test_pinned_phrogs_pair_records_exact_unsearchable_profile_families():
    """Known release omissions are explicit; arbitrary annotation/profile drift is still rejected."""
    missing_ids = ("phrog_49658", "phrog_50550", "phrog_77239", "phrog_81686", "phrog_87299")
    rows = [["phrog_1", "Integrase", "integration and excision", "high_confidence", "integrase"]]
    rows.extend(
        [profile_id, "Integrase", "integration and excision", "high_confidence", "integrase"]
        for profile_id in missing_ids
    )

    searchable, excluded = external_assets._reconcile_phrogs_profile_lookup_rows(
        rows,
        {"phrog_1", "phrog_2"},
        annotation_sha256=external_assets.DEFAULT_PHROGS_ANNOTATION_SHA256,
    )

    assert searchable == [rows[0]]
    assert excluded == rows[1:]
    with pytest.raises(ValueError, match="IDs absent"):
        external_assets._reconcile_phrogs_profile_lookup_rows(
            [*rows, ["phrog_999", "Integrase", "integration and excision", "high_confidence", "integrase"]],
            {"phrog_1", "phrog_2"},
            annotation_sha256=external_assets.DEFAULT_PHROGS_ANNOTATION_SHA256,
        )
    with pytest.raises(ValueError, match="IDs absent"):
        external_assets._reconcile_phrogs_profile_lookup_rows(
            rows,
            {"phrog_1", "phrog_2"},
            annotation_sha256="0" * 64,
        )


def test_prepare_phrogs_safety_metadata_records_profile_identity_search_contract(tmp_path):
    """PHROGs metadata records the native profile-query contract and profile lookup join."""
    external_dir = tmp_path / "external"
    annotation_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    sequence_database = _write_phrogs_v4_fixture(annotation_path)
    profile_database = _write_phrogs_profile_database(
        external_dir / "phrogs" / "phrogs_mmseqs_db" / "phrogs_profile_db"
    )
    archive_path = external_dir / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
    manifest_path = external_dir / "safety" / "asset_manifest.yaml"

    external_assets._prepare_phrogs_safety_metadata(
        external_dir,
        manifest_path=manifest_path,
        annotation_sha256=external_assets._sha256_file(annotation_path),
        sequence_database=sequence_database,
        profile_database=profile_database,
        **_profile_metadata_kwargs(profile_database, archive_path),
    )

    profile_record = yaml.safe_load(manifest_path.read_text())["phrogs_v4"]["profile_database"]
    assert profile_record["path"] == str(profile_database.resolve())
    assert profile_record["search_orientation"] == "phrog_profile_query_vs_orf_target"
    assert profile_record["search_profile_scope"] == "full_phrogs_v4_profile_database"
    assert profile_record["lookup_join_policy"] == "classify_only_profile_ids_present_in_pinned_lookup"
    assert profile_record["output_fields"] == [
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
    ]
    assert profile_record["units"] == {"pident": "percent", "qcov": "fraction", "tcov": "fraction"}
    assert profile_record["query_id_pattern"] == r"^phrog_[1-9][0-9]*$"
    assert profile_record["query_ids_join_lookup"] is True
    profile_files = sorted(profile_database.parent.glob(f"{profile_database.name}*"))
    profile_tree_files = sorted([*profile_files, profile_database.parent / "VERSION_1_8_0"])
    assert profile_record["files"] == [str(path.resolve()) for path in profile_files]
    assert profile_record["profile_id_inventory"] == external_assets._phrogs_profile_id_inventory(profile_database)
    assert profile_record["extracted_tree"] == {
        "path": str(profile_database.parent.resolve()),
        "sha256": external_assets._sha256_file_inventory(profile_database.parent, profile_tree_files),
        "files": [str(path.resolve()) for path in profile_tree_files],
    }
    assert profile_record["provenance"] == {
        "source_url": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL,
        "archive_observed_sha256": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SHA256,
        "archive_expected_sha256": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SHA256,
        "archive_published_sha256": None,
        "archive_published_md5": "a63c485241b900a11989bd1821bfbb09",
        "archive_published_size": 656_171_247,
        "retrieved_at": "2026-08-08T00:00:00Z",
        "release": "Pharokka database v1.8.0",
        "dataset_release": "PHROGs v4",
        "doi": "10.5281/zenodo.17110353",
        "license": "CC BY 4.0",
        "citation": "Pharokka database v1.8.0 (DOI: 10.5281/zenodo.17110353).",
        "minimum_mmseqs_version": "14",
        "built_with_mmseqs_version": "18.8cc5c",
    }


def test_prepare_phrogs_safety_metadata_publishes_exact_policy_search_database(tmp_path, monkeypatch):
    """Safety publication derives its smaller search DB from the generated policy lookup."""
    external_dir = tmp_path / "external"
    annotation_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    sequence_database = _write_phrogs_v4_fixture(annotation_path)
    profile_database = _write_phrogs_profile_database(
        external_dir / "phrogs" / "phrogs_mmseqs_db" / "phrogs_profile_db"
    )
    archive_path = external_dir / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
    manifest_path = external_dir / "safety" / "asset_manifest.yaml"
    mmseqs = external_dir / "bin" / "mmseqs"
    mmseqs.parent.mkdir(parents=True)
    mmseqs.write_text("mmseqs-v15\n")
    mmseqs.chmod(0o755)
    derived_record = {"schema_version": 1, "path": "derived"}
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return derived_record

    monkeypatch.setattr(external_assets, "_prepare_phrogs_safety_search_database", fake_prepare)

    external_assets._prepare_phrogs_safety_metadata(
        external_dir,
        manifest_path=manifest_path,
        annotation_sha256=external_assets._sha256_file(annotation_path),
        sequence_database=sequence_database,
        profile_database=profile_database,
        mmseqs_path=mmseqs,
        **_profile_metadata_kwargs(profile_database, archive_path),
    )

    section = yaml.safe_load(manifest_path.read_text())["phrogs_v4"]
    assert section["search_database"] == derived_record
    assert captured == {
        "profile_database": profile_database,
        "safety_lookup": Path(section["lookup_path"]),
        "output_root": Path(section["lookup_path"]).parent / "safety_search_database",
        "mmseqs_path": mmseqs,
    }


def test_public_phrogs_metadata_rejects_caller_supplied_profile_provenance(tmp_path):
    """Caller-selected profile bytes cannot be stamped as the official full PHROGs release."""
    external_dir = tmp_path / "external"
    annotation_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    annotation_path.parent.mkdir(parents=True)
    annotation_rows = ["phrog\tcolor\tannot\tcategory"]
    annotation_rows.extend(f"phrog_{index}\t#000000\tIntegrase\tintegration and excision" for index in range(1, 58))
    annotation_rows.extend(
        f"phrog_{index}\t#000000\tPutative recombinase\tintegration and excision" for index in range(58, 110)
    )
    annotation_path.write_text("\n".join(annotation_rows) + "\n")
    sequence_database = _write_mmseqs_padded_database(annotation_path.parent / "phrogs_gpu_seq_db_pad")
    profile_database = _write_phrogs_profile_database(
        annotation_path.parent / "phrogs_mmseqs_db" / "phrogs_profile_db",
        profile_ids=tuple(f"phrog_{index}" for index in range(1, 111)),
    )
    archive_path = external_dir / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
    manifest_path = external_dir / "safety" / "asset_manifest.yaml"

    with pytest.raises(RuntimeError, match="not a public publishing API"):
        external_assets.prepare_phrogs_safety_metadata(
            external_dir,
            manifest_path=manifest_path,
            annotation_sha256=external_assets._sha256_file(annotation_path),
            sequence_database=sequence_database,
            profile_database=profile_database,
            **_profile_metadata_kwargs(profile_database, archive_path),
        )

    assert not manifest_path.exists()


def test_public_phrogs_metadata_rejects_a_forged_private_shaped_capability(tmp_path):
    """The public publisher cannot be reopened by constructing its private-looking arguments."""
    external_dir = tmp_path / "external"
    annotation_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        "phrog\tcolor\tannot\tcategory\nphrog_1\t#000000\tIntegrase\tintegration and excision\n"
    )
    sequence_database = _write_mmseqs_padded_database(annotation_path.parent / "phrogs_gpu_seq_db_pad")
    profile_database = _write_phrogs_profile_database(
        annotation_path.parent / "phrogs_mmseqs_db" / "phrogs_profile_db",
        profile_ids=("phrog_1", "phrog_2"),
    )
    archive_path = external_dir / "downloads" / "caller-supplied-pharokka.tar.gz"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"caller-supplied profile archive\n")
    profile_tree = external_assets._phrogs_profile_tree_files(profile_database)
    forged_capability = external_assets._VerifiedPhrogsProfile(
        archive_path=archive_path,
        observed_archive_sha256=external_assets._sha256_file(archive_path),
        extracted_dir=profile_database.parent,
        profile_database=profile_database,
        database_sha256=external_assets._complete_phrogs_profile_database(profile_database)[0],
        tree_sha256=external_assets._sha256_file_inventory(profile_database.parent, profile_tree),
        profile_id_inventory=external_assets._phrogs_profile_id_inventory(profile_database),
        _authority=external_assets._VERIFIED_PHROGS_PROFILE_AUTHORITY,
    )

    with pytest.raises(RuntimeError, match="not a public publishing API"):
        external_assets.prepare_phrogs_safety_metadata(
            external_dir,
            annotation_sha256=external_assets._sha256_file(annotation_path),
            sequence_database=sequence_database,
            profile_database=profile_database,
            profile_retrieved_at="2026-08-08T00:00:00Z",
            _verified_profile=forged_capability,
        )


def test_prepare_phrogs_safety_metadata_rejects_empty_or_duplicate_lookup(tmp_path):
    """An empty or ambiguous PHROGs lookup must fail before it can be trusted."""
    annotation_path = tmp_path / "external" / "phrogs" / "phrog_annot_v4.tsv"
    sequence_database = _write_phrogs_v4_fixture(annotation_path, duplicate=True)
    profile_database = annotation_path.parent / "phrogs_mmseqs_db" / "phrogs_profile_db"
    profile_archive = (
        annotation_path.parent.parent / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
    )

    with pytest.raises(ValueError, match="duplicate"):
        external_assets._prepare_phrogs_safety_metadata(
            tmp_path / "external",
            annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
            sequence_database=sequence_database,
            profile_database=profile_database,
            **_profile_metadata_kwargs(profile_database, profile_archive),
        )

    sequence_database = _write_phrogs_v4_fixture(annotation_path, empty=True)
    with pytest.raises(ValueError, match="empty"):
        external_assets._prepare_phrogs_safety_metadata(
            tmp_path / "external",
            annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
            sequence_database=sequence_database,
            profile_database=profile_database,
            **_profile_metadata_kwargs(profile_database, profile_archive),
        )


def test_prepare_phrogs_safety_metadata_real_schema_has_bounded_expected_confidence_counts(tmp_path):
    """The pinned PHROGs v4 schema yields the reviewed 109-row confidence partition."""
    annotation_path = tmp_path / "external" / "phrogs" / "phrog_annot_v4.tsv"
    annotation_path.parent.mkdir(parents=True)
    rows = ["phrog\tcolor\tannot\tcategory"]
    profile_ids = tuple(f"phrog_{index}" for index in range(1, 111))
    rows.extend(f"phrog_{index}\t#000000\tIntegrase\tintegration and excision" for index in range(1, 58))
    rows.extend(f"phrog_{index}\t#000000\tPutative recombinase\tintegration and excision" for index in range(58, 110))
    annotation_path.write_text("\n".join(rows) + "\n")
    sequence_database = _write_mmseqs_padded_database(annotation_path.parent / "phrogs_gpu_seq_db_pad")
    profile_database = _write_phrogs_profile_database(
        annotation_path.parent / "phrogs_mmseqs_db" / "phrogs_profile_db",
        profile_ids=profile_ids,
    )
    profile_archive = (
        annotation_path.parent.parent / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
    )

    asset = external_assets._prepare_phrogs_safety_metadata(
        tmp_path / "external",
        annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        sequence_database=sequence_database,
        profile_database=profile_database,
        **_profile_metadata_kwargs(profile_database, profile_archive),
    )

    confidence_rows = [line.split("\t") for line in asset.path.read_text().splitlines()[1:]]
    assert len(confidence_rows) == 109
    assert sum(row[3] == "high_confidence" for row in confidence_rows) == 57
    assert sum(row[3] == "review" for row in confidence_rows) == 52


def test_prepare_phrogs_safety_metadata_rejects_profile_limited_to_safety_lookup_subset(tmp_path):
    """The full PHROGs profile must contain families beyond the pinned 109-style safety lookup."""
    external_dir = tmp_path / "external"
    annotation_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    sequence_database = _write_phrogs_v4_fixture(annotation_path)
    profile_database = _write_phrogs_profile_database(
        external_dir / "phrogs" / "phrogs_mmseqs_db" / "phrogs_profile_db",
        profile_ids=("phrog_1", "phrog_2", "phrog_3", "phrog_4", "phrog_6"),
    )
    archive_path = external_dir / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name

    with pytest.raises(ValueError, match="beyond the pinned safety lookup"):
        external_assets._prepare_phrogs_safety_metadata(
            external_dir,
            annotation_sha256=external_assets._sha256_file(annotation_path),
            sequence_database=sequence_database,
            profile_database=profile_database,
            **_profile_metadata_kwargs(profile_database, archive_path),
        )


def test_complete_phrogs_sequence_database_rejects_a_thin_nonsearchable_prefix(tmp_path):
    """A base file plus dbtype is not a complete MMseqs padded searchable database."""
    sequence_database = tmp_path / "phrogs_gpu_seq_db_pad"
    sequence_database.write_bytes(b"MPEPTIDE\n\0")
    Path(f"{sequence_database}.dbtype").write_bytes(b"\x00\x00\x00\x00")

    with pytest.raises(FileNotFoundError, match="complete MMseqs"):
        external_assets._complete_phrogs_sequence_database(sequence_database)


@pytest.mark.parametrize(
    "missing_path",
    (
        lambda database: Path(f"{database}.lookup"),
        lambda database: Path(f"{database}.source"),
        lambda database: database.parent / "VERSION_1_8_0",
    ),
)
def test_complete_phrogs_profile_database_requires_official_pharokka_identity_sidecars(tmp_path, missing_path):
    """A complete safety profile needs Pharokka's lookup, source, and release identity files."""
    profile_database = _write_phrogs_profile_database(tmp_path / "phrogs_profile_db")
    missing_path(profile_database).unlink()

    with pytest.raises(FileNotFoundError, match="complete MMseqs profile"):
        external_assets._complete_phrogs_profile_database(profile_database)


@pytest.mark.parametrize(
    "lookup_contents, error",
    (
        (b"not-an-integer\tphrog_1\t0\n", "malformed"),
        (b"0\tnot_a_phrog\t0\n", "noncanonical"),
        (b"0\tphrog_1\t0\n0\tphrog_2\t0\n", "duplicate"),
        (b"0\tphrog_1\t0\n1\tphrog_1\t0\n", "duplicate"),
    ),
)
def test_prepare_phrogs_safety_metadata_rejects_noncanonical_or_duplicate_profile_lookup_ids(
    tmp_path, lookup_contents, error
):
    """The profile's MMseqs lookup, not a boolean, proves PHROG query identity."""
    external_dir = tmp_path / "external"
    annotation_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    sequence_database = _write_phrogs_v4_fixture(annotation_path)
    profile_database = _write_phrogs_profile_database(
        external_dir / "phrogs" / "phrogs_mmseqs_db" / "phrogs_profile_db"
    )
    Path(f"{profile_database}.lookup").write_bytes(lookup_contents)
    archive_path = external_dir / "downloads" / "pharokka_v1.8.0_databases.tar.gz"

    with pytest.raises(ValueError, match=error):
        external_assets._prepare_phrogs_safety_metadata(
            external_dir,
            annotation_sha256=external_assets._sha256_file(annotation_path),
            sequence_database=sequence_database,
            profile_database=profile_database,
            **_profile_metadata_kwargs(profile_database, archive_path),
        )


@pytest.mark.parametrize("profile_identifier", ("phrog_\uff11\uff12", "phrog_\u0660\u0661", "phrog_01"))
def test_phrogs_profile_lookup_requires_ascii_canonical_identifiers(tmp_path, profile_identifier):
    """Profile lookup IDs must be ASCII canonical PHROG family identifiers, never numeric aliases."""
    profile_database = _write_phrogs_profile_database(tmp_path / "phrogs_profile_db")
    Path(f"{profile_database}.lookup").write_text(f"0\t{profile_identifier}\t0\n")

    with pytest.raises(ValueError, match="noncanonical"):
        external_assets._phrogs_profile_ids(profile_database)


def test_prepare_phrogs_safety_metadata_rejects_annotation_ids_absent_from_profile_lookup(tmp_path):
    """Every pinned lysogeny-table PHROG must be queryable in the validated profile DB."""
    external_dir = tmp_path / "external"
    annotation_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    sequence_database = _write_phrogs_v4_fixture(annotation_path)
    profile_database = _write_phrogs_profile_database(
        external_dir / "phrogs" / "phrogs_mmseqs_db" / "phrogs_profile_db",
        profile_ids=("phrog_1",),
    )
    archive_path = external_dir / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name

    with pytest.raises(ValueError, match="absent from the verified profile database"):
        external_assets._prepare_phrogs_safety_metadata(
            external_dir,
            annotation_sha256=external_assets._sha256_file(annotation_path),
            sequence_database=sequence_database,
            profile_database=profile_database,
            **_profile_metadata_kwargs(profile_database, archive_path),
        )


def test_snapshot_phrogs_profile_database_copies_only_identity_bearing_profile_inventory(tmp_path):
    """A Pharokka bundle's unrelated CARD/VFDB files cannot enter the PHROGs safety snapshot digest."""
    source_root = tmp_path / "pharokka_bundle"
    profile_database = _write_phrogs_profile_database(source_root / "phrogs_profile_db")
    (source_root / "CARD").write_bytes(b"unrelated CARD database")
    (source_root / "vfdb").write_bytes(b"unrelated VFDB database")

    snapshot_database, snapshot_root = external_assets._snapshot_phrogs_profile_database(
        profile_database, tmp_path / "safety"
    )

    assert not (snapshot_root / "CARD").exists()
    assert not (snapshot_root / "vfdb").exists()
    snapshot_files = external_assets._phrogs_profile_tree_files(snapshot_database)
    assert [path.name for path in snapshot_files] == [
        "VERSION_1_8_0",
        "phrogs_profile_db",
        "phrogs_profile_db.dbtype",
        "phrogs_profile_db.index",
        "phrogs_profile_db.lookup",
        "phrogs_profile_db.source",
        "phrogs_profile_db_h",
        "phrogs_profile_db_h.dbtype",
        "phrogs_profile_db_h.index",
    ]


def test_safety_profile_source_is_the_versioned_pharokka_release():
    """Safety identity searches pin the compatible Pharokka database rather than PHROGs latest."""
    assert external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL == (
        "https://zenodo.org/record/17110353/files/pharokka_v1.8.0_databases.tar.gz"
    )
    assert external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5 == "a63c485241b900a11989bd1821bfbb09"


def test_prepare_phrogs_safety_profile_db_verifies_published_md5_before_extracting(tmp_path, monkeypatch):
    """A corrupt pinned Pharokka archive must fail before profile files are extracted or scanned."""
    extract_called = False

    def fake_download(_url, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"corrupt Pharokka archive")
        return output_path

    def fail_extract(*_args, **_kwargs):
        nonlocal extract_called
        extract_called = True
        pytest.fail("extraction ran before published MD5 verification")

    monkeypatch.setattr(external_assets, "_download", fake_download)
    monkeypatch.setattr(external_assets, "_verify_file_size", lambda *_args, **_kwargs: 656_171_247)
    monkeypatch.setattr(external_assets, "_extract_tar", fail_extract)

    with pytest.raises(ValueError, match="MD5 mismatch"):
        external_assets.prepare_phrogs_safety_profile_db(tmp_path / "external")

    assert extract_called is False


def test_prepare_phrogs_safety_profile_db_verifies_published_size_before_extracting(tmp_path, monkeypatch):
    """A short Pharokka archive must fail before extraction even when its MD5 verifier is reached."""
    extract_called = False

    def fake_download(_url, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"short Pharokka archive")
        return output_path

    def fail_extract(*_args, **_kwargs):
        nonlocal extract_called
        extract_called = True
        pytest.fail("extraction ran before published size verification")

    monkeypatch.setattr(external_assets, "_download", fake_download)
    monkeypatch.setattr(external_assets, "_verify_md5", lambda *_args, **_kwargs: pytest.fail("MD5 ran first"))
    monkeypatch.setattr(external_assets, "_extract_tar", fail_extract)

    with pytest.raises(ValueError, match="size mismatch"):
        external_assets.prepare_phrogs_safety_profile_db(tmp_path / "external")

    assert extract_called is False


def test_prepare_phrogs_safety_profile_db_cleanly_reextracts_the_verified_archive(tmp_path, monkeypatch):
    """A digest-named existing extraction cannot be reused as an unverified safety profile tree."""
    download_calls = []
    extract_overwrite_flags = []

    def fake_download(url, output_path, **kwargs):
        download_calls.append((url, kwargs))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"verified Pharokka archive")
        return output_path

    def fake_extract(_archive_path, output_dir, *, overwrite):
        extract_overwrite_flags.append(overwrite)
        _write_phrogs_profile_database(output_dir / "phrogs_profile_db")
        return output_dir

    monkeypatch.setattr(external_assets, "_download", fake_download)
    monkeypatch.setattr(external_assets, "_verify_file_size", lambda *_args, **_kwargs: 656_171_247)
    monkeypatch.setattr(external_assets, "_verify_md5", lambda *_args, **_kwargs: "a63c485241b900a11989bd1821bfbb09")
    monkeypatch.setattr(external_assets, "_extract_tar", fake_extract)
    _mock_reviewed_phrogs_archive_sha256(monkeypatch)

    external_assets.prepare_phrogs_safety_profile_db(tmp_path / "external", overwrite=False)

    assert download_calls == [
        (
            external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL,
            {
                "overwrite": False,
                "insecure": False,
                "expected_sha256": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SHA256,
                "expected_size": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SIZE,
                "expected_md5": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5,
                "resume": True,
            },
        )
    ]
    assert extract_overwrite_flags == [True]


def test_prepare_phrogs_safety_profile_db_materializes_release_marker_after_archive_verification(
    tmp_path, monkeypatch
):
    """The real archive's absent marker is derived only after its published size and MD5 authenticate it."""
    events = []

    def fake_download(_url, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"verified Pharokka archive")
        return output_path

    def verify_size(_path, _expected_size):
        events.append("size")
        return external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SIZE

    def verify_md5(_path, _expected_md5):
        events.append("md5")
        return external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5

    def extract_without_marker(_archive_path, output_dir, *, overwrite):
        assert overwrite is True
        events.append("extract")
        profile_database = _write_phrogs_profile_database(output_dir / "phrogs_profile_db")
        (profile_database.parent / external_assets.PHROGS_PROFILE_RELEASE_MARKER).unlink()
        return output_dir

    monkeypatch.setattr(external_assets, "_download", fake_download)
    monkeypatch.setattr(external_assets, "_verify_file_size", verify_size)
    monkeypatch.setattr(external_assets, "_verify_md5", verify_md5)
    monkeypatch.setattr(external_assets, "_extract_tar", extract_without_marker)
    _mock_reviewed_phrogs_archive_sha256(monkeypatch)

    asset = external_assets.prepare_phrogs_safety_profile_db(tmp_path / "external")

    marker = asset.path / external_assets.PHROGS_PROFILE_RELEASE_MARKER
    assert events == ["size", "md5", "extract"]
    assert marker.read_text() == "1.8.0\n"


def test_prepare_phrogs_safety_profile_db_accepts_published_empty_release_sentinel(tmp_path, monkeypatch):
    """The published Pharokka archive represents its release marker as a zero-byte sentinel."""

    def fake_download(_url, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"verified Pharokka archive")
        return output_path

    def extract_with_empty_sentinel(_archive_path, output_dir, *, overwrite):
        assert overwrite is True
        profile_database = _write_phrogs_profile_database(output_dir / "phrogs_profile_db")
        (profile_database.parent / external_assets.PHROGS_PROFILE_RELEASE_MARKER).write_bytes(b"")
        return output_dir

    monkeypatch.setattr(external_assets, "_download", fake_download)
    monkeypatch.setattr(
        external_assets,
        "_verify_file_size",
        lambda *_args, **_kwargs: external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SIZE,
    )
    monkeypatch.setattr(
        external_assets, "_verify_md5", lambda *_args, **_kwargs: external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5
    )
    monkeypatch.setattr(external_assets, "_extract_tar", extract_with_empty_sentinel)
    _mock_reviewed_phrogs_archive_sha256(monkeypatch)

    asset = external_assets.prepare_phrogs_safety_profile_db(tmp_path / "external")

    marker = asset.path / external_assets.PHROGS_PROFILE_RELEASE_MARKER
    assert marker.is_file()
    assert marker.read_bytes() == b""


def test_prepare_phrogs_safety_profile_db_rejects_conflicting_archive_release_marker(tmp_path, monkeypatch):
    """An archive-supplied marker cannot contradict the release bound by URL, size, and MD5."""

    def fake_download(_url, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"verified Pharokka archive")
        return output_path

    def extract_with_conflicting_marker(_archive_path, output_dir, *, overwrite):
        assert overwrite is True
        profile_database = _write_phrogs_profile_database(output_dir / "phrogs_profile_db")
        (profile_database.parent / external_assets.PHROGS_PROFILE_RELEASE_MARKER).write_text("9.9.9\n")
        return output_dir

    monkeypatch.setattr(external_assets, "_download", fake_download)
    monkeypatch.setattr(
        external_assets,
        "_verify_file_size",
        lambda *_args, **_kwargs: external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SIZE,
    )
    monkeypatch.setattr(
        external_assets, "_verify_md5", lambda *_args, **_kwargs: external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5
    )
    monkeypatch.setattr(external_assets, "_extract_tar", extract_with_conflicting_marker)
    _mock_reviewed_phrogs_archive_sha256(monkeypatch)

    with pytest.raises(ValueError, match="release marker"):
        external_assets.prepare_phrogs_safety_profile_db(tmp_path / "external")


def test_staged_safety_manifest_rejects_raw_phrogs_database_without_identity_search_contract(tmp_path):
    """A raw FAA-derived PHROGs prefix cannot stand in for PHROG-identity profile search."""
    safety_dir = tmp_path / "safety"
    manifest = {"schema_version": 3}
    for name in ("amrfinder", "toxins", "phrogs"):
        section, record = _materialize_mock_safety_manifest_section(name, safety_dir)
        if section == "phrogs_v4":
            record.pop("profile_database")
        manifest[section] = record

    with pytest.raises(RuntimeError, match="profile_database"):
        external_assets._validate_staged_safety_manifest(manifest, verify_asset_paths=True)


def test_staged_safety_manifest_rejects_malformed_phrogs_profile_observed_archive_digest(tmp_path):
    """A profile manifest cannot present arbitrary text as its observed archive SHA-256 evidence."""
    safety_dir = tmp_path / "safety"
    manifest = {"schema_version": 3}
    for name in ("amrfinder", "toxins", "phrogs"):
        section, record = _materialize_mock_safety_manifest_section(name, safety_dir)
        manifest[section] = record
    manifest["phrogs_v4"]["profile_database"]["provenance"]["archive_observed_sha256"] = "not-a-sha256"

    with pytest.raises(RuntimeError, match="observed archive SHA-256"):
        external_assets._validate_staged_safety_manifest(manifest, verify_asset_paths=True)


def test_staged_safety_manifest_rejects_profile_without_verified_archive_lineage(tmp_path):
    """Pinned labels cannot authenticate a caller-created profile tree without an archive lineage."""
    safety_dir = tmp_path / "safety"
    manifest = {"schema_version": 3}
    for name in ("amrfinder", "toxins", "phrogs"):
        section, record = _materialize_mock_safety_manifest_section(name, safety_dir)
        manifest[section] = record

    with pytest.raises(RuntimeError, match="verified archive cache"):
        external_assets._validate_staged_safety_manifest(manifest, verify_asset_paths=True)


def test_prepare_external_assets_with_safety_rejects_incomplete_phrogs_profile_database(tmp_path, monkeypatch):
    """Safety setup must reject a thin profile extracted from the verified Pharokka archive."""
    external_dir = tmp_path / "external"
    _write_phrogs_v4_fixture(external_dir / "phrogs" / "phrog_annot_v4.tsv")
    archive_path = external_dir / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name

    def extract_thin_profile(path, output_dir, *, overwrite):
        assert path == archive_path
        assert overwrite is True
        thin_profile = Path(output_dir) / "phrogs_profile_db"
        thin_profile.parent.mkdir(parents=True)
        thin_profile.write_bytes(b"profile\n\0")
        Path(f"{thin_profile}.dbtype").write_bytes(b"\x10\x00\x00\x00")
        return output_dir

    monkeypatch.setattr(external_assets, "_verify_file_size", lambda *_args, **_kwargs: 656_171_247)
    monkeypatch.setattr(
        external_assets, "_verify_md5", lambda *_args, **_kwargs: external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5
    )
    monkeypatch.setattr(external_assets, "_extract_tar", extract_thin_profile)
    _mock_reviewed_phrogs_archive_sha256(monkeypatch)
    monkeypatch.setattr(
        external_assets,
        "prepare_amrfinder_plus",
        lambda *_args, **_kwargs: pytest.fail("AMRFinder ran before PHROGs profile preflight"),
    )

    with pytest.raises(FileNotFoundError, match="complete MMseqs profile"):
        prepare_external_assets(
            external_dir,
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=False,
            download_arc_evo2=False,
            download_large_databases=False,
            configure_lovis4u=False,
            with_safety=True,
        )


def test_prepare_external_assets_with_safety_rejects_unverified_reused_pharokka_archive(tmp_path, monkeypatch):
    """A nonempty shared archive cannot bless an arbitrary cached PHROGs profile tree."""
    external_dir = tmp_path / "external"
    _write_phrogs_v4_fixture(external_dir / "phrogs" / "phrog_annot_v4.tsv")
    monkeypatch.setattr(
        external_assets,
        "prepare_amrfinder_plus",
        lambda *_args, **_kwargs: pytest.fail("AMRFinder ran before reused Pharokka archive verification"),
    )

    with pytest.raises(ValueError, match="size mismatch"):
        prepare_external_assets(
            external_dir,
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=False,
            download_arc_evo2=False,
            download_large_databases=False,
            configure_lovis4u=False,
            with_safety=True,
        )


def test_prepare_external_assets_reuses_profile_extracted_from_verified_archive_not_shared_tree(tmp_path, monkeypatch):
    """A full safety profile must come from the verified archive, not a mutable 109-ID shared tree."""
    external_dir = tmp_path / "external"
    _write_phrogs_v4_fixture(external_dir / "phrogs" / "phrog_annot_v4.tsv")
    mmseqs = external_dir / "bin" / "mmseqs"
    mmseqs.parent.mkdir(parents=True)
    mmseqs.write_text("#!/usr/bin/env bash\n")
    mmseqs.chmod(0o755)
    shared_profile = _write_phrogs_profile_database(
        external_dir / "phrogs" / "phrogs_mmseqs_db" / "phrogs_profile_db",
        profile_ids=tuple(f"phrog_{index}" for index in range(1, 110)),
    )
    archive_path = external_dir / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
    verification_calls = []
    extraction_calls = []
    native_amrfinder_bin_dir = tmp_path / "native-amrfinder" / "bin"

    def verify_size(path, expected_size):
        verification_calls.append(("size", path, expected_size))
        assert path == archive_path
        assert expected_size == external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SIZE
        return expected_size

    def verify_md5(path, expected_md5):
        verification_calls.append(("md5", path, expected_md5))
        assert path == archive_path
        assert expected_md5 == external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5
        return expected_md5

    def extract_verified_archive(path, output_dir, *, overwrite):
        extraction_calls.append((path, output_dir, overwrite))
        assert path == archive_path
        assert overwrite is True
        _write_phrogs_profile_database(
            output_dir / "phrogs_profile_db",
            profile_ids=tuple(f"phrog_{index}" for index in range(1, 111)),
        )
        return output_dir

    def fail_after_profile_snapshot(*_args, **kwargs):
        snapshot_database = Path(kwargs["safety_dir"]) / "phrogs" / "profile_database" / "phrogs_profile_db"
        assert "phrog_110" in external_assets._phrogs_profile_ids(snapshot_database)
        assert external_assets._phrogs_profile_id_inventory(snapshot_database)["count"] == 110
        assert shared_profile != snapshot_database
        assert kwargs["source_bin_dir"] == native_amrfinder_bin_dir
        assert kwargs["source_repository"] == "https://github.com/ncbi/amr.git"
        assert kwargs["source_revision"] == "76bf8527b3cabc6a08fdc0e20783e594bba162e3"
        raise RuntimeError("injected failure after verified profile snapshot")

    monkeypatch.setattr(external_assets, "_verify_file_size", verify_size)
    monkeypatch.setattr(external_assets, "_verify_md5", verify_md5)
    monkeypatch.setattr(external_assets, "_extract_tar", extract_verified_archive)
    _mock_reviewed_phrogs_archive_sha256(monkeypatch)
    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", fail_after_profile_snapshot)

    with pytest.raises(RuntimeError, match="injected failure after verified profile snapshot"):
        prepare_external_assets(
            external_dir,
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=False,
            download_arc_evo2=False,
            download_large_databases=False,
            configure_lovis4u=False,
            with_safety=True,
            amrfinder_source_bin_dir=native_amrfinder_bin_dir,
            amrfinder_source_repository="https://github.com/ncbi/amr.git",
            amrfinder_source_revision="76bf8527b3cabc6a08fdc0e20783e594bba162e3",
        )

    expected_verification_pair = [
        ("size", archive_path, external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SIZE),
        ("md5", archive_path, external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5),
    ]
    assert verification_calls == expected_verification_pair * 4
    assert len(extraction_calls) == 2
    assert all(path == archive_path and overwrite is True for path, _output, overwrite in extraction_calls)
    assert extraction_calls[0][1] != extraction_calls[1][1]


def test_prepare_phrogs_gpu_sequence_db_requests_a_lookup_sidecar(tmp_path, monkeypatch):
    """The generated padded target DB includes the lookup required by the pinned complete contract."""
    external_dir = tmp_path / "external"
    (external_dir / "phrogs").mkdir(parents=True)
    extracted_dir = tmp_path / "FAA_phrog"
    extracted_dir.mkdir()
    (extracted_dir / "phrogs.faa").write_text(">phrog_1\nMPEPTIDE\n")
    archive_path = tmp_path / "FAA_phrog.tar.gz"
    archive_path.write_bytes(b"archive")
    calls = []

    def fake_download(*_args, **_kwargs):
        return archive_path

    def fake_extract(*_args, **_kwargs):
        return extracted_dir

    def fake_run(command, check):
        calls.append((command, check))
        if command[1] == "makepaddedseqdb":
            _write_mmseqs_padded_database(Path(command[3]))

    monkeypatch.setattr(external_assets, "_download", fake_download)
    monkeypatch.setattr(external_assets, "_extract_tar", fake_extract)
    monkeypatch.setattr("subprocess.run", fake_run)

    asset = external_assets.prepare_phrogs_gpu_sequence_db(external_dir)
    sequence_db = external_dir / "phrogs" / "phrogs_gpu_seq_db"
    padded_db = external_dir / "phrogs" / "phrogs_gpu_seq_db_pad"
    assert calls == [
        (["mmseqs", "createdb", str(external_dir / "phrogs" / "FAA_phrog_combined.faa"), str(sequence_db)], True),
        (
            [
                "mmseqs",
                "makepaddedseqdb",
                str(sequence_db),
                str(padded_db),
                "--write-lookup",
                "1",
            ],
            True,
        ),
    ]
    assert asset.path == padded_db


def test_phage_safety_assets_recipe_pins_scanner_sources_and_expected_roles():
    """The tracked recipe fixes scanner provenance while runtime digests stay out of Git."""
    recipe_path = Path(__file__).resolve().parents[3] / "configs" / "phage_safety_assets.yaml"

    recipe = yaml.safe_load(recipe_path.read_text())

    assert recipe["schema_version"] == 3
    assert recipe["amrfinder_plus"]["release"] == "amrfinder_v4.2.7"
    assert recipe["amrfinder_plus"]["release_url"] == (
        "https://github.com/ncbi/amr/releases/download/amrfinder_v4.2.7/amrfinder_binaries_v4.2.7.tar.gz"
    )
    assert recipe["amrfinder_plus"]["archive_sha256"] == (
        "68045a8bccdbe3c5dcdf941bebe2352ed419758a9914c41f48f0bbbd6fbade56"
    )
    assert recipe["amrfinder_plus"]["citation"]
    assert recipe["amrfinder_plus"]["require_recorded_amrfinder_index"] is True
    assert recipe["amrfinder_plus"]["require_contained_latest_symlink"] is True
    assert external_assets.DEFAULT_AMRFINDER_URL == recipe["amrfinder_plus"]["release_url"]
    assert external_assets.DEFAULT_AMRFINDER_SHA256 == recipe["amrfinder_plus"]["archive_sha256"]
    assert recipe["toxin_reference"]["query"] == DEFAULT_UNIPROT_TOXIN_QUERY
    assert recipe["toxin_reference"]["license"] == "CC BY 4.0"
    assert recipe["phrogs_v4"]["annotation_url"].endswith("phrog_annot_v4.tsv")
    assert recipe["phrogs_v4"]["annotation_sha256"] == (
        "502f96101597c21133bcce5711803e0b95e0c162cd4e86425c352549bd95e8c2"
    )
    assert external_assets.DEFAULT_PHROGS_ANNOTATION_SHA256 == recipe["phrogs_v4"]["annotation_sha256"]
    assert recipe["phrogs_v4"]["integration_excision_category"] == "integration and excision"
    assert recipe["phrogs_v4"]["selection_scope"] == external_assets.PHROGS_LYSOGENY_SELECTION_SCOPE
    assert recipe["phrogs_v4"]["additional_high_confidence_terms"] == list(
        external_assets.PHROGS_ADDITIONAL_LIFECYCLE_HIGH_CONFIDENCE_TERMS
    )
    assert recipe["phrogs_v4"]["citation"]
    assert recipe["phrogs_v4"]["use_terms"]
    profile_recipe = recipe["phrogs_v4"]["profile_database"]
    assert profile_recipe == {
        "source_url": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL,
        "release": external_assets.PHROGS_PROFILE_RELEASE,
        "dataset_release": external_assets.PHROGS_PROFILE_DATASET_RELEASE,
        "archive_published_sha256": None,
        "archive_expected_sha256": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SHA256,
        "archive_published_md5": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5,
        "archive_published_size": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SIZE,
        "archive_digest_provenance": (
            "verify the separately reviewed expected SHA-256 plus the published MD5 and size before extraction; "
            "record the matching observed SHA-256 at preparation; no source-published SHA-256 is asserted"
        ),
        "doi": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_DOI,
        "license": external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_LICENSE,
        "citation": external_assets.PHROGS_PROFILE_SOURCE_CITATION,
        "minimum_mmseqs_version": external_assets.PHROGS_PROFILE_MIN_MMSEQS_VERSION,
        "built_with_mmseqs_version": external_assets.PHROGS_PROFILE_BUILDER_MMSEQS_VERSION,
        "release_marker": external_assets.PHROGS_PROFILE_RELEASE_MARKER,
        "search_orientation": "phrog_profile_query_vs_orf_target",
        "search_profile_scope": "full_phrogs_v4_profile_database",
        "lookup_join_policy": "classify_only_profile_ids_present_in_pinned_lookup",
        "output_fields": [
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
        ],
        "units": {"pident": "percent", "qcov": "fraction", "tcov": "fraction"},
        "query_id_pattern": "^phrog_[1-9][0-9]*$",
        "query_ids_join_lookup": True,
        "required_files": [
            "phrogs_profile_db",
            "phrogs_profile_db.dbtype",
            "phrogs_profile_db.index",
            "phrogs_profile_db.lookup",
            "phrogs_profile_db.source",
            "phrogs_profile_db_h",
            "phrogs_profile_db_h.dbtype",
            "phrogs_profile_db_h.index",
            "VERSION_1_8_0",
        ],
        "require_complete_profile_database": True,
    }
    assert recipe["phrogs_v4"]["high_confidence_terms"] == [
        "integrase",
        "excisionase",
        "site-specific recombinase",
        "lysogeny repressor",
    ]
    assert set(recipe["expected_file_roles"]) >= {
        "amrfinder_binary",
        "amrfinder_index",
        "amrfinder_database",
        "toxin_annotations",
        "toxin_fasta",
        "toxin_diamond_database",
        "phrogs_profile_database",
        "phrogs_lysogeny_table",
    }


def test_prepare_phrogs_annotation_enforces_the_reviewed_digest_and_scoped_tls_exception(tmp_path, monkeypatch):
    """Only the pinned PHROGs TSV may use the reviewed TLS exception."""
    calls = []
    annotation_path = tmp_path / "external" / "phrogs" / "phrog_annot_v4.tsv"

    def fake_download(url, output_path, *, overwrite, insecure, expected_sha256):
        calls.append((url, output_path, overwrite, insecure, expected_sha256))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("phrog\tcolor\tannot\tcategory\n")
        return output_path

    monkeypatch.setattr(external_assets, "_download", fake_download)

    asset = external_assets.prepare_phrogs_annotation(tmp_path / "external", insecure_downloads=True)

    assert asset.path == annotation_path
    assert calls == [
        (
            external_assets.DEFAULT_PHROGS_ANNOTATION_URL,
            annotation_path,
            False,
            True,
            "502f96101597c21133bcce5711803e0b95e0c162cd4e86425c352549bd95e8c2",
        )
    ]


def test_prepare_external_assets_with_safety_prepares_pinned_profile_without_raw_sequence_database(
    tmp_path, monkeypatch
):
    """Clean safety setup prepares its pinned profile without opting into the unpinned raw FAA database."""
    calls = []

    def require_profile(_external_dir, **kwargs):
        calls.append((kwargs["overwrite"], kwargs["verified_archive_path"]))
        raise RuntimeError("pinned profile requested")

    monkeypatch.setattr(external_assets, "prepare_phrogs_safety_profile_db", require_profile)
    monkeypatch.setattr(
        external_assets,
        "prepare_phrogs_gpu_sequence_db",
        lambda *_args, **_kwargs: pytest.fail("optional raw FAA database was prepared by default"),
    )
    monkeypatch.setattr(
        external_assets,
        "prepare_amrfinder_plus",
        lambda *_args, **_kwargs: pytest.fail("AMRFinder ran before the pinned PHROGs profile was prepared"),
    )

    with pytest.raises(RuntimeError, match="pinned profile requested"):
        prepare_external_assets(
            tmp_path / "external",
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=False,
            download_arc_evo2=False,
            download_large_databases=False,
            configure_lovis4u=False,
            with_safety=True,
        )

    assert calls == [(False, None)]


def test_prepare_external_assets_with_safety_prepares_all_safety_assets_after_prerequisites(tmp_path, monkeypatch):
    """Safety prepares its pinned profile before AMR, toxin, and PHROGs metadata after regular assets."""
    calls = []
    original_prepare_metadata = external_assets._prepare_verified_phrogs_safety_metadata

    def fake_asset(name):
        def prepare(*_args, **kwargs):
            calls.append(name)
            if name == "phrogs":
                metadata = original_prepare_metadata(
                    *_args,
                    annotation_sha256=external_assets._sha256_file(Path(kwargs["annotation_path"])),
                    **kwargs,
                )
                return PreparedAsset(name, metadata.path, metadata.detail)
            if name in {"amrfinder", "toxins", "phrogs"} and isinstance(kwargs.get("manifest"), dict):
                section, record = _materialize_mock_safety_manifest_section(name, Path(kwargs["safety_dir"]))
                kwargs["manifest"][section] = record
            return PreparedAsset(name, tmp_path / name, name)

        return prepare

    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_pyrodigal_wrapper", fake_asset("prodigal"))
    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_diamond", fake_asset("diamond"))
    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_hmmer", fake_asset("hmmer"))
    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_amrfinder_plus", fake_asset("amrfinder"))
    monkeypatch.setattr("bionemo.evo2_phage_gen.external_assets.prepare_toxin_reference", fake_asset("toxins"))
    monkeypatch.setattr(
        "bionemo.evo2_phage_gen.external_assets._prepare_verified_phrogs_safety_metadata",
        fake_asset("phrogs"),
    )
    _write_phrogs_v4_fixture(tmp_path / "external" / "phrogs" / "phrog_annot_v4.tsv")
    _mock_verified_phrogs_profile_archive(monkeypatch, tmp_path / "external")

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

    assert [asset.name for asset in assets] == [
        "prodigal",
        "diamond",
        "hmmer",
        "phrogs_safety_profile_db",
        "amrfinder",
        "toxins",
        "phrogs",
    ]
    assert calls == ["prodigal", "diamond", "hmmer", "amrfinder", "toxins", "phrogs"]


def test_prepare_external_assets_snapshots_mmseqs_into_safety_generation(tmp_path, monkeypatch):
    """PHROGs derivation receives a regular generation-owned executable, never the mutable outer link."""
    external_dir = tmp_path / "external"
    _write_phrogs_v4_fixture(external_dir / "phrogs" / "phrog_annot_v4.tsv")
    _mock_verified_phrogs_profile_archive(monkeypatch, external_dir)
    outer_mmseqs = external_dir / "bin" / "mmseqs"
    source_mmseqs = external_dir / "tools" / "mmseqs-source"
    source_mmseqs.parent.mkdir(parents=True, exist_ok=True)
    outer_mmseqs.replace(source_mmseqs)
    outer_mmseqs.symlink_to(source_mmseqs)

    monkeypatch.setattr(
        external_assets,
        "prepare_pyrodigal_wrapper",
        lambda *_args, **_kwargs: PreparedAsset("prodigal", tmp_path / "prodigal", "fixture"),
    )

    def staged_asset(name):
        def prepare(*_args, **kwargs):
            section, record = _materialize_mock_safety_manifest_section(name, Path(kwargs["safety_dir"]))
            kwargs["manifest"][section] = record
            return PreparedAsset(name, Path(kwargs["safety_dir"]) / name, name)

        return prepare

    observed = {}

    def prepare_phrogs(*_args, **kwargs):
        mmseqs = Path(kwargs["mmseqs_path"])
        assert mmseqs.is_file() and not mmseqs.is_symlink()
        assert mmseqs.is_relative_to((external_dir / "safety" / "generations").resolve())
        observed["mmseqs_bytes"] = mmseqs.read_bytes()
        raise RuntimeError("captured generation MMseqs")

    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", staged_asset("amrfinder"))
    monkeypatch.setattr(external_assets, "prepare_toxin_reference", staged_asset("toxins"))
    monkeypatch.setattr(external_assets, "_prepare_verified_phrogs_safety_metadata", prepare_phrogs)

    with pytest.raises(RuntimeError, match="captured generation MMseqs"):
        prepare_external_assets(
            external_dir,
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=False,
            download_arc_evo2=False,
            configure_lovis4u=False,
            with_safety=True,
        )

    assert outer_mmseqs.is_symlink()
    assert observed["mmseqs_bytes"] == source_mmseqs.read_bytes()


def test_prepare_external_assets_with_safety_does_not_leak_insecure_transport_to_amr_or_uniprot(tmp_path, monkeypatch):
    """The global compatibility flag is scoped away from GitHub and UniProt safety assets."""
    calls = {}
    _write_phrogs_v4_fixture(tmp_path / "external" / "phrogs" / "phrog_annot_v4.tsv")
    _mock_verified_phrogs_profile_archive(monkeypatch, tmp_path / "external")
    original_prepare_metadata = external_assets._prepare_verified_phrogs_safety_metadata

    def fake_asset(name):
        def prepare(*_args, **kwargs):
            calls[name] = kwargs.get("insecure_downloads")
            if name == "phrogs":
                metadata = original_prepare_metadata(
                    *_args,
                    annotation_sha256=external_assets._sha256_file(Path(kwargs["annotation_path"])),
                    **kwargs,
                )
                return PreparedAsset(name, metadata.path, metadata.detail)
            section, record = _materialize_mock_safety_manifest_section(name, Path(kwargs["safety_dir"]))
            kwargs["manifest"][section] = record
            return PreparedAsset(name, tmp_path / name, name)

        return prepare

    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", fake_asset("amrfinder"))
    monkeypatch.setattr(external_assets, "prepare_toxin_reference", fake_asset("toxins"))
    monkeypatch.setattr(external_assets, "_prepare_verified_phrogs_safety_metadata", fake_asset("phrogs"))

    prepare_external_assets(
        tmp_path / "external",
        download_mmseqs=False,
        download_dustmasker=False,
        download_diamond=False,
        download_hmmer=False,
        download_phrogs_annotation=False,
        download_arc_evo2=False,
        download_large_databases=False,
        configure_lovis4u=False,
        with_safety=True,
        insecure_downloads=True,
    )

    assert calls == {"amrfinder": False, "toxins": False, "phrogs": None}


def test_prepare_external_assets_rejects_structurally_incomplete_staged_manifest(tmp_path, monkeypatch):
    """All three safety records need their required digests/provenance before publication."""
    _write_phrogs_v4_fixture(tmp_path / "external" / "phrogs" / "phrog_annot_v4.tsv")
    _mock_verified_phrogs_profile_archive(monkeypatch, tmp_path / "external")

    def incomplete_asset(name):
        def prepare(*_args, **kwargs):
            section = _mock_safety_manifest_section(name)[0]
            kwargs["manifest"][section] = {"prepared": name}
            return PreparedAsset(name, tmp_path / name, name)

        return prepare

    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", incomplete_asset("amrfinder"))
    monkeypatch.setattr(external_assets, "prepare_toxin_reference", incomplete_asset("toxins"))
    monkeypatch.setattr(external_assets, "_prepare_verified_phrogs_safety_metadata", incomplete_asset("phrogs"))

    with pytest.raises(RuntimeError, match="required field"):
        prepare_external_assets(
            tmp_path / "external",
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=False,
            download_arc_evo2=False,
            download_large_databases=False,
            configure_lovis4u=False,
            with_safety=True,
        )


def test_prepare_external_assets_with_safety_failure_keeps_previous_manifest(tmp_path, monkeypatch):
    """A failed safety generation must not overwrite a previous complete trusted manifest."""
    external_dir = tmp_path / "external"
    manifest_path = external_dir / "safety" / "asset_manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    old_manifest_text = "schema_version: 1\ngeneration: old\n"
    manifest_path.write_text(old_manifest_text)
    _write_phrogs_v4_fixture(external_dir / "phrogs" / "phrog_annot_v4.tsv")
    _mock_verified_phrogs_profile_archive(monkeypatch, external_dir)

    def partial_amrfinder(*_args, **kwargs):
        staged_manifest = kwargs.get("manifest")
        if isinstance(staged_manifest, dict):
            staged_manifest["amrfinder_plus"] = {"generation": "new"}
        else:
            external_assets._update_safety_manifest(kwargs["manifest_path"], "amrfinder_plus", {"generation": "new"})
        return PreparedAsset("amrfinder_plus", tmp_path / "amrfinder", "partial")

    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", partial_amrfinder)
    monkeypatch.setattr(
        external_assets,
        "prepare_toxin_reference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected toxin failure")),
    )

    with pytest.raises(RuntimeError, match="injected toxin failure"):
        prepare_external_assets(
            external_dir,
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=False,
            download_arc_evo2=False,
            download_large_databases=False,
            configure_lovis4u=False,
            with_safety=True,
            safety_manifest=manifest_path,
        )

    assert manifest_path.read_text() == old_manifest_text


def test_safety_failure_preserves_every_digest_referenced_by_the_previous_generation(tmp_path, monkeypatch):
    """A real AMRFinder mutation followed by failure cannot alter an older trusted generation."""
    external_dir = tmp_path / "external"
    bin_dir = external_dir / "bin"
    bin_dir.mkdir(parents=True)
    old_amrfinder = bin_dir / "amrfinder"
    old_amrfinder_update = bin_dir / "amrfinder_update"
    old_amrfinder_index = bin_dir / "amrfinder_index"
    for path in (old_amrfinder, old_amrfinder_update, old_amrfinder_index):
        path.write_text(f"old {path.name}\n")
        path.chmod(0o755)
    prerequisite_paths = _write_amrfinder_prerequisites(bin_dir, label="old")
    makeblastdb = prerequisite_paths["makeblastdb"]
    hmmpress = prerequisite_paths["hmmpress"]
    old_database = external_dir / "safety" / "amrfinder" / "database" / "2026-01-01.1"
    old_database.mkdir(parents=True)
    (old_database / "AMRProt.fa").write_text(">old\nMPEPTIDE\n")
    (old_database.parent / "latest").symlink_to(old_database.name)
    old_toxin_database = external_dir / "safety" / "toxins" / "reviewed_toxins.dmnd"
    old_toxin_database.parent.mkdir(parents=True)
    old_toxin_database.write_text("old toxin index\n")
    old_lookup = external_dir / "safety" / "phrogs" / "phrogs_integration_excision_v4.tsv"
    old_lookup.parent.mkdir(parents=True)
    old_lookup.write_text("old lookup\n")
    referenced_assets = {
        old_amrfinder: external_assets._sha256_file(old_amrfinder),
        old_amrfinder_update: external_assets._sha256_file(old_amrfinder_update),
        old_amrfinder_index: external_assets._sha256_file(old_amrfinder_index),
        **{path: external_assets._sha256_file(path) for path in prerequisite_paths.values()},
        old_database: external_assets._sha256_path(old_database),
        old_toxin_database: external_assets._sha256_file(old_toxin_database),
        old_lookup: external_assets._sha256_file(old_lookup),
    }
    manifest_path = external_dir / "safety" / "asset_manifest.yaml"
    old_manifest = {
        "schema_version": 2,
        "amrfinder_plus": {
            "binary_path": str(old_amrfinder.resolve()),
            "binary_sha256": referenced_assets[old_amrfinder],
            "amrfinder_index_path": str(old_amrfinder_index.resolve()),
            "amrfinder_index_sha256": referenced_assets[old_amrfinder_index],
            "amrfinder_update_path": str(old_amrfinder_update.resolve()),
            "amrfinder_update_sha256": referenced_assets[old_amrfinder_update],
            "makeblastdb_path": str(makeblastdb.resolve()),
            "makeblastdb_sha256": referenced_assets[makeblastdb],
            "hmmpress_path": str(hmmpress.resolve()),
            "hmmpress_sha256": referenced_assets[hmmpress],
            "database_path": str(old_database.resolve()),
            "database_sha256": referenced_assets[old_database],
        },
        "toxin_reference": {
            "files": {
                "diamond_database": {
                    "path": str(old_toxin_database.resolve()),
                    "sha256": referenced_assets[old_toxin_database],
                }
            }
        },
        "phrogs_v4": {
            "lookup_path": str(old_lookup.resolve()),
            "lookup_sha256": referenced_assets[old_lookup],
        },
    }
    manifest_path.write_text(yaml.safe_dump(old_manifest, sort_keys=False))
    old_manifest_text = manifest_path.read_text()
    _write_phrogs_v4_fixture(external_dir / "phrogs" / "phrog_annot_v4.tsv")
    _mock_verified_phrogs_profile_archive(monkeypatch, external_dir)
    archive_path = _write_amrfinder_tarball(tmp_path)
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    original_prepare_amrfinder_plus = external_assets.prepare_amrfinder_plus

    def prepare_local_amrfinder(*args, **kwargs):
        kwargs["amrfinder_url"] = archive_path.as_uri()
        kwargs["amrfinder_sha256"] = archive_sha256
        return original_prepare_amrfinder_plus(*args, **kwargs)

    def fake_run(cmd, check, capture_output, text):
        del check, capture_output, text
        if cmd[0].endswith("amrfinder_update"):
            database_dir = Path(cmd[2])
            version_dir = database_dir / "2026-01-26.1"
            _write_minimal_amrfinder_database_sources(version_dir)
            (version_dir / "AMRProt.fa").write_text(">new\nMPEPTIDE\n")
            (database_dir / "latest").symlink_to(version_dir.name)
            return type("Completed", (), {"stdout": ""})()
        if cmd[-1] == "--version":
            return type("Completed", (), {"stdout": "AMRFinderPlus version 4.2.7\n"})()
        return type("Completed", (), {"stdout": "2026-01-26.1\n"})()

    def fail_toxin_after_writing_an_asset(prepared_external_dir, **kwargs):
        safety_dir = Path(kwargs.get("safety_dir", Path(prepared_external_dir) / "safety"))
        partial_database = safety_dir / "toxins" / "reviewed_toxins.dmnd"
        partial_database.parent.mkdir(parents=True, exist_ok=True)
        partial_database.write_text("new partial toxin index\n")
        raise RuntimeError("injected toxin failure")

    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", prepare_local_amrfinder)
    monkeypatch.setattr(external_assets, "prepare_toxin_reference", fail_toxin_after_writing_an_asset)
    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="injected toxin failure"):
        prepare_external_assets(
            external_dir,
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=False,
            download_arc_evo2=False,
            download_large_databases=False,
            configure_lovis4u=False,
            with_safety=True,
            safety_manifest=manifest_path,
        )

    assert manifest_path.read_text() == old_manifest_text
    for path, expected_sha256 in referenced_assets.items():
        assert path.exists()
        assert external_assets._sha256_path(path) == expected_sha256


def test_safety_overwrite_failure_preserves_previous_phrogs_source_and_database(tmp_path, monkeypatch):
    """A shared-PHROGs overwrite cannot invalidate a prior manifest's source, raw, or profile DB."""
    external_dir = tmp_path / "external"
    mmseqs = external_dir / "bin" / "mmseqs"
    mmseqs.parent.mkdir(parents=True)
    mmseqs.write_text("#!/usr/bin/env bash\n")
    mmseqs.chmod(0o755)
    source_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("old PHROGs annotation\n")
    sequence_database = _write_mmseqs_padded_database(external_dir / "phrogs" / "phrogs_gpu_seq_db_pad")
    sequence_database.write_bytes(b"old PHROGs sequence database\n")
    sequence_database_sha256, sequence_database_files = external_assets._complete_phrogs_sequence_database(
        sequence_database
    )
    profile_database = _write_phrogs_profile_database(
        external_dir / "phrogs" / "phrogs_mmseqs_db" / "phrogs_profile_db"
    )
    profile_database_files = sorted(profile_database.parent.glob(f"{profile_database.name}*"))
    profile_tree_files = external_assets._phrogs_profile_tree_files(profile_database)
    referenced_assets = {source_path: external_assets._sha256_file(source_path)}
    referenced_assets.update({path: external_assets._sha256_file(path) for path in sequence_database_files})
    referenced_assets.update({path: external_assets._sha256_file(path) for path in profile_tree_files})
    manifest_path = external_dir / "safety" / "asset_manifest.yaml"
    manifest_path.parent.mkdir()
    old_manifest = {
        "schema_version": 2,
        "phrogs_v4": {
            "source_path": str(source_path.resolve()),
            "source_sha256": referenced_assets[source_path],
            "sequence_database": {
                "path": str(sequence_database.resolve()),
                "sha256": sequence_database_sha256,
                "files": [str(path.resolve()) for path in sequence_database_files],
            },
            "profile_database": {
                "path": str(profile_database.resolve()),
                "sha256": external_assets._complete_phrogs_profile_database(profile_database)[0],
                "files": [str(path.resolve()) for path in profile_database_files],
            },
        },
    }
    manifest_path.write_text(yaml.safe_dump(old_manifest, sort_keys=False))
    old_manifest_text = manifest_path.read_text()
    calls = []

    def rebuild_annotation(prepared_external_dir, **kwargs):
        assert kwargs["overwrite"] is True
        calls.append("annotation")
        rebuilt_source = Path(prepared_external_dir) / "phrogs" / "phrog_annot_v4.tsv"
        rebuilt_source.parent.mkdir(parents=True, exist_ok=True)
        rebuilt_source.write_text("new PHROGs annotation\n")
        return PreparedAsset("phrogs_annotation", rebuilt_source, "rebuilt")

    def rebuild_profile_database(prepared_external_dir, **kwargs):
        assert kwargs["overwrite"] is True
        calls.append("profile")
        profile_root = Path(prepared_external_dir) / "phrogs" / "phrogs_mmseqs_db"
        rebuilt_profile = _write_phrogs_profile_database(profile_root / "phrogs_profile_db")
        rebuilt_profile.write_bytes(b"new PHROGs profile database\n\0")
        archive_path = (
            Path(prepared_external_dir) / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"new PHROGs profile archive\n")
        return PreparedAsset("phrogs_safety_profile_db", profile_root, "rebuilt")

    def rebuild_sequence_database(prepared_external_dir, **kwargs):
        assert kwargs["overwrite"] is True
        calls.append("sequence")
        rebuilt_database = _write_mmseqs_padded_database(
            Path(prepared_external_dir) / "phrogs" / "phrogs_gpu_seq_db_pad"
        )
        return PreparedAsset("phrogs_gpu_seq_db", rebuilt_database, "rebuilt")

    def fail_amrfinder(*_args, **_kwargs):
        calls.append("amrfinder")
        raise RuntimeError("injected AMRFinder failure")

    monkeypatch.setattr(external_assets, "prepare_phrogs_annotation", rebuild_annotation)
    monkeypatch.setattr(external_assets, "prepare_phrogs_safety_profile_db", rebuild_profile_database)
    monkeypatch.setattr(external_assets, "prepare_phrogs_gpu_sequence_db", rebuild_sequence_database)
    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", fail_amrfinder)
    monkeypatch.setattr(
        external_assets,
        "_verify_file_size",
        lambda _path, expected_size: expected_size,
    )
    monkeypatch.setattr(
        external_assets,
        "_verify_md5",
        lambda _path, expected_md5: expected_md5,
    )

    def extract_verified_profile(_archive_path, output_dir, *, overwrite):
        assert overwrite is True
        _write_phrogs_profile_database(Path(output_dir) / "phrogs_profile_db")
        return output_dir

    monkeypatch.setattr(external_assets, "_extract_tar", extract_verified_profile)
    _mock_reviewed_phrogs_archive_sha256(monkeypatch)

    with pytest.raises(RuntimeError, match="injected AMRFinder failure"):
        prepare_external_assets(
            external_dir,
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=True,
            download_arc_evo2=False,
            download_large_databases=True,
            download_checkv=False,
            configure_lovis4u=False,
            with_safety=True,
            safety_manifest=manifest_path,
            overwrite=True,
        )

    assert calls == ["annotation", "profile", "amrfinder"]
    assert manifest_path.read_text() == old_manifest_text
    for path, expected_sha256 in referenced_assets.items():
        assert path.exists()
        assert external_assets._sha256_file(path) == expected_sha256
    observed_database_sha256, observed_database_files = external_assets._complete_phrogs_sequence_database(
        sequence_database
    )
    assert observed_database_sha256 == sequence_database_sha256
    assert [str(path.resolve()) for path in observed_database_files] == old_manifest["phrogs_v4"]["sequence_database"][
        "files"
    ]


def test_safety_manifest_records_a_generation_owned_phrogs_snapshot(tmp_path, monkeypatch):
    """A published safety manifest points at a validated PHROGs snapshot, never the mutable shared cache."""
    external_dir = tmp_path / "external"
    source_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    _write_phrogs_v4_fixture(source_path)
    _mock_verified_phrogs_profile_archive(monkeypatch, external_dir)
    original_prepare_metadata = external_assets._prepare_verified_phrogs_safety_metadata

    def staged_asset(name):
        def prepare(*_args, **kwargs):
            section, record = _materialize_mock_safety_manifest_section(name, Path(kwargs["safety_dir"]))
            kwargs["manifest"][section] = record
            return PreparedAsset(name, Path(kwargs["safety_dir"]) / name, name)

        return prepare

    def prepare_snapshot_metadata(*args, **kwargs):
        selected_annotation_path = Path(kwargs.get("annotation_path", source_path))
        return original_prepare_metadata(
            *args,
            annotation_sha256=external_assets._sha256_file(selected_annotation_path),
            **kwargs,
        )

    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", staged_asset("amrfinder"))
    monkeypatch.setattr(external_assets, "prepare_toxin_reference", staged_asset("toxins"))
    monkeypatch.setattr(external_assets, "_prepare_verified_phrogs_safety_metadata", prepare_snapshot_metadata)

    prepare_external_assets(
        external_dir,
        download_mmseqs=False,
        download_dustmasker=False,
        download_diamond=False,
        download_hmmer=False,
        download_phrogs_annotation=False,
        download_arc_evo2=False,
        download_large_databases=False,
        configure_lovis4u=False,
        with_safety=True,
    )

    manifest_path = external_dir / "safety" / "asset_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    phrogs_record = manifest["phrogs_v4"]
    generation_root = (external_dir / "safety" / "generations").resolve()
    shared_phrogs_dir = (external_dir / "phrogs").resolve()
    snapshot_source_path = Path(phrogs_record["source_path"])
    snapshot_profile_path = Path(phrogs_record["profile_database"]["path"])

    assert snapshot_source_path.is_relative_to(generation_root)
    assert snapshot_profile_path.is_relative_to(generation_root)
    assert not snapshot_source_path.is_relative_to(shared_phrogs_dir)
    assert not snapshot_profile_path.is_relative_to(shared_phrogs_dir)
    assert "sequence_database" not in phrogs_record
    assert snapshot_source_path.read_bytes() == source_path.read_bytes()
    snapshot_profile_sha256, snapshot_profile_files = external_assets._complete_phrogs_profile_database(
        snapshot_profile_path
    )
    assert snapshot_profile_sha256 == phrogs_record["profile_database"]["sha256"]
    assert [str(path.resolve()) for path in snapshot_profile_files] == phrogs_record["profile_database"]["files"]
    assert all(Path(path).is_relative_to(generation_root) for path in phrogs_record["profile_database"]["files"])
    external_assets._validate_staged_safety_manifest(manifest, verify_asset_paths=True)


def test_safety_large_database_preparation_keeps_unpinned_sequence_database_outside_manifest(tmp_path, monkeypatch):
    """Explicit raw FAA preparation stays external while safety retains only pinned profile lineage."""
    external_dir = tmp_path / "external"
    _mock_phrogs_search_database_builder(monkeypatch, external_dir)
    original_prepare_metadata = external_assets._prepare_verified_phrogs_safety_metadata
    verification_calls = []

    def staged_asset(name):
        def prepare(*_args, **kwargs):
            section, record = _materialize_mock_safety_manifest_section(name, Path(kwargs["safety_dir"]))
            kwargs["manifest"][section] = record
            return PreparedAsset(name, Path(kwargs["safety_dir"]) / name, name)

        return prepare

    def prepare_annotation(prepared_external_dir, **_kwargs):
        annotation_path = Path(prepared_external_dir) / "phrogs" / "phrog_annot_v4.tsv"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            "phrog\tcolor\tannot\tcategory\nphrog_1\t#000000\tIntegrase\tintegration and excision\n"
        )
        return PreparedAsset("phrogs_annotation", annotation_path, "prepared")

    def prepare_profile_database(prepared_external_dir, **_kwargs):
        profile_root = Path(prepared_external_dir) / "phrogs" / "phrogs_mmseqs_db"
        _write_phrogs_profile_database(profile_root / "phrogs_profile_db")
        (profile_root / "CARD").write_bytes(b"unrelated Pharokka CARD database")
        archive_path = (
            Path(prepared_external_dir) / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"prepared Pharokka profile archive\n")
        return PreparedAsset("phrogs_safety_profile_db", profile_root, "prepared")

    def prepare_sequence_database(prepared_external_dir, **_kwargs):
        sequence_database = _write_mmseqs_padded_database(
            Path(prepared_external_dir) / "phrogs" / "phrogs_gpu_seq_db_pad"
        )
        return PreparedAsset("phrogs_gpu_seq_db", sequence_database, "prepared")

    def prepare_snapshot_metadata(*args, **kwargs):
        selected_annotation_path = Path(kwargs["annotation_path"])
        return original_prepare_metadata(
            *args,
            annotation_sha256=external_assets._sha256_file(selected_annotation_path),
            **kwargs,
        )

    def verify_size(path, expected_size):
        verification_calls.append(("size", Path(path), expected_size))
        assert (
            Path(path).name == Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
            or Path(path).parent.name == "phrogs_safety_profile_archives"
        )
        assert expected_size == external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SIZE
        return expected_size

    def verify_md5(path, expected_md5):
        verification_calls.append(("md5", Path(path), expected_md5))
        assert (
            Path(path).name == Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
            or Path(path).parent.name == "phrogs_safety_profile_archives"
        )
        assert expected_md5 == external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_MD5
        return expected_md5

    def extract_cached_verified_archive(_path, output_dir, *, overwrite):
        assert overwrite is True
        _write_phrogs_profile_database(Path(output_dir) / "phrogs_profile_db")
        return output_dir

    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", staged_asset("amrfinder"))
    monkeypatch.setattr(external_assets, "prepare_toxin_reference", staged_asset("toxins"))
    monkeypatch.setattr(external_assets, "prepare_phrogs_annotation", prepare_annotation)
    monkeypatch.setattr(external_assets, "prepare_phrogs_safety_profile_db", prepare_profile_database)
    monkeypatch.setattr(external_assets, "prepare_phrogs_gpu_sequence_db", prepare_sequence_database)
    monkeypatch.setattr(external_assets, "_prepare_verified_phrogs_safety_metadata", prepare_snapshot_metadata)
    monkeypatch.setattr(external_assets, "_verify_file_size", verify_size)
    monkeypatch.setattr(external_assets, "_verify_md5", verify_md5)
    monkeypatch.setattr(external_assets, "_extract_tar", extract_cached_verified_archive)
    _mock_reviewed_phrogs_archive_sha256(monkeypatch)

    prepare_external_assets(
        external_dir,
        download_mmseqs=False,
        download_dustmasker=False,
        download_diamond=False,
        download_hmmer=False,
        download_phrogs_annotation=True,
        download_arc_evo2=False,
        download_large_databases=True,
        download_phrogs_sequence_database=True,
        download_checkv=False,
        configure_lovis4u=False,
        with_safety=True,
    )

    manifest = yaml.safe_load((external_dir / "safety" / "asset_manifest.yaml").read_text())
    phrogs_record = manifest["phrogs_v4"]
    snapshot_source_path = Path(phrogs_record["source_path"])
    snapshot_profile_path = Path(phrogs_record["profile_database"]["path"])
    generation_root = (external_dir / "safety" / "generations").resolve()
    shared_phrogs_dir = (external_dir / "phrogs").resolve()

    assert snapshot_source_path.is_relative_to(generation_root)
    assert snapshot_profile_path.is_relative_to(generation_root)
    assert "sequence_database" not in phrogs_record
    external_assets._validate_staged_safety_manifest(manifest, verify_asset_paths=True)

    shared_source_path = shared_phrogs_dir / "phrog_annot_v4.tsv"
    shared_database_path = shared_phrogs_dir / "phrogs_gpu_seq_db_pad"
    assert shared_source_path.read_bytes() == snapshot_source_path.read_bytes()
    _shared_database_sha256, shared_database_files = external_assets._complete_phrogs_sequence_database(
        shared_database_path
    )
    assert shared_database_files
    shared_profile_path = shared_phrogs_dir / "phrogs_mmseqs_db" / "phrogs_profile_db"
    shared_profile_sha256, shared_profile_files = external_assets._complete_phrogs_profile_database(
        shared_profile_path
    )
    assert shared_profile_sha256 == phrogs_record["profile_database"]["sha256"]
    assert [path.name for path in shared_profile_files] == [
        Path(path).name for path in phrogs_record["profile_database"]["files"]
    ]
    generation_profile_root = snapshot_profile_path.parent.parent / "phrogs_mmseqs_db"
    assert not generation_profile_root.exists()
    assert not (
        snapshot_profile_path.parents[2] / "downloads" / Path(external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_URL).name
    ).exists()
    cached_archive_path = Path(phrogs_record["profile_database"]["provenance"]["verified_archive"]["path"])
    assert cached_archive_path.is_file()
    assert cached_archive_path.parent.name == "phrogs_safety_profile_archives"

    prepare_external_assets(
        external_dir,
        download_mmseqs=False,
        download_dustmasker=False,
        download_diamond=False,
        download_hmmer=False,
        download_phrogs_annotation=False,
        download_arc_evo2=False,
        download_large_databases=False,
        configure_lovis4u=False,
        with_safety=True,
    )

    repeat_manifest = yaml.safe_load((external_dir / "safety" / "asset_manifest.yaml").read_text())
    repeat_profile_path = Path(repeat_manifest["phrogs_v4"]["profile_database"]["path"])
    assert repeat_profile_path.exists()
    assert repeat_profile_path != snapshot_profile_path
    assert cached_archive_path.is_file()
    external_assets._validate_staged_safety_manifest(repeat_manifest, verify_asset_paths=True)
    assert [call[0] for call in verification_calls] == ["size", "md5"] * 11


def test_safety_profile_preparation_can_exclude_unpinned_arc_sequence_database(tmp_path, monkeypatch):
    """Safety preparation must not require the unrelated, unpinned legacy PHROGs FAA archive."""
    external_dir = tmp_path / "external"
    _mock_phrogs_search_database_builder(monkeypatch, external_dir)
    original_prepare_metadata = external_assets._prepare_verified_phrogs_safety_metadata
    cached_archive_payload = b"previously authenticated Pharokka profile archive\n"
    cached_archive_sha256 = external_assets.DEFAULT_PHROGS_SAFETY_PROFILE_SHA256
    cached_archive_path = (
        external_dir / "downloads" / "phrogs_safety_profile_archives" / f"{cached_archive_sha256}.tar.gz"
    )
    cached_archive_path.parent.mkdir(parents=True)
    cached_archive_path.write_bytes(cached_archive_payload)

    def staged_asset(name):
        def prepare(*_args, **kwargs):
            section, record = _materialize_mock_safety_manifest_section(name, Path(kwargs["safety_dir"]))
            kwargs["manifest"][section] = record
            return PreparedAsset(name, Path(kwargs["safety_dir"]) / name, name)

        return prepare

    def prepare_annotation(prepared_external_dir, **_kwargs):
        annotation_path = Path(prepared_external_dir) / "phrogs" / "phrog_annot_v4.tsv"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            "phrog\tcolor\tannot\tcategory\nphrog_1\t#000000\tIntegrase\tintegration and excision\n"
        )
        return PreparedAsset("phrogs_annotation", annotation_path, "prepared")

    def prepare_profile_database(prepared_external_dir, **kwargs):
        assert Path(kwargs["verified_archive_path"]) == cached_archive_path
        profile_root = Path(prepared_external_dir) / "phrogs" / "phrogs_mmseqs_db"
        _write_phrogs_profile_database(profile_root / "phrogs_profile_db")
        return PreparedAsset("phrogs_safety_profile_db", profile_root, "prepared")

    def reject_unpinned_sequence_database(*_args, **_kwargs):
        raise AssertionError("the unpinned legacy PHROGs FAA must not be downloaded for safety")

    def prepare_snapshot_metadata(*args, **kwargs):
        selected_annotation_path = Path(kwargs["annotation_path"])
        return original_prepare_metadata(
            *args,
            annotation_sha256=external_assets._sha256_file(selected_annotation_path),
            **kwargs,
        )

    def extract_cached_verified_archive(_path, output_dir, *, overwrite):
        assert overwrite is True
        _write_phrogs_profile_database(Path(output_dir) / "phrogs_profile_db")
        return output_dir

    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", staged_asset("amrfinder"))
    monkeypatch.setattr(external_assets, "prepare_toxin_reference", staged_asset("toxins"))
    monkeypatch.setattr(external_assets, "prepare_phrogs_annotation", prepare_annotation)
    monkeypatch.setattr(external_assets, "prepare_phrogs_safety_profile_db", prepare_profile_database)
    monkeypatch.setattr(external_assets, "prepare_phrogs_gpu_sequence_db", reject_unpinned_sequence_database)
    monkeypatch.setattr(external_assets, "_prepare_verified_phrogs_safety_metadata", prepare_snapshot_metadata)
    monkeypatch.setattr(external_assets, "_verify_file_size", lambda _path, expected: expected)
    monkeypatch.setattr(external_assets, "_verify_md5", lambda _path, expected: expected)
    monkeypatch.setattr(external_assets, "_extract_tar", extract_cached_verified_archive)
    _mock_reviewed_phrogs_archive_sha256(monkeypatch)

    prepare_external_assets(
        external_dir,
        download_mmseqs=False,
        download_dustmasker=False,
        download_diamond=False,
        download_hmmer=False,
        download_phrogs_annotation=True,
        download_arc_evo2=False,
        download_large_databases=True,
        download_checkv=False,
        configure_lovis4u=False,
        with_safety=True,
    )

    manifest = yaml.safe_load((external_dir / "safety" / "asset_manifest.yaml").read_text())
    assert "sequence_database" not in manifest["phrogs_v4"]
    assert Path(manifest["phrogs_v4"]["profile_database"]["path"]).is_file()
    external_assets._validate_staged_safety_manifest(manifest, verify_asset_paths=True)


def test_safety_postpublication_legacy_profile_failure_retains_published_generation(tmp_path, monkeypatch):
    """A legacy profile-copy error after manifest publication cannot delete the trusted generation."""
    external_dir = tmp_path / "external"
    source_path = external_dir / "phrogs" / "phrog_annot_v4.tsv"
    _write_phrogs_v4_fixture(source_path)
    _mock_verified_phrogs_profile_archive(monkeypatch, external_dir)
    original_prepare_metadata = external_assets._prepare_verified_phrogs_safety_metadata

    def staged_asset(name):
        def prepare(*_args, **kwargs):
            section, record = _materialize_mock_safety_manifest_section(name, Path(kwargs["safety_dir"]))
            kwargs["manifest"][section] = record
            return PreparedAsset(name, Path(kwargs["safety_dir"]) / name, name)

        return prepare

    def prepare_annotation(prepared_external_dir, **_kwargs):
        annotation_path = Path(prepared_external_dir) / "phrogs" / "phrog_annot_v4.tsv"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            "phrog\tcolor\tannot\tcategory\nphrog_1\t#000000\tIntegrase\tintegration and excision\n"
        )
        return PreparedAsset("phrogs_annotation", annotation_path, "prepared")

    def prepare_snapshot_metadata(*args, **kwargs):
        selected_annotation_path = Path(kwargs["annotation_path"])
        return original_prepare_metadata(
            *args,
            annotation_sha256=external_assets._sha256_file(selected_annotation_path),
            **kwargs,
        )

    def fail_legacy_publication(*_args, **_kwargs):
        raise RuntimeError("injected legacy PHROGs profile publication failure")

    monkeypatch.setattr(external_assets, "prepare_amrfinder_plus", staged_asset("amrfinder"))
    monkeypatch.setattr(external_assets, "prepare_toxin_reference", staged_asset("toxins"))
    monkeypatch.setattr(external_assets, "prepare_phrogs_annotation", prepare_annotation)
    monkeypatch.setattr(external_assets, "_prepare_verified_phrogs_safety_metadata", prepare_snapshot_metadata)
    monkeypatch.setattr(external_assets, "_publish_phrogs_legacy_profile_database", fail_legacy_publication)

    with pytest.raises(RuntimeError, match="injected legacy PHROGs profile publication failure"):
        prepare_external_assets(
            external_dir,
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=True,
            download_arc_evo2=False,
            download_large_databases=False,
            configure_lovis4u=False,
            with_safety=True,
        )

    manifest = yaml.safe_load((external_dir / "safety" / "asset_manifest.yaml").read_text())
    generation_root = (external_dir / "safety" / "generations").resolve()
    assert Path(manifest["phrogs_v4"]["source_path"]).is_relative_to(generation_root)
    assert "sequence_database" not in manifest["phrogs_v4"]
    assert Path(manifest["phrogs_v4"]["profile_database"]["path"]).is_relative_to(generation_root)
    external_assets._validate_staged_safety_manifest(manifest, verify_asset_paths=True)


def test_update_safety_manifest_publishes_with_replace_and_fsync(tmp_path, monkeypatch):
    """Single-section publication is crash-safe: write/fsync a sibling temp file, then replace."""
    manifest_path = tmp_path / "safety" / "asset_manifest.yaml"
    replace_calls = []
    fsync_calls = []
    original_replace = external_assets.os.replace

    def recording_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(external_assets.os, "replace", recording_replace)
    monkeypatch.setattr(external_assets.os, "fsync", lambda descriptor: fsync_calls.append(descriptor))

    external_assets._update_safety_manifest(manifest_path, "test_asset", {"value": "new"})

    assert yaml.safe_load(manifest_path.read_text())["test_asset"] == {"value": "new"}
    assert replace_calls and replace_calls[0][1] == manifest_path
    assert len(fsync_calls) >= 2


def test_write_safety_manifest_commits_when_directory_fsync_fails_after_replace(tmp_path, monkeypatch):
    """A post-replace directory fsync error is reported as a committed manifest, not a false rollback."""
    manifest_path = tmp_path / "safety" / "asset_manifest.yaml"
    manifest_path.parent.mkdir()
    manifest_path.write_text("schema_version: 1\ngeneration: old\n")
    fsync_calls = []
    original_fsync = external_assets.os.fsync

    def fail_after_replace(descriptor):
        fsync_calls.append(descriptor)
        if len(fsync_calls) == 2:
            raise OSError("injected directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(external_assets.os, "fsync", fail_after_replace)

    external_assets._write_safety_manifest_atomic(manifest_path, {"schema_version": 1, "generation": "new"})

    assert yaml.safe_load(manifest_path.read_text())["generation"] == "new"
    assert len(fsync_calls) == 2


def test_write_safety_manifest_is_readable_outside_the_preparation_container(tmp_path):
    """Published provenance is non-secret metadata and must survive a container UID mismatch."""
    manifest_path = tmp_path / "safety" / "asset_manifest.yaml"

    external_assets._write_safety_manifest_atomic(manifest_path, {"schema_version": 1})

    assert manifest_path.stat().st_mode & 0o777 == 0o644


def test_main_leaves_safety_manifest_unset_for_custom_external_directory(tmp_path, monkeypatch):
    """CLI parsing lets preparation resolve the default manifest under the selected external root."""
    calls = {}

    def fake_prepare(external_dir, **kwargs):
        calls["external_dir"] = external_dir
        calls.update(kwargs)
        return []

    monkeypatch.setattr(external_assets, "prepare_external_assets", fake_prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_external_assets",
            "--external-dir",
            str(tmp_path / "custom-external"),
            "--skip-mmseqs",
            "--skip-dustmasker",
            "--skip-diamond",
            "--skip-hmmer",
            "--skip-phrogs-annotation",
            "--skip-arc-evo2",
            "--skip-lovis4u-config",
            "--skip-checkv",
        ],
    )

    external_assets.main()

    assert calls["external_dir"] == tmp_path / "custom-external"
    assert calls["safety_manifest"] is None


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
    hmmer_archive = _write_multi_executable_tarball(
        tmp_path / "hmmer",
        ("hmmsearch", "hmmpress"),
        subdir="bin",
        archive_name="hmmer.tar.gz",
    )
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


def test_verified_phrogs_profile_rejects_wrong_expected_sha256_before_extraction(tmp_path, monkeypatch):
    """Published MD5 and size are insufficient when the reviewed SHA-256 differs."""
    archive = tmp_path / "pharokka.tar.gz"
    archive.write_bytes(b"wrong archive bytes")
    monkeypatch.setattr(external_assets, "_verify_file_size", lambda *_args: len(archive.read_bytes()))
    monkeypatch.setattr(external_assets, "_verify_md5", lambda *_args: "published-md5")
    monkeypatch.setattr(
        external_assets,
        "_extract_tar",
        lambda *_args, **_kwargs: pytest.fail("archive extraction ran before SHA-256 verification"),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        external_assets._extract_verified_phrogs_safety_profile_archive(archive, tmp_path / "extracted")


def test_download_with_headers_resumes_and_authenticates_a_partial(tmp_path, monkeypatch):
    """An interrupted large transfer resumes only from a valid Content-Range boundary."""
    payload = b"reviewed large archive payload"
    output = tmp_path / "archive.tar.gz"
    partial = tmp_path / "archive.tar.gz.tmp"
    partial.write_bytes(payload[:9])
    requests = []

    class Response(io.BytesIO):
        status = 206

        def __init__(self):
            super().__init__(payload[9:])
            self.headers = {"Content-Range": f"bytes 9-{len(payload) - 1}/{len(payload)}"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def urlopen(request, **_kwargs):
        requests.append(request)
        assert request.get_header("Range") == "bytes=9-"
        return Response()

    monkeypatch.setattr(external_assets.urllib.request, "urlopen", urlopen)

    downloaded, headers = external_assets._download_with_headers(
        "https://mirror.example/archive.tar.gz",
        output,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        expected_size=len(payload),
        resume=True,
    )

    assert downloaded.read_bytes() == payload
    assert headers["content-range"].startswith("bytes 9-")
    assert not partial.exists()
    assert len(requests) == 1


def test_with_safety_refuses_unsupported_arm_toolchain_before_mutation(tmp_path, monkeypatch):
    """ARM BLAST support must not masquerade as complete ARM safety support."""
    monkeypatch.setattr(external_assets.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        external_assets,
        "prepare_pyrodigal_wrapper",
        lambda *_args, **_kwargs: pytest.fail("asset preparation started before architecture validation"),
    )

    with pytest.raises(RuntimeError, match="full sequence-safety toolchain"):
        prepare_external_assets(
            tmp_path / "external",
            download_mmseqs=False,
            download_dustmasker=False,
            download_diamond=False,
            download_hmmer=False,
            download_phrogs_annotation=False,
            download_arc_evo2=False,
            download_large_databases=False,
            download_checkv=False,
            configure_lovis4u=False,
            with_safety=True,
        )


def test_optional_phrogs_sequence_database_defaults_off():
    """The unpinned raw FAA compatibility database must require an explicit opt-in."""
    import inspect

    parameter = inspect.signature(prepare_external_assets).parameters["download_phrogs_sequence_database"]

    assert parameter.default is False
