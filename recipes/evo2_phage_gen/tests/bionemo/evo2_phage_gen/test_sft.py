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

"""Tests for ``bionemo.evo2_phage_gen.sft``."""

from __future__ import annotations

import hashlib
import json
import stat
import tomllib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pytest
import yaml

from bionemo.evo2_phage_gen import host_evidence as host_evidence_module
from bionemo.evo2_phage_gen import sft
from bionemo.evo2_phage_gen.design_scope import HostDomain
from bionemo.evo2_phage_gen.host_evidence import (
    HostEvidenceTable,
    HostEvidenceTableRow,
    load_host_evidence_table,
    resolve_ncbi_host_evidence,
    write_host_evidence_table,
)
from bionemo.evo2_phage_gen.sequence_safety_cli import parse_fasta_records


RECIPE_ROOT = Path(__file__).parents[3]
PREPROCESS_CONFIG = RECIPE_ROOT / "configs" / "sft_microviridae_preprocess.yaml"
DATASET_CONFIG = RECIPE_ROOT / "configs" / "sft_microviridae_dataset.yaml"
SYNTHETIC_SOURCE_BYTES = (
    b">b_pass\n+!ACGT\n>a_pass\n+#TGCA\n>ba_fail\n+$CCCC\n>euk\n+^GGGG\n>unknown\n+~NNNN\n>duplicate\n+!ACGT\n"
)
SYNTHETIC_PHIX_BYTES = b">NC_001422.1 phiX174\nACGTACGT\n"


def test_preprocess_config_points_only_at_safety_pass_fasta_and_manifest() -> None:
    """Restoring the historical source path would bypass the SFT safety gate."""
    config = yaml.safe_load(PREPROCESS_CONFIG.read_text())

    assert config[0]["datapaths"] == ["data/curated/microviridae_sft_training_data_safety_pass.fna"]
    assert config[0]["safety_manifest"] == "data/curated/SAFETY_MANIFEST.yaml"
    assert config[0]["output_prefix"] == "microviridae_sft_safety_pass"
    assert config[0]["hf_tokenizer_model_path"] == "tokenizers/nucleotide_fast_tokenizer_512"
    assert config[0]["force_uppercase"] is False


def test_dataset_config_uses_expected_preprocessed_prefixes() -> None:
    """Dataset prefixes should match preprocess_evo2's tokenizer-suffixed output names."""
    config = yaml.safe_load(DATASET_CONFIG.read_text())

    assert [entry["dataset_split"] for entry in config] == ["train", "validation", "test"]
    assert config[0]["dataset_prefix"].endswith("microviridae_sft_safety_pass_nucleotide_fast_tokenizer_512_train")
    assert config[1]["dataset_prefix"].endswith("microviridae_sft_safety_pass_nucleotide_fast_tokenizer_512_val")
    assert config[2]["dataset_prefix"].endswith("microviridae_sft_safety_pass_nucleotide_fast_tokenizer_512_test")


def test_prepare_sft_data_downloads_processed_file_by_default(monkeypatch) -> None:
    """The SFT helper should download only the processed training FASTA unless raw is requested."""
    calls = []

    def fake_download(url: str, output_path: Path, *, overwrite: bool = False) -> Path:
        calls.append((url, output_path, overwrite))
        return output_path

    monkeypatch.setattr(sft, "_download", fake_download)

    paths = sft.prepare_sft_data()

    assert paths == [sft.DEFAULT_SFT_PROCESSED]
    assert calls == [(sft.DEFAULT_SFT_PROCESSED_URL, sft.DEFAULT_SFT_PROCESSED, False)]


def test_prepare_sft_data_can_include_raw_fasta(monkeypatch) -> None:
    """The raw FASTA is optional because training uses the processed soft-prompt file."""
    calls = []

    def fake_download(url: str, output_path: Path, *, overwrite: bool = False) -> Path:
        calls.append((url, output_path, overwrite))
        return output_path

    monkeypatch.setattr(sft, "_download", fake_download)

    paths = sft.prepare_sft_data(include_raw=True)

    assert paths == [sft.DEFAULT_SFT_PROCESSED, sft.DEFAULT_SFT_RAW]
    assert calls == [
        (sft.DEFAULT_SFT_PROCESSED_URL, sft.DEFAULT_SFT_PROCESSED, False),
        (sft.DEFAULT_SFT_RAW_URL, sft.DEFAULT_SFT_RAW, False),
    ]


def test_zenodo_source_is_immutable_even_when_legacy_overwrite_is_requested(tmp_path: Path, monkeypatch) -> None:
    """A legacy overwrite flag must not mutate an already downloaded source corpus."""
    source = tmp_path / "microviridae.fna"
    original = b">one\n+!ACGT\n"
    source.write_bytes(original)
    monkeypatch.setattr(sft, "DEFAULT_SFT_PROCESSED", source)
    monkeypatch.setattr(sft, "DEFAULT_SFT_RAW", tmp_path / "raw.fna")

    with pytest.raises(sft.SFTSafetyError, match="immutable"):
        sft.prepare_sft_data(overwrite=True)

    assert source.read_bytes() == original


def test_default_processed_download_reuse_requires_the_published_source_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An arbitrary existing file at the production destination must not be accepted as the Zenodo corpus."""
    processed = tmp_path / sft.PRODUCTION_SFT_SOURCE_IDENTITY.file_name
    processed.write_bytes(b">synthetic\n+!ACGT\n")
    monkeypatch.setattr(sft, "DEFAULT_SFT_PROCESSED", processed)

    with pytest.raises(sft.SFTSafetyError, match="published source identity"):
        sft.prepare_sft_data()

    generic = tmp_path / "injected-example.fna"
    generic.write_bytes(b">injected\n+!TGCA\n")
    assert sft._download("https://example.invalid/injected.fna", generic) == generic


def test_zenodo_download_uses_exclusive_staging_without_following_predictable_temp_symlinks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A pre-planted historical `.tmp` symlink must not clobber its target or become the source artifact."""
    output = tmp_path / "microviridae.fna"
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"preserve-me")
    predictable_temp = output.with_suffix(output.suffix + ".tmp")
    predictable_temp.symlink_to(victim)
    staging_paths: list[Path] = []

    def fake_urlretrieve(_url: str, filename: str | Path):
        staging = Path(filename)
        staging_paths.append(staging)
        staging.write_bytes(b">one\n+!ACGT\n")
        return str(staging), None

    monkeypatch.setattr(sft.urllib.request, "urlretrieve", fake_urlretrieve)

    assert sft._download("https://example.invalid/source.fna", output) == output
    assert victim.read_bytes() == b"preserve-me"
    assert output.read_bytes() == b">one\n+!ACGT\n"
    assert not output.is_symlink()
    assert predictable_temp.is_symlink()
    assert len(staging_paths) == 1
    assert staging_paths[0] != predictable_temp
    assert staging_paths[0].name.startswith(f".{output.name}.download.")
    assert not staging_paths[0].exists()


def test_failed_download_publication_removes_only_its_linked_artifact_and_all_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A directory-fsync failure after hard-linking must leave no artifact that a retry could trust."""
    output = tmp_path / "microviridae.fna"
    download_calls: list[Path] = []

    def fake_urlretrieve(_url: str, filename: str | Path):
        staging = Path(filename)
        download_calls.append(staging)
        staging.write_bytes(b">one\n+!ACGT\n")
        return str(staging), None

    real_fsync = sft.os.fsync
    fail_directory_fsync = True

    def injected_fsync(descriptor: int) -> None:
        if fail_directory_fsync and stat.S_ISDIR(sft.os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(sft.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(sft.os, "fsync", injected_fsync)

    with pytest.raises(OSError, match="injected directory fsync failure"):
        sft._download("https://example.invalid/source.fna", output)
    assert not output.exists()
    assert list(tmp_path.glob(".microviridae.fna.download.*")) == []

    fail_directory_fsync = False
    assert sft._download("https://example.invalid/source.fna", output) == output
    assert output.read_bytes() == b">one\n+!ACGT\n"
    assert len(download_calls) == 2


def test_failed_download_cleanup_stays_bound_to_replaced_destination_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A renamed destination parent must not strand either the publication link or its staging inode."""
    destination_parent = tmp_path / "download"
    destination_parent.mkdir()
    displaced_parent = tmp_path / "download-displaced"
    output = destination_parent / "microviridae.fna"
    swapped = False

    def fake_urlretrieve(_url: str, filename: str | Path):
        staging = Path(filename)
        staging.write_bytes(b">one\n+!ACGT\n")
        return str(staging), None

    real_link = sft.os.link

    def replace_parent_after_link(source, destination, *args, **kwargs):
        nonlocal swapped
        result = real_link(source, destination, *args, **kwargs)
        destination_parent.rename(displaced_parent)
        destination_parent.mkdir()
        swapped = True
        return result

    real_fsync = sft.os.fsync

    def fail_bound_directory_fsync(descriptor: int) -> None:
        if swapped and stat.S_ISDIR(sft.os.fstat(descriptor).st_mode):
            raise OSError("injected post-link directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(sft.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(sft.os, "link", replace_parent_after_link)
    monkeypatch.setattr(sft.os, "fsync", fail_bound_directory_fsync)

    with pytest.raises(OSError):
        sft._download("https://example.invalid/source.fna", output)

    assert swapped is True
    assert not (displaced_parent / output.name).exists()
    assert list(displaced_parent.glob(f".{output.name}.download.*")) == []
    assert not output.exists()


def test_download_cannot_report_success_after_destination_parent_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A publication moved out from under its returned pathname must fail and remove every bound link."""
    destination_parent = tmp_path / "download"
    destination_parent.mkdir()
    displaced_parent = tmp_path / "download-displaced"
    output = destination_parent / "microviridae.fna"
    swapped = False

    def fake_urlretrieve(_url: str, filename: str | Path):
        staging = Path(filename)
        staging.write_bytes(b">one\n+!ACGT\n")
        return str(staging), None

    real_link = sft.os.link

    def replace_parent_after_link(source, destination, *args, **kwargs):
        nonlocal swapped
        result = real_link(source, destination, *args, **kwargs)
        destination_parent.rename(displaced_parent)
        destination_parent.mkdir()
        swapped = True
        return result

    monkeypatch.setattr(sft.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(sft.os, "link", replace_parent_after_link)

    with pytest.raises(sft.SFTSafetyError, match="destination.*(parent|path|identity).*(changed|replaced)"):
        sft._download("https://example.invalid/source.fna", output)

    assert swapped is True
    assert not output.exists()
    assert not (displaced_parent / output.name).exists()
    assert list(displaced_parent.glob(f".{output.name}.download.*")) == []


def test_conditioning_parser_strips_exactly_one_allowed_two_byte_prefix_for_scanning(tmp_path: Path) -> None:
    """Task 4 must see only ACGTN while curated SFT records retain their conditioning token and bytes."""
    source = tmp_path / "source.fna"
    payload = b">one description\r\n+!acgt\r\n>two\n+#TGCA\n"
    source.write_bytes(payload)

    records = sft.parse_conditioned_fasta(source)
    scanner = tmp_path / "scanner.fna"
    sft.write_scanner_fasta(scanner, records)
    curated = sft.conditioned_records_bytes(records, {"one", "two"})

    assert [record.conditioning_prefix for record in records] == [b"+!", b"+#"]
    assert [record.biological_sequence for record in records] == ["ACGT", "TGCA"]
    assert [(record.source_start, record.source_end) for record in records] == [(0, 26), (26, len(payload))]
    assert scanner.read_bytes() == b">sft_000000\nACGT\n>sft_000001\nTGCA\n"
    assert curated == payload
    assert records[0].source_record_sha256 == sft.sha256_bytes(payload[:26])
    assert records[0].scanner_record_sha256 == sft.sha256_bytes(b">sft_000000\nACGT\n")


@pytest.mark.parametrize("sequence", [b"ACGT", b"++ACGT", b"+?ACGT", b"+!+!ACGT"])
def test_conditioning_parser_rejects_missing_unknown_or_repeated_prefixes(tmp_path: Path, sequence: bytes) -> None:
    """Ambiguous conditioning removal must fail instead of changing scanner semantics."""
    source = tmp_path / "source.fna"
    source.write_bytes(b">one\n" + sequence + b"\n")

    with pytest.raises(sft.SFTSafetyError, match="conditioning prefix|biological sequence"):
        sft.parse_conditioned_fasta(source)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (4999, "BLOCKED_BELOW_MINIMUM"),
        (5000, "MINIMUM_SIZE_MET"),
        (9999, "MINIMUM_SIZE_MET"),
        (10000, "PREFERRED_SIZE_MET"),
    ],
)
def test_corpus_adequacy_uses_exact_distinct_genome_boundaries(count: int, expected: str) -> None:
    """Off-by-one changes at 5k or 10k would either over-block or understate the preferred target."""
    decision = sft.assess_corpus_adequacy(count)
    assert decision.state.value == expected
    assert decision.distinct_genome_count == count
    assert decision.ready is (count >= 5000)


def test_below_minimum_requires_scoped_count_only_authorization() -> None:
    """An upfront completion mandate may waive only corpus count, never biological gates."""
    authorization = sft.CorpusCountAuthorization(
        kind=sft.CorpusAuthorizationKind.UPFRONT_COMPLETION_MANDATE,
        verbatim_statement="Just do whatever it takes to get the run done.",
        authorized_at=datetime(2026, 8, 8, tzinfo=timezone.utc),
        scope="corpus_count_only",
        minimum_accepted_count=None,
    )
    decision = sft.assess_corpus_adequacy(3000, authorization=authorization)

    assert decision.state is sft.CorpusAdequacy.AUTHORIZED_BELOW_MINIMUM
    assert decision.ready is True
    assert decision.authorization == authorization

    with pytest.raises(sft.SFTSafetyError, match="corpus_count_only"):
        sft.CorpusCountAuthorization(
            kind=sft.CorpusAuthorizationKind.UPFRONT_COMPLETION_MANDATE,
            verbatim_statement="Do it",
            authorized_at=datetime(2026, 8, 8, tzinfo=timezone.utc),
            scope="all_safety_gates",
            minimum_accepted_count=None,
        )


def _host_row(record_id: str, domain: HostDomain, *, confirmed: bool = True) -> HostEvidenceTableRow:
    return HostEvidenceTableRow.create(
        record_id=record_id,
        accession="NC_001422.1" if record_id == "NC_001422.1" else None,
        normalized_host_domain=domain,
        confirmed=confirmed,
        evidence_source="reviewed_multi_source_catalog",
        evidence_id=f"reviewed:{record_id}",
        evidence_version="2026-08-08",
        retrieved_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        raw_response_path=None,
        raw_response_sha256=None,
        reason_codes=("REVIEWED_REPLICATION_HOST",),
    )


def _fake_task4_runtime(*, phix_state: str = "PASS", diagnostic_label: str | None = None):
    def task4_runner(request: sft.Task4SafetyRequest) -> sft.Task4SafetyArtifacts:
        records = parse_fasta_records(request.input_fasta)
        states = {
            record.sequence_id: (
                phix_state
                if request.label == "phix_nc_001422_1"
                else "FAIL"
                if record.sequence_id == "sft_000002"
                else "PASS"
            )
            for record in records
        }
        scan_dir = request.output_root / "scan"
        filter_dir = request.output_root / "filter"
        scan_dir.mkdir(parents=True)
        filter_dir.mkdir(parents=True)
        scan_type = "sequence_safety_diagnostic" if request.label == diagnostic_label else "sequence_safety_scan"
        scan = {
            "schema_version": 1,
            "manifest_type": scan_type,
            "input_fasta": {
                "path": str(request.input_fasta.absolute()),
                "sha256": hashlib.sha256(request.input_fasta.read_bytes()).hexdigest(),
                "count": len(records),
            },
            "resolved_profile": {"host_domain": request.host_domain.value},
            "records": [
                {"record_id": record.sequence_id, "input_index": index, "state": states[record.sequence_id]}
                for index, record in enumerate(records)
            ],
            "aggregate": {
                "state": "FAIL"
                if "FAIL" in states.values()
                else "INDETERMINATE"
                if "INDETERMINATE" in states.values()
                else "PASS"
            },
        }
        scan_path = scan_dir / "manifest.json"
        scan_path.write_text(json.dumps(scan, sort_keys=True))
        filter_manifest = {
            **scan,
            "manifest_type": "sequence_safety_filter",
            "source_scan_manifest": {
                "path": str(scan_path.absolute()),
                "sha256": hashlib.sha256(scan_path.read_bytes()).hexdigest(),
            },
        }
        filter_path = filter_dir / "manifest.json"
        filter_path.write_text(json.dumps(filter_manifest, sort_keys=True))
        return sft.Task4SafetyArtifacts(scan_manifest=scan_path, filter_manifest=filter_path)

    def manifest_validator(path: str | Path, *, expected_type: str | None = None) -> Mapping[str, object]:
        payload = json.loads(Path(path).read_text())
        if expected_type is not None and payload.get("manifest_type") != expected_type:
            raise sft.SFTSafetyError(f"expected trusted {expected_type} manifest")
        return payload

    return sft.SFTAuditRuntime(
        task4_runner=task4_runner,
        task4_manifest_validator=manifest_validator,
        clock=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        expected_source_identity=sft.SFTSourceIdentity(
            provenance_kind="INJECTED_TEST_FIXTURE",
            repository="test-fixture",
            record_id="synthetic-microviridae",
            doi="test:synthetic-microviridae",
            file_name="microviridae_sft_training_data_processed.fna",
            url="test://synthetic-microviridae",
            size_bytes=len(SYNTHETIC_SOURCE_BYTES),
            md5=hashlib.md5(SYNTHETIC_SOURCE_BYTES, usedforsecurity=False).hexdigest(),
        ),
        expected_phix_identity=sft.PhiXReferenceIdentity(
            provenance_kind="INJECTED_TEST_FIXTURE",
            accession="NC_001422.1",
            source="test-fixture",
            dataset_report_url="test://NC_001422.1/report",
            fasta_url="test://NC_001422.1/fasta",
            sequence_length=8,
            sequence_sha256=hashlib.sha256(b"ACGTACGT").hexdigest(),
            ncbi_sequence_hash="TEST0000",
        ),
    )


def _audit_request(tmp_path: Path) -> tuple[sft.SFTAuditRequest, bytes]:
    source = tmp_path / "data" / "external" / "zenodo" / "microviridae_sft_training_data_processed.fna"
    source.parent.mkdir(parents=True)
    source_bytes = SYNTHETIC_SOURCE_BYTES
    source.write_bytes(source_bytes)
    rows = (
        _host_row("b_pass", HostDomain.BACTERIA),
        _host_row("a_pass", HostDomain.ARCHAEA),
        _host_row("ba_fail", HostDomain.BACTERIA_AND_ARCHAEA),
        _host_row("euk", HostDomain.EUKARYOTA),
        _host_row("unknown", HostDomain.UNKNOWN, confirmed=False),
        _host_row("duplicate", HostDomain.BACTERIA),
    )
    evidence_path = tmp_path / "evidence" / "HOST_EVIDENCE.yaml"
    write_host_evidence_table(
        evidence_path,
        HostEvidenceTable(
            table_id="microviridae-hosts-v1",
            created_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            rows=rows,
        ),
    )
    phix = tmp_path / "references" / "NC_001422.1.fna"
    phix.parent.mkdir(parents=True)
    phix.write_bytes(SYNTHETIC_PHIX_BYTES)
    phix_evidence_path = tmp_path / "evidence" / "PHIX_HOST_EVIDENCE.yaml"
    write_host_evidence_table(
        phix_evidence_path,
        HostEvidenceTable(
            table_id="phix-host-v1",
            created_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            rows=(_host_row("NC_001422.1", HostDomain.BACTERIA),),
        ),
    )
    curated = tmp_path / "data" / "curated" / "microviridae_sft_training_data_safety_pass.fna"
    safety_manifest = curated.parent / "SAFETY_MANIFEST.yaml"
    preprocess = tmp_path / "configs" / "sft_microviridae_preprocess.yaml"
    preprocess.parent.mkdir(parents=True)
    config = yaml.safe_load(PREPROCESS_CONFIG.read_text())
    config[0]["datapaths"] = [str(curated.relative_to(tmp_path))]
    config[0]["safety_manifest"] = str(safety_manifest.relative_to(tmp_path))
    preprocess.write_text(yaml.safe_dump(config, sort_keys=False))
    prerequisite_paths = []
    for name in ("policy.yaml", "assets.yaml", "diamond.json", "mmseqs.json"):
        path = tmp_path / "pins" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        prerequisite_paths.append(path)
    return (
        sft.SFTAuditRequest(
            source_fasta=source,
            host_evidence_table=evidence_path,
            phix_fasta=phix,
            phix_host_evidence_table=phix_evidence_path,
            curated_output=curated,
            safety_manifest=safety_manifest,
            audit_root=tmp_path / "audit",
            preprocess_config=preprocess,
            policy=prerequisite_paths[0],
            asset_manifest=prerequisite_paths[1],
            diamond_tool_pin=prerequisite_paths[2],
            mmseqs_tool_pin=prerequisite_paths[3],
            minimum_genomes=1,
            preferred_genomes=2,
        ),
        source_bytes,
    )


@pytest.mark.parametrize(
    "input_field",
    [
        "source_fasta",
        "host_evidence_table",
        "phix_fasta",
        "phix_host_evidence_table",
        "preprocess_config",
        "policy",
        "asset_manifest",
        "diamond_tool_pin",
        "mmseqs_tool_pin",
    ],
)
def test_audit_request_rejects_every_output_input_overlap(tmp_path: Path, input_field: str) -> None:
    """No output may alias any immutable input or prerequisite, regardless of the input's role."""
    request, _source = _audit_request(tmp_path)

    with pytest.raises(sft.SFTSafetyError, match="overlap|overwrite|disjoint"):
        replace(request, curated_output=getattr(request, input_field))


@pytest.mark.parametrize("shape", ["equal_files", "equal_root", "directory_ancestor", "file_ancestor"])
def test_audit_request_rejects_equal_or_ancestor_output_shapes(tmp_path: Path, shape: str) -> None:
    """Audit outputs must be pairwise disjoint and may not contain one another."""
    request, _source = _audit_request(tmp_path)
    changes = {
        "equal_files": {"safety_manifest": request.curated_output},
        "equal_root": {"audit_root": request.curated_output},
        "directory_ancestor": {"audit_root": request.curated_output.parent},
        "file_ancestor": {"safety_manifest": request.curated_output / "SAFETY_MANIFEST.yaml"},
    }[shape]

    with pytest.raises(sft.SFTSafetyError, match="overlap|disjoint|ancestor"):
        replace(request, **changes)


def test_lexical_dotdot_output_alias_cannot_overwrite_or_delete_the_source(tmp_path: Path) -> None:
    """Lexically distinct `new/../source` output syntax must normalize before destructive ownership checks."""
    request, source_bytes = _audit_request(tmp_path)
    alias = request.source_fasta.parent / "new-component" / ".." / request.source_fasta.name

    with pytest.raises(sft.SFTSafetyError, match="overlap|overwrite|disjoint"):
        dangerous_request = replace(request, curated_output=alias)
        sft.audit_and_filter(dangerous_request, runtime=_fake_task4_runtime())

    assert request.source_fasta.read_bytes() == source_bytes


def _injected_source_identity(path: Path):
    payload = path.read_bytes()
    return sft.SFTSourceIdentity(
        provenance_kind="INJECTED_TEST_FIXTURE",
        repository="test-fixture",
        record_id="synthetic-microviridae",
        doi="test:synthetic-microviridae",
        file_name=path.name,
        url="test://synthetic-microviridae",
        size_bytes=len(payload),
        md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    )


def _injected_phix_identity(path: Path):
    records = parse_fasta_records(path)
    assert len(records) == 1
    sequence = records[0].normalized_sequence.encode("ascii")
    return sft.PhiXReferenceIdentity(
        provenance_kind="INJECTED_TEST_FIXTURE",
        accession="NC_001422.1",
        source="test-fixture",
        dataset_report_url="test://NC_001422.1/report",
        fasta_url="test://NC_001422.1/fasta",
        sequence_length=len(sequence),
        sequence_sha256=hashlib.sha256(sequence).hexdigest(),
        ncbi_sequence_hash="TEST0000",
    )


def _ncbi_host_report(accession: str) -> bytes:
    return json.dumps(
        {
            "reports": [
                {
                    "accession": accession,
                    "host": {
                        "tax_id": 561,
                        "organism_name": "Escherichia",
                        "lineage": [
                            {"tax_id": 131567, "name": "cellular organisms"},
                            {"tax_id": 2, "name": "Bacteria"},
                            {"tax_id": 561, "name": "Escherichia"},
                        ],
                    },
                }
            ],
            "total_count": 1,
        },
        separators=(",", ":"),
    ).encode()


def _write_upfront_authorization(tmp_path: Path) -> Path:
    authorization = sft.CorpusCountAuthorization(
        kind=sft.CorpusAuthorizationKind.UPFRONT_COMPLETION_MANDATE,
        verbatim_statement="Just do whatever it takes to get the run done.",
        authorized_at=datetime(2026, 8, 8, 11, 59, tzinfo=timezone.utc),
        scope="corpus_count_only",
        minimum_accepted_count=None,
    )
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(authorization.to_dict()))
    return path


def _audit_cli_args(request: sft.SFTAuditRequest, *, phix_table: Path, authorization_path: Path) -> list[str]:
    return [
        "audit-and-filter",
        "--source-fasta",
        str(request.source_fasta),
        "--host-evidence-table",
        str(request.host_evidence_table),
        "--phix-fasta",
        str(request.phix_fasta),
        "--phix-host-evidence-table",
        str(phix_table),
        "--curated-output",
        str(request.curated_output),
        "--safety-manifest",
        str(request.safety_manifest),
        "--audit-root",
        str(request.audit_root),
        "--preprocess-config",
        str(request.preprocess_config),
        "--policy",
        str(request.policy),
        "--asset-manifest",
        str(request.asset_manifest),
        "--diamond-tool-pin",
        str(request.diamond_tool_pin),
        "--mmseqs-tool-pin",
        str(request.mmseqs_tool_pin),
        "--authorization-json",
        str(authorization_path),
    ]


def test_production_identity_pins_reject_synthetic_source_and_fake_phix(tmp_path: Path) -> None:
    """Official labels on arbitrary bytes must not authenticate either production trust anchor."""
    source_request, _source = _audit_request(tmp_path / "source")
    source_runtime = replace(
        _fake_task4_runtime(),
        expected_source_identity=sft.PRODUCTION_SFT_SOURCE_IDENTITY,
        expected_phix_identity=_injected_phix_identity(source_request.phix_fasta),
    )
    with pytest.raises(sft.SFTSafetyError, match="published source identity"):
        sft.audit_and_filter(source_request, runtime=source_runtime)

    phix_request, _source = _audit_request(tmp_path / "phix")
    phix_runtime = replace(
        _fake_task4_runtime(),
        expected_source_identity=_injected_source_identity(phix_request.source_fasta),
        expected_phix_identity=sft.PRODUCTION_PHIX_REFERENCE_IDENTITY,
    )
    with pytest.raises(sft.SFTSafetyError, match="canonical PhiX sequence identity"):
        sft.audit_and_filter(phix_request, runtime=phix_runtime)


def test_source_identity_hash_parse_and_curation_share_one_authenticated_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A source that presents published bytes to identity/hash reads and altered bytes to parsing cannot certify."""
    request, source_bytes = _audit_request(tmp_path)
    altered_source = source_bytes.replace(b">b_pass\n+!ACGT\n", b">b_pass\n+!AAAA\n", 1)
    real_read = sft._read_regular_file_bytes

    def split_source_reads(path: str | Path, *, label: str) -> bytes:
        if Path(sft.os.path.abspath(sft.os.fspath(path))) == request.source_fasta:
            return altered_source if label == "conditioned FASTA" else source_bytes
        return real_read(path, label=label)

    monkeypatch.setattr(sft, "_read_regular_file_bytes", split_source_reads)
    runtime = _fake_task4_runtime()

    result = sft.audit_and_filter(request, runtime=runtime)
    validated = sft.validate_safety_manifest(
        request.safety_manifest,
        task4_manifest_validator=runtime.task4_manifest_validator,
        expected_minimum_genomes=request.minimum_genomes,
        expected_preferred_genomes=request.preferred_genomes,
        expected_source_identity=runtime.expected_source_identity,
        expected_phix_identity=runtime.expected_phix_identity,
    )

    assert result.exit_code == 0
    assert request.curated_output.read_bytes() == b">b_pass\n+!ACGT\n>a_pass\n+#TGCA\n"
    assert validated["curated_output"]["count"] == 2


def test_corpus_evidence_digest_and_rows_share_one_authenticated_snapshot(tmp_path: Path, monkeypatch) -> None:
    """An all-UNKNOWN table on disk cannot borrow previously positive rows from a separate loader read."""
    request, _source = _audit_request(tmp_path)
    positive_payload = request.host_evidence_table.read_bytes()
    current = load_host_evidence_table(request.host_evidence_table)
    write_host_evidence_table(
        request.host_evidence_table,
        HostEvidenceTable(
            table_id=current.table_id,
            created_at=current.created_at,
            rows=tuple(_host_row(row.record_id, HostDomain.UNKNOWN, confirmed=False) for row in current.rows),
        ),
    )
    real_read = host_evidence_module._read_regular_file_bytes

    def stale_positive_loader(path: str | Path, *, label: str) -> bytes:
        if (
            Path(host_evidence_module.os.path.abspath(host_evidence_module.os.fspath(path)))
            == request.host_evidence_table
        ):
            return positive_payload
        return real_read(path, label=label)

    monkeypatch.setattr(host_evidence_module, "_read_regular_file_bytes", stale_positive_loader)
    runtime = _fake_task4_runtime()

    result = sft.audit_and_filter(request, runtime=runtime)
    validated = sft.validate_safety_manifest(
        request.safety_manifest,
        task4_manifest_validator=runtime.task4_manifest_validator,
        require_ready=False,
        expected_minimum_genomes=request.minimum_genomes,
        expected_preferred_genomes=request.preferred_genomes,
        expected_source_identity=runtime.expected_source_identity,
        expected_phix_identity=runtime.expected_phix_identity,
    )

    assert result.exit_code == 3
    assert result.readiness == "BLOCKED"
    assert validated["curated_output"]["count"] == 0


def test_phix_evidence_digest_and_row_share_one_authenticated_snapshot(tmp_path: Path, monkeypatch) -> None:
    """An UNKNOWN PhiX table cannot borrow a positive bacterial row from a separate loader read."""
    request, _source = _audit_request(tmp_path)
    positive_payload = request.phix_host_evidence_table.read_bytes()
    current = load_host_evidence_table(request.phix_host_evidence_table)
    write_host_evidence_table(
        request.phix_host_evidence_table,
        HostEvidenceTable(
            table_id=current.table_id,
            created_at=current.created_at,
            rows=(_host_row("NC_001422.1", HostDomain.UNKNOWN, confirmed=False),),
        ),
    )
    real_read = host_evidence_module._read_regular_file_bytes

    def stale_positive_loader(path: str | Path, *, label: str) -> bytes:
        if (
            Path(host_evidence_module.os.path.abspath(host_evidence_module.os.fspath(path)))
            == request.phix_host_evidence_table
        ):
            return positive_payload
        return real_read(path, label=label)

    monkeypatch.setattr(host_evidence_module, "_read_regular_file_bytes", stale_positive_loader)
    runtime = _fake_task4_runtime()

    with pytest.raises(sft.SFTSafetyError, match="positive.*bacterial"):
        sft.audit_and_filter(request, runtime=runtime)
        sft.validate_safety_manifest(
            request.safety_manifest,
            task4_manifest_validator=runtime.task4_manifest_validator,
            expected_minimum_genomes=request.minimum_genomes,
            expected_preferred_genomes=request.preferred_genomes,
            expected_source_identity=runtime.expected_source_identity,
            expected_phix_identity=runtime.expected_phix_identity,
        )


def test_parent_validator_defaults_cannot_trust_injected_fixture_identities(tmp_path: Path) -> None:
    """A test-only identity is usable only when the caller explicitly injects the same expected identity."""
    request, _source = _audit_request(tmp_path)
    runtime = replace(
        _fake_task4_runtime(),
        expected_source_identity=_injected_source_identity(request.source_fasta),
        expected_phix_identity=_injected_phix_identity(request.phix_fasta),
    )
    sft.audit_and_filter(request, runtime=runtime)

    with pytest.raises(sft.SFTSafetyError, match="published source identity"):
        sft.validate_safety_manifest(
            request.safety_manifest,
            task4_manifest_validator=runtime.task4_manifest_validator,
            expected_minimum_genomes=request.minimum_genomes,
            expected_preferred_genomes=request.preferred_genomes,
        )


def test_sft_input_reads_reject_symlinked_parent_components(tmp_path: Path) -> None:
    """A non-symlink FASTA leaf beneath a symlinked directory is not a trusted immutable input."""
    real = tmp_path / "real"
    real.mkdir()
    source = real / "source.fna"
    source.write_bytes(b">one\n+!ACGT\n")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(sft.SFTSafetyError, match="symlink component"):
        sft.parse_conditioned_fasta(alias / source.name)


def test_regular_file_read_binds_each_parent_descriptor_against_replacement(tmp_path: Path, monkeypatch) -> None:
    """Replacing a checked parent with a symlink before final open must not redirect the authenticated read."""
    trusted_parent = tmp_path / "trusted"
    trusted_parent.mkdir()
    source = trusted_parent / "source.fna"
    source.write_bytes(b"trusted-bytes")
    saved_parent = tmp_path / "trusted-saved"
    malicious_parent = tmp_path / "malicious"
    malicious_parent.mkdir()
    (malicious_parent / source.name).write_bytes(b"malicious-bytes")
    real_open = sft.os.open
    swapped = False

    def swapping_open(path, flags, *args, dir_fd=None, **kwargs):
        nonlocal swapped
        path_text = sft.os.fspath(path)
        opens_full_path = dir_fd is None and Path(path_text) == source
        opens_bound_final = dir_fd is not None and path_text == source.name
        if not swapped and (opens_full_path or opens_bound_final):
            trusted_parent.rename(saved_parent)
            trusted_parent.symlink_to(malicious_parent, target_is_directory=True)
            swapped = True
        return real_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(sft.os, "open", swapping_open)

    assert sft._read_regular_file_bytes(source, label="race fixture") == b"trusted-bytes"
    assert swapped is True


def test_phix_identity_parses_the_authenticated_snapshot_without_reopening(tmp_path: Path, monkeypatch) -> None:
    """PhiX identity parsing must consume the no-follow byte snapshot, not reopen the mutable path."""
    request, _source = _audit_request(tmp_path)

    def forbidden_reopen(_path: Path):
        pytest.fail("PhiX identity validator reopened the authenticated path")

    monkeypatch.setattr(sft, "parse_fasta_records", forbidden_reopen, raising=False)

    sft._validate_phix_identity(request.phix_fasta, _injected_phix_identity(request.phix_fasta))


def test_audit_reuses_the_authenticated_phix_snapshot_without_reopening(tmp_path: Path, monkeypatch) -> None:
    """The audit must pass the already-authenticated PhiX record to later checks and Task 4."""
    request, _source = _audit_request(tmp_path)

    def forbidden_reopen(_path: Path):
        pytest.fail("SFT audit reopened the authenticated PhiX path")

    monkeypatch.setattr(sft, "parse_fasta_records", forbidden_reopen, raising=False)

    result = sft.audit_and_filter(request, runtime=_fake_task4_runtime())
    assert result.exit_code == 0


def test_phix_task4_consumes_an_audit_owned_authenticated_snapshot(tmp_path: Path, monkeypatch) -> None:
    """Changing the canonical PhiX path after identity validation must not change Task 4's input bytes."""
    request, _source = _audit_request(tmp_path)
    base_runtime = _fake_task4_runtime()
    observed: list[tuple[Path, bytes]] = []

    def capture_task4(task_request: sft.Task4SafetyRequest) -> sft.Task4SafetyArtifacts:
        if task_request.label == "phix_nc_001422_1":
            observed.append((task_request.input_fasta.absolute(), task_request.input_fasta.read_bytes()))
        return base_runtime.task4_runner(task_request)

    real_validate = sft._validate_phix_identity

    def validate_then_swap(path: Path, expected: sft.PhiXReferenceIdentity):
        record = real_validate(path, expected)
        path.write_bytes(b">NC_001422.1 phiX174\nAAAAAAAA\n")
        return record

    monkeypatch.setattr(sft, "_validate_phix_identity", validate_then_swap)
    runtime = replace(base_runtime, task4_runner=capture_task4)

    with pytest.raises(sft.SFTSafetyError, match="PhiX reference changed"):
        sft.audit_and_filter(request, runtime=runtime)

    assert observed == [(request.audit_root / "scanner_inputs" / "phix_nc_001422_1.fna", SYNTHETIC_PHIX_BYTES)]
    assert not request.audit_root.exists()


def test_parent_validator_reuses_the_authenticated_phix_snapshot_without_reopening(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Recursive parent validation must not reopen PhiX after authenticating its byte snapshot."""
    request, _source = _audit_request(tmp_path)
    runtime = _fake_task4_runtime()
    sft.audit_and_filter(request, runtime=runtime)

    def forbidden_reopen(_path: Path):
        pytest.fail("parent validator reopened the authenticated PhiX path")

    monkeypatch.setattr(sft, "parse_fasta_records", forbidden_reopen, raising=False)

    sft.validate_safety_manifest(
        request.safety_manifest,
        task4_manifest_validator=runtime.task4_manifest_validator,
        expected_minimum_genomes=request.minimum_genomes,
        expected_preferred_genomes=request.preferred_genomes,
        expected_source_identity=runtime.expected_source_identity,
        expected_phix_identity=runtime.expected_phix_identity,
    )


def test_phix_evidence_builder_reuses_the_authenticated_snapshot_without_reopening(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """PhiX evidence preparation must resolve the header retained in the authenticated snapshot."""
    request, _source = _audit_request(tmp_path)

    def forbidden_reopen(_path: Path):
        pytest.fail("PhiX evidence builder reopened the authenticated path")

    monkeypatch.setattr(sft, "parse_fasta_records", forbidden_reopen, raising=False)

    output = sft.prepare_phix_host_evidence_table(
        phix_fasta=request.phix_fasta,
        output_table=tmp_path / "prepared" / "PHIX_HOST_EVIDENCE.yaml",
        cache_dir=tmp_path / "prepared" / "cache",
        table_id="phix-snapshot-v1",
        ncbi_fetcher=lambda accession: _ncbi_host_report(accession),
        clock=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        expected_identity=_injected_phix_identity(request.phix_fasta),
    )
    assert load_host_evidence_table(output).rows[0].accession == "NC_001422.1"


def test_future_count_authorization_cannot_be_recast_as_upfront_permission(tmp_path: Path) -> None:
    """A mandate recorded after audit start must not authorize a below-minimum corpus."""
    request, _source = _audit_request(tmp_path)
    authorization = sft.CorpusCountAuthorization(
        kind=sft.CorpusAuthorizationKind.UPFRONT_COMPLETION_MANDATE,
        verbatim_statement="Just do whatever it takes to get the run done.",
        authorized_at=datetime(2999, 1, 1, tzinfo=timezone.utc),
        scope="corpus_count_only",
        minimum_accepted_count=None,
    )
    request = replace(request, authorization=authorization, minimum_genomes=10, preferred_genomes=20)

    with pytest.raises(sft.SFTSafetyError, match="authorized before the audit started"):
        sft.audit_and_filter(request, runtime=_fake_task4_runtime())


def test_parent_validator_rechecks_authorization_time_ordering(tmp_path: Path) -> None:
    """Editing an authorization timestamp after publication must invalidate parent readiness."""
    request, _source = _audit_request(tmp_path)
    authorization = sft.CorpusCountAuthorization(
        kind=sft.CorpusAuthorizationKind.UPFRONT_COMPLETION_MANDATE,
        verbatim_statement="Just do whatever it takes to get the run done.",
        authorized_at=datetime(2026, 8, 8, 11, 59, tzinfo=timezone.utc),
        scope="corpus_count_only",
        minimum_accepted_count=None,
    )
    request = replace(request, authorization=authorization, minimum_genomes=10, preferred_genomes=20)
    runtime = _fake_task4_runtime()
    sft.audit_and_filter(request, runtime=runtime)
    manifest = yaml.safe_load(request.safety_manifest.read_text())
    manifest["authorization"]["authorized_at"] = "2999-01-01T00:00:00Z"
    request.safety_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(sft.SFTSafetyError, match="authorized before the audit started"):
        sft.validate_safety_manifest(
            request.safety_manifest,
            task4_manifest_validator=runtime.task4_manifest_validator,
            require_ready=True,
            expected_minimum_genomes=request.minimum_genomes,
            expected_preferred_genomes=request.preferred_genomes,
            expected_source_identity=runtime.expected_source_identity,
            expected_phix_identity=runtime.expected_phix_identity,
        )


def test_source_header_accession_must_match_its_ordered_evidence_row(tmp_path: Path) -> None:
    """Ordered record IDs alone must not attach evidence to a different accession in the header."""
    request, source = _audit_request(tmp_path)
    changed = source.replace(b">b_pass\n", b">b_pass OQ123456.1\n")
    request.source_fasta.write_bytes(changed)
    runtime = replace(
        _fake_task4_runtime(),
        expected_source_identity=_injected_source_identity(request.source_fasta),
    )

    with pytest.raises(sft.SFTSafetyError, match="header accession"):
        sft.audit_and_filter(request, runtime=runtime)


def _request_with_accessionless_header_but_positive_ncbi_row(
    request: sft.SFTAuditRequest,
    tmp_path: Path,
) -> sft.SFTAuditRequest:
    table = load_host_evidence_table(request.host_evidence_table)
    ncbi_row = resolve_ncbi_host_evidence(
        record_id="b_pass",
        header=">OQ123456.1 cultured phage",
        cache_dir=tmp_path / "ncbi-cache",
        fetcher=lambda accession: _ncbi_host_report(accession),
        clock=lambda: datetime(2026, 8, 8, 11, 58, tzinfo=timezone.utc),
    )
    evidence_path = tmp_path / "evidence" / "HOST_EVIDENCE_WITH_UNBOUND_ACCESSION.yaml"
    write_host_evidence_table(
        evidence_path,
        HostEvidenceTable(
            table_id="unbound-accession-v1",
            created_at=datetime(2026, 8, 8, 11, 58, tzinfo=timezone.utc),
            rows=(ncbi_row, *table.rows[1:]),
        ),
    )
    return replace(request, host_evidence_table=evidence_path)


def test_accessionless_source_header_cannot_borrow_a_positive_ncbi_accession_row(tmp_path: Path) -> None:
    """Bidirectional binding must reject a positive accession row when its source header has no accession."""
    request, _source = _audit_request(tmp_path)
    request = _request_with_accessionless_header_but_positive_ncbi_row(request, tmp_path)

    with pytest.raises(sft.SFTSafetyError, match="header accession"):
        sft.audit_and_filter(request, runtime=_fake_task4_runtime())


def test_parent_manifest_rechecks_bidirectional_source_accession_binding(tmp_path: Path, monkeypatch) -> None:
    """Recursive validation must independently reject a manifest created with an unbound positive NCBI row."""
    request, _source = _audit_request(tmp_path)
    request = _request_with_accessionless_header_but_positive_ncbi_row(request, tmp_path)
    runtime = _fake_task4_runtime()

    def bypass_binding(_records, _rows) -> None:
        return None

    with monkeypatch.context() as patch:
        patch.setattr(sft, "_validate_source_accession_bindings", bypass_binding)
        sft.audit_and_filter(request, runtime=runtime)

    with pytest.raises(sft.SFTSafetyError, match="header accession"):
        sft.validate_safety_manifest(
            request.safety_manifest,
            task4_manifest_validator=runtime.task4_manifest_validator,
            expected_minimum_genomes=request.minimum_genomes,
            expected_preferred_genomes=request.preferred_genomes,
            expected_source_identity=runtime.expected_source_identity,
            expected_phix_identity=runtime.expected_phix_identity,
        )


def test_audit_and_cli_revalidate_phix_host_evidence_raw_artifacts(tmp_path: Path, capsys) -> None:
    """Neither direct audit nor its CLI may trust a PhiX row after its cached NCBI response drifts."""
    request, _source = _audit_request(tmp_path)
    row = resolve_ncbi_host_evidence(
        record_id="NC_001422.1",
        header=">NC_001422.1 phiX174",
        cache_dir=tmp_path / "phix-cache",
        fetcher=lambda accession: _ncbi_host_report(accession),
        clock=lambda: datetime(2026, 8, 8, 11, 58, tzinfo=timezone.utc),
    )
    phix_table = tmp_path / "evidence" / "PHIX_HOST_EVIDENCE_NCBI.yaml"
    write_host_evidence_table(
        phix_table,
        HostEvidenceTable(
            table_id="phix-ncbi-v2alpha",
            created_at=datetime(2026, 8, 8, 11, 58, tzinfo=timezone.utc),
            rows=(row,),
        ),
    )
    Path(row.raw_response_path).write_bytes(b"tampered")
    authorization_path = _write_upfront_authorization(tmp_path)

    with pytest.raises(sft.SFTSafetyError, match="PhiX host-evidence.*raw response"):
        sft.audit_and_filter(
            replace(request, phix_host_evidence_table=phix_table),
            runtime=_fake_task4_runtime(),
        )

    assert (
        sft.main(
            _audit_cli_args(request, phix_table=phix_table, authorization_path=authorization_path),
            runtime=_fake_task4_runtime(),
        )
        == 3
    )
    assert "raw response" in capsys.readouterr().err


def test_parent_manifest_revalidates_bound_phix_evidence_table(tmp_path: Path) -> None:
    """A PhiX raw response changed after audit must invalidate the completed parent manifest."""
    request, _source = _audit_request(tmp_path)
    row = resolve_ncbi_host_evidence(
        record_id="NC_001422.1",
        header=">NC_001422.1 phiX174",
        cache_dir=tmp_path / "phix-cache",
        fetcher=lambda accession: _ncbi_host_report(accession),
        clock=lambda: datetime(2026, 8, 8, 11, 58, tzinfo=timezone.utc),
    )
    phix_table = tmp_path / "evidence" / "PHIX_HOST_EVIDENCE_NCBI.yaml"
    write_host_evidence_table(
        phix_table,
        HostEvidenceTable(
            table_id="phix-ncbi-v2alpha",
            created_at=datetime(2026, 8, 8, 11, 58, tzinfo=timezone.utc),
            rows=(row,),
        ),
    )
    runtime = _fake_task4_runtime()
    authorization_path = _write_upfront_authorization(tmp_path)
    assert (
        sft.main(
            _audit_cli_args(request, phix_table=phix_table, authorization_path=authorization_path),
            runtime=runtime,
        )
        == 0
    )
    Path(row.raw_response_path).write_bytes(b"tampered-after-audit")

    with pytest.raises(sft.SFTSafetyError, match="PhiX host-evidence.*raw response"):
        sft.validate_safety_manifest(
            request.safety_manifest,
            task4_manifest_validator=runtime.task4_manifest_validator,
            expected_source_identity=runtime.expected_source_identity,
            expected_phix_identity=runtime.expected_phix_identity,
        )


@pytest.mark.parametrize(
    "metadata_path",
    [
        ("host_evidence_table",),
        ("phix_reference", "host_evidence_table"),
    ],
)
def test_parent_manifest_rejects_boolean_host_evidence_schema_versions(
    tmp_path: Path,
    metadata_path: tuple[str, ...],
) -> None:
    """YAML booleans must not satisfy either integer host-evidence schema-version field."""
    request, _source = _audit_request(tmp_path)
    runtime = _fake_task4_runtime()
    sft.audit_and_filter(request, runtime=runtime)
    manifest = yaml.safe_load(request.safety_manifest.read_text())
    metadata = manifest
    for key in metadata_path:
        metadata = metadata[key]
    metadata["schema_version"] = True
    request.safety_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(sft.SFTSafetyError, match="host-evidence table lineage drift"):
        sft.validate_safety_manifest(
            request.safety_manifest,
            task4_manifest_validator=runtime.task4_manifest_validator,
            expected_minimum_genomes=request.minimum_genomes,
            expected_preferred_genomes=request.preferred_genomes,
            expected_source_identity=runtime.expected_source_identity,
            expected_phix_identity=runtime.expected_phix_identity,
        )


def test_malformed_or_missing_phix_evidence_maps_to_cli_exit_three(tmp_path: Path, capsys) -> None:
    """Host-evidence validation/configuration errors are indeterminate, never uncaught or biological FAIL."""
    request, _source = _audit_request(tmp_path)
    malformed = tmp_path / "evidence" / "malformed.yaml"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("not: [valid")
    authorization_path = _write_upfront_authorization(tmp_path)

    assert (
        sft.main(
            _audit_cli_args(request, phix_table=malformed, authorization_path=authorization_path),
            runtime=_fake_task4_runtime(),
        )
        == 3
    )
    assert "error" in capsys.readouterr().err


def test_corpus_host_evidence_builder_resolves_accessions_and_requires_supplemental_rows(tmp_path: Path) -> None:
    """The production builder must emit one ordered fail-closed row without guessing non-accession hosts."""
    source = tmp_path / "source.fna"
    source.write_bytes(b">OQ123456.1 cultured phage\n+!ACGT\n>IMGVR_UViG_1\n+#TGCA\n")
    supplemental_path = tmp_path / "SUPPLEMENTAL_HOST_EVIDENCE.yaml"
    write_host_evidence_table(
        supplemental_path,
        HostEvidenceTable(
            table_id="reviewed-imgvr-v1",
            created_at=datetime(2026, 8, 8, 11, 57, tzinfo=timezone.utc),
            rows=(_host_row("IMGVR_UViG_1", HostDomain.ARCHAEA),),
        ),
    )
    calls: list[str] = []

    def fetch(accession: str) -> bytes:
        calls.append(accession)
        return _ncbi_host_report(accession)

    output = tmp_path / "HOST_EVIDENCE.yaml"
    sft.prepare_host_evidence_table(
        source_fasta=source,
        output_table=output,
        cache_dir=tmp_path / "cache",
        table_id="complete-host-evidence-v1",
        supplemental_table=supplemental_path,
        ncbi_fetcher=fetch,
        clock=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
    )
    table = load_host_evidence_table(output)

    assert calls == ["OQ123456.1"]
    assert [row.record_id for row in table.rows] == ["OQ123456.1", "IMGVR_UViG_1"]
    assert [row.evidence_source for row in table.rows] == ["NCBI_DATASETS", "reviewed_multi_source_catalog"]
    assert [row.normalized_host_domain for row in table.rows] == [HostDomain.BACTERIA, HostDomain.ARCHAEA]

    with pytest.raises(sft.SFTSafetyError, match="supplemental"):
        sft.prepare_host_evidence_table(
            source_fasta=source,
            output_table=tmp_path / "incomplete.yaml",
            cache_dir=tmp_path / "cache-2",
            table_id="incomplete",
            supplemental_table=None,
            ncbi_fetcher=fetch,
            clock=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        )


def test_host_evidence_builder_never_leaves_or_clobbers_an_untrusted_output(tmp_path: Path) -> None:
    """Post-write reconciliation failure and an existing destination must both remain transaction-safe."""
    source = tmp_path / "source.fna"
    source.write_bytes(b">OQ123456.1 cultured phage\n+!ACGT\n")
    failed_output = tmp_path / "failed.yaml"

    with pytest.raises(sft.SFTSafetyError, match="generated host-evidence table"):
        sft.prepare_host_evidence_table(
            source_fasta=source,
            output_table=failed_output,
            cache_dir=tmp_path / "bad-cache",
            table_id="must-not-publish",
            ncbi_fetcher=lambda _accession: b"{}",
            clock=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        )
    assert not failed_output.exists()

    existing_output = tmp_path / "existing.yaml"
    existing_output.write_bytes(b"preserve-reviewed-artifact")
    with pytest.raises(sft.SFTSafetyError, match="already exists"):
        sft.prepare_host_evidence_table(
            source_fasta=source,
            output_table=existing_output,
            cache_dir=tmp_path / "good-cache",
            table_id="must-not-clobber",
            ncbi_fetcher=lambda accession: _ncbi_host_report(accession),
            clock=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        )
    assert existing_output.read_bytes() == b"preserve-reviewed-artifact"


def test_host_evidence_publication_cannot_succeed_through_a_replaced_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The builder must fail and clean its bound inodes when its destination parent is renamed after link."""
    destination_parent = tmp_path / "evidence"
    destination_parent.mkdir()
    displaced_parent = tmp_path / "evidence-displaced"
    destination = destination_parent / "HOST_EVIDENCE.yaml"
    table = HostEvidenceTable(
        table_id="reviewed-hosts-v1",
        created_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        rows=(_host_row("reviewed-one", HostDomain.BACTERIA),),
    )
    real_link = sft.os.link
    swapped = False

    def replace_parent_after_link(source, output, *args, **kwargs):
        nonlocal swapped
        result = real_link(source, output, *args, **kwargs)
        destination_parent.rename(displaced_parent)
        destination_parent.mkdir()
        swapped = True
        return result

    monkeypatch.setattr(sft.os, "link", replace_parent_after_link)

    with pytest.raises(sft.SFTSafetyError, match="destination.*(parent|path|identity).*(changed|replaced)"):
        sft._publish_validated_host_evidence_table(destination, table, label="host-evidence table")

    assert swapped is True
    assert not destination.exists()
    assert list(destination_parent.iterdir()) == []
    assert list(displaced_parent.iterdir()) == []


def test_owned_byte_publication_cannot_succeed_through_a_replaced_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The shared no-clobber publisher must bind successful return to the caller-visible parent and name."""
    destination_parent = tmp_path / "curated"
    destination_parent.mkdir()
    displaced_parent = tmp_path / "curated-displaced"
    destination = destination_parent / "safety-pass.fna"
    real_link = sft.os.link
    swapped = False

    def replace_parent_after_link(source, output, *args, **kwargs):
        nonlocal swapped
        result = real_link(source, output, *args, **kwargs)
        destination_parent.rename(displaced_parent)
        destination_parent.mkdir()
        swapped = True
        return result

    monkeypatch.setattr(sft.os, "link", replace_parent_after_link)

    with pytest.raises(sft.SFTSafetyError, match="curated output.*(parent|path|identity).*(changed|replaced)"):
        sft._publish_owned_bytes(destination, b">one\n+!ACGT\n", label="curated output")

    assert swapped is True
    assert not destination.exists()
    assert list(destination_parent.iterdir()) == []
    assert list(displaced_parent.iterdir()) == []


def test_host_evidence_cli_builds_corpus_and_phix_tables_without_live_network(tmp_path: Path) -> None:
    """The installed workflow must create both tables using a bounded mocked resolver in tests."""
    source = tmp_path / "source.fna"
    source.write_bytes(b">OQ123456.1 cultured phage\n+!ACGT\n")
    corpus_output = tmp_path / "HOST_EVIDENCE.yaml"

    def fetch(accession: str) -> bytes:
        return _ncbi_host_report(accession)

    assert (
        sft.host_evidence_main(
            [
                "corpus",
                "--source-fasta",
                str(source),
                "--output-table",
                str(corpus_output),
                "--cache-dir",
                str(tmp_path / "corpus-cache"),
                "--table-id",
                "corpus-v1",
            ],
            ncbi_fetcher=fetch,
            clock=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        )
        == 0
    )
    assert [row.record_id for row in load_host_evidence_table(corpus_output).rows] == ["OQ123456.1"]

    request, _source = _audit_request(tmp_path / "phix-fixture")
    phix_output = tmp_path / "PHIX_HOST_EVIDENCE.yaml"
    assert (
        sft.host_evidence_main(
            [
                "phix",
                "--phix-fasta",
                str(request.phix_fasta),
                "--output-table",
                str(phix_output),
                "--cache-dir",
                str(tmp_path / "phix-cache"),
                "--table-id",
                "phix-v1",
            ],
            ncbi_fetcher=fetch,
            clock=lambda: datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            expected_phix_identity=_injected_phix_identity(request.phix_fasta),
        )
        == 0
    )
    assert [row.accession for row in load_host_evidence_table(phix_output).rows] == ["NC_001422.1"]


def test_audit_groups_prokaryotic_hosts_filters_task4_pass_and_preserves_conditioned_source_bytes(
    tmp_path: Path,
) -> None:
    """Bypassing domain scans, PhiX, duplicates, or original-byte curation must invalidate the SFT artifact."""
    request, source_bytes = _audit_request(tmp_path)
    runtime = _fake_task4_runtime()

    result = sft.audit_and_filter(request, runtime=runtime)
    validated = sft.validate_safety_manifest(
        request.safety_manifest,
        task4_manifest_validator=runtime.task4_manifest_validator,
        require_ready=True,
        expected_minimum_genomes=request.minimum_genomes,
        expected_preferred_genomes=request.preferred_genomes,
        expected_source_identity=runtime.expected_source_identity,
        expected_phix_identity=runtime.expected_phix_identity,
    )

    assert result.exit_code == 0
    assert request.source_fasta.read_bytes() == source_bytes
    assert request.curated_output.read_bytes() == b">b_pass\n+!ACGT\n>a_pass\n+#TGCA\n"
    assert validated["source"]["sha256_before"] == validated["source"]["sha256_after"]
    assert validated["quality"]["duplicate_count"] == 1
    assert validated["curated_output"]["count"] == 2
    assert validated["curated_output"]["distinct_genome_hash_count"] == 2
    assert validated["adequacy"]["state"] == "PREFERRED_SIZE_MET"
    assert validated["phix_reference"]["accession"] == "NC_001422.1"
    assert validated["phix_reference"]["state"] == "PASS"
    assert Path(validated["phix_reference"]["source_fasta"]["path"]) == request.phix_fasta
    assert Path(validated["phix_reference"]["input_fasta"]["path"]) != request.phix_fasta
    assert request.audit_root in Path(validated["phix_reference"]["input_fasta"]["path"]).parents
    assert Path(validated["phix_reference"]["input_fasta"]["path"]).read_bytes() == SYNTHETIC_PHIX_BYTES
    assert [child["host_domain"] for child in validated["domain_children"]] == [
        "BACTERIA",
        "ARCHAEA",
        "BACTERIA_AND_ARCHAEA",
    ]
    decisions = {row["record_id"]: row for row in validated["record_decisions"]}
    assert decisions["b_pass"]["eligible_for_sft"] is True
    assert decisions["a_pass"]["eligible_for_sft"] is True
    assert decisions["ba_fail"]["reason_codes"] == ["SEQUENCE_SAFETY_FAIL"]
    assert decisions["euk"]["reason_codes"] == ["EUKARYOTIC_REPLICATION_HOST"]
    assert decisions["unknown"]["reason_codes"] == ["INCOMPLETE_HOST_EVIDENCE"]
    assert decisions["duplicate"]["reason_codes"] == ["DUPLICATE_BIOLOGICAL_GENOME"]
    assert [entry["conditioning_prefix"] for entry in validated["conditioning_lineage"]] == [
        "+!",
        "+#",
        "+$",
        "+^",
        "+~",
        "+!",
    ]
    assert validated["conditioning_summary"] == {
        "allowed_prefixes": ["+!", "+#", "+$", "+^", "+~"],
        "observed_counts": {"+!": 2, "+#": 1, "+$": 1, "+^": 1, "+~": 1},
        "historical_zenodo_counts": {"+!": 52, "+#": 388, "+$": 13729, "+^": 166, "+~": 131},
        "historical_total": 14466,
    }
    with pytest.raises(sft.SFTSafetyError, match="nonstandard corpus thresholds"):
        sft.validate_safety_manifest(
            request.safety_manifest,
            task4_manifest_validator=runtime.task4_manifest_validator,
            expected_source_identity=runtime.expected_source_identity,
            expected_phix_identity=runtime.expected_phix_identity,
        )


@pytest.mark.parametrize(
    ("runtime", "exit_code", "message"),
    [
        (_fake_task4_runtime(diagnostic_label="bacteria"), 3, "diagnostic"),
        (_fake_task4_runtime(phix_state="FAIL"), 2, "PhiX"),
        (_fake_task4_runtime(phix_state="INDETERMINATE"), 3, "PhiX"),
    ],
)
def test_audit_rejects_diagnostics_and_requires_independent_phix_pass(
    tmp_path: Path,
    runtime: sft.SFTAuditRuntime,
    exit_code: int,
    message: str,
) -> None:
    """Neither missing scanner prerequisites nor a historical reference identity may bypass Task 4 PASS."""
    request, source_bytes = _audit_request(tmp_path)

    with pytest.raises(sft.SFTSafetyError, match=message) as error:
        sft.audit_and_filter(request, runtime=runtime)

    assert error.value.exit_code == exit_code
    assert request.source_fasta.read_bytes() == source_bytes
    assert not request.safety_manifest.exists()
    assert not request.curated_output.exists()
    assert not request.audit_root.exists()

    retry = sft.audit_and_filter(request, runtime=_fake_task4_runtime())
    assert retry.exit_code == 0


def test_audit_preflight_never_removes_a_preexisting_output_path(tmp_path: Path) -> None:
    """Rollback ownership begins only after the all-outputs-absent preflight succeeds."""
    request, _source = _audit_request(tmp_path)
    request.audit_root.mkdir()
    sentinel = request.audit_root / "operator-owned.txt"
    sentinel.write_bytes(b"preserve-me")

    with pytest.raises(sft.SFTSafetyError, match="must not already exist"):
        sft.audit_and_filter(request, runtime=_fake_task4_runtime())

    assert sentinel.read_bytes() == b"preserve-me"


def test_audit_validates_task4_manifests_from_stable_same_directory_snapshots(tmp_path: Path) -> None:
    """Hash/type checks and the injected Task 4 validator must consume the same captured manifest bytes."""
    request, _source = _audit_request(tmp_path)
    base_runtime = _fake_task4_runtime()
    emitted: set[Path] = set()
    validator_paths: list[Path] = []

    def record_artifacts(task_request: sft.Task4SafetyRequest) -> sft.Task4SafetyArtifacts:
        artifacts = base_runtime.task4_runner(task_request)
        emitted.update({artifacts.scan_manifest.absolute(), artifacts.filter_manifest.absolute()})
        return artifacts

    def validate_snapshot(path: str | Path, *, expected_type: str | None = None) -> Mapping[str, object]:
        snapshot = Path(path).absolute()
        validator_paths.append(snapshot)
        assert snapshot not in emitted
        assert any(snapshot.parent == original.parent for original in emitted)
        return base_runtime.task4_manifest_validator(snapshot, expected_type=expected_type)

    runtime = replace(base_runtime, task4_runner=record_artifacts, task4_manifest_validator=validate_snapshot)
    result = sft.audit_and_filter(request, runtime=runtime)

    assert result.exit_code == 0
    assert len(validator_paths) == 16
    assert all(not path.exists() for path in validator_paths)


def test_parent_validation_reads_each_bound_artifact_once_and_snapshots_task4_manifests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Recursive validation must not hash one open and parse or delegate a second mutable open."""
    request, _source = _audit_request(tmp_path)
    runtime = _fake_task4_runtime()
    sft.audit_and_filter(request, runtime=runtime)
    manifest = yaml.safe_load(request.safety_manifest.read_text())
    task4_paths = {
        Path(child[key]["path"]).absolute()
        for child in manifest["domain_children"]
        for key in ("scan_manifest", "filter_manifest")
    }
    task4_paths.update(
        Path(manifest["phix_reference"][key]["path"]).absolute() for key in ("scan_manifest", "filter_manifest")
    )
    bound_paths = {
        request.source_fasta,
        request.host_evidence_table,
        request.curated_output,
        request.preprocess_config,
        request.phix_fasta,
        request.phix_host_evidence_table,
        *(Path(child["input_fasta"]["path"]).absolute() for child in manifest["domain_children"]),
        Path(manifest["phix_reference"]["input_fasta"]["path"]).absolute(),
        *task4_paths,
    }
    read_counts = {path: 0 for path in bound_paths}
    real_sft_read = sft._read_regular_file_bytes
    real_host_read = host_evidence_module._read_regular_file_bytes

    def count_sft_read(path: str | Path, *, label: str) -> bytes:
        absolute = Path(sft.os.path.abspath(sft.os.fspath(path)))
        if absolute in read_counts:
            read_counts[absolute] += 1
        return real_sft_read(path, label=label)

    def count_host_read(path: str | Path, *, label: str) -> bytes:
        absolute = Path(host_evidence_module.os.path.abspath(host_evidence_module.os.fspath(path)))
        if absolute in read_counts:
            read_counts[absolute] += 1
        return real_host_read(path, label=label)

    validator_paths: list[Path] = []

    def validate_snapshot(path: str | Path, *, expected_type: str | None = None) -> Mapping[str, object]:
        snapshot = Path(path).absolute()
        validator_paths.append(snapshot)
        assert snapshot not in task4_paths
        assert any(snapshot.parent == original.parent for original in task4_paths)
        return runtime.task4_manifest_validator(snapshot, expected_type=expected_type)

    monkeypatch.setattr(sft, "_read_regular_file_bytes", count_sft_read)
    monkeypatch.setattr(host_evidence_module, "_read_regular_file_bytes", count_host_read)

    sft.validate_safety_manifest(
        request.safety_manifest,
        task4_manifest_validator=validate_snapshot,
        expected_minimum_genomes=request.minimum_genomes,
        expected_preferred_genomes=request.preferred_genomes,
        expected_source_identity=runtime.expected_source_identity,
        expected_phix_identity=runtime.expected_phix_identity,
    )

    assert read_counts == dict.fromkeys(bound_paths, 1)
    assert len(validator_paths) == 8
    assert all(not path.exists() for path in validator_paths)


def test_audit_rollback_preserves_a_foreign_output_created_after_preflight(tmp_path: Path) -> None:
    """Cleanup must remove only inode-owned outputs and must not mask the initiating failure."""
    request, _source = _audit_request(tmp_path)
    sentinel = b"concurrent-writer-owned"

    def failing_runner(_task_request: sft.Task4SafetyRequest) -> sft.Task4SafetyArtifacts:
        request.curated_output.parent.mkdir(parents=True, exist_ok=True)
        request.curated_output.write_bytes(sentinel)
        raise sft.SFTSafetyError("injected primary failure")

    runtime = replace(_fake_task4_runtime(), task4_runner=failing_runner)

    with pytest.raises(sft.SFTSafetyError, match="injected primary failure"):
        sft.audit_and_filter(request, runtime=runtime)

    assert request.curated_output.read_bytes() == sentinel
    assert not request.audit_root.exists()


def test_audit_never_writes_through_a_replaced_claimed_root_path(tmp_path: Path) -> None:
    """Renaming the inode-owned root and planting a replacement must fail before any audit write or certification."""
    request, _source = _audit_request(tmp_path)
    displaced_owned_root = tmp_path / "displaced-owned-audit-root"
    foreign_sentinel = b"concurrent-writer-owned"
    swapped = False

    def swapping_clock() -> datetime:
        nonlocal swapped
        if not swapped:
            request.audit_root.rename(displaced_owned_root)
            request.audit_root.mkdir()
            (request.audit_root / "operator-owned.txt").write_bytes(foreign_sentinel)
            swapped = True
        return datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)

    runtime = replace(_fake_task4_runtime(), clock=swapping_clock)
    caught: sft.SFTSafetyError | None = None
    try:
        sft.audit_and_filter(request, runtime=runtime)
    except sft.SFTSafetyError as error:
        caught = error

    assert swapped is True
    assert (request.audit_root / "operator-owned.txt").read_bytes() == foreign_sentinel
    assert {path.name for path in request.audit_root.iterdir()} == {"operator-owned.txt"}
    assert caught is not None
    assert "audit root" in str(caught).lower()
    assert not request.curated_output.exists()
    assert not request.safety_manifest.exists()
    assert displaced_owned_root.exists()
    assert list(displaced_owned_root.iterdir()) == []


def test_task4_callback_stays_on_claimed_root_when_named_root_is_replaced(tmp_path: Path) -> None:
    """All callback I/O must remain rooted at the claimed directory inode across a pathname swap."""
    request, _source = _audit_request(tmp_path)
    base_runtime = _fake_task4_runtime()
    displaced_owned_root = tmp_path / "displaced-task4-audit-root"
    foreign_sentinel = b"concurrent-writer-owned"
    callback_paths_are_relative: list[bool] = []
    swapped = False

    def swap_inside_task4(task_request: sft.Task4SafetyRequest) -> sft.Task4SafetyArtifacts:
        nonlocal swapped
        if not swapped:
            request.audit_root.rename(displaced_owned_root)
            request.audit_root.mkdir()
            (request.audit_root / "operator-owned.txt").write_bytes(foreign_sentinel)
            swapped = True
        callback_paths = (
            task_request.input_fasta,
            task_request.output_root,
            task_request.policy,
            task_request.asset_manifest,
            task_request.diamond_tool_pin,
            task_request.mmseqs_tool_pin,
        )
        callback_paths_are_relative.append(all(not path.is_absolute() for path in callback_paths))
        marker = task_request.output_root / "callback-owned-marker.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"owned-callback-output")
        for prerequisite in (
            task_request.input_fasta,
            task_request.policy,
            task_request.asset_manifest,
            task_request.diamond_tool_pin,
            task_request.mmseqs_tool_pin,
        ):
            assert prerequisite.read_bytes()
        return base_runtime.task4_runner(task_request)

    runtime = replace(base_runtime, task4_runner=swap_inside_task4)
    with pytest.raises(sft.SFTSafetyError, match="audit root.*(changed|replaced)"):
        sft.audit_and_filter(request, runtime=runtime)

    assert swapped is True
    assert callback_paths_are_relative == [True]
    assert (request.audit_root / "operator-owned.txt").read_bytes() == foreign_sentinel
    assert {path.name for path in request.audit_root.iterdir()} == {"operator-owned.txt"}
    assert displaced_owned_root.exists()
    assert list(displaced_owned_root.iterdir()) == []
    assert not request.curated_output.exists()
    assert not request.safety_manifest.exists()


def test_task4_receives_stable_audit_owned_prerequisite_snapshots(tmp_path: Path) -> None:
    """Policy, assets, and tool pins handed to Task 4 must not be mutable caller-owned paths."""
    request, _source = _audit_request(tmp_path)
    base_runtime = _fake_task4_runtime()
    fields = ("policy", "asset_manifest", "diamond_tool_pin", "mmseqs_tool_pin")
    originals = {field: getattr(request, field).read_bytes() for field in fields}
    observed: list[dict[str, tuple[Path, bytes]]] = []
    mutated = False

    def capture_task4(task_request: sft.Task4SafetyRequest) -> sft.Task4SafetyArtifacts:
        nonlocal mutated
        if not mutated:
            for field in fields:
                getattr(request, field).write_bytes(b"attacker-controlled")
            mutated = True
        observed.append(
            {
                field: (getattr(task_request, field).absolute(), getattr(task_request, field).read_bytes())
                for field in fields
            }
        )
        return base_runtime.task4_runner(task_request)

    result = sft.audit_and_filter(request, runtime=replace(base_runtime, task4_runner=capture_task4))

    assert result.exit_code == 0
    assert observed
    for task_inputs in observed:
        for field in fields:
            snapshot_path, snapshot_bytes = task_inputs[field]
            assert snapshot_path != getattr(request, field)
            assert request.audit_root in snapshot_path.parents
            assert snapshot_bytes == originals[field]


def test_parent_manifest_and_guarded_preprocess_reject_tampering_before_delegation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The shared preprocessor must never receive historical, stale, or unmanifested input."""
    request, _source_bytes = _audit_request(tmp_path)
    runtime = _fake_task4_runtime()
    sft.audit_and_filter(request, runtime=runtime)
    calls: list[list[dict[str, object]]] = []
    snapshots: list[tuple[Path, bytes]] = []
    expected_curated = request.curated_output.read_bytes()
    monkeypatch.chdir(tmp_path)

    def capture_snapshot(batch: list[dict[str, object]]) -> None:
        calls.append(batch)
        snapshot = Path(batch[0]["datapaths"][0])
        snapshots.append((snapshot, snapshot.read_bytes()))

    exit_code = sft.preprocess_main(
        ["--config", str(request.preprocess_config)],
        delegate=capture_snapshot,
        safety_manifest_validator=lambda path, **kwargs: sft.validate_safety_manifest(
            path,
            task4_manifest_validator=runtime.task4_manifest_validator,
            expected_minimum_genomes=request.minimum_genomes,
            expected_preferred_genomes=request.preferred_genomes,
            expected_source_identity=runtime.expected_source_identity,
            expected_phix_identity=runtime.expected_phix_identity,
            **kwargs,
        ),
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert "safety_manifest" not in calls[0][0]
    assert snapshots[0][0] != request.curated_output
    assert snapshots[0][1] == expected_curated
    assert not snapshots[0][0].exists()
    request.curated_output.write_bytes(request.curated_output.read_bytes() + b">tamper\n+!AAAA\n")
    with pytest.raises(sft.SFTSafetyError, match="curated output digest"):
        sft.preprocess_main(
            ["--config", str(request.preprocess_config)],
            delegate=lambda batch: calls.append(batch),
            safety_manifest_validator=lambda path, **kwargs: sft.validate_safety_manifest(
                path,
                task4_manifest_validator=runtime.task4_manifest_validator,
                expected_minimum_genomes=request.minimum_genomes,
                expected_preferred_genomes=request.preferred_genomes,
                expected_source_identity=runtime.expected_source_identity,
                expected_phix_identity=runtime.expected_phix_identity,
                **kwargs,
            ),
        )
    assert len(calls) == 1

    historical = yaml.safe_load(request.preprocess_config.read_text())
    historical[0]["datapaths"] = ["data/external/zenodo/microviridae_sft_training_data_processed.fna"]
    historical_path = tmp_path / "configs" / "historical.yaml"
    historical_path.write_text(yaml.safe_dump(historical, sort_keys=False))
    with pytest.raises(sft.SFTSafetyError, match="safety-pass curated FASTA"):
        sft.preprocess_main(
            ["--config", str(historical_path)],
            delegate=lambda batch: calls.append(batch),
            safety_manifest_validator=lambda path, **kwargs: {},
        )


def test_preprocess_never_delegates_bytes_swapped_after_parent_validation(tmp_path: Path, monkeypatch) -> None:
    """A post-validation path swap must fail before the shared preprocessor can consume malicious bytes."""
    request, _source = _audit_request(tmp_path)
    runtime = _fake_task4_runtime()
    sft.audit_and_filter(request, runtime=runtime)
    monkeypatch.chdir(tmp_path)
    delegated: list[list[dict[str, object]]] = []

    def validate_then_swap(path: Path, **kwargs):
        manifest = sft.validate_safety_manifest(
            path,
            task4_manifest_validator=runtime.task4_manifest_validator,
            expected_minimum_genomes=request.minimum_genomes,
            expected_preferred_genomes=request.preferred_genomes,
            expected_source_identity=runtime.expected_source_identity,
            expected_phix_identity=runtime.expected_phix_identity,
            **kwargs,
        )
        request.curated_output.write_bytes(b">malicious\n+!AAAA\n")
        return manifest

    with pytest.raises(sft.SFTSafetyError, match="curated.*(changed|digest|snapshot)"):
        sft.preprocess_main(
            ["--config", str(request.preprocess_config)],
            delegate=lambda batch: delegated.append(batch),
            safety_manifest_validator=validate_then_swap,
        )

    assert delegated == []


def test_preprocess_config_batch_and_parent_digest_share_one_snapshot(tmp_path: Path, monkeypatch) -> None:
    """A delegate batch parsed from bytes other than the parent-bound config snapshot must never run."""
    request, _source = _audit_request(tmp_path)
    runtime = _fake_task4_runtime()
    sft.audit_and_filter(request, runtime=runtime)
    monkeypatch.chdir(tmp_path)
    canonical_config = request.preprocess_config.read_bytes()
    altered = yaml.safe_load(canonical_config)
    altered[0]["output_dir"] = "attacker-controlled-output"
    altered[0]["force_uppercase"] = True
    altered_config = yaml.safe_dump(altered, sort_keys=False).encode()
    real_read = sft._read_regular_file_bytes
    config_reads = 0

    def split_config_reads(path: str | Path, *, label: str) -> bytes:
        nonlocal config_reads
        absolute = Path(sft.os.path.abspath(sft.os.fspath(path)))
        if absolute == request.preprocess_config:
            config_reads += 1
            return altered_config if config_reads == 1 else canonical_config
        return real_read(path, label=label)

    monkeypatch.setattr(sft, "_read_regular_file_bytes", split_config_reads)
    delegated: list[list[dict[str, object]]] = []

    with pytest.raises(sft.SFTSafetyError, match="preprocess config.*(digest|snapshot|changed)"):
        sft.preprocess_main(
            ["--config", str(request.preprocess_config)],
            delegate=lambda batch: delegated.append(batch),
            safety_manifest_validator=lambda path, **kwargs: sft.validate_safety_manifest(
                path,
                task4_manifest_validator=runtime.task4_manifest_validator,
                expected_minimum_genomes=request.minimum_genomes,
                expected_preferred_genomes=request.preferred_genomes,
                expected_source_identity=runtime.expected_source_identity,
                expected_phix_identity=runtime.expected_phix_identity,
                **kwargs,
            ),
        )

    assert config_reads == 2
    assert delegated == []


def test_preprocess_private_snapshot_cannot_be_replaced_before_trusted_consumption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request, _source = _audit_request(tmp_path)
    runtime = _fake_task4_runtime()
    sft.audit_and_filter(request, runtime=runtime)
    expected = request.curated_output.read_bytes()
    malicious = b">malicious\n+!AAAA\n"
    consumed: list[bytes] = []
    exposed: list[Path] = []
    monkeypatch.chdir(tmp_path)

    def concurrent_swap_then_trusted_consumer(batch: list[dict[str, object]]) -> None:
        snapshot = Path(batch[0]["datapaths"][0])
        exposed.append(snapshot)
        try:
            snapshot.unlink()
            snapshot.write_bytes(malicious)
        except OSError:
            pass
        consumed.append(snapshot.read_bytes())

    assert (
        sft.preprocess_main(
            ["--config", str(request.preprocess_config)],
            delegate=concurrent_swap_then_trusted_consumer,
            safety_manifest_validator=lambda path, **kwargs: sft.validate_safety_manifest(
                path,
                task4_manifest_validator=runtime.task4_manifest_validator,
                expected_minimum_genomes=request.minimum_genomes,
                expected_preferred_genomes=request.preferred_genomes,
                expected_source_identity=runtime.expected_source_identity,
                expected_phix_identity=runtime.expected_phix_identity,
                **kwargs,
            ),
        )
        == 0
    )
    assert consumed == [expected]
    assert not exposed[0].exists()


def test_shared_preprocess_sanitized_config_cannot_be_rewritten_before_consumption(monkeypatch) -> None:
    """The production delegate must expose immutable config bytes for the shared parser's full read."""
    import bionemo.evo2.data as shared_data

    expected = [{"datapaths": ["/trusted/curated.fna"], "output_prefix": "safety-pass"}]
    malicious = [{"datapaths": ["/attacker/changed.fna"], "output_prefix": "unsafe"}]
    consumed: list[object] = []
    exposed: list[Path] = []

    def concurrent_rewrite_then_shared_parser() -> None:
        config_path = Path(sft.sys.argv[-1])
        exposed.append(config_path)
        try:
            config_path.write_bytes(yaml.safe_dump(malicious, sort_keys=False).encode())
        except OSError:
            pass
        consumed.append(yaml.safe_load(config_path.read_bytes()))

    class SharedPreprocessStub:
        main = staticmethod(concurrent_rewrite_then_shared_parser)

    monkeypatch.setattr(shared_data, "preprocess", SharedPreprocessStub, raising=False)

    sft._delegate_shared_preprocess(expected)

    assert consumed == [expected]
    assert not exposed[0].exists()


def test_preprocess_entry_point_is_the_safety_validating_wrapper() -> None:
    """Changing the console script back to the shared delegate would bypass manifest validation."""
    pyproject = tomllib.loads((RECIPE_ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"]["preprocess_evo2"] == "bionemo.evo2_phage_gen.sft:preprocess_main"
    assert (
        pyproject["project"]["scripts"]["evo2_phage_prepare_sft_host_evidence"]
        == "bionemo.evo2_phage_gen.sft:host_evidence_main"
    )


def test_malformed_audit_cli_is_indeterminate_not_biological_fail(capsys) -> None:
    """Usage errors must return three so exit two remains reserved for a trusted biological FAIL."""
    assert sft.main(["audit-and-filter"]) == 3
    assert "error" in capsys.readouterr().err


def test_below_minimum_audit_is_published_but_preprocessing_remains_blocked(tmp_path: Path) -> None:
    """A valid two-genome corpus remains blocked when the minimum is three and no user authorized it."""
    request, _source = _audit_request(tmp_path)
    request = replace(request, minimum_genomes=3, preferred_genomes=4)
    runtime = _fake_task4_runtime()

    result = sft.audit_and_filter(request, runtime=runtime)
    blocked = sft.validate_safety_manifest(
        request.safety_manifest,
        task4_manifest_validator=runtime.task4_manifest_validator,
        require_ready=False,
        expected_minimum_genomes=request.minimum_genomes,
        expected_preferred_genomes=request.preferred_genomes,
        expected_source_identity=runtime.expected_source_identity,
        expected_phix_identity=runtime.expected_phix_identity,
    )

    assert result.exit_code == 3
    assert result.readiness == "BLOCKED"
    assert blocked["adequacy"]["state"] == "BLOCKED_BELOW_MINIMUM"
    with pytest.raises(sft.SFTSafetyError, match="blocked without user permission"):
        sft.validate_safety_manifest(
            request.safety_manifest,
            task4_manifest_validator=runtime.task4_manifest_validator,
            require_ready=True,
            expected_minimum_genomes=request.minimum_genomes,
            expected_preferred_genomes=request.preferred_genomes,
            expected_source_identity=runtime.expected_source_identity,
            expected_phix_identity=runtime.expected_phix_identity,
        )


def test_count_authorization_cannot_waive_phix_or_child_manifest_gates(tmp_path: Path) -> None:
    """The corpus-count-only scope must remain powerless over reference and sequence safety."""
    request, _source = _audit_request(tmp_path)
    authorization = sft.CorpusCountAuthorization(
        kind=sft.CorpusAuthorizationKind.UPFRONT_COMPLETION_MANDATE,
        verbatim_statement="Just do whatever it takes to get the run done.",
        authorized_at=datetime(2026, 8, 8, tzinfo=timezone.utc),
        scope="corpus_count_only",
        minimum_accepted_count=None,
    )

    with pytest.raises(sft.SFTSafetyError, match="PhiX") as error:
        sft.audit_and_filter(
            replace(request, authorization=authorization), runtime=_fake_task4_runtime(phix_state="FAIL")
        )
    assert error.value.exit_code == 2
    assert not request.safety_manifest.exists()


def test_audit_and_filter_cli_records_upfront_count_authorization(tmp_path: Path) -> None:
    """The installed download command must expose a real audited path without weakening the 5k default."""
    request, _source = _audit_request(tmp_path)
    phix_table = request.phix_host_evidence_table
    authorization = sft.CorpusCountAuthorization(
        kind=sft.CorpusAuthorizationKind.UPFRONT_COMPLETION_MANDATE,
        verbatim_statement="Just do whatever it takes to get the run done.",
        authorized_at=datetime(2026, 8, 8, tzinfo=timezone.utc),
        scope="corpus_count_only",
        minimum_accepted_count=None,
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization.to_dict()))
    runtime = _fake_task4_runtime()

    exit_code = sft.main(
        [
            "audit-and-filter",
            "--source-fasta",
            str(request.source_fasta),
            "--host-evidence-table",
            str(request.host_evidence_table),
            "--phix-fasta",
            str(request.phix_fasta),
            "--phix-host-evidence-table",
            str(phix_table),
            "--curated-output",
            str(request.curated_output),
            "--safety-manifest",
            str(request.safety_manifest),
            "--audit-root",
            str(request.audit_root),
            "--preprocess-config",
            str(request.preprocess_config),
            "--policy",
            str(request.policy),
            "--asset-manifest",
            str(request.asset_manifest),
            "--diamond-tool-pin",
            str(request.diamond_tool_pin),
            "--mmseqs-tool-pin",
            str(request.mmseqs_tool_pin),
            "--authorization-json",
            str(authorization_path),
        ],
        runtime=runtime,
    )

    assert exit_code == 0
    validated = sft.validate_safety_manifest(
        request.safety_manifest,
        task4_manifest_validator=runtime.task4_manifest_validator,
        expected_source_identity=runtime.expected_source_identity,
        expected_phix_identity=runtime.expected_phix_identity,
    )
    assert validated["adequacy"]["state"] == "AUTHORIZED_BELOW_MINIMUM"
    assert validated["authorization"] == authorization.to_dict()


def test_explicit_count_override_applies_its_authorized_floor() -> None:
    """An explicit 3k override must not silently authorize a smaller corpus."""
    authorization = sft.CorpusCountAuthorization(
        kind=sft.CorpusAuthorizationKind.EXPLICIT_COUNT_OVERRIDE,
        verbatim_statement="You may proceed with at least 3,000 distinct genomes.",
        authorized_at=datetime(2026, 8, 8, tzinfo=timezone.utc),
        scope="corpus_count_only",
        minimum_accepted_count=3000,
    )
    assert (
        sft.assess_corpus_adequacy(2999, authorization=authorization).state is sft.CorpusAdequacy.BLOCKED_BELOW_MINIMUM
    )
    assert (
        sft.assess_corpus_adequacy(3000, authorization=authorization).state
        is sft.CorpusAdequacy.AUTHORIZED_BELOW_MINIMUM
    )
