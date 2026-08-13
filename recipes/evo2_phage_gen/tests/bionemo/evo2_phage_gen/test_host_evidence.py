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

"""Tests for strict, versioned SFT replication-host evidence."""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bionemo.evo2_phage_gen import host_evidence as host_evidence_module
from bionemo.evo2_phage_gen.design_scope import HostDomain, evaluate_host_evidence
from bionemo.evo2_phage_gen.host_evidence import (
    HostEvidenceError,
    HostEvidenceTable,
    HostEvidenceTableRow,
    _default_ncbi_fetcher,
    extract_accession,
    load_host_evidence_table,
    resolve_ncbi_host_evidence,
    validate_host_evidence_artifacts,
    write_host_evidence_table,
)


NOW = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)


def _ncbi_dataset_report(
    accession: str = "NC_001422.1",
    *,
    host_tax_id: int = 561,
    host_lineage: list[dict[str, object]] | None = None,
    include_host: bool = True,
) -> bytes:
    """Mirror the complete field shape returned by the pinned v2alpha virus report endpoint."""
    if host_lineage is None:
        host_lineage = [
            {"tax_id": 131567, "name": "cellular organisms"},
            {"tax_id": 2, "name": "Bacteria"},
            {"tax_id": 1224, "name": "Pseudomonadota"},
            {"tax_id": 561, "name": "Escherichia"},
        ]
    report: dict[str, object] = {
        "accession": accession,
        "is_annotated": True,
        "source_database": "RefSeq",
        "protein_count": 11,
        "virus": {
            "tax_id": 2886930,
            "organism_name": "Escherichia phage phiX174",
            "lineage": [
                {"tax_id": 10239, "name": "Viruses"},
                {"tax_id": 10841, "name": "Microviricetes"},
                {"tax_id": 2886930, "name": "Escherichia phage phiX174"},
            ],
        },
        "bioprojects": ["PRJNA485481"],
        "update_date": "2023-01-11T00:00:00Z",
        "release_date": "1993-04-28T00:00:00Z",
        "completeness": "COMPLETE",
        "length": 5386,
        "gene_count": 11,
        "nucleotide": {"sequence_hash": "2FAF564E"},
        "submitter": {
            "names": ["National Center for Biotechnology Information"],
            "affiliation": "Bacteria research text must not determine host domain",
            "country": "USA",
        },
    }
    if include_host:
        report["host"] = {
            "tax_id": host_tax_id,
            "organism_name": "Escherichia",
            "lineage": host_lineage,
        }
    return json.dumps({"reports": [report], "total_count": 1}, separators=(",", ":")).encode()


def _copy_row(row: HostEvidenceTableRow, **changes: object) -> HostEvidenceTableRow:
    values = {
        "record_id": row.record_id,
        "accession": row.accession,
        "normalized_host_domain": row.normalized_host_domain,
        "confirmed": row.confirmed,
        "evidence_source": row.evidence_source,
        "evidence_id": row.evidence_id,
        "evidence_version": row.evidence_version,
        "retrieved_at": row.retrieved_at,
        "raw_response_path": row.raw_response_path,
        "raw_response_sha256": row.raw_response_sha256,
        "reason_codes": row.reason_codes,
    }
    values.update(changes)
    return HostEvidenceTableRow.create(**values)


def _row(
    record_id: str,
    domain: HostDomain,
    *,
    accession: str | None = None,
    source: str = "reviewed_catalog",
    confirmed: bool = True,
) -> HostEvidenceTableRow:
    return HostEvidenceTableRow.create(
        record_id=record_id,
        accession=accession,
        normalized_host_domain=domain,
        confirmed=confirmed,
        evidence_source=source,
        evidence_id=f"{source}:{record_id}",
        evidence_version="2026-08-08",
        retrieved_at=NOW,
        raw_response_path=f"cache/{record_id}.json",
        raw_response_sha256="1" * 64,
        reason_codes=("REVIEWED_REPLICATION_HOST",),
    )


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (">NC_001422.1 Enterobacteria phage phiX174", "NC_001422.1"),
        (">ref|NC_001422.1| phiX174", "NC_001422.1"),
        (">OQ123456.1 cultured phage isolate", "OQ123456.1"),
        (">IMGVR_UViG_1 no RefSeq accession", None),
    ],
)
def test_extract_accession_uses_accession_bearing_header_tokens(header: str, expected: str | None) -> None:
    """Removing accession recognition must prevent NCBI-backed evidence resolution."""
    assert extract_accession(header) == expected


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.payload


def test_default_ncbi_fetcher_retries_transient_errors_with_bounded_backoff(monkeypatch) -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    def urlopen(request, *, timeout):
        attempts.append(request.full_url)
        assert timeout == 60
        if len(attempts) < 3:
            raise urllib.error.URLError("temporary failure")
        return _Response(b"resolved")

    monkeypatch.setattr(host_evidence_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", sleeps.append)

    assert _default_ncbi_fetcher("NC_001422.1") == b"resolved"
    assert len(attempts) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize(
    "error",
    [urllib.error.URLError("offline"), TimeoutError("timed out")],
    ids=["url-error", "timeout"],
)
def test_default_ncbi_fetcher_wraps_exhausted_transport_errors(monkeypatch, error) -> None:
    attempts = 0

    def urlopen(_request, *, timeout):
        nonlocal attempts
        attempts += 1
        assert timeout == 60
        raise error

    monkeypatch.setattr(host_evidence_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _delay: None)

    with pytest.raises(HostEvidenceError, match="cannot fetch NCBI metadata for NC_001422.1"):
        _default_ncbi_fetcher("NC_001422.1")
    assert attempts == 3


def test_ncbi_resolution_caches_exact_raw_response_and_produces_versioned_evidence(tmp_path: Path) -> None:
    """Dropping raw-response caching or source versions must make resolved evidence untrustworthy."""
    raw = _ncbi_dataset_report()
    calls: list[str] = []

    def fetch(accession: str) -> bytes:
        calls.append(accession)
        return raw

    row = resolve_ncbi_host_evidence(
        record_id="phiX",
        header=">ref|NC_001422.1| phiX174",
        cache_dir=tmp_path / "cache",
        fetcher=fetch,
        clock=lambda: NOW,
    )

    assert calls == ["NC_001422.1"]
    assert row.accession == "NC_001422.1"
    assert row.normalized_host_domain is HostDomain.BACTERIA
    assert row.confirmed is True
    assert row.evidence_source == "NCBI_DATASETS"
    assert row.evidence_id == "ncbi-virus-accession:NC_001422.1:host-taxon:561"
    assert row.evidence_version == "ncbi-datasets-v2alpha-virus-report-host-v2"
    assert row.reason_codes == ("NCBI_STRUCTURED_BACTERIAL_HOST",)
    assert (tmp_path / row.raw_response_path).read_bytes() == raw
    assert evaluate_host_evidence(row.to_task1_host_evidence()).allowed is True


def test_ncbi_resolution_caches_malformed_or_conflicting_evidence_before_rejecting_it(tmp_path: Path) -> None:
    """Parser failure or prokaryote/eukaryote conflict must never lose evidence or become eligible."""
    conflicting = _ncbi_dataset_report(
        "NC_9.1",
        host_tax_id=999,
        host_lineage=[
            {"tax_id": 131567, "name": "cellular organisms"},
            {"tax_id": 2, "name": "Bacteria"},
            {"tax_id": 2759, "name": "Eukaryota"},
        ],
    )
    row = resolve_ncbi_host_evidence(
        record_id="conflict",
        header=">NC_9.1 conflict",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: conflicting,
        clock=lambda: NOW,
    )

    assert (tmp_path / row.raw_response_path).read_bytes() == conflicting
    assert row.normalized_host_domain is HostDomain.UNKNOWN
    assert row.confirmed is False
    decision = evaluate_host_evidence(row.to_task1_host_evidence())
    assert decision.allowed is False
    assert row.reason_codes == ("CONFLICTING_STRUCTURED_HOST_DOMAINS",)

    malformed = b"not-json"
    malformed_row = resolve_ncbi_host_evidence(
        record_id="malformed",
        header=">NC_10.1 malformed",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: malformed,
        clock=lambda: NOW,
    )
    assert (tmp_path / malformed_row.raw_response_path).read_bytes() == malformed
    assert malformed_row.confirmed is False
    assert malformed_row.reason_codes == ("NCBI_METADATA_UNRESOLVED",)


def test_ncbi_resolution_never_infers_host_domain_from_arbitrary_free_text(tmp_path: Path) -> None:
    """Virus names and affiliations mentioning bacteria cannot substitute for structured host taxonomy."""
    row = resolve_ncbi_host_evidence(
        record_id="no-structured-host",
        header=">NC_001422.1 free text says Escherichia phage",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: _ncbi_dataset_report(include_host=False),
        clock=lambda: NOW,
    )

    assert row.normalized_host_domain is HostDomain.UNKNOWN
    assert row.confirmed is False
    assert row.reason_codes == ("NCBI_STRUCTURED_HOST_UNRESOLVED",)
    assert evaluate_host_evidence(row.to_task1_host_evidence()).allowed is False


def test_versioned_multi_source_table_round_trips_and_retains_domain_semantics(tmp_path: Path) -> None:
    """Collapsing Archaea or mixed prokaryotic evidence into unknown would wrongly quarantine safe hosts."""
    table = HostEvidenceTable(
        table_id="microviridae-hosts-2026-08-08",
        created_at=NOW,
        rows=(
            _row("b", HostDomain.BACTERIA, accession="NC_1.1", source="NCBI_DATASETS"),
            _row("a", HostDomain.ARCHAEA, source="IMGVR"),
            _row("ba", HostDomain.BACTERIA_AND_ARCHAEA, source="PHAGESCOPE"),
            _row("e", HostDomain.EUKARYOTA),
            _row("u", HostDomain.UNKNOWN, confirmed=False),
        ),
    )
    path = tmp_path / "HOST_EVIDENCE.yaml"

    write_host_evidence_table(path, table)
    loaded = load_host_evidence_table(path)

    assert loaded == table
    assert [row.record_id for row in loaded.rows] == ["b", "a", "ba", "e", "u"]
    assert [evaluate_host_evidence(row.to_task1_host_evidence()).allowed for row in loaded.rows] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert {row.evidence_source for row in loaded.rows} == {
        "NCBI_DATASETS",
        "IMGVR",
        "PHAGESCOPE",
        "reviewed_catalog",
    }


def test_table_loader_rejects_duplicate_keys_rows_and_evidence_digest_drift(tmp_path: Path) -> None:
    """Ambiguous YAML and self-inconsistent evidence must fail rather than silently choose a value."""
    path = tmp_path / "HOST_EVIDENCE.yaml"
    path.write_text(
        "schema_version: 1\n"
        "table_type: replication_host_evidence\n"
        "table_id: one\n"
        "table_id: shadow\n"
        "created_at: '2026-08-08T12:00:00Z'\n"
        "rows: []\n"
    )
    with pytest.raises(HostEvidenceError, match="duplicate key"):
        load_host_evidence_table(path)

    table = HostEvidenceTable(table_id="one", created_at=NOW, rows=(_row("dup", HostDomain.BACTERIA),))
    write_host_evidence_table(path, table)
    payload = path.read_text().replace("evidence_digest:", "evidence_digest: deadbeef #")
    path.write_text(payload)
    with pytest.raises(HostEvidenceError, match="evidence digest"):
        load_host_evidence_table(path)

    with pytest.raises(HostEvidenceError, match="duplicate record_id"):
        HostEvidenceTable(table_id="duplicates", created_at=NOW, rows=(table.rows[0], table.rows[0]))


def test_raw_ncbi_evidence_is_revalidated_from_the_table_boundary(tmp_path: Path) -> None:
    """Changing cached metadata after table creation must invalidate its positive host evidence."""
    raw = _ncbi_dataset_report("OQ123456.1")
    row = resolve_ncbi_host_evidence(
        record_id="phage",
        header=">OQ123456.1 phage",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: raw,
        clock=lambda: NOW,
    )
    table_path = tmp_path / "HOST_EVIDENCE.yaml"
    table = HostEvidenceTable(table_id="ncbi-v1", created_at=NOW, rows=(row,))
    write_host_evidence_table(table_path, table)

    validate_host_evidence_artifacts(table, table_path=table_path)
    Path(row.raw_response_path).write_bytes(b"tampered")

    with pytest.raises(HostEvidenceError, match="raw response digest"):
        validate_host_evidence_artifacts(table, table_path=table_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accession", "OQ999999.1"),
        ("normalized_host_domain", HostDomain.EUKARYOTA),
        ("confirmed", False),
        ("evidence_id", "ncbi-virus-accession:OQ123456.1:host-taxon:999"),
        ("evidence_version", "unreviewed-parser-v999"),
        ("reason_codes", ("NCBI_STRUCTURED_EUKARYOTIC_HOST",)),
    ],
)
def test_cached_ncbi_response_must_exactly_reconcile_with_every_derived_row_field(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """A self-consistent row digest must not let claims diverge from its cached primary response."""
    row = resolve_ncbi_host_evidence(
        record_id="phage",
        header=">OQ123456.1 phage",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: _ncbi_dataset_report("OQ123456.1"),
        clock=lambda: NOW,
    )
    forged = _copy_row(row, **{field: value})
    table_path = tmp_path / "HOST_EVIDENCE.yaml"
    table = HostEvidenceTable(table_id="ncbi-v1", created_at=NOW, rows=(forged,))
    write_host_evidence_table(table_path, table)

    with pytest.raises(HostEvidenceError, match="does not reconcile"):
        validate_host_evidence_artifacts(table, table_path=table_path)


def test_host_evidence_reads_reject_symlinked_parent_components(tmp_path: Path) -> None:
    """Checking only the leaf would let a trusted table or raw response traverse a swapped parent."""
    real = tmp_path / "real"
    real.mkdir()
    row = resolve_ncbi_host_evidence(
        record_id="phage",
        header=">OQ123456.1 phage",
        cache_dir=real / "cache",
        fetcher=lambda _accession: _ncbi_dataset_report("OQ123456.1"),
        clock=lambda: NOW,
    )
    table_path = real / "HOST_EVIDENCE.yaml"
    write_host_evidence_table(table_path, HostEvidenceTable(table_id="ncbi-v1", created_at=NOW, rows=(row,)))
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(HostEvidenceError, match="symlink component"):
        load_host_evidence_table(alias / table_path.name)

    aliased_row = _copy_row(row, raw_response_path=str(alias / "cache" / Path(row.raw_response_path).name))
    with pytest.raises(HostEvidenceError, match="symlink component"):
        validate_host_evidence_artifacts(
            HostEvidenceTable(table_id="aliased", created_at=NOW, rows=(aliased_row,)),
            table_path=table_path,
        )


def test_host_evidence_read_binds_each_parent_descriptor_against_replacement(tmp_path: Path, monkeypatch) -> None:
    """A checked parent swapped to a symlink before final open must not redirect host-evidence bytes."""
    trusted_parent = tmp_path / "trusted"
    trusted_parent.mkdir()
    source = trusted_parent / "response.json"
    source.write_bytes(b"trusted-bytes")
    saved_parent = tmp_path / "trusted-saved"
    malicious_parent = tmp_path / "malicious"
    malicious_parent.mkdir()
    (malicious_parent / source.name).write_bytes(b"malicious-bytes")
    real_open = host_evidence_module.os.open
    swapped = False

    def swapping_open(path, flags, *args, dir_fd=None, **kwargs):
        nonlocal swapped
        path_text = host_evidence_module.os.fspath(path)
        opens_full_path = dir_fd is None and Path(path_text) == source
        opens_bound_final = dir_fd is not None and path_text == source.name
        if not swapped and (opens_full_path or opens_bound_final):
            trusted_parent.rename(saved_parent)
            trusted_parent.symlink_to(malicious_parent, target_is_directory=True)
            swapped = True
        return real_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(host_evidence_module.os, "open", swapping_open)

    assert host_evidence_module._read_regular_file_bytes(source, label="race fixture") == b"trusted-bytes"
    assert swapped is True
