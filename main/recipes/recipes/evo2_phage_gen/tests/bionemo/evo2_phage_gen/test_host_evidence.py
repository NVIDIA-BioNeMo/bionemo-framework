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

import json
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bionemo.evo2_phage_gen import host_evidence
from bionemo.evo2_phage_gen.design_scope import HostDomain, evaluate_host_evidence
from bionemo.evo2_phage_gen.host_evidence import (
    HostEvidenceError,
    HostEvidenceTable,
    HostEvidenceTableRow,
    extract_accession,
    load_host_evidence_table,
    resolve_ncbi_host_evidence,
    validate_host_evidence_artifacts,
    write_host_evidence_table,
)


NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)


def _report(
    accession: str = "NC_001422.1",
    *,
    lineage: list[dict[str, object]] | None = None,
    include_host: bool = True,
) -> bytes:
    lineage = lineage or [
        {"tax_id": 131567, "name": "cellular organisms"},
        {"tax_id": 2, "name": "Bacteria"},
        {"tax_id": 561, "name": "Escherichia"},
    ]
    report: dict[str, object] = {
        "accession": accession,
        "length": 5386,
        "new_provider_field": {"may": "appear in future releases"},
    }
    if include_host:
        report["host"] = {"tax_id": 561, "lineage": lineage}
    return json.dumps({"reports": [report], "total_count": 1}).encode()


@pytest.mark.parametrize(
    ("header", "expected"),
    (
        (">NC_001422.1 phiX174", "NC_001422.1"),
        (">ref|NC_001422.1| phiX174", "NC_001422.1"),
        (">IMGVR_UViG_1", None),
    ),
)
def test_extract_accession(header: str, expected: str | None) -> None:
    assert extract_accession(header) == expected


def test_ncbi_resolution_records_observed_source_and_domain(tmp_path: Path) -> None:
    row = resolve_ncbi_host_evidence(
        record_id="phix",
        header=">NC_001422.1 phiX174",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: _report(),
        clock=lambda: NOW,
    )
    assert row.normalized_host_domain is HostDomain.BACTERIA
    assert row.confirmed is True
    assert row.accession == "NC_001422.1"
    assert Path(row.raw_response_path).read_bytes() == _report()
    assert evaluate_host_evidence(row.to_task1_host_evidence()).allowed is True


def test_conflicting_host_domains_are_not_eligible(tmp_path: Path) -> None:
    raw = _report(
        lineage=[
            {"tax_id": 2, "name": "Bacteria"},
            {"tax_id": 2759, "name": "Eukaryota"},
        ]
    )
    row = resolve_ncbi_host_evidence(
        record_id="conflict",
        header=">NC_001422.1",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: raw,
        clock=lambda: NOW,
    )
    assert row.normalized_host_domain is HostDomain.UNKNOWN
    assert row.confirmed is False
    assert row.reason_codes == ("CONFLICTING_STRUCTURED_HOST_DOMAINS",)


def test_free_text_does_not_substitute_for_structured_host_data(tmp_path: Path) -> None:
    row = resolve_ncbi_host_evidence(
        record_id="unknown",
        header=">NC_001422.1 Escherichia phage",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: _report(include_host=False),
        clock=lambda: NOW,
    )
    assert row.normalized_host_domain is HostDomain.UNKNOWN
    assert row.confirmed is False


def test_malformed_provider_response_is_recorded_as_unresolved(tmp_path: Path) -> None:
    row = resolve_ncbi_host_evidence(
        record_id="malformed",
        header=">NC_001422.1",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: b"not-json",
        clock=lambda: NOW,
    )
    assert row.reason_codes == ("NCBI_METADATA_UNRESOLVED",)
    assert Path(row.raw_response_path).read_bytes() == b"not-json"


def test_table_round_trip_preserves_scientific_fields(tmp_path: Path) -> None:
    row = resolve_ncbi_host_evidence(
        record_id="phix",
        header=">NC_001422.1",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: _report(),
        clock=lambda: NOW,
    )
    table = HostEvidenceTable("microviridae-hosts", NOW, (row,))
    path = write_host_evidence_table(tmp_path / "HOST_EVIDENCE.yaml", table)
    loaded = load_host_evidence_table(path)
    assert loaded == table
    validate_host_evidence_artifacts(loaded, table_path=path)


def test_recorded_response_must_still_support_recorded_domain(tmp_path: Path) -> None:
    row = resolve_ncbi_host_evidence(
        record_id="phix",
        header=">NC_001422.1",
        cache_dir=tmp_path / "cache",
        fetcher=lambda _accession: _report(),
        clock=lambda: NOW,
    )
    table = HostEvidenceTable("hosts", NOW, (row,))
    path = write_host_evidence_table(tmp_path / "HOST_EVIDENCE.yaml", table)
    Path(row.raw_response_path).write_bytes(_report(include_host=False))
    with pytest.raises(HostEvidenceError, match="no longer agrees"):
        validate_host_evidence_artifacts(table, table_path=path)


def test_non_ncbi_rows_do_not_require_raw_api_responses(tmp_path: Path) -> None:
    row = HostEvidenceTableRow(
        "curated",
        None,
        HostDomain.ARCHAEA,
        True,
        "reviewed_catalog",
        "catalog:curated",
        "2026-08",
        NOW,
        None,
        ("REVIEWED_REPLICATION_HOST",),
    )
    table = HostEvidenceTable("hosts", NOW, (row,))
    validate_host_evidence_artifacts(table, table_path=tmp_path / "HOST_EVIDENCE.yaml")


def test_ncbi_fetcher_retries_transient_errors(monkeypatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"ok"

    def urlopen(_request, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError("temporary")
        return Response()

    monkeypatch.setattr(host_evidence.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(host_evidence.time, "sleep", sleeps.append)
    assert host_evidence._default_ncbi_fetcher("NC_001422.1") == b"ok"
    assert attempts == 3
    assert sleeps == [1, 2]
