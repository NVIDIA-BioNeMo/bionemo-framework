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

"""Resolve and record replication-host evidence for phage sequences."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bionemo.evo2_phage_gen.design_scope import HostDomain, HostEvidence


HOST_EVIDENCE_SCHEMA_VERSION = 1
HOST_EVIDENCE_TABLE_TYPE = "replication_host_evidence"
NCBI_RESOLVER_VERSION = "ncbi-datasets-virus-report"
_ACCESSION_RE = re.compile(r"(?:^|[>|\s])((?:[A-Z]{2}_[0-9]+|[A-Z]{1,4}[0-9]{5,9})\.[1-9][0-9]*)(?:[|\s]|$)")
_DOMAIN_TAXA = {
    2: (HostDomain.BACTERIA, "NCBI_STRUCTURED_BACTERIAL_HOST"),
    2157: (HostDomain.ARCHAEA, "NCBI_STRUCTURED_ARCHAEAL_HOST"),
    2759: (HostDomain.EUKARYOTA, "NCBI_STRUCTURED_EUKARYOTIC_HOST"),
}


class HostEvidenceError(ValueError):
    """Report invalid or incomplete host-evidence data."""

    pass


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise HostEvidenceError("timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise HostEvidenceError("timestamp must be text")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise HostEvidenceError("timestamp is invalid") from error


def _text(value: object, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise HostEvidenceError(f"{label} must be nonempty text")
    return value


@dataclass(frozen=True)
class HostEvidenceTableRow:
    """Describe host evidence for one sequence record."""

    record_id: str
    accession: str | None
    normalized_host_domain: HostDomain
    confirmed: bool
    evidence_source: str
    evidence_id: str
    evidence_version: str
    retrieved_at: datetime
    raw_response_path: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the normalized row."""
        _text(self.record_id, "record_id")
        _text(self.accession, "accession", optional=True)
        object.__setattr__(self, "normalized_host_domain", HostDomain(self.normalized_host_domain))
        if type(self.confirmed) is not bool:
            raise HostEvidenceError("confirmed must be boolean")
        _text(self.evidence_source, "evidence_source")
        _text(self.evidence_id, "evidence_id")
        _text(self.evidence_version, "evidence_version")
        _timestamp(self.retrieved_at)
        _text(self.raw_response_path, "raw_response_path", optional=True)
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        if not self.reason_codes or any(not isinstance(reason, str) or not reason for reason in self.reason_codes):
            raise HostEvidenceError("reason_codes must contain text")

    @classmethod
    def create(cls, **values) -> "HostEvidenceTableRow":
        """Create a validated row from keyword values."""
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        """Serialize the row."""
        return {
            "record_id": self.record_id,
            "accession": self.accession,
            "normalized_host_domain": self.normalized_host_domain.value,
            "confirmed": self.confirmed,
            "evidence_source": self.evidence_source,
            "evidence_id": self.evidence_id,
            "evidence_version": self.evidence_version,
            "retrieved_at": _timestamp(self.retrieved_at),
            "raw_response_path": self.raw_response_path,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, value: object) -> "HostEvidenceTableRow":
        """Parse a row from serialized data."""
        if not isinstance(value, Mapping):
            raise HostEvidenceError("host-evidence row must be a mapping")
        reasons = value.get("reason_codes")
        if not isinstance(reasons, list):
            raise HostEvidenceError("reason_codes must be a list")
        return cls(
            record_id=_text(value.get("record_id"), "record_id"),
            accession=_text(value.get("accession"), "accession", optional=True),
            normalized_host_domain=HostDomain(value.get("normalized_host_domain")),
            confirmed=value.get("confirmed"),
            evidence_source=_text(value.get("evidence_source"), "evidence_source"),
            evidence_id=_text(value.get("evidence_id"), "evidence_id"),
            evidence_version=_text(value.get("evidence_version"), "evidence_version"),
            retrieved_at=_parse_timestamp(value.get("retrieved_at")),
            raw_response_path=_text(value.get("raw_response_path"), "raw_response_path", optional=True),
            reason_codes=tuple(reasons),
        )

    def to_task1_host_evidence(self) -> HostEvidence:
        """Convert the row to the design-scope host-evidence model."""
        return HostEvidence(
            source=self.evidence_source,
            source_version=self.evidence_version,
            replication_host_domains=frozenset({self.normalized_host_domain}),
            confirmed=self.confirmed,
            metadata={
                "record_id": self.record_id,
                "accession": self.accession,
                "evidence_id": self.evidence_id,
                "retrieved_at": _timestamp(self.retrieved_at),
                "raw_response_path": self.raw_response_path,
                "reason_codes": list(self.reason_codes),
            },
        )


@dataclass(frozen=True)
class HostEvidenceTable:
    """Collect host-evidence rows for a dataset."""

    table_id: str
    created_at: datetime
    rows: tuple[HostEvidenceTableRow, ...]

    def __post_init__(self) -> None:
        """Validate table identity and unique record IDs."""
        _text(self.table_id, "table_id")
        _timestamp(self.created_at)
        object.__setattr__(self, "rows", tuple(self.rows))
        ids = [row.record_id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise HostEvidenceError("host-evidence record IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        """Serialize the table."""
        return {
            "schema_version": HOST_EVIDENCE_SCHEMA_VERSION,
            "table_type": HOST_EVIDENCE_TABLE_TYPE,
            "table_id": self.table_id,
            "created_at": _timestamp(self.created_at),
            "rows": [row.to_dict() for row in self.rows],
        }


def write_host_evidence_table(path: str | Path, table: HostEvidenceTable) -> Path:
    """Write a host-evidence table as YAML."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(table.to_dict(), sort_keys=False))
    return destination


def load_host_evidence_table(path: str | Path) -> HostEvidenceTable:
    """Load and validate a host-evidence table."""
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, Mapping):
        raise HostEvidenceError("host-evidence table must be a mapping")
    if (
        payload.get("schema_version") != HOST_EVIDENCE_SCHEMA_VERSION
        or payload.get("table_type") != HOST_EVIDENCE_TABLE_TYPE
    ):
        raise HostEvidenceError("unsupported host-evidence table")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise HostEvidenceError("host-evidence rows must be a list")
    return HostEvidenceTable(
        table_id=_text(payload.get("table_id"), "table_id"),
        created_at=_parse_timestamp(payload.get("created_at")),
        rows=tuple(HostEvidenceTableRow.from_dict(row) for row in rows),
    )


def extract_accession(header: str) -> str | None:
    """Extract an accession.version from a FASTA header when present."""
    if not isinstance(header, str):
        raise HostEvidenceError("FASTA header must be text")
    match = _ACCESSION_RE.search(header)
    return match.group(1) if match else None


def _default_ncbi_fetcher(accession: str) -> bytes:
    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/virus/accession/{accession}/dataset_report"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    raise HostEvidenceError(f"cannot fetch NCBI metadata for {accession}") from last_error


def _parse_ncbi_response(raw: bytes, *, accession: str) -> tuple[HostDomain, bool, str, str, tuple[str, ...]]:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise HostEvidenceError("NCBI metadata is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise HostEvidenceError("NCBI metadata must be a mapping")
    reports = payload.get("reports")
    if not isinstance(reports, list) or len(reports) != 1 or not isinstance(reports[0], Mapping):
        raise HostEvidenceError("NCBI metadata must contain exactly one report")
    report = reports[0]
    if report.get("accession") != accession:
        raise HostEvidenceError("NCBI metadata accession mismatch")
    host = report.get("host")
    if not isinstance(host, Mapping):
        return (
            HostDomain.UNKNOWN,
            False,
            f"ncbi-virus-accession:{accession}:host-unresolved",
            NCBI_RESOLVER_VERSION,
            ("NCBI_STRUCTURED_HOST_UNRESOLVED",),
        )
    lineage = host.get("lineage")
    if not isinstance(lineage, list):
        raise HostEvidenceError("NCBI host lineage is missing")
    domains: set[HostDomain] = set()
    reasons: dict[HostDomain, str] = {}
    for item in lineage:
        if not isinstance(item, Mapping) or type(item.get("tax_id")) is not int:
            continue
        recognized = _DOMAIN_TAXA.get(item["tax_id"])
        if recognized:
            domain, reason = recognized
            domains.add(domain)
            reasons[domain] = reason
    evidence_id = f"ncbi-virus-accession:{accession}:host-taxon:{host.get('tax_id', 'unknown')}"
    if len(domains) > 1:
        return (
            HostDomain.UNKNOWN,
            False,
            evidence_id,
            NCBI_RESOLVER_VERSION,
            ("CONFLICTING_STRUCTURED_HOST_DOMAINS",),
        )
    if not domains:
        return (
            HostDomain.UNKNOWN,
            False,
            evidence_id,
            NCBI_RESOLVER_VERSION,
            ("NCBI_STRUCTURED_HOST_UNRESOLVED",),
        )
    domain = next(iter(domains))
    return domain, True, evidence_id, NCBI_RESOLVER_VERSION, (reasons[domain],)


def _cache_response(cache_dir: Path, accession: str, raw: bytes) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{accession}.json"
    path.write_bytes(raw)
    return path.resolve()


def resolve_ncbi_host_evidence(
    *,
    record_id: str,
    header: str,
    cache_dir: str | Path,
    fetcher: Callable[[str], bytes] = _default_ncbi_fetcher,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> HostEvidenceTableRow:
    """Resolve bacterial-host evidence for one NCBI accession."""
    accession = extract_accession(header)
    retrieved_at = clock()
    if accession is None:
        return HostEvidenceTableRow(
            record_id,
            None,
            HostDomain.UNKNOWN,
            False,
            "NCBI_DATASETS",
            f"unresolved-header:{record_id}",
            NCBI_RESOLVER_VERSION,
            retrieved_at,
            None,
            ("MISSING_NCBI_ACCESSION",),
        )
    raw = fetcher(accession)
    if not isinstance(raw, bytes):
        raise HostEvidenceError("NCBI fetcher must return bytes")
    cache_path = _cache_response(Path(cache_dir), accession, raw)
    try:
        domain, confirmed, evidence_id, version, reasons = _parse_ncbi_response(raw, accession=accession)
    except HostEvidenceError:
        domain, confirmed = HostDomain.UNKNOWN, False
        evidence_id, version, reasons = (
            f"accession:{accession}",
            NCBI_RESOLVER_VERSION,
            ("NCBI_METADATA_UNRESOLVED",),
        )
    return HostEvidenceTableRow(
        record_id,
        accession,
        domain,
        confirmed,
        "NCBI_DATASETS",
        evidence_id,
        version,
        retrieved_at,
        str(cache_path),
        reasons,
    )


def validate_host_evidence_artifacts(table: HostEvidenceTable, *, table_path: str | Path) -> None:
    """Check that evidence artifacts referenced by a table are readable."""
    base = Path(table_path).resolve().parent
    for row in table.rows:
        if row.evidence_source != "NCBI_DATASETS":
            continue
        if row.accession is None:
            if row.raw_response_path is not None:
                raise HostEvidenceError(f"{row.record_id} has a response without an accession")
            continue
        if row.raw_response_path is None:
            raise HostEvidenceError(f"{row.record_id} lacks its recorded NCBI response")
        path = Path(row.raw_response_path)
        if not path.is_absolute():
            path = base / path
        raw = path.read_bytes()
        try:
            observed = _parse_ncbi_response(raw, accession=row.accession)
        except HostEvidenceError:
            observed = (
                HostDomain.UNKNOWN,
                False,
                f"accession:{row.accession}",
                NCBI_RESOLVER_VERSION,
                ("NCBI_METADATA_UNRESOLVED",),
            )
        expected = (
            row.normalized_host_domain,
            row.confirmed,
            row.evidence_id,
            row.evidence_version,
            row.reason_codes,
        )
        if observed != expected:
            raise HostEvidenceError(f"recorded NCBI response no longer agrees with {row.record_id}")
