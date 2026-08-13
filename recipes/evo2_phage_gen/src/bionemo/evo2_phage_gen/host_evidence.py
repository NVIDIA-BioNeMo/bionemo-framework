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

"""Strict replication-host evidence used to gate phage SFT corpora."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from bionemo.evo2_phage_gen.design_scope import HostDomain, HostEvidence


HOST_EVIDENCE_SCHEMA_VERSION = 1
HOST_EVIDENCE_TABLE_TYPE = "replication_host_evidence"
NCBI_RESOLVER_VERSION = "ncbi-datasets-v2alpha-virus-report-host-v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ACCESSION_RE = re.compile(r"(?<![A-Za-z0-9_])((?:[A-Z]{2}_[0-9]+|[A-Z]{1,4}[0-9]{5,9})(?:\.[0-9]+)?)(?![A-Za-z0-9_])")


class HostEvidenceError(ValueError):
    """Host evidence is missing, ambiguous, stale, or structurally invalid."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise HostEvidenceError(f"duplicate key in host-evidence table: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _strict_mapping(value: object, *, label: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys or not all(isinstance(key, str) for key in value):
        raise HostEvidenceError(f"{label} keys must be exactly {sorted(keys)}")
    return value


def _canonical_sha256(value: Mapping[str, object]) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise HostEvidenceError(f"host evidence is not canonical JSON: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise HostEvidenceError("retrieval timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise HostEvidenceError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00"  # noqa: FURB162  # Python 3.10 datetime.fromisoformat does not accept Z
        )
    except ValueError as error:
        raise HostEvidenceError(f"{label} must be an RFC3339 UTC timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise HostEvidenceError(f"{label} must be UTC")
    return parsed


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HostEvidenceError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label=label)


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise HostEvidenceError(f"{label} path contains a symlink component")
        except FileNotFoundError:
            return
        except OSError as error:
            raise HostEvidenceError(f"cannot inspect {label} path: {error}") from error


def _read_regular_file_bytes(path: str | Path, *, label: str) -> bytes:
    """Read one stable regular-file snapshot without following symlink path components."""
    source = Path(os.path.abspath(os.fspath(path)))
    parts = source.parts
    if len(parts) < 2:
        raise HostEvidenceError(f"{label} must identify a regular file")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_descriptor = os.open(source.anchor, directory_flags)
    except OSError as error:
        raise HostEvidenceError(f"cannot open {label} root directory: {error}") from error
    try:
        for part in parts[1:-1]:
            try:
                next_descriptor = os.open(part, directory_flags, dir_fd=parent_descriptor)
            except OSError as error:
                raise HostEvidenceError(
                    f"{label} path contains an unreadable or symlink component: {error}"
                ) from error
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise HostEvidenceError(f"{label} path contains an unreadable or symlink component: {error}") from error
    finally:
        os.close(parent_descriptor)
    chunks: list[bytes] = []
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise HostEvidenceError(f"{label} must be a regular file")
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
    except OSError as error:
        raise HostEvidenceError(f"cannot read {label}: {error}") from error
    finally:
        os.close(descriptor)
    return b"".join(chunks)


@dataclass(frozen=True)
class HostEvidenceTableRow:
    """One immutable, source-versioned replication-host attribution."""

    record_id: str
    accession: str | None
    normalized_host_domain: HostDomain
    confirmed: bool
    evidence_source: str
    evidence_id: str
    evidence_version: str
    retrieved_at: datetime
    raw_response_path: str | None
    raw_response_sha256: str | None
    reason_codes: tuple[str, ...]
    evidence_digest: str

    def __post_init__(self) -> None:
        """Validate every provenance field and its canonical evidence digest."""
        _require_text(self.record_id, label="record_id")
        _optional_text(self.accession, label="accession")
        try:
            object.__setattr__(self, "normalized_host_domain", HostDomain(self.normalized_host_domain))
        except (TypeError, ValueError) as error:
            raise HostEvidenceError("normalized_host_domain is unsupported") from error
        if type(self.confirmed) is not bool:
            raise HostEvidenceError("confirmed must be a boolean")
        _require_text(self.evidence_source, label="evidence_source")
        _require_text(self.evidence_id, label="evidence_id")
        _require_text(self.evidence_version, label="evidence_version")
        _timestamp(self.retrieved_at)
        if (self.raw_response_path is None) != (self.raw_response_sha256 is None):
            raise HostEvidenceError("raw response path and digest must both be present or absent")
        _optional_text(self.raw_response_path, label="raw_response_path")
        if self.raw_response_sha256 is not None and not _SHA256_RE.fullmatch(self.raw_response_sha256):
            raise HostEvidenceError("raw_response_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.reason_codes, tuple):
            object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        if not self.reason_codes or any(not isinstance(reason, str) or not reason for reason in self.reason_codes):
            raise HostEvidenceError("reason_codes must contain non-empty strings")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise HostEvidenceError("reason_codes must be unique")
        if not _SHA256_RE.fullmatch(self.evidence_digest):
            raise HostEvidenceError("evidence digest must be a lowercase SHA-256 digest")
        if self.evidence_digest != _canonical_sha256(self._unsigned_dict()):
            raise HostEvidenceError("evidence digest does not match the row")

    @classmethod
    def create(
        cls,
        *,
        record_id: str,
        accession: str | None,
        normalized_host_domain: HostDomain,
        confirmed: bool,
        evidence_source: str,
        evidence_id: str,
        evidence_version: str,
        retrieved_at: datetime,
        raw_response_path: str | None,
        raw_response_sha256: str | None,
        reason_codes: Sequence[str],
    ) -> HostEvidenceTableRow:
        """Create a row whose digest binds every provenance and decision field."""
        values = {
            "record_id": record_id,
            "accession": accession,
            "normalized_host_domain": HostDomain(normalized_host_domain).value,
            "confirmed": confirmed,
            "evidence_source": evidence_source,
            "evidence_id": evidence_id,
            "evidence_version": evidence_version,
            "retrieved_at": _timestamp(retrieved_at),
            "raw_response_path": raw_response_path,
            "raw_response_sha256": raw_response_sha256,
            "reason_codes": list(reason_codes),
        }
        return cls(
            record_id=record_id,
            accession=accession,
            normalized_host_domain=HostDomain(normalized_host_domain),
            confirmed=confirmed,
            evidence_source=evidence_source,
            evidence_id=evidence_id,
            evidence_version=evidence_version,
            retrieved_at=retrieved_at,
            raw_response_path=raw_response_path,
            raw_response_sha256=raw_response_sha256,
            reason_codes=tuple(reason_codes),
            evidence_digest=_canonical_sha256(values),
        )

    def _unsigned_dict(self) -> dict[str, object]:
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
            "raw_response_sha256": self.raw_response_sha256,
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize the row using JSON/YAML-stable primitives."""
        return {**self._unsigned_dict(), "evidence_digest": self.evidence_digest}

    @classmethod
    def from_dict(cls, value: object) -> HostEvidenceTableRow:
        """Parse one exact-schema row and revalidate its evidence digest."""
        strict = _strict_mapping(
            value,
            label="host-evidence row",
            keys=frozenset(
                {
                    "record_id",
                    "accession",
                    "normalized_host_domain",
                    "confirmed",
                    "evidence_source",
                    "evidence_id",
                    "evidence_version",
                    "retrieved_at",
                    "raw_response_path",
                    "raw_response_sha256",
                    "reason_codes",
                    "evidence_digest",
                }
            ),
        )
        reasons = strict["reason_codes"]
        if not isinstance(reasons, list):
            raise HostEvidenceError("reason_codes must be a list")
        try:
            domain = HostDomain(strict["normalized_host_domain"])
        except (TypeError, ValueError) as error:
            raise HostEvidenceError("normalized_host_domain is unsupported") from error
        return cls(
            record_id=_require_text(strict["record_id"], label="record_id"),
            accession=_optional_text(strict["accession"], label="accession"),
            normalized_host_domain=domain,
            confirmed=strict["confirmed"],
            evidence_source=_require_text(strict["evidence_source"], label="evidence_source"),
            evidence_id=_require_text(strict["evidence_id"], label="evidence_id"),
            evidence_version=_require_text(strict["evidence_version"], label="evidence_version"),
            retrieved_at=_parse_timestamp(strict["retrieved_at"], label="retrieved_at"),
            raw_response_path=_optional_text(strict["raw_response_path"], label="raw_response_path"),
            raw_response_sha256=_optional_text(strict["raw_response_sha256"], label="raw_response_sha256"),
            reason_codes=tuple(reasons),
            evidence_digest=_require_text(strict["evidence_digest"], label="evidence_digest"),
        )

    def to_task1_host_evidence(self) -> HostEvidence:
        """Convert this provenance row to the central typed host-scope contract."""
        return HostEvidence(
            source=self.evidence_source,
            source_version=self.evidence_version,
            replication_host_domains=frozenset({self.normalized_host_domain}),
            confirmed=self.confirmed,
            metadata={
                "record_id": self.record_id,
                "accession": self.accession,
                "evidence_id": self.evidence_id,
                "evidence_digest": self.evidence_digest,
                "retrieved_at": _timestamp(self.retrieved_at),
                "raw_response_path": self.raw_response_path,
                "raw_response_sha256": self.raw_response_sha256,
                "reason_codes": list(self.reason_codes),
            },
        )


@dataclass(frozen=True)
class HostEvidenceTable:
    """Ordered multi-source evidence table with one row per source record."""

    table_id: str
    created_at: datetime
    rows: tuple[HostEvidenceTableRow, ...]

    def __post_init__(self) -> None:
        """Validate table identity, timestamp, row types, and unique record IDs."""
        _require_text(self.table_id, label="table_id")
        _timestamp(self.created_at)
        if not isinstance(self.rows, tuple):
            object.__setattr__(self, "rows", tuple(self.rows))
        if any(not isinstance(row, HostEvidenceTableRow) for row in self.rows):
            raise HostEvidenceError("host-evidence rows must be HostEvidenceTableRow instances")
        record_ids = [row.record_id for row in self.rows]
        if len(record_ids) != len(set(record_ids)):
            raise HostEvidenceError("duplicate record_id in host-evidence table")

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete versioned table contract."""
        return {
            "schema_version": HOST_EVIDENCE_SCHEMA_VERSION,
            "table_type": HOST_EVIDENCE_TABLE_TYPE,
            "table_id": self.table_id,
            "created_at": _timestamp(self.created_at),
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class HostEvidenceTableSnapshot:
    """One byte snapshot bound to both a strict table parse and its SHA-256 digest."""

    table: HostEvidenceTable
    payload: bytes
    sha256: str


def write_host_evidence_table(path: str | Path, table: HostEvidenceTable) -> Path:
    """Atomically write a deterministic evidence table without following a destination symlink."""
    destination = Path(path)
    if destination.is_symlink():
        raise HostEvidenceError("host-evidence table destination must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(table.to_dict(), sort_keys=False, allow_unicode=False).encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_host_evidence_table_snapshot(
    path: str | Path,
    *,
    payload: bytes | None = None,
) -> HostEvidenceTableSnapshot:
    """Strictly parse and hash exactly one caller-captured or locally captured table snapshot."""
    source = Path(path).absolute()
    try:
        raw = _read_regular_file_bytes(source, label="host-evidence table") if payload is None else payload
        if not isinstance(raw, bytes):
            raise HostEvidenceError("host-evidence table snapshot must be bytes")
        parsed = yaml.load(raw, Loader=_UniqueKeyLoader)
    except HostEvidenceError:
        raise
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as error:
        raise HostEvidenceError(f"cannot load host-evidence table: {error}") from error
    strict = _strict_mapping(
        parsed,
        label="host-evidence table",
        keys=frozenset({"schema_version", "table_type", "table_id", "created_at", "rows"}),
    )
    if type(strict["schema_version"]) is not int or strict["schema_version"] != HOST_EVIDENCE_SCHEMA_VERSION:
        raise HostEvidenceError("unsupported host-evidence schema version")
    if strict["table_type"] != HOST_EVIDENCE_TABLE_TYPE:
        raise HostEvidenceError("unsupported host-evidence table type")
    rows = strict["rows"]
    if not isinstance(rows, list):
        raise HostEvidenceError("host-evidence rows must be a list")
    table = HostEvidenceTable(
        table_id=_require_text(strict["table_id"], label="table_id"),
        created_at=_parse_timestamp(strict["created_at"], label="created_at"),
        rows=tuple(HostEvidenceTableRow.from_dict(row) for row in rows),
    )
    return HostEvidenceTableSnapshot(table=table, payload=raw, sha256=hashlib.sha256(raw).hexdigest())


def load_host_evidence_table(path: str | Path) -> HostEvidenceTable:
    """Load the exact versioned table schema and validate every evidence digest."""
    return load_host_evidence_table_snapshot(path).table


def validate_host_evidence_artifacts(table: HostEvidenceTable, *, table_path: str | Path) -> None:
    """Revalidate every recorded raw response and require it for NCBI-derived evidence."""
    source = Path(table_path).absolute()
    for row in table.rows:
        if row.evidence_source == "NCBI_DATASETS" and row.raw_response_path is None:
            raise HostEvidenceError(f"NCBI row {row.record_id} lacks its cached raw response")
        if row.raw_response_path is None:
            continue
        raw_path = Path(row.raw_response_path)
        if not raw_path.is_absolute():
            raw_path = (source.parent / raw_path).absolute()
        raw = _read_regular_file_bytes(raw_path, label=f"raw response for {row.record_id}")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != row.raw_response_sha256:
            raise HostEvidenceError(f"raw response digest drift for {row.record_id}")
        if row.evidence_source == "NCBI_DATASETS":
            if row.accession is None:
                raise HostEvidenceError(f"NCBI row {row.record_id} lacks an accession")
            observed = (
                row.normalized_host_domain,
                row.confirmed,
                row.evidence_id,
                row.evidence_version,
                row.reason_codes,
            )
            if "NCBI_METADATA_UNRESOLVED" in row.reason_codes:
                unresolved = (
                    HostDomain.UNKNOWN,
                    False,
                    f"accession:{row.accession}",
                    NCBI_RESOLVER_VERSION,
                    ("NCBI_METADATA_UNRESOLVED",),
                )
                if observed != unresolved:
                    raise HostEvidenceError(f"unresolved NCBI row does not reconcile for {row.record_id}")
                try:
                    _parse_ncbi_response(raw, accession=row.accession)
                except HostEvidenceError:
                    continue
                raise HostEvidenceError(f"cached unresolved NCBI response now parses for {row.record_id}")
            try:
                expected = _parse_ncbi_response(raw, accession=row.accession)
            except HostEvidenceError as error:
                raise HostEvidenceError(
                    f"cached NCBI response does not reconcile with row {row.record_id}: {error}"
                ) from error
            if observed != expected:
                raise HostEvidenceError(f"cached NCBI response does not reconcile with row {row.record_id}")


def extract_accession(header: str) -> str | None:
    """Extract a versioned NCBI nucleotide accession without treating arbitrary IDs as accessions."""
    if not isinstance(header, str):
        raise HostEvidenceError("FASTA header must be text")
    match = _ACCESSION_RE.search(header)
    return None if match is None else match.group(1)


def _cache_raw_response(cache_dir: Path, accession: str, raw: bytes) -> Path:
    _reject_symlink_components(cache_dir.absolute(), label="NCBI cache directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    cache_path = cache_dir / f"{accession}.{digest}.json"
    if cache_path.exists():
        if _read_regular_file_bytes(cache_path, label="cached NCBI response") != raw:
            raise HostEvidenceError("cached NCBI response conflicts with its content digest")
        return cache_path.absolute()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{accession}.", dir=cache_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return cache_path.absolute()


def _default_ncbi_fetcher(accession: str) -> bytes:
    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/virus/accession/{accession}/dataset_report"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    last_error: urllib.error.URLError | TimeoutError | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    raise HostEvidenceError(f"cannot fetch NCBI metadata for {accession} after 3 attempts") from last_error


def _reject_json_constant(value: str) -> object:
    raise HostEvidenceError(f"non-finite JSON value in NCBI metadata: {value}")


_NCBI_REPORT_KEYS = frozenset(
    {
        "accession",
        "is_annotated",
        "isolate",
        "source_database",
        "protein_count",
        "host",
        "virus",
        "bioprojects",
        "location",
        "update_date",
        "release_date",
        "completeness",
        "length",
        "gene_count",
        "mature_peptide_count",
        "biosample",
        "mol_type",
        "nucleotide",
        "purpose_of_sampling",
        "sra_accessions",
        "submitter",
        "lab_host",
        "is_lab_host",
        "is_vaccine_strain",
        "segment",
    }
)
_NCBI_ORGANISM_KEYS = frozenset(
    {"tax_id", "organism_name", "common_name", "lineage", "pangolin_classification", "infraspecific_names"}
)
_NCBI_DOMAIN_TAXA = {
    2: ("Bacteria", HostDomain.BACTERIA, "NCBI_STRUCTURED_BACTERIAL_HOST"),
    2157: ("Archaea", HostDomain.ARCHAEA, "NCBI_STRUCTURED_ARCHAEAL_HOST"),
    2759: ("Eukaryota", HostDomain.EUKARYOTA, "NCBI_STRUCTURED_EUKARYOTIC_HOST"),
}


def _parse_ncbi_response(raw: bytes, *, accession: str) -> tuple[HostDomain, bool, str, str, tuple[str, ...]]:
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=lambda pairs: _json_unique_pairs(pairs),
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, HostEvidenceError, TypeError, ValueError) as error:
        raise HostEvidenceError(f"NCBI metadata is not strict JSON: {error}") from error
    root = _strict_mapping(
        payload,
        label="NCBI v2alpha virus dataset report",
        keys=frozenset({"reports", "total_count"}),
    )
    reports = root["reports"]
    if type(root["total_count"]) is not int or root["total_count"] != 1:
        raise HostEvidenceError("NCBI metadata must report exactly one accession")
    if not isinstance(reports, list) or len(reports) != 1 or not isinstance(reports[0], Mapping):
        raise HostEvidenceError("NCBI metadata must contain exactly one virus report")
    report = reports[0]
    if not all(isinstance(key, str) for key in report) or not set(report) <= _NCBI_REPORT_KEYS:
        raise HostEvidenceError("NCBI virus report contains fields outside the pinned v2alpha schema")
    if report.get("accession") != accession:
        raise HostEvidenceError("NCBI metadata accession mismatch")
    host = report.get("host")
    unresolved_id = f"ncbi-virus-accession:{accession}:host-unresolved"
    if host is None:
        return (
            HostDomain.UNKNOWN,
            False,
            unresolved_id,
            NCBI_RESOLVER_VERSION,
            ("NCBI_STRUCTURED_HOST_UNRESOLVED",),
        )
    if not isinstance(host, Mapping) or not all(isinstance(key, str) for key in host):
        raise HostEvidenceError("NCBI structured host must be an object")
    if not set(host) <= _NCBI_ORGANISM_KEYS or "tax_id" not in host or "lineage" not in host:
        raise HostEvidenceError("NCBI structured host does not match the pinned Organism schema")
    host_tax_id = host["tax_id"]
    lineage = host["lineage"]
    if type(host_tax_id) is not int or host_tax_id < 1:
        raise HostEvidenceError("NCBI structured host tax_id must be a positive integer")
    if not isinstance(lineage, list) or not lineage:
        raise HostEvidenceError("NCBI structured host lineage must be a non-empty list")
    domains: set[HostDomain] = set()
    reason_by_domain: dict[HostDomain, str] = {}
    for lineage_item in lineage:
        item = _strict_mapping(
            lineage_item,
            label="NCBI structured host lineage item",
            keys=frozenset({"tax_id", "name"}),
        )
        tax_id = item["tax_id"]
        name = item["name"]
        if type(tax_id) is not int or tax_id < 1 or not isinstance(name, str) or not name:
            raise HostEvidenceError("NCBI structured host lineage item is malformed")
        recognized = _NCBI_DOMAIN_TAXA.get(tax_id)
        if recognized is None:
            continue
        expected_name, domain, reason = recognized
        if name != expected_name:
            raise HostEvidenceError("NCBI structured host domain taxon/name mismatch")
        domains.add(domain)
        reason_by_domain[domain] = reason
    evidence_id = f"ncbi-virus-accession:{accession}:host-taxon:{host_tax_id}"
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
    return domain, True, evidence_id, NCBI_RESOLVER_VERSION, (reason_by_domain[domain],)


def _json_unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HostEvidenceError(f"duplicate key in NCBI metadata: {key!r}")
        result[key] = value
    return result


def resolve_ncbi_host_evidence(
    *,
    record_id: str,
    header: str,
    cache_dir: str | Path,
    fetcher: Callable[[str], bytes] = _default_ncbi_fetcher,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> HostEvidenceTableRow:
    """Resolve one accession header, caching exact raw bytes before interpreting them."""
    accession = extract_accession(header)
    retrieved_at = clock()
    if accession is None:
        return HostEvidenceTableRow.create(
            record_id=record_id,
            accession=None,
            normalized_host_domain=HostDomain.UNKNOWN,
            confirmed=False,
            evidence_source="NCBI_DATASETS",
            evidence_id=f"unresolved-header:{record_id}",
            evidence_version=NCBI_RESOLVER_VERSION,
            retrieved_at=retrieved_at,
            raw_response_path=None,
            raw_response_sha256=None,
            reason_codes=("MISSING_NCBI_ACCESSION",),
        )
    raw = fetcher(accession)
    if not isinstance(raw, bytes):
        raise HostEvidenceError("NCBI fetcher must return bytes")
    cache_path = _cache_raw_response(Path(cache_dir), accession, raw)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        domain, confirmed, evidence_id, version, reason_codes = _parse_ncbi_response(raw, accession=accession)
    except HostEvidenceError:
        domain = HostDomain.UNKNOWN
        confirmed = False
        evidence_id = f"accession:{accession}"
        version = NCBI_RESOLVER_VERSION
        reason_codes = ("NCBI_METADATA_UNRESOLVED",)
    return HostEvidenceTableRow.create(
        record_id=record_id,
        accession=accession,
        normalized_host_domain=domain,
        confirmed=confirmed,
        evidence_source="NCBI_DATASETS",
        evidence_id=evidence_id,
        evidence_version=version,
        retrieved_at=retrieved_at,
        raw_response_path=str(cache_path),
        raw_response_sha256=raw_sha256,
        reason_codes=reason_codes,
    )
