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

"""Utilities for reproducing the Microviridae SFT stage."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import tempfile
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import yaml

from bionemo.evo2_phage_gen.design_scope import HostDomain, HostEvidence, evaluate_host_evidence
from bionemo.evo2_phage_gen.host_evidence import (
    HOST_EVIDENCE_SCHEMA_VERSION,
    HostEvidenceError,
    HostEvidenceTable,
    HostEvidenceTableRow,
    HostEvidenceTableSnapshot,
    extract_accession,
    load_host_evidence_table_snapshot,
    resolve_ncbi_host_evidence,
    validate_host_evidence_artifacts,
)
from bionemo.evo2_phage_gen.sequence_safety_cli import (
    CLIValidationError,
    FastaRecord,
    validate_manifest_file,
)
from bionemo.evo2_phage_gen.sequence_safety_cli import (
    main as sequence_safety_main,
)


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ZENODO_DIR = RECIPE_ROOT / "data" / "external" / "zenodo"
DEFAULT_SFT_PROCESSED = DEFAULT_ZENODO_DIR / "microviridae_sft_training_data_processed.fna"
DEFAULT_SFT_RAW = DEFAULT_ZENODO_DIR / "microviridae_sft_training_data_raw.fna"
DEFAULT_SFT_PROCESSED_URL = (
    "https://zenodo.org/records/17101843/files/microviridae_sft_training_data_processed.fna?download=1"
)
DEFAULT_SFT_RAW_URL = "https://zenodo.org/records/17101843/files/microviridae_sft_training_data_raw.fna?download=1"
ALLOWED_SFT_CONDITIONING_PREFIXES = (b"+!", b"+#", b"+$", b"+^", b"+~")
HISTORICAL_ZENODO_CONDITIONING_PREFIX_COUNTS = {
    b"+!": 52,
    b"+#": 388,
    b"+$": 13_729,
    b"+^": 166,
    b"+~": 131,
}
MINIMUM_SFT_GENOMES = 5_000
PREFERRED_SFT_GENOMES = 10_000


class SFTSafetyError(ValueError):
    """An SFT input cannot cross the immutable-source safety boundary."""

    def __init__(self, message: str, *, exit_code: int = 3) -> None:
        """Attach the CLI exit code for an unaccepted result to a stable diagnostic."""
        super().__init__(message)
        self.exit_code = exit_code


class _SFTArgumentParser(argparse.ArgumentParser):
    """Map malformed safety CLI input to INDETERMINATE instead of biological FAIL."""

    def error(self, message: str) -> None:
        """Raise the typed exit-three error used by the public CLI boundary."""
        raise SFTSafetyError(message)


class CorpusAdequacy(StrEnum):
    """Readiness state computed from distinct post-filter biological genomes."""

    PREFERRED_SIZE_MET = "PREFERRED_SIZE_MET"
    MINIMUM_SIZE_MET = "MINIMUM_SIZE_MET"
    AUTHORIZED_BELOW_MINIMUM = "AUTHORIZED_BELOW_MINIMUM"
    BLOCKED_BELOW_MINIMUM = "BLOCKED_BELOW_MINIMUM"


class CorpusAuthorizationKind(StrEnum):
    """The only user-authorized corpus-count exceptions."""

    EXPLICIT_COUNT_OVERRIDE = "explicit_count_override"
    UPFRONT_COMPLETION_MANDATE = "upfront_completion_mandate"


@dataclass(frozen=True)
class CorpusCountAuthorization:
    """Verbatim, time-bound authorization whose scope cannot waive a safety gate."""

    kind: CorpusAuthorizationKind
    verbatim_statement: str
    authorized_at: datetime
    scope: str
    minimum_accepted_count: int | None

    def __post_init__(self) -> None:
        """Restrict authorization to a verbatim, time-bound corpus-count waiver."""
        try:
            object.__setattr__(self, "kind", CorpusAuthorizationKind(self.kind))
        except (TypeError, ValueError) as error:
            raise SFTSafetyError("unsupported corpus-count authorization kind") from error
        if not isinstance(self.verbatim_statement, str) or not self.verbatim_statement.strip():
            raise SFTSafetyError("authorization must preserve a verbatim user statement")
        if self.authorized_at.tzinfo is None or self.authorized_at.utcoffset() is None:
            raise SFTSafetyError("authorization timestamp must be timezone-aware")
        if self.scope != "corpus_count_only":
            raise SFTSafetyError("authorization scope must be corpus_count_only")
        if self.kind is CorpusAuthorizationKind.EXPLICIT_COUNT_OVERRIDE:
            if type(self.minimum_accepted_count) is not int or self.minimum_accepted_count < 1:
                raise SFTSafetyError("an explicit count override requires a positive minimum_accepted_count")
        elif self.minimum_accepted_count is not None:
            raise SFTSafetyError("an upfront completion mandate must not invent a count threshold")

    def to_dict(self) -> dict[str, object]:
        """Serialize the exact count-only authorization record."""
        return {
            "kind": self.kind.value,
            "verbatim_statement": self.verbatim_statement,
            "authorized_at": self.authorized_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "scope": self.scope,
            "minimum_accepted_count": self.minimum_accepted_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> CorpusCountAuthorization:
        """Parse and validate an exact-schema authorization record."""
        if not isinstance(value, Mapping) or set(value) != {
            "kind",
            "verbatim_statement",
            "authorized_at",
            "scope",
            "minimum_accepted_count",
        }:
            raise SFTSafetyError("invalid corpus-count authorization schema")
        timestamp = value["authorized_at"]
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise SFTSafetyError("authorization timestamp must be RFC3339 UTC")
        try:
            authorized_at = datetime.fromisoformat(
                timestamp[:-1] + "+00:00"  # noqa: FURB162  # Python 3.10 datetime.fromisoformat does not accept Z
            )
        except ValueError as error:
            raise SFTSafetyError("authorization timestamp must be RFC3339 UTC") from error
        return cls(
            kind=value["kind"],
            verbatim_statement=value["verbatim_statement"],
            authorized_at=authorized_at,
            scope=value["scope"],
            minimum_accepted_count=value["minimum_accepted_count"],
        )


@dataclass(frozen=True)
class CorpusAdequacyDecision:
    """Exact corpus-size result used by audit and preprocess readiness gates."""

    state: CorpusAdequacy
    distinct_genome_count: int
    ready: bool
    authorization: CorpusCountAuthorization | None


@dataclass(frozen=True)
class ConditionedFastaRecord:
    """One source record and its deterministic unconditioned scanner representation."""

    input_index: int
    record_id: str
    scanner_record_id: str
    original_bytes: bytes
    source_start: int
    source_end: int
    conditioning_prefix: bytes
    conditioned_sequence_sha256: str
    biological_sequence: str
    biological_sequence_sha256: str
    source_record_sha256: str
    scanner_record_sha256: str

    @property
    def scanner_bytes(self) -> bytes:
        """Return the canonical unconditioned Task 4 FASTA record."""
        return f">{self.scanner_record_id}\n{self.biological_sequence}\n".encode("ascii")


@dataclass(frozen=True)
class SFTSourceIdentity:
    """Expected immutable identity for the processed SFT source artifact."""

    provenance_kind: str
    repository: str
    record_id: str
    doi: str
    file_name: str
    url: str
    size_bytes: int
    md5: str

    def __post_init__(self) -> None:
        """Validate a published or explicitly injected source identity."""
        if self.provenance_kind not in {"PUBLISHED_ZENODO_FILE", "INJECTED_TEST_FIXTURE"}:
            raise SFTSafetyError("unsupported SFT source provenance kind")
        for label, value in (
            ("repository", self.repository),
            ("record_id", self.record_id),
            ("doi", self.doi),
            ("file_name", self.file_name),
            ("url", self.url),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SFTSafetyError(f"SFT source identity {label} must be non-empty")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise SFTSafetyError("SFT source identity size_bytes must be positive")
        if re.fullmatch(r"[0-9a-f]{32}", self.md5) is None:
            raise SFTSafetyError("SFT source identity MD5 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete expected source identity."""
        return {
            "provenance_kind": self.provenance_kind,
            "repository": self.repository,
            "record_id": self.record_id,
            "doi": self.doi,
            "file_name": self.file_name,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "md5": self.md5,
        }


@dataclass(frozen=True)
class PhiXReferenceIdentity:
    """Expected accession-version and biological-sequence identity for PhiX."""

    provenance_kind: str
    accession: str
    source: str
    dataset_report_url: str
    fasta_url: str
    sequence_length: int
    sequence_sha256: str
    ncbi_sequence_hash: str

    def __post_init__(self) -> None:
        """Validate a production or explicitly injected PhiX identity."""
        if self.provenance_kind not in {"PUBLISHED_NCBI_ACCESSION", "INJECTED_TEST_FIXTURE"}:
            raise SFTSafetyError("unsupported PhiX provenance kind")
        if self.accession != "NC_001422.1":
            raise SFTSafetyError("PhiX identity must pin accession NC_001422.1")
        for label, value in (
            ("source", self.source),
            ("dataset_report_url", self.dataset_report_url),
            ("fasta_url", self.fasta_url),
            ("ncbi_sequence_hash", self.ncbi_sequence_hash),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SFTSafetyError(f"PhiX identity {label} must be non-empty")
        if type(self.sequence_length) is not int or self.sequence_length < 1:
            raise SFTSafetyError("PhiX sequence length must be positive")
        if re.fullmatch(r"[0-9a-f]{64}", self.sequence_sha256) is None:
            raise SFTSafetyError("PhiX sequence SHA-256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete expected PhiX identity."""
        return {
            "provenance_kind": self.provenance_kind,
            "accession": self.accession,
            "source": self.source,
            "dataset_report_url": self.dataset_report_url,
            "fasta_url": self.fasta_url,
            "sequence_length": self.sequence_length,
            "sequence_sha256": self.sequence_sha256,
            "ncbi_sequence_hash": self.ncbi_sequence_hash,
        }


PRODUCTION_SFT_SOURCE_IDENTITY = SFTSourceIdentity(
    provenance_kind="PUBLISHED_ZENODO_FILE",
    repository="Zenodo",
    record_id="17101843",
    doi="10.5281/zenodo.17101843",
    file_name="microviridae_sft_training_data_processed.fna",
    url=DEFAULT_SFT_PROCESSED_URL,
    size_bytes=72_513_830,
    md5="9cc0906f28fa0b5f0b9aff18adc30126",
)
PRODUCTION_PHIX_REFERENCE_IDENTITY = PhiXReferenceIdentity(
    provenance_kind="PUBLISHED_NCBI_ACCESSION",
    accession="NC_001422.1",
    source="NCBI RefSeq and NCBI Datasets v2alpha virus report",
    dataset_report_url="https://api.ncbi.nlm.nih.gov/datasets/v2alpha/virus/accession/NC_001422.1/dataset_report",
    fasta_url=(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        "db=nuccore&id=NC_001422.1&rettype=fasta&retmode=text"
    ),
    sequence_length=5_386,
    sequence_sha256="97038c7e1edea2297667d7f0426ba942b322c74cb30e072ec66ba47f9c0448d0",
    ncbi_sequence_hash="2FAF564E",
)


@dataclass(frozen=True)
class Task4SafetyRequest:
    """One final-path Task 4 scan/filter request."""

    label: str
    input_fasta: Path
    output_root: Path
    host_domain: HostDomain
    host_evidence: HostEvidence
    policy: Path
    asset_manifest: Path
    diamond_tool_pin: Path
    mmseqs_tool_pin: Path


@dataclass(frozen=True)
class Task4SafetyArtifacts:
    """Trusted child manifests produced for one host-domain batch."""

    scan_manifest: Path
    filter_manifest: Path


@dataclass(frozen=True)
class SFTAuditRequest:
    """All frozen inputs and final output paths for one immutable-source SFT audit."""

    source_fasta: Path
    host_evidence_table: Path
    phix_fasta: Path
    phix_host_evidence_table: Path
    curated_output: Path
    safety_manifest: Path
    audit_root: Path
    preprocess_config: Path
    policy: Path
    asset_manifest: Path
    diamond_tool_pin: Path
    mmseqs_tool_pin: Path
    authorization: CorpusCountAuthorization | None = None
    minimum_genomes: int = MINIMUM_SFT_GENOMES
    preferred_genomes: int = PREFERRED_SFT_GENOMES

    def __post_init__(self) -> None:
        """Normalize paths and prohibit source/output aliasing or invalid thresholds."""
        input_fields = (
            "source_fasta",
            "host_evidence_table",
            "phix_fasta",
            "phix_host_evidence_table",
            "preprocess_config",
            "policy",
            "asset_manifest",
            "diamond_tool_pin",
            "mmseqs_tool_pin",
        )
        output_fields = ("curated_output", "safety_manifest", "audit_root")
        for field_name in (*input_fields, *output_fields):
            normalized = Path(os.path.abspath(os.fspath(getattr(self, field_name))))
            object.__setattr__(self, field_name, normalized)
        inputs = [(field_name, getattr(self, field_name)) for field_name in input_fields]
        outputs = [(field_name, getattr(self, field_name)) for field_name in output_fields]
        for index, (left_name, left_path) in enumerate(outputs):
            for right_name, right_path in outputs[index + 1 :]:
                if left_path == right_path or left_path in right_path.parents or right_path in left_path.parents:
                    raise SFTSafetyError(f"audit outputs {left_name} and {right_name} overlap; paths must be disjoint")
            for input_name, input_path in inputs:
                if left_path == input_path or left_path in input_path.parents or input_path in left_path.parents:
                    raise SFTSafetyError(
                        f"audit output {left_name} and input {input_name} overlap; paths must be disjoint"
                    )
        assess_corpus_adequacy(
            0,
            minimum=self.minimum_genomes,
            preferred=self.preferred_genomes,
        )


@dataclass(frozen=True)
class SFTAuditRuntime:
    """Injectable Task 4 and clock boundaries for deterministic local tests."""

    task4_runner: Callable[[Task4SafetyRequest], Task4SafetyArtifacts]
    task4_manifest_validator: Callable[..., Mapping[str, object]]
    clock: Callable[[], datetime]
    expected_source_identity: SFTSourceIdentity = PRODUCTION_SFT_SOURCE_IDENTITY
    expected_phix_identity: PhiXReferenceIdentity = PRODUCTION_PHIX_REFERENCE_IDENTITY


@dataclass(frozen=True)
class SFTAuditResult:
    """Published parent-manifest result and process exit semantics."""

    exit_code: int
    readiness: str
    safety_manifest: Path
    curated_output: Path


@dataclass(frozen=True)
class _SFTSourceSnapshot:
    """One immutable source payload bound to identity, digest, and conditioned records."""

    path: Path
    payload: bytes
    sha256: str
    records: tuple[ConditionedFastaRecord, ...]


@dataclass(frozen=True)
class _ArtifactSnapshot:
    """One regular-file payload bound to its path and digest."""

    path: Path
    payload: bytes
    sha256: str


@dataclass
class _OwnedAuditEntry:
    """One exclusively created output bound to its parent descriptor and inode."""

    path: Path
    parent_descriptor: int
    name: str
    identity: tuple[int, int]
    is_directory: bool
    descriptor: int | None = None


@dataclass
class _AuditOwnership:
    """All output inodes created by one audit attempt."""

    root: _OwnedAuditEntry
    files: list[_OwnedAuditEntry]


_DOMAIN_ORDER = (
    HostDomain.BACTERIA,
    HostDomain.ARCHAEA,
    HostDomain.BACTERIA_AND_ARCHAEA,
)
_SFT_CLAIM_BOUNDARY = {
    "label": "EMA-draft-aligned first-order SFT eligibility gate",
    "pass_meaning": (
        "Only positive versioned bacterial/archaeal host evidence and Task 4 sequence-safety PASS are eligible. "
        "This does not prove product safety, strict lysis, regulatory compliance, or EMA acceptance."
    ),
    "count_authorization_scope": "corpus_count_only",
}


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest used throughout SFT lineage records."""
    return hashlib.sha256(payload).hexdigest()


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise SFTSafetyError(f"{label} path contains a symlink component")
        except FileNotFoundError:
            return
        except OSError as error:
            raise SFTSafetyError(f"cannot inspect {label} path: {error}") from error


def _read_regular_file_bytes(path: str | Path, *, label: str) -> bytes:
    """Read one stable regular-file snapshot without following symlink path components."""
    source = Path(os.path.abspath(os.fspath(path)))
    parts = source.parts
    if len(parts) < 2:
        raise SFTSafetyError(f"{label} must identify a regular file")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_descriptor = os.open(source.anchor, directory_flags)
    except OSError as error:
        raise SFTSafetyError(f"cannot open {label} root directory: {error}") from error
    try:
        for part in parts[1:-1]:
            try:
                next_descriptor = os.open(part, directory_flags, dir_fd=parent_descriptor)
            except OSError as error:
                raise SFTSafetyError(f"{label} path contains an unreadable or symlink component: {error}") from error
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise SFTSafetyError(f"{label} path contains an unreadable or symlink component: {error}") from error
    finally:
        os.close(parent_descriptor)
    chunks: list[bytes] = []
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SFTSafetyError(f"{label} must be a regular file")
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
    except OSError as error:
        raise SFTSafetyError(f"cannot read {label}: {error}") from error
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _open_directory_descriptor(path: str | Path, *, label: str) -> int:
    """Open and bind every component of one directory without following symlinks."""
    directory = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(directory.anchor, flags)
        for part in directory.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise SFTSafetyError(f"cannot bind {label}: {error}") from error
    return descriptor


def _create_bound_staging(
    parent_descriptor: int,
    destination_name: str,
    *,
    marker: str = "download",
) -> tuple[int, str, Path]:
    """Create an exclusive staging inode relative to an already-bound parent directory."""
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(128):
        name = f".{destination_name}.{marker}.{secrets.token_hex(12)}"
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        return descriptor, name, Path(f"/proc/self/fd/{parent_descriptor}") / name
    raise SFTSafetyError("cannot allocate an exclusive Zenodo download staging file")


def _read_descriptor_bytes(descriptor: int, *, label: str) -> bytes:
    """Read one regular-file payload through an already-bound descriptor without changing its offset."""
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SFTSafetyError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise SFTSafetyError(f"cannot read {label}: {error}") from error
    if offset != before.st_size or (after.st_dev, after.st_ino, after.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ):
        raise SFTSafetyError(f"{label} changed while it was read")
    return b"".join(chunks)


@contextmanager
def _sealed_memfd_snapshot(payload: bytes, *, label: str) -> Iterator[Path]:
    """Expose immutable in-memory bytes through a descriptor path for one trusted consumer."""
    required_os = ("memfd_create", "MFD_ALLOW_SEALING")
    required_fcntl = ("F_ADD_SEALS", "F_GET_SEALS", "F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    if any(not hasattr(os, name) for name in required_os) or any(not hasattr(fcntl, name) for name in required_fcntl):
        raise SFTSafetyError(f"{label} requires Linux sealed-memfd support")
    flags = os.MFD_ALLOW_SEALING | getattr(os, "MFD_CLOEXEC", 0)
    descriptor = -1
    try:
        descriptor = os.memfd_create("evo2-phage-validated-snapshot", flags)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise SFTSafetyError(f"cannot materialize {label}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals != seals:
            raise SFTSafetyError(f"cannot seal {label}")
        if _read_descriptor_bytes(descriptor, label=label) != payload:
            raise SFTSafetyError(f"{label} changed while it was sealed")
        yield Path(f"/proc/self/fd/{descriptor}")
    except SFTSafetyError:
        raise
    except OSError as error:
        raise SFTSafetyError(f"cannot create sealed {label}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_visible_bound_entry(
    path: Path,
    *,
    parent_descriptor: int,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    """Require the caller-visible parent and leaf to resolve to retained bound inodes."""
    visible_parent = -1
    try:
        visible_parent = _open_directory_descriptor(path.parent, label=f"{label} visible parent")
        bound_parent = os.fstat(parent_descriptor)
        visible_parent_stat = os.fstat(visible_parent)
        if (visible_parent_stat.st_dev, visible_parent_stat.st_ino) != (
            bound_parent.st_dev,
            bound_parent.st_ino,
        ):
            raise SFTSafetyError(f"{label} parent path changed or was replaced")
        visible = os.stat(path.name, dir_fd=visible_parent, follow_symlinks=False)
        if (visible.st_dev, visible.st_ino) != expected_identity:
            raise SFTSafetyError(f"{label} path identity changed or was replaced")
    except SFTSafetyError:
        raise
    except OSError as error:
        raise SFTSafetyError(f"{label} parent or path identity changed or was replaced: {error}") from error
    finally:
        if visible_parent >= 0:
            os.close(visible_parent)


def _require_claimed_audit_root(ownership: _AuditOwnership) -> None:
    """Require the caller-visible audit root to remain the exclusively claimed directory."""
    root = ownership.root
    if root.descriptor is None:
        raise SFTSafetyError("audit root ownership descriptor is unavailable")
    try:
        observed = os.fstat(root.descriptor)
    except OSError as error:
        raise SFTSafetyError(f"audit root ownership changed or was replaced: {error}") from error
    if not stat.S_ISDIR(observed.st_mode) or (observed.st_dev, observed.st_ino) != root.identity:
        raise SFTSafetyError("audit root ownership identity changed or was replaced")
    _require_visible_bound_entry(
        root.path,
        parent_descriptor=root.parent_descriptor,
        expected_identity=root.identity,
        label="audit root",
    )


def _write_owned_audit_bytes(
    ownership: _AuditOwnership,
    relative_path: Path,
    payload: bytes,
    *,
    label: str,
) -> Path:
    """Create an internal audit artifact relative to the retained root descriptor."""
    root = ownership.root
    if root.descriptor is None:
        raise SFTSafetyError("audit root ownership descriptor is unavailable")
    relative = Path(relative_path)
    parts = relative.parts
    if relative.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise SFTSafetyError(f"{label} requires a strict audit-root-relative path")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.dup(root.descriptor)
    descriptor = -1
    created = False
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o755, dir_fd=current)
            except FileExistsError:
                pass
            next_descriptor = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = next_descriptor
        descriptor = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=current,
        )
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise SFTSafetyError(f"cannot write {label} inside the bound audit root")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.fsync(current)
    except SFTSafetyError:
        raise
    except OSError as error:
        raise SFTSafetyError(f"cannot write {label} inside the bound audit root: {error}") from error
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if created and descriptor >= 0:
            try:
                os.unlink(parts[-1], dir_fd=current)
            except OSError:
                pass
        os.close(current)
    return root.path.joinpath(*parts)


@contextmanager
def _bound_audit_root_cwd(ownership: _AuditOwnership) -> Iterator[None]:
    """Resolve relative callback paths from the retained audit-root directory inode."""
    root = ownership.root
    if root.descriptor is None:
        raise SFTSafetyError("audit root ownership descriptor is unavailable")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    previous = os.open(".", directory_flags)
    try:
        os.fchdir(root.descriptor)
        yield
    except OSError as error:
        raise SFTSafetyError(f"cannot use the bound audit root for Task 4: {error}") from error
    finally:
        try:
            os.fchdir(previous)
        finally:
            os.close(previous)


def _audit_relative_path(path: Path, ownership: _AuditOwnership, *, label: str) -> Path:
    """Convert one known audit child to a strict callback-relative path."""
    try:
        relative = Path(path).relative_to(ownership.root.path)
    except ValueError as error:
        raise SFTSafetyError(f"{label} is outside the claimed audit root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise SFTSafetyError(f"{label} is not a strict audit-root-relative path")
    return relative


def _claim_owned_directory(path: Path, *, label: str) -> _OwnedAuditEntry:
    """Exclusively create and bind one output directory without following path components."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path, label=label)
    parent_descriptor = _open_directory_descriptor(path.parent, label=f"{label} parent")
    try:
        os.mkdir(path.name, mode=0o755, dir_fd=parent_descriptor)
    except FileExistsError as error:
        os.close(parent_descriptor)
        raise SFTSafetyError(f"{label} already exists") from error
    try:
        observed = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    except Exception:
        try:
            os.rmdir(path.name, dir_fd=parent_descriptor)
        finally:
            os.close(parent_descriptor)
        raise
    return _OwnedAuditEntry(
        path=path,
        parent_descriptor=parent_descriptor,
        name=path.name,
        identity=(observed.st_dev, observed.st_ino),
        is_directory=True,
        descriptor=descriptor,
    )


def _publish_owned_bytes(path: Path, payload: bytes, *, label: str) -> _OwnedAuditEntry:
    """Publish bytes without clobbering and retain a descriptor-bound inode ownership proof."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path, label=label)
    parent_descriptor = _open_directory_descriptor(path.parent, label=f"{label} parent")
    descriptor, staging_name, _staging_path = _create_bound_staging(
        parent_descriptor,
        path.name,
        marker="publish",
    )
    linked_identity: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        descriptor = -1
        staged = os.stat(staging_name, dir_fd=parent_descriptor, follow_symlinks=False)
        try:
            os.link(
                staging_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise SFTSafetyError(f"{label} appeared during no-clobber publication") from error
        linked_identity = (staged.st_dev, staged.st_ino)
        published = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(published.st_mode) or (published.st_dev, published.st_ino) != linked_identity:
            raise SFTSafetyError(f"{label} publication identity changed")
        os.fsync(parent_descriptor)
        _require_visible_bound_entry(
            path,
            parent_descriptor=parent_descriptor,
            expected_identity=linked_identity,
            label=label,
        )
        os.unlink(staging_name, dir_fd=parent_descriptor)
        return _OwnedAuditEntry(
            path=path,
            parent_descriptor=parent_descriptor,
            name=path.name,
            identity=linked_identity,
            is_directory=False,
        )
    except Exception:
        if linked_identity is not None:
            try:
                current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == linked_identity:
                    os.unlink(path.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        try:
            os.unlink(staging_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)
        raise


def _clear_owned_directory(descriptor: int) -> None:
    """Recursively remove entries relative to one already-bound owned directory descriptor."""
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for name in os.listdir(descriptor):
        observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            child = os.open(name, directory_flags, dir_fd=descriptor)
            try:
                _clear_owned_directory(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


def _rollback_owned_audit_outputs(ownership: _AuditOwnership) -> None:
    """Best-effort rollback that removes only entries whose bound inode still matches."""
    for entry in reversed(ownership.files):
        try:
            observed = os.stat(entry.name, dir_fd=entry.parent_descriptor, follow_symlinks=False)
            if (observed.st_dev, observed.st_ino) == entry.identity:
                os.unlink(entry.name, dir_fd=entry.parent_descriptor)
        except OSError:
            pass
    root = ownership.root
    if root.descriptor is not None:
        try:
            _clear_owned_directory(root.descriptor)
        except OSError:
            pass
    try:
        observed = os.stat(root.name, dir_fd=root.parent_descriptor, follow_symlinks=False)
        if (observed.st_dev, observed.st_ino) == root.identity:
            os.rmdir(root.name, dir_fd=root.parent_descriptor)
    except OSError:
        pass


def _close_audit_ownership(ownership: _AuditOwnership) -> None:
    """Close retained ownership descriptors without changing successful outputs."""
    for entry in ownership.files:
        try:
            os.close(entry.parent_descriptor)
        except OSError:
            pass
    if ownership.root.descriptor is not None:
        try:
            os.close(ownership.root.descriptor)
        except OSError:
            pass
    try:
        os.close(ownership.root.parent_descriptor)
    except OSError:
        pass


def _sha256_file(path: str | Path) -> str:
    return sha256_bytes(_read_regular_file_bytes(path, label="required artifact"))


def _capture_artifact(path: str | Path, *, label: str) -> _ArtifactSnapshot:
    """Capture one regular-file payload and derive its digest without reopening it."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    payload = _read_regular_file_bytes(absolute, label=label)
    return _ArtifactSnapshot(path=absolute, payload=payload, sha256=sha256_bytes(payload))


@contextmanager
def _temporary_validation_snapshot(
    reference_path: Path,
    payload: bytes,
    *,
    label: str,
) -> Iterator[Path]:
    """Expose captured bytes as an exclusive same-directory file for one validator call."""
    parent_descriptor = _open_directory_descriptor(reference_path.parent, label=f"{label} parent")
    descriptor = -1
    staging_name = ""
    try:
        descriptor, staging_name, _bound_path = _create_bound_staging(
            parent_descriptor,
            reference_path.name,
            marker="validation",
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        descriptor = -1
        yield reference_path.parent / staging_name
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if staging_name:
            try:
                os.unlink(staging_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _validate_source_identity_payload(path: Path, payload: bytes, expected: SFTSourceIdentity) -> None:
    """Validate the published identity against the exact payload used downstream."""
    observed_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    if path.name != expected.file_name or len(payload) != expected.size_bytes or observed_md5 != expected.md5:
        raise SFTSafetyError("immutable SFT file does not match the expected published source identity")


def _validate_source_identity(path: Path, expected: SFTSourceIdentity) -> None:
    payload = _read_regular_file_bytes(path, label="immutable SFT source")
    _validate_source_identity_payload(path, payload, expected)


def _capture_source_snapshot(path: Path, expected: SFTSourceIdentity) -> _SFTSourceSnapshot:
    """Capture once, then derive published identity, digest, and conditioned records from those bytes."""
    payload = _read_regular_file_bytes(path, label="immutable SFT source")
    _validate_source_identity_payload(path, payload, expected)
    return _SFTSourceSnapshot(
        path=path,
        payload=payload,
        sha256=sha256_bytes(payload),
        records=_parse_conditioned_fasta_payload(payload),
    )


def _parse_single_fasta_snapshot(payload: bytes, *, label: str) -> FastaRecord:
    """Parse exactly one strict FASTA record from an already-authenticated byte snapshot."""
    lines = payload.splitlines(keepends=True)
    if not lines or not lines[0].startswith(b">") or any(line.startswith(b">") for line in lines[1:]):
        raise SFTSafetyError(f"{label} requires exactly one FASTA record")
    header = lines[0].rstrip(b"\r\n")[1:].strip()
    if not header:
        raise SFTSafetyError(f"{label} FASTA header is empty")
    try:
        sequence_id = header.split(None, 1)[0].decode("ascii")
        normalized_sequence = b"".join(b"".join(lines[1:]).split()).decode("ascii").upper()
    except UnicodeDecodeError as error:
        raise SFTSafetyError(f"{label} FASTA identifier and sequence must be ASCII") from error
    if re.fullmatch(r"[A-Za-z0-9_.-]+", sequence_id) is None:
        raise SFTSafetyError(f"{label} FASTA record ID is not byte-stable")
    if not normalized_sequence or re.fullmatch(r"[ACGTN]+", normalized_sequence) is None:
        raise SFTSafetyError(f"{label} FASTA sequence must contain only ACGTN")
    return FastaRecord(
        sequence_id=sequence_id,
        original_bytes=payload,
        normalized_sequence=normalized_sequence,
    )


def _validate_phix_identity_payload(
    path: Path,
    payload: bytes,
    expected: PhiXReferenceIdentity,
) -> FastaRecord:
    """Validate PhiX identity against the exact retained payload used downstream."""
    record = _parse_single_fasta_snapshot(payload, label="canonical PhiX sequence identity")
    sequence = record.normalized_sequence.encode("ascii")
    if (
        record.sequence_id != expected.accession
        or len(sequence) != expected.sequence_length
        or sha256_bytes(sequence) != expected.sequence_sha256
    ):
        raise SFTSafetyError("PhiX reference does not match the canonical PhiX sequence identity")
    return record


def _validate_phix_identity(path: Path, expected: PhiXReferenceIdentity) -> FastaRecord:
    payload = _read_regular_file_bytes(path, label="PhiX reference")
    return _validate_phix_identity_payload(path, payload, expected)


def _validate_authorization_time(
    authorization: CorpusCountAuthorization | None,
    *,
    audit_started_at: datetime,
) -> None:
    if authorization is not None and authorization.authorized_at.astimezone(
        timezone.utc
    ) > audit_started_at.astimezone(timezone.utc):
        raise SFTSafetyError("corpus-count permission must be authorized before the audit started")


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SFTSafetyError("audit timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _artifact_ref(path: Path) -> dict[str, object]:
    return {"path": str(path.absolute()), "sha256": _sha256_file(path)}


def _artifact_ref_from_digest(path: Path, digest: str) -> dict[str, object]:
    """Bind a path to a digest already derived from its retained byte snapshot."""
    return {"path": str(path.absolute()), "sha256": digest}


def _load_validated_host_evidence_table(path: Path, *, label: str) -> HostEvidenceTableSnapshot:
    """Capture one complete table snapshot and reconcile every recorded raw-response artifact."""
    try:
        payload = _read_regular_file_bytes(path, label=label)
        snapshot = load_host_evidence_table_snapshot(path, payload=payload)
        validate_host_evidence_artifacts(snapshot.table, table_path=path)
    except HostEvidenceError as error:
        raise SFTSafetyError(f"{label} is untrusted: {error}") from error
    return snapshot


def _validate_source_accession_bindings(
    records: Sequence[ConditionedFastaRecord], rows: Sequence[HostEvidenceTableRow]
) -> None:
    """Bind each accession-bearing source header to its ordered evidence row."""
    for record, row in zip(records, rows, strict=True):
        header = record.original_bytes.splitlines()[0][1:].decode("ascii")
        try:
            header_accession = extract_accession(header)
        except HostEvidenceError as error:
            raise SFTSafetyError(f"cannot parse source header accession: {error}") from error
        if row.accession != header_accession:
            raise SFTSafetyError(
                f"source header accession {header_accession!r} does not match its ordered host-evidence row"
            )


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise SFTSafetyError(f"duplicate key in YAML: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _parse_strict_yaml_snapshot(raw: bytes, *, label: str) -> object:
    """Strictly parse one already-captured YAML payload."""
    try:
        for token in yaml.scan(raw):
            if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
                raise SFTSafetyError(f"{label} must not contain YAML anchors or aliases")
        payload = yaml.load(raw, Loader=_UniqueKeyLoader)
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except SFTSafetyError:
        raise
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise SFTSafetyError(f"cannot load strict {label}: {error}") from error
    return payload


def _load_strict_yaml(path: str | Path, *, label: str) -> object:
    source = Path(path).absolute()
    return _parse_strict_yaml_snapshot(_read_regular_file_bytes(source, label=label), label=label)


def _strict_mapping(value: object, *, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys or not all(isinstance(key, str) for key in value):
        raise SFTSafetyError(f"{label} keys must be exactly {sorted(keys)}")
    return value


def _write_yaml_atomic(path: Path, value: Mapping[str, object]) -> None:
    _write_bytes_atomic(path, yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=False).encode())


def _artifact_location(value: object, *, label: str) -> tuple[Path, str]:
    record = _strict_mapping(value, label=label, keys={"path", "sha256"})
    if not isinstance(record["path"], str) or not isinstance(record["sha256"], str):
        raise SFTSafetyError(f"{label} path/digest must be strings")
    path = Path(record["path"])
    if not path.is_absolute():
        raise SFTSafetyError(f"{label} path must be absolute")
    return path, record["sha256"]


def _validate_artifact_ref(value: object, *, label: str) -> Path:
    path, expected_digest = _artifact_location(value, label=label)
    if _sha256_file(path) != expected_digest:
        raise SFTSafetyError(f"{label} digest drift")
    return path


def _validate_artifact_snapshot(value: object, *, label: str) -> _ArtifactSnapshot:
    """Capture, hash, and validate one referenced artifact without reopening it."""
    path, expected_digest = _artifact_location(value, label=label)
    snapshot = _capture_artifact(path, label=label)
    if snapshot.sha256 != expected_digest:
        raise SFTSafetyError(f"{label} digest drift")
    return snapshot


def _parse_manifest_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SFTSafetyError(f"{label} must be RFC3339 UTC")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00"  # noqa: FURB162  # Python 3.10 datetime.fromisoformat does not accept Z
        )
    except ValueError as error:
        raise SFTSafetyError(f"{label} must be RFC3339 UTC") from error
    return parsed


def _load_preprocess_batch(
    config_path: Path,
    *,
    payload: bytes | None = None,
) -> tuple[list[dict[str, object]], Path, Path]:
    parsed = (
        _load_strict_yaml(config_path, label="preprocess config")
        if payload is None
        else _parse_strict_yaml_snapshot(payload, label="preprocess config")
    )
    if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], Mapping):
        raise SFTSafetyError("preprocess config must contain exactly one mapping")
    entry = dict(parsed[0])
    datapaths = entry.get("datapaths")
    manifest_value = entry.get("safety_manifest")
    if (
        not isinstance(datapaths, list)
        or len(datapaths) != 1
        or not isinstance(datapaths[0], str)
        or not datapaths[0].endswith("data/curated/microviridae_sft_training_data_safety_pass.fna")
    ):
        raise SFTSafetyError("preprocess config must use only the safety-pass curated FASTA")
    if not isinstance(manifest_value, str) or not manifest_value.endswith("data/curated/SAFETY_MANIFEST.yaml"):
        raise SFTSafetyError("preprocess config requires the companion SAFETY_MANIFEST.yaml")
    if entry.get("output_prefix") != "microviridae_sft_safety_pass":
        raise SFTSafetyError("preprocess output prefix must be safety-qualified")
    config_root = config_path.parent.parent
    datapath = Path(datapaths[0])
    manifest_path = Path(manifest_value)
    if not datapath.is_absolute():
        datapath = (config_root / datapath).absolute()
    if not manifest_path.is_absolute():
        manifest_path = (config_root / manifest_path).absolute()
    return [entry], datapath, manifest_path


def _expected_groups(
    records: Sequence[ConditionedFastaRecord],
    rows: Sequence[HostEvidenceTableRow],
) -> dict[HostDomain, list[ConditionedFastaRecord]]:
    result = {domain: [] for domain in _DOMAIN_ORDER}
    seen_hashes: set[str] = set()
    for record, row in zip(records, rows, strict=True):
        if record.biological_sequence_sha256 in seen_hashes:
            continue
        seen_hashes.add(record.biological_sequence_sha256)
        if evaluate_host_evidence(row.to_task1_host_evidence()).allowed:
            result[row.normalized_host_domain].append(record)
    return result


def validate_safety_manifest(
    path: str | Path,
    *,
    task4_manifest_validator: Callable[..., Mapping[str, object]] = validate_manifest_file,
    require_ready: bool = True,
    expected_minimum_genomes: int = MINIMUM_SFT_GENOMES,
    expected_preferred_genomes: int = PREFERRED_SFT_GENOMES,
    expected_source_identity: SFTSourceIdentity = PRODUCTION_SFT_SOURCE_IDENTITY,
    expected_phix_identity: PhiXReferenceIdentity = PRODUCTION_PHIX_REFERENCE_IDENTITY,
) -> Mapping[str, object]:
    """Recursively validate SFT lineage, every Task 4 child, PhiX, config, and exact curated bytes."""
    manifest_path = Path(path).absolute()
    payload = _load_strict_yaml(manifest_path, label="SFT SAFETY_MANIFEST")
    manifest = _strict_mapping(
        payload,
        label="SFT SAFETY_MANIFEST",
        keys={
            "schema_version",
            "manifest_type",
            "created_at",
            "completed_at",
            "source",
            "quality",
            "host_evidence_table",
            "conditioning_lineage",
            "conditioning_summary",
            "domain_children",
            "record_decisions",
            "curated_output",
            "phix_reference",
            "adequacy",
            "authorization",
            "readiness",
            "preprocess",
            "claim_boundary",
        },
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise SFTSafetyError("unsupported SFT safety manifest schema version")
    if manifest["manifest_type"] != "microviridae_sft_safety":
        raise SFTSafetyError("unsupported SFT safety manifest type")
    created = _parse_manifest_timestamp(manifest["created_at"], label="created_at")
    completed = _parse_manifest_timestamp(manifest["completed_at"], label="completed_at")
    if completed < created:
        raise SFTSafetyError("SFT safety manifest completed before it was created")

    source_record = _strict_mapping(
        manifest["source"],
        label="source",
        keys={"path", "sha256_before", "sha256_after", "record_count", "immutable", "published_identity"},
    )
    if (
        not isinstance(source_record["path"], str)
        or source_record["immutable"] is not True
        or source_record["published_identity"] != expected_source_identity.to_dict()
        or source_record["sha256_before"] != source_record["sha256_after"]
    ):
        raise SFTSafetyError("immutable published source identity or lineage drift")
    source_path = Path(source_record["path"])
    if not source_path.is_absolute():
        raise SFTSafetyError("immutable Zenodo source digest drift")
    source_snapshot = _capture_source_snapshot(source_path, expected_source_identity)
    if source_snapshot.sha256 != source_record["sha256_before"]:
        raise SFTSafetyError("immutable Zenodo source digest drift")
    records = source_snapshot.records
    if type(source_record["record_count"]) is not int or source_record["record_count"] != len(records):
        raise SFTSafetyError("source record count drift")

    table_record = _strict_mapping(
        manifest["host_evidence_table"],
        label="host_evidence_table",
        keys={"path", "sha256", "schema_version", "table_id", "row_count", "sources"},
    )
    table_path, expected_table_sha256 = _artifact_location(
        {"path": table_record["path"], "sha256": table_record["sha256"]},
        label="host-evidence table",
    )
    table_snapshot = _load_validated_host_evidence_table(table_path, label="host-evidence table")
    table = table_snapshot.table
    if (
        table_snapshot.sha256 != expected_table_sha256
        or type(table_record["schema_version"]) is not int
        or table_record["schema_version"] != HOST_EVIDENCE_SCHEMA_VERSION
        or table_record["table_id"] != table.table_id
        or type(table_record["row_count"]) is not int
        or table_record["row_count"] != len(table.rows)
        or table_record["sources"] != sorted({row.evidence_source for row in table.rows})
        or [row.record_id for row in table.rows] != [record.record_id for record in records]
    ):
        raise SFTSafetyError("host-evidence table lineage drift")
    _validate_source_accession_bindings(records, table.rows)

    if manifest["conditioning_lineage"] != [_lineage_record(record) for record in records]:
        raise SFTSafetyError("conditioning lineage drift")
    if manifest["conditioning_summary"] != _conditioning_summary(records):
        raise SFTSafetyError("conditioning-prefix summary drift")
    distinct_hashes = {record.biological_sequence_sha256 for record in records}
    duplicate_count = len(records) - len(distinct_hashes)
    quality = _strict_mapping(
        manifest["quality"],
        label="quality",
        keys={"valid_record_count", "distinct_input_genome_hash_count", "duplicate_count"},
    )
    if quality != {
        "valid_record_count": len(records),
        "distinct_input_genome_hash_count": len(distinct_hashes),
        "duplicate_count": duplicate_count,
    } or any(type(value) is not int for value in quality.values()):
        raise SFTSafetyError("quality or duplicate counts drift")

    expected_groups = _expected_groups(records, table.rows)
    expected_domains = [domain for domain in _DOMAIN_ORDER if expected_groups[domain]]
    children = manifest["domain_children"]
    if not isinstance(children, list) or len(children) != len(expected_domains):
        raise SFTSafetyError("Task 4 domain child inventory drift")
    all_states: dict[str, str] = {}
    for child_value, domain in zip(children, expected_domains, strict=True):
        child = _strict_mapping(
            child_value,
            label="domain child",
            keys={"label", "host_domain", "input_fasta", "scan_manifest", "filter_manifest"},
        )
        expected_records = expected_groups[domain]
        expected_ids = [record.scanner_record_id for record in expected_records]
        label = domain.value.lower()
        if child["label"] != label or child["host_domain"] != domain.value:
            raise SFTSafetyError("Task 4 domain child order/profile drift")
        input_record = _strict_mapping(
            child["input_fasta"],
            label="domain scanner input",
            keys={"path", "sha256", "count", "record_ids"},
        )
        input_snapshot = _validate_artifact_snapshot(
            {"path": input_record["path"], "sha256": input_record["sha256"]},
            label="domain scanner input",
        )
        input_path = input_snapshot.path
        if (
            type(input_record["count"]) is not int
            or input_record["count"] != len(expected_ids)
            or input_record["record_ids"] != expected_ids
            or input_snapshot.payload != b"".join(record.scanner_bytes for record in expected_records)
        ):
            raise SFTSafetyError("domain scanner input bytes/order drift")
        scan_snapshot = _validate_artifact_snapshot(child["scan_manifest"], label="domain scan manifest")
        filter_snapshot = _validate_artifact_snapshot(child["filter_manifest"], label="domain filter manifest")
        states, _scan, _filtered = _validate_task4_pair(
            Task4SafetyArtifacts(scan_snapshot.path, filter_snapshot.path),
            input_fasta=input_path,
            input_snapshot=input_snapshot,
            scan_snapshot=scan_snapshot,
            filter_snapshot=filter_snapshot,
            host_domain=domain,
            expected_ids=expected_ids,
            validator=task4_manifest_validator,
        )
        all_states.update(states)

    decisions, selected_ids = _decision_rows(records, table.rows, all_states)
    if manifest["record_decisions"] != decisions:
        raise SFTSafetyError("record eligibility/quarantine decisions drift")
    curated_record = _strict_mapping(
        manifest["curated_output"],
        label="curated_output",
        keys={
            "path",
            "sha256",
            "count",
            "record_ids",
            "distinct_genome_hash_count",
            "preserves_original_conditioned_source_bytes",
        },
    )
    if not isinstance(curated_record["path"], str) or not isinstance(curated_record["sha256"], str):
        raise SFTSafetyError("curated output path/digest is malformed")
    curated_snapshot = _validate_artifact_snapshot(
        {"path": curated_record["path"], "sha256": curated_record["sha256"]},
        label="curated output",
    )
    curated_path = curated_snapshot.path
    selected_order = [record.record_id for record in records if record.record_id in selected_ids]
    expected_curated = conditioned_records_bytes(records, selected_ids)
    selected_hashes = {record.biological_sequence_sha256 for record in records if record.record_id in selected_ids}
    if (
        curated_snapshot.payload != expected_curated
        or type(curated_record["count"]) is not int
        or curated_record["count"] != len(selected_order)
        or curated_record["record_ids"] != selected_order
        or type(curated_record["distinct_genome_hash_count"]) is not int
        or curated_record["distinct_genome_hash_count"] != len(selected_hashes)
        or curated_record["preserves_original_conditioned_source_bytes"] is not True
    ):
        raise SFTSafetyError("curated output bytes/order/count drift")

    phix = _strict_mapping(
        manifest["phix_reference"],
        label="phix_reference",
        keys={
            "accession",
            "state",
            "reference_identity",
            "source_fasta",
            "input_fasta",
            "host_evidence_table",
            "host_evidence",
            "scan_manifest",
            "filter_manifest",
        },
    )
    phix_source = _strict_mapping(
        phix["source_fasta"],
        label="PhiX canonical source",
        keys={"path", "sha256", "count"},
    )
    phix_input = _strict_mapping(
        phix["input_fasta"],
        label="PhiX input",
        keys={"path", "sha256", "count"},
    )
    phix_input_snapshot = _validate_artifact_snapshot(
        {"path": phix_input["path"], "sha256": phix_input["sha256"]},
        label="PhiX input",
    )
    phix_path = phix_input_snapshot.path
    if phix["reference_identity"] != expected_phix_identity.to_dict():
        raise SFTSafetyError("canonical PhiX sequence identity drift")
    phix_source_snapshot = _validate_artifact_snapshot(
        {"path": phix_source["path"], "sha256": phix_source["sha256"]},
        label="PhiX canonical source",
    )
    phix_source_path = phix_source_snapshot.path
    source_phix_record = _validate_phix_identity_payload(
        phix_source_path,
        phix_source_snapshot.payload,
        expected_phix_identity,
    )
    phix_record = _validate_phix_identity_payload(phix_path, phix_input_snapshot.payload, expected_phix_identity)
    if (
        type(phix_source["count"]) is not int
        or phix_source["count"] != 1
        or sha256_bytes(source_phix_record.original_bytes) != phix_source["sha256"]
        or source_phix_record.original_bytes != phix_record.original_bytes
        or sha256_bytes(phix_record.original_bytes) != phix_input["sha256"]
    ):
        raise SFTSafetyError("canonical PhiX byte snapshot digest drift")
    phix_table_record = _strict_mapping(
        phix["host_evidence_table"],
        label="PhiX host-evidence table",
        keys={"path", "sha256", "schema_version", "table_id", "row_count", "sources"},
    )
    phix_table_path, expected_phix_table_sha256 = _artifact_location(
        {"path": phix_table_record["path"], "sha256": phix_table_record["sha256"]},
        label="PhiX host-evidence table",
    )
    phix_table_snapshot = _load_validated_host_evidence_table(
        phix_table_path,
        label="PhiX host-evidence table",
    )
    phix_table = phix_table_snapshot.table
    if (
        phix_table_snapshot.sha256 != expected_phix_table_sha256
        or type(phix_table_record["schema_version"]) is not int
        or phix_table_record["schema_version"] != HOST_EVIDENCE_SCHEMA_VERSION
        or phix_table_record["table_id"] != phix_table.table_id
        or type(phix_table_record["row_count"]) is not int
        or phix_table_record["row_count"] != 1
        or len(phix_table.rows) != 1
        or phix_table_record["sources"] != sorted({row.evidence_source for row in phix_table.rows})
    ):
        raise SFTSafetyError("PhiX host-evidence table lineage drift")
    phix_evidence = phix_table.rows[0]
    if (
        phix["accession"] != "NC_001422.1"
        or phix["state"] != "PASS"
        or type(phix_input["count"]) is not int
        or phix_input["count"] != 1
        or phix_record.sequence_id != "NC_001422.1"
        or phix_evidence.record_id != "NC_001422.1"
        or phix_evidence.accession != "NC_001422.1"
        or phix["host_evidence"] != phix_evidence.to_dict()
        or phix_evidence.normalized_host_domain is not HostDomain.BACTERIA
        or not evaluate_host_evidence(phix_evidence.to_task1_host_evidence()).allowed
    ):
        raise SFTSafetyError("PhiX reference identity or positive bacterial evidence drift")
    phix_scan_snapshot = _validate_artifact_snapshot(phix["scan_manifest"], label="PhiX scan manifest")
    phix_filter_snapshot = _validate_artifact_snapshot(phix["filter_manifest"], label="PhiX filter manifest")
    phix_states, _scan, _filtered = _validate_task4_pair(
        Task4SafetyArtifacts(phix_scan_snapshot.path, phix_filter_snapshot.path),
        input_fasta=phix_path,
        input_snapshot=phix_input_snapshot,
        scan_snapshot=phix_scan_snapshot,
        filter_snapshot=phix_filter_snapshot,
        host_domain=HostDomain.BACTERIA,
        expected_ids=["NC_001422.1"],
        validator=task4_manifest_validator,
    )
    if phix_states != {"NC_001422.1": "PASS"}:
        raise SFTSafetyError("PhiX independent Task 4 result is not PASS")

    adequacy_record = _strict_mapping(
        manifest["adequacy"],
        label="adequacy",
        keys={
            "state",
            "distinct_genome_count",
            "minimum_genomes",
            "preferred_genomes",
            "preferred_is_non_blocking",
            "ready",
        },
    )
    authorization = (
        None if manifest["authorization"] is None else CorpusCountAuthorization.from_dict(manifest["authorization"])
    )
    _validate_authorization_time(authorization, audit_started_at=created)
    if any(
        type(adequacy_record[key]) is not int
        for key in ("distinct_genome_count", "minimum_genomes", "preferred_genomes")
    ):
        raise SFTSafetyError("adequacy counts must be integers")
    if (
        adequacy_record["minimum_genomes"] != expected_minimum_genomes
        or adequacy_record["preferred_genomes"] != expected_preferred_genomes
    ):
        raise SFTSafetyError("nonstandard corpus thresholds are not trusted by this validator")
    expected_adequacy = assess_corpus_adequacy(
        len(selected_hashes),
        authorization=authorization,
        minimum=adequacy_record["minimum_genomes"],
        preferred=adequacy_record["preferred_genomes"],
    )
    if adequacy_record != {
        "state": expected_adequacy.state.value,
        "distinct_genome_count": expected_adequacy.distinct_genome_count,
        "minimum_genomes": adequacy_record["minimum_genomes"],
        "preferred_genomes": adequacy_record["preferred_genomes"],
        "preferred_is_non_blocking": True,
        "ready": expected_adequacy.ready,
    }:
        raise SFTSafetyError("corpus adequacy drift")
    expected_authorization = (
        None if expected_adequacy.authorization is None else expected_adequacy.authorization.to_dict()
    )
    if manifest["authorization"] != expected_authorization:
        raise SFTSafetyError("corpus-count authorization drift")

    preprocess_record = _strict_mapping(
        manifest["preprocess"],
        label="preprocess",
        keys={"path", "sha256", "datapath", "safety_manifest", "output_prefix"},
    )
    preprocess_snapshot = _validate_artifact_snapshot(
        {"path": preprocess_record["path"], "sha256": preprocess_record["sha256"]},
        label="preprocess config",
    )
    preprocess_path = preprocess_snapshot.path
    _batch, configured_datapath, configured_manifest = _load_preprocess_batch(
        preprocess_path,
        payload=preprocess_snapshot.payload,
    )
    if (
        configured_datapath != curated_path
        or configured_manifest != manifest_path
        or preprocess_record["datapath"] != str(curated_path)
        or preprocess_record["safety_manifest"] != str(manifest_path)
        or preprocess_record["output_prefix"] != "microviridae_sft_safety_pass"
    ):
        raise SFTSafetyError("preprocess config/manifest lineage drift")
    completion = "READY" if expected_adequacy.ready and bool(selected_ids) else "BLOCKED"
    readiness = _strict_mapping(manifest["readiness"], label="readiness", keys={"state", "reason_codes"})
    expected_readiness = {
        "state": completion,
        "reason_codes": [] if completion == "READY" else ["CORPUS_BELOW_MINIMUM_WITHOUT_AUTHORIZATION"],
    }
    if readiness != expected_readiness:
        raise SFTSafetyError("SFT readiness drift")
    if manifest["claim_boundary"] != _SFT_CLAIM_BOUNDARY:
        raise SFTSafetyError("SFT claim boundary drift")
    if require_ready and completion != "READY":
        raise SFTSafetyError("SFT blocked without user permission for the below-minimum corpus count")
    return manifest


def _delegate_shared_preprocess(batch: list[dict[str, object]]) -> None:
    from bionemo.evo2.data import preprocess as shared_preprocess

    payload = yaml.safe_dump(batch, sort_keys=False).encode()
    original_argv = sys.argv
    with _sealed_memfd_snapshot(payload, label="sanitized preprocess config") as config_snapshot:
        try:
            sys.argv = ["preprocess_evo2", "--config", str(config_snapshot)]
            shared_preprocess.main()
        finally:
            sys.argv = original_argv


def preprocess_main(
    argv: Sequence[str] | None = None,
    *,
    delegate: Callable[[list[dict[str, object]]], object] | None = None,
    safety_manifest_validator: Callable[..., Mapping[str, object]] = validate_safety_manifest,
) -> int:
    """Validate the parent and child safety lineage before delegating to shared preprocessing."""
    parser = _SFTArgumentParser(description="Safety-gated Evo 2 FASTA preprocessing")
    parser.add_argument("-c", "--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = args.config.absolute()
    config_snapshot = _capture_artifact(config_path, label="preprocess config")
    batch, datapath, manifest_path = _load_preprocess_batch(config_path, payload=config_snapshot.payload)
    curated_snapshot = _capture_artifact(datapath, label="curated preprocess input")
    validated_manifest = safety_manifest_validator(manifest_path, require_ready=True)
    curated_record = _strict_mapping(
        validated_manifest.get("curated_output") if isinstance(validated_manifest, Mapping) else None,
        label="validated curated_output",
        keys={
            "path",
            "sha256",
            "count",
            "record_ids",
            "distinct_genome_hash_count",
            "preserves_original_conditioned_source_bytes",
        },
    )
    preprocess_record = _strict_mapping(
        validated_manifest.get("preprocess") if isinstance(validated_manifest, Mapping) else None,
        label="validated preprocess config",
        keys={"path", "sha256", "datapath", "safety_manifest", "output_prefix"},
    )
    if (
        preprocess_record["path"] != str(config_snapshot.path)
        or preprocess_record["sha256"] != config_snapshot.sha256
        or preprocess_record["datapath"] != str(datapath)
        or preprocess_record["safety_manifest"] != str(manifest_path)
        or preprocess_record["output_prefix"] != "microviridae_sft_safety_pass"
    ):
        raise SFTSafetyError("validated preprocess config path or digest changed from the parsed snapshot")
    if curated_record["path"] != str(datapath) or curated_record["sha256"] != curated_snapshot.sha256:
        raise SFTSafetyError("validated curated snapshot path or digest drift")
    if _read_regular_file_bytes(datapath, label="curated preprocess input") != curated_snapshot.payload:
        raise SFTSafetyError("curated preprocess input changed after safety validation")
    sanitized = [{key: value for key, value in entry.items() if key != "safety_manifest"} for entry in batch]
    selected_delegate = _delegate_shared_preprocess if delegate is None else delegate
    with _sealed_memfd_snapshot(curated_snapshot.payload, label="validated curated preprocess input") as snapshot_path:
        sanitized[0]["datapaths"] = [str(snapshot_path)]
        selected_delegate(sanitized)
    return 0


def _default_task4_runner(request: Task4SafetyRequest) -> Task4SafetyArtifacts:
    """Run Task 4 directly while retaining biological FAIL records for partitioning."""
    scan_dir = request.output_root / "scan"
    filter_dir = request.output_root / "filter"
    host_json = json.dumps(request.host_evidence.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    scan_exit = sequence_safety_main(
        [
            "scan",
            "--input-fasta",
            str(request.input_fasta),
            "--output-dir",
            str(scan_dir),
            "--policy",
            str(request.policy),
            "--asset-manifest",
            str(request.asset_manifest),
            "--host-domain",
            request.host_domain.value,
            "--host-evidence-json",
            host_json,
            "--diamond-tool-pin",
            str(request.diamond_tool_pin),
            "--mmseqs-tool-pin",
            str(request.mmseqs_tool_pin),
        ]
    )
    scan_manifest = scan_dir / "manifest.json"
    if scan_exit == 3:
        if scan_manifest.exists():
            try:
                if json.loads(scan_manifest.read_text()).get("manifest_type") == "sequence_safety_diagnostic":
                    raise SFTSafetyError("diagnostic Task 4 manifest cannot authorize SFT")
            except json.JSONDecodeError:
                pass
        raise SFTSafetyError(f"Task 4 scan was indeterminate for {request.label}")
    if scan_exit not in {0, 2}:
        raise SFTSafetyError(f"unexpected Task 4 scan exit code for {request.label}: {scan_exit}")
    filter_exit = sequence_safety_main(
        [
            "filter-fasta",
            "--input-fasta",
            str(request.input_fasta),
            "--scan-manifest",
            str(scan_manifest),
            "--output-dir",
            str(filter_dir),
        ]
    )
    if filter_exit not in {0, 2}:
        raise SFTSafetyError(f"Task 4 filtering was indeterminate for {request.label}")
    return Task4SafetyArtifacts(scan_manifest=scan_manifest, filter_manifest=filter_dir / "manifest.json")


def _default_audit_runtime() -> SFTAuditRuntime:
    return SFTAuditRuntime(
        task4_runner=_default_task4_runner,
        task4_manifest_validator=validate_manifest_file,
        clock=lambda: datetime.now(timezone.utc),
    )


def _parse_task4_manifest_snapshot(snapshot: _ArtifactSnapshot, *, label: str) -> dict[str, object]:
    """Strictly decode exactly the captured Task 4 manifest bytes without a second open."""

    def unique_mapping(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SFTSafetyError(f"duplicate key in {label}: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise SFTSafetyError(f"non-finite JSON value in {label}: {value}")

    try:
        payload = json.loads(
            snapshot.payload,
            object_pairs_hook=unique_mapping,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SFTSafetyError(f"cannot parse {label}: {error}") from error
    if not isinstance(payload, dict):
        raise SFTSafetyError(f"{label} must be a mapping")
    return payload


def _validate_task4_pair(
    artifacts: Task4SafetyArtifacts,
    *,
    input_fasta: Path,
    input_snapshot: _ArtifactSnapshot | None = None,
    scan_snapshot: _ArtifactSnapshot | None = None,
    filter_snapshot: _ArtifactSnapshot | None = None,
    host_domain: HostDomain,
    expected_ids: Sequence[str],
    validator: Callable[..., Mapping[str, object]],
) -> tuple[dict[str, str], _ArtifactSnapshot, _ArtifactSnapshot]:
    captured_input = input_snapshot or _capture_artifact(input_fasta, label="Task 4 input FASTA")
    captured_scan = scan_snapshot or _capture_artifact(artifacts.scan_manifest, label="Task 4 scan manifest")
    captured_filter = filter_snapshot or _capture_artifact(artifacts.filter_manifest, label="Task 4 filter manifest")
    scan = _parse_task4_manifest_snapshot(captured_scan, label="Task 4 scan manifest")
    filtered = _parse_task4_manifest_snapshot(captured_filter, label="Task 4 filter manifest")
    if scan.get("manifest_type") == "sequence_safety_diagnostic":
        raise SFTSafetyError("diagnostic Task 4 manifests cannot authorize SFT")
    if (
        scan.get("manifest_type") != "sequence_safety_scan"
        or filtered.get("manifest_type") != "sequence_safety_filter"
    ):
        raise SFTSafetyError("unsupported Task 4 child manifest type")
    expected_path = str(captured_input.path)
    expected_digest = captured_input.sha256
    for name, payload in (("scan", scan), ("filter", filtered)):
        input_record = payload.get("input_fasta")
        if not isinstance(input_record, Mapping) or (
            input_record.get("path") != expected_path
            or input_record.get("sha256") != expected_digest
            or input_record.get("count") != len(expected_ids)
        ):
            raise SFTSafetyError(f"Task 4 {name} input lineage drift")
        profile = payload.get("resolved_profile")
        if not isinstance(profile, Mapping) or profile.get("host_domain") != host_domain.value:
            raise SFTSafetyError(f"Task 4 {name} host-domain drift")
    source_scan = filtered.get("source_scan_manifest")
    if not isinstance(source_scan, Mapping) or source_scan != {
        "path": str(captured_scan.path),
        "sha256": captured_scan.sha256,
    }:
        raise SFTSafetyError("Task 4 filter source-scan lineage drift")
    scan_for_validation = json.loads(json.dumps(scan, sort_keys=True, allow_nan=False))
    filter_for_validation = json.loads(json.dumps(filtered, sort_keys=True, allow_nan=False))
    with _temporary_validation_snapshot(
        captured_input.path,
        captured_input.payload,
        label="Task 4 input validation snapshot",
    ) as validation_input:
        scan_for_validation["input_fasta"] = {**scan_for_validation["input_fasta"], "path": str(validation_input)}
        scan_validation_payload = json.dumps(
            scan_for_validation, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        with _temporary_validation_snapshot(
            captured_scan.path,
            scan_validation_payload,
            label="Task 4 scan validation snapshot",
        ) as validation_scan:
            filter_for_validation["input_fasta"] = {
                **filter_for_validation["input_fasta"],
                "path": str(validation_input),
            }
            filter_for_validation["source_scan_manifest"] = {
                "path": str(validation_scan),
                "sha256": sha256_bytes(scan_validation_payload),
            }
            filter_validation_payload = json.dumps(
                filter_for_validation, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            with _temporary_validation_snapshot(
                captured_filter.path,
                filter_validation_payload,
                label="Task 4 filter validation snapshot",
            ) as validation_filter:
                try:
                    validated_scan = validator(validation_scan, expected_type="sequence_safety_scan")
                    validated_filter = validator(validation_filter, expected_type="sequence_safety_filter")
                except (CLIValidationError, OSError, TypeError, ValueError) as error:
                    raise SFTSafetyError(f"untrusted Task 4 child manifest: {error}") from error
    if validated_scan != scan_for_validation or validated_filter != filter_for_validation:
        raise SFTSafetyError("Task 4 validator returned bytes other than its captured validation snapshots")
    scan_rows = scan.get("records")
    filter_rows = filtered.get("records")
    if not isinstance(scan_rows, list) or not isinstance(filter_rows, list) or len(scan_rows) != len(expected_ids):
        raise SFTSafetyError("Task 4 record inventory drift")
    states: dict[str, str] = {}
    for index, (expected_id, scan_row, filter_row) in enumerate(
        zip(expected_ids, scan_rows, filter_rows, strict=True)
    ):
        if not isinstance(scan_row, Mapping) or not isinstance(filter_row, Mapping):
            raise SFTSafetyError("Task 4 record row is malformed")
        state = scan_row.get("state")
        if (
            scan_row.get("record_id") != expected_id
            or scan_row.get("input_index") != index
            or filter_row.get("record_id") != expected_id
            or filter_row.get("input_index") != index
            or filter_row.get("state") != state
            or state not in {"PASS", "FAIL", "INDETERMINATE"}
        ):
            raise SFTSafetyError("Task 4 record order/state drift")
        states[expected_id] = state
    return states, captured_scan, captured_filter


def _canonical_task4_artifact_path(path: Path, ownership: _AuditOwnership, *, label: str) -> Path:
    """Canonicalize one callback result and require it to remain under the named owned root."""
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(ownership.root.path)
    except ValueError as error:
        raise SFTSafetyError(f"{label} is outside the claimed audit root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise SFTSafetyError(f"{label} is not a strict audit-root artifact path")
    return ownership.root.path / relative


def _run_bound_task4_pair(
    *,
    ownership: _AuditOwnership,
    runtime: SFTAuditRuntime,
    label: str,
    input_fasta: Path,
    input_snapshot: _ArtifactSnapshot,
    output_root: Path,
    host_domain: HostDomain,
    host_evidence: HostEvidence,
    trusted_inputs: Mapping[str, Path],
    expected_ids: Sequence[str],
) -> tuple[dict[str, str], _ArtifactSnapshot, _ArtifactSnapshot]:
    """Run and validate Task 4 while every callback path resolves from the claimed root inode."""
    relative_input = _audit_relative_path(input_fasta, ownership, label=f"{label} Task 4 input")
    relative_output = _audit_relative_path(output_root, ownership, label=f"{label} Task 4 output")
    relative_trusted = {
        field: _audit_relative_path(path, ownership, label=f"{label} Task 4 {field}")
        for field, path in trusted_inputs.items()
    }
    _require_claimed_audit_root(ownership)
    with _bound_audit_root_cwd(ownership):
        artifacts = runtime.task4_runner(
            Task4SafetyRequest(
                label=label,
                input_fasta=relative_input,
                output_root=relative_output,
                host_domain=host_domain,
                host_evidence=host_evidence,
                policy=relative_trusted["policy"],
                asset_manifest=relative_trusted["asset_manifest"],
                diamond_tool_pin=relative_trusted["diamond_tool_pin"],
                mmseqs_tool_pin=relative_trusted["mmseqs_tool_pin"],
            )
        )
        _require_claimed_audit_root(ownership)
        canonical_artifacts = Task4SafetyArtifacts(
            scan_manifest=_canonical_task4_artifact_path(
                artifacts.scan_manifest,
                ownership,
                label=f"{label} Task 4 scan manifest",
            ),
            filter_manifest=_canonical_task4_artifact_path(
                artifacts.filter_manifest,
                ownership,
                label=f"{label} Task 4 filter manifest",
            ),
        )
        result = _validate_task4_pair(
            canonical_artifacts,
            input_fasta=input_fasta,
            input_snapshot=input_snapshot,
            host_domain=host_domain,
            expected_ids=expected_ids,
            validator=runtime.task4_manifest_validator,
        )
        _require_claimed_audit_root(ownership)
    _require_claimed_audit_root(ownership)
    return result


def _combined_host_evidence(rows: Sequence[HostEvidenceTableRow], domain: HostDomain, table_id: str) -> HostEvidence:
    return HostEvidence(
        source="SFT_HOST_EVIDENCE_TABLE",
        source_version=table_id,
        replication_host_domains=frozenset({domain}),
        confirmed=True,
        metadata={"record_evidence_digests": [row.evidence_digest for row in rows]},
    )


def _lineage_record(record: ConditionedFastaRecord) -> dict[str, object]:
    return {
        "input_index": record.input_index,
        "record_id": record.record_id,
        "scanner_record_id": record.scanner_record_id,
        "source_byte_span": [record.source_start, record.source_end],
        "source_record_sha256": record.source_record_sha256,
        "conditioning_prefix": record.conditioning_prefix.decode("ascii"),
        "conditioned_sequence_sha256": record.conditioned_sequence_sha256,
        "biological_sequence_sha256": record.biological_sequence_sha256,
        "scanner_record_sha256": record.scanner_record_sha256,
    }


def _conditioning_summary(records: Sequence[ConditionedFastaRecord]) -> dict[str, object]:
    observed = {prefix.decode("ascii"): 0 for prefix in ALLOWED_SFT_CONDITIONING_PREFIXES}
    for record in records:
        observed[record.conditioning_prefix.decode("ascii")] += 1
    historical = {
        prefix.decode("ascii"): count for prefix, count in HISTORICAL_ZENODO_CONDITIONING_PREFIX_COUNTS.items()
    }
    return {
        "allowed_prefixes": [prefix.decode("ascii") for prefix in ALLOWED_SFT_CONDITIONING_PREFIXES],
        "observed_counts": observed,
        "historical_zenodo_counts": historical,
        "historical_total": sum(historical.values()),
    }


def _decision_rows(
    records: Sequence[ConditionedFastaRecord],
    evidence_rows: Sequence[HostEvidenceTableRow],
    states: Mapping[str, str],
) -> tuple[list[dict[str, object]], set[str]]:
    decisions: list[dict[str, object]] = []
    selected: set[str] = set()
    first_by_hash: dict[str, str] = {}
    for record, evidence in zip(records, evidence_rows, strict=True):
        duplicate_of = first_by_hash.get(record.biological_sequence_sha256)
        if duplicate_of is None:
            first_by_hash[record.biological_sequence_sha256] = record.record_id
        host_decision = evaluate_host_evidence(evidence.to_task1_host_evidence())
        state = states.get(record.scanner_record_id)
        if duplicate_of is not None:
            reasons = ["DUPLICATE_BIOLOGICAL_GENOME"]
            state = None
        elif not host_decision.allowed:
            reasons = list(host_decision.reason_codes)
            state = None
        elif state == "PASS":
            reasons = ["SEQUENCE_SAFETY_PASS"]
            selected.add(record.record_id)
        elif state == "FAIL":
            reasons = ["SEQUENCE_SAFETY_FAIL"]
        elif state == "INDETERMINATE":
            reasons = ["SEQUENCE_SAFETY_INDETERMINATE"]
        else:
            raise SFTSafetyError(f"eligible record lacks a Task 4 result: {record.record_id}")
        decisions.append(
            {
                "input_index": record.input_index,
                "record_id": record.record_id,
                "scanner_record_id": record.scanner_record_id,
                "host_domain": evidence.normalized_host_domain.value,
                "host_evidence_digest": evidence.evidence_digest,
                "duplicate_of": duplicate_of,
                "safety_state": state,
                "eligible_for_sft": record.record_id in selected,
                "reason_codes": reasons,
            }
        )
    return decisions, selected


def audit_and_filter(request: SFTAuditRequest, *, runtime: SFTAuditRuntime | None = None) -> SFTAuditResult:
    """Audit immutable conditioned input and roll back every owned output on failure."""
    output_paths = (request.safety_manifest, request.curated_output, request.audit_root)
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise SFTSafetyError("audit outputs must not already exist")
    ownership = _AuditOwnership(
        root=_claim_owned_directory(request.audit_root, label="audit root"),
        files=[],
    )
    try:
        return _audit_and_filter_owned(request, runtime=runtime, ownership=ownership)
    except BaseException:
        _rollback_owned_audit_outputs(ownership)
        raise
    finally:
        _close_audit_ownership(ownership)


def _audit_and_filter_owned(
    request: SFTAuditRequest,
    *,
    runtime: SFTAuditRuntime | None,
    ownership: _AuditOwnership,
) -> SFTAuditResult:
    """Run one audit after the public wrapper has established exclusive output ownership."""
    selected_runtime = _default_audit_runtime() if runtime is None else runtime
    audit_started_at = selected_runtime.clock()
    _require_claimed_audit_root(ownership)
    created_at = _timestamp(audit_started_at)
    _validate_authorization_time(request.authorization, audit_started_at=audit_started_at)
    source_snapshot = _capture_source_snapshot(request.source_fasta, selected_runtime.expected_source_identity)
    phix_record = _validate_phix_identity(request.phix_fasta, selected_runtime.expected_phix_identity)
    source_before = source_snapshot.sha256
    preprocess_snapshot = _capture_artifact(request.preprocess_config, label="preprocess config")
    preprocess_sha256 = preprocess_snapshot.sha256
    phix_sha256 = sha256_bytes(phix_record.original_bytes)
    records = source_snapshot.records
    table_snapshot = _load_validated_host_evidence_table(request.host_evidence_table, label="host-evidence table")
    table = table_snapshot.table
    evidence_sha256 = table_snapshot.sha256
    if [row.record_id for row in table.rows] != [record.record_id for record in records]:
        raise SFTSafetyError("host-evidence table must contain exactly one ordered row per source record")
    _validate_source_accession_bindings(records, table.rows)
    phix_table_snapshot = _load_validated_host_evidence_table(
        request.phix_host_evidence_table,
        label="PhiX host-evidence table",
    )
    phix_table = phix_table_snapshot.table
    phix_evidence_sha256 = phix_table_snapshot.sha256
    if len(phix_table.rows) != 1:
        raise SFTSafetyError("PhiX host-evidence table must contain exactly NC_001422.1")
    trusted_inputs: dict[str, Path] = {}
    for field_name in ("policy", "asset_manifest", "diamond_tool_pin", "mmseqs_tool_pin"):
        source_path = getattr(request, field_name)
        payload = _read_regular_file_bytes(source_path, label=f"Task 4 {field_name}")
        snapshot_path = _write_owned_audit_bytes(
            ownership,
            Path("trusted_inputs") / f"{field_name}-{source_path.name}",
            payload,
            label=f"Task 4 {field_name} snapshot",
        )
        trusted_inputs[field_name] = snapshot_path
    phix_scanner_input = _write_owned_audit_bytes(
        ownership,
        Path("scanner_inputs") / "phix_nc_001422_1.fna",
        phix_record.original_bytes,
        label="PhiX scanner input",
    )
    _require_claimed_audit_root(ownership)
    phix_input_snapshot = _ArtifactSnapshot(phix_scanner_input, phix_record.original_bytes, phix_sha256)
    duplicate_ids: set[str] = set()
    seen_hashes: set[str] = set()
    groups: dict[HostDomain, list[tuple[ConditionedFastaRecord, HostEvidenceTableRow]]] = {
        domain: [] for domain in _DOMAIN_ORDER
    }
    for record, evidence in zip(records, table.rows, strict=True):
        if record.biological_sequence_sha256 in seen_hashes:
            duplicate_ids.add(record.record_id)
            continue
        seen_hashes.add(record.biological_sequence_sha256)
        if evaluate_host_evidence(evidence.to_task1_host_evidence()).allowed:
            groups[evidence.normalized_host_domain].append((record, evidence))
    all_states: dict[str, str] = {}
    children: list[dict[str, object]] = []
    for domain in _DOMAIN_ORDER:
        grouped = groups[domain]
        if not grouped:
            continue
        label = domain.value.lower()
        grouped_records = [item[0] for item in grouped]
        grouped_evidence = [item[1] for item in grouped]
        scanner_payload = b"".join(record.scanner_bytes for record in grouped_records)
        scanner_input = _write_owned_audit_bytes(
            ownership,
            Path("scanner_inputs") / f"{label}.fna",
            scanner_payload,
            label=f"{label} scanner input",
        )
        scanner_snapshot = _ArtifactSnapshot(scanner_input, scanner_payload, sha256_bytes(scanner_payload))
        expected_ids = [record.scanner_record_id for record in grouped_records]
        states, scan_snapshot, filter_snapshot = _run_bound_task4_pair(
            ownership=ownership,
            runtime=selected_runtime,
            label=label,
            input_fasta=scanner_input,
            input_snapshot=scanner_snapshot,
            output_root=request.audit_root / "task4" / label,
            host_domain=domain,
            host_evidence=_combined_host_evidence(grouped_evidence, domain, table.table_id),
            trusted_inputs=trusted_inputs,
            expected_ids=expected_ids,
        )
        all_states.update(states)
        children.append(
            {
                "label": label,
                "host_domain": domain.value,
                "input_fasta": {
                    **_artifact_ref_from_digest(scanner_input, scanner_snapshot.sha256),
                    "count": len(expected_ids),
                    "record_ids": expected_ids,
                },
                "scan_manifest": _artifact_ref_from_digest(scan_snapshot.path, scan_snapshot.sha256),
                "filter_manifest": _artifact_ref_from_digest(filter_snapshot.path, filter_snapshot.sha256),
            }
        )

    if phix_record.sequence_id != "NC_001422.1":
        raise SFTSafetyError("PhiX reference must be accession NC_001422.1")
    phix_evidence = phix_table.rows[0]
    if (
        phix_evidence.record_id != "NC_001422.1"
        or phix_evidence.accession != "NC_001422.1"
        or phix_evidence.normalized_host_domain is not HostDomain.BACTERIA
        or not evaluate_host_evidence(phix_evidence.to_task1_host_evidence()).allowed
    ):
        raise SFTSafetyError("PhiX requires positive versioned bacterial replication-host evidence")
    phix_states, phix_scan_snapshot, phix_filter_snapshot = _run_bound_task4_pair(
        ownership=ownership,
        runtime=selected_runtime,
        label="phix_nc_001422_1",
        input_fasta=phix_scanner_input,
        input_snapshot=phix_input_snapshot,
        output_root=request.audit_root / "task4" / "phix_nc_001422_1",
        host_domain=HostDomain.BACTERIA,
        host_evidence=phix_evidence.to_task1_host_evidence(),
        trusted_inputs=trusted_inputs,
        expected_ids=["NC_001422.1"],
    )
    phix_state = phix_states["NC_001422.1"]
    if phix_state != "PASS":
        exit_code = 2 if phix_state == "FAIL" else 3
        raise SFTSafetyError(
            f"PhiX NC_001422.1 requires independent Task 4 PASS; got {phix_state}", exit_code=exit_code
        )

    decisions, selected_ids = _decision_rows(records, table.rows, all_states)
    curated_payload = conditioned_records_bytes(records, selected_ids)
    distinct_output_hashes = {
        record.biological_sequence_sha256 for record in records if record.record_id in selected_ids
    }
    adequacy = assess_corpus_adequacy(
        len(distinct_output_hashes),
        authorization=request.authorization,
        minimum=request.minimum_genomes,
        preferred=request.preferred_genomes,
    )
    source_after = _sha256_file(request.source_fasta)
    if source_after != source_before:
        raise SFTSafetyError("immutable Zenodo source changed during audit")
    if _sha256_file(request.host_evidence_table) != evidence_sha256:
        raise SFTSafetyError("host-evidence table changed during audit")
    if _sha256_file(request.phix_fasta) != phix_sha256:
        raise SFTSafetyError("PhiX reference changed during audit")
    if _sha256_file(request.phix_host_evidence_table) != phix_evidence_sha256:
        raise SFTSafetyError("PhiX host-evidence table changed during audit")
    if _sha256_file(request.preprocess_config) != preprocess_sha256:
        raise SFTSafetyError("preprocess config changed during audit")
    _require_claimed_audit_root(ownership)
    ownership.files.append(_publish_owned_bytes(request.curated_output, curated_payload, label="curated SFT output"))
    _require_claimed_audit_root(ownership)
    completion = "READY" if adequacy.ready and selected_ids else "BLOCKED"
    completed_at = _timestamp(selected_runtime.clock())
    _require_claimed_audit_root(ownership)
    manifest = {
        "schema_version": 1,
        "manifest_type": "microviridae_sft_safety",
        "created_at": created_at,
        "completed_at": completed_at,
        "source": {
            "path": str(request.source_fasta),
            "sha256_before": source_before,
            "sha256_after": source_after,
            "record_count": len(records),
            "immutable": True,
            "published_identity": selected_runtime.expected_source_identity.to_dict(),
        },
        "quality": {
            "valid_record_count": len(records),
            "distinct_input_genome_hash_count": len(seen_hashes),
            "duplicate_count": len(duplicate_ids),
        },
        "host_evidence_table": {
            **_artifact_ref_from_digest(request.host_evidence_table, evidence_sha256),
            "schema_version": HOST_EVIDENCE_SCHEMA_VERSION,
            "table_id": table.table_id,
            "row_count": len(table.rows),
            "sources": sorted({row.evidence_source for row in table.rows}),
        },
        "conditioning_lineage": [_lineage_record(record) for record in records],
        "conditioning_summary": _conditioning_summary(records),
        "domain_children": children,
        "record_decisions": decisions,
        "curated_output": {
            **_artifact_ref_from_digest(request.curated_output, sha256_bytes(curated_payload)),
            "count": len(selected_ids),
            "record_ids": [record.record_id for record in records if record.record_id in selected_ids],
            "distinct_genome_hash_count": len(distinct_output_hashes),
            "preserves_original_conditioned_source_bytes": True,
        },
        "phix_reference": {
            "accession": "NC_001422.1",
            "state": phix_state,
            "reference_identity": selected_runtime.expected_phix_identity.to_dict(),
            "source_fasta": {**_artifact_ref_from_digest(request.phix_fasta, phix_sha256), "count": 1},
            "input_fasta": {**_artifact_ref_from_digest(phix_scanner_input, phix_sha256), "count": 1},
            "host_evidence_table": {
                **_artifact_ref_from_digest(request.phix_host_evidence_table, phix_evidence_sha256),
                "schema_version": HOST_EVIDENCE_SCHEMA_VERSION,
                "table_id": phix_table.table_id,
                "row_count": len(phix_table.rows),
                "sources": sorted({row.evidence_source for row in phix_table.rows}),
            },
            "host_evidence": phix_evidence.to_dict(),
            "scan_manifest": _artifact_ref_from_digest(phix_scan_snapshot.path, phix_scan_snapshot.sha256),
            "filter_manifest": _artifact_ref_from_digest(phix_filter_snapshot.path, phix_filter_snapshot.sha256),
        },
        "adequacy": {
            "state": adequacy.state.value,
            "distinct_genome_count": adequacy.distinct_genome_count,
            "minimum_genomes": request.minimum_genomes,
            "preferred_genomes": request.preferred_genomes,
            "preferred_is_non_blocking": True,
            "ready": adequacy.ready,
        },
        "authorization": None if adequacy.authorization is None else adequacy.authorization.to_dict(),
        "readiness": {
            "state": completion,
            "reason_codes": [] if completion == "READY" else ["CORPUS_BELOW_MINIMUM_WITHOUT_AUTHORIZATION"],
        },
        "preprocess": {
            **_artifact_ref_from_digest(request.preprocess_config, preprocess_sha256),
            "datapath": str(request.curated_output),
            "safety_manifest": str(request.safety_manifest),
            "output_prefix": "microviridae_sft_safety_pass",
        },
        "claim_boundary": _SFT_CLAIM_BOUNDARY,
    }
    manifest_payload = yaml.safe_dump(dict(manifest), sort_keys=False, allow_unicode=False).encode()
    _require_claimed_audit_root(ownership)
    ownership.files.append(
        _publish_owned_bytes(request.safety_manifest, manifest_payload, label="SFT SAFETY_MANIFEST")
    )
    _require_claimed_audit_root(ownership)
    validate_safety_manifest(
        request.safety_manifest,
        task4_manifest_validator=selected_runtime.task4_manifest_validator,
        require_ready=False,
        expected_minimum_genomes=request.minimum_genomes,
        expected_preferred_genomes=request.preferred_genomes,
        expected_source_identity=selected_runtime.expected_source_identity,
        expected_phix_identity=selected_runtime.expected_phix_identity,
    )
    _require_claimed_audit_root(ownership)
    return SFTAuditResult(
        exit_code=0 if completion == "READY" else 3,
        readiness=completion,
        safety_manifest=request.safety_manifest,
        curated_output=request.curated_output,
    )


def assess_corpus_adequacy(
    distinct_genome_count: int,
    *,
    authorization: CorpusCountAuthorization | None = None,
    minimum: int = MINIMUM_SFT_GENOMES,
    preferred: int = PREFERRED_SFT_GENOMES,
) -> CorpusAdequacyDecision:
    """Apply the exact preferred, minimum, and scoped below-minimum semantics."""
    if type(distinct_genome_count) is not int or distinct_genome_count < 0:
        raise SFTSafetyError("distinct genome count must be a non-negative integer")
    if type(minimum) is not int or type(preferred) is not int or minimum < 1 or preferred < minimum:
        raise SFTSafetyError("corpus thresholds must satisfy 1 <= minimum <= preferred")
    if distinct_genome_count >= preferred:
        return CorpusAdequacyDecision(CorpusAdequacy.PREFERRED_SIZE_MET, distinct_genome_count, True, None)
    if distinct_genome_count >= minimum:
        return CorpusAdequacyDecision(CorpusAdequacy.MINIMUM_SIZE_MET, distinct_genome_count, True, None)
    if authorization is not None:
        if (
            authorization.kind is CorpusAuthorizationKind.EXPLICIT_COUNT_OVERRIDE
            and distinct_genome_count < authorization.minimum_accepted_count
        ):
            return CorpusAdequacyDecision(CorpusAdequacy.BLOCKED_BELOW_MINIMUM, distinct_genome_count, False, None)
        return CorpusAdequacyDecision(
            CorpusAdequacy.AUTHORIZED_BELOW_MINIMUM,
            distinct_genome_count,
            True,
            authorization,
        )
    return CorpusAdequacyDecision(CorpusAdequacy.BLOCKED_BELOW_MINIMUM, distinct_genome_count, False, None)


def _parse_conditioned_fasta_payload(payload: bytes) -> tuple[ConditionedFastaRecord, ...]:
    """Parse one already-captured byte-exact conditioned FASTA payload."""
    records: list[ConditionedFastaRecord] = []
    seen_ids: set[str] = set()
    lines = payload.splitlines(keepends=True)
    current_start: int | None = None
    current_lines: list[bytes] = []
    offset = 0

    def finish(end: int) -> None:
        if current_start is None:
            return
        original = payload[current_start:end]
        header = current_lines[0].rstrip(b"\r\n")
        try:
            header_body = header[1:].strip().decode("ascii")
        except UnicodeDecodeError as error:
            raise SFTSafetyError("conditioned FASTA headers must be ASCII") from error
        if not header_body:
            raise SFTSafetyError("conditioned FASTA header is empty")
        record_id = header_body.split(None, 1)[0]
        if record_id in seen_ids:
            raise SFTSafetyError(f"duplicate conditioned FASTA record ID: {record_id}")
        conditioned = b"".join(b"".join(current_lines[1:]).split())
        prefix = conditioned[:2]
        if prefix not in ALLOWED_SFT_CONDITIONING_PREFIXES:
            raise SFTSafetyError(f"record {record_id} lacks one allowed conditioning prefix")
        biological = conditioned[2:]
        if biological[:2] in ALLOWED_SFT_CONDITIONING_PREFIXES:
            raise SFTSafetyError(f"record {record_id} has a repeated conditioning prefix")
        try:
            biological_text = biological.decode("ascii").upper()
        except UnicodeDecodeError as error:
            raise SFTSafetyError(f"record {record_id} biological sequence must be ASCII ACGTN") from error
        if not biological_text or re.fullmatch(r"[ACGTN]+", biological_text) is None:
            raise SFTSafetyError(f"record {record_id} biological sequence must contain only ACGTN")
        scanner_id = f"sft_{len(records):06d}"
        scanner_bytes = f">{scanner_id}\n{biological_text}\n".encode("ascii")
        records.append(
            ConditionedFastaRecord(
                input_index=len(records),
                record_id=record_id,
                scanner_record_id=scanner_id,
                original_bytes=original,
                source_start=current_start,
                source_end=end,
                conditioning_prefix=prefix,
                conditioned_sequence_sha256=sha256_bytes(conditioned),
                biological_sequence=biological_text,
                biological_sequence_sha256=sha256_bytes(biological_text.encode("ascii")),
                source_record_sha256=sha256_bytes(original),
                scanner_record_sha256=sha256_bytes(scanner_bytes),
            )
        )
        seen_ids.add(record_id)

    for line in lines:
        if line.startswith(b">"):
            finish(offset)
            current_start = offset
            current_lines = [line]
        else:
            if current_start is None:
                raise SFTSafetyError("bytes precede the first conditioned FASTA header")
            current_lines.append(line)
        offset += len(line)
    finish(len(payload))
    if not records:
        raise SFTSafetyError("conditioned FASTA is empty")
    return tuple(records)


def parse_conditioned_fasta(path: str | Path) -> tuple[ConditionedFastaRecord, ...]:
    """Parse byte-exact SFT FASTA and remove one reviewed two-byte token for scanning."""
    source = Path(os.path.abspath(os.fspath(path)))
    return _parse_conditioned_fasta_payload(_read_regular_file_bytes(source, label="conditioned FASTA"))


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_scanner_fasta(path: str | Path, records: Sequence[ConditionedFastaRecord]) -> Path:
    """Write deterministic unconditioned Task 4 input records."""
    destination = Path(path)
    _write_bytes_atomic(destination, b"".join(record.scanner_bytes for record in records))
    return destination


def conditioned_records_bytes(records: Sequence[ConditionedFastaRecord], selected_ids: set[str]) -> bytes:
    """Select original source bytes in source order without reconstructing conditioned records."""
    known_ids = {record.record_id for record in records}
    if not selected_ids <= known_ids:
        raise SFTSafetyError("selected conditioned record IDs are not an exact source subset")
    return b"".join(record.original_bytes for record in records if record.record_id in selected_ids)


def _recipe_relative(path: Path) -> str:
    """Return a path relative to the recipe root when possible."""
    path = Path(path)
    try:
        return path.resolve().relative_to(RECIPE_ROOT).as_posix()
    except ValueError:
        return str(path)


def _production_download_identity(url: str, destination: Path) -> SFTSourceIdentity | None:
    """Return the published identity only for the exact production processed-file request."""
    if url == DEFAULT_SFT_PROCESSED_URL and destination == Path(DEFAULT_SFT_PROCESSED).absolute():
        return PRODUCTION_SFT_SOURCE_IDENTITY
    return None


def _download(url: str, output_path: Path, *, overwrite: bool = False) -> Path:
    """Download through exclusive same-directory staging and publish without clobbering."""
    output_path = Path(output_path)
    destination = output_path.absolute()
    expected_identity = _production_download_identity(url, destination)
    if overwrite:
        raise SFTSafetyError("downloaded Zenodo source files are immutable and cannot be overwritten")
    _reject_symlink_components(destination, label="Zenodo download destination")
    if destination.exists():
        _read_regular_file_bytes(destination, label="downloaded Zenodo source")
        if expected_identity is not None:
            _validate_source_identity(destination, expected_identity)
        return output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination, label="Zenodo download destination")
    parent_descriptor = _open_directory_descriptor(destination.parent, label="Zenodo download destination parent")
    descriptor, staging_name, staging = _create_bound_staging(parent_descriptor, destination.name)
    linked_inode: tuple[int, int] | None = None
    publication_complete = False
    try:
        urllib.request.urlretrieve(url, staging)
        try:
            descriptor_stat = os.fstat(descriptor)
            staging_stat = os.stat(staging_name, dir_fd=parent_descriptor, follow_symlinks=False)
        except OSError as error:
            raise SFTSafetyError(f"cannot inspect Zenodo download staging file: {error}") from error
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(staging_stat.st_mode)
            or (descriptor_stat.st_dev, descriptor_stat.st_ino) != (staging_stat.st_dev, staging_stat.st_ino)
        ):
            raise SFTSafetyError("Zenodo download staging identity changed")
        os.fsync(descriptor)
        try:
            os.link(
                staging_name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise SFTSafetyError("Zenodo download destination appeared during no-clobber publication") from error
        linked_inode = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        published_stat = os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(published_stat.st_mode) or (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            published_stat.st_dev,
            published_stat.st_ino,
        ):
            raise SFTSafetyError("published Zenodo download identity changed")
        os.fsync(parent_descriptor)
        if expected_identity is not None:
            _validate_source_identity_payload(
                destination,
                _read_descriptor_bytes(descriptor, label="downloaded Zenodo source"),
                expected_identity,
            )
        _require_visible_bound_entry(
            destination,
            parent_descriptor=parent_descriptor,
            expected_identity=linked_inode,
            label="Zenodo download destination",
        )
        publication_complete = True
        return output_path
    finally:
        if linked_inode is not None and not publication_complete:
            try:
                current_stat = os.stat(destination.name, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (current_stat.st_dev, current_stat.st_ino) == linked_inode:
                    os.unlink(destination.name, dir_fd=parent_descriptor)
        os.close(descriptor)
        try:
            os.unlink(staging_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def prepare_sft_data(*, include_raw: bool = False, overwrite: bool = False) -> list[Path]:
    """Download Microviridae SFT FASTA files from the Zenodo paper record."""
    if overwrite:
        raise SFTSafetyError("downloaded Zenodo source files are immutable and cannot be overwritten")
    paths = [_download(DEFAULT_SFT_PROCESSED_URL, DEFAULT_SFT_PROCESSED, overwrite=False)]
    if include_raw:
        paths.append(_download(DEFAULT_SFT_RAW_URL, DEFAULT_SFT_RAW, overwrite=False))
    return paths


def _original_fasta_header(original_bytes: bytes, *, label: str) -> str:
    """Decode the complete first FASTA header line retained by a strict parser."""
    first_line = original_bytes.splitlines()[0]
    if not first_line.startswith(b">"):
        raise SFTSafetyError(f"{label} does not start with a FASTA header")
    try:
        return first_line[1:].strip().decode("ascii")
    except UnicodeDecodeError as error:
        raise SFTSafetyError(f"{label} header must be ASCII") from error


def _resolve_ncbi_row(
    *,
    record_id: str,
    header: str,
    cache_dir: Path,
    ncbi_fetcher: Callable[[str], bytes] | None,
    clock: Callable[[], datetime],
) -> HostEvidenceTableRow:
    """Use the production resolver while retaining a bounded injectable test boundary."""
    if ncbi_fetcher is None:
        return resolve_ncbi_host_evidence(
            record_id=record_id,
            header=header,
            cache_dir=cache_dir,
            clock=clock,
        )
    return resolve_ncbi_host_evidence(
        record_id=record_id,
        header=header,
        cache_dir=cache_dir,
        fetcher=ncbi_fetcher,
        clock=clock,
    )


def _publish_validated_host_evidence_table(
    destination: Path,
    table: HostEvidenceTable,
    *,
    label: str,
) -> Path:
    """Validate a same-directory staging file, then publish it atomically without clobbering."""
    destination = Path(destination).absolute()
    _reject_symlink_components(destination, label=f"{label} destination")
    if destination.exists() or destination.is_symlink():
        raise SFTSafetyError(f"{label} destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination, label=f"{label} destination")
    payload = yaml.safe_dump(table.to_dict(), sort_keys=False, allow_unicode=False).encode()
    try:
        validated = load_host_evidence_table_snapshot(destination, payload=payload)
        validate_host_evidence_artifacts(validated.table, table_path=destination)
    except HostEvidenceError as error:
        raise SFTSafetyError(f"{label} is untrusted: {error}") from error
    if validated.table != table:
        raise SFTSafetyError(f"{label} changed during deterministic serialization")
    owned = _publish_owned_bytes(destination, payload, label=f"{label} destination")
    os.close(owned.parent_descriptor)
    return destination


def prepare_host_evidence_table(
    *,
    source_fasta: str | Path,
    output_table: str | Path,
    cache_dir: str | Path,
    table_id: str,
    supplemental_table: str | Path | None = None,
    ncbi_fetcher: Callable[[str], bytes] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Path:
    """Build one ordered corpus table from NCBI accessions plus reviewed supplemental rows."""
    records = parse_conditioned_fasta(source_fasta)
    supplemental_by_id: dict[str, HostEvidenceTableRow] = {}
    if supplemental_table is not None:
        supplemental_path = Path(supplemental_table).absolute()
        reviewed_snapshot = _load_validated_host_evidence_table(
            supplemental_path,
            label="supplemental host-evidence table",
        )
        reviewed = reviewed_snapshot.table
        supplemental_by_id = {row.record_id: row for row in reviewed.rows}

    headers = {
        record.record_id: _original_fasta_header(
            record.original_bytes,
            label=f"source record {record.record_id}",
        )
        for record in records
    }
    source_ids = {record.record_id for record in records}
    extra_supplemental = set(supplemental_by_id) - source_ids
    if extra_supplemental:
        raise SFTSafetyError("supplemental host-evidence table contains rows outside the source corpus")

    rows: list[HostEvidenceTableRow] = []
    for record in records:
        header = headers[record.record_id]
        try:
            accession = extract_accession(header)
        except HostEvidenceError as error:
            raise SFTSafetyError(f"cannot parse source header accession: {error}") from error
        if accession is not None:
            if record.record_id in supplemental_by_id:
                raise SFTSafetyError("supplemental rows may only cover source headers without an NCBI accession")
            row = _resolve_ncbi_row(
                record_id=record.record_id,
                header=header,
                cache_dir=Path(cache_dir),
                ncbi_fetcher=ncbi_fetcher,
                clock=clock,
            )
        else:
            row = supplemental_by_id.pop(record.record_id, None)
            if row is None:
                raise SFTSafetyError(
                    f"source record {record.record_id} requires a reviewed supplemental host-evidence row"
                )
            if row.accession is not None:
                raise SFTSafetyError(
                    "a supplemental row for a non-accession source record must not invent an accession"
                )
        rows.append(row)

    if supplemental_by_id:
        raise SFTSafetyError("supplemental host-evidence rows were not consumed exactly once")
    _validate_source_accession_bindings(records, rows)
    destination = Path(output_table).absolute()
    table = HostEvidenceTable(table_id=table_id, created_at=clock(), rows=tuple(rows))
    return _publish_validated_host_evidence_table(
        destination,
        table,
        label="generated host-evidence table",
    )


def prepare_phix_host_evidence_table(
    *,
    phix_fasta: str | Path,
    output_table: str | Path,
    cache_dir: str | Path,
    table_id: str,
    ncbi_fetcher: Callable[[str], bytes] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    expected_identity: PhiXReferenceIdentity = PRODUCTION_PHIX_REFERENCE_IDENTITY,
) -> Path:
    """Build the independently resolved PhiX table after authenticating its sequence identity."""
    source = Path(phix_fasta).absolute()
    record = _validate_phix_identity(source, expected_identity)
    header = _original_fasta_header(record.original_bytes, label="PhiX reference")
    row = _resolve_ncbi_row(
        record_id=record.sequence_id,
        header=header,
        cache_dir=Path(cache_dir),
        ncbi_fetcher=ncbi_fetcher,
        clock=clock,
    )
    if row.record_id != "NC_001422.1" or row.accession != "NC_001422.1":
        raise SFTSafetyError("PhiX host evidence must resolve exact accession NC_001422.1")
    destination = Path(output_table).absolute()
    table = HostEvidenceTable(table_id=table_id, created_at=clock(), rows=(row,))
    return _publish_validated_host_evidence_table(
        destination,
        table,
        label="generated PhiX host-evidence table",
    )


def _host_evidence_parser() -> argparse.ArgumentParser:
    parser = _SFTArgumentParser(description="Prepare strict versioned SFT replication-host evidence")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    corpus = subparsers.add_parser("corpus", help="Prepare the ordered corpus host-evidence table")
    corpus.add_argument("--source-fasta", type=Path, required=True)
    corpus.add_argument("--output-table", type=Path, required=True)
    corpus.add_argument("--cache-dir", type=Path, required=True)
    corpus.add_argument("--table-id", required=True)
    corpus.add_argument("--supplemental-table", type=Path)
    phix = subparsers.add_parser("phix", help="Prepare the independent PhiX host-evidence table")
    phix.add_argument("--phix-fasta", type=Path, required=True)
    phix.add_argument("--output-table", type=Path, required=True)
    phix.add_argument("--cache-dir", type=Path, required=True)
    phix.add_argument("--table-id", required=True)
    return parser


def host_evidence_main(
    argv: Sequence[str] | None = None,
    *,
    ncbi_fetcher: Callable[[str], bytes] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    expected_phix_identity: PhiXReferenceIdentity = PRODUCTION_PHIX_REFERENCE_IDENTITY,
) -> int:
    """Create complete corpus or PhiX host-evidence tables; never resolve silently during audit."""
    try:
        args = _host_evidence_parser().parse_args(argv)
        if args.mode == "corpus":
            output = prepare_host_evidence_table(
                source_fasta=args.source_fasta,
                output_table=args.output_table,
                cache_dir=args.cache_dir,
                table_id=args.table_id,
                supplemental_table=args.supplemental_table,
                ncbi_fetcher=ncbi_fetcher,
                clock=clock,
            )
        else:
            output = prepare_phix_host_evidence_table(
                phix_fasta=args.phix_fasta,
                output_table=args.output_table,
                cache_dir=args.cache_dir,
                table_id=args.table_id,
                ncbi_fetcher=ncbi_fetcher,
                clock=clock,
                expected_identity=expected_phix_identity,
            )
        print(output)
        return 0
    except (SFTSafetyError, HostEvidenceError, OSError) as error:
        print(f"evo2_phage_prepare_sft_host_evidence: error: {error}", file=sys.stderr)
        return 3


def _reject_duplicate_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SFTSafetyError(f"duplicate key in authorization JSON: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise SFTSafetyError(f"authorization JSON contains non-finite value: {value}")


def _load_authorization(path: Path | None) -> CorpusCountAuthorization | None:
    if path is None:
        return None
    try:
        payload = json.loads(
            _read_regular_file_bytes(path, label="authorization JSON"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except SFTSafetyError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SFTSafetyError(f"cannot load authorization JSON: {error}") from error
    return CorpusCountAuthorization.from_dict(payload)


def _audit_parser() -> argparse.ArgumentParser:
    parser = _SFTArgumentParser(description="Audit and safety-filter immutable Microviridae SFT data")
    parser.add_argument("--source-fasta", type=Path, required=True)
    parser.add_argument("--host-evidence-table", type=Path, required=True)
    parser.add_argument("--phix-fasta", type=Path, required=True)
    parser.add_argument("--phix-host-evidence-table", type=Path, required=True)
    parser.add_argument("--curated-output", type=Path, required=True)
    parser.add_argument("--safety-manifest", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--preprocess-config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--diamond-tool-pin", type=Path, required=True)
    parser.add_argument("--mmseqs-tool-pin", type=Path, required=True)
    parser.add_argument("--authorization-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None, *, runtime: SFTAuditRuntime | None = None) -> int:
    """Download immutable source data or run the complete Task 4-gated SFT audit."""
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        if values and values[0] == "audit-and-filter":
            args = _audit_parser().parse_args(values[1:])
            result = audit_and_filter(
                SFTAuditRequest(
                    source_fasta=args.source_fasta,
                    host_evidence_table=args.host_evidence_table,
                    phix_fasta=args.phix_fasta,
                    phix_host_evidence_table=args.phix_host_evidence_table,
                    curated_output=args.curated_output,
                    safety_manifest=args.safety_manifest,
                    audit_root=args.audit_root,
                    preprocess_config=args.preprocess_config,
                    policy=args.policy,
                    asset_manifest=args.asset_manifest,
                    diamond_tool_pin=args.diamond_tool_pin,
                    mmseqs_tool_pin=args.mmseqs_tool_pin,
                    authorization=_load_authorization(args.authorization_json),
                ),
                runtime=runtime,
            )
            return result.exit_code
        if values and values[0] == "download":
            values = values[1:]
        parser = _SFTArgumentParser(description="Download immutable Zenodo Microviridae SFT FASTA files")
        parser.add_argument("--include-raw", action="store_true", help="Also download the raw SFT FASTA")
        parser.add_argument("--overwrite", action="store_true", help=argparse.SUPPRESS)
        args = parser.parse_args(values)
        for path in prepare_sft_data(include_raw=args.include_raw, overwrite=args.overwrite):
            print(_recipe_relative(path))
        return 0
    except SFTSafetyError as error:
        print(f"evo2_phage_download_sft_data: error: {error}", file=sys.stderr)
        return error.exit_code
    except (HostEvidenceError, OSError) as error:
        print(f"evo2_phage_download_sft_data: error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
