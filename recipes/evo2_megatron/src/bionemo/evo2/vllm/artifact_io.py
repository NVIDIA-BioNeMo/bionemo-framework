# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Fail-closed immutable byte snapshots for admitted JSON artifacts."""

from __future__ import annotations

import ctypes
import errno
import gzip
import hashlib
import json
import os
import stat
import uuid
import zlib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO, Literal, TypeVar


_RENAME_NOREPLACE = 1
_T = TypeVar("_T")


class ArtifactSnapshotError(ValueError):
    """An artifact could not be captured and decoded without ambiguity."""


class DuplicateJsonKeyError(ArtifactSnapshotError):
    """A JSON object contains a duplicate key."""


@dataclass(frozen=True)
class JsonSnapshot:
    """One parsed JSON value and the digest of the exact bytes that produced it."""

    path: Path
    payload: bytes
    value: Any
    sha256: str
    size_bytes: int
    device: int
    inode: int


@dataclass(frozen=True)
class JsonLinesSnapshot:
    """Parsed JSONL rows and the digest of their exact stored byte snapshot."""

    path: Path
    values: tuple[Any, ...]
    sha256: str
    size_bytes: int
    device: int
    inode: int


@dataclass(frozen=True)
class ByteSnapshot:
    """Exact bytes and file identity captured through one open descriptor."""

    path: Path
    payload: bytes
    sha256: str
    size_bytes: int
    device: int
    inode: int


@dataclass(frozen=True)
class FileDigestSnapshot:
    """Streaming digest and identity of one regular file without retained bytes."""

    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int


@dataclass(frozen=True)
class PublicationReceipt:
    """Durable identity and digest of one no-clobber published regular file."""

    schema_version: str
    state: str
    final_path: str
    parent_device: int
    parent_inode: int
    final_device: int
    final_inode: int
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FilePublicationPlan:
    """Serializable worker envelope for one coordinator-reserved final file."""

    schema_version: str
    publication_key: str
    final_path: str
    marker_path: str
    staging_directory_path: str
    parent_device: int
    parent_inode: int
    marker_device: int
    marker_inode: int
    staging_device: int
    staging_inode: int
    payload_device: int
    payload_inode: int
    external_contract_sha256: str
    payload_contract_sha256: str
    marker_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FilePublicationReservation:
    """Coordinator-owned open descriptors retained across worker execution."""

    plan: FilePublicationPlan
    parent_fd: int
    marker_fd: int
    staging_fd: int
    payload_fd: int
    closed: bool = False

    def close_without_unlinking(self) -> None:
        if self.closed:
            return
        for descriptor in (self.payload_fd, self.staging_fd, self.marker_fd, self.parent_fd):
            if descriptor >= 0:
                os.close(descriptor)
        self.payload_fd = -1
        self.staging_fd = -1
        self.marker_fd = -1
        self.parent_fd = -1
        self.closed = True


def _write_all_fd(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while staging immutable artifact bytes")
        view = view[written:]


def _digest_fd(descriptor: int, *, return_payload: bool = False) -> tuple[str, int, bytes | None]:
    if type(return_payload) is not bool:
        raise TypeError("return_payload must be a built-in bool")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size_bytes = 0
    chunks = [] if return_payload else None
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            payload = b"".join(chunks) if chunks is not None else None
            return digest.hexdigest(), size_bytes, payload
        digest.update(chunk)
        size_bytes += len(chunk)
        if chunks is not None:
            chunks.append(chunk)


def _stable_file_identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _file_publication_marker_payload(plan: FilePublicationPlan) -> bytes:
    value = plan.to_dict()
    value.pop("marker_sha256")
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _coerce_file_publication_plan(plan: FilePublicationPlan | Mapping[str, Any]) -> FilePublicationPlan:
    if isinstance(plan, FilePublicationPlan):
        value = plan.to_dict()
    elif isinstance(plan, Mapping):
        value = dict(plan)
    else:
        raise TypeError("file publication plan must be a FilePublicationPlan or mapping")
    expected_fields = {field.name for field in fields(FilePublicationPlan)}
    if set(value) != expected_fields:
        raise RuntimeError("file publication plan fields are not exact")
    if value.get("schema_version") != "evo2-file-publication-plan/v2":
        raise RuntimeError("file publication plan schema is unsupported")
    for field_name in ("publication_key", "final_path", "marker_path", "staging_directory_path"):
        if type(value.get(field_name)) is not str or not value[field_name]:
            raise RuntimeError(f"file publication plan {field_name} must be a nonempty string")
    for field_name in (
        "parent_device",
        "parent_inode",
        "marker_device",
        "marker_inode",
        "staging_device",
        "staging_inode",
        "payload_device",
        "payload_inode",
    ):
        if type(value.get(field_name)) is not int or value[field_name] < 0:
            raise RuntimeError(f"file publication plan {field_name} must be a nonnegative built-in integer")
    _require_sha256(value.get("external_contract_sha256"), label="external contract")
    _require_sha256(value.get("payload_contract_sha256"), label="payload contract")
    _require_sha256(value.get("marker_sha256"), label="marker")
    return FilePublicationPlan(**value)


def _coerce_publication_receipt(receipt: PublicationReceipt | Mapping[str, Any]) -> PublicationReceipt:
    if isinstance(receipt, PublicationReceipt):
        value = receipt.to_dict()
    elif isinstance(receipt, Mapping):
        value = dict(receipt)
    else:
        raise TypeError("publication receipt must be a PublicationReceipt or mapping")
    expected_fields = {field.name for field in fields(PublicationReceipt)}
    if set(value) != expected_fields:
        raise RuntimeError("publication receipt fields are not exact")
    return PublicationReceipt(**value)


def _open_directory_nofollow(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _rename_noreplace(
    source_dir_fd: int,
    source_name: str,
    destination_dir_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename one staged name while refusing every existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError("libc renameat2(RENAME_NOREPLACE) is required") from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_dir_fd,
        os.fsencode(source_name),
        destination_dir_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _unlink_at_if_identity(
    directory_fd: int,
    name: str,
    expected_identity: tuple[int, int],
) -> bool:
    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(observed.st_mode) or (observed.st_dev, observed.st_ino) != expected_identity:
        return False
    os.unlink(name, dir_fd=directory_fd)
    return True


def publish_file_noreplace(
    path: str | Path,
    writer: Callable[[BinaryIO], _T],
    *,
    ownership_validator: Callable[[], Any] | None = None,
    publication_recorder: Callable[[PublicationReceipt, Any], Any] | None = None,
) -> tuple[PublicationReceipt, _T]:
    """Stage, fsync, and no-clobber publish exactly the inode written by ``writer``."""
    lexical_path = Path(os.path.abspath(Path(path).expanduser()))
    lexical_path.parent.mkdir(parents=True, exist_ok=True)
    parent_path = lexical_path.parent.resolve(strict=True)
    final_name = lexical_path.name
    if not final_name or final_name in {".", ".."}:
        raise ValueError("publication path must name one file")

    parent_fd = _open_directory_nofollow(parent_path)
    staging_name = f".{final_name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    staging_fd = -1
    data_fd = -1
    final_fd = -1
    staged_identity: tuple[int, int] | None = None
    ownership_token: Any = None
    published = False
    primary: BaseException | None = None
    try:
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise RuntimeError("immutable publication parent is not a directory")
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        staging_stat = os.fstat(staging_fd)
        staging_path_stat = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(staging_stat.st_mode)
            or (staging_stat.st_dev, staging_stat.st_ino)
            != (staging_path_stat.st_dev, staging_path_stat.st_ino)
            or staging_stat.st_dev != parent_stat.st_dev
        ):
            raise RuntimeError("immutable publication staging directory identity changed")
        data_fd = os.open(
            "payload",
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=staging_fd,
        )
        initial_data_stat = os.fstat(data_fd)
        staged_identity = (initial_data_stat.st_dev, initial_data_stat.st_ino)
        if not stat.S_ISREG(initial_data_stat.st_mode):
            raise RuntimeError("immutable publication staged payload is not a regular file")
        if ownership_validator is not None:
            ownership_token = ownership_validator()
        with os.fdopen(os.dup(data_fd), "wb") as handle:
            writer_result = writer(handle)
            handle.flush()
        os.fsync(data_fd)
        data_stat_before_digest = os.fstat(data_fd)
        if (
            not stat.S_ISREG(data_stat_before_digest.st_mode)
            or (data_stat_before_digest.st_dev, data_stat_before_digest.st_ino) != staged_identity
        ):
            raise RuntimeError("immutable publication retained staged descriptor identity changed")
        staged_sha256, staged_size_bytes, _ = _digest_fd(data_fd)
        data_stat = os.fstat(data_fd)
        if _stable_file_identity(data_stat_before_digest) != _stable_file_identity(data_stat):
            raise RuntimeError("immutable publication staged payload changed while hashing")
        if staged_size_bytes != data_stat.st_size:
            raise RuntimeError("immutable publication staged payload size changed")
        os.fsync(staging_fd)
        if ownership_validator is not None:
            ownership_token = ownership_validator()
        staged_path_stat = os.stat("payload", dir_fd=staging_fd, follow_symlinks=False)
        if (staged_path_stat.st_dev, staged_path_stat.st_ino) != staged_identity:
            raise RuntimeError("immutable publication staged inode changed before publication")
        _rename_noreplace(staging_fd, "payload", parent_fd, final_name)
        published = True
        os.rmdir(staging_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        final_fd = os.open(
            final_name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        final_stat = os.fstat(final_fd)
        final_path_stat = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(final_stat.st_mode)
            or (final_stat.st_dev, final_stat.st_ino) != staged_identity
            or (final_path_stat.st_dev, final_path_stat.st_ino) != staged_identity
        ):
            raise RuntimeError("immutable publication first final open differs from staged inode")
        final_stat_before_digest = os.fstat(final_fd)
        final_sha256, final_size_bytes, _ = _digest_fd(final_fd)
        final_stat_after_digest = os.fstat(final_fd)
        if _stable_file_identity(final_stat_before_digest) != _stable_file_identity(final_stat_after_digest):
            raise RuntimeError("immutable publication final payload changed while hashing")
        if final_size_bytes != staged_size_bytes or final_sha256 != staged_sha256:
            raise RuntimeError("immutable publication final bytes differ from staged bytes")
        if ownership_validator is not None:
            ownership_token = ownership_validator()
        final_path_stat = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        if (final_path_stat.st_dev, final_path_stat.st_ino) != staged_identity:
            raise RuntimeError("immutable publication final inode changed after ownership validation")
        receipt = PublicationReceipt(
            schema_version="evo2-immutable-publication/v1",
            state="durably_published",
            final_path=str(parent_path / final_name),
            parent_device=parent_stat.st_dev,
            parent_inode=parent_stat.st_ino,
            final_device=data_stat.st_dev,
            final_inode=data_stat.st_ino,
            size_bytes=staged_size_bytes,
            sha256=staged_sha256,
        )
        if publication_recorder is not None:
            publication_recorder(receipt, ownership_token)
        final_path_stat = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        if (final_path_stat.st_dev, final_path_stat.st_ino) != staged_identity:
            raise RuntimeError("immutable publication final inode changed during receipt registration")
        return receipt, writer_result
    except BaseException as error:
        primary = error
        if published and staged_identity is not None:
            try:
                if _unlink_at_if_identity(parent_fd, final_name, staged_identity):
                    os.fsync(parent_fd)
            except BaseException as cleanup_error:
                error.add_note(f"cleaning owned failed publication failed: {cleanup_error!r}")
        raise
    finally:
        if not published and staging_fd >= 0 and staged_identity is not None:
            try:
                _unlink_at_if_identity(staging_fd, "payload", staged_identity)
            except BaseException as cleanup_error:
                if primary is not None:
                    primary.add_note(f"cleaning staged publication payload failed: {cleanup_error!r}")
        if not published and staging_fd >= 0:
            try:
                os.rmdir(staging_name, dir_fd=parent_fd)
            except BaseException as cleanup_error:
                if primary is not None:
                    primary.add_note(f"cleaning staged publication directory failed: {cleanup_error!r}")
        for descriptor in (final_fd, data_fd, staging_fd, parent_fd):
            if descriptor >= 0:
                os.close(descriptor)


def publish_bytes_noreplace(
    path: str | Path,
    payload: bytes,
    *,
    ownership_validator: Callable[[], Any] | None = None,
    publication_recorder: Callable[[PublicationReceipt, Any], Any] | None = None,
) -> PublicationReceipt:
    """No-clobber publish one exact byte string and return its durable receipt."""
    if type(payload) is not bytes:
        raise TypeError("published payload must be exact bytes")

    def writer(handle: BinaryIO) -> None:
        handle.write(payload)

    receipt, _ = publish_file_noreplace(
        path,
        writer,
        ownership_validator=ownership_validator,
        publication_recorder=publication_recorder,
    )
    return receipt


def reserve_file_publication(
    path: str | Path,
    *,
    external_contract_sha256: str,
    publication_key: str,
    payload_contract_sha256: str | None = None,
) -> tuple[FilePublicationReservation, FilePublicationPlan]:
    """Reserve one final name and its exact payload inode before worker launch."""
    _require_sha256(external_contract_sha256, label="external contract")
    if payload_contract_sha256 is None:
        payload_contract_sha256 = external_contract_sha256
    _require_sha256(payload_contract_sha256, label="payload contract")
    if type(publication_key) is not str or not publication_key:
        raise TypeError("publication_key must be a nonempty built-in string")
    lexical_path = Path(os.path.abspath(Path(path).expanduser()))
    lexical_path.parent.mkdir(parents=True, exist_ok=True)
    parent_path = lexical_path.parent.resolve(strict=True)
    final_path = parent_path / lexical_path.name
    if os.path.lexists(final_path):
        raise FileExistsError(f"rank-local publication final already exists: {final_path}")
    reservation_id = uuid.uuid4().hex
    marker_name = f".{final_path.name}.reserved.{reservation_id}.json"
    marker_path = parent_path / marker_name
    staging_name = f".{final_path.name}.reserved.{reservation_id}.staging"
    staging_path = parent_path / staging_name
    parent_fd = _open_directory_nofollow(parent_path)
    marker_fd = -1
    staging_fd = -1
    payload_fd = -1
    marker_identity: tuple[int, int] | None = None
    staging_identity: tuple[int, int] | None = None
    payload_identity: tuple[int, int] | None = None
    try:
        parent_stat = os.fstat(parent_fd)
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        staging_stat = os.fstat(staging_fd)
        staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
        if not stat.S_ISDIR(staging_stat.st_mode) or staging_stat.st_dev != parent_stat.st_dev:
            raise RuntimeError("rank-local publication staging directory is invalid")
        payload_fd = os.open(
            "payload",
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=staging_fd,
        )
        payload_stat = os.fstat(payload_fd)
        payload_identity = (payload_stat.st_dev, payload_stat.st_ino)
        if not stat.S_ISREG(payload_stat.st_mode) or payload_stat.st_size != 0:
            raise RuntimeError("rank-local reserved payload inode is invalid")
        marker_fd = os.open(
            marker_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        marker_stat = os.fstat(marker_fd)
        marker_identity = (marker_stat.st_dev, marker_stat.st_ino)
        plan_without_digest = FilePublicationPlan(
            schema_version="evo2-file-publication-plan/v2",
            publication_key=publication_key,
            final_path=str(final_path),
            marker_path=str(marker_path),
            staging_directory_path=str(staging_path),
            parent_device=parent_stat.st_dev,
            parent_inode=parent_stat.st_ino,
            marker_device=marker_stat.st_dev,
            marker_inode=marker_stat.st_ino,
            staging_device=staging_stat.st_dev,
            staging_inode=staging_stat.st_ino,
            payload_device=payload_stat.st_dev,
            payload_inode=payload_stat.st_ino,
            external_contract_sha256=external_contract_sha256,
            payload_contract_sha256=payload_contract_sha256,
            marker_sha256="0" * 64,
        )
        marker_payload = _file_publication_marker_payload(plan_without_digest)
        plan = FilePublicationPlan(
            **{
                **plan_without_digest.to_dict(),
                "marker_sha256": hashlib.sha256(marker_payload).hexdigest(),
            }
        )
        _write_all_fd(marker_fd, marker_payload)
        os.fsync(marker_fd)
        os.fsync(payload_fd)
        os.fsync(staging_fd)
        marker_path_stat = os.stat(marker_name, dir_fd=parent_fd, follow_symlinks=False)
        if (marker_path_stat.st_dev, marker_path_stat.st_ino) != marker_identity:
            raise RuntimeError("rank-local publication marker changed during reservation")
        os.fsync(parent_fd)
        reservation = FilePublicationReservation(
            plan=plan,
            parent_fd=parent_fd,
            marker_fd=marker_fd,
            staging_fd=staging_fd,
            payload_fd=payload_fd,
        )
        parent_fd = -1
        marker_fd = -1
        staging_fd = -1
        payload_fd = -1
        return reservation, plan
    except BaseException as error:
        if marker_identity is not None:
            try:
                if _unlink_at_if_identity(parent_fd, marker_name, marker_identity):
                    os.fsync(parent_fd)
            except BaseException as cleanup_error:
                error.add_note(f"cleaning owned rank-local reservation failed: {cleanup_error!r}")
        if payload_identity is not None and staging_fd >= 0:
            try:
                _unlink_at_if_identity(staging_fd, "payload", payload_identity)
            except BaseException as cleanup_error:
                error.add_note(f"cleaning owned rank-local payload failed: {cleanup_error!r}")
        if staging_identity is not None and parent_fd >= 0:
            try:
                staging_path_stat = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
                if (staging_path_stat.st_dev, staging_path_stat.st_ino) == staging_identity:
                    os.rmdir(staging_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                error.add_note(f"cleaning owned rank-local staging directory failed: {cleanup_error!r}")
        raise
    finally:
        if payload_fd >= 0:
            os.close(payload_fd)
        if staging_fd >= 0:
            os.close(staging_fd)
        if marker_fd >= 0:
            os.close(marker_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def validate_file_publication_plan(
    plan: FilePublicationPlan | Mapping[str, Any],
) -> FilePublicationPlan:
    """Validate a worker plan against its marker and preowned payload inode."""
    normalized = _coerce_file_publication_plan(plan)
    final_path = Path(os.path.abspath(Path(normalized.final_path).expanduser()))
    marker_path = Path(os.path.abspath(Path(normalized.marker_path).expanduser()))
    staging_path = Path(os.path.abspath(Path(normalized.staging_directory_path).expanduser()))
    if (
        str(final_path) != normalized.final_path
        or str(marker_path) != normalized.marker_path
        or str(staging_path) != normalized.staging_directory_path
    ):
        raise RuntimeError("rank-local publication plan paths must be canonical absolute paths")
    if (
        final_path.parent != marker_path.parent
        or final_path.parent != staging_path.parent
        or len({final_path, marker_path, staging_path}) != 3
    ):
        raise RuntimeError("rank-local publication path relationship is invalid")
    parent_stat = os.stat(final_path.parent, follow_symlinks=False)
    if not stat.S_ISDIR(parent_stat.st_mode) or (parent_stat.st_dev, parent_stat.st_ino) != (
        normalized.parent_device,
        normalized.parent_inode,
    ):
        raise RuntimeError("rank-local publication parent identity changed")
    snapshot = read_json_snapshot(marker_path, label="rank-local publication reservation marker")
    expected_payload = _file_publication_marker_payload(normalized)
    if (
        snapshot.payload != expected_payload
        or snapshot.sha256 != normalized.marker_sha256
        or (snapshot.device, snapshot.inode) != (normalized.marker_device, normalized.marker_inode)
        or snapshot.value != parse_json_bytes(expected_payload, label="expected rank-local reservation marker")
    ):
        raise RuntimeError("rank-local publication reservation marker does not match its worker envelope")
    staging_stat = os.stat(staging_path, follow_symlinks=False)
    if not stat.S_ISDIR(staging_stat.st_mode) or (staging_stat.st_dev, staging_stat.st_ino) != (
        normalized.staging_device,
        normalized.staging_inode,
    ):
        raise RuntimeError("rank-local publication staging directory identity changed")
    staged_payload_path = staging_path / "payload"
    staged_exists = os.path.lexists(staged_payload_path)
    final_exists = os.path.lexists(final_path)
    if staged_exists == final_exists:
        raise RuntimeError("rank-local publication payload must occupy exactly one reserved location")
    payload_path = staged_payload_path if staged_exists else final_path
    payload_stat = os.stat(payload_path, follow_symlinks=False)
    if not stat.S_ISREG(payload_stat.st_mode) or (payload_stat.st_dev, payload_stat.st_ino) != (
        normalized.payload_device,
        normalized.payload_inode,
    ):
        raise RuntimeError("rank-local publication payload inode changed")
    return normalized


def publish_reserved_file(
    plan: FilePublicationPlan | Mapping[str, Any],
    writer: Callable[[BinaryIO], _T],
) -> tuple[PublicationReceipt, _T]:
    """Write and publish the exact payload inode preowned by the coordinator."""
    normalized = validate_file_publication_plan(plan)
    final_path = Path(normalized.final_path)
    staging_path = Path(normalized.staging_directory_path)
    if os.path.lexists(final_path) or not os.path.lexists(staging_path / "payload"):
        raise RuntimeError("rank-local reserved payload has already been published")
    parent_fd = _open_directory_nofollow(final_path.parent)
    staging_fd = os.open(
        staging_path.name,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    payload_fd = os.open(
        "payload",
        os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=staging_fd,
    )
    final_fd = -1
    try:
        parent_stat = os.fstat(parent_fd)
        staging_stat = os.fstat(staging_fd)
        payload_stat = os.fstat(payload_fd)
        if (parent_stat.st_dev, parent_stat.st_ino) != (normalized.parent_device, normalized.parent_inode):
            raise RuntimeError("rank-local publication parent identity changed before worker write")
        if (staging_stat.st_dev, staging_stat.st_ino) != (
            normalized.staging_device,
            normalized.staging_inode,
        ):
            raise RuntimeError("rank-local publication staging identity changed before worker write")
        if (
            not stat.S_ISREG(payload_stat.st_mode)
            or (payload_stat.st_dev, payload_stat.st_ino)
            != (normalized.payload_device, normalized.payload_inode)
            or payload_stat.st_size != 0
        ):
            raise RuntimeError("rank-local publication payload is not a fresh coordinator-owned inode")
        with os.fdopen(os.dup(payload_fd), "wb") as handle:
            writer_result = writer(handle)
            handle.flush()
        os.fsync(payload_fd)
        before_digest = os.fstat(payload_fd)
        payload_sha256, payload_size_bytes, _ = _digest_fd(payload_fd)
        after_digest = os.fstat(payload_fd)
        if (
            _stable_file_identity(before_digest) != _stable_file_identity(after_digest)
            or (after_digest.st_dev, after_digest.st_ino)
            != (normalized.payload_device, normalized.payload_inode)
            or payload_size_bytes != after_digest.st_size
        ):
            raise RuntimeError("rank-local reserved payload changed while hashing")
        os.fsync(staging_fd)
        validate_file_publication_plan(normalized)
        _rename_noreplace(staging_fd, "payload", parent_fd, final_path.name)
        os.fsync(parent_fd)
        final_fd = os.open(
            final_path.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        final_stat = os.fstat(final_fd)
        if (
            not stat.S_ISREG(final_stat.st_mode)
            or (final_stat.st_dev, final_stat.st_ino)
            != (normalized.payload_device, normalized.payload_inode)
        ):
            raise RuntimeError("rank-local final differs from the coordinator-owned payload inode")
        final_sha256, final_size_bytes, _ = _digest_fd(final_fd)
        if final_sha256 != payload_sha256 or final_size_bytes != payload_size_bytes:
            raise RuntimeError("rank-local final bytes differ from the staged payload")
        validate_file_publication_plan(normalized)
        return (
            PublicationReceipt(
                schema_version="evo2-immutable-publication/v1",
                state="durably_published",
                final_path=normalized.final_path,
                parent_device=normalized.parent_device,
                parent_inode=normalized.parent_inode,
                final_device=normalized.payload_device,
                final_inode=normalized.payload_inode,
                size_bytes=payload_size_bytes,
                sha256=payload_sha256,
            ),
            writer_result,
        )
    finally:
        for descriptor in (final_fd, payload_fd, staging_fd, parent_fd):
            if descriptor >= 0:
                os.close(descriptor)


def publish_reserved_bytes(
    plan: FilePublicationPlan | Mapping[str, Any],
    payload: bytes,
) -> PublicationReceipt:
    """Worker-side exact-byte publication under one coordinator-owned reservation."""
    if type(payload) is not bytes:
        raise TypeError("reserved publication payload must be exact bytes")

    def writer(handle: BinaryIO) -> None:
        handle.write(payload)

    receipt, _ = publish_reserved_file(plan, writer)
    return receipt


def finalize_reserved_publication(
    reservation: FilePublicationReservation,
    receipt: PublicationReceipt | Mapping[str, Any],
) -> PublicationReceipt:
    """Coordinator-side reopen, rehash, marker removal, and terminal fsync."""
    if not isinstance(reservation, FilePublicationReservation) or reservation.closed:
        raise RuntimeError("rank-local publication reservation is unavailable")
    plan = validate_file_publication_plan(reservation.plan)
    normalized_receipt = _coerce_publication_receipt(receipt)
    if normalized_receipt.final_path != plan.final_path:
        raise RuntimeError("rank-local publication receipt names an unreserved final path")
    parent_stat = os.fstat(reservation.parent_fd)
    marker_stat = os.fstat(reservation.marker_fd)
    staging_stat = os.fstat(reservation.staging_fd)
    payload_stat = os.fstat(reservation.payload_fd)
    if (parent_stat.st_dev, parent_stat.st_ino) != (plan.parent_device, plan.parent_inode):
        raise RuntimeError("rank-local publication retained parent descriptor changed")
    if (marker_stat.st_dev, marker_stat.st_ino) != (plan.marker_device, plan.marker_inode):
        raise RuntimeError("rank-local publication retained marker descriptor changed")
    if (staging_stat.st_dev, staging_stat.st_ino) != (plan.staging_device, plan.staging_inode):
        raise RuntimeError("rank-local publication retained staging descriptor changed")
    if (payload_stat.st_dev, payload_stat.st_ino) != (plan.payload_device, plan.payload_inode):
        raise RuntimeError("rank-local publication retained payload descriptor changed")
    validate_publication_receipt(normalized_receipt)
    marker_name = Path(plan.marker_path).name
    marker_path_stat = os.stat(marker_name, dir_fd=reservation.parent_fd, follow_symlinks=False)
    if (marker_path_stat.st_dev, marker_path_stat.st_ino) != (plan.marker_device, plan.marker_inode):
        raise RuntimeError("rank-local publication marker changed before coordinator finalization")
    try:
        os.stat("payload", dir_fd=reservation.staging_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("rank-local publication staged payload still exists after worker publication")
    staging_name = Path(plan.staging_directory_path).name
    staging_path_stat = os.stat(staging_name, dir_fd=reservation.parent_fd, follow_symlinks=False)
    if (staging_path_stat.st_dev, staging_path_stat.st_ino) != (plan.staging_device, plan.staging_inode):
        raise RuntimeError("rank-local publication staging path changed before finalization")
    os.rmdir(staging_name, dir_fd=reservation.parent_fd)
    os.unlink(marker_name, dir_fd=reservation.parent_fd)
    os.fsync(reservation.parent_fd)
    validate_publication_receipt(normalized_receipt)
    reservation.close_without_unlinking()
    return normalized_receipt


def cancel_file_publication_reservation(reservation: FilePublicationReservation) -> None:
    """Terminalize a failed plan, deleting only coordinator-owned path identities."""
    if not isinstance(reservation, FilePublicationReservation) or reservation.closed:
        raise RuntimeError("rank-local publication reservation is unavailable")
    plan = reservation.plan
    errors = []
    try:
        for label, descriptor, expected in (
            ("parent", reservation.parent_fd, (plan.parent_device, plan.parent_inode)),
            ("marker", reservation.marker_fd, (plan.marker_device, plan.marker_inode)),
            ("staging", reservation.staging_fd, (plan.staging_device, plan.staging_inode)),
            ("payload", reservation.payload_fd, (plan.payload_device, plan.payload_inode)),
        ):
            observed = os.fstat(descriptor)
            if (observed.st_dev, observed.st_ino) != expected:
                errors.append(f"retained {label} descriptor became foreign")

        final_name = Path(plan.final_path).name
        try:
            final_stat = os.stat(final_name, dir_fd=reservation.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISREG(final_stat.st_mode) and (final_stat.st_dev, final_stat.st_ino) == (
                plan.payload_device,
                plan.payload_inode,
            ):
                os.unlink(final_name, dir_fd=reservation.parent_fd)
            else:
                errors.append("final path contains a foreign inode")

        try:
            staged_stat = os.stat("payload", dir_fd=reservation.staging_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISREG(staged_stat.st_mode) and (staged_stat.st_dev, staged_stat.st_ino) == (
                plan.payload_device,
                plan.payload_inode,
            ):
                os.unlink("payload", dir_fd=reservation.staging_fd)
            else:
                errors.append("staging directory contains a foreign payload inode")

        staging_name = Path(plan.staging_directory_path).name
        try:
            staging_path_stat = os.stat(staging_name, dir_fd=reservation.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISDIR(staging_path_stat.st_mode) and (
                staging_path_stat.st_dev,
                staging_path_stat.st_ino,
            ) == (plan.staging_device, plan.staging_inode):
                try:
                    os.rmdir(staging_name, dir_fd=reservation.parent_fd)
                except OSError as error:
                    errors.append(f"owned staging directory was not empty: {error!r}")
            else:
                errors.append("staging path contains a foreign inode")

        marker_name = Path(plan.marker_path).name
        try:
            marker_stat = os.stat(marker_name, dir_fd=reservation.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISREG(marker_stat.st_mode) and (marker_stat.st_dev, marker_stat.st_ino) == (
                plan.marker_device,
                plan.marker_inode,
            ):
                os.unlink(marker_name, dir_fd=reservation.parent_fd)
            else:
                errors.append("marker path contains a foreign inode")
        os.fsync(reservation.parent_fd)
    finally:
        reservation.close_without_unlinking()
    if errors:
        raise RuntimeError("; ".join(errors))


def validate_publication_receipt(
    receipt: PublicationReceipt | Mapping[str, Any],
    *,
    return_payload: bool = False,
) -> bytes | None:
    """Reopen and rehash the exact regular file named by a publication receipt."""
    if type(return_payload) is not bool:
        raise TypeError("return_payload must be a built-in bool")
    if isinstance(receipt, PublicationReceipt):
        value = receipt.to_dict()
    elif isinstance(receipt, Mapping):
        value = dict(receipt)
    else:
        raise TypeError("publication receipt must be a PublicationReceipt or mapping")
    expected_fields = {field.name for field in fields(PublicationReceipt)}
    if set(value) != expected_fields:
        raise RuntimeError("publication receipt fields are not exact")
    if value.get("schema_version") != "evo2-immutable-publication/v1" or value.get("state") != "durably_published":
        raise RuntimeError("publication receipt schema or state is unsupported")
    for field in (
        "parent_device",
        "parent_inode",
        "final_device",
        "final_inode",
        "size_bytes",
    ):
        if type(value.get(field)) is not int or value[field] < 0:
            raise RuntimeError(f"publication receipt {field} must be a nonnegative built-in integer")
    digest = value.get("sha256")
    if type(digest) is not str or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError("publication receipt SHA256 is malformed")
    if type(value.get("final_path")) is not str or not value["final_path"]:
        raise RuntimeError("publication receipt final_path must be a nonempty string")
    path = Path(os.path.abspath(Path(value["final_path"]).expanduser()))
    parent_fd = _open_directory_nofollow(path.parent)
    final_fd = -1
    try:
        parent_stat = os.fstat(parent_fd)
        if (parent_stat.st_dev, parent_stat.st_ino) != (value["parent_device"], value["parent_inode"]):
            raise RuntimeError("publication receipt parent inode changed")
        final_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        final_stat = os.fstat(final_fd)
        expected_identity = (value["final_device"], value["final_inode"])
        if not stat.S_ISREG(final_stat.st_mode) or (final_stat.st_dev, final_stat.st_ino) != expected_identity:
            raise RuntimeError("publication receipt final inode changed")
        before_digest = os.fstat(final_fd)
        observed_sha256, observed_size_bytes, payload = _digest_fd(final_fd, return_payload=return_payload)
        after_digest = os.fstat(final_fd)
        if _stable_file_identity(before_digest) != _stable_file_identity(after_digest):
            raise RuntimeError("publication receipt final file changed while hashing")
        path_stat = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (path_stat.st_dev, path_stat.st_ino) != expected_identity:
            raise RuntimeError("publication receipt final path inode changed during validation")
        if observed_size_bytes != value["size_bytes"] or observed_sha256 != digest:
            raise RuntimeError("publication receipt final bytes changed")
        return payload
    finally:
        if final_fd >= 0:
            os.close(final_fd)
        os.close(parent_fd)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJsonKeyError(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def _reject_nonstandard_constant(value: str) -> None:
    raise ArtifactSnapshotError(f"JSON contains non-standard numeric constant {value!r}")


def parse_json_bytes(payload: bytes, *, label: str) -> Any:
    """Decode strict JSON bytes while rejecting duplicate object keys."""
    if not isinstance(payload, bytes):
        raise TypeError("JSON payload must be bytes")
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except DuplicateJsonKeyError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactSnapshotError(f"{label} is not valid UTF-8 JSON") from error


def read_byte_snapshot(path: str | Path, *, label: str) -> ByteSnapshot:
    """Capture one regular file once and bind its exact bytes, size, and identity."""
    lexical_path = Path(os.path.abspath(Path(path).expanduser()))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical_path, flags)
    except OSError as error:
        raise ArtifactSnapshotError(f"{label} could not be opened as a regular file: {lexical_path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactSnapshotError(f"{label} is not a regular file: {lexical_path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise ArtifactSnapshotError(f"{label} changed while its byte snapshot was being captured")
    return ByteSnapshot(
        path=lexical_path,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        device=after.st_dev,
        inode=after.st_ino,
    )


def read_file_digest_snapshot(path: str | Path, *, label: str) -> FileDigestSnapshot:
    """Stream one regular file once and bind its digest, size, and descriptor identity."""
    lexical_path = Path(os.path.abspath(Path(path).expanduser()))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical_path, flags)
    except OSError as error:
        raise ArtifactSnapshotError(f"{label} could not be opened as a regular file: {lexical_path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactSnapshotError(f"{label} is not a regular file: {lexical_path}")
        sha256, size_bytes, _ = _digest_fd(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stable_file_identity(before) != _stable_file_identity(after) or size_bytes != after.st_size:
        raise ArtifactSnapshotError(f"{label} changed while its digest snapshot was being captured")
    return FileDigestSnapshot(
        path=lexical_path,
        sha256=sha256,
        size_bytes=size_bytes,
        device=after.st_dev,
        inode=after.st_ino,
    )


def read_json_snapshot(path: str | Path, *, label: str) -> JsonSnapshot:
    """Parse and hash one JSON artifact from the same immutable byte snapshot."""
    snapshot = read_byte_snapshot(path, label=label)
    return JsonSnapshot(
        path=snapshot.path,
        payload=snapshot.payload,
        value=parse_json_bytes(snapshot.payload, label=label),
        sha256=snapshot.sha256,
        size_bytes=snapshot.size_bytes,
        device=snapshot.device,
        inode=snapshot.inode,
    )


def read_jsonl_snapshot(
    path: str | Path,
    *,
    label: str,
    compression: Literal["gzip"] | None = None,
) -> JsonLinesSnapshot:
    """Parse and hash JSONL rows from one immutable stored-byte snapshot."""
    snapshot = read_byte_snapshot(path, label=label)
    if compression is None:
        decoded_payload = snapshot.payload
    elif compression == "gzip":
        try:
            decoded_payload = gzip.decompress(snapshot.payload)
        except (OSError, EOFError, zlib.error) as error:
            raise ArtifactSnapshotError(f"{label} is not valid gzip data") from error
    else:
        raise ValueError(f"unsupported JSONL compression: {compression!r}")

    rows = []
    for line_number, line in enumerate(decoded_payload.splitlines(), start=1):
        if not line:
            raise ArtifactSnapshotError(f"{label} line {line_number} is blank")
        rows.append(parse_json_bytes(line, label=f"{label} line {line_number}"))
    if not rows:
        raise ArtifactSnapshotError(f"{label} contains no JSONL rows")
    return JsonLinesSnapshot(
        path=snapshot.path,
        values=tuple(rows),
        sha256=snapshot.sha256,
        size_bytes=snapshot.size_bytes,
        device=snapshot.device,
        inode=snapshot.inode,
    )


__all__ = [
    "ArtifactSnapshotError",
    "ByteSnapshot",
    "DuplicateJsonKeyError",
    "FilePublicationPlan",
    "FilePublicationReservation",
    "FileDigestSnapshot",
    "JsonLinesSnapshot",
    "JsonSnapshot",
    "PublicationReceipt",
    "cancel_file_publication_reservation",
    "finalize_reserved_publication",
    "parse_json_bytes",
    "publish_bytes_noreplace",
    "publish_file_noreplace",
    "publish_reserved_bytes",
    "publish_reserved_file",
    "read_byte_snapshot",
    "read_file_digest_snapshot",
    "read_json_snapshot",
    "read_jsonl_snapshot",
    "reserve_file_publication",
    "validate_file_publication_plan",
    "validate_publication_receipt",
]
