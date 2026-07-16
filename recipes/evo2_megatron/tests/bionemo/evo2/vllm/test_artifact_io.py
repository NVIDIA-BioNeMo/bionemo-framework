# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import gzip
import hashlib
import os
import zlib
from pathlib import Path

import pytest

import bionemo.evo2.vllm.artifact_io as artifact_io
from bionemo.evo2.vllm.artifact_io import (
    ArtifactSnapshotError,
    DuplicateJsonKeyError,
    cancel_file_publication_reservation,
    finalize_reserved_publication,
    publish_bytes_noreplace,
    publish_file_noreplace,
    publish_reserved_bytes,
    read_file_digest_snapshot,
    read_json_snapshot,
    read_jsonl_snapshot,
    reserve_file_publication,
    validate_publication_receipt,
)


def test_json_snapshot_binds_parsed_value_and_digest_to_one_byte_snapshot(tmp_path) -> None:
    path = tmp_path / "proof.json"
    original = b'{"schema_version":1,"status":"passed"}'
    path.write_bytes(original)

    snapshot = read_json_snapshot(path, label="linked proof")
    replacement = b'{"schema_version":1,"status":"foreign"}'
    path.write_bytes(replacement)

    assert snapshot.value == {"schema_version": 1, "status": "passed"}
    assert snapshot.payload == original
    assert snapshot.sha256 == hashlib.sha256(original).hexdigest()
    assert snapshot.size_bytes == len(original)
    assert path.read_bytes() == replacement


@pytest.mark.parametrize(
    "payload,duplicate",
    (
        (b'{"phase":"steady","phase":"foreign"}', "phase"),
        (b'{"phase":{"rank":0,"rank":1}}', "rank"),
    ),
)
def test_json_snapshot_rejects_duplicate_object_keys(tmp_path, payload, duplicate) -> None:
    path = tmp_path / "duplicate.json"
    path.write_bytes(payload)

    with pytest.raises(DuplicateJsonKeyError, match=duplicate):
        read_json_snapshot(path, label="proof")


def test_gzip_jsonl_snapshot_hashes_compressed_bytes_and_rejects_duplicate_keys(tmp_path) -> None:
    valid_path = tmp_path / "outputs.jsonl.gz"
    valid_payload = gzip.compress(b'{"request_id":"a","seed":42}\n{"request_id":"b","seed":43}\n')
    valid_path.write_bytes(valid_payload)

    snapshot = read_jsonl_snapshot(valid_path, label="full outputs", compression="gzip")

    assert snapshot.values == (
        {"request_id": "a", "seed": 42},
        {"request_id": "b", "seed": 43},
    )
    assert snapshot.sha256 == hashlib.sha256(valid_payload).hexdigest()
    assert snapshot.size_bytes == len(valid_payload)

    duplicate_path = tmp_path / "duplicate.outputs.jsonl.gz"
    duplicate_path.write_bytes(gzip.compress(b'{"request_id":"a","seed":42,"seed":43}\n'))
    with pytest.raises(DuplicateJsonKeyError, match="seed"):
        read_jsonl_snapshot(duplicate_path, label="full outputs", compression="gzip")


def test_jsonl_snapshot_rejects_blank_rows(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"request_id":"a"}\n\n')

    with pytest.raises(ValueError, match="line 2 is blank"):
        read_jsonl_snapshot(path, label="request manifest")


def test_gzip_jsonl_snapshot_converts_zlib_failure_to_artifact_error(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rows.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"request_id":"a"}\n'))
    monkeypatch.setattr(
        artifact_io.gzip,
        "decompress",
        lambda _payload: (_ for _ in ()).throw(zlib.error("invalid deflate stream")),
    )

    with pytest.raises(ArtifactSnapshotError, match="not valid gzip"):
        read_jsonl_snapshot(path, label="full outputs", compression="gzip")


def test_noreplace_publication_binds_durable_receipt_to_staged_inode_and_bytes(tmp_path) -> None:
    output = tmp_path / "proof.json"
    payload = b'{"status":"passed"}\n'

    receipt = publish_bytes_noreplace(output, payload)

    assert output.read_bytes() == payload
    assert receipt.final_path == str(output.resolve())
    assert receipt.size_bytes == len(payload)
    assert receipt.sha256 == hashlib.sha256(payload).hexdigest()
    assert (receipt.final_device, receipt.final_inode) == (output.stat().st_dev, output.stat().st_ino)
    assert validate_publication_receipt(receipt) is None
    assert validate_publication_receipt(receipt, return_payload=True) == payload


def test_publication_and_validation_stream_large_payloads_without_retaining_bytes(tmp_path) -> None:
    output = tmp_path / "outputs.jsonl.gz"
    chunk = b"0123456789abcdef" * 65_536
    chunk_count = 5

    def writer(handle):
        for _ in range(chunk_count):
            handle.write(chunk)
        return "writer-result"

    receipt, writer_result = publish_file_noreplace(output, writer)

    assert writer_result == "writer-result"
    assert receipt.size_bytes == len(chunk) * chunk_count
    assert receipt.sha256 == hashlib.sha256(chunk * chunk_count).hexdigest()
    assert validate_publication_receipt(receipt) is None


def test_file_digest_snapshot_streams_without_retaining_payload(tmp_path) -> None:
    path = tmp_path / "weight.safetensors"
    payload = b"weight-shard" * 1_000_000
    path.write_bytes(payload)

    snapshot = read_file_digest_snapshot(path, label="checkpoint shard")

    assert snapshot.sha256 == hashlib.sha256(payload).hexdigest()
    assert snapshot.size_bytes == len(payload)
    assert (snapshot.device, snapshot.inode) == (path.stat().st_dev, path.stat().st_ino)
    assert not hasattr(snapshot, "payload")


def test_noreplace_publication_refuses_existing_final_without_clobbering(tmp_path) -> None:
    output = tmp_path / "proof.json"
    foreign = b"foreign\n"
    output.write_bytes(foreign)

    with pytest.raises(FileExistsError):
        publish_bytes_noreplace(output, b"owned\n")

    assert output.read_bytes() == foreign


def test_noreplace_publication_rejects_early_staging_inode_substitution(tmp_path, monkeypatch) -> None:
    output = tmp_path / "proof.json"
    foreign = b"foreign-staged-inode\n"
    real_rename = artifact_io._rename_noreplace

    def substitute_then_rename(source_dir_fd, source_name, destination_dir_fd, destination_name):
        os.unlink(source_name, dir_fd=source_dir_fd)
        descriptor = os.open(
            source_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=source_dir_fd,
        )
        try:
            os.write(descriptor, foreign)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        real_rename(source_dir_fd, source_name, destination_dir_fd, destination_name)

    monkeypatch.setattr(artifact_io, "_rename_noreplace", substitute_then_rename)

    with pytest.raises(RuntimeError, match="staged inode"):
        publish_bytes_noreplace(output, b"owned\n")

    assert output.read_bytes() == foreign


def test_noreplace_publication_rejects_foreign_final_substitution(tmp_path, monkeypatch) -> None:
    output = tmp_path / "proof.json"
    foreign = b"foreign-final-inode\n"
    real_rename = artifact_io._rename_noreplace

    def rename_then_replace(source_dir_fd, source_name, destination_dir_fd, destination_name):
        real_rename(source_dir_fd, source_name, destination_dir_fd, destination_name)
        foreign_name = f".{destination_name}.foreign"
        descriptor = os.open(
            foreign_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=destination_dir_fd,
        )
        try:
            os.write(descriptor, foreign)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(
            foreign_name,
            destination_name,
            src_dir_fd=destination_dir_fd,
            dst_dir_fd=destination_dir_fd,
        )

    monkeypatch.setattr(artifact_io, "_rename_noreplace", rename_then_replace)

    with pytest.raises(RuntimeError, match="staged inode"):
        publish_bytes_noreplace(output, b"owned\n")

    assert output.read_bytes() == foreign


def test_publication_receipt_rejects_same_bytes_on_foreign_inode(tmp_path) -> None:
    output = tmp_path / "proof.json"
    payload = b'{"status":"passed"}\n'
    receipt = publish_bytes_noreplace(output, payload)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(payload)
    replacement.replace(output)

    with pytest.raises(RuntimeError, match="inode"):
        validate_publication_receipt(receipt)


def test_publication_registers_receipt_before_return_and_cleans_owned_inode_on_rejection(tmp_path) -> None:
    output = tmp_path / "proof.json"
    observed = []

    def reject_receipt(receipt, ownership_token) -> None:
        observed.append(receipt)
        assert ownership_token is None
        assert validate_publication_receipt(receipt, return_payload=True) == b"owned\n"
        raise RuntimeError("coordinator rejected receipt")

    with pytest.raises(RuntimeError, match="coordinator rejected receipt"):
        publish_bytes_noreplace(output, b"owned\n", publication_recorder=reject_receipt)

    assert len(observed) == 1
    assert not output.exists()


def test_rank_file_reservation_precedes_worker_publish_and_only_coordinator_finalizes(tmp_path) -> None:
    output = tmp_path / "proof.wave-000.dp0.outputs.jsonl.gz"
    reservation, plan = reserve_file_publication(
        output,
        external_contract_sha256="a" * 64,
        publication_key="cold/wave-000/dp-0",
    )

    assert plan.final_path == str(output.resolve())
    assert os.path.lexists(plan.marker_path)
    provisional = publish_reserved_bytes(plan, b"rank-local-output\n")
    assert os.path.lexists(plan.marker_path)

    durable = finalize_reserved_publication(reservation, provisional)

    assert durable == provisional
    assert not os.path.lexists(plan.marker_path)
    assert validate_publication_receipt(durable, return_payload=True) == b"rank-local-output\n"


def test_rank_file_reservation_cancels_owned_final_after_lost_worker_response(tmp_path) -> None:
    output = tmp_path / "proof.wave-000.dp0.outputs.jsonl.gz"
    reservation, plan = reserve_file_publication(
        output,
        external_contract_sha256="a" * 64,
        publication_key="cold/wave-000/dp-0",
    )

    provisional = publish_reserved_bytes(plan, b"published-before-ray-failure\n")
    assert (provisional.final_device, provisional.final_inode) == (
        plan.payload_device,
        plan.payload_inode,
    )

    cancel_file_publication_reservation(reservation)

    assert reservation.closed is True
    assert not output.exists()
    assert not os.path.lexists(plan.marker_path)
    assert not os.path.lexists(plan.staging_directory_path)


def test_rank_file_failed_terminalization_never_unlinks_foreign_final_or_marker(tmp_path) -> None:
    output = tmp_path / "proof.wave-000.dp0.outputs.jsonl.gz"
    reservation, plan = reserve_file_publication(
        output,
        external_contract_sha256="a" * 64,
        publication_key="cold/wave-000/dp-0",
    )
    publish_reserved_bytes(plan, b"owned\n")
    owned_backup = tmp_path / "owned-backup"
    output.replace(owned_backup)
    output.write_bytes(b"foreign-final\n")
    marker = Path(plan.marker_path)
    foreign_marker = tmp_path / "foreign-marker"
    foreign_marker.write_bytes(marker.read_bytes())
    marker.unlink()
    foreign_marker.replace(marker)

    with pytest.raises(RuntimeError, match="foreign"):
        cancel_file_publication_reservation(reservation)

    assert reservation.closed is True
    assert output.read_bytes() == b"foreign-final\n"
    assert marker.exists()
    assert owned_backup.read_bytes() == b"owned\n"


def test_rank_file_reservation_rejects_mutated_worker_envelope_before_publish(tmp_path) -> None:
    output = tmp_path / "proof.wave-000.dp0.outputs.jsonl.gz"
    reservation, plan = reserve_file_publication(
        output,
        external_contract_sha256="a" * 64,
        publication_key="cold/wave-000/dp-0",
    )
    foreign = plan.to_dict()
    foreign["external_contract_sha256"] = "b" * 64
    try:
        with pytest.raises(RuntimeError, match="reservation|contract|marker"):
            publish_reserved_bytes(foreign, b"must-not-publish\n")
        assert not output.exists()
        assert os.path.lexists(plan.marker_path)
    finally:
        cancel_file_publication_reservation(reservation)


def test_rank_file_reservation_rejects_foreign_marker_inode_without_removing_it(tmp_path) -> None:
    output = tmp_path / "proof.wave-000.dp0.outputs.jsonl.gz"
    reservation, plan = reserve_file_publication(
        output,
        external_contract_sha256="a" * 64,
        publication_key="cold/wave-000/dp-0",
    )
    foreign = tmp_path / "foreign-marker"
    foreign.write_bytes(Path(plan.marker_path).read_bytes())
    foreign.replace(plan.marker_path)

    with pytest.raises(RuntimeError, match="marker|inode"):
        publish_reserved_bytes(plan, b"must-not-publish\n")

    assert Path(plan.marker_path).exists()
    assert not output.exists()
    reservation.close_without_unlinking()
