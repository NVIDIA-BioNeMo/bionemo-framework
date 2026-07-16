# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import shutil
from pathlib import Path

import pytest

import bionemo.evo2.vllm.tokenizer_io as tokenizer_io
from bionemo.evo2.vllm.tokenizer_io import SnapshotBoundTokenizer


TOKENIZERS = Path(__file__).parents[4] / "tokenizers"
TOKENIZER_512 = TOKENIZERS / "nucleotide_fast_tokenizer_512" / "tokenizer.json"
TOKENIZER_256 = TOKENIZERS / "nucleotide_fast_tokenizer_256" / "tokenizer.json"


def test_snapshot_bound_tokenizer_consumes_captured_bytes_and_retains_identity() -> None:
    tokenizer = SnapshotBoundTokenizer.from_path(TOKENIZER_512)

    assert tokenizer.encode("+~GAGTTTTATC") == tuple(map(ord, "+~GAGTTTTATC"))
    assert tokenizer.verify_source()["source_sha256"] == tokenizer.source_sha256
    assert tokenizer.provenance()["loaded_object_sha256"]


def test_snapshot_bound_tokenizer_rejects_replacement_during_load(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tokenizer.json"
    shutil.copyfile(TOKENIZER_512, path)
    replacement = TOKENIZER_256.read_bytes()
    real_reader = tokenizer_io.read_json_snapshot
    calls = 0

    def replace_after_capture(source, *, label):
        nonlocal calls
        snapshot = real_reader(source, label=label)
        calls += 1
        if calls == 1:
            path.write_bytes(replacement)
        return snapshot

    monkeypatch.setattr(tokenizer_io, "read_json_snapshot", replace_after_capture)

    with pytest.raises(RuntimeError, match="changed during tokenizer load"):
        SnapshotBoundTokenizer.from_path(path)
    assert calls == 2


def test_snapshot_bound_tokenizer_survives_later_path_replacement_but_rescan_fails(tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    shutil.copyfile(TOKENIZER_512, path)
    tokenizer = SnapshotBoundTokenizer.from_path(path)
    expected = tokenizer.encode("+~GAGTTTTATC")

    path.write_bytes(TOKENIZER_256.read_bytes())

    assert tokenizer.encode("+~GAGTTTTATC") == expected
    with pytest.raises(RuntimeError, match="no longer matches"):
        tokenizer.verify_source()
