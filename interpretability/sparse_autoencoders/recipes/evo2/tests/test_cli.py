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

"""CLI contract tests — the `encode | batch | generate` subcommands, CPU-only.

A mocked engine (no model) drives `cli.main()` so these run in CI alongside test_server.py.
They lock the stdout JSON shapes, the clean-exit error handling (no tracebacks), and the
shared clamp/annotate paths in `core` that the CLI and the server now both go through.
Real model inference is covered by test_steering.py.
"""

import json

import pandas as pd
import pytest
from evo2_sae import cli, core


# FakeEngine lives in conftest.py — shared with test_server.py so both suites mock the same surface.


@pytest.fixture
def fake(monkeypatch, fake_engine):
    """Patch the CLI's engine factory so every subcommand runs against the shared CPU fake."""
    monkeypatch.setattr(cli, "_engine", lambda args: fake_engine)
    return fake_engine


def run(monkeypatch, *argv):
    """Invoke `cli.main()` with the given argv (no leading prog name)."""
    monkeypatch.setattr("sys.argv", ["evo2-sae", *argv])
    cli.main()


# ----------------------------------------------------------------------------- encode
def test_encode_outputs_top_features(fake, monkeypatch, capsys):
    run(monkeypatch, "encode", "--sequence", "ACGTACGT")
    out = json.loads(capsys.readouterr().out)
    assert out["bases"] == 8
    assert out["top_features"][0]["feature_id"] == 0


def test_encode_rejects_non_dna(fake, monkeypatch):
    with pytest.raises(SystemExit):  # clean exit, not a traceback
        run(monkeypatch, "encode", "--sequence", "ZZZZ")


def test_encode_rejects_unknown_organism(fake, monkeypatch):
    with pytest.raises(SystemExit):
        run(monkeypatch, "encode", "--sequence", "ACGT", "--organism", "Martian")


# --------------------------------------------------------------------------- generate
def test_generate_outputs_sequence(fake, monkeypatch, capsys):
    run(monkeypatch, "generate", "--prompt", "ACGT")
    out = json.loads(capsys.readouterr().out)
    assert out["sequence"] == "ACGT"
    assert out["steered"] is False


def test_generate_passes_raw_clamp_strings(fake, monkeypatch, capsys):
    run(monkeypatch, "generate", "--prompt", "ACGT", "--clamp", "0:5", "--clamp", "1")
    # The CLI hands raw "ID[:STRENGTH]" strings straight to the engine; core normalizes them.
    assert fake.gen_kwargs["features"] == ["0:5", "1"]
    out = json.loads(capsys.readouterr().out)
    # generate() echoes feat_meta keyed {id, label, strength} — the shape the viz consumes.
    assert out["features"] == [
        {"id": 0, "label": "feat0", "strength": 5.0},
        {"id": 1, "label": "feat1", "strength": 1.0},
    ]


def test_generate_rejects_malformed_clamp(fake, monkeypatch):
    with pytest.raises(SystemExit):
        run(monkeypatch, "generate", "--prompt", "ACGT", "--clamp", "notanumber")


# ------------------------------------------------------------------------------ batch
def test_batch_writes_parquet(fake, monkeypatch, capsys, tmp_path):
    fasta = tmp_path / "in.fa"
    fasta.write_text(">a\nACGTACGT\n>b\nTTTT\n")
    out = tmp_path / "out.parquet"
    run(monkeypatch, "batch", "--fasta", str(fasta), "--out", str(out))
    df = pd.read_parquet(out)
    assert set(df["sequence_id"]) == {"a", "b"}
    assert {"sequence_id", "bp", "rank", "feature_id"} <= set(df.columns)


# ------------------------------------------------------- shared core helpers (CLI + API)
def test_parse_clamp_spec_string_and_dict():
    assert core.parse_clamp_spec("29244:300") == {"feature_id": 29244, "strength": 300.0}
    assert core.parse_clamp_spec("7") == {"feature_id": 7, "strength": 1.0}  # default strength
    assert core.parse_clamp_spec({"feature_id": 7, "strength": 2.5}) == {"feature_id": 7, "strength": 2.5}
    assert core.parse_clamp_spec({"feature_id": 7}) == {"feature_id": 7, "strength": 1.0}


def test_parse_clamp_spec_rejects_garbage():
    with pytest.raises(ValueError):
        core.parse_clamp_spec("x:y")


def test_annotate_skips_tag_and_rejects_bad_input(fake_engine):
    dna, tag, codes, tag_len = core.annotate(fake_engine, "ACGT", organism="Human")
    assert dna == "ACGT" and tag == "|tag|" and tag_len == len("|tag|")
    assert codes.shape == (len(tag) + len(dna), fake_engine.n_features)
    with pytest.raises(ValueError):
        core.annotate(fake_engine, "ZZZZ")  # non-DNA
    with pytest.raises(ValueError):
        core.annotate(fake_engine, "ACGT", organism="Martian")  # unknown organism
