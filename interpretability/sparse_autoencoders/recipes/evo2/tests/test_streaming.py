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

"""CPU guards for scripts/train_streaming.py.

These run without CUDA or bionemo.evo2: the heavy Evo2 ``predict`` import is lazy
(inside ``Evo2ActivationProducer.__call__``), so we inject a fake ``predict`` module
that drives the same per-batch writer hook the real producer monkeypatches. This
exercises the producer/queue plumbing, the --input-dim guard, and the
--init-pre-bias rejection on CPU.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_streaming.py"


def _load_train_streaming():
    """Import scripts/train_streaming.py as a module (it is not an installed package)."""
    spec = importlib.util.spec_from_file_location("train_streaming", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _install_fake_predict(monkeypatch, width: int, n_batches: int = 1):
    """Inject a fake bionemo.evo2.run.predict whose main() drives the writer hook.

    The producer sets ``predict_mod._write_predictions_batch`` then calls ``main()``;
    our fake main() invokes that writer with ``n_batches`` of ``[1, 4, width]`` activations
    (pad_mask all ones), mimicking Evo2 emitting a residual stream of the given width.
    """
    fake = types.ModuleType("bionemo.evo2.run.predict")
    fake._write_predictions_batch = None  # replaced by the producer before main() runs

    def main():
        for b in range(n_batches):
            hidden = torch.zeros(1, 4, width)
            preds = {"hidden_embeddings": hidden, "pad_mask": torch.ones(1, 4)}
            fake._write_predictions_batch(preds, "unused", b, 0, 0)

    fake.main = main

    # Provide the parent packages so `from bionemo.evo2.run import predict` resolves to the fake.
    for name in ("bionemo", "bionemo.evo2", "bionemo.evo2.run"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["bionemo.evo2.run"].predict = fake
    monkeypatch.setitem(sys.modules, "bionemo.evo2.run.predict", fake)
    return fake


def _producer_args(ts, **overrides):
    """Minimal argparse.Namespace for Evo2ActivationProducer."""
    base = dict(
        ckpt_dir="unused", fasta="unused", layer=12, input_dim=1920, micro_batch_size=4,
        max_tokens=0, dtype="fp32", queue_size=4,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_input_dim_mismatch_raises_clear_error(monkeypatch):
    """#4: a width != --input-dim fails with a clear message, not an opaque matmul error."""
    ts = _load_train_streaming()
    _install_fake_predict(monkeypatch, width=1920)
    producer = ts.Evo2ActivationProducer(_producer_args(ts, input_dim=1024))  # deliberately wrong
    with pytest.raises(ValueError, match=r"--input-dim=1024.*width 1920"):
        list(producer())


def test_input_dim_match_streams_chunks(monkeypatch):
    """Matching width streams chunks through to the consumer and terminates on the sentinel."""
    ts = _load_train_streaming()
    _install_fake_predict(monkeypatch, width=1920, n_batches=2)
    producer = ts.Evo2ActivationProducer(_producer_args(ts, input_dim=1920))
    chunks = list(producer())
    assert len(chunks) == 2
    assert all(c.shape[1] == 1920 for c in chunks)


def test_producer_propagates_predict_failure(monkeypatch):
    """A crash in the predict thread surfaces in the consumer instead of hanging."""
    ts = _load_train_streaming()
    fake = _install_fake_predict(monkeypatch, width=1920)

    def boom():
        raise RuntimeError("predict exploded")

    fake.main = boom
    producer = ts.Evo2ActivationProducer(_producer_args(ts))
    with pytest.raises(RuntimeError, match="predict exploded"):
        list(producer())


def test_init_pre_bias_is_rejected(monkeypatch):
    """#1: --init-pre-bias is unsupported on the streaming path and fails fast (no GPU needed)."""
    ts = _load_train_streaming()
    monkeypatch.setattr(
        sys, "argv",
        ["train_streaming.py", "--ckpt-dir", "x", "--fasta", "y", "--embedding-layer", "12",
         "--input-dim", "8", "--init-pre-bias", "--no-wandb"],
    )
    with pytest.raises(NotImplementedError, match="init-pre-bias"):
        ts.main()
