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

"""Tests for producer-consumer activation streaming (sae.streaming).

All tests are model-agnostic and CPU-only: the "producer" is a plain Python
generator of tensors, so nothing here loads a base model or touches a GPU.
"""

import gc
import math

import pytest
import torch
from sae.streaming import (
    StreamingActivationDataset,
    StreamingConfig,
    make_streaming_dataloader,
)


HIDDEN_DIM = 4


def make_factory(chunk_sizes, hidden_dim=HIDDEN_DIM, on_produce=None):
    """Build a producer factory yielding chunks whose rows carry a unique id.

    Row global index ``g`` becomes the tensor row ``[g, g, ...]`` so order and
    membership can be checked exactly. ``on_produce(chunk_index)`` is called as
    each chunk is produced (for instrumentation).
    """

    def factory():
        g = 0
        for ci, n in enumerate(chunk_sizes):
            if on_produce is not None:
                on_produce(ci)
            rows = torch.arange(g, g + n, dtype=torch.float32).unsqueeze(1).repeat(1, hidden_dim)
            g += n
            yield rows

    return factory


def collect(dataset):
    """Drain a streaming dataset into a single concatenated tensor of batches."""
    batches = list(iter(dataset))
    return batches


def test_all_tokens_consumed_in_order():
    chunk_sizes = [5, 7, 4]  # 16 rows total
    total = sum(chunk_sizes)
    ds = StreamingActivationDataset(make_factory(chunk_sizes), batch_size=4, config=StreamingConfig())

    batches = collect(ds)
    out = torch.cat(batches, dim=0)

    assert out.shape == (total, HIDDEN_DIM)
    # No shuffle -> exact order preserved, every token exactly once.
    expected = torch.arange(total, dtype=torch.float32).unsqueeze(1).repeat(1, HIDDEN_DIM)
    assert torch.equal(out, expected)


def test_batch_shape_and_dtype():
    ds = StreamingActivationDataset(make_factory([10, 10]), batch_size=4, config=StreamingConfig())
    batches = collect(ds)

    # 20 rows, batch 4 -> 5 full batches, no remainder.
    assert all(b.shape == (4, HIDDEN_DIM) for b in batches)
    assert all(b.dtype == torch.float32 for b in batches)
    assert sum(b.shape[0] for b in batches) == 20


def test_partial_last_batch_kept_by_default():
    ds = StreamingActivationDataset(make_factory([10]), batch_size=4, config=StreamingConfig())
    batches = collect(ds)
    sizes = [b.shape[0] for b in batches]
    assert sizes == [4, 4, 2]  # final partial batch retained


def test_drop_last():
    ds = StreamingActivationDataset(make_factory([10]), batch_size=4, config=StreamingConfig(drop_last=True))
    batches = collect(ds)
    sizes = [b.shape[0] for b in batches]
    assert sizes == [4, 4]  # 2-row remainder dropped
    assert sum(sizes) == 8


def test_producer_exception_propagates():
    def factory():
        yield torch.zeros(4, HIDDEN_DIM)
        raise RuntimeError("boom")

    ds = StreamingActivationDataset(factory, batch_size=2, config=StreamingConfig(queue_size=1))
    with pytest.raises(RuntimeError, match="boom"):
        list(iter(ds))  # must raise, not hang


def test_multi_epoch_refreshes_producer():
    calls = []
    factory = make_factory([8], on_produce=lambda ci: calls.append(ci))
    ds = StreamingActivationDataset(factory, batch_size=4, config=StreamingConfig())

    first = torch.cat(collect(ds), dim=0)
    second = torch.cat(collect(ds), dim=0)

    # Producer factory was invoked once per pass (one chunk each).
    assert len(calls) == 2
    assert torch.equal(first, second)


def test_backpressure_bounds_in_flight():
    produced = []
    queue_size = 2
    # Each chunk is one full batch; a fast producer would otherwise race ahead.
    factory = make_factory([4] * 1000, on_produce=lambda ci: produced.append(ci))

    ds = StreamingActivationDataset(factory, batch_size=4, config=StreamingConfig(queue_size=queue_size))
    it = iter(ds)
    next(it)  # consume a single batch, then stop early
    it.close()  # triggers cleanup (stop event + drain + join)
    gc.collect()

    # With backpressure the producer cannot run away: at most queue_size buffered
    # + one blocked put + a little slack. Far below the 1000 available chunks.
    assert len(produced) <= queue_size + 3


def test_shuffle_buffer_is_seeded_permutation():
    chunk_sizes = [16, 16, 16, 16]  # 64 rows
    total = sum(chunk_sizes)
    cfg = StreamingConfig(shuffle_buffer_size=32, seed=123)

    ds_a = StreamingActivationDataset(make_factory(chunk_sizes), batch_size=8, config=cfg)
    ds_b = StreamingActivationDataset(
        make_factory(chunk_sizes), batch_size=8, config=StreamingConfig(shuffle_buffer_size=32, seed=123)
    )

    out_a = torch.cat(collect(ds_a), dim=0)
    out_b = torch.cat(collect(ds_b), dim=0)

    # Same seed -> identical (deterministic) output order.
    assert torch.equal(out_a, out_b)
    # Multiset preserved: every original token present exactly once.
    ids = out_a[:, 0].sort().values
    expected = torch.arange(total, dtype=torch.float32)
    assert torch.equal(ids, expected)
    # And it actually shuffled (not identity order).
    assert not torch.equal(out_a[:, 0], expected)


def test_trainer_fit_streaming_smoke():
    from sae.architectures import TopKSAE
    from sae.training import Trainer, TrainingConfig

    torch.manual_seed(0)
    input_dim, hidden_dim, batch_size = 8, 16, 32

    def factory():
        for _ in range(8):  # 8 chunks * 32 rows = 256 tokens
            yield torch.randn(batch_size, input_dim)

    dataloader = make_streaming_dataloader(
        factory, batch_size=batch_size, config=StreamingConfig(enabled=True, queue_size=4)
    )

    sae = TopKSAE(input_dim=input_dim, hidden_dim=hidden_dim, top_k=4)
    trainer = Trainer(sae, TrainingConfig(n_epochs=1, batch_size=batch_size, device="cpu", log_interval=1000))

    final_loss = trainer.fit(dataloader)
    assert isinstance(final_loss, float)
    assert math.isfinite(final_loss)


def test_multiple_producers_all_consumed():
    """Multiple producer factories feed one queue; every token is consumed once."""

    def make(ids):
        def factory():
            for g in ids:
                yield torch.full((1, HIDDEN_DIM), float(g))

        return factory

    a = list(range(0, 10))
    b = list(range(100, 117))
    ds = StreamingActivationDataset([make(a), make(b)], batch_size=4, config=StreamingConfig(queue_size=2))

    out = torch.cat(collect(ds), dim=0)
    assert out.shape[0] == len(a) + len(b)
    got = sorted(int(v) for v in out[:, 0].tolist())
    assert got == sorted(a + b)  # union of both producers, each row exactly once


def test_max_steps_stops_across_streaming_epochs():
    """max_steps stops at an exact step budget, looping the producer across epochs."""
    from sae.architectures import TopKSAE
    from sae.training import Trainer, TrainingConfig

    torch.manual_seed(0)
    input_dim, batch_size = 8, 4

    # 5 batches per producer pass; max_steps=12 needs ~3 passes (producer re-runs).
    def factory():
        for _ in range(5):
            yield torch.randn(batch_size, input_dim)

    dl = make_streaming_dataloader(factory, batch_size=batch_size, config=StreamingConfig(enabled=True))
    sae = TopKSAE(input_dim=input_dim, hidden_dim=16, top_k=4)
    trainer = Trainer(
        sae,
        TrainingConfig(n_epochs=1, batch_size=batch_size, device="cpu", max_steps=12, log_interval=10_000),
    )
    trainer.fit(dl)
    assert trainer.global_step == 12
