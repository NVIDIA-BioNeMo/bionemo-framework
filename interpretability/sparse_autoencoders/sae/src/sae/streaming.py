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

"""Producer-consumer streaming of activations for on-the-fly SAE training.

Instead of extracting all activations to disk and then training on them, a
background *producer* thread runs a caller-supplied activation source and pushes
activation chunks onto a bounded queue, while the SAE ``Trainer`` *consumes*
re-batched activations off the queue. This avoids persisting activations to disk
(beyond checkpoints) and bounds host memory via the queue size: when the queue
is full the producer blocks (backpressure), so at most ``queue_size`` chunks are
in flight at once.

The ``sae`` package is model-agnostic, so the activation source is supplied by
the caller as a *factory* -- a zero-argument callable that returns a fresh
iterator of activation chunks each time it is called. A factory (rather than a
bare iterator) is required so that each training epoch re-runs the source. Each
chunk is a tensor of shape ``[n_tokens, hidden_dim]`` (already flattened and
masked by the caller).

Streaming is OFF by default. Enable it with ``StreamingConfig(enabled=True)``;
the flag is consulted by callers (e.g. a recipe's training script) to decide
whether to build a streaming dataloader instead of reading a cached store.

Example:
    >>> def producer_factory():
    ...     for batch_of_texts in batches:
    ...         yield model.activations(batch_of_texts)  # [n_tokens, hidden_dim]
    >>> cfg = StreamingConfig(enabled=True, queue_size=8)
    >>> dataloader = make_streaming_dataloader(producer_factory, batch_size=4096, config=cfg)
    >>> trainer.fit(dataloader)  # Trainer consumes batches as they are produced
"""

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset


# A factory returning a fresh iterator of activation chunks, each of shape
# [n_tokens, hidden_dim]. A factory (not a bare iterator) lets each epoch
# re-run the source.
ActivationProducer = Callable[[], Iterable[torch.Tensor]]


@dataclass
class StreamingConfig:
    """Configuration for producer-consumer activation streaming.

    Attributes:
        enabled: Master flag. Off by default; callers check this to decide
            whether to stream instead of reading a cached activation store.
        queue_size: Maximum number of activation chunks buffered between the
            producer thread and the consumer. Bounds host memory and provides
            backpressure (the producer blocks when the queue is full).
        shuffle_buffer_size: If > 0, shuffle incoming tokens within a buffer of
            this many rows before emitting batches (approximate shuffle). 0
            preserves producer order.
        seed: Seed for the shuffle buffer (ignored when shuffle_buffer_size == 0).
        drop_last: If True, drop the final partial batch (keeps batch sizes
            uniform, which matters for DDP).
    """

    enabled: bool = False
    queue_size: int = 8
    shuffle_buffer_size: int = 0
    seed: Optional[int] = None
    drop_last: bool = False


# Sentinel signalling the producer has finished. A unique object so it can never
# collide with a yielded activation chunk.
_DONE = object()


def _normalize_chunk(chunk: torch.Tensor) -> torch.Tensor:
    """Coerce a producer chunk to a 2D float32 CPU tensor."""
    if not torch.is_tensor(chunk):
        chunk = torch.as_tensor(chunk)
    chunk = chunk.detach().to(device="cpu", dtype=torch.float32)
    if chunk.ndim != 2:
        raise ValueError(f"Activation chunks must be 2D [n_tokens, hidden_dim], got shape {tuple(chunk.shape)}")
    return chunk


class StreamingActivationDataset(IterableDataset):
    """IterableDataset that streams activations from one or more producers.

    One daemon producer thread per factory iterates its activation source and
    puts chunks onto a single shared bounded queue; ``__iter__`` pulls chunks
    from all producers, optionally shuffles within a buffer, and yields
    pre-formed ``[batch_size, hidden_dim]`` batches. Wrap it in
    ``DataLoader(..., batch_size=None)`` (or use ``make_streaming_dataloader``)
    so the loader passes batches through as-is.

    Multiple producers enable parallel multi-GPU extraction: give each factory a
    model replica pinned to its own GPU (and a disjoint slice of the data), and
    the threads run their forward passes concurrently (PyTorch releases the GIL
    during CUDA execution) while the consumer trains on a separate device.

    Exceptions raised by any producer are propagated to the consumer. When the
    consumer stops early (e.g. ``GeneratorExit``), producers are signalled to
    stop and the queue is drained so the threads can exit.
    """

    def __init__(
        self,
        producer_factory,
        batch_size: int,
        config: Optional[StreamingConfig] = None,
    ):
        """Initialize the streaming dataset.

        Args:
            producer_factory: A zero-arg callable returning a fresh iterator of
                activation chunks ([n_tokens, hidden_dim]), OR a list of such
                callables (one background thread is spawned per factory, all
                feeding the same queue).
            batch_size: Number of token activations per emitted batch.
            config: Streaming configuration (uses defaults if None).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        # Normalize to a list of factories (single factory -> one producer).
        if callable(producer_factory):
            self.producer_factories = [producer_factory]
        else:
            self.producer_factories = list(producer_factory)
        if not self.producer_factories:
            raise ValueError("producer_factory must be a callable or a non-empty list of callables")
        self.batch_size = batch_size
        self.config = config or StreamingConfig()

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield ``[batch_size, hidden_dim]`` batches as the producers fill the queue."""
        cfg = self.config
        batch_size = self.batch_size
        emit_threshold = max(cfg.shuffle_buffer_size, batch_size) if cfg.shuffle_buffer_size > 0 else batch_size

        q: "queue.Queue" = queue.Queue(maxsize=max(1, cfg.queue_size))
        stop_event = threading.Event()
        n_producers = len(self.producer_factories)

        def _produce(factory) -> None:
            try:
                for chunk in factory():
                    if stop_event.is_set():
                        break
                    # put() blocks when the queue is full -> backpressure.
                    while not stop_event.is_set():
                        try:
                            q.put(chunk, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                q.put(_DONE)  # one DONE marker per producer
            except Exception as exc:  # surface producer failures to the consumer
                q.put(exc)

        threads = [
            threading.Thread(target=_produce, args=(f,), name=f"sae-activation-producer-{i}", daemon=True)
            for i, f in enumerate(self.producer_factories)
        ]
        for t in threads:
            t.start()

        generator = None
        if cfg.shuffle_buffer_size > 0 and cfg.seed is not None:
            generator = torch.Generator().manual_seed(cfg.seed)

        buffer: Optional[torch.Tensor] = None

        def _shuffle(buf: torch.Tensor) -> torch.Tensor:
            if cfg.shuffle_buffer_size <= 0:
                return buf
            perm = torch.randperm(buf.shape[0], generator=generator)
            return buf[perm]

        try:
            done_count = 0
            while True:
                item = q.get()
                if item is _DONE:
                    done_count += 1
                    if done_count == n_producers:  # all producers finished
                        break
                    continue
                if isinstance(item, BaseException):
                    raise item

                chunk = _normalize_chunk(item)
                if chunk.shape[0] == 0:
                    continue
                buffer = chunk if buffer is None else torch.cat([buffer, chunk], dim=0)

                if buffer.shape[0] >= emit_threshold:
                    buffer = _shuffle(buffer)
                    n_full = (buffer.shape[0] // batch_size) * batch_size
                    for start in range(0, n_full, batch_size):
                        yield buffer[start : start + batch_size]
                    buffer = buffer[n_full:]

            # Flush whatever remains after the producer finished.
            if buffer is not None and buffer.shape[0] > 0:
                buffer = _shuffle(buffer)
                n_full = (buffer.shape[0] // batch_size) * batch_size
                for start in range(0, n_full, batch_size):
                    yield buffer[start : start + batch_size]
                remainder = buffer[n_full:]
                if remainder.shape[0] > 0 and not cfg.drop_last:
                    yield remainder
        finally:
            # Signal producers and drain the queue so any blocked put() returns.
            stop_event.set()
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            for t in threads:
                t.join(timeout=5.0)


def make_streaming_dataloader(
    producer_factory,
    batch_size: int,
    config: Optional[StreamingConfig] = None,
) -> DataLoader:
    """Build a DataLoader that streams activations from one or more producers.

    The returned DataLoader yields pre-formed ``[batch_size, hidden_dim]``
    tensors and can be passed directly to ``Trainer.fit``. ``num_workers`` is
    fixed to 0 because producers run as threads in the main process (where the
    base model replicas live on the GPUs); worker processes would need their own
    model copies.

    Args:
        producer_factory: A zero-arg callable returning a fresh iterator of
            activation chunks ([n_tokens, hidden_dim]), OR a list of such
            callables for parallel multi-GPU extraction (one thread each, all
            feeding the same queue).
        batch_size: Number of token activations per emitted batch.
        config: Streaming configuration (uses defaults if None).

    Returns:
        A DataLoader yielding ``[batch_size, hidden_dim]`` float32 tensors.
    """
    dataset = StreamingActivationDataset(producer_factory, batch_size, config)
    # batch_size=None: the dataset already yields pre-formed batches.
    return DataLoader(dataset, batch_size=None, num_workers=0)
