# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Indexed-safetensors planning for production NeMo-RL IPC refits."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from nemo_rl.models.policy.utils import calculate_aligned_size
from safetensors import safe_open


_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


@dataclass(frozen=True)
class SafetensorsTensorSpec:
    """One indexed tensor and the exact metadata consumed by NeMo-RL refit."""

    name: str
    shard: str
    shape: torch.Size
    dtype: torch.dtype
    nbytes: int
    aligned_nbytes: int


@dataclass(frozen=True)
class IndexedSafetensorsLayout:
    """Validated deterministic traversal of an indexed safetensors checkpoint."""

    root: Path
    tensors: tuple[SafetensorsTensorSpec, ...]

    @property
    def tensor_count(self) -> int:
        """Return the number of indexed source tensors."""
        return len(self.tensors)

    @property
    def total_tensor_bytes(self) -> int:
        """Return unaligned bytes across all indexed tensors."""
        return sum(spec.nbytes for spec in self.tensors)

    @property
    def largest_tensor_bytes(self) -> int:
        """Return the largest single source tensor in bytes."""
        return max((spec.nbytes for spec in self.tensors), default=0)

    @property
    def state_dict_info(self) -> dict[str, tuple[torch.Size, torch.dtype]]:
        """Return metadata consumed by NeMo-RL refit receivers."""
        return {spec.name: (spec.shape, spec.dtype) for spec in self.tensors}


@dataclass(frozen=True)
class IpcChunk:
    """One expected invocation of NeMo-RL's internal chunk loader."""

    chunk_index: int
    tensor_names: tuple[str, ...]
    used_bytes: int
    tensor_bytes: int

    @property
    def tensor_count(self) -> int:
        """Return the number of source tensors in this chunk."""
        return len(self.tensor_names)


@dataclass(frozen=True)
class IpcChunkPlan:
    """Ping-pong IPC groups using the production helper's half-buffer capacity."""

    buffer_size_bytes: int
    per_buffer_capacity_bytes: int
    chunks: tuple[IpcChunk, ...]

    @property
    def chunk_count(self) -> int:
        """Return the number of planned receiver load calls."""
        return len(self.chunks)


def _checkpoint_shard(root: Path, shard_name: str) -> Path:
    shard = (root / shard_name).resolve()
    try:
        shard.relative_to(root)
    except ValueError as error:
        raise ValueError(f"checkpoint shard escapes root: {shard_name}") from error
    if not shard.is_file():
        raise FileNotFoundError(f"indexed checkpoint shard is missing: {shard}")
    return shard


def indexed_safetensors_layout(checkpoint: str | Path) -> IndexedSafetensorsLayout:
    """Read shapes and dtypes from shard headers without materializing weights."""
    root = Path(checkpoint).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"safetensors index is missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index must contain a non-empty weight_map")

    names_by_shard: dict[str, list[str]] = {}
    for name, shard_name in weight_map.items():
        names_by_shard.setdefault(str(shard_name), []).append(str(name))

    specs = []
    for shard_name in sorted(names_by_shard):
        shard_path = _checkpoint_shard(root, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
            expected_keys = set(names_by_shard[shard_name])
            if shard_keys != expected_keys:
                raise ValueError(
                    f"safetensors index/header mismatch for {shard_name}: "
                    f"missing={sorted(expected_keys - shard_keys)}, extra={sorted(shard_keys - expected_keys)}"
                )
            for name in sorted(expected_keys):
                tensor_slice = handle.get_slice(name)
                dtype_name = tensor_slice.get_dtype()
                try:
                    dtype = _SAFETENSORS_DTYPES[dtype_name]
                except KeyError as error:
                    raise ValueError(
                        f"unsupported safetensors dtype {dtype_name!r} for {name}"
                    ) from error
                shape = torch.Size(tensor_slice.get_shape())
                nbytes = math.prod(shape) * dtype.itemsize
                specs.append(
                    SafetensorsTensorSpec(
                        name=name,
                        shard=shard_name,
                        shape=shape,
                        dtype=dtype,
                        nbytes=nbytes,
                        aligned_nbytes=calculate_aligned_size(nbytes),
                    )
                )
    return IndexedSafetensorsLayout(root=root, tensors=tuple(specs))


def iter_indexed_safetensors(
    layout: IndexedSafetensorsLayout,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Lazily materialize one CPU tensor at a time in the validated layout order."""
    for shard_name, shard_specs in groupby(layout.tensors, key=lambda spec: spec.shard):
        shard_path = _checkpoint_shard(layout.root, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for spec in shard_specs:
                tensor = handle.get_tensor(spec.name)
                if tensor.shape != spec.shape or tensor.dtype != spec.dtype:
                    raise RuntimeError(f"safetensors metadata changed while reading {spec.name}")
                yield spec.name, tensor


def plan_ipc_chunks(
    layout: IndexedSafetensorsLayout,
    *,
    buffer_size_bytes: int,
) -> IpcChunkPlan:
    """Mirror NeMo-RL's ping-pong grouping before allocating any CUDA memory."""
    if buffer_size_bytes <= 0 or buffer_size_bytes % 2:
        raise ValueError("buffer_size_bytes must be a positive even integer")
    capacity = buffer_size_bytes // 2
    chunks = []
    names: list[str] = []
    used_bytes = 0
    tensor_bytes = 0
    for spec in layout.tensors:
        if spec.aligned_nbytes > capacity:
            raise ValueError(
                f"tensor {spec.name} requires {spec.aligned_nbytes} aligned bytes, "
                f"larger than IPC half-buffer capacity {capacity}"
            )
        if names and used_bytes + spec.aligned_nbytes > capacity:
            chunks.append(IpcChunk(len(chunks), tuple(names), used_bytes, tensor_bytes))
            names = []
            used_bytes = 0
            tensor_bytes = 0
        names.append(spec.name)
        used_bytes += spec.aligned_nbytes
        tensor_bytes += spec.nbytes
    if names:
        chunks.append(IpcChunk(len(chunks), tuple(names), used_bytes, tensor_bytes))
    return IpcChunkPlan(
        buffer_size_bytes=buffer_size_bytes,
        per_buffer_capacity_bytes=capacity,
        chunks=tuple(chunks),
    )


def stream_indexed_checkpoint_to_device(
    layout: IndexedSafetensorsLayout,
    *,
    device_index: int,
    expected_device_uuid: str,
    buffer_size_bytes: int,
    set_device: Callable[[int], Any] | None = None,
    get_device_uuid: Callable[[int], str] | None = None,
    context_factory: Callable[[], Any] | None = None,
    stream_impl: Callable[..., None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    ready_event: Any | None = None,
    start_event: Any | None = None,
    start_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Stream one indexed checkpoint copy through NeMo-RL's production CUDA IPC helper."""
    import zmq

    if device_index < 0:
        raise ValueError("device_index must be nonnegative")
    if buffer_size_bytes <= 0 or buffer_size_bytes % 2:
        raise ValueError("buffer_size_bytes must be a positive even integer")
    if not expected_device_uuid:
        raise ValueError("expected_device_uuid cannot be empty")
    if set_device is None:
        set_device = torch.cuda.set_device
    if get_device_uuid is None:
        from nemo_rl.utils.nvml import get_device_uuid as nvml_device_uuid

        get_device_uuid = nvml_device_uuid
    if context_factory is None:
        context_factory = zmq.Context
    if stream_impl is None:
        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        stream_impl = stream_weights_via_ipc_zmq_impl

    set_device(device_index)
    actual_device_uuid = get_device_uuid(device_index)
    if actual_device_uuid != expected_device_uuid:
        raise RuntimeError(
            f"refit producer device {device_index} UUID {actual_device_uuid} "
            f"does not match generation worker UUID {expected_device_uuid}"
        )

    context = context_factory()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.SNDTIMEO, 120_000)
    socket.setsockopt(zmq.RCVTIMEO, 120_000)
    socket.setsockopt(zmq.LINGER, 0)
    address = f"ipc:///tmp/{actual_device_uuid}.sock"
    socket.bind(address)
    if ready_event is not None:
        ready_event.set()
    if start_event is not None and not start_event.wait(start_timeout_s):
        raise TimeoutError("timed out waiting to start checkpoint refit streaming")

    started = clock()
    try:
        stream_impl(
            params_generator=iter_indexed_safetensors(layout),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=socket,
            rank=device_index,
            worker_name=f"evo2-checkpoint-refit-tp{device_index}",
        )
        elapsed = clock() - started
    finally:
        socket.close()
        context.term()
    return {
        "device_index": device_index,
        "device_uuid": actual_device_uuid,
        "socket_address": address,
        "buffer_size_bytes": buffer_size_bytes,
        "tensor_count": layout.tensor_count,
        "tensor_bytes": layout.total_tensor_bytes,
        "stream_s": elapsed,
    }


def validate_refit_proof(
    proof: dict,
    *,
    layout: IndexedSafetensorsLayout,
    plan: IpcChunkPlan,
    expected_phase: str,
    expected_completed_transactions: int,
    expected_tp_size: int,
) -> dict:
    """Require every TP worker to prove one exact complete refit transaction."""
    if proof.get("phase") != expected_phase:
        raise AssertionError("outer refit proof phase drifted")
    device_uuids = proof.get("device_uuids")
    if not isinstance(device_uuids, list) or len(device_uuids) != expected_tp_size:
        raise AssertionError("refit proof does not identify every TP device")
    if len(set(device_uuids)) != expected_tp_size:
        raise AssertionError("refit proof contains duplicate TP device identities")
    worker_proofs = proof.get("worker_proof")
    if not isinstance(worker_proofs, list) or len(worker_proofs) != expected_tp_size:
        raise AssertionError("refit proof does not cover every TP worker")

    expected_chunks = [
        {
            "chunk_index": chunk.chunk_index,
            "tensor_count": chunk.tensor_count,
            "tensor_bytes": chunk.tensor_bytes,
            "first_name": chunk.tensor_names[0],
            "last_name": chunk.tensor_names[-1],
            "names_sha256": hashlib.sha256("\n".join(chunk.tensor_names).encode()).hexdigest(),
        }
        for chunk in plan.chunks
    ]

    chunk_counts = []
    load_call_s = []
    for worker in worker_proofs:
        if worker.get("phase") != expected_phase:
            raise AssertionError("inner refit proof phase drifted")
        chunks = worker.get("chunks")
        if worker.get("chunk_count") != plan.chunk_count or not isinstance(chunks, list):
            raise AssertionError("refit chunk count differs from the production plan")
        if len(chunks) != plan.chunk_count:
            raise AssertionError("refit chunk evidence is incomplete")
        for actual, expected in zip(chunks, expected_chunks, strict=True):
            if any(actual.get(key) != value for key, value in expected.items()):
                raise AssertionError("refit chunk boundary or tensor identity drifted")
            elapsed = float(actual.get("load_call_s", -1.0))
            if not math.isfinite(elapsed) or elapsed < 0:
                raise AssertionError("refit chunk load timing must be finite and nonnegative")
            load_call_s.append(elapsed)

        loader = worker.get("loader")
        if not isinstance(loader, dict):
            raise AssertionError("refit proof is missing loader transaction state")
        if loader.get("completed_transactions") != expected_completed_transactions:
            raise AssertionError("refit loader transaction count did not advance exactly once")
        if loader.get("loaded_parameter_count") != loader.get("required_parameter_count"):
            raise AssertionError("refit loader did not receive every mandatory parameter")
        if not loader.get("started") or not loader.get("complete"):
            raise AssertionError("refit loader transaction is incomplete")
        if loader.get("pending_fc1_layer_count") != 0:
            raise AssertionError("refit loader retained incomplete fused MLP weights")
        chunk_counts.append(len(chunks))

    return {
        "passed": True,
        "phase": expected_phase,
        "tp_worker_count": len(worker_proofs),
        "device_uuids": list(device_uuids),
        "source_tensor_count": layout.tensor_count,
        "source_tensor_bytes": layout.total_tensor_bytes,
        "chunk_count_per_worker": chunk_counts,
        "load_call_s": load_call_s,
    }


__all__ = [
    "IndexedSafetensorsLayout",
    "IpcChunk",
    "IpcChunkPlan",
    "SafetensorsTensorSpec",
    "indexed_safetensors_layout",
    "iter_indexed_safetensors",
    "plan_ipc_chunks",
    "stream_indexed_checkpoint_to_device",
    "validate_refit_proof",
]
