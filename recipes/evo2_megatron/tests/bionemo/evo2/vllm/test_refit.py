# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from bionemo.evo2.vllm.refit import (
    indexed_safetensors_layout,
    iter_indexed_safetensors,
    plan_ipc_chunks,
    stream_indexed_checkpoint_to_device,
    validate_refit_proof,
)


def _write_indexed_checkpoint(tmp_path):
    first = {
        "decoder.a": torch.arange(128, dtype=torch.float32),
        "decoder.b": torch.arange(64, dtype=torch.bfloat16),
    }
    second = {
        "decoder.c": torch.arange(96, dtype=torch.float32),
    }
    save_file(first, tmp_path / "model-00001-of-00002.safetensors")
    save_file(second, tmp_path / "model-00002-of-00002.safetensors")
    weight_map = {
        name: "model-00001-of-00002.safetensors" for name in first
    } | {name: "model-00002-of-00002.safetensors" for name in second}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return first | second


def test_indexed_safetensors_layout_and_iterator_preserve_exact_metadata(tmp_path) -> None:
    expected = _write_indexed_checkpoint(tmp_path)

    layout = indexed_safetensors_layout(tmp_path)

    assert [spec.name for spec in layout.tensors] == [
        "decoder.a",
        "decoder.b",
        "decoder.c",
    ]
    assert layout.tensor_count == 3
    assert layout.total_tensor_bytes == sum(tensor.nbytes for tensor in expected.values())
    assert layout.largest_tensor_bytes == expected["decoder.a"].nbytes
    assert layout.state_dict_info == {
        name: (tensor.shape, tensor.dtype) for name, tensor in expected.items()
    }
    actual = dict(iter_indexed_safetensors(layout))
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor)


def test_ipc_chunk_plan_matches_production_half_buffer_capacity(tmp_path) -> None:
    _write_indexed_checkpoint(tmp_path)
    layout = indexed_safetensors_layout(tmp_path)

    plan = plan_ipc_chunks(layout, buffer_size_bytes=2_048)

    assert plan.buffer_size_bytes == 2_048
    assert plan.per_buffer_capacity_bytes == 1_024
    assert plan.chunk_count == 2
    assert [chunk.tensor_names for chunk in plan.chunks] == [
        ("decoder.a", "decoder.b"),
        ("decoder.c",),
    ]
    assert sum(chunk.tensor_count for chunk in plan.chunks) == 3
    assert [chunk.tensor_bytes for chunk in plan.chunks] == [640, 384]


def test_refit_proof_requires_exact_chunks_on_every_tp_rank(tmp_path) -> None:
    _write_indexed_checkpoint(tmp_path)
    layout = indexed_safetensors_layout(tmp_path)
    plan = plan_ipc_chunks(layout, buffer_size_bytes=2_048)

    chunks = [
        {
            "chunk_index": chunk.chunk_index,
            "tensor_count": chunk.tensor_count,
            "tensor_bytes": chunk.tensor_bytes,
            "first_name": chunk.tensor_names[0],
            "last_name": chunk.tensor_names[-1],
            "names_sha256": hashlib.sha256("\n".join(chunk.tensor_names).encode()).hexdigest(),
            "load_call_s": 0.25,
        }
        for chunk in plan.chunks
    ]
    worker = {
        "phase": "refit-1",
        "chunk_count": 2,
        "chunks": chunks,
        "loader": {
            "completed_transactions": 2,
            "loaded_parameter_count": 3,
            "required_parameter_count": 3,
            "pending_fc1_layer_count": 0,
            "started": True,
            "complete": True,
            "consumed": True,
        },
    }
    proof = {
        "phase": "refit-1",
        "device_uuids": ["GPU-a", "GPU-b"],
        "worker_proof": [worker, worker],
    }

    summary = validate_refit_proof(
        proof,
        layout=layout,
        plan=plan,
        expected_phase="refit-1",
        expected_completed_transactions=2,
        expected_tp_size=2,
    )

    assert summary["passed"] is True
    assert summary["tp_worker_count"] == 2
    assert summary["chunk_count_per_worker"] == [2, 2]

    unconsumed = {
        **worker,
        "loader": {**worker["loader"], "consumed": False},
    }
    with pytest.raises(AssertionError, match="consumed"):
        validate_refit_proof(
            {**proof, "worker_proof": [worker, unconsumed]},
            layout=layout,
            plan=plan,
            expected_phase="refit-1",
            expected_completed_transactions=2,
            expected_tp_size=2,
        )

    proof["worker_proof"][1] = {**worker, "chunks": chunks[:-1], "chunk_count": 1}
    with pytest.raises(AssertionError, match="chunk"):
        validate_refit_proof(
            proof,
            layout=layout,
            plan=plan,
            expected_phase="refit-1",
            expected_completed_transactions=2,
            expected_tp_size=2,
        )


def test_streamer_uses_production_helper_and_uuid_scoped_socket(tmp_path) -> None:
    expected = _write_indexed_checkpoint(tmp_path)
    layout = indexed_safetensors_layout(tmp_path)
    events = []

    class FakeSocket:
        def setsockopt(self, option, value):
            events.append(("setsockopt", option, value))

        def bind(self, address):
            events.append(("bind", address))

        def close(self):
            events.append(("close",))

    class FakeContext:
        def socket(self, socket_type):
            events.append(("socket", socket_type))
            return FakeSocket()

        def term(self):
            events.append(("term",))

    def fake_stream(**kwargs):
        events.append(
            (
                "stream",
                kwargs["buffer_size_bytes"],
                kwargs["rank"],
                kwargs["worker_name"],
                tuple((name, tensor.clone()) for name, tensor in kwargs["params_generator"]),
            )
        )

    times = iter((10.0, 12.5))
    result = stream_indexed_checkpoint_to_device(
        layout,
        device_index=1,
        expected_device_uuid="GPU-b",
        buffer_size_bytes=2_048,
        set_device=lambda index: events.append(("set_device", index)),
        get_device_uuid=lambda index: "GPU-b",
        context_factory=FakeContext,
        stream_impl=fake_stream,
        clock=lambda: next(times),
    )

    stream_event = next(event for event in events if event[0] == "stream")
    assert stream_event[1:4] == (2_048, 1, "evo2-checkpoint-refit-tp1")
    assert [name for name, _ in stream_event[4]] == list(expected)
    for name, tensor in stream_event[4]:
        torch.testing.assert_close(tensor, expected[name])
    assert ("bind", "ipc:///tmp/GPU-b.sock") in events
    assert events[-2:] == [("close",), ("term",)]
    assert result == {
        "device_index": 1,
        "device_uuid": "GPU-b",
        "socket_address": "ipc:///tmp/GPU-b.sock",
        "buffer_size_bytes": 2_048,
        "tensor_count": 3,
        "tensor_bytes": 1_024,
        "stream_s": 2.5,
    }


def test_ipc_chunk_plan_rejects_one_tensor_larger_than_half_buffer(tmp_path) -> None:
    _write_indexed_checkpoint(tmp_path)
    layout = indexed_safetensors_layout(tmp_path)

    with pytest.raises(ValueError, match="decoder.a"):
        plan_ipc_chunks(layout, buffer_size_bytes=512)
