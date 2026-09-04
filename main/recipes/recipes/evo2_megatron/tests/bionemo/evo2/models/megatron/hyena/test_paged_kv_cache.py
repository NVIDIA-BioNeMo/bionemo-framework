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

import pytest
import torch

from bionemo.evo2.models.megatron.hyena import paged_kv_cache


def test_contiguous_cache_slice_crosses_signed_int32_at_batch_96():
    """The 96-request Evo2 cache must select 64-bit destination offsets."""
    block_stride = 256 * 32 * 128
    shape = (3843, 256, 32, 128)
    strides = (block_stride, 32 * 128, 128, 1)

    assert paged_kv_cache.storage_span_elements(shape, strides) == 4_029_677_568
    assert paged_kv_cache.requires_64bit_indexing(shape, strides)


def test_contiguous_cache_slice_keeps_32bit_path_below_limit():
    """The corresponding 48-request cache remains within signed-int32 offsets."""
    block_stride = 256 * 32 * 128
    shape = (1923, 256, 32, 128)
    strides = (block_stride, 32 * 128, 128, 1)

    assert paged_kv_cache.storage_span_elements(shape, strides) == 2_016_411_648
    assert not paged_kv_cache.requires_64bit_indexing(shape, strides)


def test_storage_span_uses_strides_instead_of_only_numel():
    """A sparse strided view can require wide offsets despite having few elements."""
    shape = (2, 2)
    strides = (2**31, 1)

    assert paged_kv_cache.storage_span_elements(shape, strides) == 2**31 + 2
    assert paged_kv_cache.requires_64bit_indexing(shape, strides)


def test_storage_span_handles_empty_tensors():
    assert paged_kv_cache.storage_span_elements((0, 256), (256, 1)) == 0
    assert not paged_kv_cache.requires_64bit_indexing((0, 256), (256, 1))


def test_narrow_span_delegates_to_mcore(monkeypatch):
    """The common narrow case must retain MCore's original implementation."""
    if not torch.cuda.is_available():
        pytest.skip("fused paged KV append requires CUDA")
    calls = []

    def record_mcore_call(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        paged_kv_cache,
        "_mcore_append_key_value_cache",
        record_mcore_call,
    )
    monkeypatch.setattr(
        paged_kv_cache,
        "requires_64bit_indexing",
        lambda _shape, _strides: False,
    )
    device = torch.device("cuda")
    key = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16, device=device)
    value = torch.zeros_like(key)
    memory_buffer = torch.zeros((2, 1, 1, 8, 2, 4), dtype=torch.bfloat16, device=device)
    block_ids = torch.zeros(1, dtype=torch.int32, device=device)
    local_positions = torch.zeros_like(block_ids)

    paged_kv_cache.triton_append_key_value_cache(
        0,
        key,
        value,
        memory_buffer,
        1,
        block_ids,
        local_positions,
    )

    assert calls == [
        (
            (0, key, value, memory_buffer, 1, block_ids, local_positions),
            {},
        )
    ]


@pytest.mark.skipif(not paged_kv_cache.HAVE_TRITON, reason="fused paged KV append requires Triton")
def test_evo2_dynamic_context_installs_large_tensor_safe_append(monkeypatch):
    """The Evo2 context factory must route MCore's append through the span guard."""
    from megatron.core.inference.contexts import dynamic_context

    from bionemo.evo2.models.evo2_provider import make_evo2_dynamic_inference_context_cls

    monkeypatch.setattr(
        dynamic_context,
        "triton_append_key_value_cache",
        paged_kv_cache._mcore_append_key_value_cache,
    )

    context_cls = make_evo2_dynamic_inference_context_cls()

    assert context_cls is dynamic_context.DynamicInferenceContext
    assert dynamic_context.triton_append_key_value_cache is paged_kv_cache.triton_append_key_value_cache


@pytest.mark.parametrize("mismatched_input", ["key", "value"])
def test_wide_append_rejects_input_cache_dtype_mismatch(mismatched_input, monkeypatch):
    if not torch.cuda.is_available() or not paged_kv_cache.HAVE_TRITON:
        pytest.skip("fused paged KV append requires CUDA and Triton")
    monkeypatch.setattr(
        paged_kv_cache,
        "requires_64bit_indexing",
        lambda _shape, _strides: True,
    )
    key = torch.zeros((1, 1, 2, 4), dtype=torch.bfloat16, device="cuda")
    value = torch.zeros_like(key)
    if mismatched_input == "key":
        key = key.float()
    else:
        value = value.float()
    memory_buffer = torch.zeros((2, 1, 1, 8, 2, 4), dtype=torch.bfloat16, device="cuda")
    block_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    local_positions = torch.zeros_like(block_ids)

    with pytest.raises(ValueError, match="dtype does not match cache dtype"):
        paged_kv_cache.triton_append_key_value_cache(
            0,
            key,
            value,
            memory_buffer,
            1,
            block_ids,
            local_positions,
        )


@pytest.mark.parametrize("force_64bit", [False, True], ids=["int32", "int64"])
def test_fused_append_matches_indexed_assignment(force_64bit, monkeypatch):
    """MCore's narrow and Evo2's widened paths must scatter K/V rows identically."""
    if not torch.cuda.is_available():
        pytest.skip("fused paged KV append requires CUDA")
    monkeypatch.setattr(
        paged_kv_cache,
        "requires_64bit_indexing",
        lambda _shape, _strides: force_64bit,
    )
    device = torch.device("cuda")
    key = torch.arange(3 * 2 * 4, dtype=torch.bfloat16, device=device).reshape(3, 1, 2, 4)
    value = -key
    memory_buffer = torch.zeros((2, 1, 4, 8, 2, 4), dtype=torch.bfloat16, device=device)
    block_ids = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    local_positions = torch.tensor([1, 4, 7], dtype=torch.int32, device=device)

    paged_kv_cache.triton_append_key_value_cache(
        layer_number=0,
        key=key,
        value=value,
        memory_buffer=memory_buffer,
        padded_active_token_count=3,
        token_to_block_idx=block_ids,
        token_to_local_position_within_kv_block=local_positions,
    )

    torch.testing.assert_close(memory_buffer[0, 0, block_ids.long(), local_positions.long()], key[:, 0])
    torch.testing.assert_close(memory_buffer[1, 0, block_ids.long(), local_positions.long()], value[:, 0])
