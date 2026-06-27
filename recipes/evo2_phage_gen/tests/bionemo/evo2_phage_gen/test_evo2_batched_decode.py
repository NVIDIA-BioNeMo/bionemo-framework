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

"""Tests for opt-in Evo2 batched dynamic decode helpers."""

import pytest
import torch
from types import SimpleNamespace

from bionemo.evo2.models.evo2_provider import bind_hyena_packed_views_to_dynamic_context_batch
from bionemo.evo2.models.megatron.hyena.hyena_mixer import (
    _reshape_dynamic_context_requests,
    _restore_dynamic_context_requests,
)
from bionemo.evo2.run.infer import (
    _native_generated_ids_hit_stop,
    _native_stop_token_ids,
    _trim_native_text_stop_markers,
)


class _DummyDynamicContext:
    def __init__(self, query_lengths: list[int]):
        self.evo2_batched_decode_enabled = True
        self.paused_request_count = 0
        self.total_request_count = len(query_lengths)
        self.active_token_count = sum(query_lengths)
        self.request_query_lengths = torch.tensor(query_lengths, dtype=torch.int32)

    def is_static_batching(self) -> bool:
        return False


class _DummyTokenizer:
    eod = 0
    eos_token_id = 99

    def detokenize(self, token_ids: list[int]) -> str:
        token_text = {
            0: "<EOD>",
            1: "A",
            2: "C",
            3: "G",
            4: "T",
            5: " STOP",
            99: "<EOS>",
        }
        return "".join(token_text[token_id] for token_id in token_ids)


def test_batched_decode_reshape_round_trips_flattened_requests():
    """Same-length flattened request tokens should unpack to Hyena batch rows and restore exactly."""
    context = _DummyDynamicContext([2, 2, 2])
    features = torch.arange(1 * 4 * 6, dtype=torch.float32).reshape(1, 4, 6)

    unpacked, layout = _reshape_dynamic_context_requests(features, context)
    restored = _restore_dynamic_context_requests(unpacked, layout)

    assert layout == (3, 2)
    assert unpacked.shape == (3, 4, 2)
    torch.testing.assert_close(unpacked[0], features[0, :, 0:2])
    torch.testing.assert_close(unpacked[1], features[0, :, 2:4])
    torch.testing.assert_close(unpacked[2], features[0, :, 4:6])
    torch.testing.assert_close(restored, features)


def test_batched_decode_reshape_rejects_mixed_query_lengths():
    """The opt-in path should fail loudly instead of mixing per-request recurrent state."""
    context = _DummyDynamicContext([2, 3])
    features = torch.zeros(1, 4, 5)

    with pytest.raises(ValueError, match="same query length"):
        _reshape_dynamic_context_requests(features, context)


def test_batched_decode_stop_helpers_trim_textual_markers():
    """Text STOP/EOS/EOD markers should trim completion text for FASTA-style generation."""
    assert _trim_native_text_stop_markers("ACGT STOP ignored") == "ACGT"
    assert _trim_native_text_stop_markers("ACGTEOS ignored") == "ACGT"
    assert _trim_native_text_stop_markers("ACGT") == "ACGT"


def test_batched_decode_stop_helpers_detect_row_local_stop():
    """Stop detection should be row-local and support both token ids and decoded markers."""
    tokenizer = _DummyTokenizer()
    stop_token_ids = _native_stop_token_ids(tokenizer)

    assert stop_token_ids == {0, 99}
    assert not _native_generated_ids_hit_stop(tokenizer, [1, 2, 3, 4], stop_token_ids)
    assert _native_generated_ids_hit_stop(tokenizer, [1, 2, 0], stop_token_ids)
    assert _native_generated_ids_hit_stop(tokenizer, [1, 2, 5], stop_token_ids)


def test_batched_hyena_binding_accepts_reverse_contiguous_slots():
    """MCore can allocate Mamba state slots in reverse order; the batched view should still bind."""
    conv_owner = object()
    ssm_owner = object()
    shapes = SimpleNamespace(
        conv_owner_id=id(conv_owner),
        ssm_owner_id=id(ssm_owner),
        ssm_shape=(3, 2),
        ssm_kind="iir",
    )
    layer = SimpleNamespace(layer_number=1, mixer=SimpleNamespace(hyena_state_shapes_per_request=lambda: None))
    decoder = SimpleNamespace(
        layers=[layer],
        hyena_state_shapes_per_request=lambda: ((2, 2), (3, 4), [shapes]),
    )
    context = SimpleNamespace(
        mamba_conv_states=torch.zeros(1, 8, 2, 2),
        mamba_ssm_states=torch.zeros(1, 8, 3, 4),
        layer_map=[0],
    )

    packed_dicts = bind_hyena_packed_views_to_dynamic_context_batch(
        decoder,
        context,
        request_slots=torch.tensor([7, 6, 5, 4]),
    )

    assert len(packed_dicts) == 3
    context.fir_filter_state_dict[id(conv_owner)] = torch.ones(4, 2, 2)
    context.iir_filter_state_dict[id(ssm_owner)] = torch.ones(4, 3, 2)
    assert context.fir_filter_state_dict[id(conv_owner)].shape == (4, 2, 2)
    assert context.iir_filter_state_dict[id(ssm_owner)].shape == (4, 3, 2)
