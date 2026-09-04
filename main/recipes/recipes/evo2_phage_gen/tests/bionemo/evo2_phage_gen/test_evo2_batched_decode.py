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

from types import SimpleNamespace

import pytest
import torch

from bionemo.evo2.models.evo2_provider import bind_hyena_packed_views_to_dynamic_context_batch
from bionemo.evo2.models.megatron.hyena.hyena_mixer import (
    _reshape_dynamic_context_requests,
    _restore_dynamic_context_requests,
)
from bionemo.evo2.run.infer import (
    _native_stop_token_ids,
    _normalize_new_request_slots_for_packed_hyena,
    _sampled_token_action,
    _sampling_rng_for_native_dynamic,
)
from bionemo.evo2_phage_gen.nemo_rl_evo2_generation import (
    _batched_sampling_value,
    _native_generated_token_ids,
    _PromptTokenProxy,
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

    def tokenize(self, text: str) -> list[int]:
        return [1 if char == "A" else 2 for char in text]


def test_prompt_token_proxy_rejects_missing_tokenizer_without_recursing():
    """A partially constructed proxy should fail with AttributeError, not recursive lookup."""
    proxy = object.__new__(_PromptTokenProxy)

    with pytest.raises(AttributeError, match="_tokenizer"):
        proxy._tokenizer


def test_native_batched_generation_requires_homogeneous_sampling_params():
    """Batched Evo2 generation should fail loudly if NeMo-RL sends mixed sampling params."""
    params = [
        SimpleNamespace(num_tokens_to_generate=8, temperature=1.0),
        SimpleNamespace(num_tokens_to_generate=16, temperature=1.0),
    ]

    with pytest.raises(ValueError, match="num_tokens_to_generate differs"):
        _batched_sampling_value(params, "num_tokens_to_generate", required=True)


def test_native_batched_generation_requires_max_new_tokens_param():
    """Missing max-token sampling params should not silently generate zero tokens."""
    with pytest.raises(ValueError, match="num_tokens_to_generate"):
        _batched_sampling_value([SimpleNamespace(temperature=1.0)], "num_tokens_to_generate", required=True)


def test_native_sampling_rng_persists_across_generate_calls():
    """Repeated prompt-file chunks should continue the RNG stream instead of reseeding."""
    native_dynamic = SimpleNamespace(evo2_seed=1234, sampling_rng=None)
    device = torch.device("cpu")

    first_rng = _sampling_rng_for_native_dynamic(native_dynamic, device)
    first_draw = torch.rand(8, generator=first_rng)

    second_rng = _sampling_rng_for_native_dynamic(native_dynamic, device)
    second_draw = torch.rand(8, generator=second_rng)

    fresh_rng = torch.Generator(device=device)
    fresh_rng.manual_seed(1234)

    assert second_rng is first_rng
    torch.testing.assert_close(first_draw, torch.rand(8, generator=fresh_rng))
    assert not torch.equal(first_draw, second_draw)


def test_native_generated_token_ids_do_not_fall_back_to_text_retokenization():
    """The NeMo adapter should use original sampled IDs, not decode->encode replay."""
    result = SimpleNamespace(generated_text="AC", generated_tokens=[10, 20], generated_log_probs=[-0.1, -0.2])

    assert _native_generated_token_ids(result) == [10, 20]


def test_native_generated_token_ids_require_sampled_ids():
    """Missing sampled IDs should fail loudly instead of retokenizing generated text."""
    result = SimpleNamespace(generated_text="AC", generated_log_probs=[-0.1, -0.2])

    with pytest.raises(ValueError, match="generated token IDs"):
        _native_generated_token_ids(result)


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


def test_batched_decode_reshape_accepts_already_batched_requests():
    """NeMo-RL can present decode tokens as Hyena-compatible request batch rows."""
    context = _DummyDynamicContext([1, 1, 1, 1])
    features = torch.arange(4 * 4 * 1, dtype=torch.float32).reshape(4, 4, 1)

    unpacked, layout = _reshape_dynamic_context_requests(features, context)
    restored = _restore_dynamic_context_requests(unpacked, layout)

    assert layout is None
    torch.testing.assert_close(unpacked, features)
    torch.testing.assert_close(restored, features)


def test_batched_decode_reshape_rejects_mixed_query_lengths():
    """The opt-in path should fail loudly instead of mixing per-request recurrent state."""
    context = _DummyDynamicContext([2, 3])
    features = torch.zeros(1, 4, 5)

    with pytest.raises(ValueError, match="same query length"):
        _reshape_dynamic_context_requests(features, context)


def test_batched_decode_stop_actions_are_row_local():
    """Each sampled token should independently append or stop its request."""
    tokenizer = _DummyTokenizer()
    stop_token_ids = _native_stop_token_ids(tokenizer)

    assert stop_token_ids == {0, 99}
    assert [_sampled_token_action(token_id, stop_token_ids, ignore_eos=False) for token_id in (1, 0, 2, 99)] == [
        (True, False),
        (False, True),
        (True, False),
        (False, True),
    ]


def test_batched_decode_can_preserve_terminal_stop_actions():
    """Opt-in preservation should retain each terminal action while stopping its row."""
    stop_token_ids = _native_stop_token_ids(_DummyTokenizer())

    assert [
        _sampled_token_action(token_id, stop_token_ids, ignore_eos=False, preserve_eos_token=True)
        for token_id in (1, 0, 2, 99)
    ] == [
        (True, False),
        (True, True),
        (True, False),
        (True, True),
    ]


def test_batched_decode_ignore_eos_omits_stop_token_without_stopping():
    """Ignoring EOS should omit the stop token while keeping the request active."""
    stop_token_ids = _native_stop_token_ids(_DummyTokenizer())

    assert _sampled_token_action(0, stop_token_ids, ignore_eos=True) == (False, False)


def test_batched_hyena_binding_normalizes_reverse_contiguous_slots():
    """MCore's reverse slot allocation should normalize before binding packed Hyena views."""
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
        mamba_metadata=SimpleNamespace(request_to_mamba_state_idx=torch.tensor([7, 6, 5, 4])),
        layer_map=[0],
    )

    request_slots = _normalize_new_request_slots_for_packed_hyena(context, request_count=4)
    packed_dicts = bind_hyena_packed_views_to_dynamic_context_batch(
        decoder,
        context,
        request_slots=request_slots,
    )

    assert request_slots.tolist() == [4, 5, 6, 7]
    assert context.mamba_metadata.request_to_mamba_state_idx.tolist() == [4, 5, 6, 7]
    assert len(packed_dicts) == 3
    assert hasattr(context, "fir_filter_state_dict")
    assert hasattr(context, "iir_filter_state_dict")
    fir_seed = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)
    iir_seed = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)

    context.fir_filter_state_dict[id(conv_owner)] = fir_seed
    context.iir_filter_state_dict[id(ssm_owner)] = iir_seed

    fir_view = context.mamba_conv_states[0, 4:8]
    iir_view = context.mamba_ssm_states[0, 4:8, :3, :2]
    assert context.fir_filter_state_dict[id(conv_owner)].data_ptr() == fir_view.data_ptr()
    assert context.iir_filter_state_dict[id(ssm_owner)].data_ptr() == iir_view.data_ptr()
    torch.testing.assert_close(fir_view, fir_seed)
    torch.testing.assert_close(iir_view, iir_seed)
