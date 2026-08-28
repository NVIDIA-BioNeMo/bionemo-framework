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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from unittest.mock import Mock

import pytest
import torch

from bionemo.evo2.models.megatron.hyena import packed_rope


def test_preindex_packed_rope_frequencies_uses_local_positions() -> None:
    frequencies = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 1, 1, 4)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.int64)

    actual = packed_rope.preindex_packed_rope_frequencies(
        frequencies,
        position_ids,
        total_tokens=5,
    )

    torch.testing.assert_close(actual, frequencies[position_ids[0]])


def test_precompute_packed_rope_cos_sin_happens_once() -> None:
    frequencies = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 1, 1, 4)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.int64)

    actual = packed_rope.precompute_packed_rope_cos_sin(
        frequencies,
        position_ids,
        total_tokens=5,
        dtype=torch.bfloat16,
    )
    selected = frequencies[position_ids[0]]
    expected = torch.cat((selected.cos(), selected.sin()), dim=-2).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected)


def test_precomputed_fused_rope_uses_stride_safe_kernel(monkeypatch) -> None:
    tokens = torch.randn(5, 2, 4)
    frequencies = torch.randn(5, 1, 2, 4)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    expected = torch.randn_like(tokens)
    packed = Mock(return_value=expected)
    thd = Mock()
    monkeypatch.setattr(packed_rope._PackedRoPE, "apply", packed)
    monkeypatch.setattr(packed_rope, "_ORIGINAL_FUSED_THD", thd)

    actual = packed_rope.apply_preindexed_fused_rope_thd(
        tokens,
        cu_seqlens,
        frequencies,
        cp_size=1,
        cp_rank=0,
        interleaved=True,
    )

    torch.testing.assert_close(actual, expected)
    packed.assert_called_once()
    call_args, call_kwargs = packed.call_args
    assert call_args[0] is tokens
    assert call_args[1] is frequencies
    assert call_args[2] is True
    assert call_kwargs == {}
    thd.assert_not_called()


def test_non_preindexed_frequencies_keep_standard_thd_kernel(monkeypatch) -> None:
    tokens = torch.randn(5, 2, 4)
    frequencies = torch.randn(3, 1, 1, 4)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    expected = torch.randn_like(tokens)
    thd = Mock(return_value=expected)
    monkeypatch.setattr(packed_rope, "_ORIGINAL_FUSED_THD", thd)

    actual = packed_rope.apply_preindexed_fused_rope_thd(
        tokens,
        cu_seqlens,
        frequencies,
        cp_size=2,
        cp_rank=1,
        interleaved=False,
    )

    torch.testing.assert_close(actual, expected)
    thd.assert_called_once_with(
        tokens,
        cu_seqlens,
        frequencies,
        cp_size=2,
        cp_rank=1,
        interleaved=False,
    )


def _rope_reference(
    tokens: torch.Tensor,
    cos_sin: torch.Tensor,
    *,
    interleaved: bool,
    inverse: bool,
) -> torch.Tensor:
    rotary_dim = cos_sin.shape[-1]
    rotary = tokens[..., :rotary_dim].float()
    if interleaved:
        first = rotary[..., 0::2]
        second = rotary[..., 1::2]
        rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    else:
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    cosine = cos_sin[:, 0, 0].float().unsqueeze(1)
    sine = cos_sin[:, 0, 1].float().unsqueeze(1)
    sign = -1.0 if inverse else 1.0
    output = rotary * cosine + sign * rotated * sine
    return torch.cat((output, tokens[..., rotary_dim:].float()), dim=-1).to(tokens.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.parametrize("interleaved", [False, True])
def test_stride_safe_rope_forward_backward_and_isolation(interleaved: bool) -> None:
    torch.manual_seed(1234)
    device = torch.device("cuda")
    lengths = [19, 7, 11]
    total_tokens = sum(lengths)
    base = torch.randn(total_tokens, 3, 4, 16, device=device, dtype=torch.bfloat16)
    tokens = base[:, 0].requires_grad_()
    positions = torch.cat([torch.arange(length) for length in lengths]).to(device)
    base_frequencies = torch.randn(max(lengths), 6, device=device)
    frequencies = (
        base_frequencies.repeat_interleave(2, dim=-1)
        if interleaved
        else torch.cat((base_frequencies, base_frequencies), dim=-1)
    )[:, None, None, :]
    cos_sin = packed_rope.precompute_packed_rope_cos_sin(
        frequencies,
        positions,
        total_tokens=total_tokens,
        dtype=torch.bfloat16,
    )

    actual = packed_rope._PackedRoPE.apply(tokens, cos_sin, interleaved)
    expected = _rope_reference(tokens, cos_sin, interleaved=interleaved, inverse=False)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    repeated = packed_rope._PackedRoPE.apply(tokens, cos_sin, interleaved)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    perturbed = tokens.detach().clone()
    perturbed[: lengths[0]].add_(1)
    perturbed_output = packed_rope._PackedRoPE.apply(perturbed, cos_sin, interleaved)
    torch.testing.assert_close(actual[lengths[0] :], perturbed_output[lengths[0] :], rtol=0, atol=0)

    cotangent = torch.randn_like(actual)
    grad = torch.autograd.grad((actual * cotangent).sum(), tokens)[0]
    expected_grad = _rope_reference(cotangent, cos_sin, interleaved=interleaved, inverse=True)
    torch.testing.assert_close(grad, expected_grad, rtol=2e-2, atol=2e-2)
