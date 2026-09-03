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

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from bionemo.evo2.models.megatron.hyena import engine
from bionemo.evo2.models.megatron.hyena import packed_kernels as packed_kernels_module
from bionemo.evo2.models.megatron.hyena.packed_kernels import (
    TRITON_AVAILABLE,
    fused_hyena_decode_from_projection,
    local_positions_from_cu_seqlens,
    modal_chunk_metadata_from_cu_seqlens,
    modal_poles,
    segmented_causal_conv1d,
    segmented_fir_from_projection,
    segmented_modal_from_projection,
    segmented_tail,
    sequence_ids_from_cu_seqlens,
)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires a CUDA GPU"),
    pytest.mark.skipif(not TRITON_AVAILABLE, reason="Test requires Triton"),
]


def _cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)


def _split_projection(projection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reshaped = projection.reshape(projection.shape[0], -1, 3)
    return reshaped[..., 0], reshaped[..., 1], reshaped[..., 2]


def _segmented_fir_reference(
    projection: torch.Tensor,
    weight: torch.Tensor,
    lengths: list[int],
    *,
    group_width: int,
    flip_filter: bool,
    pregate: bool,
    postgate: bool,
    diagonal: torch.Tensor | None,
) -> torch.Tensor:
    x1, x2, value = _split_projection(projection)
    weight_by_channel = weight.repeat_interleave(group_width, dim=0)
    outputs = []
    start = 0
    for length in lengths:
        segment_x1 = x1[start : start + length]
        segment_x2 = x2[start : start + length]
        segment_value = value[start : start + length]
        recurrent_input = segment_x2 * segment_value if pregate else segment_value
        kernel = weight_by_channel if flip_filter else weight_by_channel.flip(-1)
        convolved = F.conv1d(
            F.pad(recurrent_input.T[None].float(), (weight.shape[1] - 1, 0)),
            kernel[:, None].float(),
            groups=recurrent_input.shape[1],
        )[0].T.to(projection.dtype)
        if diagonal is not None:
            convolved = convolved + recurrent_input * diagonal
        outputs.append(segment_x1 * convolved if postgate else convolved)
        start += length
    return torch.cat(outputs)


def _segmented_modal_reference(
    projection: torch.Tensor,
    diagonal: torch.Tensor,
    residues: torch.Tensor,
    gamma: torch.Tensor,
    poles_parameter: torch.Tensor,
    lengths: list[int],
    *,
    group_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x1, x2, value = _split_projection(projection)
    decay = torch.exp(-torch.exp(poles_parameter.float()) * torch.exp(gamma.float()))
    decay = decay.repeat_interleave(group_width, dim=0)
    residues = residues.float().repeat_interleave(group_width, dim=0)
    outputs = []
    final_states = []
    start = 0
    for length in lengths:
        state = torch.zeros_like(residues)
        for token in range(start, start + length):
            recurrent_input = (x2[token] * value[token]).float()
            state = recurrent_input[:, None] + decay * state
            convolved = (residues * state).sum(-1) + diagonal.float() * recurrent_input
            outputs.append((convolved.to(projection.dtype) * x1[token]).unsqueeze(0))
        final_states.append(state)
        start += length
    return torch.cat(outputs), torch.stack(final_states)


def _decode_reference(
    projection: torch.Tensor,
    projection_state: torch.Tensor,
    projection_weight: torch.Tensor,
    mixer_state: torch.Tensor,
    mixer_weight: torch.Tensor,
    diagonal: torch.Tensor,
    residues: torch.Tensor,
    gamma: torch.Tensor,
    poles_parameter: torch.Tensor,
    *,
    operator: str,
    mixer_group_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, projection_channels = projection.shape
    channels = projection_channels // 3
    expanded_projection_weight = projection_weight.repeat_interleave(3, dim=0)[:, None]
    projected, projection_state = engine.step_fir(
        u=projection,
        fir_state=projection_state,
        weight=expanded_projection_weight,
    )
    x1, x2, value = projected.view(batch, channels, 3).unbind(dim=2)
    recurrent_input = x2 * value
    if operator in {"short", "medium"}:
        expanded_mixer_weight = mixer_weight.repeat_interleave(mixer_group_width, dim=0)[:, None]
        mixed, mixer_state = engine.step_fir(
            u=recurrent_input,
            fir_state=mixer_state,
            weight=expanded_mixer_weight,
            bias=diagonal,
            gated_bias=True,
            flip_filter=operator == "medium",
        )
        output = x1 * mixed
    else:
        expanded_residues = residues.repeat_interleave(mixer_group_width, dim=0)
        log_decay = -torch.exp(poles_parameter) * torch.exp(gamma)
        expanded_log_decay = log_decay.repeat_interleave(mixer_group_width, dim=0)[..., None]
        output, mixer_state = engine.step_iir(
            x2=x1,
            x1=x2,
            v=value,
            D=diagonal,
            residues=expanded_residues,
            poles=expanded_log_decay,
            iir_state=mixer_state,
        )
        output = output.to(projection.dtype)
    return output, projection_state, mixer_state


def test_segmented_projection_fir_matches_independent_forward_and_backward() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    lengths = [9, 2, 7]
    channels = 24
    group_width = 2
    torch.manual_seed(1122)
    packed_input = torch.randn(sum(lengths), channels, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_input = packed_input.detach().clone().requires_grad_(True)
    weight = torch.randn(channels // group_width, 3, device=device, dtype=torch.bfloat16, requires_grad=True)
    cu_seqlens = _cu_seqlens(lengths, device)
    sequence_ids = sequence_ids_from_cu_seqlens(cu_seqlens, packed_input.shape[0])

    actual = segmented_causal_conv1d(
        packed_input,
        weight,
        sequence_ids,
        group_width=group_width,
    )
    expanded_weight = weight.repeat_interleave(group_width, dim=0)
    outputs = []
    start = 0
    for length in lengths:
        segment = reference_input[start : start + length].T[None]
        outputs.append(
            F.conv1d(F.pad(segment.float(), (2, 0)), expanded_weight[:, None].float(), groups=channels)[0].T.to(
                torch.bfloat16
            )
        )
        start += length
    expected = torch.cat(outputs)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    cotangent = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * cotangent).sum(), (packed_input, weight))
    expected_grads = torch.autograd.grad((expected * cotangent).sum(), (reference_input, weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-2, atol=3e-2)


def test_projection_fir_inference_uses_boundary_checked_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    lengths = [7, 2, 5]
    channels = 24
    group_width = 2
    torch.manual_seed(2211)
    packed_input = torch.randn(sum(lengths), channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(channels // group_width, 3, device=device, dtype=torch.bfloat16)
    sequence_ids = sequence_ids_from_cu_seqlens(_cu_seqlens(lengths, device), sum(lengths))
    monkeypatch.setattr(
        packed_kernels_module,
        "causal_conv1d_fn",
        lambda *_args, **_kwargs: pytest.fail("inference must not use the i32-limited extension"),
    )

    with torch.no_grad():
        actual = segmented_causal_conv1d(packed_input, weight, sequence_ids, group_width=group_width)

    expanded_weight = weight.repeat_interleave(group_width, dim=0)
    expected_segments = []
    start = 0
    for length in lengths:
        segment = packed_input[start : start + length].T[None]
        expected_segments.append(
            F.conv1d(F.pad(segment.float(), (2, 0)), expanded_weight[:, None].float(), groups=channels)[0].T.to(
                torch.bfloat16
            )
        )
        start += length
    torch.testing.assert_close(actual, torch.cat(expected_segments), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("operator", "taps", "mixer_group_width"),
    [("short", 7, 4), ("medium", 128, 4), ("modal", 0, 1)],
)
def test_fused_hyena_decode_matches_existing_recurrence_and_state_updates(
    operator: str,
    taps: int,
    mixer_group_width: int,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    batch = 3
    channels = 16
    torch.manual_seed(7710 + taps)
    projection = torch.randn(batch, 3 * channels, device=device, dtype=torch.bfloat16)
    projection_weight = torch.randn(channels, 3, device=device, dtype=torch.bfloat16).contiguous()
    projection_state = torch.randn(batch, 3 * channels, 2, device=device, dtype=torch.float32)
    diagonal = torch.randn(channels, device=device, dtype=torch.bfloat16)
    residues = torch.randn(
        channels // mixer_group_width,
        16,
        device=device,
        dtype=torch.float32,
    ).contiguous()
    gamma = torch.empty_like(residues).uniform_(-4.5, -2.5)
    poles_parameter = torch.empty_like(residues).uniform_(-1.5, -0.5)
    mixer_weight = torch.randn(
        channels // mixer_group_width,
        max(1, taps),
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    state_width = 16 if operator == "modal" else taps - 1
    # Use a padded, non-contiguous view just like MCore's common mamba-state slot.
    mixer_state_storage = torch.randn(batch, channels, 127, device=device, dtype=torch.float32)
    mixer_state = mixer_state_storage[..., :state_width]

    expected, expected_projection_state, expected_mixer_state = _decode_reference(
        projection,
        projection_state.clone(),
        projection_weight,
        mixer_state.clone(),
        mixer_weight,
        diagonal,
        residues,
        gamma,
        poles_parameter,
        operator=operator,
        mixer_group_width=mixer_group_width,
    )
    actual_projection_state = projection_state.clone()
    actual_mixer_storage = mixer_state_storage.clone()
    actual_mixer_state = actual_mixer_storage[..., :state_width]
    actual = fused_hyena_decode_from_projection(
        projection,
        actual_projection_state,
        projection_weight,
        actual_mixer_state,
        mixer_weight,
        diagonal,
        residues,
        modal_poles(gamma, poles_parameter),
        projection_group_width=3,
        mixer_group_width=mixer_group_width,
        operator=operator,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_projection_state, expected_projection_state, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_mixer_state, expected_mixer_state, rtol=2e-4, atol=2e-4)


def test_fused_hyena_decode_keeps_batch_state_independent() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    batch = 3
    channels = 16
    group_width = 4
    taps = 128
    torch.manual_seed(7821)
    projection = torch.randn(batch, 3 * channels, device=device, dtype=torch.bfloat16)
    projection_state = torch.randn(batch, 3 * channels, 2, device=device, dtype=torch.float32)
    mixer_state = torch.randn(batch, channels, taps - 1, device=device, dtype=torch.float32)
    projection_weight = torch.randn(channels, 3, device=device, dtype=torch.bfloat16).contiguous()
    mixer_weight = torch.randn(channels // group_width, taps, device=device, dtype=torch.bfloat16).contiguous()
    diagonal = torch.randn(channels, device=device, dtype=torch.bfloat16)

    original_projection_state = projection_state.clone()
    original_mixer_state = mixer_state.clone()
    original = fused_hyena_decode_from_projection(
        projection,
        original_projection_state,
        projection_weight,
        original_mixer_state,
        mixer_weight,
        diagonal,
        None,
        None,
        projection_group_width=3,
        mixer_group_width=group_width,
        operator="medium",
    )
    perturbed_projection = projection.clone()
    perturbed_projection[0].add_(torch.randn_like(perturbed_projection[0]))
    perturbed_projection_state = projection_state.clone()
    perturbed_projection_state[0].add_(torch.randn_like(perturbed_projection_state[0]))
    perturbed_mixer_state = mixer_state.clone()
    perturbed_mixer_state[0].add_(torch.randn_like(perturbed_mixer_state[0]))
    perturbed = fused_hyena_decode_from_projection(
        perturbed_projection,
        perturbed_projection_state,
        projection_weight,
        perturbed_mixer_state,
        mixer_weight,
        diagonal,
        None,
        None,
        projection_group_width=3,
        mixer_group_width=group_width,
        operator="medium",
    )

    torch.testing.assert_close(original[1:], perturbed[1:], rtol=0, atol=0)
    torch.testing.assert_close(original_projection_state[1:], perturbed_projection_state[1:], rtol=0, atol=0)
    torch.testing.assert_close(original_mixer_state[1:], perturbed_mixer_state[1:], rtol=0, atol=0)


def test_fused_modal_decode_is_batch_invariant() -> None:
    """Identical modal requests must remain bitwise identical on production-sized grids."""
    device = torch.device("cuda", torch.cuda.current_device())
    batch = 32
    channels = 4096
    torch.manual_seed(7913)

    def repeated(shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn((1, *shape), device=device, dtype=dtype).expand(batch, *shape).contiguous()

    projection = repeated((3 * channels,), dtype=torch.bfloat16)
    projection_state = repeated((3 * channels, 2), dtype=torch.float32)
    mixer_state = repeated((channels, 16), dtype=torch.float32)
    projection_weight = torch.randn(channels, 3, device=device, dtype=torch.bfloat16).contiguous()
    diagonal = torch.randn(channels, device=device, dtype=torch.bfloat16).contiguous()
    residues = torch.randn(channels, 16, device=device, dtype=torch.float32).contiguous()
    gamma = torch.empty_like(residues).uniform_(-4.5, -2.5)
    poles_parameter = torch.empty_like(residues).uniform_(-1.5, -0.5)

    output = fused_hyena_decode_from_projection(
        projection,
        projection_state,
        projection_weight,
        mixer_state,
        projection_weight,
        diagonal,
        residues,
        modal_poles(gamma, poles_parameter),
        projection_group_width=3,
        mixer_group_width=1,
        operator="modal",
    )

    torch.testing.assert_close(output, output[:1].expand_as(output), rtol=0, atol=0)
    torch.testing.assert_close(projection_state, projection_state[:1].expand_as(projection_state), rtol=0, atol=0)
    torch.testing.assert_close(mixer_state, mixer_state[:1].expand_as(mixer_state), rtol=0, atol=0)


def test_fused_medium_decode_is_batch_invariant() -> None:
    """The production medium-FIR grid must not couple or perturb equal requests."""
    device = torch.device("cuda", torch.cuda.current_device())
    batch = 32
    channels = 4096
    group_width = 16
    taps = 128
    torch.manual_seed(7951)

    def repeated(shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn((1, *shape), device=device, dtype=dtype).expand(batch, *shape).contiguous()

    projection = repeated((3 * channels,), dtype=torch.bfloat16)
    projection_state = repeated((3 * channels, 2), dtype=torch.float32)
    mixer_state = repeated((channels, taps - 1), dtype=torch.float32)
    projection_weight = torch.randn(channels, 3, device=device, dtype=torch.bfloat16).contiguous()
    mixer_weight = torch.randn(channels // group_width, taps, device=device, dtype=torch.bfloat16).contiguous()
    diagonal = torch.randn(channels, device=device, dtype=torch.bfloat16).contiguous()

    output = fused_hyena_decode_from_projection(
        projection,
        projection_state,
        projection_weight,
        mixer_state,
        mixer_weight,
        diagonal,
        None,
        None,
        projection_group_width=3,
        mixer_group_width=group_width,
        operator="medium",
    )

    torch.testing.assert_close(output, output[:1].expand_as(output), rtol=0, atol=0)
    torch.testing.assert_close(projection_state, projection_state[:1].expand_as(projection_state), rtol=0, atol=0)
    torch.testing.assert_close(mixer_state, mixer_state[:1].expand_as(mixer_state), rtol=0, atol=0)


def test_segmented_tail_right_aligns_short_sequences() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    input = torch.arange(14, device=device, dtype=torch.bfloat16).reshape(7, 2).contiguous()
    cu_seqlens = _cu_seqlens([2, 5], device)

    actual = segmented_tail(input, cu_seqlens, tail_length=3)

    expected = torch.tensor(
        [
            [[0, 0, 2], [0, 1, 3]],
            [[8, 10, 12], [9, 11, 13]],
        ],
        device=device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("taps", "group_width", "flip_filter"),
    [(7, 4, True), (128, 4, False)],
)
def test_segmented_fir_matches_independent_sequences(taps: int, group_width: int, flip_filter: bool) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    lengths = [17, 3, 9]
    channels = 16
    torch.manual_seed(1234 + taps)
    projection = torch.randn(sum(lengths), 3 * channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(channels // group_width, taps, device=device, dtype=torch.bfloat16).contiguous()
    diagonal = torch.randn(channels, device=device, dtype=torch.bfloat16)
    cu_seqlens = _cu_seqlens(lengths, device)
    local_positions = local_positions_from_cu_seqlens(cu_seqlens, projection.shape[0])

    actual = segmented_fir_from_projection(
        projection,
        weight,
        local_positions,
        group_width=group_width,
        flip_filter=flip_filter,
        diagonal=diagonal,
    )
    expected = _segmented_fir_reference(
        projection,
        weight,
        lengths,
        group_width=group_width,
        flip_filter=flip_filter,
        pregate=True,
        postgate=True,
        diagonal=diagonal,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_segmented_fir_blocks_boundary_leakage() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    lengths = [11, 7]
    channels = 16
    torch.manual_seed(2345)
    projection = torch.randn(sum(lengths), 3 * channels, device=device, dtype=torch.bfloat16)
    perturbed = projection.clone()
    perturbed[: lengths[0]].add_(torch.randn_like(perturbed[: lengths[0]]))
    weight = torch.randn(4, 128, device=device, dtype=torch.bfloat16).contiguous()
    diagonal = torch.randn(channels, device=device, dtype=torch.bfloat16)
    local_positions = local_positions_from_cu_seqlens(_cu_seqlens(lengths, device), sum(lengths))

    original_output = segmented_fir_from_projection(
        projection,
        weight,
        local_positions,
        group_width=4,
        flip_filter=False,
        diagonal=diagonal,
    )
    perturbed_output = segmented_fir_from_projection(
        perturbed,
        weight,
        local_positions,
        group_width=4,
        flip_filter=False,
        diagonal=diagonal,
    )

    torch.testing.assert_close(original_output[lengths[0] :], perturbed_output[lengths[0] :], rtol=0, atol=0)


def test_segmented_modal_matches_independent_sequences_and_blocks_leakage() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    lengths = [13, 2, 8]
    channels = 16
    group_width = 4
    torch.manual_seed(3456)
    projection = torch.randn(sum(lengths), 3 * channels, device=device, dtype=torch.bfloat16)
    perturbed = projection.clone()
    perturbed[: lengths[0]].add_(torch.randn_like(perturbed[: lengths[0]]))
    diagonal = torch.randn(channels, device=device, dtype=torch.bfloat16)
    residues = torch.randn(channels // group_width, 16, device=device, dtype=torch.float32).contiguous()
    gamma = torch.empty_like(residues).uniform_(-4.5, -2.5)
    poles_parameter = torch.empty_like(residues).uniform_(-1.5, -0.5)
    poles = modal_poles(gamma, poles_parameter)
    cu_seqlens = _cu_seqlens(lengths, device)

    final_state = torch.empty(len(lengths), channels, 16, device=device, dtype=torch.float32)
    actual = segmented_modal_from_projection(
        projection,
        diagonal,
        residues,
        poles,
        cu_seqlens,
        group_width=group_width,
        final_state_out=final_state,
    )
    expected, expected_final_state = _segmented_modal_reference(
        projection,
        diagonal,
        residues,
        gamma,
        poles_parameter,
        lengths,
        group_width=group_width,
    )
    perturbed_output = segmented_modal_from_projection(
        perturbed,
        diagonal,
        residues,
        poles,
        cu_seqlens,
        group_width=group_width,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(final_state, expected_final_state, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual[lengths[0] :], perturbed_output[lengths[0] :], rtol=0, atol=0)


def test_chunked_segmented_modal_matches_recurrence_and_blocks_leakage() -> None:
    """Chunk-parallel prefill must preserve state across chunks, but never segments."""
    device = torch.device("cuda", torch.cuda.current_device())
    lengths = [19, 2, 9]
    channels = 16
    group_width = 4
    torch.manual_seed(4567)
    projection = torch.randn(sum(lengths), 3 * channels, device=device, dtype=torch.bfloat16)
    perturbed = projection.clone()
    perturbed[: lengths[0]].add_(torch.randn_like(perturbed[: lengths[0]]))
    diagonal = torch.randn(channels, device=device, dtype=torch.bfloat16)
    residues = torch.randn(channels // group_width, 16, device=device, dtype=torch.float32).contiguous()
    gamma = torch.empty_like(residues).uniform_(-4.5, -2.5)
    poles_parameter = torch.empty_like(residues).uniform_(-1.5, -0.5)
    poles = modal_poles(gamma, poles_parameter)
    cu_seqlens = _cu_seqlens(lengths, device)
    final_state = torch.empty(len(lengths), channels, 16, device=device, dtype=torch.float32)

    actual = segmented_modal_from_projection(
        projection,
        diagonal,
        residues,
        poles,
        cu_seqlens,
        group_width=group_width,
        final_state_out=final_state,
        chunk_size=4,
    )
    expected, expected_final_state = _segmented_modal_reference(
        projection,
        diagonal,
        residues,
        gamma,
        poles_parameter,
        lengths,
        group_width=group_width,
    )
    perturbed_output = segmented_modal_from_projection(
        perturbed,
        diagonal,
        residues,
        poles,
        cu_seqlens,
        group_width=group_width,
        chunk_size=4,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(final_state, expected_final_state, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(actual[lengths[0] :], perturbed_output[lengths[0] :], rtol=0, atol=0)


def test_modal_chunk_metadata_does_not_allocate_state_for_short_neighbors() -> None:
    """A long segment must not force one recurrent-state slab per short segment."""
    device = torch.device("cuda", torch.cuda.current_device())
    metadata = modal_chunk_metadata_from_cu_seqlens(_cu_seqlens([19, 2, 9, 1], device), chunk_size=4)

    assert metadata.chunk_starts.tolist() == [0, 4, 8, 12, 16, 21, 25, 29]
    assert metadata.chunk_lengths.tolist() == [4, 4, 4, 4, 3, 4, 4, 1]
    assert metadata.chunk_sequence_ids.tolist() == [0, 0, 0, 0, 0, 2, 2, 2]
    assert metadata.chunked_sequence_ids.tolist() == [0, 2]
    assert metadata.unchunked_sequence_ids.tolist() == [1, 3]
    assert metadata.sequence_chunk_offsets.tolist() == [0, 5, 8]


def test_chunked_segmented_modal_matches_direct_scan_across_many_chunks() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    lengths = [4097, 5, 513]
    channels = 16
    group_width = 4
    torch.manual_seed(5678)
    projection = torch.randn(sum(lengths), 3 * channels, device=device, dtype=torch.bfloat16)
    diagonal = torch.randn(channels, device=device, dtype=torch.bfloat16)
    residues = torch.randn(channels // group_width, 16, device=device, dtype=torch.float32).contiguous()
    gamma = torch.empty_like(residues).uniform_(-4.5, -2.5)
    poles_parameter = torch.empty_like(residues).uniform_(-1.5, -0.5)
    poles = modal_poles(gamma, poles_parameter)
    cu_seqlens = _cu_seqlens(lengths, device)
    direct_state = torch.empty(len(lengths), channels, 16, device=device, dtype=torch.float32)
    chunked_state = torch.empty_like(direct_state)

    direct = segmented_modal_from_projection(
        projection,
        diagonal,
        residues,
        poles,
        cu_seqlens,
        group_width=group_width,
        final_state_out=direct_state,
    )
    chunked = segmented_modal_from_projection(
        projection,
        diagonal,
        residues,
        poles,
        cu_seqlens,
        group_width=group_width,
        final_state_out=chunked_state,
        chunk_size=64,
    )

    torch.testing.assert_close(chunked, direct, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(chunked_state, direct_state, rtol=2e-3, atol=2e-3)
