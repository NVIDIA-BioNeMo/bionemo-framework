# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams

import bionemo.evo2.models.megatron.hyena.hyena_mixer as hyena_mixer_module
from bionemo.evo2.models.evo2_provider import HyenaNVTestModelProvider, HyenaTestModelProvider
from bionemo.evo2.models.megatron.hyena.hyena_config import HyenaConfig
from bionemo.evo2.models.megatron.hyena.hyena_layer_specs import hyena_stack_spec_no_te
from bionemo.evo2.models.megatron.hyena.hyena_mixer import (
    HyenaMixer,
    _packed_cuda_metadata,
    _packed_sequence_boundaries,
    _pad_padded_dynamic_context_tokens,
    _slice_padded_dynamic_context_tokens,
)

from ....utils import distributed_model_parallel_state


# Add skip decorator for GPU tests
skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")


@pytest.fixture(params=[pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float32, id="fp32")])
def dtype(request):
    """Parametrized dtype fixture."""
    return request.param


@pytest.fixture(params=[pytest.param("standard", id="non_nv"), pytest.param("nv", id="nv")])
def config_type(request):
    """Parametrized config type fixture."""
    return request.param


@pytest.fixture
def test_config(dtype, config_type) -> HyenaTestModelProvider:
    """Create a test config based on the parametrized dtype and config type."""
    if config_type == "standard":
        config = HyenaTestModelProvider()
    else:  # nv
        config = HyenaNVTestModelProvider()

    config.params_dtype = dtype
    config.finalize()
    return config


@pytest.fixture
def hyena_config() -> HyenaConfig:
    """Create a HyenaConfig instance for testing."""
    config = HyenaConfig()
    config.num_groups_hyena = 4096
    config.num_groups_hyena_short = 256
    config.num_groups_hyena_medium = 256
    return config


@pytest.fixture(params=[pytest.param("hyena_short_conv", id="short"), pytest.param("hyena_medium_conv", id="medium")])
def operator_type(request):
    """Parametrized operator type fixture."""
    return request.param


def _create_hyena_mixer(
    test_config: HyenaTestModelProvider, hyena_config: HyenaConfig, operator_type: str
) -> HyenaMixer:
    """Helper to create a HyenaMixer instance. Must be called inside a distributed context."""
    submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules
    return HyenaMixer(
        transformer_config=test_config,
        hyena_config=hyena_config,
        max_sequence_length=512,
        submodules=submodules,
        layer_number=1,
        operator_type=operator_type,
    )


def _create_small_packing_mixer(operator_type: str, dtype: torch.dtype = torch.float32) -> HyenaMixer:
    """Create a small mixer for packed-sequence parity tests."""
    test_config = HyenaTestModelProvider(
        hidden_size=64,
        num_attention_heads=8,
        ffn_hidden_size=128,
        num_groups_hyena=64,
        num_groups_hyena_short=8,
        num_groups_hyena_medium=8,
    )
    test_config.params_dtype = dtype
    test_config.use_subquadratic_ops = False
    test_config.finalize()

    hyena_config = HyenaConfig(
        num_groups_hyena=64,
        num_groups_hyena_short=8,
        num_groups_hyena_medium=8,
        fast_conv_proj=False,
        fast_conv_mixer=False,
        hyena_medium_conv_len=16,
    )
    submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules
    return HyenaMixer(
        transformer_config=test_config,
        hyena_config=hyena_config,
        max_sequence_length=32,
        submodules=submodules,
        layer_number=1,
        operator_type=operator_type,
    )


def _packed_params(lengths: list[int], device: torch.device) -> PackedSeqParams:
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
    )


def _independent_mixer_forward(mixer: HyenaMixer, packed_input: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    outputs = []
    start = 0
    for length in lengths:
        output, _ = mixer(packed_input[start : start + length], _hyena_use_cp=False)
        outputs.append(output)
        start += length
    return torch.cat(outputs, dim=0)


@skip_if_no_gpu
def test_packed_cuda_metadata_reconstructs_missing_sequence_ids() -> None:
    """Inference tensors and missing seq_idx must use cached boundary-derived metadata."""
    with torch.inference_mode():
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
        packed_seq_params = SimpleNamespace(
            cu_seqlens_q_padded=None,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=3,
        )

        _, local_positions, sequence_ids, modal_chunks = _packed_cuda_metadata(packed_seq_params, total_tokens=5)
        _, cached_positions, cached_ids, _ = _packed_cuda_metadata(packed_seq_params, total_tokens=5)

    assert local_positions.tolist() == [0, 1, 0, 1, 2]
    assert sequence_ids.tolist() == [0, 0, 1, 1, 1]
    assert cached_positions.data_ptr() == local_positions.data_ptr()
    assert cached_ids.data_ptr() == sequence_ids.data_ptr()
    assert modal_chunks is None


def test_packed_boundaries_support_inference_tensors() -> None:
    """The autograd-free packed fallback must not read an unavailable version counter."""
    with torch.inference_mode():
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        packed_seq_params = SimpleNamespace(
            qkv_format="thd",
            cu_seqlens_q_padded=None,
            cu_seqlens_q=cu_seqlens,
        )

        assert _packed_sequence_boundaries(packed_seq_params, total_tokens=5) == (0, 2, 5)
        assert _packed_sequence_boundaries(packed_seq_params, total_tokens=5) == (0, 2, 5)


@skip_if_no_gpu
def test_packed_cuda_metadata_accepts_missing_max_seqlen_q() -> None:
    """MCore's optional max_seqlen_q must not prevent packed eval."""
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
    packed_seq_params = SimpleNamespace(
        cu_seqlens_q_padded=None,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=None,
    )

    _, local_positions, sequence_ids, modal_chunks = _packed_cuda_metadata(packed_seq_params, total_tokens=5)

    assert local_positions.tolist() == [0, 1, 0, 1, 2]
    assert sequence_ids.tolist() == [0, 0, 1, 1, 1]
    assert modal_chunks is None


@skip_if_no_gpu
def test_modal_poles_are_cached_until_filter_parameters_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refit must update graph-captured modal tables without replacing their storage."""
    calls = []

    def build_poles(gamma, poles_parameter):
        calls.append((gamma, poles_parameter))
        value = float(len(calls))
        return hyena_mixer_module.ModalPoles(
            decay=torch.full(gamma.shape, value, dtype=torch.float32, device=gamma.device),
            log_decay=torch.full(gamma.shape, -value, dtype=torch.float32, device=gamma.device),
        )

    monkeypatch.setattr(hyena_mixer_module, "modal_poles", build_poles, raising=False)
    with distributed_model_parallel_state():
        mixer = _create_small_packing_mixer("hyena", torch.bfloat16).eval()

        first = mixer._packed_modal_poles()
        assert mixer._packed_modal_poles() is first
        with torch.no_grad():
            mixer.mixer.filter.p.add_(1)
        refreshed = mixer._packed_modal_poles()

    assert refreshed is first
    assert refreshed.decay.unique().item() == 2.0
    assert refreshed.log_decay.unique().item() == -2.0
    assert len(calls) == 2


class _FakeDynamicContext:
    def __init__(self, active_token_count: int, *, static: bool = False):
        self.active_token_count = active_token_count
        self._static = static

    def is_static_batching(self) -> bool:
        return self._static


class _FakePackedPrefillContext:
    def __init__(self, lengths: list[int], device: torch.device):
        self.paused_request_count = 0
        self.total_request_count = len(lengths)
        self.num_prefill_requests = len(lengths)
        self.active_token_count = sum(lengths)
        self.request_query_lengths = torch.tensor(lengths, dtype=torch.int32)
        self._cu_seqlens = _packed_params(lengths, device).cu_seqlens_q
        self.fir_filter_state_dict = {}
        self.inner_fir_filter_state_dict = {}
        self.iir_filter_state_dict = {}
        self.evo2_batched_decode_enabled = True

    def is_static_batching(self) -> bool:
        return False

    def get_active_request_count(self) -> int:
        return self.total_request_count

    def cu_query_lengths(self) -> tuple[torch.Tensor, int]:
        return self._cu_seqlens, int(self.request_query_lengths.max())


class _FakeStaticDecodeContext:
    def __init__(self):
        self.fir_filter_state_dict = {}
        self.inner_fir_filter_state_dict = {}
        self.iir_filter_state_dict = {}

    def is_static_batching(self) -> bool:
        return True


def test_flat_dynamic_prefill_rejects_empty_scheduler_state() -> None:
    mixer = MagicMock()
    mixer._supports_flat_segmented_prefill.return_value = True
    context = SimpleNamespace(
        is_static_batching=lambda: False,
        get_active_request_count=lambda: 0,
        num_prefill_requests=0,
        fir_filter_state_dict={},
    )

    assert not HyenaMixer._supports_flat_dynamic_prefill(mixer, torch.empty(0), context)


def test_slice_padded_dynamic_context_tokens_keeps_only_real_rows() -> None:
    """Dynamic-context dummy token rows are excluded before Hyena recurrence."""
    features = torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4)

    sliced, padded_token_count = _slice_padded_dynamic_context_tokens(features, _FakeDynamicContext(2))

    assert padded_token_count == 4
    assert sliced.shape == (1, 3, 2)
    torch.testing.assert_close(sliced, features[..., :2])


def test_slice_padded_dynamic_context_tokens_keeps_static_width() -> None:
    """Static contexts keep the full input width."""
    features = torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4)

    sliced, padded_token_count = _slice_padded_dynamic_context_tokens(features, _FakeDynamicContext(2, static=True))

    assert padded_token_count == 4
    assert sliced.shape == features.shape
    torch.testing.assert_close(sliced, features)


def test_pad_padded_dynamic_context_tokens_restores_dummy_width() -> None:
    """Hyena mixer output is padded back to MCore's graph width."""
    z = torch.ones((1, 3, 2), dtype=torch.float32)

    padded = _pad_padded_dynamic_context_tokens(z, 4)

    assert padded.shape == (1, 3, 4)
    torch.testing.assert_close(padded[..., :2], z)
    torch.testing.assert_close(padded[..., 2:], torch.zeros((1, 3, 2)))


def test_mixer_propagates_explicit_process_groups_to_all_parallel_components(
    monkeypatch: pytest.MonkeyPatch,
    hyena_config: HyenaConfig,
) -> None:
    """Custom process-group collections control both sides of Hyena tensor parallelism."""
    tp_group = MagicMock()
    tp_group.size.return_value = 1
    build_kwargs = []

    def record_build_module(_spec, *_args, **kwargs):
        build_kwargs.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(hyena_mixer_module, "build_module", record_build_module)
    projection_conv = MagicMock()
    short_operator = MagicMock()
    monkeypatch.setattr(hyena_mixer_module, "ParallelCausalDepthwiseConv1dWithState", projection_conv)
    monkeypatch.setattr(hyena_mixer_module, "ParallelShortHyenaOperator", short_operator)

    config = HyenaTestModelProvider()
    config.finalize()
    pg_collection = SimpleNamespace(tp=tp_group)
    HyenaMixer(
        transformer_config=config,
        hyena_config=hyena_config,
        max_sequence_length=512,
        submodules=SimpleNamespace(dense_projection=object(), dense=object()),
        operator_type="hyena_short_conv",
        pg_collection=pg_collection,
    )

    assert [kwargs.get("tp_group") for kwargs in build_kwargs] == [tp_group, tp_group]
    assert projection_conv.call_args.kwargs["pg_collection"] is pg_collection
    assert short_operator.call_args.kwargs["pg_collection"] is pg_collection


@skip_if_no_gpu
def test_mixer_enables_b2b_fusion_with_subquadratic_ops(hyena_config: HyenaConfig) -> None:
    """The accelerated subquadratic path includes its key projection/mixer fusion."""
    test_config = HyenaTestModelProvider()
    test_config.use_subquadratic_ops = True
    test_config.finalize()

    with distributed_model_parallel_state():
        hyena_mixer = _create_hyena_mixer(test_config, hyena_config, "hyena_short_conv")

    assert hyena_mixer.use_subquadratic_ops is True
    assert hasattr(hyena_mixer, "b2b_kernel")


@skip_if_no_gpu
def test_mixer_initialization(test_config: HyenaTestModelProvider, hyena_config: HyenaConfig, operator_type: str):
    """Test proper initialization of HyenaMixer with different configurations."""
    with distributed_model_parallel_state():
        hyena_mixer = _create_hyena_mixer(test_config, hyena_config, operator_type)

        # Verify basic attributes
        assert hyena_mixer.transformer_config == test_config
        assert hyena_mixer.hyena_config == hyena_config
        assert hyena_mixer.operator_type == operator_type
        assert hyena_mixer.layer_number == 1

        # Verify model parallel attributes
        assert hyena_mixer.model_parallel_size == 1
        assert hyena_mixer.hidden_size_per_partition == hyena_mixer.hidden_size

        # Verify projection attributes
        assert hyena_mixer.proj_groups == hyena_config.proj_groups
        assert hyena_mixer.tie_projection_weights == hyena_config.tie_projection_weights

        # Verify mixer type based on operator_type
        if operator_type == "hyena_short_conv":
            assert hyena_mixer.num_groups == hyena_config.num_groups_hyena_short
        elif operator_type == "hyena_medium_conv":
            assert hyena_mixer.num_groups == hyena_config.num_groups_hyena_medium
        else:
            assert hyena_mixer.num_groups == hyena_config.num_groups_hyena


@skip_if_no_gpu
def test_mixer_forward_pass(test_config: HyenaTestModelProvider, hyena_config: HyenaConfig, operator_type: str):
    """Test forward pass of HyenaMixer with different input shapes and configurations."""
    with distributed_model_parallel_state():
        hyena_mixer = _create_hyena_mixer(test_config, hyena_config, operator_type)

        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 128),  # Small batch, short sequence
            (2, 512),  # Medium batch, medium sequence
            (4, 1024),  # Large batch, long sequence
        ]

        for batch_size, seq_len in test_cases:
            # Create input tensor
            input_features = torch.rand(
                (seq_len, batch_size, hyena_mixer.hidden_size),
                dtype=hyena_mixer.transformer_config.params_dtype,
                device=torch.cuda.current_device(),
            )

            # Forward pass
            y, _bias = hyena_mixer(input_features, _hyena_use_cp=False)

            # Verify output shape
            expected_shape = (seq_len, batch_size, hyena_mixer.hidden_size)
            assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"

            # Verify output is not NaN
            assert not torch.isnan(y).any(), "Output contains NaN values"
            # Verify output is not Inf
            assert not torch.isinf(y).any(), "Output contains Inf values"


@skip_if_no_gpu
def test_mixer_dtypes(
    test_config: HyenaTestModelProvider, hyena_config: HyenaConfig, operator_type: str, dtype: torch.dtype
):
    """Test HyenaMixer with different input data types."""
    with distributed_model_parallel_state():
        hyena_mixer = _create_hyena_mixer(test_config, hyena_config, operator_type)

        batch_size = 2
        seq_len = 512

        input_features = torch.rand(
            (seq_len, batch_size, hyena_mixer.hidden_size),
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

        # Forward pass
        y, bias = hyena_mixer(input_features, _hyena_use_cp=False)

        # Verify output dtype matches input dtype
        assert y.dtype == dtype, f"Expected output dtype {dtype}, got {y.dtype}"
        assert bias.dtype == dtype, f"Expected bias dtype {dtype}, got {bias.dtype}"


@skip_if_no_gpu
def test_mixer_state_dict(test_config: HyenaTestModelProvider, hyena_config: HyenaConfig, operator_type: str):
    """Test state dict functionality of HyenaMixer."""
    with distributed_model_parallel_state():
        hyena_mixer = _create_hyena_mixer(test_config, hyena_config, operator_type)

        # Get state dict
        state_dict = hyena_mixer.state_dict()

        # Create new mixer with same config
        new_mixer = _create_hyena_mixer(test_config, hyena_config, operator_type)

        # Load state dict
        new_mixer.load_state_dict(state_dict)

        # Verify parameters match
        for (name1, param1), (name2, param2) in zip(hyena_mixer.named_parameters(), new_mixer.named_parameters()):
            assert torch.allclose(param1, param2), f"Parameter mismatch after loading state dict: {name1}"


@skip_if_no_gpu
@pytest.mark.parametrize("packed_operator", ["hyena_short_conv", "hyena_medium_conv", "hyena"])
@pytest.mark.parametrize(
    "packed_dtype",
    [pytest.param(torch.float32, id="fp32"), pytest.param(torch.bfloat16, id="bf16")],
)
def test_packed_mixer_matches_independent_forward_and_backward(
    packed_operator: str,
    packed_dtype: torch.dtype,
):
    """Packed Hyena must equal independent sequences for outputs and every gradient."""
    lengths = [9, 5, 9]
    device = torch.device("cuda", torch.cuda.current_device())

    with distributed_model_parallel_state():
        torch.manual_seed(1234)
        mixer = _create_small_packing_mixer(packed_operator, packed_dtype)
        packed_input = torch.randn(
            sum(lengths),
            1,
            mixer.input_size,
            device=device,
            dtype=packed_dtype,
            requires_grad=True,
        )
        reference_input = packed_input.detach().clone().requires_grad_(True)
        packed_seq_params = _packed_params(lengths, device)

        packed_output, _ = mixer(
            packed_input,
            packed_seq_params=packed_seq_params,
            _hyena_use_cp=False,
        )
        reference_output = _independent_mixer_forward(mixer, reference_input, lengths)

        forward_tolerance = 6e-2 if packed_dtype == torch.bfloat16 and packed_operator == "hyena" else 2e-2
        if packed_dtype == torch.float32:
            forward_tolerance = 2e-4
        torch.testing.assert_close(
            packed_output,
            reference_output,
            rtol=forward_tolerance,
            atol=forward_tolerance,
        )

        cotangent = torch.randn_like(packed_output)
        parameters = [parameter for parameter in mixer.parameters() if parameter.requires_grad]
        packed_grads = torch.autograd.grad(
            (packed_output * cotangent).sum(),
            [packed_input, *parameters],
            allow_unused=True,
        )
        reference_grads = torch.autograd.grad(
            (reference_output * cotangent).sum(),
            [reference_input, *parameters],
            allow_unused=True,
        )

        for packed_grad, reference_grad in zip(packed_grads, reference_grads):
            assert (packed_grad is None) == (reference_grad is None)
            if packed_grad is not None:
                gradient_tolerance = 8e-2 if packed_dtype == torch.bfloat16 else 5e-4
                torch.testing.assert_close(
                    packed_grad,
                    reference_grad,
                    rtol=gradient_tolerance,
                    atol=gradient_tolerance,
                )


@skip_if_no_gpu
@pytest.mark.parametrize("packed_operator", ["hyena_short_conv", "hyena_medium_conv", "hyena"])
def test_packed_mixer_blocks_boundary_leakage(packed_operator: str):
    """Changing sequence A cannot affect sequence B outputs or input gradients."""
    lengths = [11, 7]
    boundary = lengths[0]
    device = torch.device("cuda", torch.cuda.current_device())

    with distributed_model_parallel_state():
        torch.manual_seed(5678)
        mixer = _create_small_packing_mixer(packed_operator)
        original = torch.randn(sum(lengths), 1, mixer.input_size, device=device)
        perturbed = original.clone()
        perturbed[:boundary].add_(torch.randn_like(perturbed[:boundary]))
        packed_seq_params = _packed_params(lengths, device)

        original.requires_grad_(True)
        perturbed.requires_grad_(True)
        original_output, _ = mixer(
            original,
            packed_seq_params=packed_seq_params,
            _hyena_use_cp=False,
        )
        perturbed_output, _ = mixer(
            perturbed,
            packed_seq_params=packed_seq_params,
            _hyena_use_cp=False,
        )

        torch.testing.assert_close(original_output[boundary:], perturbed_output[boundary:], rtol=2e-4, atol=2e-5)

        b_cotangent = torch.randn_like(original_output[boundary:])
        original_grad = torch.autograd.grad((original_output[boundary:] * b_cotangent).sum(), original)[0]
        perturbed_grad = torch.autograd.grad((perturbed_output[boundary:] * b_cotangent).sum(), perturbed)[0]
        torch.testing.assert_close(original_grad[boundary:], perturbed_grad[boundary:], rtol=5e-4, atol=5e-5)
        torch.testing.assert_close(
            original_grad[:boundary], torch.zeros_like(original_grad[:boundary]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            perturbed_grad[:boundary], torch.zeros_like(perturbed_grad[:boundary]), rtol=0, atol=0
        )


@skip_if_no_gpu
@pytest.mark.parametrize("packed_operator", ["hyena_short_conv", "hyena_medium_conv", "hyena"])
def test_packed_eval_uses_flat_segmented_prefill(packed_operator: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Packed prediction uses one flat boundary-aware pass, never the padding oracle."""
    lengths = [13, 3, 8]
    device = torch.device("cuda", torch.cuda.current_device())

    with distributed_model_parallel_state():
        torch.manual_seed(6789)
        mixer = _create_small_packing_mixer(packed_operator, torch.bfloat16).eval()
        packed_input = torch.randn(sum(lengths), 1, mixer.input_size, device=device, dtype=torch.bfloat16)
        packed_seq_params = _packed_params(lengths, device)

        with torch.no_grad():
            reference_output = _independent_mixer_forward(mixer, packed_input, lengths)

        def fail_padding_oracle(*_args, **_kwargs):
            raise AssertionError("packed eval unexpectedly selected the padding oracle")

        monkeypatch.setattr(mixer, "_mix_packed_projected_features", fail_padding_oracle)
        with torch.no_grad():
            packed_output, _ = mixer(
                packed_input,
                packed_seq_params=packed_seq_params,
                _hyena_use_cp=False,
            )

        tolerance = 5e-2 if packed_operator == "hyena" else 2e-2
        torch.testing.assert_close(packed_output, reference_output, rtol=tolerance, atol=tolerance)


@skip_if_no_gpu
def test_non_order_16_modal_prefill_falls_back_from_flat_kernel() -> None:
    """Unsupported modal orders must select the safe existing-operator fallback."""
    device = torch.device("cuda", torch.cuda.current_device())

    with distributed_model_parallel_state():
        mixer = _create_small_packing_mixer("hyena", torch.bfloat16).eval()
        mixer.hyena_config.hyena_filter_order = 8
        projection = torch.empty(
            3,
            1,
            3 * mixer.hidden_size_per_partition,
            device=device,
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            assert not mixer._supports_flat_segmented_prefill(projection)


@skip_if_no_gpu
@pytest.mark.parametrize("packed_operator", ["hyena_short_conv", "hyena_medium_conv", "hyena"])
def test_dynamic_ragged_prefill_matches_independent_and_seeds_decode_state(packed_operator: str) -> None:
    """Infer prefill consumes native ragged boundaries and emits one state row per request."""
    lengths = [12, 3, 7]
    device = torch.device("cuda", torch.cuda.current_device())

    with distributed_model_parallel_state():
        torch.manual_seed(7890)
        mixer = _create_small_packing_mixer(packed_operator, torch.bfloat16).eval()
        packed_input = torch.randn(sum(lengths), 1, mixer.input_size, device=device, dtype=torch.bfloat16)
        context = _FakePackedPrefillContext(lengths, device)

        with torch.no_grad():
            reference_output = _independent_mixer_forward(mixer, packed_input, lengths)
            actual_output, _ = mixer(
                packed_input,
                inference_context=context,
                _hyena_use_cp=False,
            )

        tolerance = 5e-2 if packed_operator == "hyena" else 2e-2
        torch.testing.assert_close(actual_output, reference_output, rtol=tolerance, atol=tolerance)
        projection_state = context.fir_filter_state_dict[id(mixer.hyena_proj_conv)]
        assert projection_state.shape == (
            len(lengths),
            3 * mixer.hidden_size_per_partition,
            mixer.hyena_proj_conv.kernel_size - 1,
        )
        if packed_operator == "hyena_short_conv":
            mixer_state = context.fir_filter_state_dict[id(mixer.mixer.short_conv)]
            expected_shape = (
                len(lengths),
                mixer.hidden_size_per_partition,
                mixer.mixer.short_conv.kernel_size - 1,
            )
        elif packed_operator == "hyena_medium_conv":
            mixer_state = context.inner_fir_filter_state_dict[id(mixer.mixer)]
            expected_shape = (
                len(lengths),
                mixer.hidden_size_per_partition,
                mixer.mixer.kernel_size - 1,
            )
        else:
            mixer_state = context.iir_filter_state_dict[id(mixer.mixer)]
            expected_shape = (
                len(lengths),
                mixer.hidden_size_per_partition,
                mixer.hyena_config.hyena_filter_order,
            )
        assert mixer_state.shape == expected_shape
        assert mixer_state.dtype == torch.float32


@skip_if_no_gpu
@pytest.mark.parametrize("packed_operator", ["hyena_short_conv", "hyena_medium_conv", "hyena"])
def test_dynamic_ragged_prefill_state_continues_exact_batched_decode(packed_operator: str) -> None:
    """States emitted by packed prefill reproduce each request's next-token output."""
    lengths = [10, 3, 6]
    device = torch.device("cuda", torch.cuda.current_device())

    with distributed_model_parallel_state():
        torch.manual_seed(8901)
        mixer = _create_small_packing_mixer(packed_operator, torch.bfloat16).eval()
        packed_input = torch.randn(sum(lengths), 1, mixer.input_size, device=device, dtype=torch.bfloat16)
        next_input = torch.randn(len(lengths), 1, mixer.input_size, device=device, dtype=torch.bfloat16)
        context = _FakePackedPrefillContext(lengths, device)

        with torch.no_grad():
            mixer(packed_input, inference_context=context, _hyena_use_cp=False)
            context.num_prefill_requests = 0
            context.active_token_count = len(lengths)
            context.request_query_lengths = torch.ones(len(lengths), dtype=torch.int32)
            decode_output, _ = mixer(next_input, inference_context=context, _hyena_use_cp=False)

            independent_decode_segments = []
            start = 0
            for request_index, length in enumerate(lengths):
                independent_context = _FakePackedPrefillContext([length], device)
                mixer(
                    packed_input[start : start + length],
                    inference_context=independent_context,
                    _hyena_use_cp=False,
                )
                independent_context.num_prefill_requests = 0
                independent_context.active_token_count = 1
                independent_context.request_query_lengths = torch.ones(1, dtype=torch.int32)
                independent_decode, _ = mixer(
                    next_input[request_index : request_index + 1],
                    inference_context=independent_context,
                    _hyena_use_cp=False,
                )
                independent_decode_segments.append(independent_decode)
                start += length
            independent_decode_output = torch.cat(independent_decode_segments)

            reference_segments = []
            start = 0
            for request_index, length in enumerate(lengths):
                full_sequence = torch.cat(
                    [packed_input[start : start + length], next_input[request_index : request_index + 1]],
                    dim=0,
                )
                full_output, _ = mixer(full_sequence, _hyena_use_cp=False)
                reference_segments.append(full_output[-1:])
                start += length
            reference_output = torch.cat(reference_segments)

        # This isolates batched state handoff from any modal-vs-FFT numerical drift.
        torch.testing.assert_close(decode_output, independent_decode_output, rtol=2e-3, atol=2e-3)

        # Long Hyena compares the order-16 BF16 modal recurrence against the legacy
        # FFT full forward here. Their accumulation order differs; the emitted fp32
        # modal state itself is checked to 2e-4 in test_packed_kernels.py.
        cross_algorithm_tolerance = 6e-2 if packed_operator == "hyena" else 3e-2
        torch.testing.assert_close(
            decode_output,
            reference_output,
            rtol=cross_algorithm_tolerance,
            atol=cross_algorithm_tolerance,
        )


@skip_if_no_gpu
@pytest.mark.parametrize(
    ("packed_operator", "kernel_operator"),
    [("hyena_short_conv", "short"), ("hyena_medium_conv", "medium"), ("hyena", "modal")],
)
def test_dynamic_single_token_decode_selects_fused_kernel(
    packed_operator: str,
    kernel_operator: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stateful ragged decode step uses the single-launch Hyena recurrence."""
    lengths = [8, 3, 5]
    device = torch.device("cuda", torch.cuda.current_device())

    with distributed_model_parallel_state():
        torch.manual_seed(9012)
        mixer = _create_small_packing_mixer(packed_operator, torch.bfloat16).eval()
        packed_input = torch.randn(sum(lengths), 1, mixer.input_size, device=device, dtype=torch.bfloat16)
        next_input = torch.randn(len(lengths), 1, mixer.input_size, device=device, dtype=torch.bfloat16)
        context = _FakePackedPrefillContext(lengths, device)

        with torch.no_grad():
            mixer(packed_input, inference_context=context, _hyena_use_cp=False)

        calls = []
        original = hyena_mixer_module.fused_hyena_decode_from_projection

        def record_call(*args, **kwargs):
            calls.append(kwargs["operator"])
            return original(*args, **kwargs)

        monkeypatch.setattr(hyena_mixer_module, "fused_hyena_decode_from_projection", record_call)
        context.num_prefill_requests = 0
        context.active_token_count = len(lengths)
        context.request_query_lengths = torch.ones(len(lengths), dtype=torch.int32)
        with torch.no_grad():
            mixer(next_input, inference_context=context, _hyena_use_cp=False)

        assert calls == [kernel_operator]


@skip_if_no_gpu
@pytest.mark.parametrize(
    ("packed_operator", "kernel_operator"),
    [("hyena_short_conv", "short"), ("hyena_medium_conv", "medium"), ("hyena", "modal")],
)
def test_static_single_token_decode_selects_fused_kernel(
    packed_operator: str,
    kernel_operator: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static FlashAttention decode also uses the fused Hyena recurrence."""
    device = torch.device("cuda", torch.cuda.current_device())

    with distributed_model_parallel_state():
        torch.manual_seed(9013)
        mixer = _create_small_packing_mixer(packed_operator, torch.bfloat16).eval()
        # Cover the complete medium FIR ring. Short prefills are right-padded by
        # the persistent static-state binding exercised separately.
        prefill_input = torch.randn(16, 2, mixer.input_size, device=device, dtype=torch.bfloat16)
        next_input = torch.randn(1, 2, mixer.input_size, device=device, dtype=torch.bfloat16)
        context = _FakeStaticDecodeContext()

        with torch.no_grad():
            mixer(prefill_input, inference_context=context, _hyena_use_cp=False)

        calls = []
        original = hyena_mixer_module.fused_hyena_decode_from_projection

        def record_call(*args, **kwargs):
            calls.append(kwargs["operator"])
            return original(*args, **kwargs)

        monkeypatch.setattr(hyena_mixer_module, "fused_hyena_decode_from_projection", record_call)
        with torch.no_grad():
            mixer(next_input, inference_context=context, _hyena_use_cp=False)

        assert calls == [kernel_operator]


@skip_if_no_gpu
def test_static_decode_falls_back_while_medium_fir_history_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial eager FIR seed must grow through the existing step path before fusion."""
    device = torch.device("cuda", torch.cuda.current_device())

    with distributed_model_parallel_state():
        torch.manual_seed(9014)
        mixer = _create_small_packing_mixer("hyena_medium_conv", torch.bfloat16).eval()
        prefill_input = torch.randn(3, 1, mixer.input_size, device=device, dtype=torch.bfloat16)
        next_input = torch.randn(1, 1, mixer.input_size, device=device, dtype=torch.bfloat16)
        context = _FakeStaticDecodeContext()

        with torch.no_grad():
            mixer(prefill_input, inference_context=context, _hyena_use_cp=False)
        assert context.inner_fir_filter_state_dict[id(mixer.mixer)].shape[-1] == 3

        fused_calls = []
        original = hyena_mixer_module.fused_hyena_decode_from_projection

        def record_call(*args, **kwargs):
            fused_calls.append(kwargs["operator"])
            return original(*args, **kwargs)

        monkeypatch.setattr(hyena_mixer_module, "fused_hyena_decode_from_projection", record_call)
        with torch.no_grad():
            decode_output, _ = mixer(next_input, inference_context=context, _hyena_use_cp=False)
            reference_output, _ = mixer(torch.cat((prefill_input, next_input)), _hyena_use_cp=False)

        assert fused_calls == []
        assert context.inner_fir_filter_state_dict[id(mixer.mixer)].shape[-1] == 4
        torch.testing.assert_close(decode_output, reference_output[-1:], rtol=3e-2, atol=3e-2)
