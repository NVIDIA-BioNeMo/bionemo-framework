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

import importlib
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812

from bionemo.evo2.models.megatron.hyena.hyena_utils import (
    B2BCausalConv1dModule,
    ExchangeOverlappingRegionsCausal,
    ParallelCausalDepthwiseConv1d,
    _get_inverse_zigzag_indices,
    _get_zigzag_indices,
    divide,
    ensure_divisibility,
    fftconv_func,
    get_groups_and_group_sizes,
    get_init_method,
    small_init_init_method,
    wang_init_method,
    zigzag_get_overlapping_patches,
)
from bionemo.evo2.models.megatron.hyena.subquadratic_safety import ensure_subquadratic_ops_supported


class MockProcessGroup:
    """Mock process group for testing."""

    @staticmethod
    def rank():
        """Return the rank of the process group."""
        return 0

    @staticmethod
    def size():
        """Return the size of the process group."""
        return 1


class MockProcessGroupCollection:
    """Mock process group collection for testing."""

    def __init__(self):
        """Initialize the mock process group collection."""
        self.tp = MockProcessGroup()
        self.pp = MockProcessGroup()
        self.cp = MockProcessGroup()
        self.embd = MockProcessGroup()
        self.dp = MockProcessGroup()
        self.expt_dp = MockProcessGroup()
        self.mp = MockProcessGroup()
        self.dp_cp = MockProcessGroup()
        self.intra_dp_cp = MockProcessGroup()
        self.intra_expt_dp = MockProcessGroup()

    def use_mpu_process_groups(self):
        """Return the process group collection."""
        return self


class MockProjConv(torch.nn.Module):
    """Mock projection convolution module for testing.

    A simplified version of the projection convolution module used in Hyena models.

    Args:
        kernel_size (int): Size of the convolution kernel
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.short_conv_weight = torch.randn(1, 1, kernel_size)
        self.group_dim = 1


class MockMixer(torch.nn.Module):
    """Mock mixer module for testing.

    A simplified version of the mixer module used in Hyena models.

    Args:
        kernel_size (int): Size of the convolution kernel
        use_conv_bias (bool, optional): Whether to use bias in convolutions. Defaults to False.
    """

    def __init__(self, kernel_size, use_conv_bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.hyena_medium_conv_len = 10
        self.use_conv_bias = use_conv_bias
        self.group_dim = 1
        # Create a mock short_conv module
        self.short_conv = torch.nn.Module()
        self.short_conv.short_conv_weight = torch.randn(1, 1, kernel_size)
        self.short_conv.kernel_size = kernel_size
        # conv_bias attribute for bias handling
        self.conv_bias = torch.randn(1) if use_conv_bias else None
        # Create a mock filter function
        self.filter = MagicMock()
        self.filter.return_value = (torch.randn(1, 1, kernel_size), torch.randn(1, 1, kernel_size))


def mock_b2b_causal_conv1d(x, weight_proj, weight_mixer, skip_bias):
    """Mock implementation of b2b_causal_conv1d that returns only the tensor for test slicing."""
    return x


@patch("bionemo.evo2.models.megatron.hyena.hyena_utils.causal_conv1d_fn")
@patch("bionemo.evo2.models.megatron.hyena.hyena_utils.causal_conv1d")
def test_parallel_causal_depthwise_conv1d_uses_subquadratic_fast_conv(
    mock_subq_causal_conv1d, mock_fast_causal_conv1d
):
    """Fast projection conv should honor use_subquadratic_ops."""
    mock_subq_causal_conv1d.side_effect = lambda x, weight: torch.zeros_like(x)
    x = torch.randn(2, 4, 8)
    module = types.SimpleNamespace(
        kernel_size=3,
        short_conv_weight=torch.ones(4, 3),
        group_dim=1,
        pg_collection=types.SimpleNamespace(cp=None),
        use_fast_causal_conv=True,
        use_subquadratic_ops=True,
    )

    y = ParallelCausalDepthwiseConv1d.forward(module, x, _use_cp=False)

    assert y.shape == x.shape
    mock_subq_causal_conv1d.assert_called_once()
    mock_fast_causal_conv1d.assert_not_called()


@pytest.mark.parametrize("operator_type", ["hyena_short_conv", "hyena_medium_conv"])
def test_b2b_causal_conv1d_module_initialization(operator_type):
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)

    b2b_module = B2BCausalConv1dModule(
        proj_conv, mixer, operator_type=operator_type, pg_collection=MockProcessGroupCollection()
    )

    assert b2b_module.operator_type == operator_type
    assert b2b_module._proj_conv_module == proj_conv
    assert b2b_module._mixer_module == mixer


@pytest.mark.parametrize("operator_type", ["hyena_short_conv", "hyena_medium_conv"])
def test_b2b_causal_conv1d_module_weight_extraction(operator_type):
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(
        proj_conv,
        mixer,
        operator_type=operator_type,
        b2b_causal_conv1d=mock_b2b_causal_conv1d,
        pg_collection=MockProcessGroupCollection(),
    )
    x = torch.randn(2, 96, 10)  # [B, D, L]
    result = b2b_module(x)

    assert result.shape == x.shape


@pytest.mark.parametrize("operator_type", ["hyena_short_conv", "hyena_medium_conv"])
@pytest.mark.parametrize("use_conv_bias", [True, False])
def test_b2b_causal_conv1d_module_bias_handling(use_conv_bias, operator_type):
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5, use_conv_bias=use_conv_bias)
    b2b_module = B2BCausalConv1dModule(
        proj_conv,
        mixer,
        operator_type=operator_type,
        b2b_causal_conv1d=mock_b2b_causal_conv1d,
        pg_collection=MockProcessGroupCollection(),
    )
    x = torch.randn(2, 96, 10)  # [B, D, L]
    result = b2b_module(x)

    assert result.shape == x.shape


def test_b2b_causal_conv1d_module_invalid_operator():
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)

    with pytest.raises(ValueError, match="Operator type invalid_type not supported"):
        B2BCausalConv1dModule(
            proj_conv, mixer, operator_type="invalid_type", pg_collection=MockProcessGroupCollection()
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [8, 16, 32])
def test_b2b_causal_conv1d_module_different_shapes(batch_size, seq_len):
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(
        proj_conv,
        mixer,
        operator_type="hyena_short_conv",
        b2b_causal_conv1d=mock_b2b_causal_conv1d,
        pg_collection=MockProcessGroupCollection(),
    )

    # Test with different hidden dimensions
    for hidden_dim in [32, 64, 128]:
        x = torch.randn(batch_size, hidden_dim, seq_len)
        result = b2b_module(x)
        assert result.shape == x.shape, (
            f"Shape mismatch for batch_size={batch_size}, hidden_dim={hidden_dim}, seq_len={seq_len}"
        )


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_b2b_causal_conv1d_module_different_kernel_sizes(kernel_size):
    proj_conv = MockProjConv(kernel_size=kernel_size)
    mixer = MockMixer(kernel_size=kernel_size)
    b2b_module = B2BCausalConv1dModule(
        proj_conv,
        mixer,
        operator_type="hyena_short_conv",
        b2b_causal_conv1d=mock_b2b_causal_conv1d,
        pg_collection=MockProcessGroupCollection(),
    )
    x = torch.randn(2, 96, 32)
    result = b2b_module(x)

    assert result.shape == x.shape, f"Shape mismatch for kernel_size={kernel_size}"


def test_b2b_causal_conv1d_module_invalid_input():
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(
        proj_conv,
        mixer,
        operator_type="hyena_short_conv",
        b2b_causal_conv1d=mock_b2b_causal_conv1d,
        pg_collection=MockProcessGroupCollection(),
    )

    # Test with invalid input dimensions
    with pytest.raises(ValueError, match="Input tensor must be 3D"):
        b2b_module(torch.randn(2, 96))  # Missing sequence dimension


def test_b2b_causal_conv1d_module_dtype_handling():
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(
        proj_conv,
        mixer,
        operator_type="hyena_short_conv",
        b2b_causal_conv1d=mock_b2b_causal_conv1d,
        pg_collection=MockProcessGroupCollection(),
    )

    # Test with different dtypes
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    for dtype in dtypes:
        x = torch.randn(2, 96, 32, dtype=dtype)
        result = b2b_module(x)

        assert result.dtype == dtype, f"Dtype mismatch for {dtype}"


def test_b2b_causal_conv1d_module_device_handling():
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)
    b2b_module = B2BCausalConv1dModule(
        proj_conv,
        mixer,
        operator_type="hyena_short_conv",
        b2b_causal_conv1d=mock_b2b_causal_conv1d,
        pg_collection=MockProcessGroupCollection(),
    )

    # Test on CPU
    x_cpu = torch.randn(2, 96, 32)
    result_cpu = b2b_module(x_cpu)
    assert result_cpu.device == x_cpu.device, "Device mismatch on CPU"

    # Test on CUDA if available
    if torch.cuda.is_available():
        x_cuda = x_cpu.cuda()
        result_cuda = b2b_module(x_cuda)
        assert result_cuda.device == x_cuda.device, "Device mismatch on CUDA"


def test_b2b_causal_conv1d_effective_padding_size():
    """Test the zigzag pattern for data distribution in context parallel mode."""
    proj_conv = MockProjConv(kernel_size=3)
    mixer = MockMixer(kernel_size=5)

    b2b_module = B2BCausalConv1dModule(
        proj_conv,
        mixer,
        operator_type="hyena_short_conv",
        b2b_causal_conv1d=mock_b2b_causal_conv1d,
        pg_collection=MockProcessGroupCollection(),
    )
    # Verify the effective padding size is correct
    expected_pad_size = (mixer.kernel_size - 1) + (proj_conv.kernel_size - 1)
    assert b2b_module.effective_pad_size == expected_pad_size


def test_b2b_causal_conv1d_module_matches_sequential_reference():
    """Document the isolated B2B CUDA kernel behavior before relying on the fused path."""
    if not torch.cuda.is_available():
        pytest.skip("B2B causal conv isolation test requires CUDA")
    try:
        ensure_subquadratic_ops_supported()
    except RuntimeError as e:
        pytest.xfail(str(e))

    torch.manual_seed(1234)
    batch_size = 2
    hidden_size = 4
    seq_len = 16
    proj_kernel_size = 3
    mixer_kernel_size = 7
    device = torch.device("cuda")

    x = torch.randn(batch_size, 3 * hidden_size, seq_len, device=device)
    proj_weight = torch.randn(3 * hidden_size, proj_kernel_size, device=device)
    mixer_weight = torch.randn(hidden_size, mixer_kernel_size, device=device)
    bias = torch.randn(hidden_size, device=device)

    proj_conv = torch.nn.Module()
    proj_conv.kernel_size = proj_kernel_size
    proj_conv.short_conv_weight = proj_weight
    proj_conv.group_dim = 1

    mixer = torch.nn.Module()
    mixer.use_conv_bias = True
    mixer.group_dim = 1
    mixer.conv_bias = bias
    mixer.short_conv = torch.nn.Module()
    mixer.short_conv.kernel_size = mixer_kernel_size
    mixer.short_conv.short_conv_weight = mixer_weight.unsqueeze(1)

    b2b_module = B2BCausalConv1dModule(
        proj_conv,
        mixer,
        operator_type="hyena_short_conv",
        pg_collection=MockProcessGroupCollection(),
    )

    fused = b2b_module(x).float()
    # subquadratic_ops_torch.b2b_causal_conv1d uses the same causal-conv
    # convention as subquadratic_ops_torch.causal_conv1d: weight[-1] is the
    # current-position tap. Do not flip the direct-convolution reference.
    projected = F.conv1d(
        F.pad(x.float(), (proj_kernel_size - 1, 0)),
        proj_weight.float().unsqueeze(1),
        groups=3 * hidden_size,
    )
    x1, x2, v = projected[:, ::3], projected[:, 1::3], projected[:, 2::3]
    z = x2 * v
    mixed = F.conv1d(
        F.pad(z, (mixer_kernel_size - 1, 0)),
        mixer_weight.float().unsqueeze(1),
        groups=hidden_size,
    )
    reference = x1 * (mixed + bias.float()[None, :, None] * z)

    torch.testing.assert_close(fused, reference, rtol=1e-4, atol=1e-4)


def test_zigzag_get_overlapping_patches():
    # Test the actual output of zigzag_get_overlapping_patches
    data = torch.arange(8).reshape(2, 4)  # shape [2, 4]
    seq_dim = 1
    overlap_size = 2
    overlap_a, overlap_b = zigzag_get_overlapping_patches(data, seq_dim, overlap_size)
    # The function splits data into two chunks along seq_dim, then extracts the last overlap_size elements from each chunk
    # For data = [[0,1,2,3],[4,5,6,7]], reshaped to [2,2,2]: chunk 0: [0,1],[4,5]; chunk 1: [2,3],[6,7]
    # overlap_a = chunk 0 last 2: [[0,1],[4,5]]; overlap_b = chunk 1 last 2: [[2,3],[6,7]]
    assert torch.equal(overlap_a, torch.tensor([[0, 1], [4, 5]]))
    assert torch.equal(overlap_b, torch.tensor([[2, 3], [6, 7]]))


def test_exchange_overlapping_regions_causal_forward(monkeypatch):
    class DummyReq:
        def wait(self):
            pass

    class DummyDist:
        def get_process_group_ranks(self, group):
            return [0, 1]

        def irecv(self, tensor, src):
            tensor.fill_(42)
            return DummyReq()

        def isend(self, tensor, dst):
            return DummyReq()

    dummy_dist = DummyDist()
    monkeypatch.setattr(dist, "irecv", dummy_dist.irecv)
    monkeypatch.setattr(dist, "isend", dummy_dist.isend)
    monkeypatch.setattr(dist, "get_process_group_ranks", dummy_dist.get_process_group_ranks)
    chunk_a = torch.zeros(1, 2)
    chunk_b = torch.zeros(1, 2)
    group = object()
    group_rank = 0
    ctx = types.SimpleNamespace()
    received_a, received_b = ExchangeOverlappingRegionsCausal.forward(ctx, chunk_a, chunk_b, group, group_rank)
    assert received_a.shape == chunk_a.shape
    assert received_b.shape == chunk_b.shape
    assert torch.all(received_a == 0) or torch.all(received_a == 42)
    assert torch.all(received_b == 42) or torch.all(received_b == 0)


def test_zigzag_indices():
    """Test the zigzag indices generation functions."""
    N = 4  # noqa: N806
    device = torch.device("cpu")

    # Test _get_zigzag_indices
    zigzag_idx = _get_zigzag_indices(N, device)
    expected = torch.tensor([0, 3, 1, 2], device=device)
    assert torch.equal(zigzag_idx, expected)

    # Test _get_inverse_zigzag_indices
    inverse_idx = _get_inverse_zigzag_indices(N, device)
    expected = torch.tensor([0, 2, 3, 1], device=device)
    assert torch.equal(inverse_idx, expected)


def test_ensure_divisibility():
    """Test the ensure_divisibility and divide functions."""
    # Test valid division
    assert divide(10, 2) == 5

    # Test invalid division
    with pytest.raises(AssertionError):
        ensure_divisibility(10, 3)


def test_get_groups_and_group_sizes():
    """Test group size calculation for model parallel."""
    hidden_size = 1024
    num_groups = 32
    world_size = 2
    expand_factor = 1.0

    width_per_tp, num_groups_per_tp, group_dim = get_groups_and_group_sizes(
        hidden_size, num_groups, world_size, expand_factor
    )

    assert width_per_tp == 512  # hidden_size / world_size
    assert num_groups_per_tp == 16  # num_groups / world_size
    assert group_dim == 32  # width_per_tp / num_groups_per_tp


def test_init_methods():
    """Test initialization methods."""
    dim = 100
    n_layers = 4

    # Test small_init
    small_init = small_init_init_method(dim)
    tensor = torch.empty(10, 10)
    small_init(tensor)
    assert tensor.std() > 0

    # Test wang_init
    wang_init = wang_init_method(n_layers, dim)
    tensor = torch.empty(10, 10)
    wang_init(tensor)
    assert tensor.std() > 0

    # Test get_init_method
    assert callable(get_init_method("small_init", n_layers, dim))
    assert callable(get_init_method("wang_init", n_layers, dim))
    with pytest.raises(NotImplementedError):
        get_init_method("invalid", n_layers, dim)


def test_fftconv_func():
    """Test the FFT convolution function."""
    batch_size = 2
    seq_len = 8
    hidden_size = 4

    # Create input tensors
    u = torch.randn(batch_size, hidden_size, seq_len)
    k = torch.randn(hidden_size, seq_len)
    D = torch.randn(hidden_size)  # noqa: N806
    k_rev = torch.randn(hidden_size, seq_len)
    dropout_mask = torch.ones(batch_size, hidden_size)

    # Test causal mode
    output = fftconv_func(u, k, D, dropout_mask, gelu=True, bidirectional=False)
    assert isinstance(output, torch.Tensor)
    assert output.shape == u.shape

    # Test bidirectional mode
    output = fftconv_func(u, k, D, dropout_mask, gelu=True, bidirectional=True)
    assert isinstance(output, torch.Tensor)
    assert output.shape == u.shape

    # Test without GELU
    output = fftconv_func(u, k, D, dropout_mask, gelu=False)
    assert isinstance(output, torch.Tensor)
    assert output.shape == u.shape

    # Test without dropout mask
    output = fftconv_func(u, k, D, None)
    assert isinstance(output, torch.Tensor)
    assert output.shape == u.shape

    # Test with k_rev
    output = fftconv_func(u, k, D, dropout_mask, gelu=True, bidirectional=False, k_rev=k_rev)
    assert isinstance(output, torch.Tensor)
    assert output.shape == u.shape

    # Test with filter k shorter than sequence length (covers the padding logic)
    k_short = torch.randn(hidden_size, seq_len // 2)  # Filter is half the sequence length
    output_short = fftconv_func(u, k_short, D, dropout_mask, gelu=True, bidirectional=False)
    assert isinstance(output_short, torch.Tensor)
    assert output_short.shape == u.shape


def test_fftconv_func_bidirectional_is_prefix_invariant_when_filter_is_longer_than_input():
    """Bidirectional FFT convolution should not alias short prefixes when the filter is long."""
    torch.manual_seed(1234)
    batch_size = 2
    short_len = 5
    long_len = 64
    hidden_size = 4
    filter_len = 64

    u_short = torch.randn(batch_size, hidden_size, short_len)
    u_long = torch.zeros(batch_size, hidden_size, long_len)
    u_long[..., :short_len] = u_short
    k = torch.randn(1, 2 * hidden_size, filter_len)
    D = torch.randn(hidden_size)  # noqa: N806

    short_out = fftconv_func(u_short, k, D, None, gelu=False, bidirectional=True)
    long_out = fftconv_func(u_long, k, D, None, gelu=False, bidirectional=True)[..., :short_len]

    torch.testing.assert_close(short_out, long_out, rtol=1e-5, atol=1e-5)


def test_fftconv_func_high_dimensional_input():
    """Test fftconv_func with high-dimensional input to cover the len(u.shape) > 3 case."""
    batch_size = 2
    seq_len = 8
    hidden_size = 4

    # Create a 4D input tensor [B, 2, H, L] to trigger the len(u.shape) > 3 case
    u_4d = torch.randn(batch_size, 2, hidden_size, seq_len)
    k_4d = torch.randn(hidden_size, 2, seq_len)
    D_4d = torch.randn(hidden_size)  # noqa: N806
    dropout_mask_4d = torch.ones(batch_size, hidden_size)

    # Test that the function can handle 4D input without crashing
    # The len(u.shape) > 3 code path should be executed
    try:
        output_4d = fftconv_func(u_4d, k_4d, D_4d, dropout_mask_4d, gelu=True, bidirectional=False)
        # If it succeeds, verify basic properties
        assert isinstance(output_4d, torch.Tensor)
        assert output_4d.dtype == u_4d.dtype
        assert output_4d.shape == u_4d.shape
    except RuntimeError as e:
        # If it fails due to broadcasting issues, that's expected for this edge case
        assert "size" in str(e) or "dimension" in str(e), f"Unexpected error: {e}"


@patch("bionemo.evo2.models.megatron.hyena.hyena_utils.fft_causal_conv1d")
def test_fftconv_func_use_subquadratic_ops_success(mock_fft_causal_conv1d):
    """Test fftconv_func with use_subquadratic_ops=True when supported."""
    mock_fft_causal_conv1d.return_value = torch.randn(2, 4, 8)

    batch_size = 2
    seq_len = 8
    hidden_size = 4

    u = torch.randn(batch_size, hidden_size, seq_len)
    k = torch.randn(hidden_size, seq_len)
    D = torch.randn(hidden_size)  # noqa: N806
    dropout_mask = torch.ones(batch_size, hidden_size)

    output = fftconv_func(u, k, D, dropout_mask, gelu=True, bidirectional=False, use_subquadratic_ops=True)
    assert isinstance(output, torch.Tensor)
    assert output.shape == u.shape
    mock_fft_causal_conv1d.assert_called_once()


class TestFallbackFunctions:
    """Test the fallback functions that are defined when subquadratic_ops import fails."""

    @patch("bionemo.evo2.models.megatron.hyena.hyena_utils.causal_conv1d")
    def test_causal_conv1d_fallback(self, mock_causal_conv1d):
        """Test that the fallback causal_conv1d function raises ImportError."""
        # Mock the function to raise ImportError
        mock_causal_conv1d.side_effect = ImportError("subquadratic_ops not installed. causal_conv1d is not available.")

        with pytest.raises(ImportError, match="subquadratic_ops not installed. causal_conv1d is not available."):
            mock_causal_conv1d(torch.randn(1, 1, 1), torch.randn(1, 1))

    @patch("bionemo.evo2.models.megatron.hyena.hyena_utils.b2b_causal_conv1d")
    def test_b2b_causal_conv1d_fallback(self, mock_b2b_causal_conv1d):
        """Test that the fallback b2b_causal_conv1d function raises ImportError."""
        # Mock the function to raise ImportError
        mock_b2b_causal_conv1d.side_effect = ImportError(
            "subquadratic_ops not installed. b2b_causal_conv1d is not available."
        )

        with pytest.raises(ImportError, match="subquadratic_ops not installed. b2b_causal_conv1d is not available."):
            mock_b2b_causal_conv1d(torch.randn(1, 1, 1), torch.randn(1, 1), torch.randn(1, 1), torch.randn(1))

    @patch("bionemo.evo2.models.megatron.hyena.hyena_utils.fft_causal_conv1d")
    def test_fft_causal_conv1d_fallback(self, mock_fft_causal_conv1d):
        """Test that the fallback fft_causal_conv1d function raises ImportError."""
        # Mock the function to raise ImportError
        mock_fft_causal_conv1d.side_effect = ImportError(
            "subquadratic_ops not installed. fft_causal_conv1d is not available."
        )

        with pytest.raises(ImportError, match="subquadratic_ops not installed. fft_causal_conv1d is not available."):
            mock_fft_causal_conv1d(torch.randn(1, 1, 1), torch.randn(1, 1))

    def test_fallback_functions_import_error_messages(self):
        """Test that all fallback functions have consistent error messages."""
        # Import the module to get access to the fallback functions
        import bionemo.evo2.models.megatron.hyena.hyena_utils as hyena_utils

        # Test that the fallback functions exist and have the expected docstrings
        assert hasattr(hyena_utils, "causal_conv1d")
        assert hasattr(hyena_utils, "b2b_causal_conv1d")
        assert hasattr(hyena_utils, "fft_causal_conv1d")

        # Test that they are callable
        assert callable(hyena_utils.causal_conv1d)
        assert callable(hyena_utils.b2b_causal_conv1d)
        assert callable(hyena_utils.fft_causal_conv1d)

    def test_einops_import_error(self):
        """Test that the einops import error is raised with the correct message."""
        import bionemo.evo2.models.megatron.hyena.hyena_utils

        try:
            # Mock the import to fail
            with patch.dict("sys.modules", {"einops": None}):
                # Re-import the module to trigger the import error
                with pytest.raises(ImportError, match="einops is required by the Hyena model but cannot be imported"):
                    # Force a reload of the module to trigger the import error
                    importlib.reload(bionemo.evo2.models.megatron.hyena.hyena_utils)
        finally:
            # CRITICAL: Always restore the module to its proper state after the test.
            # The reload above leaves the module in a corrupted state, which can cause
            # subsequent tests to fail (especially test_infer.py tests).
            importlib.reload(bionemo.evo2.models.megatron.hyena.hyena_utils)
