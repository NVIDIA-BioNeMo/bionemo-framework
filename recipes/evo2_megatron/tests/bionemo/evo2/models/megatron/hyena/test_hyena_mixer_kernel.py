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


import copy
import importlib.util

import pytest
import torch
from einops import rearrange

from bionemo.evo2.models.evo2_provider import HyenaNVTestModelProvider, HyenaTestModelProvider
from bionemo.evo2.models.megatron.hyena.hyena_config import HyenaConfig
from bionemo.evo2.models.megatron.hyena.hyena_layer_specs import hyena_stack_spec_no_te
from bionemo.evo2.models.megatron.hyena.hyena_mixer import HyenaMixer
from bionemo.evo2.models.megatron.hyena.hyena_utils import ImplicitModalFilter
from bionemo.evo2.models.megatron.hyena.subquadratic_safety import ensure_subquadratic_ops_supported

from ....utils import distributed_model_parallel_state


try:
    import subquadratic_ops_torch  # noqa: F401

    HAVE_SUBQUADRATIC_OPS = True
except ImportError:
    HAVE_SUBQUADRATIC_OPS = False


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
def hyena_config() -> HyenaConfig:  # noqa: D103
    config = HyenaConfig()
    config.num_groups_hyena = 4096
    config.num_groups_hyena_short = 256
    config.num_groups_hyena_medium = 256
    return config


@pytest.fixture(
    params=[
        pytest.param("hyena_short_conv", id="short"),
        pytest.param("hyena_medium_conv", id="medium"),
        pytest.param("hyena", id="long"),
    ]
)
def operator_type(request):
    """Parametrized operator type fixture."""
    return request.param


class MixerModuleWrapper(torch.nn.Module):  # noqa: D101
    def __init__(  # noqa: D107
        self, hyena_config, hyena_test_config, seq_len, use_subquadratic_ops=False, operator_type="hyena_medium_conv"
    ):
        super().__init__()

        # Create necessary submodules - use the mixer submodules like in the regular mixer fixture
        submodules = hyena_stack_spec_no_te.submodules.hyena_layer.submodules.mixer.submodules

        # Set the b2b parameter in the config
        hyena_test_config.use_subquadratic_ops = use_subquadratic_ops
        self.use_subquadratic_ops = use_subquadratic_ops
        self.operator_type = operator_type

        print("Creating HyenaMixer...")
        self.mixer = HyenaMixer(
            transformer_config=hyena_test_config,
            hyena_config=hyena_config,
            max_sequence_length=seq_len,
            submodules=submodules,
            layer_number=1,
            operator_type=self.operator_type,
        )

    def forward(self, x, _use_cp=False):  # noqa: D102
        if self.use_subquadratic_ops and self.operator_type != "hyena":
            z = self.mixer.b2b_kernel(x, _use_cp=_use_cp)
        else:  # long `hyena` operator internally sets use_subquadratic_ops from config
            features = self.mixer.hyena_proj_conv(x, _use_cp=_use_cp)
            x1, x2, v = rearrange(
                features, "b (g dg p) l -> b (g dg) p l", p=3, g=self.mixer.num_groups_per_tp_rank
            ).unbind(dim=2)
            z = self.mixer.mixer(x1, x2, v, _hyena_use_cp=_use_cp)
        return z


@pytest.fixture
def mixer_kernel_hyena_only(test_config: HyenaTestModelProvider, hyena_config: HyenaConfig):
    """Create a HyenaMixer instance for testing with CUDA kernel implementation - only for hyena operator."""
    with distributed_model_parallel_state():
        mixer_kernel = MixerModuleWrapper(
            hyena_config, test_config, seq_len=512, use_subquadratic_ops=True, operator_type="hyena"
        )
        yield mixer_kernel


@pytest.mark.skipif(
    importlib.util.find_spec("subquadratic_ops_torch") is None or not HAVE_SUBQUADRATIC_OPS,
    reason="subquadratic_ops_torch is not installed",
)
def test_implicit_filter(mixer_kernel_hyena_only: MixerModuleWrapper):
    """Test that the implicit filter is properly initialized with correct parameters and attributes."""
    # Check that the filter is the correct type
    assert isinstance(mixer_kernel_hyena_only.mixer.mixer.filter, ImplicitModalFilter), (
        f"mixer_kernel_hyena_only.mixer.mixer.filter must be an ImplicitModalFilter, "
        f"got {type(mixer_kernel_hyena_only.mixer.mixer.filter)}"
    )

    filter_obj = mixer_kernel_hyena_only.mixer.mixer.filter

    # Check that the filter has the required attributes
    assert hasattr(filter_obj, "implicit_filter"), "Filter must have 'implicit_filter' attribute"
    assert callable(filter_obj.implicit_filter), "implicit_filter attribute must be callable"
    assert hasattr(filter_obj, "use_subquadratic_ops"), "Filter must have 'use_subquadratic_ops' attribute"

    # Verify that use_subquadratic_ops is True for kernel implementation
    assert filter_obj.use_subquadratic_ops is True, (
        f"Filter use_subquadratic_ops should be True for kernel implementation, got {filter_obj.use_subquadratic_ops}"
    )

    # create a reference filter with use_subquadratic_ops = False
    pg_backup = filter_obj.pg_collection
    cp_backup = filter_obj.context_parallel_group
    filter_obj.pg_collection = None
    filter_obj.context_parallel_group = None
    reference_filter = copy.deepcopy(filter_obj)
    filter_obj.pg_collection = pg_backup  # bypass the pg_collection since it doesn't work with deepcopy
    filter_obj.context_parallel_group = cp_backup
    reference_filter.pg_collection = pg_backup
    reference_filter.context_parallel_group = cp_backup
    reference_filter.use_subquadratic_ops = False
    reference_filter.implicit_filter = None
    reference_filter.t = None

    # Test forward pass comparison
    L = 10  # noqa: N806

    # Get outputs from both filters - handle the return value correctly
    filter_output = filter_obj.filter(L)
    reference_output = reference_filter.filter(L)

    # Handle case where compute_filter returns (h, None) but implicit_filter returns just h
    if isinstance(filter_output, tuple):
        filter_output = filter_output[0]
    if isinstance(reference_output, tuple):
        reference_output = reference_output[0]

    # Verify forward pass output properties
    assert filter_output is not None, "Filter output should not be None"
    assert reference_output is not None, "Reference filter output should not be None"
    assert filter_output.shape == (
        1,
        filter_obj.d_model,
        L,
    ), f"Filter output should have shape (1, {filter_obj.d_model}, {L}), got {filter_output.shape}"
    assert reference_output.shape == (
        1,
        filter_obj.d_model,
        L,
    ), f"Reference filter output should have shape (1, {filter_obj.d_model}, {L}), got {reference_output.shape}"

    # Compare forward outputs between the two implementations
    torch.testing.assert_close(filter_output, reference_output, msg=f"Filter outputs do not match for L={L}")

    # Test backward pass comparison between filter and reference filter
    # Create input tensor that requires gradients
    input_tensor = torch.randn(1, filter_obj.d_model, L, device=filter_obj.device, requires_grad=True)

    # Test filter backward pass
    filter_loss = torch.sum(filter_output * input_tensor)
    filter_loss.backward()

    # Check that gradients were computed for the filter parameters
    assert filter_obj.gamma.grad is not None, f"gamma.grad should not be None for L={L}"
    assert filter_obj.R.grad is not None, f"R.grad should not be None for L={L}"
    assert filter_obj.p.grad is not None, f"p.grad should not be None for L={L}"

    # Store filter gradients
    filter_gamma_grad = filter_obj.gamma.grad.clone()
    filter_R_grad = filter_obj.R.grad.clone()  # noqa: N806
    filter_p_grad = filter_obj.p.grad.clone()

    # Clear gradients
    filter_obj.zero_grad()
    input_tensor.grad = None

    # Test reference filter backward pass
    reference_loss = torch.sum(reference_output * input_tensor)
    reference_loss.backward()

    # Check that gradients were computed for the reference filter parameters
    assert reference_filter.gamma.grad is not None, f"reference_filter.gamma.grad should not be None for L={L}"
    assert reference_filter.R.grad is not None, f"reference_filter.R.grad should not be None for L={L}"
    assert reference_filter.p.grad is not None, f"reference_filter.p.grad should not be None for L={L}"

    # Store reference filter gradients
    reference_gamma_grad = reference_filter.gamma.grad.clone()
    reference_R_grad = reference_filter.R.grad.clone()  # noqa: N806
    reference_p_grad = reference_filter.p.grad.clone()

    # Clear gradients
    reference_filter.zero_grad()
    input_tensor.grad = None

    # Compare gradients between filter and reference filter
    torch.testing.assert_close(filter_gamma_grad, reference_gamma_grad, msg=f"gamma gradients do not match for L={L}")
    torch.testing.assert_close(filter_R_grad, reference_R_grad, msg=f"R gradients do not match for L={L}")
    torch.testing.assert_close(filter_p_grad, reference_p_grad, msg=f"p gradients do not match for L={L}")


@pytest.mark.skipif(
    importlib.util.find_spec("subquadratic_ops_torch") is None or not HAVE_SUBQUADRATIC_OPS,
    reason="subquadratic_ops_torch is not installed",
)
def test_subquadratic_ops_kernel(  # noqa: D103
    test_config: HyenaTestModelProvider, hyena_config: HyenaConfig, config_type, operator_type
):
    # Skip bf16 with short convolution due to numerical instability
    if test_config.params_dtype == torch.bfloat16 and operator_type == "hyena_short_conv":
        pytest.skip("bf16 with short convolution is skipped due to numerical instability")
    try:
        ensure_subquadratic_ops_supported()
    except RuntimeError as e:
        pytest.xfail(str(e))

    with distributed_model_parallel_state():
        # Create both models inside the same distributed context
        mixer = MixerModuleWrapper(
            hyena_config, test_config, seq_len=512, use_subquadratic_ops=False, operator_type=operator_type
        )
        mixer_kernel = MixerModuleWrapper(
            hyena_config, test_config, seq_len=512, use_subquadratic_ops=True, operator_type=operator_type
        )

        # Copy all parameters from mixer_kernel to mixer to ensure identical initialization
        mixer.load_state_dict(mixer_kernel.state_dict())

        # Verify parameters are now identical
        for (name1, param1), (name2, param2) in zip(mixer.named_parameters(), mixer_kernel.named_parameters()):
            assert name1 == name2, f"Parameter name mismatch {name1} != {name2}"
            assert torch.equal(param1, param2), f"Parameter mismatch for {name1}"

        batch_size = 2
        seq_len = 512
        input_features = torch.rand(
            (batch_size, mixer.mixer.hidden_size * 3, seq_len),
            dtype=mixer.mixer.transformer_config.params_dtype,
            device=torch.cuda.current_device(),
        )

        # PyTorch Mixer
        output_features = mixer(input_features)
        assert output_features.shape == (
            batch_size,
            mixer.mixer.hidden_size,
            seq_len,
        ), (
            f"output_features.shape: {output_features.shape}, batch_size: {batch_size}, mixer.mixer.hidden_size: {mixer.mixer.hidden_size}, seq_len: {seq_len}"
        )

        loss = output_features.float().mean()
        loss.backward()

        # Store the gradients for later comparison.
        grads = []
        for n, p in mixer.named_parameters():
            if p.grad is not None:
                grads.append((n, p.grad.clone()))

        mixer.zero_grad()

        # CUDA kernel in Mixer
        output_features_kernel = mixer_kernel(input_features)
        assert output_features_kernel.shape == (
            batch_size,
            mixer_kernel.mixer.hidden_size,
            seq_len,
        ), (
            f"output_features_kernel.shape: {output_features_kernel.shape}, batch_size: {batch_size}, mixer_kernel.mixer.hidden_size: {mixer_kernel.mixer.hidden_size}, seq_len: {seq_len}"
        )

        loss_kernel = output_features_kernel.float().mean()
        loss_kernel.backward()

        # Store the gradients for later comparison.
        grads_kernel = []
        for n, p in mixer_kernel.named_parameters():
            if p.grad is not None:
                grads_kernel.append((n, p.grad.clone()))

        mixer_kernel.zero_grad()

        # Compare results between PyTorch and CUDA kernel implementations
        torch.testing.assert_close(output_features, output_features_kernel, rtol=0.02, atol=2e-4)
        torch.testing.assert_close(loss, loss_kernel, msg=f"Loss mismatch for {operator_type}")

        # Compare gradients
        assert len(grads) == len(grads_kernel), f"Gradient count mismatch for {operator_type}"

        gradient_mismatch = False
        for (n, g), (n_kernel, g_kernel) in zip(grads, grads_kernel):
            try:
                torch.testing.assert_close(g, g_kernel, msg=f"Gradient mismatch for {operator_type} - {n}")
            except AssertionError as e:  # noqa: PERF203
                gradient_mismatch = True
                print(f"Gradient mismatch for {operator_type} - {n}: {e}")

        if gradient_mismatch:
            print(f"There were gradient mismatches for {operator_type}!")
        else:
            print(f"All gradients matched successfully for {operator_type}!")
