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

"""Tests that verify actual precision of intermediate tensors during forward passes.

Checks that:
- Triangular multiplication batched GEMMs stay in FP32 by default
- tri_einsum toggle controls FP32 vs ambient dtype for triangular matmuls
- te.autocast correctly enables/disables FP8 for specific blocks
- Component precision overrides correctly keep components out of FP8
"""

import sys
from pathlib import Path

import torch
import transformer_engine.pytorch as te


sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor

from miniformer_te import BlockTE, MiniFormerTE, TransitionUpdateTE, TriangularUpdateTE
from model_te import FoldingTrunkTE
from quantization import ComponentPrecisionConfig
from te_utils import te_linear_nd, tri_mul_bmm, tri_mul_einsum, tri_mul_fp8_bdnn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, N, DIM = 2, 16, 64


class TestTriMulPrecision:
    """Verify the triangular multiplication batched GEMMs run in the expected precision."""

    def test_bmm_intermediates_are_fp32_by_default(self):
        """Even with BF16 input, the bmm should compute in FP32 via .float() when tri_einsum="off"."""
        mod = TriangularUpdateTE(dim=DIM).to(DEVICE)
        x_bf16 = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        captured = {}
        orig_bmm = torch.bmm

        def patched_bmm(a, b):
            captured["input_dtype"] = a.dtype
            result = orig_bmm(a, b)
            captured["output_dtype"] = result.dtype
            return result

        torch.bmm = patched_bmm
        try:
            mod(x_bf16, mask)
        finally:
            torch.bmm = orig_bmm

        assert captured["input_dtype"] == torch.float32, f"BMM input should be FP32 but got {captured['input_dtype']}"
        assert captured["output_dtype"] == torch.float32, (
            f"BMM output should be FP32 but got {captured['output_dtype']}"
        )

    def test_bmm_fp32_with_fp8_block(self):
        """BMM stays FP32 even when block is wrapped in te.autocast(enabled=True)."""
        mod = TriangularUpdateTE(dim=DIM).to(DEVICE)
        x_bf16 = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        captured = {}
        orig_bmm = torch.bmm

        def patched_bmm(a, b):
            captured["dtype"] = a.dtype
            return orig_bmm(a, b)

        torch.bmm = patched_bmm
        try:
            with te.autocast(enabled=True):
                mod(x_bf16, mask)
        finally:
            torch.bmm = orig_bmm

        assert captured["dtype"] == torch.float32, f"BMM should be FP32 inside te.autocast but got {captured['dtype']}"

    def test_bmm_bf16_when_tri_einsum_bf16(self):
        """BMM runs in BF16 when tri_einsum="bf16" (ambient dtype, no .float() cast)."""
        cp = ComponentPrecisionConfig(tri_einsum="bf16")
        mod = TriangularUpdateTE(dim=DIM, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x_bf16 = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        captured = {}
        orig_bmm = torch.bmm

        def patched_bmm(a, b):
            captured["dtype"] = a.dtype
            return orig_bmm(a, b)

        torch.bmm = patched_bmm
        try:
            mod(x_bf16, mask)
        finally:
            torch.bmm = orig_bmm

        assert captured["dtype"] == torch.bfloat16, (
            f"BMM should be BF16 when tri_einsum='bf16' but got {captured['dtype']}"
        )

    def test_bmm_bf16_backward_compat_bool_true(self):
        """Bool True normalizes to 'bf16' via __post_init__."""
        cp = ComponentPrecisionConfig(tri_einsum=True)
        assert cp.tri_einsum == "bf16"

    def test_bmm_bf16_backward_compat_bool_false(self):
        """Bool False normalizes to 'off' via __post_init__."""
        cp = ComponentPrecisionConfig(tri_einsum=False)
        assert cp.tri_einsum == "off"

    def test_tri_impl_defaults_to_bmm(self):
        """Hydra config should default to the current batched-matmul path."""
        cp = ComponentPrecisionConfig()
        assert cp.tri_impl == "bmm"

    def test_tri_impl_validation(self):
        """Unsupported triangular multiplication backends should fail fast."""
        try:
            ComponentPrecisionConfig(tri_impl="not_a_backend")
        except ValueError as exc:
            assert "tri_impl must be one of" in str(exc)
        else:
            raise AssertionError("Expected invalid tri_impl to raise ValueError")

    def test_tri_impl_accepts_einsum(self):
        """The original einsum path should remain Hydra-selectable."""
        cp = ComponentPrecisionConfig(tri_impl="einsum")
        assert cp.tri_impl == "einsum"

    def test_tri_impl_accepts_fp8_bmm(self):
        """The MXFP8 batched GEMM path should be Hydra-selectable."""
        cp = ComponentPrecisionConfig(tri_impl="fp8_bmm")
        assert cp.tri_impl == "fp8_bmm"

    def test_tri_impl_accepts_flash_trimul(self):
        """The fused Triton gate+tri-mul path should be Hydra-selectable."""
        cp = ComponentPrecisionConfig(tri_impl="flash_trimul")
        assert cp.tri_impl == "flash_trimul"

    def test_tri_impl_accepts_fp8_cublaslt(self):
        """The cuBLASLt MXFP8 path should be Hydra-selectable."""
        cp = ComponentPrecisionConfig(tri_impl="fp8_cublaslt")
        assert cp.tri_impl == "fp8_cublaslt"

    def test_tri_impl_accepts_fused_fp8(self):
        """The fused FP8 recipe path should be Hydra-selectable."""
        cp = ComponentPrecisionConfig(tri_impl="fused_fp8")
        assert cp.tri_impl == "fused_fp8"

    def test_bmm_fp32_when_tri_einsum_off(self):
        """BMM stays FP32 when tri_einsum="off" (explicit .float() cast)."""
        cp = ComponentPrecisionConfig(tri_einsum="off")
        mod = TriangularUpdateTE(dim=DIM, component_precision=cp).to(DEVICE)
        x_bf16 = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        captured = {}
        orig_bmm = torch.bmm

        def patched_bmm(a, b):
            captured["dtype"] = a.dtype
            return orig_bmm(a, b)

        torch.bmm = patched_bmm
        try:
            mod(x_bf16, mask)
        finally:
            torch.bmm = orig_bmm

        assert captured["dtype"] == torch.float32, (
            f"BMM should be FP32 when tri_einsum='off' but got {captured['dtype']}"
        )

    def test_bmm_fp32_with_no_component_precision(self):
        """Without ComponentPrecisionConfig, BMM defaults to FP32."""
        mod = TriangularUpdateTE(dim=DIM).to(DEVICE)
        x_bf16 = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        captured = {}
        orig_bmm = torch.bmm

        def patched_bmm(a, b):
            captured["dtype"] = a.dtype
            return orig_bmm(a, b)

        torch.bmm = patched_bmm
        try:
            mod(x_bf16, mask)
        finally:
            torch.bmm = orig_bmm

        assert captured["dtype"] == torch.float32, (
            f"BMM should default to FP32 without component_precision but got {captured['dtype']}"
        )

    def test_cublas_xbdnn_backend_is_driven_by_component_precision(self):
        """The backend selector should come from ComponentPrecisionConfig, not only env vars."""
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(tri_einsum="bf16", tri_impl="cublas_xbdnn")
        mod = TriangularUpdateTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(B, N, N, 128, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        assert out.shape == (B, N, N, 128)
        assert out.dtype == torch.bfloat16

    def test_fp8_bmm_backend_runs_forward_and_backward(self):
        """The fp8_bmm backend should produce finite activations and gradients at production-aligned shapes."""
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(tri_einsum="bf16", tri_impl="fp8_bmm")
        mod = TriangularUpdateTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(B, 32, 32, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(B, 32, 32, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        loss = out.float().square().mean()
        loss.backward()
        assert out.shape == (B, 32, 32, 128)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        assert torch.isfinite(x.grad).all()

    def test_tri_saved_tensor_fp8_hook_runs_forward_and_backward(self):
        """Experimental FP8 saved-tensor packing should preserve a usable backward path."""
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(tri_saved_tensors_fp8=True)
        mod = TriangularUpdateTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(1, 16, 16, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(1, 16, 16, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        loss = out.float().square().mean()
        loss.backward()
        assert out.shape == (1, 16, 16, 128)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        assert torch.isfinite(x.grad).all()


class TestSavedTensorPacking:
    """Verify experimental saved-activation FP8 packing paths remain functional."""

    def test_ffn_saved_tensor_fp8_hook_runs_forward_and_backward(self):
        """TransitionUpdateTE should still train when BF16 saved tensors are packed to FP8."""
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(ffn_saved_tensors_fp8=True)
        mod = TransitionUpdateTE(dim=128, hidden=512, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(1, 16, 16, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        out = mod(x)
        loss = out.float().square().mean()
        loss.backward()
        assert out.shape == (1, 16, 16, 128)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        assert torch.isfinite(x.grad).all()

    def test_ffn_checkpoint_saved_tensor_fp8_hook_hits_checkpoint_input(self, monkeypatch):
        """Checkpointed FFN packing should attach at the checkpoint boundary and see the saved BF16 input."""
        if not torch.cuda.is_available():
            return

        import miniformer_te as miniformer_te_module

        seen = []
        original_ctx = miniformer_te_module.saved_tensor_fp8_context

        def instrumented_ctx(*, enabled, include_bf16=True, include_fp32=False, include_leaf_tensors=False):
            base = original_ctx(
                enabled=enabled,
                include_bf16=include_bf16,
                include_fp32=include_fp32,
                include_leaf_tensors=include_leaf_tensors,
            )
            if not enabled:
                return base

            def _pack(tensor):
                if isinstance(tensor, torch.Tensor):
                    seen.append(
                        {
                            "shape": tuple(tensor.shape),
                            "dtype": tensor.dtype,
                            "is_leaf": bool(tensor.is_leaf),
                        }
                    )
                return tensor

            def _unpack(x):
                return x

            return torch.autograd.graph.saved_tensors_hooks(_pack, _unpack)

        monkeypatch.setattr(miniformer_te_module, "saved_tensor_fp8_context", instrumented_ctx)

        cp = ComponentPrecisionConfig(ffn=True, ffn_saved_tensors_fp8=True, ffn_checkpoint_reentrant=True)
        mod = BlockTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(1, 16, 16, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(1, 16, 16, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        loss = out.float().square().mean()
        loss.backward()

        assert any(entry["shape"] == (1, 16, 16, 128) and entry["dtype"] == torch.bfloat16 for entry in seen)

    def test_tri_checkpoint_saved_tensor_fp8_hook_hits_checkpoint_inputs(self, monkeypatch):
        """Checkpointed triangular packing should attach at the checkpoint boundary and see saved inputs."""
        if not torch.cuda.is_available():
            return

        import miniformer_te as miniformer_te_module

        seen = []
        original_ctx = miniformer_te_module.saved_tensor_fp8_context

        def instrumented_ctx(*, enabled, include_bf16=True, include_fp32=False, include_leaf_tensors=False):
            base = original_ctx(
                enabled=enabled,
                include_bf16=include_bf16,
                include_fp32=include_fp32,
                include_leaf_tensors=include_leaf_tensors,
            )
            if not enabled:
                return base

            def _pack(tensor):
                if isinstance(tensor, torch.Tensor):
                    seen.append(
                        {
                            "shape": tuple(tensor.shape),
                            "dtype": tensor.dtype,
                            "is_leaf": bool(tensor.is_leaf),
                        }
                    )
                return tensor

            def _unpack(x):
                return x

            return torch.autograd.graph.saved_tensors_hooks(_pack, _unpack)

        monkeypatch.setattr(miniformer_te_module, "saved_tensor_fp8_context", instrumented_ctx)

        cp = ComponentPrecisionConfig(
            tri_saved_tensors_fp8=True,
            tri_checkpoint_reentrant=True,
        )
        mod = BlockTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(1, 16, 16, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(1, 16, 16, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        loss = out.float().square().mean()
        loss.backward()

        assert any(entry["shape"] == (1, 16, 16, 128) and entry["dtype"] == torch.bfloat16 for entry in seen)
        assert any(entry["shape"] == (1, 16, 16) and entry["dtype"] == torch.bfloat16 for entry in seen)

    def test_tri_prefix_checkpoint_reentrant_runs_forward_and_backward(self):
        """Checkpointing only the triangular projection/gating prefix should still train."""
        if not torch.cuda.is_available():
            return

        cp = ComponentPrecisionConfig(
            tri_saved_tensors_fp8=True,
            tri_prefix_checkpoint_reentrant=True,
        )
        mod = BlockTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(1, 16, 16, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(1, 16, 16, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        loss = out.float().square().mean()
        loss.backward()

        assert out.shape == (1, 16, 16, 128)
        assert torch.isfinite(out).all()
        assert torch.isfinite(x.grad).all()

    def test_ffn_fused_subgraph_fp8_runs_forward_and_backward(self):
        """Custom-autograd TransitionUpdateTE should still backprop through TE modules."""
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(ffn=True, ffn_fused_subgraph_fp8=True)
        mod = TransitionUpdateTE(dim=128, hidden=512, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(1, 16, 16, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        out = mod(x)
        loss = out.float().square().mean()
        loss.backward()
        assert out.shape == (1, 16, 16, 128)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        assert torch.isfinite(x.grad).all()

    def test_fp8_grouped_backend_runs_forward_and_backward(self):
        """The bulk-packed grouped FP8 backend should produce finite activations and gradients."""
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(tri_einsum="bf16", tri_impl="fp8_grouped")
        mod = TriangularUpdateTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(B, 32, 32, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(B, 32, 32, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        loss = out.float().square().mean()
        loss.backward()
        assert out.shape == (B, 32, 32, 128)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        assert torch.isfinite(x.grad).all()

    def test_fp8_cublaslt_backend_runs_forward_and_backward(self):
        """The cuBLASLt FP8 backend should produce finite activations and gradients."""
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(tri_einsum="bf16", tri_impl="fp8_cublaslt")
        mod = TriangularUpdateTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(B, 32, 32, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(B, 32, 32, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        loss = out.float().square().mean()
        loss.backward()
        assert out.shape == (B, 32, 32, 128)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        assert torch.isfinite(x.grad).all()

    def test_fused_fp8_backend_runs_forward_and_backward(self):
        """The fused FP8 recipe backend should produce finite activations and gradients."""
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(tri_einsum="bf16", tri_impl="fused_fp8")
        mod = TriangularUpdateTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(B, 32, 32, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(B, 32, 32, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        loss = out.float().square().mean()
        loss.backward()
        assert out.shape == (B, 32, 32, 128)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        assert torch.isfinite(x.grad).all()

    def test_fp8_bdnn_matches_bf16_path_within_fp8_noise(self):
        """The one-permute FP8 backend should stay close to the BF16 bmm reference."""
        if not torch.cuda.is_available():
            return
        x_bdnn = torch.randn(B, 128, 32, 32, device=DEVICE, dtype=torch.bfloat16)
        out_fp8 = tri_mul_fp8_bdnn(x_bdnn, out_dtype=torch.float32)
        a1, b1, a2, b2 = torch.chunk(x_bdnn, 4, dim=1)
        out_ref = torch.cat(
            [
                tri_mul_bmm(a1.permute(0, 2, 3, 1), b1.permute(0, 2, 3, 1), k_dim=2, mode="bf16").float(),
                tri_mul_bmm(a2.permute(0, 2, 3, 1), b2.permute(0, 2, 3, 1), k_dim=1, mode="bf16").float(),
            ],
            dim=-1,
        )
        delta = (out_fp8 - out_ref).abs()
        assert delta.mean().item() < 0.25
        assert delta.flatten().quantile(0.99).item() < 0.8

    def test_tri_mul_einsum_matches_bmm_reference(self):
        """The literal einsum backend should match the reshape-to-bmm backend."""
        a = torch.randn(B, N, N, 16, device=DEVICE, dtype=torch.float32)
        b = torch.randn(B, N, N, 16, device=DEVICE, dtype=torch.float32)
        out_einsum_2 = tri_mul_einsum(a, b, k_dim=2)
        out_bmm_2 = tri_mul_bmm(a, b, k_dim=2)
        out_einsum_1 = tri_mul_einsum(a, b, k_dim=1)
        out_bmm_1 = tri_mul_bmm(a, b, k_dim=1)
        assert torch.allclose(out_einsum_2, out_bmm_2)
        assert torch.allclose(out_einsum_1, out_bmm_1)


class TestFP8StateInBlocks:
    """Verify te.autocast correctly enables/disables FP8 per block."""

    def test_fp8_enabled_in_fp8_block(self):
        """FP8GlobalStateManager.is_fp8_enabled() should be True inside an FP8 block's forward."""
        fp8_states = []

        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=["fp8", None]).to(DEVICE)

        for i, block in enumerate(mod.blocks):

            def make_hook(block_idx):
                def hook(module, input, output):
                    fp8_states.append((block_idx, FP8GlobalStateManager.is_fp8_enabled()))

                return hook

            block.register_forward_hook(make_hook(i))

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)
        mod(x, mask)

        assert fp8_states[0] == (0, True), f"Block 0 should have FP8 enabled but got {fp8_states[0]}"
        assert fp8_states[1] == (1, False), f"Block 1 should have FP8 disabled but got {fp8_states[1]}"

    def test_fp8_disabled_in_bf16_block(self):
        """All blocks BF16 — FP8 should never be enabled."""
        fp8_states = []

        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=[None, None]).to(DEVICE)

        for i, block in enumerate(mod.blocks):

            def make_hook(block_idx):
                def hook(module, input, output):
                    fp8_states.append((block_idx, FP8GlobalStateManager.is_fp8_enabled()))

                return hook

            block.register_forward_hook(make_hook(i))

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)
        mod(x, mask)

        for block_idx, fp8_enabled in fp8_states:
            assert not fp8_enabled, f"Block {block_idx} should have FP8 disabled but it's enabled"

    def test_all_blocks_fp8(self):
        """All blocks FP8 — FP8 should be enabled in every block."""
        fp8_states = []

        mod = MiniFormerTE(dim=DIM, blocks=3, block_precision=["fp8", "fp8", "fp8"]).to(DEVICE)

        for i, block in enumerate(mod.blocks):

            def make_hook(block_idx):
                def hook(module, input, output):
                    fp8_states.append((block_idx, FP8GlobalStateManager.is_fp8_enabled()))

                return hook

            block.register_forward_hook(make_hook(i))

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)
        mod(x, mask)

        for block_idx, fp8_enabled in fp8_states:
            assert fp8_enabled, f"Block {block_idx} should have FP8 enabled but it's disabled"

    def test_mixed_precision_pattern(self):
        """Alternating FP8/BF16 blocks — verify correct per-block FP8 state."""
        fp8_states = []

        mod = MiniFormerTE(dim=DIM, blocks=4, block_precision=["fp8", None, "fp8", None]).to(DEVICE)

        for i, block in enumerate(mod.blocks):

            def make_hook(block_idx):
                def hook(module, input, output):
                    fp8_states.append((block_idx, FP8GlobalStateManager.is_fp8_enabled()))

                return hook

            block.register_forward_hook(make_hook(i))

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)
        mod(x, mask)

        expected = [(0, True), (1, False), (2, True), (3, False)]
        for (idx, actual), (_, expect) in zip(fp8_states, expected):
            assert actual == expect, f"Block {idx}: expected FP8={expect}, got FP8={actual}"


class TestTEFloat8Boundaries:
    """Verify where TE FP8 outputs can persist and where the graph must break back to BF16."""

    def test_te_linear_nd_can_return_float8_tensor(self):
        """The N-D TE wrapper should preserve Float8Tensor outputs when fp8_output is requested."""
        if not torch.cuda.is_available():
            return
        mod = te.Linear(128, 128, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(B, 32, 32, 128, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            out = te_linear_nd(mod, x, fp8_output=True)

        assert isinstance(out, QuantizedTensor)
        assert out.shape == x.shape

    def test_te_float8_output_breaks_to_bf16_at_sigmoid_and_mul(self):
        """Sigmoid and elementwise multiply are mandatory higher-precision boundaries."""
        if not torch.cuda.is_available():
            return
        mod = te.Linear(128, 128, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(B, 32, 32, 128, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            q = te_linear_nd(mod, x, fp8_output=True)

        assert isinstance(q, QuantizedTensor)

        sigma = q.sigmoid()
        prod = q * sigma

        assert isinstance(sigma, torch.Tensor)
        assert not isinstance(sigma, QuantizedTensor)
        assert sigma.dtype == torch.bfloat16
        assert isinstance(prod, torch.Tensor)
        assert not isinstance(prod, QuantizedTensor)
        assert prod.dtype == torch.bfloat16


class TestComponentPrecisionOverrides:
    """Verify component-level precision overrides actually change FP8 state for sub-operations."""

    def test_ffn_excluded_from_fp8(self):
        """FFN runs with FP8 disabled even when block is in FP8."""
        cp = ComponentPrecisionConfig(ffn=False)
        mod = TransitionUpdateTE(dim=DIM, hidden=DIM * 4, component_precision=cp).to(DEVICE)

        fp8_in_ffn = []

        def hook(module, input, output):
            fp8_in_ffn.append(FP8GlobalStateManager.is_fp8_enabled())

        mod.fc1.register_forward_hook(hook)

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            mod(x)

        assert len(fp8_in_ffn) == 1
        assert not fp8_in_ffn[0], "FFN fc1 should have FP8 disabled when component_precision.ffn=False"

    def test_ffn_included_in_fp8(self):
        """FFN runs with FP8 enabled when component_precision.ffn=True."""
        cp = ComponentPrecisionConfig(ffn=True)
        mod = TransitionUpdateTE(dim=DIM, hidden=DIM * 4, component_precision=cp).to(DEVICE)

        fp8_in_ffn = []

        def hook(module, input, output):
            fp8_in_ffn.append(FP8GlobalStateManager.is_fp8_enabled())

        mod.fc1.register_forward_hook(hook)

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            mod(x)

        assert len(fp8_in_ffn) == 1
        assert fp8_in_ffn[0], "FFN fc1 should have FP8 enabled when component_precision.ffn=True"

    def test_tri_proj_excluded_from_fp8(self):
        """Triangular projections run with FP8 disabled when tri_proj=False."""
        cp = ComponentPrecisionConfig(tri_proj=False, tri_gate=True)
        mod = TriangularUpdateTE(dim=DIM, component_precision=cp).to(DEVICE)

        fp8_states = {"proj": [], "gate": []}

        def proj_hook(module, input, output):
            fp8_states["proj"].append(FP8GlobalStateManager.is_fp8_enabled())

        def gate_hook(module, input, output):
            fp8_states["gate"].append(FP8GlobalStateManager.is_fp8_enabled())

        mod.pi.register_forward_hook(proj_hook)
        mod.gi.register_forward_hook(gate_hook)

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            mod(x, mask)

        assert not any(fp8_states["proj"]), "tri_proj should have FP8 disabled"
        assert all(fp8_states["gate"]), "tri_gate should have FP8 enabled"

    def test_tri_gate_excluded_from_fp8(self):
        """Triangular gates run with FP8 disabled when tri_gate=False."""
        cp = ComponentPrecisionConfig(tri_proj=True, tri_gate=False)
        mod = TriangularUpdateTE(dim=DIM, component_precision=cp).to(DEVICE)

        fp8_states = {"proj": [], "gate": []}

        def proj_hook(module, input, output):
            fp8_states["proj"].append(FP8GlobalStateManager.is_fp8_enabled())

        def gate_hook(module, input, output):
            fp8_states["gate"].append(FP8GlobalStateManager.is_fp8_enabled())

        mod.pi.register_forward_hook(proj_hook)
        mod.gi.register_forward_hook(gate_hook)

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            mod(x, mask)

        assert all(fp8_states["proj"]), "tri_proj should have FP8 enabled"
        assert not any(fp8_states["gate"]), "tri_gate should have FP8 disabled"

    def test_dist_head_excluded_from_fp8(self):
        """Distogram head runs with FP8 disabled when dist_head=False."""
        cp = ComponentPrecisionConfig(dist_head=False)
        c_s, c_z = 512, DIM
        mod = FoldingTrunkTE(
            c_s=c_s,
            c_z=c_z,
            bins=32,
            disto_bins=64,
            num_layers=2,
            block_precision=["fp8", "fp8"],
            component_precision=cp,
        ).to(DEVICE)

        fp8_in_dist = []

        def hook(module, input, output):
            fp8_in_dist.append(FP8GlobalStateManager.is_fp8_enabled())

        mod.fc_out_1.register_forward_hook(hook)

        s_s = torch.randn(B, N, c_s, device=DEVICE, dtype=torch.bfloat16)
        s_z = torch.randn(B, N, N, c_z, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, N, device=DEVICE, dtype=torch.bfloat16)

        mod.eval()
        with torch.no_grad():
            mod(s_s, s_z, mask, num_recycling=0)

        assert len(fp8_in_dist) >= 1
        assert not any(fp8_in_dist), "dist_head fc_out_1 should have FP8 disabled"

    def test_no_component_precision_all_fp8(self):
        """Without component_precision, all te.Linear layers in FP8 block run in FP8."""
        mod = TransitionUpdateTE(dim=DIM, hidden=DIM * 4).to(DEVICE)

        fp8_in_ffn = []

        def hook(module, input, output):
            fp8_in_ffn.append(FP8GlobalStateManager.is_fp8_enabled())

        mod.fc1.register_forward_hook(hook)

        x = torch.randn(B, N, N, DIM, device=DEVICE, dtype=torch.bfloat16)

        with te.autocast(enabled=True):
            mod(x)

        assert len(fp8_in_ffn) == 1
        assert fp8_in_ffn[0], "Without component_precision, FFN should run in FP8"


class TestTriMulBmmEquivalence:
    """Verify tri_mul_bmm produces identical results to the original einsum."""

    def test_outgoing_einsum_equivalence(self):
        """tri_mul_bmm(a, b, k_dim=2) == torch.einsum('bikd,bjkd->bijd', a, b)."""
        torch.manual_seed(42)
        a = torch.randn(B, N, N, DIM // 4, device=DEVICE, dtype=torch.float32)
        b = torch.randn(B, N, N, DIM // 4, device=DEVICE, dtype=torch.float32)

        expected = torch.einsum("bikd,bjkd->bijd", a, b)
        actual = tri_mul_bmm(a, b, k_dim=2)

        assert torch.allclose(actual, expected, atol=1e-5), (
            f"k_dim=2 mismatch: max diff {(actual - expected).abs().max()}"
        )

    def test_incoming_einsum_equivalence(self):
        """tri_mul_bmm(a, b, k_dim=1) == torch.einsum('bkid,bkjd->bijd', a, b)."""
        torch.manual_seed(42)
        a = torch.randn(B, N, N, DIM // 4, device=DEVICE, dtype=torch.float32)
        b = torch.randn(B, N, N, DIM // 4, device=DEVICE, dtype=torch.float32)

        expected = torch.einsum("bkid,bkjd->bijd", a, b)
        actual = tri_mul_bmm(a, b, k_dim=1)

        # Small differences expected due to different CUDA kernel reduction orders
        assert torch.allclose(actual, expected, atol=1e-2), (
            f"k_dim=1 mismatch: max diff {(actual - expected).abs().max()}"
        )

    def test_bf16_close_to_fp32(self):
        """BF16 mode produces results close to FP32 reference."""
        if not torch.cuda.is_available():
            return
        torch.manual_seed(42)
        a = torch.randn(B, N, N, DIM // 4, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(B, N, N, DIM // 4, device="cuda", dtype=torch.bfloat16)

        ref = tri_mul_bmm(a.float(), b.float(), k_dim=2).to(torch.bfloat16)
        bf16_result = tri_mul_bmm(a, b, k_dim=2, mode="bf16")

        assert torch.allclose(bf16_result, ref, atol=0.5, rtol=0.05), (
            f"BF16 vs FP32 mismatch: max diff {(bf16_result - ref).abs().max()}"
        )
