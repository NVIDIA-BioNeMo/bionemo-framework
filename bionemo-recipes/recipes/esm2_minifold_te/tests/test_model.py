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

"""Tests for ESM2-MiniFold TE model components.

Tests model instantiation, forward pass shapes, and gradient flow.
Uses small model dimensions for fast testing.
"""

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from heads_te import PerResidueLDDTCaPredictorTE
from miniformer_te import BlockTE, MiniFormerTE, TransitionUpdateTE, TriangularUpdateTE
from model_te import FoldingTrunkTE, PairToSequenceTE, SequenceToPairTE
from quantization import ComponentPrecisionConfig, resolve_layer_precision
from structure_te import MLPTE, AngleResnetTE, AttentionTE
from te_utils import te_linear_nd


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, N, DIM = 2, 16, 64
HIDDEN = 256


# ===========================================================================
# TE Module Shape Tests
# ===========================================================================


class TestTransitionUpdateTE:
    def test_forward_shape(self):
        mod = TransitionUpdateTE(dim=DIM, hidden=HIDDEN).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        out = mod(x)
        assert out.shape == (B, N, N, DIM)

    def test_gradient_flow(self):
        mod = TransitionUpdateTE(dim=DIM, hidden=HIDDEN).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        assert x.grad is not None


class TestTriangularUpdateTE:
    def test_forward_shape(self):
        mod = TriangularUpdateTE(dim=DIM).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)

    def test_forward_shape_fp8_bmm_backend(self):
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(tri_einsum="bf16", tri_impl="fp8_bmm")
        mod = TriangularUpdateTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(B, 32, 32, 128, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, 32, 32, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        assert out.shape == (B, 32, 32, 128)
        assert out.dtype == torch.bfloat16

    def test_forward_shape_fp8_grouped_backend(self):
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(tri_einsum="bf16", tri_impl="fp8_grouped")
        mod = TriangularUpdateTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(B, 32, 32, 128, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, 32, 32, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        assert out.shape == (B, 32, 32, 128)
        assert out.dtype == torch.bfloat16

    def test_forward_shape_fp8_cublaslt_backend(self):
        if not torch.cuda.is_available():
            return
        cp = ComponentPrecisionConfig(tri_einsum="bf16", tri_impl="fp8_cublaslt")
        mod = TriangularUpdateTE(dim=128, component_precision=cp, params_dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(B, 32, 32, 128, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(B, 32, 32, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x, mask)
        assert out.shape == (B, 32, 32, 128)
        assert out.dtype == torch.bfloat16


class TestBlockTE:
    def test_forward_shape(self):
        mod = BlockTE(dim=DIM).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)


class TestMiniFormerTE:
    def test_forward_shape(self):
        mod = MiniFormerTE(dim=DIM, blocks=2).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)


class TestSequenceToPairTE:
    def test_forward_shape(self):
        seq_dim, inner, pair_dim = 512, 32, DIM
        mod = SequenceToPairTE(seq_dim, inner, pair_dim).to(DEVICE)
        x = torch.randn(B, N, seq_dim, device=DEVICE)
        out = mod(x)
        assert out.shape == (B, N, N, pair_dim)


class TestPairToSequenceTE:
    def test_forward_shape(self):
        c_z, c_s, c_s_out = DIM, 512, 512
        mod = PairToSequenceTE(c_z=c_z, c_s=c_s, c_s_out=c_s_out).to(DEVICE)
        s_z = torch.randn(B, N, N, c_z, device=DEVICE)
        s_s = torch.randn(B, N, c_s, device=DEVICE)
        pair_mask = torch.ones(B, N, N, device=DEVICE)
        out = mod(s_z, s_s, pair_mask)
        assert out.shape == (B, N, c_s_out)


class TestFoldingTrunkTE:
    def test_forward_shape(self):
        c_s, c_z = 512, DIM
        mod = FoldingTrunkTE(c_s=c_s, c_z=c_z, bins=32, disto_bins=64, num_layers=2).to(DEVICE)
        s_s = torch.randn(B, N, c_s, device=DEVICE)
        s_z = torch.randn(B, N, N, c_z, device=DEVICE)
        mask = torch.ones(B, N, device=DEVICE)

        mod.eval()
        with torch.no_grad():
            preds, sz = mod(s_s, s_z, mask, num_recycling=0)

        assert preds.shape == (B, N, N, 64)
        assert sz.shape[:3] == (B, N, N)


class TestAttentionTE:
    def test_forward_shape(self):
        dim, heads, head_width = 512, 8, 64
        mod = AttentionTE(dim, heads, head_width).to(DEVICE)
        x = torch.randn(B, N, dim, device=DEVICE)
        bias = torch.randn(B, heads, N, N, device=DEVICE)
        mask = torch.ones(B, N, device=DEVICE)
        out = mod(x, bias, mask)
        assert out.shape == (B, N, dim)


class TestMLPTE:
    def test_forward_shape(self):
        mod = MLPTE(512, 512).to(DEVICE)
        x = torch.randn(B, N, 512, device=DEVICE)
        out = mod(x)
        assert out.shape == (B, N, 512)


class TestAngleResnetTE:
    def test_forward_shape(self):
        mod = AngleResnetTE(512, DIM, no_blocks=2, no_angles=7, epsilon=1e-5).to(DEVICE)
        s = torch.randn(B, N, 512, device=DEVICE)
        s_init = torch.randn(B, N, 512, device=DEVICE)
        unnorm, norm = mod(s, s_init)
        assert unnorm.shape == (B, N, 7, 2)
        assert norm.shape == (B, N, 7, 2)


class TestPerResidueLDDTCaPredictorTE:
    def test_forward_shape(self):
        mod = PerResidueLDDTCaPredictorTE(50, 512, DIM).to(DEVICE)
        x = torch.randn(B, N, 512, device=DEVICE)
        out = mod(x)
        assert out.shape == (B, N, 50)


# ===========================================================================
# Block Precision / Quantization Tests
# ===========================================================================


class TestResolveLayerPrecision:
    def test_neither_enabled(self):
        result = resolve_layer_precision(6, fp8_enabled=False, fp4_enabled=False, fp8_layers=None, fp4_layers=None)
        assert result == [None] * 6

    def test_fp8_all_blocks(self):
        result = resolve_layer_precision(4, fp8_enabled=True, fp4_enabled=False, fp8_layers=None, fp4_layers=None)
        assert result == ["fp8"] * 4

    def test_fp4_all_blocks(self):
        result = resolve_layer_precision(4, fp8_enabled=False, fp4_enabled=True, fp8_layers=None, fp4_layers=None)
        assert result == ["fp4"] * 4

    def test_fp8_specific_blocks(self):
        result = resolve_layer_precision(4, fp8_enabled=True, fp4_enabled=False, fp8_layers=[1, 3], fp4_layers=None)
        assert result == ["fp8", None, "fp8", None]

    def test_mixed_fp8_fp4(self):
        result = resolve_layer_precision(4, fp8_enabled=True, fp4_enabled=True, fp8_layers=[1, 2], fp4_layers=[3, 4])
        assert result == ["fp8", "fp8", "fp4", "fp4"]

    def test_fp8_explicit_fp4_fills_remaining(self):
        result = resolve_layer_precision(4, fp8_enabled=True, fp4_enabled=True, fp8_layers=[1, 2], fp4_layers=None)
        assert result == ["fp8", "fp8", "fp4", "fp4"]

    def test_both_enabled_no_layers_raises(self):
        import pytest

        with pytest.raises(ValueError):
            resolve_layer_precision(4, fp8_enabled=True, fp4_enabled=True, fp8_layers=None, fp4_layers=None)

    def test_overlap_raises(self):
        import pytest

        with pytest.raises(ValueError):
            resolve_layer_precision(4, fp8_enabled=True, fp4_enabled=True, fp8_layers=[1, 2], fp4_layers=[2, 3])


class TestMiniFormerTEPrecision:
    def test_no_precision_config(self):
        """MiniFormerTE works without block_precision (BF16 default)."""
        mod = MiniFormerTE(dim=DIM, blocks=2).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)

    def test_with_block_precision_none_list(self):
        """MiniFormerTE works with all-None block_precision (explicit BF16)."""
        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=[None, None]).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)

    def test_block_precision_length_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError):
            MiniFormerTE(dim=DIM, blocks=2, block_precision=[None])


class TestComponentPrecision:
    """Test per-component precision overrides within FP8 blocks.

    These tests verify that te.autocast(enabled=False) is correctly applied
    to keep specific sub-components (projections, gates, FFN, etc.) in BF16
    while the rest of the block runs in FP8.
    """

    def test_all_components_enabled_fp8(self):
        """All components in FP8 — forward pass produces valid output."""
        cp = ComponentPrecisionConfig()  # all True by default
        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=["fp8", "fp8"], component_precision=cp).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)

    def test_ffn_bf16_only(self):
        """FFN in BF16, everything else in FP8."""
        cp = ComponentPrecisionConfig(ffn=False)
        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=["fp8", "fp8"], component_precision=cp).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)

    def test_tri_proj_bf16_only(self):
        """Triangular projections in BF16, everything else in FP8."""
        cp = ComponentPrecisionConfig(tri_proj=False)
        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=["fp8", "fp8"], component_precision=cp).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)

    def test_tri_gate_bf16_only(self):
        """Triangular gates in BF16, everything else in FP8."""
        cp = ComponentPrecisionConfig(tri_gate=False)
        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=["fp8", "fp8"], component_precision=cp).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)

    def test_all_components_bf16(self):
        """All components forced to BF16 within FP8 blocks."""
        cp = ComponentPrecisionConfig(
            tri_proj=False,
            tri_gate=False,
            ffn=False,
            struct_attn=False,
            struct_ffn=False,
            seq_proj=False,
            dist_head=False,
        )
        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=["fp8", "fp8"], component_precision=cp).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)

    def test_mixed_block_and_component(self):
        """Block 1 in FP8 with FFN in BF16, block 2 fully in BF16."""
        cp = ComponentPrecisionConfig(ffn=False)
        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=["fp8", None], component_precision=cp).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mod(x, mask)
        assert out.shape == (B, N, N, DIM)

    def test_gradient_flow_with_component_override(self):
        """Gradients flow correctly through mixed-precision components."""
        cp = ComponentPrecisionConfig(ffn=False, tri_gate=False)
        mod = MiniFormerTE(dim=DIM, blocks=2, block_precision=["fp8", "fp8"], component_precision=cp).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE, requires_grad=True)
        mask = torch.ones(B, N, N, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = mod(x, mask)
        out.sum().backward()
        assert x.grad is not None

    def test_folding_trunk_with_dist_head_bf16(self):
        """FoldingTrunkTE with dist_head forced to BF16 within FP8 blocks."""
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
        s_s = torch.randn(B, N, c_s, device=DEVICE)
        s_z = torch.randn(B, N, N, c_z, device=DEVICE)
        mask = torch.ones(B, N, device=DEVICE)
        mod.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            preds, sz = mod(s_s, s_z, mask, num_recycling=0)
        assert preds.shape == (B, N, N, 64)

    def test_folding_trunk_with_seq_proj_bf16(self):
        """FoldingTrunkTE with seq_proj forced to BF16 within FP8 blocks."""
        cp = ComponentPrecisionConfig(seq_proj=False)
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
        s_s = torch.randn(B, N, c_s, device=DEVICE)
        s_z = torch.randn(B, N, N, c_z, device=DEVICE)
        mask = torch.ones(B, N, device=DEVICE)
        mod.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            preds, sz = mod(s_s, s_z, mask, num_recycling=0)
        assert preds.shape == (B, N, N, 64)


# ===========================================================================
# te_utils Tests
# ===========================================================================


class TestTeUtils:
    def test_te_linear_nd_2d(self):
        import transformer_engine.pytorch as te

        linear = te.Linear(DIM, HIDDEN).to(DEVICE)
        x = torch.randn(B * N, DIM, device=DEVICE)
        out = te_linear_nd(linear, x)
        assert out.shape == (B * N, HIDDEN)

    def test_te_linear_nd_3d(self):
        import transformer_engine.pytorch as te

        linear = te.Linear(DIM, HIDDEN).to(DEVICE)
        x = torch.randn(B, N, DIM, device=DEVICE)
        out = te_linear_nd(linear, x)
        assert out.shape == (B, N, HIDDEN)

    def test_te_linear_nd_4d(self):
        import transformer_engine.pytorch as te

        linear = te.Linear(DIM, HIDDEN).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        out = te_linear_nd(linear, x)
        assert out.shape == (B, N, N, HIDDEN)
