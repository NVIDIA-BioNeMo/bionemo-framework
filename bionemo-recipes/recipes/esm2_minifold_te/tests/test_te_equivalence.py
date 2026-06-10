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

"""Numerical equivalence tests between original MiniFold modules and TE versions.

Each test:
1. Creates original and TE modules with matching dimensions
2. Copies weights from original -> TE
3. Runs both on the same random input (fixed seed)
4. Asserts outputs match within tolerance

Run with: pytest tests/test_te_equivalence.py -v
"""

import sys
from pathlib import Path

import pytest
import torch


# Add recipe root to path for TE modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add minifold root to path for original modules
sys.path.insert(0, "/workspaces/minifold")

# Original modules
from minifold.model.heads import PerResidueLDDTCaPredictor
from minifold.model.miniformer import Block, MiniFormer, TransitionUpdate, TriangularUpdate
from minifold.model.model import FoldingTrunk, PairToSequence, SequenceToPair
from minifold.model.structure import MLP, AngleResnet, AngleResnetBlock, Attention

# TE modules (recipe-local)
from heads_te import PerResidueLDDTCaPredictorTE
from miniformer_te import BlockTE, MiniFormerTE, TransitionUpdateTE, TriangularUpdateTE
from model_te import FoldingTrunkTE, PairToSequenceTE, SequenceToPairTE
from structure_te import MLPTE, AngleResnetBlockTE, AngleResnetTE, AttentionTE

# Weight copy (recipe-local)
from weight_copy import (
    copy_angle_resnet_block_to_te,
    copy_angle_resnet_to_te,
    copy_attention_to_te,
    copy_block_to_te,
    copy_folding_trunk_to_te,
    copy_miniformer_to_te,
    copy_mlp_to_te,
    copy_pair_to_seq_to_te,
    copy_plddt_to_te,
    copy_seq_to_pair_to_te,
    copy_transition_update_from_te,
    copy_transition_update_to_te,
    copy_triangular_update_to_te,
)


DEVICE = "cuda"
ATOL = 1e-5
RTOL = 1e-5
SEED = 42
INPUT_SEED = 123
DIM = 128
HIDDEN = 512
B = 2
N = 16


@pytest.fixture(autouse=True)
def set_seed():
    """Set seed before each test for reproducibility."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ===========================================================================
# TransitionUpdate
# ===========================================================================


class TestTransitionUpdate:
    def test_equivalence(self):
        orig = TransitionUpdate(dim=DIM, hidden=HIDDEN, kernels=False).to(DEVICE)
        te_mod = TransitionUpdateTE(dim=DIM, hidden=HIDDEN).to(DEVICE)
        copy_transition_update_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(B, N, N, DIM, device=DEVICE)

        with torch.no_grad():
            out_orig = orig(x)
            out_te = te_mod(x)

        assert out_orig.shape == out_te.shape
        assert torch.allclose(out_orig, out_te, atol=ATOL, rtol=RTOL), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )

    def test_weight_roundtrip(self):
        orig = TransitionUpdate(dim=DIM, hidden=HIDDEN, kernels=False).to(DEVICE)
        te_mod = TransitionUpdateTE(dim=DIM, hidden=HIDDEN).to(DEVICE)
        orig_copy = TransitionUpdate(dim=DIM, hidden=HIDDEN, kernels=False).to(DEVICE)

        copy_transition_update_to_te(orig, te_mod)
        copy_transition_update_from_te(te_mod, orig_copy)

        for (n1, p1), (n2, p2) in zip(orig.named_parameters(), orig_copy.named_parameters()):
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_gradient_flow(self):
        te_mod = TransitionUpdateTE(dim=DIM, hidden=HIDDEN).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE, requires_grad=True)
        out = te_mod(x)
        out.sum().backward()
        for name, param in te_mod.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ===========================================================================
# TriangularUpdate
# ===========================================================================


class TestTriangularUpdate:
    def test_equivalence(self):
        orig = TriangularUpdate(dim=DIM, kernels=False).to(DEVICE)
        te_mod = TriangularUpdateTE(dim=DIM).to(DEVICE)
        copy_triangular_update_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)

        with torch.no_grad():
            out_orig = orig(x, mask)
            out_te = te_mod(x, mask)

        assert out_orig.shape == out_te.shape
        assert torch.allclose(out_orig, out_te, atol=ATOL, rtol=RTOL), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )

    def test_with_mask(self):
        orig = TriangularUpdate(dim=DIM, kernels=False).to(DEVICE)
        te_mod = TriangularUpdateTE(dim=DIM).to(DEVICE)
        copy_triangular_update_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)
        mask[:, :, N // 2 :] = 0  # mask out half

        with torch.no_grad():
            out_orig = orig(x, mask)
            out_te = te_mod(x, mask)

        assert torch.allclose(out_orig, out_te, atol=ATOL, rtol=RTOL), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )

    def test_gradient_flow(self):
        te_mod = TriangularUpdateTE(dim=DIM).to(DEVICE)
        x = torch.randn(B, N, N, DIM, device=DEVICE, requires_grad=True)
        mask = torch.ones(B, N, N, device=DEVICE)
        out = te_mod(x, mask)
        out.sum().backward()
        for name, param in te_mod.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ===========================================================================
# Block
# ===========================================================================


class TestBlock:
    def test_equivalence(self):
        orig = Block(dim=DIM, kernels=False).to(DEVICE)
        te_mod = BlockTE(dim=DIM).to(DEVICE)
        copy_block_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)

        with torch.no_grad():
            out_orig = orig(x, mask)
            out_te = te_mod(x, mask)

        assert torch.allclose(out_orig, out_te, atol=ATOL, rtol=RTOL), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )


# ===========================================================================
# MiniFormer
# ===========================================================================


class TestMiniFormer:
    def test_equivalence(self):
        num_blocks = 2
        orig = MiniFormer(dim=DIM, blocks=num_blocks, kernels=False).to(DEVICE)
        te_mod = MiniFormerTE(dim=DIM, blocks=num_blocks).to(DEVICE)
        copy_miniformer_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(B, N, N, DIM, device=DEVICE)
        mask = torch.ones(B, N, N, device=DEVICE)

        with torch.no_grad():
            out_orig = orig(x, mask)
            out_te = te_mod(x, mask)

        assert torch.allclose(out_orig, out_te, atol=1e-4, rtol=1e-4), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )


# ===========================================================================
# SequenceToPair
# ===========================================================================


class TestSequenceToPair:
    def test_equivalence(self):
        seq_dim = 1024
        inner = 64
        pair_dim = DIM
        orig = SequenceToPair(seq_dim, inner, pair_dim).to(DEVICE)
        te_mod = SequenceToPairTE(seq_dim, inner, pair_dim).to(DEVICE)
        copy_seq_to_pair_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(B, N, seq_dim, device=DEVICE)

        with torch.no_grad():
            out_orig = orig(x)
            out_te = te_mod(x)

        assert out_orig.shape == out_te.shape == (B, N, N, pair_dim)
        # Slightly relaxed tolerance: outer product + diff accumulate FP32 rounding
        assert torch.allclose(out_orig, out_te, atol=5e-4, rtol=1e-4), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )


# ===========================================================================
# PairToSequence
# ===========================================================================


class TestPairToSequence:
    def test_equivalence(self):
        c_z, c_s = DIM, 1024
        orig = PairToSequence(c_z=c_z, c_s=c_s).to(DEVICE)
        te_mod = PairToSequenceTE(c_z=c_z, c_s=c_s).to(DEVICE)
        copy_pair_to_seq_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        s_z = torch.randn(B, N, N, c_z, device=DEVICE)
        s_s = torch.randn(B, N, c_s, device=DEVICE)
        pair_mask = torch.ones(B, N, N, device=DEVICE)

        with torch.no_grad():
            out_orig = orig(s_z, s_s, pair_mask)
            out_te = te_mod(s_z, s_s, pair_mask)

        assert out_orig.shape == out_te.shape == (B, N, c_s)
        assert torch.allclose(out_orig, out_te, atol=ATOL, rtol=RTOL), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )


# ===========================================================================
# FoldingTrunk
# ===========================================================================


class TestFoldingTrunk:
    def test_equivalence(self):
        c_s, c_z = 1024, DIM
        num_blocks = 2
        orig = FoldingTrunk(
            c_s=c_s,
            c_z=c_z,
            bins=32,
            disto_bins=64,
            num_layers=num_blocks,
            kernels=False,
        ).to(DEVICE)
        te_mod = FoldingTrunkTE(
            c_s=c_s,
            c_z=c_z,
            bins=32,
            disto_bins=64,
            num_layers=num_blocks,
        ).to(DEVICE)
        copy_folding_trunk_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        s_s = torch.randn(B, N, c_s, device=DEVICE)
        s_z = torch.randn(B, N, N, c_z, device=DEVICE)
        mask = torch.ones(B, N, device=DEVICE)

        orig.eval()
        te_mod.eval()

        with torch.no_grad():
            preds_orig, sz_orig = orig(s_s, s_z, mask, num_recycling=0)
            preds_te, sz_te = te_mod(s_s, s_z, mask, num_recycling=0)

        assert preds_orig.shape == preds_te.shape
        assert sz_orig.shape == sz_te.shape
        assert torch.allclose(preds_orig, preds_te, atol=1e-4, rtol=1e-4), (
            f"Preds max diff: {(preds_orig - preds_te).abs().max().item()}"
        )
        assert torch.allclose(sz_orig, sz_te, atol=1e-4, rtol=1e-4), (
            f"s_z max diff: {(sz_orig - sz_te).abs().max().item()}"
        )


# ===========================================================================
# Attention (StructureModule)
# ===========================================================================


class TestAttention:
    def test_equivalence(self):
        dim, num_heads, head_width = 1024, 16, 64
        orig = Attention(dim, num_heads, head_width).to(DEVICE)
        te_mod = AttentionTE(dim, num_heads, head_width).to(DEVICE)
        copy_attention_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(B, N, dim, device=DEVICE)
        bias = torch.randn(B, num_heads, N, N, device=DEVICE)
        mask = torch.ones(B, N, device=DEVICE)

        with torch.no_grad():
            out_orig = orig(x, bias, mask)
            out_te = te_mod(x, bias, mask)

        assert out_orig.shape == out_te.shape
        # Slightly relaxed: te.Linear kernel ordering differs from nn.Linear
        assert torch.allclose(out_orig, out_te, atol=5e-5, rtol=5e-5), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )


# ===========================================================================
# MLP (StructureModule)
# ===========================================================================


class TestMLP:
    def test_equivalence(self):
        in_dim, out_dim = 1024, 1024
        orig = MLP(in_dim, out_dim).to(DEVICE)
        te_mod = MLPTE(in_dim, out_dim).to(DEVICE)
        copy_mlp_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(B, N, in_dim, device=DEVICE)

        with torch.no_grad():
            out_orig = orig(x)
            out_te = te_mod(x)

        assert out_orig.shape == out_te.shape
        assert torch.allclose(out_orig, out_te, atol=ATOL, rtol=RTOL), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )


# ===========================================================================
# AngleResnetBlock
# ===========================================================================


class TestAngleResnetBlock:
    def test_equivalence(self):
        dim = DIM
        orig = AngleResnetBlock(dim).to(DEVICE)
        te_mod = AngleResnetBlockTE(dim).to(DEVICE)
        copy_angle_resnet_block_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(B, N, dim, device=DEVICE)

        with torch.no_grad():
            out_orig = orig(x)
            out_te = te_mod(x)

        assert torch.allclose(out_orig, out_te, atol=ATOL, rtol=RTOL), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )


# ===========================================================================
# AngleResnet
# ===========================================================================


class TestAngleResnet:
    def test_equivalence(self):
        c_in, c_hidden = 1024, DIM
        orig = AngleResnet(c_in, c_hidden, no_blocks=2, no_angles=7, epsilon=1e-5).to(DEVICE)
        te_mod = AngleResnetTE(c_in, c_hidden, no_blocks=2, no_angles=7, epsilon=1e-5).to(DEVICE)
        copy_angle_resnet_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        s = torch.randn(B, N, c_in, device=DEVICE)
        s_init = torch.randn(B, N, c_in, device=DEVICE)

        with torch.no_grad():
            unnorm_orig, norm_orig = orig(s, s_init)
            unnorm_te, norm_te = te_mod(s, s_init)

        assert torch.allclose(unnorm_orig, unnorm_te, atol=ATOL, rtol=RTOL), (
            f"Unnorm max diff: {(unnorm_orig - unnorm_te).abs().max().item()}"
        )
        assert torch.allclose(norm_orig, norm_te, atol=ATOL, rtol=RTOL), (
            f"Norm max diff: {(norm_orig - norm_te).abs().max().item()}"
        )


# ===========================================================================
# PerResidueLDDTCaPredictor
# ===========================================================================


class TestPerResidueLDDTCaPredictor:
    def test_equivalence(self):
        no_bins, c_in, c_hidden = 50, 1024, DIM
        orig = PerResidueLDDTCaPredictor(no_bins, c_in, c_hidden).to(DEVICE)
        te_mod = PerResidueLDDTCaPredictorTE(no_bins, c_in, c_hidden).to(DEVICE)
        copy_plddt_to_te(orig, te_mod)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(B, N, c_in, device=DEVICE)

        with torch.no_grad():
            out_orig = orig(x)
            out_te = te_mod(x)

        assert out_orig.shape == out_te.shape == (B, N, no_bins)
        assert torch.allclose(out_orig, out_te, atol=ATOL, rtol=RTOL), (
            f"Max diff: {(out_orig - out_te).abs().max().item()}"
        )
