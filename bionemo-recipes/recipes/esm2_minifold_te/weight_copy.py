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

"""Weight copy utilities between original MiniFold modules and their TE counterparts.

Both nn.Linear and te.Linear store weights as (out_features, in_features),
so direct copy works without transposition. The raw nn.Parameter tensors used
in TransitionUpdate/TriangularUpdate also use (out, in) layout via F.linear.
"""

import torch


def _copy_param(src, dst):
    """Copy parameter data from src to dst, with shape validation."""
    assert src.shape == dst.shape, f"Shape mismatch: src {src.shape} vs dst {dst.shape}"
    with torch.no_grad():
        dst.copy_(src)


# ---------------------------------------------------------------------------
# TransitionUpdate <-> TransitionUpdateTE
# ---------------------------------------------------------------------------


def copy_transition_update_to_te(orig, te_mod):
    """Copy weights from TransitionUpdate to TransitionUpdateTE."""
    _copy_param(orig.wn, te_mod.norm.weight)
    _copy_param(orig.bn, te_mod.norm.bias)
    _copy_param(orig.w1, te_mod.fc1.weight)
    _copy_param(orig.b1, te_mod.fc1.bias)
    _copy_param(orig.w2, te_mod.fc2.weight)
    _copy_param(orig.b2, te_mod.fc2.bias)


def copy_transition_update_from_te(te_mod, orig):
    """Copy weights from TransitionUpdateTE to TransitionUpdate."""
    _copy_param(te_mod.norm.weight, orig.wn)
    _copy_param(te_mod.norm.bias, orig.bn)
    _copy_param(te_mod.fc1.weight, orig.w1)
    _copy_param(te_mod.fc1.bias, orig.b1)
    _copy_param(te_mod.fc2.weight, orig.w2)
    _copy_param(te_mod.fc2.bias, orig.b2)


# ---------------------------------------------------------------------------
# TriangularUpdate <-> TriangularUpdateTE
# ---------------------------------------------------------------------------


def copy_triangular_update_to_te(orig, te_mod):
    """Copy weights from TriangularUpdate to TriangularUpdateTE."""
    _copy_param(orig.ni_w, te_mod.input_norm.weight)
    _copy_param(orig.ni_b, te_mod.input_norm.bias)
    _copy_param(orig.pi_w, te_mod.pi.weight)
    _copy_param(orig.pi_b, te_mod.pi.bias)
    _copy_param(orig.gi_w, te_mod.gi.weight)
    _copy_param(orig.gi_b, te_mod.gi.bias)

    _copy_param(orig.no_w, te_mod.output_norm.weight)
    _copy_param(orig.no_b, te_mod.output_norm.bias)
    _copy_param(orig.po_w, te_mod.po.weight)
    _copy_param(orig.po_b, te_mod.po.bias)
    _copy_param(orig.go_w, te_mod.go.weight)
    _copy_param(orig.go_b, te_mod.go.bias)


def copy_triangular_update_from_te(te_mod, orig):
    """Copy weights from TriangularUpdateTE to TriangularUpdate."""
    _copy_param(te_mod.input_norm.weight, orig.ni_w)
    _copy_param(te_mod.input_norm.bias, orig.ni_b)
    _copy_param(te_mod.pi.weight, orig.pi_w)
    _copy_param(te_mod.pi.bias, orig.pi_b)
    _copy_param(te_mod.gi.weight, orig.gi_w)
    _copy_param(te_mod.gi.bias, orig.gi_b)

    _copy_param(te_mod.output_norm.weight, orig.no_w)
    _copy_param(te_mod.output_norm.bias, orig.no_b)
    _copy_param(te_mod.po.weight, orig.po_w)
    _copy_param(te_mod.po.bias, orig.po_b)
    _copy_param(te_mod.go.weight, orig.go_w)
    _copy_param(te_mod.go.bias, orig.go_b)


# ---------------------------------------------------------------------------
# Block <-> BlockTE
# ---------------------------------------------------------------------------


def copy_block_to_te(orig, te_mod):
    """Copy weights from Block to BlockTE."""
    copy_triangular_update_to_te(orig.triangular, te_mod.triangular)
    copy_transition_update_to_te(orig.transition, te_mod.transition)


def copy_block_from_te(te_mod, orig):
    """Copy weights from BlockTE to Block."""
    copy_triangular_update_from_te(te_mod.triangular, orig.triangular)
    copy_transition_update_from_te(te_mod.transition, orig.transition)


# ---------------------------------------------------------------------------
# MiniFormer <-> MiniFormerTE
# ---------------------------------------------------------------------------


def copy_miniformer_to_te(orig, te_mod):
    """Copy weights from MiniFormer to MiniFormerTE."""
    assert len(orig.blocks) == len(te_mod.blocks)
    for orig_block, te_block in zip(orig.blocks, te_mod.blocks):
        copy_block_to_te(orig_block, te_block)


def copy_miniformer_from_te(te_mod, orig):
    """Copy weights from MiniFormerTE to MiniFormer."""
    assert len(orig.blocks) == len(te_mod.blocks)
    for orig_block, te_block in zip(orig.blocks, te_mod.blocks):
        copy_block_from_te(te_block, orig_block)


# ---------------------------------------------------------------------------
# nn.Linear <-> te.Linear (generic helper)
# ---------------------------------------------------------------------------


def _copy_linear_to_te(orig_linear, te_linear):
    """Copy from nn.Linear to te.Linear."""
    _copy_param(orig_linear.weight, te_linear.weight)
    if orig_linear.bias is not None and te_linear.bias is not None:
        _copy_param(orig_linear.bias, te_linear.bias)


def _copy_linear_from_te(te_linear, orig_linear):
    """Copy from te.Linear to nn.Linear."""
    _copy_param(te_linear.weight, orig_linear.weight)
    if orig_linear.bias is not None and te_linear.bias is not None:
        _copy_param(te_linear.bias, orig_linear.bias)


def _copy_layernorm_to_te(orig_ln, te_ln):
    """Copy from nn.LayerNorm to te.LayerNorm."""
    _copy_param(orig_ln.weight, te_ln.weight)
    _copy_param(orig_ln.bias, te_ln.bias)


def _copy_layernorm_from_te(te_ln, orig_ln):
    """Copy from te.LayerNorm to nn.LayerNorm."""
    _copy_param(te_ln.weight, orig_ln.weight)
    _copy_param(te_ln.bias, orig_ln.bias)


# ---------------------------------------------------------------------------
# SequenceToPair <-> SequenceToPairTE
# ---------------------------------------------------------------------------


def copy_seq_to_pair_to_te(orig, te_mod):
    _copy_layernorm_to_te(orig.layernorm, te_mod.layernorm)
    _copy_linear_to_te(orig.proj, te_mod.proj)
    _copy_linear_to_te(orig.o_proj, te_mod.o_proj)


def copy_seq_to_pair_from_te(te_mod, orig):
    _copy_layernorm_from_te(te_mod.layernorm, orig.layernorm)
    _copy_linear_from_te(te_mod.proj, orig.proj)
    _copy_linear_from_te(te_mod.o_proj, orig.o_proj)


# ---------------------------------------------------------------------------
# PairToSequence <-> PairToSequenceTE
# ---------------------------------------------------------------------------


def copy_pair_to_seq_to_te(orig, te_mod):
    # s_z_mlp: Sequential(LayerNorm, Linear, ReLU, Linear) -> separate TE modules
    _copy_layernorm_to_te(orig.s_z_mlp[0], te_mod.s_z_norm)
    _copy_linear_to_te(orig.s_z_mlp[1], te_mod.s_z_fc1)
    _copy_linear_to_te(orig.s_z_mlp[3], te_mod.s_z_fc2)
    # combiner: Sequential(Linear) -> te.Linear
    _copy_linear_to_te(orig.combiner[0], te_mod.combiner)


def copy_pair_to_seq_from_te(te_mod, orig):
    _copy_layernorm_from_te(te_mod.s_z_norm, orig.s_z_mlp[0])
    _copy_linear_from_te(te_mod.s_z_fc1, orig.s_z_mlp[1])
    _copy_linear_from_te(te_mod.s_z_fc2, orig.s_z_mlp[3])
    _copy_linear_from_te(te_mod.combiner, orig.combiner[0])


# ---------------------------------------------------------------------------
# FoldingTrunk <-> FoldingTrunkTE
# ---------------------------------------------------------------------------


def copy_folding_trunk_to_te(orig, te_mod):
    # Positional embedding
    _copy_param(orig.positional_embedding.embedding.weight, te_mod.positional_embedding.embedding.weight)
    copy_seq_to_pair_to_te(orig.seq_to_pair, te_mod.seq_to_pair)
    _copy_linear_to_te(orig.projection, te_mod.projection)
    _copy_linear_to_te(orig.recycle, te_mod.recycle)
    copy_miniformer_to_te(orig.miniformer, te_mod.miniformer)
    # fc_out: Sequential(Linear, ReLU, Linear) -> fc_out_1, fc_out_2
    _copy_linear_to_te(orig.fc_out[0], te_mod.fc_out_1)
    _copy_linear_to_te(orig.fc_out[2], te_mod.fc_out_2)


def copy_folding_trunk_from_te(te_mod, orig):
    _copy_param(te_mod.positional_embedding.embedding.weight, orig.positional_embedding.embedding.weight)
    copy_seq_to_pair_from_te(te_mod.seq_to_pair, orig.seq_to_pair)
    _copy_linear_from_te(te_mod.projection, orig.projection)
    _copy_linear_from_te(te_mod.recycle, orig.recycle)
    copy_miniformer_from_te(te_mod.miniformer, orig.miniformer)
    _copy_linear_from_te(te_mod.fc_out_1, orig.fc_out[0])
    _copy_linear_from_te(te_mod.fc_out_2, orig.fc_out[2])


# ---------------------------------------------------------------------------
# Attention <-> AttentionTE
# ---------------------------------------------------------------------------


def copy_attention_to_te(orig, te_mod):
    _copy_layernorm_to_te(orig.layer_norm, te_mod.layer_norm)
    _copy_linear_to_te(orig.proj, te_mod.proj)
    _copy_linear_to_te(orig.o_proj, te_mod.o_proj)
    _copy_linear_to_te(orig.g_proj, te_mod.g_proj)


def copy_attention_from_te(te_mod, orig):
    _copy_layernorm_from_te(te_mod.layer_norm, orig.layer_norm)
    _copy_linear_from_te(te_mod.proj, orig.proj)
    _copy_linear_from_te(te_mod.o_proj, orig.o_proj)
    _copy_linear_from_te(te_mod.g_proj, orig.g_proj)


# ---------------------------------------------------------------------------
# MLP <-> MLPTE
# ---------------------------------------------------------------------------


def copy_mlp_to_te(orig, te_mod):
    # orig.mlp: Sequential(LayerNorm, Linear, ReLU, Linear)
    _copy_layernorm_to_te(orig.mlp[0], te_mod.norm)
    _copy_linear_to_te(orig.mlp[1], te_mod.fc1)
    _copy_linear_to_te(orig.mlp[3], te_mod.fc2)


def copy_mlp_from_te(te_mod, orig):
    _copy_layernorm_from_te(te_mod.norm, orig.mlp[0])
    _copy_linear_from_te(te_mod.fc1, orig.mlp[1])
    _copy_linear_from_te(te_mod.fc2, orig.mlp[3])


# ---------------------------------------------------------------------------
# AngleResnetBlock <-> AngleResnetBlockTE
# ---------------------------------------------------------------------------


def copy_angle_resnet_block_to_te(orig, te_mod):
    # orig.mlp: Sequential(ReLU, Linear, ReLU, Linear)
    _copy_linear_to_te(orig.mlp[1], te_mod.fc1)
    _copy_linear_to_te(orig.mlp[3], te_mod.fc2)


def copy_angle_resnet_block_from_te(te_mod, orig):
    _copy_linear_from_te(te_mod.fc1, orig.mlp[1])
    _copy_linear_from_te(te_mod.fc2, orig.mlp[3])


# ---------------------------------------------------------------------------
# AngleResnet <-> AngleResnetTE
# ---------------------------------------------------------------------------


def copy_angle_resnet_to_te(orig, te_mod):
    _copy_linear_to_te(orig.linear_in, te_mod.linear_in)
    _copy_linear_to_te(orig.linear_initial, te_mod.linear_initial)
    _copy_linear_to_te(orig.linear_out, te_mod.linear_out)
    for orig_layer, te_layer in zip(orig.layers, te_mod.layers):
        copy_angle_resnet_block_to_te(orig_layer, te_layer)


def copy_angle_resnet_from_te(te_mod, orig):
    _copy_linear_from_te(te_mod.linear_in, orig.linear_in)
    _copy_linear_from_te(te_mod.linear_initial, orig.linear_initial)
    _copy_linear_from_te(te_mod.linear_out, orig.linear_out)
    for orig_layer, te_layer in zip(orig.layers, te_mod.layers):
        copy_angle_resnet_block_from_te(te_layer, orig_layer)


# ---------------------------------------------------------------------------
# StructureModule <-> StructureModuleTE
# ---------------------------------------------------------------------------


def copy_structure_module_to_te(orig, te_mod):
    _copy_layernorm_to_te(orig.layer_norm_s, te_mod.layer_norm_s)
    _copy_layernorm_to_te(orig.layer_norm_z, te_mod.layer_norm_z)
    _copy_linear_to_te(orig.linear_in, te_mod.linear_in)
    _copy_linear_to_te(orig.linear_b, te_mod.linear_b)
    _copy_linear_to_te(orig.bb_update, te_mod.bb_update)

    for orig_attn, te_attn in zip(orig.attn, te_mod.attn):
        copy_attention_to_te(orig_attn, te_attn)
    for orig_trans, te_trans in zip(orig.transitions, te_mod.transitions):
        copy_mlp_to_te(orig_trans, te_trans)

    copy_angle_resnet_to_te(orig.angle_resnet, te_mod.angle_resnet)


def copy_structure_module_from_te(te_mod, orig):
    _copy_layernorm_from_te(te_mod.layer_norm_s, orig.layer_norm_s)
    _copy_layernorm_from_te(te_mod.layer_norm_z, orig.layer_norm_z)
    _copy_linear_from_te(te_mod.linear_in, orig.linear_in)
    _copy_linear_from_te(te_mod.linear_b, orig.linear_b)
    _copy_linear_from_te(te_mod.bb_update, orig.bb_update)

    for orig_attn, te_attn in zip(orig.attn, te_mod.attn):
        copy_attention_from_te(te_attn, orig_attn)
    for orig_trans, te_trans in zip(orig.transitions, te_mod.transitions):
        copy_mlp_from_te(te_trans, orig_trans)

    copy_angle_resnet_from_te(te_mod.angle_resnet, orig.angle_resnet)


# ---------------------------------------------------------------------------
# PerResidueLDDTCaPredictor <-> PerResidueLDDTCaPredictorTE
# ---------------------------------------------------------------------------


def copy_plddt_to_te(orig, te_mod):
    _copy_layernorm_to_te(orig.layer_norm, te_mod.layer_norm)
    _copy_linear_to_te(orig.linear_1, te_mod.linear_1)
    _copy_linear_to_te(orig.linear_2, te_mod.linear_2)
    _copy_linear_to_te(orig.linear_3, te_mod.linear_3)


def copy_plddt_from_te(te_mod, orig):
    _copy_layernorm_from_te(te_mod.layer_norm, orig.layer_norm)
    _copy_linear_from_te(te_mod.linear_1, orig.linear_1)
    _copy_linear_from_te(te_mod.linear_2, orig.linear_2)
    _copy_linear_from_te(te_mod.linear_3, orig.linear_3)
