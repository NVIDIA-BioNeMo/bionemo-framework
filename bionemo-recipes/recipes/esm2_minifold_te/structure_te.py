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

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from einops import rearrange, repeat
from torch import Tensor

from minifold_utils import init
from minifold_utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from minifold_utils.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from minifold_utils.rigid_utils import Rigid
from minifold_utils.tensor_utils import dict_multimap, permute_final_dims
from te_utils import te_layernorm_nd, te_linear_nd


class AttentionTE(nn.Module):
    """TE version of gated self-attention used in the StructureModule."""

    def __init__(self, dim: int, num_heads: int, head_width: int, params_dtype: torch.dtype = torch.float32):
        super().__init__()
        assert dim == num_heads * head_width

        self.dim = dim
        self.num_heads = num_heads
        self.head_width = head_width
        self.rescale_factor = self.head_width**-0.5

        # Cannot fuse LN+proj because g_proj also reads the LN output
        self.layer_norm = te.LayerNorm(dim, eps=1e-5, params_dtype=params_dtype)
        self.proj = te.Linear(dim, dim * 3, bias=False, params_dtype=params_dtype)
        self.o_proj = te.Linear(dim, dim, bias=True, params_dtype=params_dtype)
        self.g_proj = te.Linear(dim, dim, bias=True, params_dtype=params_dtype)

        torch.nn.init.zeros_(self.o_proj.bias)
        torch.nn.init.zeros_(self.g_proj.weight)
        torch.nn.init.ones_(self.g_proj.bias)

    def forward(self, x: Tensor, bias: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, N, D).
            bias: External attention bias (B, H, N, N).
            mask: Mask tensor (B, N).

        Returns:
            Output tensor (B, N, D).
        """
        x = te_layernorm_nd(self.layer_norm, x)

        t = rearrange(te_linear_nd(self.proj, x), "... l (h c) -> ... h l c", h=self.num_heads)
        q, k, v = t.chunk(3, dim=-1)

        q = self.rescale_factor * q
        a = torch.einsum("...qc,...kc->...qk", q, k)

        # Add external attention bias
        a = a + bias

        # Mask padding tokens
        mask = repeat(mask, "... lk -> ... h lq lk", h=self.num_heads, lq=q.shape[-2])
        a = a.masked_fill(mask == 0, -np.inf)
        a = F.softmax(a, dim=-1)

        y = torch.einsum("...hqk,...hkc->...qhc", a, v)
        y = rearrange(y, "... h c -> ... (h c)", h=self.num_heads)
        y = te_linear_nd(self.g_proj, x).sigmoid() * y
        y = te_linear_nd(self.o_proj, y)

        return y


class MLPTE(nn.Module):
    """TE version of the MLP used in StructureModule transitions."""

    def __init__(self, in_dim: int, out_dim: int, params_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.norm = te.LayerNorm(in_dim, eps=1e-5, params_dtype=params_dtype)
        self.fc1 = te.Linear(in_dim, in_dim, params_dtype=params_dtype)
        self.fc2 = te.Linear(in_dim, out_dim, params_dtype=params_dtype)

        init.he_normal_init_(self.fc1.weight)
        init.final_init_(self.fc2.weight)
        init.bias_init_zero_(self.fc1.bias)
        init.bias_init_zero_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (..., D_in).

        Returns:
            Output tensor (..., D_out).
        """
        x = te_layernorm_nd(self.norm, x)
        x = te_linear_nd(self.fc1, x)
        x = F.relu(x)
        x = te_linear_nd(self.fc2, x)
        return x


class AngleResnetBlockTE(nn.Module):
    """TE version of AngleResnetBlock."""

    def __init__(self, dim, params_dtype=torch.float32):
        super().__init__()
        self.fc1 = te.Linear(dim, dim, params_dtype=params_dtype)
        self.fc2 = te.Linear(dim, dim, params_dtype=params_dtype)

        init.he_normal_init_(self.fc1.weight)
        init.final_init_(self.fc2.weight)
        init.bias_init_zero_(self.fc1.bias)
        init.bias_init_zero_(self.fc2.bias)

    def forward(self, a: Tensor) -> Tensor:
        x = F.relu(a)
        x = te_linear_nd(self.fc1, x)
        x = F.relu(x)
        x = te_linear_nd(self.fc2, x)
        return a + x


class AngleResnetTE(nn.Module):
    """TE version of AngleResnet."""

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon, params_dtype=torch.float32):
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = te.Linear(self.c_in, self.c_hidden, params_dtype=params_dtype)
        self.linear_initial = te.Linear(self.c_in, self.c_hidden, params_dtype=params_dtype)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            self.layers.append(AngleResnetBlockTE(dim=self.c_hidden, params_dtype=params_dtype))

        self.linear_out = te.Linear(self.c_hidden, self.no_angles * 2, params_dtype=params_dtype)

        init.lecun_normal_init_(self.linear_in.weight)
        init.lecun_normal_init_(self.linear_initial.weight)
        init.final_init_(self.linear_out.weight)

        init.bias_init_zero_(self.linear_in.bias)
        init.bias_init_zero_(self.linear_initial.bias)
        init.bias_init_zero_(self.linear_out.bias)

        self.relu = nn.ReLU()

    def forward(self, s: Tensor, s_initial: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            s: Single embedding [*, C_hidden].
            s_initial: Initial single embedding [*, C_hidden].

        Returns:
            Tuple of (unnormalized_angles, normalized_angles), each [*, no_angles, 2].
        """
        s_initial = self.relu(s_initial)
        s_initial = te_linear_nd(self.linear_initial, s_initial)
        s = self.relu(s)
        s = te_linear_nd(self.linear_in, s)
        s = s + s_initial

        for layer in self.layers:
            s = layer(s)

        s = self.relu(s)
        s = te_linear_nd(self.linear_out, s)

        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s


class StructureModuleTE(nn.Module):
    """TE version of the StructureModule."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_resnet: int,
        head_dim: int,
        no_heads: int,
        no_blocks: int,
        no_resnet_blocks: int,
        no_angles: int,
        trans_scale_factor: float,
        epsilon: float,
        inf: float,
        params_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_resnet = c_resnet
        self.no_heads = no_heads
        self.head_dim = head_dim
        self.no_blocks = no_blocks
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        self.layer_norm_s = te.LayerNorm(self.c_s, eps=1e-5, params_dtype=params_dtype)
        self.layer_norm_z = te.LayerNorm(self.c_z, eps=1e-5, params_dtype=params_dtype)
        self.linear_in = te.Linear(self.c_s, self.c_s, params_dtype=params_dtype)
        self.linear_b = te.Linear(self.c_z, self.no_blocks * self.no_heads, params_dtype=params_dtype)

        self.attn = nn.ModuleList(
            [
                AttentionTE(self.c_s, self.no_heads, self.head_dim, params_dtype=params_dtype)
                for _ in range(self.no_blocks)
            ]
        )
        self.transitions = nn.ModuleList(
            [MLPTE(self.c_s, self.c_s, params_dtype=params_dtype) for _ in range(self.no_blocks)]
        )

        self.bb_update = te.Linear(self.c_s, 9, params_dtype=params_dtype)
        self.angle_resnet = AngleResnetTE(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
            params_dtype=params_dtype,
        )

        # Initialize weights
        init.lecun_normal_init_(self.linear_in.weight)
        init.bias_init_zero_(self.linear_in.bias)
        init.lecun_normal_init_(self.bb_update.weight)
        init.bias_init_zero_(self.bb_update.bias)
        init.lecun_normal_init_(self.linear_b.weight)
        init.bias_init_zero_(self.linear_b.bias)

        # Initialize buffers
        frames = torch.tensor(restype_rigid_group_default_frame)
        groups = torch.tensor(restype_atom14_to_rigid_group)
        atom_mask = torch.tensor(restype_atom14_mask)
        positions = torch.tensor(restype_atom14_rigid_group_positions)

        self.register_buffer("default_frames", frames, persistent=False)
        self.register_buffer("group_idx", groups, persistent=False)
        self.register_buffer("atom_mask", atom_mask, persistent=False)
        self.register_buffer("lit_positions", positions, persistent=False)

    def forward(self, s, z, aatype, mask):
        """Forward pass.

        Args:
            s: Single representation (B, N, c_s).
            z: Pair representation (B, N, N, c_z).
            aatype: Amino acid types (B, N).
            mask: Residue mask (B, N).

        Returns:
            Dictionary with angles, frames, positions, states.
        """
        # Input projection
        s = te_layernorm_nd(self.layer_norm_s, s)
        s_initial = s
        s = te_linear_nd(self.linear_in, s)

        # Pairwise bias
        B, N = s.shape[:2]
        z = te_layernorm_nd(self.layer_norm_z, z)
        b = te_linear_nd(self.linear_b, z)
        b = permute_final_dims(b, (2, 0, 1))
        b = b.reshape(B, self.no_blocks, self.no_heads, N, N)

        # Apply transformer layers
        outputs = []
        for i in range(self.no_blocks):
            s = s + self.attn[i](s, b[:, i], mask)
            s = s + self.transitions[i](s)

        # Predict angles
        unnormalized_angles, angles = self.angle_resnet(s, s_initial)

        # Predict positions (in FP32 via explicit .float() cast)
        n, ca, c = te_linear_nd(self.bb_update, s.float()).chunk(3, dim=-1)
        rigids = Rigid.make_transform_from_reference(n, ca, c, eps=1e-7)
        scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

        all_frames_to_global = torsion_angles_to_frames(scaled_rigids, angles, aatype, self.default_frames)
        pred_xyz = frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            aatype,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
        outputs.append(
            {
                "angles": angles,
                "unnormalized_angles": unnormalized_angles,
                "frames": scaled_rigids.to_tensor_4x4(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "positions": pred_xyz,
                "states": s,
            }
        )

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s
        return outputs
