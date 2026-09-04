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


import torch
from torch import nn


def rotate_half(x):
    """Rotates the last dimension of a tensor by 180 degrees.

    Args:
        x (torch.Tensor): Input tensor of shape (..., dim) where dim is even.

    Returns:
        torch.Tensor: Rotated tensor of same shape as input.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    """Applies rotary positional embeddings to a tensor.

    Args:
        x (torch.Tensor): Input tensor to apply rotary embeddings to.
        cos (torch.Tensor): Cosine values for the rotation.
        sin (torch.Tensor): Sine values for the rotation.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    return ((x * cos) + (rotate_half(x) * sin)).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Basic implementation of Rotary Positional Embeddings with theta rescaling.

    This module implements rotary positional embeddings (RoPE) with NTK-aware theta
    rescaling for better handling of longer sequences. The implementation is based on
    the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding" and
    includes improvements from the NTK literature for better length extrapolation.

    The theta rescaling is based on the work by bloc97 (Reddit) which has connections
    to NTK literature and allows for better handling of longer sequences without
    fine-tuning.

    Args:
        dim (int): Embedding dimension.
        theta (float, optional): Base constant for computing rotation angles.
            Defaults to 10000.
        theta_rescale_factor (float, optional): Factor to rescale theta for better
            handling of longer sequences. Defaults to 1.0.

    Attributes:
        inv_freq (torch.Tensor): Inverse frequencies used for rotation.
        seq_len_cached (int): Cached sequence length.
        cos_cached (torch.Tensor): Cached cosine values.
        sin_cached (torch.Tensor): Cached sine values.
    """

    def __init__(self, dim, theta=10000, theta_rescale_factor=1.0):  # noqa: D107
        super().__init__()
        # Apply NTK-aware theta rescaling
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        # Calculate inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for sequence length
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        """Forward pass to compute rotary embeddings.
        Assumes sequence length is the 2nd dimension
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (cos_cached, sin_cached) tensors for applying rotary embeddings.
        """  # noqa: D205
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached
