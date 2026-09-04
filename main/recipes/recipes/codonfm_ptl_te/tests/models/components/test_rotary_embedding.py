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

from src.models.components.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb, rotate_half


class TestRotaryEmbeddingHelpers:
    def test_rotate_half(self):
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        rotated = rotate_half(x)
        expected = torch.tensor([[-3, -4, 1, 2], [-7, -8, 5, 6]])
        assert torch.equal(rotated, expected)

    def test_apply_rotary_pos_emb(self):
        x = torch.randn(2, 4, 8, 16)  # B, S, H, D
        cos = torch.randn(1, 4, 1, 16)
        sin = torch.randn(1, 4, 1, 16)

        output = apply_rotary_pos_emb(x, cos, sin)
        assert output.shape == x.shape
        assert output.dtype == x.dtype

        # Check a manual calculation for a simple case
        x_simple = torch.tensor([1.0, 2.0, 3.0, 4.0])
        cos_simple = torch.cos(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        sin_simple = torch.sin(torch.tensor([0.5, 0.5, 0.5, 0.5]))

        # apply_rotary_pos_emb expects (B, S, H, D) or similar, let's adapt
        x_simple = x_simple.view(1, 1, 1, 4)
        cos_simple = cos_simple.view(1, 1, 1, 4)
        sin_simple = sin_simple.view(1, 1, 1, 4)

        output_simple = apply_rotary_pos_emb(x_simple, cos_simple, sin_simple)

        x_rotated = rotate_half(x_simple)
        expected = (x_simple * cos_simple) + (x_rotated * sin_simple)

        assert torch.allclose(output_simple, expected)


class TestRotaryEmbeddingModule:
    def test_init(self):
        module = RotaryEmbedding(dim=64)
        assert module.inv_freq.shape == (32,)

    def test_forward_pass_and_caching(self):
        module = RotaryEmbedding(dim=64)
        x = torch.randn(2, 10, 8, 64)  # B, S, H, D

        # First forward pass
        cos, sin = module(x)

        assert cos.shape == (1, 10, 1, 64)
        assert sin.shape == (1, 10, 1, 64)
        assert module.seq_len_cached == 10

        # Cache should be used on second pass with same seq_len
        cos_cached, sin_cached = module.cos_cached, module.sin_cached
        cos2, sin2 = module(x)
        assert torch.equal(cos_cached, cos2)
        assert torch.equal(sin_cached, sin2)

        # Different seq_len should recompute
        x2 = torch.randn(2, 20, 8, 64)
        cos3, sin3 = module(x2)
        assert module.seq_len_cached == 20
        assert not torch.equal(cos_cached, cos3)

    def test_theta_rescaling(self):
        # theta *= theta_rescale_factor ** (dim / (dim - 2))
        dim = 64
        theta = 10000

        # No rescaling
        module1 = RotaryEmbedding(dim=dim, theta=theta, theta_rescale_factor=1.0)

        # With rescaling
        rescale_factor = 2.0
        module2 = RotaryEmbedding(dim=dim, theta=theta, theta_rescale_factor=rescale_factor)

        # Check that inv_freq is different
        assert not torch.allclose(module1.inv_freq, module2.inv_freq)

        # Manual check of the formula
        expected_theta_rescaled = theta * (rescale_factor ** (dim / (dim - 2)))
        expected_inv_freq = 1.0 / (expected_theta_rescaled ** (torch.arange(0, dim, 2).float() / dim))

        assert torch.allclose(module2.inv_freq, expected_inv_freq)
