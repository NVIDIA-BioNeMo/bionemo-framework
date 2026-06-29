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
from transformer_engine.pytorch.attention.rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)
from transformers import AutoConfig

from amplify.rotary import apply_rotary_emb, precompute_freqs_cis


def test_apply_rotary_pos_emb():
    rng = torch.Generator().manual_seed(42)
    query = torch.randn([2, 72, 10, 64], dtype=torch.bfloat16, generator=rng).to("cuda")
    key = torch.randn([2, 72, 10, 64], dtype=torch.bfloat16, generator=rng).to("cuda")

    # AMPLIFY HF Rope
    hf_config = AutoConfig.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True, revision="d918a9e8")

    freqs_cis = precompute_freqs_cis(hf_config.hidden_size // hf_config.num_attention_heads, 72).to("cuda")
    q_post, k_post = apply_rotary_emb(query, key, freqs_cis)

    # TE Rope
    rope_layer = RotaryPositionEmbedding(hf_config.hidden_size // hf_config.num_attention_heads, interleaved=True)
    rope_layer.to("cuda")
    freqs = rope_layer.forward(72)
    q_post_te = apply_rotary_pos_emb(
        query,
        freqs,
        tensor_format="bshd",
        fused=True,
        interleaved=True,
    )

    k_post_te = apply_rotary_pos_emb(
        key,
        freqs,
        tensor_format="bshd",
        fused=True,
        interleaved=True,
    )

    torch.testing.assert_close(q_post, q_post_te, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(k_post, k_post_te, atol=1e-2, rtol=1e-3)
