# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from transformers import AutoConfig

from bionemo.amplify.convert import maybe_mock_xformers
from bionemo.amplify.hf_rotary import apply_rotary_emb, precompute_freqs_cis
from bionemo.amplify.model import AMPLIFYConfig


def test_rope_embeddings():
    # Mock the xformers module to allow this test to succeed without needing to install xformers.
    maybe_mock_xformers()

    rng = torch.Generator().manual_seed(42)
    q = torch.randn([2, 72, 10, 64], dtype=torch.float32, generator=rng)
    k = torch.randn([2, 72, 10, 64], dtype=torch.float32, generator=rng)

    # AMPLIFY HF Rope
    hf_config = AutoConfig.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True)

    freqs_cis = precompute_freqs_cis(hf_config.hidden_size // hf_config.num_attention_heads, hf_config.max_length)
    freqs_cis = freqs_cis[: q.shape[1]]
    q_post, k_post = apply_rotary_emb(q, k, freqs_cis)

    # NeMo Rope
    nemo_config = AMPLIFYConfig(apply_rope_fusion=False, rotary_interleaved=True)
    rotary_pos_layer = RotaryEmbedding(
        kv_channels=nemo_config.kv_channels,
        rotary_percent=nemo_config.rotary_percent,
        rotary_interleaved=nemo_config.rotary_interleaved,
        seq_len_interpolation_factor=nemo_config.seq_len_interpolation_factor,
    )
    rotary_pos_emb = rotary_pos_layer(q.shape[1])
    # Note: Use the backend implementation of the RoPE to avoid
    # getting or instantiating a CP process group.
    q_post_nemo = _apply_rotary_pos_emb_bshd(
        q.transpose(0, 1).cuda(),
        rotary_pos_emb.cuda(),
        rotary_interleaved=nemo_config.rotary_interleaved,
        multi_latent_attention=nemo_config.multi_latent_attention,
    ).cpu()
    k_post_nemo = _apply_rotary_pos_emb_bshd(
        k.transpose(0, 1).cuda(),
        rotary_pos_emb.cuda(),
        rotary_interleaved=nemo_config.rotary_interleaved,
        multi_latent_attention=nemo_config.multi_latent_attention,
    ).cpu()

    torch.testing.assert_close(q_post, q_post_nemo.transpose(0, 1))
    torch.testing.assert_close(k_post, k_post_nemo.transpose(0, 1))
