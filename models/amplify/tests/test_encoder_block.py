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

import pytest
import torch
from transformer_engine.pytorch import TransformerLayer
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import AutoConfig

from amplify.amplify_hf import EncoderBlock
from amplify.rotary import apply_rotary_emb, precompute_freqs_cis


class ForwardHook:
    """A forward hook to extract a desired intermediate tensor for later comparison."""

    def __init__(self, transform_fn) -> None:
        """A forward hook to extract a desired intermediate tensor for later comparison.

        The resulting tensor is saved in the `data` attribute of the hook.

        Args:
            transform_fn: A function that maps the input and output tensors of the module to the desired tensor.
        """
        self._transform_fn = transform_fn
        self._data: torch.Tensor | None = None

    def __call__(self, module, module_in, module_out):
        """The forward hook function."""
        if not isinstance(module_out, tuple):
            module_out = (module_out,)
        if not isinstance(module_in, tuple):
            module_in = (module_in,)

        self._data = self._transform_fn(module_in, module_out).detach().cpu()

    @property
    def data(self) -> torch.Tensor:
        """The extracted tensor from the forward hook."""
        if self._data is None:
            raise ValueError("No data has been saved in this hook.")
        return self._data


@pytest.fixture
def config():
    config = AutoConfig.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True, revision="d918a9e8")
    config.dtype = torch.bfloat16
    return config


@pytest.fixture
def inputs(config):
    batch_size = 12
    torch.manual_seed(42)

    hidden_states = torch.randn(batch_size, config.max_length, config.hidden_size, dtype=torch.bfloat16).to("cuda")

    attention_mask = torch.zeros(batch_size, config.max_length).to("cuda")
    attention_mask[0, -5:] = 1
    attention_mask[1, -10:] = 1
    attention_mask[2, -15:] = 1
    attention_mask = attention_mask.bool()

    return hidden_states, attention_mask


def test_encoder_block_forward(inputs, config):
    hidden_states, attention_mask = inputs

    # Process attention mask for HF xformers
    additive_attention_mask = torch.where(attention_mask == 1, float(0.0), float("-inf")).to(torch.bfloat16)

    additive_attention_mask = (
        additive_attention_mask.unsqueeze(1)
        .unsqueeze(1)
        .repeat(1, config.num_attention_heads, additive_attention_mask.size(-1), 1)
    )

    encoder_block = EncoderBlock(config).to("cuda", dtype=torch.bfloat16)

    freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length).to("cuda")

    # Add hooks for HF layer intermediates
    hf_query_pre_rot_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    encoder_block.q.register_forward_hook(hf_query_pre_rot_hook)

    hf_key_pre_rot_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    encoder_block.k.register_forward_hook(hf_key_pre_rot_hook)

    hf_value_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    encoder_block.v.register_forward_hook(hf_value_hook)

    # The output of the attention layer is the same as the output of the linear layer, but the actual attention function
    # isn't wrapped in a nn.Module.
    hf_attn_output_hook = ForwardHook(lambda inputs, outputs: inputs[0])
    encoder_block.wo.register_forward_hook(hf_attn_output_hook)

    hf_attn_linear_output_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    encoder_block.wo.register_forward_hook(hf_attn_linear_output_hook)

    output_hf, attentions = encoder_block(
        hidden_states,
        attention_mask=additive_attention_mask,
        freqs_cis=freqs_cis,
        output_attentions=False,
    )

    # These post-rotary embeddings are applied in the forward pass of the model, so we need to apply them here.
    xq = hf_query_pre_rot_hook.data.view(
        hidden_states.shape[0],
        hidden_states.shape[1],
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
    )
    xk = hf_key_pre_rot_hook.data.view(
        hidden_states.shape[0],
        hidden_states.shape[1],
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
    )
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis[: hidden_states.shape[1]].cpu())

    hf_data = {
        "query_post_rot": xq.flatten(-2, -1),
        "key_post_rot": xk.flatten(-2, -1),
        "value": hf_value_hook.data,
        "attn_output": hf_attn_output_hook.data,
        "attn_linear_output": hf_attn_linear_output_hook.data,
    }

    assert output_hf.shape == hidden_states.shape

    # Apply some strange AMPLIFY correction.
    multiple_of = 8
    intermediate_size = int(2 * config.intermediate_size / 3)
    te_ffn_hidden_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

    transformer_block = TransformerLayer(
        hidden_size=config.hidden_size,
        ffn_hidden_size=te_ffn_hidden_size,
        num_attention_heads=config.num_attention_heads,
        layernorm_epsilon=config.norm_eps,
        hidden_dropout=config.dropout_prob,
        attention_dropout=config.dropout_prob,
        apply_residual_connection_post_layernorm=False,
        layer_type="encoder",
        self_attn_mask_type="padding",
        normalization="RMSNorm",
        fuse_qkv_params=False,
        output_layernorm=False,
        bias=False,
        activation=config.hidden_act.lower(),
        attn_input_format="bshd",
        layer_number=1,
        name="encoder_block",
        window_size=(-1, -1),
        rotary_pos_interleaved=True,
        seq_length=config.max_length,
        params_dtype=config.dtype,
    ).to("cuda", dtype=torch.bfloat16)

    state_dict_mapping = {
        "q.weight": "self_attention.layernorm_qkv.query_weight",
        "k.weight": "self_attention.layernorm_qkv.key_weight",
        "v.weight": "self_attention.layernorm_qkv.value_weight",
        "wo.weight": "self_attention.proj.weight",
        "ffn.w12.weight": "layernorm_mlp.fc1_weight",
        "ffn.w3.weight": "layernorm_mlp.fc2_weight",
        "attention_norm.weight": "self_attention.layernorm_qkv.layer_norm_weight",
        "ffn_norm.weight": "layernorm_mlp.layer_norm_weight",
    }

    transformer_block.state_dict()

    hf_state_dict = encoder_block.state_dict()
    te_state_dict = {te_key: hf_state_dict[hf_key] for hf_key, te_key in state_dict_mapping.items()}
    transformer_block.load_state_dict(te_state_dict, strict=False)

    rope_layer = RotaryPositionEmbedding(
        config.hidden_size // config.num_attention_heads,
        interleaved=True,
    )
    rope_layer.to("cuda")
    freqs = rope_layer.forward(config.max_length)

    # Add hooks for TE layer intermediates
    te_query_post_rot_hook = ForwardHook(lambda inputs, outputs: inputs[0].flatten(-2, -1))
    transformer_block.self_attention.core_attention.register_forward_hook(te_query_post_rot_hook)

    te_key_post_rot_hook = ForwardHook(lambda inputs, outputs: inputs[1].flatten(-2, -1))
    transformer_block.self_attention.core_attention.register_forward_hook(te_key_post_rot_hook)

    te_value_post_rot_hook = ForwardHook(lambda inputs, outputs: inputs[2].flatten(-2, -1))
    transformer_block.self_attention.core_attention.register_forward_hook(te_value_post_rot_hook)

    te_attn_output_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    transformer_block.self_attention.core_attention.register_forward_hook(te_attn_output_hook)

    te_attn_linear_output_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    transformer_block.self_attention.proj.register_forward_hook(te_attn_linear_output_hook)

    output_te = transformer_block(
        hidden_states,
        attention_mask=attention_mask,
        rotary_pos_emb=freqs,
    )

    te_data = {
        "query_post_rot": te_query_post_rot_hook.data,
        "key_post_rot": te_key_post_rot_hook.data,
        "value": te_value_post_rot_hook.data,
        "attn_output": te_attn_output_hook.data,
        "attn_linear_output": te_attn_linear_output_hook.data,
    }

    total_params = sum(p.numel() for p in encoder_block.parameters())
    total_params_te = sum(p.numel() for p in transformer_block.parameters())

    assert total_params == total_params_te

    torch.testing.assert_close(hf_data["query_post_rot"], te_data["query_post_rot"], atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(hf_data["key_post_rot"], te_data["key_post_rot"], atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(hf_data["value"], te_data["value"], atol=1e-1, rtol=1e-1)
    # torch.testing.assert_close(
    #     hf_data["attn_output"], te_data["attn_output"], atol=1e-1, rtol=1e-1
    # )
    # torch.testing.assert_close(
    #     hf_data["attn_linear_output"],
    #     te_data["attn_linear_output"],
    #     atol=1e-1,
    #     rtol=1e-1,
    # )

    torch.testing.assert_close(output_hf, output_te, atol=1.0, rtol=0.01)
