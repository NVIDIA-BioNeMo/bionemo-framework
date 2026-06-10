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

import gc
import re

import torch
import transformer_engine as te
from torch import nn
from torch import Tensor
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import AutoConfig
from transformers import LlamaForCausalLM
from transformers import LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NVLlamaDecoderLayer(te.pytorch.TransformerLayer):
    """NVLlamaEncoder is a TransformerEngine-optimized Llama encoder."""

    def __init__(self, config: LlamaConfig, te_rope_emb: Tensor | tuple[Tensor, Tensor] | None, *args, **kwargs):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=config.num_key_value_heads,
        )
        self.te_rope_emb = te_rope_emb

    def forward(self, hidden_states, *args, attention_mask, **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return super().forward(hidden_states, attention_mask=attention_mask, rotary_pos_emb=self.te_rope_emb)


@staticmethod
def _convert_llama_state_dict_to_nv_llama(hf_state_dict: dict, layer_prefix_pattern: str, config: LlamaConfig):
    # collect all layer prefixes to update
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        m = re.match(layer_prefix_pattern, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    for layer_prefix in all_layer_prefixes:
        # When loading weights into models with less number of layers, skip the
        # copy if the corresponding layer doesn't exist in HF model
        if layer_prefix + "input_layernorm.weight" in hf_state_dict:
            hf_state_dict[layer_prefix + "self_attention.layernorm_qkv.layer_norm_weight"] = hf_state_dict[layer_prefix + "input_layernorm.weight"]
            del hf_state_dict[layer_prefix + "input_layernorm.weight"]

        if layer_prefix + "self_attn.q_proj.weight" in hf_state_dict:
            hf_state_dict[layer_prefix + "self_attention.layernorm_qkv.query_weight"] = hf_state_dict[layer_prefix + "self_attn.q_proj.weight"]
            del hf_state_dict[layer_prefix + "self_attn.q_proj.weight"]

        if layer_prefix + "self_attn.k_proj.weight" in hf_state_dict:
            hf_state_dict[layer_prefix + "self_attention.layernorm_qkv.key_weight"] = hf_state_dict[layer_prefix + "self_attn.k_proj.weight"]
            del hf_state_dict[layer_prefix + "self_attn.k_proj.weight"]

        if layer_prefix + "self_attn.v_proj.weight" in hf_state_dict:
            hf_state_dict[layer_prefix + "self_attention.layernorm_qkv.value_weight"] = hf_state_dict[layer_prefix + "self_attn.v_proj.weight"]
            del hf_state_dict[layer_prefix + "self_attn.v_proj.weight"]

        if layer_prefix + "self_attn.o_proj.weight" in hf_state_dict:
            hf_state_dict[layer_prefix + "self_attention.proj.weight"] = hf_state_dict[
                layer_prefix + "self_attn.o_proj.weight"
            ]
            del hf_state_dict[layer_prefix + "self_attn.o_proj.weight"]

        if layer_prefix + "post_attention_layernorm.weight" in hf_state_dict:
            hf_state_dict[layer_prefix + "layernorm_mlp.layer_norm_weight"] = hf_state_dict[
                layer_prefix + "post_attention_layernorm.weight"
            ]
            del hf_state_dict[layer_prefix + "post_attention_layernorm.weight"]

        # It may happen that gate_proj.weight and up_proj.weight will be in the different files, so we need to
        # load them separately.
        if layer_prefix + "mlp.gate_proj.weight" in hf_state_dict:
            if hf_state_dict.get(layer_prefix + "layernorm_mlp.fc1_weight") is None:
                t = hf_state_dict[layer_prefix + "mlp.gate_proj.weight"]
                rows, cols = t.shape
                hf_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"] = torch.empty((2 * rows, cols), dtype=t.dtype, device=t.device)

            hf_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[: config.intermediate_size] = hf_state_dict[layer_prefix + "mlp.gate_proj.weight"].data
            del hf_state_dict[layer_prefix + "mlp.gate_proj.weight"]

        if layer_prefix + "mlp.up_proj.weight" in hf_state_dict:
            if hf_state_dict.get(layer_prefix + "layernorm_mlp.fc1_weight") is None:
                t = hf_state_dict[layer_prefix + "mlp.gate_proj.weight"]
                rows, cols = t.shape
                hf_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"] = torch.empty((2 * rows, cols), dtype=t.dtype, device=t.device)

            hf_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[config.intermediate_size :] = hf_state_dict[layer_prefix + "mlp.up_proj.weight"].data
            del hf_state_dict[layer_prefix + "mlp.up_proj.weight"]

        if layer_prefix + "mlp.down_proj.weight" in hf_state_dict:
            hf_state_dict[layer_prefix + "layernorm_mlp.fc2_weight"]= hf_state_dict[
                layer_prefix + "mlp.down_proj.weight"
            ]
            del hf_state_dict[layer_prefix + "mlp.down_proj.weight"]


# @balvisio: Might be better to inherit from PreTrainedModel instead of overriding
class NVLlamaPreTrainedModel(LlamaPreTrainedModel):
    """An abstract class to handle weights initialization and pretrained model loading."""
    _no_split_modules = ["TransformerLayer"]
    _can_record_outputs = {
        "hidden_states": te.pytorch.TransformerLayer,
        "attentions": te.pytorch.MultiheadAttention,
    }


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, *model_args, config=None, **kwargs):
        # Before loading the model, set the default dtype for torch
        torch_dtype = kwargs.get("torch_dtype", None)
        if torch_dtype is not None:
            torch.set_default_dtype(torch_dtype)

        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config._attn_implementation = "flash_attention_2"

        hf_cls, layer_prefix_pattern = NV_TO_HF_CLASS_MAP[cls]
        # 1. If user passed state_dict directly, just convert and load
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is not None:
            _convert_llama_state_dict_to_nv_llama(state_dict, layer_prefix_pattern, config)
            return super().from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, state_dict=state_dict, **kwargs
            )

        # 2. Try loading baseline LLaMA for extraction
        try:
            temp_model = hf_cls.from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
            state_dict = temp_model.state_dict()

            # Detect if it's baseline LLaMA or already NVLlama.
            if any("self_attn.q_proj" in k for k in state_dict.keys()):
                # Looks like baseline LLaMA → convert
                _convert_llama_state_dict_to_nv_llama(state_dict, layer_prefix_pattern, config)
                # Since we will be passing the 'state_dict', we don't pass the name of the model in 'pretrained_model_name_or_path'
                pretrained_model_name_or_path = None
                # To avoid printing WARNING about missing keys in 'state_dict' such as 'layers.0.layernorm_mlp._extra_state'
                kwargs["_keys_to_ignore_on_load_missing"] = [r".*_extra_state$"]
                # Pass the generation_config for generation but set 'use_cache' to False
                generation_config = temp_model.generation_config
                generation_config.use_cache = False
                kwargs["generation_config"] = generation_config

            del temp_model
            gc.collect()
        except Exception as e:
            logger.warning(f"Could not load as LlamaModel, falling back to standard loading: {e}")

        # Load NVLlama using maybe (if 'state_dict' is not None) converted weights
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, config=config, state_dict=state_dict, **kwargs
        )
        model = model.cuda()

        # Needed for the cases when using TELlamaForCausalLM
        model.config.use_cache = False

        # Make sure memory is freed.
        del state_dict
        gc.collect()

        return model


class NoOp(nn.Module):
    def __init__(self, return_input=True):
        super().__init__()
        self.return_input = return_input

    def forward(self, *x):
        if self.return_input:
            return x
        return None


class NVLlamaModel(NVLlamaPreTrainedModel, LlamaModel):
    """Llama model. This model uses NVDIA's TransformerEngine to optimize attention layer training and inference."""
    def __init__(self, config: LlamaConfig, **kwargs):
        NVLlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # The TE RotaryPositionEmbedding layer works differently from the LlamaRotaryEmbedding. The forward() method of:
        # - TE RotaryPositionEmbedding: Takes the 'max_seq_length' and creates rotary position embedding frequencies.
        #   They are passed to the Transformer Layer.
        # - LlamaRotaryEmbedding: Calculates the positional embeddings given the embeddings and positions_ids. These
        #   are passed to the Decoder Layer.
        #
        # When using the TE/NVDecoderLayer we ignore the 'position_embeddings' passed by in the model.forward().
        self.decoder_rotary_emb = RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads,
            rotary_base=config.rope_theta,
        )
        self.te_rope_emb = self.decoder_rotary_emb(max_seq_len=config.max_position_embeddings).cuda()

        # For the rotary_emb we just do a NoOp since we don't use its outputs.
        self.rotary_emb = NoOp(return_input=False)

        self.layers = nn.ModuleList(
            [
                NVLlamaDecoderLayer(config, self.te_rope_emb)
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        _keys_to_ignore_on_load_missing = kwargs.pop("_keys_to_ignore_on_load_missing", None)
        if _keys_to_ignore_on_load_missing:
            self.__class__._keys_to_ignore_on_load_missing = _keys_to_ignore_on_load_missing

        # Initialize weights and apply final processing
        self.post_init()


class NVLlamaForCausalLM(NVLlamaPreTrainedModel, LlamaForCausalLM):
    def __init__(self, config, **kwargs):
        _keys_to_ignore_on_load_missing = kwargs.pop("_keys_to_ignore_on_load_missing", None)
        if _keys_to_ignore_on_load_missing:
            self.__class__._keys_to_ignore_on_load_missing = _keys_to_ignore_on_load_missing

        NVLlamaPreTrainedModel.__init__(self, config)
        self.model = NVLlamaModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


# Used to know which HF model we should instantiate to get the appropriate 'state_dict'.
# The second element is the pattern used to rename 'state_dict' keys to make it NV compatible.
NV_TO_HF_CLASS_MAP = {
    NVLlamaModel: (LlamaModel, "layers.\d+."),
    NVLlamaForCausalLM: (LlamaForCausalLM, "model.layers.\d+.")
}
