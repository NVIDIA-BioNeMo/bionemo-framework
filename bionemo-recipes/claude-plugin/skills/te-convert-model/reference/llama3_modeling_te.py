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

"""Reference: Llama3 TransformerEngine Model (Decoder/Causal LM pattern).

This file shows how to wrap TransformerEngine layers in a HuggingFace-compatible
decoder model. Key patterns:
- NVLlamaConfig extends LlamaConfig with minimal TE fields
- NVLlamaModel uses te.TransformerLayer with RMSNorm, SwiGLU, GQA
- Automatic BSHD->THD conversion for efficient packed-sequence processing
- NVLlamaForCausalLM with GenerationMixin for text generation support
- LM head in higher precision (autocast disabled)
- AUTO_MAP dict for AutoModelForCausalLM.from_pretrained() compatibility
"""

from contextlib import nullcontext
from typing import ClassVar, ContextManager, Unpack

import torch
import torch.nn as nn
import transformer_engine.common.recipe
import transformer_engine.pytorch
import transformers
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import LlamaConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.utils.generic import TransformersKwargs


# NOTE: AUTO_MAP keys must match the model classes you want Auto** to resolve to.
# The value prefix must match the filename used in the exported checkpoint.
AUTO_MAP = {
    "AutoConfig": "modeling_llama_te.NVLlamaConfig",
    "AutoModel": "modeling_llama_te.NVLlamaModel",
    "AutoModelForCausalLM": "modeling_llama_te.NVLlamaForCausalLM",
}


class NVLlamaConfig(LlamaConfig):
    """Extended Llama config with TE-specific fields.

    NOTE: Decoder models need fewer TE-specific fields than encoder models because
    TE's TransformerLayer handles most config via constructor args directly.
    """

    # NOTE: "thd" is preferred for decoders - enables packed sequence processing
    attn_input_format: str = "thd"
    self_attn_mask_type: str = "padding_causal"

    def __init__(
        self,
        layer_precision: list[str | None] | None = None,
        use_quantized_model_init: bool = False,
        **kwargs,
    ):
        """Initialize NVLlamaConfig.

        Args:
            layer_precision: Per-layer quantization precision list.
            use_quantized_model_init: Use quantized_model_init for layer init.
            **kwargs: Additional config options passed to LlamaConfig.
        """
        super().__init__(**kwargs)
        self.layer_precision = layer_precision
        self.use_quantized_model_init = use_quantized_model_init


class NVLlamaPreTrainedModel(PreTrainedModel):
    """Base class for NVLlama models."""

    config_class = NVLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ("TransformerLayer",)
    _skip_keys_device_placement = ("past_key_values",)

    def init_empty_weights(self):
        """Move model from meta device to CUDA and initialize weights.

        NOTE: TE layers use reset_parameters(). Non-TE layers (embed_tokens)
        use standard HF init. Rotary embeddings are recomputed fresh.
        """
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.model.embed_tokens.to_empty(device="cuda")
        self.model.embed_tokens.apply(self._init_weights)
        self.model.rotary_emb.inv_freq = LlamaRotaryEmbedding(config=self.model.config).inv_freq.to("cuda")
        self.tie_weights()

    def _init_weights(self, module):
        """Skip TE modules (they handle their own init via reset_parameters)."""
        if module.__module__.startswith("transformer_engine.pytorch"):
            return
        super()._init_weights(module)

    def state_dict(self, *args, **kwargs):
        """Filter out TE's _extra_state keys for HF compatibility."""
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state_dict.items() if not k.endswith("_extra_state")}


class NVLlamaModel(NVLlamaPreTrainedModel):
    """Llama3 decoder model with TransformerEngine layers."""

    def __init__(
        self,
        config: LlamaConfig,
        fp8_recipe: transformer_engine.common.recipe.Recipe | None = None,
        fp4_recipe: transformer_engine.common.recipe.Recipe | None = None,
    ):
        """Initialize NVLlamaModel.

        Args:
            config: The model configuration.
            fp8_recipe: FP8 recipe for the model.
            fp4_recipe: FP4 recipe for the model.
        """
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self._fp8_recipe = fp8_recipe
        self._fp4_recipe = fp4_recipe

        # NOTE: Default layer_precision from recipe if not explicitly set
        if self.config.layer_precision is None and fp8_recipe is not None:
            self.config.layer_precision = ["fp8"] * self.config.num_hidden_layers

        # NOTE: Embedding is standard nn.Embedding (no padding needed for decoder models)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.dtype)

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        # NOTE: Key TransformerLayer differences from encoder:
        # - bias=False (Llama uses no bias)
        # - normalization="RMSNorm" (not LayerNorm)
        # - activation="swiglu" (gated activation)
        # - num_gqa_groups=num_key_value_heads (GQA, not MHA)
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            with self.get_autocast_context(layer_idx, init=True):
                layers.append(
                    transformer_engine.pytorch.TransformerLayer(
                        hidden_size=config.hidden_size,
                        ffn_hidden_size=config.intermediate_size,
                        num_attention_heads=config.num_attention_heads,
                        bias=False,
                        layernorm_epsilon=config.rms_norm_eps,
                        hidden_dropout=0,
                        attention_dropout=0,
                        fuse_qkv_params=True,
                        qkv_weight_interleaved=True,
                        normalization="RMSNorm",
                        activation="swiglu",
                        attn_input_format=config.attn_input_format,
                        self_attn_mask_type=config.self_attn_mask_type,
                        num_gqa_groups=config.num_key_value_heads,
                        layer_number=layer_idx + 1,
                        params_dtype=config.dtype,
                        device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                        init_method=_init_method,
                        output_layer_init_method=_init_method,
                    )
                )

        self.layers = nn.ModuleList(layers)
        self.norm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
        )

        # NOTE: Use TE's RotaryPositionEmbedding but with HF's inv_freq for compatibility
        self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        self.rotary_emb.inv_freq = LlamaRotaryEmbedding(config=config).inv_freq

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Forward pass with BSHD to THD dynamic conversion."""
        all_hidden_states = []
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # NOTE: Auto-convert BSHD -> THD for efficient packed-sequence processing.
        # This enables HF-style generation (which provides BSHD inputs) to work
        # transparently with TE's THD-optimized attention.
        has_thd_input = [x in kwargs for x in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]]
        should_pack_inputs = not any(has_thd_input) and self.config.attn_input_format == "thd"

        if should_pack_inputs:
            assert attention_mask is not None
            batch_size = hidden_states.size(0)
            padded_seq_len = input_ids.size(1)
            hidden_states, indices, cu_seqlens, max_seqlen, _ = _unpad_input(hidden_states, attention_mask)
            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens
            kwargs["max_length_q"] = kwargs["max_length_k"] = max_seqlen

        if self.config.attn_input_format == "thd" and hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)

        if self.config.attn_input_format == "bshd" and attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = ~attention_mask[:, None, None, :].bool()

        # NOTE: Rotary embeddings in higher precision
        with torch.autocast(device_type="cuda", enabled=False):
            te_rope_emb = self.rotary_emb(max_seq_len=self.config.max_position_embeddings)

        with self.get_autocast_context(None, outer=True):
            for layer_idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = (*all_hidden_states, hidden_states)
                with self.get_autocast_context(layer_idx):
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=None if self.config.attn_input_format == "thd" else attention_mask,
                        rotary_pos_emb=te_rope_emb,
                        inference_params=past_key_values,
                        cu_seqlens_q=kwargs.get("cu_seq_lens_q", None),
                        cu_seqlens_kv=kwargs.get("cu_seq_lens_k", None),
                        max_seqlen_q=kwargs.get("max_length_q", None),
                        max_seqlen_kv=kwargs.get("max_length_k", None),
                    )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        # NOTE: Convert THD back to BSHD for HF-compatible output
        if should_pack_inputs:
            hidden_states = _pad_input(hidden_states, indices, batch_size, padded_seq_len)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )

    def get_autocast_context(
        self, layer_number: int | None, init: bool = False, outer: bool = False
    ) -> ContextManager:
        """Return appropriate TE autocast context for a given layer.

        Same pattern as encoder -- see NVEsmEncoder.get_autocast_context for details.
        """
        if self.config.layer_precision is None:
            return nullcontext()

        if outer:
            if "fp8" not in self.config.layer_precision:
                return nullcontext()
            return transformer_engine.pytorch.autocast(enabled=True, recipe=self._fp8_recipe)

        precision = self.config.layer_precision[layer_number]
        recipe = {"fp8": self._fp8_recipe, "fp4": self._fp4_recipe}.get(precision)

        if init and self.config.use_quantized_model_init:
            if precision in ("fp8", "fp4"):
                return transformer_engine.pytorch.quantized_model_init(recipe=recipe)
            return nullcontext()

        if precision == "fp8":
            return transformer_engine.pytorch.autocast(enabled=True, recipe=recipe)
        if precision == "fp4":
            if recipe is None:
                raise RuntimeError("No FP4 recipe provided, but layer precision is set to FP4.")
            return transformer_engine.pytorch.autocast(enabled=True, recipe=recipe)
        return transformer_engine.pytorch.autocast(enabled=False)


class NVLlamaForCausalLM(NVLlamaPreTrainedModel, transformers.GenerationMixin):
    """Llama3 causal LM with generation support.

    NOTE: Inherits GenerationMixin for HF generate() compatibility.
    """

    # NOTE: Tied weights - lm_head.weight points to embed_tokens.weight
    _tied_weights_keys: ClassVar[dict[str, str]] = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config, fp8_recipe=None, fp4_recipe=None):
        """Initialize NVLlamaForCausalLM.

        Args:
            config: The model configuration.
            fp8_recipe: FP8 recipe for the model.
            fp4_recipe: FP4 recipe for the model.
        """
        super().__init__(config)
        self.model = NVLlamaModel(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
        self.vocab_size = config.vocab_size

        # NOTE: LM head created with quantized_model_init DISABLED for numerical stability
        with transformer_engine.pytorch.quantized_model_init(enabled=False):
            self.lm_head = transformer_engine.pytorch.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                params_dtype=config.dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
            )
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        shift_labels=None,
        logits_to_keep=0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass for causal language modeling."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # NOTE: LM head with autocast DISABLED for numerical stability
        with transformer_engine.pytorch.autocast(enabled=False):
            if hidden_states.ndim == 3:
                logits = self.lm_head(hidden_states[:, slice_indices, :])
            else:  # THD format: batch and sequence collapsed in first dimension
                logits = self.lm_head(hidden_states[slice_indices, :])

        loss = None
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, shift_labels=shift_labels, vocab_size=self.config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


# --- Helper functions for BSHD <-> THD conversion ---

torch._dynamo.config.capture_scalar_outputs = True


@torch.compile
def _pad_input(hidden_states, indices, batch, seqlen):
    """Convert THD tensor back to BSHD format."""
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)


@torch.compile
def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Convert BSHD tensor to THD format by removing padding.

    Returns:
        hidden_states: (total_tokens, hidden_size)
        indices: indices of non-padding tokens in flattened input
        cu_seqlens: cumulative sequence lengths for TE
        max_seqlen: maximum sequence length in batch
        seqused: number of used tokens per sequence
    """
    batch_size = hidden_states.size(0)
    seq_length = hidden_states.size(1)

    if attention_mask.shape[1] != seq_length:  # Generation mode with kv-caching
        return (
            hidden_states.squeeze(1),
            torch.arange(batch_size, dtype=torch.int64, device=hidden_states.device),
            torch.arange(batch_size + 1, dtype=torch.int32, device=hidden_states.device),
            1,
            1,
        )

    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    return (
        hidden_states.reshape(-1, *hidden_states.shape[2:])[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )
