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

"""Reference: ESM2 TransformerEngine Model (Encoder/BERT-like pattern).

This file shows how to wrap TransformerEngine layers in a HuggingFace-compatible
encoder model. Key patterns:
- NVEsmConfig extends EsmConfig with TE-specific fields
- NVEsmEncoder uses te.TransformerLayer with fused LN+QKV and LN+MLP
- NVEsmPreTrainedModel handles meta-device init, _init_weights, state_dict filtering
- LM head runs in higher precision (autocast disabled) for numerical stability
- AUTO_MAP dict enables AutoModel.from_pretrained() compatibility
"""

from contextlib import nullcontext
from typing import ClassVar, ContextManager, Literal, Optional, Unpack

import torch
import transformer_engine.common.recipe
import transformer_engine.pytorch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import EsmPooler, EsmPreTrainedModel
from transformers.utils.generic import TransformersKwargs


# NOTE: AUTO_MAP tells HuggingFace's Auto** classes which module contains our model.
# The prefix (e.g., "esm_nv.") must match the filename used in the exported checkpoint.
AUTO_MAP = {
    "AutoConfig": "esm_nv.NVEsmConfig",
    "AutoModel": "esm_nv.NVEsmModel",
    "AutoModelForMaskedLM": "esm_nv.NVEsmForMaskedLM",
    "AutoModelForTokenClassification": "esm_nv.NVEsmForTokenClassification",
}


class NVEsmConfig(EsmConfig):
    """Extended ESM config with TransformerEngine-specific fields."""

    model_type: str = "nv_esm"

    def __init__(
        self,
        # NOTE: These are the key TE-specific config fields to add for any encoder model
        qkv_weight_interleaved: bool = True,
        encoder_activation: str = "gelu",
        attn_input_format: Literal["bshd", "thd"] = "bshd",
        fuse_qkv_params: bool = True,
        micro_batch_size: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        padded_vocab_size: Optional[int] = 64,
        attn_mask_type: str = "padding",
        layer_precision: list[str | None] | None = None,
        use_quantized_model_init: bool = False,
        **kwargs,
    ):
        """Initialize NVEsmConfig.

        Args:
            qkv_weight_interleaved: If True, QKV weight is interleaved per-head [q0,k0,v0,q1,...].
                If False, it's concatenated [Q,K,V]. Must match the conversion code.
            encoder_activation: Activation function ("gelu", "swiglu", etc.).
            attn_input_format: "bshd" for padded batches, "thd" for packed sequences.
            fuse_qkv_params: Expose single fused QKV parameter (enables QKV fusion).
            micro_batch_size: Micro batch size for JIT warmup.
            max_seq_length: Max sequence length for JIT warmup.
            padded_vocab_size: Pad embedding to this size for FP8 alignment.
            attn_mask_type: Attention mask type ("padding" for encoder).
            layer_precision: Per-layer quantization: "fp8", "fp4", or None per layer.
            use_quantized_model_init: Use quantized_model_init context during construction.
            **kwargs: Additional config options passed to EsmConfig.
        """
        super().__init__(**kwargs)
        self.qkv_weight_interleaved = qkv_weight_interleaved
        self.encoder_activation = encoder_activation
        self.attn_input_format = attn_input_format
        self.fuse_qkv_params = fuse_qkv_params
        self.micro_batch_size = micro_batch_size
        self.max_seq_length = max_seq_length
        self.attn_mask_type = attn_mask_type
        self.layer_precision = layer_precision
        self.use_quantized_model_init = use_quantized_model_init

        # NOTE: padded_vocab_size must be >= vocab_size, used for FP8 alignment
        self.padded_vocab_size = padded_vocab_size or self.vocab_size
        if self.padded_vocab_size is not None and self.vocab_size is not None:
            assert self.padded_vocab_size >= self.vocab_size


class NVEsmEncoder(nn.Module):
    """TransformerEngine-optimized ESM encoder stack."""

    def __init__(
        self,
        config: NVEsmConfig,
        fp8_recipe: transformer_engine.common.recipe.Recipe | None = None,
        fp4_recipe: transformer_engine.common.recipe.Recipe | None = None,
    ):
        """Initialize NVEsmEncoder.

        Args:
            config: The model configuration.
            fp8_recipe: FP8 recipe for the encoder.
            fp4_recipe: FP4 recipe for the encoder.
        """
        super().__init__()
        self.config = config
        self._fp8_recipe = fp8_recipe
        self._fp4_recipe = fp4_recipe

        # NOTE: Default layer_precision from recipe if not explicitly set
        if self.config.layer_precision is None and fp8_recipe is not None:
            self.config.layer_precision = ["fp8"] * self.config.num_hidden_layers

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        # NOTE: Each layer is created inside get_autocast_context for proper FP8/FP4 init
        layers = []
        for i in range(config.num_hidden_layers):
            with self.get_autocast_context(i, init=True):
                layers.append(
                    transformer_engine.pytorch.TransformerLayer(
                        hidden_size=config.hidden_size,
                        ffn_hidden_size=config.intermediate_size,
                        num_attention_heads=config.num_attention_heads,
                        layernorm_epsilon=config.layer_norm_eps,
                        hidden_dropout=config.hidden_dropout_prob,
                        attention_dropout=config.attention_probs_dropout_prob,
                        qkv_weight_interleaved=config.qkv_weight_interleaved,
                        layer_number=i + 1,
                        layer_type="encoder",
                        self_attn_mask_type=config.attn_mask_type,
                        activation=config.encoder_activation,
                        attn_input_format=config.attn_input_format,
                        seq_length=config.max_seq_length,
                        micro_batch_size=config.micro_batch_size,
                        # NOTE: For MHA (not GQA), num_gqa_groups == num_attention_heads
                        num_gqa_groups=config.num_attention_heads,
                        fuse_qkv_params=config.fuse_qkv_params,
                        params_dtype=config.dtype,
                        device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                        init_method=_init_method,
                        output_layer_init_method=_init_method,
                    )
                )

        self.layers = nn.ModuleList(layers)

        # NOTE: Post-encoder LayerNorm (not fused into any TransformerLayer)
        self.emb_layer_norm_after = transformer_engine.pytorch.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            params_dtype=config.dtype,
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
        )
        if config.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)

    def forward(self, hidden_states, attention_mask=None, **kwargs: Unpack[TransformersKwargs]):
        """Forward pass through encoder stack."""
        all_hidden_states = ()

        if self.config.attn_input_format == "thd" and hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)

        # NOTE: Rotary embeddings must be computed in higher precision
        with torch.autocast(device_type="cuda", enabled=False):
            te_rope_emb = self.rotary_embeddings(max_seq_len=self.config.max_position_embeddings)
            te_rope_emb = te_rope_emb.to(hidden_states.device, non_blocking=True)

        with self.get_autocast_context(None, outer=True):
            for layer_idx, layer_module in enumerate(self.layers):
                if kwargs.get("output_hidden_states", False):
                    all_hidden_states = (*all_hidden_states, hidden_states)
                with self.get_autocast_context(layer_idx):
                    hidden_states = layer_module(
                        hidden_states,
                        attention_mask,
                        rotary_pos_emb=te_rope_emb,
                        cu_seqlens_q=kwargs.get("cu_seq_lens_q", None),
                        cu_seqlens_kv=kwargs.get("cu_seq_lens_k", None),
                        max_seqlen_q=kwargs.get("max_length_q", None),
                        max_seqlen_kv=kwargs.get("max_length_k", None),
                    )

        hidden_states = self.emb_layer_norm_after(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states or None)

    def get_autocast_context(
        self, layer_number: int | None, init: bool = False, outer: bool = False
    ) -> ContextManager:
        """Return the appropriate TE autocast context for a given layer.

        NOTE: This handles three cases:
        - init=True: Return quantized_model_init context for layer construction
        - outer=True: Return global te.autocast wrapping the entire encoder forward
        - Otherwise: Return per-layer te.autocast based on layer_precision
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


class NVEsmPreTrainedModel(EsmPreTrainedModel):
    """Base class handling TE-specific weight init, meta device support, and state_dict filtering."""

    config_class = NVEsmConfig
    base_model_prefix = "esm"
    supports_gradient_checkpointing = False
    _no_split_modules = ("TransformerLayer", "EsmEmbeddings")

    def init_empty_weights(self):
        """Move model from meta device to CUDA and initialize weights.

        NOTE: For TE layers, reset_parameters() handles both device placement
        and weight initialization. For non-TE layers (embeddings), use standard init.
        """
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.esm.embeddings.word_embeddings.to_empty(device="cuda")
        self.esm.embeddings.apply(self._init_weights)
        self.tie_weights()

    def _init_weights(self, module):
        """Initialize weights, skipping TE modules which handle their own init.

        NOTE: Must skip TE modules because the default HF _init_weights assumes any class
        with 'LayerNorm' in the name should have weights=1.0, which breaks
        LayerNormLinear/LayerNormMLP that use 'weight' for the linear part.
        """
        if module.__module__.startswith("transformer_engine.pytorch"):
            if hasattr(module, "reset_parameters") and not getattr(module, "primary_weights_in_fp8", False):
                module.reset_parameters()
            return
        super()._init_weights(module)

    def state_dict(self, *args, **kwargs):
        """Filter out TE's _extra_state keys for HF compatibility.

        NOTE: TE layers add _extra_state attributes that break HF model loading.
        """
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state_dict.items() if not k.endswith("_extra_state")}


class NVEsmForMaskedLM(NVEsmPreTrainedModel):
    """ESM2 masked language model with TransformerEngine encoder."""

    # NOTE: Tied weights - decoder weight points to embedding weight
    _tied_weights_keys: ClassVar[dict[str, str]] = {"lm_head.decoder.weight": "esm.embeddings.word_embeddings.weight"}

    def __init__(
        self,
        config: NVEsmConfig,
        fp8_recipe=None,
        fp4_recipe=None,
    ):
        """Initialize NVEsmForMaskedLM.

        Args:
            config: The model configuration.
            fp8_recipe: FP8 recipe for the encoder.
            fp4_recipe: FP4 recipe for the encoder.
        """
        super().__init__(config)
        self.esm = NVEsmModel(config, add_pooling_layer=False, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
        self.lm_head = NVEsmLMHead(config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Forward pass for masked language modeling."""
        outputs = self.esm(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]

        # NOTE: LM head runs with autocast DISABLED for numerical stability
        with transformer_engine.pytorch.autocast(enabled=False):
            prediction_scores = self.lm_head(sequence_output)

        # NOTE: Truncate logits back to original vocab_size (remove FP8 padding)
        if self.config.padded_vocab_size != self.config.vocab_size:
            prediction_scores = prediction_scores[..., : self.config.vocab_size]

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss()(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.to(prediction_scores.device).view(-1),
            )

        return MaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=outputs.hidden_states)


class NVEsmLMHead(nn.Module):
    """LM head: Linear -> GELU -> LayerNormLinear (to vocab).

    NOTE: Uses quantized_model_init(enabled=False) to ensure LM head stays in
    higher precision even when the rest of the model uses FP8/FP4.
    """

    def __init__(self, config: NVEsmConfig):
        """Initialize NVEsmLMHead.

        Args:
            config: The model configuration.
        """
        super().__init__()
        with transformer_engine.pytorch.quantized_model_init(enabled=False):
            self.dense = transformer_engine.pytorch.Linear(
                config.hidden_size,
                config.hidden_size,
                params_dtype=config.dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
            )
            self.decoder = transformer_engine.pytorch.LayerNormLinear(
                config.hidden_size,
                config.padded_vocab_size or config.vocab_size,
                bias=True,
                eps=config.layer_norm_eps,
                params_dtype=config.dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
            )

    def forward(self, features, **kwargs):
        """Forward pass through dense + gelu + decoder projection."""
        # NOTE: Keep LM head in higher precision to avoid numerical instability
        with transformer_engine.pytorch.autocast(enabled=False):
            x = self.dense(features)
            x = torch.nn.functional.gelu(x)
            x = self.decoder(x)
        return x


class NVEsmModel(NVEsmPreTrainedModel):
    """ESM encoder-only model with TE-optimized layers."""

    def __init__(self, config, add_pooling_layer=True, fp8_recipe=None, fp4_recipe=None):
        """Initialize NVEsmModel.

        Args:
            config: The model configuration.
            add_pooling_layer: Whether to add a pooling layer.
            fp8_recipe: FP8 recipe for the encoder.
            fp4_recipe: FP4 recipe for the encoder.
        """
        super().__init__(config)
        self.config = config
        self.embeddings = NVEsmEmbeddings(config)
        self.encoder = NVEsmEncoder(config, fp8_recipe, fp4_recipe)
        self.pooler = EsmPooler(config) if add_pooling_layer else None
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        """Forward pass through the ESM model."""
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        # NOTE: TE expects boolean mask where True=masked, opposite of HF convention
        extended_attention_mask = extended_attention_mask < -1

        embedding_output = self.embeddings(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None if self.config.attn_input_format == "thd" else extended_attention_mask,
            **kwargs,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class NVEsmEmbeddings(nn.Module):
    """Embedding layer with padded vocab size for FP8 compatibility."""

    def __init__(self, config):
        """Initialize NVEsmEmbeddings.

        Args:
            config: The model configuration.
        """
        super().__init__()
        # NOTE: Use padded_vocab_size for the embedding table
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size, config.hidden_size, padding_idx=config.pad_token_id, dtype=config.dtype
        )
        self.layer_norm = (
            transformer_engine.pytorch.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
                params_dtype=config.dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
            )
            if config.emb_layer_norm_before
            else None
        )
        if config.position_embedding_type != "rotary":
            raise ValueError("TE ESM-2 only supports rotary position embeddings")

        self.padding_idx = config.pad_token_id
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        """Compute token embeddings."""
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        return embeddings
