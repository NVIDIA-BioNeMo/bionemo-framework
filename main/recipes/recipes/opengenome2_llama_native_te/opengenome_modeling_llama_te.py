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

"""OpenGenome2 Llama model with TransformerEngine layers.

Extends the base NVLlama model (modeling_llama_te.py) with OG2-specific features:
- Megatron-style scaled initialization for residual output layers (proj/fc2)
- Spike-No-More embedding initialization (std=1.0)
- FP8 training with configurable first/last layer BF16 override
- RoPE theta fix for transformers >=5.0 compatibility

The base modeling_llama_te.py is kept as an exact CI-synced copy of models/llama3/modeling_llama_te.py.
This file defines OG2-specific config and model classes that train_fsdp2.py imports.
"""

import logging
import math
from collections import OrderedDict
from contextlib import nullcontext
from typing import ClassVar, Unpack

import torch
import torch.nn as nn
import transformer_engine.pytorch
import transformers
from transformer_engine.pytorch.attention import InferenceParams
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import LlamaConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.utils.generic import TransformersKwargs

# Shared utilities from the base modeling file (CI-synced with models/llama3)
from modeling_llama_te import HFInferenceParams as HFInferenceParams
from modeling_llama_te import _pad_input, _unpad_input


logger = logging.getLogger(__name__)


def _ensure_rope_theta(config):
    """Ensure config.rope_theta is set for LlamaRotaryEmbedding compatibility.

    Transformers >=5.0 may nullify rope_theta during config processing when rope_scaling is present.
    This function restores a sensible default (500000.0 for Llama-3) to prevent TypeError in
    LlamaRotaryEmbedding initialization.
    """
    if getattr(config, "rope_theta", None) is None:
        config.rope_theta = 500000.0
    return config


class NVLlamaConfig(LlamaConfig):
    """NVLlama configuration with OG2-specific features.

    Additional attributes:
        attn_input_format: Input format for attention ("thd" or "bshd").
        self_attn_mask_type: Attention mask type for self-attention.
        embedding_init_std: Standard deviation for embedding initialization.
            If None, uses initializer_range (typically 0.02).
            Set to 1.0 for Spike-No-More paper approach.
        use_megatron_scaled_init: Whether to use Megatron's scaled initialization
            for residual output layers (attention proj, MLP fc2).
            Scaled init uses std / sqrt(2 * num_layers) for these layers.
        fp8_first_last_bf16: When True, keeps first and last N transformer layers
            in bf16 for FP8 numerical stability. The lm_head is always kept in bf16.
        num_layers_at_start_in_bf16: Number of layers at the start to keep in BF16.
        num_layers_at_end_in_bf16: Number of layers at the end to keep in BF16.
    """

    attn_input_format: str = "thd"
    self_attn_mask_type: str = "padding_causal"
    embedding_init_std: float | None = None  # None means use initializer_range
    use_megatron_scaled_init: bool = False  # Use scaled init for proj/fc2 (std/sqrt(2*n))
    fp8_first_last_bf16: bool = False  # Keep first/last transformer layers in bf16 for FP8 stability
    num_layers_at_start_in_bf16: int = 1  # Number of layers at start to keep in BF16
    num_layers_at_end_in_bf16: int = 1  # Number of layers at end to keep in BF16


class NVLlamaPreTrainedModel(PreTrainedModel):
    """Base class for OG2 NVLlama models with custom initialization."""

    config_class = NVLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ("TransformerLayer",)
    _skip_keys_device_placement = ("past_key_values",)

    def state_dict(self, *args, **kwargs):
        """Override state_dict to filter out TransformerEngine's _extra_state keys.

        TransformerEngine layers add _extra_state attributes that are not compatible with
        standard PyTorch/HuggingFace model loading. These are filtered out to ensure
        checkpoints can be loaded with from_pretrained().
        """
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state_dict.items() if not k.endswith("_extra_state")}

    def init_empty_weights(self):
        """Move model from meta device to CUDA and initialize weights.

        Known issue: meta-device init breaks Megatron-style scaled initialization. The scaled std
        for proj/fc2 (std / sqrt(2*num_layers)) is not correctly applied when initializing from
        meta device -- TE's reset_parameters() does not use the output_layer_init_method passed
        during TransformerLayer construction. The manual fixup below attempts to re-apply scaled
        init, but does not produce the same distributions as direct CUDA initialization.
        Use use_meta_device=false when using use_megatron_scaled_init or spike_no_more_embedding_init.
        """
        # For TE layers, calling `reset_parameters` is sufficient to move them to the cuda device and apply the weight
        # initialization we passed them during module creation.
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        # The embed_tokens layer is the only non-TE layer in this model we need to deal with. We use
        # `model._init_weights` rather than `reset_parameters` to ensure we honor the original config standard
        # deviation.
        self.model.embed_tokens.to_empty(device="cuda")
        self.model.embed_tokens.apply(self._init_weights)

        self.model.rotary_emb.inv_freq = LlamaRotaryEmbedding(
            config=_ensure_rope_theta(self.model.config)
        ).inv_freq.to("cuda")

        # TE's reset_parameters() doesn't use output_layer_init_method for proj/fc2.
        # If use_megatron_scaled_init is enabled, we need to manually apply scaled init.
        use_scaled_init = getattr(self.config, "use_megatron_scaled_init", False)
        if use_scaled_init:
            std = getattr(self.config, "initializer_range", 0.02)
            num_layers = getattr(self.config, "num_hidden_layers", 32)
            output_std = std / math.sqrt(2.0 * num_layers)

            # Apply scaled init to attention proj and MLP fc2 in each TransformerLayer
            for layer in self.model.layers:
                # Attention output projection
                if hasattr(layer, "self_attention") and hasattr(layer.self_attention, "proj"):
                    proj = layer.self_attention.proj
                    if hasattr(proj, "weight") and proj.weight is not None:
                        proj.weight.data.normal_(mean=0.0, std=output_std)

                # MLP fc2 (output layer)
                if hasattr(layer, "layernorm_mlp") and hasattr(layer.layernorm_mlp, "fc2_weight"):
                    layer.layernorm_mlp.fc2_weight.data.normal_(mean=0.0, std=output_std)

        # Meta-device init seems to break weight tying, so we re-tie the weights here.
        self.tie_weights()

    def _init_weights(self, module):
        """Initialize module weights with OG2-specific initialization.

        Initialization strategy:
        - Embeddings: config.embedding_init_std if set, else initializer_range (0.02)
          - For Spike-No-More: set embedding_init_std=1.0 in config
        - QKV, fc1, LM head: regular init = initializer_range (0.02)
        - Attention proj, MLP fc2: depends on config.use_megatron_scaled_init:
          - If False (default): regular init = initializer_range (0.02)
          - If True: scaled init = initializer_range / sqrt(2 * num_layers)

        Args:
            module: The module to initialize the weights for.
        """
        # Get base std from config (typically 0.02)
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        # Check if we should use Megatron's scaled init for residual output layers
        use_scaled_init = getattr(self.config, "use_megatron_scaled_init", False)
        if use_scaled_init:
            num_layers = getattr(self.config, "num_hidden_layers", 32)
            output_std = std / math.sqrt(2.0 * num_layers)
        else:
            output_std = std

        # Embedding init std: default to regular std (0.02), use 1.0 only if explicitly set
        embedding_init_std = getattr(self.config, "embedding_init_std", None)
        embedding_std = embedding_init_std if embedding_init_std is not None else std

        # Embeddings
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=embedding_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        # Regular Linear layers: LM head and other Linear use regular std (0.02)
        if isinstance(
            module, (nn.Linear, transformer_engine.pytorch.Linear, transformer_engine.pytorch.LayerNormLinear)
        ):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, transformer_engine.pytorch.LayerNorm):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, transformer_engine.pytorch.RMSNorm):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
        if isinstance(module, transformer_engine.pytorch.LayerNormLinear):
            module.layer_norm_weight.data.fill_(1.0)
            if module.layer_norm_bias is not None:
                module.layer_norm_bias.data.zero_()

        # MLP: fc1 uses regular std, fc2 uses output_std (scaled if use_megatron_scaled_init=True)
        if isinstance(module, transformer_engine.pytorch.LayerNormMLP):
            module.layer_norm_weight.data.fill_(1.0)
            if hasattr(module, "fc1_weight") and module.fc1_weight is not None:
                module.fc1_weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "fc2_weight") and module.fc2_weight is not None:
                module.fc2_weight.data.normal_(mean=0.0, std=output_std)
            if hasattr(module, "fc1_bias") and module.fc1_bias is not None and module.fc1_bias.numel() > 0:
                module.fc1_bias.data.zero_()
            if hasattr(module, "fc2_bias") and module.fc2_bias is not None and module.fc2_bias.numel() > 0:
                module.fc2_bias.data.zero_()

        # TE TransformerLayer: attention output projection uses output_std
        if isinstance(module, transformer_engine.pytorch.TransformerLayer):
            if hasattr(module, "self_attention") and hasattr(module.self_attention, "proj"):
                proj = module.self_attention.proj
                if hasattr(proj, "weight") and proj.weight is not None:
                    proj.weight.data.normal_(mean=0.0, std=output_std)
                if hasattr(proj, "bias") and proj.bias is not None:
                    proj.bias.data.zero_()

        if isinstance(module, RotaryPositionEmbedding) and hasattr(module, "inv_freq"):
            module.inv_freq = LlamaRotaryEmbedding(config=_ensure_rope_theta(self.config)).inv_freq.to(
                module.inv_freq.device
            )


class NVLlamaModel(NVLlamaPreTrainedModel):
    """OpenGenome2 Llama3 model implemented in Transformer Engine."""

    def __init__(self, config: LlamaConfig):
        """Initialize the OG2 NVLlama model."""
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.dtype)

        # TE init_method for QKV, fc1 (standard initializer_range).
        # Scaled init for proj/fc2 is handled by _init_weights via post_init(), not by TE's
        # output_layer_init_method, because post_init() overwrites TE's initialization and
        # TE's reset_parameters() does not reliably apply output_layer_init_method anyway.
        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        self.layers = nn.ModuleList(
            [
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
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
        )

        # We use TE's RotaryPositionEmbedding, but we ensure that we use the same inv_freq as the original
        # LlamaRotaryEmbedding.
        self.rotary_emb = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        hf_rope = LlamaRotaryEmbedding(config=_ensure_rope_theta(config))
        self.rotary_emb.inv_freq = hf_rope.inv_freq

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: InferenceParams | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Forward pass for the OG2 NVLlama model.

        Args:
            input_ids: The input ids.
            attention_mask: The attention mask.
            position_ids: The position ids.
            past_key_values: The past key values.
            inputs_embeds: The inputs embeds.
            use_cache: Whether to use cache.
            **kwargs: Additional keyword arguments.

        Returns:
            BaseModelOutputWithPast: The output of the model.
        """
        all_hidden_states = []
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # TE-specific input handling.
        has_thd_input = [x in kwargs for x in ["cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"]]
        should_pack_inputs = not any(has_thd_input) and self.config.attn_input_format == "thd"

        if should_pack_inputs:
            assert attention_mask is not None, "Attention mask is required when packing BSHD inputs."
            batch_size = hidden_states.size(0)
            padded_seq_len = input_ids.size(1)
            hidden_states, indices, cu_seqlens, max_seqlen, _ = _unpad_input(hidden_states, attention_mask)
            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens
            kwargs["max_length_q"] = kwargs["max_length_k"] = max_seqlen

        if self.config.attn_input_format == "thd" and hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)

        if self.config.attn_input_format == "bshd" and attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = ~attention_mask[:, None, None, :].bool()

        if isinstance(past_key_values, InferenceParams):
            lengths = (
                attention_mask.sum(dim=1).tolist()
                if attention_mask.shape == input_ids.shape
                else [1] * input_ids.shape[0]
            )
            past_key_values.pre_step(OrderedDict(zip(list(range(len(lengths))), lengths)))

        # Ensure that rotary embeddings are computed at higher precision
        with torch.autocast(device_type="cuda", enabled=False):
            te_rope_emb = self.rotary_emb(max_seq_len=self.config.max_position_embeddings)
            assert te_rope_emb.dtype == torch.float32, "RoPE embeddings should be float32 for optimal performance"

        num_layers = self.config.num_hidden_layers
        for layer_idx, decoder_layer in enumerate(self.layers[:num_layers]):
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            # Optionally keep first and last N layers in bf16 for FP8 numerical stability
            num_start_bf16 = getattr(self.config, "num_layers_at_start_in_bf16", 1)
            num_end_bf16 = getattr(self.config, "num_layers_at_end_in_bf16", 1)
            use_bf16_for_layer = getattr(self.config, "fp8_first_last_bf16", False) and (
                layer_idx < num_start_bf16 or layer_idx >= num_layers - num_end_bf16
            )

            with transformer_engine.pytorch.autocast(enabled=False) if use_bf16_for_layer else nullcontext():
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=None if self.config.attn_input_format == "thd" else attention_mask,
                    rotary_pos_emb=te_rope_emb,
                    inference_params=past_key_values,
                    cu_seqlens_q=kwargs.get("cu_seq_lens_q", None),
                    cu_seqlens_kv=kwargs.get("cu_seq_lens_k", None),
                    cu_seqlens_q_padded=kwargs.get("cu_seq_lens_q_padded", None),
                    cu_seqlens_kv_padded=kwargs.get("cu_seq_lens_k_padded", None),
                    max_seqlen_q=kwargs.get("max_length_q", None),
                    max_seqlen_kv=kwargs.get("max_length_k", None),
                    pad_between_seqs=kwargs.get("pad_between_seqs", None),
                )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if should_pack_inputs:
            hidden_states = _pad_input(hidden_states, indices, batch_size, padded_seq_len)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class NVLlamaForCausalLM(NVLlamaPreTrainedModel, transformers.GenerationMixin):
    """OpenGenome2 Llama3 model with causal language head."""

    _tied_weights_keys: ClassVar[dict[str, str]] = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        """Initialize the OG2 NVLlamaForCausalLM model."""
        super().__init__(config)
        self.model = NVLlamaModel(config)
        self.vocab_size = config.vocab_size
        with transformer_engine.pytorch.quantized_model_init(enabled=False):
            self.lm_head = transformer_engine.pytorch.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                params_dtype=config.dtype,
                device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                init_method=lambda x: torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range),
            )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        shift_labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Forward pass for the OG2 NVLlamaForCausalLM model.

        Args:
            input_ids: The input ids.
            attention_mask: The attention mask.
            position_ids: The position ids.
            past_key_values: The past key values.
            inputs_embeds: The inputs embeds.
            labels: The labels.
            shift_labels: Labels that have already been shifted by the dataloader.
            use_cache: Whether to use cache.
            cache_position: The cache position.
            logits_to_keep: Whether to keep only the last logits.
            **kwargs: Additional keyword arguments.

        Returns:
            CausalLMOutputWithPast: The output of the model.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        with transformer_engine.pytorch.autocast(enabled=False):
            if hidden_states.ndim == 3:
                logits = self.lm_head(hidden_states[:, slice_indices, :])
            else:  # With THD inputs, batch and sequence dimensions are collapsed in the first dimension.
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
            attentions=outputs.attentions,
        )
