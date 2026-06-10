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

"""TransformerEngine-optimized CodonFM model for masked language modeling."""

from __future__ import annotations

import math
import warnings
from contextlib import nullcontext
from typing import ContextManager, Optional

import torch
import torch.nn as nn
import transformer_engine.common.recipe
import transformer_engine.pytorch
from torch.nn import CrossEntropyLoss
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput


class CodonFMConfig(PretrainedConfig):
    """Configuration for the CodonFM model."""

    model_type: str = "codonfm"

    def __init__(
        self,
        vocab_size: int = 69,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 3,
        mask_token_id: int = 4,
        max_position_embeddings: int = 2048,
        attn_input_format: str = "bshd",
        self_attn_mask_type: str = "padding",
        # TE-specific options
        qkv_weight_interleaved: bool = True,
        fuse_qkv_params: bool = True,
        # Layer-wise precision options
        layer_precision: list[str | None] | None = None,
        use_quantized_model_init: bool = False,
        **kwargs,
    ):
        """Initialize the CodonFMConfig.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            hidden_size: Dimensionality of the encoder layers.
            num_hidden_layers: Number of hidden layers in the encoder.
            num_attention_heads: Number of attention heads.
            intermediate_size: Dimensionality of the feed-forward layer.
            hidden_act: Activation function for the feed-forward layer.
            hidden_dropout_prob: Dropout probability for hidden layers.
            attention_probs_dropout_prob: Dropout probability for attention.
            initializer_range: Standard deviation for weight initialization.
            layer_norm_eps: Epsilon for layer normalization.
            pad_token_id: Token ID for padding.
            mask_token_id: Token ID for masking.
            max_position_embeddings: Maximum sequence length.
            attn_input_format: Attention input format ("bshd" or "thd").
            self_attn_mask_type: Self-attention mask type for TE TransformerLayer.
            qkv_weight_interleaved: Whether to interleave QKV weights.
            fuse_qkv_params: Whether to fuse QKV parameters.
            layer_precision: Per-layer quantization precision list.
            use_quantized_model_init: Whether to use quantized model init.
            **kwargs: Additional config options passed to PretrainedConfig.
        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.mask_token_id = mask_token_id
        self.max_position_embeddings = max_position_embeddings
        self.attn_input_format = attn_input_format
        self.self_attn_mask_type = self_attn_mask_type
        self.qkv_weight_interleaved = qkv_weight_interleaved
        self.fuse_qkv_params = fuse_qkv_params
        self.layer_precision = layer_precision
        self.use_quantized_model_init = use_quantized_model_init

        # Validation
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.hidden_act not in ("gelu", "relu", "silu"):
            raise ValueError(f"hidden_act must be one of: gelu, relu, silu, got {self.hidden_act}")
        if self.layer_precision is not None:
            if len(self.layer_precision) != self.num_hidden_layers:
                raise ValueError(
                    f"layer_precision must be a list of length {self.num_hidden_layers}, "
                    f"got {len(self.layer_precision)}"
                )
            for precision in self.layer_precision:
                if precision not in {"fp8", "fp4", None}:
                    raise ValueError(f'layer_precision element must be "fp8", "fp4", or None, got {precision!r}')


MODEL_PRESETS: dict[str, dict] = {
    "encodon_200k": {
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
    },
    "encodon_80m": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
    },
    "encodon_600m": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 12,
    },
    "encodon_1b": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 18,
    },
    "encodon_5b": {
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
    },
}


class CodonEmbedding(nn.Module):
    """Codon embedding layer with post-embedding LayerNorm and dropout."""

    def __init__(self, config: CodonFMConfig):
        """Initialize the CodonEmbedding.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.post_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_length].

        Returns:
            Embeddings of shape [batch_size, seq_length, hidden_size].
        """
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.post_ln(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CodonFMEncoder(nn.Module):
    """CodonFM encoder using standard TransformerEngine TransformerLayer."""

    def __init__(
        self,
        config: CodonFMConfig,
        fp8_recipe: transformer_engine.common.recipe.Recipe | None = None,
        fp4_recipe: transformer_engine.common.recipe.Recipe | None = None,
    ):
        """Initialize the encoder.

        Args:
            config: Model configuration.
            fp8_recipe: The FP8 recipe for the encoder.
            fp4_recipe: The FP4 recipe for the encoder.
        """
        super().__init__()
        self.config = config
        self._fp8_recipe: transformer_engine.common.recipe.Recipe | None = fp8_recipe
        self._fp4_recipe: transformer_engine.common.recipe.Recipe | None = fp4_recipe

        if self.config.layer_precision is None:
            if fp8_recipe is not None and fp4_recipe is not None:
                raise RuntimeError("Both FP8 and FP4 recipes provided, but no layer precision provided.")
            if fp8_recipe is not None:
                warnings.warn("No layer precision provided, using FP8 recipe for all layers.", UserWarning)
                self.config.layer_precision = ["fp8"] * self.config.num_hidden_layers
            elif fp4_recipe is not None:
                raise RuntimeError(
                    "FP4 recipe provided but no layer_precision configured. "
                    "Set layer_precision explicitly when using FP4."
                )

        if self.config.layer_precision is not None and "fp4" in self.config.layer_precision and fp4_recipe is None:
            raise RuntimeError("layer_precision contains 'fp4' entries but no fp4_recipe was provided.")

        device = "meta" if torch.get_default_device() == torch.device("meta") else "cuda"

        layers: list[transformer_engine.pytorch.TransformerLayer] = []
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
                        self_attn_mask_type=config.self_attn_mask_type,
                        activation=config.hidden_act,
                        attn_input_format=config.attn_input_format,
                        seq_length=config.max_position_embeddings,
                        num_gqa_groups=config.num_attention_heads,
                        fuse_qkv_params=config.fuse_qkv_params,
                        window_size=(-1, -1),
                        device=device,
                    )
                )

        self.layers = nn.ModuleList(layers)
        self.rotary_embeddings = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)

    def get_autocast_context(
        self, layer_number: int | None, init: bool = False, outer: bool = False
    ) -> ContextManager:
        """Return the appropriate TE autocast context manager for a given layer.

        Handles both the quantized_model_init during layer creation and the te.autocast() during forward.

        Args:
            layer_number: The 0-indexed layer number.
            init: Whether to return a ``quantized_model_init`` context for layer initialization.
            outer: Whether to return a global te.autocast() context to wrap the entire encoder stack.
        """
        if self.config.layer_precision is None:
            return nullcontext()

        if outer:
            if "fp8" not in self.config.layer_precision:
                return nullcontext()
            if self._fp8_recipe is None:
                warnings.warn("No FP8 recipe provided, using default recipe.", UserWarning)
            return transformer_engine.pytorch.autocast(enabled=True, recipe=self._fp8_recipe)

        precision = self.config.layer_precision[layer_number]
        recipe = {"fp8": self._fp8_recipe, "fp4": self._fp4_recipe}.get(precision)

        if init and self.config.use_quantized_model_init:
            if precision == "fp4" and recipe is None:
                raise RuntimeError("No FP4 recipe provided, but layer precision is set to FP4.")
            if precision in ("fp8", "fp4"):
                return transformer_engine.pytorch.quantized_model_init(recipe=recipe)
            return nullcontext()

        if precision == "fp8":
            if recipe is None:
                warnings.warn("No FP8 recipe provided, using default recipe.", UserWarning)
            return transformer_engine.pytorch.autocast(enabled=True, recipe=recipe)
        if precision == "fp4":
            if recipe is None:
                raise RuntimeError("No FP4 recipe provided, but layer precision is set to FP4.")
            return transformer_engine.pytorch.autocast(enabled=True, recipe=recipe)
        return transformer_engine.pytorch.autocast(enabled=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Forward pass.

        Args:
            hidden_states: Input tensor.
            attention_mask: Attention mask for BSHD format.
            output_hidden_states: Whether to return all hidden states.
            **kwargs: Additional arguments (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k for THD).

        Returns:
            Tuple of (hidden_states, all_hidden_states or None).
        """
        if self.config.attn_input_format == "thd" and hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)

        all_hidden_states: tuple[torch.Tensor, ...] = ()

        with torch.autocast(device_type="cuda", enabled=False):
            te_rope_emb = self.rotary_embeddings(max_seq_len=self.config.max_position_embeddings)
            te_rope_emb = te_rope_emb.to(hidden_states.device, non_blocking=True)

        with self.get_autocast_context(None, outer=True):
            for layer_idx, layer_module in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = (*all_hidden_states, hidden_states)

                with self.get_autocast_context(layer_idx):
                    if self.config.attn_input_format == "bshd":
                        hidden_states = layer_module(
                            hidden_states,
                            attention_mask=attention_mask,
                            rotary_pos_emb=te_rope_emb,
                        )
                    else:
                        hidden_states = layer_module(
                            hidden_states,
                            attention_mask=None,
                            rotary_pos_emb=te_rope_emb,
                            cu_seqlens_q=kwargs.get("cu_seq_lens_q"),
                            cu_seqlens_kv=kwargs.get("cu_seq_lens_k"),
                            max_seqlen_q=kwargs.get("max_length_q"),
                            max_seqlen_kv=kwargs.get("max_length_k"),
                        )

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return hidden_states, all_hidden_states or None


class CodonFMLMHead(nn.Module):
    """Prediction head for masked language modeling."""

    def __init__(self, config: CodonFMConfig):
        """Initialize the LM head.

        Args:
            config: Model configuration.
        """
        super().__init__()
        device = "meta" if torch.get_default_device() == torch.device("meta") else "cuda"
        _act_fns = {
            "gelu": torch.nn.functional.gelu,
            "relu": torch.nn.functional.relu,
            "silu": torch.nn.functional.silu,
        }
        self.activation_fn = _act_fns[config.hidden_act]

        # Disable quantization for the LM head to avoid numerical instability.
        with transformer_engine.pytorch.quantized_model_init(enabled=False):
            self.dense = transformer_engine.pytorch.Linear(
                config.hidden_size,
                config.hidden_size,
                device=device,
            )
            self.layer_norm_linear = transformer_engine.pytorch.LayerNormLinear(
                config.hidden_size,
                config.vocab_size,
                bias=True,
                eps=config.layer_norm_eps,
                device=device,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Encoder output.

        Returns:
            Logits of shape [..., vocab_size].
        """
        # Keep the LM head in higher precision to avoid numerical instability.
        with transformer_engine.pytorch.autocast(enabled=False):
            x = self.dense(hidden_states)
            x = self.activation_fn(x)
            x = self.layer_norm_linear(x)
        return x


class CodonFMPreTrainedModel(PreTrainedModel):
    """Base class for CodonFM models, handling weight initialization and pretrained model loading."""

    config_class = CodonFMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ("TransformerLayer",)

    def init_empty_weights(self):
        """Move model from meta device to CUDA and initialize weights."""
        # TE layers handle their own meta -> CUDA transition via reset_parameters.
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        # Move the entire embeddings module (word_embeddings, post_ln, dropout) from meta to CUDA.
        self.embeddings.to_empty(device="cuda")
        self._magneto_init_weights()

    def _init_weights(self, module):
        """Initialize module weights (called by HuggingFace's init machinery).

        We skip TE modules here since they handle their own initialization via reset_parameters.
        """
        if module.__module__.startswith("transformer_engine.pytorch"):
            if hasattr(module, "reset_parameters") and not getattr(module, "primary_weights_in_fp8", False):
                module.reset_parameters()
            return

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)

    def _magneto_init_weights(self):
        """Initialize weights using MAGNETO initialization scheme.

        This applies xavier_normal with scaled gain to all linear layers
        (except Q/K which use gain=1.0), and standard init to embeddings/LayerNorms.
        """
        scale_factor = math.sqrt(math.log(2 * self.config.num_hidden_layers))
        gain = self.config.initializer_range * scale_factor

        for name, module in self.named_modules():
            # Skip TE modules with FP8 primary weights — xavier_normal_ is incompatible with QuantizedTensor.
            if getattr(module, "primary_weights_in_fp8", False):
                continue
            if isinstance(module, (nn.Linear, transformer_engine.pytorch.Linear)):
                is_qk = "query" in name or "key" in name or "qkv" in name
                w_gain = 1.0 if is_qk else gain
                if getattr(module, "weight", None) is not None:
                    nn.init.xavier_normal_(module.weight, gain=w_gain)
                    if getattr(module, "bias", None) is not None:
                        module.bias.data.zero_()
                if getattr(module, "query_weight", None) is not None:
                    nn.init.xavier_normal_(module.query_weight, gain=w_gain)
                    if getattr(module, "query_bias", None) is not None:
                        module.query_bias.data.zero_()
                if getattr(module, "key_weight", None) is not None:
                    nn.init.xavier_normal_(module.key_weight, gain=w_gain)
                    if getattr(module, "key_bias", None) is not None:
                        module.key_bias.data.zero_()
                if getattr(module, "value_weight", None) is not None:
                    nn.init.xavier_normal_(module.value_weight, gain=w_gain)
                    if getattr(module, "value_bias", None) is not None:
                        module.value_bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, (nn.LayerNorm, transformer_engine.pytorch.LayerNorm)):
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.zero_()
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.data.fill_(1.0)

            if isinstance(module, transformer_engine.pytorch.LayerNormLinear):
                if hasattr(module, "layer_norm_weight"):
                    module.layer_norm_weight.data.fill_(1.0)
                if hasattr(module, "layer_norm_bias") and module.layer_norm_bias is not None:
                    module.layer_norm_bias.data.zero_()

    def state_dict(self, *args, **kwargs):
        """Override state_dict to filter out non-loadable TE-specific keys."""
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.endswith("_extra_state") and not k.endswith(".inv_freq")}


class CodonFMForMaskedLM(CodonFMPreTrainedModel):
    """CodonFM model for masked language modeling with TransformerEngine layers."""

    # Patterns for modules that should NOT be quantized (kept in higher precision).
    _do_not_quantize = ("lm_head.dense", "lm_head.layer_norm_linear")

    def __init__(
        self,
        config: CodonFMConfig,
        fp8_recipe: transformer_engine.common.recipe.Recipe | None = None,
        fp4_recipe: transformer_engine.common.recipe.Recipe | None = None,
    ):
        """Initialize the model.

        Args:
            config: Model configuration.
            fp8_recipe: The FP8 recipe for the encoder.
            fp4_recipe: The FP4 recipe for the encoder.
        """
        super().__init__(config)
        self.config = config
        self.embeddings = CodonEmbedding(config)
        self.encoder = CodonFMEncoder(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
        self.lm_head = CodonFMLMHead(config)
        self.post_init()
        self._magneto_init_weights()

    def _get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Convert a 2D attention mask to a 4D boolean mask for TE.

        Args:
            attention_mask: Mask of shape [batch_size, seq_length].

        Returns:
            Boolean mask of shape [batch_size, 1, 1, seq_length] where True means masked.
        """
        extended = attention_mask[:, None, None, :]
        extended = extended.to(dtype=torch.float32)
        extended = (1.0 - extended) * torch.finfo(torch.float32).min
        return extended < -1

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> MaskedLMOutput:
        """Forward pass.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask (for BSHD format).
            labels: Labels for MLM loss computation. -100 for tokens to ignore.
            output_hidden_states: Whether to return all hidden states.
            **kwargs: Additional arguments for THD format.

        Returns:
            MaskedLMOutput with loss, logits, and hidden states.
        """
        hidden_states = self.embeddings(input_ids)

        extended_mask = None
        if self.config.attn_input_format == "bshd" and attention_mask is not None:
            extended_mask = self._get_extended_attention_mask(attention_mask)

        hidden_states, all_hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=all_hidden_states)
