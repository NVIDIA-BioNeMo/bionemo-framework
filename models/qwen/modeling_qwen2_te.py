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

"""TransformerEngine-optimized Qwen2 model."""

import warnings
from collections import OrderedDict
from contextlib import nullcontext
from typing import ClassVar, ContextManager, Unpack

import torch
import torch.nn as nn
import transformer_engine.common.recipe
import transformer_engine.pytorch
import transformers
from transformer_engine.pytorch.attention import InferenceParams
from transformer_engine.pytorch.attention.inference import PagedKVCacheManager
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers import PreTrainedModel, Qwen2Config
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.utils.generic import TransformersKwargs


AUTO_MAP = {
    "AutoConfig": "modeling_qwen2_te.NVQwen2Config",
    "AutoModel": "modeling_qwen2_te.NVQwen2Model",
    "AutoModelForCausalLM": "modeling_qwen2_te.NVQwen2ForCausalLM",
}


class NVQwen2Config(Qwen2Config):
    """NVQwen2 configuration."""

    # Attention input format:
    #   "bshd" = Batch, Sequence, Head, Dimension (standard padded format)
    #   "thd"  = Total tokens (packed/unpadded), Head, Dimension (sequence packing format)
    attn_input_format: str = "thd"
    self_attn_mask_type: str = "padding_causal"

    def __init__(
        self,
        layer_precision: list[str | None] | None = None,
        use_quantized_model_init: bool = False,
        **kwargs,
    ):
        """Initialize the NVQwen2Config with additional TE-related config options.

        Args:
            layer_precision: Per-layer quantization precision, a list of length ``num_hidden_layers``
                where each element is ``"fp8"``, ``"fp4"``, or ``None`` (BF16 fallback). ``None``
                (the default) means no quantization is configured.
            use_quantized_model_init: Whether to use `quantized_model_init` for layer initialization.
            **kwargs: Additional config options to pass to Qwen2Config.
        """
        super().__init__(**kwargs)
        self.layer_precision = layer_precision
        self.use_quantized_model_init = use_quantized_model_init

        if layer_precision is not None:
            if len(layer_precision) != self.num_hidden_layers:
                raise ValueError(f"layer_precision must be a list of length {self.num_hidden_layers}")
            for precision in layer_precision:
                if precision not in {"fp8", "fp4", None}:
                    raise ValueError(f'layer_precision element must be "fp8", "fp4", or None, got {precision!r}')


class NVQwen2PreTrainedModel(PreTrainedModel):
    """Base class for NVQwen2 models."""

    config_class = NVQwen2Config
    base_model_prefix = "model"
    _no_split_modules = ("TransformerLayer",)
    _skip_keys_device_placement = ("past_key_values",)

    def init_empty_weights(self):
        """Handles moving the model from the meta device to the cuda device and initializing the weights."""
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

        self.model.rotary_emb.inv_freq = Qwen2RotaryEmbedding(config=self.model.config).inv_freq.to("cuda")

        # Meta-device init seems to break weight tying, so we re-tie the weights here.
        self.tie_weights()

    def _init_weights(self, module):
        """Initialize module weights.

        We only use this method for standard pytorch modules, TE modules handle their own weight initialization through
        `init_method` parameters and the `reset_parameters` method.
        """
        if module.__module__.startswith("transformer_engine.pytorch"):
            # Notably, we need to avoid calling this method for TE modules, since the default _init_weights will assume
            # any class with `LayerNorm` in the name should have weights initialized to 1.0; breaking `LayerNormLinear`
            # and `LayerNormMLP` modules that use `weight` for the linear layer and `layer_norm_weight` for the layer
            # norm.
            return

        super()._init_weights(module)

    def state_dict(self, *args, **kwargs):
        """Override state_dict to filter out TransformerEngine's _extra_state keys.

        TransformerEngine layers add _extra_state attributes that are not compatible with
        standard PyTorch/HuggingFace model loading. These are filtered out to ensure
        checkpoints can be loaded with from_pretrained().
        """
        state_dict = super().state_dict(*args, **kwargs)
        # Filter out _extra_state keys which are TransformerEngine-specific and not loadable
        return {k: v for k, v in state_dict.items() if not k.endswith("_extra_state")}


class NVQwen2Model(NVQwen2PreTrainedModel):
    """Qwen2 model implemented in Transformer Engine."""

    def __init__(
        self,
        config: Qwen2Config,
        fp8_recipe: transformer_engine.common.recipe.Recipe | None = None,
        fp4_recipe: transformer_engine.common.recipe.Recipe | None = None,
    ):
        """Initialize the NVQwen2 model.

        Args:
            config: The configuration of the model.
            fp8_recipe: The FP8 recipe for the model.
            fp4_recipe: The FP4 recipe for the model.
        """
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
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

        head_dim = config.hidden_size // config.num_attention_heads

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=config.dtype)

        def _init_method(x):
            torch.nn.init.normal_(x, mean=0.0, std=config.initializer_range)

        layers: list[transformer_engine.pytorch.TransformerLayer] = []
        for layer_idx in range(config.num_hidden_layers):
            with self.get_autocast_context(layer_idx, init=True):
                layers += [
                    transformer_engine.pytorch.TransformerLayer(
                        hidden_size=config.hidden_size,
                        ffn_hidden_size=config.intermediate_size,
                        num_attention_heads=config.num_attention_heads,
                        bias=True,
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
                        kv_channels=head_dim,
                        window_size=(config.sliding_window, config.sliding_window)
                        if config.layer_types[layer_idx] == "sliding_attention" and config.sliding_window is not None
                        else None,
                        layer_number=layer_idx + 1,
                        params_dtype=config.dtype,
                        device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
                        init_method=_init_method,
                        output_layer_init_method=_init_method,
                    )
                ]

        self.layers = nn.ModuleList(layers)
        self.norm = transformer_engine.pytorch.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
            device="meta" if torch.get_default_device() == torch.device("meta") else "cuda",
        )

        # We use TE's RotaryPositionEmbedding, but we ensure that we use the same inv_freq as the original
        # Qwen2RotaryEmbedding.
        self.rotary_emb = RotaryPositionEmbedding(head_dim)
        self.rotary_emb.inv_freq = Qwen2RotaryEmbedding(config=config).inv_freq

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
        """Forward pass for the NVQwen2 model.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.Tensor): The position ids.
            past_key_values (tuple[tuple[torch.Tensor, ...], ...]): The past key values.
            inputs_embeds (torch.Tensor): The inputs embeds.
            use_cache (bool): Whether to use cache.
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
            # Left-side padding is not supported in TE layers, so to make huggingface-style generation work with TE we
            # dynamically convert to THD-style inputs in our forward pass, and then convert back to BSHD for the output.
            # This lets the entire transformer stack run in THD mode. This might be slower for BSHD + padding with fused
            # attention backend, but it should be faster for the flash attention backend.
            assert attention_mask is not None, "Attention mask is required when packing BSHD inputs."
            batch_size = hidden_states.size(0)
            padded_seq_len = input_ids.size(1) if input_ids is not None else hidden_states.size(1)
            hidden_states, indices, cu_seqlens, max_seqlen, _ = _unpad_input(hidden_states, attention_mask)
            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens
            kwargs["max_length_q"] = kwargs["max_length_k"] = max_seqlen

        if self.config.attn_input_format == "thd" and hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            # For THD, the embedding output is a 3-dimensional tensor with shape [1, total_tokens, hidden_size], but TE
            # expects a 2-dimensional tensor with shape [total_tokens, hidden_size].
            hidden_states = hidden_states.squeeze(0)

        if self.config.attn_input_format == "bshd" and attention_mask is not None and attention_mask.dim() == 2:
            # Convert HF mask (1=attend, 0=pad) to TE boolean mask (True=masked, False=attend)
            attention_mask = ~attention_mask[:, None, None, :].bool()

        if isinstance(past_key_values, InferenceParams):  # InferenceParams is TE's way of managing kv-caching.
            # In generation mode, we set the length to 1 for each batch index. Otherwise, we use the attention mask to
            # compute the lengths of each sequence in the batch.
            lengths = (
                attention_mask.sum(dim=1).tolist()
                if attention_mask.shape == input_ids.shape
                else [1] * input_ids.shape[0]
            )
            past_key_values.pre_step(OrderedDict(zip(list(range(len(lengths))), lengths)))

        # Ensure that rotary embeddings are computed with at a higher precision
        with torch.autocast(device_type="cuda", enabled=False):
            te_rope_emb = self.rotary_emb(max_seq_len=self.config.max_position_embeddings)
            if te_rope_emb.dtype != torch.float32:
                warnings.warn("Rotary embeddings should be in float32 for optimal performance.", UserWarning)

        with self.get_autocast_context(None, outer=True):
            for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
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
                        cu_seqlens_q_padded=kwargs.get("cu_seq_lens_q_padded", None),
                        cu_seqlens_kv_padded=kwargs.get("cu_seq_lens_k_padded", None),
                        max_seqlen_q=kwargs.get("max_length_q", None),
                        max_seqlen_kv=kwargs.get("max_length_k", None),
                        pad_between_seqs=kwargs.get("pad_between_seqs", None),
                    )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer. Note that these will be in THD format; we could possibly pad
        # these with the same _pad_input call as below if we wanted them returned in BSHD format.
        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if should_pack_inputs:
            # If we've converted BSHD to THD for our TE layers, we need to convert back to BSHD for the output.
            hidden_states = _pad_input(hidden_states, indices, batch_size, padded_seq_len)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )

    def get_autocast_context(
        self, layer_number: int | None, init: bool = False, outer: bool = False
    ) -> ContextManager:
        """Return the appropriate TE autocast context manager for a given layer.

        This function handles both the quantized_model_init during layer creation and the te.autocast() during layer
        forward pass.

        Args:
            layer_number: The 0-indexed layer number.
            init: Whether to return a `quantized_model_init` context for layer initialization.
            outer: Whether to return a global te.autocast() context to wrap the entire model stack.
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


class NVQwen2ForCausalLM(NVQwen2PreTrainedModel, transformers.GenerationMixin):
    """Qwen2 model with causal language head."""

    _tied_weights_keys: ClassVar[dict[str, str]] = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(
        self,
        config,
        fp8_recipe: transformer_engine.common.recipe.Recipe | None = None,
        fp4_recipe: transformer_engine.common.recipe.Recipe | None = None,
    ):
        """Initialize the NVQwen2ForCausalLM model.

        Args:
            config: The configuration of the model.
            fp8_recipe: The FP8 recipe for the model.
            fp4_recipe: The FP4 recipe for the model.
        """
        super().__init__(config)
        self.model = NVQwen2Model(config, fp8_recipe=fp8_recipe, fp4_recipe=fp4_recipe)
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
        """Forward pass for the NVQwen2ForCausalLM model.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            position_ids (torch.Tensor): The position ids.
            past_key_values (tuple[tuple[torch.Tensor, ...], ...]): The past key values.
            inputs_embeds (torch.Tensor): The inputs embeds.
            labels (torch.Tensor): The labels.
            shift_labels (torch.Tensor): Labels that have already been shifted by the dataloader, to be used instead of
                labels for the loss function. For context parallelism, it is more reliable to shift the labels before
                splitting the batch into shards.
            use_cache (bool): Whether to use cache.
            cache_position (torch.Tensor): The cache position.
            logits_to_keep (int | torch.Tensor): Whether to keep only the last logits to reduce the memory footprint of
                the model during generation.
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
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


torch._dynamo.config.capture_scalar_outputs = True


@torch.compile
def _pad_input(hidden_states, indices, batch, seqlen):
    """Convert a THD tensor to a BSHD equivalent tensor.

    Adapted from huggingface/transformers/modeling_flash_attention_utils.py

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.

    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)


@torch.compile
def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Convert a BSHD tensor to a THD equivalent tensor.

    Adapted from huggingface/transformers/modeling_flash_attention_utils.py

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.

    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    batch_size = hidden_states.size(0)
    seq_length = hidden_states.size(1)

    if attention_mask.shape[1] != seq_length:  # Likely in generation mode with kv-caching
        return (
            hidden_states.squeeze(1),  # hidden_states
            torch.arange(batch_size, dtype=torch.int64, device=hidden_states.device),  # indices
            torch.arange(batch_size + 1, dtype=torch.int32, device=hidden_states.device),  # cu_seqlens
            1,  # max_seqlen
            1,  # seqused
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


class HFInferenceParams(InferenceParams):
    """Extension of the InferenceParams class to support beam search."""

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current cached sequence length.

        Required by HuggingFace transformers generate() to determine how many
        tokens have already been cached.
        """
        if not self.sequences:
            return 0
        return max(self.sequences.values())

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder the cache based on the beam indices."""
        if isinstance(self.cache_manager, PagedKVCacheManager):
            raise NotImplementedError("Beam search is not supported for paged cache manager.")
        for layer_number, (key_cache, value_cache) in self.cache_manager.cache.items():
            updated_key_cache = key_cache.index_select(0, beam_idx)
            updated_value_cache = value_cache.index_select(0, beam_idx)
            self.cache_manager.cache[layer_number] = (updated_key_cache, updated_value_cache)

    @property
    def is_compileable(self) -> bool:
        """Return False as this cache is not compatible with torch.compile."""
        return False
