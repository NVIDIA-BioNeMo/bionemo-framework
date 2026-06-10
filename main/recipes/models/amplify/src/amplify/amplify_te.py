# noqa: license-check
# SPDX-FileCopyrightText: Copyright (c) 2024 chandar-lab
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 chandar-lab
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted from https://huggingface.co/chandar-lab/AMPLIFY_120M/blob/main/amplify.py

import torch
import transformer_engine.pytorch
from torch import nn
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel


class AMPLIFYConfig(PretrainedConfig):
    """AMPLIFY model configuration."""

    model_type = "AMPLIFY"

    # All config parameters must have a default value.
    def __init__(
        self,
        hidden_size: int = 960,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 15,
        intermediate_size: int = 3840,
        dropout_prob: float = 0,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        rms_norm: bool = True,
        norm_eps: float = 1e-05,
        hidden_act: str = "SwiGLU",
        layer_norm_after_embedding: bool = False,
        layer_norm_before_last_layer: bool = True,
        vocab_size: int = 27,
        padded_vocab_size: int = 32,
        ffn_bias: bool = False,
        att_bias: bool = False,
        pad_token_id: int = 0,
        max_length: int = 2048,
        **kwargs,
    ):
        """Initialize a AMPLIFYConfig.

        Args:
            hidden_size (int): The hidden size of the model.
            num_hidden_layers (int): The number of hidden layers in the model.
            num_attention_heads (int): The number of attention heads in the model.
            intermediate_size (int): The intermediate size of the model.
            dropout_prob (float): The dropout probability of the model.
            embedding_init_range (float): The range of the embedding initialization.
            decoder_init_range (float): The range of the decoder initialization.
            rms_norm (bool): Whether to use RMSNorm.
            norm_eps (float): The epsilon for the normalization.
            hidden_act (str): The activation function of the model.
            layer_norm_after_embedding (bool): Whether to use layer normalization after the embedding.
            layer_norm_before_last_layer (bool): Whether to use layer normalization before the last layer.
            vocab_size (int): The vocabulary size of the model.
            padded_vocab_size (int): The padded vocabulary size of the model to support fp8.
            ffn_bias (bool): Whether to use bias in the feedforward network.
            att_bias (bool): Whether to use bias in the attention.
            pad_token_id (int): The padding token id.
            max_length (int): The maximum length of the sequence.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_prob = dropout_prob
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.rms_norm = rms_norm
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.layer_norm_after_embedding = layer_norm_after_embedding
        self.layer_norm_before_last_layer = layer_norm_before_last_layer
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.ffn_bias = ffn_bias
        self.att_bias = att_bias
        self.pad_token_id = pad_token_id
        self.max_length = max_length

        assert self.padded_vocab_size >= self.vocab_size, (
            "padded_vocab_size must be greater than or equal to vocab_size"
        )


class AMPLIFYPreTrainedModel(PreTrainedModel):
    """AMPLIFY pre-trained model."""

    config: AMPLIFYConfig
    config_class = AMPLIFYConfig
    base_model_prefix = "amplify"

    def _init_weights(self, module):
        if isinstance(
            module, (nn.Linear, transformer_engine.pytorch.Linear, transformer_engine.pytorch.LayerNormLinear)
        ):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)


class AMPLIFY(AMPLIFYPreTrainedModel):
    """The main model class."""

    def __init__(self, config: AMPLIFYConfig, **kwargs):
        """Initialize a AMPLIFY model.

        Args:
            config (AMPLIFYConfig): The configuration of the model.
            **kwargs: Additional arguments.
        """
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(
            config.padded_vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=config.dtype,
        )

        if config.layer_norm_after_embedding:
            self.layer_norm_1 = (
                transformer_engine.pytorch.RMSNorm(config.hidden_size, config.norm_eps, params_dtype=config.dtype)
                if config.rms_norm
                else transformer_engine.pytorch.LayerNorm(
                    config.hidden_size, config.norm_eps, params_dtype=config.dtype
                )
            )

        if config.hidden_act.lower() == "swiglu":
            # To keep the number of parameters and the amount of computation constant, we reduce the
            # number of hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and
            # make it a multiple of 8 to avoid RuntimeError due to misaligned operand
            multiple_of = 8
            intermediate_size = int(2 * config.intermediate_size / 3)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

        else:
            intermediate_size = config.intermediate_size

        self.transformer_encoder = nn.ModuleList()
        for layer_num in range(config.num_hidden_layers):
            self.transformer_encoder.append(
                transformer_engine.pytorch.TransformerLayer(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    layernorm_epsilon=config.norm_eps,
                    hidden_dropout=config.dropout_prob,
                    attention_dropout=config.dropout_prob,
                    apply_residual_connection_post_layernorm=False,
                    layer_type="encoder",
                    self_attn_mask_type="padding",
                    normalization="RMSNorm" if config.rms_norm else "LayerNorm",
                    fuse_qkv_params=True,
                    qkv_weight_interleaved=True,
                    output_layernorm=False,
                    bias=False,
                    activation=config.hidden_act.lower(),
                    attn_input_format="bshd",
                    layer_number=layer_num + 1,
                    name="encoder_block",
                    window_size=(-1, -1),
                    rotary_pos_interleaved=True,
                    seq_length=config.max_length,
                    params_dtype=config.dtype,
                )
            )

        self.freqs_cis = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads, interleaved=True)(
            config.max_length
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """Get the input embeddings of the model."""
        return self.encoder

    def set_input_embeddings(self, value: nn.Embedding):
        """Set the input embeddings of the model.

        Args:
            value (nn.Embedding): The input embeddings.
        """
        self.encoder = value

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False,
        labels=None,
    ) -> BaseModelOutput:
        """Forward pass of the AMPLIFY model.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            output_hidden_states (bool): Whether to output the hidden states.
            output_attentions (bool): Whether to output the attention weights.
            labels (torch.Tensor): The labels.

        Returns:
            BaseModelOutput: The output of the model.
        """
        # Initialize
        hidden_states = []

        # Attention mask
        if attention_mask is not None and attention_mask.dtype is torch.int64:
            # TE expects a boolean attention mask, where "True" indicates a token to be masked.
            attention_mask = ~attention_mask.to(bool)

        # RoPE
        self.freqs_cis = self.freqs_cis.to(input_ids.device, non_blocking=True)
        freqs_cis = self.freqs_cis[: input_ids.shape[1]]

        # Embedding
        x = self.encoder(input_ids)
        if self.config.layer_norm_after_embedding:
            x = self.layer_norm_1(x)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x = layer(x, attention_mask, rotary_pos_emb=freqs_cis)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                raise ValueError("output_attentions is not supported for TE")

        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=tuple(hidden_states) if hidden_states else None,
            attentions=None,
        )


class AMPLIFYForMaskedLM(AMPLIFYPreTrainedModel):
    """AMPLIFY for masked language modeling."""

    def __init__(self, config: AMPLIFYConfig, **kwargs):
        """Initialize a AMPLIFYForMaskedLM model.

        Args:
            config (AMPLIFYConfig): The configuration of the model.
            **kwargs: Additional arguments.
        """
        super().__init__(config)
        self.amplify = AMPLIFY(config, **kwargs)

        if config.layer_norm_before_last_layer:
            self.decoder = transformer_engine.pytorch.LayerNormLinear(
                config.hidden_size,
                config.padded_vocab_size,
                config.norm_eps,
                params_dtype=config.dtype,
                normalization="RMSNorm" if config.rms_norm else "LayerNorm",
                init_method=lambda x: torch.nn.init.uniform_(
                    x, -self.config.decoder_init_range, self.config.decoder_init_range
                ),
            )

        else:
            self.decoder = transformer_engine.pytorch.Linear(
                config.hidden_size, config.vocab_size, params_dtype=config.dtype
            )

    def get_input_embeddings(self):
        """Get the input embeddings of the model."""
        return self.amplify.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        """Set the input embeddings of the model.

        Args:
            value (nn.Embedding): The input embeddings.
        """
        self.amplify.set_input_embeddings(value)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False,
        labels=None,
    ) -> MaskedLMOutput:
        """Forward pass of the AMPLIFYForMaskedLM model.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            output_hidden_states (bool): Whether to output the hidden states.
            output_attentions (bool): Whether to output the attention weights.
            labels (torch.Tensor): The labels.

        Returns:
            MaskedLMOutput: The output of the model.
        """
        outputs = self.amplify(
            input_ids,
            attention_mask,
            output_hidden_states,
            output_attentions,
            labels,
        )

        # Classification head with layer norm
        logits = self.decoder(outputs.last_hidden_state)
        if self.config.padded_vocab_size != self.config.vocab_size:
            logits = logits[:, :, : self.config.vocab_size]

        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        else:
            loss = None

        # Return logits or the output of the last hidden layer
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
