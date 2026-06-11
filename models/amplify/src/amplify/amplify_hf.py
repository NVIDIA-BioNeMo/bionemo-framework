# noqa: license-check
# SPDX-FileCopyrightText: Copyright (c) 2024 chandar-lab
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 chandar-lab
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
# From https://huggingface.co/chandar-lab/AMPLIFY_120M/blob/main/amplify.py
# From https://stackoverflow.com/a/23689767
# From https://github.com/pytorch/pytorch/issues/97899
# From https://github.com/facebookresearch/llama/blob/main/llama/model.py

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from xformers.ops import SwiGLU, memory_efficient_attention

from .rmsnorm import RMSNorm
from .rotary import apply_rotary_emb, precompute_freqs_cis


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
        self.ffn_bias = ffn_bias
        self.att_bias = att_bias
        self.pad_token_id = pad_token_id
        self.max_length = max_length


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: AMPLIFYConfig):
        """Initialize a EncoderBlock.

        Args:
            config (AMPLIFYConfig): The configuration of the model.
        """
        super().__init__()

        self.config = config
        self.d_head = config.hidden_size // config.num_attention_heads

        # Attention
        self.q = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=config.att_bias,
        )
        self.k = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=config.att_bias,
        )
        self.v = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=config.att_bias,
        )
        self.wo = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=config.att_bias,
        )
        self.resid_dropout = nn.Dropout(config.dropout_prob)

        # Feedforward network
        match config.hidden_act.lower():
            case "swiglu":
                # To keep the number of parameters and the amount of computation constant, we reduce the number of
                # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
                # avoid RuntimeError due to misaligned operand
                multiple_of = 8
                intermediate_size = int(2 * config.intermediate_size / 3)
                intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
                self.ffn = SwiGLU(
                    config.hidden_size,
                    intermediate_size,
                    config.hidden_size,
                    bias=config.ffn_bias,
                )
            case "relu":
                self.ffn = nn.Sequential(
                    nn.Linear(
                        config.hidden_size,
                        config.intermediate_size,
                        bias=config.ffn_bias,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        config.intermediate_size,
                        config.hidden_size,
                        bias=config.ffn_bias,
                    ),
                )
            case "gelu":
                self.ffn = nn.Sequential(
                    nn.Linear(
                        config.hidden_size,
                        config.intermediate_size,
                        bias=config.ffn_bias,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        config.intermediate_size,
                        config.hidden_size,
                        bias=config.ffn_bias,
                    ),
                )

        self.attention_norm = (
            RMSNorm(config.hidden_size, config.norm_eps)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )
        self.ffn_norm = (
            RMSNorm(config.hidden_size, config.norm_eps)
            if config.rms_norm
            else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        self.ffn_dropout = nn.Dropout(config.dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        output_attentions: bool,
    ):
        """Forward pass of the EncoderBlock.

        Args:
            x (torch.Tensor): The input tensor.
            attention_mask (torch.Tensor): The attention mask.
            freqs_cis (torch.Tensor): The frequency tensor.
            output_attentions (bool): Whether to output the attention weights.

        Returns:
            tuple(torch.Tensor, torch.Tensor): The output tensor and the attention weights.
        """
        attn, contact = self._att_block(self.attention_norm(x), attention_mask, freqs_cis, output_attentions)
        x = x + attn
        x = x + self._ff_block(self.ffn_norm(x))
        return x, contact

    def _att_block(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        output_attentions: bool,
    ):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.config.num_attention_heads, self.d_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Compute the attention weight
        attn_weights = None
        if output_attentions:
            attn_weights = xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) / (xq.size(-1) ** 0.5)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.softmax(-1)

        # Compute the attention using xformers if the tensors are on GPU
        if x.is_cuda:
            # Input and output are of dimension (B, M, H, K) where B is the batch size, M the sequence length,
            # H the number of heads, and K the embeding size per head
            attn = memory_efficient_attention(
                query=xq,
                key=xk,
                value=xv,
                attn_bias=attention_mask,
                p=self.config.dropout_prob if self.training else 0,
            )
        else:
            # Input and output are of dimension (B, H, M, K)
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=self.config.dropout_prob if self.training else 0,
            ).transpose(1, 2)

        attn_scores = self.wo(attn.view(batch_size, seq_len, self.config.num_attention_heads * self.d_head))
        return (self.resid_dropout(attn_scores), attn_weights)

    def _ff_block(self, x: torch.Tensor):
        return self.ffn_dropout(self.ffn(x))


class AMPLIFYPreTrainedModel(PreTrainedModel):
    """AMPLIFY pre-trained model."""

    config_class = AMPLIFYConfig

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
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

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.layer_norm_after_embedding:
            self.layer_norm_1 = (
                RMSNorm(config.hidden_size, config.norm_eps)
                if config.rms_norm
                else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        if config.layer_norm_before_last_layer:
            self.layer_norm_2 = (
                RMSNorm(config.hidden_size, config.norm_eps)
                if config.rms_norm
                else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False,
        labels=None,
        **kwargs,
    ) -> MaskedLMOutput:
        """Forward pass of the AMPLIFY model.

        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (torch.Tensor): The attention mask.
            output_hidden_states (bool): Whether to output the hidden states.
            output_attentions (bool): Whether to output the attention weights.
            labels (torch.Tensor): The labels.
            **kwargs: Additional arguments.

        Returns:
            MaskedLMOutput: The output of the model.
        """
        # Initialize
        hidden_states, attentions = [], []

        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if attention_mask is not None and not torch.all(attention_mask == 0):
            attention_mask = torch.where(attention_mask == 1, float(0.0), float("-inf")).to(torch.bfloat16)
            attention_mask = (
                attention_mask.unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, self.config.num_attention_heads, attention_mask.size(-1), 1)
            )
        else:
            attention_mask = None

        # RoPE
        self.freqs_cis = self.freqs_cis.to(input_ids.device, non_blocking=True)
        freqs_cis = self.freqs_cis[: input_ids.shape[1]]

        # Embedding
        x = self.encoder(input_ids)
        if self.config.layer_norm_after_embedding:
            x = self.layer_norm_1(x)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x, attn = layer(x, attention_mask, freqs_cis, output_attentions)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn)

        # Classification head with layer norm
        logits = self.decoder(self.layer_norm_2(x) if self.config.layer_norm_before_last_layer else x)

        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        else:
            loss = None

        # Return logits or the output of the last hidden layer
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
