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


from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence, Type, TypeVar

import torch
from megatron.core import tensor_parallel
from megatron.core.models.bert.pooler import Pooler
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer import spec_utils
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import get_linear_layer
from torch import Tensor
from torch.nn.functional import silu
from torch.optim import Optimizer

# from bionemo.amplify.data.tokenizer import BioNeMoAMPLIFYTokenizer
from bionemo.llm.api import MegatronLossType
from bionemo.llm.model.biobert.model import BioBertConfig, MegatronBioBertModel, PositionEmbeddingKinds
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.utils import iomixin_utils as iom


__all__: Sequence[str] = (
    "AMPLIFYConfig",
    "AMPLIFYModel",
)


class AMPLIFYLMHead(MegatronModule):
    """LM head for AMPLIFY.

    Args:
        hidden_size: hidden size
        config (TransformerConfig): TransformerConfig object
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.head = IdentityOp()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.head(hidden_states)


class AMPLIFYModel(MegatronBioBertModel):
    """AMPLIFY protein language model."""

    def __init__(
        self,
        config: TransformerConfig,
        num_tokentypes: int,
        transformer_layer_spec: spec_utils.ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        tokenizer: Optional[Any] = None,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        add_binary_head: bool = True,
        return_embeddings: bool = False,
        include_embeddings: bool = False,
        include_input_ids: bool = False,
        use_full_attention_mask: bool = False,
        include_hiddens: bool = False,
        skip_logits: bool = False,
    ) -> None:
        """Initialize the AMPLIFY model.

        Args:
            config (TransformerConfig): transformer config
            num_tokentypes (int): Set to 2 when args.bert_binary_head is True, and 0 otherwise. Defaults to 0.
            transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
            vocab_size (int): vocabulary size
            max_sequence_length (int): maximum size of sequence. This is used for positional embedding
            tokenizer (AutoTokenizer): optional tokenizer object (currently only used in the constructor of ESM2Model)
            pre_process (bool): Include embedding layer (used with pipeline parallelism)
            post_process (bool): Include an output layer (used with pipeline parallelism)
            fp16_lm_cross_entropy: Whether to move the cross entropy unreduced loss calculation for lm head to fp16.
            parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks
            share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are shared. Defaults to False.
            position_embedding_type (string): Position embedding type. Options ['learned_absolute', 'rope'].
                Defaults is 'learned_absolute'.
            rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
                Defaults to 1.0 (100%). Ignored unless position_embedding_type is 'rope'.
            seq_len_interpolation_factor (Optional[float]): Interpolation factor for sequence length. Defaults to None.
            add_binary_head (bool): Whether to add a binary head. Defaults to True.
            return_embeddings (bool): Whether to return embeddings. Defaults to False.
            include_embeddings (bool): Whether to include embeddings in the output dictionary. Defaults to False.
            include_input_ids (bool): Whether to include input_ids in the output dictionary. Defaults to False.
            use_full_attention_mask (bool): Whether to use full attention mask. Defaults to False.
            include_hiddens (bool): Whether to include hidden states in the output dictionary. Defaults to False.
            skip_logits (bool): Skip writing the token logits in output dict
        """
        super(MegatronBioBertModel, self).__init__(config=config)
        self.post_process = post_process
        self.add_binary_head = add_binary_head
        if return_embeddings:
            assert self.post_process, "only return embeddings on the last pipeline stage"
        # `b` = batch, `s` = sequence.
        # The old flash attention mechanism apparently wants you to use a b x 1 x s x s attention mask while
        #  the new one wants a b x 1 x 1 x s attention mask. This is a hack to allow us to switch between the two.
        self.use_full_attention_mask = use_full_attention_mask
        self.config: TransformerConfig = config
        self.transformer_layer_spec: spec_utils.ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.add_binary_head = add_binary_head
        self.return_embeddings = return_embeddings
        self.include_embeddings = include_embeddings
        self.include_hiddens = include_hiddens
        self.include_input_ids = include_input_ids
        self.skip_logits = skip_logits

        if config.gated_linear_unit:
            multiple_of = 8
            intermediate_size = int(2 * config.ffn_hidden_size / 3)
            config.ffn_hidden_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        # Embeddings.
        if self.pre_process:
            self.register_buffer(
                "bert_position_id_tensor",
                torch.arange(max_sequence_length, dtype=torch.long, requires_grad=False).unsqueeze(0),
                persistent=False,
            )
            # ESM2 Customization: ESM2Embedding instead of LanguageModelEmbedding
            # TODO: call super, overwrite the self.embedding, and setup_embeddings_and_output_layer in constructor.
            # Note: need to avoid calling setup twice: skip with super (super(skip_setup=True))
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                num_tokentypes=num_tokentypes,
            )

        if self.position_embedding_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
            )

        # Transformer.
        self.encoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            # TODO: Make sure you are passing in the mpu_vocab_size properly
            self.lm_head = AMPLIFYLMHead(config)

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=True,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights,
            )

            self.binary_head = None
            if self.add_binary_head:
                # TODO: Shoudl switch this to TE ?
                self.binary_head = get_linear_layer(
                    config.hidden_size, 2, config.init_method, config.perform_initialization
                )

                self.pooler = Pooler(config.hidden_size, config.init_method, config, config.sequence_parallel)

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def embedding_forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        tokentype_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Produce embeddings."""
        return self.embedding(input_ids=input_ids, position_ids=position_ids, tokentype_ids=tokentype_ids)


AMPLIFYModelT = TypeVar("AMPLIFYModelT", bound=AMPLIFYModel)


@dataclass
class AMPLIFYConfig(BioBertConfig[AMPLIFYModelT, MegatronLossType], iom.IOMixinWithGettersSetters):
    """Configuration class for AMPLIFY model.

    Attributes:
        num_layers: Number of layers in the model.
        hidden_size: Hidden size of the model.
        num_attention_heads: Number of attention heads in the model.
        ffn_hidden_size: Hidden size of the feed-forward network.
        hidden_dropout: Dropout rate for hidden layers.
        attention_dropout: Dropout rate for attention layers.
        apply_residual_connection_post_layernorm: Whether to apply residual connection after layer normalization.
        layernorm_epsilon: Epsilon value for layer normalization.
        layernorm_zero_centered_gamma: Whether to zero-center the gamma parameter in layer normalization.
        activation_func: Activation function used in the model.
        init_method_std: Standard deviation for weight initialization.
        apply_query_key_layer_scaling: Whether to apply scaling to query and key layers.
        masked_softmax_fusion: Whether to use a kernel that fuses attention softmax with its mask.
        fp16_lm_cross_entropy: Whether to move the cross entropy unreduced loss calculation for lm head to fp16.
        share_embeddings_and_output_weights: Whether to share embeddings and output weights.
        enable_autocast: Whether to enable autocast for mixed precision.
        biobert_spec_option: BiobertSpecOption for the model.
        position_embedding_type: Type of position embedding used in the model.
        seq_length: Length of the input sequence.
        make_vocab_size_divisible_by: Make the vocabulary size divisible by this value.
        token_dropout: Whether to apply token dropout.
        use_attention_mask: Whether to use attention mask.
        use_esm_attention: Whether to use ESM attention.
        attention_softmax_in_fp32: Whether to use fp32 for attention softmax.
        optimizer_fn: Optional optimizer function for the model.
        parallel_output: Whether to use parallel output.
        rotary_base: Base value for rotary positional encoding.
        rotary_percent: Percentage of rotary positional encoding.
        seq_len_interpolation_factor: Interpolation factor for sequence length.
        get_attention_mask_from_fusion: Whether to get attention mask from fusion.
        nemo1_ckpt_path: Path to NEMO1 checkpoint.
        return_only_hidden_states: Whether to return only hidden states.
        loss_reduction_class: Loss reduction class for the model. Default to BERTMLMLossWithReduction.
    """

    # When overriding fields in a dataclass _always_ declare types: https://github.com/python/cpython/issues/123269
    model_cls: Type[AMPLIFYModelT] = AMPLIFYModel
    seq_length: int = 512
    num_layers: int = 24  # 32 for 350M, 24 for 120M
    hidden_size: int = 640  # 960 for 350M, 640 for 120M
    num_attention_heads: int = 10  # 15 for 350M, 10 for 120M
    ffn_hidden_size: int = 2560  # Transformer FFN hidden size. Usually 4 * hidden_size.
    hidden_dropout: float = 0  # AMPLIFY removes dropout from hidden layers and attention
    attention_dropout: float = 0.0  # AMPLIFY does not use attention dropout
    apply_residual_connection_post_layernorm: bool = False  # TODO: farhadr False is new default, True was BERT pub.
    layernorm_epsilon: float = 1.0e-5
    init_method_std: float = 0.02

    # embedding
    token_dropout: bool = False
    use_attention_mask: bool = True

    # core attention
    use_esm_attention: bool = False  # Skip ESM2 custom attention for TE acceleration. Still passes golden value test.
    attention_softmax_in_fp32: bool = False
    normalize_attention_scores: bool = False

    # From megatron.core.models.gpt.bert_model.GPTModel
    fp16_lm_cross_entropy: bool = False  # Move the cross entropy unreduced loss calculation for lm head to fp16
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    make_vocab_size_divisible_by: int = 32
    position_embedding_type: PositionEmbeddingKinds = "rope"
    rotary_interleaved: bool = True
    rotary_base: int = 10_000
    rotary_percent: float = 1.0

    # AMPLIFY specific configuration
    add_bias_linear: bool = False  # AMPLIFY does not use bias in linear layers
    bias_swiglu_fusion: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = False
    gated_linear_unit: bool = True
    masked_softmax_fusion: bool = True
    activation_func: Callable = silu
    normalization: str = "RMSNorm"  # AMPLIFY uses RMSNorm instead of LayerNorm
    layernorm_zero_centered_gamma: bool = False  # Zero centered gamma not supported for RMSNorm
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.amplify_with_transformer_engine_spec
    apply_query_key_layer_scaling = False

    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False

    optimizer_fn: Optional[Callable[[MegatronBioBertModel], Optimizer]] = None
    # TODO (@skothenhill,@georgea) update to use the nemo2 checkpoint mixins
    #  support HF (requires weight interleaving on qkv layer) and nemo1 checkpoints ideally.
    nemo1_ckpt_path: str | None = None
    # The following checkpoint path is for nemo2 checkpoints. Config parameters not present in
    #  self.override_parent_fields will be loaded from the checkpoint and override those values here.
    initial_ckpt_path: str | None = None
    # TODO (@jstjohn) come up with a cleaner way in the biobert module to return user requested
    #  things as part of the workflow for inference and fine-tuning.
    return_embeddings: bool = False
    include_embeddings: bool = False
    skip_logits: bool = False
    return_only_hidden_states: bool = False  # return logits
