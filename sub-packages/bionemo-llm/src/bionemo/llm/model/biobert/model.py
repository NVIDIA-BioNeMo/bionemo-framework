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


import logging
import tarfile
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypedDict,
    TypeVar,
)

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.bert.bert_lm_head import BertLMHead
from megatron.core.models.bert.pooler import Pooler
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import get_linear_layer
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning import get_vocab_size
from torch import Tensor
from torch.optim import Optimizer

from bionemo.llm.api import MegatronLossType
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption, get_biobert_spec
from bionemo.llm.model.config import (
    OVERRIDE_BIONEMO_CONFIG_DEFAULTS,
    MegatronBioNeMoTrainableModelConfig,
    TorchmetricsConfig,
)
from bionemo.llm.model.loss import BERTMLMLossWithReduction
from bionemo.llm.utils.weight_utils import nemo1_to_nemo2_biobert_key_mapping


__all__: Sequence[str] = (
    "OVERRIDE_BIOBERT_CONFIG_DEFAULTS",
    "BioBertConfig",
    "BioBertOutput",
    "BioBertOutputCore",
    "MegatronBioBertModel",
    "MegatronBioBertModelType",
    "PositionEmbeddingKinds",
)

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
)

logger = logging.getLogger(__file__)

# Add some fields specific to the BIOBERT config that we want to override by default
# TODO automatically determine which fields a user is trying to override in the future.
_OVERRIDE_BIOBERT_CONFIG_DEFAULTS: List[str] = OVERRIDE_BIONEMO_CONFIG_DEFAULTS + [
    "return_only_hidden_states",
    "include_embeddings",
    "include_input_ids",
    "include_hiddens",
    # Precision override for starting from a checkpoint and casting to a different precision
    "params_dtype",
    "pipeline_dtype",
    "autocast_dtype",
    # Model parallelism settings! Important to override these if the user requests different settings from how
    #  a model was trained (common). See https://github.com/NVIDIA/bionemo-framework/issues/275
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "sequence_parallel",
    "context_parallel_size",
    "expert_model_parallel_size",
    "apply_rope_fusion",
    "bias_dropout_fusion",
    "bias_activation_fusion",
    "attention_softmax_in_fp32",
    "get_attention_mask_from_fusion",
    "activation_func",  # FIXME hack: update the ESM2 checkpoint with the updated activation function and don't override
    "moe_extended_tp",
    "skip_logits",
]

# A copy that we do not use internally. Useful for external users who want to
#  start with these defaults and add some new keys that they want to not override.
OVERRIDE_BIOBERT_CONFIG_DEFAULTS = deepcopy(_OVERRIDE_BIOBERT_CONFIG_DEFAULTS)


class BioBertOutputCore(TypedDict):
    """Keys always present in the bionemo bert model inference output."""

    token_logits: Tensor
    binary_logits: Optional[Tensor]


class BioBertOutput(BioBertOutputCore, total=False):
    """The megatron bionemo bert model inference type."""

    hidden_states: Tensor


PositionEmbeddingKinds = Literal["learned_absolute", "rope"]
"""Kinds of supported positional embeddings."""


# TODO make this a base class without the language head and pooler
class MegatronBioBertModel(LanguageModule):
    """Transformer language model.

    Args:
        config: transformer config
        num_tokentypes: Set to 2 when args.bert_binary_head is True, and 0 otherwise. Defaults to 0.
        transformer_layer_spec: Specifies module to use for transformer layers
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This is used for positional embedding
        pre_process: Include embedding layer (used with pipeline parallelism)
        post_process: Include an output layer (used with pipeline parallelism)
        parallel_output: Do not gather the outputs, keep them split across tensor parallel ranks
        share_embeddings_and_output_weights: When True, input embeddings and output logit weights are shared.
            Defaults to False.
        position_embedding_type: Position embedding type. Options ["learned_absolute", "rope"].
            Defaults is 'learned_absolute'.
        rotary_percent: Percent of rotary dimension to use for rotary position embeddings.
            Defaults to 1.0 (100%). Ignored unless position_embedding_type is 'rope'.
    """

    def __init__(  # noqa: D107
        self,
        config: TransformerConfig,
        num_tokentypes: int,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        tokenizer: Optional[AutoTokenizer] = None,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: PositionEmbeddingKinds = "learned_absolute",
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        add_binary_head: bool = False,
        return_embeddings: bool = False,
        include_embeddings: bool = False,
        use_full_attention_mask: bool = False,
        include_hiddens: bool = False,
        include_input_ids: bool = False,
        skip_logits: bool = False,  # Useful for inference time.
    ):
        # TODO (@jstjohn) come up with a cleaner way for this model to return a set of things the user wants.
        #  hidden states, embeddings, logits, etc. The defaults should work for training but we need to make it
        #  customizable and easy to tell how to make it work well for inference as well as trouble shooting.
        #  Also make sure that everything returned that the user wants gets transposed to the b,s,h format.
        super(MegatronBioBertModel, self).__init__(config=config)
        self.post_process = post_process
        self.add_binary_head = add_binary_head
        self.skip_logits = skip_logits
        if return_embeddings:
            assert self.post_process, "only return embeddings on the last pipeline stage"
        # `b` = batch, `s` = sequence.
        # The old flash attention mechanism apparently wants you to use a b x 1 x s x s attention mask while
        #  the new one wants a b x 1 x 1 x s attention mask. This is a hack to allow us to switch between the two.
        self.use_full_attention_mask = use_full_attention_mask
        self.config: TransformerConfig = config
        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
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

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder
        # Embeddings.
        if self.pre_process:
            self.register_buffer(
                "bert_position_id_tensor",
                torch.arange(max_sequence_length, dtype=torch.long, requires_grad=False).unsqueeze(0),
                persistent=False,
            )
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
                # bug in megatron: they list the type as `float` but they default to `None` so it should be `Optional[float]`
                seq_len_interpolation_factor=seq_len_interpolation_factor,  # type: ignore
            )

        # Transformer.
        self.encoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,  # NOTE: in bionemo1 this is hard-coded to True
        )

        # Output
        if post_process:
            # TODO: Make sure you are passing in the mpu_vocab_size properly
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.lm_head = BertLMHead(
                config.hidden_size,
                config,
            )

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                is_expert=False,
                bias=True,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=pre_process and share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
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

    def bert_extended_attention_mask(self, attention_mask: Tensor) -> Tensor:
        """Creates the extended attention mask

        Converts the attention mask of dimension [batch size, 1, seq len] to [batch size, 1, seq len, seq len] and makes it binary

        Args:
            attention_mask (Tensor): The input attention mask

        Returns:
            Tensor: The extended binary attention mask
        """  # noqa: D415
        # We create a 3D attention mask from a 2D tensor mask.
        # [b, 1, s]
        attention_mask_b1s = attention_mask.unsqueeze(1)

        if self.use_full_attention_mask:
            # [b, s, 1]
            attention_mask_bs1 = attention_mask.unsqueeze(2)
            # [b, s, s]
            attention_mask_bss = attention_mask_b1s * attention_mask_bs1
            # [b, 1, s, s]
            extended_attention_mask = attention_mask_bss.unsqueeze(1)
        else:
            # Tensor Engine requires a 1x1xS attention mask which it internally
            #  converts into a 1xSxS mask.
            # [b, 1, 1, s]
            extended_attention_mask = attention_mask_b1s.unsqueeze(1)

        # Convert attention mask to binary, and flip the values from 0 to 1 and vice versa so that
        #  extended_attention_mask._mask_fill(-1000) that megatron does internally result in
        #  masking out pad positions.
        extended_attention_mask = extended_attention_mask < 0.5

        return extended_attention_mask

    def bert_position_ids(self, token_ids):  # noqa: D102
        # Create position ids
        seq_length = token_ids.size(1)
        if seq_length != self.max_sequence_length:
            return self.bert_position_id_tensor[:, :seq_length]
        return self.bert_position_id_tensor  # No need to subset so skip the slice op

    def embedding_forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        tokentype_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Produce embeddings."""
        return self.embedding(input_ids=input_ids, position_ids=position_ids, tokentype_ids=tokentype_ids)

    def set_input_tensor(self, input_tensor: Tensor | list[Tensor]) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor: Sets the input tensor for the model.

        Raises:
            ValueError: Iff the input tensor is a list that doesn't have exactly 1 tensor.
        """
        # This is usually handled in schedules.py but some inference code still gives us non-lists or None.
        if isinstance(input_tensor, list):
            if len(input_tensor) != 1:
                raise ValueError(f"input_tensor should only be length 1 for gpt/bert, not length: {len(input_tensor)}")
            single_input_tensor: Tensor = input_tensor[0]
        else:
            single_input_tensor = input_tensor
        self.encoder.set_input_tensor(single_input_tensor)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tokentype_ids: Optional[Tensor] = None,
        lm_labels: Optional[Tensor] = None,
        inference_params: Any | None = None,
        runtime_gather_output: Optional[bool] = None,
    ) -> BioBertOutput | Tensor:
        """Forward function of BERT model

        Forward function of the BERT Model This function passes the input tensors
        through the embedding layer, and then the encoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given or the final hidden units.
        """  # noqa: D415
        # TODO! If we upgrade to TE 1.7 why does bit flipping back to 1 help the loss in TE 1.7? It claimed that they now follow standards, did
        #  nemo/megatron flip again internally to be compatible wtih TE somewhere?
        #  change the following line to ~self.bert... and see if it helps if we upgrade to TE 1.7 and NeMo/Megatron have not compensated.
        extended_attention_mask = self.bert_extended_attention_mask(attention_mask)

        if parallel_state.is_pipeline_first_stage():
            using_input_ids: Optional[Tensor] = input_ids
            using_position_ids: Optional[Tensor] = self.bert_position_ids(input_ids)
        else:
            using_input_ids = None
            using_position_ids = None

        # Encoder embedding.
        if self.pre_process:
            encoder_input: Optional[Tensor] = self.embedding_forward(
                input_ids=using_input_ids,
                position_ids=using_position_ids,
                tokentype_ids=tokentype_ids,
                attention_mask=attention_mask,
            )
        else:
            # intermediate stage of pipeline
            # encoder will get hidden_states from encoder.input_tensor
            encoder_input = None

        # Rotary positional embeddings (Why not move this into BERT/GPTEmberdding ?)
        rotary_pos_emb = None
        if self.position_embedding_type == "rope":
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params,
                self.encoder,
                encoder_input,
                self.config,
                packed_seq_params=None,  # TODO @sichu: upstream to Megatron-LM
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run encoder.
        hidden_states = self.encoder(
            hidden_states=encoder_input,
            attention_mask=extended_attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
        )

        if not self.post_process:
            return hidden_states

        if self.add_binary_head:
            pooled_output = self.pooler(hidden_states, 0)

        if self.return_embeddings or self.include_embeddings:
            embeddings = torch.transpose(hidden_states, 0, 1)
            masks = torch.sum(attention_mask, dim=1)
            # Collect masked embeddings.
            output_embeddings = torch.zeros(
                size=(embeddings.shape[0], embeddings.shape[2]),
                dtype=embeddings.dtype,
                device=torch.cuda.current_device(),
            )
            for i, (embedding, mask) in enumerate(zip(embeddings, masks)):
                output_embeddings[i, :] = torch.mean(embedding[1 : mask - 1], dim=0)

        if self.return_embeddings:
            return output_embeddings

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        hidden_states_after_lm_head = self.lm_head(hidden_states=hidden_states)
        if not self.skip_logits:
            # TODO add , runtime_gather_output=runtime_gather_output once supported in ColumnParallelLinear
            logits, _ = self.output_layer(hidden_states_after_lm_head, weight=output_weight)
        else:
            logits = None

        binary_logits = None
        if self.binary_head is not None:
            binary_logits = self.binary_head(pooled_output)

        output = {"token_logits": logits, "binary_logits": binary_logits}
        if self.include_hiddens:
            output["hidden_states"] = hidden_states.transpose(0, 1).contiguous()  # [s b h] => [b s h]
        if self.include_input_ids:
            output["input_ids"] = input_ids
        if self.include_embeddings:
            output["embeddings"] = output_embeddings
        return output


# Typevar that works for all children of MegatronBioBertModel
MegatronBioBertModelType = TypeVar("MegatronBioBertModelType", bound=MegatronBioBertModel)
"""A megatron model that is or extends the MegatronBioBertModel."""


@dataclass
class BioBertConfig(
    MegatronBioNeMoTrainableModelConfig[MegatronBioBertModelType, MegatronLossType],
):
    """Config class for BioBert model, responsible for the partial configuration of Transformer models.

    NOTE: do not use this config directly, define a child config that overrides items from this parent config

    `configure_model()` is ultimately called by the LightningModule using PTL lightning module hooks.
    """

    # From megatron.core.models.gpt.bert_model.GPTModel
    kv_channels: int | None = None
    fp16_lm_cross_entropy: bool = False
    apply_rope_fusion: bool = True
    parallel_output: bool = True
    bias_dropout_fusion: bool = True
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    get_attention_mask_from_fusion: bool = True
    share_embeddings_and_output_weights: bool = False  # try True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: PositionEmbeddingKinds = "learned_absolute"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 1024
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_layers: int = 6
    init_method_std: float = 0.02
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec

    optimizer_fn: Optional[Callable[["MegatronBioBertModel"], Optimizer]] = None
    # TODO (@skothenhill,@georgea) update to use the nemo2 checkpoint mixins
    #  support HF (requires weight interleaving on qkv layer) and nemo1 checkpoints ideally.
    # TODO (@skothenhill,@jstjohn) come up with a nice way of doing fine-tuning checkpoint loading,
    #  where some acceptible layers (eg lm_head) may or may not be absent from the model, and others
    #  (like a new head) may be new and missing from the initial checkpoint.
    nemo1_ckpt_path: Optional[str] = None

    initial_ckpt_path: Optional[str] = None
    # TODO(@jstjohn, @skothenhill) Was this supposed to be only on the child?
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=list)
    # Used if initializing from a checkpoint, set this to any fields you want to override rather than re-set.
    #  by default all fields will be overridden.
    override_parent_fields: List[str] = field(default_factory=lambda: _OVERRIDE_BIOBERT_CONFIG_DEFAULTS)
    return_embeddings: bool = False
    include_embeddings: bool = False
    return_only_hidden_states: bool = False
    include_hiddens: bool = False  # Include hidden layers in the output of the model
    include_input_ids: bool = False
    skip_logits: bool = False  # useful for inference
    core_attention_override: Type[torch.nn.Module] | None = None

    # loss reduction class
    loss_reduction_class: Type[MegatronLossType] = BERTMLMLossWithReduction

    # metric logging
    train_metric: Optional[TorchmetricsConfig] = None
    valid_metric: Optional[TorchmetricsConfig] = None

    def configure_model(self, tokenizer: AutoTokenizer) -> MegatronBioBertModelType:  # noqa: D102
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        # The local specs all require the standard full attention mask.
        use_full_attention_mask: bool = "transformer_engine" not in self.biobert_spec_option
        do_next_sentence = False
        if self.model_cls is None:
            raise ValueError(
                f"You must supply `model_cls` to the {type(self)} for module to initialization in `configure_model`."
            )

        if self.initial_ckpt_path:
            self.load_settings_from_checkpoint(self.initial_ckpt_path)
        model = self.model_cls(
            self,
            transformer_layer_spec=get_biobert_spec(
                self.biobert_spec_option,
                qk_layernorm=self.qk_layernorm,
                core_attention=self.core_attention_override,
            ),
            num_tokentypes=2 if do_next_sentence else 0,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            tokenizer=tokenizer,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            return_embeddings=self.return_embeddings,
            include_embeddings=self.include_embeddings,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),  # set to False for inference
            add_binary_head=do_next_sentence,
            use_full_attention_mask=use_full_attention_mask,
            include_hiddens=self.include_hiddens,
            skip_logits=self.skip_logits,
            include_input_ids=self.include_input_ids,
        )
        # TODO (@skothenhill) this is a hack to load the old checkpoint.
        # This should be removed once we have a proper checkpoint conversion
        # see NeMo/nemo/collections/llm/gpt/model/mixtral.py for how we should do it.
        # We should eventually have an adapter for nemo1 checkpoints, HF checkpoints (at least for ESM2 @georgea)
        # and an adapter may also be the right way to handle expected missing/extra keys when importing
        # a checkpoint for fine-tuning (eg ignore misisng lm_head, if not there in model, etc).
        if self.nemo1_ckpt_path is not None:
            assert self.initial_ckpt_path is None, "Mutually exclusive checkpoint path used twice"
            te_mapping = "transformer_engine" in self.biobert_spec_option.value
            with tarfile.open(self.nemo1_ckpt_path, "r") as old_ckpt:
                ckpt_file = old_ckpt.extractfile("./model_weights.ckpt")
                if ckpt_file is None:
                    raise ValueError(f"Failure to read checkpoint file: {old_ckpt}/model_weights/ckpt")
                old_weights = torch.load(ckpt_file)
                new_state_dict_from_old = {}
                for k, v in old_weights.items():
                    new_key = nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="", te_mapping=te_mapping)
                    new_state_dict_from_old[new_key] = v
                # TE adds non-null ._extra_state objects to layers, which store some kind of buffer bits
                #  so we need to allow those to pass through if we're loading from bionemo1 which did not
                #  use TE.
                model.load_state_dict(new_state_dict_from_old, strict=not te_mapping)
        if self.initial_ckpt_path is not None:
            assert self.nemo1_ckpt_path is None, "Mutually exclusive checkpoint path used twice"
            self.update_model_from_checkpoint(model, self.initial_ckpt_path)

        # TODO (@jstjohn) come up with a cleaner way in the biobert module to return hidden states.
        #  maybe a suite of options like hugging face has so a user can ask for several or only one thing.
        if self.return_only_hidden_states:
            # this applies the final layernorm in the encoder to the hidden states which was
            #  the default in nemo1.
            model.post_process = False
            model.encoder.post_process = True
            model.encoder.post_layer_norm = True
        return model

    def get_loss_reduction_class(self) -> Type[MegatronLossType]:  # noqa: D102
        # You could optionally return a different loss reduction class here based on the config settings.
        return self.loss_reduction_class
