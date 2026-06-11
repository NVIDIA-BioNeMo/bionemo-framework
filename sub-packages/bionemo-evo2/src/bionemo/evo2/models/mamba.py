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
from dataclasses import dataclass
from typing import Callable

import megatron.core.models.mamba.mamba_model
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from nemo.collections.llm.gpt.model.base import GPTModel, gpt_data_step
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import make_upper_case, reweighted_cross_entropy
from nemo.collections.llm.gpt.model.ssm import (
    NemotronHConfigBase,
)
from nemo.lightning import get_vocab_size
from typing_extensions import override

from bionemo.evo2.utils.loss.embedding_variance import SquaredErrorTargetedVarianceLoss


logger = logging.getLogger(__name__)


def mamba_forward_step(model, batch) -> torch.Tensor:
    """Forward step function for Mamba models, similar to hyena_forward_step.

    Args:
        model: The Mamba model
        batch: Dictionary containing input batch data

    Returns:
        torch.Tensor: Output from the model forward pass
    """
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
        "loss_mask": batch["loss_mask"],
    }
    forward_args["attention_mask"] = None
    return model(**forward_args)


class MambaModel(GPTModel):
    """Mamba model that extends GPTModel for integration with NeMo.

    Note that the loss calculation is handled by CustomMCoreMambaModel instead.
    """

    @override
    def get_inference_wrapper(
        self, params_dtype, inference_batch_times_seqlen_threshold, inference_max_seq_length=8192
    ) -> GPTInferenceWrapper:
        """Gets the inference wrapper for the Mamba model."""
        from megatron.core.models.mamba import MambaModel as MCoreMambaModel

        # Find MCoreMambaModel instance
        mcore_model = self.module
        while mcore_model:
            if isinstance(mcore_model, (MCoreMambaModel, Evo2StyleMCoreMambaModel)):
                break
            mcore_model = getattr(mcore_model, "module", None)
        if mcore_model is None or not isinstance(mcore_model, (MCoreMambaModel, Evo2StyleMCoreMambaModel)):
            raise ValueError("Mamba model instance not found in the model structure.")

        vocab_size = None
        if self.tokenizer is not None:
            vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        else:
            raise ValueError("Unable to find vocab size.")

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=mcore_model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=vocab_size,
            inference_max_seq_length=inference_max_seq_length,
        )

        model_inference_wrapper = GPTInferenceWrapper(mcore_model, inference_wrapper_config)
        return model_inference_wrapper

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        decoder_input: torch.Tensor | None = None,
        inference_context=None,
        packed_seq_params=None,
        inference_params=None,
        runtime_gather_output: bool | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass that delegates to CustomMCoreMambaModel, which handles loss calculation."""
        extra_kwargs = {"packed_seq_params": packed_seq_params} if packed_seq_params is not None else {}
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,  # Pass labels to the Megatron module
            inference_params=inference_params,
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            loss_mask=loss_mask,  # Pass loss_mask to the Megatron module
            **extra_kwargs,
        )

        # Return whatever CustomMCoreMambaModel.forward returns
        # (logits during inference, loss during training)
        return output_tensor


# Custom MCoreMambaModel with reweighted loss calculation
class Evo2StyleMCoreMambaModel(megatron.core.models.mamba.mamba_model.MambaModel):
    """Custom version of MCoreMambaModel that implements reweighted loss calculation.

    Note that this is similar to the HyenaModel for uppercase/lowercase handling.
    """

    def __init__(self, *args, **kwargs):
        """Initializes `Evo2StyleMCoreMambaModel` with unique parameters for the Evo2 variant of `MCoreMambaModel`."""
        super().__init__(*args, **kwargs)
        if self.config.use_targeted_variance_loss:
            if not hasattr(self.config, "embedding_init_method_std"):
                logger.warning("embedding_init_method_std is not supported in this config, please upgrade Megatron-LM")
            # 1.0 is the suggested value for embedding_init_method_std from the
            # [Spike No More](https://arxiv.org/abs/2312.16903) paper.
            embedding_init_method_std: float = getattr(self.config, "embedding_init_method_std", 1.0)
            self.targeted_variance_loss = SquaredErrorTargetedVarianceLoss(
                loss_coeff=self.config.targeted_variance_loss_loss_coeff,
                var_target=embedding_init_method_std**2,
            )

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inference_context=None,
        runtime_gather_output: bool | None = None,
        *,
        inference_params=None,
        loss_mask: torch.Tensor | None = None,
    ):
        """Forward pass with custom loss calculation for uppercase/lowercase reweighting.

        Note that this mimics the behavior in hyena_model.py lines 273-292.

        Forward function of the Mamba model. This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        rotary_pos_emb = None
        if self.position_embedding_type == "rope":
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Wrap decoder_input to allow the decoder (MambaBlock) to delete the
        # reference held by this caller function, enabling early garbage collection
        # for inference.
        if inference_context is not None and not self.training:
            decoder_input = WrappedTensor(decoder_input)

        # The following assert will currently fail when running inference.
        # Commented out for now.
        # TODO (duncan/rwaleffe): (1) confirm that the externally-generated
        #   attention mask is not needed and is ignored by the model in
        #   inference mode, (2) reduce the size of the externally-generated
        #   attention mask to prevent CPU OOM (as we did for training), (3)
        #   force the attention mask passed to the model in inference mode to
        #   be None, so this assert will succeed.
        # assert attention_mask is None, "The attention mask is ignored and should be set to None"

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if (
            not self.training
            and inference_context is not None
            and inference_context.materialize_only_last_token_logits
        ):
            hidden_states = hidden_states[-1, :, :].unsqueeze(0)

        logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        # Apply reweighted loss calculation for uppercase/lowercase handling
        labels, lowercase_mask = make_upper_case(labels)
        loss = self.compute_language_model_loss(labels, logits)
        normalize_per_batch = True if self.config.to_upper == "normalized_weighted" else False
        loss = reweighted_cross_entropy(
            loss,
            (labels, loss_mask, lowercase_mask),
            lowercase_weight=self.config.lowercase_loss_reweighting,
            normalize_per_batch=normalize_per_batch,
        )
        if self.training and self.config.use_targeted_variance_loss:
            # Only use this in training, not validation etc.
            var_loss = self.targeted_variance_loss(self.embedding.word_embeddings.weight)
            loss += var_loss
        return loss


def mamba_no_weight_decay_cond(name, param, exclude_embeddings: bool = False):
    """Condition for no weight decay for Mamba parameters.

    Note that this follows the same pattern as in the original Mamba implementation.
    """
    # Mamba-specific parameters that should not have weight decay
    if (
        name.endswith("dt_bias")
        or name.endswith("A_log")
        or name.endswith("D")
        or ("embedding" in name and exclude_embeddings)
        or getattr(param, "_no_weight_decay", False)
    ):
        no_wd = True
    # All other parameters - use default MCore behavior:
    # Do not regularize biases and norm parameters
    # (See megatron.core.optimizer._get_pram_groups)
    # TODO exclude embeddings
    else:
        no_wd = name.endswith(".bias") or len(param.shape) == 1
    return no_wd


def mamba_no_weight_decay_cond_with_embeddings(name, param):
    """Condition for no weight decay for Mamba parameters with embeddings.

    Note that this follows the same pattern as in the original Mamba implementation but also skips WD on embeddings.
    """
    return mamba_no_weight_decay_cond(name, param, exclude_embeddings=True)


@dataclass
class HybridMambaConfig8BEvo2Loss(NemotronHConfigBase):
    """Config for 8B hybrid Mamba model."""

    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    num_layers: int = 52
    seq_length: int = 8192
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 8
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    ffn_hidden_size: int = 21504
    num_attention_heads: int = 32
    init_method_std: float = 0.014
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128
    tokenizer_library: str = "byte-level"  # Use Evo2 tokenizer
    tokenizer_name: str = None
    masked_softmax_fusion: bool = True
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    attention_softmax_in_fp32: bool = False
    vocab_size: int = 512
    first_last_layers_bf16: bool = True
    is_hybrid_model: bool = True
    forward_step_fn: Callable = mamba_forward_step
    data_step_fn: Callable = gpt_data_step
    # Set a reasonable default for to_upper to match HyenaModel behavior
    to_upper: str = "normalized_weighted"
    # Set lowercase loss reweighting factor
    lowercase_loss_reweighting: float = 1.0
    activation_func: Callable = lambda x: torch.square(F.relu(x))  # lambda x: torch.pow(F.relu(x), 2)
    # The trainer is responsible for using this when initializing the optimizer state:
    #  opt = MegatronOptimizerModule(opt_config, sched, no_weight_decay_cond=model_config.hyena_no_weight_decay_cond_fn)
    hyena_no_weight_decay_cond_fn: Callable = mamba_no_weight_decay_cond
    spike_no_more_embedding_init: bool = False  # TODO: remove this.
    layernorm_embeddings: bool = False
    # If set to true, use targeted variance loss which encourages the word embedding weight variances
    # to be close to a target value (1.0).
    use_targeted_variance_loss: bool = False
    targeted_variance_loss_loss_coeff: float = 0.1
    share_embeddings_and_output_weights: bool = False

    def __post_init__(self):
        """Post-init logic for Evo2 to enable backwards compatibility with old configs."""
        # Specific post_init logic for Evo2 to enable backwards compatibility with old configs.
        if not hasattr(self, "embedding_init_method_std"):
            raise ValueError("embedding_init_method_std is not supported in this config, please upgrade Megatron-LM")
        if self.spike_no_more_embedding_init and self.embedding_init_method_std is None:
            logger.warning(
                "spike_no_more_embedding_init is deprecated, please set "
                "embedding_init_method_std=[desired_stdev] in the future. To get the old behavior set to 1.0. "
                "For now setting to 1.0."
            )
            self.embedding_init_method_std = 1.0
        # Continue with the remaining post-init logic defined in NemotronHConfigBase and/or TransformerConfig.
        super().__post_init__()

    @override
    def configure_model(
        self, tokenizer, pre_process=None, post_process=None, vp_stage: int | None = None
    ) -> Evo2StyleMCoreMambaModel:
        """Configures the model for training or inference."""
        mamba_stack_spec = self.mamba_stack_spec
        if not isinstance(mamba_stack_spec, ModuleSpec):
            mamba_stack_spec = mamba_stack_spec()

        assert getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None, (
            "Virtual pipeline model parallelism is temporarily unsupported in SSM/Mamaba "
            "models due to upstream MCore MambaModel API dependency"
        )
        return Evo2StyleMCoreMambaModel(
            self,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
            post_process=post_process or parallel_state.is_pipeline_last_stage(),
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
        )


# Dictionary mapping model size names to config classes
MAMBA_MODEL_OPTIONS: dict[str, type[NemotronHConfigBase]] = {
    "hybrid_mamba_8b": HybridMambaConfig8BEvo2Loss,
}
