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
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Type, TypedDict

import torch
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.collections.llm import fn
from nemo.collections.llm.fn.mixin import FNMixin
from nemo.collections.llm.peft.lora import LoRA, LoRALinear
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import ParallelLinearAdapter
from nemo.lightning.megatron_parallel import masked_token_loss
from torch import Tensor, nn

from bionemo.llm.model.biobert.model import BioBertConfig, BioBertOutput, MegatronBioBertModel
from bionemo.llm.model.loss import (
    BERTMLMLossWithReduction,
    DataParallelGroupLossAndIO,
    PerTokenLossDict,
    SameSizeLossDict,
    unreduced_token_loss_fn,
)
from bionemo.llm.utils import iomixin_utils as iom
from bionemo.llm.utils.megatron_utils import average_losses_across_data_parallel_group


# This package demonstrates how you can take a pretrained geneformer module and fine-tune the classifier
# token to output cell type predictions.

__all__: Sequence[str] = (
    "FineTuneSeqLenBioBertConfig",
    "LoRAForGeneFormerTokenRegressor",
    "MegatronBioBertFineTuneSeqLengthModel",
    "MegatronFineTuneOutput",
    "MegatronRegressionMLPHead",
    "SequenceLengthRMSEPlusBERTMLMLossWithReduction",
)


class SeqLenRmsepBatch(TypedDict):
    labels: Tensor
    attention_mask: Tensor
    loss_mask: Tensor
    num_valid_tokens_in_ub: int


class MegatronFineTuneOutput(BioBertOutput):
    """Inference output type for MegatronBioBertFineTuneSeqLengthModel."""

    regression_output: Tensor


class SequenceLengthRMSEPlusBERTMLMLossWithReduction(BERTMLMLossWithReduction):
    """Loss function."""

    def forward(
        self,
        batch: SeqLenRmsepBatch,
        forward_out: Dict[str, Tensor],
    ) -> Tuple[Tensor, PerTokenLossDict | SameSizeLossDict | DataParallelGroupLossAndIO]:
        """Computes loss of `labels` in the batch vs `token_logits` in the forward output currently.

        In the future this will be extended to handle other loss types like sequence loss if it is present in the
        forward_out and batch.

        Args:
            batch: The batch of data. Each tensor should be of shape [batch_size, *, *],
                and match the corresponding dimension for that particular key in the batch output.
                For example, the "labels" and "token_logits" key should have a tensor of shape [batch_size, sequence_length].
            forward_out: The forward output from the model. Each tensor should be of shape [batch_size, *, *]

        Taken from:
        https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L951-L976
        """
        if "labels" not in batch:
            raise ValueError("Labels not provided in the batch. These are required for this loss computation.")

        unreduced_token_loss = unreduced_token_loss_fn(forward_out["token_logits"], batch["labels"])
        regression_output = forward_out["regression_output"]
        n_tokens = batch["attention_mask"].sum(dim=-1, keepdim=True).to(dtype=regression_output.dtype)
        assert len(n_tokens.shape) == 2
        assert n_tokens.shape[-1] == 1
        rmse_loss = torch.nn.functional.mse_loss(regression_output, n_tokens)

        # TODO(@jstjohn) also handle different output keys, like the sequence loss.

        # Compute loss over "valid" tokens in the microbatch, i.e. the non-masked tokens.
        # The loss is not normalized, so you need to divide by the number of non-masked
        # tokens (loss_mask.sum()) to compute the mean loss per token.
        loss_for_microbatch, num_valid_tokens_in_microbatch = masked_token_loss(
            unreduced_token_loss, batch["loss_mask"]
        )

        # Get the context parallel size for some normalizations and reductions.
        cp_size = parallel_state.get_context_parallel_world_size()

        # If we do not drop the last partial batch of validation, we need to do fancy reduction handling to support
        #  reducing the loss across the data parallel group.
        if self.validation_step and not self.val_drop_last:
            if loss_for_microbatch.isnan():
                # TODO(@jomitchell): Add a unit test for this. This is the case where there are no valid tokens in the microbatch for the loss
                #  to be computed over, so we expect a NaN loss (divide by zero for a mean) but we make this an expected and non-breaking case,
                #  re-defining it as a 0 loss. This is standard in NeMo/NeMo2.
                if batch["loss_mask"].count_nonzero() != 0:
                    raise ValueError("Got NaN loss with non-empty input")
                loss_sum_for_microbatch = torch.zeros_like(num_valid_tokens_in_microbatch)
            else:
                loss_sum_for_microbatch = loss_for_microbatch

            # In this case we need to store the loss sum as well as the number of valid tokens in the microbatch.
            loss_sum_and_microbatch_size_all_gpu = torch.cat(
                [
                    loss_sum_for_microbatch.clone().detach().view(1),
                    torch.tensor([num_valid_tokens_in_microbatch]).cuda().clone().detach(),
                ]
            )
            torch.distributed.all_reduce(
                loss_sum_and_microbatch_size_all_gpu,
                group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            return loss_for_microbatch * cp_size, {
                "loss_sum_and_microbatch_size": loss_sum_and_microbatch_size_all_gpu
            }

        loss_for_microbatch = loss_for_microbatch + rmse_loss  # add in the RMSE loss after reducing the logit loss

        # average the losses across the data parallel group, but also return the unreduced loss
        reduced_loss: Tensor = (
            average_losses_across_data_parallel_group([loss_for_microbatch], with_context_parallel=True)
            / num_valid_tokens_in_microbatch
        )
        return loss_for_microbatch * cp_size, {"avg": reduced_loss}


class MegatronRegressionMLPHead(MegatronModule):
    """A megatron MLP head."""

    def __init__(self, config: TransformerConfig):
        """Constructor."""
        super().__init__(config)
        # FC layer over just the [CLS] token embedding
        # TODO use bias/activation fusion if requested
        self.linear_fc1 = nn.Linear(in_features=config.hidden_size, out_features=config.ffn_hidden_size)
        self.activation_function = config.activation_func
        self.linear_fc2 = nn.Linear(in_features=config.ffn_hidden_size, out_features=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Inference."""
        return self.linear_fc2(self.activation_function(self.linear_fc1(hidden_states)))


class MegatronBioBertFineTuneSeqLengthModel(MegatronBioBertModel):
    """Megatron model for biobert finetuning with sequence length."""

    def __init__(self, config, *args, include_hiddens: bool = False, post_process: bool = True, **kwargs):
        """Constructor."""
        super().__init__(config, *args, include_hiddens=True, post_process=post_process, **kwargs)
        self.include_hiddens_finetuning = (
            include_hiddens  # this include_hiddens is for the final output of fine-tuning
        )
        # If post_process is True that means that we are at the last megatron parallelism stage and we can
        #   apply the head.
        if post_process:
            # if we are doing post process (eg pipeline last stage) then we need to add the output layers
            self.regression_head = MegatronRegressionMLPHead(config)

    def forward(self, *args, **kwargs) -> MegatronFineTuneOutput | BioBertOutput | Tensor:
        """Inference."""
        output: MegatronFineTuneOutput | BioBertOutput | Tensor = super().forward(*args, **kwargs)
        # Stop early if we are not in post_process mode (for example if we are in the middle of model parallelism)
        if not self.post_process:
            return output  # we are not at the last pipeline stage so just return what the parent has
        # Double check that the output from the parent has everything we need to do prediction in this head.
        if not isinstance(output, dict) or ("hidden_states" not in output):
            raise ValueError(
                f"Expected to find 'hidden_states' in the output, and output to be dictionary-like, found {output},\n"
                "Make sure include_hiddens=True in the call to super().__init__"
            )
        # Get the hidden state from the parent output, and pull out the [CLS] token for this task
        hidden_states: Tensor = output["hidden_states"][:, 0]  # [b s h] => [b h], use [CLS] (first) token for reg
        # Predict our 1d regression target
        regression_output = self.regression_head(hidden_states)
        if not self.include_hiddens_finetuning:
            del output["hidden_states"]
        output["regression_output"] = regression_output
        return output


@dataclass
class FineTuneSeqLenBioBertConfig(
    BioBertConfig[MegatronBioBertFineTuneSeqLengthModel, SequenceLengthRMSEPlusBERTMLMLossWithReduction],
    iom.IOMixinWithGettersSetters,
):
    """BioBert fine-tuning sequence length model configuration."""

    # When overriding fields in a dataclass _always_ declare types: https://github.com/python/cpython/issues/123269
    model_cls: Type[MegatronBioBertFineTuneSeqLengthModel] = MegatronBioBertFineTuneSeqLengthModel
    # typical case is fine-tune the base biobert that doesn't have this head. If you are instead loading a checkpoint
    # that has this new head and want to keep using these weights, please drop this next line or set to []
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: ["regression_head"])

    def get_loss_reduction_class(self) -> Type[SequenceLengthRMSEPlusBERTMLMLossWithReduction]:
        """Loss function type."""
        return SequenceLengthRMSEPlusBERTMLMLossWithReduction


class LoRAForGeneFormerTokenRegressor(LoRA):
    """LoRA for Genformer Token Regression.

    There are a few tricky things here to get everything to work right:

    1. Freezing logic for the transformer has to be updated in order to not freeze the new head layers.
    2. The LoRA adapter logic has to be updated to pull the input/output sizes of the layers to be adapted from the
       modules that are passed (the previous method was compatible with nn and TE, but not megatrons tensor_parallel
       modules that are currently used by geneformer). This method contains a suggested refactor to make these methods
       a little more general and extensible with structural pattern matching as well. We should push this
       requirement onto NeMo, since we shouldn't duplicate the adapter method.
    3. There's a ton of assumptions in NeMo about which module is being called and that it inherits specific mixins.
       This could break the if it is updated from a megatron module to a torch module or something else. Functional
       calls are generally favored for this reason and some have been made here to avoid updating inheritance throughout
       the code base.
    """

    def input_size_getter(self, m: nn.Module) -> int:
        """Gets the input size of the supplied model."""
        match m:
            case object(input_size=n):
                return n
            case object(in_features=n):
                return n
            case _:
                raise ValueError(f"Module {m} does not have a supported input size calculation.")

    def output_size_getter(self, m: nn.Module) -> int:
        """Gets the output size of the supplied model."""
        match m:
            case object(output_size=n):
                return n
            case object(out_features=n):
                return n
            case _:
                raise ValueError(f"Module {m} does not have a supported output size calculation.")

    def __call__(self, model: nn.Module) -> nn.Module:
        """Inference."""
        fn.walk(model, self.selective_freeze)
        fn.walk(model, self.transform)
        return model

    def selective_freeze(self, m: nn.Module, name: str | None = None, prefix: str | None = None) -> nn.Module:
        """Freezes either 'encoder' or 'embedding' parameters of the input model (`m`) iff name is one of these."""
        if name in ["encoder", "embedding"]:
            FNMixin.freeze(m)
        return m

    def transform(self, m: nn.Module, name: str | None = None, prefix: str | None = None) -> nn.Module | LoRALinear:
        """Transforms the input model if the name is in the target modules."""
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if name in self.target_modules:
            # m.in_features and m.out_features are divided by tp_size already,
            # but in_features and out_features passed to ParallelLinearAdapter are not.
            if prefix is not None and "regression_head" in prefix:
                return m
            if name in ["linear_qkv", "linear_fc1"]:
                # Column Parallel Linear
                input_is_parallel = False
                in_features = self.input_size_getter(
                    m
                )  # TODO(@georgea) note that this could break depending on the impl of `m`
                out_features = self.output_size_getter(m) * tp_size
                # LoRA is applied after layernorm, so layernorm output must be returned
                m.return_layernorm_output = True
                # perf optimization for LoRA + SP
                if m.config.sequence_parallel and not m.ub_overlap_ag:
                    m.return_layernorm_output_gathered = True
            else:  # name in ['linear_proj', 'linear_fc2']
                # Row Parallel Linear
                input_is_parallel = True
                in_features = (
                    self.input_size_getter(m) * tp_size
                )  # TODO(@georgea) note this could break depending on the impl of `m`
                out_features = self.output_size_getter(m)

            adapter = ParallelLinearAdapter(
                in_features,
                out_features,
                self.dim,
                activation="identity",
                norm_position=None,
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(m, "config", None),
                alpha=self.alpha,
            )
            return LoRALinear(m, adapter)
        return m
