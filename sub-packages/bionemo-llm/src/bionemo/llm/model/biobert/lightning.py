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


from typing import Callable, Dict, Iterable, Optional, Protocol, Sequence, TypedDict, cast

import lightning.pytorch as pl
import torch.distributed
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from torch import Tensor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from bionemo.llm.lightning import (
    BionemoLightningModule,
    DataStep,
    ForwardStep,
    default_megatron_optimizer,
)
from bionemo.llm.model.biobert.model import BioBertConfig, MegatronBioBertModel


__all__: Sequence[str] = (
    "BertBatch",
    "BertModel",
    "SequenceBatch",
    "bert_forward_step",
    "biobert_data_step",
    "biobert_lightning_module",
    "get_batch_on_this_context_parallel_rank",
    "get_packed_seq_params",
)


class BertModel(Protocol[DataT]):
    """Interface for BERT-like models."""

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> DataT:
        """Inference for BERT-like models.

        Inference for BERT-like models require their tokenized inputs by IDs, an attention mask over the input,
        and the original sequence lengths if the sequences are packed into a dense batch.
        """
        ...


class BertBatchCore(TypedDict):
    """Input datatype for inference with BERT-like models."""

    text: Tensor
    attention_mask: Tensor


DataStepOutput = Dict[str, torch.Tensor | PackedSeqParams]
DataStepFunction = Callable[[Iterable], DataStepOutput]
ForwardStepFunction = Callable[[pl.LightningModule, DataStepOutput], DataT]


class BertBatch(BertBatchCore, total=False):
    """Input datatype for inference with BERT-like models."""

    cu_seqlens: Tensor


class SequenceBatchCore(TypedDict):
    """Input datatype for inference with BERT-like models."""

    cu_seqlens: Tensor


class SequenceBatch(SequenceBatchCore, total=False):
    """Input datatype for inference with BERT-like models."""

    cu_seqlens_argmin: Tensor
    max_seqlen: Tensor


def biobert_data_step(dataloader_iter) -> Dict[str, Tensor]:
    """Preprocesses a batch of data for the GeneFormer model, and ingest a single batch of data from the dataloader iterator.
        only necessary batch keys are subsetted and passed to the model's forward pass, and the loss forward pass, depending on stage.
        TODO document how parallel_state pipeline stages work.

    Args:
        dataloader_iter: An iterator over the dataloader.

    Returns:
        output: A dictionary of this batch limiting to relevant keys.

    """  # noqa: D205
    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    if isinstance(batch, tuple) and len(batch) == 3:
        _batch: dict = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")
    if parallel_state.is_pipeline_first_stage():
        required_keys.add("text")
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask", "types", "is_random"))
    # if self.get_attention_mask_from_fusion:
    #     required_keys.remove('attention_mask')

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def bert_forward_step(model: BertModel[DataT], batch: BertBatch) -> DataT:
    """Performs the model's forward pass using the batch, for Megatron compatibility.

    This subsets the batch keys to the ones actually used by forward pass of the model, and then calls the model's
    forward pass. if "cu_seqsens" are defined in the batch, then the packed sequence parameters are also passed to the
    model for forward pass efficiency.
    """
    if "cu_seqlens" in batch:
        forward_results = model.forward(
            input_ids=batch["text"],
            attention_mask=batch["attention_mask"],
            packed_seq_params=get_packed_seq_params(cast(SequenceBatch, batch)),
        )
    else:
        forward_results = model.forward(input_ids=batch["text"], attention_mask=batch["attention_mask"])
    # TODO support losses that also include the binary head, this means doing something more fancy than the one
    #      default GPT reduction function above MaskedTokenLossReduction()
    return forward_results


def biobert_lightning_module(
    config: BioBertConfig[MegatronBioBertModel, MegatronLossReduction],
    optimizer: Optional[MegatronOptimizerModule] = None,
    tokenizer: Optional[TokenizerSpec | PreTrainedTokenizerBase] = None,
    data_step: DataStep = biobert_data_step,
    forward_step: ForwardStep = bert_forward_step,
    model_transform: Optional[Callable] = None,
    **model_construct_args,
) -> BionemoLightningModule[MegatronBioBertModel, MegatronLossReduction]:
    """A pytorch lightning module for BioBert-derived models.

    This module is designed to be used with the Megatron-LM strategy and nemo 2.0 conventions.
    To change your loss, pass in a different config object that returns a different loss reduction class.
    To change your model and what it outputs, pass in a different config object that returns a different model.
    Do not modify this function unless you need to change higher level logic. You may need to modify the various step
    and forward functions towards the bottom of this file to handle new/different keys in the batch. In the future some
    of those functions may need to be refactored out into the config object or a different place so that they live
    closer to the model definition.
    """
    return BionemoLightningModule(
        config=config,
        optimizer=optimizer if optimizer is not None else default_megatron_optimizer(),
        data_step=data_step,
        forward_step=forward_step,
        tokenizer=tokenizer,
        model_transform=model_transform,
        **model_construct_args,
    )


def get_batch_on_this_context_parallel_rank(batch: Dict[str, Tensor], in_place: bool = True) -> Dict[str, Tensor]:
    """Ensures that the input batch is in the right format for context parallel rank.

    Modifies the batch data based on the context parallel rank, if the context parallel world size is greater than 1.
    Otherwise, the batch is returned as-is.


    Args:
        batch: The input batch data.
        in_place: If true, then the input is mutated. The returned dict is a reference to the input.
                  Otherwise, the input data is always shallow-copied and this copy is modified and returned.

    Returns:
        dict: The modified batch data based on the context parallel rank.
    """
    if not in_place:
        batch: dict[str, Tensor] = dict(**batch)

    if cp_size := parallel_state.get_context_parallel_world_size() > 1:
        num_valid_tokens_in_ub: Tensor | None = None
        if "loss_mask" in batch and batch["loss_mask"] is not None:
            num_valid_tokens_in_ub = batch["loss_mask"].sum()

        cp_rank = parallel_state.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != "attention_mask" else 2
                _val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
                    non_blocking=True
                )
                _val = _val.index_select(seq_dim, index)
                _val = _val.view(*val.shape[0:seq_dim], -1, *_val.shape[(seq_dim + 2) :])
                batch[key] = _val
        batch["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub  # type: ignore

    return batch


def get_packed_seq_params(batch: SequenceBatch) -> PackedSeqParams:
    """Get the packed sequence parameters for the given batch.

    This function should only be called if `cu_seqlens` is defined in the batch.

    Args:
        batch: The input batch to pack.

    Returns:
        PackedSeqParams: The packed sequence parameters containing the following attributes:
            - cu_seqlens_q (Tensor): The sequence lengths for query.
            - cu_seqlens_kv (Tensor): The sequence lengths for key and value.
            - max_seqlen_q (Tensor, optional): The maximum sequence length for query.
            - max_seqlen_kv (Tensor, optional): The maximum sequence length for key and value.
            - qkv_format (str): The format of query, key, and value tensors.

    """
    cu_seqlens = batch["cu_seqlens"].squeeze()  # remove batch size dimension (mbs=1)
    # remove -1 "paddings" added in collate_fn
    if cu_seqlens_argmin := batch.get("cu_seqlens_argmin", None) is not None:
        # pre-compute cu_seqlens_argmin in dataset class for perf
        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

    # pre-compute max_seqlens in dataset class for perf
    max_seqlen = batch["max_seqlen"].squeeze() if "max_seqlen" in batch else None

    # these args are passed eventually into TEDotProductAttention.forward()
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )


class BioBertLightningModule(BionemoLightningModule):
    def __init__(
        self,
        *args,
        data_step_function: DataStepFunction = biobert_data_step,
        forward_step_function: ForwardStepFunction = bert_forward_step,
        **kwargs,
    ):
        """DEPRECATED! Please use BionemoLightningModule. This is here so we can load older checkpoints.
        This maps the old name `forward_step_function` to the new name `forward_step` and `data_step_function` to
        `data_step`.

        Args:
            *args: all args are passed through to BionemoLightningModule
            data_step_function (DataStepFunction, optional): The data step function. Defaults to biobert_data_step.
            forward_step_function (ForwardStepFunction, optional): The forward step function. Defaults to bert_forward_step.
            **kwargs: all other kwargs are passed through to BionemoLightningModule.
        """  # noqa: D205
        super().__init__(*args, forward_step=forward_step_function, data_step=data_step_function, **kwargs)
