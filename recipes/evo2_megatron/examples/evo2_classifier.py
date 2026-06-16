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

"""Sequence-classification fine-tuning of Evo2 with megatron-bridge.

Provides:

* :class:`HyenaForSequenceClassification` — Megatron ``HyenaModel`` with a small
  MLP classification head on top of pooled hidden states.
* :class:`Evo2ClassifierDatasetProvider` — a :class:`DatasetProvider` that yields
  per-example ``{input_ids, pool_mask, labels}`` batches from JSONL files.
* :func:`classifier_forward_step` — the per-microbatch step that returns
  ``(logits, loss_fn)`` where ``loss_fn`` returns ``(loss, count, reporting)``.
* :func:`evo2_1b_classifier_config` — builds a :class:`ConfigContainer`
  configured for ``pretrain()``: classifier provider + dataset provider +
  ``Evo2LoRA`` (or a no-adapter freeze-only variant for the head-only baseline)
  + checkpoint loading from a pretrained MBridge backbone.
* :func:`predict` — self-contained post-training inference helper that reads the
  saved ``run_config.yaml`` to rebuild the model and (if applicable) reapply the
  PEFT structure before overlaying the trained adapter+head tensors.
* CLI ``main`` for launching training via ``torchrun evo2_classifier.py …``.
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.checkpointing import (
    _generate_model_state_dict,
    _load_model_weights_from_checkpoint,
    apply_peft_adapter_filter_to_state_dict,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DatasetBuildContext,
    DatasetProvider,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.checkpoint_utils import (
    get_checkpoint_run_config_filename,
    read_run_config,
)
from megatron.bridge.utils.instantiate_utils import instantiate, register_allowed_target_prefix
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import HuggingFaceTokenizer
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import ModuleSpec
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
from bionemo.evo2.models.evo2_lora import Evo2LoRA
from bionemo.evo2.models.evo2_provider import (
    Hyena1bModelProvider,
    HyenaModelProvider,
    HyenaOptimizerConfigOverrideProvider,
)
from bionemo.evo2.models.megatron.hyena.hyena_layer_specs import get_hyena_stack_spec
from bionemo.evo2.models.megatron.hyena.hyena_model import HyenaModel as MCoreHyenaModel
from bionemo.evo2.run.predict import initialize_inference_distributed, resolve_checkpoint_path


logger: logging.Logger = logging.getLogger(__name__)


# This example is launched as ``evo2_classifier.py`` (e.g. ``torchrun ... evo2_classifier.py``), so
# the providers defined below are serialized into a trained checkpoint's run_config with a
# ``_target_`` of ``evo2_classifier.<Provider>``. Megatron-Bridge's ``instantiate`` only resolves
# targets under an allow-listed module prefix, so register this module's prefix here (mirroring
# ``bionemo.evo2.`` in evo2_provider.py). Without it, rebuilding the model at predict time in
# ``_build_classifier_from_checkpoint`` raises InstantiationException.
register_allowed_target_prefix("evo2_classifier.")


# ─────────────────────────────────────────────────────────────────────────────
# Model: subclass of HyenaModel with a classification head
# ─────────────────────────────────────────────────────────────────────────────


class HyenaForSequenceClassification(MCoreHyenaModel):
    """Hyena backbone with a sequence-classification head.

    A small MLP is applied on top of a pooled hidden-state representation
    taken from the last decoder layer's output. Pooling is mean-over-sequence
    by default; an optional per-token ``pool_mask`` lets the caller exclude
    padding positions.

    The base ``output_layer`` (and its tie to the input embedding) is left in
    place so MBridge pretrained checkpoints load without needing custom
    handling for missing/unexpected keys — we simply do not call it during
    classification.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        classifier_hidden_size: Optional[int] = None,
        classifier_dropout: float = 0.1,
        pool: str = "mean",
        **hyena_kwargs,
    ) -> None:
        """Build the Hyena backbone (via the parent ``__init__``) and the head MLP."""
        super().__init__(**hyena_kwargs)
        if pool not in ("mean", "last"):
            raise ValueError(f"pool must be 'mean' or 'last', got {pool!r}")
        self.num_classes = num_classes
        self.pool = pool

        hidden = self.transformer_config.hidden_size
        head_hidden = classifier_hidden_size or hidden

        self.classification_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(head_hidden, num_classes),
        )
        self._init_classification_head()

    def _init_classification_head(self) -> None:
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                std = 1.0 / math.sqrt(module.in_features)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pool(self, hidden_states_bsh: Tensor, pool_mask: Optional[Tensor]) -> Tensor:
        if self.pool == "mean":
            if pool_mask is None:
                return hidden_states_bsh.mean(dim=1)
            mask = pool_mask.to(hidden_states_bsh.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (hidden_states_bsh * mask).sum(dim=1) / denom
        if pool_mask is None:
            return hidden_states_bsh[:, -1, :]
        last_idx = pool_mask.long().sum(dim=1).clamp(min=1) - 1
        gather = last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states_bsh.size(-1))
        return hidden_states_bsh.gather(dim=1, index=gather).squeeze(1)

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        pool_mask: Optional[Tensor] = None,
        **_unused,
    ) -> Tensor:
        """Run backbone + pool + head, returning logits.

        ``labels`` is accepted but ignored; cross-entropy is computed downstream
        in :func:`classifier_forward_step`.
        """
        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=None,
            inference_context=None,
            packed_seq_params=None,
        )

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=None,
            sequence_len_offset=sequence_len_offset,
        )

        # Megatron returns sequence-first tensors: [s, b, h] → [b, s, h]
        hidden_states_bsh = hidden_states.transpose(0, 1).contiguous()
        pooled = self._pool(hidden_states_bsh, pool_mask)
        head_dtype = next(self.classification_head.parameters()).dtype
        return self.classification_head(pooled.to(head_dtype))


@dataclass
class HyenaForSequenceClassificationProvider(HyenaModelProvider):
    """Provider that builds a :class:`HyenaForSequenceClassification`.

    Subclasses of this dataclass should also subclass a size-specific
    provider (e.g. :class:`Hyena1bModelProvider`) so all backbone
    hyperparameters are fixed to their pretrained values.
    """

    num_classes: int = 2
    classifier_hidden_size: Optional[int] = None
    classifier_dropout: float = 0.1
    pool: str = "mean"

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> HyenaForSequenceClassification:
        """Construct the classification model with the configured backbone."""
        self.bias_activation_fusion = False if self.remove_activation_post_first_layer else self.bias_activation_fusion

        assert getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None, (
            "Virtual pipeline model parallelism is unsupported in Hyena classifier."
        )
        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()."

        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        hyena_stack_spec: ModuleSpec = get_hyena_stack_spec(
            use_te=self.use_te,
            vortex_style_fp8=self.vortex_style_fp8,
            unfused_rmsnorm=self.unfused_rmsnorm,
            plain_row_linear=self.plain_row_linear,
        )

        model = HyenaForSequenceClassification(
            num_classes=self.num_classes,
            classifier_hidden_size=self.classifier_hidden_size,
            classifier_dropout=self.classifier_dropout,
            pool=self.pool,
            transformer_config=self,
            hyena_stack_spec=hyena_stack_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self.seq_length,
            num_groups_hyena=self.num_groups_hyena,
            num_groups_hyena_medium=self.num_groups_hyena_medium,
            num_groups_hyena_short=self.num_groups_hyena_short,
            hybrid_override_pattern=self.hybrid_override_pattern,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=(
                False
                if self.pre_process is False
                else (pre_process if pre_process is not None else parallel_state.is_pipeline_first_stage())
            ),
            post_process=(
                False
                if self.post_process is False
                else (post_process if post_process is not None else parallel_state.is_pipeline_last_stage())
            ),
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            hyena_init_method=self.hyena_init_method,
            hyena_output_layer_init_method=self.hyena_output_layer_init_method,
            remove_activation_post_first_layer=self.remove_activation_post_first_layer,
            add_attn_proj_bias=self.add_attn_proj_bias,
        )
        return model


@dataclass
class Hyena1bClassifierProvider(Hyena1bModelProvider, HyenaForSequenceClassificationProvider):
    """1B Evo2 backbone with a classification head."""


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> tuple[list[str], list[int]]:
    sequences: list[str] = []
    labels: list[int] = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            sequences.append(obj["sequence"])
            labels.append(int(obj["label"]))
    return sequences, labels


class Evo2ClassifierDataset(Dataset):
    """A torch ``Dataset`` of tokenized DNA sequences with sequence-level labels.

    Each ``__getitem__`` returns a dict with ``input_ids`` (long ``[S]``),
    ``pool_mask`` (float ``[S]`` — 1 inside, 0 on padding) and ``labels``
    (scalar long).
    """

    def __init__(
        self,
        sequences: Sequence[str],
        labels: Sequence[int],
        tokenizer: HuggingFaceTokenizer,
        seq_length: int,
        pad_token_id: Optional[int] = None,
    ) -> None:
        """Tokenize once into pre-allocated tensors so ``__getitem__`` is just a slice."""
        if pad_token_id is None:
            try:
                pad_token_id = tokenizer.pad_id
            except (NotImplementedError, AttributeError):
                pad_token_id = None
            if pad_token_id is None:
                pad_token_id = 1

        n = len(sequences)
        if len(labels) != n:
            raise ValueError(f"sequences/labels length mismatch: {n} vs {len(labels)}")

        self.input_ids = torch.full((n, seq_length), pad_token_id, dtype=torch.long)
        self.pool_mask = torch.zeros((n, seq_length), dtype=torch.float32)
        self.labels = torch.tensor(list(labels), dtype=torch.long)
        self.seq_length = seq_length

        for i, s in enumerate(sequences):
            ids = tokenizer.tokenize(s) if hasattr(tokenizer, "tokenize") else tokenizer.text_to_ids(s)
            ids = list(ids)[:seq_length]
            n_ids = len(ids)
            self.input_ids[i, :n_ids] = torch.tensor(ids, dtype=torch.long)
            self.pool_mask[i, :n_ids] = 1.0

    def __len__(self) -> int:
        """Return the number of examples in the split."""
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return the ``idx``-th example as a dict of tensors."""
        return {
            "input_ids": self.input_ids[idx],
            "pool_mask": self.pool_mask[idx],
            "labels": self.labels[idx],
        }


@dataclass
class Evo2ClassifierDatasetProvider(DatasetProvider):
    """:class:`DatasetProvider` for sequence-classification fine-tuning.

    Reads ``{sequence, label}`` JSONL files for each split, tokenizes once
    with the HuggingFace fast tokenizer that ships with the Evo2 recipe, and
    returns three :class:`Evo2ClassifierDataset` instances. The framework's
    default torch ``DataLoader`` collates the dicts cleanly because every
    field is a tensor of consistent shape.
    """

    train_jsonl: Optional[str] = None
    val_jsonl: Optional[str] = None
    test_jsonl: Optional[str] = None
    seq_length: int = 512
    tokenizer_path: str = DEFAULT_HF_TOKENIZER_MODEL_PATH_512
    pad_token_id: Optional[int] = None
    dataloader_type: str = "cyclic"

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Tokenize the JSONL splits into :class:`Evo2ClassifierDataset` objects."""
        tokenizer = context.tokenizer if context.tokenizer is not None else HuggingFaceTokenizer(self.tokenizer_path)

        def _build(path: Optional[str]) -> Optional[Evo2ClassifierDataset]:
            if path is None:
                return None
            sequences, labels = _read_jsonl(path)
            return Evo2ClassifierDataset(
                sequences=sequences,
                labels=labels,
                tokenizer=tokenizer,
                seq_length=self.seq_length,
                pad_token_id=self.pad_token_id,
            )

        return _build(self.train_jsonl), _build(self.val_jsonl), _build(self.test_jsonl)


# ─────────────────────────────────────────────────────────────────────────────
# Forward step
# ─────────────────────────────────────────────────────────────────────────────


def _classification_loss_fn(labels: Tensor, output_tensor: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Sum-of-CE loss with batch-size as the averaging count.

    The bridge averages microbatch losses by the returned count, so summing
    here yields a per-sample mean once the framework divides through.
    """
    losses = F.cross_entropy(output_tensor.float(), labels.long(), reduction="none")
    loss = losses.sum()
    n = torch.tensor(labels.numel(), dtype=torch.int, device=labels.device)
    correct = (output_tensor.argmax(dim=-1) == labels.long()).sum()
    reporting = torch.cat([loss.detach().clone().view(1), n.view(1)])
    accuracy_reporting = torch.cat([correct.detach().clone().view(1), n.view(1)])
    return loss, n, {"ce loss": reporting, "accuracy": accuracy_reporting}


def classifier_forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: HyenaForSequenceClassification,
    return_schedule_plan: bool = False,
) -> tuple[Tensor, partial]:
    """Per-microbatch closure invoked by :func:`pretrain`.

    Pulls one batch from ``data_iterator``, runs the model to obtain logits,
    and returns ``(logits, partial(loss_fn, labels))``. The framework then
    calls the loss closure with ``logits`` to get ``(loss, count, reporting)``.
    """
    if return_schedule_plan:
        raise NotImplementedError("Schedule plans are not used by the classifier forward step.")

    batch = next(data_iterator)
    input_ids = batch["input_ids"].cuda(non_blocking=True)
    pool_mask = batch["pool_mask"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)

    logits = model(input_ids=input_ids, pool_mask=pool_mask)
    return logits, partial(_classification_loss_fn, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────


# Default LoRA targets attention and MLP projections. The Hyena 1B hybrid
# pattern is "SDH*SDHSDH*SDHSDH*SDHSDH*" with 4 attention layers; including
# linear_qkv / linear_proj puts adapters on those, while linear_fc1 / fc2
# additionally target every MLP block. dense_projection / dense add adapters
# inside the Hyena mixer.
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
    "dense_projection",
    "dense",
)
HEAD_SKIP_PATTERN = "*classification_head*"
# Pattern that matches no real Hyena/Evo2 module — used by the head-only
# baseline to invoke Evo2LoRA's freeze + pretrained-checkpoint-load logic
# without wrapping any module in a LoRA adapter.
_NO_MATCH_PATTERN = "__none_zzz__"


def evo2_1b_classifier_config(  # noqa: D417
    *,
    base_ckpt_dir: Path,
    train_jsonl: Path,
    val_jsonl: Optional[Path],
    test_jsonl: Optional[Path],
    num_classes: int,
    result_dir: Path,
    experiment_name: str,
    seq_length_tokens: int = 512,
    backbone_seq_length: int = 8192,
    train_iters: int = 300,
    global_batch_size: int = 8,
    micro_batch_size: int = 2,
    lr: float = 1e-3,
    min_lr: float = 1e-4,
    warmup_iters: int = 20,
    decay_steps: Optional[int] = None,
    weight_decay: float = 0.0,
    eval_interval: int = 50,
    eval_iters: int = 10,
    save_interval: Optional[int] = None,
    log_interval: int = 10,
    seed: int = 1234,
    classifier_hidden_size: Optional[int] = None,
    classifier_dropout: float = 0.1,
    pool: str = "mean",
    use_lora: bool = True,
    lora_dim: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
    tokenizer_path: str = DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
    no_activation_checkpointing: bool = True,
    precision_recipe: str = "bf16_mixed",
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_save_dir: Optional[Path] = None,
) -> ConfigContainer:
    """Build a :class:`ConfigContainer` for ``pretrain()`` + :func:`classifier_forward_step`.

    Args:
        base_ckpt_dir: Path to a pretrained MBridge backbone checkpoint
            (parent of ``iter_*`` subdirs is auto-resolved).
        num_classes: Number of classification labels.
        use_lora: If ``False``, run the head-only baseline — backbone is
            loaded and frozen, but no LoRA adapters are added.
        no_activation_checkpointing: Disable Hyena's full recompute. Short
            sequences fit comfortably without it.

    Remaining keyword arguments are forwarded to the corresponding fields of
    the underlying mbridge config dataclasses (``TrainingConfig``,
    ``CheckpointConfig``, ``LoggerConfig``, the LoRA config, etc.).
    """
    base_output_dir = result_dir / experiment_name
    checkpoint_dir = base_output_dir / "checkpoints"
    tensorboard_dir = base_output_dir / "tb_logs"

    if num_classes <= 1:
        raise ValueError(f"num_classes must be ≥ 2 for classification, got {num_classes}.")

    # ── Model provider ────────────────────────────────────────────────────
    model_cfg = Hyena1bClassifierProvider(
        num_classes=num_classes,
        classifier_hidden_size=classifier_hidden_size,
        classifier_dropout=classifier_dropout,
        pool=pool,
        seq_length=backbone_seq_length,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        pipeline_dtype=torch.bfloat16,
        perform_initialization=True,
    )
    if no_activation_checkpointing:
        model_cfg.recompute_granularity = None
        model_cfg.recompute_method = None
        model_cfg.recompute_num_layers = None

    # ── Dataset provider ──────────────────────────────────────────────────
    dataset_cfg = Evo2ClassifierDatasetProvider(
        train_jsonl=str(train_jsonl),
        val_jsonl=str(val_jsonl) if val_jsonl is not None else None,
        test_jsonl=str(test_jsonl) if test_jsonl is not None else None,
        seq_length=seq_length_tokens,
        tokenizer_path=tokenizer_path,
        num_workers=0,  # tokenization is in-memory, workers add only overhead
        persistent_workers=False,
        pin_memory=True,
        data_sharding=False,
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    resolved_decay_steps = decay_steps if decay_steps is not None else max(train_iters - warmup_iters, 1)
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=warmup_iters,
        lr_decay_iters=resolved_decay_steps,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=weight_decay,
        max_lr=lr,
        min_lr=min_lr,
    )

    # ── PEFT ──────────────────────────────────────────────────────────────
    if use_lora:
        peft = Evo2LoRA(
            target_modules=list(lora_target_modules),
            dim=lora_dim,
            alpha=lora_alpha,
            dropout=lora_dropout,
            skip_freeze_modules=[HEAD_SKIP_PATTERN],
        )
    else:
        # Head-only baseline: a target_modules pattern that matches nothing,
        # combined with the head's skip-freeze pattern, gives us "freeze the
        # backbone, leave the head trainable, add zero adapters".
        peft = Evo2LoRA(
            target_modules=[_NO_MATCH_PATTERN],
            dim=1,
            alpha=1,
            dropout=0.0,
            skip_freeze_modules=[HEAD_SKIP_PATTERN],
        )

    # ── Training / logging / checkpoint configs ──────────────────────────
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_config,
        optimizer_config_override_provider=HyenaOptimizerConfigOverrideProvider(no_weight_decay_embeddings=False),
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            align_param_gather=False,
            use_distributed_optimizer=True,
        ),
        dataset=dataset_cfg,
        logger=LoggerConfig(
            log_interval=log_interval,
            tensorboard_dir=str(tensorboard_dir),
            log_params_norm=False,
            log_throughput=False,
            log_progress=True,
            wandb_project=wandb_project or "",
            wandb_exp_name=wandb_run_name or experiment_name,
            wandb_entity=wandb_entity or "",
            wandb_save_dir=str(wandb_save_dir) if wandb_save_dir is not None else "",
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=str(tokenizer_path),
        ),
        checkpoint=CheckpointConfig(
            save_interval=save_interval if save_interval is not None else max(train_iters, eval_interval),
            save=str(checkpoint_dir),
            load=str(checkpoint_dir),
            ckpt_format="torch_dist",
            fully_parallel_load=True,
            dist_ckpt_optim_fully_reshardable=False,
            finetune=True,
            # For megatron-bridge <= 0.3.0, pass parent_dir of iter_* subdir. Otherwise, model weights
            # won't be initialized correctly.
            pretrained_checkpoint=str(base_ckpt_dir),
            dist_ckpt_strictness="ignore_all",  # head + (sometimes) LoRA params aren't in the base ckpt
            exit_on_missing_checkpoint=False,
        ),
        rng=RNGConfig(seed=seed),
        mixed_precision=get_mixed_precision_config(precision_recipe),
        peft=peft,
    )
    cfg.dataset.num_workers = 0
    return cfg


def train_classifier(cfg: ConfigContainer) -> None:
    """Run :func:`pretrain` with :func:`classifier_forward_step`."""
    pretrain(cfg, classifier_forward_step)


# ─────────────────────────────────────────────────────────────────────────────
# Inference (post-training)
# ─────────────────────────────────────────────────────────────────────────────


_LORA_PARAM_TOKENS: tuple[str, ...] = (".linear_in.", ".linear_out.", ".adapter.")


def count_classifier_params(model: nn.Module) -> dict[str, int]:
    """Count parameters by structural role: head, LoRA adapters, total.

    Megatron-Bridge LoRA produces two parameter-naming patterns depending on
    the wrapped class. ``LoRALinear``/``TEFusedLoRALinear`` wrappers nest the
    adapter under ``.adapter.``, while ``LinearAdapter``/``TELinearAdapter``
    subclass the original linear and attach ``linear_in``/``linear_out``
    directly at the wrapped FQN with no ``.adapter.`` prefix. Match all three
    tokens so neither variant is undercounted.
    """
    total = 0
    head = 0
    adapter = 0
    for name, p in model.named_parameters():
        total += p.numel()
        lower_name = name.lower()
        if "classification_head" in name:
            head += p.numel()
        elif any(tok in lower_name for tok in _LORA_PARAM_TOKENS):
            adapter += p.numel()
    fine_tuned = head + adapter
    return {
        "total": total,
        "head": head,
        "lora_adapters": adapter,
        "fine_tuned": fine_tuned,
        "fine_tuned_fraction": fine_tuned / total if total else 0.0,
    }


def _cleanup_inference_distributed() -> None:
    """Tear down parallel_state, the microbatch calculator, and the process group."""
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    destroy_num_microbatches_calculator()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _build_classifier_from_checkpoint(
    trained_ckpt_dir: Path,
) -> tuple[list[nn.Module], HuggingFaceTokenizer, int]:
    """Rebuild a trained classifier model from a saved checkpoint.

    Reads the ``run_config.yaml`` saved alongside the trained checkpoint to
    recover model hyperparameters, the tokenizer, the base-backbone path, and
    the PEFT config (if any). Returns the loaded model, the tokenizer used at
    training time, and the per-example token sequence length.
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve and load configuration from checkpoint
    # -------------------------------------------------------------------------
    resolved_ckpt_dir = resolve_checkpoint_path(trained_ckpt_dir)
    run_config_filename = get_checkpoint_run_config_filename(str(resolved_ckpt_dir))
    run_config = read_run_config(run_config_filename)
    model_provider = instantiate(run_config["model"])
    logger.info(f"Instantiated model provider: {type(model_provider).__name__}")

    # -------------------------------------------------------------------------
    # Step 2: Override parallelism and precision settings
    # -------------------------------------------------------------------------
    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.context_parallel_size = 1
    model_provider.sequence_parallel = False

    if "mixed_precision" in run_config and run_config["mixed_precision"] is not None:
        mp_value = run_config["mixed_precision"]
        if isinstance(mp_value, str):
            mp_config = get_mixed_precision_config(mp_value)
            logger.info(f"Using mixed precision recipe from checkpoint: {mp_value}")
        else:
            mp_config = instantiate(mp_value)
            logger.info("Using mixed precision config from checkpoint")
    else:
        mp_config = get_mixed_precision_config("bf16_mixed")

    mp_config.finalize()
    mp_config.setup(model_provider)

    # -------------------------------------------------------------------------
    # Step 3: Load tokenizer
    # -------------------------------------------------------------------------
    tokenizer_dir = resolved_ckpt_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = HuggingFaceTokenizer(str(tokenizer_dir))
    else:
        tokenizer = HuggingFaceTokenizer(DEFAULT_HF_TOKENIZER_MODEL_PATH_512)

    model_provider.vocab_size = tokenizer.vocab_size
    model_provider.should_pad_vocab = True

    # -------------------------------------------------------------------------
    # Step 4: Initialize distributed environment
    # -------------------------------------------------------------------------
    if not parallel_state.model_parallel_is_initialized():
        rng_config = instantiate(run_config.get("rng")) if run_config.get("rng") else RNGConfig(seed=1234)
        dist_config = instantiate(run_config.get("dist")) if run_config.get("dist") else DistributedInitConfig()
        destroy_num_microbatches_calculator()
        initialize_inference_distributed(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            micro_batch_size=1,
            global_batch_size=1,
            rng_config=rng_config,
            dist_config=dist_config,
        )
        logger.info("Initialized distributed environment")

    # -------------------------------------------------------------------------
    # Step 5: Create model and load weights
    # -------------------------------------------------------------------------
    logger.info("Creating model...")
    model_provider.finalize()

    model = model_provider.provide_distributed_model(
        ddp_config=None,
        wrap_with_ddp=False,
        data_parallel_random_init=False,
        bf16=mp_config.bf16,
        fp16=mp_config.fp16,
        mixed_precision_wrapper=Float16Module if (mp_config.bf16 or mp_config.fp16) else None,
    )

    for model_module in model:
        model_module.eval()

    peft_section = run_config.get("peft")
    if peft_section is not None:
        pretrained_ckpt = resolve_checkpoint_path(Path(run_config["checkpoint"]["pretrained_checkpoint"]))
        logger.info(f"Loading base model weights from: {pretrained_ckpt}")
        _load_model_weights_from_checkpoint(
            checkpoint_path=str(pretrained_ckpt),
            model=model,
            dist_ckpt_strictness="ignore_all",
        )

        unwrapped = [m.module if hasattr(m, "module") else m for m in model]
        peft_cfg = instantiate(peft_section)
        # Apply with training=True so adapters + skip-freeze (classification head)
        # keep requires_grad=True; this is what set_params_to_save reads. Calling
        # with training=False would freeze everything first and leave params_to_save
        # empty, dropping the head and LinearAdapter-style LoRA from the load filter.
        peft_cfg(unwrapped, training=True)
        peft_cfg.set_params_to_save(unwrapped)
        for chunk in unwrapped:
            for p in chunk.parameters():
                p.requires_grad = False
            chunk.eval()

        logger.info(f"Loading adapter weights from: {resolved_ckpt_dir}")
        sharded_sd = _generate_model_state_dict(unwrapped, {})
        sharded_sd = apply_peft_adapter_filter_to_state_dict(sharded_sd, peft_cfg)
        loaded = dist_checkpointing.load(sharded_sd, str(resolved_ckpt_dir), strict="ignore_all")
        if len(unwrapped) == 1:
            unwrapped[0].load_state_dict(loaded["model"], strict=False)
        else:
            for i, inner in enumerate(unwrapped):
                inner.load_state_dict(loaded[f"model{i}"], strict=False)
    else:
        logger.info(f"Loading weights from: {resolved_ckpt_dir}")
        _load_model_weights_from_checkpoint(
            checkpoint_path=str(resolved_ckpt_dir),
            model=model,
            dist_ckpt_strictness="ignore_all",
        )
    logger.info("Weights loaded successfully")

    dataset_section: dict = run_config.get("dataset", {})
    seq_length_tokens = int(dataset_section["seq_length"])

    return model, tokenizer, seq_length_tokens


def count_params_from_checkpoint(trained_ckpt_dir: Path) -> dict[str, int]:
    """Load a trained classifier and return its parameter breakdown.

    Convenience wrapper around :func:`count_classifier_params` that handles
    distributed init + checkpoint loading + teardown. Use this when you want
    the head / LoRA-adapter / total counts without running inference.
    """
    try:
        model, _tokenizer, _seq_length_tokens = _build_classifier_from_checkpoint(trained_ckpt_dir)
        model_module = model[0].module if hasattr(model[0], "module") else model[0]
        return count_classifier_params(model_module)
    finally:
        _cleanup_inference_distributed()


def _classifier_predict_step(
    model: torch.nn.Module,
    batch: dict[str, Tensor],
) -> Optional[dict[str, Tensor]]:
    """Run a single classifier prediction step."""
    if not parallel_state.is_pipeline_last_stage():
        return None

    logits = model(input_ids=batch["input_ids"], pool_mask=batch["pool_mask"])
    return {"logits": logits}


def predict(
    *,
    trained_ckpt_dir: Path,
    sequences: Sequence[str],
    micro_batch_size: int = 8,
    return_logits: bool = False,
) -> dict[str, Any]:
    """Run inference using a trained classifier checkpoint.

    Args:
        trained_ckpt_dir: Path to a saved ``iter_*`` checkpoint (parent
            directory with ``iter_*`` subdirs is auto-resolved).
        sequences: List of DNA strings to classify.
        micro_batch_size: Forward-pass batch size.
        return_logits: If True, include the full ``[N, num_classes]`` logits
            tensor in the result alongside the argmax predictions.

    Returns:
        Dict with ``predictions`` (``[N]`` long tensor of class IDs), and
        optionally ``logits`` when ``return_logits=True``. Use
        :func:`count_params_from_checkpoint` for the parameter breakdown.
    """
    try:
        # -------------------------------------------------------------------------
        # Step 1-5: Build model, tokenizer, and initialize distributed environment
        # -------------------------------------------------------------------------
        logger.info(f"Loading classifier checkpoint from: {trained_ckpt_dir}")
        model, tokenizer, seq_length_tokens = _build_classifier_from_checkpoint(trained_ckpt_dir)

        # -------------------------------------------------------------------------
        # Step 6: Create dataset and dataloader
        # -------------------------------------------------------------------------
        dataset = Evo2ClassifierDataset(
            sequences=sequences,
            labels=[0] * len(sequences),
            tokenizer=tokenizer,
            seq_length=seq_length_tokens,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        # -------------------------------------------------------------------------
        # Step 7: Run prediction loop
        # -------------------------------------------------------------------------
        logger.info("Starting prediction loop...")
        outputs: list[Tensor] = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                batch_gpu = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()
                }

                step_out = _classifier_predict_step(model=model[0], batch=batch_gpu)
                if step_out is not None:
                    outputs.append(step_out["logits"].cpu())

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

        logits_all = torch.cat(outputs, dim=0) if outputs else torch.empty(0)
        result: dict[str, Any] = {"predictions": logits_all.argmax(dim=-1)}
        if return_logits:
            result["logits"] = logits_all
        logger.info("Prediction complete!")
        return result
    finally:
        _cleanup_inference_distributed()


__all__ = [
    "DEFAULT_LORA_TARGET_MODULES",
    "Evo2ClassifierDataset",
    "Evo2ClassifierDatasetProvider",
    "Hyena1bClassifierProvider",
    "HyenaForSequenceClassification",
    "HyenaForSequenceClassificationProvider",
    "classifier_forward_step",
    "count_classifier_params",
    "count_params_from_checkpoint",
    "evo2_1b_classifier_config",
    "main",
    "predict",
    "train_classifier",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — used by the notebook via `torchrun evo2_classifier.py ...`
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Evo2 for sequence classification using megatron-bridge pretrain().",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-jsonl", type=Path, required=True, help="JSONL with {sequence, label} per line.")
    p.add_argument("--val-jsonl", type=Path, default=None, help="Optional validation JSONL.")
    p.add_argument("--test-jsonl", type=Path, default=None, help="Optional test JSONL.")
    p.add_argument("--base-ckpt-dir", type=Path, required=True, help="Path to MBridge backbone checkpoint.")
    p.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Root directory for tensorboard logs and adapter checkpoints.",
    )
    p.add_argument("--experiment-name", required=True, help="Subdirectory of --result-dir for this run.")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--seq-length-tokens", type=int, default=512)
    p.add_argument("--backbone-seq-length", type=int, default=8192)
    p.add_argument("--train-iters", type=int, default=300)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=20)
    p.add_argument(
        "--decay-steps",
        type=int,
        default=None,
        help="Cosine-annealing decay length. Defaults to (train_iters - warmup_iters).",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--classifier-hidden-size", type=int, default=None)
    p.add_argument("--classifier-dropout", type=float, default=0.1)
    p.add_argument("--pool", choices=["mean", "last"], default="mean")
    p.add_argument("--lora-finetune", action="store_true", help="Enable LoRA adapters (default: head-only baseline).")
    p.add_argument("--lora-dim", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument(
        "--lora-target-modules",
        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
        default=list(DEFAULT_LORA_TARGET_MODULES),
        help="Comma-separated list of LoRA target module short names.",
    )
    p.add_argument("--tokenizer-path", default=DEFAULT_HF_TOKENIZER_MODEL_PATH_512)
    p.add_argument("--precision-recipe", default="bf16_mixed")
    p.add_argument("--wandb-project", default=None, help="W&B project name. Disables wandb when omitted.")
    p.add_argument("--wandb-entity", default=None, help="W&B team/user posting the run.")
    p.add_argument("--wandb-run-name", default=None, help="W&B run name. Defaults to --experiment-name.")
    p.add_argument("--wandb-save-dir", type=Path, default=None, help="Local directory for wandb metadata.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: parse args, build the config, run training."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    cfg = evo2_1b_classifier_config(
        base_ckpt_dir=args.base_ckpt_dir,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        test_jsonl=args.test_jsonl,
        num_classes=args.num_classes,
        result_dir=args.result_dir,
        experiment_name=args.experiment_name,
        seq_length_tokens=args.seq_length_tokens,
        backbone_seq_length=args.backbone_seq_length,
        train_iters=args.train_iters,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        decay_steps=args.decay_steps,
        weight_decay=args.weight_decay,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        seed=args.seed,
        classifier_hidden_size=args.classifier_hidden_size,
        classifier_dropout=args.classifier_dropout,
        pool=args.pool,
        use_lora=args.lora_finetune,
        lora_dim=args.lora_dim,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        tokenizer_path=args.tokenizer_path,
        precision_recipe=args.precision_recipe,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_save_dir=args.wandb_save_dir,
    )
    train_classifier(cfg)


if __name__ == "__main__":
    Hyena1bClassifierProvider.__module__ = "evo2_classifier"
    Evo2ClassifierDatasetProvider.__module__ = "evo2_classifier"
    main(sys.argv[1:])
