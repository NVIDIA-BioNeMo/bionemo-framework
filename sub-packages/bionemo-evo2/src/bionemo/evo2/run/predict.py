# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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


import argparse
import functools
import tempfile
from pathlib import Path
from typing import Any, Literal

import nemo.lightning as nl
import torch
from lightning.pytorch import LightningDataModule
from megatron.core import parallel_state
from megatron.core.enums import Fp8Recipe
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim
from megatron.core.utils import get_batch_on_this_cp_rank
from nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset import Evo2Dataset
from nemo.collections.llm.gpt.model.base import GPTModel, get_packed_seq_params
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS, HyenaModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.data import WrappedDataLoader
from nemo.utils import logging as logger
from torch import Tensor

from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset
from bionemo.evo2.models.llama import LLAMA_MODEL_OPTIONS

# Add import for Mamba models
from bionemo.evo2.models.mamba import MAMBA_MODEL_OPTIONS, MambaModel
from bionemo.evo2.models.peft import Evo2LoRA
from bionemo.evo2.run.utils import infer_model_type, patch_eden_tokenizer
from bionemo.llm.data import collate
from bionemo.llm.lightning import LightningPassthroughPredictionMixin
from bionemo.llm.utils.callbacks import PredictionWriter


CheckpointFormats = Literal["torch_dist", "zarr"]

SHUFFLE_MESSAGE = (
    "Per token log probabilities are not supported when using context parallelism. The results will be "
    "zigzag shuffled along the sequence dimension. Raise a feature request if you need this and do "
    "not want to manually do the unshuffling yourself. You need to undo the shuffling that happened in "
    "`megatron.core.utils.get_batch_on_this_cp_rank`."
)


def parse_args():
    """Parse arguments for Evo2 inference."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use for prediction, defaults to 1.")
    ap.add_argument(
        "--devices",
        type=int,
        help="Number of devices to use for prediction, defaults to tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size.",
    )
    ap.add_argument(
        "--eden-tokenizer",
        action="store_true",
        help="Patch the tokenizer to work with the one used in training the Eden model.",
    )
    ap.add_argument("--fasta", type=Path, required=True, help="Fasta path from which to generate logit predictions.")
    ap.add_argument("--ckpt-dir", type=Path, required=True, help="NeMo2 checkpoint directory for inference.")
    ap.add_argument("--min-length", type=int, required=False, help="Minimum sequence length for padding.")
    ap.add_argument("--prepend-bos", action="store_true", help="Prepend BOS token to sequences. Defaults to False.")
    ap.add_argument(
        "--mask-phylogenetic-tags",
        action="store_true",
        help="Mask phylogenetic tags in loss computation. Defaults to False.",
    )
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1.")
    ap.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        choices=[1],
        default=1,
        help="Order of pipeline parallelism. Defaults to 1 and currently only 1 is supported.",
    )
    ap.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )
    ap.add_argument(
        "--fp8-recipe",
        type=str,
        default="delayed",
        choices=list(Fp8Recipe.__members__.keys()),
        help="FP8 recipe to use for FP8 tensors in the forward and backward pass. Note that some recipes are only "
        "supported by certain architectures. For example 'mxfp8' requires at least blackwell, and 'blockwise' is only "
        "implemented for hopper (but not blackwell). 'tensorwise' and 'delayed' are currently supported by all "
        "architectures, but 'tensorwise' is preferred over 'delayed' which is the default for historical reasons.",
    )
    ap.add_argument(
        "--no-sequence-parallel",
        action="store_true",
        help="When using TP, skip sequence parallelism. Otherwise sequence parallelism is used whenever tensor "
        "parallelism is used. sequence parallelism should save a small amount of GPU memory so it's on"
        " by default.",
    )
    ap.add_argument("--micro-batch-size", type=int, default=1, help="Batch size for prediction. Defaults to 1.")
    ap.add_argument(
        "--write-interval",
        type=str,
        default="epoch",
        choices=["epoch", "batch"],
        help="Interval to write predictions to disk. If doing very large predictions, you may want to set this to 'batch'.",
    )
    ap.add_argument(
        "--model-size",
        type=str,
        default="7b_arc_longcontext",
        choices=sorted(
            list(HYENA_MODEL_OPTIONS.keys()) + list(MAMBA_MODEL_OPTIONS.keys()) + list(LLAMA_MODEL_OPTIONS.keys())
        ),
        help="Model size to use. Defaults to '7b_arc_longcontext'.",
    )
    # output args:
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir that will contain the generated text produced by the Evo2 model. If not provided, the output will be logged.",
    )
    ap.add_argument(
        "--files-per-subdir",
        type=int,
        help="Number of files to write to each subdirectory. If provided, subdirectories with N files each will be created. Ignored unless --write-interval is 'batch'.",
    )
    ap.add_argument(
        "--full-fp8",
        action="store_true",
        help="Use full FP8 precision (faster but less accurate) rather than vortex style which "
        "only applies FP8 to the projection layer of the hyena mixer, when using FP8.",
    )
    ap.add_argument("--fp8", action="store_true", help="Use FP8 precision. Defaults to BF16.")
    # extra:
    ap.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated.",
    )
    ap.add_argument(
        "--output-log-prob-seqs", action="store_true", help="Output log probability of sequences. Defaults to False."
    )
    ap.add_argument(
        "--log-prob-collapse-option",
        choices=["sum", "mean", "per_token"],
        default="mean",
        help="How to collapse the log probabilities across the sequence dimension.",
    )
    ap.add_argument(
        "--hybrid-override-pattern",
        type=str,
        help="Override the hybrid override pattern in the config (specifies hyena layer ordering and type).",
    )
    ap.add_argument(
        "--num-layers", type=int, help="If set, override the number of layers specified in the requested config."
    )
    ap.add_argument(
        "--seq-len-interpolation-factor",
        type=int,
        help="If set, override the sequence length interpolation factor specified in the requested config. If you "
        "know a model was trained with a specific interpolation factor for ROPE, provide it here, it can make a big "
        "difference in accuracy.",
    )
    ap.add_argument(
        "--lora-checkpoint-path",
        type=Path,
        required=False,
        default=None,
        help="Path to the lora states to restore from.",
    )
    return ap.parse_args()


def _gather_along_cp_dim(input_, seq_dim: int = 1):
    """Gather tensors and concatenate along the last dimension."""
    world_size = parallel_state.get_context_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    # TODO: handle zigzag packing here. Currently this just gathers along ranks, but if you want to see the sequence in
    #   the original order you need to undo the zigzag packing that happens in
    #   `megatron.core.utils.get_batch_on_this_cp_rank`.
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=parallel_state.get_context_parallel_group()
    )
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=seq_dim).contiguous()

    return output


def _to_cpu(inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.cpu() for k, v in inputs.items()}


def _identity(inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    return inputs


class BasePredictor(LightningPassthroughPredictionMixin):
    """Base predictor for GPT-style models."""

    def __init__(
        self,
        *args,
        output_log_prob_seqs: bool = False,
        include_tokens_with_logprob_seqs: bool = False,
        log_prob_collapse_option: Literal["sum", "mean", "per_token"] = "mean",
        **kwargs,
    ):
        """Initialize the base predictor with arguments needed for writing predictions."""
        super().__init__(*args, **kwargs)
        self.output_log_prob_seqs = output_log_prob_seqs
        self.log_prob_collapse_option = log_prob_collapse_option
        self.include_tokens_with_logprob_seqs = include_tokens_with_logprob_seqs
        self.shuffle_warning_raised = False

    def predict_step(
        self, batch, batch_idx: int | None = None, to_cpu: bool = True
    ) -> Tensor | dict[str, Tensor] | None:
        """Alias for forward_step, also log the pad mask since sequences may not all have the same length."""
        if len(batch) == 0:
            return
        assert self.training is False, "predict_step should be called in eval mode"
        with torch.no_grad():
            forward_out = self.forward_step(batch)
        if not parallel_state.is_pipeline_last_stage():
            return None
        # Reminder: the model's predictions for input i land at output i+1. To get everything to align, we prepend the
        # EOS token to the input sequences and take the outputs for all but the first token.
        forward_out_tp_gathered = _gather_along_last_dim(
            forward_out, group=parallel_state.get_tensor_model_parallel_group()
        )

        forward_out_gathered = _gather_along_cp_dim(forward_out_tp_gathered)
        loss_mask_gathered = _gather_along_cp_dim(batch["loss_mask"])
        tokens_gathered = _gather_along_cp_dim(batch["tokens"])
        cp_group_size = max(parallel_state.get_context_parallel_world_size(), 1)
        assert self.tokenizer.vocab_size == forward_out_gathered.shape[-1]
        to_cpu_fn = _to_cpu if to_cpu else _identity
        if self.output_log_prob_seqs:
            if self.log_prob_collapse_option == "per_token" and cp_group_size > 1 and not self.shuffle_warning_raised:
                logger.warning(SHUFFLE_MESSAGE)
                self.shuffle_warning_raised = True
            softmax_logprobs = torch.log_softmax(forward_out_gathered, dim=-1)
            softmax_logprobs = softmax_logprobs[:, :-1]
            input_ids = tokens_gathered[:, 1:]
            if softmax_logprobs.shape[1] != input_ids.shape[1]:
                raise RuntimeError(
                    f"Softmax logprobs shape {softmax_logprobs.shape} does not match input ids shape {input_ids.shape}"
                )

            logprobs = torch.gather(
                softmax_logprobs,  # Gather likelihoods...
                2,  # along the vocab dimension...
                input_ids.unsqueeze(-1),  # using the token ids to index.
            ).squeeze(-1)
            log_prob_per_token = logprobs * loss_mask_gathered[:, 1:].float()
            if self.log_prob_collapse_option == "per_token":
                return to_cpu_fn(
                    {
                        "log_probs_seqs": log_prob_per_token,
                        "seq_idx": batch["seq_idx"],
                        "loss_mask": loss_mask_gathered[:, 1:],
                    }
                )
            else:
                log_prob_seqs = torch.sum(log_prob_per_token, dim=1)
                if self.log_prob_collapse_option == "mean":
                    log_prob_seqs = log_prob_seqs / torch.clamp(loss_mask_gathered[:, 1:].float().sum(dim=-1), min=1.0)
                return to_cpu_fn({"log_probs_seqs": log_prob_seqs, "seq_idx": batch["seq_idx"]})
        else:
            # If the user wants to match back to logits, then they will need to do the offsetting logic themselves.
            if cp_group_size > 1 and not self.shuffle_warning_raised:
                logger.warning(SHUFFLE_MESSAGE)
                self.shuffle_warning_raised = True
            logprob_seqs_result = {
                "token_logits": forward_out_gathered,
                "pad_mask": loss_mask_gathered,
                "seq_idx": batch["seq_idx"],
            }
            if self.include_tokens_with_logprob_seqs:
                logprob_seqs_result["tokens"] = tokens_gathered
            # Note, to match up tokens with logprobs, you need to offset by 1. Eg something like this:
            #  shifted_token_logits = token_logits[:, :-1]
            #  shifted_pad_mask = pad_mask[:, 1:]
            #  shifted_tokens = tokens[:, 1:]
            return to_cpu_fn(logprob_seqs_result)


class HyenaPredictor(BasePredictor, HyenaModel):
    """A predictor for the Hyena model. This adds in the predict step and the passthrough method."""

    def configure_model(self, *args, **kwargs) -> None:
        """Configure the model."""
        super().configure_model(*args, **kwargs)
        self.trainer.strategy._init_model_parallel = True


class MambaPredictor(BasePredictor, MambaModel):
    """Mamba model for prediction with additional metrics."""


class LlamaPredictor(BasePredictor, GPTModel):
    """Llama model for prediction with additional metrics."""


def hyena_predict_forward_step(model, batch) -> torch.Tensor:
    """Performs a forward step for the Hyena model.

    Args:
        model: The Hyena model
        batch: Dictionary containing input batch data with keys:
            - tokens: Input token IDs
            - position_ids: Position IDs
            - labels: Labels for loss computation
            - loss_mask: Mask for loss computation

    Returns:
        torch.Tensor: Output from the model forward pass
    """
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        # "labels": batch["labels"],
        # "loss_mask": batch["loss_mask"],
    }

    forward_args["attention_mask"] = None
    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)
    return model(**forward_args)


def hyena_predict_data_step(dataloader_iter) -> dict[str, torch.Tensor]:
    """Data step for the Hyena model prediction. Modified from the original gpt data step to include the seq_idx."""
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_device_keys = set()
    required_host_keys = set()

    required_device_keys.add("attention_mask")
    if "cu_seqlens" in _batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    if parallel_state.is_pipeline_first_stage():
        required_device_keys.update(("tokens", "position_ids"))
    include_seq_idx = False
    if parallel_state.is_pipeline_last_stage():
        include_seq_idx = True
        required_device_keys.update(("labels", "tokens", "loss_mask"))

    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_cp_rank(_batch_required_keys)
    if include_seq_idx:
        output["seq_idx"] = _batch["seq_idx"].cuda(non_blocking=True)
    return output


class PredictDataModule(LightningDataModule):
    """Create a dataloader for prediction."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 1,
        tokenizer=None,
        min_length: int | None = None,
    ):
        """Create a dataloader for prediction."""
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.min_length = min_length
        default_pad_id = 0
        self.pad_token_id = getattr(tokenizer, "pad_id", default_pad_id) if tokenizer is not None else default_pad_id

    def setup(self, stage: str | None = None) -> None:
        """Set up the dataloader."""
        pass

    def predict_dataloader(self):
        """Create a dataloader for prediction."""
        # need to use this to communicate that we are in predict mode and safe to not drop last batch
        return WrappedDataLoader(
            mode="predict",
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=functools.partial(
                collate.padding_collate_fn,
                padding_values={"tokens": self.pad_token_id, "position_ids": self.pad_token_id, "loss_mask": False},
                min_length=self.min_length,
                max_length=None,
            ),
        )


def predict(
    fasta_path: Path,
    ckpt_dir: str,
    output_dir: Path,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    num_nodes: int = 1,
    devices: int | None = None,
    eden_tokenizer: bool = False,
    model_size: str = "7b",
    ckpt_format: CheckpointFormats = "torch_dist",
    fp8: bool = False,
    full_fp8: bool = False,
    fp8_recipe: str = "delayed",
    work_dir: Path | None = None,
    micro_batch_size: int = 1,
    output_log_prob_seqs: bool = False,
    log_prob_collapse_option: Literal["sum", "mean", "per_token"] = "mean",
    write_interval: Literal["epoch", "batch"] = "epoch",
    prepend_bos: bool = False,
    no_sequence_parallel: bool = False,
    hybrid_override_pattern: str | None = None,
    num_layers: int | None = None,
    seq_len_interpolation_factor: int | None = None,
    files_per_subdir: int | None = None,
    lora_checkpoint_path: Path | None = None,
    mask_phylogenetic_tags: bool = False,
    min_length: int | None = None,
    extra_callbacks: list | None = None,  # use this for making testing the predict loop easier.
):
    """Inference workflow for Evo2.

    Returns:
        None
    """
    if fp8 and not full_fp8 and fp8_recipe != "delayed":
        logger.warning(
            "fp8_recipe is ignored when using fp8 and not full_fp8 since it is set inside of the layer "
            "config to match vortex style FP8."
        )
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    if files_per_subdir is None and write_interval == "batch":
        logger.warning(
            "--files-per-subdir is not set with --write-interval batch, will write all predictions to a "
            "single directory. This may cause problems if you are predicting on a very large dataset."
        )
    sequence_parallel = tensor_parallel_size > 1 and not no_sequence_parallel
    output_dir.mkdir(parents=True, exist_ok=True)  # Make sure the output directory exists, files will be written here.
    model_parallel_size = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    if devices is None:
        devices = model_parallel_size
    world_size = num_nodes * devices
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size must be divisible by model_parallel_size, got {world_size} and"
            f" {model_parallel_size}. Please set --num-nodes and --devices such that num_nodes * devices is divisible "
            "by model_parallel_size, which is TP * CP * PP."
        )
    global_batch_size = micro_batch_size * world_size // model_parallel_size

    callbacks = [
        PredictionWriter(
            output_dir=output_dir,
            write_interval=write_interval,
            batch_dim_key_defaults={"token_logits": 0},
            seq_dim_key_defaults={"token_logits": 1},
            files_per_subdir=files_per_subdir,
            save_all_model_parallel_ranks=False,  # only write one copy of predictions.
        )
    ]
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    # The following two config options are really only used for testing, but may also be useful for getting output from
    #   specific layers of the model.
    config_modifiers_init: dict[str, Any] = {
        "distribute_saved_activations": False if sequence_parallel and tensor_parallel_size > 1 else True,
    }
    if hybrid_override_pattern is not None:
        config_modifiers_init["hybrid_override_pattern"] = hybrid_override_pattern
    if num_layers is not None:
        config_modifiers_init["num_layers"] = num_layers
    if seq_len_interpolation_factor is not None:
        config_modifiers_init["seq_len_interpolation_factor"] = seq_len_interpolation_factor

    tokenizer = get_nmt_tokenizer("byte-level")
    if eden_tokenizer:
        patch_eden_tokenizer(tokenizer)

    model_type = infer_model_type(model_size)

    # Select model config based on model type
    if model_type == "hyena":
        if model_size not in HYENA_MODEL_OPTIONS:
            raise ValueError(f"Invalid model size for Hyena: {model_size}")
        config = HYENA_MODEL_OPTIONS[model_size](
            forward_step_fn=hyena_predict_forward_step,
            data_step_fn=hyena_predict_data_step,
            # Only use vortex style FP8 in the model config if using FP8 and not full FP8. This will only apply FP8 to
            #   the projection layer of the hyena mixer.
            vortex_style_fp8=fp8 and not full_fp8,
            **config_modifiers_init,
        )

        if lora_checkpoint_path:
            model_transform = Evo2LoRA(peft_ckpt_path=str(lora_checkpoint_path))
            callbacks.append(model_transform)
        else:
            model_transform = None

        model = HyenaPredictor(
            config,
            tokenizer=tokenizer,
            output_log_prob_seqs=output_log_prob_seqs,
            log_prob_collapse_option=log_prob_collapse_option,
            model_transform=model_transform,
        )
    elif model_type == "mamba":  # mamba
        if model_size not in MAMBA_MODEL_OPTIONS:
            raise ValueError(f"Invalid model size for Mamba: {model_size}")
        config = MAMBA_MODEL_OPTIONS[model_size](
            forward_step_fn=hyena_predict_forward_step,  # Can reuse the same forward steps
            data_step_fn=hyena_predict_data_step,
            **config_modifiers_init,
        )

        model = MambaPredictor(
            config,
            tokenizer=tokenizer,
            output_log_prob_seqs=output_log_prob_seqs,
            log_prob_collapse_option=log_prob_collapse_option,
        )
    elif model_type == "llama":
        if model_size not in LLAMA_MODEL_OPTIONS:
            raise ValueError(f"Invalid model size for Llama: {model_size}")
        config = LLAMA_MODEL_OPTIONS[model_size](
            forward_step_fn=hyena_predict_forward_step,
            data_step_fn=hyena_predict_data_step,
            **config_modifiers_init,
        )
        model = LlamaPredictor(
            config,
            tokenizer=tokenizer,
            output_log_prob_seqs=output_log_prob_seqs,
            log_prob_collapse_option=log_prob_collapse_option,
        )
    else:
        # This shouldn't be possible to reach.
        raise ValueError(f"Invalid model type: {model_type}.")

    # Create PTL trainer.
    trainer = nl.Trainer(
        accelerator="gpu",
        num_nodes=num_nodes,
        devices=devices,
        strategy=nl.MegatronStrategy(
            drop_last_batch=False,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            sequence_parallel=sequence_parallel,
            save_ckpt_format=ckpt_format,
            ckpt_load_strictness="log_all",
            setup_optimizers=False,
            store_optimizer_states=False,
            configure_optimizers=False,
            data_sampler=nl.MegatronDataSampler(
                micro_batch_size=micro_batch_size,
                global_batch_size=global_batch_size,
                seq_len=8192,
                output_log=False,  # this is needed for predict step to work
            ),
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            # Only use FP8 in this plugin when using full FP8 precision and FP8.
            #   Otherwise use vortex_style_fp8 in the model config.
            fp8_recipe=fp8_recipe,
            fp8="hybrid" if fp8 and full_fp8 else None,
            fp8_amax_history_len=16 if fp8 and full_fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 and full_fp8 else "most_recent",
        ),
    )

    nemo_logger = NeMoLogger(log_dir=str(work_dir))
    nemo_logger.setup(trainer, resume_if_exists=True)
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=False,
        resume_past_end=False,
        resume_from_path=str(ckpt_dir),
        restore_config=None,
    )

    resume.setup(trainer, model)  # this pulls weights from the starting checkpoint.

    if mask_phylogenetic_tags:

        def custom_loss_masker(tokens):
            # Run the evo2 dataset mask_phylogenetic_tags function
            return Evo2Dataset.mask_phylogenetic_tags(
                tokens,
                Evo2Dataset.TAG_BOUNDS,
                Evo2Dataset.TAG_CHARS,
                tokenizer.eod if tokenizer is not None else Evo2Dataset.DEFAULT_EOD,
                Evo2Dataset.MAX_TAG_LEN,
            )
    else:
        custom_loss_masker = None

    dataset = SimpleFastaDataset(fasta_path, tokenizer, prepend_bos=prepend_bos, custom_loss_masker=custom_loss_masker)
    datamodule = PredictDataModule(dataset, batch_size=micro_batch_size, tokenizer=tokenizer, min_length=min_length)
    trainer.predict(model, datamodule=datamodule)  # TODO return_predictions=False
    dataset.write_idx_map(
        output_dir
    )  # Finally write out the index map so we can match the predictions to the original sequences.


def main():
    """Entrypoint for Evo2 prediction (single inference step, no new tokens)."""
    args = parse_args()
    predict(
        num_nodes=args.num_nodes,
        devices=args.devices,
        fasta_path=args.fasta,
        ckpt_dir=args.ckpt_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        output_dir=args.output_dir,
        model_size=args.model_size,
        ckpt_format=args.ckpt_format,
        fp8=args.fp8,
        full_fp8=args.full_fp8,
        fp8_recipe=args.fp8_recipe,
        micro_batch_size=args.micro_batch_size,
        output_log_prob_seqs=args.output_log_prob_seqs,
        log_prob_collapse_option=args.log_prob_collapse_option,
        prepend_bos=args.prepend_bos,
        no_sequence_parallel=args.no_sequence_parallel,
        hybrid_override_pattern=args.hybrid_override_pattern,
        seq_len_interpolation_factor=args.seq_len_interpolation_factor,
        num_layers=args.num_layers,
        files_per_subdir=args.files_per_subdir,
        write_interval=args.write_interval,
        lora_checkpoint_path=args.lora_checkpoint_path,
        mask_phylogenetic_tags=args.mask_phylogenetic_tags,
        min_length=args.min_length,
        eden_tokenizer=args.eden_tokenizer,
    )


if __name__ == "__main__":
    main()
