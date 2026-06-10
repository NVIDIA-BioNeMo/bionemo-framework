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

r"""Prediction (inference) workflow for Eden using Megatron Bridge.

This module provides functionality to run inference on Eden models using MBridge checkpoints.
It supports various parallelism strategies (TP, CP, DP) and can output either full logits
or collapsed log probabilities.

Usage (CLI):
    # Single GPU inference
    torchrun --nproc_per_node 1 -m bionemo.eden.run.predict \
        --fasta input.fasta --ckpt-dir /path/to/mbridge/checkpoint \
        --output-dir /path/to/output

    # Multi-GPU with tensor parallelism
    torchrun --nproc_per_node 2 -m bionemo.eden.run.predict \
        --fasta input.fasta --ckpt-dir /path/to/mbridge/checkpoint \
        --output-dir /path/to/output --tensor-parallel-size 2

    # With context parallelism for long sequences
    torchrun --nproc_per_node 2 -m bionemo.eden.run.predict \
        --fasta input.fasta --ckpt-dir /path/to/mbridge/checkpoint \
        --output-dir /path/to/output --context-parallel-size 2

Output Format:
    Batch mode (--write-interval batch):
    - predictions__rank_{global_rank}__dp_rank_{dp_rank}__batch_{batch_idx}.pt
    - With --files-per-subdir: subdir_{N}/predictions__rank_...
    - Each file includes batch_idx tensor for reconstruction

    Epoch mode (--write-interval epoch, default):
    - predictions__rank_{global_rank}__dp_rank_{dp_rank}.pt
    - All batches collated into single file

    Both modes:
    - seq_idx_map.json: Mapping from sequence names to indices in predictions

Key Functions:
    - predict(): Main prediction workflow
    - batch_collator(): Collate predictions from multiple batches/ranks
    - initialize_inference_distributed(): Set up distributed environment for inference
"""

import argparse
import datetime
import logging
import os
import random
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.bridge.training.config import DistributedInitConfig, RNGConfig
from megatron.bridge.training.mixed_precision import MIXED_PRECISION_RECIPES, get_mixed_precision_config
from megatron.bridge.training.tokenizers.tokenizer import _HuggingFaceTokenizer
from megatron.bridge.training.utils.checkpoint_utils import (
    file_exists,
    get_checkpoint_run_config_filename,
    read_run_config,
)
from megatron.bridge.utils.common_utils import (
    get_local_rank_preinit,
    get_master_addr_safe,
    get_master_port_safe,
    get_rank_safe,
    get_world_size_safe,
)
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.core import parallel_state, tensor_parallel
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_batch_on_this_cp_rank
from torch import Tensor

from bionemo.eden.data.fasta_dataset import SimpleFastaDataset
from bionemo.recipeutils.inference.collation import batch_collator


_REPO_BASE_DIR = Path(__file__).resolve().parents[4]
DEFAULT_HF_TOKENIZER_MODEL_PATH = str(_REPO_BASE_DIR / "tokenizers" / "nucleotide_fast_tokenizer_256")


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =============================================================================
# Checkpoint Path Resolution
# =============================================================================


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    """Resolve a checkpoint path to the actual checkpoint directory.

    MBridge checkpoints can be organized in two ways:
    1. Direct checkpoint: A directory containing run_config.yaml directly
       (e.g., after conversion or for single checkpoints)
    2. Training output: A parent directory containing iter_XXXXXXX subdirectories

    This function handles both cases:
    - If run_config.yaml exists in the given path, return it as-is
    - Otherwise, find the latest iter_XXXXXXX subdirectory and return that

    Args:
        checkpoint_path: Path to either a direct checkpoint or a training output directory.

    Returns:
        Path to the checkpoint directory containing run_config.yaml.

    Raises:
        FileNotFoundError: If the path doesn't exist or no valid checkpoint is found.
        NotADirectoryError: If the path is not a directory.

    Examples:
        >>> # Direct checkpoint path
        >>> resolve_checkpoint_path(Path("/checkpoints/eden_1b_mbridge"))
        PosixPath('/checkpoints/eden_1b_mbridge')

        >>> # Training output with iter_* subdirectories
        >>> resolve_checkpoint_path(Path("/training/output"))
        PosixPath('/training/output/iter_0007000')  # Returns latest
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' does not exist.")
    if not checkpoint_path.is_dir():
        raise NotADirectoryError(f"Checkpoint path '{checkpoint_path}' must be a directory.")

    # Check if run_config.yaml exists directly in this path
    run_config_path = get_checkpoint_run_config_filename(str(checkpoint_path))
    if file_exists(run_config_path):
        return checkpoint_path

    # Look for iter_* subdirectories
    iter_dirs = [
        (child.name, child) for child in checkpoint_path.iterdir() if child.is_dir() and child.name.startswith("iter_")
    ]

    if not iter_dirs:
        raise FileNotFoundError(
            f"No valid checkpoint found at '{checkpoint_path}'. "
            "Expected either run_config.yaml in the directory or iter_* subdirectories."
        )

    # Find the latest iteration by parsing the iteration number
    def _parse_iter_num(item: tuple[str, Path]) -> int:
        try:
            return int(item[0].replace("iter_", ""))
        except ValueError:
            return -1

    _, latest_iter_path = max(iter_dirs, key=_parse_iter_num)

    # Verify the selected iter directory has run_config.yaml
    run_config_path = get_checkpoint_run_config_filename(str(latest_iter_path))
    if not file_exists(run_config_path):
        raise FileNotFoundError(f"Latest checkpoint directory '{latest_iter_path}' does not contain run_config.yaml.")

    logger.info(f"Resolved checkpoint path to: {latest_iter_path}")
    return latest_iter_path


# =============================================================================
# Distributed Initialization
# =============================================================================


def initialize_inference_distributed(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    micro_batch_size: int = 1,
    global_batch_size: int = 1,
    rng_config: Optional[RNGConfig] = None,
    dist_config: Optional[DistributedInitConfig] = None,
) -> None:
    """Initialize distributed environment for inference.

    Sets up the minimal distributed infrastructure needed for model-parallel inference:
    1. torch.distributed process group
    2. Model parallel groups (TP, PP, CP, DP)
    3. Microbatch calculator (for batch scheduling)
    4. Random seeds for reproducibility

    This is a lightweight alternative to full Megatron initialization, skipping
    training-specific components like the rerun state machine.

    Args:
        tensor_model_parallel_size: Tensor parallelism degree (splits model across GPUs)
        pipeline_model_parallel_size: Pipeline parallelism degree (must be 1 for inference)
        context_parallel_size: Context parallelism degree (splits sequence across GPUs)
        micro_batch_size: Batch size per forward pass
        global_batch_size: Total batch size across all DP ranks
        rng_config: Random number generator configuration. Defaults to seed=1234.
        dist_config: Distributed backend configuration. Defaults to NCCL backend.

    Note:
        This function must be called before creating the model. It initializes
        parallel_state which is used throughout the codebase.
    """
    # Apply defaults
    if rng_config is None:
        rng_config = RNGConfig(seed=1234)
    if dist_config is None:
        dist_config = DistributedInitConfig()

    assert torch.cuda.is_available(), "Inference requires CUDA."

    device_count = torch.cuda.device_count()
    world_size = get_world_size_safe()
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    data_parallel_size = world_size // model_parallel_size

    # Initialize microbatch calculator
    init_num_microbatches_calculator(
        rank=get_rank_safe(),
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
        decrease_batch_size_if_needed=False,
    )

    # Initialize torch.distributed
    if not torch.distributed.is_initialized():
        if get_rank_safe() == 0:
            print("> initializing torch distributed for inference ...", flush=True)

        if device_count > 0:
            torch.cuda.set_device(get_local_rank_preinit())

        # Ensure environment variables are set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = get_master_addr_safe()
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(get_master_port_safe())

        torch.distributed.init_process_group(
            backend=dist_config.distributed_backend,
            world_size=world_size,
            rank=get_rank_safe(),
            timeout=datetime.timedelta(minutes=dist_config.distributed_timeout_minutes),
        )
        torch.distributed.barrier(device_ids=[get_local_rank_preinit()])
    else:
        if get_rank_safe() == 0:
            print("torch distributed is already initialized, skipping ...", flush=True)

    # Initialize model parallel groups
    if device_count > 0 and not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            distributed_timeout_minutes=dist_config.distributed_timeout_minutes,
        )
        if get_rank_safe() == 0:
            print(
                f"> initialized tensor model parallel with size {parallel_state.get_tensor_model_parallel_world_size()}"
            )
            print(
                f"> initialized pipeline model parallel with size {parallel_state.get_pipeline_model_parallel_world_size()}"
            )
            print(f"> initialized data parallel with size {parallel_state.get_data_parallel_world_size()}")
    elif get_rank_safe() == 0:
        print("model parallel is already initialized", flush=True)

    # Set random seeds
    if get_rank_safe() == 0:
        print(f"> setting random seeds to {rng_config.seed} ...", flush=True)

    seed = rng_config.seed + (100 * parallel_state.get_pipeline_model_parallel_rank())
    if rng_config.data_parallel_random_init:
        seed = seed + (10 * parallel_state.get_data_parallel_rank())

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)

    if device_count > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(
            seed,
            rng_config.te_rng_tracker,
            rng_config.inference_rng_tracker,
        )


# =============================================================================
# Context Parallelism Utilities
# =============================================================================


def _gather_along_cp_dim(input_: Tensor, seq_dim: int = 1, unshuffle_zigzag: bool = True) -> Tensor:
    """Gather tensors from all CP ranks and restore original sequence order.

    When using context parallelism (CP), sequences are split across multiple GPUs using a
    "zigzag" pattern for load balancing. This function gathers the split tensors from all
    CP ranks and optionally restores the original sequence order.

    Zigzag Pattern (CP=2 example):
        Original sequence: [chunk0, chunk1, chunk2, chunk3]
        CP rank 0 receives: [chunk0, chunk3]  (positions 0 and 3)
        CP rank 1 receives: [chunk1, chunk2]  (positions 1 and 2)

    After gathering and unshuffling, the original order is restored.

    Args:
        input_: Input tensor with shape [B, S/CP, ...] where S is full sequence length
        seq_dim: Sequence dimension in the tensor. Default 1.
        unshuffle_zigzag: If True, restore original sequence order after gathering.
            Set to False only if you need the raw gathered order. Default True.

    Returns:
        Gathered tensor with shape [B, S, ...] in original sequence order.
        If CP=1, returns input unchanged.

    Note:
        This function requires parallel_state to be initialized with CP groups.
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size == 1:
        return input_

    # Gather from all CP ranks
    # After all_gather: [B * cp_size, seq_len_per_rank, ...]
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * cp_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=parallel_state.get_context_parallel_group()
    )

    # Chunk by batch dimension and concatenate by sequence dimension
    # Result: [B, seq_len_per_rank * cp_size, ...]
    tensor_list = output.chunk(cp_size, dim=0)
    output = torch.cat(tensor_list, dim=seq_dim).contiguous()

    if not unshuffle_zigzag:
        return output

    # Undo the zigzag pattern from get_batch_on_this_cp_rank
    # The zigzag assigns chunk i and (2*cp_size - i - 1) to rank i
    seq_len = output.shape[seq_dim]
    num_chunks = 2 * cp_size
    chunk_size = seq_len // num_chunks

    chunks = output.split(chunk_size, dim=seq_dim)

    # Build the order in which chunks appear after gathering:
    # [rank0_first, rank0_second, rank1_first, rank1_second, ...]
    # where rank_i has chunks (i, 2*cp_size - i - 1)
    gathered_order = []
    for rank in range(cp_size):
        gathered_order.append(rank)
        gathered_order.append(2 * cp_size - rank - 1)

    # Create inverse mapping: original_position -> gathered_position
    inverse_order = [0] * num_chunks
    for pos, orig_idx in enumerate(gathered_order):
        inverse_order[orig_idx] = pos

    # Reorder to original sequence order [0, 1, 2, ..., 2*cp_size-1]
    reordered_chunks = [chunks[inverse_order[i]] for i in range(num_chunks)]
    return torch.cat(reordered_chunks, dim=seq_dim).contiguous()


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Eden inference.

    Returns:
        Parsed arguments namespace
    """
    ap = argparse.ArgumentParser(
        description="Run inference on Eden models using MBridge checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    ap.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="Path to input FASTA file containing sequences for prediction",
    )
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="Path to MBridge checkpoint directory (must contain run_config.yaml)",
    )

    # Output arguments
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output predictions. If not set, predictions are discarded.",
    )
    ap.add_argument(
        "--write-interval",
        type=str,
        default="epoch",
        choices=["epoch", "batch"],
        help="When to write predictions: 'epoch' writes all at end, 'batch' writes after each batch",
    )
    ap.add_argument(
        "--files-per-subdir",
        type=int,
        help="Group output files into subdirectories. Only used with --write-interval batch.",
    )

    # Parallelism arguments
    ap.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed inference")
    ap.add_argument(
        "--devices",
        type=int,
        help="Number of GPUs per node. Default: TP * PP * CP",
    )
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism degree")
    ap.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        choices=[1],
        default=1,
        help="Pipeline parallelism degree (only 1 supported)",
    )
    ap.add_argument("--context-parallel-size", type=int, default=1, help="Context parallelism degree")
    ap.add_argument(
        "--no-sequence-parallel",
        action="store_true",
        help="Disable sequence parallelism when using TP > 1",
    )

    # Model/precision arguments
    ap.add_argument(
        "--mixed-precision-recipe",
        type=str,
        choices=list(MIXED_PRECISION_RECIPES.keys()),
        help="Override mixed precision recipe (default: use checkpoint setting)",
    )

    # Batch/sequence arguments
    ap.add_argument("--micro-batch-size", type=int, default=1, help="Batch size per forward pass")
    ap.add_argument("--min-length", type=int, help="Minimum sequence length (pad shorter sequences)")
    ap.add_argument("--prepend-bos", action="store_true", help="Prepend BOS token to sequences")

    # Output format arguments
    ap.add_argument(
        "--output-log-prob-seqs",
        action="store_true",
        help="Output log probabilities instead of raw logits",
    )
    ap.add_argument(
        "--log-prob-collapse-option",
        choices=["sum", "mean", "per_token"],
        default="mean",
        help="How to aggregate per-token log probs: sum, mean, or keep per_token",
    )

    # Embedding extraction arguments
    ap.add_argument(
        "--embedding-layer",
        type=int,
        help="Extract embeddings from a specific transformer layer instead of logits. "
        "Supports Python-style negative indexing (e.g., -1 for last layer, -2 for second-to-last). "
        "For a 25-layer model, layer 24 and layer -1 both refer to the last layer.",
    )

    return ap.parse_args()


def on_writing_rank() -> bool:
    """Returns True if the current rank is one that should own writing predictions."""
    return (
        (parallel_state.is_pipeline_last_stage())
        and (parallel_state.get_tensor_model_parallel_rank() == 0)
        and (parallel_state.get_context_parallel_rank() == 0)
    )


# =============================================================================
# Data Loading Utilities
# =============================================================================


def _padding_collate_fn_factory(
    pad_token_id: int = 0,
    min_length: Optional[int] = None,
):
    """Create a collate function that pads sequences to uniform length.

    Args:
        pad_token_id: Token ID to use for padding
        min_length: Minimum sequence length (pad shorter sequences to this)

    Returns:
        Collate function compatible with DataLoader
    """

    def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        return _padding_collate_fn(batch, pad_token_id, min_length)

    return collate_fn


def _padding_collate_fn(
    batch: list[dict[str, Tensor]],
    pad_token_id: int = 0,
    min_length: Optional[int] = None,
) -> dict[str, Tensor]:
    """Pad sequences in a batch to the same length.

    Handles the following keys specially:
    - tokens: Padded with pad_token_id
    - position_ids: Extended with consecutive positions
    - loss_mask: Padded with 0 (masked)
    - seq_idx: Not padded (scalar per sample)
    - Other keys: Padded with 0

    Args:
        batch: List of sample dictionaries from the dataset
        pad_token_id: Token ID for padding
        min_length: Minimum length to pad to

    Returns:
        Dictionary with batched and padded tensors
    """
    max_len = max(sample["tokens"].shape[0] for sample in batch)
    if min_length is not None:
        max_len = max(max_len, min_length)

    padded_batch: dict[str, list[Tensor]] = {key: [] for key in batch[0].keys()}

    for sample in batch:
        seq_len = sample["tokens"].shape[0]
        pad_len = max_len - seq_len

        for key, value in sample.items():
            if key == "tokens":
                padded = torch.nn.functional.pad(value, (0, pad_len), value=pad_token_id)
            elif key == "position_ids":
                if pad_len > 0:
                    padded = torch.cat([value, torch.arange(seq_len, max_len, dtype=value.dtype)])
                else:
                    padded = value
            elif key == "loss_mask":
                padded = torch.nn.functional.pad(value, (0, pad_len), value=0)
            elif key == "seq_idx":
                padded = value  # Scalar, no padding
            else:
                padded = torch.nn.functional.pad(value, (0, pad_len), value=0)
            padded_batch[key].append(padded)

    return {key: torch.stack(values) for key, values in padded_batch.items()}


# =============================================================================
# Prediction Step
# =============================================================================


def _predict_step(
    model: torch.nn.Module,
    batch: dict[str, Tensor],
    output_log_prob_seqs: bool = False,
    log_prob_collapse_option: Literal["sum", "mean", "per_token"] = "mean",
    context_parallel_size: int = 1,
    output_embeddings: bool = False,
) -> Optional[dict[str, Tensor]]:
    """Run a single prediction step and gather outputs across parallel ranks.

    Args:
        model: The Eden model to run inference with
        batch: Input batch containing:
            - tokens: Input token IDs [B, S]
            - position_ids: Position indices [B, S]
            - loss_mask: Mask indicating valid tokens [B, S]
            - seq_idx: Original sequence indices [B]
        output_log_prob_seqs: If True, return log probabilities instead of logits
        log_prob_collapse_option: How to aggregate log probs ('sum', 'mean', or 'per_token')
        context_parallel_size: CP size (for warning about per_token output)
        output_embeddings: If True, return embeddings instead of logits (model must have
            post_process=False)

    Returns:
        Dictionary containing predictions:
        - If output_embeddings=True: hidden_embeddings, pad_mask, seq_idx, tokens
        - If output_log_prob_seqs=False: token_logits, pad_mask, seq_idx, tokens
        - If output_log_prob_seqs=True with sum/mean: log_probs_seqs, seq_idx
        - If output_log_prob_seqs=True with per_token: log_probs_seqs, seq_idx, loss_mask
        Returns None if not on the last pipeline stage.
    """
    if not parallel_state.is_pipeline_last_stage():
        return None

    # Forward pass
    output_tensor = model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=None,
    )

    # Gather across tensor parallel ranks
    # For logits (post_process=True): gather along vocabulary dimension (last dim is sharded)
    # For embeddings (post_process=False): hidden states are not sharded across TP, skip gathering
    if output_embeddings:
        # Hidden states are not sharded across TP ranks, just use the output directly
        forward_out_tp_gathered = output_tensor
    else:
        # Logits have the vocab dimension sharded across TP ranks
        forward_out_tp_gathered = _gather_along_last_dim(
            output_tensor, group=parallel_state.get_tensor_model_parallel_group()
        )

    # Gather across context parallel ranks (sequence dimension)
    forward_out_gathered = _gather_along_cp_dim(forward_out_tp_gathered)
    loss_mask_gathered = _gather_along_cp_dim(batch["loss_mask"])
    tokens_gathered = _gather_along_cp_dim(batch["tokens"])

    if output_embeddings:
        # When extracting embeddings, the model output is hidden states, not logits
        # Model outputs [S, B, H] (sequence-first format), transpose to [B, S, H] for consistency
        hidden_embeddings = forward_out_gathered.transpose(0, 1).contiguous()
        return {
            "hidden_embeddings": hidden_embeddings,
            "pad_mask": loss_mask_gathered,
            "seq_idx": batch["seq_idx"],
            "tokens": tokens_gathered,
        }
    elif output_log_prob_seqs:
        return _compute_log_probs(
            logits=forward_out_gathered,
            tokens=tokens_gathered,
            loss_mask=loss_mask_gathered,
            seq_idx=batch["seq_idx"],
            collapse_option=log_prob_collapse_option,
            context_parallel_size=context_parallel_size,
        )
    else:
        return {
            "token_logits": forward_out_gathered,
            "pad_mask": loss_mask_gathered,
            "seq_idx": batch["seq_idx"],
            "tokens": tokens_gathered,
        }


def _compute_log_probs(
    logits: Tensor,
    tokens: Tensor,
    loss_mask: Tensor,
    seq_idx: Tensor,
    collapse_option: Literal["sum", "mean", "per_token"],
    context_parallel_size: int,
) -> dict[str, Tensor]:
    """Compute log probabilities from model logits.

    Computes P(token_i | token_0, ..., token_{i-1}) for each token.

    Args:
        logits: Model output logits [B, S, V]
        tokens: Input token IDs [B, S]
        loss_mask: Mask for valid tokens [B, S]
        seq_idx: Sequence indices [B]
        collapse_option: How to aggregate: 'sum', 'mean', or 'per_token'
        context_parallel_size: CP size (for per_token warning)

    Returns:
        Dictionary with log_probs_seqs and seq_idx (and loss_mask if per_token)
    """
    # Predictions for token i are at position i, labels are at i+1
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]  # [B, S-1, V]
    target_tokens = tokens[:, 1:]  # [B, S-1]

    if softmax_logprobs.shape[1] != target_tokens.shape[1]:
        raise RuntimeError(f"Shape mismatch: logprobs {softmax_logprobs.shape} vs targets {target_tokens.shape}")

    # Gather log probs for actual tokens
    log_probs_per_token = torch.gather(softmax_logprobs, 2, target_tokens.unsqueeze(-1)).squeeze(-1)

    # Apply loss mask (zero out padding)
    loss_mask_shifted = loss_mask[:, 1:].float()
    log_probs_per_token = log_probs_per_token * loss_mask_shifted

    if collapse_option == "per_token":
        if context_parallel_size > 1:
            logger.warning(
                "Per-token log probabilities with CP>1 will have zigzag-shuffled order. "
                "Use 'sum' or 'mean' to get correctly aggregated results."
            )
        return {
            "log_probs_seqs": log_probs_per_token,
            "seq_idx": seq_idx,
            "loss_mask": loss_mask_shifted.bool(),
        }

    # Sum log probs across sequence
    log_prob_seqs = torch.sum(log_probs_per_token, dim=1)

    if collapse_option == "mean":
        # Divide by number of valid tokens
        valid_token_count = torch.clamp(loss_mask_shifted.sum(dim=-1), min=1.0)
        log_prob_seqs = log_prob_seqs / valid_token_count

    return {"log_probs_seqs": log_prob_seqs, "seq_idx": seq_idx}


# =============================================================================
# Output Writing
# =============================================================================


def _write_predictions_batch(
    predictions: dict[str, Tensor],
    output_dir: Path,
    batch_idx: int,
    global_rank: int,
    dp_rank: int,
    files_per_subdir: Optional[int] = None,
    num_files_written: int = 0,
    data_parallel_world_size: int = 1,
) -> tuple[Path, int, int]:
    """Write predictions to disk as a PyTorch file (batch mode).

    File naming follows the original PredictionWriter convention:
    predictions__rank_{global_rank}__dp_rank_{dp_rank}__batch_{batch_idx}.pt

    Subdirectory structure (when files_per_subdir is set):
    subdir_{num}/predictions__rank_...

    The subdirectory numbering starts from 1 and increments when the number of files
    written (across all DP ranks) reaches files_per_subdir.

    Args:
        predictions: Dictionary of prediction tensors to save
        output_dir: Base output directory
        batch_idx: Batch index for file naming
        global_rank: Global rank of this process
        dp_rank: Data parallel rank (included in filename for multi-GPU)
        files_per_subdir: If set, organize files into subdirectories
        num_files_written: Number of files already written in current subdir
        data_parallel_world_size: Number of data parallel ranks

    Returns:
        Tuple of (output_path, updated_num_files_written, updated_num_subdirs)
    """
    if (not predictions) or (not on_writing_rank()):
        return output_dir, num_files_written, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Track subdirectory state
    current_output_dir = output_dir
    num_subdirs_written = 0

    if files_per_subdir is not None:
        # Calculate how many subdirs we've created based on total files written
        # (counting all DP ranks)
        effective_files = num_files_written * data_parallel_world_size
        if effective_files >= files_per_subdir:
            # Need a new subdirectory
            num_subdirs_written = effective_files // files_per_subdir + 1
            current_output_dir = output_dir / f"subdir_{num_subdirs_written}"
            current_output_dir.mkdir(parents=True, exist_ok=True)
            num_files_written = 0

    filename = f"predictions__rank_{global_rank}__dp_rank_{dp_rank}__batch_{batch_idx}.pt"
    output_path = current_output_dir / filename

    # Add batch_idx to predictions (matching original PredictionWriter behavior)
    predictions["batch_idx"] = torch.tensor([batch_idx], dtype=torch.int64)

    torch.save(predictions, output_path)
    logger.info(f"Inference predictions are stored in {output_path}\n{predictions.keys()}")

    return output_path, num_files_written + 1, num_subdirs_written


def _write_predictions_epoch(
    predictions: dict[str, Tensor],
    output_dir: Path,
    global_rank: int,
    dp_rank: int,
) -> Path:
    """Write predictions to disk as a PyTorch file (epoch mode).

    File naming follows the original PredictionWriter convention:
    predictions__rank_{global_rank}__dp_rank_{dp_rank}.pt

    Args:
        predictions: Dictionary of prediction tensors to save
        output_dir: Base output directory
        global_rank: Global rank of this process
        dp_rank: Data parallel rank

    Returns:
        Path to the saved file
    """
    if (not predictions) or (not on_writing_rank()):
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"predictions__rank_{global_rank}__dp_rank_{dp_rank}.pt"
    output_path = output_dir / filename

    torch.save(predictions, output_path)
    logger.info(f"Inference predictions are stored in {output_path}\n{predictions.keys()}")

    return output_path


# =============================================================================
# Main Prediction Workflow
# =============================================================================


def predict(
    fasta_path: Path,
    ckpt_dir: Path,
    output_dir: Optional[Path] = None,
    *,
    # Parallelism settings
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    no_sequence_parallel: bool = False,
    # Precision settings
    mixed_precision_recipe: Optional[str] = None,
    # Batch/sequence settings
    micro_batch_size: int = 1,
    min_length: Optional[int] = None,
    prepend_bos: bool = False,
    # Output settings
    write_interval: Literal["epoch", "batch"] = "epoch",
    files_per_subdir: Optional[int] = None,
    output_log_prob_seqs: bool = False,
    log_prob_collapse_option: Literal["sum", "mean", "per_token"] = "mean",
    # Embedding extraction
    embedding_layer: Optional[int] = None,
) -> None:
    """Run the complete Eden prediction workflow.

    This function orchestrates the full inference pipeline:
    1. Load model configuration from MBridge checkpoint
    2. Override parallelism and precision settings
    3. Initialize distributed environment
    4. Create and configure the model
    5. Load model weights
    6. Process FASTA sequences and write predictions

    Args:
        fasta_path: Path to input FASTA file containing sequences for prediction.
        ckpt_dir: Path to MBridge checkpoint directory (must contain run_config.yaml).
        output_dir: Directory for output predictions. If None, predictions are discarded.
        tensor_parallel_size: Tensor parallelism degree (splits model across GPUs).
        pipeline_model_parallel_size: Pipeline parallelism degree (must be 1).
        context_parallel_size: Context parallelism degree (splits sequence across GPUs).
        no_sequence_parallel: Disable sequence parallelism when using TP > 1.
        mixed_precision_recipe: Override mixed precision recipe (default: use checkpoint).
        micro_batch_size: Batch size per forward pass.
        min_length: Minimum sequence length (pad shorter sequences to this).
        prepend_bos: Prepend BOS token to sequences.
        write_interval: When to write predictions: 'epoch' or 'batch'.
        files_per_subdir: Group output files into subdirectories (batch mode only).
        output_log_prob_seqs: Output log probabilities instead of raw logits.
        log_prob_collapse_option: How to aggregate log probs: 'sum', 'mean', 'per_token'.
        embedding_layer: Extract embeddings from a specific layer instead of logits.
            Supports Python-style negative indexing (-1 for last layer, -2 for second-to-last).
            For a 25-layer model, layer 24 and -1 both refer to the last layer.

    Raises:
        ValueError: If pipeline parallelism > 1 is requested.
        FileNotFoundError: If checkpoint run_config.yaml is missing.

    Example:
        >>> from pathlib import Path
        >>> predict(
        ...     fasta_path=Path("sequences.fasta"),
        ...     ckpt_dir=Path("/path/to/mbridge/checkpoint"),
        ...     output_dir=Path("/path/to/output"),
        ...     tensor_parallel_size=2,
        ...     micro_batch_size=4,
        ... )
    """
    if pipeline_model_parallel_size != 1:
        raise ValueError("Pipeline parallelism > 1 is not currently supported for prediction.")

    # -------------------------------------------------------------------------
    # Step 1: Resolve and load configuration from checkpoint
    # -------------------------------------------------------------------------
    # Handle both direct checkpoint paths and training output directories with iter_* subdirs
    resolved_ckpt_dir = resolve_checkpoint_path(ckpt_dir)
    logger.info(f"Loading configuration from checkpoint: {resolved_ckpt_dir}")

    run_config_filename = get_checkpoint_run_config_filename(str(resolved_ckpt_dir))

    run_config = read_run_config(run_config_filename)
    model_provider = instantiate(run_config["model"])
    logger.info(f"Instantiated model provider: {type(model_provider).__name__}")

    # -------------------------------------------------------------------------
    # Step 2: Override parallelism and precision settings
    # -------------------------------------------------------------------------
    model_provider.tensor_model_parallel_size = tensor_parallel_size
    model_provider.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_provider.context_parallel_size = context_parallel_size
    model_provider.sequence_parallel = tensor_parallel_size > 1 and not no_sequence_parallel

    # Configure mixed precision
    if mixed_precision_recipe is not None:
        mp_config = get_mixed_precision_config(mixed_precision_recipe)
    elif "mixed_precision" in run_config and run_config["mixed_precision"] is not None:
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
        tokenizer = _HuggingFaceTokenizer(tokenizer_dir)
    else:
        tokenizer = _HuggingFaceTokenizer(DEFAULT_HF_TOKENIZER_MODEL_PATH)

    model_provider.vocab_size = tokenizer.vocab_size
    model_provider.should_pad_vocab = True

    # -------------------------------------------------------------------------
    # Step 3.5: Handle embedding layer extraction
    # -------------------------------------------------------------------------
    # Get the original number of layers from the checkpoint config
    original_num_layers = model_provider.num_layers
    output_embeddings = embedding_layer is not None

    if output_embeddings:
        # Validate and resolve the embedding layer index
        # Support Python-style negative indexing
        if embedding_layer < 0:
            # Convert negative index to positive (e.g., -1 -> last layer)
            target_num_layers = original_num_layers + embedding_layer + 1
        else:
            # Positive index: layer N means we need N+1 layers (0-indexed)
            target_num_layers = embedding_layer + 1

        if target_num_layers <= 0 or target_num_layers > original_num_layers:
            raise ValueError(
                f"Invalid embedding_layer={embedding_layer} for model with {original_num_layers} layers. "
                f"Valid range: -{original_num_layers} to {original_num_layers - 1}."
            )

        # Set the model to use fewer layers and skip post-processing (output heads)
        model_provider.num_layers = target_num_layers
        model_provider.post_process = False

        # Disable remove_activation_post_first_layer if we only have 1 layer, since it requires at least 2 layers
        if target_num_layers == 1 and hasattr(model_provider, "remove_activation_post_first_layer"):
            if model_provider.remove_activation_post_first_layer:
                model_provider.remove_activation_post_first_layer = False
                logger.info("Disabled remove_activation_post_first_layer (requires at least 2 layers)")

        logger.info(
            f"Embedding extraction mode: extracting from layer {embedding_layer} "
            f"(using {target_num_layers} of {original_num_layers} layers, post_process=False)"
        )

        # Cannot use log prob output with embedding mode
        if output_log_prob_seqs:
            raise ValueError("Cannot use --output-log-prob-seqs with --embedding-layer. Embeddings are not logits.")

    # -------------------------------------------------------------------------
    # Step 4: Initialize distributed environment
    # -------------------------------------------------------------------------
    rng_config = instantiate(run_config.get("rng")) if run_config.get("rng") else RNGConfig(seed=1234)
    dist_config = instantiate(run_config.get("dist")) if run_config.get("dist") else DistributedInitConfig()

    model_parallel_size = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    world_size = get_world_size_safe()
    data_parallel_size = world_size // model_parallel_size
    global_batch_size = micro_batch_size * data_parallel_size

    initialize_inference_distributed(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
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

    # Log model layer information
    # Access the underlying model to get layer count
    model_for_inspection = model[0]
    if hasattr(model_for_inspection, "module"):
        # Handle Float16Module wrapper
        model_for_inspection = model_for_inspection.module
    if hasattr(model_for_inspection, "decoder") and hasattr(model_for_inspection.decoder, "layers"):
        actual_num_layers = len(model_for_inspection.decoder.layers)
        logger.info(f"Model initialized with {actual_num_layers} layers")
        if output_embeddings:
            logger.info(
                f"Embedding extraction: model has {actual_num_layers} layers "
                f"(from original {original_num_layers} layers)"
            )
    else:
        logger.warning("Could not determine number of layers from model structure")

    logger.info(f"Loading weights from: {resolved_ckpt_dir}")
    _load_model_weights_from_checkpoint(
        checkpoint_path=str(resolved_ckpt_dir),
        model=model,
        dist_ckpt_strictness="ignore_all",
    )
    logger.info("Weights loaded successfully")

    # -------------------------------------------------------------------------
    # Step 6: Create dataset and dataloader
    # -------------------------------------------------------------------------
    logger.info(f"Loading dataset from: {fasta_path}")
    dataset = SimpleFastaDataset(
        fasta_path=fasta_path,
        tokenizer=tokenizer,
        prepend_bos=prepend_bos,
        custom_loss_masker=None,
    )

    data_parallel_rank = parallel_state.get_data_parallel_rank()
    data_parallel_size = parallel_state.get_data_parallel_world_size()

    dataloader = build_pretraining_data_loader(
        dataset=dataset,
        consumed_samples=0,
        dataloader_type="single",
        micro_batch_size=micro_batch_size,
        num_workers=4,
        data_sharding=False,
        collate_fn=_padding_collate_fn_factory(
            pad_token_id=getattr(tokenizer, "pad_id", 0),
            min_length=min_length,
        ),
        pin_memory=True,
        persistent_workers=False,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # Step 7: Run prediction loop
    # -------------------------------------------------------------------------
    logger.info("Starting prediction loop...")
    predictions: list[dict[str, Tensor]] = []

    # Get ranks for file naming (matching original PredictionWriter behavior)
    global_rank = get_rank_safe()
    num_files_written = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Move to GPU
            batch_gpu = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()
            }

            # Apply context parallel slicing (seq_idx must NOT be sliced)
            if context_parallel_size > 1:
                seq_idx = batch_gpu.pop("seq_idx", None)
                batch_gpu = get_batch_on_this_cp_rank(batch_gpu)
                if seq_idx is not None:
                    batch_gpu["seq_idx"] = seq_idx

            # Forward pass
            result = _predict_step(
                model=model[0],
                batch=batch_gpu,
                output_log_prob_seqs=output_log_prob_seqs,
                log_prob_collapse_option=log_prob_collapse_option,
                context_parallel_size=context_parallel_size,
                output_embeddings=output_embeddings,
            )

            if result is not None:
                predictions.append({k: v.cpu() for k, v in result.items()})

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

            # Write at batch interval
            if write_interval == "batch" and output_dir is not None and predictions:
                _, num_files_written, _ = _write_predictions_batch(
                    predictions=predictions[0],
                    output_dir=output_dir,
                    batch_idx=batch_idx,
                    global_rank=global_rank,
                    dp_rank=data_parallel_rank,
                    files_per_subdir=files_per_subdir,
                    num_files_written=num_files_written,
                    data_parallel_world_size=data_parallel_size,
                )
                predictions = []

    # Write at epoch end
    if write_interval == "epoch" and output_dir is not None and predictions:
        combined = batch_collator(
            predictions,
            batch_dim=0,
            seq_dim=1,
            batch_dim_key_defaults={},
            seq_dim_key_defaults={},
        )
        _write_predictions_epoch(
            predictions=combined,
            output_dir=output_dir,
            global_rank=global_rank,
            dp_rank=data_parallel_rank,
        )

    # Write sequence index map
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.write_idx_map(output_dir)

    logger.info("Prediction complete!")

    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for Eden prediction."""
    args = parse_args()
    predict(
        fasta_path=args.fasta,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
        # Parallelism settings
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        no_sequence_parallel=args.no_sequence_parallel,
        # Precision settings
        mixed_precision_recipe=args.mixed_precision_recipe,
        # Batch/sequence settings
        micro_batch_size=args.micro_batch_size,
        min_length=args.min_length,
        prepend_bos=args.prepend_bos,
        # Output settings
        write_interval=args.write_interval,
        files_per_subdir=args.files_per_subdir,
        output_log_prob_seqs=args.output_log_prob_seqs,
        log_prob_collapse_option=args.log_prob_collapse_option,
        # Embedding extraction
        embedding_layer=args.embedding_layer,
    )


if __name__ == "__main__":
    main()
