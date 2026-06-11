# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Data collators for sequence packing and context parallel training.

This should eventually get moved to a separate package, or possibly upstreamed into `transformers`.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, TypedDict

import datasets
import nvtx
import torch
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp
from transformers import DataCollator, DataCollatorForLanguageModeling


logger = logging.getLogger(__name__)


@dataclass
class DataCollatorWithFlattening:
    """Data collator that wraps a DataCollatorForLanguageModeling and flattens inputs for flash-attention.

    This collator enables efficient training on batches containing variable-length sequences, by first flattening
    (packing) multiple input sequences into a single contiguous tensor without padding between sequences. Then, it
    applies masked language modeling (MLM) masking using the provided DataCollatorForLanguageModeling instance.

    The collator also generates metadata required for Flash Attention or context-parallel attention:
      - `cu_seq_lens_q` and `cu_seq_lens_k` tensors, denoting cumulative sequence lengths so that sequence boundaries
        within the packed tensor are known during attention computation.

    Optionally, the collator can:
      - Pad the total number of tokens in the batch to be divisible by `pad_to_multiple_of` (by appending a mock
        sequence).
      - Pad each individual sequence to be divisible by `pad_sequences_to_be_divisible_by` if provided.

    Only PyTorch tensors (`return_tensors="pt"`) are supported.

    Args:
        collator (DataCollatorForLanguageModeling): The collator to use for MLM masking. This is a captive
            collator and should be constructed externally and passed in.
        return_position_ids (bool): Whether to return position ids (default False).
        pad_to_multiple_of (int, optional): If set, pads the total sequence length to be divisible by this number.
        pad_sequences_to_be_divisible_by (int, optional): If set, each individual sequence is padded to this value.
        separator_id (int, optional): A label to insert between sequences, typically should be -100 for causal LM.

    Example:
        >>> from transformers import AutoTokenizer, DataCollatorForLanguageModeling
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        >>> mlm_collator = DataCollatorForLanguageModeling(tokenizer)
        >>> flat_collator = DataCollatorWithFlattening(
        ...     collator=mlm_collator,
        ...     pad_to_multiple_of=8,
        ... )
        >>>
        >>> # Input: variable length protein sequences
        >>> sequences = [
        ...     {"input_ids": [0, 5, 6, 7, 2]},      # 5 tokens
        ...     {"input_ids": [0, 8, 9, 10, 11, 2]}, # 6 tokens
        ...     {"input_ids": [0, 12, 13, 2]},       # 4 tokens
        ... ]  # Total: 15 tokens
        >>> batch = flat_collator(sequences)
        >>> print(batch['input_ids'].shape)    # torch.Size([1, 16])
        >>> print(batch['labels'].shape)       # torch.Size([1, 16])
        >>> print(batch['cu_seq_lens_q'])      # tensor([0, 5, 11, 15, 16], dtype=torch.int32)

    Note:
        The output is a THD-format (Total, Height, Depth) batch, where all input sequences are packed without
        inter-sequence padding. Sequence boundaries are preserved using `cu_seq_lens_q`/`cu_seq_lens_k`, enabling
        Flash Attention or context-parallelism without traditional attention masks.
    """

    collator: DataCollatorForLanguageModeling
    return_position_ids: bool = False
    pad_to_multiple_of: int | None = None
    pad_sequences_to_be_divisible_by: int | None = None
    separator_id: int | None = None

    def __post_init__(self):
        """Ensure padding options are not used together."""
        if self.pad_sequences_to_be_divisible_by is not None and self.pad_to_multiple_of is not None:
            raise ValueError("pad_sequences_to_be_divisible_by and pad_to_multiple_of cannot be used together")

    def __call__(self, features, return_tensors=None):
        """Process a batch of variable-length sequences for Flash Attention with MLM.

        This method performs the following steps:
        1. Flattens multiple sequences into a single packed tensor with Flash Attention metadata
        2. Applies MLM masking to the flattened sequence while preserving special tokens
        3. Optionally pads to a multiple of a specified number for hardware optimization

        Args:
            features (List[Dict[str, List[int]]]): List of tokenized sequences, each containing
                'input_ids' and optionally 'attention_mask'. Example:
                [
                    {"input_ids": [0, 5, 6, 7, 2]},      # Protein sequence 1
                    {"input_ids": [0, 8, 9, 10, 11, 2]}, # Protein sequence 2
                    {"input_ids": [0, 12, 13, 2]}        # Protein sequence 3
                ]
            return_tensors (str, optional): Format for returned tensors. Only "pt" (PyTorch)
                is supported. Defaults to None (uses collator default).

        Returns:
            Dict[str, torch.Tensor]: Batch dictionary containing:
                - input_ids (torch.Tensor): Flattened and MLM-masked token sequences.
                  Shape: [1, total_tokens] where total_tokens = sum of all sequence lengths
                  (plus padding if pad_to_multiple_of is specified).
                - labels (torch.Tensor): MLM labels with -100 for non-masked tokens and
                  original token IDs for masked positions. Same shape as input_ids.
                - cu_seq_lens_q (torch.IntTensor): Cumulative sequence lengths for queries.
                  Shape: [num_sequences + 1] or [num_sequences + 2] if padding is added.
                  Example: [0, 5, 11, 15] or [0, 5, 11, 15, 16] with padding.
                - cu_seq_lens_k (torch.IntTensor): Cumulative sequence lengths for keys.
                  Same as cu_seq_lens_q for self-attention.
                - max_length_q (int): Maximum sequence length in the batch.
                - max_length_k (int): Same as max_length_q for self-attention.
                - attention_mask (torch.Tensor): Attention mask with 1s for actual tokens
                  and 0s for padding tokens (if any).

        Raises:
            NotImplementedError: If return_tensors is not "pt".

        Example:
            >>> # Input features
            >>> features = [
            ...     {"input_ids": [0, 5, 6, 7, 2]},      # 5 tokens
            ...     {"input_ids": [0, 8, 9, 10, 11, 2]}, # 6 tokens
            ... ]
            >>>
            >>> batch = collator(features)
            >>>
            >>> # Output shapes and values
            >>> batch['input_ids'].shape          # torch.Size([1, 11]) or larger if padded
            >>> batch['labels'].shape             # torch.Size([1, 11]) or larger if padded
            >>> batch['cu_seq_lens_q']            # tensor([0, 5, 11], dtype=torch.int32) or larger

        Note:
            The output is in THD (Total, Height, Depth) format with batch_size=1 and
            sequence_length=total_tokens, optimized for Flash Attention's variable-length
            sequence processing capabilities. When pad_to_multiple_of is used, an additional
            mock sequence is appended to reach the desired total length.
        """
        if return_tensors is not None and return_tensors != "pt":
            raise NotImplementedError(f"Only return_tensors='pt' is supported, got '{return_tensors}'")

        # Perform the masking with the BSHD collator.
        bshd_batch = self.collator(features, return_tensors=return_tensors)

        # Create the flattened batch to get the cu_seq_lens_q and cu_seq_lens_k values.
        packed_batch = _pt_flatten_collate(features, return_position_ids=self.return_position_ids)

        # Get the masked input_ids and labels from the BSHD batch.
        masked_input_ids = bshd_batch["input_ids"][bshd_batch["attention_mask"].bool()].unsqueeze(0)
        masked_labels = bshd_batch["labels"][bshd_batch["attention_mask"].bool()].unsqueeze(0)

        if self.separator_id is not None:
            masked_labels[:, packed_batch["cu_seq_lens_q"][1:-1]] = self.separator_id

        # Update the packed batch with the masked input_ids and labels.
        packed_batch["input_ids"] = masked_input_ids
        packed_batch["labels"] = masked_labels

        if self.pad_to_multiple_of is not None:
            packed_batch = self._pad_batch_to_multiple_of(packed_batch)

        elif self.pad_sequences_to_be_divisible_by is not None:
            packed_batch = self._pad_sequences_to_be_divisible_by(packed_batch)

        return packed_batch

    def _pad_batch_to_multiple_of(self, batch):
        """Add a mock sequence to make the total number of tokens divisible by pad_to_multiple_of."""
        # Ensure token_pad is an integer, defaulting to 1 if pad_token_id is None or invalid
        pad_token_id = self.collator.tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            logger.warning(f"tokenizer.pad_token_id is not an integer, using 1 instead: {pad_token_id}")
            pad_token_id = 1

        assert self.pad_to_multiple_of is not None, "pad_to_multiple_of must be set"

        return _pt_pad_to_multiple_of(
            batch,
            self.pad_to_multiple_of,
            token_pad=pad_token_id,
            label_pad=-100,
        )

    def _pad_sequences_to_be_divisible_by(self, batch):
        """Pad individual sequences using cu_seq_lens_*_padded for context parallelism."""
        pad_token_id = self.collator.tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            logger.warning(f"tokenizer.pad_token_id is not an integer, using 1 instead: {pad_token_id}")
            pad_token_id = 1

        assert self.pad_sequences_to_be_divisible_by is not None, "pad_sequences_to_be_divisible_by must be set"

        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            batch["input_ids"],
            batch["labels"],
            batch["cu_seq_lens_q"],
            self.pad_sequences_to_be_divisible_by,
            padding_token_id=pad_token_id,
            padding_label_id=-100,
        )

        batch["input_ids"] = input_ids_padded.unsqueeze(0)
        batch["labels"] = labels_padded.unsqueeze(0)
        batch["cu_seq_lens_q_padded"] = cu_seqlens_padded.to(torch.int32)
        batch["cu_seq_lens_k_padded"] = cu_seqlens_padded.to(torch.int32)
        batch["pad_between_seqs"] = True
        return batch


@dataclass
class TokenPackingDataset(torch.utils.data.IterableDataset):
    """Dataset that uses sequence packing to construct batches with variable length up to a maximum number of tokens."""

    dataset: datasets.IterableDataset
    """Dataset to pack."""
    max_tokens_per_batch: int
    """Maximum number of tokens per batch."""
    drop_last: bool = True
    """Whether to drop the last batch if it's less than max_length."""
    split_samples: bool = False
    """Whether to split samples to ensure batches have exactly max_tokens_per_batch tokens."""
    pad_sequences_to_be_divisible_by: int | None = None
    """If set, account for per-sequence padding when accumulating batches.

    Each sequence's contribution to the batch length is rounded up to the nearest multiple of this value,
    matching the padding behavior of DataCollatorWithFlattening with the same parameter. When used with
    split_samples=True, the split point is chosen so that the first part (after padding) exactly fills
    the remaining batch capacity.
    """

    def __post_init__(self):
        """Validate padding configuration."""
        if (
            self.pad_sequences_to_be_divisible_by is not None
            and self.max_tokens_per_batch % self.pad_sequences_to_be_divisible_by != 0
        ):
            logger.warning(
                "max_tokens_per_batch (%d) is not divisible by pad_sequences_to_be_divisible_by (%d). "
                "Batches may not fill to exactly max_tokens_per_batch when split_samples=True.",
                self.max_tokens_per_batch,
                self.pad_sequences_to_be_divisible_by,
            )

    def _padded_len(self, length: int) -> int:
        """Return the padded length of a sequence, rounding up to the nearest multiple of pad_sequences_to_be_divisible_by."""
        if self.pad_sequences_to_be_divisible_by is None:
            return length
        return -(-length // self.pad_sequences_to_be_divisible_by) * self.pad_sequences_to_be_divisible_by

    def __iter__(self):
        """Yield batches of samples, each with a variable number of tokens up to the maximum length.

        When split_samples=True, ensures each batch has exactly max_tokens_per_batch by splitting
        the final sample if needed. The remaining tokens from the split sample start the next batch.

        When pad_sequences_to_be_divisible_by is set, each sequence's padded length is used when
        accumulating batch sizes, so the total padded length of the batch matches max_tokens_per_batch.

        Returns:
            A generator of batches of samples, each with a variable number of tokens up to the maximum length.
        """
        samples = []
        current_length = 0
        for sample in iter(self.dataset):
            sample_length = len(sample["input_ids"])
            padded_len = self._padded_len(sample_length)
            if padded_len > self.max_tokens_per_batch:
                raise ValueError(
                    f"TokenPackingDataset: Padded sample length ({padded_len}) exceeds max_tokens_per_batch "
                    f"({self.max_tokens_per_batch}). Set truncation or a maximum length in your tokenizer or dataset to"
                    " ensure all samples fit within max_tokens_per_batch."
                )

            current_length += padded_len
            if current_length == self.max_tokens_per_batch:
                yield [*samples, sample]
                samples = []
                current_length = 0

            elif current_length > self.max_tokens_per_batch:
                if not self.split_samples:
                    # Yield the current batch (before this sample) and start a new one with this sample.
                    if samples:
                        yield samples
                    samples = [sample]
                    current_length = padded_len
                else:
                    # Calculate how many padded tokens are already in the batch.
                    tokens_in_batch = current_length - padded_len
                    # Calculate how many tokens we can fit from this sample, ensuring the
                    # padded length doesn't exceed the remaining capacity.
                    tokens_available = self.max_tokens_per_batch - tokens_in_batch
                    if self.pad_sequences_to_be_divisible_by is not None:
                        d = self.pad_sequences_to_be_divisible_by
                        tokens_available = (tokens_available // d) * d
                    if tokens_available <= 0:
                        # Remaining capacity is less than pad_sequences_to_be_divisible_by;
                        # can't fit any tokens from this sample. Yield current batch and start fresh.
                        if samples:
                            yield samples
                        samples = [sample]
                        current_length = padded_len
                    else:
                        first_part, remaining_part = _split_sample_by_num_tokens(sample, tokens_available)
                        yield [*samples, first_part]
                        samples = [remaining_part]
                        current_length = self._padded_len(len(samples[0]["input_ids"]))
            else:
                samples.append(sample)

        if not self.drop_last and samples:
            yield samples

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset."""
        self.dataset.set_epoch(epoch)


@dataclass
class DataCollatorForContextParallel:
    """A collator that is aware of context parallelism.

    For the case of context parallelism, padded sequences will be returned from the wrapped collator, and then split
    into shards for each context parallelism rank.

    The shards are then typically sent to the ContextParallelDataLoaderWrapper which will scatter them to the
    appropriate GPUs.

    Note:
        When used with the ContextParallelDataLoaderWrapper and both context parallelism and tensor parallelism are
        used, the collator inspects the ordering of the mesh dimensions to determine the layout of the flattened batch.

        If "cp" comes before "tp" in the mesh dimension names (CP row-major), the flattened batch will be:
        [(cp0, tp0), (cp0, tp1), ..., (cp1, tp0), (cp1, tp1), ...]

        If "tp" comes before "cp" (TP row-major), the flattened batch will be:
        [(tp0, cp0), (tp0, cp1), ..., (tp1, cp0), (tp1, cp1), ...]

    Args:
        collator: The collator to use for the batch.
        device_mesh: The device mesh with named dimensions. Must contain either a "cp" dimension for context parallelism
            and/or a "tp" dimension for tensor parallelism.
        qkv_format: The format of the query-key-value (QKV) tensor.
        is_causal_lm: Whether the collator is for a causal language model. If True, the labels will be shifted before
            being split into CP shards, and will be returned in the `shift_labels` field.

    """

    collator: DataCollator
    device_mesh: torch.distributed.device_mesh.DeviceMesh
    qkv_format: str = "thd"
    is_causal_lm: bool = False

    # Derived fields, initialized in __post_init__.
    cp_world_size: int = field(init=False)
    tp_world_size: int | None = field(init=False)
    _is_cp_row_major: bool = field(init=False)

    def __post_init__(self):
        """Initialize the cp_world_size, tp_world_size, and _is_cp_row_major fields based on the device mesh."""
        dim_names = self.device_mesh.mesh_dim_names
        if dim_names is None:
            raise ValueError("device_mesh must have mesh_dim_names")

        self.cp_world_size = self.device_mesh.size(dim_names.index("cp")) if "cp" in dim_names else 1
        self.tp_world_size = self.device_mesh.size(dim_names.index("tp")) if "tp" in dim_names else None

        # Determine whether CP is the row (outer) dimension of the 2D mesh.
        # When flattened, the row-major dimension's index changes slowest.
        # If "cp" comes before "tp" in mesh_dim_names, CP is the row dimension.
        if "cp" in dim_names and "tp" in dim_names:
            self._is_cp_row_major = dim_names.index("cp") < dim_names.index("tp")
        else:
            self._is_cp_row_major = True

    def __call__(self, features) -> list[dict[str, Any]]:
        """Process batches of data and create shards for each context parallelism rank.

        Args:
            features: List of tokenized sequences, each containing 'input_ids' and optionally 'labels'.

        Returns:
            A list of dictionaries, each containing a shard of the batch for a given context parallelism rank.
        """
        batch = self.collator(features)

        # Remove the attention mask from the batch, it's not valid for CP.
        batch.pop("attention_mask", None)

        if self.is_causal_lm:
            labels = torch.nn.functional.pad(batch["labels"], (0, 1), value=-100)
            batch["labels"] = labels[..., 1:].contiguous()

        combined_batch = []
        for cp_rank in range(self.cp_world_size):
            input_ids_sharded, labels_sharded = _split_batch_by_cp_rank(
                cu_seqlens_padded=batch.get("cu_seq_lens_q_padded", None),  # This will be None for BSHD format.
                input_ids_padded=batch["input_ids"],
                labels_padded=batch["labels"],
                qvk_format=self.qkv_format,
                cp_rank=cp_rank,
                cp_world_size=self.cp_world_size,
            )
            batch_shard = dict(batch)
            batch_shard["input_ids"] = input_ids_sharded
            if self.is_causal_lm:
                batch_shard["shift_labels"] = labels_sharded
                batch_shard["labels"] = None
            else:
                batch_shard["labels"] = labels_sharded
            # Now determine the max length of the sequence.
            if self.qkv_format == "thd":
                seqlens_q = batch_shard["cu_seq_lens_q_padded"][1:] - batch_shard["cu_seq_lens_q_padded"][:-1]
                max_length = seqlens_q.max().item()
            elif self.qkv_format == "bshd":
                max_length = batch["input_ids"].shape[1]
            else:
                raise ValueError(f"Unsupported qvk_format: {self.qkv_format}!")

            batch_shard["max_length_k"] = batch_shard["max_length_q"] = ((max_length + 63) // 64) * 64
            combined_batch.append(batch_shard)

        if self.tp_world_size is not None:
            # Replicate each CP shard for TP ranks. The ordering depends on which dimension forms the rows in the
            # flattened mesh.
            if self._is_cp_row_major:
                # Flattened mesh: [(cp0,tp0), (cp0,tp1), (cp1,tp0), (cp1,tp1)]
                # Output: [cp0, cp0, cp1, cp1]
                combined_batch = [batch for batch in combined_batch for _ in range(self.tp_world_size)]
            else:
                # Flattened mesh: [(tp0,cp0), (tp0,cp1), (tp1,cp0), (tp1,cp1)]
                # Output: [cp0, cp1, cp0, cp1]
                combined_batch = [
                    combined_batch[cp_rank] for _ in range(self.tp_world_size) for cp_rank in range(self.cp_world_size)
                ]

        return combined_batch


class ContextParallelDataLoaderWrapper:
    """A dataloader that is aware of context and tensor parallelism."""

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader | None,
        cp_tp_mesh: torch.distributed.device_mesh.DeviceMesh,
    ):
        """A dataloader wrapper that distributes the data across the context and tensor parallelism groups.

        This class materializes a single dataloader for each data parallel mesh rank, and splits / replicates the data
        from this dataloader across the context and tensor parallelism groups.

        Args:
            dataloader: The dataloader to use.
            cp_tp_mesh: The context parallel mesh, or a flattened, combined context parallel and tensor parallel mesh.
                If a flattened mesh is provided, the cp / tp dimensions should be in the order they appeared in the
                mesh_dim_names as passed to DataCollatorForContextParallel.
        """
        if cp_tp_mesh.get_local_rank() == 0:
            assert dataloader is not None, "dataloader must be provided on rank 0"
            self.dataloader = dataloader

        else:
            assert dataloader is None, "Dataloader on non-rank 0 will not be used"

        self.cp_tp_rank = cp_tp_mesh.get_local_rank()
        self.cp_tp_group = cp_tp_mesh.get_group()
        self.num_cp_tp_ranks = cp_tp_mesh.size()
        self._iterator = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_result: Any = None
        self._cuda_device: int | None = None

        logger.debug(
            "Created ContextParallelDataLoaderWrapper on global rank %s, cp rank %s",
            torch.distributed.get_rank() if torch.distributed.is_initialized() else "<not initialized>",
            self.cp_tp_rank,
        )

    def __iter__(self):
        """Make the dataloader iterable."""
        if self.cp_tp_rank == 0:
            self._iterator = iter(self.dataloader)  # < --- collator output.
        self.close()
        # Capture CUDA device from main thread; torch.cuda.set_device is per-thread,
        # so the background thread needs to set it explicitly.
        self._cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._kick_prefetch()
        return self

    @nvtx.annotate("ContextParallelDataLoaderWrapper __next__", color="blue")
    def __next__(self):
        """Get the batch from the dataloader for the current CP rank."""
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
        result = self._prefetch_result
        if isinstance(result, Exception):
            self._prefetch_thread = None
            raise result
        self._kick_prefetch()
        return result

    def _kick_prefetch(self):
        """Start a background thread to prefetch exactly one batch via scatter."""
        self._prefetch_thread = threading.Thread(target=self._do_one_prefetch, daemon=True)
        self._prefetch_thread.start()

    def _do_one_prefetch(self):
        """Fetch one batch in the background.

        This function calls the _send_data_to_cp_tp_ranks function to materialize the next batches for all ranks in the
        given CP/TP group, and uses torch.distributed.scatter_object_list to scatter these batches to their
        corresponding ranks. The result is stored in _prefetch_result, and returned when __next__ is called.
        """
        if self._cuda_device is not None:
            torch.cuda.set_device(self._cuda_device)
        try:
            self._prefetch_result = self._send_data_to_cp_tp_ranks()
        except StopIteration as e:
            self._prefetch_result = e
        except Exception as e:
            self._prefetch_result = e

    def close(self):
        """Stop the prefetch thread. Must be called before destroy_process_group()."""
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=10)
            self._prefetch_thread = None

    @nvtx.annotate("ContextParallelDataLoaderWrapper _send_data_to_cp_tp_ranks", color="green")
    def _send_data_to_cp_tp_ranks(self):
        """Send data to all the CP/TP ranks.

        This function will get the batch from the dataloader on CP rank 0, and then determine
        the shards for all the different CP group members.
        combined_batch = [<cp_rank_0_shard>, <cp_rank_1_shard>, ..., <cp_rank_n_shard>]
        Then it will scatter the shards to the different CP group members.
        The shards are then combined into a single batch and returned to the caller
        for the current CP rank.

        If tensor parallelism is also being used, the combined batch will look like:
        combined_batch = [<cp0_shard>, <cp0_shard>, ..., <cp1_shard>, <cp1_shard>, ...]
        where there are cp_world_size shards, and each shard is replicated tp_world_size times. The ordering of the
        shards depends on which dimension forms the rows in the flattened mesh.

        Scalability:
            Rank 0's work grows linearly with CP size, but the other ranks do not need to store all the shards so they
            do not grow linearly with CP size.

        Args:
            None

        Returns:
            batch: The batch for the current CP/TP rank.

        """
        try:
            with nvtx.annotate("ContextParallelDataLoaderWrapper next batch", color="green"):
                combined_batch = next(self._iterator) if self.cp_tp_rank == 0 else None
        except StopIteration as ex:
            # If we encounter a StopIteration in the dataloader, we want to raise this error on all the CP ranks, so
            # that the dataloader can be restarted.
            combined_batch = [ex] * self.num_cp_tp_ranks

        batch_on_this_rank = _scatter_batch_to_cp_tp_ranks(combined_batch, self.cp_tp_group)

        if isinstance(batch_on_this_rank, StopIteration):
            raise batch_on_this_rank

        return batch_on_this_rank

    def state_dict(self):
        """Get the state dict by delegating to the dataloader."""
        if self.cp_tp_rank != 0:
            return {}
        elif hasattr(self.dataloader, "state_dict"):
            return {"dataloader": self.dataloader.state_dict()}
        else:
            logger.warning(
                "Attempting to get the state dict of the dataloader, but the dataloader does not support state_dict, "
                "returning empty dict"
            )
            return {"dataloader": {}}

    def load_state_dict(self, state_dict):
        """Load the state dict by delegating to the dataloader."""
        if self.cp_tp_rank != 0:
            return
        elif hasattr(self.dataloader, "load_state_dict"):
            self.dataloader.load_state_dict(state_dict["dataloader"])
        else:
            logger.warning(
                "Attempting to load the state dict of the dataloader, but the dataloader does not support "
                "load_state_dict, returning without loading the state dict."
            )
            return

    @property
    def num_workers(self):
        """Get the number of workers of the dataloader."""
        if self.cp_tp_rank != 0:
            return 0
        else:
            return self.dataloader.num_workers


def _split_sample_by_num_tokens(sample: dict[str, Any], num_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a sample dictionary at a specified number of tokens.

    This function splits a sample into two parts: the first part contains exactly `num_tokens` tokens,
    and the second part contains the remaining tokens. All fields that are sequences (input_ids, attention_mask,
    token_type_ids, labels, etc.) are split accordingly.

    Args:
        sample: Dictionary containing sample data with fields like input_ids, attention_mask, token_type_ids, labels, etc.
        num_tokens: Number of tokens to include in the first part of the split.

    Returns:
        A tuple of two dictionaries: (first_part, remaining_part), where:
        - first_part contains the first `num_tokens` tokens from each sequence field
        - remaining_part contains the remaining tokens from each sequence field

    Example:
        >>> sample = {
        ...     "input_ids": [0, 5, 6, 7, 8, 9, 2],
        ...     "attention_mask": [1, 1, 1, 1, 1, 1, 1],
        ...     "labels": [0, 5, 6, 7, 8, 9, 2]
        ... }
        >>> first, remaining = split_sample_by_num_tokens(sample, 3)
        >>> first["input_ids"]  # [0, 5, 6]
        >>> remaining["input_ids"]  # [7, 8, 9, 2]
    """
    sample_length = len(sample["input_ids"])
    if num_tokens >= sample_length:
        raise ValueError(
            f"num_tokens ({num_tokens}) must be less than sample length ({sample_length}) to split the sample"
        )
    if num_tokens <= 0:
        raise ValueError(f"num_tokens ({num_tokens}) must be positive")

    first_part = {}
    remaining_part = {}

    # Fields that should be split by tokens (sequence fields)
    sequence_fields = ["input_ids", "attention_mask", "token_type_ids", "token_type", "labels"]

    for key, value in sample.items():
        if key in sequence_fields:
            # Handle both list and tensor inputs
            if isinstance(value, torch.Tensor):
                first_part[key] = value[:num_tokens].clone()
                remaining_part[key] = value[num_tokens:].clone()
            elif isinstance(value, list):
                first_part[key] = value[:num_tokens]
                remaining_part[key] = value[num_tokens:]
            else:
                # For other types, try to slice if possible
                try:
                    first_part[key] = value[:num_tokens]
                    remaining_part[key] = value[num_tokens:]
                except (TypeError, IndexError):
                    # If slicing doesn't work, copy the value to both parts
                    # This handles fields that shouldn't be split (like metadata)
                    first_part[key] = value
                    remaining_part[key] = value
        else:
            # For non-sequence fields, copy to both parts
            # This handles metadata fields that shouldn't be split
            first_part[key] = value
            remaining_part[key] = value

    return first_part, remaining_part


def _pt_flatten_collate(features: list[dict[str, list[int]]], return_position_ids: bool = False):
    """Flatten a list of tokenized samples into a single packed batch with cumulative sequence lengths.

    Args:
        features: List of tokenized samples, each containing at least ``input_ids``.
        return_position_ids: Whether to return position ids for each token.

    Returns:
        A dictionary with packed ``input_ids``, ``cu_seq_lens_q``/``cu_seq_lens_k``, and
        ``max_length_q``/``max_length_k``.
    """
    is_labels_provided = "labels" in features[0]
    sample_lengths = [len(sample["input_ids"]) for sample in features]

    batch = {}
    batch["max_length_q"] = batch["max_length_k"] = max(sample_lengths)
    batch["input_ids"] = torch.tensor(
        [[token for sample in features for token in sample["input_ids"]]], dtype=torch.int64
    )
    if is_labels_provided:
        batch["labels"] = torch.tensor(
            [[label for sample in features for label in sample["labels"]]], dtype=torch.int64
        )
    cu_seq_lens = torch.zeros(len(features) + 1, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(torch.tensor(sample_lengths), dim=0, dtype=torch.int32)
    batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = cu_seq_lens
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.tensor(
            [[v for sample in features for v in sample["attention_mask"]]], dtype=torch.int64
        )
    if return_position_ids:
        batch["position_ids"] = torch.hstack(
            [torch.arange(sample_len, dtype=torch.int64) for sample_len in sample_lengths]
        ).unsqueeze(0)

    return batch


def _find_seq_dim(tensor: torch.Tensor, seq_len: int) -> int:
    """Find which dimension of tensor matches the expected sequence length.

    Args:
        tensor: The tensor to inspect.
        seq_len: The expected sequence length to match against tensor dimensions.

    Returns:
        The dimension index that matches the sequence length.

    Raises:
        ValueError: If no dimension matches the expected sequence length.
    """
    if tensor.ndim == 1:
        if tensor.shape[0] == seq_len:
            return 0
        raise ValueError(f"1D tensor shape {tensor.shape} doesn't match sequence length {seq_len}")
    elif tensor.ndim >= 2:
        if tensor.shape[1] == seq_len:
            return 1
        elif tensor.shape[0] == seq_len:
            return 0
        raise ValueError(f"Tensor shape {tensor.shape} doesn't match sequence length {seq_len} in dim 0 or 1")
    raise ValueError(f"Unexpected tensor ndim={tensor.ndim}")


def _process_tensor_thd(
    val: torch.Tensor | None,
    seq_len: int,
    slice_sizes: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    cp_rank: int,
    total_slices: int,
) -> torch.Tensor | None:
    """Extract the THD context-parallel shard for a single tensor.

    For each sequence in the batch, selects two slices (one from the beginning and one from the end)
    corresponding to the given CP rank, following the zigzag CP sharding pattern.

    Args:
        val: The tensor to shard, or None (returned as-is).
        seq_len: Total sequence length (from cu_seqlens_padded[-1]).
        slice_sizes: Per-sequence slice sizes, computed as sequence_lengths // total_slices.
        cu_seqlens_padded: Cumulative sequence lengths including padding.
        cp_rank: The context parallelism rank index.
        total_slices: Total number of slices per sequence (2 * cp_world_size).

    Returns:
        The sharded tensor for the given CP rank, or None if val is None.
    """
    if val is None:
        return val

    seq_dim = _find_seq_dim(val, seq_len)

    cp_rank_slices = []
    for slice_size, seq_start in zip(slice_sizes, cu_seqlens_padded[:-1]):
        # 1st segment
        cp_rank_slices.append(
            torch.arange(
                seq_start + (cp_rank * slice_size),
                seq_start + ((cp_rank + 1) * slice_size),
                device=val.device,
            )
        )

        # 2nd segment
        cp_rank_slices.append(
            torch.arange(
                seq_start + ((total_slices - cp_rank - 1) * slice_size),
                seq_start + ((total_slices - cp_rank) * slice_size),
                device=val.device,
            )
        )

    return val.index_select(seq_dim, torch.cat(cp_rank_slices))


def _process_tensor_bshd(
    val: torch.Tensor | None,
    cp_rank: int,
    cp_world_size: int,
) -> torch.Tensor | None:
    """Extract the BSHD context-parallel shard for a single tensor.

    Splits a BSHD-format tensor along the sequence dimension (dim=1) into 2*cp_world_size chunks,
    then selects the two chunks corresponding to the given CP rank (zigzag pattern).

    Args:
        val: The tensor to shard, or None (returned as-is).
        cp_rank: The context parallelism rank index.
        cp_world_size: Total number of context parallelism ranks.

    Returns:
        The sharded tensor for the given CP rank, or None if val is None.

    Raises:
        ValueError: If the tensor has fewer than 2 dimensions or its sequence length
            is not divisible by 2 * cp_world_size.
    """
    if val is None:
        return val

    if val.ndim < 2:
        raise ValueError(f"BSHD format requires at least 2D tensors, got {val.ndim}D")

    seq_len = val.shape[1]

    # Calculate chunk size
    total_chunks = 2 * cp_world_size
    chunk_size = seq_len // total_chunks

    if seq_len % total_chunks != 0:
        raise ValueError(
            f"Sequence length {seq_len} must be divisible by {total_chunks} "
            f"(2 * cp_world_size) for BSHD context parallelism"
        )

    # Determine which chunks this rank should get
    # Rank 0 gets chunks [0, total_chunks-1]
    # Rank 1 gets chunks [1, total_chunks-2]
    # Rank k gets chunks [k, total_chunks-k-1]
    chunk_indices = [cp_rank, total_chunks - cp_rank - 1]

    # Collect slices for this rank
    rank_slices = []
    for chunk_idx in chunk_indices:
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size
        rank_slices.append(torch.arange(start_idx, end_idx, device=val.device))

    # Concatenate indices for all chunks this rank should get
    indices = torch.cat(rank_slices)

    # Select along sequence dimension (dim=1)
    return val.index_select(1, indices)


def _pt_pad_to_multiple_of(batch: dict[str, Any], pad_to_multiple_of: int, token_pad: int, label_pad: int):
    """Pad a batch to a multiple of pad_to_multiple_of.

    Appends a mock sequence to the end of the batch with the given token_pad and label_pad to make the total number of
    tokens divisible by pad_to_multiple_of.

    Args:
        batch: Input batch, possibly containing labels and/or cu_seq_lens / max_length keys.
        pad_to_multiple_of: Multiple to pad to.
        token_pad: Token to pad with.
        label_pad: Label to pad with.

    Returns:
        Batch dictionary with padded input_ids, labels, cu_seq_lens_q, cu_seq_lens_k, max_length_q, and max_length_k.
    """
    # Number of tokens we need to pad to make the total number of tokens divisible by pad_to_multiple_of
    remainder = -batch["input_ids"].numel() % pad_to_multiple_of

    if remainder == 0:
        return batch

    batch["input_ids"] = torch.cat(
        [batch["input_ids"], torch.full((1, remainder), token_pad, dtype=batch["input_ids"].dtype)], dim=1
    )

    if "labels" in batch:
        batch["labels"] = torch.cat(
            [batch["labels"], torch.full((1, remainder), label_pad, dtype=batch["labels"].dtype)], dim=1
        )

    if "cu_seq_lens_q" in batch:
        batch["cu_seq_lens_q"] = torch.cat(
            [
                batch["cu_seq_lens_q"],
                torch.tensor([batch["cu_seq_lens_q"][-1] + remainder], dtype=batch["cu_seq_lens_q"].dtype),
            ],
            dim=0,
        )
        batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]

    if "max_length_q" in batch:
        batch["max_length_q"] = max(batch["max_length_q"], remainder)
        batch["max_length_k"] = batch["max_length_q"]

    if "attention_mask" in batch:
        batch["attention_mask"] = torch.cat(
            [batch["attention_mask"], torch.zeros((1, remainder), dtype=batch["attention_mask"].dtype)], dim=1
        )

    if "position_ids" in batch:
        batch["position_ids"] = torch.cat(
            [batch["position_ids"], torch.arange(remainder, dtype=batch["position_ids"].dtype).unsqueeze(0)], dim=1
        )

    return batch


# TODO(@jomitchell): Once this gets merged: https://github.com/NVIDIA/TransformerEngine/pull/2387
# we can replace this with the one in TransformerEngine.
@nvtx.annotate("collator._split_batch_by_cp_rank", color="green")
def _split_batch_by_cp_rank(
    cu_seqlens_padded: torch.Tensor | None,
    input_ids_padded: torch.Tensor,
    labels_padded: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup | None = None,
    qvk_format: str = "thd",
    cp_rank: int | None = None,
    cp_world_size: int | None = None,
):
    """Slice batch input along sequence dimension into multiple chunks for THD or BSHD format.

    This function is intended for use in self attention. It will not work for cross attention because
    it does not handle the case where the sequence length of the query and key are different.
    Which are parallelized across GPUs in a context parallel group.
    This version works with variable-length sequences using cumulative sequence lengths for THD format,
    and with padded sequences for BSHD format.

    Args:
        cu_seqlens_padded: Cumulative sequence length. Required for THD format, optional for BSHD format.
        input_ids_padded: Input IDs.
        labels_padded: Labels.
        cp_group: Context parallel group.
        qvk_format: Format of the input data ("thd" or "bshd").
        cp_world_size: The size of the context parallelism group. If provided, the function will use this value to determine the rank.
        cp_rank: Optional manual CP rank index. When provided, the function shards tensors as if it
            were executing on that rank without querying `torch.distributed.get_rank`.
    """
    if qvk_format not in ["thd", "bshd", "sbhd"]:
        raise ValueError(f"Unsupported qvk_format: {qvk_format}!")

    if cp_world_size is None or cp_world_size <= 1:
        # No splitting needed
        return input_ids_padded, labels_padded

    if cp_rank is None:
        cp_rank = torch.distributed.get_rank(group=cp_group)
    elif not (0 <= cp_rank < cp_world_size):
        raise ValueError(f"cp_rank must be in [0, {cp_world_size}), but received {cp_rank}.")

    if qvk_format == "thd":
        if cu_seqlens_padded is None:
            raise ValueError("cu_seqlens_padded is required for THD format")

        # Calculate the chunk sizes for each sequence
        total_slices_of_any_sequence = 2 * cp_world_size
        slice_sizes = (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]) // total_slices_of_any_sequence

        # Ensure cu_seqlens_padded[-1] is a Python int, not a 0-dim tensor
        last_elem = cu_seqlens_padded[-1]
        seq_len_val = last_elem.item() if isinstance(last_elem, torch.Tensor) else last_elem

        input_ids_padded = _process_tensor_thd(
            input_ids_padded, seq_len_val, slice_sizes, cu_seqlens_padded, cp_rank, total_slices_of_any_sequence
        )
        labels_padded = _process_tensor_thd(
            labels_padded, seq_len_val, slice_sizes, cu_seqlens_padded, cp_rank, total_slices_of_any_sequence
        )

    elif qvk_format == "bshd":
        input_ids_padded = _process_tensor_bshd(input_ids_padded, cp_rank, cp_world_size)
        labels_padded = _process_tensor_bshd(labels_padded, cp_rank, cp_world_size)

    else:
        raise ValueError(f"Support not implemented yet for qvk_format: {qvk_format}!")

    return input_ids_padded, labels_padded


class BatchType(TypedDict):
    """The fields in the batch dictionary for THD context parallel."""

    input_ids: torch.Tensor
    labels: torch.Tensor | None
    shift_labels: torch.Tensor | None
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor
    cu_seq_lens_q_padded: torch.Tensor
    cu_seq_lens_k_padded: torch.Tensor
    max_length_q: int
    max_length_k: int
    pad_between_seqs: bool


@nvtx.annotate("collator._scatter_batch_to_cp_tp_ranks", color="green")
def _scatter_batch_to_cp_tp_ranks(
    all_batches: list[BatchType] | list[StopIteration], cp_tp_group: torch.distributed.ProcessGroup | None = None
) -> BatchType | StopIteration:
    """Scatter a batch to all the CP ranks.

    Args:
        all_batches (list[BatchType] | list[StopIteration]): A list of already-sharded batches to scatter to the CP/TP
            ranks.
        cp_tp_group (torch.distributed.ProcessGroup | None): The process group to scatter the batches to.

    Returns:
        BatchType | StopIteration: The batch on this rank.
    """
    scatter_object_output_list = [None]
    # Note: This does not provide an async_op handle. Thus its blocking.
    torch.distributed.scatter_object_list(
        scatter_object_output_list=scatter_object_output_list,
        scatter_object_input_list=all_batches,
        group=cp_tp_group,
        group_src=0,
    )
    return scatter_object_output_list[0]
