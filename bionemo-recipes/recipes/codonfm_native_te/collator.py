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

"""Context parallel data collation and distribution utilities.

Provides model-agnostic utilities for context parallelism (CP):
- DataCollatorForContextParallel: Splits THD/BSHD batches into CP shards
- ContextParallelDataLoaderWrapper: Distributes shards across CP ranks
- _split_batch_by_cp_rank: Core batch splitting logic

Adapted from bionemo-recipes/models/esm2/collator.py for use with CodonFM's
custom CodonTHDCollator. The utilities here work with any collator that produces
THD format output (input_ids, labels, cu_seq_lens_q/k).
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, TypedDict

import nvtx
import torch
from transformers import DataCollator


logger = logging.getLogger(__name__)


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
        cp_world_size: The size of the context parallelism group.
        cp_rank: Optional manual CP rank index.
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
