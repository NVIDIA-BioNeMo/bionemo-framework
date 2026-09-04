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

import copy
import threading
from typing import Dict, Iterator, List
from unittest import mock

import pytest
import torch
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp
from transformers import DataCollatorForLanguageModeling

from collator import (
    BatchType,
    ContextParallelDataLoaderWrapper,
    DataCollatorForContextParallel,
    DataCollatorWithFlattening,
    _split_batch_by_cp_rank,
)


def get_dummy_data_thd_with_padding_dp0(cp_size: int):
    pid = 1  # The pad token id.
    label_pad = -100  # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([1, 2, 3, 5, 6])
    labels = torch.tensor(
        [
            10,
            20,
            30,
            50,
            60,
        ]
    )
    cu_seqlens_q = torch.tensor([0, 3, 5])
    divisibility_factor = 2 * cp_size

    input_ids_padded, labels_padded, cu_seqlens_q_padded = pad_thd_sequences_for_cp(
        input_ids.unsqueeze(0),
        labels.unsqueeze(0),
        cu_seqlens_q,
        divisibility_factor,
        padding_token_id=pid,
        padding_label_id=label_pad,
    )

    batch = {
        "input_ids": input_ids_padded.unsqueeze(0).to(torch.int64),  # Add batch dim: [1, seq_len]
        "labels": labels_padded.unsqueeze(0).to(torch.int64),  # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q_padded.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q_padded.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_with_padding_dp1(cp_size: int):
    pid = 1  # The pad token id.
    label_pad = -100  # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor(
        [
            9,
            10,
            11,
            13,
            14,
            15,
        ]
    )
    labels = torch.tensor(
        [
            90,
            100,
            110,
            130,
            140,
            150,
        ]
    )
    cu_seqlens_q = torch.tensor([0, 3, 6])
    divisibility_factor = 2 * cp_size

    input_ids_padded, labels_padded, cu_seqlens_q_padded = pad_thd_sequences_for_cp(
        input_ids.unsqueeze(0),
        labels.unsqueeze(0),
        cu_seqlens_q,
        divisibility_factor,
        padding_token_id=pid,
        padding_label_id=label_pad,
    )

    batch = {
        "input_ids": input_ids_padded.unsqueeze(0).to(torch.int64),  # Add batch dim: [1, seq_len]
        "labels": labels_padded.unsqueeze(0).to(torch.int64),  # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q_padded.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q_padded.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_dp0_nopadding():
    # Make some fake data.
    input_ids = torch.tensor(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,  # 8 tokens
        ]
    )
    labels = torch.tensor(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
        ]
    )
    cu_seqlens_q = torch.tensor([0, 8])
    batch = {
        "input_ids": input_ids.unsqueeze(0).to(torch.int64),  # Add batch dim: [1, seq_len]
        "labels": labels.unsqueeze(0).to(torch.int64),  # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


def get_dummy_data_thd_dp1_nopadding():
    # Make some fake data.
    input_ids = torch.tensor(
        [
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,  # 8 tokens
        ]
    )
    labels = torch.tensor(
        [
            90,
            100,
            110,
            120,
            130,
            140,
            150,
            160,
        ]
    )
    cu_seqlens_q = torch.tensor([0, 8])
    batch = {
        "input_ids": input_ids.unsqueeze(0).to(torch.int64),  # Add batch dim: [1, seq_len]
        "labels": labels.unsqueeze(0).to(torch.int64),  # [1, seq_len]
        "cu_seq_lens_q_padded": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k_padded": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_q": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "cu_seq_lens_k": cu_seqlens_q.to(torch.int32),  # Keep 1D - int32
        "max_length_q": 8,
        "max_length_k": 8,
    }
    return batch


class _DummyCollatorMesh:
    """Dummy device mesh for testing DataCollatorForContextParallel.

    Supports the `.mesh_dim_names` and `.size(dim_name)` interface that the collator needs.
    """

    def __init__(self, cp_size: int, tp_size: int | None = None, tp_first: bool = False):
        if tp_size is not None:
            if tp_first:
                self._sizes = {"tp": tp_size, "cp": cp_size}
            else:
                self._sizes = {"cp": cp_size, "tp": tp_size}
        else:
            self._sizes = {"cp": cp_size}

    @property
    def mesh_dim_names(self):
        return tuple(self._sizes.keys())

    def size(self, dim=None):
        if dim is None:
            result = 1
            for v in self._sizes.values():
                result *= v
            return result
        if isinstance(dim, int):
            return list(self._sizes.values())[dim]
        return self._sizes[dim]


class _DummyLoader:
    """Minimal iterable that always yields the same object (batch or list)."""

    def __init__(self, batch):
        self._batch = batch

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        return copy.deepcopy(self._batch)


class _DummyDeviceMesh:
    """Dummy device mesh for testing ContextParallelDataLoaderWrapper."""

    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank
        self._group = mock.MagicMock()  # Mock process group

    def get_local_rank(self) -> int:
        """Return the local rank within this mesh."""
        return self._rank

    def get_group(self):
        """Return the process group."""
        return self._group

    def size(self) -> int:
        """Return the size of the mesh."""
        return self._size


def test_pad_thd_sequences_for_cp():
    pid = 1  # The pad token id.
    label_pad = -100  # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # 7 tokens
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,  # 11 tokens
            3,
            3,
            3,
            3,
            3,  # 5 tokens
        ]
    )
    labels = torch.tensor([10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 5, 6, 7, 8, 9])
    cu_seqlens_q = torch.tensor([0, 7, 18, 23])
    divisibility_factor = 4

    input_ids_padded, labels_padded, cu_seqlens_q_padded = pad_thd_sequences_for_cp(
        input_ids.unsqueeze(0),
        labels.unsqueeze(0),
        cu_seqlens_q,
        divisibility_factor,
        padding_token_id=pid,
        padding_label_id=label_pad,
    )
    expected_input_ids = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, pid, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, pid, 3, 3, 3, 3, 3, pid, pid, pid]
    )

    expected_labels = torch.tensor(
        [
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            label_pad,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            label_pad,
            5,
            6,
            7,
            8,
            9,
            label_pad,
            label_pad,
            label_pad,
        ]
    )

    expected_cu_seqlens_padded = torch.tensor([0, 8, 20, 28])

    assert torch.equal(input_ids_padded, expected_input_ids)
    assert torch.equal(labels_padded, expected_labels)
    assert torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_padded)


def test_dataloader_scatter_nopadding():
    """
    Test single sequence on two dataloader ranks with no additional padding, with CP=2, DP=2, ensure that the data is scattered correctly.
    There are going to be 4 shards. CP0, CP1 (for context parallel) and DP0, DP1 (for data parallel).
    DP0 will return [1,2,3,4,5,6,7,8] and DP1 will return [9,10,11,12,13,14,15,16].
    CP0 will receive [1,2,7,8] and CP1 will receive [3,4,5,6]. (from DP0)
    CP1 will receive [9,10,15,16] and CP0 will receive [11,12,13,14]. (from DP1)

        |   DP0   |    DP1        |
    CP0 | 1,2,7,8 | 9, 10, 15, 16 |
    CP1 | 3,4,5,6 | 11, 12, 13, 14|
    """
    cp_size = 2

    def run_roundtrip(base_batch):
        combined_batch = [
            dict(
                base_batch,
                **{
                    "input_ids": _split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_size,
                    )[0],
                    "labels": _split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_size,
                    )[1],
                },
            )
            for cp_rank in range(cp_size)
        ]
        cp_mesh_rank0 = _DummyDeviceMesh(size=cp_size, rank=0)
        cp_mesh_rank1 = _DummyDeviceMesh(size=cp_size, rank=1)
        loader_rank0 = ContextParallelDataLoaderWrapper(_DummyLoader(combined_batch), cp_mesh_rank0)
        loader_rank1 = ContextParallelDataLoaderWrapper(None, cp_mesh_rank1)

        scatter_payload: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        data_ready = threading.Event()

        def fake_scatter(
            *,
            scatter_object_output_list,
            scatter_object_input_list,
            group,
            group_src,
        ):
            if scatter_object_input_list is not None:
                # Rank 0: store the full payload and return shard 0
                scatter_payload["data"] = scatter_object_input_list
                data_ready.set()
                scatter_object_output_list[0] = scatter_object_input_list[0]
            else:
                # Rank 1: wait for rank 0's data, then return shard 1
                data_ready.wait(timeout=5)
                scatter_object_output_list[0] = scatter_payload["data"][1]

        with (
            mock.patch("collator.torch.distributed.scatter_object_list", side_effect=fake_scatter),
            mock.patch("collator.torch.distributed.barrier", return_value=None),
        ):
            iter(loader_rank0)
            iter(loader_rank1)

            batch_cp0 = next(loader_rank0)
            batch_cp1 = next(loader_rank1)

        return batch_cp0, batch_cp1

    batch_dp0_cp0, batch_dp0_cp1 = run_roundtrip(get_dummy_data_thd_dp0_nopadding())
    torch.testing.assert_close(batch_dp0_cp0["input_ids"], torch.tensor([[1, 2, 7, 8]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp0["labels"], torch.tensor([[10, 20, 70, 80]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp1["input_ids"], torch.tensor([[3, 4, 5, 6]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp1["labels"], torch.tensor([[30, 40, 50, 60]], dtype=torch.int64))

    batch_dp1_cp0, batch_dp1_cp1 = run_roundtrip(get_dummy_data_thd_dp1_nopadding())

    torch.testing.assert_close(batch_dp1_cp0["input_ids"], torch.tensor([[9, 10, 15, 16]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp0["labels"], torch.tensor([[90, 100, 150, 160]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp1["input_ids"], torch.tensor([[11, 12, 13, 14]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp1["labels"], torch.tensor([[110, 120, 130, 140]], dtype=torch.int64))


def test_dataloader_scatter_with_pad_between_seqs():
    """
    Here we are going to test two sequences using two dataloaders with padding. We use CP=2, DP=2 and
    ensure that the data is scattered correctly.
    There are going to be 4 shards. CP0, CP1 (for context parallel) and DP0, DP1 (for data parallel).
    DP0 will return [1,2,3,<p> | 5,6,<p>,<p>]
    DP1 will return [9,10,11,<p> | 13,14,15,<p>]

    We notice that CP sharding grabs slices of each sequence. Thus, CP0 grabs the first and last slices, while CP1 grabs the middle slices.
        |   DP0       |       DP1      |
    CP0 | 1,<p>,5,<p> | 9, <p>, 13, <p>|
    CP1 | 2,3,6, <p>  | 10, 11, 14, 15 |
    """
    cp_size = 2

    def run_roundtrip(base_batch):
        combined_batch = [
            dict(
                base_batch,
                **{
                    "input_ids": _split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_size,
                    )[0],
                    "labels": _split_batch_by_cp_rank(
                        cu_seqlens_padded=base_batch["cu_seq_lens_q_padded"],
                        input_ids_padded=base_batch["input_ids"],
                        labels_padded=base_batch["labels"],
                        qvk_format="thd",
                        cp_rank=cp_rank,
                        cp_world_size=cp_size,
                    )[1],
                },
            )
            for cp_rank in range(cp_size)
        ]
        cp_mesh_rank0 = _DummyDeviceMesh(size=cp_size, rank=0)
        cp_mesh_rank1 = _DummyDeviceMesh(size=cp_size, rank=1)
        loader_rank0 = ContextParallelDataLoaderWrapper(_DummyLoader(combined_batch), cp_mesh_rank0)
        loader_rank1 = ContextParallelDataLoaderWrapper(None, cp_mesh_rank1)

        scatter_payload: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        data_ready = threading.Event()

        def fake_scatter(
            *,
            scatter_object_output_list,
            scatter_object_input_list,
            group,
            group_src,
        ):
            if scatter_object_input_list is not None:
                # Rank 0: store the full payload and return shard 0
                scatter_payload["data"] = scatter_object_input_list
                data_ready.set()
                scatter_object_output_list[0] = scatter_object_input_list[0]
            else:
                # Rank 1: wait for rank 0's data, then return shard 1
                data_ready.wait(timeout=5)
                scatter_object_output_list[0] = scatter_payload["data"][1]

        with (
            mock.patch("collator.torch.distributed.scatter_object_list", side_effect=fake_scatter),
            mock.patch("collator.torch.distributed.barrier", return_value=None),
        ):
            iter(loader_rank0)
            iter(loader_rank1)

            batch_cp0 = next(loader_rank0)
            batch_cp1 = next(loader_rank1)

        return batch_cp0, batch_cp1

    batch_dp0_cp0, batch_dp0_cp1 = run_roundtrip(get_dummy_data_thd_with_padding_dp0(cp_size=2))

    torch.testing.assert_close(batch_dp0_cp0["input_ids"], torch.tensor([[1, 1, 5, 1]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp0_cp1["input_ids"], torch.tensor([[2, 3, 6, 1]], dtype=torch.int64))

    batch_dp1_cp0, batch_dp1_cp1 = run_roundtrip(get_dummy_data_thd_with_padding_dp1(cp_size=2))

    torch.testing.assert_close(batch_dp1_cp0["input_ids"], torch.tensor([[9, 1, 13, 1]], dtype=torch.int64))
    torch.testing.assert_close(batch_dp1_cp1["input_ids"], torch.tensor([[10, 11, 14, 15]], dtype=torch.int64))


def get_dummy_data_bshd_single_sequence(cp_size: int, seq_len: int = 8):
    """Create dummy BSHD format data with a single sequence.

    Args:
        cp_size: The size of the context parallelism group.
        seq_len: The sequence length (must be divisible by 2*cp_size).

    Returns:
        A dictionary containing input_ids and labels in BSHD format.
    """
    if seq_len % (2 * cp_size) != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by 2*cp_size ({2 * cp_size})")

    # Create a simple sequence: [1, 2, 3, ..., seq_len]
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.int64).unsqueeze(0)  # [1, seq_len]
    labels = torch.arange(10, 10 + seq_len, dtype=torch.int64).unsqueeze(0)  # [1, seq_len]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def get_dummy_data_bshd_multiple_sequences(cp_size: int, batch_size: int = 2, seq_len: int = 8):
    """Create dummy BSHD format data with multiple sequences.

    Args:
        cp_size: The size of the context parallelism group.
        batch_size: The batch size.
        seq_len: The sequence length (must be divisible by 2*cp_size).

    Returns:
        A dictionary containing input_ids and labels in BSHD format.
    """
    if seq_len % (2 * cp_size) != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by 2*cp_size ({2 * cp_size})")

    # Create sequences: each sequence starts at a different offset
    input_ids_list = []
    labels_list = []
    for i in range(batch_size):
        seq_input_ids = torch.arange(i * 100 + 1, i * 100 + seq_len + 1, dtype=torch.int64)
        seq_labels = torch.arange(i * 1000 + 10, i * 1000 + seq_len + 10, dtype=torch.int64)
        input_ids_list.append(seq_input_ids)
        labels_list.append(seq_labels)

    input_ids = torch.stack(input_ids_list)  # [batch_size, seq_len]
    labels = torch.stack(labels_list)  # [batch_size, seq_len]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def test_split_batch_by_cp_rank_bshd_single_sequence():
    """Test BSHD format splitting for a single sequence with CP=2.

    For a sequence of length 8 with CP=2:
    - Total chunks = 2 * 2 = 4
    - Chunk size = 8 / 4 = 2
    - CP rank 0 gets chunks [0, 3]: indices [0:2] and [6:8] -> [1,2,7,8]
    - CP rank 1 gets chunks [1, 2]: indices [2:4] and [4:6] -> [3,4,5,6]
    """
    cp_size = 2
    seq_len = 8
    batch = get_dummy_data_bshd_single_sequence(cp_size=cp_size, seq_len=seq_len)

    # Test CP rank 0
    input_ids_cp0, labels_cp0 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="bshd",
        cp_rank=0,
        cp_world_size=cp_size,
    )

    # CP rank 0 should get chunks [0, 3]: [1,2] and [7,8]
    expected_input_ids_cp0 = torch.tensor([[1, 2, 7, 8]], dtype=torch.int64)
    expected_labels_cp0 = torch.tensor([[10, 11, 16, 17]], dtype=torch.int64)

    torch.testing.assert_close(input_ids_cp0, expected_input_ids_cp0)
    torch.testing.assert_close(labels_cp0, expected_labels_cp0)

    # Test CP rank 1
    input_ids_cp1, labels_cp1 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="bshd",
        cp_rank=1,
        cp_world_size=cp_size,
    )

    # CP rank 1 should get chunks [1, 2]: [3,4] and [5,6]
    expected_input_ids_cp1 = torch.tensor([[3, 4, 5, 6]], dtype=torch.int64)
    expected_labels_cp1 = torch.tensor([[12, 13, 14, 15]], dtype=torch.int64)

    torch.testing.assert_close(input_ids_cp1, expected_input_ids_cp1)
    torch.testing.assert_close(labels_cp1, expected_labels_cp1)


def test_split_batch_by_cp_rank_bshd_multiple_sequences():
    """Test BSHD format splitting for multiple sequences with CP=2.

    For batch_size=2, seq_len=8 with CP=2:
    - Each sequence is split independently
    - Sequence 0: [1,2,3,4,5,6,7,8] (i=0, starts at 0*100+1=1)
    - Sequence 1: [101,102,103,104,105,106,107,108] (i=1, starts at 1*100+1=101)
    - CP rank 0 gets chunks [0, 3] from each sequence
    - CP rank 1 gets chunks [1, 2] from each sequence
    """
    cp_size = 2
    batch_size = 2
    seq_len = 8
    batch = get_dummy_data_bshd_multiple_sequences(cp_size=cp_size, batch_size=batch_size, seq_len=seq_len)

    # Test CP rank 0
    input_ids_cp0, labels_cp0 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="bshd",
        cp_rank=0,
        cp_world_size=cp_size,
    )

    # CP rank 0 should get chunks [0, 3] from each sequence
    # Sequence 0: [1,2] and [7,8] -> [1,2,7,8]
    # Sequence 1: [101,102] and [107,108] -> [101,102,107,108]
    expected_input_ids_cp0 = torch.tensor([[1, 2, 7, 8], [101, 102, 107, 108]], dtype=torch.int64)
    expected_labels_cp0 = torch.tensor([[10, 11, 16, 17], [1010, 1011, 1016, 1017]], dtype=torch.int64)

    torch.testing.assert_close(input_ids_cp0, expected_input_ids_cp0)
    torch.testing.assert_close(labels_cp0, expected_labels_cp0)

    # Test CP rank 1
    input_ids_cp1, labels_cp1 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=batch["input_ids"],
        labels_padded=batch["labels"],
        qvk_format="bshd",
        cp_rank=1,
        cp_world_size=cp_size,
    )

    # CP rank 1 should get chunks [1, 2] from each sequence
    # Sequence 0: [3,4] and [5,6] -> [3,4,5,6]
    # Sequence 1: [103,104] and [105,106] -> [103,104,105,106]
    expected_input_ids_cp1 = torch.tensor([[3, 4, 5, 6], [103, 104, 105, 106]], dtype=torch.int64)
    expected_labels_cp1 = torch.tensor([[12, 13, 14, 15], [1012, 1013, 1014, 1015]], dtype=torch.int64)

    torch.testing.assert_close(input_ids_cp1, expected_input_ids_cp1)
    torch.testing.assert_close(labels_cp1, expected_labels_cp1)


def test_split_batch_by_cp_rank_bshd_cp4():
    """Test BSHD format splitting with CP=4.

    For a sequence of length 16 with CP=4:
    - Total chunks = 2 * 4 = 8
    - Chunk size = 16 / 8 = 2
    - CP rank 0 gets chunks [0, 7]: [1,2] and [15,16]
    - CP rank 1 gets chunks [1, 6]: [3,4] and [13,14]
    - CP rank 2 gets chunks [2, 5]: [5,6] and [11,12]
    - CP rank 3 gets chunks [3, 4]: [7,8] and [9,10]
    """
    cp_size = 4
    seq_len = 16
    batch = get_dummy_data_bshd_single_sequence(cp_size=cp_size, seq_len=seq_len)

    # Test each CP rank
    for cp_rank in range(cp_size):
        input_ids_shard, labels_shard = _split_batch_by_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=batch["input_ids"],
            labels_padded=batch["labels"],
            qvk_format="bshd",
            cp_rank=cp_rank,
            cp_world_size=cp_size,
        )

        # Verify shape: should be [1, 4] (batch_size=1, 2 chunks * chunk_size=2)
        assert input_ids_shard.shape == (1, 4), (
            f"CP rank {cp_rank}: expected shape (1, 4), got {input_ids_shard.shape}"
        )
        assert labels_shard.shape == (1, 4), f"CP rank {cp_rank}: expected shape (1, 4), got {labels_shard.shape}"

        # Verify that all values are unique (no duplicates)
        unique_values = torch.unique(input_ids_shard)
        assert len(unique_values) == 4, f"CP rank {cp_rank}: expected 4 unique values, got {len(unique_values)}"

    # Verify that all ranks together reconstruct the original sequence
    all_shards = []
    for cp_rank in range(cp_size):
        input_ids_shard, _ = _split_batch_by_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=batch["input_ids"],
            labels_padded=batch["labels"],
            qvk_format="bshd",
            cp_rank=cp_rank,
            cp_world_size=cp_size,
        )
        all_shards.append(input_ids_shard.squeeze(0))

    # Concatenate all shards
    reconstructed = torch.cat(all_shards)
    # Sort to compare with original (chunks are interleaved)
    reconstructed_sorted = torch.sort(reconstructed)[0]
    original_sorted = torch.sort(batch["input_ids"].squeeze(0))[0]

    torch.testing.assert_close(reconstructed_sorted, original_sorted)


def test_split_batch_by_cp_rank_bshd_3d_tensor():
    """Test BSHD format splitting for 3D tensors (e.g., [batch, seq_len, hidden_dim]).

    This tests that the function works correctly for tensors with more than 2 dimensions.
    """
    cp_size = 2
    batch_size = 2
    seq_len = 8
    hidden_dim = 128

    # Create 3D tensors
    input_ids = torch.randn(batch_size, seq_len, hidden_dim)
    labels = torch.randn(batch_size, seq_len, hidden_dim)

    # Test CP rank 0
    input_ids_cp0, labels_cp0 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=input_ids,
        labels_padded=labels,
        qvk_format="bshd",
        cp_rank=0,
        cp_world_size=cp_size,
    )

    # Should split along seq_len dimension (dim=1)
    # CP rank 0 gets chunks [0, 3]: indices [0:2] and [6:8]
    expected_shape = (batch_size, 4, hidden_dim)  # 2 chunks * chunk_size=2
    assert input_ids_cp0.shape == expected_shape, f"Expected shape {expected_shape}, got {input_ids_cp0.shape}"
    assert labels_cp0.shape == expected_shape, f"Expected shape {expected_shape}, got {labels_cp0.shape}"

    # Verify the chunks are correct by checking indices
    # First chunk should be original[:, 0:2, :]
    torch.testing.assert_close(input_ids_cp0[:, 0:2, :], input_ids[:, 0:2, :])
    # Second chunk should be original[:, 6:8, :]
    torch.testing.assert_close(input_ids_cp0[:, 2:4, :], input_ids[:, 6:8, :])

    # Test CP rank 1
    input_ids_cp1, labels_cp1 = _split_batch_by_cp_rank(
        cu_seqlens_padded=None,
        input_ids_padded=input_ids,
        labels_padded=labels,
        qvk_format="bshd",
        cp_rank=1,
        cp_world_size=cp_size,
    )

    assert input_ids_cp1.shape == expected_shape
    # CP rank 1 gets chunks [1, 2]: indices [2:4] and [4:6]
    torch.testing.assert_close(input_ids_cp1[:, 0:2, :], input_ids[:, 2:4, :])
    torch.testing.assert_close(input_ids_cp1[:, 2:4, :], input_ids[:, 4:6, :])
    torch.testing.assert_close(labels_cp1[:, 0:2, :], labels[:, 2:4, :])
    torch.testing.assert_close(labels_cp1[:, 2:4, :], labels[:, 4:6, :])


def test_bshd_and_thd_equivalence(tokenizer):
    """Test that BSHD and THD formats produce equivalent CP shards for real protein sequences.

    This test verifies that when we shard data for context parallelism using both BSHD and THD
    formats, the actual protein tokens (excluding padding) end up on the same CP ranks.

    For CP=2, each sequence is split into 4 chunks (2*cp_size). Each CP rank gets 2 chunks:
    - CP rank 0 gets chunks [0, 3] (first and last)
    - CP rank 1 gets chunks [1, 2] (middle)
    """
    cp_size = 2
    divisibility_factor = 2 * cp_size  # = 4

    # Use proteins with lengths that will result in clean padding
    protein1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLL"  # 30 AA -> 32 tokens with special tokens
    protein2 = "MSHHWGYGKHNGPEHWHKDFPIAKGERFLL"  # 30 AA -> 32 tokens with special tokens

    tok1 = tokenizer(protein1, add_special_tokens=True)
    tok2 = tokenizer(protein2, add_special_tokens=True)

    # For BSHD format: pad each sequence to same length (multiple of divisibility_factor)
    # We need the pad length to match what THD will use
    assert len(tok1["input_ids"]) == len(tok2["input_ids"])
    assert len(tok1["input_ids"]) % divisibility_factor == 0

    # Create BSHD collator - no MLM masking to make comparison deterministic
    data_collator_bshd_base = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    batch_bshd = data_collator_bshd_base([tok1, tok2])

    # Create THD collator with per-sequence padding for CP
    data_collator_thd = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        pad_sequences_to_be_divisible_by=divisibility_factor,
    )
    batch_thd = data_collator_thd([tok1, tok2])

    # Verify the unsharded batches have the expected structure
    # BSHD: [batch_size, seq_len] with padding at the end of each sequence
    assert batch_bshd["input_ids"].shape[0] == 2, "BSHD batch should have 2 sequences"
    # THD: [1, total_tokens] with sequences concatenated
    assert batch_thd["input_ids"].shape[0] == 1, "THD batch should have batch_size=1"

    # Get sequence lengths for THD
    cu_seqlens_padded = batch_thd["cu_seq_lens_q_padded"]
    seq_lengths_thd = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]

    # Now split both formats by CP rank and verify equivalence
    for cp_rank in range(cp_size):
        # Split BSHD batch
        bshd_input_ids, bshd_labels = _split_batch_by_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=batch_bshd["input_ids"],
            labels_padded=batch_bshd["labels"],
            qvk_format="bshd",
            cp_rank=cp_rank,
            cp_world_size=cp_size,
        )

        # Split THD batch
        thd_input_ids, thd_labels = _split_batch_by_cp_rank(
            cu_seqlens_padded=batch_thd["cu_seq_lens_q_padded"],
            input_ids_padded=batch_thd["input_ids"],
            labels_padded=batch_thd["labels"],
            qvk_format="thd",
            cp_rank=cp_rank,
            cp_world_size=cp_size,
        )

        # Extract per-sequence shards from THD format
        # THD shards should match BSHD shards when extracted per-sequence
        thd_seq1_shard_len = seq_lengths_thd[0].item() // (2 * cp_size) * 2
        thd_seq2_shard_len = seq_lengths_thd[1].item() // (2 * cp_size) * 2

        thd_seq1_shard = thd_input_ids[0, :thd_seq1_shard_len]
        thd_seq2_shard = thd_input_ids[0, thd_seq1_shard_len : thd_seq1_shard_len + thd_seq2_shard_len]

        # Compare BSHD sequence shards with THD sequence shards
        bshd_seq1_shard = bshd_input_ids[0]
        bshd_seq2_shard = bshd_input_ids[1]

        # The tokens should match (accounting for any padding differences)
        torch.testing.assert_close(
            bshd_seq1_shard,
            thd_seq1_shard,
            msg=f"CP rank {cp_rank}: Sequence 1 shards don't match between BSHD and THD",
        )
        torch.testing.assert_close(
            bshd_seq2_shard,
            thd_seq2_shard,
            msg=f"CP rank {cp_rank}: Sequence 2 shards don't match between BSHD and THD",
        )

    # Verify that all CP ranks together reconstruct the original sequences
    all_bshd_shards_seq1 = []
    all_bshd_shards_seq2 = []
    for cp_rank in range(cp_size):
        bshd_input_ids, _ = _split_batch_by_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=batch_bshd["input_ids"],
            labels_padded=batch_bshd["labels"],
            qvk_format="bshd",
            cp_rank=cp_rank,
            cp_world_size=cp_size,
        )
        all_bshd_shards_seq1.append(bshd_input_ids[0])
        all_bshd_shards_seq2.append(bshd_input_ids[1])

    # Sort and verify all tokens are present
    reconstructed_seq1 = torch.cat(all_bshd_shards_seq1)
    reconstructed_seq2 = torch.cat(all_bshd_shards_seq2)

    torch.testing.assert_close(
        torch.sort(reconstructed_seq1)[0],
        torch.sort(batch_bshd["input_ids"][0])[0],
        msg="Reconstructed sequence 1 doesn't match original",
    )
    torch.testing.assert_close(
        torch.sort(reconstructed_seq2)[0],
        torch.sort(batch_bshd["input_ids"][1])[0],
        msg="Reconstructed sequence 2 doesn't match original",
    )


@pytest.mark.parametrize("cp_world_size", [2, 4])
def test_data_collator_for_context_parallel_returns_correct_list_size(tokenizer, cp_world_size):
    """Test that DataCollatorForContextParallel returns a list of the correct size."""
    divisibility_factor = 2 * cp_world_size

    # Create the wrapped collator that produces padded THD batches
    base_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
        pad_sequences_to_be_divisible_by=divisibility_factor,
    )

    # Create the context parallel collator
    cp_collator = DataCollatorForContextParallel(
        collator=base_collator, device_mesh=_DummyCollatorMesh(cp_size=cp_world_size)
    )

    # Create test sequences
    features = [
        {"input_ids": [0, 5, 6, 7, 8, 9, 10, 2], "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1]},  # 8 tokens
        {"input_ids": [0, 11, 12, 13, 14, 15, 16, 17, 2], "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1]},  # 9 tokens
    ]

    # Call the collator
    result = cp_collator(features)

    # Assert that the result is a list of the correct size
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == cp_world_size, f"Expected list of size {cp_world_size}, got {len(result)}"


def test_data_collator_for_context_parallel_thd(tokenizer):
    """Test that each shard from DataCollatorForContextParallel has all required keys from BatchType."""

    cp_world_size = 2
    divisibility_factor = 2 * cp_world_size

    # Create the wrapped collator that produces padded THD batches
    base_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
        pad_sequences_to_be_divisible_by=divisibility_factor,
    )

    # Create the context parallel collator
    cp_collator = DataCollatorForContextParallel(
        collator=base_collator, device_mesh=_DummyCollatorMesh(cp_size=cp_world_size)
    )

    # Create test sequences
    features = [
        {"input_ids": [0, 5, 6, 7, 8, 9, 10, 2]},  # 8 tokens
        {"input_ids": [0, 11, 12, 13, 14, 15, 16, 17, 2]},  # 9 tokens
    ]

    # Call the collator
    result = cp_collator(features)

    assert len(result) == cp_world_size, f"Expected list of size {cp_world_size}, got {len(result)}"

    # Define the required keys from BatchType. Shift labels wont appear when we are not using a causal language model.
    required_keys = set(BatchType.__annotations__.keys()) - {"shift_labels"}

    # Assert each shard has all required keys
    for cp_rank, shard in enumerate(result):
        assert set(shard.keys()) == required_keys, (
            f"CP rank {cp_rank}: difference: {set(shard.keys()) - required_keys}"
        )


def test_data_collator_for_context_parallel_thd_causal_lm(tokenizer):
    """Test that each shard from DataCollatorForContextParallel has all required keys from BatchType."""

    cp_world_size = 2
    divisibility_factor = 2 * cp_world_size

    # Create the wrapped collator that produces padded THD batches
    base_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        pad_sequences_to_be_divisible_by=divisibility_factor,
        separator_id=-100,
    )

    # Create the context parallel collator
    cp_collator = DataCollatorForContextParallel(
        collator=base_collator, device_mesh=_DummyCollatorMesh(cp_size=cp_world_size), is_causal_lm=True
    )

    # Create test sequences
    features = [
        {"input_ids": [0, 5, 6, 7, 8, 9, 10, 2]},  # 8 tokens
        {"input_ids": [0, 11, 12, 13, 14, 15, 16, 17, 2]},  # 9 tokens
    ]

    # Call the collator
    result = cp_collator(features)

    assert len(result) == cp_world_size, f"Expected list of size {cp_world_size}, got {len(result)}"

    # Define the required keys from BatchType
    required_keys = set(BatchType.__annotations__.keys())

    # Assert each shard has all required keys
    for cp_rank, shard in enumerate(result):
        assert set(shard.keys()) == required_keys, (
            f"CP rank {cp_rank}: difference: {set(shard.keys()) - required_keys}"
        )

    assert result[0]["labels"] is None
    assert result[1]["labels"] is None

    expected_rank0_input_ids = torch.tensor([[0, 5, 10, 2, 0, 11, 12, 1, 1, 1]], dtype=torch.int64)
    torch.testing.assert_close(result[0]["input_ids"], expected_rank0_input_ids)

    expected_rank0_shift_labels = torch.tensor([[5, 6, 2, -100, 11, 12, 13, -100, -100, -100]], dtype=torch.int64)
    torch.testing.assert_close(result[0]["shift_labels"], expected_rank0_shift_labels)

    expected_rank1_input_ids = torch.tensor([[6, 7, 8, 9, 13, 14, 15, 16, 17, 2]], dtype=torch.int64)
    torch.testing.assert_close(result[1]["input_ids"], expected_rank1_input_ids)

    expected_rank1_shift_labels = torch.tensor([[7, 8, 9, 10, 14, 15, 16, 17, 2, -100]], dtype=torch.int64)
    torch.testing.assert_close(result[1]["shift_labels"], expected_rank1_shift_labels)


def test_data_collator_for_context_parallel_thd_correctness(tokenizer):
    """Test that DataCollatorForContextParallel returns correct values for THD format.

    This test verifies:
    1. max_length_q and max_length_k are correctly rounded up to a multiple of 64
    2. input_ids and labels have the correct shape (sharded by cp_world_size)
    3. cu_seq_lens_* tensors are preserved correctly
    4. All shards together reconstruct the original data
    """
    cp_world_size = 2
    divisibility_factor = 2 * cp_world_size

    # Create the wrapped collator that produces padded THD batches - disable MLM for deterministic testing
    base_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        pad_sequences_to_be_divisible_by=divisibility_factor,
    )

    # Create the context parallel collator
    cp_collator = DataCollatorForContextParallel(
        collator=base_collator, device_mesh=_DummyCollatorMesh(cp_size=cp_world_size), qkv_format="thd"
    )

    # Create test sequences - 8 tokens each for easy division
    features = [
        {"input_ids": [0, 5, 6, 7, 8, 9, 10, 2]},  # 8 tokens
        {"input_ids": [0, 11, 12, 13, 14, 15, 16, 2]},  # 8 tokens
    ]

    # Call the collator
    result = cp_collator(features)

    assert len(result) == cp_world_size

    # Verify max_length_q and max_length_k are rounded up to a multiple of 64
    for cp_rank, shard in enumerate(result):
        max_length = shard["max_length_q"]
        assert max_length == shard["max_length_k"], "max_length_q and max_length_k should be equal"
        assert max_length % 64 == 0, f"CP rank {cp_rank}: max_length {max_length} should be a multiple of 64"
        # Since our sequences are 8 tokens, padded to divisibility_factor=4, max_seqlen should be 8,
        # and rounded up to 64
        assert max_length == 64, f"CP rank {cp_rank}: expected max_length=64, got {max_length}"

    # Verify input_ids shape - should be sharded along sequence dimension
    # Original total tokens: 16 (8+8), each shard should have 16 / cp_world_size = 8 tokens
    for cp_rank, shard in enumerate(result):
        assert shard["input_ids"].shape[1] == 8, (
            f"CP rank {cp_rank}: expected input_ids shape [1, 8], got {shard['input_ids'].shape}"
        )
        assert shard["labels"].shape[1] == 8, (
            f"CP rank {cp_rank}: expected labels shape [1, 8], got {shard['labels'].shape}"
        )

    # Verify that all shards together contain all the original tokens
    all_input_ids = torch.cat([shard["input_ids"] for shard in result], dim=1)
    # Check that all original tokens are present (sorted comparison since order may differ due to sharding)
    expected_tokens = torch.tensor([[0, 5, 6, 7, 8, 9, 10, 2, 0, 11, 12, 13, 14, 15, 16, 2]], dtype=torch.int64)
    torch.testing.assert_close(
        torch.sort(all_input_ids.flatten())[0],
        torch.sort(expected_tokens.flatten())[0],
        msg="Sharded tokens don't match original tokens",
    )

    # Verify cu_seq_lens_q_padded and cu_seq_lens_k_padded are preserved in each shard
    for cp_rank, shard in enumerate(result):
        assert "cu_seq_lens_q_padded" in shard
        assert "cu_seq_lens_k_padded" in shard
        torch.testing.assert_close(shard["cu_seq_lens_q_padded"], shard["cu_seq_lens_k_padded"])

    # Verify pad_between_seqs is True for THD format
    for cp_rank, shard in enumerate(result):
        assert shard["pad_between_seqs"] is True, f"CP rank {cp_rank}: pad_between_seqs should be True for THD format"


@pytest.mark.parametrize(
    "max_seqlen,expected_rounded",
    [
        (8, 64),  # Small value rounds up to 64
        (64, 64),  # Exactly 64 stays 64
        (65, 128),  # Just over 64 rounds up to 128
        (100, 128),  # 100 rounds up to 128
        (128, 128),  # Exactly 128 stays 128
        (129, 192),  # Just over 128 rounds up to 192
    ],
)
def test_data_collator_for_context_parallel_thd_max_length_rounding(tokenizer, max_seqlen, expected_rounded):
    """Test that max_length_q/k is correctly rounded up to a multiple of 64 for various sequence lengths."""
    cp_world_size = 2
    divisibility_factor = 2 * cp_world_size

    # Create input_ids of the specified length (must be divisible by divisibility_factor)
    # We pad to the next multiple of divisibility_factor if needed
    padded_len = ((max_seqlen + divisibility_factor - 1) // divisibility_factor) * divisibility_factor
    input_ids = [0, *list(range(5, 5 + padded_len - 2)), 2]  # [CLS] + tokens + [SEP]
    input_ids = input_ids[:padded_len]  # Truncate to exact length

    # Create the collators
    base_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        pad_sequences_to_be_divisible_by=divisibility_factor,
    )
    cp_collator = DataCollatorForContextParallel(
        collator=base_collator, device_mesh=_DummyCollatorMesh(cp_size=cp_world_size), qkv_format="thd"
    )

    # Use a single sequence to ensure max_seqlen is exactly what we expect after padding
    features = [{"input_ids": input_ids}]
    result = cp_collator(features)

    # The actual max_seqlen after padding may differ, but it should still round correctly
    for shard in result:
        max_length = shard["max_length_q"]
        assert max_length % 64 == 0, f"max_length {max_length} should be a multiple of 64"
        # Verify the rounding formula: ((x + 63) // 64) * 64
        actual_seqlen = (shard["cu_seq_lens_q_padded"][1:] - shard["cu_seq_lens_q_padded"][:-1]).max().item()
        expected = ((actual_seqlen + 63) // 64) * 64
        assert max_length == expected, f"Expected max_length={expected} for seqlen={actual_seqlen}, got {max_length}"


def test_data_collator_for_context_parallel_bshd(tokenizer):
    """Test that each shard from DataCollatorForContextParallel has all required keys from BatchType."""

    cp_world_size = 2
    divisibility_factor = 2 * cp_world_size

    # Create the wrapped collator that produces padded THD batches
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=divisibility_factor,
    )

    # Create the context parallel collator
    cp_collator = DataCollatorForContextParallel(
        collator=base_collator, device_mesh=_DummyCollatorMesh(cp_size=cp_world_size), qkv_format="bshd"
    )

    # Create test sequences
    features = [
        {"input_ids": [0, 5, 6, 7, 8, 9, 10, 2]},  # 8 tokens
        {"input_ids": [0, 11, 12, 13, 14, 15, 16, 17, 2]},  # 9 tokens
    ]

    # Call the collator
    result = cp_collator(features)

    assert len(result) == cp_world_size, f"Expected list of size {cp_world_size}, got {len(result)}"

    # Define the required keys from BatchType
    required_keys = {"input_ids", "labels", "max_length_q", "max_length_k"}

    # Assert each shard has all required keys
    for cp_rank, shard in enumerate(result):
        assert set(shard.keys()) == required_keys, (
            f"CP rank {cp_rank}: expected keys {required_keys}, got {set(shard.keys())}"
        )


def test_data_collator_for_context_parallel_with_tp(tokenizer):
    """Test that DataCollatorForContextParallel duplicates batches for TP ranks when tp_world_size is set."""
    cp_world_size = 2
    tp_world_size = 2
    divisibility_factor = 2 * cp_world_size

    # Create the wrapped collator that produces padded THD batches
    base_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
        pad_sequences_to_be_divisible_by=divisibility_factor,
    )

    # Create the context parallel collator with TP
    cp_collator = DataCollatorForContextParallel(
        collator=base_collator, device_mesh=_DummyCollatorMesh(cp_size=cp_world_size, tp_size=tp_world_size)
    )

    # Create test sequences
    features = [
        {"input_ids": [0, 5, 6, 7, 8, 9, 10, 2]},  # 8 tokens
        {"input_ids": [0, 11, 12, 13, 14, 15, 16, 17, 2]},  # 9 tokens
    ]

    # Call the collator
    result = cp_collator(features)

    # Assert that the result list has length cp_world_size * tp_world_size
    expected_length = cp_world_size * tp_world_size
    assert len(result) == expected_length, f"Expected list of size {expected_length}, got {len(result)}"

    # Assert that batches are duplicated for TP ranks
    # The structure should be: [cp0_tp0, cp0_tp1, cp1_tp0, cp1_tp1]
    # So consecutive pairs should be identical for the same CP rank
    for cp_rank in range(cp_world_size):
        base_idx = cp_rank * tp_world_size
        for tp_rank in range(1, tp_world_size):
            # Compare each TP rank's batch with the first TP rank's batch for this CP rank
            first_batch = result[base_idx]
            current_batch = result[base_idx + tp_rank]

            # Check that all keys are the same
            assert set(first_batch.keys()) == set(current_batch.keys()), (
                f"CP rank {cp_rank}, TP rank {tp_rank}: keys don't match"
            )

            # Check that tensor values are identical
            for key in first_batch.keys():
                if isinstance(first_batch[key], torch.Tensor):
                    torch.testing.assert_close(
                        first_batch[key],
                        current_batch[key],
                        msg=f"CP rank {cp_rank}, TP rank {tp_rank}: '{key}' tensors don't match",
                    )
                else:
                    assert first_batch[key] == current_batch[key], (
                        f"CP rank {cp_rank}, TP rank {tp_rank}: '{key}' values don't match"
                    )


def test_data_collator_for_context_parallel_with_tp_first(tokenizer):
    """Test that DataCollatorForContextParallel produces the correct interleaving when TP is the row dimension.

    With tp_first=True (mesh_dim_names=("tp", "cp")), the flattened mesh is:
    [(tp0,cp0), (tp0,cp1), (tp1,cp0), (tp1,cp1)]

    So the collator output should be: [cp0, cp1, cp0, cp1] (CP shards interleaved).
    Compare with the default (CP row-major): [cp0, cp0, cp1, cp1] (CP shards grouped).
    """
    cp_world_size = 2
    tp_world_size = 2
    divisibility_factor = 2 * cp_world_size

    base_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        pad_sequences_to_be_divisible_by=divisibility_factor,
    )

    features = [
        {"input_ids": [0, 5, 6, 7, 8, 9, 10, 2]},
        {"input_ids": [0, 11, 12, 13, 14, 15, 16, 17, 2]},
    ]

    # CP row-major (default): output is [cp0, cp0, cp1, cp1]
    cp_collator_cp_first = DataCollatorForContextParallel(
        collator=base_collator,
        device_mesh=_DummyCollatorMesh(cp_size=cp_world_size, tp_size=tp_world_size, tp_first=False),
    )
    result_cp_first = cp_collator_cp_first(features)

    # TP row-major: output is [cp0, cp1, cp0, cp1]
    cp_collator_tp_first = DataCollatorForContextParallel(
        collator=base_collator,
        device_mesh=_DummyCollatorMesh(cp_size=cp_world_size, tp_size=tp_world_size, tp_first=True),
    )
    result_tp_first = cp_collator_tp_first(features)

    # Both should have the same total length
    assert len(result_cp_first) == cp_world_size * tp_world_size
    assert len(result_tp_first) == cp_world_size * tp_world_size

    # CP row-major: [cp0, cp0, cp1, cp1] — consecutive pairs are identical
    for key in result_cp_first[0]:
        if isinstance(result_cp_first[0][key], torch.Tensor):
            torch.testing.assert_close(result_cp_first[0][key], result_cp_first[1][key])
            torch.testing.assert_close(result_cp_first[2][key], result_cp_first[3][key])

    # TP row-major: [cp0, cp1, cp0, cp1] — alternating pairs are identical
    for key in result_tp_first[0]:
        if isinstance(result_tp_first[0][key], torch.Tensor):
            torch.testing.assert_close(result_tp_first[0][key], result_tp_first[2][key])
            torch.testing.assert_close(result_tp_first[1][key], result_tp_first[3][key])

    # Verify the same CP shards exist in both orderings, just in different positions
    # CP row-major shard at index 0 (cp0) should match TP row-major shard at index 0 (also cp0)
    for key in result_cp_first[0]:
        if isinstance(result_cp_first[0][key], torch.Tensor):
            torch.testing.assert_close(result_cp_first[0][key], result_tp_first[0][key])
            torch.testing.assert_close(result_cp_first[2][key], result_tp_first[1][key])
