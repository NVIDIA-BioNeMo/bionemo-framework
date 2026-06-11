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

"""Tests for ContextParallelDataLoaderWrapper using torchrun with CPU-only distributed backend.

These tests verify the data distribution functionality of ContextParallelDataLoaderWrapper
without requiring GPUs, allowing multi-process testing on CPU-only machines.

The tests use DataCollatorForContextParallel as the collate_fn in a real torch DataLoader,
testing both THD and BSHD formats, as well as combinations of context parallelism (CP)
and tensor parallelism (TP).

Test configurations:
- CP only: Data is sharded across CP ranks
- TP only: Data is replicated across TP ranks
- CP + TP: Data is sharded across CP ranks and replicated across TP ranks within each CP group
"""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# When launched via torchrun, conftest.py sys.path setup doesn't run.
# Ensure the model directory (parent of tests/) is on sys.path for bare module imports.
sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from collator import (
    ContextParallelDataLoaderWrapper,
    DataCollatorForContextParallel,
    DataCollatorWithFlattening,
)


@dataclass(frozen=True)
class DistributedConfig:
    """Class to track distributed ranks and handle basic distributed training setup.

    If torch distributed environment variables are not set, we set them to default values for single-process training.

    Attributes:
        rank: The rank of the process.
        local_rank: The local rank of the process.
        world_size: The total number of processes.
    """

    rank: int = field(default_factory=lambda: int(os.environ.setdefault("RANK", "0")))
    local_rank: int = field(default_factory=lambda: int(os.environ.setdefault("LOCAL_RANK", "0")))
    world_size: int = field(default_factory=lambda: int(os.environ.setdefault("WORLD_SIZE", "1")))
    _master_addr: str = field(default_factory=lambda: os.environ.setdefault("MASTER_ADDR", "localhost"))
    _master_port: str = field(default_factory=lambda: os.environ.setdefault("MASTER_PORT", "12356"))

    def is_main_process(self) -> bool:
        """This is the global rank 0 process, to be used for wandb logging, etc."""
        return self.rank == 0


# Test protein sequences of varying lengths
TEST_PROTEINS = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLLEVL",  # 33 AA
    "MSHHWGYGKHNGPEHWHKDFPIAKGERFL",  # 30 AA
    "MLSATEEKLSDYISSLFASVSIINSI",  # 27 AA
    "MFVFFAGTLVNQDTLNFRDQLNINVVGTVRGIAQ",  # 34 AA
]


class TokenizedProteinDataset(Dataset):
    """A simple dataset of pre-tokenized protein sequences."""

    def __init__(self, tokenized_sequences: list[dict]):
        """Initialize the dataset with tokenized sequences.

        Args:
            tokenized_sequences: List of tokenized sequences (dicts with 'input_ids', etc.)
        """
        self.sequences = tokenized_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def get_tokenizer():
    """Get the ESM fast tokenizer."""
    return AutoTokenizer.from_pretrained("esm_fast_tokenizer")


def create_cp_collator_thd(tokenizer, device_mesh):
    """Create a DataCollatorForContextParallel for THD format.

    Args:
        tokenizer: The tokenizer to use.
        device_mesh: The device mesh with named dimensions (must contain "cp").

    Returns:
        A DataCollatorForContextParallel configured for THD format.
    """
    dim_names = device_mesh.mesh_dim_names
    cp_world_size = device_mesh.size(dim_names.index("cp")) if "cp" in dim_names else 1
    divisibility_factor = 2 * cp_world_size

    # Create the base THD collator with per-sequence padding for CP
    base_collator = DataCollatorWithFlattening(
        collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Disable MLM for deterministic testing
        ),
        pad_sequences_to_be_divisible_by=divisibility_factor,
    )

    return DataCollatorForContextParallel(
        collator=base_collator,
        device_mesh=device_mesh,
        qkv_format="thd",
    )


def create_cp_collator_bshd(tokenizer, device_mesh):
    """Create a DataCollatorForContextParallel for BSHD format.

    Args:
        tokenizer: The tokenizer to use.
        device_mesh: The device mesh with named dimensions (must contain "cp").

    Returns:
        A DataCollatorForContextParallel configured for BSHD format.
    """
    dim_names = device_mesh.mesh_dim_names
    cp_world_size = device_mesh.size(dim_names.index("cp")) if "cp" in dim_names else 1
    divisibility_factor = 2 * cp_world_size

    # Create the base BSHD collator with padding
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Disable MLM for deterministic testing
        pad_to_multiple_of=divisibility_factor,
    )

    return DataCollatorForContextParallel(
        collator=base_collator,
        device_mesh=device_mesh,
        qkv_format="bshd",
    )


def create_dataloader(
    tokenized_sequences: list[dict],
    collate_fn,
    batch_size: int = 2,
) -> DataLoader:
    """Create a DataLoader with the given collate function.

    Args:
        tokenized_sequences: List of tokenized sequences.
        collate_fn: The collate function (DataCollatorForContextParallel).
        batch_size: Number of sequences per batch.

    Returns:
        A PyTorch DataLoader.
    """
    dataset = TokenizedProteinDataset(tokenized_sequences)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


# =============================================================================
# CP-only tests (2 processes)
# =============================================================================


def test_cp_dataloader_wrapper_thd_2_processes():
    """Test ContextParallelDataLoaderWrapper with THD format and 2 processes (CP only)."""
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
        "test_thd_scatter",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


def test_cp_dataloader_wrapper_bshd_2_processes():
    """Test ContextParallelDataLoaderWrapper with BSHD format and 2 processes (CP only)."""
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
        "test_bshd_scatter",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


def test_cp_dataloader_wrapper_stop_iteration_thd():
    """Test that StopIteration is properly propagated to all CP ranks with THD format."""
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
        "test_stop_iteration_thd",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


def test_cp_dataloader_wrapper_stop_iteration_bshd():
    """Test that StopIteration is properly propagated to all CP ranks with BSHD format."""
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
        "test_stop_iteration_bshd",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


def test_cp_dataloader_wrapper_multiple_batches_thd():
    """Test ContextParallelDataLoaderWrapper with multiple batches using THD format."""
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
        "test_multiple_batches_thd",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


def test_cp_dataloader_wrapper_multiple_batches_bshd():
    """Test ContextParallelDataLoaderWrapper with multiple batches using BSHD format."""
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
        "test_multiple_batches_bshd",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


# =============================================================================
# TP-only tests (2 processes, CP=1, TP=2)
# Data should be replicated across TP ranks
# =============================================================================


def test_cp_dataloader_wrapper_tp_only_thd():
    """Test ContextParallelDataLoaderWrapper with TP only (CP=1, TP=2) using THD format.

    Data should be replicated across TP ranks.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
        "test_tp_only_thd",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


def test_cp_dataloader_wrapper_tp_only_bshd():
    """Test ContextParallelDataLoaderWrapper with TP only (CP=1, TP=2) using BSHD format.

    Data should be replicated across TP ranks.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        os.path.relpath(__file__),
        "test_tp_only_bshd",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


# =============================================================================
# CP + TP tests (4 processes, CP=2, TP=2)
# Data should be sharded across CP ranks and replicated within TP groups
# =============================================================================


def test_cp_dataloader_wrapper_cp_tp_thd():
    """Test ContextParallelDataLoaderWrapper with CP=2, TP=2 using THD format.

    Data should be sharded across CP ranks and replicated within TP groups.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        os.path.relpath(__file__),
        "test_cp_tp_thd",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


def test_cp_dataloader_wrapper_cp_tp_bshd():
    """Test ContextParallelDataLoaderWrapper with CP=2, TP=2 using BSHD format.

    Data should be sharded across CP ranks and replicated within TP groups.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        os.path.relpath(__file__),
        "test_cp_tp_bshd",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


# =============================================================================
# TP + CP tests (4 processes, TP=2, CP=2, TP is the row dimension)
# Data should be sharded across CP ranks and replicated within TP groups
# =============================================================================


def test_cp_dataloader_wrapper_tp_cp_thd():
    """Test ContextParallelDataLoaderWrapper with TP=2, CP=2 using THD format.

    TP is the row dimension (mesh_dim_names=("tp", "cp")).
    Data should be sharded across CP ranks and replicated within TP groups.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        os.path.relpath(__file__),
        "test_tp_cp_thd",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


def test_cp_dataloader_wrapper_tp_cp_bshd():
    """Test ContextParallelDataLoaderWrapper with TP=2, CP=2 using BSHD format.

    TP is the row dimension (mesh_dim_names=("tp", "cp")).
    Data should be sharded across CP ranks and replicated within TP groups.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        os.path.relpath(__file__),
        "test_tp_cp_bshd",
    ]
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


# =============================================================================
# Distributed test implementations (CP only)
# =============================================================================


def _run_test_thd_scatter():
    """Test that THD data is correctly scattered from rank 0 to all CP ranks."""
    dist_config = DistributedConfig()

    # Initialize distributed with gloo backend (CPU-only)
    torch.distributed.init_process_group(backend="gloo")

    cp_size = dist_config.world_size
    device_mesh = init_device_mesh(
        "cpu",
        mesh_shape=(1, cp_size),
        mesh_dim_names=("ddp", "cp"),
    )

    cp_mesh = device_mesh["cp"]
    cp_rank = cp_mesh.get_local_rank()

    # Create tokenizer and collator
    tokenizer = get_tokenizer()
    cp_collator = create_cp_collator_thd(tokenizer, device_mesh=cp_mesh)

    # Tokenize the test proteins
    tokenized_proteins = [tokenizer(p, add_special_tokens=True) for p in TEST_PROTEINS[:2]]

    # Create a real DataLoader with DataCollatorForContextParallel as collate_fn (only on rank 0)
    if cp_rank == 0:
        dataloader = create_dataloader(tokenized_proteins, collate_fn=cp_collator, batch_size=2)
        # Get expected batch for verification
        expected_sharded_batches = cp_collator(tokenized_proteins)
    else:
        dataloader = None
        expected_sharded_batches = None

    # Broadcast expected batch to all ranks for verification
    expected_list = [expected_sharded_batches]
    torch.distributed.broadcast_object_list(expected_list, src=0)
    expected_sharded_batches = expected_list[0]

    # Create the wrapper
    wrapper = ContextParallelDataLoaderWrapper(dataloader=dataloader, cp_tp_mesh=cp_mesh)

    # Iterate and verify
    iter(wrapper)
    batch_on_rank = next(wrapper)

    # Verify that this rank received the correct shard
    expected_batch = expected_sharded_batches[cp_rank]

    torch.testing.assert_close(
        batch_on_rank["input_ids"],
        expected_batch["input_ids"],
        msg=f"Rank {cp_rank}: input_ids mismatch",
    )
    torch.testing.assert_close(
        batch_on_rank["labels"],
        expected_batch["labels"],
        msg=f"Rank {cp_rank}: labels mismatch",
    )

    # Verify THD-specific fields
    assert "cu_seq_lens_q_padded" in batch_on_rank, f"Rank {cp_rank}: missing cu_seq_lens_q_padded"
    assert "cu_seq_lens_k_padded" in batch_on_rank, f"Rank {cp_rank}: missing cu_seq_lens_k_padded"
    assert "pad_between_seqs" in batch_on_rank, f"Rank {cp_rank}: missing pad_between_seqs"

    torch.distributed.destroy_process_group()


def _run_test_bshd_scatter():
    """Test that BSHD data is correctly scattered from rank 0 to all CP ranks."""
    dist_config = DistributedConfig()

    # Initialize distributed with gloo backend (CPU-only)
    torch.distributed.init_process_group(backend="gloo")

    cp_size = dist_config.world_size
    device_mesh = init_device_mesh(
        "cpu",
        mesh_shape=(1, cp_size),
        mesh_dim_names=("ddp", "cp"),
    )

    cp_mesh = device_mesh["cp"]
    cp_rank = cp_mesh.get_local_rank()

    # Create tokenizer and collator
    tokenizer = get_tokenizer()
    cp_collator = create_cp_collator_bshd(tokenizer, device_mesh=cp_mesh)

    # Tokenize the test proteins
    tokenized_proteins = [tokenizer(p, add_special_tokens=True) for p in TEST_PROTEINS[:2]]

    # Create a real DataLoader with DataCollatorForContextParallel as collate_fn (only on rank 0)
    if cp_rank == 0:
        dataloader = create_dataloader(tokenized_proteins, collate_fn=cp_collator, batch_size=2)
        # Get expected batch for verification
        expected_sharded_batches = cp_collator(tokenized_proteins)
    else:
        dataloader = None
        expected_sharded_batches = None

    # Broadcast expected batch to all ranks for verification
    expected_list = [expected_sharded_batches]
    torch.distributed.broadcast_object_list(expected_list, src=0)
    expected_sharded_batches = expected_list[0]

    # Create the wrapper
    wrapper = ContextParallelDataLoaderWrapper(dataloader=dataloader, cp_tp_mesh=cp_mesh)

    # Iterate and verify
    iter(wrapper)
    batch_on_rank = next(wrapper)

    # Verify that this rank received the correct shard
    expected_batch = expected_sharded_batches[cp_rank]

    torch.testing.assert_close(
        batch_on_rank["input_ids"],
        expected_batch["input_ids"],
        msg=f"Rank {cp_rank}: input_ids mismatch",
    )
    torch.testing.assert_close(
        batch_on_rank["labels"],
        expected_batch["labels"],
        msg=f"Rank {cp_rank}: labels mismatch",
    )

    # Verify BSHD format: should NOT have THD-specific fields like cu_seq_lens_q_padded
    assert "cu_seq_lens_q_padded" not in batch_on_rank, f"Rank {cp_rank}: BSHD should not have cu_seq_lens_q_padded"
    # BSHD format removes attention_mask
    assert "attention_mask" not in batch_on_rank, f"Rank {cp_rank}: BSHD should not have attention_mask"

    torch.distributed.destroy_process_group()


def _run_test_stop_iteration(qkv_format: str):
    """Test that StopIteration is properly propagated to all CP ranks.

    Args:
        qkv_format: Either "thd" or "bshd".
    """
    dist_config = DistributedConfig()

    # Initialize distributed with gloo backend (CPU-only)
    torch.distributed.init_process_group(backend="gloo")

    cp_size = dist_config.world_size
    device_mesh = init_device_mesh(
        "cpu",
        mesh_shape=(1, cp_size),
        mesh_dim_names=("ddp", "cp"),
    )

    cp_mesh = device_mesh["cp"]
    cp_rank = cp_mesh.get_local_rank()

    # Create tokenizer and collator
    tokenizer = get_tokenizer()
    if qkv_format == "thd":
        cp_collator = create_cp_collator_thd(tokenizer, device_mesh=cp_mesh)
    else:
        cp_collator = create_cp_collator_bshd(tokenizer, device_mesh=cp_mesh)

    # Tokenize the test proteins - use only 2 for a single batch
    tokenized_proteins = [tokenizer(p, add_special_tokens=True) for p in TEST_PROTEINS[:2]]

    # Create a real DataLoader (only on rank 0)
    if cp_rank == 0:
        dataloader = create_dataloader(tokenized_proteins, collate_fn=cp_collator, batch_size=2)
    else:
        dataloader = None

    wrapper = ContextParallelDataLoaderWrapper(dataloader=dataloader, cp_tp_mesh=cp_mesh)

    # Get the first (and only) batch
    iter(wrapper)
    _ = next(wrapper)

    # The next call should raise StopIteration on all ranks
    stop_iteration_raised = False
    try:
        _ = next(wrapper)
    except StopIteration:
        stop_iteration_raised = True

    assert stop_iteration_raised, f"Rank {cp_rank}: StopIteration was not raised"

    # Verify all ranks got StopIteration (use all_reduce to check)
    stop_count = torch.tensor([1 if stop_iteration_raised else 0], dtype=torch.int32)
    torch.distributed.all_reduce(stop_count, op=torch.distributed.ReduceOp.SUM)

    assert stop_count.item() == cp_size, f"Not all ranks received StopIteration: {stop_count.item()} vs {cp_size}"

    torch.distributed.destroy_process_group()


def _run_test_multiple_batches(qkv_format: str):
    """Test ContextParallelDataLoaderWrapper with multiple batches.

    Args:
        qkv_format: Either "thd" or "bshd".
    """
    dist_config = DistributedConfig()

    # Initialize distributed with gloo backend (CPU-only)
    torch.distributed.init_process_group(backend="gloo")

    cp_size = dist_config.world_size
    device_mesh = init_device_mesh(
        "cpu",
        mesh_shape=(1, cp_size),
        mesh_dim_names=("ddp", "cp"),
    )

    cp_mesh = device_mesh["cp"]
    cp_rank = cp_mesh.get_local_rank()

    # Create tokenizer and collator
    tokenizer = get_tokenizer()
    if qkv_format == "thd":
        cp_collator = create_cp_collator_thd(tokenizer, device_mesh=cp_mesh)
    else:
        cp_collator = create_cp_collator_bshd(tokenizer, device_mesh=cp_mesh)

    # Tokenize all proteins - with batch_size=2, we'll get 2 batches
    tokenized_proteins = [tokenizer(p, add_special_tokens=True) for p in TEST_PROTEINS]

    # Create a real DataLoader (only on rank 0)
    if cp_rank == 0:
        dataloader = create_dataloader(tokenized_proteins, collate_fn=cp_collator, batch_size=2)
        # Pre-compute expected batches for verification
        expected_batches = list(dataloader)
        # Re-create dataloader since we consumed it
        dataloader = create_dataloader(tokenized_proteins, collate_fn=cp_collator, batch_size=2)
    else:
        dataloader = None
        expected_batches = None

    # Broadcast expected batches to all ranks for verification
    expected_list = [expected_batches]
    torch.distributed.broadcast_object_list(expected_list, src=0)
    expected_batches = expected_list[0]

    wrapper = ContextParallelDataLoaderWrapper(dataloader=dataloader, cp_tp_mesh=cp_mesh)

    # Iterate through all batches
    received_batches = list(wrapper)

    # Verify we got the correct number of batches
    num_batches = len(expected_batches)
    assert len(received_batches) == num_batches, (
        f"Rank {cp_rank}: Expected {num_batches} batches, got {len(received_batches)}"
    )

    # Verify each batch has the correct content for this rank
    for i, batch in enumerate(received_batches):
        expected_batch = expected_batches[i][cp_rank]
        torch.testing.assert_close(
            batch["input_ids"],
            expected_batch["input_ids"],
            msg=f"Rank {cp_rank}, Batch {i}: input_ids mismatch",
        )
        torch.testing.assert_close(
            batch["labels"],
            expected_batch["labels"],
            msg=f"Rank {cp_rank}, Batch {i}: labels mismatch",
        )

    torch.distributed.destroy_process_group()


# =============================================================================
# Distributed test implementations (TP only - data replication)
# =============================================================================


def _run_test_tp_only(qkv_format: str):
    """Test that data is replicated across TP ranks when CP=1.

    Args:
        qkv_format: Either "thd" or "bshd".
    """
    dist_config = DistributedConfig()

    # Initialize distributed with gloo backend (CPU-only)
    torch.distributed.init_process_group(backend="gloo")

    # TP=2, CP=1 configuration
    tp_size = dist_config.world_size

    # Create a 1D mesh with TP only (no CP dimension)
    device_mesh = init_device_mesh(
        "cpu",
        mesh_shape=(tp_size,),
        mesh_dim_names=("tp",),
    )

    # Flatten the CP+TP mesh for the dataloader wrapper
    flat_rank = device_mesh.get_local_rank()

    # Get individual TP rank (no CP dimension in this mesh)
    tp_rank = device_mesh.get_local_rank("tp")

    # Create tokenizer and collator with TP replication
    tokenizer = get_tokenizer()
    if qkv_format == "thd":
        cp_collator = create_cp_collator_thd(tokenizer, device_mesh=device_mesh)
    else:
        cp_collator = create_cp_collator_bshd(tokenizer, device_mesh=device_mesh)

    # Tokenize the test proteins
    tokenized_proteins = [tokenizer(p, add_special_tokens=True) for p in TEST_PROTEINS[:2]]

    # Create a real DataLoader (only on flat rank 0)
    if flat_rank == 0:
        dataloader = create_dataloader(tokenized_proteins, collate_fn=cp_collator, batch_size=2)
        # Get expected batch for verification
        expected_sharded_batches = cp_collator(tokenized_proteins)
    else:
        dataloader = None
        expected_sharded_batches = None

    # Broadcast expected batch to all ranks for verification
    expected_list = [expected_sharded_batches]
    torch.distributed.broadcast_object_list(expected_list, src=0)
    expected_sharded_batches = expected_list[0]

    # Create the wrapper
    wrapper = ContextParallelDataLoaderWrapper(dataloader=dataloader, cp_tp_mesh=device_mesh)

    # Iterate and verify
    iter(wrapper)
    batch_on_rank = next(wrapper)

    # Verify that this rank received the correct shard
    # With CP=1 and TP=2, shards are: [cp0_tp0, cp0_tp1] = [shard0, shard0]
    # All TP ranks should get the same data (replicated)
    expected_batch = expected_sharded_batches[0]

    torch.testing.assert_close(
        batch_on_rank["input_ids"],
        expected_batch["input_ids"],
        msg=f"Flat rank {flat_rank} (tp={tp_rank}): input_ids mismatch",
    )
    torch.testing.assert_close(
        batch_on_rank["labels"],
        expected_batch["labels"],
        msg=f"Flat rank {flat_rank} (tp={tp_rank}): labels mismatch",
    )

    # Verify that all TP ranks within the same CP group have identical data
    # Gather all batches to rank 0 and verify
    all_input_ids = [None] * dist_config.world_size
    torch.distributed.all_gather_object(all_input_ids, batch_on_rank["input_ids"])

    if flat_rank == 0:
        # All ranks should have the same data since CP=1
        for i in range(1, len(all_input_ids)):
            torch.testing.assert_close(
                all_input_ids[0],
                all_input_ids[i],
                msg=f"TP replication failed: rank 0 and rank {i} have different data",
            )

    torch.distributed.destroy_process_group()


# =============================================================================
# Distributed test implementations (CP + TP combined)
# =============================================================================


def _run_test_cp_tp(qkv_format: str):
    """Test that data is sharded across CP ranks and replicated across TP ranks.

    With CP=2, TP=2:
    - 4 total ranks
    - Mesh shape: (cp=2, tp=2)
    - Flattened order: [cp0_tp0, cp0_tp1, cp1_tp0, cp1_tp1]
    - Expected batches: [cp0_shard, cp0_shard, cp1_shard, cp1_shard]

    Args:
        qkv_format: Either "thd" or "bshd".
    """
    dist_config = DistributedConfig()

    # Initialize distributed with gloo backend (CPU-only)
    torch.distributed.init_process_group(backend="gloo")

    # CP=2, TP=2 configuration
    cp_size = 2
    tp_size = 2

    # Create a 2D mesh with CP=2, TP=2
    device_mesh = init_device_mesh(
        "cpu",
        mesh_shape=(cp_size, tp_size),
        mesh_dim_names=("cp", "tp"),
    )

    # Flatten the CP+TP mesh for the dataloader wrapper
    cp_tp_mesh = device_mesh[("cp", "tp")]._flatten("cp_tp")
    flat_rank = cp_tp_mesh.get_local_rank()

    # Get individual CP and TP ranks
    cp_rank = device_mesh.get_local_rank("cp")
    tp_rank = device_mesh.get_local_rank("tp")

    # Create tokenizer and collator with TP replication
    tokenizer = get_tokenizer()
    if qkv_format == "thd":
        cp_collator = create_cp_collator_thd(tokenizer, device_mesh=device_mesh)
    else:
        cp_collator = create_cp_collator_bshd(tokenizer, device_mesh=device_mesh)

    # Tokenize the test proteins
    tokenized_proteins = [tokenizer(p, add_special_tokens=True) for p in TEST_PROTEINS[:2]]

    # Create a real DataLoader (only on flat rank 0)
    if flat_rank == 0:
        dataloader = create_dataloader(tokenized_proteins, collate_fn=cp_collator, batch_size=2)
        # Get expected batch for verification
        expected_sharded_batches = cp_collator(tokenized_proteins)
    else:
        dataloader = None
        expected_sharded_batches = None

    # Broadcast expected batch to all ranks for verification
    expected_list = [expected_sharded_batches]
    torch.distributed.broadcast_object_list(expected_list, src=0)
    expected_sharded_batches = expected_list[0]

    # Create the wrapper
    wrapper = ContextParallelDataLoaderWrapper(dataloader=dataloader, cp_tp_mesh=cp_tp_mesh)

    # Iterate and verify
    iter(wrapper)
    batch_on_rank = next(wrapper)

    # Verify that this rank received the correct shard
    # With CP=2 and TP=2, shards are: [cp0_tp0, cp0_tp1, cp1_tp0, cp1_tp1]
    # = [cp0_shard, cp0_shard, cp1_shard, cp1_shard]
    expected_batch = expected_sharded_batches[flat_rank]

    torch.testing.assert_close(
        batch_on_rank["input_ids"],
        expected_batch["input_ids"],
        msg=f"Flat rank {flat_rank} (cp={cp_rank}, tp={tp_rank}): input_ids mismatch",
    )
    torch.testing.assert_close(
        batch_on_rank["labels"],
        expected_batch["labels"],
        msg=f"Flat rank {flat_rank} (cp={cp_rank}, tp={tp_rank}): labels mismatch",
    )

    # Gather all batches to verify sharding and replication patterns
    all_input_ids = [None] * dist_config.world_size
    torch.distributed.all_gather_object(all_input_ids, batch_on_rank["input_ids"])

    if flat_rank == 0:
        # Verify TP replication: ranks with same CP rank should have identical data
        # Rank 0 (cp=0, tp=0) should match Rank 1 (cp=0, tp=1)
        torch.testing.assert_close(
            all_input_ids[0],
            all_input_ids[1],
            msg="TP replication failed: cp=0 ranks have different data",
        )
        # Rank 2 (cp=1, tp=0) should match Rank 3 (cp=1, tp=1)
        torch.testing.assert_close(
            all_input_ids[2],
            all_input_ids[3],
            msg="TP replication failed: cp=1 ranks have different data",
        )

        # Verify CP sharding: ranks with different CP ranks should have different data
        # Rank 0 (cp=0) should differ from Rank 2 (cp=1)
        assert not torch.equal(all_input_ids[0], all_input_ids[2]), (
            "CP sharding failed: different CP ranks have the same data"
        )

    torch.distributed.destroy_process_group()


# =============================================================================
# Distributed test implementations (TP + CP combined, TP row-major)
# =============================================================================


def _run_test_tp_cp(qkv_format: str):
    """Test that data is sharded across CP ranks and replicated across TP ranks with TP as the row dimension.

    With TP=2, CP=2 and mesh_dim_names=("tp", "cp"):
    - 4 total ranks
    - Mesh shape: (tp=2, cp=2) — TP is the row dimension
    - Flattened order: [tp0_cp0, tp0_cp1, tp1_cp0, tp1_cp1]
    - Expected batches from collator: [cp0_shard, cp1_shard, cp0_shard, cp1_shard]

    Args:
        qkv_format: Either "thd" or "bshd".
    """
    dist_config = DistributedConfig()

    # Initialize distributed with gloo backend (CPU-only)
    torch.distributed.init_process_group(backend="gloo")

    # TP=2, CP=2 configuration with TP as the row dimension
    tp_size = 2
    cp_size = 2

    # Create a 2D mesh with TP as the first (row) dimension
    device_mesh = init_device_mesh(
        "cpu",
        mesh_shape=(tp_size, cp_size),
        mesh_dim_names=("tp", "cp"),
    )

    # Flatten the TP+CP mesh for the dataloader wrapper (TP first)
    cp_tp_mesh = device_mesh[("tp", "cp")]._flatten("tp_cp")
    flat_rank = cp_tp_mesh.get_local_rank()

    # Get individual CP and TP ranks
    cp_rank = device_mesh.get_local_rank("cp")
    tp_rank = device_mesh.get_local_rank("tp")

    # Create tokenizer and collator — pass the 2D mesh so collator reads dimension order
    tokenizer = get_tokenizer()
    if qkv_format == "thd":
        cp_collator = create_cp_collator_thd(tokenizer, device_mesh=device_mesh)
    else:
        cp_collator = create_cp_collator_bshd(tokenizer, device_mesh=device_mesh)

    # Tokenize the test proteins
    tokenized_proteins = [tokenizer(p, add_special_tokens=True) for p in TEST_PROTEINS[:2]]

    # Create a real DataLoader (only on flat rank 0)
    if flat_rank == 0:
        dataloader = create_dataloader(tokenized_proteins, collate_fn=cp_collator, batch_size=2)
        # Get expected batch for verification
        expected_sharded_batches = cp_collator(tokenized_proteins)
    else:
        dataloader = None
        expected_sharded_batches = None

    # Broadcast expected batch to all ranks for verification
    expected_list = [expected_sharded_batches]
    torch.distributed.broadcast_object_list(expected_list, src=0)
    expected_sharded_batches = expected_list[0]

    # Create the wrapper
    wrapper = ContextParallelDataLoaderWrapper(dataloader=dataloader, cp_tp_mesh=cp_tp_mesh)

    # Iterate and verify
    iter(wrapper)
    batch_on_rank = next(wrapper)

    # Verify that this rank received the correct shard
    # With TP row-major, flattened order: [tp0_cp0, tp0_cp1, tp1_cp0, tp1_cp1]
    # Collator output: [cp0_shard, cp1_shard, cp0_shard, cp1_shard]
    expected_batch = expected_sharded_batches[flat_rank]

    torch.testing.assert_close(
        batch_on_rank["input_ids"],
        expected_batch["input_ids"],
        msg=f"Flat rank {flat_rank} (tp={tp_rank}, cp={cp_rank}): input_ids mismatch",
    )
    torch.testing.assert_close(
        batch_on_rank["labels"],
        expected_batch["labels"],
        msg=f"Flat rank {flat_rank} (tp={tp_rank}, cp={cp_rank}): labels mismatch",
    )

    # Gather all batches to verify sharding and replication patterns
    all_input_ids = [None] * dist_config.world_size
    torch.distributed.all_gather_object(all_input_ids, batch_on_rank["input_ids"])

    if flat_rank == 0:
        # Verify TP replication: ranks with same CP rank but different TP rank should have identical data
        # With TP row-major flattened: [tp0_cp0(0), tp0_cp1(1), tp1_cp0(2), tp1_cp1(3)]
        # Rank 0 (tp=0, cp=0) should match Rank 2 (tp=1, cp=0)
        torch.testing.assert_close(
            all_input_ids[0],
            all_input_ids[2],
            msg="TP replication failed: cp=0 ranks (flat 0 and 2) have different data",
        )
        # Rank 1 (tp=0, cp=1) should match Rank 3 (tp=1, cp=1)
        torch.testing.assert_close(
            all_input_ids[1],
            all_input_ids[3],
            msg="TP replication failed: cp=1 ranks (flat 1 and 3) have different data",
        )

        # Verify CP sharding: ranks with different CP ranks should have different data
        # Rank 0 (cp=0) should differ from Rank 1 (cp=1)
        assert not torch.equal(all_input_ids[0], all_input_ids[1]), (
            "CP sharding failed: different CP ranks have the same data"
        )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_cp_dataloader.py <test_name>")
        sys.exit(1)

    test_name = sys.argv[1]

    # CP-only tests
    if test_name == "test_thd_scatter":
        _run_test_thd_scatter()
    elif test_name == "test_bshd_scatter":
        _run_test_bshd_scatter()
    elif test_name == "test_stop_iteration_thd":
        _run_test_stop_iteration(qkv_format="thd")
    elif test_name == "test_stop_iteration_bshd":
        _run_test_stop_iteration(qkv_format="bshd")
    elif test_name == "test_multiple_batches_thd":
        _run_test_multiple_batches(qkv_format="thd")
    elif test_name == "test_multiple_batches_bshd":
        _run_test_multiple_batches(qkv_format="bshd")
    # TP-only tests
    elif test_name == "test_tp_only_thd":
        _run_test_tp_only(qkv_format="thd")
    elif test_name == "test_tp_only_bshd":
        _run_test_tp_only(qkv_format="bshd")
    # CP + TP tests (CP row-major)
    elif test_name == "test_cp_tp_thd":
        _run_test_cp_tp(qkv_format="thd")
    elif test_name == "test_cp_tp_bshd":
        _run_test_cp_tp(qkv_format="bshd")
    # TP + CP tests (TP row-major)
    elif test_name == "test_tp_cp_thd":
        _run_test_tp_cp(qkv_format="thd")
    elif test_name == "test_tp_cp_bshd":
        _run_test_tp_cp(qkv_format="bshd")
    else:
        print(f"Unknown test: {test_name}")
        sys.exit(1)
