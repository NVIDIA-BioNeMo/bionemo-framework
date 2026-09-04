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

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh

from checkpoint import load_dataloader, save_dataloader
from dataset import DistributedConfig, create_bshd_dataloader, create_cp_dataloader, create_thd_dataloader


requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


@dataclass
class MockDistributedConfig:
    rank: int
    local_rank: int
    world_size: int

    def is_main_process(self) -> bool:
        return self.rank == 0


def test_load_dataset_state_from_latest_checkpoint(tmp_path):
    dataloader_path = tmp_path / "dl_test"
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    reference_dataloader, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    for i, _ in enumerate(reference_dataloader):
        if i in [1, 5, 9]:
            dataloader_path = dataloader_path / f"step_{i}"
            os.makedirs(dataloader_path, exist_ok=True)
            save_dataloader(
                dataloader=reference_dataloader,
                ckpt_path=dataloader_path,
                dist_config=dist_config,
            )
        if i == 9:
            break

    new_dataloader, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    new_dataloader = load_dataloader(
        dataloader=new_dataloader,
        ckpt_path=dataloader_path,
        dist_config=dist_config,
    )

    assert new_dataloader.state_dict()["_snapshot"]["_snapshot_step"] == 10


def test_map_style_stateful_dataloader_resumption_multi_process(tmp_path):
    dataloader_path = tmp_path / "dl_test"
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": False,
    }

    rank0_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )
    rank1_config = MockDistributedConfig(
        rank=1,
        local_rank=1,
        world_size=2,
    )

    # Based on local rank.
    # Create dataloader for process 0
    rank0_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank0_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    rank1_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    dataloader_path_step_4 = dataloader_path / "step_4"
    dataloader_path_step_5 = dataloader_path / "step_5"
    os.makedirs(dataloader_path_step_4, exist_ok=True)
    os.makedirs(dataloader_path_step_5, exist_ok=True)

    # Run 10 batches, save state at step 5
    reference_rank0_batches = []
    for i, batch in enumerate(rank0_dataloader):
        reference_rank0_batches.append(batch["input_ids"])
        if i == 5:
            save_dataloader(
                dataloader=rank0_dataloader,
                ckpt_path=dataloader_path_step_5,
                dist_config=rank0_config,
            )
        if i == 9:
            break

    # Run 10 batches, save state at step 4
    reference_rank1_batches = []

    for i, batch in enumerate(rank1_dataloader):
        reference_rank1_batches.append(batch["input_ids"])
        if i == 4:
            save_dataloader(
                dataloader=rank1_dataloader,
                ckpt_path=dataloader_path_step_4,
                dist_config=rank1_config,
            )
        if i == 9:
            break

    # Load rank0 dataloader state at step 5
    rank0_dataloader_info_reloaded, _ = create_bshd_dataloader(
        distributed_config=rank0_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    rank0_dataloader_reloaded = load_dataloader(
        dataloader=rank0_dataloader_info_reloaded,
        ckpt_path=dataloader_path_step_5,
        dist_config=rank0_config,
    )

    # Load rank1 dataloader state at step 4
    rank1_dataloader_info_reloaded, _ = create_bshd_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    rank1_dataloader_reloaded = load_dataloader(
        dataloader=rank1_dataloader_info_reloaded,
        ckpt_path=dataloader_path_step_4,
        dist_config=rank1_config,
    )

    # Run 3 more steps on loaded_rank0_dataloader and save the batches
    loaded_rank0_batches = []
    for i, batch in enumerate(rank0_dataloader_reloaded):
        loaded_rank0_batches.append(batch["input_ids"])
        if i == 2:  # Collect 3 batches (indices 0-2) to match with reference batches 7-9
            break
    # Run 4 more steps on loaded_rank1_dataloader and save the batches
    loaded_rank1_batches = []
    for i, batch in enumerate(rank1_dataloader_reloaded):
        loaded_rank1_batches.append(batch["input_ids"])
        if i == 3:  # Collect 4 batches (indices 0-3) to match with reference batches 6-9
            break

    assert torch.equal(loaded_rank0_batches[0], reference_rank0_batches[6])
    assert torch.equal(loaded_rank0_batches[1], reference_rank0_batches[7])
    assert torch.equal(loaded_rank0_batches[2], reference_rank0_batches[8])

    assert torch.equal(loaded_rank1_batches[0], reference_rank1_batches[5])
    assert torch.equal(loaded_rank1_batches[1], reference_rank1_batches[6])
    assert torch.equal(loaded_rank1_batches[2], reference_rank1_batches[7])
    assert torch.equal(loaded_rank1_batches[3], reference_rank1_batches[8])

    shutil.rmtree(dataloader_path)


def test_iterable_stateful_dataloader_resumption_multi_process(tmp_path):
    dataloader_path = tmp_path / "dl_test"
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    rank0_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )
    rank1_config = MockDistributedConfig(
        rank=1,
        local_rank=1,
        world_size=2,
    )

    # Based on local rank.
    # Create dataloader for process 0
    rank0_dataloader_info, _ = create_thd_dataloader(
        distributed_config=rank0_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    rank1_dataloader_info, _ = create_thd_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )
    dataloader_path_step_4 = dataloader_path / "step_4"
    dataloader_path_step_5 = dataloader_path / "step_5"
    os.makedirs(dataloader_path_step_4, exist_ok=True)
    os.makedirs(dataloader_path_step_5, exist_ok=True)

    # Run 10 batches, save state at step 5
    reference_rank0_batches = []
    for i, batch in enumerate(rank0_dataloader_info):
        reference_rank0_batches.append(batch["input_ids"])
        if i == 5:
            save_dataloader(
                dataloader=rank0_dataloader_info,
                ckpt_path=dataloader_path_step_5,
                dist_config=rank0_config,
            )
        if i == 9:
            break

    # Run 10 batches, save state at step 4
    reference_rank1_batches = []

    for i, batch in enumerate(rank1_dataloader_info):
        reference_rank1_batches.append(batch["input_ids"])
        if i == 4:
            save_dataloader(
                dataloader=rank1_dataloader_info,
                ckpt_path=dataloader_path_step_4,
                dist_config=rank1_config,
            )
        if i == 9:
            break

    # Load rank0 dataloader state at step 5
    rank0_dataloader_reloaded, _ = create_thd_dataloader(
        distributed_config=rank0_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    rank0_dataloader_reloaded = load_dataloader(
        dataloader=rank0_dataloader_reloaded,
        ckpt_path=dataloader_path_step_5,
        dist_config=rank0_config,
    )

    # Load rank1 dataloader state at step 4
    rank1_dataloader_reloaded, _ = create_thd_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    rank1_dataloader_reloaded = load_dataloader(
        dataloader=rank1_dataloader_reloaded,
        ckpt_path=dataloader_path_step_4,
        dist_config=rank1_config,
    )

    # Run 3 more steps on loaded_rank0_dataloader and save the batches
    loaded_rank0_batches = []
    for i, batch in enumerate(rank0_dataloader_reloaded):
        loaded_rank0_batches.append(batch["input_ids"])
        if i == 2:  # Collect 3 batches (indices 0-2) to match with reference batches 7-9
            break
    # Run 4 more steps on loaded_rank1_dataloader and save the batches
    loaded_rank1_batches = []
    for i, batch in enumerate(rank1_dataloader_reloaded):
        loaded_rank1_batches.append(batch["input_ids"])
        if i == 3:  # Collect 4 batches (indices 0-3) to match with reference batches 6-9
            break

    assert torch.equal(loaded_rank0_batches[0], reference_rank0_batches[6])
    assert torch.equal(loaded_rank0_batches[1], reference_rank0_batches[7])
    assert torch.equal(loaded_rank0_batches[2], reference_rank0_batches[8])

    assert torch.equal(loaded_rank1_batches[0], reference_rank1_batches[5])
    assert torch.equal(loaded_rank1_batches[1], reference_rank1_batches[6])
    assert torch.equal(loaded_rank1_batches[2], reference_rank1_batches[7])
    assert torch.equal(loaded_rank1_batches[3], reference_rank1_batches[8])

    shutil.rmtree(dataloader_path)


def test_stateful_dataloader_works_save_dataloader_and_load_dataloader_single_process(tmp_path):
    # Test uses rank 0.
    dataloader_path = tmp_path / "dl_test"
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    # First, collect reference batches from a fresh dataloader
    reference_dataloader_info, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    # Collect 10 batches in total. Save the state of the sixth batch at iteration 5.
    reference_batches = []
    for i, batch in enumerate(reference_dataloader_info):
        reference_batches.append(batch["input_ids"])
        if i == 5:
            # save the state of the fifth batch
            save_dataloader(
                dataloader=reference_dataloader_info,
                ckpt_path=dataloader_path,
                dist_config=dist_config,
            )
        if i == 9:  # Collect 10 batches total
            break

    # Now test checkpoint/restore
    new_dataloader_info, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    new_dataloader = load_dataloader(
        dataloader=new_dataloader_info,
        ckpt_path=dataloader_path,
        dist_config=dist_config,
    )

    loaded_batches = []
    for i, batch in enumerate(new_dataloader):
        loaded_batches.append(batch["input_ids"])
        if i == 2:
            break

    assert len(reference_batches) == 10
    assert len(loaded_batches) == 3

    assert torch.equal(loaded_batches[0], reference_batches[6])
    assert torch.equal(loaded_batches[1], reference_batches[7])
    assert torch.equal(loaded_batches[2], reference_batches[8])

    shutil.rmtree(dataloader_path)


def test_stateful_dataloader():
    """Test that the stateful dataloader works with streaming = False.
    First we create a fresh dataloader and collect 10 batches, specified by 0th first index [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Save the state of the dataloader after the sixth batch (at iteration 5).
    Then we create another dataloader called loaded_dataloader and collect 3 batches which should be [6, 7, 8].
    then we compare the first 3 batches of the loaded_dataloader to batches 6, 7, 8 of the reference_batches.
    """

    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    # First, collect reference batches from a fresh dataloader
    reference_dataloader_info, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    # Collect 10 batches in total. Save the state of the sixth batch at iteration 5.
    reference_batches = []
    for i, batch in enumerate(reference_dataloader_info):
        reference_batches.append(batch["input_ids"])
        if i == 5:
            # save the state of the fifth batch
            dataloader_state = reference_dataloader_info.state_dict()
        if i == 9:  # Collect 10 batches total
            break

    # Now test checkpoint/restore
    new_dataloader_info, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    new_dataloader = new_dataloader_info
    new_dataloader.load_state_dict(dataloader_state)

    # Note: Maybe the transform is non deterministic? Like the lazy loading map function.
    # Get three batches of data
    loaded_batches = []
    for i, batch in enumerate(new_dataloader):
        loaded_batches.append(batch["input_ids"])
        if i == 2:
            break

    assert len(reference_batches) == 10
    assert len(loaded_batches) == 3

    assert torch.equal(loaded_batches[0], reference_batches[6])
    assert torch.equal(loaded_batches[1], reference_batches[7])
    assert torch.equal(loaded_batches[2], reference_batches[8])


def test_stateful_dataloader_with_multiple_workers(tmp_path):
    """Test that the stateful dataloader works with multiple GPUs."""
    dataloader_path = tmp_path / "dl_test_multi_workers"
    shutil.rmtree(dataloader_path, ignore_errors=True)
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": False,
    }

    dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    # First, collect reference batches from a fresh dataloader
    reference_dataloader, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=2,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    # Collect 10 batches in total. Save the state of the sixth batch at iteration 5.
    reference_batches = []
    for i, batch in enumerate(reference_dataloader):
        reference_batches.append(batch["input_ids"])
        if i == 5:
            # save the state of the fifth batch
            dataloader_path = dataloader_path / f"step_{i}"
            os.makedirs(dataloader_path, exist_ok=True)
            save_dataloader(
                dataloader=reference_dataloader,
                ckpt_path=dataloader_path,
                dist_config=dist_config,
            )
        if i == 9:  # Collect 10 batches total
            break

    # Now test checkpoint/restore
    new_dataloader, _ = create_bshd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=2,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    load_dataloader(
        dataloader=new_dataloader,
        ckpt_path=dataloader_path,
        dist_config=dist_config,
    )

    loaded_batches = []
    for i, batch in enumerate(new_dataloader):
        loaded_batches.append(batch["input_ids"])
        if i == 2:
            break

    assert len(reference_batches) == 10
    assert len(loaded_batches) == 3

    assert torch.equal(loaded_batches[0], reference_batches[6])
    assert torch.equal(loaded_batches[1], reference_batches[7])
    assert torch.equal(loaded_batches[2], reference_batches[8])


def test_iterable_dataloader_yields_different_values_per_rank():
    """Test that the iterable dataloader yields different values per rank."""
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
        # The only difference here is that this dataset doesn't set streaming to True
    }

    rank1_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )

    rank1_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_duplicate_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_stateful_dataloader=True,
    )

    rank2_config = MockDistributedConfig(
        rank=1,
        local_rank=1,
        world_size=2,
    )

    rank2_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank2_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_stateful_dataloader=True,
    )

    rank1_batch = next(iter(rank1_dataloader))
    rank1_duplicate_batch = next(iter(rank1_duplicate_dataloader))
    rank2_batch = next(iter(rank2_dataloader))

    for key, value in rank1_batch.items():
        assert rank1_batch[key] is not None
        assert (value != rank2_batch[key]).any()
        torch.testing.assert_close(value, rank1_duplicate_batch[key])


def test_map_dataset_dataloader_yields_different_values_per_rank():
    """Test that the map-style dataloader yields different values per rank."""

    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        # The only difference here is that this dataset doesn't set streaming to True
    }

    rank1_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )

    rank1_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_duplicate_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank1_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank2_config = MockDistributedConfig(
        rank=1,
        local_rank=1,
        world_size=2,
    )

    rank2_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank2_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
    )

    rank1_batch = next(iter(rank1_dataloader))
    rank1_duplicate_batch = next(iter(rank1_duplicate_dataloader))
    rank2_batch = next(iter(rank2_dataloader))

    for key, value in rank1_batch.items():
        assert (value != rank2_batch[key]).any()
        torch.testing.assert_close(value, rank1_duplicate_batch[key])


def test_lazy_tokenization_returns_batch():
    """Test that the lazy tokenization works."""

    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": False,
    }

    config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )

    dataloader, _ = create_bshd_dataloader(
        distributed_config=config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        use_lazy_tokenization=True,
    )

    batch = next(iter(dataloader))
    assert batch is not None


def test_stateful_dataloader_load_fails_if_num_workers_mismatch(tmp_path, caplog):
    dataloader_path = tmp_path / "dl_test_num_workers_mismatch"
    shutil.rmtree(dataloader_path, ignore_errors=True)
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": False,
    }

    rank0_dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    reference_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank0_dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    save_dataloader(
        dataloader=reference_dataloader,
        ckpt_path=dataloader_path,
        dist_config=rank0_dist_config,
    )

    del reference_dataloader

    reference_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank0_dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=2,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    with caplog.at_level(logging.WARNING):
        load_dataloader(
            dataloader=reference_dataloader,
            ckpt_path=dataloader_path,
            dist_config=rank0_dist_config,
        )

    assert (
        "Dataloader num_workers mismatch: 2 != 1 or num_ranks mismatch: 1 != 1, starting dataloader from scratch."
        in caplog.text
    )


def test_stateful_dataloader_load_fails_if_num_ranks_mismatch(tmp_path, caplog):
    dataloader_path = tmp_path / "dl_test_num_workers_mismatch"
    shutil.rmtree(dataloader_path, ignore_errors=True)
    os.makedirs(dataloader_path, exist_ok=True)
    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": False,
    }

    rank0_dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    reference_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank0_dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    save_dataloader(
        dataloader=reference_dataloader,
        ckpt_path=dataloader_path,
        dist_config=rank0_dist_config,
    )

    del reference_dataloader
    del rank0_dist_config

    rank2_dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=2,
    )

    reference_dataloader, _ = create_bshd_dataloader(
        distributed_config=rank2_dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=4,
        num_workers=1,
        mlm_probability=0,
        use_stateful_dataloader=True,
    )

    with caplog.at_level(logging.WARNING):
        load_dataloader(
            dataloader=reference_dataloader,
            ckpt_path=dataloader_path,
            dist_config=rank2_dist_config,
        )

    assert (
        "Dataloader num_workers mismatch: 1 != 1 or num_ranks mismatch: 2 != 1, starting dataloader from scratch."
        in caplog.text
    )


def test_token_packing_dataloader():
    """Test that the token packing dataloader works."""

    tokenizer_name = "facebook/esm2_t6_8M_UR50D"
    load_dataset_kwargs = {
        "path": "parquet",
        "split": "train",
        "data_files": "train.parquet",
        "streaming": True,
    }

    dist_config = MockDistributedConfig(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    dataloader, _ = create_thd_dataloader(
        distributed_config=dist_config,
        tokenizer_name=tokenizer_name,
        load_dataset_kwargs=load_dataset_kwargs,
        token_micro_batch_size=8 * 1024,
    )

    batch = next(iter(dataloader))
    assert batch["input_ids"].shape[1] == 8 * 1024
    assert batch["labels"].shape[1] == 8 * 1024


@requires_multi_gpu
def test_cp_dataloader(recipe_path):
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(recipe_path)

    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "tests/test_dataset.py",
    ]

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
        cwd=str(recipe_path),
        env=env,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command failed with exit code {result.returncode}")


if __name__ == "__main__":
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    device_mesh = init_device_mesh("cuda", mesh_shape=(1, 2), mesh_dim_names=("dp", "cp"))

    dataloader, _ = create_cp_dataloader(
        distributed_config=dist_config,
        cp_mesh=device_mesh["cp"],
        tokenizer_name="facebook/esm2_t6_8M_UR50D",
        load_dataset_kwargs={
            "path": "parquet",
            "split": "train",
            "data_files": "train.parquet",
            "streaming": True,
        },
        token_micro_batch_size=8 * 1024,
        num_workers=1,
    )

    batch = next(iter(dataloader))
    # With CP size 2, each sequence is split into 2 * cp_world_size = 4 slices.
    # Each rank gets 2 slices (beginning and end), so each rank gets approximately
    # (8 * 1024) / 2 = 4096 tokens per rank
    # Note: Sequences are padded to be divisible by pad_sequences_to_be_divisible_by
    # (which defaults to cp_mesh.size() * 2 = 4 if not provided)
    # The actual token count per rank can vary due to:
    # 1. Sequence packing (variable-length sequences packed up to token_micro_batch_size)
    # 2. Per-sequence padding to be divisible by pad_sequences_to_be_divisible_by
    # 3. CP splitting logic that takes slices from beginning and end
    expected_tokens_per_rank = (8 * 1024) // device_mesh["cp"].size()
    actual_shape = batch["input_ids"].shape[1]
    # Allow for variance due to sequence packing, padding, and CP splitting
    # The actual shape should be close to expected_tokens_per_rank but can vary
    # Allow up to 100 tokens of variance (both above and below) to account for
    # sequence packing and padding effects
    assert actual_shape >= expected_tokens_per_rank - 100, (
        f"Expected at least {expected_tokens_per_rank - 100} tokens, got {actual_shape}"
    )
    assert actual_shape <= expected_tokens_per_rank + 100, (
        f"Expected at most {expected_tokens_per_rank + 100} tokens, got {actual_shape}"
    )
    assert batch["labels"].shape[1] == actual_shape

    dataloader.close()
    torch.distributed.destroy_process_group()
