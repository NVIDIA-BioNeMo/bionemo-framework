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

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from nemo import lightning as nl

from bionemo.testing import megatron_parallel_state_utils


MAX_WORLD_SIZE = 4
AVAILABLE_WORLD_SIZE = torch.cuda.device_count()
WORLD_SIZES = range(min(MAX_WORLD_SIZE, torch.cuda.device_count()))


def _test_all_reduce_sum(rank: int, world_size: int):
    """Private test function for torch.distributed mean reduce."""
    with megatron_parallel_state_utils.distributed_model_parallel_state(rank=rank, world_size=world_size):
        tensor = torch.tensor([rank + 1]).cuda(rank)
        dist.all_reduce(tensor)
        assert tensor.item() == world_size * (world_size + 1) / 2


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_all_reduce_sum(world_size: int):
    """Multiprocessing test of _test_all_reduce_sum."""
    torch.multiprocessing.spawn(
        fn=_test_all_reduce_sum,
        args=(world_size,),
        nprocs=world_size,
    )


def _test_data_parallel_group(rank: int, world_size: int):
    """Private test function for dp parallel state."""
    with megatron_parallel_state_utils.distributed_model_parallel_state(rank=rank, world_size=world_size):
        assert parallel_state.get_data_parallel_rank() == rank
        assert parallel_state.get_data_parallel_world_size() == world_size
        assert parallel_state.get_data_parallel_src_rank() == 0


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_data_parallel_group(world_size: int):
    """Multiprocessing test of _test_data_parallel_group."""
    torch.multiprocessing.spawn(
        fn=_test_data_parallel_group,
        args=(world_size,),
        nprocs=world_size,
    )


def _test_tensor_model_parallel_group(rank: int, world_size: int):
    """Private test function for tp parallel state."""
    with megatron_parallel_state_utils.distributed_model_parallel_state(
        rank=rank, world_size=world_size, tensor_model_parallel_size=world_size
    ):
        assert parallel_state.get_tensor_model_parallel_rank() == rank
        assert parallel_state.get_tensor_model_parallel_world_size() == world_size
        assert parallel_state.get_tensor_model_parallel_src_rank() == 0


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_tensor_model_parallel_group(world_size: int):
    """Multiprocessing test of _test_tensor_model_parallel_group."""
    torch.multiprocessing.spawn(
        fn=_test_tensor_model_parallel_group,
        args=(world_size,),
        nprocs=world_size,
    )


def _test_pipeline_model_parallel_group(rank: int, world_size: int):
    """Private test function for pp parallel state."""
    with megatron_parallel_state_utils.distributed_model_parallel_state(
        rank=rank, world_size=world_size, pipeline_model_parallel_size=world_size
    ):
        assert parallel_state.get_pipeline_model_parallel_rank() == rank
        assert parallel_state.get_pipeline_model_parallel_world_size() == world_size
        if rank == 0:
            assert parallel_state.is_pipeline_first_stage()
        if rank == world_size:
            assert parallel_state.is_pipeline_last_stage()


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_pipeline_model_parallel_group(world_size: int):
    """Multiprocessing test of _test_pipeline_model_parallel_group."""
    torch.multiprocessing.spawn(
        fn=_test_pipeline_model_parallel_group,
        args=(world_size,),
        nprocs=world_size,
    )


def test_load_megatron_strategy():
    # This will clean up most of the megatron global state that can get created
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)
        assert strategy.tensor_model_parallel_size == 1


def test_construct_nemo_lightning_trainer():
    # This will clean up most of the megatron global state that can get created
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        trainer = nl.Trainer(
            devices=1,
            max_steps=5,
            accelerator="gpu",
            strategy=nl.MegatronStrategy(tensor_model_parallel_size=1),
        )
        assert trainer.max_steps == 5


def test_mock_rank0_first_pipeline():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=8, rank=0, pipeline_model_parallel_size=8
    ):
        assert parallel_state.is_pipeline_first_stage()
        assert not parallel_state.is_pipeline_last_stage()


def test_mock_rank4_mid_pipeline():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=8, rank=4, pipeline_model_parallel_size=8
    ):
        assert not parallel_state.is_pipeline_first_stage()
        assert not parallel_state.is_pipeline_last_stage()


def test_mock_rank7_last_pipeline():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(
        world_size=8, rank=7, pipeline_model_parallel_size=8
    ):
        assert not parallel_state.is_pipeline_first_stage()
        assert parallel_state.is_pipeline_last_stage()


def test_mock_get_pp_group():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, pipeline_model_parallel_size=2):
        assert parallel_state.get_pipeline_model_parallel_group() is not None


def test_mock_get_tp_group():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, tensor_model_parallel_size=2):
        assert parallel_state.get_tensor_model_parallel_group() is not None


def test_mock_get_cp_group():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, context_parallel_size=2):
        assert parallel_state.get_context_parallel_group() is not None


def test_mock_all_reduce():
    # Adapted from https://github.com/pytorch/pytorch/blob/main/test/distributed/test_fake_pg.py
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, rank=1):
        output = torch.ones(3, 3).cuda() * dist.get_rank()
        dist.all_reduce(output)
        assert tuple(output.shape) == (3, 3)


def test_mock_allgather():
    # Adapted from https://github.com/pytorch/pytorch/blob/main/test/distributed/test_fake_pg.py
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, rank=1):
        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [torch.empty_like(input_tensor) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for _, out_tensor in enumerate(output_tensors):
            assert tuple(out_tensor.shape) == (3, 3)


def test_mock_reduce_scatter():
    # Adapted from https://github.com/pytorch/pytorch/blob/main/test/distributed/test_fake_pg.py
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, rank=1):
        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        output_tensor = torch.empty(3, 3)

        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        assert tuple(output_tensor.shape) == (3, 3)


def test_mock_all_reduce_sum():
    with megatron_parallel_state_utils.mock_distributed_parallel_state(world_size=2, rank=1):
        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor)
        assert tensor.item() == 2  # there is no actual communication in mock distributed
