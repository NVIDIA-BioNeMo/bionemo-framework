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


import nemo.lightning as nl
import pytest
import torch
from torch import nn

from bionemo.llm import lightning as bnptl
from bionemo.llm.lightning import batch_collator, get_dtype_device
from bionemo.testing import megatron_parallel_state_utils


def test_batch_collate_seqdim_and_singleton_with_padding(batch_size=2, num_batches=5):
    raw_batches = [
        # Try making the data with an unusual dtype (uint8) to verify that it is left unchanged with padding.
        {"idx": torch.tensor([i] * batch_size), "seq": torch.ones(batch_size, i + 1, dtype=torch.uint8)}
        for i in range(num_batches)
    ]
    result = batch_collator(raw_batches)
    assert isinstance(result, dict), "expect output container to be the same type as input (dict)"
    torch.testing.assert_close(result["idx"], torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))
    # Make sure the padding is correct, and that the dtype is left as it was.
    expected_result = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(result["seq"], expected_result)


def test_batch_collate_tuple():
    result = batch_collator(tuple((torch.tensor([i]), torch.tensor([i + 1])) for i in range(10)))
    assert isinstance(result, tuple), "expect output container to be the same type as input (tuple)"
    assert torch.equal(result[0], torch.tensor(list(range(10))))
    assert torch.equal(result[1], torch.tensor([i + 1 for i in range(10)]))


def test_batch_collate_dict():
    result = batch_collator(
        [{"fixed key1": torch.tensor([i]), "fixed key2": torch.tensor([i + 1])} for i in range(10)]
    )
    assert isinstance(result, dict), "expect output container to be the same type as input (dict)"
    assert torch.equal(result["fixed key1"], torch.tensor(list(range(10))))
    assert torch.equal(result["fixed key2"], torch.tensor([i + 1 for i in range(10)]))


def test_batch_collate_list():
    result = batch_collator([[torch.tensor([i]), torch.tensor([i + 1])] for i in range(10)])
    assert isinstance(result, list), "expect output container to be the same type as input (list)"
    assert torch.equal(result[0], torch.tensor(list(range(10))))
    assert torch.equal(result[1], torch.tensor([i + 1 for i in range(10)]))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires at least 2 GPUs")
def test_batch_collate_multi_gpu():
    # Create tensors on different GPUs
    tensor1 = torch.tensor([1, 2, 3], device="cuda:0")
    tensor2 = torch.tensor([4, 5, 6], device="cuda:1")

    result = batch_collator([tensor1, tensor2])

    # Result should be on the first GPU and contain all values
    expected = torch.tensor([1, 2, 3, 4, 5, 6], device="cuda:0")
    assert result.device == torch.device("cuda:0")
    assert torch.equal(result, expected)


def test_batch_collate_none():
    assert batch_collator([None, None]) is None


def test_batch_collator_tensor_fails():
    with pytest.raises(ValueError, match="Unsupported input structure in batch_collator"):
        batch_collator(torch.tensor([[torch.tensor([i]), torch.tensor([i + 1])] for i in range(10)]))


def test_batch_collator_primitive_fails():
    with pytest.raises(ValueError, match="Unsupported input structure in batch_collator"):
        batch_collator(4)


def test_batch_collator_emptylist_fails():
    with pytest.raises(ValueError, match="Cannot process an empty sequence"):
        batch_collator([])


def test_batch_collator_emptytuple_fails():
    with pytest.raises(ValueError, match="Cannot process an empty sequence"):
        batch_collator(())


def test_batch_collator_emptyset_fails():
    with pytest.raises(ValueError, match="Unsupported input structure in batch_collator"):
        batch_collator(set())


def test_batch_collator_emptydict_fails():
    with pytest.raises(ValueError, match="Unsupported input structure in batch_collator"):
        batch_collator({})


def test_tensor_dtype():
    tensor = torch.tensor(4.0, dtype=torch.float32)
    dtype, _ = get_dtype_device(tensor)
    assert dtype == torch.float32


def test_module_dtype():
    module = MyModule(dtype=torch.float32)
    dtype, _ = get_dtype_device(module)
    assert dtype == torch.float32


def test_nested_dtype():
    module = MyModule(dtype=torch.float32)
    nested = NestedModule(module)
    dtype, _ = get_dtype_device(nested)
    assert dtype == torch.float32


def test_dict_tensor_dtype():
    dtype, _ = get_dtype_device({"tensor": torch.tensor(5, dtype=torch.float32)})
    assert dtype == torch.float32


# Handles the cases where we pass in a valid type, but it does not have an associated dtype
def test_empty_module():
    # Module with no underlying parameters
    empty = MyModuleEmpty()
    with pytest.raises(ValueError, match="Cannot get dtype on a torch module with no parameters."):
        get_dtype_device(empty)


def test_none_fails():
    with pytest.raises(ValueError, match="non-None value not found"):
        get_dtype_device([None, None])


def test_empty_dict_fails():
    with pytest.raises(ValueError, match="Looking up dtype on an empty dict"):
        get_dtype_device({})


def test_empty_list_fails():
    with pytest.raises(ValueError, match="Looking up dtype on an empty list"):
        get_dtype_device([])


def test_garbage_fails():
    # String not a valid input type, should work for other garbage values too.
    with pytest.raises(TypeError, match="Got something we didnt expect"):
        get_dtype_device("flkasdflasd")


class MyModule(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10, dtype=dtype) for i in range(10)])
        self.others = nn.ModuleList([nn.Linear(10, 10, dtype=dtype) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x


class MyModuleEmpty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class NestedModule(nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other

    def forward(self, x):
        return self.other(x)


# TODO rewrite unittest and potentially LightningPassthroughPredictionMixin
@pytest.mark.xfail(reason="MegatronStrategy no longer has '_get_loss_reduction' attribute")
def test_mixin_strategy_contract_get_loss_reduction():
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            ddp="megatron",
            find_unused_parameters=True,
            always_save_context=False,
        )
        strategy.connect(bnptl.LightningPassthroughPredictionMixin())
        mixin = bnptl.LightningPassthroughPredictionMixin()
        strategy_reduction_function = strategy._get_loss_reduction("predict")
        assert isinstance(strategy_reduction_function(mixin), bnptl.PassthroughLossReduction)
