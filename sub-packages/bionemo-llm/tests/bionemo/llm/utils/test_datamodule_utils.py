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


from bionemo.llm.utils.datamodule_utils import float_or_int_or_none, infer_global_batch_size


def test_float_or_int_or_none_type_float():
    """Test that float_or_int_or_none returns a float when given a float on edge case 1.0"""
    assert isinstance(float_or_int_or_none(1.0), float)
    assert isinstance(float_or_int_or_none("1.0"), float)


def test_float_or_int_or_none_type_int():
    """Test that float_or_int_or_none returns an int when given an int on edge case 1"""
    assert isinstance(float_or_int_or_none(1), int)
    assert isinstance(float_or_int_or_none("1"), int)


def test_float_or_int_or_none_type_none():
    """Test that float_or_int_or_none returns None when given None"""
    assert float_or_int_or_none(None) is None
    assert float_or_int_or_none("None") is None


def test_infer_global_batch_size():
    """Test that infer_global_batch_size returns the correct global batch size"""
    assert infer_global_batch_size(micro_batch_size=1, num_nodes=1, devices=1) == 1  # single node, single device
    assert infer_global_batch_size(micro_batch_size=1, num_nodes=1, devices=8) == 8  # single node, multi device
    assert (
        infer_global_batch_size(
            micro_batch_size=1,
            num_nodes=2,
            devices=8,
        )
        == 16
    )  # multi node, multi device
    assert (
        infer_global_batch_size(micro_batch_size=1, num_nodes=2, devices=8, pipeline_model_parallel_size=2) == 8
    )  # multi node, multi device with pipeline parallel
    assert (
        infer_global_batch_size(
            micro_batch_size=1, num_nodes=2, devices=8, pipeline_model_parallel_size=2, tensor_model_parallel_size=2
        )
        == 4
    )  # multi node, multi device with pipeline and tensor parallel
    assert (
        infer_global_batch_size(
            micro_batch_size=1,
            num_nodes=2,
            devices=8,
            pipeline_model_parallel_size=2,
            tensor_model_parallel_size=2,
            accumulate_grad_batches=2,
        )
        == 8
    )  # multi node, multi device with pipeline and tensor parallel, and accumulate grad batches
