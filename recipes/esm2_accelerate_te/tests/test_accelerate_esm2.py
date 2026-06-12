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

# Local helper function import, resolved in conftest.py
from launch import launch_accelerate, requires_gpu, requires_multi_gpu


def test_te_with_default_config(tmp_path):
    train_loss = launch_accelerate("default.yaml", tmp_path, 1, "L0_sanity", "model_tag=./example_8m_checkpoint")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


def test_te_with_fsdp1_config(tmp_path):
    train_loss = launch_accelerate("fsdp1_te.yaml", tmp_path, 1, "L0_sanity", "model_tag=./example_8m_checkpoint")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


def test_te_with_fsdp2_config(tmp_path):
    train_loss = launch_accelerate("fsdp2_te.yaml", tmp_path, 1, "L0_sanity", "model_tag=./example_8m_checkpoint")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


def test_te_with_dynamo_config(tmp_path):
    train_loss = launch_accelerate("dynamo.yaml", tmp_path, 1, "L0_sanity", "model_tag=./example_8m_checkpoint")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_gpu
def test_te_with_fp8_config(tmp_path):
    train_loss = launch_accelerate("fp8.yaml", tmp_path, 1, "L0_sanity", "model_tag=./example_8m_checkpoint")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_gpu
def test_hf_with_default_config(tmp_path):
    train_loss = launch_accelerate("default.yaml", tmp_path, 1, "L0_sanity", "model_tag=facebook/esm2_t6_8M_UR50D")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_gpu
def test_hf_with_fsdp2_config(tmp_path):
    train_loss = launch_accelerate("fsdp2_hf.yaml", tmp_path, 1, "L0_sanity", "model_tag=facebook/esm2_t6_8M_UR50D")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_multi_gpu
def test_te_with_default_config_two_gpus(tmp_path):
    train_loss = launch_accelerate("default.yaml", tmp_path, 2, "L0_sanity", "model_tag=./example_8m_checkpoint")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_multi_gpu
def test_te_with_fsdp1_config_two_gpus(tmp_path):
    train_loss = launch_accelerate("fsdp1_te.yaml", tmp_path, 2, "L0_sanity", "model_tag=./example_8m_checkpoint")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_multi_gpu
def test_te_with_fsdp2_config_two_gpus(tmp_path):
    train_loss = launch_accelerate("fsdp2_te.yaml", tmp_path, 2, "L0_sanity", "model_tag=./example_8m_checkpoint")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_multi_gpu
def test_hf_with_default_config_two_gpus(tmp_path):
    train_loss = launch_accelerate("default.yaml", tmp_path, 2, "L0_sanity", "model_tag=facebook/esm2_t6_8M_UR50D")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_multi_gpu
def test_hf_with_fsdp1_config_two_gpus(tmp_path):
    train_loss = launch_accelerate("fsdp1_hf.yaml", tmp_path, 2, "L0_sanity", "model_tag=facebook/esm2_t6_8M_UR50D")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_multi_gpu
def test_hf_with_fsdp2_config_two_gpus(tmp_path):
    train_loss = launch_accelerate("fsdp2_hf.yaml", tmp_path, 2, "L0_sanity", "model_tag=facebook/esm2_t6_8M_UR50D")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"
