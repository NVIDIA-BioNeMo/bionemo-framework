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

"""
These are tests for the AMPLIFY model, an ESM-2 variant with a modified different architecture and tokenizer. See
https://doi.org/10.1101/2024.09.23.614603 and huggingface.co/chandar-lab/amplify_350m for more details. Note: in these
tests, we don't test the original xformers-based model, since we don't install xformers in our base image for these
recipe tests.
"""

# Local helper function import, resolved in conftest.py
from launch import launch_accelerate, requires_multi_gpu


def test_te_with_default_config(tmp_path):
    train_loss = launch_accelerate("default.yaml", tmp_path, 1, "L0_sanity_amplify")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


def test_te_with_dynamo_config(tmp_path):
    train_loss = launch_accelerate("dynamo.yaml", tmp_path, 1, "L0_sanity_amplify")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


def test_te_with_fp8_config(tmp_path):
    train_loss = launch_accelerate("fp8.yaml", tmp_path, 1, "L0_sanity_amplify")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


def test_te_with_fsdp2_config(tmp_path):
    train_loss = launch_accelerate("fsdp2_te.yaml", tmp_path, 1, "L0_sanity_amplify")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_multi_gpu
def test_te_with_default_config_two_gpus(tmp_path):
    train_loss = launch_accelerate("default.yaml", tmp_path, 2, "L0_sanity_amplify")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_multi_gpu
def test_te_with_fp8_config_two_gpus(tmp_path):
    train_loss = launch_accelerate("fp8.yaml", tmp_path, 2, "L0_sanity_amplify")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"


@requires_multi_gpu
def test_te_with_fsdp2_config_two_gpus(tmp_path):
    train_loss = launch_accelerate("fsdp2_te.yaml", tmp_path, 2, "L0_sanity_amplify")
    assert train_loss < 3.0, f"Final train_loss {train_loss} should be less than 3.0"
