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

from bionemo.esm2.data import tokenizer
from bionemo.esm2.model.finetune.token_model import (
    ESM2FineTuneTokenConfig,
    ESM2FineTuneTokenModel,
    MegatronConvNetHead,
)
from bionemo.testing import megatron_parallel_state_utils


@pytest.fixture
def config():
    return ESM2FineTuneTokenConfig(encoder_frozen=True, cnn_dropout=0.1, cnn_hidden_size=32, cnn_num_classes=5)


@pytest.fixture
def finetune_token_model(config):
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        model = config.configure_model(tokenizer.get_tokenizer())
        yield model


def test_ft_config(config):
    assert config.initial_ckpt_skip_keys_with_these_prefixes == ["classification_head"]
    assert config.encoder_frozen
    assert config.cnn_dropout == 0.1
    assert config.cnn_hidden_size == 32
    assert config.cnn_num_classes == 5


def test_ft_model_initialized(finetune_token_model):
    assert isinstance(finetune_token_model, ESM2FineTuneTokenModel)
    assert isinstance(finetune_token_model.classification_head, MegatronConvNetHead)
    assert finetune_token_model.post_process
    assert not finetune_token_model.include_hiddens_finetuning
