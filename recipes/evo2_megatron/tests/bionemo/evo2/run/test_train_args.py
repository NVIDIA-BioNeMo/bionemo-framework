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

from unittest.mock import MagicMock

import pytest

import bionemo.evo2.run.train as train_module
from bionemo.evo2.run.train import parse_args


def test_no_save_optim_flag_defaults_to_false():
    args = parse_args(["--mock-data"])

    assert args.no_save_optim is False


def test_no_save_optim_flag_can_be_enabled():
    args = parse_args(["--mock-data", "--no-save-optim"])

    assert args.no_save_optim is True


@pytest.mark.parametrize(
    ("extra_args", "expected_save_optim"),
    [
        ([], True),
        (["--no-save-optim"], False),
    ],
)
def test_train_assigns_save_optim_from_no_save_optim_flag(monkeypatch, extra_args, expected_save_optim):
    cfg = MagicMock()
    cfg.checkpoint.load = None
    mocked_pretrain_config = MagicMock(return_value=cfg)
    mocked_pretrain = MagicMock()
    monkeypatch.setattr(train_module, "pretrain_config", mocked_pretrain_config)
    monkeypatch.setattr(train_module, "pretrain", mocked_pretrain)
    monkeypatch.setattr(train_module, "get_rank_safe", lambda: 1)
    monkeypatch.setattr(train_module.torch.distributed, "is_initialized", lambda: False)

    args = parse_args(["--mock-data", *extra_args])
    train_module.train(args)
    mocked_pretrain.assert_called_once_with(cfg, train_module.hyena_forward_step)

    assert cfg.checkpoint.save_optim is expected_save_optim
    mocked_pretrain_config.assert_called_once()
