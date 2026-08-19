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


def test_best_checkpoint_args():
    args = parse_args(
        [
            "--mock-data",
            "--keep-best-k",
            "3",
            "--most-recent-k",
            "1",
            "--checkpoint-metric-name",
            "lm loss",
            "--checkpoint-metric-step-tolerance",
            "3",
            "--eval-interval",
            "20",
            "--save-interval",
            "100",
        ]
    )

    assert args.keep_best_k == 3
    assert args.most_recent_k == 1
    assert args.checkpoint_metric_name == "lm loss"
    assert args.checkpoint_metric_mode == "min"
    assert args.checkpoint_metric_step_tolerance == 3
    assert args.save_interval == 100


def test_best_checkpoint_args_check_cadence():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mock-data",
                "--keep-best-k",
                "3",
                "--eval-interval",
                "20",
                "--save-interval",
                "30",
            ]
        )


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


def test_train_installs_metric_retention(monkeypatch):
    cfg = MagicMock()
    cfg.checkpoint.load = None
    mocked_pretrain = MagicMock()
    monkeypatch.setattr(train_module, "pretrain_config", MagicMock(return_value=cfg))
    monkeypatch.setattr(train_module, "pretrain", mocked_pretrain)
    monkeypatch.setattr(train_module, "get_rank_safe", lambda: 1)
    monkeypatch.setattr(train_module.torch.distributed, "is_initialized", lambda: False)

    args = parse_args(
        [
            "--mock-data",
            "--eval-interval",
            "20",
            "--save-interval",
            "100",
            "--keep-best-k",
            "3",
            "--most-recent-k",
            "1",
            "--strict-checkpoint-metric",
            "--checkpoint-metric-step-tolerance",
            "3",
        ]
    )
    train_module.train(args)

    assert cfg.checkpoint.most_recent_k == -1
    assert cfg.checkpoint.save_interval == 100
    callbacks = mocked_pretrain.call_args.kwargs["callbacks"]
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert callback.metric_name == "lm loss"
    assert callback.keep_best_k == 3
    assert callback.keep_recent_k == 1
    assert callback.step_tolerance == 3
    assert callback.strict_metric is True
