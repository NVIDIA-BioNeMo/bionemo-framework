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

from unittest.mock import MagicMock, patch

import torch

from src.tasks import evaluate as evaluate_fn
from src.tasks import train as train_fn


def _make_min_config():
    logger = MagicMock()
    data = MagicMock()
    trainer_kwargs = {"max_steps": 1}
    model = MagicMock()
    callbacks = {"cb": MagicMock()}
    return {
        "log": logger,
        "data": data,
        "trainer": trainer_kwargs,
        "model": model,
        "callbacks": callbacks,
    }


@patch("src.tasks.Trainer")
def test_train_with_existing_ckpt(mock_trainer, tmp_path):
    config = _make_min_config()
    ckpt_path = str(tmp_path / "last.ckpt")
    # create a real checkpoint file
    torch.save({"state_dict": {"model.layer": 1}}, ckpt_path)

    trainer = MagicMock()
    mock_trainer.return_value = trainer

    train_fn(
        config=config,
        ckpt_path=ckpt_path,
        seed=123,
        config_dict={"foo": "bar"},
        out_dir=str(tmp_path),
    )

    # model.configure_model called with state dict from ckpt
    config["model"].configure_model.assert_called_once()
    _, kwargs = config["model"].configure_model.call_args
    assert kwargs["state_dict"] == {"model.layer": 1}

    # trainer.fit called with ckpt_path pointing to existing ckpt
    trainer.fit.assert_called_once()
    _, k = trainer.fit.call_args
    assert k.get("ckpt_path") == ckpt_path


@patch("src.tasks.Trainer")
def test_train_without_ckpt_starts_fresh(mock_trainer, tmp_path):
    config = _make_min_config()
    trainer = MagicMock()
    mock_trainer.return_value = trainer

    train_fn(
        config=config,
        ckpt_path=str(tmp_path / "last.ckpt"),
        seed=123,
        config_dict={},
        out_dir=str(tmp_path),
    )

    config["model"].configure_model.assert_called_once()
    _, kwargs = config["model"].configure_model.call_args
    assert kwargs == {}

    # ckpt_path should be None on first-time train
    _, k = trainer.fit.call_args
    assert k.get("ckpt_path") is None


@patch("src.tasks.Trainer")
def test_evaluate_with_ckpt_loads_datamodule_and_sets_counter(mock_trainer, tmp_path):
    config = _make_min_config()
    # simulate datamodule with init_global_step
    config["data"].init_global_step = 777

    model_ckpt_path = str(tmp_path / "model_state.ckpt")
    # write a small checkpoint
    torch.save({"any": "thing"}, model_ckpt_path)

    trainer = MagicMock()
    mock_trainer.return_value = trainer

    evaluate_fn(
        config=config,
        config_dict={"cfg": 1},
        model_ckpt_path=model_ckpt_path,
        out_dir=str(tmp_path),
        seed=321,
    )

    # datamodule loads state_dict and prediction_counter set
    config["data"].load_state_dict.assert_called_once()
    assert config["model"].prediction_counter == 777

    trainer.predict.assert_called_once()


@patch("src.tasks.Trainer")
def test_evaluate_without_ckpt_skips_loading(mock_trainer, tmp_path):
    config = _make_min_config()
    trainer = MagicMock()
    mock_trainer.return_value = trainer

    evaluate_fn(
        config=config,
        config_dict={"cfg": 2},
        model_ckpt_path=str(tmp_path / "missing.ckpt"),
        out_dir=str(tmp_path),
        seed=1,
    )

    config["data"].load_state_dict.assert_not_called()
    trainer.predict.assert_called_once()
