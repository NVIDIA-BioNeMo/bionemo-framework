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

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.tasks import finetune as finetune_fn


def _make_min_config():
    """Create a minimal config dict structure expected by tasks.finetune."""
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


@patch("src.tasks.load_checkpoint")
@patch("src.tasks.Trainer")
@patch("src.tasks.os.path.exists")
def test_finetune_with_safetensors_loads_and_uses_none_ckpt(mock_exists, mock_trainer, mock_load_checkpoint, tmp_path):
    config = _make_min_config()

    # Pretend pretrained safetensors exists and target ckpt does not
    pretrained = str(tmp_path / "pretrained.safetensors")
    ckpt_path = str(tmp_path / "last.ckpt")

    def _exists(p):
        if p == pretrained:
            return True
        if p == ckpt_path:
            return False
        return Path(p).exists()

    mock_exists.side_effect = _exists
    mock_load_checkpoint.return_value = {"state_dict": {"some": "weights"}}

    trainer = MagicMock()
    mock_trainer.return_value = trainer

    finetune_fn(
        config=config,
        pretrained_ckpt_path=pretrained,
        seed=123,
        resume_trainer_state=False,
        config_dict={"foo": "bar"},
        out_dir=str(tmp_path),
        ckpt_path=ckpt_path,
    )

    # configure_model called with safetensors state_dict
    config["model"].configure_model.assert_called_once()
    args, kwargs = config["model"].configure_model.call_args
    assert kwargs["state_dict"] == {"some": "weights"}

    # trainer.fit called with ckpt_path=None (first-time finetune)
    trainer.fit.assert_called_once()
    _, k = trainer.fit.call_args
    assert k.get("ckpt_path") is None


@patch("src.tasks.load_checkpoint")
@patch("src.tasks.Trainer")
@patch("src.tasks.os.path.exists")
def test_finetune_with_ckpt_loads_and_maybe_resumes(mock_exists, mock_trainer, mock_load_checkpoint, tmp_path):
    config = _make_min_config()
    pretrained = str(tmp_path / "pretrained.ckpt")
    ckpt_path = str(tmp_path / "last.ckpt")

    # Pretrained exists, target ckpt does not

    def _exists(p):
        if p == pretrained:
            return True
        if p == ckpt_path:
            return False
        return Path(p).exists()

    mock_exists.side_effect = _exists
    mock_load_checkpoint.return_value = {"state_dict": {"model.layer": 1}}

    trainer = MagicMock()
    mock_trainer.return_value = trainer

    # Case 1: resume_trainer_state=False -> ckpt_path=None
    finetune_fn(
        config=config,
        pretrained_ckpt_path=pretrained,
        seed=123,
        resume_trainer_state=False,
        config_dict={},
        out_dir=str(tmp_path),
        ckpt_path=ckpt_path,
    )
    config["model"].configure_model.assert_called()
    _, k = trainer.fit.call_args
    assert k.get("ckpt_path") is None

    # Case 2: resume_trainer_state=True -> ckpt_path points to pretrained ckpt
    config2 = _make_min_config()
    trainer2 = MagicMock()
    mock_trainer.return_value = trainer2
    finetune_fn(
        config=config2,
        pretrained_ckpt_path=pretrained,
        seed=123,
        resume_trainer_state=True,
        config_dict={},
        out_dir=str(tmp_path),
        ckpt_path=ckpt_path,
    )
    _, k2 = trainer2.fit.call_args
    assert k2.get("ckpt_path") == pretrained


@patch("src.tasks.load_checkpoint")
@patch("src.tasks.Trainer")
@patch("src.tasks.os.path.exists", return_value=False)
def test_finetune_without_pretrained_starts_from_scratch(mock_exists, mock_trainer, mock_load_checkpoint, tmp_path):
    config = _make_min_config()
    trainer = MagicMock()
    mock_trainer.return_value = trainer

    finetune_fn(
        config=config,
        pretrained_ckpt_path=str(tmp_path / "missing.ckpt"),
        seed=123,
        resume_trainer_state=False,
        config_dict={},
        out_dir=str(tmp_path / "fresh_out"),
        ckpt_path=str(tmp_path / "last.ckpt"),
    )

    # Should call configure_model() without state_dict
    config["model"].configure_model.assert_called_once()
    args, kwargs = config["model"].configure_model.call_args
    assert "state_dict" not in kwargs
