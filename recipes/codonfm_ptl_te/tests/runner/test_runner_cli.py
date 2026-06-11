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

import sys
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("use_te", [False, True])
@patch("src.runner.get_config")
@patch("src.runner.fdl.build")
@patch("src.runner.finetune")
def test_runner_main_finetune_dispatch(mock_finetune, mock_build, mock_get_config, monkeypatch, use_te):
    # Build a fake argv for finetune mode
    argv = [
        "prog",
        "finetune",
        "--exp_name",
        "run1",
        "--data_path",
        "/data",
        "--process_item",
        "codon_sequence",
        "--dataset_name",
        "CodonBertDataset",
        "--model_name",
        "encodon_80m",
        "--out_dir",
        "/out",
        "--checkpoints_dir",
        "/out/ckpts",
        "--checkpoint_path",
        "/pretrained/model.ckpt",
        "--resume_trainer_state",
        "--finetune_strategy",
        "full",
    ]
    if use_te:
        argv.append("--use_transformer_engine")

    monkeypatch.setenv("WANDB_API_KEY", "")  # Ensure branch without W&B
    with patch.object(sys, "argv", argv):
        # Import inside to use patched argv
        import importlib

        mod = importlib.import_module("src.runner")
        # Spy on parser to avoid running the job by setting dryrun
        with patch.object(mod, "get_parser") as mock_get_parser:
            parser = mod.get_parser()
            parser.set_defaults(dryrun=True)
            mock_get_parser.return_value = parser
            mod.main()

    # Ensure config was constructed and finetune selected
    assert mock_get_config.called
    assert mock_build.called
    # In dryrun, finetune should not be called
    mock_finetune.assert_not_called()


@pytest.mark.parametrize("use_te", [False, True])
@patch("src.runner.get_config")
@patch("src.runner.fdl.build")
@patch("src.runner.finetune")
def test_runner_finetune_args_propagation(mock_finetune, mock_build, mock_get_config, monkeypatch, use_te):
    argv = [
        "prog",
        "finetune",
        "--exp_name",
        "run2",
        "--data_path",
        "/data",
        "--process_item",
        "codon_sequence",
        "--dataset_name",
        "CodonBertDataset",
        "--model_name",
        "encodon_80m",
        "--out_dir",
        "/out",
        "--checkpoints_dir",
        "/out/ckpts",
        "--pretrained_ckpt_path",
        "/pretrained/initial.ckpt",
        "--train_batch_size",
        "16",
    ]
    if use_te:
        argv.append("--use_transformer_engine")

    monkeypatch.setenv("WANDB_API_KEY", "")
    with patch.object(sys, "argv", argv):
        import importlib

        mod = importlib.import_module("src.runner")
        # Let it run fully to call finetune (not dryrun)
        mod.main()

    # finetune should be invoked exactly once with keyword args
    assert mock_finetune.called
    _, kwargs = mock_finetune.call_args
    assert "config" in kwargs and "pretrained_ckpt_path" in kwargs and "ckpt_path" in kwargs
    assert kwargs["pretrained_ckpt_path"] is None or isinstance(kwargs["pretrained_ckpt_path"], str)


@pytest.mark.parametrize("use_te", [False, True])
@patch("src.runner.get_config")
@patch("src.runner.fdl.build")
@patch("src.runner.train")
def test_runner_pretrain_dispatch(mock_train, mock_build, mock_get_config, monkeypatch, use_te):
    argv = [
        "prog",
        "pretrain",
        "--exp_name",
        "run_pre",
        "--data_path",
        "/data",
        "--process_item",
        "codon_sequence",
        "--dataset_name",
        "CodonBertDataset",
        "--model_name",
        "encodon_80m",
        "--out_dir",
        "/out",
        "--checkpoints_dir",
        "/out/ckpts",
        "--train_batch_size",
        "16",
    ]
    if use_te:
        argv.append("--use_transformer_engine")

    monkeypatch.setenv("WANDB_API_KEY", "")
    with patch.object(sys, "argv", argv):
        import importlib

        mod = importlib.import_module("src.runner")
        mod.main()

    assert mock_train.called
    _, kwargs = mock_train.call_args
    assert "config" in kwargs and "ckpt_path" in kwargs and kwargs["ckpt_path"].endswith("last.ckpt")


@pytest.mark.parametrize("use_te", [False, True])
@patch("src.runner.get_config")
@patch("src.runner.fdl.build")
@patch("src.runner.evaluate")
def test_runner_eval_dispatch(mock_evaluate, mock_build, mock_get_config, monkeypatch, use_te):
    argv = [
        "prog",
        "eval",
        "--exp_name",
        "run_eval",
        "--data_path",
        "/data",
        "--process_item",
        "codon_sequence",
        "--dataset_name",
        "CodonBertDataset",
        "--model_name",
        "encodon_80m",
        "--out_dir",
        "/out",
        "--checkpoints_dir",
        "/out/ckpts",
        "--checkpoint_path",
        "/ckpt/model.ckpt",
        "--val_batch_size",
        "16",
    ]
    if use_te:
        argv.append("--use_transformer_engine")

    monkeypatch.setenv("WANDB_API_KEY", "")
    with patch.object(sys, "argv", argv):
        import importlib

        mod = importlib.import_module("src.runner")
        mod.main()

    assert mock_evaluate.called
    _, kwargs = mock_evaluate.call_args
    assert kwargs["model_ckpt_path"] == "/ckpt/model.ckpt"


def test_runner_eval_requires_checkpoint(monkeypatch):
    argv = [
        "prog",
        "eval",
        "--exp_name",
        "run_eval",
        "--data_path",
        "/data",
        "--process_item",
        "codon_sequence",
        "--dataset_name",
        "CodonBertDataset",
        "--model_name",
        "encodon_80m",
        "--out_dir",
        "/out",
        "--checkpoints_dir",
        "/out/ckpts",
    ]
    monkeypatch.setenv("WANDB_API_KEY", "")
    with patch.object(sys, "argv", argv):
        import importlib

        mod = importlib.import_module("src.runner")
        with pytest.raises(SystemExit):
            mod.main()


def test_runner_wandb_requires_project_and_entity(monkeypatch):
    argv = [
        "prog",
        "finetune",
        "--exp_name",
        "run_wb",
        "--data_path",
        "/data",
        "--process_item",
        "codon_sequence",
        "--dataset_name",
        "CodonBertDataset",
        "--model_name",
        "encodon_80m",
        "--out_dir",
        "/out",
        "--checkpoints_dir",
        "/out/ckpts",
        "--enable_wandb",
    ]
    monkeypatch.setenv("WANDB_API_KEY", "key")
    with patch.object(sys, "argv", argv):
        import importlib

        mod = importlib.import_module("src.runner")
        with pytest.raises(SystemExit):
            mod.main()


def test_runner_finetune_thd_not_supported_with_te(monkeypatch):
    """Test that using THD (sequence packing) with finetuning raises an error."""
    argv = [
        "prog",
        "finetune",
        "--exp_name",
        "run_thd_error",
        "--data_path",
        "/data",
        "--process_item",
        "codon_sequence",
        "--dataset_name",
        "CodonBertDataset",
        "--model_name",
        "encodon_80m",
        "--out_dir",
        "/out",
        "--checkpoints_dir",
        "/out/ckpts",
        "--pretrained_ckpt_path",
        "/pretrained/model.ckpt",
        "--attn_input_format",
        "thd",
        "--use_transformer_engine",
        "--train_batch_size",
        "16",
        "--val_batch_size",
        "16",
    ]
    monkeypatch.setenv("WANDB_API_KEY", "")
    with patch.object(sys, "argv", argv):
        import importlib

        mod = importlib.import_module("src.runner")
        with pytest.raises(ValueError, match="THD format is not supported for finetuning"):
            mod.main()


def test_runner_pretrain_thd_requires_transformer_engine(monkeypatch):
    """Test that using THD (sequence packing) without transformer engine raises an error."""
    argv = [
        "prog",
        "pretrain",
        "--exp_name",
        "run_thd_no_te_error",
        "--data_path",
        "/data",
        "--process_item",
        "codon_sequence",
        "--dataset_name",
        "CodonBertDataset",
        "--model_name",
        "encodon_80m",
        "--out_dir",
        "/out",
        "--checkpoints_dir",
        "/out/ckpts",
        "--attn_input_format",
        "thd",
        # Note: NOT using --use_transformer_engine
    ]

    monkeypatch.setenv("WANDB_API_KEY", "")
    with patch.object(sys, "argv", argv):
        import importlib

        mod = importlib.import_module("src.runner")
        with pytest.raises(ValueError, match="THD format requires transformer engine"):
            mod.main()
