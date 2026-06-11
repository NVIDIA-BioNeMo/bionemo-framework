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

import logging
import pathlib

import pytest
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from nemo import lightning as nl
from nemo.lightning.nemo_logger import NeMoLogger

from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


@pytest.fixture
def project_name() -> str:
    return "test_project"


@pytest.fixture
def wandb_config(project_name):
    return WandbConfig(
        entity=None,
        project=project_name,
        tags=["tag1", "tag2"],
        group="test_group",
        job_type="test_job",
        offline=True,  # ensure no actual communication with wandb servers
        id=None,
        anonymous=False,
        log_model=False,
    )


def test_construct_logger_no_loggers():
    logger = setup_nemo_lightning_logger("test")
    assert logger.name == "test"
    assert logger.wandb is None
    assert logger.tensorboard is None


def test_setup_logger_all_loggers(tmp_path, wandb_config, project_name, caplog):
    # Use a unique experiment name
    exp_name = "unit-test-loggers"
    root_dir = tmp_path  # provided by pytest as a temporary directory

    with caplog.at_level(logging.WARNING):
        logger = setup_nemo_lightning_logger(
            name=exp_name,
            root_dir=root_dir,
            initialize_tensorboard_logger=True,
            wandb_config=wandb_config,
            ckpt_callback=None,
        )

    # Checks on the returned logger
    assert isinstance(logger, NeMoLogger), "The returned logger should be a NeMoLogger instance."
    assert logger.name == exp_name

    # Check that directories are set up correctly
    expected_save_dir = root_dir / exp_name
    assert logger.log_dir == str(root_dir), "NeMoLogger save_dir should match expected path."
    # assert not expected_save_dir.exists(), "Expected experiment directory should not be created yet."

    # Check TensorBoard logger initialization
    tb_logger = logger.tensorboard
    assert isinstance(tb_logger, TensorBoardLogger), "TensorBoardLogger should be created."
    tb_log_dir = pathlib.Path(tb_logger.log_dir)
    # assert not tb_log_dir.is_dir(), "TensorBoard log directory should not exist yet."
    assert tb_log_dir == (expected_save_dir / "dev"), "TensorBoardLogger name should match experiment name."

    # Check WandB logger initialization
    wandb_logger = logger.wandb
    assert isinstance(wandb_logger, WandbLogger), "WandBLogger should be created."
    # Validate that wandb_logger uses correct save_dir and name
    # WandbLogger's experiment is lazily created, so just check configured values
    assert wandb_logger.name != exp_name, "WandBLogger name should not match experiment name."
    assert wandb_logger.name == project_name, "WandBLogger name should match project name."
    assert pathlib.Path(wandb_logger.save_dir) == expected_save_dir, "WandBLogger save_dir should match expected path."
    # Since we provided wandb_config and tensorboard was enabled, we should NOT see
    # the warnings about them being turned off.
    assert "WandB is currently turned off." not in caplog.text
    assert "User-set tensorboard is currently turned off." not in caplog.text


def test_nemo_logger_initialized(tmp_path, wandb_config, project_name, caplog):
    # Use a unique experiment name
    exp_name = "unit-test-loggers"
    root_dir = tmp_path  # provided by pytest as a temporary directory
    trainer = nl.Trainer(devices=1, accelerator="gpu", num_nodes=1)

    logger = setup_nemo_lightning_logger(
        name=exp_name,
        root_dir=root_dir,
        initialize_tensorboard_logger=True,
        wandb_config=wandb_config,
        ckpt_callback=None,
    )

    # as in https://github.com/NVIDIA/NeMo/blob/bb895bc4b28ba99d707cb907c4496297a2a7b533/nemo/collections/llm/api.py#L852C22-L856C6
    logger.setup(trainer=trainer)

    # Check that directories are set up correctly
    expected_save_dir = root_dir / exp_name
    assert expected_save_dir.exists(), "Expected experiment directory should not be created yet."

    # Check TensorBoard logger initialization
    tb_logger = logger.tensorboard
    tb_log_dir = pathlib.Path(tb_logger.log_dir)
    # assert not tb_log_dir.is_dir(), "TensorBoard log directory should not exist yet."
    assert tb_log_dir == (expected_save_dir / "dev")

    # Trigger lazy creation of experiment in loggers so loggers have their metadata available
    # following trainer setup at the start of the training in
    # https://github.com/Lightning-AI/pytorch-lightning/blob/de7c28ae865b5c9fd3ff21debebb994605f7f420/src/lightning/pytorch/trainer/trainer.py#L944
    # which executes
    # https://github.com/Lightning-AI/pytorch-lightning/blob/caa9e1e59436913e365bf52eeb2b07e3bf67efac/src/lightning/pytorch/trainer/call.py#L94C1-L97C34
    for _logger in trainer.loggers:
        if hasattr(_logger, "experiment"):
            _ = _logger.experiment


def test_setup_logger_wandb_experiment(tmp_path, wandb_config, project_name, caplog):
    exp_name = "unit-test-loggers"
    root_dir = tmp_path  # provided by pytest as a temporary directory

    logger = setup_nemo_lightning_logger(
        name=exp_name,
        root_dir=root_dir,
        initialize_tensorboard_logger=True,
        wandb_config=wandb_config,
        ckpt_callback=None,
    )
    wandb_logger = logger.wandb
    _ = wandb_logger.experiment
