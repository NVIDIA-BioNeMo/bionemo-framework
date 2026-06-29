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


import os
from typing import Any, Dict

from lightning.pytorch import Trainer, seed_everything

from src.utils import RankedLogger
from src.utils.load_checkpoint import load_checkpoint


# Initialize logger at module level so it's available to all functions
logging = RankedLogger(__name__, rank_zero_only=True)


def train(config: Dict[str, Any], ckpt_path: str, seed: int, config_dict: Dict[str, Any], out_dir: str):  # noqa: D417
    """Launches the pre-training process for the Encodon model.

    Args:
        config: A dictionary containing the configuration for the model, data, and trainer.
        ckpt_path: The path to the checkpoint file to resume training from.
        seed: The random seed to use for reproducibility.
    """
    seed_everything(seed, workers=True)

    os.makedirs(out_dir, exist_ok=True)

    logger, data, trainer_kwargs, model, callbacks = (
        config["log"],
        config["data"],
        config["trainer"],
        config["model"],
        config["callbacks"],
    )
    trainer = Trainer(**trainer_kwargs)

    if logger and hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(config_dict)

    if os.path.exists(ckpt_path):
        state_dict = load_checkpoint(ckpt_path, map_location="cpu")
        model.configure_model(state_dict=state_dict.get("state_dict"))
    else:
        model.configure_model()

    trainer.callbacks = list(callbacks.values())
    trainer.logger = logger
    logging.info(f"Starting pre-training from {ckpt_path}")
    trainer.fit(model, datamodule=data, ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None)


def finetune(  # noqa: D417
    config: Dict[str, Any],
    pretrained_ckpt_path: str,
    seed: int,
    resume_trainer_state: bool,
    config_dict: Dict[str, Any],
    out_dir: str,
    ckpt_path: str,
):
    """Launches the fine-tuning process for the Encodon model.

    Args:
        config: A dictionary containing the configuration for the model, data, and trainer.
        ckpt_path: The path to save the fine-tuned model checkpoint.
        pretrained_ckpt_path: The path to the pre-trained model checkpoint to start fine-tuning from.
        seed: The random seed to use for reproducibility.
        resume_trainer_state: Whether to resume the trainer state from the checkpoint.
    """
    seed_everything(seed, workers=True)

    os.makedirs(out_dir, exist_ok=True)

    logger, data, trainer_kwargs, model, callbacks = (
        config["log"],
        config["data"],
        config["trainer"],
        config["model"],
        config["callbacks"],
    )
    trainer = Trainer(**trainer_kwargs)

    if logger and hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(config_dict)
    if os.path.exists(pretrained_ckpt_path) and not os.path.exists(ckpt_path):
        state_dict = load_checkpoint(pretrained_ckpt_path, map_location="cpu")
        model.configure_model(state_dict=state_dict.get("state_dict"))
    else:
        logging.info(f"No pretrained checkpoint found at {pretrained_ckpt_path}, starting from scratch")
        model.configure_model()

    trainer.callbacks = list(callbacks.values())
    trainer.logger = logger

    if resume_trainer_state and os.path.exists(pretrained_ckpt_path) and not os.path.exists(ckpt_path):
        trainer_ckpt_path = pretrained_ckpt_path
    else:
        trainer_ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None

    logging.info(
        f"Starting finetuning from {trainer_ckpt_path} \
        with resume_trainer_state={resume_trainer_state} and ckpt_path={ckpt_path}"
    )
    trainer.fit(model, datamodule=data, ckpt_path=trainer_ckpt_path)


def evaluate(  # noqa: D417
    config: Dict[str, Any],
    config_dict: Dict[str, Any],
    model_ckpt_path: str,
    out_dir: str,
    seed: int = 123,
) -> None:
    """Launches the evaluation process for the Encodon model.

    Args:
        config: A dictionary containing the configuration for the model, data, and trainer.
        config_dict: A dictionary containing the configuration for the model, data, and trainer.

    Note: Evaluation must be run in a single run as resuming the trainer state is not supported for prediction.
    """
    seed_everything(seed, workers=True)
    os.makedirs(out_dir, exist_ok=True)

    logger, data, trainer_kwargs, model, callbacks = (
        config["log"],
        config["data"],
        config["trainer"],
        config["model"],
        config["callbacks"],
    )

    trainer = Trainer(**trainer_kwargs)

    if logger and hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(config_dict)

    model.configure_model()

    data.setup("test")
    if os.path.exists(model_ckpt_path):
        logging.info(f"Loading dataset checkpoint from {model_ckpt_path}")
        data.load_state_dict(load_checkpoint(model_ckpt_path, map_location="cpu"))
        model.prediction_counter = data.init_global_step

    trainer.logger = logger
    trainer.callbacks = list(callbacks.values())

    logging.info("Starting Evaluation!")
    trainer.predict(model, datamodule=data, return_predictions=False)
    return
