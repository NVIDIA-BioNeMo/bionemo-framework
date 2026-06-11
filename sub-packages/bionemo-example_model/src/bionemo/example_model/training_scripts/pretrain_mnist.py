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


from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import NeMoLogger, resume

from bionemo.example_model.lightning.lightning_basic import (
    BionemoLightningModule,
    PretrainConfig,
    checkpoint_callback,
    data_module,
)


def run_pretrain(name: str, directory_name: str):
    """Run the pretraining step.

    Args:
        name: The experiment name.
        directory_name: The directory to write the output
    Returns:
        str: the path of the trained model.
    """
    # Setup the logger train the model
    save_dir = Path(directory_name) / "pretrain"

    nemo_logger = NeMoLogger(
        log_dir=str(save_dir),
        name=name,
        tensorboard=TensorBoardLogger(save_dir=save_dir, name=name),
        ckpt=checkpoint_callback,
        extra_loggers=[CSVLogger(save_dir / "logs", name=name)],
    )

    # Set up the training module
    lightning_module = BionemoLightningModule(config=PretrainConfig())
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        always_save_context=True,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        limit_val_batches=5,
        val_check_interval=5,
        max_steps=100,
        max_epochs=10,
        num_nodes=1,
        log_every_n_steps=5,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    # This trains the model
    llm.train(
        model=lightning_module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
    return Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))


if __name__ == "__main__":
    directory_name = "sample_models"
    name = "example"
    pretrain_ckpt_dirpath = run_pretrain(name, directory_name)

    print(pretrain_ckpt_dirpath)
