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

import logging
import os
from pathlib import Path

import hydra
import torch
import transformers
import wandb
from accelerate import PartialState
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from callbacks import StepTimingCallback, StopAfterNStepsCallback
from dataset import create_datasets_and_collator
from metrics import compute_metrics


logger = logging.getLogger(__name__)


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig):
    """Entrypoint."""
    # Initialize Accelerate's distributed state early so torch device is set per process
    state = PartialState()
    logger.info(
        "Accelerate initialized (local_process_index=%s, num_processes=%s, device=%s)",
        state.local_process_index,
        state.num_processes,
        state.device,
    )

    # add wandb logging on main process
    if state.is_main_process:
        wandb.init(config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True), **args.wandb_init_args)

    config = AutoConfig.from_pretrained(args.model_tag, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True, dtype=torch.bfloat16)

    train_dataset, eval_dataset, data_collator = create_datasets_and_collator(**args.dataset)

    training_args = TrainingArguments(**args.trainer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[
            StopAfterNStepsCallback(args.stop_after_n_steps),
            StepTimingCallback(),
        ],
    )

    if training_args.do_train:
        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        last_checkpoint = training_args.resume_from_checkpoint or transformers.trainer_utils.get_last_checkpoint(
            training_args.output_dir
        )
        if last_checkpoint is not None:
            logger.info("Resuming from checkpoint: %s", last_checkpoint)
        else:
            logger.info("No checkpoint found, starting from scratch")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        logger.info("Training complete. Metrics: %s", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_model(str(Path(training_args.output_dir) / "checkpoint-last"))

    if training_args.do_eval:
        trainer.evaluate()

    if wandb.run is not None:
        wandb.finish()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Set required environment variables for distributed training (if not already set) to enable `python train.py` to
    # succeed.
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WANDB_MODE", "offline")
    main()
