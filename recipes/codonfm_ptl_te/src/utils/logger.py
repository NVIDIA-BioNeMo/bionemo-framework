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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import nemo_run as run
import yaml
from lightning.pytorch.loggers import WandbLogger
from nemo_run.core.serialization.yaml import YamlSerializer


log = logging.getLogger(__name__)


def wandb_logger(
    project: str,
    name: str,
    output_dir: str,
    entity: Optional[str] = None,
    offline: bool = False,
    log_model: bool = False,
    tags: List[str] = [],
    config: Optional[Dict] = None,
) -> run.Config[WandbLogger]:
    """Create a configuration for a Weights & Biases logger.

    Args:
        project: The Weights & Biases project name.
        name: The run name to appear in the Weights & Biases UI.
        output_dir: Directory where artifacts and logs are stored.
        entity: Optional Weights & Biases entity (team or user).
        offline: If True, initializes the logger in offline mode.
        log_model: If True, logs model checkpoints to Weights & Biases.
        tags: Optional list of string tags to attach to the run.
        config: Optional configuration dictionary to log as run config.

    Returns:
        A `nemo_run.Config` that constructs a `WandbLogger` when executed.
    """
    cfg = run.Config(
        WandbLogger,
        project=project,
        name=name,
        save_dir=output_dir,
        offline=offline,
        log_model=log_model,
        tags=tags,
        config=config,
    )

    if entity:
        cfg.entity = entity
    return cfg


@dataclass(kw_only=True)
class WandbPlugin(run.Plugin):
    """Plugin to configure Weights & Biases logging for `nemo_run` tasks.
    This plugin attaches a PyTorch Lightning `WandbLogger` to tasks and, when
    enabled, logs the task configuration for reproducibility. The plugin only
    activates if the `WANDB_API_KEY` environment variable is present; the key is
    also propagated to the executor environment.

    Attributes:
        name: Logical task name used for run metadata.
        logger_fn: A `nemo_run.Config` that constructs a `WandbLogger`.
        log_task_config: Whether to log the (partial) task configuration to W&B.
    """  # noqa: D205

    name: str
    logger_fn: run.Config[WandbLogger]
    log_task_config: bool = True

    def setup(self, task: run.Partial | run.Script, executor: run.Executor):
        """Set up the Weights & Biases logger on the provided task.

        If the task is a `run.Script`, the plugin has no effect. When a W&B API
        key is available, attaches the configured logger and optionally logs a
        partial task configuration enriched with executor metadata.

        Args:
            task: The `nemo_run` task to configure.
            executor: The executor that will run the task.
        """
        if isinstance(task, run.Script):
            log.info(
                f"The {self.__class__.__name__} will have no effect on the task as it's an instance of run.Script"
            )
            return

        if "WANDB_API_KEY" in os.environ:
            executor.env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

            if hasattr(task, "log"):
                task.log = self.logger_fn
                if self.log_task_config:
                    partial_config = yaml.safe_load(YamlSerializer().serialize(task))
                    partial_config["experiment"] = {
                        "task_name": self.name,
                        "executor": executor.info(),
                        "remote_directory": (
                            os.path.join(executor.tunnel.job_dir, Path(executor.job_dir).name)
                            if isinstance(executor, run.SlurmExecutor)
                            else None
                        ),
                        "local_directory": executor.job_dir,
                    }
                    task.log.config = partial_config
        else:
            log.info(
                f"The {self.__class__.__name__} will have no effect as WANDB_API_KEY environment variable is not set."
            )
