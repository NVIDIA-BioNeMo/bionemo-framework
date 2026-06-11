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


from typing import List, Optional, Sequence, TypedDict

from nemo.lightning.pytorch.optim.lr_scheduler import LRSchedulerModule
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from torch.optim.lr_scheduler import _LRScheduler

from bionemo.llm.model.biobert.model import MegatronBioBertModel


__all__: Sequence[str] = (
    "SchedulerOutput",
    "WarmupAnnealDecayHold",
    "WarmupAnnealDecayHoldScheduler",
)


class SchedulerOutput(TypedDict):
    """Output of the scheduler method."""

    optimizer: MegatronOptimizerModule
    lr_scheduler: dict
    monitor: str


class WarmupAnnealDecayHold(_LRScheduler):
    """Warmup Anneal Decay Hold learning rate scheduler."""

    def __init__(
        self,
        optimizer: MegatronOptimizerModule,
        *,
        warmup_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        max_lr: Optional[float] = None,
        min_lr: float = 4e-5,
        anneal_percentage: float = 0.10,
        last_epoch: int = -1,
    ) -> None:
        """Initializes the WarmupAnnealDecayHold learning rate scheduler.

        Args:
            optimizer: Optimizer to apply the learning rate scheduler.
            warmup_steps (int): Number of steps for the linear warm-up.
            max_steps (int): Total number of training steps.
            max_lr (float): Peak learning rate to be achieved after warm-up.
            min_lr (float): Minimum learning rate.
            anneal_percentage (float): Percentage of the max_lr to hold after decay.
            last_epoch (int): The index of the last epoch.
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.anneal_percentage = anneal_percentage
        self.last_epoch = last_epoch

        for group in optimizer.param_groups:
            group.setdefault("initial_lr", max_lr)

        super(WarmupAnnealDecayHold, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get the learning rate at the current step."""
        step_num = self.last_epoch
        if step_num < self.warmup_steps:
            lr = self.min_lr + (self.max_lr - self.min_lr) * step_num / self.warmup_steps
        else:
            decay_steps = self.max_steps - self.warmup_steps
            lr = self.max_lr * (1 - (step_num - self.warmup_steps) / decay_steps)
            lr = max(lr, self.max_lr * self.anneal_percentage)

        return [lr for _ in self.optimizer.param_groups]


class WarmupAnnealDecayHoldScheduler(LRSchedulerModule):
    """Warmup Policy Learning Rate Scheduler."""

    def __init__(
        self,
        warmup_steps: int = 2000,
        max_steps: int = 500_000,
        max_lr: float = 4e-4,
        min_lr: float = 4e-5,
        anneal_percentage: float = 0.10,
        interval: str = "step",
        frequency: int = 1,
        monitor: str = "val_loss",
    ) -> None:
        """Initializes the WarmupAnnealDecayHoldScheduler."""
        super().__init__()
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.anneal_percentage = anneal_percentage
        self.interval = interval
        self.frequency = frequency
        self.monitor = monitor

    def scheduler(self, model: MegatronBioBertModel, optimizer: MegatronOptimizerModule) -> SchedulerOutput:
        """Returns the scheduler output."""
        lr_scheduler = WarmupAnnealDecayHold(
            optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            anneal_percentage=self.anneal_percentage,
        )
        return {
            "optimizer": optimizer,
            # REQUIRED: The scheduler instance
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                # `interval` is the unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": self.interval,
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": self.frequency,
            },
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.monitor,
        }
