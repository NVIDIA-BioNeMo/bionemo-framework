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


import os
import signal
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from nemo.lightning import io
from nemo.lightning.data import MegatronPretrainingSampler
from nemo.lightning.megatron_parallel import CallbackMethods, DataT, MegatronLossReduction, MegatronStep
from overrides import override

from bionemo.testing.harnesses.mode import Mode
from bionemo.testing.torch import recursive_detach


class StopAfterValidEpochEndCallback(Callback, CallbackMethods):
    """A callback that stops training after the validation epoch.

    Use this callback for pytest based Stop and go tests.
    """

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        if trainer.sanity_checking:
            return
        trainer.should_stop = True


class SignalAfterGivenStepCallback(Callback, CallbackMethods):
    """A callback that emits a given signal to the current process at the defined step.

    Use this callback for pytest based Stop and go tests.
    """

    def __init__(
        self,
        stop_step: int,
        signal_: signal.Signals = signal.SIGUSR2,
        use_trainer_should_stop: bool = False,
        stop_before_step: bool = False,
    ):
        """Initializes the callback with the given stop_step."""
        # Note that the stop step will be one less than the requested step if stop_before_step is True.
        #  this is because the first step is 0 so you get i+1 steps normally.
        if stop_before_step:
            self.stop_step = stop_step - 1
        else:
            self.stop_step = stop_step
        self.signal = signal_
        # If True, ask the trainer to stop by setting should_stop to True rather than emitting a kill signal.
        self.use_trainer_should_stop = use_trainer_should_stop

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        """Stop training if the global step is greater than or equal to the stop_step."""
        if step.trainer.global_step >= self.stop_step:
            if self.use_trainer_should_stop:
                # Ask the trainer to stop by setting should_stop to True rather than emitting a kill signal.
                step.trainer.should_stop = True
            else:
                os.kill(os.getpid(), self.signal)
        return step


class BaseInterruptedVsContinuousCallback(Callback, CallbackMethods, io.IOMixin):
    """Base class for serializable stop-and-go callback to compare continuous to interrupted training.

    This class is used by extending a callback and collecting data into the `self.data` attribute. This data is then
    compared between continuous and interrupted training.

    See nemo.lightning.megatron_parallel.CallbackMethods for the available callback methods.
    """

    def __init__(self):
        """Initializes the callback."""
        self.data = []

    def __deepcopy__(self, memo):
        """Don't actually attempt to copy this data when this callback is being serialized."""
        ...


class LearningRateCallback(BaseInterruptedVsContinuousCallback):
    """Stop-and-go callback for learning rate before pausing and after resuming training."""

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        """Get learning rate as metadata."""
        if step.trainer.training:
            self.data.append(np.array(step.trainer.optimizers[0].param_groups[0]["lr"]))
        return step


class GlobalStepStateCallback(BaseInterruptedVsContinuousCallback):
    """Stop-and-go callback for global_step before pausing and after resuming training."""

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        """Get learning rate as metadata."""
        if step.trainer.training:
            self.data.append(np.array(step.trainer.global_step))
        return step


class ConsumedSamplesCallback(BaseInterruptedVsContinuousCallback):
    """Stop-and-go callback to check consumed samples before pausing and after resuming training."""

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        """Get consumed samples as metadata."""
        if step.trainer.training:
            data_sampler = step.trainer.datamodule.data_sampler
            consumed_samples = data_sampler.compute_consumed_samples(
                step.trainer.global_step - step.trainer.datamodule.init_global_step
            )
            self.data.append(np.array(consumed_samples))
        return step


class TrainInputCallback(BaseInterruptedVsContinuousCallback):
    """Collect training input samples for comparison."""

    def on_megatron_microbatch_end(
        self,
        step: MegatronStep,
        batch: DataT,
        forward_callback: "MegatronLossReduction",
        output: Any,
    ) -> None:
        """Get consumed samples as metadata."""
        if step.trainer.training:
            self.data.append(recursive_detach(batch))


class ValidInputCallback(BaseInterruptedVsContinuousCallback):
    """Collect validation input samples for comparison."""

    def on_megatron_microbatch_end(
        self,
        step: MegatronStep,
        batch: DataT,
        forward_callback: "MegatronLossReduction",
        output: Any,
    ) -> None:
        """Get consumed samples as metadata."""
        if step.trainer.validating:
            self.data.append(recursive_detach(batch))


class TrainOutputCallback(BaseInterruptedVsContinuousCallback):
    """Collect training output samples for comparison."""

    def on_megatron_microbatch_end(
        self,
        step: MegatronStep,
        batch: DataT,
        forward_callback: "MegatronLossReduction",
        output: Any,
    ) -> None:
        """Get consumed samples as metadata."""
        if step.trainer.training:
            self.data.append(recursive_detach(output))


class ValidOutputCallback(BaseInterruptedVsContinuousCallback):
    """Collect validation output samples for comparison."""

    def on_megatron_microbatch_end(
        self,
        step: MegatronStep,
        batch: DataT,
        forward_callback: "MegatronLossReduction",
        output: Any,
    ) -> None:
        """Get consumed samples as metadata."""
        if step.trainer.validating:
            self.data.append(recursive_detach(output))


class TrainLossCallback(BaseInterruptedVsContinuousCallback):
    """Collect training loss samples for comparison."""

    def on_megatron_step_end(
        self,
        step: MegatronStep,
        microbatch_outputs: List[Any],
        reduced: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """Get consumed samples as metadata."""
        if step.trainer.training:
            self.data.append(recursive_detach(reduced))


class ValidLossCallback(BaseInterruptedVsContinuousCallback):
    """Collect training loss samples for comparison."""

    def on_megatron_step_end(
        self,
        step: MegatronStep,
        microbatch_outputs: List[Any],
        reduced: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """Get consumed samples as metadata."""
        if step.trainer.validating:
            self.data.append(recursive_detach(reduced))


class OptimizerStateCallback(BaseInterruptedVsContinuousCallback):
    """Stop-and-go callback to check optimizer states before pausing and after resuming training."""

    def on_megatron_step_start(self, step: MegatronStep) -> MegatronStep:
        """Get optimizer states as metadata."""
        if step.trainer.training:
            self.data.append(
                recursive_detach(
                    [
                        optimizer.mcore_optimizer.optimizer.state_dict()["state"]
                        for optimizer in step.trainer.optimizers
                    ]
                )
            )
        return step


class AbstractStopAndGoCallback(ABC, BaseInterruptedVsContinuousCallback):
    """Abstract base class for stop-and-go callback to compare metadata before pausing and after resuming training.

    This base class provides utility methods to help streamline stop and go comparison.

    Provided methods:
        - __init__: initializes the callback with the given mode.
        - get_metadata: abstract method that should be overridden to get metadata from the trainer and pl_module.

    Default behaviors:
        - in stop mode, metadata is gotten and compared on_validation_epoch_end.
        - in go mode, metadata is gotten and saved on_train_epoch_start.

    Override these behaviors if necessary.
    """

    def __init__(self, mode: Mode = Mode.STOP):
        """Initialize StopAndGoCallback.

        Args:
            mode (str, optional): Mode to run in. Must be either Mode.STOP or Mode.RESUME. Defaults to Mode.STOP.

        Notes:
            User must override get_metadata to get metadata from the trainer and pl_module.
        """
        if mode not in [Mode.STOP, Mode.RESUME]:
            raise ValueError(f"mode must be 'stop' or 'go', got {mode}")
        self.mode = mode
        super().__init__()

    @abstractmethod
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> Any:
        """Get metadata from trainer and pl_module."""
        raise NotImplementedError

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        if self.mode == Mode.RESUME:
            self.data = self.get_metadata(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):  # noqa: D102
        if not trainer.sanity_checking and self.mode == Mode.STOP:
            self.data = self.get_metadata(trainer, pl_module)


class TrainValInitConsumedSamplesStopAndGoCallback(AbstractStopAndGoCallback):
    """Stop-and-go callback to check consumed samples before pausing and after resuming training.

    This is currently the only callback that doesn't fit with the new pattern of directly comparing continuous and
    interrupted training, since the dataloaders don't track their consumed_samples before and after checkpoint
    resumption.
    """

    @override
    def get_metadata(self, trainer: Trainer, pl_module: LightningModule) -> Any:
        """Get consumed samples as metadata."""
        # return trainer.datamodule.state_dict()["consumed_samples"]  # TODO why state_dict can be empty despite working lines below
        train_data_sampler: MegatronPretrainingSampler = trainer.train_dataloader.batch_sampler
        val_data_sampler: MegatronPretrainingSampler = trainer.val_dataloaders.batch_sampler
        return train_data_sampler.consumed_samples, val_data_sampler.consumed_samples
