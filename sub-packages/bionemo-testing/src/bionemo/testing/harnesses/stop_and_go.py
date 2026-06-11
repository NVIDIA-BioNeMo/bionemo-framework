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


import pathlib
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Literal, Sequence, Type, TypeVar

import lightning.pytorch as pl
import nemo.lightning as nl
import pytest
from nemo.collections import llm
from nemo.lightning import resume
from nemo.lightning.nemo_logger import NeMoLogger
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.strategies import MegatronStrategy
from nemo.utils import logging

from bionemo.testing import testing_callbacks
from bionemo.testing.harnesses.mode import Mode
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state
from bionemo.testing.torch import recursive_assert_approx_equal


__all__: Sequence[str] = ("CallbackDict", "StopAndGoHarness", "get_callback")


Callback = TypeVar("Callback", bound=pl.Callback)
CallbackDict = Dict[Mode, Dict[Type[pl.Callback], pl.Callback]]


def get_callback(callbacks: CallbackDict, mode: Mode, callback_type: Type[Callback]) -> Callback:
    """Returns the callback with the given name and mode.

    Convenience function to make type hinting easier.

    Args:
        callbacks: The dictionary of callbacks.
        mode: The mode indicating whether to stop or go.
        callback_type: The type of the callback.

    Returns:
        pl.Callback: The callback with the given name and mode.
    """
    return callbacks[mode][callback_type]  # type: ignore


class StopAndGoHarness(ABC):
    """Abstract base class for testing consistency between interrupted and continuous training.

    Users should override cls.setup_model and update cls.setup_class to customize the downstream test cases. Metadata
    are collected through callbacks and users can add new unit tests by comparing the metadata for the interrupted and
    continuous cases.

    By default, learning rate, global step, optimizer state, consumed samples, input and output tensors, and loss are
    compared. Users can add additional metrics by adding new callbacks to `cls.callbacks` and associated test functions.

    Stop and go tests act as follows:
        - setup a clean model for a brief training run, set callbacks to track.
        - interrupt training via the StopAndGoException in the callback Raise.
        - train the model resumed from the checkpoint with the same set of callbacks.
        - train the model continuously without interruption with a new set of the same callbacks.
        - compare each pair of interrupted and continuous callbacks to check for equality.

    Considerations when implementing this class:
        - The derived test name should start with `Test`, and test methods should start with `test_` to enable pytest
          discovery.
        - devices, pipeline_model_parallel, and tensor_model_parallel may impact the setup of DataModule. Certain
            datasets expect a known global batch size, which depends on the number of devices and conditional tensor
            model parallel/ pipeline model parallel settings. By default, we are testing only on single device without
            parallelism.
        - 'mode' is useful in some cases, but not in all cases. Implement conditions based on these when useful. As an
            example, it may be useful to implement a test that stops and resumes.
            - changing callbacks to test metadata integrity (core feature of stop-and-go tests).
            - changing the model construction to use different hyperparameters.
            - ... etc
            Each of the above tests cases may be useful for automated testing of various expected behavior.
        - stop(), resume(), continuous() or collectively run_stop_and_go() are provided methods which execute the actual
          tests, leveraging the conditions in the various setup methods, respecting 'mode' where necessary.

    Attributes:
        root_dir: The root directory.
        val_check_interval: The validation check interval. Stored as an attribute to ensure consistency.
        exp_name: The experiment name.
        extra_metrics_dict: A dictionary of metrics and their corresponding functions.

    See Also: bionemo.testing.callbacks.
    """

    # class variables that need to be overridden
    num_steps: int
    val_check_interval: int
    limit_val_batches: int
    lr: float = 1e-4
    precision: Literal["16-mixed", "bf16-mixed", "32"]
    output_tensor_atol: float = 1e-3  # Absolute tolerance for model precision between output tensors.
    output_tensor_rtol: float = 1e-4  # Relative tolerance for model precision between output tensors.

    # class variables that will be setup in setUpClass
    tempdir: tempfile.TemporaryDirectory
    metadata_dir: pathlib.Path
    exp_name: str
    callbacks: CallbackDict
    nemo_logger: NeMoLogger

    @classmethod
    def setup_class(cls) -> None:
        """Sets up the class by creating a temporary directory, metadata_dir, exp_name and callbacks."""
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.metadata_dir = pathlib.Path(cls.tempdir.name) / "metadata"
        cls.exp_name = cls.__name__

        cls.callbacks = cls.get_default_callbacks()

        cls.nemo_logger = NeMoLogger(
            log_dir=cls.tempdir.name,
            name=cls.exp_name,
            use_datetime_version=False,
            version=None,
            tensorboard=None,
            wandb=None,
            ckpt=None,
        )

    @classmethod
    def teardown_class(cls) -> None:
        """Tears down the class by cleaning up the temporary directory."""
        cls.tempdir.cleanup()

    @classmethod
    @abstractmethod
    def setup_model(cls, mode: Mode) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        """Constructs the model, data, and optimizer for the test harness.

        Optionally supports separate code paths for 'stop'/'resume'/'continuous', although implementors are encouraged
        to use the same code path for both.

        Args:
            mode: The mode indicating whether to stop or go.

        Returns:
            tuple: A tuple containing the model, data, and optimizer.
        """
        raise NotImplementedError()

    @classmethod
    def setup_trainer(
        cls,
        mode: Mode,
    ) -> nl.Trainer:
        """Setup trainer by passing stop, resume, or continuous callbacks according to mode.

        Args:
            mode (Mode): The mode indicating whether to stop, resume, or train continuously.

        Returns:
            (nl.Trainer): NeMo Lightning trainer object.
        """
        strategy = MegatronStrategy(
            ddp="megatron",
            find_unused_parameters=True,
            ckpt_include_optimizer=True,
            ckpt_async_save=False,
        )

        trainer = nl.Trainer(
            devices=1,
            max_steps=cls.num_steps,
            accelerator="gpu",
            strategy=strategy,
            limit_val_batches=cls.limit_val_batches,
            val_check_interval=cls.val_check_interval,
            log_every_n_steps=cls.val_check_interval,
            num_nodes=1,
            callbacks=list(cls.callbacks[mode].values()),
            plugins=nl.MegatronMixedPrecision(precision=cls.precision),
        )
        return trainer

    @classmethod
    def get_default_callbacks(cls) -> CallbackDict:
        """Returns a list of callbacks based on the specified mode. Base implementation provides reasonable defaults.

        To extend this method, call the super and append to the callbacks, depending on which mode you are in:

        ```python
        callbacks = super().get_callbacks()
        callbacks[mode]["MyCustomCallback"] = MyCustomCallback()
        return callbacks
        ```

        Returns:
            A dictionary of callbacks based on the specified mode, each of which maps a callback name to a callback
            object.
        """
        callbacks: CallbackDict = {}

        def make_callbacks() -> Dict[Type[pl.Callback], pl.Callback]:
            return {
                testing_callbacks.LearningRateCallback: testing_callbacks.LearningRateCallback(),
                testing_callbacks.GlobalStepStateCallback: testing_callbacks.GlobalStepStateCallback(),
                testing_callbacks.ConsumedSamplesCallback: testing_callbacks.ConsumedSamplesCallback(),
                testing_callbacks.OptimizerStateCallback: testing_callbacks.OptimizerStateCallback(),
                testing_callbacks.TrainInputCallback: testing_callbacks.TrainInputCallback(),
                testing_callbacks.TrainOutputCallback: testing_callbacks.TrainOutputCallback(),
                testing_callbacks.TrainLossCallback: testing_callbacks.TrainLossCallback(),
                testing_callbacks.ValidInputCallback: testing_callbacks.ValidInputCallback(),
                testing_callbacks.ValidOutputCallback: testing_callbacks.ValidOutputCallback(),
                testing_callbacks.ValidLossCallback: testing_callbacks.ValidLossCallback(),
            }

        interrupted_callbacks = make_callbacks()
        callbacks[Mode.CONTINUOUS] = make_callbacks()

        for mode in [Mode.STOP, Mode.RESUME]:
            consumed_samples_cls = testing_callbacks.TrainValInitConsumedSamplesStopAndGoCallback
            callbacks[mode] = {
                consumed_samples_cls: consumed_samples_cls(mode=mode),
                **interrupted_callbacks,
            }

        callbacks[Mode.STOP].update(
            {
                testing_callbacks.StopAfterValidEpochEndCallback: testing_callbacks.StopAfterValidEpochEndCallback(),
                nl_callbacks.ModelCheckpoint: nl_callbacks.ModelCheckpoint(
                    save_last=True,
                    monitor="val_loss",
                    save_top_k=2,
                    always_save_context=True,
                    filename="{epoch}-{step}-{val_loss:.2f}",
                ),
            }
        )

        return callbacks

    # stop() and resume() are provided methods and run the requisite methods with the appropriate mode.
    @classmethod
    def stop(cls) -> None:
        """Runs pre-training and 'stops' after the first checkpoint is saved.

        This method sets up the model, data, and optimizer for the Mode.STOP mode.
        It then sets up the trainer and strategy for the Mode.STOP mode with the given metrics.
        The training process is executed using the `llm.train` function, passing the model, data, trainer, logger, optimizer, and resume options.
        If a `testing_callbacks.StopAndGoException` is raised during training, it is caught and no action is taken.

        Raises:
            testing_callbacks.StopAndGoException: If a stop and go exception occurs during training.
        """
        logging.info("Running stop()...")

        model, data, opt = cls.setup_model(mode=Mode.STOP)
        trainer = cls.setup_trainer(Mode.STOP)
        with distributed_model_parallel_state():
            llm.train(
                model=model,
                data=data,
                trainer=trainer,
                log=cls.nemo_logger,
                optim=opt,
                resume=resume.AutoResume(
                    resume_if_exists=False,  # Looks for the -last checkpoint to continue training.
                    resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
                ),
            )

    @classmethod
    def resume(cls) -> None:
        """Resumes the model from the checkpoint saved at the end of `stop()` and verifies the metadata integrity."""
        logging.info("Running resume()...")

        model, data, opt = cls.setup_model(mode=Mode.RESUME)
        trainer = cls.setup_trainer(Mode.RESUME)
        with distributed_model_parallel_state():
            llm.train(
                model=model,
                data=data,
                trainer=trainer,
                log=cls.nemo_logger,
                optim=opt,
                resume=resume.AutoResume(
                    resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
                    resume_ignore_no_checkpoint=False,  # When false this will throw an error with no existing checkpoint.
                ),
            )

    @classmethod
    def continuous(cls) -> None:
        """Trains the model in one continuous path without stopping."""
        logging.info("Running continuous()...")

        model, data, opt = cls.setup_model(mode=Mode.CONTINUOUS)
        trainer = cls.setup_trainer(Mode.CONTINUOUS)
        with distributed_model_parallel_state():
            llm.train(model=model, data=data, trainer=trainer, log=cls.nemo_logger, optim=opt)

    @classmethod
    def run_stop_and_go(cls):
        """Executes training both continuously and with a checkpoint interruption."""
        # Interrupted model training
        cls.stop()
        cls.resume()

        # Cleanup and reinitialize the temporary directory so we don't conflict with a previous checkpoint.
        cls.tempdir.cleanup()
        cls.tempdir = tempfile.TemporaryDirectory()

        # Continuous model training.
        cls.continuous()

    @pytest.mark.parametrize(
        "callback_type",
        [
            testing_callbacks.LearningRateCallback,
            testing_callbacks.GlobalStepStateCallback,
            testing_callbacks.ConsumedSamplesCallback,
            testing_callbacks.OptimizerStateCallback,
            testing_callbacks.TrainInputCallback,
            testing_callbacks.TrainOutputCallback,
            testing_callbacks.TrainLossCallback,
            testing_callbacks.ValidInputCallback,
            testing_callbacks.ValidOutputCallback,
            testing_callbacks.ValidLossCallback,
        ],
    )
    def test_stop_and_go_consistency(self, callback_type):
        """Tests the consistency of the callback data between the interrupted and continuous checks."""
        interrupted_callback = get_callback(self.callbacks, Mode.RESUME, callback_type)
        continuous_callback = get_callback(self.callbacks, Mode.CONTINUOUS, callback_type)
        assert interrupted_callback.data, f"No data found for {callback_type}"

        if callback_type in {testing_callbacks.TrainOutputCallback, testing_callbacks.ValidOutputCallback}:
            atol, rtol = self.output_tensor_atol, self.output_tensor_rtol
        else:
            atol, rtol = 1e-4, 1e-4

        recursive_assert_approx_equal(
            interrupted_callback.data,
            continuous_callback.data,
            atol=atol,
            rtol=rtol,
        )

    def test_train_val_init_consumed_samples(self):
        """Tests the initial consumed samples in stop-and-go scenario."""
        train_consumed_stop, val_consumed_stop = get_callback(
            self.callbacks, Mode.STOP, testing_callbacks.TrainValInitConsumedSamplesStopAndGoCallback
        ).data
        train_consumed_go, val_consumed_go = get_callback(
            self.callbacks, Mode.RESUME, testing_callbacks.TrainValInitConsumedSamplesStopAndGoCallback
        ).data

        assert val_consumed_stop == 0
        assert val_consumed_go == 0
        assert train_consumed_stop == 0
        assert train_consumed_go > 0
