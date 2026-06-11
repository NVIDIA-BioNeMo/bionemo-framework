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

"""This is intended to be a minimal self-container NeMo2 example."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, Type, TypedDict, TypeVar

import torch
from megatron.core import ModelParallelConfig
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from nemo.lightning import io
from nemo.lightning.megatron_parallel import MegatronLossReduction
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import lightning.pytorch as pl
from bionemo.core import BIONEMO_CACHE_DIR
from bionemo.core.data.multi_epoch_dataset import IdentityMultiEpochDatasetWrapper, MultiEpochDatasetResampler
from bionemo.llm.api import MegatronLossType
from bionemo.llm.lightning import LightningPassthroughPredictionMixin
from bionemo.llm.model.config import OVERRIDE_BIONEMO_CONFIG_DEFAULTS, MegatronBioNeMoTrainableModelConfig
from bionemo.llm.utils import iomixin_utils as iom


__all__: Sequence[str] = (
    "BionemoLightningModule",
    "ExampleFineTuneOutput",
    "ExampleModel",
    "ExampleModelOutput",
    "MNISTCustomDataset",
    "MNISTDataModule",
    "MSELossReduction",
    "MnistItem",
    "PretrainConfig",
    "SameSizeLossDict",
    "checkpoint_callback",
    "data_module",
)

#############################################################################################
# Losses: here we define some loss functions. The output of forward happens in parallel
#  and that is where backward happens. def reduce is only used for collecting forward output
#  for inference, as well as for logging.


class SameSizeLossDict(TypedDict):
    """This is the return type for a loss that is computed for the entire batch, where all microbatches are the same size."""

    avg: Tensor


class MnistItem(TypedDict):
    """Training input for the MNIST dataset."""

    data: Tensor
    label: Tensor
    idx: int


class ExampleModelOutput(TypedDict):
    """Output for the example model implementation."""

    x_hat: Tensor
    z: Tensor


class ExampleFineTuneOutput(ExampleModelOutput):
    """Output for the fine-tuned example model implementation."""

    digit_logits: Tensor


class MSELossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: MnistItem, forward_out: Dict[str, Tensor]) -> Tuple[Tensor, SameSizeLossDict]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        x = batch["data"]
        x_hat = forward_out["x_hat"]
        xview = x.view(x.size(0), -1).to(x_hat.dtype)
        loss = nn.functional.mse_loss(x_hat, xview)

        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[SameSizeLossDict]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()


class MSEPlusClassifierLossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: MnistItem, forward_out: ExampleFineTuneOutput) -> Tuple[Tensor, SameSizeLossDict]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        x = batch["data"]
        digits = batch["label"]
        x_hat = forward_out["x_hat"]
        digit_logits = forward_out["digit_logits"]
        xview = x.view(x.size(0), -1).to(x_hat.dtype)
        mse_loss = nn.functional.mse_loss(x_hat, xview)
        classifier_loss = nn.functional.cross_entropy(digit_logits, digits)
        loss = classifier_loss + mse_loss
        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[SameSizeLossDict]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()


class ClassifierLossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: MnistItem, forward_out: Tensor) -> Tuple[Tensor, SameSizeLossDict]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        digits = batch["label"]
        digit_logits = forward_out
        loss = nn.functional.cross_entropy(digit_logits, digits)
        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[SameSizeLossDict]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()


#######################################################################################
# Data methods. The dataset has no changes vs a vanilla pytorch dataset. The data module
#  has a data_sampler in it which is a nemo2 peculiarity. Also the sampler will not
#  shuffle your data! So you need to wrap your dataset in a dataset shuffler that maps
#  sequential ids to random ids in your dataset.


class MNISTCustomDataset(MNIST):
    """A Wrapper for the MNIST Dataset."""

    def __getitem__(self, idx: int) -> MnistItem:
        """Wraps the getitem method of the MNIST dataset such that we return a Dict.

        This is instead of a Tuple or tensor.

        Args:
            idx: The index we want to grab, an int.

        Returns:
            A dict containing the data ("x"), label ("y"), and index ("idx").
        """
        data, label = super().__getitem__(idx)

        return {
            "data": data,
            "label": label,
            "idx": idx,
        }


#######################################################################################
# Data module needs a data_sampler for handling the mcore strategy nemo2 runner.
class MNISTDataModule(pl.LightningDataModule):
    """A Megatron Compatible Data Module for MNIST.

    Attributes:
    data_dir: data directory
    micro_batch_size: batch_size
    global_batch_size: global batch size
    max_len: maximal sequence length for megatron sampler
    rampup_batch_size: ramp up batch size
    num_workers: number of workers
    data_sampler: data_sampler set to be a megatron one
    """

    def __init__(
        self,
        data_dir: str | os.PathLike = str(BIONEMO_CACHE_DIR),
        batch_size: int = 32,
        num_workers: int = 0,
        global_batch_size: int | None = None,
        output_log: bool = True,
    ) -> None:
        """Initialize class.

        Args:
            data_dir: data directory
            batch_size: batch_size
            global_batch_size: global batch size
            num_workers: number of workers
            output_log: whether to output logs

        """
        super().__init__()
        self.data_dir = data_dir
        self.micro_batch_size = batch_size
        self.global_batch_size = global_batch_size or batch_size
        self.max_len = 1048
        # We need to define a "seq_length" for OneLogger, but we just set it to max_len
        self.seq_length = self.max_len
        self.rampup_batch_size = None
        self.num_workers = num_workers
        #  Note that this sampler is sequential, meaning it does not do any shuffling. Let's wrap our data in a shuffler.
        # Wraps the datasampler with the MegatronDataSampler. The MegatronDataSampler is a wrapper that allows the sampler
        # to be used with megatron. It sets up the capability to utilize micro-batching and gradient accumulation. It is also
        # the place where the global batch size is constructed.
        self.data_sampler = MegatronDataSampler(
            seq_len=self.max_len,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
            output_log=output_log,
        )

    def setup(self, stage: str) -> None:
        """Sets up the datasets.

        Args:
            stage: can be one of train / test / predict.
        """
        self.mnist_test = MultiEpochDatasetResampler(
            IdentityMultiEpochDatasetWrapper(
                MNISTCustomDataset(self.data_dir, download=True, transform=transforms.ToTensor(), train=False)
            ),
            seed=43,
            shuffle=False,
        )
        mnist_full = MNISTCustomDataset(self.data_dir, download=True, transform=transforms.ToTensor(), train=True)
        mnist_train, mnist_val = torch.utils.data.random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        self.mnist_train = MultiEpochDatasetResampler(
            IdentityMultiEpochDatasetWrapper(mnist_train), seed=44, shuffle=True
        )

        self.mnist_val = MultiEpochDatasetResampler(
            IdentityMultiEpochDatasetWrapper(mnist_val),
            seed=45,
            shuffle=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the training dataloader."""
        return DataLoader(self.mnist_train, batch_size=self.micro_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader."""
        return DataLoader(self.mnist_val, batch_size=self.micro_batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        """Returns the prediction dataloader."""
        return DataLoader(self.mnist_test, batch_size=self.micro_batch_size, num_workers=self.num_workers)


#########################################################
# Models: These need to be megatron modules. At the most basic level this just means:
#  1. they need a config argument of type ModelParallelConfig
#  2. they need a self.model_type:ModelType enum defined (ModelType.encoder_or_decoder is probably usually fine)
#  3. def set_input_tensor(self, input_tensor) needs to be present. This is used in model parallelism


class ExampleModelTrunk(MegatronModule):
    def __init__(self, config: ModelParallelConfig) -> None:
        """Constructor of the model.

        Args:
            config: The config object is responsible for telling the strategy what model to create.
        """
        super().__init__(config)
        # FIXME add an assertion that the user is not trying to do tensor parallelism since this doesn't use
        #  parallelizable megatron linear layers.
        self.model_type: ModelType = ModelType.encoder_or_decoder
        self.vp_stage = None  # Add vp_stage attribute for compatibility with virtual pipeline
        self.linear1 = nn.Linear(28 * 28, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 3)

    def forward(self, x: Tensor) -> Tensor:
        # we could return a dictionary of strings to tensors here, but let's demonstrate this is not necessary
        x = x.view(x.size(0), -1)
        z = self.linear1(x)
        z = self.relu(z)
        z = self.linear2(z)
        return z

    def set_input_tensor(self, input_tensor: Optional[Tensor]) -> None:
        """This _would_ be needed for model parallel and other kinds of more complicated forward passes in megatron."""
        pass


class ExampleModel(ExampleModelTrunk):
    """An example model."""

    def __init__(self, config: ModelParallelConfig) -> None:
        """Constructor of the model.

        Args:
            config: The config object is responsible for telling the strategy what model to create.
        """
        super().__init__(config)
        self.linear3 = nn.Linear(3, 64)
        self.relu2 = nn.ReLU()
        self.linear4 = nn.Linear(64, 28 * 28)

    def forward(self, x: Tensor) -> ExampleModelOutput:
        """Forward pass of the model.

        Args:
            x: The input data.

        Returns:
            x_hat: The result of the last linear layer of the network.
        """
        z: Tensor = super().forward(x)
        x_hat = self.linear3(z)
        x_hat = self.relu2(x_hat)
        x_hat = self.linear4(x_hat)
        return {"x_hat": x_hat, "z": z}


class ExampleFineTuneBothModel(ExampleModel):
    """Example of taking the example model and adding an output task."""

    def __init__(self, config: ModelParallelConfig):
        super().__init__(config)
        # 10 output digits, and use the latent output layer (z) for making predictions
        self.digit_classifier = nn.Linear(self.linear2.out_features, 10)

    def forward(self, x: Tensor) -> ExampleFineTuneOutput:
        parent_out: ExampleModelOutput = super().forward(x)
        digit_logits = self.digit_classifier(parent_out["z"])
        return {
            "x_hat": parent_out["x_hat"],
            "z": parent_out["z"],
            "digit_logits": digit_logits,
        }


class ExampleFineTuneModel(ExampleModelTrunk):
    """Example of taking the example model and replacing output task."""

    def __init__(self, config: ModelParallelConfig):
        super().__init__(config)
        # 10 output digits, and use the latent output layer (z) for making predictions
        self.digit_classifier = nn.Linear(self.linear2.out_features, 10)

    def forward(self, x: Tensor) -> Tensor:
        z: Tensor = super().forward(x)
        digit_logits = self.digit_classifier(z)  # to demonstrate flexibility, in this case we return a tensor
        return digit_logits


#################################################################################################################
# Model+Loss Configs: these have a configure_model function which allows the megatron strategy to lazily initialize
#  the model after the parallel computing environment has been setup. These also handle loading starting weights
#  for fine-tuning cases. Additionally these configs tell the trainer which loss you want to use with a matched
#  model.


# typevar for capturing subclasses of ExampleModelTrunk. Useful for Generic type hints as below.
ExampleModelT = TypeVar("ExampleModelT", bound=ExampleModelTrunk)


@dataclass
class ExampleGenericConfig(
    Generic[ExampleModelT, MegatronLossType], MegatronBioNeMoTrainableModelConfig[ExampleModelT, MegatronLossType]
):
    """ExampleGenericConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    loss_cls: Type[MegatronLossType] = MSELossReduction  # type: ignore  # this will get overriden by children
    hidden_size: int = 64  # Needs to be set to avoid zero division error in megatron :(
    num_attention_heads: int = 1  # Needs to be set to avoid zero division error in megatron :(
    num_layers: int = 1  # Needs to be set to avoid zero division error in megatron :(
    # IMPORTANT: Since we're adding/overriding the loss_cls, and that's not how we generally track this, we need to
    #   add this into the list of config settings that we do not draw from the loaded checkpoint when restoring.
    override_parent_fields: List[str] = field(default_factory=lambda: OVERRIDE_BIONEMO_CONFIG_DEFAULTS + ["loss_cls"])

    def configure_model(self) -> ExampleModelT:
        """Uses model_cls and loss_cls to configure the model.

        Note: Must pass self into Model since model requires having a config object.

        Returns:
            The model object.
        """
        # 1. first load any settings that may exist in the checkpoint related to the model.
        if self.initial_ckpt_path:
            self.load_settings_from_checkpoint(self.initial_ckpt_path)
        # 2. then initialize the model
        model = self.model_cls(self)
        # 3. Load weights from the checkpoint into the model
        if self.initial_ckpt_path:
            self.update_model_from_checkpoint(model, self.initial_ckpt_path)
        return model

    def get_loss_reduction_class(self) -> Type[MegatronLossType]:
        """Use loss_cls to configure the loss, since we do not change the settings of the loss based on the config."""
        return self.loss_cls


# The configs below simply define which model class to pair with which loss, since the abstractions around getting the
#  model and loss are handled in the ExampleGenericConfig class.
@dataclass
class PretrainConfig(ExampleGenericConfig["ExampleModel", "MSELossReduction"], iom.IOMixinWithGettersSetters):
    """PretrainConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    model_cls: Type[ExampleModel] = ExampleModel
    loss_cls: Type[MSELossReduction] = MSELossReduction


@dataclass
class ExampleFineTuneBothConfig(
    ExampleGenericConfig["ExampleFineTuneBothModel", "MSEPlusClassifierLossReduction"], iom.IOMixinWithGettersSetters
):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    model_cls: Type[ExampleFineTuneBothModel] = ExampleFineTuneBothModel
    loss_cls: Type[MSEPlusClassifierLossReduction] = MSEPlusClassifierLossReduction


@dataclass
class ExampleFineTuneConfig(
    ExampleGenericConfig["ExampleFineTuneConfig", "ClassifierLossReduction"], iom.IOMixinWithGettersSetters
):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    model_cls: Type[ExampleFineTuneModel] = ExampleFineTuneModel
    loss_cls: Type[ClassifierLossReduction] = ClassifierLossReduction


################################################################################
# General training wrapper that can be re-used for all model/loss combos
#  just specify different configs.


class BionemoLightningModule(pl.LightningModule, io.IOMixin, LightningPassthroughPredictionMixin):
    """A very basic lightning module for testing the megatron strategy and the megatron-nemo2-bionemo contract."""

    def __init__(self, config: MegatronBioNeMoTrainableModelConfig):
        """Initializes the model.

        Args:
            config: a Config object necessary to construct the actual nn.Module (the thing that has the parameters).
        """
        super().__init__()
        self.config = config
        self.vp_stage = None  # Add vp_stage attribute for compatibility with virtual pipeline
        self.optim = MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=1e-4,
                optimizer="adam",
                use_distributed_optimizer=True,
                bf16=config.bf16,
                fp16=config.fp16,
                params_dtype=config.params_dtype,
            ),
        )
        # Bind the configure_optimizers method to the model
        self.optim.connect(self)

    def forward(self, batch: Dict, batch_idx: int) -> Any:
        """This forward will be called by the megatron scheduler and it will be wrapped.

        !!! note

            The `training_step` defines the training loop and is independent of the `forward` method here.

        Args:
            batch: A dictionary of data.
            batch_idx: The index of the batch.

        Returns:
            The output of the model.
        """
        x = batch["data"]
        return self.module(x)

    def training_step(self, batch, batch_idx: Optional[int] = None):
        """The training step is where the loss is calculated and the backpropagation is done.

        Background:
        - NeMo's Strategy overrides this method.
        - The strategies' training step will call the forward method of the model.
        - That forward method then calls the wrapped forward step of MegatronParallel which wraps the forward method of the model.
        - That wrapped forward step is then executed inside the Mcore scheduler, which calls the `_forward_step` method from the
            MegatronParallel class.
        - Which then calls the training_step function here.

        In this particular use case, we simply call the forward method of this class, the lightning module.

        Args:
            batch: A dictionary of data. requires `batch_idx` as default None.
            batch_idx: The index of the batch.
        """
        # Forward pass
        predictions = self(batch, batch_idx)

        # Calculate loss using the training loss reduction function
        loss_reduction = self.training_loss_reduction()
        loss_reduction.setup(batch)
        loss = loss_reduction(predictions)

        # Log the training loss
        self.log("train_loss", loss[1]["avg"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return predictions

    def validation_step(self, batch, batch_idx: Optional[int] = None):
        """Alias for forward step at validation."""
        predictions = self(batch, batch_idx)

        # Calculate loss using the validation loss reduction function
        loss_reduction = self.validation_loss_reduction()
        loss_reduction.setup(batch)
        loss = loss_reduction(predictions)
        # Log the validation loss
        self.log(
            "val_loss",
            loss[1]["avg"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return predictions

    def predict_step(self, batch, batch_idx: Optional[int] = None):
        """Alias for forward step at prediction."""
        return self(batch, batch_idx)

    def training_loss_reduction(self) -> MegatronLossReduction:
        """This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss.

        Returns:
        A MegatronLossReduction
        """
        return self.loss_reduction_class()()

    def validation_loss_reduction(self) -> MegatronLossReduction:
        """This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss.

        Returns:
        A MegatronLossReduction
        """
        return self.loss_reduction_class()()

    def test_loss_reduction(self) -> MegatronLossReduction:
        """This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss.

        Returns:
        A MegatronLossReduction
        """
        return self.loss_reduction_class()()

    def configure_model(self) -> None:
        """This configures the model. It is called lazily by the megatron strategy."""
        self.module = self.config.configure_model()
        # Ensure vp_stage attribute is set for compatibility with virtual pipeline
        if not hasattr(self.module, "vp_stage"):
            self.module.vp_stage = None

    def loss_reduction_class(self) -> Type[MegatronLossReduction]:
        """Get the loss reduction class the user has specified in their config."""
        return self.config.get_loss_reduction_class()


"""Training Elements"""
checkpoint_callback = nl_callbacks.ModelCheckpoint(
    save_last=True,
    save_on_train_epoch_end=True,
    monitor="val_loss",
    always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
)

# Set up the data module
data_module = MNISTDataModule(data_dir=str(BIONEMO_CACHE_DIR), batch_size=128)
# metric_tracker = MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"])
