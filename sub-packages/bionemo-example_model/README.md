# bionemo-example_model

This is a minimalist package containing an example model that makes use of bionemo2 and nemo conventions. It contains the necessary models, dataloaders, datasets, and custom loss functions. The referenced classes and functions are in `bionemo.example_model.lightning.lightning_basic`.

This tutorial demonstrates the creation of a simple MNIST model. This should be run in a BioNeMo container. The BioNeMo Framework container can run in a brev.dev launchable: [![ Click here to deploy.](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdeploynavy.svg)](https://console.brev.dev/launchable/deploy?launchableID=env-2pPDA4sJyTuFf3KsCv5KWRbuVlU). It takes about 10 minutes to deploy this notebook as a Launchable. As of this writing, we are working on a free tier so a credit card may be required. You can reach out to your NVIDIA rep for credit. Notebooks and a shell interface can be launched by clicking `Open Notebook`. (Note: This links to the nightly release and may be out of sync with these docs.)

For this tutorial, we will reuse elements from the BioNeMo example_model package.

`Megatron`/`NeMo` modules and datasets are special derivatives of PyTorch modules and datasets that extend and accelerate the distributed training and inference capabilities of PyTorch.

Some distinctions of Megatron/NeMo are:

- `torch.nn.Module`/`LightningModule` changes into `MegatronModule`.
- Loss functions should extend the `MegatronLossReduction` module and implement a `reduce` method for aggregating loss across multiple micro-batches.
- Megatron configuration classes (for example `megatron.core.transformer.TransformerConfig`) are extended with a `configure_model` method that defines how model weights are initialized and loaded in a way that is compliant with training via NeMo2.
- Various modifications and extensions to common PyTorch classes, such as adding a `MegatronDataSampler` (and re-sampler such as `PRNGResampleDataset` or `MultiEpochDatasetResampler`) to your `LightningDataModule`.

# Loss Functions

First, we define a simple loss function in `bionemo.example_model.lightning.lightning_basic`. These should extend the `MegatronLossReduction` class. The output of forward and backward passes happen in parallel. There should be a forward function that calculates the loss defined. The reduce function is required.

Loss functions used here are `MSELossReduction` and `ClassifierLossReduction`. These functions return a Tensor, which contain the losses for the microbatches, and a `SameSizeLossDict` containing the average loss. This is a Typed Dictionary that is the return type for a loss that is computed for the entire batch, where all microbatches are the same size.

# Datasets and Datamodules

Datasets used for model training must be compatible with Megatron datasets. To enable this, the output of a given index and epoch must be deterministic. However, we may wish to have a different ordering in every epoch. To enable this, the items in the dataset should be accessible by both the epoch and the index. This can be done by accessing elements of the dataset with `EpochIndex` from `bionemo.core.data.multi_epoch_dataset`. A simple way of doing this is to wrap a dataset with `IdentityMultiEpochDatasetWrapper` imported from `bionemo.core.data.multi_epoch_dataset`. In this example, in in `bionemo.example_model.lightning.lightning_basic`, we use a custom dataset `MNISTCustomDataset` that wraps the `__getitem__` method of the MNIST dataset such that it returns a dict instead of a Tuple or tensor. The `MNISTCustomDataset` returns elements of type `MnistItem`, which is a `TypedDict`.

In the data module/data loader class, it is necessary to have a data_sampler attribute to shuffle the data and that allows the sampler to be used with Megatron. This is a nemo2 peculiarity. A `nemo.lightning.pytorch.plugins.MegatronDataSampler` is the best choice. It sets up the capability to utilize micro-batching and gradient accumulation. It is also the place where the global batch size is constructed.

Also the sampler will not shuffle your data. So you need to wrap your dataset in a dataset shuffler that maps sequential IDs to random IDs in your dataset. This can be done with `MultiEpochDatasetResampler` from `bionemo.core.data.multi_epoch_dataset`.

This is implemented in the `MNISTDataModule`. In the setup method of the dataloader, the train, test and validation sets are `MNISTCustomDataset` are wrapped in the `IdentityMultiEpochDatasetWrapper`. These are then wrapped in the `MultiEpochDatasetResampler`. More information about `MegatronCompatability` and how to set up more complicated datasets can be found in [`docs.user-guide.background.megatron_datasets.md`](https://docs.nvidia.com/bionemo-framework/latest/user-guide/background/megatron_datasets/).

We also define a `train_dataloader`, `val_dataloader`, and `predict_dataloader` methods that return the corresponding dataloaders.

# Models

Models need to be Megatron modules. At the most basic level this just means:

1. They extend `MegatronModule` from megatron.core.transformer.module.
2. They need a config argument of type `megatron.core.ModelParallelConfig`. An easy way of implementing this is to inherit from `bionemo.llm.model.config.MegatronBioNeMoTrainableModelConfig`. This is a class for BioNeMo that supports usage with Megatron models, as NeMo2 requires. This class also inherits `ModelParallelConfig`.
3. They need a self.`model_type:megatron.core.transformer.enums.ModelType` enum defined (`ModelType.encoder_or_decoder` is a good option.)
4. `def set_input_tensor(self, input_tensor)` needs to be present. This is used in model parallelism. This function can be a stub/placeholder function.

The following models are implemented in `bionemo.example_model.lightning.lightning_basic`.

`ExampleModelTrunk` is a base model. This returns a tensor. `ExampleModel` is a model that extends the base model with a few linear layers and it is used for pretraining. This returns the output of the base model and of the full model.

`ExampleFineTuneModel` extends the `ExampleModelTrunk` by adding a classification layer. This returns a tensor of logits over the 10 potential digits.

# Model Configs

The model config class is used to instantiate the model. These configs must have:

1. A `configure_model` method which allows the Megatron strategy to lazily initialize the model after the parallel computing environment has been setup. These also handle loading starting weights for fine-tuning cases. Additionally these configs tell the trainer which loss you want to use with a matched model.
2. A `get_loss_reduction_class` method that defines the loss function.

The following configs are implemented in `bionemo.example_model.lightning.lightning_basic`.

Here, a base generic config `ExampleGenericConfig` is defined. `PretrainConfig` extends this class. This defines the model class and the loss class in:

```
class PretrainConfig(ExampleGenericConfig["PretrainModel", "MSELossReduction"], iom.IOMixinWithGettersSetters):

    model_cls: Type[PretrainModel] = PretrainModel
    loss_cls: Type[MSELossReduction] = MSELossReduction

```

Similarly, `ExampleFineTuneConfig` extends `ExampleGenericConfig` for finetuning.

# Training Module

It is helpful to have a training module that inherits from `lightning.pytorch.LightningModule` which organizes the model architecture, training, validation, and testing logic while abstracting away boilerplate code, enabling easier and more scalable training. This wrapper can be used for all model and loss combinations specified in the config.
In `bionemo.example_model.lightning.lightning_basic`, we define `BionemoLightningModule`.

In this example, `training_step`, `validation_step`, and `predict_step` define the training, validation, and prediction loops are independent of the forward method. In nemo:

1. NeMo's Strategy overrides the `train_step`, `validation_step` and `prediction_step` methods.
2. The strategies' training step will call the forward method of the model.
3. That forward method then calls the wrapped forward step of `MegatronParallel` which wraps the forward method of the model.
4. That wrapped forward step is then executed inside the `MegatronCore` scheduler, which calls the `_forward_step` method from the `MegatronParallel` class.
5. Which then calls the `training_step`, `validation_step` and `prediction_step` function here.

Additionally, during these steps, we log the validation, testing, and training loss. This is done similarly to https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html. These logs can then be exported to wandb, or other metric viewers. For more complicated tracking, it may be necessary to use pytorch callbacks: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html.

Further `loss_reduction_class()`, `training_loss_reduction()`, `validation_loss_reduction(),` and` test_loss_reduction()` are defined based on what's in the config. Additionally, `configure_model()` is defined based on the config.

# Training the models

In `bionemo.example_model.lightning.lightning_basic` a checkpoint_callback variable is defined. This enables .nemo file-like checkpointing.

The remaining functions are defined in the training scripts: `pretrain_mnist.py`, `finetune_mnist.py`, and `predict_mnist.py`.

We specify a training strategy of type `nemo.lightning.MegatronStrategy`. This strategy implements model parallelism using NVIDIA's Megatron-LM framework. It supports various forms of parallelism including tensor model parallelism, pipeline model parallelism, sequence parallelism, and expert parallelism for efficient training of large language models.

We specify a trainer of type `nemo.lightning.Trainer`, which is an extension of the pytorch lightning trainer. This is where the devices, validation intervals, maximal steps, maximal number of epochs, and how frequently to log are specified.

We specify a nemo-logger. We can set TensorBoard and WandB logging, along with extra loggers. Here, we specify a `CSVLogger` from lightning.pytorch.loggers.

We can now proceed to training. The first pre-training scripts is `bionemo/example_model/training_scripts/pretrain_mnist.py`

Then, we train the model with the `BionemoLightningModule`, `MNISTDataModule`, trainer and nemo_logger.

This script will print out the location of the final model: \<pretrain_directory>

Then we can run a finetuning-script:

```
python src/bionemo/example_model/training_scripts/finetune_mnist.py ---pretrain_ckpt_dirpath <pretrain_directory>
```

A nuance here is that in the config file, we specify the initial checkpoint path, along with which keys to skip. In the previous model checkpoint, we did not have a head labelled "digit_classifier", so we specify it as a head to be skipped.
This script will print the location of the finetuned directory: \<finetune_dir>.

Finally, we can run a classification task with

```

python src/bionemo/example_model/training_scripts/predict_mnist.py  --finetune_dir <finetune_dir>.
```

The results can be viewed with TensorBoardLogger if that is configured, or as a CSV file created by the `CSVLogger`.
