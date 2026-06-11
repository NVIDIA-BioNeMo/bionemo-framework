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

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Type, get_args

import yaml
from nemo import lightning as nl

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.dataset import InMemoryProteinDataset
from bionemo.esm2.model.finetune.peft import ESM2LoRA
from bionemo.esm2.model.finetune.sequence_model import ESM2FineTuneSeqConfig
from bionemo.esm2.model.finetune.token_model import ESM2FineTuneTokenConfig
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.biobert.model import BioBertConfig
from bionemo.llm.utils.callbacks import IntervalT, PredictionWriter
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size


__all__: Sequence[str] = ("infer_model",)


SUPPORTED_CONFIGS = {
    "ESM2Config": ESM2Config,
    "ESM2FineTuneSeqConfig": ESM2FineTuneSeqConfig,
    "ESM2FineTuneTokenConfig": ESM2FineTuneTokenConfig,
}


def infer_model(
    data_path: Path,
    checkpoint_path: Path,
    results_path: Path,
    min_seq_length: int = 1024,
    include_hiddens: bool = False,
    include_embeddings: bool = False,
    include_logits: bool = False,
    include_input_ids: bool = False,
    micro_batch_size: int = 64,
    precision: PrecisionTypes = "bf16-mixed",
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    devices: int = 1,
    num_nodes: int = 1,
    prediction_interval: IntervalT = "epoch",
    config_class: Type[BioBertConfig] = ESM2Config,
    lora_checkpoint_path: str | None = None,
    initial_ckpt_skip_keys_with_these_prefixes: list[str] = [],
) -> None:
    """Runs inference on a BioNeMo ESM2 model using PyTorch Lightning.

    Args:
        data_path (Path): Path to the input data.
        checkpoint_path (Path): Path to the model checkpoint.
        results_path (Path): Path to save the inference results.
        min_seq_length (int): minimum sequence length to be padded. This should be at least equal to the length of largest sequence in the dataset
        include_hiddens (bool, optional): Whether to include hidden states in the output. Defaults to False.
        include_embeddings (bool, optional): Whether to include embeddings in the output. Defaults to False.
        include_logits (bool, Optional): Whether to include token logits in the output. Defaults to False.
        include_input_ids (bool, Optional): Whether to include input_ids in the output. Defaults to False.
        micro_batch_size (int, optional): Micro batch size for inference. Defaults to 64.
        precision (PrecisionTypes, optional): Precision type for inference. Defaults to "bf16-mixed".
        tensor_model_parallel_size (int, optional): Tensor model parallel size for distributed inference. Defaults to 1.
        pipeline_model_parallel_size (int, optional): Pipeline model parallel size for distributed inference. Defaults to 1.
        devices (int, optional): Number of devices to use for inference. Defaults to 1.
        num_nodes (int, optional): Number of nodes to use for distributed inference. Defaults to 1.
        prediction_interval (IntervalT, optional): Intervals to write predict method output into disck for DDP inference. Defaults to epoch.
        config_class (Type[BioBertConfig]): The config class for configuring the model using checkpoint provided
        lora_checkpoint_path (Optional[str]): path to the lora checkpoint file.
        initial_ckpt_skip_keys_with_these_prefixes (Optional[List[str]]): List of keys to skip when loading the initial checkpoint. Needed for loading the initial checkpoint from a different model.
    """
    # create the directory to save the inference results
    os.makedirs(results_path, exist_ok=True)

    # Setup the strategy and trainer
    global_batch_size = infer_global_batch_size(
        micro_batch_size=micro_batch_size,
        num_nodes=num_nodes,
        devices=devices,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        ddp="megatron",
        find_unused_parameters=True,
        ckpt_parallel_load=True,
    )

    prediction_writer = PredictionWriter(output_dir=results_path, write_interval=prediction_interval)
    callbacks = [prediction_writer]

    # Setup data
    dataset = InMemoryProteinDataset.from_csv(data_path, ignore_labels=True)
    datamodule = ESM2FineTuneDataModule(
        predict_dataset=dataset,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        min_seq_length=min_seq_length,
    )

    # Setup model
    config = config_class(
        params_dtype=get_autocast_dtype(precision),
        pipeline_dtype=get_autocast_dtype(precision),
        autocast_dtype=get_autocast_dtype(precision),
        include_hiddens=include_hiddens,
        include_embeddings=include_embeddings,
        include_input_ids=include_input_ids,
        skip_logits=not include_logits,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_path=str(checkpoint_path),
    )

    tokenizer = get_tokenizer()

    # Initialize LoRA adapter if needed
    # Initialize base model with or without LoRA

    if lora_checkpoint_path:
        # check that lora checkpoint has a model.yaml file under "context" subfolder and load the yaml file to correctly initialize the
        lora_context_model_path = lora_checkpoint_path / "context" / "model.yaml"
        if not lora_context_model_path.exists():
            raise FileNotFoundError(
                f"LoRA checkpoint {lora_context_model_path} is required to be a valid LoRA checkpoint"
            )
        with open(lora_context_model_path, "r") as f:
            lora_context_model_config = yaml.safe_load(f)["config"]
        if config_class == ESM2FineTuneSeqConfig:
            config.mlp_ft_dropout = lora_context_model_config["mlp_ft_dropout"]
            config.mlp_hidden_size = lora_context_model_config["mlp_hidden_size"]
            config.mlp_target_size = lora_context_model_config["mlp_target_size"]
        elif config_class == ESM2FineTuneTokenConfig:
            config.cnn_dropout = lora_context_model_config["cnn_dropout"]
            config.cnn_hidden_size = lora_context_model_config["cnn_hidden_size"]
            config.cnn_num_classes = lora_context_model_config["cnn_num_classes"]

        if config_class != ESM2Config:
            task_type = lora_context_model_config["task_type"]
            if task_type == "regression":
                initial_ckpt_skip_keys_with_these_prefixes.extend(["regression_head"])
            elif task_type == "classification":
                initial_ckpt_skip_keys_with_these_prefixes.extend(["classification_head"])
            config.task_type = task_type
        peft = ESM2LoRA(peft_ckpt_path=lora_checkpoint_path)
        callbacks.append(peft)
        config.initial_ckpt_skip_keys_with_these_prefixes = initial_ckpt_skip_keys_with_these_prefixes
        module = biobert_lightning_module(config=config, tokenizer=tokenizer, model_transform=peft)
        module.configure_init_model_parallel = True
    else:
        # In this case, the weights of the heads will be in the fine-tuned files and should be read
        # from there as opposed to the base model checkpoint.
        config.initial_ckpt_skip_keys_with_these_prefixes = initial_ckpt_skip_keys_with_these_prefixes
        module = biobert_lightning_module(config=config, tokenizer=tokenizer)

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(precision=precision),
        max_steps=100,
    )

    # Run prediction
    trainer.predict(module, datamodule=datamodule)


def infer_esm2_entrypoint():
    """Entrypoint for running inference on a geneformer checkpoint and data."""
    # 1. get arguments
    parser = get_parser()
    args = parser.parse_args()

    # 2. Call infer with args
    infer_model(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        results_path=args.results_path,
        include_hiddens=args.include_hiddens,
        include_embeddings=args.include_embeddings,
        include_logits=args.include_logits,
        include_input_ids=args.include_input_ids,
        micro_batch_size=args.micro_batch_size,
        precision=args.precision,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        prediction_interval=args.prediction_interval,
        config_class=args.config_class,
        lora_checkpoint_path=args.lora_checkpoint_path,
        initial_ckpt_skip_keys_with_these_prefixes=args.initial_ckpt_skip_keys_with_these_prefixes,
    )


def get_parser():
    """Return the cli parser for this tool."""
    parser = argparse.ArgumentParser(description="Infer ESM2.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to the ESM2 finetuned checkpoint or pretrained checkpoint. If you use LoRA this needs to be the ESM2 pretrained checkpoint.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the CSV file containing sequences and label columns",
    )
    parser.add_argument("--results-path", type=Path, required=True, help="Path to the results directory.")

    parser.add_argument(
        "--precision",
        type=str,
        choices=get_args(PrecisionTypes),
        required=False,
        default="bf16-mixed",
        help="Precision type to use for training.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=False,
        default=1,
        help="Number of GPUs to use for training. Default is 1.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=False,
        default=1,
        help="Number of nodes to use for training. Default is 1.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=2,
        help="Micro-batch size. Global batch size is inferred from this.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Pipeline model parallel size. Default is 1.",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="Tensor model parallel size. Default is 1.",
    )
    parser.add_argument(
        "--prediction-interval",
        type=str,
        required=False,
        choices=get_args(IntervalT),
        default="epoch",
        help="Intervals to write DDP predictions into disk",
    )
    parser.add_argument(
        "--include-hiddens",
        action="store_true",
        default=False,
        help="Include hiddens in output of inference",
    )
    parser.add_argument(
        "--include-input-ids",
        action="store_true",
        default=False,
        help="Include input_ids in output of inference",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        default=False,
        help="Include embeddings in output of inference",
    )
    parser.add_argument(
        "--include-logits", action="store_true", default=False, help="Include per-token logits in output."
    )
    config_class_options: dict[str, Type[BioBertConfig]] = SUPPORTED_CONFIGS

    def config_class_type(desc: str) -> Type[BioBertConfig]:
        try:
            return config_class_options[desc]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"Do not recognize key {desc}, valid options are: {config_class_options.keys()}"
            )

    parser.add_argument(
        "--config-class",
        type=config_class_type,
        default="ESM2Config",
        help="Model configs link model classes with losses, and handle model initialization (including from a prior "
        "checkpoint). This is how you can fine-tune a model. First train with one config class that points to one model "
        "class and loss, then implement and provide an alternative config class that points to a variant of that model "
        "and alternative loss. In the future this script should also provide similar support for picking different data "
        f"modules for fine-tuning with different data types. Choices: {config_class_options.keys()}",
    )
    parser.add_argument(
        "--lora-checkpoint-path",
        type=Path,
        required=False,
        default=None,
        help="Path to the lora states to restore from.",
    )
    parser.add_argument(
        "--initial-ckpt-skip-keys-with-these-prefixes",
        nargs="+",
        required=False,
        default=[],
        help="List of keys to skip when loading the initial checkpoint. Needed for loading the model weights from a different checkpoint.",
    )

    return parser


if __name__ == "__main__":
    infer_esm2_entrypoint()
