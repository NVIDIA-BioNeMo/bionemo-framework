# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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
import logging
from pathlib import Path
from typing import Type

from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.collections.llm.gpt.model.hyena import (
    HyenaModel,
)
from nemo.lightning import io, teardown

from bionemo.evo2.models.mamba import MambaModel


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the Evo2 un-sharded (MP1) model checkpoint file, or a Hugging Face model name. Any model "
        "from the Savanna Evo2 family is supported such as 'hf://arcinstitute/savanna_evo2_1b_base'.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory path for the converted model.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["hyena", "mamba", "llama"],
        default="hyena",
        help="Model architecture to use, choose between 'hyena', 'mamba', or 'llama'.",
    )
    return parser.parse_args()


class _OptimizerRemoverBase:
    MODEL_CLS: Type

    """Base class for optimizer remover importers."""

    def __new__(cls, path: str | Path, model_config=None):
        """Creates a new importer instance.

        Args:
            path: Path to the PyTorch model
            model_config: Optional model configuration

        Returns:
            PyTorchHyenaImporter instance
        """
        instance = super().__new__(cls, path)
        if model_config is None:
            model_config = io.load_context(path, subpath="model.config")
        instance.model_config = model_config
        return instance

    def init(self):
        """Initializes a new HyenaModel instance.

        Returns:
            HyenaModel: Initialized model
        """
        return self.MODEL_CLS(self.config, tokenizer=self.tokenizer)

    def get_source_model(self):
        """Returns the source model."""
        model, _ = self.nemo_load(self)
        return model

    def apply(self, output_path: Path, checkpoint_format: str = "torch_dist", **kwargs) -> Path:
        """Applies the model conversion from PyTorch to NeMo format.

        Args:
            output_path: Path to save the converted model
            checkpoint_format: Format for saving checkpoints
            **kwargs: Additional keyword arguments to pass to the nemo_setup and nemo_save methods

        Returns:
            Path: Path to the saved NeMo model
        """
        source = self.get_source_model()

        target = self.init()
        trainer = self.nemo_setup(target, ckpt_async_save=False, save_ckpt_format=checkpoint_format, **kwargs)
        source.to(self.config.params_dtype)
        target.to(self.config.params_dtype)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer, **kwargs)

        logging.info(f"Converted Hyena model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """Converts the state dictionary from source format to target format.

        Args:
            source: Source model state
            target: Target model

        Returns:
            Result of applying state transforms
        """
        mapping = {k: k for k in source.module.state_dict().keys()}
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
        )

    @property
    def tokenizer(self):
        """Gets the tokenizer for the model.

        Returns:
            Tokenizer instance
        """
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        tokenizer = get_nmt_tokenizer(
            library=getattr(self.model_config, "tokenizer_library", "byte-level"),
        )

        return tokenizer

    @property
    def config(self):
        """Gets the model configuration.

        Returns:
            HyenaConfig: Model configuration
        """
        return self.model_config


@io.model_importer(HyenaModel, "pytorch")
class HyenaOptimizerRemover(_OptimizerRemoverBase, io.ModelConnector["HyenaModel", HyenaModel]):
    """Removes the optimizer state from a nemo2 format model checkpoint."""

    MODEL_CLS = HyenaModel


@io.model_importer(GPTModel, "pytorch")
class LlamaOptimizerRemover(_OptimizerRemoverBase, io.ModelConnector["GPTModel", GPTModel]):
    """Removes the optimizer state from a nemo2 format model checkpoint."""

    MODEL_CLS = GPTModel


@io.model_importer(MambaModel, "pytorch")
class MambaOptimizerRemover(_OptimizerRemoverBase, io.ModelConnector["MambaModel", MambaModel]):
    """Removes the optimizer state from a nemo2 format model checkpoint."""

    MODEL_CLS = MambaModel


def main():
    """Convert a PyTorch Evo2 model checkpoint to a NeMo model checkpoint."""
    args = parse_args()
    if args.model_type == "hyena":
        optimizer_remover = HyenaOptimizerRemover(args.model_path)
    elif args.model_type == "mamba":
        optimizer_remover = MambaOptimizerRemover(args.model_path)
    elif args.model_type == "llama":
        optimizer_remover = LlamaOptimizerRemover(args.model_path)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}.")
    optimizer_remover.apply(args.output_dir)


if __name__ == "__main__":
    main()
