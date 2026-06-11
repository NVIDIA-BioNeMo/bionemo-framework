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


from dataclasses import dataclass, field
from typing import List, Literal, Sequence, Type

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from bionemo.esm2.api import ESM2GenericConfig, ESM2Model
from bionemo.esm2.model.finetune.loss import ClassifierLossReduction
from bionemo.llm.model.biobert.model import BioBertOutput
from bionemo.llm.utils import iomixin_utils as iom


"""This package demonstrates how you can take a pretrained ESM2 module and fine-tune the classifier
token to output secondary structure predictions.
"""

__all__: Sequence[str] = (
    "ESM2FineTuneTokenConfig",
    "ESM2FineTuneTokenModel",
    "MegatronConvNetHead",
)


class MegatronConvNetHead(MegatronModule):
    """A convolutional neural network class for residue-level classification."""

    def __init__(self, config: TransformerConfig):
        """Constructor."""
        super().__init__(config)

        self.finetune_model = torch.nn.Sequential(
            torch.nn.Conv2d(config.hidden_size, config.cnn_hidden_size, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(config.cnn_dropout),
        )
        # class_heads (torch.nn.ModuleList): A list of convolutional layers, each corresponding to a different class head.
        # These are used for producing logits scores of varying sizes as specified in `output_sizes`.
        self.class_heads = torch.nn.Conv2d(32, config.cnn_num_classes, kernel_size=(7, 1), padding=(3, 0))

    def forward(self, hidden_states: Tensor) -> List[Tensor]:
        """Inference."""
        # [b, s, h] -> [b, h, s, 1]
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(dim=-1)
        hidden_states = self.finetune_model(hidden_states)  # [b, 32, s, 1]
        output = self.class_heads(hidden_states).squeeze(dim=-1).permute(0, 2, 1)  # [b, s, output_size]
        return output


class ESM2FineTuneTokenModel(ESM2Model):
    """An ESM2 model that is suitable for fine tuning."""

    def __init__(self, config, *args, include_hiddens: bool = False, post_process: bool = True, **kwargs):
        """Constructor."""
        super().__init__(config, *args, include_hiddens=True, post_process=post_process, **kwargs)

        # freeze encoder parameters
        if config.encoder_frozen:
            for _, param in self.named_parameters():
                param.requires_grad = False

        self.include_hiddens_finetuning = (
            include_hiddens  # this include_hiddens is for the final output of fine-tuning
        )
        # If post_process is True that means that we are at the last megatron parallelism stage and we can
        #   apply the head.
        if post_process:
            self.task_type = config.task_type
            # if we are doing post process (eg pipeline last stage) then we need to add the output layers
            self.head_name = f"{self.task_type}_head"  # Example: 'regression_head' or 'classification_head'
            setattr(self, self.head_name, MegatronConvNetHead(config))

    def forward(self, *args, **kwargs) -> Tensor | BioBertOutput:
        """Inference."""
        output = super().forward(*args, **kwargs)
        # Stop early if we are not in post_process mode (for example if we are in the middle of model parallelism)
        if not self.post_process:
            return output  # we are not at the last pipeline stage so just return what the parent has
        # Double check that the output from the parent has everything we need to do prediction in this head.
        if not isinstance(output, dict) or "hidden_states" not in output:
            raise ValueError(
                f"Expected to find 'hidden_states' in the output, and output to be dictionary-like, found {output},\n"
                "Make sure include_hiddens=True in the call to super().__init__"
            )
        # Get the hidden state from the parent output, and pull out the [CLS] token for this task
        hidden_states: Tensor = output["hidden_states"]
        # Predict our 1d regression target
        task_head = getattr(self, self.head_name)
        output[f"{self.task_type}_output"] = task_head(hidden_states)
        if not self.include_hiddens_finetuning:
            del output["hidden_states"]
        return output


@dataclass
class ESM2FineTuneTokenConfig(
    ESM2GenericConfig[ESM2FineTuneTokenModel, ClassifierLossReduction], iom.IOMixinWithGettersSetters
):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    model_cls: Type[ESM2FineTuneTokenModel] = ESM2FineTuneTokenModel
    # typical case is fine-tune the base biobert that doesn't have this head. If you are instead loading a checkpoint
    # that has this new head and want to keep using these weights, please drop this next line or set to []
    initial_ckpt_skip_keys_with_these_prefixes: List[str] = field(default_factory=lambda: ["classification_head"])

    task_type: Literal["classification", "regression"] = "classification"
    encoder_frozen: bool = True  # freeze encoder parameters
    cnn_num_classes: int = 3  # number of classes in each label
    cnn_dropout: float = 0.25
    cnn_hidden_size: int = 32  # The number of output channels in the bottleneck layer of the convolution.

    def get_loss_reduction_class(self) -> Type[ClassifierLossReduction]:
        """The loss function type."""
        return ClassifierLossReduction
