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


from abc import ABC, abstractmethod

from lightning import LightningModule


class BaseInference(LightningModule, ABC):
    """Base class for all inference tasks.
    Provides a standard interface for configuring a model and running predict
    steps. Subclasses must implement `configure_model` and `_predict_step`.

    Args:
        model_path: Path to the model checkpoint.
        task_type: Type of task to perform.
        use_transformer_engine: Whether to use Transformer Engine.
    """  # noqa: D205

    def __init__(  # noqa: D107
        self,
        model_path: str,
        task_type: str,
        use_transformer_engine: bool = False,
        attn_input_format: str = "bshd",
    ):
        super().__init__()
        self.task_type = task_type
        self.model_path = model_path
        self.use_transformer_engine = use_transformer_engine
        self.attn_input_format = attn_input_format
        self.model = None
        self.prediction_counter = 0  # Initialize prediction counter
        self.save_hyperparameters()

    @abstractmethod
    def configure_model(self):
        """Configure the underlying model for inference.
        Must be implemented by subclasses to initialize and load weights.
        """  # noqa: D205
        pass

    def predict_step(self, batch, batch_idx):
        """Perform a prediction step and increment the counter."""
        self.prediction_counter += 1
        return self._predict_step(batch, batch_idx)

    @abstractmethod
    def _predict_step(self, batch, batch_idx):
        """Perform the actual prediction step. Must be implemented by subclasses."""
        pass
