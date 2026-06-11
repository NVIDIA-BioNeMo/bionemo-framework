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

import gc
import os

import torch
from lightning.pytorch import Callback


class _FirstBatchCudaSync(Callback):
    # TEMPORARY CALLBACK. Remove once bug is fixed.
    # First batch CUDA sync callback: adds barriers for the first training batch to avoid race condition
    # See https://github.com/NVIDIA/bionemo-framework/issues/1301 for more details.
    def __init__(self):
        self._done = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._done and torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_after_backward(self, trainer, pl_module):
        if not self._done and torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._done and torch.cuda.is_available():
            torch.cuda.synchronize()
            # Unset blocking for subsequent batches
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
            self._done = True


class GarbageCollectAtInferenceTime(Callback):
    """Callback to clean up CUDA memory before validation to prevent initialization errors."""

    def on_validation_start(self, trainer, pl_module) -> None:
        """Clean up CUDA memory before validation to prevent initialization errors."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                current_device = torch.cuda.current_device()
                torch.cuda.set_device(current_device)
                torch.cuda.synchronize()
                gc.collect()
            except Exception as e:
                print(f"Warning: CUDA cleanup failed: {e}")
