# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# sae/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module, ABC):
    """Minimal interface for Sparse Autoencoders.

    Subclasses must implement encode/decode.
    Optionally override loss() and post_step().
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """Initialize the sparse autoencoder with input and hidden dimensions."""
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input -> sparse codes. Shape: (..., hidden_dim)."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Map sparse codes -> reconstruction. Shape: (..., input_dim)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, codes)."""
        codes = self.encode(x)
        recon = self.decode(codes)
        return recon, codes

    def loss(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Loss with sparsity metrics for logging."""
        recon, codes = self(x)
        mse = F.mse_loss(recon, x)

        # Sparsity metrics (for logging, not loss computation)
        l0 = (codes != 0).float().sum(dim=-1).mean()  # avg non-zero activations

        return {
            "total": mse,
            "sparsity": l0,
        }

    def post_step(self) -> None:
        """Optional hook called by Trainer after optimizer.step()."""
        return
