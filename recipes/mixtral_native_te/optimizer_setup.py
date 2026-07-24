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

"""Transformer Engine optimizer setup that is intentionally hidden from the training walkthrough."""

import logging
from dataclasses import dataclass, field

import torch
from torch.distributed.tensor import DTensor, distribute_tensor
from transformer_engine.pytorch.optimizers import FusedAdam


logger = logging.getLogger(__name__)


def _local_tensor(param: torch.Tensor) -> torch.Tensor:
    return param._local_tensor if isinstance(param, DTensor) else param


@dataclass
class HighPrecisionInitValues:
    """Bridge TE quantized initialization values across FSDP2 wrapping into FusedAdam.

    Use this only with persistent quantized parameters created by
    ``quantized_model_init(preserve_high_precision_init_val=True)``. TE deliberately keeps the
    pre-quantization values on CPU so an optimizer master can avoid an MXFP8 round trip, but TE
    currently leaves extraction, FSDP sharding, FusedAdam state initialization, and cleanup to the
    caller. FSDP2 may replace the original parameter with a DTensor and lose its dynamically
    attached TE value, so :meth:`capture` runs before wrapping; meta-device initialization instead
    creates the values after wrapping, and :meth:`initialize_master_weights` handles that fallback.

    This class should be removed once TE's FusedAdam can initialize its master weights directly from
    TE's preserved values through FSDP2 DTensors. PyTorch cannot fix the whole issue because the
    getter, clearer, and master-state API are TE-specific, though preserving tensor subclass metadata
    across FSDP parameter replacement would eliminate the pre-wrap capture step.
    """

    values: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def capture(cls, model: torch.nn.Module) -> "HighPrecisionInitValues":
        """Detach CPU master values before FSDP2 may replace their parameter objects."""
        values = {}
        for name, param in model.named_parameters():
            local = _local_tensor(param)
            if not hasattr(local, "get_high_precision_init_val"):
                continue
            value = local.get_high_precision_init_val()
            if value is not None:
                values[name] = value
                local.clear_high_precision_init_val()
        return cls(values)

    def initialize_master_weights(
        self,
        optimizer: FusedAdam,
        model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        """Initialize FusedAdam FP32 masters from captured or still-attached TE values."""
        count = 0
        for name, param in model.named_parameters():
            local = _local_tensor(param)
            value = self.values.pop(name, None)
            if value is None and hasattr(local, "get_high_precision_init_val"):
                value = local.get_high_precision_init_val()
                if value is not None:
                    local.clear_high_precision_init_val()
            if value is None:
                continue

            # This path initializes a full FP32 master for a quantized parameter. The remainder
            # representation requires a BF16 parameter base and must not be combined with this path.
            optimizer.initialize_state(param, store_param_remainders=False)
            value = value.to(device=device)
            if isinstance(param, DTensor):
                value = distribute_tensor(value, param.device_mesh, param.placements).to_local()
            optimizer.set_scaled_state(param, "master_param", value.to(dtype=torch.float32))
            count += 1

        self.values.clear()
        logger.info("Initialized %d master weight(s) from high-precision values", count)
