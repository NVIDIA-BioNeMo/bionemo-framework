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
import math
from typing import Sequence

import torch
import transformer_engine as te
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide


__all__: Sequence[str] = ("ESM2QueryScaling", "TELayerNorm")


class TELayerNorm(te.pytorch.LayerNorm):  # noqa: D101
    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:  # noqa: D417
        """A wrapper around transformer engine layernorm that allows it to be initialized with a TransformerConfig.
            This allows this method to be used in a megatron layerspec.

        Args:
            config (TransformerConfig): The megatron config. This is used for extracing sequence_parallel and zero_centered_gamma.
                The rest of the config is not used.
        """  # noqa: D205
        # Eps tends to get passed through properly, as does hidden_size, but not other params from the config.
        super().__init__(
            *args,
            zero_centered_gamma=config.layernorm_zero_centered_gamma,
            sequence_parallel=config.sequence_parallel,
            **kwargs,
        )


class ESM2QueryScaling(torch.nn.Module):  # noqa: D101
    def __init__(self, config: TransformerConfig, *args, **kwargs) -> None:  # noqa: D417
        """A custom layer that scales quary values.

        This layer should replace the q_layernorm=IdentityOp in ESM2 ModuleSpec to reproduce ESM2
        which apply 1/sqrt(hidden_size_per_attention_head) scaling prior to apply_rotary_pos_emb()

        Args:
            config (TransformerConfig): The megatron config. This is used for computing projection_size
        """
        super().__init__()
        projection_size = config.kv_channels * config.num_attention_heads
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.sqrt_val = math.sqrt(self.hidden_size_per_attention_head)

    @torch.compile
    def forward(self, query, *args, **kwargs):  # noqa: D102
        return query / self.sqrt_val
