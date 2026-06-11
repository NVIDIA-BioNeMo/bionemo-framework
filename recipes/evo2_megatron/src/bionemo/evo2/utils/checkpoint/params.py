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


from dataclasses import dataclass


@dataclass
class Param:
    """A dataclass representing a parameter in a checkpoint.

    Attributes:
        name (str): The name of the parameter in the checkpoint.
        partition_dim (int): The dimension index that gets sharded. `None` for no sharding.
        hidden_dim (int): The hidden dimension index. `None` for no hidden dimension.
    """

    name: str  # Name of the parameter in the checkpoint.
    partition_dim: int  # The dimension index that gets sharded. `None` for no sharding.
    hidden_dim: int  # The hidden dimension index. `None` for no hidden dimension.


EVO2_PARAMS = [
    # Only layer_00.
    Param("word_embeddings.weight", 0, 1),  # torch.Size([64, 8192])
    Param("input_layernorm.weight", None, 0),  # torch.Size([8192])
    Param("post_attention_layernorm.weight", None, 0),  # torch.Size([8192])
    Param("pre_mlp_layernorm.weight", None, 0),  # torch.Size([8192])
    Param("outer_mlp_layernorm.weight", None, 0),  # torch.Size([8192])
    Param("mixer.dense_projection.weight", 0, 1),  # torch.Size([3072, 8192]),
    Param("mixer.hyena_proj_conv.short_conv_weight", 0, None),  # torch.Size([3072, 3]),
    Param("mixer.mixer.conv_bias", 0, None),  # torch.Size([1024]),
    Param("mixer.mixer.filter.decay", 0, None),  # torch.Size([64, 8192]),
    Param("mixer.mixer.filter.gamma", 0, None),  # torch.Size([1024, 16]),
    Param("mixer.mixer.filter.h", 0, None),  # torch.Size([64, 8192]),
    Param("mixer.mixer.filter.p", 0, None),  # torch.Size([1024, 16]),
    Param("mixer.mixer.filter.R", 0, None),  # torch.Size([1024, 16]),
    Param("mixer.mixer.filter.t", None, 0),  # torch.Size([1, 1, seqlen]),
    Param("mixer.mixer.short_conv.short_conv_weight", 0, None),  # torch.Size([64, 1, 7]),
    Param("mixer.rotary_emb.inv_freq", None, None),  # torch.Size([64])
    Param("mixer.dense.weight", 1, 0),  # torch.Size([8192, 2048]),
    Param("mixer.dense.bias", None, 0),  # torch.Size([8192])
    Param("mlp.w1.weight", 0, 1),  # torch.Size([2736, 8192]),
    Param("mlp.w2.weight", 0, 1),  # torch.Size([2736, 8192]),
    Param("mlp.w3.weight", 1, 0),  # torch.Size([8192, 2736]),
    # Only last layer.
    Param("norm.weight", None, 0),  # torch.Size([8192]),
]
