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

# Copyright The Lightning AI team.
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

import torch
from lightning.pytorch import Callback
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim


class TEVCallback(Callback):
    """Callback for logging TEV statistics before each optimizer step.

    This callback handles different parallelism strategies:
    - Pipeline Parallelism: Only computes on first pipeline stage
    - Tensor Parallelism: Gathers embedding shards across TP ranks
    - Context Parallelism: Gathers across CP ranks
    - Data Parallelism: Only logs on rank 0 of each model parallel group
    """

    @torch.no_grad()
    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        """Called before each optimizer step during training.

        This method calculates and logs Token Embedding Variance (TEV) statistics:
        1. Gets embedding parameter only on pipeline rank 0 (where embeddings live)
        2. Gathers embedding shards across tensor and context parallel ranks
        3. Calculates the token embedding variance (TEV)
        4. Logs the mean and standard deviation of TEV values only on data parallel rank 0

        Args:
            trainer: The Lightning trainer instance
            pl_module: The current Lightning module being trained
            optimizer: The optimizer being used

        Note:
            The callback assumes embeddings live on pipeline rank 0, which is the standard
            configuration in Megatron-LM.
        """
        # Only compute on pipeline rank 0 where embeddings live
        if not parallel_state.is_pipeline_first_stage():
            return

        # Get all named parameters from the model
        named_params = dict(pl_module.named_parameters())

        # Find all parameter keys containing 'embed'
        embed_keys = [key for key in named_params.keys() if "embed" in key]

        # Validate we have exactly one embedding layer
        if len(embed_keys) == 0:
            raise ValueError("No embed keys found.")
        if len(embed_keys) > 1:
            raise ValueError("Multiple embed keys found.")

        # Get the embedding parameter
        embed = named_params[embed_keys[0]]

        # If using tensor parallelism, gather embedding shards
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            embed = _gather_along_last_dim(embed, group=parallel_state.get_tensor_model_parallel_group())

        # If using context parallelism, gather across context parallel ranks
        if parallel_state.get_context_parallel_world_size() > 1:
            world_size = parallel_state.get_context_parallel_world_size()
            dim_size = list(embed.size())
            dim_size[0] = dim_size[0] * world_size

            output = torch.empty(dim_size, dtype=embed.dtype, device=torch.cuda.current_device())
            torch.distributed.all_gather_into_tensor(
                output, embed.contiguous(), group=parallel_state.get_context_parallel_group()
            )
            embed = output

        # Calculate token embedding variance (TEV)
        # First center the embeddings by subtracting the mean
        # Then calculate the mean squared deviation (variance)
        # Finally take the square root to get standard deviation
        tev = torch.sqrt(torch.mean(torch.pow(embed - embed.mean(dim=0), 2), dim=0))

        # Calculate statistics of the TEV values
        tev_mean = torch.mean(tev).item()
        tev_sd = torch.std(tev).item()

        # Only log on data parallel rank 0 to avoid duplicate logging
        if parallel_state.get_data_parallel_rank() == 0:
            # Log the TEV statistics
            pl_module.log("tev_mean", tev_mean, on_step=True, on_epoch=False, sync_dist=False)
            pl_module.log("tev_sd", tev_sd, on_step=True, on_epoch=False, sync_dist=False)
