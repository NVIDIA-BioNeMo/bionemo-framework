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
import contextlib
from unittest.mock import MagicMock, patch

import torch
from megatron.bridge.training.config import OptimizerConfig, OptimizerConfigOverrideProviderContext, SchedulerConfig
from megatron.core.optimizer import _get_param_groups

from bionemo.evo2.models.evo2_provider import HyenaNVTestModelProvider, HyenaOptimizerConfigOverrideProvider


class _FakePGCollection:
    cp = None
    pp = None
    tp = None
    embd = None
    dp = None
    expt_dp = None
    mp = None
    dp_cp = None
    intra_dp_cp = None
    intra_expt_dp = None


@contextlib.contextmanager
def _no_op_context_manager():
    yield


def _mock_all_gather_object(object_list, obj, group=None):
    object_list[:] = [obj]


def test_weight_decay_conditions():
    """Verify that our custom no_weight_decay_cond function is used correctly and changes param groups."""
    with (
        patch("megatron.core.process_groups_config.ProcessGroupCollection.use_mpu_process_groups") as mock_use_mpu,
        patch("megatron.core.tensor_parallel.layers.get_cuda_rng_tracker") as mock_tracker_getter,
        patch("bionemo.evo2.models.megatron.hyena.hyena_utils.get_cuda_rng_tracker") as mock_tracker_getter,
        patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size", return_value=1),
        patch("megatron.core.parallel_state.get_tensor_model_parallel_group", return_value=None),
        patch("megatron.core.parallel_state.get_context_parallel_group", return_value=None),
        patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size", return_value=1),
        patch("megatron.core.parallel_state.get_context_parallel_world_size", return_value=1),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.all_gather_object", side_effect=_mock_all_gather_object),
    ):
        # Mock ProcessGroupCollection
        mock_use_mpu.return_value = _FakePGCollection()

        # Mock get_cuda_rng_tracker().fork()
        mock_tracker = MagicMock()
        mock_tracker.fork.side_effect = _no_op_context_manager
        mock_tracker_getter.return_value = mock_tracker

        config = HyenaNVTestModelProvider(
            vocab_size=256,
            kv_channels=128,
            num_query_groups=1,
            rotary_percent=1.0,
            init_method=torch.nn.init.normal_,
            embedding_init_method=torch.nn.init.normal_,
        )
        config.finalize()
        assert config.init_method is not None
        model = config.provide(pre_process=True, post_process=True)
        optimizer_config_override_provider = HyenaOptimizerConfigOverrideProvider(
            no_weight_decay_embeddings=False,
        )
        optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=1.0,
            weight_decay=1.0,
        )
        scheduler_config = SchedulerConfig(
            lr_decay_style="linear",
            lr_decay_iters=1000,
            lr_decay_samples=1000000,
        )
        hyena_config_overrides = optimizer_config_override_provider.build_config_overrides(
            context=OptimizerConfigOverrideProviderContext(
                model=model,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config,
            )
        )
        param_groups = _get_param_groups(
            model_chunks=[model],
            config=optimizer_config,
            config_overrides=None,  # default config overrides
        )
        param_groups2 = _get_param_groups(
            model_chunks=[model],
            config=optimizer_config,
            config_overrides=hyena_config_overrides,
        )
        assert len(param_groups2) == len(param_groups)
        assert len(param_groups2) == 2
        assert set(param_groups2[0]["params"]) != set(param_groups[0]["params"])
        assert set(param_groups2[1]["params"]) != set(param_groups[1]["params"])
