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
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from megatron.bridge.training.config import OptimizerConfig, OptimizerConfigOverrideProviderContext, SchedulerConfig
from megatron.core.inference.utils import InferenceMode
from megatron.core.optimizer import _get_param_groups, get_standard_config_overrides
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.mlp import MLP

from bionemo.evo2.models.evo2_provider import HyenaNVTestModelProvider, HyenaOptimizerConfigOverrideProvider
from bionemo.evo2.models.megatron.hyena.hyena_block import HyenaStack
from bionemo.evo2.models.megatron.hyena.hyena_layer import HyenaLayer
from bionemo.evo2.models.megatron.hyena.hyena_model import HyenaModel

from .tp_reference import (
    get_tp_reference_hyena_stack_spec,
    merge_strided_column_shards,
    select_strided_column_shard,
)


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


def test_flash_decode_requires_inference_context_when_inference_mode_is_active():
    model = HyenaModel.__new__(HyenaModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(flash_decode=True)

    with (
        InferenceMode.active(),
        pytest.raises(
            AssertionError,
            match="Flash decode is only supported in inference mode, but no inference_context is provided",
        ),
    ):
        model.forward(
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            inference_context=None,
            runtime_gather_output=True,
        )


def test_hyena_stack_does_not_create_full_iteration_manager_for_empty_scope():
    stack = HyenaStack.__new__(HyenaStack)
    torch.nn.Module.__init__(stack)
    stack.config = SimpleNamespace(cuda_graph_scope=[])

    with patch(
        "bionemo.evo2.models.megatron.hyena.hyena_block.CudaGraphManager",
        return_value=object(),
    ):
        stack.create_mcore_cudagraph_manager(stack.config)

    assert not hasattr(stack, "cudagraph_manager")


def test_hyena_stack_uses_cudagraph_manager_config_scope():
    stack = HyenaStack.__new__(HyenaStack)
    torch.nn.Module.__init__(stack)
    stack.config = SimpleNamespace(cuda_graph_scope=[])
    config = SimpleNamespace(cuda_graph_scope=[CudaGraphScope.full_iteration])
    manager = object()

    with patch(
        "bionemo.evo2.models.megatron.hyena.hyena_block.CudaGraphManager",
        return_value=manager,
    ) as manager_cls:
        stack.create_mcore_cudagraph_manager(config)

    assert stack.cudagraph_manager is manager
    manager_cls.assert_called_once_with(config)


def test_hyena_layer_cuda_graph_cache_key_includes_evo2_request_shape():
    layer = HyenaLayer.__new__(HyenaLayer)
    torch.nn.Module.__init__(layer)
    layer.eval()
    layer.config = SimpleNamespace(cuda_graph_impl="local", cuda_graph_scope=[])
    layer.cudagraph_manager = MagicMock(return_value="graph output")
    padded_batch_dimensions = object()
    inference_context = SimpleNamespace(
        evo2_max_batched_decode_requests=4,
        evo2_batched_decode_enabled=True,
        total_request_count=2,
        paused_request_count=0,
        padded_batch_dimensions=padded_batch_dimensions,
        is_static_batching=lambda: False,
        using_cuda_graph_this_step=lambda: True,
    )
    hidden_states = torch.zeros(1)

    output = layer(hidden_states, attention_mask=None, inference_context=inference_context)

    assert output == "graph output"
    layer.cudagraph_manager.assert_called_once_with(
        layer,
        (hidden_states,),
        {"attention_mask": None, "inference_context": inference_context},
        cache_key=(padded_batch_dimensions, 2, True),
    )


def test_tp_reference_stack_uses_test_only_fp32_linears():
    spec = get_tp_reference_hyena_stack_spec()

    hyena_submodules = spec.submodules.hyena_layer.submodules
    attention_submodules = spec.submodules.attention_layer.submodules
    attention_mlp = attention_submodules.mlp
    assert isinstance(attention_mlp, partial)
    assert attention_mlp.func == MLP.as_mlp_submodule
    attention_mlp_submodules = attention_mlp.keywords["submodules"]
    row_linear_modules = (
        hyena_submodules.mixer.submodules.dense,
        hyena_submodules.mlp.submodules.linear_fc2,
        attention_submodules.self_attention.submodules.linear_proj,
        attention_mlp_submodules.linear_fc2,
    )
    column_linear_modules = (
        hyena_submodules.mixer.submodules.dense_projection,
        hyena_submodules.mlp.submodules.linear_fc1,
        attention_submodules.self_attention.submodules.linear_qkv,
        attention_mlp_submodules.linear_fc1,
    )

    assert {module.__name__ for module in row_linear_modules} == {"TpReferenceRowParallelLinear"}
    assert {module.__name__ for module in column_linear_modules} == {"TpReferenceLayerNormColumnParallelLinear"}


@pytest.mark.parametrize(("tp_size", "stride"), [(1, 1), (2, 1), (2, 2), (4, 2)])
def test_strided_column_shards_round_trip(tp_size: int, stride: int):
    """Logical GLU ordering survives TP shard selection and reconstruction."""
    width = tp_size * stride * 3
    full_output = torch.arange(2 * 3 * width).reshape(2, 3, width)

    output_shards = [
        select_strided_column_shard(full_output, tp_rank=tp_rank, tp_size=tp_size, stride=stride)
        for tp_rank in range(tp_size)
    ]
    restored_output = merge_strided_column_shards(
        [shard.movedim(-1, 0) for shard in output_shards],
        stride=stride,
    ).movedim(0, -1)

    assert torch.equal(restored_output, full_output)


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
            config_overrides=get_standard_config_overrides(optimizer_config),
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
