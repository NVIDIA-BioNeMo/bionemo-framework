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
from megatron.core.transformer.enums import CudaGraphScope, InferenceCudaGraphScope
from megatron.core.transformer.mlp import MLP
from megatron.core.utils import WrappedTensor

from bionemo.evo2.models.evo2_provider import HyenaNVTestModelProvider, HyenaOptimizerConfigOverrideProvider
from bionemo.evo2.models.megatron.hyena.hyena_block import HyenaStack
from bionemo.evo2.models.megatron.hyena.hyena_layer import HyenaLayer
from bionemo.evo2.models.megatron.hyena.hyena_model import HyenaModel, _static_sequence_len_offset

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


class _TupleIdentity(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states, None


class _RecordingMixer(_TupleIdentity):
    def __init__(self):
        super().__init__()
        self.packed_seq_params = None

    def forward(self, hidden_states, *, packed_seq_params=None, **kwargs):
        self.packed_seq_params = packed_seq_params
        return super().forward(hidden_states)


class _BiasDropoutIdentity(torch.nn.Module):
    def forward(self, training, fused):
        del training, fused

        def apply(output_with_bias, residual, dropout):
            del residual, dropout
            return output_with_bias[0]

        return apply


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


def test_static_sequence_len_offset_reuses_graph_stable_storage():
    context = SimpleNamespace(sequence_len_offset=7)

    first = _static_sequence_len_offset(context, batch_size=3, device=torch.device("cpu"))
    first_pointer = first.data_ptr()
    context.sequence_len_offset = 11
    second = _static_sequence_len_offset(context, batch_size=3, device=torch.device("cpu"))

    assert second.data_ptr() == first_pointer
    assert second.dtype == torch.int32
    assert second.tolist() == [11, 11, 11]


def test_packed_rope_uses_full_token_count_for_sequence_parallel_input():
    model = HyenaModel.__new__(HyenaModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(multi_latent_attention=False, flash_decode=False, apply_rope_fusion=True)
    model.position_embedding_type = "rope"
    model.decoder = object()
    model.rotary_pos_emb = MagicMock()
    model.rotary_pos_emb.get_rotary_seq_len.return_value = 5
    frequencies = torch.empty(5, 1, 1, 4)
    model.rotary_pos_emb.return_value = frequencies
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    sequence_parallel_input = torch.empty(3, 1, 8, dtype=torch.bfloat16)
    packed_seq_params = SimpleNamespace(qkv_format="thd")
    fused_frequencies = torch.empty(5, 1, 2, 4, dtype=torch.bfloat16)

    with patch(
        "bionemo.evo2.models.megatron.hyena.hyena_model.precompute_packed_rope_cos_sin",
        return_value=fused_frequencies,
    ) as precompute:
        _, rotary_pos_emb, _, _, _ = model._preprocess(
            input_ids=torch.empty(0, dtype=torch.long),
            position_ids=position_ids,
            decoder_input=sequence_parallel_input,
            packed_seq_params=packed_seq_params,
        )

    assert rotary_pos_emb is fused_frequencies
    precompute.assert_called_once_with(
        frequencies,
        position_ids,
        total_tokens=position_ids.numel(),
        dtype=sequence_parallel_input.dtype,
    )


def test_packed_rope_keeps_frequencies_for_unfused_path():
    """Unfused THD RoPE must receive frequencies, not the fused cos/sin table."""
    model = HyenaModel.__new__(HyenaModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(multi_latent_attention=False, flash_decode=False, apply_rope_fusion=False)
    model.position_embedding_type = "rope"
    model.decoder = object()
    model.rotary_pos_emb = MagicMock()
    model.rotary_pos_emb.get_rotary_seq_len.return_value = 5
    frequencies = torch.empty(5, 1, 1, 4)
    model.rotary_pos_emb.return_value = frequencies
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    decoder_input = torch.empty(5, 1, 8, dtype=torch.bfloat16)
    packed_seq_params = SimpleNamespace(qkv_format="thd")

    with patch(
        "bionemo.evo2.models.megatron.hyena.hyena_model.precompute_packed_rope_cos_sin",
    ) as precompute:
        _, rotary_pos_emb, _, _, _ = model._preprocess(
            input_ids=torch.empty(0, dtype=torch.long),
            position_ids=position_ids,
            decoder_input=decoder_input,
            packed_seq_params=packed_seq_params,
        )

    assert rotary_pos_emb is frequencies
    precompute.assert_not_called()


def test_packed_rope_uses_config_dtype_without_pipeline_decoder_input():
    model = HyenaModel.__new__(HyenaModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        multi_latent_attention=False,
        flash_decode=False,
        apply_rope_fusion=True,
        params_dtype=torch.bfloat16,
    )
    model.pre_process = False
    model.position_embedding_type = "rope"
    model.decoder = object()
    model.rotary_pos_emb = MagicMock()
    model.rotary_pos_emb.get_rotary_seq_len.return_value = 5
    frequencies = torch.empty(5, 1, 1, 4)
    model.rotary_pos_emb.return_value = frequencies
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    packed_seq_params = SimpleNamespace(qkv_format="thd")
    fused_frequencies = torch.empty(5, 1, 2, 4, dtype=torch.bfloat16)

    with patch(
        "bionemo.evo2.models.megatron.hyena.hyena_model.precompute_packed_rope_cos_sin",
        return_value=fused_frequencies,
    ) as precompute:
        _, rotary_pos_emb, _, _, _ = model._preprocess(
            input_ids=torch.empty(0, dtype=torch.long),
            position_ids=position_ids,
            decoder_input=None,
            packed_seq_params=packed_seq_params,
        )

    assert rotary_pos_emb is fused_frequencies
    precompute.assert_called_once_with(
        frequencies,
        position_ids,
        total_tokens=position_ids.numel(),
        dtype=torch.bfloat16,
    )


def test_hyena_stack_does_not_create_full_iteration_manager_for_empty_scope():
    stack = HyenaStack.__new__(HyenaStack)
    torch.nn.Module.__init__(stack)
    stack.config = SimpleNamespace(
        cuda_graph_scope=[],
        inference_cuda_graph_scope=InferenceCudaGraphScope.layer,
    )

    with patch(
        "bionemo.evo2.models.megatron.hyena.hyena_block.CudaGraphManager",
        return_value=object(),
    ):
        stack.create_mcore_cudagraph_manager(stack.config)

    assert not hasattr(stack, "cudagraph_manager")


def test_hyena_stack_uses_cudagraph_manager_config_scope():
    stack = HyenaStack.__new__(HyenaStack)
    torch.nn.Module.__init__(stack)
    stack.config = SimpleNamespace(
        cuda_graph_scope=[],
        inference_cuda_graph_scope=InferenceCudaGraphScope.layer,
    )
    config = SimpleNamespace(
        cuda_graph_scope=[CudaGraphScope.full_iteration],
        inference_cuda_graph_scope=InferenceCudaGraphScope.layer,
    )
    manager = object()

    with patch(
        "bionemo.evo2.models.megatron.hyena.hyena_block.CudaGraphManager",
        return_value=manager,
    ) as manager_cls:
        stack.create_mcore_cudagraph_manager(config)

    assert stack.cudagraph_manager is manager
    manager_cls.assert_called_once_with(config)


def test_hyena_stack_uses_block_inference_cudagraph_manager():
    stack = HyenaStack.__new__(HyenaStack)
    torch.nn.Module.__init__(stack)
    config = SimpleNamespace(
        cuda_graph_scope=[],
        inference_cuda_graph_scope=InferenceCudaGraphScope.block,
    )
    stack.config = config
    manager = object()

    with patch(
        "bionemo.evo2.models.megatron.hyena.hyena_block.CudaGraphManager",
        return_value=manager,
    ) as manager_cls:
        stack.create_mcore_cudagraph_manager(config)

    assert stack.cudagraph_manager is manager
    manager_cls.assert_called_once_with(config)


def test_hyena_stack_non_preprocess_forward_honors_explicit_pipeline_input():
    """A graph-bound PP activation must not be replaced by the stack's stale input tensor."""
    stack = HyenaStack.__new__(HyenaStack)
    torch.nn.Module.__init__(stack)
    stack.eval()
    stack.pre_process = False
    stack.input_tensor = torch.zeros(2, 1, 4)
    stack.layers = torch.nn.ModuleList()
    stack.final_norm = None
    stack.config = SimpleNamespace(
        sequence_parallel=False,
        fp8=None,
        fp4=None,
        recompute_granularity=None,
    )
    graph_input = torch.ones(2, 1, 4)

    output = stack.forward(graph_input, attention_mask=None)

    torch.testing.assert_close(output, graph_input)


def test_hyena_stack_block_graph_receives_current_pipeline_input():
    """PP decode replay must copy each newly received activation into its graph input buffer."""
    stack = HyenaStack.__new__(HyenaStack)
    torch.nn.Module.__init__(stack)
    stack.eval()
    stack.pre_process = False
    pipeline_input = torch.ones(2, 1, 4)
    stack.input_tensor = pipeline_input
    stack.config = SimpleNamespace(
        cuda_graph_scope=[],
        inference_cuda_graph_scope=InferenceCudaGraphScope.block,
    )
    stack.cudagraph_manager = MagicMock(return_value=("graph output",))
    inference_context = SimpleNamespace(
        evo2_max_batched_decode_requests=1,
        evo2_batched_decode_enabled=True,
        total_request_count=1,
        paused_request_count=0,
        padded_batch_dimensions=(1, 1),
        is_static_batching=lambda: False,
        is_decode_only=lambda: True,
        using_cuda_graph_this_step=lambda: True,
    )

    output = stack(
        hidden_states=WrappedTensor(None),
        attention_mask=None,
        inference_context=inference_context,
    )

    assert output == "graph output"
    assert stack.cudagraph_manager.call_args.kwargs["cache_key"] == ((1, 1), 1, True)
    assert stack.cudagraph_manager.call_args.args[2]["hidden_states"] is pipeline_input


def test_hyena_stack_block_cuda_graph_cache_key_includes_evo2_request_shape():
    stack = HyenaStack.__new__(HyenaStack)
    torch.nn.Module.__init__(stack)
    stack.eval()
    stack.pre_process = True
    stack.config = SimpleNamespace(
        cuda_graph_scope=[],
        inference_cuda_graph_scope=InferenceCudaGraphScope.block,
    )
    stack.cudagraph_manager = MagicMock(return_value=("graph output",))
    padded_batch_dimensions = object()
    inference_context = SimpleNamespace(
        evo2_max_batched_decode_requests=4,
        evo2_batched_decode_enabled=True,
        total_request_count=2,
        paused_request_count=0,
        padded_batch_dimensions=padded_batch_dimensions,
        is_static_batching=lambda: False,
        is_decode_only=lambda: True,
        using_cuda_graph_this_step=lambda: True,
    )
    hidden_states = torch.zeros(1)

    output = stack(hidden_states=hidden_states, attention_mask=None, inference_context=inference_context)

    assert output == "graph output"
    stack.cudagraph_manager.assert_called_once_with(
        stack,
        (),
        {
            "hidden_states": hidden_states,
            "attention_mask": None,
            "inference_context": inference_context,
            "dynamic_inference_decode_only": True,
        },
        cache_key=(padded_batch_dimensions, 2, True),
    )


def test_hyena_stack_block_cuda_graph_cache_key_includes_static_cache_shape():
    """Static graphs cannot alias contexts with different batch or sequence capacities."""
    stack = HyenaStack.__new__(HyenaStack)
    torch.nn.Module.__init__(stack)
    stack.eval()
    stack.pre_process = True
    stack.config = SimpleNamespace(
        cuda_graph_scope=[],
        inference_cuda_graph_scope=InferenceCudaGraphScope.block,
    )
    stack.cudagraph_manager = MagicMock(return_value=("graph output",))
    inference_context = SimpleNamespace(
        max_batch_size=3,
        max_sequence_length=4096,
        is_static_batching=lambda: True,
        is_decode_only=lambda: True,
    )
    hidden_states = torch.zeros(1)

    output = stack(hidden_states=hidden_states, attention_mask=None, inference_context=inference_context)

    assert output == "graph output"
    assert stack.cudagraph_manager.call_args.kwargs["cache_key"] == ("evo2-static", 3, 4096)


def test_hyena_layer_does_not_create_manager_inside_block_inference_graph():
    layer = HyenaLayer.__new__(HyenaLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        cuda_graph_impl="local",
        cuda_graph_scope=[],
        inference_cuda_graph_scope=InferenceCudaGraphScope.block,
    )

    with patch(
        "bionemo.evo2.models.megatron.hyena.hyena_layer.CudaGraphManager",
        return_value=object(),
    ):
        layer.create_mcore_cudagraph_manager(layer.config)

    assert not hasattr(layer, "cudagraph_manager")


def test_hyena_layer_cuda_graph_cache_key_includes_evo2_request_shape():
    layer = HyenaLayer.__new__(HyenaLayer)
    torch.nn.Module.__init__(layer)
    layer.eval()
    layer.config = SimpleNamespace(
        cuda_graph_impl="local",
        cuda_graph_scope=[],
        inference_cuda_graph_scope=InferenceCudaGraphScope.layer,
    )
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


def test_hyena_layer_cuda_graph_cache_key_includes_static_cache_shape():
    layer = HyenaLayer.__new__(HyenaLayer)
    torch.nn.Module.__init__(layer)
    layer.eval()
    layer.config = SimpleNamespace(
        cuda_graph_impl="local",
        cuda_graph_scope=[],
        inference_cuda_graph_scope=InferenceCudaGraphScope.layer,
    )
    layer.cudagraph_manager = MagicMock(return_value="graph output")
    inference_context = SimpleNamespace(
        max_batch_size=2,
        max_sequence_length=8192,
        is_static_batching=lambda: True,
        is_decode_only=lambda: True,
    )
    hidden_states = torch.zeros(1)

    output = layer(hidden_states=hidden_states, attention_mask=None, inference_context=inference_context)

    assert output == "graph output"
    assert layer.cudagraph_manager.call_args.kwargs["cache_key"] == ("evo2-static", 2, 8192)


def test_hyena_layer_forwards_packed_sequence_metadata():
    layer = HyenaLayer.__new__(HyenaLayer)
    torch.nn.Module.__init__(layer)
    layer.transformer_config = SimpleNamespace(params_dtype=torch.float32, bias_dropout_fusion=False)
    layer.hidden_dropout = 0.0
    layer.residual_in_fp32 = False
    layer.norm = torch.nn.Identity()
    layer.mixer = _RecordingMixer()
    layer.hyena_bda = _BiasDropoutIdentity()
    layer.pre_mlp_layernorm = torch.nn.Identity()
    layer.mlp = _TupleIdentity()
    layer.mlp_bda = _BiasDropoutIdentity()
    packed_seq_params = object()

    hidden_states = torch.randn(4, 1, 8)
    layer.forward(hidden_states, attention_mask=None, packed_seq_params=packed_seq_params)

    assert layer.mixer.packed_seq_params is packed_seq_params


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
