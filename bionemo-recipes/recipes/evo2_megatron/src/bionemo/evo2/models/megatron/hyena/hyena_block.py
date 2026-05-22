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

import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Union

import torch
from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import get_cpu_offload_context
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor
from torch import Tensor, nn

from bionemo.evo2.models.megatron.hyena.hyena_config import HyenaConfig
from bionemo.evo2.models.megatron.hyena.hyena_hybrid_layer_allocation import Symbols as LayerSymbols
from bionemo.evo2.models.megatron.hyena.hyena_hybrid_layer_allocation import allocate_layers


try:
    from megatron.core.extensions.transformer_engine import TENorm, te_checkpoint

    HAVE_TE = True
    LayerNormImpl = TENorm

except ImportError:
    HAVE_TE = False

    try:
        from apex.normalization import FusedLayerNorm

        LayerNormImpl = FusedLayerNorm

    except ImportError:
        from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

        LayerNormImpl = WrappedTorchLayerNorm


HYENA_LAYER_MAP = {
    LayerSymbols.HYENA_SHORT: "hyena_short_conv",
    LayerSymbols.HYENA_MEDIUM: "hyena_medium_conv",
    LayerSymbols.HYENA: "hyena",
}


@dataclass
class HyenaStackSubmodules:
    """A class for the module specs for the HyenaStack."""

    hyena_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp


class HyenaStack(GraphableMegatronModule, MegatronModule):
    """A class for the HyenaStack."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        hyena_config: HyenaConfig,
        hybrid_override_pattern,
        max_sequence_length,
        submodules: HyenaStackSubmodules,
        pre_process: bool = True,
        post_process: bool = True,
        post_layer_norm: bool = False,
        pg_collection=None,
    ) -> None:
        """Initialize the HyenaStack."""
        super().__init__(config=transformer_config)

        self.transformer_config = transformer_config
        self.hyena_config = hyena_config
        self.submodules = submodules
        self.hybrid_override_pattern = hybrid_override_pattern
        self.pre_process = pre_process
        self.post_process = post_process
        self.post_layer_norm = post_layer_norm
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        # Required for pipeline parallel schedules
        self.input_tensor = None

        layer_type_list = allocate_layers(self.transformer_config.num_layers, self.hybrid_override_pattern)

        pp_layer_offset = 0
        if self.pg_collection.pp is not None and self.pg_collection.pp.size() > 1:
            pp_layer_offset, layer_type_list = self._select_layers_for_pipeline_parallel(layer_type_list)

        if get_cpu_offload_context is not None:
            # Megatron Core changed this helper from six to seven positional arguments
            # across releases. Pass only the arguments accepted by the installed version.
            offload_args = [
                self.config.cpu_offloading,
                self.config.cpu_offloading_num_layers,
                self.config.num_layers,
                self.config.cpu_offloading_activations,
                self.config.cpu_offloading_weights,
                self.config.cpu_offloading_double_buffering,
                getattr(self.config, "cpu_offloading_retain_pinned_cpu_buffers", False),
            ]
            num_offload_params = len(inspect.signature(get_cpu_offload_context).parameters)
            (self.offload_context, self.group_prefetch_offload_commit_async) = get_cpu_offload_context(
                *offload_args[:num_offload_params],
            )
            self.config._cpu_offloading_context = self.offload_context if self.config.cpu_offloading else None
        else:
            assert self.config.cpu_offloading is False, "CPU Offloading is enabled when TE is not present"

            self.offload_context, self.group_prefetch_offload_commit_async = nullcontext(), None
            self.config._cpu_offloading_context = None

        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(layer_type_list):
            if layer_type in HYENA_LAYER_MAP:
                # Get appropriate quantization context (FP8 and FP4 are mutually exclusive)
                if transformer_config.fp8:
                    quantization_context = get_fp8_context(
                        transformer_config,
                        i + pp_layer_offset,
                        is_init=True,  # 0 based global layer index
                    )
                elif transformer_config.fp4:
                    quantization_context = get_fp4_context(
                        transformer_config,
                        i + pp_layer_offset,
                        is_init=True,  # 0 based global layer index
                    )
                else:
                    quantization_context = nullcontext()

                with quantization_context:
                    layer = build_module(
                        submodules.hyena_layer,
                        self.transformer_config,
                        self.hyena_config,
                        operator_type=HYENA_LAYER_MAP.get(layer_type),
                        max_sequence_length=max_sequence_length,
                        layer_number=i + 1 + pp_layer_offset,
                        pg_collection=self.pg_collection,
                    )
            elif layer_type == LayerSymbols.ATTENTION:
                # Transformer layers apply their own pp_layer_offset
                # Get appropriate quantization context (FP8 and FP4 are mutually exclusive)
                if transformer_config.fp8:
                    quantization_context = get_fp8_context(
                        transformer_config,
                        i + pp_layer_offset,
                        is_init=True,  # 0 based global layer index
                    )
                elif transformer_config.fp4:
                    quantization_context = get_fp4_context(
                        transformer_config,
                        i + pp_layer_offset,
                        is_init=True,  # 0 based global layer index
                    )
                else:
                    quantization_context = nullcontext()

                with quantization_context:
                    layer = build_module(
                        submodules.attention_layer,
                        config=self.transformer_config,
                        layer_number=i + 1,
                        pg_collection=self.pg_collection,
                    )
            else:
                assert True, "unexpected layer_type"
            self.layers.append(layer)
        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_norm = TENorm(
                config=self.transformer_config,
                hidden_size=self.transformer_config.hidden_size,
                eps=self.transformer_config.layernorm_epsilon,
            )
        else:
            # Ensure final_norm is always defined to avoid AttributeError when post_process=False
            self.final_norm = None
        # Required for activation recomputation
        self.num_layers_per_pipeline_rank = len(self.layers)

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func
        """
        self.input_tensor = input_tensor

    def _select_layers_for_pipeline_parallel(self, layer_type_list):
        pipeline_rank = self.pg_collection.pp.rank() if self.pg_collection.pp is not None else 0
        num_layers_per_pipeline_rank = self.transformer_config.num_layers // (
            self.pg_collection.pp.size() if self.pg_collection.pp is not None else 1
        )

        assert getattr(self.transformer_config, "virtual_pipeline_model_parallel_size", None) is None, (
            "The Hyena hybrid model does not currently support virtual/interleaved pipeline parallelism"
        )

        offset = pipeline_rank * num_layers_per_pipeline_rank
        selected_list = layer_type_list[offset : offset + num_layers_per_pipeline_rank]

        return offset, selected_list

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(hidden_states, attention_mask, context, context_mask, rotary_pos_emb):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                        # TODO: check if fp4 is supported in this case
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()
                    with inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`."""
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                upper_layer_idx = min(layer_idx + self.config.recompute_num_layers, self.num_layers_per_pipeline_rank)
                hidden_states, context = checkpoint_handler(custom(layer_idx, upper_layer_idx))
                new_n_layers = upper_layer_idx - layer_idx
                layer_idx += new_n_layers

        elif self.config.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                # TODO: check if fp4 is supported in this case
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states, attention_mask, context, context_mask, rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """Check if we should call the local cudagraph path."""
        if not self.training and (
            hasattr(self, "cudagraph_manager")
            and kwargs["attention_mask"] is None
            and (kwargs.get("inference_context") is not None or kwargs.get("inference_params") is not None)
            and CudaGraphScope.full_iteration in self.config.cuda_graph_scope
        ):
            if kwargs["inference_context"].is_static_batching():
                using_cuda_graph = kwargs["inference_context"].is_decode_only()
            else:
                using_cuda_graph = kwargs["inference_context"].using_cuda_graph_this_step()

            if using_cuda_graph:
                return True
        return False

    def __call__(self, *args, **kwargs):
        """Capture the call to this function and first check whether to call the local cudagraph path."""
        if self._should_call_local_cudagraph(*args, **kwargs):
            kwargs["hidden_states"] = (
                kwargs["hidden_states"].unwrap()
                if isinstance(kwargs["hidden_states"], WrappedTensor)
                else kwargs["hidden_states"]
            )
            # dynamic_inference_decode_only is not a real argument to forward, it is only used
            # to differentiate the cuda graph used for decode from the one used for non-decode
            # inference.
            dynamic_inference_decode_only = kwargs["inference_context"].is_decode_only()
            # cudagraphmanager returns a singleton tuple, whereas the
            # normal forward returns a tensor, therefore we need
            # to extract the tensor from the tuple
            return super().__call__(*args, dynamic_inference_decode_only=dynamic_inference_decode_only, **kwargs)[0]
        # If not calling the local cudagraph path, call the normal forward path.
        return super().__call__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ):
        """Perform the forward pass through the Hyena block, based on the transformerblock.

        See https://github.com/NVIDIA/Megatron-LM/blob/1eed1d2/megatron/core/transformer/transformer_block.py#L583

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            sequence_len_offset (Tensor, optional): Offset for sequence length when computing RoPE.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            inference_params (BaseInferenceContext, optional): Object for storing inference-time
                context.
            dynamic_inference_decode_only: Optional[bool]: If true, indicates that the current
                inference context is for decode-only. This args is only used to uniquely
                identify decode and non-decode cuda graph runners in the cuda graph manager.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)
        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # For FP4: NVFP4BlockScaling doesn't have delayed scaling, always uses inner context
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
            )
        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            # No quantization
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()
                    with self.offload_context, inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )
                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

            # The attention layer (currently a simplified transformer layer)
            # outputs a tuple of (hidden_states, context). Context is intended
            # for cross-attention, and is not needed in our model.
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        # Final layer norm.
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_norm:
            hidden_states = hidden_states.clone()

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """Returns a sharded state dictionary for the current object.

        This function constructs a sharded state dictionary by iterating over the layers
        in the current object, computing the sharded state dictionary for each layer,
        and combining the results into a single dictionary.

        Parameters:
            prefix (str): The prefix to use for the state dictionary keys.
            sharded_offsets (tuple): The sharded offsets to use for the state dictionary.
            metadata (dict): Additional metadata to use when computing the sharded state dictionary.

        Returns:
            dict: The sharded state dictionary for the current object.
        """
        sharded_state_dict = {}
        layer_prefix = f"{prefix}layers."

        for local_layer_idx, layer in enumerate(self.layers):
            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f"{layer_prefix}{local_layer_idx}."  # module list index in HyenaBlock

            sharded_prefix = f"{layer_prefix}{global_layer_offset}."
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(state_dict_prefix, sharded_pp_offset, metadata)

            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if module is not self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=self.pg_collection.tp
                    )
                )

        return sharded_state_dict
