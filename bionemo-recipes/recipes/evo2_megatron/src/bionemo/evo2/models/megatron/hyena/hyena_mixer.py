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

import logging
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default

from bionemo.evo2.models.megatron.hyena.hyena_config import HyenaConfig
from bionemo.evo2.models.megatron.hyena.hyena_utils import (
    B2BCausalConv1dModule,
    ParallelCausalDepthwiseConv1dWithState,
    ParallelHyenaOperator,
    ParallelShortHyenaOperator,
    divide,
)


logger = logging.getLogger(__name__)


try:
    from transformer_engine.common.recipe import DelayedScaling, Format
except ImportError:

    def DelayedScaling(*args, **kwargs):  # noqa: N802
        """Not imported: DelayedScaling. An error will be raised if this is called."""
        raise ImportError("transformer_engine not installed. Using default recipe.")

    def Format(*args, **kwargs):  # noqa: N802
        """Not imported: Format. An error will be raised if this is called."""
        raise ImportError("transformer_engine not installed. Using default recipe.")

    class _te:  # noqa: N801
        """If this dummy module is accessed, a not imported error will be raised."""

        def __getattribute__(self, name: str) -> None:
            """Not imported: te. An error will be raised if this is called like a module."""
            raise ImportError("transformer_engine not installed. Using default recipe.")

    te = _te()  # if a user accesses anything in this module, an error will be raised
    logger.warning("WARNING: transformer_engine not installed. Using default recipe.")

try:
    from subquadratic_ops_torch.rearrange import rearrange as subquadratic_ops_rearrange
except ImportError as e:
    error = e
    msg = f"Imporrt error with subquadratic_ops: {e}. subquadratic_ops_rearrange is not available."

    def subquadratic_ops_rearrange(*args, **kwargs):
        """Not imported: subquadratic_ops_rearrange. An error will be raised if this is called."""
        raise ImportError(msg) from error


def set_format_recipe():
    """Set the fp8 format recipe. for Hyena."""
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    return fp8_recipe


@dataclass
class HyenaMixerSubmodules:
    """Contains the module specs for the input and output linear layers."""

    dense_projection: Union[ModuleSpec, type] = None
    dense: Union[ModuleSpec, type] = None


@dataclass
class HyenaMixerStateShapes:
    """Per-request recurrent decode-state layout for one Hyena mixer.

    Returned by :meth:`HyenaMixer.hyena_state_shapes_per_request`. Describes the two
    recurrent states every Hyena layer carries during decode, which dynamic inference packs
    into the context's two Mamba slots:

    * ``conv_*`` — the ``hyena_proj_conv`` FIR ring (uniform across all Hyena layer types).
    * ``ssm_*`` — the operator's single mixer state, whose shape/kind varies by operator
      type (``fir`` for short, ``inner_fir`` for medium, ``iir`` for long).

    ``*_owner_id`` are the ``id(module)`` keys the Hyena ops use to index their
    ``*_filter_state_dict`` (see ``hyena_utils.update_filter_state``/``get_filter_state``);
    the packed-slot adapter routes those exact ids to the dynamic context slots.
    """

    conv_shape: tuple  # (proj_channels, K_proj - 1)
    conv_owner_id: int  # id(self.hyena_proj_conv)
    ssm_shape: tuple  # (width, K_mixer - 1) for FIR, or (width, order) for IIR
    ssm_kind: str  # "fir" | "inner_fir" | "iir"  -> the *_filter_state_dict bucket
    ssm_owner_id: int  # id(mixer.short_conv) for short, else id(mixer)


class HyenaMixer(MegatronModule):
    """A class for the HyenaMixer."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        hyena_config: HyenaConfig,
        max_sequence_length,
        submodules,
        layer_number=1,
        operator_type="H",
        pg_collection=None,
    ):
        """Initialize the HyenaMixer."""
        super().__init__(transformer_config)
        self.transformer_config = transformer_config
        self.hyena_config = hyena_config
        self.operator_type = operator_type
        self.layer_number = layer_number
        self.grouped_attention = self.hyena_config.grouped_attention
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = self.pg_collection.tp
        self.fast_conv_proj = self.hyena_config.fast_conv_proj
        self.fast_conv_mixer = self.hyena_config.fast_conv_mixer

        self.use_subquadratic_ops = self.transformer_config.use_subquadratic_ops

        # Per attention head and per partition values.
        assert torch.distributed.is_initialized()
        self.model_parallel_size = self.tp_group.size() if self.tp_group is not None else 1
        world_size: int = self.model_parallel_size

        # Width expansion for Hyena
        self.hyena_width_expansion = self.hyena_config.hyena_width_expansion

        # we might expand the hidden size for hyena
        self.input_size = self.transformer_config.hidden_size
        self.hidden_size = int(self.transformer_config.hidden_size * self.hyena_width_expansion)

        # ensures parallizable
        if self.hyena_width_expansion > 1:
            multiple_of = 32
            self.hidden_size = int(multiple_of * ((self.hidden_size + multiple_of - 1) // multiple_of))

        # checks on the hidden size divisibility
        assert self.hidden_size % world_size == 0, (
            f"Hidden size {self.hidden_size} is not divisible by the world size {world_size}"
        )
        self.hidden_size_per_partition = divide(self.hidden_size, world_size)
        self.proj_groups = self.hyena_config.proj_groups

        self.tie_projection_weights = self.hyena_config.tie_projection_weights

        self.grouped_proj_size = self.transformer_config.hidden_size // self.proj_groups

        # Strided linear layer.
        if self.tie_projection_weights:
            # we'll repeat the output 3 times instead
            projections_size = self.hidden_size
        else:
            projections_size = 3 * self.hidden_size

        # qkv projections
        self.dense_projection = build_module(
            submodules.dense_projection,
            self.input_size,
            projections_size,
            config=self.transformer_config,
            init_method=self.transformer_config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.tp_group,
        )

        hyena_proj_groups = self.proj_groups if not self.grouped_attention else 1
        grouped_proj_size = self.hidden_size_per_partition // hyena_proj_groups

        self.hyena_proj_conv = ParallelCausalDepthwiseConv1dWithState(
            self.hidden_size_per_partition + 2 * grouped_proj_size,
            self.transformer_config,
            self.hyena_config,
            kernel_size=self.hyena_config.short_conv_L,
            init_method=transformer_config.init_method,
            bias=False,  # bias not currently supported (self.hyena_config.conv_proj_bias),
            use_fast_causal_conv=self.fast_conv_proj,
        )

        if self.operator_type == "hyena_short_conv":
            self.num_groups = self.hyena_config.num_groups_hyena_short
            self.num_groups_per_tp_rank = self.num_groups // self.model_parallel_size

            self.mixer = ParallelShortHyenaOperator(
                self.hidden_size,  # pass hidden size here to avoid recalculating
                self.transformer_config,
                self.hyena_config,
                self.transformer_config.init_method,
                short_conv_class=ParallelCausalDepthwiseConv1dWithState,
                use_fast_causal_conv=self.fast_conv_mixer,
                use_conv_bias=self.transformer_config.use_short_conv_bias,
            )

            if self.use_subquadratic_ops:
                # The B2B kernel is guarded in hyena_utils and fails early if the local CUDA stack
                # cannot run subquadratic_ops_torch correctly.
                self.b2b_kernel = B2BCausalConv1dModule(
                    self.hyena_proj_conv,
                    self.mixer,
                    operator_type=self.operator_type,
                    flip_mixer_weight=False,
                )

        if self.operator_type in [
            "hyena",
            "hyena_medium_conv",
        ]:
            if self.operator_type == "hyena_medium_conv":
                self.num_groups = self.hyena_config.num_groups_hyena_medium
            else:
                self.num_groups = self.hyena_config.num_groups_hyena
            self.num_groups_per_tp_rank = self.num_groups // self.model_parallel_size

            # subquadratic_ops LI layer is handled internally in the ParallelHyenaOperator
            # by transformer_configs.use_subquadratic_ops
            self.mixer = ParallelHyenaOperator(
                self.hidden_size,  # pass hidden size here to avoid recalculating
                self.transformer_config,
                self.hyena_config,
                self.transformer_config.init_method,
                operator_type,
                max_sequence_length,
            )

            if self.use_subquadratic_ops and self.operator_type == "hyena_medium_conv":
                # The B2B kernel is guarded in hyena_utils and fails early if the local CUDA stack
                # cannot run subquadratic_ops_torch correctly.
                self.b2b_kernel = B2BCausalConv1dModule(
                    self.hyena_proj_conv,
                    self.mixer,
                    operator_type=self.operator_type,
                    flip_mixer_weight=True,
                )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.dropout_p = self.transformer_config.attention_dropout
        self.attention_dropout = nn.Dropout(self.dropout_p)

        # When using non-parallel row linears, we allow PyTorch's Linear to
        # add bias: this is faster for TP=1 inference. For other cases (and
        # training), a more complex path is used, where bias is added as a
        # separate step.
        dense_skip_bias_add = not self.transformer_config.plain_row_linear

        self.dense = build_module(
            submodules.dense,
            self.hidden_size,
            self.input_size,
            config=self.transformer_config,
            init_method=self.transformer_config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=dense_skip_bias_add,
            is_expert=False,
            tp_comm_buffer_name="fc2",
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharded state dictionary for the HyenaMixer."""
        sharded_state_dict = {}
        # Submodules
        for name, module in self.named_children():
            if name != "attention_dropout" and name != "b2b_kernel":  # Don't register b2b_kernel (it's a wrapper)
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=self.pg_collection.tp
                )

                sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict

    def hyena_state_shapes_per_request(self) -> "HyenaMixerStateShapes":
        """Per-request recurrent decode-state shapes for this Hyena mixer.

        The Hyena analog of :meth:`megatron.core.ssm.mamba_mixer.MambaMixer.mamba_state_shapes_per_request`
        (mcore ``mamba_mixer.py:1195``). Every Hyena layer — regardless of operator type
        (``hyena_short_conv`` / ``hyena_medium_conv`` / ``hyena``) — carries **exactly two**
        recurrent states during decode, mirroring Mamba's ``(conv_state, ssm_state)`` slots:

        1. **conv slot** — the FIR ring buffer of ``self.hyena_proj_conv`` (the shared input
           projection conv present in *all* Hyena layer types). Stored eagerly under
           ``inference_context.fir_filter_state_dict[id(self.hyena_proj_conv)]`` with shape
           ``(B, proj_channels, K_proj-1)`` (see ``hyena_utils.ParallelCausalDepthwiseConv1dWithState.forward``
           and ``engine.parallel_fir``). Per-request shape (drop B): ``(proj_channels, K_proj-1)``.

        2. **ssm slot** — the operator's single mixer state, which *differs in shape and kind*
           by operator type:
             * ``hyena_short_conv``: FIR ring of ``mixer.short_conv``, key ``fir`` keyed by
               ``id(mixer.short_conv)``, shape ``(width, K_short-1)``.
             * ``hyena_medium_conv``: FIR ring of the operator, key ``inner_fir`` keyed by
               ``id(mixer)``, shape ``(width, K_medium-1)``.
             * ``hyena`` (long): the IIR pole-recurrence of the operator, key ``iir`` keyed by
               ``id(mixer)``, shape ``(width, order)``. NOTE: in *this* Evo2 implementation the
               IIR state is **real fp32, not complex** — the poles ``p``/``gamma`` are real
               params and ``engine.step_iir`` does ``exp(real_log_poles) * iir_state`` (real),
               while the prefill seed ``engine.prefill_via_modal_fft`` explicitly drops the
               imaginary part via ``.to(torch.float32)`` (engine.py:281). So NO 2x-real
               expansion is needed; ``order`` real slots suffice.

        Both states are kept fp32 because the decode recurrences in ``engine.step_fir`` and
        ``engine.step_iir`` run in fp32. The caller (:meth:`HyenaStack.hyena_state_shapes_per_request`)
        pads each layer's mixer state up to a common ``ssm_states_shape`` so the dynamic
        context can allocate one uniform shape across all Hyena ("mamba") layers.

        Returns:
            HyenaMixerStateShapes with this layer's conv/ssm per-request shapes + the
            owner ids + the state-dict key for the ssm slot.
        """
        proj_channels = self.hyena_proj_conv.short_conv_weight.shape[0] * self.hyena_proj_conv.group_dim
        conv_shape = (proj_channels, self.hyena_proj_conv.kernel_size - 1)

        if self.operator_type == "hyena_short_conv":
            width = self.mixer.short_conv.d_model
            ssm_shape = (width, self.mixer.short_conv.kernel_size - 1)
            ssm_kind = "fir"
            ssm_owner = self.mixer.short_conv
        elif self.operator_type == "hyena_medium_conv":
            width = self.mixer.width_per_tp_group
            ssm_shape = (width, self.mixer.kernel_size - 1)
            ssm_kind = "inner_fir"
            ssm_owner = self.mixer
        elif self.operator_type == "hyena":
            width = self.mixer.width_per_tp_group
            ssm_shape = (width, self.hyena_config.hyena_filter_order)
            ssm_kind = "iir"
            ssm_owner = self.mixer
        else:
            raise ValueError(f"Unsupported operator_type for native dynamic inference: {self.operator_type}")

        return HyenaMixerStateShapes(
            conv_shape=conv_shape,
            conv_owner_id=id(self.hyena_proj_conv),
            ssm_shape=ssm_shape,
            ssm_kind=ssm_kind,
            ssm_owner_id=id(ssm_owner),
        )

    def forward(self, x, layer_past=None, inference_context=None, _hyena_use_cp=True):
        """Applies the Hyena sequence mixing operation to input embeddings.

        Args:
            x: Input tensor of shape [L, B, D] (seq_len, batch_size, hidden_dim)
            layer_past: Past layer state for inference (default: None)
            inference_context: Parameters for inference (default: None)
            _hyena_use_cp: Whether to use context parallelism (default: True)

        Returns:
            Tuple of (output tensor, bias)
        """
        # CP control: disable CP during inference because the inference path
        # does not split sequences across CP ranks (the full sequence is on each rank).
        # The AllToAll operations in Hyena operators assume sequence-split input which
        # only happens during training.
        if inference_context is not None:
            _proj_use_cp = False
        elif _hyena_use_cp:
            cp_group = self.pg_collection.cp
            cp_size = cp_group.size()
            _proj_use_cp = cp_group is not None and cp_size > 1
        else:
            _proj_use_cp = False

        features, _ = self.dense_projection(x)
        if self.use_subquadratic_ops:
            features = subquadratic_ops_rearrange(features, bhl_to_lbh=False)
        else:
            features = rearrange(features, "l b d -> b d l").contiguous()

        is_b2b_eligible = self.use_subquadratic_ops and self.operator_type in [
            "hyena_short_conv",
            "hyena_medium_conv",
        ]
        # B2B runs during training (no inference_context) or during prefill (no FIR cache yet).
        # During decode, fall back to the regular per-token step path.
        is_prefill = inference_context is not None and id(self.hyena_proj_conv) not in getattr(
            inference_context, "fir_filter_state_dict", {}
        )

        if is_b2b_eligible and (inference_context is None or is_prefill):
            z = self.b2b_kernel(features, _use_cp=_proj_use_cp)
            if is_prefill:
                self._populate_b2b_inference_state(features, inference_context)
        else:
            features = self.hyena_proj_conv(
                features, _use_cp=_proj_use_cp, inference_context=inference_context
            )  # [B, D, L]
            x1, x2, v = rearrange(features, "b (g dg p) l -> b (g dg) p l", p=3, g=self.num_groups_per_tp_rank).unbind(
                dim=2
            )
            z = self.mixer(x1, x2, v, _hyena_use_cp=_proj_use_cp, inference_context=inference_context)

        if self.use_subquadratic_ops:
            z = subquadratic_ops_rearrange(z, bhl_to_lbh=True)
        else:
            z = rearrange(z, "b d l -> l b d").contiguous()
        y, bias = self.dense(z)
        return y, bias

    def _populate_b2b_inference_state(self, features, inference_context):
        """Populate FIR state for proj_conv and mixer after a b2b prefill.

        The b2b kernel doesn't expose its post-projection intermediate, but subsequent
        decode steps need (a) the proj_conv input tail and (b) the tail of `x2 * v`
        — the gated stream that mixer's short_conv operates on. We get (b) by running
        a windowed proj_conv on just the last (K_proj + K_mixer - 2) input positions.
        """
        proj_kernel_size = self.hyena_proj_conv.kernel_size

        # (a) proj_conv FIR state: input tail in [B, D, K_proj-1]
        # fp32 persistent buffer so step_fir's ``.to(float32)`` is a no-op and the
        # in-place ring-buffer shift preserves the dynamic-context alias.
        proj_state = features[..., -(proj_kernel_size - 1) :].to(torch.float32).contiguous()
        proj_dict = getattr(inference_context, "fir_filter_state_dict", {})
        proj_dict[id(self.hyena_proj_conv)] = proj_state
        setattr(inference_context, "fir_filter_state_dict", proj_dict)

        # (b) mixer FIR state: tail of (x2 * v), the gated post-projection stream
        if self.operator_type == "hyena_short_conv":
            mixer_kernel_size = self.mixer.short_conv.kernel_size
        else:  # hyena_medium_conv
            mixer_kernel_size = self.mixer.kernel_size

        tail_in_len = proj_kernel_size + mixer_kernel_size - 2
        if features.shape[-1] < tail_in_len:
            tail_in = F.pad(features, (tail_in_len - features.shape[-1], 0))
        else:
            tail_in = features[..., -tail_in_len:].contiguous()

        # Reuse the cached transformed weight from get_weight() (lru_cache'd).
        proj_weight = self.hyena_proj_conv.get_weight()

        intermediate = F.conv1d(
            F.pad(tail_in.to(torch.float32), (proj_kernel_size - 1, 0)),
            proj_weight,
            bias=None,
            stride=1,
            padding=0,
            groups=tail_in.shape[1],
        )[..., -(mixer_kernel_size - 1) :].to(features.dtype)

        x1, x2, v = rearrange(intermediate, "b (g dg p) l -> b (g dg) p l", p=3, g=self.num_groups_per_tp_rank).unbind(
            dim=2
        )
        mixer_input_tail = (x2 * v).to(torch.float32).contiguous()  # [B, D, K_mixer-1]

        if self.operator_type == "hyena_short_conv":
            mixer_state_owner_id = id(self.mixer.short_conv)
            mixer_dict_key = "fir_filter_state_dict"
        else:  # hyena_medium_conv
            mixer_state_owner_id = id(self.mixer)
            mixer_dict_key = "inner_fir_filter_state_dict"

        mixer_dict = getattr(inference_context, mixer_dict_key, {})
        mixer_dict[mixer_state_owner_id] = mixer_input_tail
        setattr(inference_context, mixer_dict_key, mixer_dict)
