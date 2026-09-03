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
from itertools import pairwise
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from megatron.core.packed_seq_params import PackedSeqParams
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
from bionemo.evo2.models.megatron.hyena.packed_kernels import (
    PACKED_CAUSAL_CONV_AVAILABLE,
    TRITON_AVAILABLE,
    ModalChunkMetadata,
    ModalPoles,
    fused_hyena_decode_from_projection,
    local_positions_from_cu_seqlens,
    modal_chunk_metadata_from_cu_seqlens,
    modal_poles,
    segmented_causal_conv1d,
    segmented_fir_from_projection,
    segmented_modal_from_projection,
    segmented_tail,
    sequence_ids_from_cu_seqlens,
)


logger = logging.getLogger(__name__)

_PACKED_MODAL_CHUNK_SIZE = 512


def _packed_metadata_cache_key(tensor: torch.Tensor, total_tokens: int) -> tuple[int, int, int | None, int]:
    """Identify immutable packed metadata without reading inference-tensor versions."""
    version = None if torch.is_inference(tensor) else tensor._version
    return id(tensor), tensor.data_ptr(), version, total_tokens


def _packed_sequence_boundaries(packed_seq_params: PackedSeqParams, total_tokens: int) -> tuple[int, ...]:
    """Return physical THD boundaries, caching the one required device synchronization.

    ``cu_seqlens_q_padded`` describes physical offsets when MCore inserts alignment
    padding between sequences; otherwise ``cu_seqlens_q`` is already physical.  The
    same ``PackedSeqParams`` object is passed through every layer, so cache the parsed
    tuple on it rather than synchronizing the CUDA metadata once per Hyena layer.
    """
    if packed_seq_params.qkv_format != "thd":
        raise ValueError(f"Packed Hyena only supports qkv_format='thd', got {packed_seq_params.qkv_format!r}")

    cu_seqlens = packed_seq_params.cu_seqlens_q_padded
    if cu_seqlens is None:
        cu_seqlens = packed_seq_params.cu_seqlens_q
    if not torch.is_tensor(cu_seqlens):
        raise ValueError("Packed Hyena requires cu_seqlens_q metadata")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"Packed Hyena requires one-dimensional cu_seqlens_q, got shape {tuple(cu_seqlens.shape)}")

    # Inference tensors deliberately have no version counter. PackedSeqParams treats
    # cu_seqlens as immutable, so object/storage identity is sufficient for those tensors;
    # ordinary tensors retain version-based invalidation for in-place changes.
    cache_key = _packed_metadata_cache_key(cu_seqlens, total_tokens)
    cached = getattr(packed_seq_params, "_evo2_hyena_boundary_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    boundaries = tuple(int(boundary) for boundary in cu_seqlens.detach().cpu().tolist())
    if len(boundaries) < 2:
        raise ValueError("Packed Hyena requires at least one sequence")
    if boundaries[0] != 0:
        raise ValueError(f"Packed Hyena boundaries must start at zero, got {boundaries[0]}")
    if any(end <= start for start, end in pairwise(boundaries)):
        raise ValueError(f"Packed Hyena boundaries must be strictly increasing, got {boundaries}")
    if boundaries[-1] > total_tokens:
        raise ValueError(f"Packed Hyena boundary {boundaries[-1]} exceeds the physical token count {total_tokens}")
    if boundaries[-1] < total_tokens:
        # Match MCore's seq_idx convention: graph/dataset padding after the last
        # declared sequence is an isolated extra sequence, never continuation.
        boundaries = (*boundaries, total_tokens)

    setattr(packed_seq_params, "_evo2_hyena_boundary_cache", (cache_key, boundaries))
    return boundaries


def _hyena_packed_bucket_length(sequence_length: int) -> int:
    """Round a segment to a geometric bucket with less than 2x padding."""
    return 1 << (sequence_length - 1).bit_length()


def _packed_cuda_metadata(
    packed_seq_params: PackedSeqParams, total_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ModalChunkMetadata | None]:
    """Return boundaries, positions, ids, and modal chunks cached for all layers."""
    cu_seqlens = packed_seq_params.cu_seqlens_q_padded
    if cu_seqlens is None:
        cu_seqlens = packed_seq_params.cu_seqlens_q
    if (
        not torch.is_tensor(cu_seqlens)
        or not cu_seqlens.is_cuda
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.ndim != 1
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("Fast packed Hyena requires contiguous CUDA int32 cu_seqlens_q")

    cache_key = _packed_metadata_cache_key(cu_seqlens, total_tokens)
    cached = getattr(packed_seq_params, "_evo2_hyena_cuda_metadata_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    local_positions = local_positions_from_cu_seqlens(cu_seqlens, total_tokens)
    sequence_ids = getattr(packed_seq_params, "seq_idx", None)
    if (
        torch.is_tensor(sequence_ids)
        and sequence_ids.shape == (1, total_tokens)
        and sequence_ids.dtype == torch.int32
        and sequence_ids.is_cuda
    ):
        sequence_ids = sequence_ids[0]
    else:
        sequence_ids = sequence_ids_from_cu_seqlens(cu_seqlens, total_tokens)
    max_sequence_length = getattr(packed_seq_params, "max_seqlen_q", None)
    if max_sequence_length is None:
        # MCore leaves this optional. The total packed width is a conservative upper
        # bound that preserves correctness; callers that provide the maximum avoid
        # building modal chunk metadata for a pack of many individually short rows.
        max_sequence_length = total_tokens
    elif torch.is_tensor(max_sequence_length):
        max_sequence_length = int(max_sequence_length.item())
    else:
        max_sequence_length = int(max_sequence_length)
    modal_chunks = None
    if max_sequence_length > _PACKED_MODAL_CHUNK_SIZE:
        modal_chunks = modal_chunk_metadata_from_cu_seqlens(
            cu_seqlens,
            chunk_size=_PACKED_MODAL_CHUNK_SIZE,
        )
    metadata = (cu_seqlens, local_positions, sequence_ids, modal_chunks)
    setattr(packed_seq_params, "_evo2_hyena_cuda_metadata_cache", (cache_key, metadata))
    return metadata


def _packed_fir_weight(module) -> tuple[torch.Tensor, int]:
    """Normalize an Evo2 grouped depthwise weight to contiguous ``[G,K]``."""
    weight = module.short_conv_weight
    if weight.ndim == 2:
        return weight.contiguous(), module.group_dim
    if weight.ndim != 3:
        raise ValueError(f"Unsupported Hyena FIR weight shape {tuple(weight.shape)}")
    if weight.shape[1] == 1:
        return weight[:, 0].contiguous(), module.group_dim
    return weight.reshape(-1, weight.shape[-1]).contiguous(), 1


def _dynamic_packed_cuda_metadata(
    inference_context, total_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ModalChunkMetadata | None]:
    """Reuse the dynamic attention scheduler's physical ragged prefill boundaries."""
    active_request_count = int(inference_context.get_active_request_count())
    cu_seqlens, _ = inference_context.cu_query_lengths()
    cu_seqlens = cu_seqlens[: active_request_count + 1]
    if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_cuda or not cu_seqlens.is_contiguous():
        raise ValueError("Dynamic packed Hyena requires contiguous CUDA int32 cumulative query lengths")

    active_slice = slice(inference_context.paused_request_count, inference_context.total_request_count)
    query_lengths = tuple(int(length) for length in inference_context.request_query_lengths[active_slice].tolist())
    cache_key = (cu_seqlens.data_ptr(), total_tokens, query_lengths)
    cached = getattr(inference_context, "_evo2_hyena_cuda_metadata_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    local_positions = local_positions_from_cu_seqlens(cu_seqlens, total_tokens)
    sequence_ids = sequence_ids_from_cu_seqlens(cu_seqlens, total_tokens)
    modal_chunks = None
    if max(query_lengths) > _PACKED_MODAL_CHUNK_SIZE:
        modal_chunks = modal_chunk_metadata_from_cu_seqlens(
            cu_seqlens,
            chunk_size=_PACKED_MODAL_CHUNK_SIZE,
        )
    metadata = (cu_seqlens, local_positions, sequence_ids, modal_chunks)
    setattr(inference_context, "_evo2_hyena_cuda_metadata_cache", (cache_key, metadata))
    return metadata


def warm_packed_hyena_caches(model: torch.nn.Module) -> int:
    """Build every long-Hyena layer's packed modal pole tables once, ahead of any decode.

    Inference entry points call this after the model is finalized so the tables exist
    for the whole process lifetime and are never first created inside a CUDA graph
    capture. Returns the number of layers warmed.
    """
    warmed = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, HyenaMixer) and module.operator_type == "hyena":
                module._packed_modal_poles()
                warmed += 1
    return warmed


def _dynamic_context_real_token_count(inference_context, padded_token_count: int) -> int:
    """Return real active tokens in a dynamic-context flattened token batch."""
    if inference_context is None:
        return padded_token_count
    is_static_batching = getattr(inference_context, "is_static_batching", None)
    if is_static_batching is None or is_static_batching():
        return padded_token_count

    active_token_count = getattr(inference_context, "active_token_count", padded_token_count)
    if torch.is_tensor(active_token_count):
        if active_token_count.numel() != 1:
            return padded_token_count
        active_token_count = active_token_count.item()
    try:
        active_token_count = int(active_token_count)
    except (TypeError, ValueError):
        return padded_token_count
    return max(1, min(active_token_count, padded_token_count))


def _slice_padded_dynamic_context_tokens(features: torch.Tensor, inference_context) -> tuple[torch.Tensor, int]:
    """Drop dynamic-context dummy token rows before Hyena recurrent state updates."""
    padded_token_count = int(features.shape[-1])
    real_token_count = _dynamic_context_real_token_count(inference_context, padded_token_count)
    if real_token_count == padded_token_count:
        return features, padded_token_count
    return features[..., :real_token_count].contiguous(), padded_token_count


def _pad_padded_dynamic_context_tokens(z: torch.Tensor, padded_token_count: int) -> torch.Tensor:
    """Restore MCore's padded token width after Hyena recurrent computation."""
    if z.shape[-1] >= padded_token_count:
        return z
    return F.pad(z, (0, padded_token_count - z.shape[-1]))


def _reshape_dynamic_context_requests(
    features: torch.Tensor, inference_context
) -> tuple[torch.Tensor, tuple[int, int] | None]:
    """Unpack MCore's flattened active requests into Hyena's batch dimension.

    Dynamic inference usually presents active request tokens as one flattened stream shaped
    ``[1, channels, total_tokens]``. Evo2's Hyena recurrences need independent state per
    request, so the opt-in batched path reshapes same-length active requests into
    ``[num_requests, channels, tokens_per_request]`` for each Hyena layer and later restores the
    flattened layout for MCore attention/output layers. NeMo-RL's generation worker can also
    call dummy/decode forwards with requests already in that Hyena-compatible batch layout.
    """
    if inference_context is None or not bool(getattr(inference_context, "evo2_batched_decode_enabled", False)):
        return features, None

    paused_request_count = int(getattr(inference_context, "paused_request_count", 0))
    total_request_count = int(getattr(inference_context, "total_request_count", 0))
    active_request_count = total_request_count - paused_request_count
    if paused_request_count != 0 or active_request_count <= 1:
        return features, None

    request_query_lengths = inference_context.request_query_lengths[paused_request_count:total_request_count].detach()
    first_query_length = int(request_query_lengths[0].item())
    if first_query_length <= 0 or not bool((request_query_lengths == first_query_length).all().item()):
        raise ValueError(
            "Evo2 batched decode requires all active requests to have the same query length; "
            f"got {request_query_lengths.cpu().tolist()}"
        )

    if features.shape[0] == active_request_count and features.shape[-1] == first_query_length:
        return features, None

    if features.shape[0] != 1:
        raise ValueError(
            "Evo2 batched decode expects flattened dynamic input with batch=1 or already batched "
            f"input with batch={active_request_count}, got {features.shape}"
        )

    real_token_count = active_request_count * first_query_length
    if features.shape[-1] != real_token_count:
        raise ValueError(
            "Evo2 batched decode expected flattened tokens to match active requests; "
            f"tokens={features.shape[-1]}, requests={active_request_count}, query_length={first_query_length}"
        )

    unpacked = features.reshape(1, features.shape[1], active_request_count, first_query_length)
    return unpacked.squeeze(0).permute(1, 0, 2).contiguous(), (active_request_count, first_query_length)


def _restore_dynamic_context_requests(z: torch.Tensor, layout: tuple[int, int] | None) -> torch.Tensor:
    """Restore ``[num_requests, channels, query_length]`` to MCore's flattened layout."""
    if layout is None:
        return z
    active_request_count, query_length = layout
    if z.shape[0] != active_request_count or z.shape[-1] != query_length:
        raise ValueError(
            f"Evo2 batched decode Hyena output shape changed unexpectedly; output={tuple(z.shape)}, layout={layout}"
        )
    return z.permute(1, 0, 2).contiguous().reshape(1, z.shape[1], active_request_count * query_length)


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
            pg_collection=self.pg_collection,
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
                pg_collection=self.pg_collection,
            )

            if self.use_subquadratic_ops:
                # The B2B kernel is guarded in hyena_utils and fails early if the local CUDA stack
                # cannot run subquadratic_ops_torch correctly.
                self.b2b_kernel = B2BCausalConv1dModule(
                    self.hyena_proj_conv,
                    self.mixer,
                    operator_type=self.operator_type,
                    flip_mixer_weight=False,
                    pg_collection=self.pg_collection,
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
                pg_collection=self.pg_collection,
            )

            if self.use_subquadratic_ops and self.operator_type == "hyena_medium_conv":
                # The B2B kernel is guarded in hyena_utils and fails early if the local CUDA stack
                # cannot run subquadratic_ops_torch correctly.
                self.b2b_kernel = B2BCausalConv1dModule(
                    self.hyena_proj_conv,
                    self.mixer,
                    operator_type=self.operator_type,
                    flip_mixer_weight=True,
                    pg_collection=self.pg_collection,
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
            tp_group=self.tp_group,
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

    def _mix_projected_features(self, features, *, inference_context, _proj_use_cp):
        """Apply the projection FIR and selected Hyena operator to ``[B, D, L]`` features."""
        fused_decode = self._mix_fused_dynamic_decode(
            features,
            inference_context=inference_context,
            _proj_use_cp=_proj_use_cp,
        )
        if fused_decode is not None:
            return fused_decode

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
            return z

        features = self.hyena_proj_conv(
            features, _use_cp=_proj_use_cp, inference_context=inference_context
        )  # [B, D, L]
        x1, x2, v = rearrange(
            features,
            "b (g dg p) l -> b (g dg) p l",
            p=3,
            g=self.num_groups_per_tp_rank,
        ).unbind(dim=2)
        return self.mixer(x1, x2, v, _hyena_use_cp=_proj_use_cp, inference_context=inference_context)

    def _mix_fused_dynamic_decode(self, features, *, inference_context, _proj_use_cp):
        """Run one stateful decode position with one projection-plus-mixer launch."""
        if (
            inference_context is None
            or _proj_use_cp
            or not TRITON_AVAILABLE
            or torch.is_grad_enabled()
            or not features.is_cuda
            or features.dtype != torch.bfloat16
            or features.ndim != 3
            or features.shape[1] != 3 * self.hidden_size_per_partition
            or features.shape[-1] != 1
            or not 2 <= self.hyena_proj_conv.kernel_size <= 4
            or self.operator_type not in {"hyena_short_conv", "hyena_medium_conv", "hyena"}
        ):
            return None
        # Both dynamic paged decode and static FlashAttention decode expose the
        # same per-request recurrent-state dictionaries.  The fused recurrence
        # only depends on those stable tensors, not on the KV-cache layout.
        if getattr(inference_context, "is_static_batching", None) is None:
            return None

        projection_state = getattr(inference_context, "fir_filter_state_dict", {}).get(id(self.hyena_proj_conv))
        if projection_state is None or projection_state.shape[0] != features.shape[0]:
            return None
        projection_weight, projection_group_width = _packed_fir_weight(self.hyena_proj_conv)
        if projection_state.ndim == 3 and projection_state.shape[-1] < projection_weight.shape[-1] - 1:
            # The ordinary eager state path grows a short prefill history one decode
            # token at a time. Fusion requires the complete fixed-width ring.
            return None

        if self.operator_type == "hyena_short_conv":
            # The fused implementation currently models Evo2's standard gated short operator.
            if not self.mixer.pregate or not self.mixer.postgate:
                return None
            mixer_state = getattr(inference_context, "fir_filter_state_dict", {}).get(id(self.mixer.short_conv))
            mixer_weight, mixer_group_width = _packed_fir_weight(self.mixer.short_conv)
            diagonal = self.mixer.conv_bias.contiguous() if self.mixer.use_conv_bias else None
            diagonal_group_width = self.mixer.group_dim
            operator = "short"
            residues = poles = None
        elif self.operator_type == "hyena_medium_conv":
            mixer_state = getattr(inference_context, "inner_fir_filter_state_dict", {}).get(id(self.mixer))
            mixer_weight = self.mixer.filter(self.mixer.hyena_medium_conv_len)
            if isinstance(mixer_weight, tuple):
                mixer_weight = mixer_weight[0]
            mixer_weight = mixer_weight.squeeze(0).contiguous()
            mixer_group_width = self.mixer.group_dim
            diagonal = self.mixer.conv_bias.contiguous()
            diagonal_group_width = 1
            operator = "medium"
            residues = poles = None
        else:
            if self.hyena_config.hyena_filter_order != 16:
                return None
            mixer_state = getattr(inference_context, "iir_filter_state_dict", {}).get(id(self.mixer))
            mixer_weight = projection_weight  # unused by the modal constexpr branch
            mixer_group_width = self.mixer.group_dim
            diagonal = self.mixer.conv_bias.contiguous()
            diagonal_group_width = 1
            operator = "modal"
            residues = self.mixer.filter.R.contiguous()
            poles = self._packed_modal_poles()

        if mixer_state is None or mixer_state.shape[0] != features.shape[0]:
            return None
        if (
            operator in {"short", "medium"}
            and mixer_state.ndim == 3
            and mixer_state.shape[-1] < mixer_weight.shape[-1] - 1
        ):
            return None
        output = fused_hyena_decode_from_projection(
            features[..., 0].contiguous(),
            projection_state,
            projection_weight,
            mixer_state,
            mixer_weight,
            diagonal,
            residues,
            poles,
            projection_group_width=projection_group_width,
            mixer_group_width=mixer_group_width,
            operator=operator,
            diagonal_group_width=diagonal_group_width,
        )
        return output.unsqueeze(-1)

    def _mix_packed_projected_features(self, features, packed_seq_params):
        """Mix THD segments in skew-safe length buckets with autograd-safe boundaries.

        A causal FIR or FFT over the concatenated token stream leaks across sequence
        boundaries. Materializing segments along the operator batch dimension lets
        every existing short, medium, and long Hyena forward/backward path run
        independently. Geometric buckets keep padding below 2x even for highly skewed
        packs, and prevent one very long segment from inflating every short transform.

        This is a correctness and compatibility fallback, not an expected training
        throughput optimization: every Hyena layer splits, pads, and reassembles its
        segments on each forward until boundary-aware packed backward kernels exist.
        """
        if features.shape[0] != 1:
            raise ValueError(
                "Packed Hyena expects MCore THD hidden states with batch dimension 1; "
                f"got projected shape {tuple(features.shape)}"
            )

        boundaries = _packed_sequence_boundaries(packed_seq_params, features.shape[-1])
        lengths = tuple(end - start for start, end in pairwise(boundaries))
        buckets: dict[int, list[tuple[int, int, int]]] = {}
        for index, (start, end) in enumerate(pairwise(boundaries)):
            bucket_length = _hyena_packed_bucket_length(end - start)
            buckets.setdefault(bucket_length, []).append((index, start, end))

        output_segments: list[torch.Tensor | None] = [None] * len(lengths)
        for bucket_length, entries in sorted(buckets.items()):
            padded_segments = [
                F.pad(features[..., start:end], (0, bucket_length - (end - start))) for _, start, end in entries
            ]
            batched_features = torch.cat(padded_segments, dim=0)
            batched_output = self._mix_projected_features(
                batched_features,
                inference_context=None,
                _proj_use_cp=False,
            )
            for bucket_index, (segment_index, start, end) in enumerate(entries):
                output_segments[segment_index] = batched_output[bucket_index : bucket_index + 1, :, : end - start]

        assert all(segment is not None for segment in output_segments)
        return torch.cat(output_segments, dim=-1)

    def _packed_modal_poles(self) -> ModalPoles:
        """Return this layer's modal decay tables, computed once and reused by every packed kernel.

        The tables depend only on the ``p`` and ``gamma`` parameters, so they are cached on
        the module and refreshed only when either parameter's storage or version changes.
        Refresh preserves the tables' tensor storage when possible: decode CUDA graphs capture
        those addresses, while RL refits update the source parameters in place between rollouts.
        They keep the parameters' ``[num_groups, 16]`` shape for this tensor-parallel
        partition; the kernels expand groups to channels with ``group_dim``. Unlike
        ``get_logp`` there is no context-parallel rank slicing, because the packed paths
        only run with the full channel set on every rank.
        """
        p = self.mixer.filter.p
        gamma = self.mixer.filter.gamma
        cache_key = (p.data_ptr(), p._version, gamma.data_ptr(), gamma._version)
        cached = getattr(self, "_packed_modal_poles_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        refreshed = modal_poles(gamma.detach(), p.detach())
        poles = refreshed
        if cached is not None:
            cached_poles = cached[1]
            if all(
                old.shape == new.shape and old.dtype == new.dtype and old.device == new.device
                for old, new in zip(
                    (cached_poles.decay, cached_poles.log_decay),
                    (refreshed.decay, refreshed.log_decay),
                    strict=True,
                )
            ):
                cached_poles.decay.copy_(refreshed.decay)
                cached_poles.log_decay.copy_(refreshed.log_decay)
                poles = cached_poles
        self._packed_modal_poles_cache = (cache_key, poles)
        return poles

    def _supports_flat_segmented_prefill(self, projection: torch.Tensor) -> bool:
        """Whether the inference-only flat segmented kernels support this model instance."""
        if not TRITON_AVAILABLE or not PACKED_CAUSAL_CONV_AVAILABLE:
            return False
        if torch.is_grad_enabled() or not projection.is_cuda or projection.dtype != torch.bfloat16:
            return False
        if projection.ndim != 3 or projection.shape[1] != 1:
            return False
        if projection.shape[-1] != 3 * self.hidden_size_per_partition:
            return False
        if self.hyena_proj_conv.kernel_size > 4:
            return False
        if self.operator_type == "hyena":
            if self.mixer.bidirectional or self.hyena_config.hyena_filter_order != 16:
                return False
        return self.operator_type in {"hyena_short_conv", "hyena_medium_conv", "hyena"}

    def _supports_flat_dynamic_prefill(self, projection: torch.Tensor, inference_context) -> bool:
        """Select the initial ragged prefill step, leaving continuation/decode stateful."""
        if inference_context is None or not self._supports_flat_segmented_prefill(projection):
            return False
        is_static_batching = getattr(inference_context, "is_static_batching", None)
        if is_static_batching is None or is_static_batching():
            return False
        active_request_count = int(inference_context.get_active_request_count())
        if active_request_count < 1:
            return False
        if int(getattr(inference_context, "num_prefill_requests", 0)) != active_request_count:
            return False
        projection_states = getattr(inference_context, "fir_filter_state_dict", {})
        return projection_states.get(id(self.hyena_proj_conv)) is None

    def _mix_flat_segmented_prefill(
        self,
        projection: torch.Tensor,
        cu_seqlens: torch.Tensor,
        local_positions: torch.Tensor,
        sequence_ids: torch.Tensor,
        modal_chunks: ModalChunkMetadata | None,
        *,
        inference_context=None,
    ) -> torch.Tensor:
        """Apply one unpadded THD Hyena prefill using shared ``cu_seqlens`` metadata."""
        flat_projection = projection[:, 0, :]
        if not flat_projection.is_contiguous():
            flat_projection = flat_projection.contiguous()

        projection_weight, projection_group_width = _packed_fir_weight(self.hyena_proj_conv)
        projection_fir = segmented_causal_conv1d(
            flat_projection,
            projection_weight,
            sequence_ids,
            group_width=projection_group_width,
        )

        if self.operator_type == "hyena_short_conv":
            mixer_weight, group_width = _packed_fir_weight(self.mixer.short_conv)
            diagonal = None
            if self.mixer.use_conv_bias:
                diagonal = self.mixer.conv_bias.repeat_interleave(self.mixer.group_dim).contiguous()
            output = segmented_fir_from_projection(
                projection_fir,
                mixer_weight,
                local_positions,
                group_width=group_width,
                flip_filter=True,
                pregate=self.mixer.pregate,
                postgate=self.mixer.postgate,
                diagonal=diagonal,
            )
        elif self.operator_type == "hyena_medium_conv":
            mixer_weight = self.mixer.filter(self.mixer.hyena_medium_conv_len)
            if isinstance(mixer_weight, tuple):
                mixer_weight = mixer_weight[0]
            mixer_weight = mixer_weight.squeeze(0).contiguous()
            output = segmented_fir_from_projection(
                projection_fir,
                mixer_weight,
                local_positions,
                group_width=self.mixer.group_dim,
                flip_filter=False,
                diagonal=self.mixer.conv_bias.contiguous(),
            )
        else:
            final_state = None
            if inference_context is not None:
                final_state = torch.empty(
                    cu_seqlens.numel() - 1,
                    self.hidden_size_per_partition,
                    self.hyena_config.hyena_filter_order,
                    dtype=torch.float32,
                    device=projection.device,
                )
            output = segmented_modal_from_projection(
                projection_fir,
                self.mixer.conv_bias.contiguous(),
                self.mixer.filter.R.contiguous(),
                self._packed_modal_poles(),
                cu_seqlens,
                group_width=self.mixer.group_dim,
                final_state_out=final_state,
                chunk_metadata=modal_chunks,
            )

        if inference_context is not None:
            projection_state = segmented_tail(flat_projection, cu_seqlens, self.hyena_proj_conv.kernel_size - 1)
            inference_context.fir_filter_state_dict[id(self.hyena_proj_conv)] = projection_state
            if self.operator_type in {"hyena_short_conv", "hyena_medium_conv"}:
                mixer_kernel_size = (
                    self.mixer.short_conv.kernel_size
                    if self.operator_type == "hyena_short_conv"
                    else self.mixer.kernel_size
                )
                projected_tail = segmented_tail(
                    projection_fir,
                    cu_seqlens,
                    mixer_kernel_size - 1,
                    output_dtype=projection_fir.dtype,
                )
                batch_size, projected_channels, tail_length = projected_tail.shape
                projected_tail = projected_tail.view(batch_size, projected_channels // 3, 3, tail_length)
                mixer_state = (projected_tail[:, :, 1] * projected_tail[:, :, 2]).to(torch.float32)
                if self.operator_type == "hyena_short_conv":
                    inference_context.fir_filter_state_dict[id(self.mixer.short_conv)] = mixer_state
                else:
                    inference_context.inner_fir_filter_state_dict[id(self.mixer)] = mixer_state
            else:
                inference_context.iir_filter_state_dict[id(self.mixer)] = final_state
        return output

    def forward(
        self,
        x,
        layer_past=None,
        inference_context=None,
        packed_seq_params: PackedSeqParams | None = None,
        _hyena_use_cp=True,
    ):
        """Applies the Hyena sequence mixing operation to input embeddings.

        Args:
            x: Input tensor of shape [L, B, D] (seq_len, batch_size, hidden_dim)
            layer_past: Past layer state for inference (default: None)
            inference_context: Parameters for inference (default: None)
            packed_seq_params: THD boundaries for stateless packed training or prediction/scoring. Stateful
                generation instead uses ``inference_context`` for both prefill and autoregressive decode.
            _hyena_use_cp: Whether to use context parallelism (default: True)

        Returns:
            Tuple of (output tensor, bias)
        """
        # CP control: disable CP during inference because the inference path
        # does not split sequences across CP ranks (the full sequence is on each rank).
        # The AllToAll operations in Hyena operators assume sequence-split input which
        # only happens during training.
        cp_group = self.pg_collection.cp
        cp_size = cp_group.size() if cp_group is not None else 1
        if inference_context is not None:
            _proj_use_cp = False
        elif _hyena_use_cp:
            _proj_use_cp = cp_group is not None and cp_size > 1
        else:
            _proj_use_cp = False

        # ``PackedSeqParams`` belongs to stateless packed training/scoring forwards. Native generation
        # instead lets ``inference_context`` own request boundaries during ragged prefill, then reuses
        # that context for autoregressive decode without passing packed parameters.
        if packed_seq_params is not None and inference_context is not None:
            raise ValueError(
                "PackedSeqParams cannot share Hyena's stateful inference context; use native dynamic request "
                "batching for generation or an ordinary stateless packed forward for training/scoring"
            )
        if packed_seq_params is not None and _proj_use_cp:
            raise NotImplementedError("Packed Hyena with context parallel size greater than one is not yet supported")

        features, _ = self.dense_projection(x)
        if packed_seq_params is not None and self._supports_flat_segmented_prefill(features):
            cu_seqlens, local_positions, sequence_ids, modal_chunks = _packed_cuda_metadata(
                packed_seq_params, features.shape[0]
            )
            z = self._mix_flat_segmented_prefill(
                features,
                cu_seqlens,
                local_positions,
                sequence_ids,
                modal_chunks,
            )
            y, bias = self.dense(z.unsqueeze(1))
            return y, bias
        if self._supports_flat_dynamic_prefill(features, inference_context):
            real_token_count = _dynamic_context_real_token_count(inference_context, features.shape[0])
            real_projection = features[:real_token_count]
            cu_seqlens, local_positions, sequence_ids, modal_chunks = _dynamic_packed_cuda_metadata(
                inference_context, real_token_count
            )
            z = self._mix_flat_segmented_prefill(
                real_projection,
                cu_seqlens,
                local_positions,
                sequence_ids,
                modal_chunks,
                inference_context=inference_context,
            )
            if real_token_count < features.shape[0]:
                z = F.pad(z, (0, 0, 0, features.shape[0] - real_token_count))
            y, bias = self.dense(z.unsqueeze(1))
            return y, bias
        if self.use_subquadratic_ops:
            features = subquadratic_ops_rearrange(features, bhl_to_lbh=False)
        else:
            features = rearrange(features, "l b d -> b d l").contiguous()
        if packed_seq_params is not None:
            z = self._mix_packed_projected_features(features, packed_seq_params)
        else:
            features, padded_dynamic_token_count = _slice_padded_dynamic_context_tokens(features, inference_context)
            features, dynamic_request_layout = _reshape_dynamic_context_requests(features, inference_context)
            z = self._mix_projected_features(
                features,
                inference_context=inference_context,
                _proj_use_cp=_proj_use_cp,
            )
            z = _restore_dynamic_context_requests(z, dynamic_request_layout)
            z = _pad_padded_dynamic_context_tokens(z, padded_dynamic_token_count)
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

        _x1, x2, v = rearrange(
            intermediate, "b (g dg p) l -> b (g dg) p l", p=3, g=self.num_groups_per_tp_rank
        ).unbind(dim=2)
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
