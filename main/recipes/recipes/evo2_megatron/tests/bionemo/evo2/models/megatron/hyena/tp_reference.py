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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Slow FP32 tensor-parallel oracle used only by tests and debugging.

These layers gather complete logical tensors and repeat full FP32 GEMMs on every TP
rank. They demonstrate that checkpoint sharding and TP layouts represent the same
mathematical model; they are intentionally outside ``src`` because they are not a
production inference implementation and must never be selected by the Evo2 CLI.
"""

from unittest.mock import patch

import torch
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TERowParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

import bionemo.evo2.models.megatron.hyena.hyena_layer_specs as hyena_layer_specs


def _all_gather_shards(tensor: torch.Tensor, *, tp_size: int, tp_group) -> list[torch.Tensor]:
    """Gather equal-shaped tensor-parallel shards in rank order."""
    if tp_size == 1:
        return [tensor]
    shards = [torch.empty_like(tensor) for _ in range(tp_size)]
    torch.distributed.all_gather(shards, tensor.contiguous(), group=tp_group)
    return shards


def merge_strided_column_shards(shards: list[torch.Tensor], *, stride: int) -> torch.Tensor:
    """Reconstruct a full column-parallel tensor, including MCore's GLU stride layout."""
    if stride == 1:
        return torch.cat(shards, dim=0)
    rank_pieces = [torch.chunk(shard, stride, dim=0) for shard in shards]
    return torch.cat(
        [rank_pieces[rank][stride_index] for stride_index in range(stride) for rank in range(len(shards))],
        dim=0,
    )


def select_strided_column_shard(
    tensor: torch.Tensor,
    *,
    tp_rank: int,
    tp_size: int,
    stride: int,
) -> torch.Tensor:
    """Select one rank's output from a full column projection using MCore stride layout."""
    pieces = torch.chunk(tensor, tp_size * stride, dim=-1)
    return torch.cat([pieces[tp_rank + stride_index * tp_size] for stride_index in range(stride)], dim=-1)


def _current_scaling_fp8_dequantize(tensor: torch.Tensor) -> torch.Tensor:
    """Apply deterministic E4M3 per-tensor current scaling and return FP32 values."""
    tensor = tensor.float()
    amax = tensor.abs().amax()
    if amax == 0:
        return tensor
    fp8_dtype = torch.float8_e4m3fn
    scale = torch.finfo(fp8_dtype).max / amax
    return (tensor * scale).to(fp8_dtype).float() / scale


def _topology_reference_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Evaluate a complete logical linear, emulating current-scaling FP8 when active."""
    x = x.float()
    weight = weight.float()
    if FP8GlobalStateManager.is_fp8_enabled():
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if not recipe.float8_current_scaling():
            raise RuntimeError("The TP test oracle supports FP8 only with per-tensor current scaling.")
        x = _current_scaling_fp8_dequantize(x)
        weight = _current_scaling_fp8_dequantize(weight)
    return torch.nn.functional.linear(x, weight, None)


class TpReferenceLayerNormColumnParallelLinear(TELayerNormColumnParallelLinear):
    """Test-only fused RMSNorm/column projection evaluated as one FP32 GEMM."""

    def forward(self, x):  # noqa: D102
        if torch.is_grad_enabled():
            raise RuntimeError("The FP32 TP reference is inference-only.")
        if self.normalization != "RMSNorm":
            raise RuntimeError("The FP32 TP reference currently requires RMSNorm.")

        if self.tp_size > 1 and self.sequence_parallel:
            x = gather_from_sequence_parallel_region(
                x,
                tensor_parallel_output_grad=False,
                group=self._tp_group,
            )

        norm_weight = self.layer_norm_weight.float()
        if self.zero_centered_gamma:
            norm_weight = norm_weight + 1.0
        normalized = torch.nn.functional.rms_norm(
            x.float(),
            (self.in_features,),
            norm_weight,
            self.eps,
        )
        weight = self.weight
        if self.tp_size > 1:
            weight = merge_strided_column_shards(
                _all_gather_shards(weight, tp_size=self.tp_size, tp_group=self._tp_group),
                stride=self.stride,
            )

        output = _topology_reference_linear(normalized, weight).to(x.dtype)
        if self.tp_size > 1:
            output = select_strided_column_shard(
                output,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                stride=self.stride,
            )
        bias = getattr(self, "bias", None)
        if self.te_return_bias:
            return output, bias
        if self.use_bias and bias is not None:
            output = output + bias
        return output, None


class TpReferenceRowParallelLinear(TERowParallelLinear):
    """Test-only row projection evaluated as one gathered FP32 GEMM."""

    def forward(self, x):  # noqa: D102
        if torch.is_grad_enabled():
            raise RuntimeError("The FP32 TP reference is inference-only.")

        weight = self.weight
        if self.tp_size > 1:
            x = torch.cat(_all_gather_shards(x, tp_size=self.tp_size, tp_group=self.tp_group), dim=-1)
            weight = torch.cat(
                _all_gather_shards(weight, tp_size=self.tp_size, tp_group=self.tp_group),
                dim=1,
            )

        output = _topology_reference_linear(x, weight)
        if self.tp_size > 1 and self.sequence_parallel:
            tp_rank = torch.distributed.get_rank(group=self.tp_group)
            output = torch.chunk(output, self.tp_size, dim=0)[tp_rank].contiguous()

        output = output.to(x.dtype)
        bias = getattr(self, "bias", None)
        if self.te_return_bias:
            return output, bias
        if self.use_bias and bias is not None:
            output = output + bias
        return output, None


def get_tp_reference_hyena_stack_spec(
    *,
    use_te: bool = True,
    vortex_style_fp8: bool = False,
    unfused_rmsnorm: bool = False,
    plain_row_linear: bool = False,
):
    """Build a Hyena spec with the slow test-only TP oracle substituted for TE linears."""
    if not use_te or vortex_style_fp8 or unfused_rmsnorm or plain_row_linear:
        raise ValueError("The test-only TP oracle requires the standard TE layer specification.")
    with (
        patch.object(
            hyena_layer_specs,
            "TELayerNormColumnParallelLinear",
            TpReferenceLayerNormColumnParallelLinear,
        ),
        patch.object(hyena_layer_specs, "TERowParallelLinear", TpReferenceRowParallelLinear),
    ):
        return hyena_layer_specs.get_hyena_stack_spec()
