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

"""Inference-only native Blackwell MXFP8 projection kernels for Evo2.

This is deliberately separate from Hopper's Transformer Engine FP8 recipes and Evo2's
vortex-style delayed-scaling mode. Selected BF16 RMSNorm projection weights are block-quantized
once, activations use FlashInfer's fused MXFP8 quantizer, and the projection dispatches the
Blackwell block-scaled Tensor Core GEMM. Generation retains the loaded BF16 projection for
numerically sensitive single-token decode by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn


NativeFP8Policy = Literal["off", "hyena", "fc1", "expansion"]
NativeFP8Decode = Literal["none", "bf16", "fp8"]

_NATIVE_FP8_POLICIES = {"off", "hyena", "fc1", "expansion"}
_NATIVE_FP8_DECODE_MODES = {"none", "bf16", "fp8"}
_FP8_E4M3_MAX = 448.0


@dataclass(frozen=True)
class NativeFP8Preparation:
    """Summary of an inference-only model conversion."""

    policy: str
    module_names: tuple[str, ...] = ()
    original_weight_bytes: int = 0
    quantized_weight_bytes: int = 0
    decode_mode: str = "bf16"

    @property
    def converted_modules(self) -> int:
        """Number of projections converted to native FP8."""
        return len(self.module_names)


def validate_native_fp8_policy(policy: str) -> None:
    """Validate the selective projection policy."""
    if policy not in _NATIVE_FP8_POLICIES:
        expected = ", ".join(sorted(_NATIVE_FP8_POLICIES))
        raise ValueError(f"Unsupported native FP8 policy {policy!r}; expected one of: {expected}")


def validate_native_fp8_decode(decode_mode: str) -> None:
    """Validate whether decode stays BF16 or uses low-latency native FP8."""
    if decode_mode not in _NATIVE_FP8_DECODE_MODES:
        expected = ", ".join(sorted(_NATIVE_FP8_DECODE_MODES))
        raise ValueError(f"Unsupported native FP8 decode mode {decode_mode!r}; expected one of: {expected}")


def validate_native_fp8_precision(
    mixed_precision_config: Any,
    *,
    vortex_style_fp8: bool,
    native_nvfp4_policy: str,
) -> None:
    """Reject nested native, vortex, and Transformer Engine quantization contexts."""
    fp8_enabled = bool(getattr(mixed_precision_config, "fp8", None))
    fp4_enabled = bool(getattr(mixed_precision_config, "fp4", None))
    if vortex_style_fp8 or fp8_enabled or fp4_enabled or native_nvfp4_policy != "off":
        raise ValueError(
            "Selective native FP8 inference requires bf16_mixed and is mutually exclusive with "
            "vortex-style FP8, global FP8/FP4 recipes, and native NVFP4"
        )


def _target_kind(module_name: str) -> str | None:
    if module_name.endswith(".mlp.linear_fc1"):
        return "fc1"
    if module_name.endswith(".mixer.dense_projection"):
        return "hyena_projection"
    if module_name.endswith(".self_attention.linear_qkv"):
        return "attention_qkv"
    return None


def _policy_selects(kind: str | None, policy: str) -> bool:
    if policy == "hyena":
        return kind == "hyena_projection"
    if policy == "fc1":
        return kind == "fc1"
    if policy == "expansion":
        return kind is not None
    return False


def _is_te_layernorm_column_linear(module: nn.Module) -> bool:
    try:
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
    except ImportError:
        return False
    return isinstance(module, TELayerNormColumnParallelLinear)


def _te_rmsnorm(module: "NativeFP8LayerNormColumnLinear", x: torch.Tensor) -> torch.Tensor:
    """Apply the same TE inference RMSNorm kernel used by LayerNormLinear."""
    from transformer_engine.pytorch.module._common import apply_normalization

    normalized, _, _ = apply_normalization(
        inputmat=x,
        ln_out=None,
        ln_weight=module.layer_norm_weight,
        ln_bias=None,
        eps=module.eps,
        output_quantizer=None,
        output_dtype=x.dtype,
        normalization="RMSNorm",
        fwd_ln_sm_margin=module.inf_ln_sm_margin,
        zero_centered_gamma=module.zero_centered_gamma,
    )
    return normalized


def _mxfp8_quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize values and their 32-element block scales with FlashInfer's CUDA kernel."""
    import flashinfer

    return flashinfer.mxfp8_quantize(value, is_sf_swizzled_layout=True, backend="cuda")


def _native_mxfp8_mm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_block_scales: torch.Tensor,
    weight_block_scales: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Call FlashInfer's native Blackwell block-scaled MXFP8 Tensor Core GEMM."""
    import flashinfer

    return flashinfer.mm_mxfp8(
        activation,
        weight,
        activation_block_scales,
        weight_block_scales,
        out_dtype=out_dtype,
        backend="cutlass",
    )


class NativeFP8LayerNormColumnLinear(nn.Module):
    """Inference-only RMSNorm plus native Blackwell block-scaled MXFP8 projection."""

    def __init__(self, source: nn.Module, *, decode_mode: NativeFP8Decode = "bf16") -> None:
        """Prequantize one compatible TE projection."""
        super().__init__()
        validate_native_fp8_decode(decode_mode)
        if source.training:
            raise RuntimeError("Native FP8 conversion is inference-only; call model.eval() before conversion")
        if not torch.cuda.is_available() or source.weight.device.type != "cuda":
            raise RuntimeError("Native FP8 inference requires CUDA-resident weights")
        capability = torch.cuda.get_device_capability(source.weight.device)
        if capability[0] < 10:
            raise RuntimeError(
                "Native MXFP8 inference requires a Blackwell-class GPU; use a Transformer Engine FP8 recipe "
                f"on Hopper, got compute capability {capability}"
            )
        if getattr(source, "normalization", None) != "RMSNorm":
            raise ValueError("Selective native FP8 currently supports fused RMSNorm projections only")
        if bool(getattr(source, "sequence_parallel", False)):
            raise ValueError("Selective native FP8 requires sequence_parallel=False")
        if getattr(source, "te_quant_params", None) is not None:
            raise ValueError("Selective native FP8 requires an unquantized BF16/FP16 TE source module")

        weight = source.weight.detach()
        if weight.ndim != 2 or weight.dtype not in {torch.bfloat16, torch.float16}:
            raise TypeError(f"Native FP8 requires a 2D BF16/FP16 weight, got {weight.shape=} and {weight.dtype=}")
        if weight.shape[0] % 32 or weight.shape[1] % 32:
            raise ValueError(f"Native MXFP8 projection widths must be divisible by 32, got {tuple(weight.shape)}")

        try:
            import flashinfer
        except ImportError as error:
            raise RuntimeError("Selective native MXFP8 requires FlashInfer with Blackwell kernels") from error

        self.in_features = int(weight.shape[1])
        self.out_features = int(getattr(source, "out_features", weight.shape[0]))
        self.local_out_features = int(weight.shape[0])
        self.eps = float(source.eps)
        self.zero_centered_gamma = bool(getattr(source, "zero_centered_gamma", False))
        self.inf_ln_sm_margin = int(getattr(source, "inf_ln_sm_margin", 0))
        self.te_return_bias = bool(getattr(source, "te_return_bias", False))
        self.use_bias = bool(getattr(source, "use_bias", False))
        self.layer_norm_weight = source.layer_norm_weight
        self.bias = source.bias if self.use_bias else None
        self.decode_mode = decode_mode
        self.bf16_decode_module = source if decode_mode == "bf16" else None
        self._flashinfer = flashinfer

        with torch.no_grad():
            weight_mxfp8, weight_block_scales = _mxfp8_quantize(weight)
        decode_weight_fp8 = None
        fp8_activation_global_scale = None
        fp8_gemm_alpha = None
        if decode_mode == "fp8":
            if weight.shape[0] % 128 or weight.shape[1] % 128:
                raise ValueError(
                    "Low-latency native FP8 decode requires input and output widths divisible by 128, "
                    f"got {tuple(weight.shape)}"
                )
            weight_amax = torch.nan_to_num(weight.abs().amax().float(), nan=1.0, posinf=1.0).clamp_min_(1e-12)
            weight_global_scale = torch.full((1,), _FP8_E4M3_MAX, device=weight.device) / weight_amax
            weight_fp8 = (weight.float() * weight_global_scale).to(torch.float8_e4m3fn)
            decode_weight_fp8 = flashinfer.prepare_low_latency_gemm_weights(weight_fp8, {})
            gamma = self.layer_norm_weight.float()
            if self.zero_centered_gamma:
                gamma = gamma + 1.0
            gamma_amax = torch.nan_to_num(gamma.abs().amax(), nan=1.0, posinf=1.0).clamp_min_(1e-12)
            safe_activation_amax = gamma_amax * self.in_features**0.5
            fp8_activation_global_scale = torch.full((1,), _FP8_E4M3_MAX, device=weight.device) / safe_activation_amax
            fp8_gemm_alpha = (fp8_activation_global_scale * weight_global_scale).reciprocal()

        self.register_buffer("weight_mxfp8", weight_mxfp8, persistent=False)
        self.register_buffer("weight_block_scales", weight_block_scales, persistent=False)
        self.register_buffer("decode_weight_fp8", decode_weight_fp8, persistent=False)
        self.register_buffer("fp8_activation_global_scale", fp8_activation_global_scale, persistent=False)
        self.register_buffer("fp8_gemm_alpha", fp8_gemm_alpha, persistent=False)
        self.original_weight_bytes = weight.numel() * weight.element_size()
        self.quantized_weight_bytes = weight_mxfp8.nbytes + weight_block_scales.nbytes
        if decode_weight_fp8 is not None:
            self.quantized_weight_bytes += decode_weight_fp8.nbytes
        self.native_fp8 = True
        nn.Module.train(self, False)

    def train(self, mode: bool = True):
        """Keep the converted module inference-only."""
        if mode:
            raise RuntimeError("Native FP8 projection modules cannot be used for training")
        return super().train(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run exact TE RMSNorm, fused MXFP8 quantization, and a native block-scaled GEMM."""
        if torch.is_grad_enabled():
            raise RuntimeError("Native FP8 projection modules require torch.no_grad() or torch.inference_mode()")
        is_single_token_decode = x.ndim >= 3 and x.shape[0] == 1
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected hidden size {self.in_features}, got input shape {tuple(x.shape)}")
        if self.bf16_decode_module is not None and is_single_token_decode:
            return self.bf16_decode_module(x)

        leading_shape = x.shape[:-1]
        normalized = _te_rmsnorm(self, x.reshape(-1, self.in_features))
        if self.decode_mode == "fp8" and is_single_token_decode:
            if self.fp8_activation_global_scale.dtype is not torch.float32:
                raise RuntimeError("Native FP8 activation and GEMM scales must remain FP32")
            activation_fp8 = (normalized.float() * self.fp8_activation_global_scale).to(torch.float8_e4m3fn)
            output = self._flashinfer.mm_fp8(
                activation_fp8,
                self.decode_weight_fp8,
                self.fp8_gemm_alpha,
                out_dtype=x.dtype,
                backend="trtllm_low_latency",
            )
        else:
            activation_mxfp8, activation_block_scales = _mxfp8_quantize(normalized)
            output = _native_mxfp8_mm(
                activation_mxfp8,
                self.weight_mxfp8.T,
                activation_block_scales,
                self.weight_block_scales,
                out_dtype=x.dtype,
            )
        output = output.view(*leading_shape, self.local_out_features)
        if self.use_bias and not self.te_return_bias:
            output = output + self.bias
        return output, self.bias if self.te_return_bias else None

    def extra_repr(self) -> str:
        """Describe native compute and local tensor-parallel dimensions."""
        return (
            f"in_features={self.in_features}, local_out_features={self.local_out_features}, "
            f"compute=MXFP8-native-Blackwell, decode={self.decode_mode}"
        )


def prepare_model_for_native_fp8_inference(
    model: nn.Module,
    *,
    policy: NativeFP8Policy = "off",
    decode_mode: NativeFP8Decode = "bf16",
) -> NativeFP8Preparation:
    """Replace selected fused expansion projections after loading a BF16 checkpoint."""
    validate_native_fp8_policy(policy)
    validate_native_fp8_decode(decode_mode)
    if policy == "off":
        return NativeFP8Preparation(policy=policy, decode_mode=decode_mode)
    if model.training:
        raise RuntimeError("Native FP8 conversion is inference-only; call model.eval() before conversion")

    targets: list[tuple[str, nn.Module]] = []
    unsupported: list[str] = []
    for name, module in tuple(model.named_modules()):
        if not _policy_selects(_target_kind(name), policy):
            continue
        if _is_te_layernorm_column_linear(module):
            targets.append((name, module))
        else:
            unsupported.append(name)
    if unsupported:
        raise TypeError(
            "Selective native FP8 targets must be TE LayerNormColumnParallelLinear modules: " + ", ".join(unsupported)
        )
    if not targets:
        raise ValueError(f"Native FP8 policy {policy!r} did not match any Evo2 expansion projections")

    converted_names: list[str] = []
    original_weight_bytes = 0
    quantized_weight_bytes = 0
    for name, source in targets:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        replacement = NativeFP8LayerNormColumnLinear(source, decode_mode=decode_mode)
        setattr(parent, child_name, replacement)
        converted_names.append(name)
        original_weight_bytes += int(replacement.original_weight_bytes)
        quantized_weight_bytes += int(replacement.quantized_weight_bytes)

    return NativeFP8Preparation(
        policy=policy,
        module_names=tuple(converted_names),
        original_weight_bytes=original_weight_bytes,
        quantized_weight_bytes=quantized_weight_bytes,
        decode_mode=decode_mode,
    )
