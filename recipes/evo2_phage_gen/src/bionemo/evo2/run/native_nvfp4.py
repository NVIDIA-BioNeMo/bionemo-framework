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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: recipes/evo2_megatron/src/bionemo/evo2/run/native_nvfp4.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

"""Inference-only native Blackwell NVFP4 projection kernels for Evo2.

This path intentionally does not enable Transformer Engine's global NVFP4 training recipe. It
prepacks selected BF16 projection weights once, quantizes their activations to NVFP4 inside the
inference CUDA graph, and executes a native W4A4 block-scaled GEMM. ``fc1`` and ``expansion``
replace RMSNorm-fed column projections; the experimental ``full`` policy also replaces FC2 and
Hyena/attention row projections. Recurrent Hyena kernels remain BF16. Autoregressive projections
stay BF16 by default; explicit FP8 and W4A4 decode modes are benchmark-only accuracy comparators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn


NativeNVFP4Policy = Literal["off", "fc1", "expansion", "full"]
NativeNVFP4Decode = Literal["bf16", "fp8", "nvfp4"]

_NATIVE_NVFP4_POLICIES = {"off", "fc1", "expansion", "full"}
_NATIVE_NVFP4_DECODE_MODES = {"bf16", "fp8", "nvfp4"}
_FP4_E2M1_MAX = 6.0
_FP8_E4M3_MAX = 448.0
_NVFP4_COMBINED_RANGE = _FP4_E2M1_MAX * _FP8_E4M3_MAX


@dataclass(frozen=True)
class NativeNVFP4Preparation:
    """Summary of an inference-only model conversion."""

    policy: str
    module_names: tuple[str, ...] = ()
    original_weight_bytes: int = 0
    quantized_weight_bytes: int = 0
    decode_mode: str = "bf16"

    @property
    def converted_modules(self) -> int:
        """Number of projection modules converted to native W4A4."""
        return len(self.module_names)


def _validate_native_nvfp4_activation_amax(activation_amax: float | None) -> None:
    """Validate the optional calibrated RMSNorm range independently of policy selection."""
    if activation_amax is not None and not 0.0 < activation_amax < float("inf"):
        raise ValueError(f"Native NVFP4 activation amax must be finite and positive, got {activation_amax!r}")


def validate_native_nvfp4_policy(policy: str, *, activation_amax: float | None) -> None:
    """Validate the selective projection policy and calibrated RMSNorm range."""
    if policy not in _NATIVE_NVFP4_POLICIES:
        expected = ", ".join(sorted(_NATIVE_NVFP4_POLICIES))
        raise ValueError(f"Unsupported native NVFP4 policy {policy!r}; expected one of: {expected}")
    _validate_native_nvfp4_activation_amax(activation_amax)


def validate_native_nvfp4_decode(decode_mode: str) -> None:
    """Validate whether autoregressive single-token projections stay BF16 or use W4A4."""
    if decode_mode not in _NATIVE_NVFP4_DECODE_MODES:
        expected = ", ".join(sorted(_NATIVE_NVFP4_DECODE_MODES))
        raise ValueError(f"Unsupported native NVFP4 decode mode {decode_mode!r}; expected one of: {expected}")


def native_nvfp4_target_kind(module_name: str) -> str | None:
    """Return the expansion-projection kind selected by a fully qualified module name."""
    if module_name.endswith(".mlp.linear_fc1"):
        return "fc1"
    if module_name.endswith(".mixer.dense_projection"):
        return "hyena_projection"
    if module_name.endswith(".self_attention.linear_qkv"):
        return "attention_qkv"
    if module_name.endswith(".mlp.linear_fc2"):
        return "fc2"
    if module_name.endswith(".mixer.dense"):
        return "hyena_output"
    if module_name.endswith(".self_attention.linear_proj"):
        return "attention_output"
    return None


def rmsnorm_no_clip_amax(
    layer_norm_weight: torch.Tensor,
    *,
    in_features: int,
    zero_centered_gamma: bool,
) -> torch.Tensor:
    """Return a strict activation bound from RMSNorm geometry and its learned gamma.

    Before gamma, each RMS-normalized component is bounded by ``sqrt(in_features)``. Using that
    bound avoids E4M3 global-scale saturation without an activation reduction in every forward.
    """
    gamma = layer_norm_weight.float()
    if zero_centered_gamma:
        gamma = gamma + 1.0
    gamma_amax = torch.nan_to_num(gamma.abs().amax(), nan=1.0, posinf=1.0).clamp_min_(1e-12)
    return gamma_amax * in_features**0.5


def _policy_selects(kind: str | None, policy: str) -> bool:
    if policy == "fc1":
        return kind == "fc1"
    if policy == "expansion":
        return kind in {"fc1", "hyena_projection", "attention_qkv"}
    if policy == "full":
        return kind is not None
    return False


def _is_te_layernorm_column_linear(module: nn.Module) -> bool:
    """Avoid importing Megatron/TE until selective NVFP4 is explicitly requested."""
    try:
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
    except ImportError:
        return False
    return isinstance(module, TELayerNormColumnParallelLinear)


def _is_te_row_parallel_linear(module: nn.Module) -> bool:
    """Return whether a module is MCore's Transformer Engine row-parallel linear."""
    try:
        from megatron.core.extensions.transformer_engine import TERowParallelLinear
    except ImportError:
        return False
    return isinstance(module, TERowParallelLinear)


def _reduce_row_parallel_output(output: torch.Tensor, *, group: Any, tp_size: int) -> torch.Tensor:
    """Sum local row-parallel partials with the same MCore tensor-parallel mapping."""
    if tp_size == 1:
        return output
    from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region

    return reduce_from_tensor_model_parallel_region(output, group=group)


def _te_rmsnorm(module: "NativeNVFP4LayerNormColumnLinear", x: torch.Tensor) -> torch.Tensor:
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


class NativeNVFP4LayerNormColumnLinear(nn.Module):
    """Inference-only RMSNorm plus native Blackwell W4A4 column projection.

    The source BF16 weight is released when the source module is replaced. The small RMSNorm
    parameter and optional bias remain in their checkpoint dtype. FlashInfer's CUDA quantizer emits
    packed E2M1 activations/weights plus swizzled E4M3 block scales, and ``mm_fp4`` is explicitly
    constrained to its CUTLASS NVFP4 backend. The optional FP8 decode comparator uses FlashInfer's
    TensorRT-LLM low-latency E4M3 GEMM with weights converted to its block layout once.
    """

    def __init__(
        self,
        source: nn.Module,
        *,
        activation_amax: float | None = None,
        decode_mode: NativeNVFP4Decode = "bf16",
    ) -> None:
        """Prepack one compatible TE projection and retain the requested decode fallback."""
        super().__init__()
        _validate_native_nvfp4_activation_amax(activation_amax)
        validate_native_nvfp4_decode(decode_mode)

        if source.training:
            raise RuntimeError("Native NVFP4 conversion is inference-only; call model.eval() before conversion")
        if not torch.cuda.is_available() or source.weight.device.type != "cuda":
            raise RuntimeError("Native NVFP4 inference requires CUDA-resident weights")
        capability = torch.cuda.get_device_capability(source.weight.device)
        if capability[0] < 10:
            raise RuntimeError(
                f"Native NVFP4 inference requires a Blackwell-class GPU, got compute capability {capability}"
            )
        if getattr(source, "normalization", None) != "RMSNorm":
            raise ValueError("Selective native NVFP4 currently supports fused RMSNorm projections only")
        if bool(getattr(source, "sequence_parallel", False)):
            raise ValueError("Selective native NVFP4 requires sequence_parallel=False")
        if getattr(source, "te_quant_params", None) is not None:
            raise ValueError("Selective native NVFP4 requires an unquantized BF16/FP16 TE source module")

        weight = source.weight.detach()
        if weight.ndim != 2 or weight.dtype not in {torch.bfloat16, torch.float16}:
            raise TypeError(f"Native NVFP4 requires a 2D BF16/FP16 weight, got {weight.shape=} and {weight.dtype=}")
        if weight.shape[1] % 16:
            raise ValueError(f"Native NVFP4 input width must be divisible by 16, got {weight.shape[1]}")

        try:
            import flashinfer
        except ImportError as error:
            raise RuntimeError("Selective native NVFP4 requires FlashInfer with Blackwell FP4 kernels") from error

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
        # Prefill/predict can use W4A4 while numerically sensitive recurrent single-token decode
        # calls the original loaded TE module. This retains its BF16 weight without copying it.
        self.bf16_decode_module = source if decode_mode == "bf16" else None
        self._flashinfer = flashinfer

        original_weight_bytes = weight.numel() * weight.element_size()
        weight_amax = torch.nan_to_num(weight.abs().amax().float(), nan=1.0, posinf=1.0).clamp_min_(1e-12)
        weight_global_scale = torch.full((1,), _NVFP4_COMBINED_RANGE, device=weight.device) / weight_amax
        with torch.no_grad():
            packed_weight, weight_block_scales = flashinfer.fp4_quantize(
                weight,
                weight_global_scale,
                sf_vec_size=16,
                is_sf_swizzled_layout=True,
                backend="cuda",
            )

        safe_activation_amax = rmsnorm_no_clip_amax(
            self.layer_norm_weight,
            in_features=self.in_features,
            zero_centered_gamma=self.zero_centered_gamma,
        )
        if activation_amax is not None:
            safe_activation_amax = torch.maximum(
                safe_activation_amax,
                torch.tensor(activation_amax, dtype=torch.float32, device=weight.device),
            )
        activation_global_scale = (
            torch.full((1,), _NVFP4_COMBINED_RANGE, dtype=torch.float32, device=weight.device) / safe_activation_amax
        )
        gemm_alpha = (activation_global_scale * weight_global_scale).reciprocal()

        fp8_decode_weight = None
        fp8_activation_global_scale = None
        fp8_gemm_alpha = None
        if decode_mode == "fp8":
            if weight.shape[0] % 128 or weight.shape[1] % 128:
                raise ValueError(
                    "Low-latency FP8 decode requires input and output widths divisible by 128, "
                    f"got {tuple(weight.shape)}"
                )
            fp8_weight_global_scale = (
                torch.full((1,), _FP8_E4M3_MAX, dtype=torch.float32, device=weight.device) / weight_amax
            )
            fp8_weight = (weight.float() * fp8_weight_global_scale).to(torch.float8_e4m3fn)
            fp8_decode_weight = flashinfer.prepare_low_latency_gemm_weights(fp8_weight, {})
            fp8_activation_global_scale = (
                torch.full((1,), _FP8_E4M3_MAX, dtype=torch.float32, device=weight.device) / safe_activation_amax
            )
            fp8_gemm_alpha = (fp8_activation_global_scale * fp8_weight_global_scale).reciprocal()

        self.register_buffer("packed_weight", packed_weight, persistent=False)
        self.register_buffer("weight_block_scales", weight_block_scales, persistent=False)
        self.register_buffer("activation_global_scale", activation_global_scale, persistent=False)
        self.register_buffer("gemm_alpha", gemm_alpha, persistent=False)
        self.register_buffer("fp8_decode_weight", fp8_decode_weight, persistent=False)
        self.register_buffer("fp8_activation_global_scale", fp8_activation_global_scale, persistent=False)
        self.register_buffer("fp8_gemm_alpha", fp8_gemm_alpha, persistent=False)
        self.original_weight_bytes = original_weight_bytes
        self.quantized_weight_bytes = packed_weight.nbytes + weight_block_scales.nbytes
        if fp8_decode_weight is not None:
            self.quantized_weight_bytes += fp8_decode_weight.nbytes
        self.native_w4a4 = True
        nn.Module.train(self, False)

    def train(self, mode: bool = True):
        """Keep the converted module inference-only."""
        if mode:
            raise RuntimeError("Native NVFP4 projection modules cannot be used for training")
        return super().train(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Execute TE RMSNorm, FlashInfer NVFP4 quantization, and native CUTLASS W4A4 GEMM."""
        if torch.is_grad_enabled():
            raise RuntimeError("Native NVFP4 projection modules require torch.no_grad() or torch.inference_mode()")
        is_single_token_decode = x.ndim >= 3 and x.shape[0] == 1
        if self.bf16_decode_module is not None and is_single_token_decode:
            return self.bf16_decode_module(x)
        if self.activation_global_scale.dtype is not torch.float32 or self.gemm_alpha.dtype is not torch.float32:
            raise RuntimeError("Native NVFP4 global scales must remain FP32; convert after model-wide dtype casts")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected hidden size {self.in_features}, got input shape {tuple(x.shape)}")
        leading_shape = x.shape[:-1]
        normalized = _te_rmsnorm(self, x.reshape(-1, self.in_features))
        if self.decode_mode == "fp8" and is_single_token_decode:
            if self.fp8_activation_global_scale.dtype is not torch.float32:
                raise RuntimeError("Native FP8 activation and GEMM scales must remain FP32")
            activation_fp8 = (normalized.float() * self.fp8_activation_global_scale).to(torch.float8_e4m3fn)
            output = self._flashinfer.mm_fp8(
                activation_fp8,
                self.fp8_decode_weight,
                self.fp8_gemm_alpha,
                out_dtype=x.dtype,
                backend="trtllm_low_latency",
            )
            output = output.view(*leading_shape, self.local_out_features)
            if self.use_bias and not self.te_return_bias:
                output = output + self.bias
            return output, self.bias if self.te_return_bias else None

        packed_activation, activation_block_scales = self._flashinfer.fp4_quantize(
            normalized,
            self.activation_global_scale,
            sf_vec_size=16,
            is_sf_swizzled_layout=True,
            backend="cuda",
        )
        output = self._flashinfer.mm_fp4(
            packed_activation,
            self.packed_weight.T,
            activation_block_scales,
            self.weight_block_scales.T,
            self.gemm_alpha,
            out_dtype=x.dtype,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass",
            use_nvfp4=True,
        )
        output = output.view(*leading_shape, self.local_out_features)
        if self.use_bias and not self.te_return_bias:
            output = output + self.bias
        return output, self.bias if self.te_return_bias else None

    def extra_repr(self) -> str:
        """Describe the native compute and local tensor-parallel dimensions."""
        return (
            f"in_features={self.in_features}, local_out_features={self.local_out_features}, "
            f"compute=W4A4-native-NVFP4, decode={self.decode_mode}"
        )


class NativeNVFP4RowParallelLinear(nn.Module):
    """Inference-only native W4A4 replacement for an Evo2 TE row projection.

    Unlike expansion projections, FC2 and the Hyena/attention output projections are not fed by
    RMSNorm, so they have no strict geometry-derived activation bound. The default global scale of
    one represents block maxima up to the full E2M1xE4M3 range (2688) without a reduction in every
    layer. ``activation_amax`` may tighten that range after model-specific calibration. Per-block
    scales still adapt at every call. Tensor-parallel partial outputs are reduced before applying
    the replicated bias, matching ``TERowParallelLinear`` semantics.
    """

    def __init__(
        self,
        source: nn.Module,
        *,
        activation_amax: float | None = None,
        decode_mode: NativeNVFP4Decode = "bf16",
    ) -> None:
        """Prepack one compatible row-parallel projection."""
        super().__init__()
        _validate_native_nvfp4_activation_amax(activation_amax)
        validate_native_nvfp4_decode(decode_mode)
        if source.training:
            raise RuntimeError("Native NVFP4 conversion is inference-only; call model.eval() before conversion")
        if not torch.cuda.is_available() or source.weight.device.type != "cuda":
            raise RuntimeError("Native NVFP4 inference requires CUDA-resident weights")
        capability = torch.cuda.get_device_capability(source.weight.device)
        if capability[0] < 10:
            raise RuntimeError(
                f"Native NVFP4 inference requires a Blackwell-class GPU, got compute capability {capability}"
            )
        if bool(getattr(source, "sequence_parallel", False)):
            raise ValueError("Selective native NVFP4 requires sequence_parallel=False")
        if getattr(source, "te_quant_params", None) is not None:
            raise ValueError("Selective native NVFP4 requires an unquantized BF16/FP16 TE source module")

        weight = source.weight.detach()
        if weight.ndim != 2 or weight.dtype not in {torch.bfloat16, torch.float16}:
            raise TypeError(f"Native NVFP4 requires a 2D BF16/FP16 weight, got {weight.shape=} and {weight.dtype=}")
        if weight.shape[1] % 16:
            raise ValueError(f"Native NVFP4 input width must be divisible by 16, got {weight.shape[1]}")

        try:
            import flashinfer
        except ImportError as error:
            raise RuntimeError("Selective native NVFP4 requires FlashInfer with Blackwell FP4 kernels") from error

        self.in_features = int(weight.shape[1])
        self.out_features = int(getattr(source, "out_features", weight.shape[0]))
        self.local_out_features = int(weight.shape[0])
        self.te_return_bias = bool(getattr(source, "te_return_bias", False))
        self.use_bias = bool(getattr(source, "use_bias", False))
        self.bias = source.bias if self.use_bias else None
        self.decode_mode = decode_mode
        self.bf16_decode_module = source if decode_mode == "bf16" else None
        self.tp_size = int(getattr(source, "tp_size", 1))
        self.tp_group = getattr(source, "_tp_group", None)
        if self.tp_size > 1 and self.tp_group is None:
            raise ValueError("Tensor-parallel native NVFP4 row projections require the source TP process group")
        self._flashinfer = flashinfer

        original_weight_bytes = weight.numel() * weight.element_size()
        weight_amax = torch.nan_to_num(weight.abs().amax().float(), nan=1.0, posinf=1.0).clamp_min_(1e-12)
        weight_global_scale = torch.full((1,), _NVFP4_COMBINED_RANGE, device=weight.device) / weight_amax
        with torch.no_grad():
            packed_weight, weight_block_scales = flashinfer.fp4_quantize(
                weight,
                weight_global_scale,
                sf_vec_size=16,
                is_sf_swizzled_layout=True,
                backend="cuda",
            )

        contraction_amax = activation_amax if activation_amax is not None else _NVFP4_COMBINED_RANGE
        activation_global_scale = torch.full(
            (1,),
            _NVFP4_COMBINED_RANGE / contraction_amax,
            dtype=torch.float32,
            device=weight.device,
        )
        gemm_alpha = (activation_global_scale * weight_global_scale).reciprocal()

        fp8_decode_weight = None
        fp8_activation_global_scale = None
        fp8_gemm_alpha = None
        if decode_mode == "fp8":
            if weight.shape[0] % 128 or weight.shape[1] % 128:
                raise ValueError(
                    "Low-latency FP8 decode requires input and output widths divisible by 128, "
                    f"got {tuple(weight.shape)}"
                )
            fp8_weight_global_scale = (
                torch.full((1,), _FP8_E4M3_MAX, dtype=torch.float32, device=weight.device) / weight_amax
            )
            fp8_weight = (weight.float() * fp8_weight_global_scale).to(torch.float8_e4m3fn)
            fp8_decode_weight = flashinfer.prepare_low_latency_gemm_weights(fp8_weight, {})
            fp8_activation_global_scale = torch.full(
                (1,),
                _FP8_E4M3_MAX / contraction_amax,
                dtype=torch.float32,
                device=weight.device,
            )
            fp8_gemm_alpha = (fp8_activation_global_scale * fp8_weight_global_scale).reciprocal()

        self.register_buffer("packed_weight", packed_weight, persistent=False)
        self.register_buffer("weight_block_scales", weight_block_scales, persistent=False)
        self.register_buffer("activation_global_scale", activation_global_scale, persistent=False)
        self.register_buffer("gemm_alpha", gemm_alpha, persistent=False)
        self.register_buffer("fp8_decode_weight", fp8_decode_weight, persistent=False)
        self.register_buffer("fp8_activation_global_scale", fp8_activation_global_scale, persistent=False)
        self.register_buffer("fp8_gemm_alpha", fp8_gemm_alpha, persistent=False)
        self.original_weight_bytes = original_weight_bytes
        self.quantized_weight_bytes = packed_weight.nbytes + weight_block_scales.nbytes
        if fp8_decode_weight is not None:
            self.quantized_weight_bytes += fp8_decode_weight.nbytes
        self.native_w4a4 = True
        nn.Module.train(self, False)

    def train(self, mode: bool = True):
        """Keep the converted module inference-only."""
        if mode:
            raise RuntimeError("Native NVFP4 projection modules cannot be used for training")
        return super().train(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Quantize a row-projection input, run native W4A4, and reduce TP partials."""
        if torch.is_grad_enabled():
            raise RuntimeError("Native NVFP4 projection modules require torch.no_grad() or torch.inference_mode()")
        is_single_token_decode = x.ndim >= 3 and x.shape[0] == 1
        if self.bf16_decode_module is not None and is_single_token_decode:
            return self.bf16_decode_module(x)
        if self.activation_global_scale.dtype is not torch.float32 or self.gemm_alpha.dtype is not torch.float32:
            raise RuntimeError("Native NVFP4 global scales must remain FP32; convert after model-wide dtype casts")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected local input size {self.in_features}, got input shape {tuple(x.shape)}")

        leading_shape = x.shape[:-1]
        activation = x.reshape(-1, self.in_features)
        if self.decode_mode == "fp8" and is_single_token_decode:
            if self.fp8_activation_global_scale.dtype is not torch.float32:
                raise RuntimeError("Native FP8 activation and GEMM scales must remain FP32")
            activation_fp8 = (activation.float() * self.fp8_activation_global_scale).to(torch.float8_e4m3fn)
            output = self._flashinfer.mm_fp8(
                activation_fp8,
                self.fp8_decode_weight,
                self.fp8_gemm_alpha,
                out_dtype=x.dtype,
                backend="trtllm_low_latency",
            )
        else:
            packed_activation, activation_block_scales = self._flashinfer.fp4_quantize(
                activation,
                self.activation_global_scale,
                sf_vec_size=16,
                is_sf_swizzled_layout=True,
                backend="cuda",
            )
            output = self._flashinfer.mm_fp4(
                packed_activation,
                self.packed_weight.T,
                activation_block_scales,
                self.weight_block_scales.T,
                self.gemm_alpha,
                out_dtype=x.dtype,
                block_size=16,
                use_8x4_sf_layout=False,
                backend="cutlass",
                use_nvfp4=True,
            )
        output = output.view(*leading_shape, self.local_out_features)
        output = _reduce_row_parallel_output(output, group=self.tp_group, tp_size=self.tp_size)
        if self.use_bias and not self.te_return_bias:
            output = output + self.bias
        return output, self.bias if self.te_return_bias else None

    def extra_repr(self) -> str:
        """Describe native compute and local tensor-parallel dimensions."""
        return (
            f"in_features={self.in_features}, local_out_features={self.local_out_features}, "
            f"compute=W4A4-native-NVFP4-row, TP={self.tp_size}, decode={self.decode_mode}"
        )


def prepare_model_for_native_nvfp4_inference(
    model: nn.Module,
    *,
    policy: NativeNVFP4Policy = "off",
    activation_amax: float | None = None,
    decode_mode: NativeNVFP4Decode = "bf16",
) -> NativeNVFP4Preparation:
    """Replace selected expansion or full projection sets after loading a BF16 checkpoint."""
    validate_native_nvfp4_policy(policy, activation_amax=activation_amax)
    validate_native_nvfp4_decode(decode_mode)
    if policy == "off":
        return NativeNVFP4Preparation(policy=policy, decode_mode=decode_mode)
    if model.training:
        raise RuntimeError("Native NVFP4 conversion is inference-only; call model.eval() before conversion")

    targets: list[tuple[str, nn.Module, str]] = []
    unsupported: list[str] = []
    for name, module in tuple(model.named_modules()):
        kind = native_nvfp4_target_kind(name)
        if not _policy_selects(kind, policy):
            continue
        is_expansion = kind in {"fc1", "hyena_projection", "attention_qkv"}
        if is_expansion and _is_te_layernorm_column_linear(module):
            targets.append((name, module, "expansion"))
        elif not is_expansion and _is_te_row_parallel_linear(module):
            targets.append((name, module, "contraction"))
        else:
            unsupported.append(name)

    if unsupported:
        names = ", ".join(unsupported)
        raise TypeError(
            "Selective native NVFP4 targets must be TE LayerNormColumnParallelLinear expansion or "
            f"TERowParallelLinear contraction modules: {names}"
        )
    if not targets:
        raise ValueError(f"Native NVFP4 policy {policy!r} did not match any Evo2 projections")

    converted_names: list[str] = []
    original_weight_bytes = 0
    quantized_weight_bytes = 0
    for name, source, projection_kind in targets:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        replacement_type = (
            NativeNVFP4LayerNormColumnLinear if projection_kind == "expansion" else NativeNVFP4RowParallelLinear
        )
        replacement = replacement_type(source, activation_amax=activation_amax, decode_mode=decode_mode)
        setattr(parent, child_name, replacement)
        converted_names.append(name)
        original_weight_bytes += int(getattr(replacement, "original_weight_bytes", 0))
        quantized_weight_bytes += int(getattr(replacement, "quantized_weight_bytes", 0))

    return NativeNVFP4Preparation(
        policy=policy,
        module_names=tuple(converted_names),
        original_weight_bytes=original_weight_bytes,
        quantized_weight_bytes=quantized_weight_bytes,
        decode_mode=decode_mode,
    )
