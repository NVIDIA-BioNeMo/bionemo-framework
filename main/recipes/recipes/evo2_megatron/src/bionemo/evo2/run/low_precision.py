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

"""Shared low-precision setup for Evo2 inference entry points."""

from __future__ import annotations

import math
from functools import wraps
from typing import Any, Literal


def _enabled(value: Any) -> bool:
    """Treat MBridge's optional precision fields as enabled only when truthy."""
    return bool(value)


def inference_precision_kind(mixed_precision_config: Any) -> str:
    """Return a compact label for the configured inference compute format."""
    if _enabled(getattr(mixed_precision_config, "fp4", None)):
        return "nvfp4"
    if _enabled(getattr(mixed_precision_config, "fp8", None)):
        recipe = str(getattr(mixed_precision_config, "fp8_recipe", "") or "").lower()
        if "mxfp8" in recipe:
            return "mxfp8"
        return "fp8-all-layers" if getattr(mixed_precision_config, "evo2_fp8_all_layers", False) else "fp8"
    if _enabled(getattr(mixed_precision_config, "bf16", None)):
        return "bf16"
    if _enabled(getattr(mixed_precision_config, "fp16", None)):
        return "fp16"
    return "fp32"


def configure_global_fp8_layer_scope(mixed_precision_config: Any, *, all_layers: bool) -> None:
    """Optionally remove MBridge's BF16 boundary-block exclusions from global TE FP8.

    MBridge's regular current-scaling recipe keeps the first and last Transformer blocks in BF16.
    Evo2 7B is known to tolerate regular FP8 across every compatible Transformer Engine linear, so
    inference exposes an explicit override without changing the upstream recipe's safer default.
    Nonlinear Hyena recurrence, normalization, softmax, embeddings, and other non-TE operations keep
    their configured higher precision.
    """
    if not all_layers:
        return
    if not _enabled(getattr(mixed_precision_config, "fp8", None)):
        raise ValueError("--fp8-all-layers requires a global FP8 mixed-precision recipe")

    mixed_precision_config.first_last_layers_bf16 = False
    mixed_precision_config.num_layers_at_start_in_bf16 = 0
    mixed_precision_config.num_layers_at_end_in_bf16 = 0
    mixed_precision_config.evo2_fp8_all_layers = True


def inference_parameter_storage(mixed_precision_config: Any) -> str:
    """Return whether TE linear weights are stored natively quantized or in BF16."""
    if _enabled(getattr(mixed_precision_config, "fp8_param", None)) or _enabled(
        getattr(mixed_precision_config, "fp4_param", None)
    ):
        return "quantized"
    return "bf16"


def configure_quantized_parameter_storage(mixed_precision_config: Any, storage: str) -> None:
    """Select native quantized weights or BF16 weights with quantized GEMMs.

    ``recipe`` preserves the built-in MBridge recipe. ``bf16`` retains high-precision
    parameters while keeping the recipe's MXFP8/NVFP4 activation and GEMM contexts.
    The latter avoids constructing quantized target tensors before loading a BF16
    distributed checkpoint and is also a useful throughput/accuracy comparator.
    """
    if storage not in {"recipe", "bf16"}:
        raise ValueError(f"Unsupported quantized parameter storage {storage!r}; expected 'recipe' or 'bf16'")
    if storage == "recipe":
        return
    if not (
        _enabled(getattr(mixed_precision_config, "fp8", None))
        or _enabled(getattr(mixed_precision_config, "fp4", None))
    ):
        raise ValueError("BF16 parameter storage override requires an FP8 or FP4 compute recipe")

    for field_name in (
        "fp8_param",
        "fp8_param_gather",
        "fp4_param",
        "fp4_param_gather",
        "reuse_grad_buf_for_mxfp8_param_ag",
    ):
        if hasattr(mixed_precision_config, field_name):
            setattr(mixed_precision_config, field_name, False)


def configure_prediction_sequence_parallel(
    model_provider: Any,
    mixed_precision_config: Any,
    *,
    policy: Literal["auto", "on", "off"] = "auto",
    legacy_disabled: bool = False,
) -> bool:
    """Resolve prediction sequence parallelism and update the model provider."""
    if policy not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported sequence-parallel policy {policy!r}; expected 'auto', 'on', or 'off'")
    if legacy_disabled:
        if policy == "on":
            raise ValueError("--no-sequence-parallel conflicts with --sequence-parallel-policy on")
        policy = "off"

    tp_size = int(getattr(model_provider, "tensor_model_parallel_size", 1) or 1)
    global_quantization = _enabled(getattr(mixed_precision_config, "fp8", None)) or _enabled(
        getattr(mixed_precision_config, "fp4", None)
    )
    # AUTO disables SP for global FP8/FP4 because MCore's current padding shim adds
    # external collectives and double-reduces row outputs; SP-off is correct and faster
    # on the representative 7B TP2 packed workload. Revisit when MCore supplies pad-aware
    # SP with one row reduction: A/B auto/off vs on on H100 TP2 using aligned and
    # unaligned ragged batches, and require log-probability parity and lower wall time.
    enabled = tp_size > 1 and policy != "off"
    if policy == "auto" and global_quantization:
        enabled = False

    model_provider.sequence_parallel = enabled
    return enabled


def validate_inference_precision(
    mixed_precision_config: Any,
    *,
    vortex_style_fp8: bool,
) -> None:
    """Reject precision combinations whose nested quantization contexts are undefined."""
    fp8_enabled = _enabled(getattr(mixed_precision_config, "fp8", None))
    fp4_enabled = _enabled(getattr(mixed_precision_config, "fp4", None))
    if fp8_enabled and fp4_enabled:
        raise ValueError("Global FP8 and FP4 inference recipes are mutually exclusive")
    if vortex_style_fp8 and (fp8_enabled or fp4_enabled):
        raise ValueError(
            "--vortex-style-fp8 and a global FP8/FP4 mixed-precision recipe are mutually exclusive; "
            "use vortex FP8 with bf16_mixed or select one global quantization recipe"
        )


def _fp8_gemm_dims_are_aligned(tensor: Any) -> bool:
    """Match Transformer Engine's regular-FP8 GEMM dimension predicate."""
    shape = tuple(getattr(tensor, "shape", ()))
    return len(shape) >= 2 and math.prod(shape[:-1]) % 8 == 0 and shape[-1] % 16 == 0


def _is_regular_fp8_recipe(mixed_precision_config: Any) -> bool:
    """Return whether current/tensorwise or delayed regular FP8 is active."""
    if not _enabled(getattr(mixed_precision_config, "fp8", None)) or _enabled(
        getattr(mixed_precision_config, "fp4", None)
    ):
        return False
    recipe = str(getattr(mixed_precision_config, "fp8_recipe", "") or "").lower()
    return "tensorwise" in recipe or "delayed" in recipe


def _install_regular_fp8_aligned_fast_path(model: Any) -> int:
    """Bypass MCore's per-linear pad/unpad when TE's flattened GEMM is already legal.

    MCore's compatibility wrapper pads the sequence axis in isolation. Transformer Engine's
    actual regular-FP8 constraint is on the product of all leading dimensions, so a decode input
    shaped ``[1, batch, hidden]`` is already legal whenever the batch supplies eight rows. Keep
    the upstream wrapper for unaligned and sequence-parallel inputs.
    """
    from megatron.core import fp8_utils

    wrapped = 0

    def make_aligned_forward(module, padded_forward, unpadded_forward):
        @wraps(padded_forward)
        def aligned_forward(input_tensor, *args, **kwargs):
            weight = getattr(module, "weight", None)
            if (
                not getattr(module, "sequence_parallel", False)
                and _fp8_gemm_dims_are_aligned(input_tensor)
                and _fp8_gemm_dims_are_aligned(weight)
            ):
                return unpadded_forward(input_tensor, *args, **kwargs)
            return padded_forward(input_tensor, *args, **kwargs)

        return aligned_forward

    for module in model.modules():
        if not isinstance(module, fp8_utils.TE_LINEAR_TYPES) or getattr(
            module, "_evo2_regular_fp8_aligned_fast_path", False
        ):
            continue
        padded_forward = module.forward
        unpadded_forward = getattr(padded_forward, "__wrapped__", None)
        if unpadded_forward is None:
            continue
        module.forward = make_aligned_forward(module, padded_forward, unpadded_forward)
        module._evo2_regular_fp8_aligned_fast_path = True
        wrapped += 1
    return wrapped


def prepare_model_for_quantized_inference(model: Any, mixed_precision_config: Any) -> bool:
    """Pad/unpad arbitrary token counts around globally quantized TE linears.

    Megatron Core's helper retains its historical FP8 name, but its installed
    implementation checks each TE module's active FP8 *or FP4* context. Transformer
    Engine then chooses the alignment from the active recipe. This is required for
    both a one-token decode and a flat packed prefill whose total token count is not
    naturally aligned.

    For regular current/tensorwise and delayed FP8, an outer guard calls the
    unpadded TE forward directly when the flattened sequence-times-batch rows
    and weight already satisfy TE's native constraints. Unaligned or
    sequence-parallel inputs retain the upstream padding fallback.

    Returns:
        ``True`` when the model was wrapped, otherwise ``False``.
    """
    if not (
        _enabled(getattr(mixed_precision_config, "fp8", None))
        or _enabled(getattr(mixed_precision_config, "fp4", None))
    ):
        return False

    from megatron.core.fp8_utils import prepare_model_for_fp8_inference

    prepare_model_for_fp8_inference(model)
    aligned_fast_path_modules = 0
    if _is_regular_fp8_recipe(mixed_precision_config):
        aligned_fast_path_modules = _install_regular_fp8_aligned_fast_path(model)
        mixed_precision_config.evo2_regular_fp8_aligned_fast_path_modules = aligned_fast_path_modules
    model.evo2_regular_fp8_aligned_fast_path_modules = aligned_fast_path_modules
    return True
