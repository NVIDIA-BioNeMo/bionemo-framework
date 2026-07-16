#!/usr/bin/env python3
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

"""ModelOpt quantization wrapper for BioNeMo models.

Provides a clean interface to NVIDIA ModelOpt's quantization APIs, including:
  - All 22 built-in quantization configurations
  - Custom config support (e.g., quantizing all MLPs for Evo2)
  - Calibration forward loop construction from adapters
  - In-place quantization with error handling

Usage:
    from src.quantize import quantize_model, ALL_QUANT_METHODS, get_quant_config
    from src.adapters import get_adapter

    adapter = get_adapter("esm2")
    model, tok, _ = adapter.load_model(ckpt)
    quantize_model(model, "FP8_DEFAULT_CFG", adapter, tok)
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List

import torch

try:
    import modelopt.torch.quantization as mtq
except ImportError as e:
    raise ImportError(
        f"NVIDIA ModelOpt not found: {e}\n"
        "Install with: pip install nvidia-modelopt[all]"
    )


# ============================================================================
# All 22 ModelOpt quantization configurations
# ============================================================================

ALL_QUANT_METHODS: List[str] = [
    # --- FP8 family (E4M3 / E5M2) ---
    "FP8_DEFAULT_CFG",                      # FP8 E4M3 per-tensor
    "FP8_PER_CHANNEL_PER_TOKEN_CFG",        # FP8 per-channel weights, per-token activations
    "FP8_KV_CFG",                           # FP8 with KV cache quantization
    "FP8_AFFINE_KV_CFG",                    # FP8 with affine KV quantization
    "FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG",     # FP8 blockwise weight-only

    # --- INT8 family ---
    "INT8_DEFAULT_CFG",                     # INT8 per-tensor symmetric
    "INT8_SMOOTHQUANT_CFG",                 # INT8 SmoothQuant (migrates difficulty)

    # --- INT4 family ---
    "INT4_AWQ_CFG",                         # INT4 Activation-aware Weight Quantization
    "INT4_BLOCKWISE_WEIGHT_ONLY_CFG",       # INT4 blockwise weight-only

    # --- NVFP4 family (NVIDIA FP4) ---
    "NVFP4_DEFAULT_CFG",                    # NVFP4 default
    "NVFP4_AWQ_LITE_CFG",                   # NVFP4 with AWQ lite calibration
    "NVFP4_AWQ_CLIP_CFG",                   # NVFP4 with AWQ clip calibration
    "NVFP4_AWQ_FULL_CFG",                   # NVFP4 with full AWQ calibration
    "NVFP4_KV_CFG",                         # NVFP4 with KV cache quantization
    "NVFP4_KV_ROTATE_CFG",                  # NVFP4 with rotated KV quantization
    "NVFP4_AFFINE_KV_CFG",                  # NVFP4 with affine KV quantization
    "NVFP4_SVDQUANT_DEFAULT_CFG",           # NVFP4 with SVD quantization

    # --- MX (Microscaling) family ---
    "MXFP4_DEFAULT_CFG",                    # Microscaling FP4
    "MXFP6_DEFAULT_CFG",                    # Microscaling FP6
    "MXFP8_DEFAULT_CFG",                    # Microscaling FP8
    "MXINT8_DEFAULT_CFG",                   # Microscaling INT8

    # --- Mixed precision ---
    "W4A8_AWQ_BETA_CFG",                    # 4-bit weights + 8-bit activations
]

# Friendly name → ModelOpt constant name mapping
FRIENDLY_NAMES: Dict[str, str] = {
    "fp8": "FP8_DEFAULT_CFG",
    "fp8_per_channel": "FP8_PER_CHANNEL_PER_TOKEN_CFG",
    "int8": "INT8_DEFAULT_CFG",
    "int8_smoothquant": "INT8_SMOOTHQUANT_CFG",
    "int4_awq": "INT4_AWQ_CFG",
    "int4_blockwise": "INT4_BLOCKWISE_WEIGHT_ONLY_CFG",
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "w4a8": "W4A8_AWQ_BETA_CFG",
}


def get_quant_config(
    method_name: str,
    enable_all_mlp: bool = False,
) -> dict:
    """Get a ModelOpt quantization config by name.

    Args:
        method_name: Either a ModelOpt constant name (e.g., "FP8_DEFAULT_CFG")
                     or a friendly name (e.g., "fp8"). Surrounding whitespace
                     is ignored.
        enable_all_mlp: If True, removes the 'default: enable=False' entry
                        from the config to quantize ALL MLP layers
                        (relevant for Evo2's Hyena layers).

    Returns:
        Deep copy of the quantization config dict.

    Raises:
        ValueError: If the method name is not recognized.
    """
    # Normalize, then resolve a friendly name if one was given.
    method_name = method_name.strip()
    resolved = FRIENDLY_NAMES.get(method_name, method_name)

    # Only whitelisted configs may reach ModelOpt: getattr() on an arbitrary
    # string could otherwise return an unrelated mtq attribute.
    if resolved not in ALL_QUANT_METHODS:
        raise ValueError(
            f"Unknown quantization method: '{method_name}'. "
            f"Available: {ALL_QUANT_METHODS}"
        )

    cfg = getattr(mtq, resolved, None)
    if cfg is None:
        raise ValueError(
            f"'{resolved}' is not available in this ModelOpt build. "
            f"Available: {ALL_QUANT_METHODS}"
        )

    cfg = copy.deepcopy(cfg)

    if enable_all_mlp:
        # ModelOpt configs carry a 'default: {enable: False}' entry that stops
        # modules not matched by an explicit wildcard from being quantized.
        # Dropping it lets all MLP layers be quantized, including the Hyena
        # layers' TE ColumnParallelLinear / RowParallelLinear.
        #
        # This assumes quant_cfg is a mapping. If a ModelOpt version uses a
        # different layout we fail loudly rather than silently doing nothing:
        # a silent no-op would report "all-MLP" numbers for a default-scope run.
        quant_cfg = cfg.get("quant_cfg")
        if isinstance(quant_cfg, dict) and "default" in quant_cfg:
            del quant_cfg["default"]
        else:
            raise RuntimeError(
                f"enable_all_mlp: no 'default' entry found in quant_cfg "
                f"(type={type(quant_cfg).__name__}) for '{resolved}'. This "
                "ModelOpt version likely uses a different config layout, so "
                "--all-mlp would have no effect; refusing to report misleading "
                "all-MLP results."
            )

    return cfg


def build_calibration_loop(
    adapter: Any,
    tokenizer: Any,
    model_name: str,
    num_batches: int = 4,
    batch_size: int = 2,
    seq_length: int = 128,
) -> Callable:
    """Build a calibration forward loop function for mtq.quantize().

    ModelOpt's mtq.quantize() requires a callable that drives the model
    through representative inputs so it can collect activation statistics
    for calibration (needed by methods like SmoothQuant, AWQ, etc.).

    Args:
        adapter: ModelAdapter instance.
        tokenizer: Tokenizer from adapter.load_model().
        model_name: Model name string.
        num_batches: Number of calibration batches.
        batch_size: Samples per batch.
        seq_length: Sequence length per sample.

    Returns:
        Callable that takes a model and runs calibration forward passes.
    """
    # Pre-generate calibration data
    calib_batches = []
    for _ in range(num_batches):
        ids, mask = adapter.generate_test_data(tokenizer, batch_size, seq_length)
        fwd_args = adapter.build_calibration_forward_args(ids, mask)
        calib_batches.append(fwd_args)

    def forward_loop(model):
        """Drive model through calibration data for quantizer statistics."""
        failures = 0
        for kwargs in calib_batches:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                try:
                    model(**kwargs)
                except Exception as e:
                    failures += 1
                    print(f"    [calib] forward pass failed: {type(e).__name__}: {e}")
        if failures == len(calib_batches):
            print(
                f"    [calib] WARNING: all {failures} calibration passes failed; "
                "quantization scales are uncalibrated."
            )

    return forward_loop


def quantize_model(
    model: torch.nn.Module,
    method_name: str,
    adapter: Any,
    tokenizer: Any,
    enable_all_mlp: bool = False,
    num_calib_batches: int = 4,
    calib_batch_size: int = 2,
    calib_seq_length: int = 128,
) -> Dict[str, Any]:
    """Quantize a model in-place using the specified ModelOpt method.

    Args:
        model: BioNeMo model (will be modified in-place).
        method_name: Quantization method (e.g., "FP8_DEFAULT_CFG").
        adapter: ModelAdapter instance for data generation.
        tokenizer: Tokenizer for generating calibration data.
        enable_all_mlp: If True, quantize all MLP layers (for Evo2).
        num_calib_batches: Number of calibration batches.
        calib_batch_size: Batch size for calibration.
        calib_seq_length: Sequence length for calibration.

    Returns:
        Dict with 'quant_time_s'.
    """
    cfg = get_quant_config(method_name, enable_all_mlp=enable_all_mlp)

    forward_loop = build_calibration_loop(
        adapter=adapter,
        tokenizer=tokenizer,
        model_name=adapter.name,
        num_batches=num_calib_batches,
        batch_size=calib_batch_size,
        seq_length=calib_seq_length,
    )

    t0 = time.time()
    mtq.quantize(model, cfg, forward_loop=forward_loop)
    quant_time = time.time() - t0

    return {
        "quant_time_s": quant_time,
    }
