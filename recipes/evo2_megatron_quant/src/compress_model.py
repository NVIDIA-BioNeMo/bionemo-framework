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

"""Compress an Evo2 model by replacing nn.Linear layers with real low-precision storage.

Pipeline:
  1. Load Evo2 in BF16
  2. (Optional) Run ModelOpt calibration to get optimal scales
  3. Walk all nn.Linear / TE Linear layers
  4. Replace with INT8Linear (real int8 weight storage)
  5. Return compressed model with real memory savings
"""

import gc

import torch
import torch.nn as nn

try:
    from .compressed_linear import METHOD_TO_PRECISION, compress_linear
except ImportError:  # allow running with src/ placed directly on sys.path
    from compressed_linear import METHOD_TO_PRECISION, compress_linear


def get_memory_mb():
    """Current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / 1e6


def compress_model(model: nn.Module, precision: str = "int8",
                   skip_patterns: tuple = ("embedding", "layernorm", "ln_", "norm"),
                   verbose: bool = True, **kwargs) -> dict:
    """Replace all nn.Linear layers with compressed versions.

    Args:
        model: PyTorch model (Evo2)
        precision: Target precision (currently "int8")
        skip_patterns: Module name patterns to skip (embeddings, norms)
        verbose: Print compression stats
        **kwargs: Extra args forwarded to the compressed linear constructor

    Returns:
        dict with compression statistics
    """
    stats = {
        "precision": precision,
        "total_linears": 0,
        "compressed_linears": 0,
        "skipped_linears": 0,
        "mem_before_mb": get_memory_mb(),
        "params_compressed": 0,
        "params_skipped": 0,
    }

    # Collect all (parent, name, linear) triples
    replacements = []
    for name, module in model.named_modules():
        # Check for nn.Linear or TE Linear
        is_linear = isinstance(module, nn.Linear)
        if not is_linear:
            # Check for TransformerEngine Linear
            try:
                import transformer_engine.pytorch as te
                is_linear = isinstance(module, te.Linear)
            except ImportError:
                pass

        if not is_linear:
            continue

        stats["total_linears"] += 1

        # Skip patterns
        name_lower = name.lower()
        if any(p in name_lower for p in skip_patterns):
            stats["skipped_linears"] += 1
            param_count = sum(p.numel() for p in module.parameters())
            stats["params_skipped"] += param_count
            continue

        replacements.append((name, module))

    if verbose:
        print(f"  Found {stats['total_linears']} Linear layers, "
              f"compressing {len(replacements)}, "
              f"skipping {stats['skipped_linears']}")

    # Do replacements — free old weights aggressively
    for idx, (full_name, linear) in enumerate(replacements):
        parts = full_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]

        param_count = linear.weight.numel()

        # Extract weight and bias from any Linear type
        weight_data = linear.weight.data
        bias_data = None
        if hasattr(linear, 'bias') and linear.bias is not None:
            b = linear.bias
            if hasattr(b, 'data') and b.data.numel() > 0:
                bias_data = b.data
                param_count += b.data.numel()

        in_f = weight_data.shape[1]
        out_f = weight_data.shape[0]

        # Create a wrapper nn.Linear on meta device, then assign weight/bias
        has_bias = bias_data is not None
        temp = nn.Linear(in_f, out_f, bias=has_bias, device='meta')
        temp.weight = nn.Parameter(weight_data, requires_grad=False)
        if has_bias:
            temp.bias = nn.Parameter(bias_data, requires_grad=False)

        # Compress
        compressed = compress_linear(temp, precision, **kwargs)

        # Free old weight memory
        linear.weight.data = torch.empty(0, device='cpu')
        if hasattr(linear, 'bias') and linear.bias is not None:
            try:
                linear.bias.data = torch.empty(0, device='cpu')
            except Exception:
                pass
        del temp

        # Replace in model
        setattr(parent, attr_name, compressed)
        del linear, compressed

        stats["compressed_linears"] += 1
        stats["params_compressed"] += param_count

        # Periodic GC
        if (idx + 1) % 8 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    stats["mem_after_mb"] = get_memory_mb()
    stats["mem_saved_mb"] = stats["mem_before_mb"] - stats["mem_after_mb"]
    stats["mem_saved_pct"] = (stats["mem_saved_mb"] / stats["mem_before_mb"] * 100
                              if stats["mem_before_mb"] > 0 else 0)

    if verbose:
        print(f"  Compressed {stats['compressed_linears']} layers ({stats['params_compressed']:,} params)")
        print(f"  Memory: {stats['mem_before_mb']:.0f} → {stats['mem_after_mb']:.0f} MB "
              f"(saved {stats['mem_saved_mb']:.0f} MB, {stats['mem_saved_pct']:.1f}%)")

    return stats


def compress_from_method(model: nn.Module, method_name: str,
                         verbose: bool = True, **kwargs) -> dict:
    """Compress model using precision determined by ModelOpt method name.

    Args:
        model: PyTorch model
        method_name: One of the 22 ModelOpt method names (e.g. "FP8_DEFAULT_CFG")
        verbose: Print stats
    """
    precision = METHOD_TO_PRECISION.get(method_name)
    if precision is None:
        raise ValueError(f"Unknown method: {method_name}. "
                         f"Known: {list(METHOD_TO_PRECISION.keys())}")
    if verbose:
        print(f"  Method {method_name} → precision: {precision}")
    return compress_model(model, precision, verbose=verbose, **kwargs)
