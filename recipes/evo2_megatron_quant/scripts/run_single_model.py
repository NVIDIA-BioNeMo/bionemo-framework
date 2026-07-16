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

"""Run the ModelOpt PTQ harness on a single BioNeMo model.

This is a manual GPU runner (needs a CUDA device and an NGC checkpoint), not a
pytest test — the CPU-only L0 test lives in
tests/test_compressed_linear_roundtrip.py.

Loads the model once, gets a BF16 baseline, then quantizes in-place and
compares quality metrics. For each quant method, reloads a fresh model
to ensure clean state. Exits non-zero if any method FAILs or ERRORs.

Usage:
    # Single method
    python scripts/run_single_model.py --model esm2 --quant INT8_DEFAULT_CFG

    # Multiple methods
    python scripts/run_single_model.py --model esm2 \
        --quant INT8_DEFAULT_CFG,INT8_SMOOTHQUANT_CFG

    # Evo2 with all-MLP quantization
    python scripts/run_single_model.py --model evo2 \
        --quant INT8_DEFAULT_CFG --all-mlp
"""

import argparse
import gc
import os
import sys
import time
import traceback

import torch

# Add parent directory to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters import get_adapter, list_models
from src.quantize import quantize_model
from src.metrics import compute_metrics, print_summary, save_csv


def run_single_method(
    adapter, model_name, ckpt_path, tokenizer, input_type,
    input_ids, attention_mask, bf16_logits, quant_name,
    enable_all_mlp=False,
):
    """Test one quantization method by reloading the model fresh.

    This ensures clean state — no leftover quantizers from previous methods.

    Args:
        adapter: ModelAdapter instance.
        model_name: Model name string.
        ckpt_path: Checkpoint path for reloading.
        tokenizer: Tokenizer from first load (reused for data generation).
        input_type: Data type string.
        input_ids: Test input tensor.
        attention_mask: Test attention mask.
        bf16_logits: Baseline logits from unquantized model.
        quant_name: Quantization method name.
        enable_all_mlp: Whether to quantize all MLPs (Evo2).

    Returns:
        Result dict with status, metrics, and timing info.
    """
    result = {
        "model": model_name,
        "quant_method": quant_name,
        "status": "ERROR",
        "error": "",
        "cos_sim_avg": 0.0, "cos_sim_min": 0.0,
        "top1_agree": 0.0, "top5_overlap": 0.0, "mse": 0.0,
        "quant_time_s": 0.0, "infer_time_ms": 0.0,
        "num_quantizers": 0,
        "mode": "all-mlp" if enable_all_mlp else "default",
    }

    model_copy = None
    try:
        # Reload fresh model for this quant method
        model_copy, _, _ = adapter.load_model(ckpt_path)
        result["num_params_m"] = sum(p.numel() for p in model_copy.parameters()) / 1e6

        # Quantize in-place
        seq_len = 256 if model_name == "evo2" else 128
        q_info = quantize_model(
            model_copy, quant_name, adapter, tokenizer,
            enable_all_mlp=enable_all_mlp,
            calib_seq_length=seq_len,
        )
        result.update(q_info)

        # Forward pass with quantized model
        t0 = time.time()
        quant_logits = adapter.run_forward(model_copy, input_ids, attention_mask)
        result["infer_time_ms"] = (time.time() - t0) * 1000

        if quant_logits is None:
            result["error"] = "Forward returned None"
            return result

        # Compare with baseline
        metrics = compute_metrics(bf16_logits, quant_logits)
        result.update(metrics)
        result["status"] = (
            "PASS" if metrics["cos_sim_avg"] >= 0.90 and metrics["top1_agree"] >= 0.50
            else "FAIL"
        )

    except Exception as e:
        result["error"] = traceback.format_exc()[-200:]
        result["status"] = "ERROR"

    finally:
        if model_copy is not None:
            del model_copy
        gc.collect()
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test quantization methods on a BioNeMo model",
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=list_models(),
        help="Model to test",
    )
    parser.add_argument(
        "--quant", type=str, default="INT8_DEFAULT_CFG",
        help="Comma-separated list of quant methods (default: INT8_DEFAULT_CFG)",
    )
    parser.add_argument(
        "--all-mlp", action="store_true",
        help="Quantize all MLP layers (relevant for Evo2 Hyena layers)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="CSV output path (default: results/<model>_quant.csv)",
    )
    args = parser.parse_args()

    methods = [m.strip() for m in args.quant.split(",") if m.strip()]
    if not methods:
        parser.error("--quant did not contain any method names")
    adapter = get_adapter(args.model)

    print(f"\n{'=' * 80}")
    print(f"  {adapter.description}")
    print(f"  Testing {len(methods)} quantization method(s)")
    print(f"  Mode: {'all-mlp' if args.all_mlp else 'default'}")
    print(f"{'=' * 80}")

    # 1. Download checkpoint
    print(f"\n  Downloading {args.model} checkpoint...")
    ckpt_path = adapter.download_checkpoint()

    # 2. Load model for BF16 baseline
    print(f"  Loading model for BF16 baseline...")
    t0 = time.time()
    model, tokenizer, input_type = adapter.load_model(ckpt_path)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ Loaded: {num_params:.1f}M params in {time.time() - t0:.1f}s")

    # 3. Generate test data
    seq_length = 256 if args.model == "evo2" else 128
    input_ids, attention_mask = adapter.generate_test_data(
        tokenizer, batch_size=4, seq_length=seq_length,
    )
    print(f"  ✓ Test data: {input_ids.shape}")

    # 4. BF16 baseline
    print(f"  Running BF16 baseline...")
    bf16_logits = adapter.run_forward(model, input_ids, attention_mask)
    print(f"  ✓ BF16 output: {bf16_logits.shape}")

    # Free baseline model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Test each method
    results = []
    for i, qname in enumerate(methods):
        print(f"\n  [{i+1}/{len(methods)}] {qname}...", end=" ", flush=True)
        result = run_single_method(
            adapter, args.model, ckpt_path, tokenizer, input_type,
            input_ids, attention_mask, bf16_logits, qname,
            enable_all_mlp=args.all_mlp,
        )
        icon = {"PASS": "✅", "FAIL": "❌"}.get(result["status"], "⚠️")
        if result["status"] in ("PASS", "FAIL"):
            print(f"{icon} cos={result['cos_sim_avg']:.4f} "
                  f"top1={result['top1_agree']*100:.1f}% "
                  f"t={result['quant_time_s']:.1f}s")
        else:
            print(f"{icon} {result.get('error', '')[:80]}")
        results.append(result)

    # 6. Summary
    print_summary(results, f"{args.model.upper()} — Quantization Results")

    # 7. Save CSV
    output_path = args.output or f"results/{args.model}_quant.csv"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_csv(results, output_path)

    # 8. Reflect outcome in the exit code so CI/automation can detect failures.
    n_not_pass = sum(1 for r in results if r["status"] != "PASS")
    return 1 if n_not_pass else 0


if __name__ == "__main__":
    sys.exit(main())
