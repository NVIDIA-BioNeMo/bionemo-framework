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

"""Quality metrics for comparing BF16 baseline vs. quantized model outputs.

Metrics computed:
  - Cosine Similarity (avg & min): measures directional alignment of logit vectors.
    A value of 1.0 means the logits point in the exact same direction.
  - Top-1 Agreement: fraction of positions where the predicted token matches.
    Directly measures whether quantization changes the model's predictions.
  - Top-5 Overlap: fraction of top-5 predicted tokens that overlap.
    More lenient than top-1 — captures whether quantization preserves ranking.
  - MSE: mean squared error between logit values.
    Measures raw numerical difference (sensitive to scale).

Usage:
    from src.metrics import compute_metrics, print_summary, save_csv

    metrics = compute_metrics(bf16_logits, quant_logits)
    print_summary(results)
    save_csv(results, "results.csv")
"""

import csv
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


def compute_metrics(
    bf16_logits: torch.Tensor,
    quant_logits: torch.Tensor,
) -> Dict[str, float]:
    """Compare BF16 baseline logits against quantized model logits.

    Args:
        bf16_logits: Baseline logits tensor, shape [B, L, V] or [B*L, V].
        quant_logits: Quantized model logits, same shape as bf16_logits.

    Returns:
        Dictionary with keys: cos_sim_avg, cos_sim_min, top1_agree,
        top5_overlap, mse.
    """
    # Flatten to [N, vocab_size] for per-position comparison
    bf = bf16_logits.reshape(-1, bf16_logits.shape[-1]).float()
    qf = quant_logits.reshape(-1, quant_logits.shape[-1]).float()

    # Cosine similarity: per-position, then aggregate
    cos = F.cosine_similarity(bf, qf, dim=-1)

    # Top-1 agreement: does the argmax match?
    top1_bf = bf.argmax(dim=-1)
    top1_q = qf.argmax(dim=-1)

    # Top-5 overlap: what fraction of top-5 tokens overlap?
    top5_bf = bf.topk(5, dim=-1).indices
    top5_q = qf.topk(5, dim=-1).indices
    overlap = sum(
        len(set(top5_bf[i].tolist()) & set(top5_q[i].tolist())) / 5.0
        for i in range(top5_bf.shape[0])
    ) / top5_bf.shape[0]

    return {
        "cos_sim_avg": cos.mean().item(),
        "cos_sim_min": cos.min().item(),
        "top1_agree": (top1_bf == top1_q).float().mean().item(),
        "top5_overlap": overlap,
        "mse": F.mse_loss(bf, qf).item(),
    }


def print_summary(
    results: List[Dict[str, Any]],
    title: str = "Quantization Results",
) -> None:
    """Print a formatted summary table of quantization results.

    Args:
        results: List of result dicts, each containing 'quant_method',
                 'status', 'cos_sim_avg', 'top1_agree', etc.
        title: Title for the summary table.
    """
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(
        f"  {'Method':<35} {'Status':>7} {'CosSim':>8} {'Top1%':>7} "
        f"{'Top5%':>7} {'MSE':>10} {'#Q':>4} {'Time':>7}"
    )
    print(
        f"  {'-'*35} {'-'*7} {'-'*8} {'-'*7} "
        f"{'-'*7} {'-'*10} {'-'*4} {'-'*7}"
    )

    for r in results:
        if r["status"] in ("PASS", "FAIL"):
            icon = "✅" if r["status"] == "PASS" else "❌"
            nq = r.get("num_quantizers", r.get("num_quant_modules", 0))
            print(
                f"  {r['quant_method']:<35} {icon:>7} "
                f"{r['cos_sim_avg']:>8.4f} "
                f"{r['top1_agree']*100:>6.1f}% "
                f"{r.get('top5_overlap', 0)*100:>6.1f}% "
                f"{r['mse']:>10.6f} "
                f"{nq:>4} "
                f"{r.get('quant_time_s', 0):>6.1f}s"
            )
        else:
            err = r.get("error", "")[:40]
            print(
                f"  {r['quant_method']:<35} {'⚠️':>7} "
                f"{'—':>8} {'—':>7} {'—':>7} {'—':>10} {'—':>4} {err}"
            )

    print(f"{'=' * 100}")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_err = sum(1 for r in results if r["status"] == "ERROR")
    print(f"  Total: {n_pass} PASS, {n_fail} FAIL, {n_err} ERROR out of {len(results)}")


def save_csv(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Save quantization results to a CSV file.

    Args:
        results: List of result dictionaries.
        output_path: Path to the output CSV file.
    """
    if not results:
        return

    fieldnames = [
        "model", "quant_method", "status",
        "cos_sim_avg", "cos_sim_min", "top1_agree", "top5_overlap", "mse",
        "num_quantizers", "num_quant_modules",
        "quant_time_s", "infer_time_ms", "num_params_m",
        "mode", "error",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  📄 Results saved to: {output_path}")
