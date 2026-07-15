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

"""Benchmark Evo2 with REAL INT8 weight compression.

Unlike ModelOpt simulated quantization, this script truly stores weights in
int8, achieving real GPU-memory savings (~2x on the compressed Linear layers).

For each config:
  1. Load Evo2 in BF16
  2. Run BF16 inference → baseline logits
  3. Compress weights to int8 (in-place)
  4. Measure real memory savings
  5. Run compressed inference → compare quality
  6. Binary search for max sequence length

Usage (inside BioNeMo Docker):
  python scripts/benchmark_real_quant.py --precisions int8
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time


# === Subprocess worker ===
WORKER_SCRIPT = r'''
import gc, json, os, sys, time
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", os.environ.get("WORKER_PORT", "29500"))
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
import torch.nn.functional as F
import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", world_size=1, rank=0)
from megatron.core import parallel_state
if not parallel_state.is_initialized():
    parallel_state.initialize_model_parallel(1, 1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(42)

import nemo.lightning as nl
from bionemo.core.data.load import load as bionemo_load
from bionemo.evo2.data.tokenizer import Evo2Tokenizer

# Add src to path
sys.path.insert(0, "/workspace/quantization/src")
from compressed_linear import METHOD_TO_PRECISION
from compress_model import compress_model

# Parse args
precision = sys.argv[1]          # int8
method_name = sys.argv[2]        # e.g. FP8_DEFAULT_CFG
eval_seq_len = int(sys.argv[3])  # seq length for quality eval
max_seqlen_search = sys.argv[4] == "1"  # whether to binary-search max seqlen
output_file = sys.argv[5]

result = {
    "method": method_name,
    "precision": precision,
    "eval_seq_len": eval_seq_len,
    "status": "ERROR", "error": "",
    # Memory
    "mem_bf16_mb": 0, "mem_compressed_mb": 0, "mem_saved_mb": 0, "mem_saved_pct": 0,
    # Quality
    "cos_sim_avg": 0.0, "cos_sim_min": 0.0, "top1_agree": 0.0, "mse": 0.0,
    # Speed
    "infer_bf16_ms": 0.0, "infer_compressed_ms": 0.0,
    # Compression stats
    "linears_compressed": 0, "params_compressed": 0,
    # Max sequence length
    "max_seqlen": 0, "max_seqlen_peak_gb": 0.0,
}

try:
    # Load model
    ckpt = str(bionemo_load("evo2/7b-8k:1.0", source="ngc"))
    ctx = nl.io.load_context(ckpt)
    config = ctx.model.config
    config.initial_ckpt_path = ckpt
    tokenizer = Evo2Tokenizer()
    tokenizer.vocab_size = tokenizer.tokenizer.vocab_size
    model = config.configure_model(tokenizer)
    model = model.bfloat16().cuda().eval()

    mem_bf16 = torch.cuda.memory_allocated() / 1e6
    result["mem_bf16_mb"] = mem_bf16

    # Generate eval input
    input_ids = torch.randint(0, 512, (1, eval_seq_len), dtype=torch.long).cuda()
    position_ids = torch.arange(eval_seq_len, device="cuda").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    # BF16 baseline
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # Warmup
        warmup_out = model(input_ids=input_ids[:, :256], position_ids=position_ids[:, :256],
                           attention_mask=attention_mask[:, :256])
        torch.cuda.synchronize()
        del warmup_out

        t0 = time.perf_counter()
        bf16_out = model(input_ids=input_ids, position_ids=position_ids,
                         attention_mask=attention_mask)
        torch.cuda.synchronize()
        result["infer_bf16_ms"] = (time.perf_counter() - t0) * 1000

    bf16_logits = bf16_out if isinstance(bf16_out, torch.Tensor) else \
        getattr(bf16_out, "token_logits", getattr(bf16_out, "logits", bf16_out))

    # Compress model
    print(f"  Compressing to {precision}...")
    stats = compress_model(model, precision=precision, verbose=True)
    result["mem_compressed_mb"] = stats["mem_after_mb"]
    result["mem_saved_mb"] = stats["mem_saved_mb"]
    result["mem_saved_pct"] = stats["mem_saved_pct"]
    result["linears_compressed"] = stats["compressed_linears"]
    result["params_compressed"] = stats["params_compressed"]

    # Compressed inference
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        t0 = time.perf_counter()
        q_out = model(input_ids=input_ids, position_ids=position_ids,
                      attention_mask=attention_mask)
        torch.cuda.synchronize()
        result["infer_compressed_ms"] = (time.perf_counter() - t0) * 1000

    q_logits = q_out if isinstance(q_out, torch.Tensor) else \
        getattr(q_out, "token_logits", getattr(q_out, "logits", q_out))

    # Quality metrics
    bf = bf16_logits.reshape(-1, bf16_logits.shape[-1]).float()
    qf = q_logits.reshape(-1, q_logits.shape[-1]).float()
    cos = F.cosine_similarity(bf, qf, dim=-1)
    result["cos_sim_avg"] = cos.mean().item()
    result["cos_sim_min"] = cos.min().item()
    top1_bf = bf.argmax(dim=-1)
    top1_q = qf.argmax(dim=-1)
    result["top1_agree"] = (top1_bf == top1_q).float().mean().item()
    result["mse"] = F.mse_loss(bf, qf).item()

    # Free eval tensors
    del bf16_logits, q_logits, bf16_out, q_out, bf, qf, input_ids, position_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()

    # Max sequence length: linear sweep over candidate lengths; report the largest that fits.
    if max_seqlen_search:
        print("  Sweeping candidate sequence lengths...")
        test_lens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        best_len = 0
        best_peak = 0.0

        for slen in test_lens:
            ids = pos = mask = probe_out = None
            oom = False
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                ids = torch.randint(0, 512, (1, slen), dtype=torch.long).cuda()
                pos = torch.arange(slen, device="cuda").unsqueeze(0)
                mask = torch.ones_like(ids)

                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    probe_out = model(input_ids=ids, position_ids=pos, attention_mask=mask)
                    torch.cuda.synchronize()

                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"    seq_len={slen:>8d}: OK (peak={peak:.1f} GB)")
                best_len = slen
                best_peak = peak
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    print(f"    seq_len={slen:>8d}: OOM")
                    oom = True
                else:
                    raise
            finally:
                del ids, pos, mask, probe_out
                gc.collect()
                torch.cuda.empty_cache()
            if oom:
                break

        result["max_seqlen"] = best_len
        result["max_seqlen_peak_gb"] = best_peak

    result["status"] = "PASS"

except Exception as e:
    import traceback
    result["error"] = traceback.format_exc()[-800:]

with open(output_file, "w") as f:
    json.dump(result, f, indent=2)
print(f"  Result saved to {output_file}")
'''


def run_test(precision, method_name, eval_seq_len, max_seqlen_search, port, output_dir):
    """Run one compression test in isolated subprocess."""
    output_file = os.path.join(output_dir, f"result_{method_name}_{precision}.json")
    worker_file = os.path.join(output_dir, "_worker_real.py")

    with open(worker_file, "w") as f:
        f.write(WORKER_SCRIPT)

    # Remove any stale result so a crashed worker cannot be mistaken for a fresh run.
    if os.path.exists(output_file):
        os.remove(output_file)

    env = os.environ.copy()
    env["WORKER_PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        sys.executable, worker_file,
        precision, method_name, str(eval_seq_len),
        "1" if max_seqlen_search else "0",
        output_file,
    ]

    print(f"\n{'─' * 70}")
    print(f"  {method_name} → {precision.upper()}")
    print(f"{'─' * 70}")

    t0 = time.time()
    try:
        subprocess.run(cmd, env=env, capture_output=False, timeout=1200)
    except subprocess.TimeoutExpired:
        return {
            "method": method_name, "precision": precision,
            "status": "TIMEOUT", "error": "Worker exceeded the 1200s timeout",
            "wall_time_s": time.time() - t0,
        }
    elapsed = time.time() - t0

    if os.path.exists(output_file):
        with open(output_file) as f:
            result = json.load(f)
        result["wall_time_s"] = elapsed
        return result
    return {
        "method": method_name, "precision": precision,
        "status": "CRASH", "error": "No output file produced",
        "wall_time_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Evo2 Real Compression Benchmark")
    parser.add_argument("--precisions", nargs="+", default=["int8"], choices=["int8"],
                        help="Precision families to test")
    parser.add_argument("--eval-seq-len", type=int, default=2048,
                        help="Sequence length for quality evaluation")
    parser.add_argument("--max-seqlen", action=argparse.BooleanOptionalAction, default=True,
                        help="Sweep candidate sequence lengths and report the largest that fits "
                             "(use --no-max-seqlen to skip)")
    parser.add_argument("--output-dir", default="/workspace/quantization/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Map precision families to representative method names
    precision_methods = {
        "int8": "INT8_DEFAULT_CFG",
    }

    configs = [(p, precision_methods[p]) for p in args.precisions if p in precision_methods]

    print("=" * 80)
    print("  Evo2 REAL Weight Compression Benchmark")
    print(f"  Configs: {[(p, m) for p, m in configs]}")
    print(f"  Eval seq_len: {args.eval_seq_len}")
    print(f"  Max seqlen search: {args.max_seqlen}")
    print("=" * 80)

    all_results = []
    port = 29700

    for precision, method_name in configs:
        port += 1
        result = run_test(precision, method_name, args.eval_seq_len,
                          args.max_seqlen, port, args.output_dir)

        status = result.get("status", "?")
        mem_bf16 = result.get("mem_bf16_mb", 0)
        mem_comp = result.get("mem_compressed_mb", 0)
        saved = result.get("mem_saved_pct", 0)
        cos = result.get("cos_sim_avg", 0)
        top1 = result.get("top1_agree", 0)
        mse = result.get("mse", 0)
        max_sl = result.get("max_seqlen", 0)

        print(f"\n  ► {status}: {precision.upper()}")
        print(f"    Memory: {mem_bf16:.0f} → {mem_comp:.0f} MB (saved {saved:.1f}%)")
        print(f"    Quality: CosSim={cos:.6f}  Top1={top1:.4f}  MSE={mse:.6f}")
        if max_sl > 0:
            print(f"    Max SeqLen: {max_sl:,}")

        if result.get("error"):
            print(f"    Error: {result['error'][:300]}")

        all_results.append(result)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "real_quant_benchmark.csv")
    if all_results:
        keys = sorted(set(k for r in all_results for k in r.keys()))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_results:
                w.writerow({k: r.get(k, "") for k in keys})

    # Summary
    print(f"\n{'=' * 90}")
    print("  REAL COMPRESSION RESULTS SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Precision':<8s} {'Method':<25s} {'BF16 MB':<10s} {'Comp MB':<10s} {'Saved%':<8s} "
          f"{'CosSim':<10s} {'Top1':<8s} {'MaxSeqLen':<12s}")
    print(f"  {'-' * 85}")
    for r in all_results:
        print(f"  {r.get('precision','?'):<8s} {r.get('method','?'):<25s} "
              f"{r.get('mem_bf16_mb',0):>8.0f}  {r.get('mem_compressed_mb',0):>8.0f}  "
              f"{r.get('mem_saved_pct',0):>6.1f}  {r.get('cos_sim_avg',0):>8.6f}  "
              f"{r.get('top1_agree',0):>6.4f}  {r.get('max_seqlen',0):>10,}")

    print(f"\n✅ Results saved to: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
