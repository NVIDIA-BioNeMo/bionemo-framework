#!/usr/bin/env python3
"""HuggingFace naive-pipeline-parallel baseline for Mixtral-8x7B (device_map="auto", 8 GPUs).

Deliberately the "worst" reference: a single process spreads the model layer-by-layer across all 8
GPUs (naive model/pipeline parallel — only one stage active at a time). We measure the SAME quantity
the bionemo recipe's perf_logger reports — tokens/s/GPU — on the same seq=8192 workload, then derive
PFLOP/s/GPU with the identical 6*N_active factor so it drops straight into the 8xB300 CSV.

Random-init HF Mixtral is fine: throughput is weight-independent, and this avoids any conversion.
"""

import os
import statistics
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = os.environ.get("HF_MIXTRAL", "mistralai/Mixtral-8x7B-v0.1")
SEQ = int(os.environ.get("HF_SEQ", "8192"))
BATCH = int(os.environ.get("HF_BATCH", "8"))
WARMUP = int(os.environ.get("HF_WARMUP", "3"))
STEPS = int(os.environ.get("HF_STEPS", "10"))
N_ACTIVE = 12_748_587_008
TOKENS_TO_PFLOPS = 6 * N_ACTIVE / 1e15
PEAK_BF16 = 2.5  # B300 dense bf16 PFLOP/s/GPU (matches the recipe CSV mfu_pct derivation)


def main():
    torch.manual_seed(0)
    n_gpus = torch.cuda.device_count()
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.use_cache = False

    print(f"loading {MODEL} bf16 device_map=auto across {n_gpus} GPUs ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        config=cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.train()
    in_dev = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, foreach=False)

    def do_step():
        ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ), device=in_dev)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(out.loss.detach())

    for i in range(WARMUP):
        loss = do_step()
        print(f"[warmup {i}] loss {loss:.2f}", flush=True)
    torch.cuda.synchronize()

    times = []
    for i in range(STEPS):
        t0 = time.perf_counter()
        loss = do_step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[step {i}] {dt * 1000:.0f} ms  loss {loss:.2f}", flush=True)

    med = statistics.median(times)
    tokens = BATCH * SEQ
    tps_gpu = tokens / med / n_gpus
    pflops = TOKENS_TO_PFLOPS * tps_gpu
    mem = max(torch.cuda.max_memory_allocated(d) / 1e9 for d in range(n_gpus))
    print(
        "HF_RESULT "
        f"tokens_per_s_per_gpu={tps_gpu:.1f} pflops_per_gpu={pflops:.4f} "
        f"mfu_pct={pflops / PEAK_BF16 * 100:.2f} step_time_s={med:.3f} mem_gb={mem:.1f} "
        f"num_gpus={n_gpus} batch={BATCH} seq={SEQ} last_loss={loss:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
