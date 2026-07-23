#!/usr/bin/env python3
"""Optimized HuggingFace Mixtral-8x7B baseline — naive pipeline parallel (device_map="auto", 8 GPUs).

This is the reference point the native-TE expert-parallel recipe is compared against, so it is made as
strong as HF reasonably allows (not a strawman):

  * REAL pretrained checkpoint (`from_pretrained`) fed REAL text tokens. MoE routing is *learned*, so
    the expert loads are only balanced/representative with the trained router on in-distribution text.
    Random weights — or even real weights on random token ids — collapse routing to a few experts and
    give an unrepresentative (and usually inflated) throughput.
  * `experts_implementation="grouped_mm"` — a single grouped GEMM over tokens ordered by expert
    (torch.nn.functional.grouped_mm, PyTorch 2.9+), the most efficient / compile-friendly HF backend.
  * `torch.compile(model.forward, mode="max-autotune-no-cudagraphs")` — grouped_mm requires the
    no-cudagraphs mode.
  * `attn_implementation="sdpa"` — compile-friendly. (`flex_attention` was tried but hit an Inductor
    lowering bug on this torch build; set HF_ATTN=flex_attention to retry.)

Measures the same axis as the recipe's perf_logger: tokens/s/GPU (= batch*seq / step_time / n_gpus,
divided by n_gpus because naive PP has one pipeline stage active at a time), then derives PFLOP/s/GPU
with the identical 6*N_active factor so it drops into the 8xB300 CSV.
"""

import os
import statistics
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("HF_MIXTRAL", "mistralai/Mixtral-8x7B-v0.1")
SEQ = int(os.environ.get("HF_SEQ", "8192"))
BATCH = int(os.environ.get("HF_BATCH", "8"))
WARMUP = int(os.environ.get("HF_WARMUP", "5"))
STEPS = int(os.environ.get("HF_STEPS", "10"))
EXPERTS = os.environ.get("HF_EXPERTS", "grouped_mm")
ATTN = os.environ.get("HF_ATTN", "sdpa")
COMPILE = os.environ.get("HF_COMPILE", "1") == "1"
CORPUS = os.environ.get("BENCH_CORPUS", "/lustre/fsw/coreai_prod_infbench/faradawny/mixtral_bench_8xB300/bench_corpus.parquet")

N_ACTIVE = 12_748_587_008
TOKENS_TO_PFLOPS = 6 * N_ACTIVE / 1e15
PEAK_BF16 = 2.5  # B300 dense bf16 PFLOP/s/GPU (matches the recipe CSV mfu_pct derivation)


def real_token_batch(device):
    """One (BATCH, SEQ) batch of REAL wikitext tokens so the learned router load-balances normally."""
    import pyarrow.parquet as pq

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    need = BATCH * SEQ + 1
    texts = pq.read_table(CORPUS)["text"]
    ids: list[int] = []
    for x in texts:
        s = x.as_py()
        if not s:
            continue
        ids.extend(tokenizer(s, add_special_tokens=False)["input_ids"])
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise RuntimeError(f"corpus too small: got {len(ids)} tokens, need {need}")
    t = torch.tensor(ids[: BATCH * SEQ], dtype=torch.long, device=device)
    return t.view(BATCH, SEQ)


def main():
    torch.manual_seed(0)
    n_gpus = torch.cuda.device_count()
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.use_cache = False

    print(f"loading {MODEL} bf16 device_map=auto over {n_gpus} GPUs "
          f"(experts={EXPERTS}, attn={ATTN}, compile={COMPILE}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        config=cfg,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=ATTN,
        experts_implementation=EXPERTS,
    )
    model.train()
    print("experts:", model.get_experts_implementation(), flush=True)
    if COMPILE:
        model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs")

    in_dev = next(model.parameters()).device
    ids = real_token_batch(in_dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, foreach=False)

    def do_step():
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(out.loss.detach())

    for i in range(WARMUP):
        t0 = time.perf_counter()
        loss = do_step()
        torch.cuda.synchronize()
        print(f"[warmup {i}] {(time.perf_counter() - t0) * 1000:.0f} ms  loss {loss:.2f}", flush=True)

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
        f"num_gpus={n_gpus} batch={BATCH} seq={SEQ} experts={EXPERTS} attn={ATTN} compile={COMPILE} "
        f"last_loss={loss:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
