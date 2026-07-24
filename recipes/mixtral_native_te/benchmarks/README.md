# Mixtral-8x7B B200 and B300 benchmarks

This directory contains the measured training throughput for Mixtral-8x7B on one node with eight
B200 or B300 GPUs. The results use the native Transformer Engine recipe, pretrained Mixtral weights,
packed THD batches, and a pinned local copy of DCLM.

The committed results and plots are:

- `mixtral_8x7b_8xB200.csv`
- `mixtral_8x7b_B200_pflops.png`
- `mixtral_8x7b_8xB300.csv`
- `mixtral_8x7b_B300_pflops.png`

## Environment

The results were collected inside the NVIDIA PyTorch 26.06 image on single Slurm nodes with either
8×B200 or 8×B300 GPUs. From the repository root, install the recipe requirements:

```bash
export PIP_CACHE_DIR="${TMPDIR:-/tmp}/pip-cache"
python -m pip install -r recipes/mixtral_native_te/requirements.txt
```

Keep the Hugging Face cache off a space-constrained home directory:

```bash
export HF_HOME="${TMPDIR:-/tmp}/mixtral_native_te_hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME/te_checkpoints"
```

The model and dataset are public, but Hugging Face credentials may still be supplied through the
usual `huggingface-cli login`, `HF_TOKEN`, or `~/.netrc` mechanisms.

## Prepare the pretrained checkpoint

The benchmark intentionally has no random-initialization mode. It requires a BF16 Transformer
Engine checkpoint converted from `mistralai/Mixtral-8x7B-v0.1` at this fixed location:

```text
$HF_HOME/te_checkpoints/mixtral_8x7b_fused_bf16.pt
```

The checkpoint used for these results was prepared from the repository root with:

```bash
export MIXTRAL_TE_PRETRAINED_CHECKPOINT="$HF_HOME/te_checkpoints/mixtral_8x7b_fused_bf16.pt"
PYTHONPATH=models/mixtral python - <<'PY'
import os
from pathlib import Path

import torch
from export import export_hf_state_dict


export_hf_state_dict(
    "mistralai/Mixtral-8x7B-v0.1",
    Path(os.environ["MIXTRAL_TE_PRETRAINED_CHECKPOINT"]),
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    expert_ffn_mode="fused_grouped_mlp",
)
PY
```

This conversion is performed once, before benchmarking, and is not included in the reported elapsed
times.

## Run the matrix

Run all four `(dp, ep)` layouts in both MXFP8 and BF16:

```bash
export HF_HOME="${TMPDIR:-/tmp}/mixtral_native_te_hf"
recipes/mixtral_native_te/benchmarks/benchmark_8xGPU.sh
```

To run only one layout or precision, override the matrix:

```bash
PARALLEL_CONFIGS="2,4" PRECISIONS="fp8 bf16" \
    recipes/mixtral_native_te/benchmarks/benchmark_8xGPU.sh
```

The script:

1. Downloads all nine parquet shards from `codelion/dclm-baseline-1B` at revision
   `2b7b056aae2fde089e234563fb32c678caea6bca`.
2. Caches the tokenizer before launching distributed training.
3. Enables Hugging Face offline mode for `torchrun`, so `streaming: true` reads only the downloaded
   local parquet files and performs no network I/O during training.
4. Runs 60 steps on 8 GPUs and writes per-row logs and elapsed times under `OUTPUT_DIR`.
5. Detects B200 versus B300, applies the hardware-specific defaults below, and writes CSV-formatted
   steady-state results to `RESULTS_CSV`.

The training settings live in `../hydra_config/L1_8x7B_B200.yaml`. The B200 runs use:

- `max_seq_length=4096`
- `token_micro_batch_size=4096`
- online THD sequence packing
- `use_meta_device=false`
- pretrained initialization on every run
- TE FusedAdam with FP32-precision main weights

For MXFP8, quantized model initialization preserves the pretrained high-precision values used to
seed full FP32 optimizer masters. For BF16, `store_param_remainders=true` represents each FP32 main
weight as the BF16 parameter plus its 16-bit remainder.

We tested `max_seq_length=8192` with an 8192-token microbatch, but it did not fit on B200 during the
first optimizer step. Meta-device initialization was also disabled because materialization after
FSDP2 wrapping invalidated the sharded-parameter reference in this software stack.

## Hardware-specific defaults

The runner detects the first GPU reported by `nvidia-smi` and selects:

| GPU  | `max_seq_length` | `token_micro_batch_size` | FP8 peak | BF16 peak |
| ---- | ---------------- | ------------------------ | -------- | --------- |
| B200 | 4096             | 4096                     | 4.5      | 2.25      |
| B300 | 8192             | 16384                    | 5.0      | 2.5       |

All values remain available as environment overrides. A normal B300 run is:

```bash
export HF_HOME="${TMPDIR:-/tmp}/mixtral_native_te_hf"
recipes/mixtral_native_te/benchmarks/benchmark_8xGPU.sh
```

By default, logs and the generated CSV are written under
`${TMPDIR:-/tmp}/mixtral_native_te_8xB300`. Set `RESULTS_CSV` to write the final summary to a
different location. The peak-PFLOP values only affect the reported MFU; they do not affect training.
The old `benchmark_8xB200.sh` name remains as a compatibility wrapper around the hardware-aware
runner.

## Steady-state metrics

The first two reporting windows, at steps 10 and 20, include kernel compilation and warmup. Each CSV
row averages the final three reported windows at steps 30, 40, and 50.

PFLOP/s/GPU is calculated as:

```text
6 × 12,748,587,008 active parameters × tokens/s/GPU ÷ 1e15
```

MFU uses the hardware-specific dense peaks listed above. Regenerate the B200 chart with:

```bash
python recipes/mixtral_native_te/benchmarks/plot_perf.py
```

Regenerate the B300 chart with:

```bash
python recipes/mixtral_native_te/benchmarks/plot_perf.py \
    --csv recipes/mixtral_native_te/benchmarks/mixtral_8x7b_8xB300.csv \
    --out recipes/mixtral_native_te/benchmarks/mixtral_8x7b_B300_pflops.png \
    --title "Mixtral-8x7B training throughput — 8×B300" \
    --subtitle "Pretrained weights, local DCLM parquet, THD packing, token_mb=16384, max_seq=8192. MFU vs dense B300 peaks (fp8 5.0, bf16 2.5 PFLOP/s)."
```
