# evo2_megatron_quant

Inference-side **INT8 real weight compression** and a **ModelOpt PTQ analysis
harness** for NVIDIA BioNeMo biological foundation models (ESM-2, Geneformer,
Evo2), on top of
[NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

Upstream `bionemo-recipes` covers low-precision *training* (TransformerEngine
FP8/MXFP8/NVFP4); this recipe adds an *inference*-side path:

1. **Real INT8 weight compression** — a drop-in `nn.Linear` replacement
   (`INT8Linear`) that stores weights in int8 (per-channel symmetric scaling)
   and dequantizes on the fly, for genuine GPU-memory savings.
2. **ModelOpt PTQ harness** — wires BioNeMo model loading to `mtq.quantize()`
   (simulated Q/DQ) with an automated quality report (cosine similarity, top-k
   agreement, MSE) vs. the BF16 baseline.

> This is the first, deliberately narrow slice. FP8/INT4 real weight compression
> and KV-cache experiments are intentionally left out of this PR and will follow
> once their GPU quality is re-measured.

## Results (INT8 real weight compression)

Measured on **Evo2 7B** (H200), compressing the TE `Linear` layers:

| Precision | Model memory | Saved | Cosine sim | Top-1 |
|-----------|:------------:|:-----:|:----------:|:-----:|
| BF16 (baseline) | 13,035 MB | — | 1.000 | 100% |
| **INT8** (per-channel) | **11,057 MB** | **~15%** | **0.998** | **100%** |

Most of Evo2's parameters live in non-`Linear` Hyena/SSM components that this
pass does not touch, so the saving reflects compressing the Linear layers only.

## Supported models

| Model | Architecture | Domain | Params | Checkpoint (NGC) |
|-------|-------------|--------|:------:|-------------------|
| ESM-2 | Transformer encoder | Protein | 8M | `esm2/8m:2.0` |
| Geneformer | Transformer encoder | Gene expression | 10M | `geneformer/10M_241113:2.0` |
| Evo2 | Hyena/Mamba SSM | DNA | 7B | `evo2/7b-8k:1.0` |

## Layout

```text
evo2_megatron_quant/
├── src/
│   ├── adapters.py            # per-model load / tokenize / forward adapters
│   ├── quantize.py            # ModelOpt mtq.quantize() wrapper
│   ├── metrics.py             # cosine sim / top-k / MSE
│   ├── compressed_linear.py   # INT8Linear (real int8 weight storage)
│   └── compress_model.py      # walk model, swap Linear -> INT8Linear
├── tests/
│   └── test_compressed_linear_roundtrip.py   # L0 sanity: CPU-only, no checkpoint
├── scripts/
│   ├── download_models.sh
│   ├── run_single_model.py                   # ModelOpt PTQ harness runner (GPU + NGC)
│   └── benchmark_real_quant.py               # INT8 memory + quality benchmark (GPU)
├── configs/quant_methods.yaml
├── docker/start_container.sh
├── requirements.txt
└── README.md
```

## Quick start

```bash
# L0 sanity tests (CPU-only, run in CI — no GPU or checkpoint needed):
pytest tests/test_compressed_linear_roundtrip.py -v

# Real INT8 weight compression benchmark (inside the BioNeMo container, GPU + NGC):
python scripts/benchmark_real_quant.py --precisions int8

# ModelOpt PTQ harness on one model (GPU + NGC):
python scripts/run_single_model.py --model esm2 --quant INT8_DEFAULT_CFG
```

Programmatic use:

```python
from src.compress_model import compress_model

stats = compress_model(evo2_model, precision="int8")   # swaps Linear -> INT8Linear
# -> ~15% GPU memory reduction on Evo2 7B, cosine sim ~0.998 vs BF16
```

## How INT8 weight compression works

1. Extract weight/bias from each `nn.Linear` / TE `Linear`.
2. Quantize to int8 on CPU (avoids a GPU memory spike), per output channel:
   `scale = amax(|W|, dim=1) / 127`, `W_int8 = round(W / scale)`.
3. Replace the layer with an `INT8Linear` that dequantizes (`W_int8 * scale`) on
   the fly for the matmul.
4. Free the original BF16 weights.

## Tests

`tests/` holds only the CPU-only L0 gate,
`tests/test_compressed_linear_roundtrip.py`: it round-trips INT8 on tiny random
layers and asserts near-lossless reconstruction, no systematic weight bias, and a
real reduction in the stored buffer size. It needs neither a GPU nor a
checkpoint, so it runs in normal CI.

The GPU runners live under `scripts/` (`run_single_model.py`,
`benchmark_real_quant.py`) — they require a CUDA device and NGC checkpoints and
are meant to be run manually, so they are deliberately not pytest tests.
