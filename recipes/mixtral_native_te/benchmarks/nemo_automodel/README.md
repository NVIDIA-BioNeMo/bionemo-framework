# Mixtral-8x7B benchmark with NeMo AutoModel

This benchmark searches for the fastest supported NeMo AutoModel topology on one 8×B200 node while
holding the important native-recipe inputs fixed:

- pretrained `mistralai/Mixtral-8x7B-v0.1` weights at snapshot
  `fc7ac94680e38d7348cfa806e51218e6273104b0`;
- the same nine DCLM parquet shards at revision
  `2b7b056aae2fde089e234563fb32c678caea6bca`;
- padding-free 4,096-token packs on every data-parallel rank;
- 32,768 tokens per optimizer step;
- 60 steps, with the first 30 excluded as compilation and warmup;
- BF16 parameters, FP32 gradient reduction, and TE FusedAdam with FP32-precision main weights.

Run the fastest supported layout from the repository root:

```bash
export HF_HOME="${TMPDIR:-/tmp}/mixtral_automodel_hf"
export OUTPUT_DIR="${TMPDIR:-/tmp}/mixtral_automodel_8xB200"
recipes/mixtral_native_te/benchmarks/nemo_automodel/benchmark_8xB200.sh
```

The first invocation downloads the roughly 90 GB pretrained checkpoint and the pinned DCLM sample
before training starts. Set `HF_TOKEN` if the shared Hugging Face IP is rate-limited. To use an
already-downloaded snapshot and parquet files:

```bash
MODEL_PATH=/path/to/Mixtral-8x7B-v0.1 \
DATA_FILE='/path/to/dclm/*.parquet' \
    recipes/mixtral_native_te/benchmarks/nemo_automodel/benchmark_8xB200.sh
```

The default is `DP8`, which is the fastest and only memory-safe layout found in the 26.06 image. It
keeps the global batch at eight packs without gradient accumulation. You can explicitly retry the
experimental TP sweep in a newer image with:

```bash
LAYOUTS='dp8tp1 dp4tp2 dp2tp4 dp1tp8' \
    recipes/mixtral_native_te/benchmarks/nemo_automodel/benchmark_8xB200.sh
```

Failed experimental layouts are retained as logs and do not discard successful results.

Results are written as per-layout logs and JSON plus `mixtral_8x7b_8xB200.csv` under `OUTPUT_DIR`.
The CSV uses the same active-parameter throughput estimate as the native benchmark:

```text
6 × 12,748,587,008 × tokens/s/GPU
```

## Why this does not use EP

The NeMo AutoModel 26.06 image can load and train Hugging Face Mixtral. Its expert-parallel
implementation, however, requires an AutoModel-native MoE class that inherits
`MoEFSDPSyncMixin`; the Hugging Face `MixtralForCausalLM` used by this release does not. Setting
`distributed.ep_size > 1` therefore fails validation rather than providing Mixtral EP.

Transformers 5's Mixtral TP plan is also not usable directly by this AutoModel build
(`colwise_gather_output` is unknown), so AutoModel falls back to its generic Llama-style plan. That
plan shards attention projections but deliberately skips the Hugging Face MoE styles: every rank
stores and computes all expert weights. On 8×B200, `DP4×TP2` reached roughly 170 GiB allocated per
GPU and ran out of memory on its first accumulated backward pass. Larger TP degrees are consequently
not useful candidates in this image.

For the fastest available HF path, the recipe explicitly selects:

- FlashAttention 2 with position-reset variable-length packing, which preserves document
  boundaries in the packed attention call;
- Transformers' `grouped_mm` Mixtral expert implementation;
- Liger disabled, because its Mixtral patch replaces the complete expert module with an eager
  per-expert loop and bypasses the configured grouped-GEMM backend;
- TorchAO's Blackwell MXFP8 grouped training GEMM for the MXFP8 row, including MXFP8
  weight-gradient GEMMs;
- TorchAO's Blackwell MXFP8 dense training GEMM for the attention projections, including MXFP8
  activation-gradient and weight-gradient GEMMs. The router and LM head stay BF16, as they do in
  the native recipe. The packaged TorchAO parameter-wrapper API requires a newer PyTorch nightly
  than this image, so the local `MXFP8Linear` uses the same underlying autograd kernel while
  preserving the ordinary parameter layout required by FSDP2;
- FSDP2 for the data-parallel dimension.

This is an AutoModel throughput comparison against the native TE results, not a kernel-for-kernel
reproduction. In particular, the native recipe's fused TE GroupedMLP and MXFP8 expert path are not
available to Hugging Face Mixtral through AutoModel in this image.

Set `MXFP8_DENSE=false` to measure experts-only MXFP8. That variant reached 4,149 tok/s/GPU, but
the primary result below enables dense MXFP8 because the native TE benchmark applies its MXFP8
autocast to every decoder layer.

## Measured 8×B200 results

These rows were measured in the 26.06 image with checkpoint snapshot
`fc7ac94680e38d7348cfa806e51218e6273104b0`, 30 warmup steps, and 30 measured steps. Both use
exactly 32,768 tokens per optimizer step.

| stack     | precision | topology | tok/s/GPU | PFLOP/s/GPU |    MFU | step (s) | memory (GiB) |
| --------- | --------- | -------- | --------: | ----------: | -----: | -------: | -----------: |
| AutoModel | BF16      | DP8      |     4,284 |      0.3277 | 14.56% |    0.956 |       120.86 |
| Native TE | BF16      | DP8      |     4,447 |      0.3401 | 15.12% |    0.938 |         65.8 |
| AutoModel | MXFP8     | DP8      |     4,055 |      0.3102 |  6.89% |    1.010 |       126.69 |
| Native TE | MXFP8     | DP8      |     6,580 |      0.5033 | 11.18% |    0.634 |         88.2 |

The AutoModel MXFP8 row quantizes the attention and expert forward, activation-gradient, and
weight-gradient GEMMs. At DP8, this TorchAO MXFP8 path is 5.3% slower than AutoModel BF16; its
dynamic quantization and 128-row expert-group padding cost more than the MXFP8 GEMMs save. Its
throughput is 38.4% below native TE MXFP8 at the same DP8 topology.

`mfu_pct` is calculated externally using the same active-parameter convention as the native CSV,
not AutoModel's built-in MFU field. The source result is
[`mixtral_8x7b_8xB200.csv`](mixtral_8x7b_8xB200.csv).
