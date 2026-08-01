# Precision selection

Phases 2 and 3. The goal is a *measured* recommendation, not a heuristic — and an honest statement
of how much of the measured GEMM speedup will actually survive to end-to-end training.

## Phase 2: what the hardware probe tells you

`assets/probe_hardware.py` delegates entirely to Transformer Engine rather than doing its own
compute-capability arithmetic. This mirrors `$BIONEMO_RECIPES/models/esm2/tests/common/fixtures.py::_check_recipe_support`:

| Recipe                 | Support check                           |
| ---------------------- | --------------------------------------- |
| `DelayedScaling`       | `fp8.check_fp8_support()`               |
| `Float8CurrentScaling` | `fp8.check_fp8_support()`               |
| `Float8BlockScaling`   | `fp8.check_fp8_block_scaling_support()` |
| `MXFP8BlockScaling`    | `fp8.check_mxfp8_support()`             |
| `NVFP4BlockScaling`    | `fp8.check_nvfp4_support()`             |

Each returns `(supported: bool, reason: str)`. Report the reason string verbatim when a recipe is
unavailable — it is more specific than anything you would write.

### Caveats the probe emits that TE's own checks do not

- **sm_120 (consumer Blackwell, RTX 50xx).** `$BIONEMO_RECIPES/recipes/eden_megatron/tests/bionemo/eden/utils.py`
  bounds `is_mxfp8_supported()` and `is_fp4_supported()` at `< (12, 0)`, excluding sm_120 even
  where TE may report support. Treat MXFP8/NVFP4 on sm_120 as unvalidated.
- **`fused_attn` on sm_80 and sm_120.** `$BIONEMO_RECIPES/models/esm2/tests/common/test_modeling_common.py` xfails
  the fused-attention THD golden-value test on these. Sequence packing still works via
  `flash_attn`; note it rather than blocking.
- **Data-center GPU gate.** `$BIONEMO_RECIPES/models/esm2/tests/common/__init__.py` exports `HAS_DATA_CENTER_GPU`,
  true only for H100/H200/B100/B200/B300. Several reference tests are gated on it.

### Transformer Engine version

"Optimal version of Transformer Engine" resolves to a **recommendation plus a patch**, never a
silent upgrade of the user's environment. Report three things:

1. The version currently installed in the target environment.
2. The NGC image this repo standardizes on — `nvcr.io/nvidia/pytorch:26.04-py3`, per the
   `Dockerfile` in `$BIONEMO_RECIPES/recipes/esm2_native_te/`, `$BIONEMO_RECIPES/recipes/llama3_native_te/`, and others. In these
   images TE comes from the image, and `requirements.txt` lists an unversioned
   `transformer_engine[pytorch]`.
3. The repo's only explicit pin, for non-NGC environments:
   `transformer-engine[pytorch]==2.9.0` with `torch==2.9.0` in
   `$BIONEMO_RECIPES/recipes/esm2_native_te/Dockerfile.cuda`, installed `--no-build-isolation`.

Emit a suggested `Dockerfile` / `requirements.txt` diff into `.bionemo-accel/`. Apply it only if the
user asks.

API-surface note that depends on TE version: the current recipes use
`transformer_engine.pytorch.autocast(enabled=..., recipe=...)`. The legacy
`te.fp8_autocast(fp8_recipe=...)` still appears in `$BIONEMO_RECIPES/recipes/geneformer_native_te_mfsdp_fp8/train.py`.
Generate the modern form; if the target's installed TE lacks it, that is a version-upgrade finding
for the report, not a reason to emit legacy code silently.

## Phase 3: the GEMM benchmark

The tool is Transformer Engine's own `benchmarks/gemm/benchmark_gemm.py`, documented at
<https://nvidia.github.io/TransformerEngine/examples/gemm_profiling/gemm_profiling.html>. It ships in
the TE **source tree**, not the pip wheel — `assets/run_gemm_benchmark.py` searches for it and, if
absent, shallow-clones TE at the installed version tag into `.bionemo-accel/`.

Nothing in this repo does this job. `$BIONEMO_RECIPES/recipes/fp8_analysis/analyze_and_create_heatmap.py` is often
mistaken for it — it only post-processes `nvdlfw_inspect` logs from a run that already chose a
recipe, into a per-layer gradient-underflow heatmap. Useful as an accuracy-risk signal *after*
selection; not a selector.

### Model config mode (default)

```bash
python benchmarks/gemm/benchmark_gemm.py \
  --hidden_size 4096 --intermediate_size 16384 \
  --num_attention_heads 32 --num_hidden_layers 24 \
  --micro_batch_size 31 --sequence_length 512 \
  -o ./gemm_speedup.png
```

On Hopper add `--no-fp8 --no-fp4` to skip the block-scaled and FP4 precisions it cannot run.

### Manual shape mode

For MoE experts or any projection set that does not match the standard four, pass explicit `MxKxN`
triplets (mutually exclusive with the model-config flags):

```bash
python benchmarks/gemm/benchmark_gemm.py -o roofline_fprop.png \
  --shapes 15872x4096x12288,15872x4096x4096,15872x4096x16384,15872x16384x4096
```

Derivation: `M = micro_batch_size × sequence_length`; for `Y = X @ W`, `A` is `[M, K]` and `B` is
`[K, N]`. The four forward projections are QKV (`M×H×3H'`), attention output (`M×H×H`), MLP up
(`M×H×I`), MLP down (`M×I×H`). Backward yields two GEMMs per op (dX, dW), so total ≈ 3× forward.

### Autocast vs pre-quantized — run both

- **Autocast (default)** quantizes inputs before each GEMM, so it includes quantization cost. This
  is what training actually experiences.
- **`--pre-quantize`** quantizes once before timing, isolating raw kernel performance.

The gap between them is the dynamic quantization overhead and must appear in the report. In the
tutorial's B300 example, NVFP4-vs-BF16 falls from **3.48× pre-quantized to 1.98× under autocast**.
`DelayedScaling` always runs in autocast mode regardless of the flag, because it needs amax history.

## Interpreting the results

- **GEMM speedup ≈ training speedup** → GEMMs are the bottleneck; the recipe will deliver.
- **GEMM speedup ≫ training speedup** → something outside the GEMMs dominates: attention,
  communication, or quantization-specific ops.
- **GEMM speedup ≈ 1.0** → do **not** conclude "no benefit". This commonly indicates a silent
  fallback to a lower-precision kernel. Re-run with `NVTE_LOG_LEVEL=1` and confirm kernel dispatch
  before recommending or dismissing that recipe.

### GEMM speedup is an upper bound

Precision settings affect **only the linear projections**. These remain precision-agnostic and are
not represented in the benchmark at all:

- Attention itself (FlashAttention, BF16/FP16)
- LayerNorm / RMSNorm (typically FP32)
- Element-wise activations (memory-bound)
- AllReduce / AllGather for DDP or FSDP

So report the GEMM number as a ceiling and get the real number from the vendored `PerfLogger`
(`train/tokens_per_second_per_gpu`) on a short before/after run.

### NVFP4 has costs the benchmark does not capture

Random Hadamard transforms on Wgrad inputs, stochastic rounding on gradients, 2D block scaling for
weights, and per-tensor amax passes all sit outside the GEMM kernels. Discount NVFP4's headline
number accordingly, and prefer MXFP8 when the margin is narrow.

## Producing the recommendation

Rank the supported recipes by autocast speedup. Record the winner, the runner-up, and the margin in
`.bionemo-accel/precision.json`. Then generate the config **disabled**:

```yaml
fp8_config:
  enabled: false                                                  # user opts in
  fp8_recipe: transformer_engine.common.recipe.MXFP8BlockScaling  # the recommendation
  fp8_format: "E4M3"
  fp8_recipe_kwargs: {}
```

This mirrors `$BIONEMO_RECIPES/recipes/esm2_native_te/hydra_config/defaults.yaml`, which also ships
`enabled: false`. The dotted-path form is what makes switching recipes a one-line change — see
`references/te-conversion.md`.

Rough expectations from the tutorial's worked examples, useful as a sanity check on your own
numbers — not as a substitute for running it:

| GPU      | Recipe                     | Autocast speedup vs BF16                  |
| -------- | -------------------------- | ----------------------------------------- |
| B300     | NVFP4                      | 1.98×                                     |
| B300     | MXFP8                      | ≈1.42× (NVFP4 is 1.39× faster than MXFP8) |
| H200 NVL | FP8 `DelayedScaling`       | 1.69×                                     |
| H200 NVL | FP8 `Float8CurrentScaling` | 1.58×                                     |
| H200 NVL | `Float8BlockScaling`       | 1.40×                                     |

If your measured numbers are wildly different from these for a similarly-sized model, suspect
misconfiguration or kernel fallback before believing the result.
