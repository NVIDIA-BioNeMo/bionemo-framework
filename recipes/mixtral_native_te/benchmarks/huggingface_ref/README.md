# Hugging Face Mixtral reference benchmark

This benchmark trains the upstream Hugging Face `MixtralForCausalLM`; it does not import or use
Transformer Engine. It matches the native-TE B200 benchmark's pretrained Mixtral-8x7B weights,
4096-token sequences, pinned DCLM data, AdamW schedule, BF16 parameters, and gradient clipping.

The optimized 8xB200 configuration uses:

- Hugging Face native EP8, with one expert's weights on each B200
- Hugging Face `grouped_mm` for BF16 experts
- TorchAO differentiable grouped MXFP8 GEMMs for the MXFP8 run
- PyTorch SDPA, which selects fused CUDA attention kernels
- every complete decoder block compiled with `torch.compile(fullgraph=True, dynamic=False)`

Install and run from the repository root:

```bash
PIP_CONSTRAINT= pip install --no-build-isolation \
  -r recipes/mixtral_native_te/benchmarks/huggingface_ref/requirements.txt
recipes/mixtral_native_te/benchmarks/huggingface_ref/benchmark_8xB200.sh
```

The launcher downloads the exact pinned Mixtral and DCLM revisions before starting workers, then
enforces offline mode. JSON results and logs default to `/tmp/mixtral_hf_ref_8xB200`. By default it
runs both `EP_SIZES="1 8"` and `PRECISIONS="bf16 mxfp8"`; narrow either list for a single case.

## Measured results

These are full forward/backward/AdamW steps on this 8xB200 system, using pretrained weights, the
same DCLM batch source, sequence length 4096, and microbatch size 1:

| Layout | Precision     | Global tokens/step |  Median step | Global tokens/s | Peak/GPU |
| ------ | ------------- | -----------------: | -----------: | --------------: | -------: |
| FSDP8  | BF16          |             32,768 |     0.8862 s |      **36,974** | 79.8 GiB |
| FSDP8  | MXFP8 experts |             32,768 |     0.9222 s |          35,534 | 84.6 GiB |
| HF EP8 | BF16          |              4,096 |     0.3224 s |          12,706 | 73.9 GiB |
| HF EP8 | MXFP8 experts |              4,096 | **0.3120 s** |          13,128 | 76.3 GiB |

FSDP8 BF16 is the best aggregate-throughput configuration. MXFP8 improves EP8 latency by about
3.3%, but quantization overhead slightly outweighs its expert-GEMM saving in FSDP8. The layouts do
different amounts of useful data work per step, so their raw step latency is not directly
comparable.

## What EP means in this implementation

Transformers 5.14.1's Mixtral EP plan shards the fused expert weights and router computation across
the full process group. Each rank receives the same token batch, masks routes for experts it does
not own, evaluates its local expert contribution, and a differentiable all-reduce combines the
partial MoE outputs.

This is expert-weight parallelism, but not Transformer Engine's all-to-all token-dispatch EP.
Mixtral's installed plan does not shard attention, embeddings, norms, router weights, or the LM
head, so those non-expert layers are replicated—not TP8. Consequently EP8 processes one shared
4096-token batch per step, while the FSDP8 comparison (`EP_SIZES=1`) processes eight distinct
4096-token batches. Compare step time as well as globally processed tokens/sec.

The stable Transformers loader builds EP over the entire torchrun world. Hybrid EP+FSDP would need
a custom multidimensional device mesh and extra model integration, so it is intentionally outside
this minimal documented baseline.

## MXFP8 details

The Transformers `TorchAoConfig` loader quantizes `nn.Linear` and embedding weights for
weight-only/inference-style integrations; Mixtral experts are fused 3-D parameters. This benchmark
instead follows TorchAO's differentiable MoE training operation: expert activations and weights are
dynamically block-scaled to MXFP8 for forward and input-gradient grouped GEMMs, while trainable
weights, optimizer state, and weight-gradient GEMMs remain BF16.

TorchAO `0.18.0+git28e6aca5` was roughly two orders of magnitude slower in this container's
Blackwell grouped path. The requirements pin commit
`693b573f6f3eae8be467db0ef3e49a1553ae2731` (`0.18.0+git693b573`), which fixes that regression.
PyTorch remains the container build, `2.13.0a0+8145d630e8.nv26.06`.

## Tuning notes

SDPA outperformed the available alternatives while preserving full-graph compilation.
FlashAttention 2's current Transformers wrapper has tensor-dependent Python control flow that
Dynamo rejects with `fullgraph=True`. `grouped_mm` also cannot use CUDA graphs, and Inductor's
default compile mode was faster than `max-autotune-no-cudagraphs` in the measured BF16 sweep.

Composable FSDP gather/reshard hooks cannot live inside the Dynamo graph, so the FSDP8 case compiles
each complete decoder block and leaves only the nested FSDP hooks outside. Disabling reshard after
forward was both much slower and substantially more memory-hungry.

HF Mixtral does not expose a TE-style THD/`cu_seqlens` training interface. Packing variable-length
documents would require a block-diagonal attention mask (or a custom varlen attention integration);
it gives no benefit for this benchmark's already-full 4096-token samples.

Useful overrides:

```bash
# FSDP8 comparison
PRECISIONS=bf16 EP_SIZES=1 \
  recipes/mixtral_native_te/benchmarks/huggingface_ref/benchmark_8xB200.sh

# Test a larger shared EP batch if memory permits
PRECISIONS=bf16 EP_SIZES=8 MICRO_BATCH_SIZE=4 \
  recipes/mixtral_native_te/benchmarks/huggingface_ref/benchmark_8xB200.sh
```
