# Evo2 vLLM Packed Inference Design

## Objective

Build and evaluate an out-of-tree vLLM implementation of Evo2/Vortex that can replace the
recipe-specific Megatron rollout engine for NeMo-RL GDPO. The implementation must preserve Evo2
generation accuracy, execute HCS/HCM/HCL layers over logically independent packed requests, support
the current sampling and recurrent-state lifecycle, and match or exceed the current MCore path on
the real 7B Microviridae rollout workload.

This is an independent backend experiment. It does not require MCore inference to succeed. Once the
vLLM work is otherwise complete, the experiment will review the API produced in
`/data/jstjohn/evo2-mcore-pr5274-lab` and recommend changes that would make the MCore work more useful
to external inference backends or allow low-level Evo2 kernels and state contracts to be shared.

## Isolation And Revisions

All work lives under `/data/jstjohn/evo2-vllm-lab`:

- `bionemo-recipes/`: implementation worktree at baseline
  `51e8b7afebf12547cce800f08747dc9248481909`.
- `nemo-rl/`: clean NeMo-RL checkout at
  `a2957f41c3860d5928832be0689b548813c6904a`.
- `vllm/`: read-only reference checkout of vLLM `v0.20.0` at
  `88d34c6409e9fb3c7b8ca0c04756f061d2099eb1`.
- `baseline/`: immutable manifests and comparison outputs.
- `artifacts/`: correctness records, profiler traces, benchmark samples, and environment manifests.

The production implementation belongs to the `bionemo-evo2` package under
`recipes/evo2_megatron`. vLLM 0.20.0 is an optional, exactly pinned package extra and must not become
a dependency of ordinary training or Megatron inference installs. The implementation must not
require source changes to the vLLM checkout. Experimental instrumentation of vLLM is allowed only
to diagnose a problem and cannot remain necessary for the final path.

The active `/data/jstjohn/evo2-mcore-pr5274-lab` workspace is read-only to this effort. No command may
send input to its tmux pane or alter its files, processes, branches, or environment.

## Supported Workload

The primary production target is the base Evo2 7B Microviridae checkpoint used by the current GDPO
recipe, with bf16 model parameters and fp32 recurrent state. Required rollout behavior includes:

- a global generation batch of 96 requests;
- prompt lengths from 4 through 12 tokens, including a mixed-length batch;
- the recipe's full generated sequence length and stop behavior;
- temperature, top-k, and top-p sampling;
- processed log-probabilities from the distribution that sampled each token;
- deterministic replay for a fixed seed and topology;
- CUDA-graph and compile-enabled decode;
- vLLM sleep/wake and post-optimizer weight refit;
- TP2/DP1 and TP1/DP2 on the available two H100 GPUs;
- a valid design for TP greater than one together with DP greater than one.

The 1B checkpoint remains the bounded correctness fixture for existing second-half identity tests.
The 7B checkpoint is mandatory for the final accuracy, memory, distributed, and throughput gates.

Prefix caching and speculative decoding are not current Evo2 rollout features and are not release
requirements. They must be rejected explicitly rather than silently producing incorrect state.

## Package Boundary

The implementation is an out-of-tree model plugin under
`recipes/evo2_megatron/src/bionemo/evo2/vllm/`:

```text
bionemo/evo2/vllm/
  plugin.py           vllm.general_plugins registration
  config.py           checkpoint/config normalization and layer pattern
  model.py            Evo2ForCausalLM and top-level vLLM model
  layers.py           attention, MLP, norm, and residual blocks
  hyena.py            HCS/HCM/HCL mixers and MambaBase cache interface
  packed_fir.py       packed FIR reference and optimized kernels
  packed_iir.py       packed HCL reference and optimized kernels
  weights.py          Vortex/MBridge loading and TP-aware refit mapping
  benchmark.py        reproducible standalone benchmark entry point
```

The plugin registers `Evo2ForCausalLM` lazily through `vllm.general_plugins`. Every vLLM worker
loads the same package. NeMo-RL uses its ordinary vLLM generation worker, scheduler, sampler,
processed-logprob mode, sleep/wake lifecycle, and refit transport.

The model uses vLLM's tensor-parallel embeddings, linear layers, logits processor, and attention
layer. Evo2-specific modules own only Evo2 architecture and recurrent math. The implementation must
not wrap the Megatron model or invoke the Megatron inference engine from inside vLLM.

## Packed Sequence Semantics

Packing is physical, not semantic. A flattened activation tensor contains adjacent request segments,
but every recurrent operation receives `query_start_loc`/`cu_seqlens`, state-slot indices, and
`has_initial_state`. No FIR tap, IIR recurrence, output, or final state may cross a segment boundary.

For a packed batch with lengths `[L0, L1, ...]`, request `i` occupies
`[cu_seqlens[i], cu_seqlens[i + 1])`. Its initial state comes only from its assigned vLLM cache slot,
and its final state is written only to that slot. Reordering, compaction, graph padding, partial
batches, and slot reuse must not change these rules.

Mixed prefill/decode forwards are split using vLLM's recurrent metadata and reassembled in original
token order. Decode never loops over requests in Python. Prefill may bucket requests by sequence
length for an FFT operation, but it must scatter outputs and final states back by authoritative
request indices and cannot expose semantic padding to another layer.

## Recurrent State Contract

Every Hyena layer exposes exactly two fp32 state tensors through `MambaBase`:

1. Projection FIR state for the common width-3 input projection convolution.
2. Operator state:
   - HCS: width-7 FIR history;
   - HCM: width-128 FIR history;
   - HCL: 16-value real modal state per channel.

The cache-facing shape may be padded to the largest operator state when required by vLLM's hybrid
cache allocator. Each layer slices only its logical state shape, and tests place nonzero sentinels in
the padding to prove it is neither read nor modified. TP shards the channel dimension; no recurrent
state is replicated across TP ranks unless the corresponding activation is replicated by the model.

Prefix-cache block snapshots are disabled. One terminal state per active request is sufficient for
the current autoregressive rollout and chunked-prefill continuation lifecycle.

## Kernel Design

### Correctness References

The existing Vortex-derived `parallel_fir`, `step_fir`, `parallel_iir`, and `step_iir` implementations
are the independent dense references. Tests are written against those functions before optimized
code is added. A scalar token loop is the final authority for boundary, short-history, and final-state
behavior.

### Packed FIR

vLLM 0.20's Mamba causal-convolution kernel provides useful request metadata and cache conventions,
but its optimized paths are specialized around short Mamba widths. Evo2 needs widths 3, 7, and 128.
The Evo2 plugin therefore owns a packed FIR operation with this logical interface:

```python
packed_causal_fir(
    x, weight, bias, state_cache, query_start_loc, state_indices,
    has_initial_state, gated_bias=False, flip_filter=False,
) -> output
```

The operation updates `state_cache` in place. Widths 3 and 7 use a direct unrolled Triton kernel.
Width 128 starts with a direct segmented kernel and is compared against a length-bucketed
convolution/FFT path. The profiler decides which path is retained for short prefill, long prefill,
and single-token decode. The implementation may specialize these regimes, but all variants share
the same tests and state layout.

An optional fused projection-plus-operator path for HCS/HCM is added only if profiling shows that the
two packed FIR launches prevent throughput parity. It must be equivalent to the unfused reference.

### Packed HCL

HCL uses the fixed real diagonal recurrence:

```text
h[t] = exp(log_poles) * h[t-1] + x1[t] * v[t]
y[t] = x2[t] * (sum(residues * h[t]) + D * x1[t] * v[t])
```

Stock vLLM selective scan accepts packed request metadata, but its public operation expects B/C
values contiguous over tokens. Expanding Evo2's static residues across every token would create a
large temporary and is prohibited.

Single-token decode and short 4-12 token prefill use a dedicated fixed-coefficient segmented Triton
kernel. Long prefill initially uses the existing batched FFT implementation grouped into exact or
power-of-two length buckets. Right padding is internal to a bucket: outputs are gathered only at real
positions, and modal state is gathered at each request's true final position before any padded zero
could decay it. This preserves prefix invariance.

A parallel affine scan may replace FFT bucketing only if it is both exact within the established
tolerance and faster. No token-serial Python loop is allowed in the final 7B path.

## Model And Weight Loading

`Evo2ForCausalLM` reproduces Vortex layer order, normalization, projections, attention, MLP, tied or
duplicated embeddings as required by the checkpoint, and uppercase tokenizer behavior. Configuration
contains an explicit per-layer block pattern so vLLM identifies attention and recurrent layers.

The loader accepts an exported Vortex checkpoint for standalone validation and a stable vLLM
checkpoint directory for engine startup. Tensor-parallel loaders shard projections, attention heads,
MLP weights, filters, residues, poles, and biases along the same dimensions as the forward path.

For NeMo-RL, the refit mapping consumes the training model's MBridge/Megatron parameters and writes
the vLLM representation after every optimizer update. Nonlinear checkpoint conversions must be
identified explicitly and tested. Refit must be streamed and TP-aware; it cannot construct a second
complete 7B state dict on each GPU or silently leave filter parameters stale.

Two-rollout testing proves that updated weights are visible after refit and that recurrent state and
CUDA graphs from the first rollout do not leak into the second.

## Distributed Execution

TP uses vLLM's tensor-parallel process group and native parallel linear layers. Every TP rank receives
the same request metadata and sampling result while owning only its parameter and recurrent-state
channel shard.

DP uses independent vLLM engines/workers as orchestrated by NeMo-RL. Each replica receives a distinct
request shard and deterministic seed stream. Results are restored to global input order exactly once.

Required hardware proofs are TP2/DP1 and TP1/DP2. The TP2/DP2 design must specify rank groups, model
and state sharding, refit collectives, seed derivation, and result gathering. It is recorded as
unexecuted unless four GPUs become available; no two-GPU test may be represented as proof of a
four-GPU topology.

## Correctness Gates

Focused kernel tests cover:

- widths 3, 7, and 128;
- HCL order 16 with real fp32 state;
- zero-length padding entries and 1-token through long segments;
- mixed 4-12 token lengths in one packed tensor;
- absent and present initial state;
- arbitrary and reverse state-slot indices plus repeated vLLM null-block entries (block `0`);
- state padding sentinels and freed-slot reuse;
- packed output and final-state equivalence to independent dense and scalar references;
- full prefill versus multiple chunked-prefill calls;
- mixed prefill and decode in one scheduler step;
- eager versus compile and CUDA-graph decode.

End-to-end accuracy covers:

- exact repeated greedy token IDs for one topology and execution layout;
- close first-step logits and processed log-probabilities versus Vortex/MCore;
- the existing 1B second-half generation identity goldens;
- the 7B Microviridae checkpoint on representative biological prompts;
- stochastic replay with temperature/top-k/top-p;
- every request returned once in original order;
- TP2/DP1 and TP1/DP2 output validity;
- two NeMo-RL rollout/refit cycles.

Batch-layout numerical differences may cause later greedy divergence only after first-token logits
and log-probabilities pass tight dtype-appropriate tolerances. Biological identity must remain within
five percentage points of the serial reference and cannot fall below the existing test threshold.

## Performance Gate

The immutable MCore baseline and vLLM candidate use the same:

- base 7B Microviridae checkpoint and dtype;
- tokenizer and tokenized prompt corpus;
- 96-request global batch;
- homogeneous prompt controls and a deterministic 4-12 token mixed-length batch;
- generation length, stop handling, sampling parameters, and seed;
- TP/DP topology;
- CUDA-graph scope, compile configuration, warmup count, and GPU clocks/environment.

Measure generation tokens/second, requests/second, TTFT, median and p95 inter-token latency, peak
allocated/reserved memory, and profiler-visible launch/copy/synchronization counts. Initial tuning
uses two warmups and five interleaved repetitions. The final gate uses three warmups and at least ten
interleaved baseline/candidate repetitions, reporting medians and dispersion.

The candidate passes only if sustained generation throughput and requests/second match or exceed
MCore within measured run variance. It also must not regress median or p95 inter-token latency, exceed
MCore peak memory by more than 5%, introduce a per-token host/device synchronization or copy, use a
serial request fallback, or reduce accuracy. A failed gate triggers profiling and optimization; it is
not waived because vLLM simplifies orchestration.

## No-Shortcut Rules

- No concatenated genome may be treated as one Hyena recurrence.
- No per-request Python loop may remain in decode or the production short-prompt prefill path.
- No hidden serial generation fallback may satisfy a packed or distributed test.
- No output-only smoke test may substitute for state and logit equivalence.
- No tiny/random model result may substitute for the real 1B identity and 7B workload gates.
- No eager-only result may substitute for CUDA-graph and compile measurements.
- No TP1 result may substitute for TP2, and no single replica may substitute for DP2.
- No startup-only load may substitute for a post-optimizer refit and second rollout.
- No vLLM source patch may be required by the final plugin.

## Final MCore API Review

After all vLLM correctness, distributed, and performance work is complete, inspect the then-current
PR5274 lab API read-only. Produce a separate report that answers:

1. Which state-shape, packed-metadata, projection, kernel, sampling-logprob, seed, and lifecycle
   contracts can be shared between the MCore and vLLM implementations?
2. Which MCore API changes would make Evo2 an ordinary external model without coupling MCore to
   Vortex or vLLM?
3. Should packed FIR/HCL kernels live in `bionemo-evo2` beneath both backends, and what backend-neutral
   signatures should they expose?
4. Which PR5274 changes remain useful if production GDPO inference moves entirely to vLLM?
5. Which changes are unnecessary or should be kept out of the MCore PR?

This review is advisory only. It does not edit or interrupt the MCore lab.

## Completion Evidence

Completion requires committed source and tests, exact environment and checkpoint manifests, raw
correctness outputs, profiler traces, distributed run logs, benchmark tables, the final MCore API
review, and a requirement-by-requirement audit. The experiment remains incomplete if any required
gate is missing, indirect, or measured on a smaller substitute workload.
