# Evo2 vLLM GDPO runtime contract

Use this contract when vLLM generates NeMo-RL rollouts. Keep the current
backend as a matched control until complete train and validation steps pass.

## Build and runtime environments

On the actual execution host, run `./.ci_build.sh` and only then
`source .ci_test_env.sh`. The build retains the exact recursive NeMo-RL source,
applies the recipe-owned patches, and creates a locked vLLM actor environment
under `$NEMO_RL_VENV_DIR`. The main environment remains on the container's
BioNeMo/Megatron Python and Torch stack; vLLM actors use the separately verified
Python, Torch, and official vLLM versions selected by the pinned NeMo-RL lock.

Resolve the actor interpreter instead of inventing a `PYTHONPATH`:

```bash
VLLM_PYTHON="$(python -c 'from bionemo.evo2_phage_gen.nemo_rl_patches import vllm_actor_python_executable; print(vllm_actor_python_executable())')"
test -x "$VLLM_PYTHON"
```

Older container Torch builds may lack `torch._opaque_base`, which current vLLM
compilation requires. In that case the isolated actor environment is required.
Newer Torch versions, including verified 2.13 builds, may expose the API in the
main environment, but do not bypass the recipe pin without a matched rebuild and
qualification. Never patch upstream vLLM core to work around an environment
mismatch.

## Batch and prompt groups

Define the logical batch as `P` prompt groups times `K` stochastic rollouts per
prompt: `P*K=train_global_batch_size`. The local physical generation batch is
`GBS/DP`; vLLM may pack that local set into one request batch or explicit waves
without semantically padding prompts to one length.

`policy.train_micro_batch_size` is the MCore forward/backward chunk. Test the
full local batch `N=GBS/DP` first. If measured memory or throughput requires a
smaller value, use the largest stable divisor of `N` and accumulate `N/MBS`
chunks. Capacity-test `policy.logprob_batch_size` independently. Neither field
defines prompt count, prompt length, rollout count, advantage groups, or vLLM
wave size.

The primary mixed workload is eight fixed PhiX174 prompt-length strata, 4
through 11, with 12 rollouts per stratum for GBS96. TP1/DP2 assigns six
rollouts from every stratum to each rank. Retain prompt ID, length stratum,
rollout ordinal, global request index, generation call, DP rank, and seed in
every row. Assemble all K rewards before within-prompt advantage normalization,
including when a prompt group spans DP ranks. Validation uses the frozen mixed
manifest and reports every stratum plus a predeclared equal-weight aggregate.
The historical length-10 bank is a control, not the primary selection metric.

## Checkpoint export and standalone generation

Export each selected MBridge policy to a fresh safetensors directory before
standalone generation or filtering:

```bash
python -m bionemo.evo2.vllm.export \
  /path/to/policy/weights/iter_0000000 \
  /path/to/fresh-vllm-export \
  --max-shard-size 2GiB
```

Hash the export manifest, config, tokenizer inventory, and safetensors index.
Run standalone vLLM modules with `$VLLM_PYTHON`, set `VLLM_PLUGINS=evo2`, and
select topology from assigned GPUs. The qualified two-H100 reference is
O2/balanced, compilation mode 3, exact-batch `FULL_AND_PIECEWISE` CUDA graphs,
the multiprocess executor, and async scheduling. TP2 is a measured reference,
not a universal maximum: test wider TP and disjoint DP engine groups when more
GPUs are assigned, recompute local batch/capture sizes, and retain the same
correctness gates.

Use the public inference path and independently retained RL authority for final
generation/filtering:

```bash
"$VLLM_PYTHON" -m bionemo.evo2.vllm.infer \
  --model /path/to/fresh-vllm-export \
  --rl-checkpoint /path/to/policy/weights/iter_0000000 \
  --rl-tokenizer-json /path/to/tokenizer.json \
  --prompt-file /path/to/frozen-prompts.jsonl \
  --max-new-tokens 5988 \
  --temperature 1.0 --top-p 1.0 --top-k 4 \
  --tensor-parallel-size auto \
  --batch-size 96 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.91 \
  --optimization-level 2 --performance-mode balanced \
  --async-scheduling \
  --output-file /path/to/final-generations.jsonl
```

The run manifest must retain the successful RL/export load-parity record. Size
the physical batch and exact graph capture to the assigned topology; do not pin
TP2 when a different tested TP uses the available GPUs more effectively.

## Provenance and patches

Freeze the official vLLM version and source tag or wheel hash, BioNeMo Evo2
plugin/model revision, NeMo-RL revision, exported model/config/tokenizer hashes,
and every recipe-owned dependency patch path/hash. Prove each patch against its
pinned clean base with forward/reverse or runtime checks before CUDA. Required
behavior may not exist only in a lab checkout or `/tmp`.

An upstream vLLM sampler, scheduler, worker, or other core patch or runtime
monkeypatch is a stop-and-review condition. Diagnostic instrumentation does not
belong in the promoted runtime.

## Topology and numerical gates

Run actual TP2/DP1 and TP1/DP2 one-step loops on two GPUs. Prove disjoint DP
request, global-index, and seed streams. Separately test TP>1 plus DP>1 resource
group composition and fail closed when resources are insufficient; do not claim
a four-GPU runtime on two GPUs.

For every bounded smoke and final comparison require:

- exact A/C/G/T/N completion alphabet and requested length, with no retained EOS;
- finite aligned chosen rollout logprobs and the unchanged sequence-logprob threshold;
- rollout-versus-policy token absolute-delta mean/p95/max, sequence-sum delta,
  importance-ratio, and clipping diagnostics;
- finite rewards, prompt-group advantages, loss, gradients, and optimizer state;
- an observable optimizer-step counter and parameter/checksum change.

After the optimizer step, export/convert MCore weights, refit vLLM, synchronize
all ranks, and prove the new weights are active before validation or the next
rollout. Any failed or stale refit stops the attempt.

## Timing and final cleanup

Compare matched current-backend and vLLM train and validation steps. Record
rollout, reward/QC, policy-logprob forward, reference/KL work, backward,
optimizer, export/conversion, refit/sync, barriers, validation, peak memory, and
outer total wall time. vLLM wins only when generation savings offset refit and
integration overhead in the matched total step.

Start with a bounded mixed one-step smoke, then GBS96 and repeated steps. Before
promotion, remove diagnostic/timing monkeypatches and superseded patches, retain
useful behavior as tests, reconstruct from recipe-owned artifacts, and rerun
correctness, topology, numerical, refit, timing, lint, and skill-eval gates.
