# Tensor-parallel TopK SAE — stepwise plan (tests-first)

Goal: shard the SAE latents across GPUs so the latent count reaches ~1M (and beyond)
on 8× H100. Standard latent-sharding design (the OpenAI public repo does **not**
include TP — only the single-device model + the kernels we ported — so this is a
from-design build informed by the paper).

## Principles

- **Tests first.** Every step writes its test before its code; the test defines correctness.
- **Additive / minimal blast radius.** New code in new files. **Do not modify** the dense
  `TopKSAE` or the `Trainer` DDP path. The only edits to shared training code are a narrow,
  additive TP path guarded by `tp_size > 1`. Recipe changes are additive config + one branch.
- **CPU-first validation.** Almost everything is verified with a `gloo` multi-process
  harness (world_size 2 and 4) on CPU — no GPU needed until the final 1M run.
- **The parity oracle** (testing backbone): take a single-process dense `TopKSAE`, split its
  weights into shards, load them into the sharded model across ranks, and assert the sharded
  forward/loss/grads match the dense model within fp tolerance.

## Design recap (rank `r` owns latents `[r·L, (r+1)·L)`, `L = n/P`)

- Shard `W_enc_local [L,d]`, `latent_bias_local [L]`, `W_dec_local [d,L]`; `pre_bias [d]` replicated; `x` replicated.
- Encode → ReLU → **local** top-k → `all_gather` candidates (indices offset by `r·L`) → **global** top-k over `P·k`.
- Decode local selection via the Triton sparse kernel → partial recon → `all_reduce(sum)` → `+ pre_bias`.
- Backward flows through the collectives; optimizer is free-sharded; replicated `pre_bias` grad is all-reduced.

______________________________________________________________________

## Phase A — sharded architecture (CPU/gloo, no GPU)

> Status: **Phase A complete & green** (A0–A5; CPU/gloo, world 2 & 4). `ShardedTopKSAE`
> matches the dense `TopKSAE` exactly on forward, all gradients, full `loss()` metrics
> (`total/fvu/sparsity/mse/variance_explained/dead_pct`), per-shard dead stats, and the
> auxk path. Harness uses spawn (fork-after-autograd is unsafe in the mixed GPU/CPU
> suite). Full suite: 37 passed. Next: Phase B (B0 sharded checkpoints).

### A0. Distributed test harness ← test infra first

- New `sae/tests/_dist_utils.py`: `run_distributed(fn, world_size, backend="gloo")` via
  `torch.multiprocessing.spawn`, init/destroy process group, collect per-rank asserts.
- Test: trivial all-reduce/all-gather sanity across world=2 to prove the harness works.

### A1. Comms + TP process group

- New `sae/src/sae/parallel/comms.py`: thin `all_gather_cat`, `all_reduce_sum` (autograd-aware
  where needed) + TP-group helpers (reuse `process_group_manager.py`).
- **Test first** (`test_tp_comms.py`): all_gather concatenation and all_reduce-sum correctness,
  and that an autograd `all_reduce_sum` passes `gradcheck` (gloo, world=2).

### A2. Global top-k across shards ← the trickiest bit, isolated

- `global_topk(pre_act_local, k, rank, L)` in `parallel/`: local top-k → all_gather (offset
  indices) → global top-k.
- **Test first** (`test_tp_global_topk.py`): for random `pre_act` split across world∈{2,4},
  the sharded `(values, global_indices)` **exactly equal** single-process `torch.topk` over the
  concatenated latents (order-insensitive set compare + value match). Edge cases: ties, k=1,
  a latent's mass concentrated on one rank.

### A3. `ShardedTopKSAE.forward` / encode / decode

- New `sae/src/sae/architectures/topk_tp.py`: `ShardedTopKSAE` (own sharded params; small
  helpers like `_normalize` duplicated from `TopKSAE` so we don't touch it). Decode uses the
  existing `TritonDecoderAutograd` (dense fallback on CPU/no-triton for tests).
- **Test first** (`test_tp_topk.py::forward_parity`): build dense `TopKSAE`, shard weights into
  `ShardedTopKSAE` across world∈{2,4}, assert recon (fwd) and all param grads (bwd) match dense
  within tolerance; `normalize_input` on/off.

### A4. `ShardedTopKSAE.loss` + metrics + dead-latent stats

- Mirror dense `loss()` keys (`total/fvu/mse/sparsity/variance_explained/dead_pct`); dead stats
  tracked on the local shard.
- **Test first**: loss-dict parity vs dense (`total`, `mse`, `fvu`, `l0`) and dead-stat parity.

### A5. auxk (dead latents) under sharding

- Dead latents are local; compute aux loss per shard, `all_reduce` the scalar.
- **Test first**: aux-enabled loss parity vs dense within tolerance (document any approximation).

______________________________________________________________________

## Phase B — TP training, checkpoints, recipe

> Status: **Phase B COMPLETE (B0–B4) — TP converges matching dense.** 1.03M-latent SAE
> trains across 8× H100 at 16.2 GB/rank; a 172k-latent run matches the single-GPU dense
> baseline almost exactly (fvu→0.49, min loss 0.382 vs dense 0.380) at ~76k samples/s.
> sharded init_pre_bias + PerfLogger/W&B wired; eval.py merges sharded→dense (B4). 45 tests pass.
>
> Three bugs found & fixed during B3 (none were in the parity-tested per-step logic):
>
> 1. missing decoder normalization in the TP loop -> added (global dim=1 row-norm via all-reduce, matches dense).
> 2. x_var's replicated pre_bias gradient over-counted x world_size -> 1/world_size grad scaling (B1 caught it).
> 3. **each TP rank pulled a different batch from its own streaming dataloader** (the divergence cause)
>    -> rank 0 broadcasts the batch (and pre_bias after init) to the whole TP group.
>    Follow-up (minor): LR-scaling in the TP loop; a real 1M run for many steps.

### B0. Sharded checkpoints ← test first

- New `sae/src/sae/parallel/checkpoint.py`: `save_sharded` (per-rank slice + meta) and
  `load_and_merge` (→ a single dense `TopKSAE` state_dict for eval).
- **Test first** (`test_tp_checkpoint.py`): shard→save→merge round-trips to the original dense
  weights exactly; a merged checkpoint loads into dense `TopKSAE` and reproduces its outputs.

### B1. TP training path

- Narrow, additive `tp_size` support: per-rank device, **no DDP wrap** of sharded params,
  all-reduce of replicated (`pre_bias`) grads, optimizer over local params. (Either a small
  `_setup_tensor_parallel` hook in `Trainer` guarded by `tp_size>1`, or a thin standalone TP
  trainer in the recipe — chosen to leave the DDP path byte-identical.)
- **Test first** (gloo, world=2): one optimizer step on the sharded model matches the dense
  model's post-step weights (gather shards → compare), within tolerance.

### B2. Recipe wiring (additive)

- `configs`: `parallel.tp_size` (default 1). `train.py`: when `tp_size>1`, build
  `ShardedTopKSAE` and train **from cache** (extract→cache→TP-train workflow). No change to the
  default dense/DDP/streaming paths.
- `scripts/tp_1m.sh`: `torchrun --nproc_per_node=8 … parallel.tp_size=8 model.expansion_factor≈390`.

### B3. Real GPU validation (the 1M run)

- `torchrun --nproc_per_node=8`, 1M latents, short run from a cached activation store: assert
  per-GPU mem ~16–18 GB, loss decreases, PerfLogger metrics sane, sharded checkpoint written.

### B4. Eval

- `eval.py`: load via `load_and_merge` (sharded → dense on one GPU) so the existing dense eval +
  loss-recovered path is reused unchanged.

______________________________________________________________________

## Files

**New:** `sae/src/sae/parallel/{__init__,comms,checkpoint}.py`,
`sae/src/sae/architectures/topk_tp.py`,
`sae/tests/{_dist_utils,test_tp_comms,test_tp_global_topk,test_tp_topk,test_tp_checkpoint}.py`,
`recipes/nemotron/scripts/tp_1m.sh`.
**Minimal additive edits:** `architectures/__init__.py` (export `ShardedTopKSAE`),
`sae/__init__.py` (exports), recipe `configs` (`parallel.tp_size`), `train.py` (one `tp_size>1`
branch), `eval.py` (merge-load), and the single guarded TP hook in training.
**Untouched:** dense `TopKSAE`, the `Trainer` DDP path, streaming, the kernels.

## Acceptance criteria

1. Phases A–B0 fully verified on CPU (gloo, world 2 & 4): sharded == dense for top-k, forward,
   loss, grads, one optimizer step, and checkpoint round-trip.
2. `torchrun --nproc_per_node=8` trains a 1M-latent SAE from cache within H100 memory; loss
   decreases; sharded checkpoint saved and merge-loads for eval.
3. All existing tests still pass; dense/DDP/streaming behavior unchanged.

## Sequencing

A0 → A1 → A2 → A3 → A4 → A5 → B0 → B1 → B2 → B3 → B4. Phases A–B1 need no GPU and are the bulk
of the correctness work; only B3 requires the 8× H100s.
