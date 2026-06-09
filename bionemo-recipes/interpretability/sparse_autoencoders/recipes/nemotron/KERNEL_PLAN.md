# Sparse TopK SAE kernels — scaling plan

Goal: scale the TopK SAE latent count (toward ~1M+, ≈370× expansion on Nemotron's
d=2688) by avoiding the dense `[batch, n_latents]` code tensor and the full decoder
matmul. Mirrors [openai/sparse_autoencoder](https://github.com/openai/sparse_autoencoder).

## Latent-size targets (from OpenAI Gao et al. 2024 + Anthropic Scaling Monosemanticity 2024)

- k (sparsity): 32–64 (k=32 sweet spot).
- Ladder for d=2688: 8× (21.5k) → 32× (86k) → 128× (344k) → **~1M (≈370×)** → 2–4M (stretch).
- Dense `[batch=4096, n]` fp32 buffer: 8×=0.35GB, 32×=1.4GB, 128×=5.6GB, 256×=11.3GB,
  1M=16.4GB ×2 (enc+dec). Kernels become necessary at ≥128×.

## Phase 1 — Triton sparse decoder ✅ implemented (validation pending GPU)

Files:

- `sae/src/sae/kernels/triton_decoder.py` — ported kernels + `TritonDecoderAutograd`
  (`triton_sparse_dense_matmul`, `triton_sparse_transpose_dense_matmul`,
  `triton_dense_dense_sparseout_matmul`). Triton imported lazily (`HAS_TRITON`).
- `sae/src/sae/kernels/reference.py` — dense autograd oracle.
- `TopKSAE(decoder_impl="dense"|"triton")` — `"triton"` routes `forward()`/`loss()`
  through `_decode_topk_triton` / `_loss_triton`; dense path byte-identical; weights
  and checkpoints interchangeable (decoder_impl is not persisted in `_get_config`).
- Recipe: `model.decoder_impl` config knob; wired in `build_sae`.
- Tests: `sae/tests/test_kernels.py` (GPU-gated: fwd/bwd vs reference, end-to-end SAE
  parity incl. auxk).
- Benchmark: `python -m sae.benchmarks.bench_decoder` (dense vs triton latency + peak
  memory across the ladder; OOM cells are the headline).

### Validated on H100 (8/8 tests pass)

Correctness: kernel matches the fp64 reference to ~1e-5 — *more* accurate than TF32
cuBLAS (the test disables TF32 so the dense oracle is exact fp32).

SAE training path (`loss()` fwd+bwd, fp32, batch 4096, d=2688, k=32):

| expansion | n     | dense (fwd/bwd, mem) | triton (fwd/bwd, mem) |
| --------- | ----- | -------------------- | --------------------- |
| 8×        | 21.5k | 6.3/7.2ms, 2.2GB     | 13.7/15.2ms, 1.9GB    |
| 32×       | 86k   | 21/27ms, 8.1GB       | 22/29ms, 7.2GB        |
| 128×      | 344k  | 83/108ms, 31.6GB     | **54/80ms**, 28.2GB   |
| 256×      | 688k  | 184/230ms, 63.0GB    | **110/154ms**, 56.2GB |
| 390×      | 1.05M | **OOM**              | **OOM**               |

Findings:

- Triton is **1.5–1.7× faster at ≥128×** and uses ~10% less memory; at 8× it's slower
  (use dense there — `decoder_impl` default stays dense).
- **1M OOMs for BOTH** → Phase 1 (sparse decode) is not sufficient alone. The wall is
  now the dense encoder `pre_act [batch, n]` (16GB at 1M) + fp32 param grads, **not**
  the decoder. This confirms Phase 2 (encoder tiling) is required to reach 1M.
- atomic-add in the weight-grad kernel → grads nondeterministic at ~1e-3 (tests tolerate).

### Phase 1.5 (perf follow-up) — decoder layout

Our `nn.Linear` decoder is `[d, n]` row-major; the kernel reads its transpose with
uncoalesced loads (why 8× is slow). Maintaining a contiguous `[n, d]` decode view
(OpenAI's layout) should make triton faster across the board. Deferred to keep
checkpoint compatibility; revisit alongside Phase 2.

## Phase 2 — encoder tiling

Tile `x @ W_enc` over `n` with a streaming/running top-k so the dense `[b,n]` pre-act
is never resident — needed to push past ~256× toward 1M+.

## Phase 3 — CUDA kernels

Hand-written CUDA for the hottest paths (fused top-k + sparse decode, transpose-sparse
matmul for `grad_W`), bf16/fp16, beyond Triton's ceiling.
