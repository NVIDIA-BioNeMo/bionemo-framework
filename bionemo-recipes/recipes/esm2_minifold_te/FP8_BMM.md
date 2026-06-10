# FP8 BMM Triangular Multiplication

`tri_impl: "fp8_bmm"` enables the original MXFP8 batched-GEMM path for MiniFold triangular multiplication.

`tri_impl: "fp8_cublaslt"` enables the newer raw cuBLASLt MXFP8 batched-GEMM path.

## CUTLASS Note

The lower-level native-kernel exploration now depends on a local CUTLASS checkout:

- repo: `https://github.com/NVIDIA/cutlass.git`
- pinned ref: `08185b9c3e90510ee2b656662ed0d53b06d28157`
- expected path in the container: `/workspace/third_party/cutlass`

Important architecture note:

- this development box is `sm103` (`NVIDIA B300 SXM6 AC`)
- the stock CUTLASS MXFP8 grouped examples we probed (`81_*`, `92_*`) are gated for `100a`
- that means the current CUTLASS checkout is useful for source reference and custom-kernel development here, but the out-of-the-box MXFP8 grouped example binaries are not a direct benchmark on this machine

This needs to stay in future reporting so we do not mistake an example-ISA mismatch for a kernel-quality issue in the tri-mul codepath.

## What It Does

- quantizes each triangular-multiply operand to real MXFP8 with explicit block scales
- executes the matmul through the `bmm_ext` Transformer-Engine/cuBLAS FP8 backend
- accumulates in FP32
- returns BF16 or FP32 outputs
- reuses saved FP8 tensors and scales in backward instead of saving BF16 activations
- preserves TE `Float8Tensor` outputs across `pi`, `gi`, `po`, and `go` linear boundaries when the enclosing block is in FP8 autocast

The recipe routes this backend through the same one-permute `(B, D, N, N)` flow used by the BF16 `bmm` and `cublas_xbdnn` paths.

## Config

```yaml
component_precision:
  tri_einsum: "bf16"
  tri_impl: "fp8_cublaslt"
```

Example override:

```bash
torchrun --nproc_per_node=8 train_fsdp2.py \
  --config-name run_100_real_3B \
  component_precision.tri_einsum=bf16 \
  component_precision.tri_impl=fp8_cublaslt
```

For before/after comparisons, keep `seed` fixed so the synthetic dataset and
training loop are reproducible across backend changes.

## Shape Constraints

The current MXFP8 path assumes production-aligned shapes:

- sequence length `N` must be divisible by `32`
- each triangular branch width `D_chunk` must be divisible by `32`
- the backend expects square `N x N` slices

These constraints match the current MiniFold recipe regime where `c_z = 128`, so `D_chunk = 32`, and the target sequence lengths are block-aligned.

## Implementation Notes

- forward path:
  - `pi`, `gi`, `po`, and `go` can emit TE `Float8Tensor` outputs when the block is in FP8 autocast
  - `sigmoid`, gating multiply, masking multiply, layernorm, and residual/add are forced higher-precision boundaries
  - after the input gate and mask, `TriangularUpdateTE` permutes once into `(B, D, N, N)`
  - `tri_mul_fp8_bdnn(...)` consumes the post-gating tensor through the packed MXFP8 path
- backward path:
  - upstream gradients are quantized fresh to MXFP8
  - saved FP8 forward operands and scales are reused
  - gradients are computed with two FP8 GEMM calls

## Precision Boundaries

This backend does not keep one single FP8 tensor alive through the entire
`proj -> gate -> tri-mul -> proj -> gate` chain. That would be numerically wrong
for MiniFold because:

- `sigmoid` must break back to BF16/FP32
- the elementwise gating multiply must run in higher precision
- mask application, layernorm, and residual/add also break the FP8 chain

The correct interpretation is:

1. keep TE linear outputs quantized where the API supports it
2. break to BF16 at nonlinear / elementwise boundaries
3. repack the post-gating tensor once for the triangular FP8 GEMMs

## Backend Status

- `fp8_bmm`: older TE-backed MXFP8 route kept for comparison
- `fp8_cublaslt`: current recommended MXFP8 route

On the measured MiniFold-aligned tri-mul shapes, `fp8_cublaslt` is materially faster than `fp8_bmm` while preserving the same MXFP8 output as the TE raw backend.

## Files

- recipe integration: [te_utils.py](/workspace/bionemo-framework/bionemo-recipes/recipes/esm2_minifold_te/te_utils.py)
- module dispatch: [miniformer_te.py](/workspace/bionemo-framework/bionemo-recipes/recipes/esm2_minifold_te/miniformer_te.py)
- precision config: [quantization.py](/workspace/bionemo-framework/bionemo-recipes/recipes/esm2_minifold_te/quantization.py)
- FP8 kernel wrapper: [/workspace/claude_tasks/trimul/bmm/bmm_ext/ops.py](/workspace/claude_tasks/trimul/bmm/bmm_ext/ops.py)

## Tests

Relevant coverage includes:

- extension-level FP8 quantization and backward tests in `/workspace/claude_tasks/trimul/bmm/tests/test_api.py`
- recipe precision/backend tests in [tests/test_precisions.py](/workspace/bionemo-framework/bionemo-recipes/recipes/esm2_minifold_te/tests/test_precisions.py)
- module shape smoke tests in [tests/test_model.py](/workspace/bionemo-framework/bionemo-recipes/recipes/esm2_minifold_te/tests/test_model.py)
