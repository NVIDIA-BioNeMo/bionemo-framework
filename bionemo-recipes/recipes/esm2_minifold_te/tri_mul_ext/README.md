# tri_mul_ext

`tri_mul_ext` is the staging area for a fused triangular-multiplication kernel
for the ESM2 MiniFold recipe.

Current state:

- public API is in place: `tri_mul_fused(a, b, k_dim=...)`
- a specialized full-layout cuBLAS backend is available:
  `tri_mul_xbdnn_cublas(x_bdnn)`
- optional CUDA extension build is scaffolded
- current direct-layout CUDA and Triton kernels remain experimental
- benchmark script sweeps sequence length so scaling behavior can be evaluated
- current specialization targets the real training shape `D_chunk=32` only

The intended end state is a custom forward kernel that:

- reads the original `(B, N, N, D)` layout directly
- avoids `permute(...).contiguous()` materialization
- accumulates in FP32
- can later grow an internal low-precision path if forward scaling warrants it

Current constraints:

- `tri_mul_xbdnn_cublas` currently requires BF16 input and `x_bdnn.shape[1] == 128`
- direct fused kernels only support `a.shape[-1] == b.shape[-1] == 32`
- intended for the main MiniFold configs where `c_z=128` and tri-mul uses
  `D_chunk = c_z / 4 = 32`
- the recipe can opt into the low-level cuBLAS path with
  `BIONEMO_TRI_MUL_IMPL=cublas_xbdnn`

Build:

```bash
cd /workspace/bionemo-framework/bionemo-recipes/recipes/esm2_minifold_te/tri_mul_ext
python setup.py build_ext --inplace
# or let the recipe Dockerfile install it during image build
```

Benchmark:

```bash
cd /workspace/bionemo-framework/bionemo-recipes/recipes/esm2_minifold_te
PYTHONPATH=. python tri_mul_ext/examples/bench_tri_mul.py --sequence-lengths 128 256 384 512
```

Recipe integration:

```bash
export PYTHONPATH=/workspace/bionemo-framework/bionemo-recipes/recipes/esm2_minifold_te:$PYTHONPATH
export BIONEMO_TRI_MUL_IMPL=cublas_xbdnn
```

With the env var unset, the recipe stays on the current `torch.bmm` path. The
`cublas_xbdnn` backend is intended for `tri_einsum="bf16"` mode.
