# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton kernels for the sparse TopK SAE decoder.

Adapted from OpenAI's sparse autoencoder kernels
(https://github.com/openai/sparse_autoencoder, MIT License, Copyright (c) OpenAI).

The decoder of a TopK SAE only touches ``k`` of ``n_latents`` columns per token, so
materializing the dense ``[batch, n_latents]`` code tensor and running a full
``[batch, n] @ [n, d]`` matmul is wasteful. These kernels operate directly on the
top-k ``(indices, values)`` so the decode is ``O(batch * k * d)`` instead of
``O(batch * n * d)`` and never allocates the dense code tensor -- which is what
lets the latent count scale to ~1M+.

``TritonDecoderAutograd.apply(indices, values, decoder_weight)`` computes the
reconstruction (pre-bias) and its gradients. ``decoder_weight`` is ``[d, n]``
(i.e. ``nn.Linear(n, d, bias=False).weight``); the kernel uses its transpose.

Triton is imported lazily so this module is importable on CPU-only machines; the
kernels still require a CUDA device at call time.
"""

import torch


try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on environment
    HAS_TRITON = False


if HAS_TRITON:

    def triton_sparse_dense_matmul(
        sparse_indices: torch.Tensor,
        sparse_values: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``sparse @ dense`` (reduce along the uncollated dim of sparse).

        sparse_indices/sparse_values are (A, k); dense is (N, B); output is (A, B).
        ``dense`` must be contiguous along dim 0 (i.e. ``dense.T`` is contiguous).
        """
        N = dense.shape[0]
        assert sparse_indices.shape == sparse_values.shape
        assert sparse_indices.is_contiguous()
        assert sparse_values.is_contiguous()
        # NOTE: OpenAI asserts dense.is_contiguous(). Our decoder weight is a
        # standard nn.Linear [d, n], so dense = weight.T is a strided [n, d] view.
        # The kernel reads via stride_dn/stride_db so it is correct on the strided
        # view (loads along B are uncoalesced -> a perf, not correctness, follow-up).

        A = sparse_indices.shape[0]
        K = sparse_indices.shape[1]
        B = dense.shape[1]

        out = torch.zeros(A, B, device=dense.device, dtype=sparse_values.dtype)

        triton_sparse_dense_matmul_kernel[(A,)](
            sparse_indices,
            sparse_values,
            dense,
            out,
            stride_dn=dense.stride(0),
            stride_db=dense.stride(1),
            A=A,
            B=B,
            N=N,
            K=K,
            BLOCK_SIZE_K=triton.next_power_of_2(K),
            BLOCK_SIZE_B=triton.next_power_of_2(B),
        )
        return out

    @triton.jit
    def triton_sparse_dense_matmul_kernel(  # noqa: D103 - low-level Triton kernel (ported)
        sparse_indices_ptr,
        sparse_values_ptr,
        dense_ptr,
        out_ptr,
        stride_dn,
        stride_db,
        A,
        B,
        N,
        K,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_B: tl.constexpr,
    ):
        pid = tl.program_id(0)

        offsets_k = tl.arange(0, BLOCK_SIZE_K)
        sparse_indices = tl.load(sparse_indices_ptr + pid * K + offsets_k, mask=offsets_k < K)  # (K,)
        sparse_values = tl.load(sparse_values_ptr + pid * K + offsets_k, mask=offsets_k < K)  # (K,)

        accum = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)
        offsets_b = tl.arange(0, BLOCK_SIZE_B)

        for k in range(K):
            # workaround to do sparse_indices[k]
            i = tl.sum(
                tl.where(
                    tl.arange(0, BLOCK_SIZE_K) == k,
                    sparse_indices,
                    tl.zeros((BLOCK_SIZE_K,), dtype=tl.int64),
                )
            )
            # workaround to do sparse_values[k]
            v = tl.sum(
                tl.where(
                    tl.arange(0, BLOCK_SIZE_K) == k,
                    sparse_values,
                    tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32),
                )
            )

            tl.device_assert(i < N)
            if v != 0:
                accum += v * tl.load(dense_ptr + i * stride_dn + offsets_b * stride_db, mask=offsets_b < B)

        tl.store(out_ptr + pid * B + offsets_b, accum.to(sparse_values.dtype), mask=offsets_b < B)

    def triton_sparse_transpose_dense_matmul(
        sparse_indices: torch.Tensor,
        sparse_values: torch.Tensor,
        dense: torch.Tensor,
        N: int,
        BLOCK_SIZE_AK=128,
    ) -> torch.Tensor:
        """Compute ``sparse.T @ dense`` (reduce along the collated dim of sparse).

        sparse_indices/sparse_values are (A, k); dense is (A, B); output is (N, B).
        """
        assert sparse_indices.shape == sparse_values.shape
        assert sparse_indices.is_contiguous()
        assert sparse_values.is_contiguous()
        assert dense.is_contiguous()  # contiguous along B

        K = sparse_indices.shape[1]
        A = dense.shape[0]
        assert sparse_indices.shape[0] == A

        # COO-format and sorted (by latent index) so equal latents are contiguous.
        sorted_indices = sparse_indices.view(-1).sort()
        coo_indices = torch.stack(
            [
                torch.arange(A, device=sparse_indices.device).repeat_interleave(K)[sorted_indices.indices],
                sorted_indices.values,
            ]
        )  # (2, A * K)
        coo_values = sparse_values.view(-1)[sorted_indices.indices]  # (A * K,)
        return triton_coo_sparse_dense_matmul(coo_indices, coo_values, dense, N, BLOCK_SIZE_AK)

    def triton_coo_sparse_dense_matmul(  # noqa: D103 - low-level Triton kernel wrapper (ported)
        coo_indices: torch.Tensor,
        coo_values: torch.Tensor,
        dense: torch.Tensor,
        N: int,
        BLOCK_SIZE_AK=128,
    ) -> torch.Tensor:
        AK = coo_indices.shape[1]
        B = dense.shape[1]

        out = torch.zeros(N, B, device=dense.device, dtype=coo_values.dtype)

        grid = lambda META: (triton.cdiv(AK, META["BLOCK_SIZE_AK"]), 1)  # noqa: E731
        triton_sparse_transpose_dense_matmul_kernel[grid](
            coo_indices,
            coo_values,
            dense,
            out,
            stride_da=dense.stride(0),
            stride_db=dense.stride(1),
            B=B,
            N=N,
            AK=AK,
            BLOCK_SIZE_AK=BLOCK_SIZE_AK,
            BLOCK_SIZE_B=triton.next_power_of_2(B),
        )
        return out

    @triton.jit
    def triton_sparse_transpose_dense_matmul_kernel(  # noqa: D103 - low-level Triton kernel (ported)
        coo_indices_ptr,
        coo_values_ptr,
        dense_ptr,
        out_ptr,
        stride_da,
        stride_db,
        B,
        N,
        AK,
        BLOCK_SIZE_AK: tl.constexpr,
        BLOCK_SIZE_B: tl.constexpr,
    ):
        pid_ak = tl.program_id(0)
        pid_b = tl.program_id(1)

        coo_offsets = tl.arange(0, BLOCK_SIZE_AK)
        b_offsets = tl.arange(0, BLOCK_SIZE_B)

        A_coords = tl.load(
            coo_indices_ptr + pid_ak * BLOCK_SIZE_AK + coo_offsets,
            mask=pid_ak * BLOCK_SIZE_AK + coo_offsets < AK,
        )
        K_coords = tl.load(
            coo_indices_ptr + pid_ak * BLOCK_SIZE_AK + coo_offsets + AK,
            mask=pid_ak * BLOCK_SIZE_AK + coo_offsets < AK,
        )
        values = tl.load(
            coo_values_ptr + pid_ak * BLOCK_SIZE_AK + coo_offsets,
            mask=pid_ak * BLOCK_SIZE_AK + coo_offsets < AK,
        )

        last_k = tl.min(K_coords)
        accum = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)

        for ind in range(BLOCK_SIZE_AK):
            if ind + pid_ak * BLOCK_SIZE_AK < AK:
                # workaround to do A_coords[ind]
                a = tl.sum(
                    tl.where(tl.arange(0, BLOCK_SIZE_AK) == ind, A_coords, tl.zeros((BLOCK_SIZE_AK,), dtype=tl.int64))
                )
                k = tl.sum(
                    tl.where(tl.arange(0, BLOCK_SIZE_AK) == ind, K_coords, tl.zeros((BLOCK_SIZE_AK,), dtype=tl.int64))
                )
                v = tl.sum(
                    tl.where(tl.arange(0, BLOCK_SIZE_AK) == ind, values, tl.zeros((BLOCK_SIZE_AK,), dtype=tl.float32))
                )

                tl.device_assert(k < N)

                if k != last_k:
                    tl.atomic_add(
                        out_ptr + last_k * B + BLOCK_SIZE_B * pid_b + b_offsets,
                        accum,
                        mask=BLOCK_SIZE_B * pid_b + b_offsets < B,
                    )
                    accum *= 0
                    last_k = k

                if v != 0:
                    accum += v * tl.load(dense_ptr + a * stride_da + b_offsets, mask=b_offsets < B)

        tl.atomic_add(
            out_ptr + last_k * B + BLOCK_SIZE_B * pid_b + b_offsets,
            accum,
            mask=BLOCK_SIZE_B * pid_b + b_offsets < B,
        )

    def triton_dense_dense_sparseout_matmul(
        dense1: torch.Tensor,
        dense2: torch.Tensor,
        at_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Equivalent to ``(dense1 @ dense2).gather(1, at_indices)``.

        dense1 is (A, B); dense2 is (B, N); at_indices is (A, K); output is (A, K).
        """
        A, B = dense1.shape
        N = dense2.shape[1]
        assert dense2.shape[0] == B
        assert at_indices.shape[0] == A
        K = at_indices.shape[1]
        assert at_indices.is_contiguous()
        assert dense1.stride(1) == 1, "dense1 must be contiguous along B"
        # dense2 ([d, n] decoder weight) is read via stride_d2b/stride_d2n, so a
        # row-major [d, n] (stride(0)=n) is fine even though OpenAI assumes stride(0)==1.

        if K > 512:
            # naive is more efficient for large K
            return (dense1 @ dense2).gather(1, at_indices)

        out = torch.zeros(A, K, device=dense1.device, dtype=dense1.dtype)

        triton_dense_dense_sparseout_matmul_kernel[(A,)](
            dense1,
            dense2,
            at_indices,
            out,
            stride_d1a=dense1.stride(0),
            stride_d1b=dense1.stride(1),
            stride_d2b=dense2.stride(0),
            stride_d2n=dense2.stride(1),
            A=A,
            B=B,
            N=N,
            K=K,
            BLOCK_SIZE_B=triton.next_power_of_2(B),
            BLOCK_SIZE_N=triton.next_power_of_2(N),
            BLOCK_SIZE_K=triton.next_power_of_2(K),
        )
        return out

    @triton.jit
    def triton_dense_dense_sparseout_matmul_kernel(  # noqa: D103 - low-level Triton kernel (ported)
        dense1_ptr,
        dense2_ptr,
        at_indices_ptr,
        out_ptr,
        stride_d1a,
        stride_d1b,
        stride_d2b,
        stride_d2n,
        A,
        B,
        N,
        K,
        BLOCK_SIZE_B: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid = tl.program_id(0)

        offsets_k = tl.arange(0, BLOCK_SIZE_K)
        at_indices = tl.load(at_indices_ptr + pid * K + offsets_k, mask=offsets_k < K)  # (K,)

        offsets_b = tl.arange(0, BLOCK_SIZE_B)
        dense1 = tl.load(dense1_ptr + pid * stride_d1a + offsets_b * stride_d1b, mask=offsets_b < B)  # (B,)

        accum = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)

        for k in range(K):
            # workaround to do at_indices[k]
            i = tl.sum(
                tl.where(tl.arange(0, BLOCK_SIZE_K) == k, at_indices, tl.zeros((BLOCK_SIZE_K,), dtype=tl.int64))
            )
            tl.device_assert(i < N)

            dense2col = tl.load(dense2_ptr + offsets_b * stride_d2b + i * stride_d2n, mask=offsets_b < B)  # (B,)
            # NOTE: fixed vs upstream OpenAI, which used tl.int64 zeros here for a
            # float32 accumulator -- a type bug that truncates value gradients on
            # current Triton. The else-branch must be float32 to match `accum`.
            accum += tl.where(
                tl.arange(0, BLOCK_SIZE_K) == k,
                tl.sum(dense1 * dense2col),
                tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32),
            )

        tl.store(out_ptr + pid * K + offsets_k, accum, mask=offsets_k < K)

    class TritonDecoderAutograd(torch.autograd.Function):
        """Sparse TopK decode with custom forward/backward (mirrors OpenAI)."""

        @staticmethod
        def forward(ctx, sparse_indices, sparse_values, decoder_weight):
            """Reconstruction = sparse(top-k) @ decoder_weight.T (no dense codes)."""
            ctx.save_for_backward(sparse_indices, sparse_values, decoder_weight)
            return triton_sparse_dense_matmul(sparse_indices, sparse_values, decoder_weight.T)

        @staticmethod
        def backward(ctx, grad_output):
            """Gradients for sparse_values (gathered) and decoder_weight (sparse-transpose)."""
            sparse_indices, sparse_values, decoder_weight = ctx.saved_tensors

            # The transpose/sparseout kernels require a contiguous grad_output.
            grad_output = grad_output.contiguous()

            decoder_grad = triton_sparse_transpose_dense_matmul(
                sparse_indices, sparse_values, grad_output, N=decoder_weight.shape[1]
            ).T

            return (
                None,
                triton_dense_dense_sparseout_matmul(grad_output, decoder_weight, sparse_indices),
                # decoder is contiguous when transposed so this is a matching layout
                decoder_grad,
                None,
            )

else:  # pragma: no cover - exercised only when triton is unavailable

    class TritonDecoderAutograd:
        """Placeholder that errors clearly when Triton is unavailable."""

        @staticmethod
        def apply(*args, **kwargs):
            """Error: Triton unavailable, so the sparse decoder cannot run."""
            raise RuntimeError(
                "Triton is not available, so decoder_impl='triton' cannot run. "
                "Install triton (ships with recent PyTorch) or use decoder_impl='dense'."
            )
