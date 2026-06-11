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

"""Pure-PyTorch dense reference implementations of the sparse decoder ops.

These are the correctness oracle for the Triton kernels: simple, obviously-correct,
fully autograd-differentiable, and device/dtype agnostic. Tests compare the Triton
kernels against these (and against autograd through reference_decode for gradients).
"""

import torch


def reference_sparse_dense_matmul(
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    """Dense equivalent of ``triton_sparse_dense_matmul``: ``sparse @ dense``.

    sparse_indices/sparse_values are (A, k); dense is (N, B); output is (A, B).
    """
    # Gather the active rows of `dense` and weight by values: out[a] = sum_k vals[a,k]*dense[idx[a,k]].
    gathered = dense[sparse_indices]  # (A, k, B)
    return (gathered * sparse_values.unsqueeze(-1)).sum(dim=1)  # (A, B)


def reference_decode(
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    decoder_weight: torch.Tensor,
) -> torch.Tensor:
    """Dense, differentiable reference for ``TritonDecoderAutograd``.

    Builds the dense code tensor and runs the standard decoder matmul, so autograd
    through it yields reference gradients for ``sparse_values`` and ``decoder_weight``.
    ``decoder_weight`` is (d, n) (``nn.Linear(n, d).weight``); output is (A, d).
    """
    a, _ = sparse_indices.shape
    n = decoder_weight.shape[1]
    codes = torch.zeros(a, n, device=sparse_values.device, dtype=sparse_values.dtype)
    codes = codes.scatter(-1, sparse_indices, sparse_values)
    return codes @ decoder_weight.T  # (A, d)
