#include "common.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <vector>

namespace tri_mul_ext {

namespace {

void check_input(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 4, name, " must have shape (B, N, N, D)");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

c10::ScalarType parse_out_dtype(const std::string& out_dtype) {
  if (out_dtype == "bfloat16") {
    return c10::ScalarType::BFloat16;
  }
  if (out_dtype == "float16") {
    return c10::ScalarType::Half;
  }
  if (out_dtype == "float32") {
    return c10::ScalarType::Float;
  }
  TORCH_CHECK(false, "Unsupported out_dtype: ", out_dtype);
}

void check_bdnn_input(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 4, name, " must have shape (B, D, N, N)");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.size(1) == 32, name, " must use D_chunk == 32");
  TORCH_CHECK(t.size(2) == t.size(3), name, " must have square spatial dimensions");
}

void launch_batched_cublas(
    cublasHandle_t handle,
    const at::Tensor& a_ptrs,
    const at::Tensor& b_ptrs,
    const at::Tensor& c_ptrs,
    int64_t n,
    int64_t k_dim,
    c10::ScalarType out_type) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cublasOperation_t transa = k_dim == 2 ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb = k_dim == 2 ? CUBLAS_OP_N : CUBLAS_OP_T;
  const auto c_dtype = out_type == c10::ScalarType::Float ? CUDA_R_32F : CUDA_R_16BF;
  TORCH_CUDABLAS_CHECK(cublasGemmBatchedEx(
      handle,
      transa,
      transb,
      static_cast<int>(n),
      static_cast<int>(n),
      static_cast<int>(n),
      &alpha,
      reinterpret_cast<const void* const*>(b_ptrs.data_ptr<int64_t>()),
      CUDA_R_16BF,
      static_cast<int>(n),
      reinterpret_cast<const void* const*>(a_ptrs.data_ptr<int64_t>()),
      CUDA_R_16BF,
      static_cast<int>(n),
      &beta,
      reinterpret_cast<void* const*>(c_ptrs.data_ptr<int64_t>()),
      c_dtype,
      static_cast<int>(n),
      static_cast<int>(a_ptrs.numel()),
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

at::Tensor rowmajor_square_bmm_strided(
    const at::Tensor& left,
    const at::Tensor& right,
    bool trans_left,
    bool trans_right,
    c10::ScalarType out_type) {
  TORCH_CHECK(left.is_cuda() && right.is_cuda(), "rowmajor_square_bmm_strided expects CUDA tensors");
  TORCH_CHECK(left.dim() == 3 && right.dim() == 3, "rowmajor_square_bmm_strided expects 3D tensors");
  TORCH_CHECK(left.is_contiguous() && right.is_contiguous(), "rowmajor_square_bmm_strided expects contiguous tensors");
  TORCH_CHECK(left.scalar_type() == right.scalar_type(), "rowmajor_square_bmm_strided expects matching dtypes");
  TORCH_CHECK(left.size(0) == right.size(0), "batch dimensions must match");

  const int64_t batch = left.size(0);
  const int64_t n = left.size(1);
  TORCH_CHECK(left.size(2) == n, "left matrices must be square");
  TORCH_CHECK(right.size(1) == n && right.size(2) == n, "right matrices must match left square shape");

  auto out = at::empty({batch, n, n}, left.options().dtype(out_type));
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  auto stream = at::cuda::getDefaultCUDAStream();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int64_t stride = n * n;
  const void* left_ptr = nullptr;
  const void* right_ptr = nullptr;
  if (left.scalar_type() == c10::ScalarType::BFloat16) {
    left_ptr = left.data_ptr<at::BFloat16>();
    right_ptr = right.data_ptr<at::BFloat16>();
  } else if (left.scalar_type() == c10::ScalarType::Half) {
    left_ptr = left.data_ptr<at::Half>();
    right_ptr = right.data_ptr<at::Half>();
  } else {
    left_ptr = left.data_ptr<float>();
    right_ptr = right.data_ptr<float>();
  }
  void* out_ptr = out.data_ptr();
  const auto in_dtype =
      left.scalar_type() == c10::ScalarType::BFloat16 ? CUDA_R_16BF
      : left.scalar_type() == c10::ScalarType::Half   ? CUDA_R_16F
                                                      : CUDA_R_32F;
  const auto out_dtype =
      out_type == c10::ScalarType::BFloat16 ? CUDA_R_16BF
      : out_type == c10::ScalarType::Half   ? CUDA_R_16F
                                            : CUDA_R_32F;

  // Row-major GEMM via cuBLAS column-major view:
  // C_row = op(left_row) @ op(right_row)
  // C_col = C_row^T = op(right_row)^T @ op(left_row)^T
  // Therefore:
  // - cuBLAS A points to right storage with transa = op(right_row)
  // - cuBLAS B points to left storage with transb = op(left_row)
  const cublasOperation_t transa = trans_right ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb = trans_left ? CUBLAS_OP_T : CUBLAS_OP_N;

  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle,
      transa,
      transb,
      static_cast<int>(n),
      static_cast<int>(n),
      static_cast<int>(n),
      &alpha,
      right_ptr,
      in_dtype,
      static_cast<int>(n),
      stride,
      left_ptr,
      in_dtype,
      static_cast<int>(n),
      stride,
      &beta,
      out_ptr,
      out_dtype,
      static_cast<int>(n),
      stride,
      static_cast<int>(batch),
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  return out;
}

template <typename scalar_t>
__device__ inline float to_float(scalar_t x);

template <>
__device__ inline float to_float<float>(float x) {
  return x;
}

template <>
__device__ inline float to_float<at::Half>(at::Half x) {
  return __half2float(static_cast<__half>(x));
}

template <>
__device__ inline float to_float<at::BFloat16>(at::BFloat16 x) {
  return __bfloat162float(static_cast<__nv_bfloat16>(x));
}

template <typename scalar_t>
__device__ inline scalar_t from_float(float x);

template <>
__device__ inline float from_float<float>(float x) {
  return x;
}

template <>
__device__ inline at::Half from_float<at::Half>(float x) {
  return static_cast<at::Half>(__float2half_rn(x));
}

template <>
__device__ inline at::BFloat16 from_float<at::BFloat16>(float x) {
  return static_cast<at::BFloat16>(__float2bfloat16(x));
}

template <typename in_t, typename out_t, int BLOCK_I, int BLOCK_J, int MAX_D, bool K_DIM_TWO>
__global__ void tri_mul_tiled_dvec_kernel(
    const in_t* __restrict__ a,
    const in_t* __restrict__ b,
    out_t* __restrict__ out,
    int B,
    int N,
    int D) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.z;
  const int i = blockIdx.y * BLOCK_I + threadIdx.y;
  const int j = blockIdx.x * BLOCK_J + threadIdx.x;

  __shared__ in_t a_vec[BLOCK_I][MAX_D];
  __shared__ in_t b_vec[BLOCK_J][MAX_D];

  float acc[MAX_D];
  #pragma unroll
  for (int d = 0; d < MAX_D; ++d) {
    acc[d] = 0.0f;
  }

  for (int k = 0; k < N; ++k) {
    for (int idx = tid; idx < BLOCK_I * D; idx += BLOCK_I * BLOCK_J) {
      const int ii = idx / D;
      const int dd = idx % D;
      const int global_i = blockIdx.y * BLOCK_I + ii;
      if (global_i < N) {
        size_t a_idx;
        if constexpr (K_DIM_TWO) {
          a_idx = ((static_cast<size_t>(batch_idx) * N + global_i) * N + k) * D + dd;
        } else {
          a_idx = ((static_cast<size_t>(batch_idx) * N + k) * N + global_i) * D + dd;
        }
        a_vec[ii][dd] = a[a_idx];
      }
    }
    for (int idx = tid; idx < BLOCK_J * D; idx += BLOCK_I * BLOCK_J) {
      const int jj = idx / D;
      const int dd = idx % D;
      const int global_j = blockIdx.x * BLOCK_J + jj;
      if (global_j < N) {
        size_t b_idx;
        if constexpr (K_DIM_TWO) {
          b_idx = ((static_cast<size_t>(batch_idx) * N + global_j) * N + k) * D + dd;
        } else {
          b_idx = ((static_cast<size_t>(batch_idx) * N + k) * N + global_j) * D + dd;
        }
        b_vec[jj][dd] = b[b_idx];
      }
    }

    __syncthreads();

    if (i < N && j < N) {
      #pragma unroll
      for (int d = 0; d < MAX_D; ++d) {
        if (d < D) {
          acc[d] += to_float(a_vec[threadIdx.y][d]) * to_float(b_vec[threadIdx.x][d]);
        }
      }
    }

    __syncthreads();
  }

  if (i < N && j < N) {
    const size_t out_base = ((static_cast<size_t>(batch_idx) * N + i) * N + j) * D;
    #pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
      if (d < D) {
        out[out_base + d] = from_float<out_t>(acc[d]);
      }
    }
  }
}

template <typename in_t, typename out_t>
void launch_tri_mul_direct(
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& out,
    int64_t k_dim) {
  constexpr int BLOCK_I = 8;
  constexpr int BLOCK_J = 8;
  constexpr int MAX_D = 32;
  const int B = static_cast<int>(a.size(0));
  const int N = static_cast<int>(a.size(1));
  const int D = static_cast<int>(a.size(3));
  const dim3 block(BLOCK_J, BLOCK_I);
  const dim3 grid((N + BLOCK_J - 1) / BLOCK_J, (N + BLOCK_I - 1) / BLOCK_I, B);

  auto stream = at::cuda::getDefaultCUDAStream();
  if (k_dim == 2) {
    tri_mul_tiled_dvec_kernel<in_t, out_t, BLOCK_I, BLOCK_J, MAX_D, true>
        <<<grid, block, 0, stream>>>(
            a.data_ptr<in_t>(),
            b.data_ptr<in_t>(),
            out.data_ptr<out_t>(),
            B,
            N,
            D);
  } else {
    tri_mul_tiled_dvec_kernel<in_t, out_t, BLOCK_I, BLOCK_J, MAX_D, false>
        <<<grid, block, 0, stream>>>(
            a.data_ptr<in_t>(),
            b.data_ptr<in_t>(),
            out.data_ptr<out_t>(),
            B,
            N,
            D);
  }
}

}  // namespace

at::Tensor tri_mul_fused_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t k_dim,
    const std::string& out_dtype) {
  check_input(a, "a");
  check_input(b, "b");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have matching shapes");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have matching dtypes");
  TORCH_CHECK(k_dim == 1 || k_dim == 2, "k_dim must be 1 or 2");

  c10::cuda::CUDAGuard guard(a.device());
  const auto out_type = parse_out_dtype(out_dtype);

  const auto B = a.size(0);
  const auto N1 = a.size(1);
  const auto N2 = a.size(2);
  const auto D = a.size(3);
  TORCH_CHECK(N1 == N2, "Current fused tri-mul kernel expects square spatial dimensions");
  TORCH_CHECK(D == 32, "Current fused tri-mul kernel specialization requires D == 32");

  auto out = at::empty({B, N1, N2, D}, a.options().dtype(out_type));
  AT_DISPATCH_SWITCH(
      a.scalar_type(),
      "tri_mul_fused_cuda_in",
      AT_DISPATCH_CASE(
          c10::ScalarType::BFloat16,
          [&] {
            using in_t = at::BFloat16;
            AT_DISPATCH_SWITCH(
                out.scalar_type(),
                "tri_mul_fused_cuda_out_bf16",
                AT_DISPATCH_CASE(
                    c10::ScalarType::BFloat16,
                    [&] { launch_tri_mul_direct<in_t, at::BFloat16>(a, b, out, k_dim); })
                    AT_DISPATCH_CASE(
                        c10::ScalarType::Half,
                        [&] { launch_tri_mul_direct<in_t, at::Half>(a, b, out, k_dim); })
                    AT_DISPATCH_CASE(
                        c10::ScalarType::Float,
                        [&] { launch_tri_mul_direct<in_t, float>(a, b, out, k_dim); }));
          })
          AT_DISPATCH_CASE(
              c10::ScalarType::Half,
              [&] {
                using in_t = at::Half;
                AT_DISPATCH_SWITCH(
                    out.scalar_type(),
                    "tri_mul_fused_cuda_out_half",
                    AT_DISPATCH_CASE(
                        c10::ScalarType::BFloat16,
                        [&] { launch_tri_mul_direct<in_t, at::BFloat16>(a, b, out, k_dim); })
                        AT_DISPATCH_CASE(
                            c10::ScalarType::Half,
                            [&] { launch_tri_mul_direct<in_t, at::Half>(a, b, out, k_dim); })
                        AT_DISPATCH_CASE(
                            c10::ScalarType::Float,
                            [&] { launch_tri_mul_direct<in_t, float>(a, b, out, k_dim); }));
              })
          AT_DISPATCH_CASE(
              c10::ScalarType::Float,
              [&] {
                using in_t = float;
                AT_DISPATCH_SWITCH(
                    out.scalar_type(),
                    "tri_mul_fused_cuda_out_float",
                    AT_DISPATCH_CASE(
                        c10::ScalarType::BFloat16,
                        [&] { launch_tri_mul_direct<in_t, at::BFloat16>(a, b, out, k_dim); })
                        AT_DISPATCH_CASE(
                            c10::ScalarType::Half,
                            [&] { launch_tri_mul_direct<in_t, at::Half>(a, b, out, k_dim); })
                        AT_DISPATCH_CASE(
                            c10::ScalarType::Float,
                            [&] { launch_tri_mul_direct<in_t, float>(a, b, out, k_dim); }));
              }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> tri_mul_fused_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t k_dim) {
  check_input(grad, "grad");
  check_input(a, "a");
  check_input(b, "b");
  TORCH_CHECK(grad.sizes() == a.sizes() && a.sizes() == b.sizes(), "grad, a, and b must have matching shapes");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have matching dtypes");
  TORCH_CHECK(k_dim == 1 || k_dim == 2, "k_dim must be 1 or 2");

  c10::cuda::CUDAGuard guard(a.device());
  const int64_t B = a.size(0);
  const int64_t N = a.size(1);
  const int64_t D = a.size(3);
  TORCH_CHECK(a.size(2) == N, "tri_mul_fused_backward requires square spatial dimensions");
  TORCH_CHECK(D == 32, "tri_mul_fused_backward currently requires D == 32");

  auto grad_in = grad.scalar_type() == a.scalar_type() ? grad : grad.to(a.scalar_type());
  auto a_bdnn = a.permute({0, 3, 1, 2}).contiguous().reshape({B * D, N, N});
  auto b_bdnn = b.permute({0, 3, 1, 2}).contiguous().reshape({B * D, N, N});
  auto grad_bdnn = grad_in.permute({0, 3, 1, 2}).contiguous().reshape({B * D, N, N});

  at::Tensor grad_a_bdnn;
  at::Tensor grad_b_bdnn;
  if (k_dim == 2) {
    // Forward: C = A @ B^T
    grad_a_bdnn = rowmajor_square_bmm_strided(grad_bdnn, b_bdnn, false, false, a.scalar_type());
    grad_b_bdnn = rowmajor_square_bmm_strided(grad_bdnn, a_bdnn, true, false, b.scalar_type());
  } else {
    // Forward: C = A^T @ B
    grad_a_bdnn = rowmajor_square_bmm_strided(b_bdnn, grad_bdnn, false, true, a.scalar_type());
    grad_b_bdnn = rowmajor_square_bmm_strided(a_bdnn, grad_bdnn, false, false, b.scalar_type());
  }

  auto grad_a = grad_a_bdnn.reshape({B, D, N, N}).permute({0, 2, 3, 1}).contiguous();
  auto grad_b = grad_b_bdnn.reshape({B, D, N, N}).permute({0, 2, 3, 1}).contiguous();
  return {std::move(grad_a), std::move(grad_b)};
}

at::Tensor tri_mul_bdnn_cublas_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t k_dim,
    const std::string& out_dtype) {
  check_bdnn_input(a, "a");
  check_bdnn_input(b, "b");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have matching shapes");
  TORCH_CHECK(a.scalar_type() == c10::ScalarType::BFloat16, "cuBLAS path currently requires bfloat16 inputs");
  TORCH_CHECK(b.scalar_type() == c10::ScalarType::BFloat16, "cuBLAS path currently requires bfloat16 inputs");
  TORCH_CHECK(k_dim == 1 || k_dim == 2, "k_dim must be 1 or 2");

  c10::cuda::CUDAGuard guard(a.device());
  const auto out_type = parse_out_dtype(out_dtype);
  TORCH_CHECK(
      out_type == c10::ScalarType::BFloat16 || out_type == c10::ScalarType::Float,
      "cuBLAS path supports out_dtype in {bfloat16, float32}");

  const int64_t B = a.size(0);
  const int64_t D = a.size(1);
  const int64_t N = a.size(2);
  const int64_t batchCount = B * D;
  const int64_t stride = N * N;

  auto out_bdnn = at::empty({B, D, N, N}, a.options().dtype(out_type));
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  auto stream = at::cuda::getDefaultCUDAStream();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const void* a_ptr = a.data_ptr<at::BFloat16>();
  const void* b_ptr = b.data_ptr<at::BFloat16>();
  void* c_ptr = out_bdnn.data_ptr();
  const auto c_dtype = out_type == c10::ScalarType::Float ? CUDA_R_32F : CUDA_R_16BF;

  const cublasOperation_t transa = k_dim == 2 ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb = k_dim == 2 ? CUBLAS_OP_N : CUBLAS_OP_T;

  // We use the standard row-major trick:
  // - the batch matrices are contiguous in (B*D, N, N) row-major layout
  // - cuBLAS sees the same storage as column-major and we swap operand roles
  //   so the logical result matches the desired row-major triangular multiply
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle,
      transa,
      transb,
      static_cast<int>(N),
      static_cast<int>(N),
      static_cast<int>(N),
      &alpha,
      b_ptr,
      CUDA_R_16BF,
      static_cast<int>(N),
      stride,
      a_ptr,
      CUDA_R_16BF,
      static_cast<int>(N),
      stride,
      &beta,
      c_ptr,
      c_dtype,
      static_cast<int>(N),
      stride,
      static_cast<int>(batchCount),
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  return out_bdnn.permute({0, 2, 3, 1});
}

at::Tensor tri_mul_xbdnn_cublas_cuda(
    const at::Tensor& x_bdnn,
    const std::string& out_dtype) {
  check_input(x_bdnn, "x_bdnn");
  TORCH_CHECK(x_bdnn.scalar_type() == c10::ScalarType::BFloat16, "x_bdnn must be bfloat16");
  TORCH_CHECK(x_bdnn.size(1) == 128, "x_bdnn must have shape (B, 128, N, N)");
  TORCH_CHECK(x_bdnn.size(2) == x_bdnn.size(3), "x_bdnn must have square spatial dimensions");

  c10::cuda::CUDAGuard guard(x_bdnn.device());
  const auto out_type = parse_out_dtype(out_dtype);
  TORCH_CHECK(
      out_type == c10::ScalarType::BFloat16 || out_type == c10::ScalarType::Float,
      "cuBLAS path supports out_dtype in {bfloat16, float32}");

  const int64_t B = x_bdnn.size(0);
  const int64_t N = x_bdnn.size(2);
  constexpr int64_t D = 32;
  constexpr int64_t DT = 128;
  const int64_t matrix_elems = N * N;

  auto out1_bdnn = at::empty({B, D, N, N}, x_bdnn.options().dtype(out_type));
  auto out2_bdnn = at::empty({B, D, N, N}, x_bdnn.options().dtype(out_type));

  auto handle = at::cuda::getCurrentCUDABlasHandle();
  auto stream = at::cuda::getDefaultCUDAStream();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));

  const auto* x_ptr = x_bdnn.data_ptr<at::BFloat16>();
  auto* out1_ptr = out1_bdnn.data_ptr();
  auto* out2_ptr = out2_bdnn.data_ptr();

  std::vector<int64_t> a1_ptrs_host;
  std::vector<int64_t> b1_ptrs_host;
  std::vector<int64_t> a2_ptrs_host;
  std::vector<int64_t> b2_ptrs_host;
  std::vector<int64_t> c1_ptrs_host;
  std::vector<int64_t> c2_ptrs_host;
  a1_ptrs_host.reserve(B * D);
  b1_ptrs_host.reserve(B * D);
  a2_ptrs_host.reserve(B * D);
  b2_ptrs_host.reserve(B * D);
  c1_ptrs_host.reserve(B * D);
  c2_ptrs_host.reserve(B * D);

  for (int64_t batch_idx = 0; batch_idx < B; ++batch_idx) {
    const int64_t batch_base = batch_idx * DT * matrix_elems;
    const int64_t out_base = batch_idx * D * matrix_elems;
    for (int64_t d = 0; d < D; ++d) {
      a1_ptrs_host.push_back(reinterpret_cast<int64_t>(x_ptr + batch_base + d * matrix_elems));
      b1_ptrs_host.push_back(reinterpret_cast<int64_t>(x_ptr + batch_base + (D + d) * matrix_elems));
      a2_ptrs_host.push_back(reinterpret_cast<int64_t>(x_ptr + batch_base + (2 * D + d) * matrix_elems));
      b2_ptrs_host.push_back(reinterpret_cast<int64_t>(x_ptr + batch_base + (3 * D + d) * matrix_elems));
      c1_ptrs_host.push_back(reinterpret_cast<int64_t>(
          static_cast<char*>(out1_ptr) + (out_base + d * matrix_elems) * out1_bdnn.element_size()));
      c2_ptrs_host.push_back(reinterpret_cast<int64_t>(
          static_cast<char*>(out2_ptr) + (out_base + d * matrix_elems) * out2_bdnn.element_size()));
    }
  }

  const auto ptr_options = x_bdnn.options().dtype(c10::ScalarType::Long);
  auto a1_ptrs = at::empty({B * D}, ptr_options);
  auto b1_ptrs = at::empty({B * D}, ptr_options);
  auto a2_ptrs = at::empty({B * D}, ptr_options);
  auto b2_ptrs = at::empty({B * D}, ptr_options);
  auto c1_ptrs = at::empty({B * D}, ptr_options);
  auto c2_ptrs = at::empty({B * D}, ptr_options);

  auto copy_stream = at::cuda::getDefaultCUDAStream();
  AT_CUDA_CHECK(cudaMemcpyAsync(
      a1_ptrs.data_ptr<int64_t>(),
      a1_ptrs_host.data(),
      a1_ptrs_host.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      copy_stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      b1_ptrs.data_ptr<int64_t>(),
      b1_ptrs_host.data(),
      b1_ptrs_host.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      copy_stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      a2_ptrs.data_ptr<int64_t>(),
      a2_ptrs_host.data(),
      a2_ptrs_host.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      copy_stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      b2_ptrs.data_ptr<int64_t>(),
      b2_ptrs_host.data(),
      b2_ptrs_host.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      copy_stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      c1_ptrs.data_ptr<int64_t>(),
      c1_ptrs_host.data(),
      c1_ptrs_host.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      copy_stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      c2_ptrs.data_ptr<int64_t>(),
      c2_ptrs_host.data(),
      c2_ptrs_host.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      copy_stream));

  launch_batched_cublas(handle, a1_ptrs, b1_ptrs, c1_ptrs, N, 2, out_type);
  launch_batched_cublas(handle, a2_ptrs, b2_ptrs, c2_ptrs, N, 1, out_type);

  return at::cat({out1_bdnn, out2_bdnn}, 1).permute({0, 2, 3, 1});
}

}  // namespace tri_mul_ext
