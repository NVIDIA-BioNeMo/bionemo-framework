#include "common.h"

#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/swizzle.h>
#include <transformer_engine/transformer_engine.h>

#include <cmath>
#include <memory>

namespace bmm_ext {

namespace {

constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;
constexpr float kNvfp4NeutralAmax = 6.0f * 448.0f;

#define CUBLASLT_CHECK(EXPR) TORCH_CHECK((EXPR) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error at ", #EXPR)

void check_cuda_tensor(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

std::vector<int64_t> vec_from_sizes(c10::IntArrayRef sizes) {
  return std::vector<int64_t>(sizes.begin(), sizes.end());
}

std::vector<int64_t> require_shape_override(
    const at::Tensor& storage,
    const std::vector<int64_t>& override_shape,
    const char* name) {
  if (override_shape.empty()) {
    TORCH_CHECK(storage.dim() == 3, name, " must be 3D");
    return vec_from_sizes(storage.sizes());
  }
  TORCH_CHECK(override_shape.size() == 3, name, " logical shape override must have length 3");
  return override_shape;
}

void check_same_device(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale,
    const at::Tensor& b_scale) {
  TORCH_CHECK(
      a.device() == b.device() && a.device() == a_scale.device() && a.device() == b_scale.device(),
      "all tensors must be on the same CUDA device");
}

std::vector<int64_t> scale_shape_from_a(int64_t batch, int64_t m, int64_t k, int64_t block) {
  return {batch, m, k / block};
}

std::vector<int64_t> scale_shape_from_b(int64_t batch, int64_t k, int64_t n, int64_t block) {
  return {batch, k / block, n};
}

void check_shape_eq(
    const at::Tensor& t,
    const std::vector<int64_t>& expected,
    const char* name) {
  TORCH_CHECK(
      vec_from_sizes(t.sizes()) == expected,
      name,
      " must have shape ",
      expected,
      ", got ",
      t.sizes());
}

void validate_sm100_plus() {
  auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props != nullptr, "could not query CUDA device properties");
  const int sm = props->major * 10 + props->minor;
  TORCH_CHECK(sm >= 100, "block-scaled bmm requires SM100+, found SM", sm);
}

int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

transformer_engine::DType to_te_dtype(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float8_e4m3fn:
      return transformer_engine::DType::kFloat8E4M3;
    case c10::ScalarType::Float8_e5m2:
      return transformer_engine::DType::kFloat8E5M2;
    case c10::ScalarType::Half:
      return transformer_engine::DType::kFloat16;
    case c10::ScalarType::BFloat16:
      return transformer_engine::DType::kBFloat16;
    case c10::ScalarType::Float:
      return transformer_engine::DType::kFloat32;
    case c10::ScalarType::Byte:
      return transformer_engine::DType::kByte;
    default:
      TORCH_CHECK(false, "unsupported dtype for Transformer Engine path: ", dtype);
  }
}

at::Tensor make_swizzled_scale_rowwise(
    const at::Tensor& scale,
    int64_t rows,
    int64_t cols) {
  TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous before swizzling");
  const int64_t padded_rows = ceil_div(rows, 128) * 128;
  const int64_t padded_cols = ceil_div(cols, 4) * 4;
  auto padded = at::full(
      {padded_rows, padded_cols},
      std::pow(2.0f, -127.0f),
      scale.options());
  padded.slice(0, 0, rows).slice(1, 0, cols).copy_(scale.reshape({rows, cols}));
  auto swizzled = padded.view({padded_rows / 128, 4, 32, padded_cols / 4, 4})
                      .permute({0, 3, 2, 1, 4})
                      .contiguous()
                      .view({padded_rows, padded_cols});
  return swizzled;
}

at::Tensor pad_nvfp4_scale_rowwise(
    const at::Tensor& scale,
    int64_t rows,
    int64_t cols_blocks) {
  TORCH_CHECK(scale.is_contiguous(), "nvfp4 scale must be contiguous before padding");
  const int64_t padded_rows = ceil_div(rows, 128) * 128;
  const int64_t padded_cols = ceil_div(cols_blocks, 4) * 4;
  auto padded = at::full(
      {padded_rows, padded_cols},
      1.0,
      scale.options());
  padded.slice(0, 0, rows).slice(1, 0, cols_blocks).copy_(scale.reshape({rows, cols_blocks}));
  return padded;
}

at::Tensor swizzle_nvfp4_scale_rowwise(
    const at::Tensor& scale,
    int64_t rows,
    int64_t cols) {
  auto output = at::empty_like(scale);
  transformer_engine::TensorWrapper input_nvte(NVTE_NVFP4_1D_SCALING);
  input_nvte.set_rowwise_data(nullptr, transformer_engine::DType::kFloat4E2M1,
                              std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
  input_nvte.set_rowwise_scale_inv(
      scale.data_ptr(),
      transformer_engine::DType::kFloat8E4M3,
      std::vector<size_t>{static_cast<size_t>(scale.size(0)), static_cast<size_t>(scale.size(1))});
  transformer_engine::TensorWrapper output_nvte(NVTE_NVFP4_1D_SCALING);
  output_nvte.set_rowwise_data(nullptr, transformer_engine::DType::kFloat4E2M1,
                               std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
  output_nvte.set_rowwise_scale_inv(
      output.data_ptr(),
      transformer_engine::DType::kFloat8E4M3,
      std::vector<size_t>{static_cast<size_t>(output.size(0)), static_cast<size_t>(output.size(1))});
  output_nvte.set_with_gemm_swizzled_scales(true);
  nvte_swizzle_scaling_factors(input_nvte.data(), output_nvte.data(), at::cuda::getCurrentCUDAStream());
  return output;
}

at::Tensor transpose_pack_fp4_codes(const at::Tensor& packed, int64_t rows, int64_t cols) {
  auto packed_i64 = packed.to(at::kLong);
  auto lo_codes = at::bitwise_and(packed_i64, 0x0F);
  auto hi_codes = at::bitwise_and(at::bitwise_right_shift(packed_i64, 4), 0x0F);
  auto codes = at::stack({lo_codes, hi_codes}, -1).reshape({rows, cols});
  auto codes_t = codes.transpose(0, 1).contiguous();
  auto lo_t = codes_t.slice(1, 0, cols, 2);
  auto hi_t = codes_t.slice(1, 1, cols, 2);
  return at::bitwise_or(lo_t, at::bitwise_left_shift(hi_t, 4)).to(at::kByte);
}

at::Tensor run_mxfp8_nvte_bmm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale,
    const at::Tensor& b_scale,
    c10::ScalarType out_dtype,
    int64_t sf_vec_size,
    int64_t batch,
    int64_t m,
    int64_t k,
    int64_t n) {
  auto b_t = b.transpose(1, 2).contiguous();
  auto b_scale_t = b_scale.transpose(1, 2).contiguous();
  auto out_t = at::empty({batch, n, m}, a.options().dtype(out_dtype));
  auto workspaces = at::empty({batch, static_cast<int64_t>(kWorkspaceBytes)}, a.options().dtype(at::kByte));

  std::vector<at::Tensor> swizzled_a_scales;
  std::vector<at::Tensor> swizzled_b_scales;
  std::vector<transformer_engine::TensorWrapper> a_wrappers;
  std::vector<transformer_engine::TensorWrapper> b_wrappers;
  std::vector<transformer_engine::TensorWrapper> d_wrappers;
  std::vector<transformer_engine::TensorWrapper> bias_wrappers;
  std::vector<transformer_engine::TensorWrapper> pre_gelu_wrappers;
  std::vector<transformer_engine::TensorWrapper> workspace_wrappers;
  std::vector<NVTETensor> a_tensors;
  std::vector<NVTETensor> b_tensors;
  std::vector<NVTETensor> d_tensors;
  std::vector<NVTETensor> bias_tensors;
  std::vector<NVTETensor> pre_gelu_tensors;
  std::vector<NVTETensor> workspace_tensors;

  swizzled_a_scales.reserve(batch);
  swizzled_b_scales.reserve(batch);
  a_wrappers.reserve(batch);
  b_wrappers.reserve(batch);
  d_wrappers.reserve(batch);
  bias_wrappers.reserve(batch);
  pre_gelu_wrappers.reserve(batch);
  workspace_wrappers.reserve(batch);
  a_tensors.reserve(batch);
  b_tensors.reserve(batch);
  d_tensors.reserve(batch);
  bias_tensors.reserve(batch);
  pre_gelu_tensors.reserve(batch);
  workspace_tensors.reserve(batch);

  for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    auto a_i = a.select(0, batch_idx);
    auto b_i_t = b_t.select(0, batch_idx);
    swizzled_a_scales.emplace_back(
        make_swizzled_scale_rowwise(a_scale.select(0, batch_idx).contiguous(), m, k / sf_vec_size));
    swizzled_b_scales.emplace_back(
        make_swizzled_scale_rowwise(b_scale_t.select(0, batch_idx).contiguous(), n, k / sf_vec_size));

    a_wrappers.emplace_back(NVTE_MXFP8_1D_SCALING);
    auto& a_te = a_wrappers.back();
    a_te.set_rowwise_data(
        a_i.data_ptr(),
        to_te_dtype(a.scalar_type()),
        std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(k)});
    a_te.set_rowwise_scale_inv(
        swizzled_a_scales.back().data_ptr(),
        transformer_engine::DType::kFloat8E8M0,
        std::vector<size_t>{
            static_cast<size_t>(swizzled_a_scales.back().size(0)),
            static_cast<size_t>(swizzled_a_scales.back().size(1))});
    a_te.set_with_gemm_swizzled_scales(true);

    b_wrappers.emplace_back(NVTE_MXFP8_1D_SCALING);
    auto& b_te = b_wrappers.back();
    b_te.set_rowwise_data(
        b_i_t.data_ptr(),
        to_te_dtype(b.scalar_type()),
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
    b_te.set_rowwise_scale_inv(
        swizzled_b_scales.back().data_ptr(),
        transformer_engine::DType::kFloat8E8M0,
        std::vector<size_t>{
            static_cast<size_t>(swizzled_b_scales.back().size(0)),
            static_cast<size_t>(swizzled_b_scales.back().size(1))});
    b_te.set_with_gemm_swizzled_scales(true);

    d_wrappers.emplace_back(
        out_t.select(0, batch_idx).data_ptr(),
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(m)},
        to_te_dtype(out_dtype));
    bias_wrappers.emplace_back();
    pre_gelu_wrappers.emplace_back();
    workspace_wrappers.emplace_back(
        workspaces.select(0, batch_idx).data_ptr(),
        std::vector<size_t>{kWorkspaceBytes},
        transformer_engine::DType::kByte);
  }

  for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    a_tensors.emplace_back(a_wrappers[batch_idx].data());
    b_tensors.emplace_back(b_wrappers[batch_idx].data());
    d_tensors.emplace_back(d_wrappers[batch_idx].data());
    bias_tensors.emplace_back(bias_wrappers[batch_idx].data());
    pre_gelu_tensors.emplace_back(pre_gelu_wrappers[batch_idx].data());
    workspace_tensors.emplace_back(workspace_wrappers[batch_idx].data());
  }

  nvte_multi_tensor_gemm(
      a_tensors.data(),
      b_tensors.data(),
      d_tensors.data(),
      bias_tensors.data(),
      pre_gelu_tensors.data(),
      static_cast<int>(batch),
      true,
      false,
      false,
      workspace_tensors.data(),
      false,
      false,
      0,
      at::cuda::getCurrentCUDAStream());

  return out_t.transpose(1, 2).contiguous();
}

cudaDataType_t to_cuda_dtype(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float8_e4m3fn:
      return CUDA_R_8F_E4M3;
    case c10::ScalarType::Float8_e5m2:
      return CUDA_R_8F_E5M2;
    case c10::ScalarType::Half:
      return CUDA_R_16F;
    case c10::ScalarType::BFloat16:
      return CUDA_R_16BF;
    case c10::ScalarType::Float:
      return CUDA_R_32F;
    case c10::ScalarType::Float8_e8m0fnu:
      return CUDA_R_8F_UE8M0;
    default:
      TORCH_CHECK(false, "unsupported CUDA dtype: ", dtype);
  }
}

struct CublasLtBmmPlan {
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int device_index = -1;
  int64_t batch = -1;
  int64_t m = -1;
  int64_t k = -1;
  int64_t n = -1;
  c10::ScalarType a_dtype = c10::ScalarType::Undefined;
  c10::ScalarType b_dtype = c10::ScalarType::Undefined;
  c10::ScalarType out_dtype = c10::ScalarType::Undefined;
  bool rhs_direct = false;

  ~CublasLtBmmPlan() {
    if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
    if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
    if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
    if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
    if (d_desc != nullptr) cublasLtMatrixLayoutDestroy(d_desc);
    if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
  }

  bool matches(
      int device,
      int64_t batch_,
      int64_t m_,
      int64_t k_,
      int64_t n_,
      c10::ScalarType a_dtype_,
      c10::ScalarType b_dtype_,
      c10::ScalarType out_dtype_,
      bool rhs_direct_) const {
    return device_index == device && batch == batch_ && m == m_ && k == k_ && n == n_ &&
        a_dtype == a_dtype_ && b_dtype == b_dtype_ && out_dtype == out_dtype_ && rhs_direct == rhs_direct_;
  }
};

void init_cublaslt_bmm_plan(
    CublasLtBmmPlan* plan,
    int device_index,
    int64_t batch,
    int64_t m,
    int64_t k,
    int64_t n,
    c10::ScalarType a_dtype,
    c10::ScalarType b_dtype,
    c10::ScalarType out_dtype,
    const void* a_scale_ptr,
    const void* b_scale_ptr,
    bool rhs_direct,
    cublasLtHandle_t lt_handle) {
  if (plan->preference != nullptr) cublasLtMatmulPreferenceDestroy(plan->preference);
  if (plan->a_desc != nullptr) cublasLtMatrixLayoutDestroy(plan->a_desc);
  if (plan->b_desc != nullptr) cublasLtMatrixLayoutDestroy(plan->b_desc);
  if (plan->c_desc != nullptr) cublasLtMatrixLayoutDestroy(plan->c_desc);
  if (plan->d_desc != nullptr) cublasLtMatrixLayoutDestroy(plan->d_desc);
  if (plan->op_desc != nullptr) cublasLtMatmulDescDestroy(plan->op_desc);
  plan->preference = nullptr;
  plan->a_desc = nullptr;
  plan->b_desc = nullptr;
  plan->c_desc = nullptr;
  plan->d_desc = nullptr;
  plan->op_desc = nullptr;

  const cublasOperation_t transa = CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;
  const int32_t order_row = CUBLASLT_ORDER_ROW;
  const int32_t b_order = rhs_direct ? CUBLASLT_ORDER_ROW : CUBLASLT_ORDER_COL;
  const int32_t batch_count_i32 = static_cast<int32_t>(batch);
  const int64_t a_batch_stride = m * k;
  const int64_t b_batch_stride = k * n;
  const uint64_t b_ld = rhs_direct ? n : k;
  const int64_t c_batch_stride = m * n;
  const int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  const size_t workspace_size = kWorkspaceBytes;

  CUBLASLT_CHECK(cublasLtMatmulDescCreate(&plan->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CUBLASLT_CHECK(
      cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLASLT_CHECK(
      cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));

  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->a_desc, to_cuda_dtype(a_dtype), m, k, k));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->b_desc, to_cuda_dtype(b_dtype), k, n, b_ld));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->c_desc, to_cuda_dtype(out_dtype), m, n, n));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->d_desc, to_cuda_dtype(out_dtype), m, n, n));

  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));

  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count_i32, sizeof(batch_count_i32)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count_i32, sizeof(batch_count_i32)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count_i32, sizeof(batch_count_i32)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->d_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count_i32, sizeof(batch_count_i32)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_batch_stride, sizeof(a_batch_stride)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_batch_stride, sizeof(b_batch_stride)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_batch_stride, sizeof(c_batch_stride)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan->d_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_batch_stride, sizeof(c_batch_stride)));

  CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&plan->preference));
  CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
  int returned_results = 0;
  CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt_handle,
      plan->op_desc,
      plan->a_desc,
      plan->b_desc,
      plan->c_desc,
      plan->d_desc,
      plan->preference,
      1,
      &plan->heuristic,
      &returned_results));
  TORCH_CHECK(returned_results > 0, "cuBLASLt found no heuristic for MXFP8 batched GEMM");

  plan->device_index = device_index;
  plan->batch = batch;
  plan->m = m;
  plan->k = k;
  plan->n = n;
  plan->a_dtype = a_dtype;
  plan->b_dtype = b_dtype;
  plan->out_dtype = out_dtype;
  plan->rhs_direct = rhs_direct;
}

at::Tensor run_mxfp8_cublaslt_bmm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    c10::ScalarType out_dtype,
    int64_t batch,
    int64_t m,
    int64_t k,
    int64_t n,
    bool rhs_direct) {
  static cublasLtHandle_t lt_handle = nullptr;
  if (lt_handle == nullptr) {
    CUBLASLT_CHECK(cublasLtCreate(&lt_handle));
  }

  auto out = at::empty({batch, m, n}, a.options().dtype(out_dtype));
  auto workspace = at::empty({static_cast<int64_t>(kWorkspaceBytes)}, a.options().dtype(at::kByte));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  thread_local std::unique_ptr<CublasLtBmmPlan> cached_plan;
  if (cached_plan == nullptr) {
    cached_plan = std::make_unique<CublasLtBmmPlan>();
  }
  const int device_index = static_cast<int>(a.get_device());
  const void* a_scale_ptr = a_scale_swizzled.data_ptr();
  const void* b_scale_ptr = b_scale_swizzled.data_ptr();
  if (!cached_plan->matches(device_index, batch, m, k, n, a.scalar_type(), b.scalar_type(), out_dtype, rhs_direct)) {
    init_cublaslt_bmm_plan(
        cached_plan.get(),
        device_index,
        batch,
        m,
        k,
        n,
        a.scalar_type(),
        b.scalar_type(),
        out_dtype,
        a_scale_ptr,
        b_scale_ptr,
        rhs_direct,
        lt_handle);
  }
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      cached_plan->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      cached_plan->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));

  CUBLASLT_CHECK(cublasLtMatmul(
      lt_handle,
      cached_plan->op_desc,
      &alpha,
      a.data_ptr(),
      cached_plan->a_desc,
      b.data_ptr(),
      cached_plan->b_desc,
      &beta,
      out.data_ptr(),
      cached_plan->c_desc,
      out.data_ptr(),
      cached_plan->d_desc,
      &cached_plan->heuristic.algo,
      workspace.data_ptr(),
      kWorkspaceBytes,
      at::cuda::getCurrentCUDAStream()));
  return out;
}

at::Tensor run_nvfp4_nvte_bmm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale,
    const at::Tensor& b_scale,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    c10::ScalarType out_dtype,
    int64_t sf_vec_size,
    int64_t batch,
    int64_t m,
    int64_t k,
    int64_t n,
    bool b_rhs_transposed) {
  auto out = at::empty({batch, m, n}, a.options().dtype(out_dtype));
  auto workspace = at::empty({static_cast<int64_t>(kWorkspaceBytes)}, a.options().dtype(at::kByte));
  auto workspace_te = transformer_engine::TensorWrapper(
      workspace.data_ptr(),
      std::vector<size_t>{kWorkspaceBytes},
      transformer_engine::DType::kByte);
  transformer_engine::MatmulConfigWrapper config;
  config.set_use_split_accumulator(false);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    auto a_i = a.select(0, batch_idx);
    auto b_i = b.select(0, batch_idx);
    auto b_i_t = b_rhs_transposed ? b_i : transpose_pack_fp4_codes(b_i, k, n);
    at::Tensor a_scale_i;
    if (a_scale.scalar_type() == c10::ScalarType::Byte) {
      a_scale_i = swizzle_nvfp4_scale_rowwise(a_scale.select(0, batch_idx).contiguous(), m, k);
    } else {
      a_scale_i = swizzle_nvfp4_scale_rowwise(
          pad_nvfp4_scale_rowwise(a_scale.select(0, batch_idx).contiguous(), m, k / sf_vec_size),
          m,
          k);
    }
    at::Tensor b_scale_i;
    if (b_scale.scalar_type() == c10::ScalarType::Byte) {
      b_scale_i = swizzle_nvfp4_scale_rowwise(b_scale.select(0, batch_idx).contiguous(), n, k);
    } else {
      b_scale_i = swizzle_nvfp4_scale_rowwise(
          pad_nvfp4_scale_rowwise(
              (b_rhs_transposed
                  ? b_scale.select(0, batch_idx).contiguous()
                  : b_scale.select(0, batch_idx).transpose(0, 1).contiguous()),
              n,
              k / sf_vec_size),
          n,
          k);
    }
    auto amax_a = a_amax.numel() == 0
        ? at::full({1}, kNvfp4NeutralAmax, a.options().dtype(at::kFloat))
        : a_amax.select(0, batch_idx).reshape({1}).contiguous();
    auto amax_b = b_amax.numel() == 0
        ? at::full({1}, kNvfp4NeutralAmax, b.options().dtype(at::kFloat))
        : b_amax.select(0, batch_idx).reshape({1}).contiguous();
    auto out_i_t = at::empty({n, m}, out.options());

    auto a_te = transformer_engine::TensorWrapper(NVTE_NVFP4_1D_SCALING);
    a_te.set_rowwise_data(a_i.data_ptr(), transformer_engine::DType::kFloat4E2M1,
                          std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(k)});
    a_te.set_rowwise_scale_inv(
        a_scale_i.data_ptr(),
        transformer_engine::DType::kFloat8E4M3,
        std::vector<size_t>{static_cast<size_t>(a_scale_i.size(0)), static_cast<size_t>(a_scale_i.size(1))});
    a_te.set_amax(amax_a.data_ptr(), transformer_engine::DType::kFloat32, std::vector<size_t>{1});
    a_te.set_with_gemm_swizzled_scales(true);

    auto b_te = transformer_engine::TensorWrapper(NVTE_NVFP4_1D_SCALING);
    b_te.set_rowwise_data(b_i_t.data_ptr(), transformer_engine::DType::kFloat4E2M1,
                          std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
    b_te.set_rowwise_scale_inv(
        b_scale_i.data_ptr(),
        transformer_engine::DType::kFloat8E4M3,
        std::vector<size_t>{static_cast<size_t>(b_scale_i.size(0)), static_cast<size_t>(b_scale_i.size(1))});
    b_te.set_amax(amax_b.data_ptr(), transformer_engine::DType::kFloat32, std::vector<size_t>{1});
    b_te.set_with_gemm_swizzled_scales(true);

    auto d_te = transformer_engine::TensorWrapper(
        out_i_t.data_ptr(),
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(m)},
        to_te_dtype(out_dtype));
    nvte_cublas_gemm_v2(
        1,
        0,
        &alpha,
        a_te.data(),
        b_te.data(),
        &beta,
        d_te.data(),
        d_te.data(),
        workspace_te.data(),
        config,
        at::cuda::getCurrentCUDAStream());
    out.select(0, batch_idx).copy_(out_i_t.transpose(0, 1));
  }

  return out;
}

at::Tensor unpack_fp4_tensor(const at::Tensor& packed, const std::vector<int64_t>& logical_shape) {
  static const float kCodebookHost[16] = {
      -0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  auto codebook = at::tensor(kCodebookHost, at::TensorOptions().dtype(at::kFloat).device(packed.device()));
  auto packed_i64 = packed.to(at::kLong);
  auto lo_codes = at::bitwise_and(packed_i64, 0x0F);
  auto hi_codes = at::bitwise_and(at::bitwise_right_shift(packed_i64, 4), 0x0F);

  auto lo_values = codebook.index_select(0, lo_codes.reshape({-1})).reshape(lo_codes.sizes());
  auto hi_values = codebook.index_select(0, hi_codes.reshape({-1})).reshape(hi_codes.sizes());
  auto unpacked = at::stack({lo_values, hi_values}, -1).reshape(logical_shape);
  return unpacked;
}

at::Tensor expand_scale_a(const at::Tensor& scale, int64_t sf_vec_size, int64_t k) {
  auto expanded = scale.to(at::kFloat).repeat_interleave(sf_vec_size, -1);
  return expanded.slice(-1, 0, k);
}

at::Tensor expand_scale_b(const at::Tensor& scale, int64_t sf_vec_size, int64_t k) {
  auto expanded = scale.to(at::kFloat).repeat_interleave(sf_vec_size, 1);
  return expanded.slice(1, 0, k);
}

at::Tensor dequantize_nvfp4(
    const at::Tensor& packed,
    const at::Tensor& scale,
    const std::vector<int64_t>& logical_shape,
    bool is_a,
    int64_t sf_vec_size,
    int64_t k) {
  auto unpacked = unpack_fp4_tensor(packed, logical_shape);
  auto expanded_scale = is_a ? expand_scale_a(scale, sf_vec_size, k)
                             : expand_scale_b(scale, sf_vec_size, k);
  return unpacked * expanded_scale;
}

}  // namespace

std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t>& dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

at::Tensor bmm_block_scaled_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale,
    const at::Tensor& b_scale,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const std::string& format_str,
    const std::string& out_dtype_str,
    int64_t sf_vec_size,
    std::vector<int64_t> a_shape_override,
    std::vector<int64_t> b_shape_override,
    bool a_rhs_transposed,
    bool b_rhs_transposed) {
  const auto format = parse_format(format_str);
  const auto out_dtype = parse_out_dtype(out_dtype_str);

  check_cuda_tensor(a, "a");
  check_cuda_tensor(b, "b");
  check_cuda_tensor(a_scale, "a_scale");
  check_cuda_tensor(b_scale, "b_scale");
  check_same_device(a, b, a_scale, b_scale);
  if (a_amax.numel() != 0 || b_amax.numel() != 0) {
    check_cuda_tensor(a_amax, "a_amax");
    check_cuda_tensor(b_amax, "b_amax");
    TORCH_CHECK(a.device() == a_amax.device() && a.device() == b_amax.device(),
                "nvfp4 amax tensors must be on the same CUDA device");
  }
  TORCH_CHECK(
      !a.requires_grad() && !b.requires_grad() && !a_scale.requires_grad() && !b_scale.requires_grad() &&
          !a_amax.requires_grad() && !b_amax.requires_grad(),
      "bmm_block_scaled is forward-only and rejects tensors requiring grad");

  c10::cuda::CUDAGuard guard(a.device());
  validate_sm100_plus();

  const auto a_shape = require_shape_override(a, a_shape_override, "a");
  const auto b_shape = require_shape_override(b, b_shape_override, "b");
  TORCH_CHECK(a_shape.size() == 3 && b_shape.size() == 3, "logical inputs must be 3D");
  const int64_t batch = a_shape[0];
  const int64_t m = a_shape[1];
  const int64_t k = a_shape[2];
  TORCH_CHECK(b_shape[0] == batch, "batch dimensions must match exactly");
  TORCH_CHECK(b_shape[1] == k, "inner dimensions must match exactly");
  const int64_t n = b_shape[2];

  TORCH_CHECK(sf_vec_size > 0, "sf_vec_size must be positive");
  TORCH_CHECK(k % sf_vec_size == 0, "K must be divisible by sf_vec_size");

  if (format == BlockScaledFormat::kMXFP8) {
    TORCH_CHECK(
        sf_vec_size == 32,
        "mxfp8 currently requires sf_vec_size=32 for the cuBLASLt backend");
    TORCH_CHECK(a.scalar_type() == c10::ScalarType::Float8_e4m3fn || a.scalar_type() == c10::ScalarType::Float8_e5m2,
        "mxfp8 a must use torch.float8_e4m3fn or torch.float8_e5m2");
    TORCH_CHECK(b.scalar_type() == c10::ScalarType::Float8_e4m3fn || b.scalar_type() == c10::ScalarType::Float8_e5m2,
        "mxfp8 b must use torch.float8_e4m3fn or torch.float8_e5m2");
    TORCH_CHECK(a_scale.scalar_type() == c10::ScalarType::Float8_e8m0fnu, "mxfp8 a_scale must use torch.float8_e8m0fnu");
    TORCH_CHECK(b_scale.scalar_type() == c10::ScalarType::Float8_e8m0fnu, "mxfp8 b_scale must use torch.float8_e8m0fnu");
    TORCH_CHECK(vec_from_sizes(a.sizes()) == a_shape, "mxfp8 a storage shape must match logical shape");
    TORCH_CHECK(vec_from_sizes(b.sizes()) == b_shape, "mxfp8 b storage shape must match logical shape");
  } else {
    TORCH_CHECK(!a_rhs_transposed, "nvfp4 lhs does not support rhs_transposed metadata");
    TORCH_CHECK(sf_vec_size == 16, "nvfp4 currently requires sf_vec_size=16");
    TORCH_CHECK(a.scalar_type() == c10::ScalarType::Byte, "nvfp4 a packed storage must use torch.uint8");
    TORCH_CHECK(b.scalar_type() == c10::ScalarType::Byte, "nvfp4 b packed storage must use torch.uint8");
    TORCH_CHECK(
        a_scale.scalar_type() == c10::ScalarType::Float8_e4m3fn || a_scale.scalar_type() == c10::ScalarType::Byte,
        "nvfp4 a_scale must use torch.float8_e4m3fn or torch.uint8");
    TORCH_CHECK(
        b_scale.scalar_type() == c10::ScalarType::Float8_e4m3fn || b_scale.scalar_type() == c10::ScalarType::Byte,
        "nvfp4 b_scale must use torch.float8_e4m3fn or torch.uint8");
    TORCH_CHECK(a_shape[2] % 2 == 0, "nvfp4 logical K for a must be even");
    TORCH_CHECK(b_shape[2] % 2 == 0, "nvfp4 logical N for b must be even");
    TORCH_CHECK(vec_from_sizes(a.sizes()) == std::vector<int64_t>({batch, m, k / 2}), "nvfp4 a storage shape must be (B, M, K/2)");
    const auto expected_b_shape =
        b_rhs_transposed ? std::vector<int64_t>({batch, n, k / 2}) : std::vector<int64_t>({batch, k, n / 2});
    TORCH_CHECK(vec_from_sizes(b.sizes()) == expected_b_shape, "nvfp4 b storage shape mismatch for selected layout");
  }

  if (format == BlockScaledFormat::kMXFP8 || a_scale.scalar_type() != c10::ScalarType::Byte) {
    check_shape_eq(a_scale, scale_shape_from_a(batch, m, k, sf_vec_size), "a_scale");
  } else {
    check_shape_eq(
        a_scale,
        {batch, ceil_div(m, 128) * 128, ceil_div(k / sf_vec_size, 4) * 4},
        "a_scale");
  }
  if (format == BlockScaledFormat::kMXFP8 || b_scale.scalar_type() != c10::ScalarType::Byte) {
    check_shape_eq(b_scale, scale_shape_from_b(batch, k, n, sf_vec_size), "b_scale");
  } else {
    check_shape_eq(
        b_scale,
        {batch, ceil_div(n, 128) * 128, ceil_div(k / sf_vec_size, 4) * 4},
        "b_scale");
  }

  if (format == BlockScaledFormat::kMXFP8) {
    return run_mxfp8_nvte_bmm(a, b, a_scale, b_scale, out_dtype, sf_vec_size, batch, m, k, n);
  }

  return run_nvfp4_nvte_bmm(a, b, a_scale, b_scale, a_amax, b_amax, out_dtype, sf_vec_size, batch, m, k, n, b_rhs_transposed);
}

at::Tensor mxfp8_cublaslt_bmm_cuda(
    const at::Tensor& a,
    const at::Tensor& b_t,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const std::string& out_dtype_str) {
  const auto out_dtype = parse_out_dtype(out_dtype_str);

  check_cuda_tensor(a, "a");
  check_cuda_tensor(b_t, "b_t");
  check_cuda_tensor(a_scale_swizzled, "a_scale_swizzled");
  check_cuda_tensor(b_scale_swizzled, "b_scale_swizzled");
  TORCH_CHECK(a.dim() == 3 && b_t.dim() == 3, "mxfp8_cublaslt_bmm expects 3D tensors");
  TORCH_CHECK(a.scalar_type() == c10::ScalarType::Float8_e4m3fn || a.scalar_type() == c10::ScalarType::Float8_e5m2,
              "a must be FP8");
  TORCH_CHECK(b_t.scalar_type() == a.scalar_type(), "a and b_t must have matching FP8 dtype");
  TORCH_CHECK(a_scale_swizzled.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
              "a_scale_swizzled must use torch.float8_e8m0fnu");
  TORCH_CHECK(b_scale_swizzled.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
              "b_scale_swizzled must use torch.float8_e8m0fnu");
  TORCH_CHECK(a.size(0) == b_t.size(0), "batch dimensions must match");
  TORCH_CHECK(a.size(2) == b_t.size(2), "K dimensions must match");
  TORCH_CHECK(a.size(1) % 32 == 0 && a.size(2) % 32 == 0 && b_t.size(1) % 32 == 0,
              "MXFP8 cuBLASLt path requires M, N, and K divisible by 32");

  c10::cuda::CUDAGuard guard(a.device());
  validate_sm100_plus();
  check_same_device(a, b_t, a_scale_swizzled, b_scale_swizzled);

  const int64_t batch = a.size(0);
  const int64_t m = a.size(1);
  const int64_t k = a.size(2);
  const int64_t n = b_t.size(1);

  return run_mxfp8_cublaslt_bmm(a, b_t, a_scale_swizzled, b_scale_swizzled, out_dtype, batch, m, k, n, false);
}

at::Tensor mxfp8_cublaslt_bmm_rhs_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const std::string& out_dtype_str) {
  const auto out_dtype = parse_out_dtype(out_dtype_str);

  check_cuda_tensor(a, "a");
  check_cuda_tensor(b, "b");
  check_cuda_tensor(a_scale_swizzled, "a_scale_swizzled");
  check_cuda_tensor(b_scale_swizzled, "b_scale_swizzled");
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3, "mxfp8_cublaslt_bmm_rhs expects 3D tensors");
  TORCH_CHECK(a.scalar_type() == c10::ScalarType::Float8_e4m3fn || a.scalar_type() == c10::ScalarType::Float8_e5m2,
              "a must be FP8");
  TORCH_CHECK(b.scalar_type() == a.scalar_type(), "a and b must have matching FP8 dtype");
  TORCH_CHECK(a_scale_swizzled.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
              "a_scale_swizzled must use torch.float8_e8m0fnu");
  TORCH_CHECK(b_scale_swizzled.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
              "b_scale_swizzled must use torch.float8_e8m0fnu");
  TORCH_CHECK(a.size(0) == b.size(0), "batch dimensions must match");
  TORCH_CHECK(a.size(2) == b.size(1), "K dimensions must match");
  TORCH_CHECK(a.size(1) % 32 == 0 && a.size(2) % 32 == 0 && b.size(2) % 32 == 0,
              "MXFP8 cuBLASLt RHS path requires M, N, and K divisible by 32");

  c10::cuda::CUDAGuard guard(a.device());
  validate_sm100_plus();
  check_same_device(a, b, a_scale_swizzled, b_scale_swizzled);

  const int64_t batch = a.size(0);
  const int64_t m = a.size(1);
  const int64_t k = a.size(2);
  const int64_t n = b.size(2);

  return run_mxfp8_cublaslt_bmm(a, b, a_scale_swizzled, b_scale_swizzled, out_dtype, batch, m, k, n, true);
}

std::vector<at::Tensor> mxfp8_cublaslt_tri_mul_pair_cuda(
    const at::Tensor& a1,
    const at::Tensor& b1,
    const at::Tensor& a2_t,
    const at::Tensor& b2_rhs,
    const at::Tensor& a1_scale_swizzled,
    const at::Tensor& b1_scale_swizzled,
    const at::Tensor& a2_t_scale_swizzled,
    const at::Tensor& b2_rhs_scale_swizzled,
    const std::string& out_dtype_str) {
  const auto out_dtype = parse_out_dtype(out_dtype_str);
  check_cuda_tensor(a1, "a1");
  check_cuda_tensor(b1, "b1");
  check_cuda_tensor(a2_t, "a2_t");
  check_cuda_tensor(b2_rhs, "b2_rhs");
  check_cuda_tensor(a1_scale_swizzled, "a1_scale_swizzled");
  check_cuda_tensor(b1_scale_swizzled, "b1_scale_swizzled");
  check_cuda_tensor(a2_t_scale_swizzled, "a2_t_scale_swizzled");
  check_cuda_tensor(b2_rhs_scale_swizzled, "b2_rhs_scale_swizzled");
  check_same_device(a1, b1, a1_scale_swizzled, b1_scale_swizzled);
  TORCH_CHECK(a1.device() == a2_t.device() && a1.device() == b2_rhs.device() &&
                  a1.device() == a2_t_scale_swizzled.device() && a1.device() == b2_rhs_scale_swizzled.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(a1.dim() == 3 && b1.dim() == 3 && a2_t.dim() == 3 && b2_rhs.dim() == 3, "inputs must be 3D");
  TORCH_CHECK(a1.sizes() == b1.sizes() && a1.sizes() == a2_t.sizes() && a1.sizes() == b2_rhs.sizes(),
              "paired tri-mul inputs must share the same 3D shape");

  c10::cuda::CUDAGuard guard(a1.device());
  validate_sm100_plus();
  const int64_t batch = a1.size(0);
  const int64_t m = a1.size(1);
  const int64_t k = a1.size(2);
  TORCH_CHECK(b1.size(1) == k && b1.size(2) == m, "b1 must have compatible shape");
  TORCH_CHECK(a2_t.size(1) == m && a2_t.size(2) == k, "a2_t must have compatible shape");
  TORCH_CHECK(b2_rhs.size(1) == k && b2_rhs.size(2) == m, "b2_rhs must have compatible shape");

  auto x1 = run_mxfp8_cublaslt_bmm(
      a1, b1, a1_scale_swizzled, b1_scale_swizzled, out_dtype, batch, m, k, m, false);
  auto x2 = run_mxfp8_cublaslt_bmm(
      a2_t, b2_rhs, a2_t_scale_swizzled, b2_rhs_scale_swizzled, out_dtype, batch, m, k, m, true);
  return {std::move(x1), std::move(x2)};
}

std::vector<at::Tensor> mxfp8_cublaslt_tri_mul_pair_backward_cuda(
    const at::Tensor& g1,
    const at::Tensor& g1_t,
    const at::Tensor& g2,
    const at::Tensor& g2_t,
    const at::Tensor& a1_t,
    const at::Tensor& b1_t,
    const at::Tensor& a2,
    const at::Tensor& b2,
    const at::Tensor& g1_scale_swizzled,
    const at::Tensor& g1_t_scale_swizzled,
    const at::Tensor& g2_scale_swizzled,
    const at::Tensor& g2_t_scale_swizzled,
    const at::Tensor& a1_t_scale_swizzled,
    const at::Tensor& b1_t_scale_swizzled,
    const at::Tensor& a2_scale_swizzled,
    const at::Tensor& b2_scale_swizzled) {
  check_cuda_tensor(g1, "g1");
  check_cuda_tensor(g1_t, "g1_t");
  check_cuda_tensor(g2, "g2");
  check_cuda_tensor(g2_t, "g2_t");
  check_cuda_tensor(a1_t, "a1_t");
  check_cuda_tensor(b1_t, "b1_t");
  check_cuda_tensor(a2, "a2");
  check_cuda_tensor(b2, "b2");
  check_cuda_tensor(g1_scale_swizzled, "g1_scale_swizzled");
  check_cuda_tensor(g1_t_scale_swizzled, "g1_t_scale_swizzled");
  check_cuda_tensor(g2_scale_swizzled, "g2_scale_swizzled");
  check_cuda_tensor(g2_t_scale_swizzled, "g2_t_scale_swizzled");
  check_cuda_tensor(a1_t_scale_swizzled, "a1_t_scale_swizzled");
  check_cuda_tensor(b1_t_scale_swizzled, "b1_t_scale_swizzled");
  check_cuda_tensor(a2_scale_swizzled, "a2_scale_swizzled");
  check_cuda_tensor(b2_scale_swizzled, "b2_scale_swizzled");
  check_same_device(g1, g1_t, g1_scale_swizzled, g1_t_scale_swizzled);
  TORCH_CHECK(g1.device() == g2.device() && g1.device() == g2_t.device() && g1.device() == a1_t.device() &&
                  g1.device() == b1_t.device() && g1.device() == a2.device() && g1.device() == b2.device() &&
                  g1.device() == g2_scale_swizzled.device() && g1.device() == g2_t_scale_swizzled.device() &&
                  g1.device() == a1_t_scale_swizzled.device() && g1.device() == b1_t_scale_swizzled.device() &&
                  g1.device() == a2_scale_swizzled.device() && g1.device() == b2_scale_swizzled.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(g1.dim() == 3 && g1_t.dim() == 3 && g2.dim() == 3 && g2_t.dim() == 3, "grad inputs must be 3D");
  TORCH_CHECK(a1_t.dim() == 3 && b1_t.dim() == 3 && a2.dim() == 3 && b2.dim() == 3, "saved inputs must be 3D");

  c10::cuda::CUDAGuard guard(g1.device());
  validate_sm100_plus();
  const int64_t batch = g1.size(0);
  const int64_t m = g1.size(1);
  const int64_t k = g1.size(2);
  TORCH_CHECK(g1_t.size(0) == batch && g1_t.size(1) == m && g1_t.size(2) == k, "g1_t must match g1 shape");
  TORCH_CHECK(g2.sizes() == g1.sizes() && g2_t.sizes() == g1.sizes(), "g2/g2_t must match g1 shape");
  TORCH_CHECK(a1_t.sizes() == g1.sizes() && b1_t.sizes() == g1.sizes() && a2.sizes() == g1.sizes() &&
                  b2.sizes() == g1.sizes(),
              "saved inputs must match grad shape");

  auto grad_a1 = run_mxfp8_cublaslt_bmm(
      g1, b1_t, g1_scale_swizzled, b1_t_scale_swizzled, c10::ScalarType::Float, batch, m, k, m, false);
  auto grad_b1 = run_mxfp8_cublaslt_bmm(
      g1_t, a1_t, g1_t_scale_swizzled, a1_t_scale_swizzled, c10::ScalarType::Float, batch, m, k, m, false);
  auto grad_a2 = run_mxfp8_cublaslt_bmm(
      b2, g2, b2_scale_swizzled, g2_scale_swizzled, c10::ScalarType::Float, batch, m, k, m, false);
  auto grad_b2 = run_mxfp8_cublaslt_bmm(
      a2, g2_t, a2_scale_swizzled, g2_t_scale_swizzled, c10::ScalarType::Float, batch, m, k, m, false);
  return {std::move(grad_a1), std::move(grad_b1), std::move(grad_a2), std::move(grad_b2)};
}

}  // namespace bmm_ext
