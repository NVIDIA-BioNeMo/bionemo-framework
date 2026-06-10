#include "common.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdlib>
#include <memory>

namespace minifold_native_ext {

namespace {

constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;
constexpr float kFp8Max = 448.0f;
constexpr float kMinScale = 1e-12f;
constexpr float kMinPow2Scale = 5.877471754111438e-39f;

#define CUBLASLT_CHECK(EXPR) TORCH_CHECK((EXPR) == CUBLAS_STATUS_SUCCESS, "cuBLASLt error at ", #EXPR)

void check_cuda_tensor(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
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

void validate_sm100_plus() {
  auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props != nullptr, "could not query CUDA device properties");
  const int sm = props->major * 10 + props->minor;
  TORCH_CHECK(sm >= 100, "MiniFold native FP8 kernels require SM100+, found SM", sm);
}

int64_t get_transition_norm_fc1_chunk_rows() {
  constexpr int64_t kDefaultChunkRows = 1 << 16;
  const char* value = std::getenv("MINIFOLD_NATIVE_TRANSITION_FC1_CHUNK_ROWS");
  if (value == nullptr || value[0] == '\0') {
    return kDefaultChunkRows;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value || parsed <= 0) {
    return kDefaultChunkRows;
  }
  const int64_t aligned = std::max<int64_t>(32, (static_cast<int64_t>(parsed) / 32) * 32);
  return aligned > 0 ? aligned : kDefaultChunkRows;
}

bool get_gatepack_use_warp() {
  const char* value = std::getenv("MINIFOLD_NATIVE_GATEPACK_WARP");
  if (value == nullptr) {
    return false;
  }
  return value[0] == '1' || value[0] == 't' || value[0] == 'T' || value[0] == 'y' || value[0] == 'Y';
}

inline cudaStream_t current_cuda_stream() {
  return static_cast<cudaStream_t>(0);
}

void set_cuda_device(const at::Tensor& t) {
  TORCH_CHECK(cudaSetDevice(static_cast<int>(t.get_device())) == cudaSuccess, "failed to set CUDA device");
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

at::Tensor swizzle_scale_rowwise(const at::Tensor& scale) {
  TORCH_CHECK(scale.dim() == 3, "rowwise scale tensor must be 3D");
  const int64_t batch = scale.size(0);
  const int64_t rows = scale.size(1);
  const int64_t cols = scale.size(2);
  const int64_t padded_rows = ((rows + 127) / 128) * 128;
  const int64_t padded_cols = ((cols + 3) / 4) * 4;
  at::Tensor padded = scale;
  if (padded_rows != rows || padded_cols != cols) {
    padded = at::full({batch, padded_rows, padded_cols}, kMinPow2Scale, scale.options());
    padded.slice(1, 0, rows).slice(2, 0, cols).copy_(scale);
  }
  return padded.view({batch, padded_rows / 128, 4, 32, padded_cols / 4, 4})
      .permute({0, 1, 4, 3, 2, 5})
      .contiguous()
      .view({batch, padded_rows, padded_cols});
}

at::Tensor unswizzle_scale_rowwise(const at::Tensor& scale, int64_t rows, int64_t cols) {
  TORCH_CHECK(scale.dim() == 3, "swizzled rowwise scale tensor must be 3D");
  const int64_t batch = scale.size(0);
  const int64_t padded_rows = ((rows + 127) / 128) * 128;
  const int64_t padded_cols = ((cols + 3) / 4) * 4;
  TORCH_CHECK(
      scale.size(1) == padded_rows && scale.size(2) == padded_cols,
      "swizzled rowwise scale tensor has incompatible padded shape");
  auto unswizzled = scale.view({batch, padded_rows / 128, padded_cols / 4, 32, 4, 4})
                        .permute({0, 1, 4, 3, 2, 5})
                        .contiguous()
                        .view({batch, padded_rows, padded_cols});
  return unswizzled.slice(1, 0, rows).slice(2, 0, cols).contiguous();
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
  bool use_bias = false;
  bool use_c_scale = false;
  bool use_d_scale = false;
  bool use_d_out_scale = false;
  c10::ScalarType bias_dtype = c10::ScalarType::Undefined;
  int epilogue = static_cast<int>(CUBLASLT_EPILOGUE_DEFAULT);

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
      bool rhs_direct_,
      bool use_bias_,
      bool use_c_scale_,
      bool use_d_scale_,
      bool use_d_out_scale_,
      c10::ScalarType bias_dtype_,
      int epilogue_) const {
    return device_index == device && batch == batch_ && m == m_ && k == k_ && n == n_ &&
        a_dtype == a_dtype_ && b_dtype == b_dtype_ && out_dtype == out_dtype_ && rhs_direct == rhs_direct_ &&
        use_bias == use_bias_ && use_c_scale == use_c_scale_ && use_d_scale == use_d_scale_ &&
        use_d_out_scale == use_d_out_scale_ &&
        bias_dtype == bias_dtype_ && epilogue == epilogue_;
  }
};

void invalidate_cublaslt_bmm_plan(CublasLtBmmPlan* plan) {
  plan->device_index = -1;
  plan->batch = -1;
  plan->m = -1;
  plan->k = -1;
  plan->n = -1;
  plan->a_dtype = c10::ScalarType::Undefined;
  plan->b_dtype = c10::ScalarType::Undefined;
  plan->out_dtype = c10::ScalarType::Undefined;
  plan->rhs_direct = false;
  plan->use_bias = false;
  plan->use_c_scale = false;
  plan->use_d_scale = false;
  plan->use_d_out_scale = false;
  plan->bias_dtype = c10::ScalarType::Undefined;
  plan->epilogue = static_cast<int>(CUBLASLT_EPILOGUE_DEFAULT);
}

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
    bool use_bias,
    bool use_c_scale,
    bool use_d_scale,
    bool use_d_out_scale,
    c10::ScalarType bias_dtype,
    const void* bias_ptr,
    const void* c_scale_ptr,
    const void* d_scale_ptr,
    const void* d_out_scale_ptr,
    void* amax_d_ptr,
    cublasLtEpilogue_t epilogue,
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
  invalidate_cublaslt_bmm_plan(plan);

  const cublasOperation_t transa = CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;
  const int32_t order_row = CUBLASLT_ORDER_ROW;
  const int32_t order_col = CUBLASLT_ORDER_COL;
  const int32_t b_order = rhs_direct ? order_row : order_col;
  const uint64_t b_ld = rhs_direct ? n : k;
  const int32_t batch_count_i32 = static_cast<int32_t>(batch);
  const int64_t a_batch_stride = m * k;
  const int64_t b_batch_stride = k * n;
  const int64_t c_batch_stride = m * n;
  const int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  const size_t workspace_size = kWorkspaceBytes;

  CUBLASLT_CHECK(cublasLtMatmulDescCreate(&plan->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));
  if (use_bias) {
    const auto bias_cuda_dtype = to_cuda_dtype(bias_dtype);
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_cuda_dtype, sizeof(bias_cuda_dtype)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
  }
  if (use_c_scale) {
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale_ptr, sizeof(c_scale_ptr)));
  }
  if (use_d_scale) {
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_ptr, sizeof(d_scale_ptr)));
  }
  if (use_d_out_scale) {
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_out_scale_ptr, sizeof(d_out_scale_ptr)));
  }
  if (amax_d_ptr != nullptr) {
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->op_desc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d_ptr, sizeof(amax_d_ptr)));
  }

  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->a_desc, to_cuda_dtype(a_dtype), m, k, k));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->b_desc, to_cuda_dtype(b_dtype), k, n, b_ld));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->c_desc, to_cuda_dtype(out_dtype), m, n, n));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->d_desc, to_cuda_dtype(out_dtype), m, n, n));

  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(plan->a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(plan->b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(plan->c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
  CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(plan->d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));

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
  TORCH_CHECK(returned_results > 0, "cuBLASLt found no heuristic for MiniFold native MXFP8 batched GEMM");

  plan->device_index = device_index;
  plan->batch = batch;
  plan->m = m;
  plan->k = k;
  plan->n = n;
  plan->a_dtype = a_dtype;
  plan->b_dtype = b_dtype;
  plan->out_dtype = out_dtype;
  plan->rhs_direct = rhs_direct;
  plan->use_bias = use_bias;
  plan->use_c_scale = use_c_scale;
  plan->use_d_scale = use_d_scale;
  plan->use_d_out_scale = use_d_out_scale;
  plan->bias_dtype = bias_dtype;
  plan->epilogue = static_cast<int>(epilogue);
}

at::Tensor run_mxfp8_cublaslt_bmm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    c10::ScalarType out_dtype,
    bool rhs_direct = false,
    const c10::optional<at::Tensor>& bias = c10::nullopt,
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT) {
  static cublasLtHandle_t lt_handle = nullptr;
  if (lt_handle == nullptr) {
    CUBLASLT_CHECK(cublasLtCreate(&lt_handle));
  }

  const int64_t batch = a.size(0);
  const int64_t m = a.size(1);
  const int64_t k = a.size(2);
  const int64_t n = rhs_direct ? b.size(2) : b.size(1);

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
  const bool use_bias = bias.has_value();
  const auto bias_dtype = use_bias ? bias->scalar_type() : c10::ScalarType::Undefined;
  const void* bias_ptr = use_bias ? bias->data_ptr() : nullptr;
  if (!cached_plan->matches(
          device_index,
          batch,
          m,
          k,
          n,
          a.scalar_type(),
          b.scalar_type(),
          out_dtype,
          rhs_direct,
          use_bias,
          false,
          false,
          false,
          bias_dtype,
          static_cast<int>(epilogue))) {
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
        use_bias,
        false,
        false,
        false,
        bias_dtype,
        bias_ptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        epilogue,
        lt_handle);
  }
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      cached_plan->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      cached_plan->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));
  if (use_bias) {
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == n, "bias shape must match output width");
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        cached_plan->op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
  }

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
      current_cuda_stream()));
  return out;
}

std::tuple<at::Tensor, at::Tensor> run_mxfp8_cublaslt_bmm_direct_output_impl(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const c10::optional<at::Tensor>& bias,
    cublasLtEpilogue_t epilogue,
    bool use_d_scale_pointer) {
  static cublasLtHandle_t lt_handle = nullptr;
  if (lt_handle == nullptr) {
    CUBLASLT_CHECK(cublasLtCreate(&lt_handle));
  }

  const int64_t batch = a.size(0);
  const int64_t m = a.size(1);
  const int64_t k = a.size(2);
  const int64_t n = b.size(1);
  TORCH_CHECK(n % 32 == 0, "direct FP8 output requires output width divisible by 32, got ", n);

  auto out = at::empty({batch, m, n}, a.options().dtype(at::kFloat8_e4m3fn));
  const int64_t groups = n / 32;
  const int64_t padded_rows = ((m + 127) / 128) * 128;
  const int64_t padded_cols = ((groups + 3) / 4) * 4;
  auto out_scale_swizzled = at::full(
      {batch, padded_rows, padded_cols},
      kMinPow2Scale,
      a.options().dtype(at::kFloat8_e8m0fnu));
  auto amax_d = at::zeros({1}, a.options().dtype(at::kFloat));
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
  const bool use_bias = bias.has_value();
  const auto bias_dtype = use_bias ? bias->scalar_type() : c10::ScalarType::Undefined;
  const void* bias_ptr = use_bias ? bias->data_ptr() : nullptr;
  const void* c_scale_ptr = out_scale_swizzled.data_ptr();
  const void* d_scale_ptr = use_d_scale_pointer ? out_scale_swizzled.data_ptr() : nullptr;
  const void* d_out_scale_ptr = out_scale_swizzled.data_ptr();
  void* amax_d_ptr = amax_d.data_ptr();
  if (!cached_plan->matches(
          device_index,
          batch,
          m,
          k,
          n,
          a.scalar_type(),
          b.scalar_type(),
          c10::ScalarType::Float8_e4m3fn,
          false,
          use_bias,
          true,
          use_d_scale_pointer,
          true,
          bias_dtype,
          static_cast<int>(epilogue))) {
    init_cublaslt_bmm_plan(
        cached_plan.get(),
        device_index,
        batch,
        m,
        k,
        n,
        a.scalar_type(),
        b.scalar_type(),
        c10::ScalarType::Float8_e4m3fn,
        a_scale_ptr,
        b_scale_ptr,
        false,
        use_bias,
        true,
        use_d_scale_pointer,
        true,
        bias_dtype,
        bias_ptr,
        c_scale_ptr,
        d_scale_ptr,
        d_out_scale_ptr,
        amax_d_ptr,
        epilogue,
        lt_handle);
  }
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      cached_plan->op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      cached_plan->op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));
  if (use_bias) {
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == n, "bias shape must match output width");
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        cached_plan->op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
  }
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      cached_plan->op_desc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale_ptr, sizeof(c_scale_ptr)));
  if (use_d_scale_pointer) {
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        cached_plan->op_desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_ptr, sizeof(d_scale_ptr)));
  }
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      cached_plan->op_desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_out_scale_ptr, sizeof(d_out_scale_ptr)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      cached_plan->op_desc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d_ptr, sizeof(amax_d_ptr)));

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
      current_cuda_stream()));
  auto out_scale = unswizzle_scale_rowwise(out_scale_swizzled, m, groups).to(at::kFloat).contiguous();
  return {out, out_scale};
}

std::tuple<at::Tensor, at::Tensor> run_mxfp8_cublaslt_bmm_direct_output(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const c10::optional<at::Tensor>& bias,
    cublasLtEpilogue_t epilogue) {
  try {
    return run_mxfp8_cublaslt_bmm_direct_output_impl(
        a, b, a_scale_swizzled, b_scale_swizzled, bias, epilogue, true);
  } catch (const c10::Error&) {
    return run_mxfp8_cublaslt_bmm_direct_output_impl(
        a, b, a_scale_swizzled, b_scale_swizzled, bias, epilogue, false);
  }
}

__device__ inline float fp8_storage_to_float(__nv_fp8_storage_t value) {
  const __half_raw raw = __nv_cvt_fp8_to_halfraw(value, __NV_E4M3);
  union {
    __half_raw raw_bits;
    __half half_value;
  } half_union;
  half_union.raw_bits = raw;
  return __half2float(half_union.half_value);
}

__device__ inline __nv_fp8_storage_t float_to_fp8_storage(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

__device__ inline __nv_fp8_storage_t float_to_e8m0_storage(float value) {
  return __nv_cvt_float_to_e8m0(value, __NV_SATFINITE, cudaRoundPosInf);
}

__device__ inline float pow2_scale_from_max(float max_abs) {
  const float base_scale = fmaxf(max_abs / kFp8Max, kMinPow2Scale);
  return fmaxf(exp2f(ceilf(log2f(base_scale))), kMinPow2Scale);
}

__device__ inline int swizzled_scale_offset(int row, int col, int padded_cols) {
  const int col_blocks = padded_cols / 4;
  const int row_block = row >> 7;
  const int row_sub4 = (row >> 5) & 0x3;
  const int row_lane = row & 0x1f;
  const int col_block = col >> 2;
  const int col_sub = col & 0x3;
  return (((((row_block * col_blocks) + col_block) * 32 + row_lane) * 4 + row_sub4) * 4 + col_sub);
}

__device__ inline float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ inline float warp_reduce_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__global__ void quantize_block32_bf16_bias_generic_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* bias,
    const __nv_fp8_storage_t* residual_payload,
    const float* residual_scale,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int rows,
    int cols,
    int groups,
    bool apply_relu) {
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (row >= rows || col >= cols) {
    return;
  }
  const int warp_id = col >> 5;
  const int lane = col & 31;
  const int idx = row * cols + col;
  float value = __bfloat162float(input[idx]);
  if (bias != nullptr) {
    value += __bfloat162float(bias[col]);
  }
  if (residual_payload != nullptr) {
    value += fp8_storage_to_float(residual_payload[idx]) * residual_scale[row * groups + warp_id];
  }
  if (apply_relu && value < 0.0f) {
    value = 0.0f;
  }
  float max_abs = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  const float scale = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs, 0));
  if (lane == 0) {
    output_scale[row * groups + warp_id] = scale;
  }
  output[idx] = float_to_fp8_storage(value / scale);
}

__global__ void quantize_block32_bf16_generic_kernel(
    const __nv_bfloat16* input,
    const __nv_fp8_storage_t* residual_payload,
    const float* residual_scale,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int rows,
    int cols,
    int groups) {
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (row >= rows || col >= cols) {
    return;
  }
  const int warp_id = col >> 5;
  const int lane = col & 31;
  const int idx = row * cols + col;
  float value = __bfloat162float(input[idx]);
  if (residual_payload != nullptr) {
    value += fp8_storage_to_float(residual_payload[idx]) * residual_scale[row * groups + warp_id];
  }
  const float scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(value)), 0));
  if (lane == 0) {
    output_scale[row * groups + warp_id] = scale;
  }
  output[idx] = float_to_fp8_storage(value / scale);
}

template <int COLS, bool HAS_BIAS, bool APPLY_RELU, bool HAS_RESIDUAL>
__global__ void quantize_block32_bf16_specialized_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* bias,
    const __nv_fp8_storage_t* residual_payload,
    const float* residual_scale,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int rows) {
  constexpr int kGroups = COLS / 32;
  constexpr int kGroupTile = (COLS == 128 || COLS == 512) ? 4 : 1;
  constexpr int kWarpsPerRow = kGroups / kGroupTile;
  constexpr int kRowsPerBlock = COLS == 128 ? 8 : (COLS == 512 ? 2 : 1);
  constexpr int kWarpsPerBlock = kWarpsPerRow * kRowsPerBlock;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  if (warp_id >= kWarpsPerBlock) {
    return;
  }
  const int row_in_block = warp_id / kWarpsPerRow;
  const int warp_in_row = warp_id % kWarpsPerRow;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= rows) {
    return;
  }

#pragma unroll
  for (int tile = 0; tile < kGroupTile; ++tile) {
    const int group = warp_in_row * kGroupTile + tile;
    const int col = group * 32 + lane;
    const int idx = row * COLS + col;
    float value = __bfloat162float(input[idx]);
    if constexpr (HAS_BIAS) {
      value += __bfloat162float(bias[col]);
    }
    if constexpr (HAS_RESIDUAL) {
      value += fp8_storage_to_float(residual_payload[idx]) * residual_scale[row * kGroups + group];
    }
    if constexpr (APPLY_RELU) {
      value = fmaxf(value, 0.0f);
    }
    const float max_abs = warp_reduce_max(fabsf(value));
    const float scale = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs, 0));
    if (lane == 0) {
      output_scale[row * kGroups + group] = scale;
    }
    output[idx] = float_to_fp8_storage(value / scale);
  }
}

template <bool HAS_RESIDUAL>
__global__ void gate_sigmoid_mul_quantize_bf16_128_kernel(
    const __nv_bfloat16* lhs,
    const __nv_bfloat16* rhs,
    const __nv_bfloat16* lhs_bias,
    const __nv_bfloat16* rhs_bias,
    const __nv_fp8_storage_t* residual_payload,
    const float* residual_scale,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int rows) {
  constexpr int kCols = 128;
  constexpr int kGroups = 4;
  constexpr int kRowsPerBlock = 8;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + warp_id;
  if (row >= rows || warp_id >= kRowsPerBlock) {
    return;
  }

  float values[kGroups];
  float scales[kGroups];
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    const int col = lane + group * 32;
    const int idx = row * kCols + col;
    float lhs_value = __bfloat162float(lhs[idx]) + __bfloat162float(lhs_bias[col]);
    float rhs_value = __bfloat162float(rhs[idx]) + __bfloat162float(rhs_bias[col]);
    float out_value = lhs_value * (1.0f / (1.0f + __expf(-rhs_value)));
    if constexpr (HAS_RESIDUAL) {
      out_value += fp8_storage_to_float(residual_payload[idx]) * residual_scale[row * kGroups + group];
    }
    values[group] = out_value;
    scales[group] = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(out_value)), 0));
  }
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    if (lane == 0) {
      output_scale[row * kGroups + group] = scales[group];
    }
    const int col = lane + group * 32;
    output[row * kCols + col] = float_to_fp8_storage(values[group] / scales[group]);
  }
}

__global__ void gate_sigmoid_mul_quantize_bf16_kernel(
    const __nv_bfloat16* lhs,
    const __nv_bfloat16* rhs,
    const __nv_bfloat16* lhs_bias,
    const __nv_bfloat16* rhs_bias,
    const __nv_fp8_storage_t* residual_payload,
    const float* residual_scale,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int rows,
    int cols,
    int groups) {
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (row >= rows || col >= cols) {
    return;
  }
  const int warp_id = col >> 5;
  const int lane = col & 31;
  const int idx = row * cols + col;
  float lhs_value = __bfloat162float(lhs[idx]);
  float rhs_value = __bfloat162float(rhs[idx]);
  if (lhs_bias != nullptr) {
    lhs_value += __bfloat162float(lhs_bias[col]);
  }
  if (rhs_bias != nullptr) {
    rhs_value += __bfloat162float(rhs_bias[col]);
  }
  float out_value = lhs_value * (1.0f / (1.0f + expf(-rhs_value)));
  if (residual_payload != nullptr) {
    out_value += fp8_storage_to_float(residual_payload[idx]) * residual_scale[row * groups + warp_id];
  }
  float max_abs = fabsf(out_value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  const float scale = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs, 0));
  if (lane == 0) {
    output_scale[row * groups + warp_id] = scale;
  }
  output[idx] = float_to_fp8_storage(out_value / scale);
}

__global__ void unary_block32_kernel(
    const __nv_fp8_storage_t* input,
    const float* input_scale,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int rows,
    int cols,
    int groups,
    int op_code) {
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (row >= rows || col >= cols) {
    return;
  }
  const int warp_id = col >> 5;
  const int lane = col & 31;
  const int idx = row * cols + col;
  float value = fp8_storage_to_float(input[idx]) * input_scale[row * groups + warp_id];
  if (op_code == 0) {
    value = fmaxf(value, 0.0f);
  } else {
    value = 1.0f / (1.0f + expf(-value));
  }
  float max_abs = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  const float scale = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs, 0));
  if (lane == 0) {
    output_scale[row * groups + warp_id] = scale;
  }
  output[idx] = float_to_fp8_storage(value / scale);
}

__global__ void binary_block32_kernel(
    const __nv_fp8_storage_t* lhs,
    const float* lhs_scale,
    const __nv_fp8_storage_t* rhs,
    const float* rhs_scale,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int rows,
    int cols,
    int groups,
    int op_code) {
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (row >= rows || col >= cols) {
    return;
  }
  const int warp_id = col >> 5;
  const int lane = col & 31;
  const int idx = row * cols + col;
  const float lhs_value = fp8_storage_to_float(lhs[idx]) * lhs_scale[row * groups + warp_id];
  const float rhs_value = fp8_storage_to_float(rhs[idx]) * rhs_scale[row * groups + warp_id];
  const float out_value = op_code == 0 ? lhs_value + rhs_value : lhs_value * rhs_value;
  float max_abs = fabsf(out_value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  const float scale = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs, 0));
  if (lane == 0) {
    output_scale[row * groups + warp_id] = scale;
  }
  output[idx] = float_to_fp8_storage(out_value / scale);
}

__global__ void layernorm_block32_generic_kernel(
    const __nv_fp8_storage_t* input,
    const float* input_scale,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int rows,
    int cols,
    int groups,
    float eps) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (row >= rows || col >= cols) {
    if (col < blockDim.x) {
      shared[col] = 0.0f;
    }
    return;
  }

  const int warp_id = col >> 5;
  const int lane = col & 31;
  const int idx = row * cols + col;
  const float x = fp8_storage_to_float(input[idx]) * input_scale[row * groups + warp_id];
  shared[col] = x;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (col < stride) {
      shared[col] += shared[col + stride];
    }
    __syncthreads();
  }
  const float mean = shared[0] / static_cast<float>(cols);
  const float centered = x - mean;
  shared[col] = centered * centered;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (col < stride) {
      shared[col] += shared[col + stride];
    }
    __syncthreads();
  }
  const float inv_std = rsqrtf(shared[0] / static_cast<float>(cols) + eps);
  const float y = centered * inv_std * __bfloat162float(weight[col]) + __bfloat162float(bias[col]);

  float max_abs = fabsf(y);
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  const float scale = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs, 0));
  if (lane == 0) {
    output_scale[row * groups + warp_id] = scale;
  }
  output[idx] = float_to_fp8_storage(y / scale);
}

template <int COLS>
__global__ void layernorm_block32_kernel_small(
    const __nv_fp8_storage_t* input,
    const float* input_scale,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int rows,
    int groups,
    float eps) {
  constexpr int kRowsPerBlock = 8;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + warp_id;
  if (row >= rows || warp_id >= kRowsPerBlock) {
    return;
  }

  constexpr int kGroups = COLS / 32;
  float x_vals[kGroups];
  float sum = 0.0f;
  float sumsq = 0.0f;
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    const int col = lane + group * 32;
    const float x = fp8_storage_to_float(input[row * COLS + col]) * input_scale[row * groups + group];
    x_vals[group] = x;
    sum += x;
    sumsq += x * x;
  }
  const float total = warp_reduce_sum(sum);
  const float total_sq = warp_reduce_sum(sumsq);
  const float mean = __shfl_sync(0xffffffff, total / static_cast<float>(COLS), 0);
  const float inv_std = __shfl_sync(
      0xffffffff,
      rsqrtf(fmaxf(total_sq / static_cast<float>(COLS) - mean * mean, 0.0f) + eps),
      0);

  float y_vals[kGroups];
  float scales[kGroups];
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    const int col = lane + group * 32;
    const float y =
        (x_vals[group] - mean) * inv_std * __bfloat162float(weight[col]) + __bfloat162float(bias[col]);
    y_vals[group] = y;
    scales[group] = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(y)), 0));
  }
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    if (lane == 0) {
      output_scale[row * groups + group] = scales[group];
    }
    const int col = lane + group * 32;
    output[row * COLS + col] = float_to_fp8_storage(y_vals[group] / scales[group]);
  }
}

__global__ void transition_norm_to_mxfp8_input_kernel(
    const __nv_fp8_storage_t* input,
    const float* input_scale,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_fp8_storage_t* output,
    __nv_fp8_storage_t* output_scale_swizzled,
    int rows,
    float eps) {
  constexpr int kCols = 128;
  constexpr int kGroups = 4;
  constexpr int kPaddedCols = 4;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + warp_id;
  if (row >= rows || warp_id >= 4) {
    return;
  }

  float x_vals[kGroups];
  float sum = 0.0f;
  float sumsq = 0.0f;
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    const int col = lane + group * 32;
    const float x = fp8_storage_to_float(input[row * kCols + col]) * input_scale[row * kGroups + group];
    x_vals[group] = x;
    sum += x;
    sumsq += x * x;
  }
  const float total = warp_reduce_sum(sum);
  const float total_sq = warp_reduce_sum(sumsq);
  const float mean = __shfl_sync(0xffffffff, total / static_cast<float>(kCols), 0);
  const float inv_std = __shfl_sync(
      0xffffffff,
      rsqrtf(fmaxf(total_sq / static_cast<float>(kCols) - mean * mean, 0.0f) + eps),
      0);

  float y_vals[kGroups];
  float scales[kGroups];
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    const int col = lane + group * 32;
    const float y =
        (x_vals[group] - mean) * inv_std * __bfloat162float(weight[col]) + __bfloat162float(bias[col]);
    y_vals[group] = y;
    scales[group] = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(y)), 0));
  }
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    const int col = lane + group * 32;
    output[row * kCols + col] = float_to_fp8_storage(y_vals[group] / scales[group]);
    if (lane == 0) {
      output_scale_swizzled[swizzled_scale_offset(row, group, kPaddedCols)] = float_to_e8m0_storage(scales[group]);
    }
  }
}

__global__ void pack_block32_to_mxfp8_lhs_pair_mask_kernel(
    const __nv_fp8_storage_t* payload,
    const float* scale,
    const bool* mask,
    __nv_fp8_storage_t* output0,
    float* output0_scale,
    __nv_fp8_storage_t* output1,
    float* output1_scale,
    int batch,
    int n) {
  const int bd = blockIdx.x;
  const int row = blockIdx.y;
  const int kblock = blockIdx.z;
  const int lane = threadIdx.x;
  const int batch_idx = bd / 32;
  const int channel = bd % 32;
  const int k = kblock * 32 + lane;
  if (batch_idx >= batch || row >= n || k >= n) {
    return;
  }

  const int base = (((batch_idx * n + row) * n + k) * 128) + channel;
  const int scale_base = (((batch_idx * n + row) * n + k) * 4);
  float value0 = fp8_storage_to_float(payload[base]) * scale[scale_base];
  float value1 = fp8_storage_to_float(payload[base + 32]) * scale[scale_base + 1];
  if (mask != nullptr) {
    const int mask_idx = ((batch_idx * n + row) * n) + k;
    const float mask_value = mask[mask_idx] ? 1.0f : 0.0f;
    value0 *= mask_value;
    value1 *= mask_value;
  }

  float max_abs0 = fabsf(value0);
  float max_abs1 = fabsf(value1);
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs0 = fmaxf(max_abs0, __shfl_down_sync(0xffffffff, max_abs0, offset));
    max_abs1 = fmaxf(max_abs1, __shfl_down_sync(0xffffffff, max_abs1, offset));
  }
  const float block_scale0 = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs0, 0));
  const float block_scale1 = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs1, 0));
  output0[((bd * n + row) * n) + k] = float_to_fp8_storage(value0 / block_scale0);
  output1[((bd * n + row) * n) + k] = float_to_fp8_storage(value1 / block_scale1);
  if (lane == 0) {
    output0_scale[((bd * n + row) * (n / 32)) + kblock] = block_scale0;
    output1_scale[((bd * n + row) * (n / 32)) + kblock] = block_scale1;
  }
}

__global__ void pack_block32_to_mxfp8_lhs_mask_kernel(
    const __nv_fp8_storage_t* payload,
    const float* scale,
    const bool* mask,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int batch,
    int n,
    int channel_group,
    bool transpose) {
  const int bd = blockIdx.x;
  const int row = blockIdx.y;
  const int kblock = blockIdx.z;
  const int lane = threadIdx.x;
  const int batch_idx = bd / 32;
  const int channel = (bd % 32) + channel_group * 32;
  const int k = kblock * 32 + lane;
  if (batch_idx >= batch || row >= n || k >= n) {
    return;
  }

  const int i = transpose ? k : row;
  const int j = transpose ? row : k;
  const int payload_idx = (((batch_idx * n + i) * n + j) * 128) + channel;
  const int scale_idx = (((batch_idx * n + i) * n + j) * 4) + channel_group;
  float value = fp8_storage_to_float(payload[payload_idx]) * scale[scale_idx];
  if (mask != nullptr) {
    const int mask_idx = ((batch_idx * n + i) * n) + j;
    value *= mask[mask_idx] ? 1.0f : 0.0f;
  }

  float max_abs = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  const float block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs, 0));
  output[((bd * n + row) * n) + k] = float_to_fp8_storage(value / block_scale);
  if (lane == 0) {
    output_scale[((bd * n + row) * (n / 32)) + kblock] = block_scale;
  }
}

__global__ void pack_block32_to_mxfp8_rhs_mask_kernel(
    const __nv_fp8_storage_t* payload,
    const float* scale,
    const bool* mask,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int batch,
    int n,
    int channel_group) {
  const int bd = blockIdx.x;
  const int kblock = blockIdx.y;
  const int col = blockIdx.z;
  const int lane = threadIdx.x;
  const int batch_idx = bd / 32;
  const int channel = (bd % 32) + channel_group * 32;
  const int k = kblock * 32 + lane;
  if (batch_idx >= batch || col >= n || k >= n) {
    return;
  }
  const int payload_idx = (((batch_idx * n + k) * n + col) * 128) + channel;
  const int scale_idx = (((batch_idx * n + k) * n + col) * 4) + channel_group;
  float value = fp8_storage_to_float(payload[payload_idx]) * scale[scale_idx];
  if (mask != nullptr) {
    const int mask_idx = ((batch_idx * n + k) * n) + col;
    value *= mask[mask_idx] ? 1.0f : 0.0f;
  }

  float max_abs = fabsf(value);
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  const float block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs, 0));
  output[((bd * n + k) * n) + col] = float_to_fp8_storage(value / block_scale);
  if (lane == 0) {
    output_scale[((bd * (n / 32) + kblock) * n) + col] = block_scale;
  }
}

__global__ void pack_block32_to_mxfp8_fused_kernel(
    const __nv_fp8_storage_t* payload,
    const float* scale,
    const bool* mask,
    __nv_fp8_storage_t* a1,
    __nv_fp8_storage_t* a1_scale,
    __nv_fp8_storage_t* b1,
    __nv_fp8_storage_t* b1_scale,
    __nv_fp8_storage_t* a2_t,
    __nv_fp8_storage_t* a2_t_scale,
    __nv_fp8_storage_t* b2_rhs,
    __nv_fp8_storage_t* b2_rhs_scale,
    int batch,
    int n,
    int padded_rows,
    int padded_cols) {
  constexpr int kWarpsPerRow = 8;
  constexpr int kRowsPerBlock = 2;
  const int batch_channel_block = blockIdx.x;
  const int kblock = blockIdx.z;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int batch_idx = batch_channel_block / 4;
  const int row_in_block = warp_id / kWarpsPerRow;
  const int warp_in_row = warp_id % kWarpsPerRow;
  const int row = blockIdx.y * kRowsPerBlock + row_in_block;
  const int channel = ((batch_channel_block % 4) << 3) + warp_in_row;
  const int k = kblock * 32 + lane;
  if (batch_idx >= batch || row >= n || k >= n || warp_id >= kWarpsPerRow * kRowsPerBlock) {
    return;
  }

  const int rowk_base = (((batch_idx * n + row) * n + k) * 128);
  const int rowk_scale_base = (((batch_idx * n + row) * n + k) * 4);
  const int krow_base = (((batch_idx * n + k) * n + row) * 128);
  const int krow_scale_base = (((batch_idx * n + k) * n + row) * 4);
  float mask_rowk = 1.0f;
  float mask_krow = 1.0f;
  if (mask != nullptr) {
    mask_rowk = mask[((batch_idx * n + row) * n) + k] ? 1.0f : 0.0f;
    mask_krow = mask[((batch_idx * n + k) * n) + row] ? 1.0f : 0.0f;
  }

  const float a1_value = fp8_storage_to_float(payload[rowk_base + channel]) * scale[rowk_scale_base] * mask_rowk;
  const float b1_value = fp8_storage_to_float(payload[rowk_base + 32 + channel]) * scale[rowk_scale_base + 1] * mask_rowk;
  const float a2_value = fp8_storage_to_float(payload[krow_base + 64 + channel]) * scale[krow_scale_base + 2] * mask_krow;
  const float b2_value = fp8_storage_to_float(payload[krow_base + 96 + channel]) * scale[krow_scale_base + 3] * mask_krow;

  const float a1_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(a1_value)), 0));
  const float b1_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(b1_value)), 0));
  const float a2_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(a2_value)), 0));
  const float b2_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(b2_value)), 0));

  const int plane = batch_idx * 32 + channel;
  a1[((plane * n + row) * n) + k] = float_to_fp8_storage(a1_value / a1_block_scale);
  b1[((plane * n + row) * n) + k] = float_to_fp8_storage(b1_value / b1_block_scale);
  a2_t[((plane * n + row) * n) + k] = float_to_fp8_storage(a2_value / a2_block_scale);
  b2_rhs[((plane * n + k) * n) + row] = float_to_fp8_storage(b2_value / b2_block_scale);

  if (lane == 0) {
    const int plane_offset = plane * padded_rows * padded_cols;
    const int scale_offset = plane_offset + swizzled_scale_offset(row, kblock, padded_cols);
    a1_scale[scale_offset] = float_to_e8m0_storage(a1_block_scale);
    b1_scale[scale_offset] = float_to_e8m0_storage(b1_block_scale);
    a2_t_scale[scale_offset] = float_to_e8m0_storage(a2_block_scale);
    b2_rhs_scale[scale_offset] = float_to_e8m0_storage(b2_block_scale);
  }
}

__global__ void gate_sigmoid_mul_pack_rowk_mxfp8_kernel(
    const __nv_bfloat16* lhs,
    const __nv_bfloat16* rhs,
    const __nv_bfloat16* lhs_bias,
    const __nv_bfloat16* rhs_bias,
    const bool* mask,
    __nv_fp8_storage_t* a1,
    __nv_fp8_storage_t* a1_scale,
    __nv_fp8_storage_t* b1,
    __nv_fp8_storage_t* b1_scale,
    int batch,
    int n,
    int padded_rows,
    int padded_cols) {
  constexpr int kRowsPerBlock = 2;
  constexpr int kWarpsPerRow = 4;
  constexpr int kWarpsPerBlock = 8;
  __shared__ float lhs_bias_shared[64];
  __shared__ float rhs_bias_shared[64];
  __shared__ float rowk_mask_shared[kRowsPerBlock][32];
  __shared__ __nv_bfloat16 lhs_rowk0_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 rhs_rowk0_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 lhs_rowk1_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 rhs_rowk1_shared[kRowsPerBlock][kWarpsPerRow][32];

  const int tid = threadIdx.x;
  if (tid < 64) {
    lhs_bias_shared[tid] = __bfloat162float(lhs_bias[tid]);
    rhs_bias_shared[tid] = __bfloat162float(rhs_bias[tid]);
  }

  const int batch_channel_block = blockIdx.x;
  const int kblock = blockIdx.z;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int batch_idx = batch_channel_block / 8;
  const int row_in_block = warp_id / kWarpsPerRow;
  const int warp_in_row = warp_id % kWarpsPerRow;
  const int row = blockIdx.y * kRowsPerBlock + row_in_block;
  const int channel = ((batch_channel_block % 8) << 2) + warp_in_row;
  const int k = kblock * 32 + lane;
  if (warp_id < kWarpsPerBlock && batch_idx < batch && row < n && k < n) {
    const int64_t rowk_position = static_cast<int64_t>(batch_idx) * n * n + static_cast<int64_t>(row) * n + k;
    const int rowk_base = static_cast<int>(rowk_position * 128);

    lhs_rowk0_shared[row_in_block][warp_in_row][lane] = lhs[rowk_base + channel];
    rhs_rowk0_shared[row_in_block][warp_in_row][lane] = rhs[rowk_base + channel];
    lhs_rowk1_shared[row_in_block][warp_in_row][lane] = lhs[rowk_base + 32 + channel];
    rhs_rowk1_shared[row_in_block][warp_in_row][lane] = rhs[rowk_base + 32 + channel];

    if (warp_in_row == 0) {
      rowk_mask_shared[row_in_block][lane] = (mask == nullptr || mask[(batch_idx * n + row) * n + k]) ? 1.0f : 0.0f;
    }
  }
  __syncthreads();
  if (batch_idx >= batch || row >= n || k >= n || warp_id >= kWarpsPerBlock) {
    return;
  }

  const float mask_rowk = rowk_mask_shared[row_in_block][lane];
  const float a1_lhs = __bfloat162float(lhs_rowk0_shared[row_in_block][warp_in_row][lane]) + lhs_bias_shared[channel];
  const float a1_rhs = __bfloat162float(rhs_rowk0_shared[row_in_block][warp_in_row][lane]) + rhs_bias_shared[channel];
  const float b1_lhs =
      __bfloat162float(lhs_rowk1_shared[row_in_block][warp_in_row][lane]) + lhs_bias_shared[32 + channel];
  const float b1_rhs =
      __bfloat162float(rhs_rowk1_shared[row_in_block][warp_in_row][lane]) + rhs_bias_shared[32 + channel];

  const float a1_value = a1_lhs * (1.0f / (1.0f + __expf(-a1_rhs))) * mask_rowk;
  const float b1_value = b1_lhs * (1.0f / (1.0f + __expf(-b1_rhs))) * mask_rowk;

  const float a1_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(a1_value)), 0));
  const float b1_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(b1_value)), 0));

  const int plane = batch_idx * 32 + channel;
  a1[((plane * n + row) * n) + k] = float_to_fp8_storage(a1_value / a1_block_scale);
  b1[((plane * n + row) * n) + k] = float_to_fp8_storage(b1_value / b1_block_scale);

  if (lane == 0) {
    const int plane_offset = plane * padded_rows * padded_cols;
    const int scale_offset = plane_offset + swizzled_scale_offset(row, kblock, padded_cols);
    a1_scale[scale_offset] = float_to_e8m0_storage(a1_block_scale);
    b1_scale[scale_offset] = float_to_e8m0_storage(b1_block_scale);
  }
}

__global__ void gate_sigmoid_mul_pack_to_mxfp8_warp_kernel(
    const __nv_bfloat16* lhs,
    const __nv_bfloat16* rhs,
    const __nv_bfloat16* lhs_bias,
    const __nv_bfloat16* rhs_bias,
    const bool* mask,
    __nv_fp8_storage_t* a1,
    __nv_fp8_storage_t* a1_scale,
    __nv_fp8_storage_t* b1,
    __nv_fp8_storage_t* b1_scale,
    __nv_fp8_storage_t* a2_t,
    __nv_fp8_storage_t* a2_t_scale,
    __nv_fp8_storage_t* b2_rhs,
    __nv_fp8_storage_t* b2_rhs_scale,
    int batch,
    int n,
    int padded_rows,
    int padded_cols) {
  const int plane = blockIdx.x;
  const int row = blockIdx.y;
  const int kblock = blockIdx.z;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp_id >= 4) {
    return;
  }

  const int batch_idx = plane / 32;
  const int channel = plane % 32;
  const int k = kblock * 32 + lane;
  if (batch_idx >= batch || row >= n || k >= n) {
    return;
  }

  const int64_t rowk_position = static_cast<int64_t>(batch_idx) * n * n + static_cast<int64_t>(row) * n + k;
  const int64_t krow_position = static_cast<int64_t>(batch_idx) * n * n + static_cast<int64_t>(k) * n + row;
  const int rowk_base = static_cast<int>(rowk_position * 128);
  const int krow_base = static_cast<int>(krow_position * 128);
  const float mask_rowk = (mask == nullptr || mask[(batch_idx * n + row) * n + k]) ? 1.0f : 0.0f;
  const float mask_krow = (mask == nullptr || mask[(batch_idx * n + k) * n + row]) ? 1.0f : 0.0f;

  float value = 0.0f;
  __nv_fp8_storage_t* output = nullptr;
  __nv_fp8_storage_t* output_scale = nullptr;
  int out_i = row;
  int out_j = k;

  if (warp_id == 0) {
    const float lhs_value = __bfloat162float(lhs[rowk_base + channel]) + __bfloat162float(lhs_bias[channel]);
    const float rhs_value = __bfloat162float(rhs[rowk_base + channel]) + __bfloat162float(rhs_bias[channel]);
    value = lhs_value * (1.0f / (1.0f + __expf(-rhs_value))) * mask_rowk;
    output = a1;
    output_scale = a1_scale;
  } else if (warp_id == 1) {
    const float lhs_value = __bfloat162float(lhs[rowk_base + 32 + channel]) + __bfloat162float(lhs_bias[32 + channel]);
    const float rhs_value = __bfloat162float(rhs[rowk_base + 32 + channel]) + __bfloat162float(rhs_bias[32 + channel]);
    value = lhs_value * (1.0f / (1.0f + __expf(-rhs_value))) * mask_rowk;
    output = b1;
    output_scale = b1_scale;
  } else if (warp_id == 2) {
    const float lhs_value = __bfloat162float(lhs[krow_base + 64 + channel]) + __bfloat162float(lhs_bias[64 + channel]);
    const float rhs_value = __bfloat162float(rhs[krow_base + 64 + channel]) + __bfloat162float(rhs_bias[64 + channel]);
    value = lhs_value * (1.0f / (1.0f + __expf(-rhs_value))) * mask_krow;
    output = a2_t;
    output_scale = a2_t_scale;
  } else {
    const float lhs_value = __bfloat162float(lhs[krow_base + 96 + channel]) + __bfloat162float(lhs_bias[96 + channel]);
    const float rhs_value = __bfloat162float(rhs[krow_base + 96 + channel]) + __bfloat162float(rhs_bias[96 + channel]);
    value = lhs_value * (1.0f / (1.0f + __expf(-rhs_value))) * mask_krow;
    output = b2_rhs;
    output_scale = b2_rhs_scale;
    out_i = k;
    out_j = row;
  }

  const float block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(value)), 0));
  output[((plane * n + out_i) * n) + out_j] = float_to_fp8_storage(value / block_scale);
  if (lane == 0) {
    const int plane_offset = plane * padded_rows * padded_cols;
    const int scale_offset = plane_offset + swizzled_scale_offset(row, kblock, padded_cols);
    output_scale[scale_offset] = float_to_e8m0_storage(block_scale);
  }
}

__global__ void gate_sigmoid_mul_pack_to_mxfp8_fused_kernel(
    const __nv_bfloat16* lhs,
    const __nv_bfloat16* rhs,
    const __nv_bfloat16* lhs_bias,
    const __nv_bfloat16* rhs_bias,
    const bool* mask,
    __nv_fp8_storage_t* a1,
    __nv_fp8_storage_t* a1_scale,
    __nv_fp8_storage_t* b1,
    __nv_fp8_storage_t* b1_scale,
    __nv_fp8_storage_t* a2_t,
    __nv_fp8_storage_t* a2_t_scale,
    __nv_fp8_storage_t* b2_rhs,
    __nv_fp8_storage_t* b2_rhs_scale,
    int batch,
    int n,
    int padded_rows,
    int padded_cols) {
  constexpr int kRowsPerBlock = 2;
  constexpr int kWarpsPerRow = 8;
  constexpr int kWarpsPerBlock = 16;
  __shared__ float lhs_bias_shared[128];
  __shared__ float rhs_bias_shared[128];
  __shared__ float rowk_mask_shared[kRowsPerBlock][32];
  __shared__ float krow_mask_shared[kRowsPerBlock][32];
  __shared__ __nv_bfloat16 lhs_rowk0_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 rhs_rowk0_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 lhs_rowk1_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 rhs_rowk1_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 lhs_krow0_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 rhs_krow0_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 lhs_krow1_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 rhs_krow1_shared[kRowsPerBlock][kWarpsPerRow][32];

  const int tid = threadIdx.x;
  if (tid < 128) {
    lhs_bias_shared[tid] = __bfloat162float(lhs_bias[tid]);
    rhs_bias_shared[tid] = __bfloat162float(rhs_bias[tid]);
  }

  const int batch_channel_block = blockIdx.x;
  const int kblock = blockIdx.z;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int batch_idx = batch_channel_block / 4;
  const int row_in_block = warp_id / kWarpsPerRow;
  const int warp_in_row = warp_id % kWarpsPerRow;
  const int row = blockIdx.y * kRowsPerBlock + row_in_block;
  const int channel = ((batch_channel_block % 4) << 3) + warp_in_row;
  const int k = kblock * 32 + lane;
  if (warp_id < kWarpsPerBlock && batch_idx < batch && row < n && k < n) {
    const int64_t rowk_position = static_cast<int64_t>(batch_idx) * n * n + static_cast<int64_t>(row) * n + k;
    const int64_t krow_position = static_cast<int64_t>(batch_idx) * n * n + static_cast<int64_t>(k) * n + row;
    const int rowk_base = static_cast<int>(rowk_position * 128);
    const int krow_base = static_cast<int>(krow_position * 128);

    lhs_rowk0_shared[row_in_block][warp_in_row][lane] = lhs[rowk_base + channel];
    rhs_rowk0_shared[row_in_block][warp_in_row][lane] = rhs[rowk_base + channel];
    lhs_rowk1_shared[row_in_block][warp_in_row][lane] = lhs[rowk_base + 32 + channel];
    rhs_rowk1_shared[row_in_block][warp_in_row][lane] = rhs[rowk_base + 32 + channel];
    lhs_krow0_shared[row_in_block][warp_in_row][lane] = lhs[krow_base + 64 + channel];
    rhs_krow0_shared[row_in_block][warp_in_row][lane] = rhs[krow_base + 64 + channel];
    lhs_krow1_shared[row_in_block][warp_in_row][lane] = lhs[krow_base + 96 + channel];
    rhs_krow1_shared[row_in_block][warp_in_row][lane] = rhs[krow_base + 96 + channel];

    if (warp_in_row == 0) {
      rowk_mask_shared[row_in_block][lane] = (mask == nullptr || mask[(batch_idx * n + row) * n + k]) ? 1.0f : 0.0f;
      krow_mask_shared[row_in_block][lane] = (mask == nullptr || mask[(batch_idx * n + k) * n + row]) ? 1.0f : 0.0f;
    }
  }
  __syncthreads();
  if (batch_idx >= batch || row >= n || k >= n || warp_id >= kWarpsPerBlock) {
    return;
  }

  const float mask_rowk = rowk_mask_shared[row_in_block][lane];
  const float mask_krow = krow_mask_shared[row_in_block][lane];

  const float a1_lhs = __bfloat162float(lhs_rowk0_shared[row_in_block][warp_in_row][lane]) + lhs_bias_shared[channel];
  const float a1_rhs = __bfloat162float(rhs_rowk0_shared[row_in_block][warp_in_row][lane]) + rhs_bias_shared[channel];
  const float b1_lhs =
      __bfloat162float(lhs_rowk1_shared[row_in_block][warp_in_row][lane]) + lhs_bias_shared[32 + channel];
  const float b1_rhs =
      __bfloat162float(rhs_rowk1_shared[row_in_block][warp_in_row][lane]) + rhs_bias_shared[32 + channel];
  const float a2_lhs =
      __bfloat162float(lhs_krow0_shared[row_in_block][warp_in_row][lane]) + lhs_bias_shared[64 + channel];
  const float a2_rhs =
      __bfloat162float(rhs_krow0_shared[row_in_block][warp_in_row][lane]) + rhs_bias_shared[64 + channel];
  const float b2_lhs =
      __bfloat162float(lhs_krow1_shared[row_in_block][warp_in_row][lane]) + lhs_bias_shared[96 + channel];
  const float b2_rhs_dense =
      __bfloat162float(rhs_krow1_shared[row_in_block][warp_in_row][lane]) + rhs_bias_shared[96 + channel];

  const float a1_value = a1_lhs * (1.0f / (1.0f + __expf(-a1_rhs))) * mask_rowk;
  const float b1_value = b1_lhs * (1.0f / (1.0f + __expf(-b1_rhs))) * mask_rowk;
  const float a2_value = a2_lhs * (1.0f / (1.0f + __expf(-a2_rhs))) * mask_krow;
  const float b2_value = b2_lhs * (1.0f / (1.0f + __expf(-b2_rhs_dense))) * mask_krow;

  const float a1_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(a1_value)), 0));
  const float b1_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(b1_value)), 0));
  const float a2_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(a2_value)), 0));
  const float b2_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(b2_value)), 0));

  const int plane = batch_idx * 32 + channel;
  a1[((plane * n + row) * n) + k] = float_to_fp8_storage(a1_value / a1_block_scale);
  b1[((plane * n + row) * n) + k] = float_to_fp8_storage(b1_value / b1_block_scale);
  a2_t[((plane * n + row) * n) + k] = float_to_fp8_storage(a2_value / a2_block_scale);
  b2_rhs[((plane * n + k) * n) + row] = float_to_fp8_storage(b2_value / b2_block_scale);

  if (lane == 0) {
    const int plane_offset = plane * padded_rows * padded_cols;
    const int scale_offset = plane_offset + swizzled_scale_offset(row, kblock, padded_cols);
    a1_scale[scale_offset] = float_to_e8m0_storage(a1_block_scale);
    b1_scale[scale_offset] = float_to_e8m0_storage(b1_block_scale);
    a2_t_scale[scale_offset] = float_to_e8m0_storage(a2_block_scale);
    b2_rhs_scale[scale_offset] = float_to_e8m0_storage(b2_block_scale);
  }
}

__global__ void gate_sigmoid_mul_pack_krow_mxfp8_kernel(
    const __nv_bfloat16* lhs,
    const __nv_bfloat16* rhs,
    const __nv_bfloat16* lhs_bias,
    const __nv_bfloat16* rhs_bias,
    const bool* mask,
    __nv_fp8_storage_t* a2_t,
    __nv_fp8_storage_t* a2_t_scale,
    __nv_fp8_storage_t* b2_rhs,
    __nv_fp8_storage_t* b2_rhs_scale,
    int batch,
    int n,
    int padded_rows,
    int padded_cols) {
  constexpr int kRowsPerBlock = 2;
  constexpr int kWarpsPerRow = 4;
  constexpr int kWarpsPerBlock = 8;
  __shared__ float lhs_bias_shared[64];
  __shared__ float rhs_bias_shared[64];
  __shared__ float krow_mask_shared[kRowsPerBlock][32];
  __shared__ __nv_bfloat16 lhs_krow0_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 rhs_krow0_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 lhs_krow1_shared[kRowsPerBlock][kWarpsPerRow][32];
  __shared__ __nv_bfloat16 rhs_krow1_shared[kRowsPerBlock][kWarpsPerRow][32];

  const int tid = threadIdx.x;
  if (tid < 64) {
    lhs_bias_shared[tid] = __bfloat162float(lhs_bias[64 + tid]);
    rhs_bias_shared[tid] = __bfloat162float(rhs_bias[64 + tid]);
  }

  const int batch_channel_block = blockIdx.x;
  const int kblock = blockIdx.z;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int batch_idx = batch_channel_block / 8;
  const int row_in_block = warp_id / kWarpsPerRow;
  const int warp_in_row = warp_id % kWarpsPerRow;
  const int row = blockIdx.y * kRowsPerBlock + row_in_block;
  const int channel = ((batch_channel_block % 8) << 2) + warp_in_row;
  const int k = kblock * 32 + lane;
  if (warp_id < kWarpsPerBlock && batch_idx < batch && row < n && k < n) {
    const int64_t krow_position = static_cast<int64_t>(batch_idx) * n * n + static_cast<int64_t>(k) * n + row;
    const int krow_base = static_cast<int>(krow_position * 128);

    lhs_krow0_shared[row_in_block][warp_in_row][lane] = lhs[krow_base + 64 + channel];
    rhs_krow0_shared[row_in_block][warp_in_row][lane] = rhs[krow_base + 64 + channel];
    lhs_krow1_shared[row_in_block][warp_in_row][lane] = lhs[krow_base + 96 + channel];
    rhs_krow1_shared[row_in_block][warp_in_row][lane] = rhs[krow_base + 96 + channel];

    if (warp_in_row == 0) {
      krow_mask_shared[row_in_block][lane] = (mask == nullptr || mask[(batch_idx * n + k) * n + row]) ? 1.0f : 0.0f;
    }
  }
  __syncthreads();
  if (batch_idx >= batch || row >= n || k >= n || warp_id >= kWarpsPerBlock) {
    return;
  }

  const float mask_krow = krow_mask_shared[row_in_block][lane];
  const float a2_lhs = __bfloat162float(lhs_krow0_shared[row_in_block][warp_in_row][lane]) + lhs_bias_shared[channel];
  const float a2_rhs = __bfloat162float(rhs_krow0_shared[row_in_block][warp_in_row][lane]) + rhs_bias_shared[channel];
  const float b2_lhs =
      __bfloat162float(lhs_krow1_shared[row_in_block][warp_in_row][lane]) + lhs_bias_shared[32 + channel];
  const float b2_rhs_dense =
      __bfloat162float(rhs_krow1_shared[row_in_block][warp_in_row][lane]) + rhs_bias_shared[32 + channel];

  const float a2_value = a2_lhs * (1.0f / (1.0f + __expf(-a2_rhs))) * mask_krow;
  const float b2_value = b2_lhs * (1.0f / (1.0f + __expf(-b2_rhs_dense))) * mask_krow;

  const float a2_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(a2_value)), 0));
  const float b2_block_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(b2_value)), 0));

  const int plane = batch_idx * 32 + channel;
  a2_t[((plane * n + row) * n) + k] = float_to_fp8_storage(a2_value / a2_block_scale);
  b2_rhs[((plane * n + k) * n) + row] = float_to_fp8_storage(b2_value / b2_block_scale);

  if (lane == 0) {
    const int plane_offset = plane * padded_rows * padded_cols;
    const int scale_offset = plane_offset + swizzled_scale_offset(row, kblock, padded_cols);
    a2_t_scale[scale_offset] = float_to_e8m0_storage(a2_block_scale);
    b2_rhs_scale[scale_offset] = float_to_e8m0_storage(b2_block_scale);
  }
}

__global__ void tri_pair_to_block32_carrier_tiled_kernel(
    const __half* x1,
    const __half* x2,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int batch,
    int n,
    int d_chunk) {
  constexpr int kWarpsPerBlock = 8;
  constexpr int kTilePositions = 16;
  __shared__ float tile[kTilePositions][65];

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp_id >= kWarpsPerBlock) {
    return;
  }

  const int batch_row = blockIdx.x;
  const int batch_idx = batch_row / n;
  const int i = batch_row % n;
  const int j_base = blockIdx.y * kTilePositions;
  if (batch_idx >= batch || i >= n) {
    return;
  }

#pragma unroll
  for (int phase = 0; phase < 8; ++phase) {
    const int plane = phase * kWarpsPerBlock + warp_id;
    if (plane >= d_chunk * 2) {
      continue;
    }
    const __half* source = plane < d_chunk ? x1 : x2;
    const int source_plane = plane < d_chunk ? plane : plane - d_chunk;
    const int base_idx = (((batch_idx * d_chunk) + source_plane) * n + i) * n + j_base;
    if (lane < (kTilePositions / 2)) {
      const int j_offset = lane * 2;
      const int j = j_base + j_offset;
      if (j + 1 < n) {
        const __half2 pair = *reinterpret_cast<const __half2*>(source + base_idx + j_offset);
        tile[j_offset][plane] = __half2float(__low2half(pair));
        tile[j_offset + 1][plane] = __half2float(__high2half(pair));
      } else if (j < n) {
        tile[j_offset][plane] = __half2float(source[base_idx + j_offset]);
        tile[j_offset + 1][plane] = 0.0f;
      } else {
        tile[j_offset][plane] = 0.0f;
        tile[j_offset + 1][plane] = 0.0f;
      }
    }
  }
  __syncthreads();

  for (int position_iter = 0; position_iter < 2; ++position_iter) {
    const int j_offset = warp_id + position_iter * kWarpsPerBlock;
    const int j = j_base + j_offset;
    if (j >= n) {
      continue;
    }
#pragma unroll
    for (int group = 0; group < 2; ++group) {
      const int col = group * 32 + lane;
      const float value = tile[j_offset][col];
      const float scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(value)), 0));
      const int out_idx = (((batch_idx * n + i) * n + j) * (d_chunk * 2)) + col;
      output[out_idx] = float_to_fp8_storage(value / scale);
      if (lane == 0) {
        output_scale[(((batch_idx * n + i) * n + j) * 2) + group] = scale;
      }
    }
  }
}

__global__ void tri_pair_layernorm_to_block32_half_kernel(
    const __half* x1,
    const __half* x2,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int batch,
    int n,
    float eps) {
  __shared__ float shared_sum[2][2];
  __shared__ float shared_sumsq[2][2];

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int position_in_block = warp_id >> 1;
  const int group = warp_id & 1;
  const int position = blockIdx.x * 2 + position_in_block;
  if (warp_id >= 4 || position >= batch * n * n) {
    return;
  }

  const int batch_idx = position / (n * n);
  const int rem = position % (n * n);
  const int i = rem / n;
  const int j = rem % n;
  const int col = group * 32 + lane;
  const float x = group == 0
      ? __half2float(x1[(((batch_idx * 32) + lane) * n + i) * n + j])
      : __half2float(x2[(((batch_idx * 32) + lane) * n + i) * n + j]);

  const float warp_sum = warp_reduce_sum(x);
  const float warp_sumsq = warp_reduce_sum(x * x);
  if (lane == 0) {
    shared_sum[position_in_block][group] = warp_sum;
    shared_sumsq[position_in_block][group] = warp_sumsq;
  }
  __syncthreads();

  const float mean = (shared_sum[position_in_block][0] + shared_sum[position_in_block][1]) / 64.0f;
  const float var = fmaxf((shared_sumsq[position_in_block][0] + shared_sumsq[position_in_block][1]) / 64.0f - mean * mean, 0.0f);
  const float inv_std = rsqrtf(var + eps);
  const float y = (x - mean) * inv_std * __bfloat162float(weight[col]) + __bfloat162float(bias[col]);
  const float group_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, warp_reduce_max(fabsf(y)), 0));
  const int out_idx = (((batch_idx * n + i) * n + j) * 64) + col;
  output[out_idx] = float_to_fp8_storage(y / group_scale);
  if (lane == 0) {
    output_scale[(((batch_idx * n + i) * n + j) * 2) + group] = group_scale;
  }
}

__global__ void tri_direct_block32_kernel(
    const __nv_fp8_storage_t* payload,
    const float* scale,
    const bool* mask,
    __nv_fp8_storage_t* output,
    float* output_scale,
    int total_positions,
    int n) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int pair = warp >> 1;
  const int group = warp & 1;
  const int position = blockIdx.x * 2 + pair;
  if (position >= total_positions) {
    return;
  }

  const int batch_idx = position / (n * n);
  const int rem = position % (n * n);
  const int i = rem / n;
  const int j = rem % n;

  float accum = 0.0f;
  if (group == 0) {
    for (int k = 0; k < n; ++k) {
      const bool mask_a = mask == nullptr ? true : mask[(batch_idx * n + i) * n + k];
      const bool mask_b = mask == nullptr ? true : mask[(batch_idx * n + j) * n + k];
      if (!(mask_a && mask_b)) {
        continue;
      }
      const int lhs_idx = (((batch_idx * n + i) * n + k) * 128) + lane;
      const int lhs_scale_idx = (((batch_idx * n + i) * n + k) * 4);
      const int rhs_idx = (((batch_idx * n + j) * n + k) * 128) + 32 + lane;
      const int rhs_scale_idx = (((batch_idx * n + j) * n + k) * 4) + 1;
      const float lhs = fp8_storage_to_float(payload[lhs_idx]) * scale[lhs_scale_idx];
      const float rhs = fp8_storage_to_float(payload[rhs_idx]) * scale[rhs_scale_idx];
      accum = fmaf(lhs, rhs, accum);
    }
  } else {
    for (int k = 0; k < n; ++k) {
      const bool mask_a = mask == nullptr ? true : mask[(batch_idx * n + k) * n + i];
      const bool mask_b = mask == nullptr ? true : mask[(batch_idx * n + k) * n + j];
      if (!(mask_a && mask_b)) {
        continue;
      }
      const int lhs_idx = (((batch_idx * n + k) * n + i) * 128) + 64 + lane;
      const int lhs_scale_idx = (((batch_idx * n + k) * n + i) * 4) + 2;
      const int rhs_idx = (((batch_idx * n + k) * n + j) * 128) + 96 + lane;
      const int rhs_scale_idx = (((batch_idx * n + k) * n + j) * 4) + 3;
      const float lhs = fp8_storage_to_float(payload[lhs_idx]) * scale[lhs_scale_idx];
      const float rhs = fp8_storage_to_float(payload[rhs_idx]) * scale[rhs_scale_idx];
      accum = fmaf(lhs, rhs, accum);
    }
  }

  const float max_abs = warp_reduce_max(fabsf(accum));
  const float out_group_scale = pow2_scale_from_max(__shfl_sync(0xffffffff, max_abs, 0));
  const int out_idx = (((batch_idx * n + i) * n + j) * 64) + group * 32 + lane;
  output[out_idx] = float_to_fp8_storage(accum / out_group_scale);
  if (lane == 0) {
    output_scale[(((batch_idx * n + i) * n + j) * 2) + group] = out_group_scale;
  }
}

std::tuple<at::Tensor, at::Tensor> quantize_block32_bf16_with_bias(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& residual_payload_opt = c10::nullopt,
    const c10::optional<at::Tensor>& residual_scale_opt = c10::nullopt,
    bool apply_relu = false) {
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::BFloat16, "quantize_block32_bf16_with_bias expects bfloat16 input");
  TORCH_CHECK(input.dim() == 3, "quantize_block32_bf16_with_bias expects a 3D tensor");
  const int64_t batch = input.size(0);
  const int64_t rows_per_batch = input.size(1);
  const int64_t cols = input.size(2);
  TORCH_CHECK(cols % 32 == 0, "block32 output quantization requires cols divisible by 32, got ", cols);
  TORCH_CHECK(cols <= 1024, "block32 output quantization currently supports cols <= 1024, got ", cols);
  const int64_t rows = batch * rows_per_batch;
  const int64_t groups = cols / 32;

  at::Tensor bias;
  const __nv_bfloat16* bias_ptr = nullptr;
  if (bias_opt.has_value()) {
    bias = bias_opt.value();
    check_cuda_tensor(bias, "bias");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == cols, "bias must be 1D with shape [", cols, "]");
    TORCH_CHECK(bias.scalar_type() == c10::ScalarType::BFloat16, "bias must be bfloat16");
    bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
  }

  at::Tensor residual_payload;
  at::Tensor residual_scale;
  const __nv_fp8_storage_t* residual_payload_ptr = nullptr;
  const float* residual_scale_ptr = nullptr;
  if (residual_payload_opt.has_value() || residual_scale_opt.has_value()) {
    TORCH_CHECK(residual_payload_opt.has_value() && residual_scale_opt.has_value(),
                "residual_payload and residual_scale must be provided together");
    residual_payload = residual_payload_opt.value();
    residual_scale = residual_scale_opt.value();
    check_cuda_tensor(residual_payload, "residual_payload");
    check_cuda_tensor(residual_scale, "residual_scale");
    TORCH_CHECK(residual_payload.scalar_type() == c10::ScalarType::Float8_e4m3fn,
                "residual_payload must be float8_e4m3fn");
    TORCH_CHECK(residual_scale.scalar_type() == c10::ScalarType::Float, "residual_scale must be float32");
    TORCH_CHECK(
        residual_payload.dim() == 3 && residual_payload.size(0) == batch && residual_payload.size(1) == rows_per_batch &&
            residual_payload.size(2) == cols,
        "residual_payload must have shape [", batch, ", ", rows_per_batch, ", ", cols, "]");
    TORCH_CHECK(
        residual_scale.dim() == 3 && residual_scale.size(0) == batch && residual_scale.size(1) == rows_per_batch &&
            residual_scale.size(2) == groups,
        "residual_scale must have shape [", batch, ", ", rows_per_batch, ", ", groups, "]");
    residual_payload_ptr = reinterpret_cast<const __nv_fp8_storage_t*>(residual_payload.data_ptr());
    residual_scale_ptr = residual_scale.data_ptr<float>();
  }

  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));
  auto output_scale = at::empty({batch, rows_per_batch, groups}, input.options().dtype(at::kFloat));
  auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  auto* output_ptr = reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr());
  auto* scale_ptr = output_scale.data_ptr<float>();
  auto stream = current_cuda_stream();
  if (cols == 128) {
    const int quantize_blocks = static_cast<int>((rows + 7) / 8);
    if (apply_relu) {
      if (residual_payload_ptr != nullptr) {
        quantize_block32_bf16_specialized_kernel<128, true, true, true><<<quantize_blocks, 256, 0, stream>>>(
            input_ptr, bias_ptr, residual_payload_ptr, residual_scale_ptr, output_ptr, scale_ptr, static_cast<int>(rows));
      } else {
        quantize_block32_bf16_specialized_kernel<128, true, true, false><<<quantize_blocks, 256, 0, stream>>>(
            input_ptr, bias_ptr, nullptr, nullptr, output_ptr, scale_ptr, static_cast<int>(rows));
      }
    } else {
      if (residual_payload_ptr != nullptr) {
        quantize_block32_bf16_specialized_kernel<128, true, false, true><<<quantize_blocks, 256, 0, stream>>>(
            input_ptr, bias_ptr, residual_payload_ptr, residual_scale_ptr, output_ptr, scale_ptr, static_cast<int>(rows));
      } else {
        quantize_block32_bf16_specialized_kernel<128, true, false, false><<<quantize_blocks, 256, 0, stream>>>(
            input_ptr, bias_ptr, nullptr, nullptr, output_ptr, scale_ptr, static_cast<int>(rows));
      }
    }
  } else if (cols == 512) {
    const int quantize_blocks = static_cast<int>((rows + 1) / 2);
    if (apply_relu) {
      if (residual_payload_ptr != nullptr) {
        quantize_block32_bf16_specialized_kernel<512, true, true, true><<<quantize_blocks, 256, 0, stream>>>(
            input_ptr, bias_ptr, residual_payload_ptr, residual_scale_ptr, output_ptr, scale_ptr, static_cast<int>(rows));
      } else {
        quantize_block32_bf16_specialized_kernel<512, true, true, false><<<quantize_blocks, 256, 0, stream>>>(
            input_ptr, bias_ptr, nullptr, nullptr, output_ptr, scale_ptr, static_cast<int>(rows));
      }
    } else {
      if (residual_payload_ptr != nullptr) {
        quantize_block32_bf16_specialized_kernel<512, true, false, true><<<quantize_blocks, 256, 0, stream>>>(
            input_ptr, bias_ptr, residual_payload_ptr, residual_scale_ptr, output_ptr, scale_ptr, static_cast<int>(rows));
      } else {
        quantize_block32_bf16_specialized_kernel<512, true, false, false><<<quantize_blocks, 256, 0, stream>>>(
            input_ptr, bias_ptr, nullptr, nullptr, output_ptr, scale_ptr, static_cast<int>(rows));
      }
    }
  } else {
    quantize_block32_bf16_bias_generic_kernel<<<rows, cols, 0, stream>>>(
        input_ptr,
        bias_ptr,
        residual_payload_ptr,
        residual_scale_ptr,
        output_ptr,
        scale_ptr,
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(groups),
        apply_relu);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, output_scale};
}

std::tuple<at::Tensor, at::Tensor> quantize_block32_bf16(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& residual_payload_opt = c10::nullopt,
    const c10::optional<at::Tensor>& residual_scale_opt = c10::nullopt) {
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::BFloat16, "quantize_block32_bf16 expects bfloat16 input");
  TORCH_CHECK(input.dim() == 3, "quantize_block32_bf16 expects a 3D tensor");
  const int64_t batch = input.size(0);
  const int64_t rows_per_batch = input.size(1);
  const int64_t cols = input.size(2);
  TORCH_CHECK(cols % 32 == 0, "block32 output quantization requires cols divisible by 32, got ", cols);
  TORCH_CHECK(cols <= 1024, "block32 output quantization currently supports cols <= 1024, got ", cols);
  const int64_t rows = batch * rows_per_batch;
  const int64_t groups = cols / 32;

  at::Tensor residual_payload;
  at::Tensor residual_scale;
  const __nv_fp8_storage_t* residual_payload_ptr = nullptr;
  const float* residual_scale_ptr = nullptr;
  if (residual_payload_opt.has_value() || residual_scale_opt.has_value()) {
    TORCH_CHECK(residual_payload_opt.has_value() && residual_scale_opt.has_value(),
                "residual_payload and residual_scale must be provided together");
    residual_payload = residual_payload_opt.value();
    residual_scale = residual_scale_opt.value();
    check_cuda_tensor(residual_payload, "residual_payload");
    check_cuda_tensor(residual_scale, "residual_scale");
    TORCH_CHECK(residual_payload.scalar_type() == c10::ScalarType::Float8_e4m3fn,
                "residual_payload must be float8_e4m3fn");
    TORCH_CHECK(residual_scale.scalar_type() == c10::ScalarType::Float, "residual_scale must be float32");
    TORCH_CHECK(
        residual_payload.dim() == 3 && residual_payload.size(0) == batch && residual_payload.size(1) == rows_per_batch &&
            residual_payload.size(2) == cols,
        "residual_payload must have shape [", batch, ", ", rows_per_batch, ", ", cols, "]");
    TORCH_CHECK(
        residual_scale.dim() == 3 && residual_scale.size(0) == batch && residual_scale.size(1) == rows_per_batch &&
            residual_scale.size(2) == groups,
        "residual_scale must have shape [", batch, ", ", rows_per_batch, ", ", groups, "]");
    residual_payload_ptr = reinterpret_cast<const __nv_fp8_storage_t*>(residual_payload.data_ptr());
    residual_scale_ptr = residual_scale.data_ptr<float>();
  }

  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));
  auto output_scale = at::empty({batch, rows_per_batch, groups}, input.options().dtype(at::kFloat));
  auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  auto* output_ptr = reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr());
  auto* scale_ptr = output_scale.data_ptr<float>();
  auto stream = current_cuda_stream();
  if (cols == 128) {
    const int quantize_blocks = static_cast<int>((rows + 7) / 8);
    if (residual_payload_ptr != nullptr) {
      quantize_block32_bf16_specialized_kernel<128, false, false, true><<<quantize_blocks, 256, 0, stream>>>(
          input_ptr, nullptr, residual_payload_ptr, residual_scale_ptr, output_ptr, scale_ptr, static_cast<int>(rows));
    } else {
      quantize_block32_bf16_specialized_kernel<128, false, false, false><<<quantize_blocks, 256, 0, stream>>>(
          input_ptr, nullptr, nullptr, nullptr, output_ptr, scale_ptr, static_cast<int>(rows));
    }
  } else if (cols == 512) {
    const int quantize_blocks = static_cast<int>((rows + 1) / 2);
    if (residual_payload_ptr != nullptr) {
      quantize_block32_bf16_specialized_kernel<512, false, false, true><<<quantize_blocks, 256, 0, stream>>>(
          input_ptr, nullptr, residual_payload_ptr, residual_scale_ptr, output_ptr, scale_ptr, static_cast<int>(rows));
    } else {
      quantize_block32_bf16_specialized_kernel<512, false, false, false><<<quantize_blocks, 256, 0, stream>>>(
          input_ptr, nullptr, nullptr, nullptr, output_ptr, scale_ptr, static_cast<int>(rows));
    }
  } else {
    quantize_block32_bf16_generic_kernel<<<rows, cols, 0, stream>>>(
        input_ptr,
        residual_payload_ptr,
        residual_scale_ptr,
        output_ptr,
        scale_ptr,
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(groups));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, output_scale};
}

std::tuple<at::Tensor, at::Tensor> gate_sigmoid_mul_quantize_bf16(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const c10::optional<at::Tensor>& lhs_bias_opt,
    const c10::optional<at::Tensor>& rhs_bias_opt,
    const c10::optional<at::Tensor>& residual_payload_opt = c10::nullopt,
    const c10::optional<at::Tensor>& residual_scale_opt = c10::nullopt) {
  TORCH_CHECK(lhs.scalar_type() == c10::ScalarType::BFloat16 && rhs.scalar_type() == c10::ScalarType::BFloat16,
              "gate_sigmoid_mul_quantize_bf16 expects bfloat16 inputs");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "lhs and rhs gate inputs must have the same shape");
  TORCH_CHECK(lhs.dim() == 3, "gate_sigmoid_mul_quantize_bf16 expects 3D tensors");
  const int64_t batch = lhs.size(0);
  const int64_t rows_per_batch = lhs.size(1);
  const int64_t cols = lhs.size(2);
  TORCH_CHECK(cols % 32 == 0 && cols <= 128, "gate_sigmoid_mul_quantize_bf16 expects cols divisible by 32 and <= 128, got ", cols);
  const int64_t rows = batch * rows_per_batch;
  const int64_t groups = cols / 32;

  at::Tensor lhs_bias;
  at::Tensor rhs_bias;
  const __nv_bfloat16* lhs_bias_ptr = nullptr;
  const __nv_bfloat16* rhs_bias_ptr = nullptr;
  if (lhs_bias_opt.has_value()) {
    lhs_bias = lhs_bias_opt.value();
    check_cuda_tensor(lhs_bias, "lhs_bias");
    TORCH_CHECK(lhs_bias.dim() == 1 && lhs_bias.size(0) == cols, "lhs_bias must be 1D with output width");
    lhs_bias_ptr = reinterpret_cast<const __nv_bfloat16*>(lhs_bias.data_ptr());
  }
  if (rhs_bias_opt.has_value()) {
    rhs_bias = rhs_bias_opt.value();
    check_cuda_tensor(rhs_bias, "rhs_bias");
    TORCH_CHECK(rhs_bias.dim() == 1 && rhs_bias.size(0) == cols, "rhs_bias must be 1D with output width");
    rhs_bias_ptr = reinterpret_cast<const __nv_bfloat16*>(rhs_bias.data_ptr());
  }

  at::Tensor residual_payload;
  at::Tensor residual_scale;
  const __nv_fp8_storage_t* residual_payload_ptr = nullptr;
  const float* residual_scale_ptr = nullptr;
  if (residual_payload_opt.has_value() || residual_scale_opt.has_value()) {
    TORCH_CHECK(residual_payload_opt.has_value() && residual_scale_opt.has_value(),
                "residual_payload and residual_scale must be provided together");
    residual_payload = residual_payload_opt.value();
    residual_scale = residual_scale_opt.value();
    check_cuda_tensor(residual_payload, "residual_payload");
    check_cuda_tensor(residual_scale, "residual_scale");
    TORCH_CHECK(residual_payload.scalar_type() == c10::ScalarType::Float8_e4m3fn,
                "residual_payload must be float8_e4m3fn");
    TORCH_CHECK(residual_scale.scalar_type() == c10::ScalarType::Float, "residual_scale must be float32");
    TORCH_CHECK(
        residual_payload.dim() == 3 && residual_payload.size(0) == batch && residual_payload.size(1) == rows_per_batch &&
            residual_payload.size(2) == cols,
        "residual_payload must have shape [", batch, ", ", rows_per_batch, ", ", cols, "]");
    TORCH_CHECK(
        residual_scale.dim() == 3 && residual_scale.size(0) == batch && residual_scale.size(1) == rows_per_batch &&
            residual_scale.size(2) == groups,
        "residual_scale must have shape [", batch, ", ", rows_per_batch, ", ", groups, "]");
    residual_payload_ptr = reinterpret_cast<const __nv_fp8_storage_t*>(residual_payload.data_ptr());
    residual_scale_ptr = residual_scale.data_ptr<float>();
  }

  auto output = at::empty(lhs.sizes(), lhs.options().dtype(at::kFloat8_e4m3fn));
  auto output_scale = at::empty({batch, rows_per_batch, groups}, lhs.options().dtype(at::kFloat));
  if (cols == 128 && lhs_bias_ptr != nullptr && rhs_bias_ptr != nullptr) {
    const int gate_blocks = static_cast<int>((rows + 7) / 8);
    if (residual_payload_ptr != nullptr) {
      gate_sigmoid_mul_quantize_bf16_128_kernel<true><<<gate_blocks, 256, 0, current_cuda_stream()>>>(
          reinterpret_cast<const __nv_bfloat16*>(lhs.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(rhs.data_ptr()),
          lhs_bias_ptr,
          rhs_bias_ptr,
          residual_payload_ptr,
          residual_scale_ptr,
          reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
          output_scale.data_ptr<float>(),
          static_cast<int>(rows));
    } else {
      gate_sigmoid_mul_quantize_bf16_128_kernel<false><<<gate_blocks, 256, 0, current_cuda_stream()>>>(
          reinterpret_cast<const __nv_bfloat16*>(lhs.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(rhs.data_ptr()),
          lhs_bias_ptr,
          rhs_bias_ptr,
          nullptr,
          nullptr,
          reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
          output_scale.data_ptr<float>(),
          static_cast<int>(rows));
    }
  } else {
    gate_sigmoid_mul_quantize_bf16_kernel<<<rows, cols, 0, current_cuda_stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(lhs.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(rhs.data_ptr()),
        lhs_bias_ptr,
        rhs_bias_ptr,
        residual_payload_ptr,
        residual_scale_ptr,
        reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
        output_scale.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(groups));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, output_scale};
}

std::tuple<at::Tensor, at::Tensor> unary_block32_impl(
    const at::Tensor& payload,
    const at::Tensor& scale,
    int op_code) {
  check_cuda_tensor(payload, "payload");
  check_cuda_tensor(scale, "scale");
  TORCH_CHECK(payload.scalar_type() == c10::ScalarType::Float8_e4m3fn, "payload must use float8_e4m3fn");
  TORCH_CHECK(scale.scalar_type() == c10::ScalarType::Float, "scale must use float32");
  const int64_t cols = payload.size(-1);
  const int64_t rows = payload.numel() / cols;
  const int64_t groups = cols / 32;
  TORCH_CHECK(cols % 32 == 0 && cols <= 512, "unary_block32 expects cols divisible by 32 and <= 512, got ", cols);
  TORCH_CHECK(scale.numel() == rows * groups, "scale shape is incompatible with payload");
  auto output = at::empty_like(payload);
  auto output_scale = at::empty_like(scale);
  unary_block32_kernel<<<rows, cols, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __nv_fp8_storage_t*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
      output_scale.data_ptr<float>(),
      static_cast<int>(rows),
      static_cast<int>(cols),
      static_cast<int>(groups),
      op_code);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, output_scale};
}

std::tuple<at::Tensor, at::Tensor> binary_block32_impl(
    const at::Tensor& lhs_payload,
    const at::Tensor& lhs_scale,
    const at::Tensor& rhs_payload,
    const at::Tensor& rhs_scale,
    int op_code) {
  check_cuda_tensor(lhs_payload, "lhs_payload");
  check_cuda_tensor(lhs_scale, "lhs_scale");
  check_cuda_tensor(rhs_payload, "rhs_payload");
  check_cuda_tensor(rhs_scale, "rhs_scale");
  TORCH_CHECK(lhs_payload.sizes() == rhs_payload.sizes(), "lhs_payload and rhs_payload must match");
  TORCH_CHECK(lhs_scale.sizes() == rhs_scale.sizes(), "lhs_scale and rhs_scale must match");
  const int64_t cols = lhs_payload.size(-1);
  const int64_t rows = lhs_payload.numel() / cols;
  const int64_t groups = cols / 32;
  TORCH_CHECK(cols % 32 == 0 && cols <= 512, "binary_block32 expects cols divisible by 32 and <= 512, got ", cols);
  auto output = at::empty_like(lhs_payload);
  auto output_scale = at::empty_like(lhs_scale);
  binary_block32_kernel<<<rows, cols, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __nv_fp8_storage_t*>(lhs_payload.data_ptr()),
      lhs_scale.data_ptr<float>(),
      reinterpret_cast<const __nv_fp8_storage_t*>(rhs_payload.data_ptr()),
      rhs_scale.data_ptr<float>(),
      reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
      output_scale.data_ptr<float>(),
      static_cast<int>(rows),
      static_cast<int>(cols),
      static_cast<int>(groups),
      op_code);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, output_scale};
}

std::tuple<at::Tensor, at::Tensor> layernorm_block32_impl(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& weight,
    const at::Tensor& bias,
    float eps) {
  check_cuda_tensor(payload, "payload");
  check_cuda_tensor(scale, "scale");
  check_cuda_tensor(weight, "weight");
  check_cuda_tensor(bias, "bias");
  TORCH_CHECK(payload.scalar_type() == c10::ScalarType::Float8_e4m3fn, "payload must use float8_e4m3fn");
  TORCH_CHECK(scale.scalar_type() == c10::ScalarType::Float, "scale must use float32");
  const int64_t cols = payload.size(-1);
  const int64_t rows = payload.numel() / cols;
  const int64_t groups = cols / 32;
  TORCH_CHECK(cols % 32 == 0 && cols <= 512, "layernorm_block32 expects cols divisible by 32 and <= 512, got ", cols);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == cols, "weight shape must match cols");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == cols, "bias shape must match cols");
  TORCH_CHECK(weight.scalar_type() == c10::ScalarType::BFloat16 && bias.scalar_type() == c10::ScalarType::BFloat16,
              "weight and bias must be bfloat16");
  auto output = at::empty_like(payload);
  auto output_scale = at::empty_like(scale);
  if (cols == 64) {
    const int layernorm_blocks_small = static_cast<int>((rows + 7) / 8);
    layernorm_block32_kernel_small<64><<<layernorm_blocks_small, 256, 0, current_cuda_stream()>>>(
        reinterpret_cast<const __nv_fp8_storage_t*>(payload.data_ptr()),
        scale.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
        output_scale.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(groups),
        eps);
  } else if (cols == 128) {
    const int layernorm_blocks_small = static_cast<int>((rows + 7) / 8);
    layernorm_block32_kernel_small<128><<<layernorm_blocks_small, 256, 0, current_cuda_stream()>>>(
        reinterpret_cast<const __nv_fp8_storage_t*>(payload.data_ptr()),
        scale.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
        output_scale.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(groups),
        eps);
  } else {
    const size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
    layernorm_block32_generic_kernel<<<rows, cols, shared_bytes, current_cuda_stream()>>>(
        reinterpret_cast<const __nv_fp8_storage_t*>(payload.data_ptr()),
        scale.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
        output_scale.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(groups),
        eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, output_scale};
}

std::tuple<at::Tensor, at::Tensor> pack_block32_to_mxfp8_lhs(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& mask_opt,
    int channel_group,
    bool transpose) {
  const int64_t batch = payload.size(0);
  const int64_t n = payload.size(1);
  auto output = at::empty({batch * 32, n, n}, payload.options());
  auto output_scale_fp32 = at::empty({batch * 32, n, n / 32}, scale.options().dtype(at::kFloat));
  const bool* mask_ptr = nullptr;
  at::Tensor mask;
  if (mask_opt.has_value()) {
    mask = mask_opt.value();
    check_cuda_tensor(mask, "mask");
    TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool, "mask must be bool");
    TORCH_CHECK(
        mask.dim() == 3 && mask.size(0) == batch && mask.size(1) == n && mask.size(2) == n,
        "mask shape must match [B, N, N]");
    mask_ptr = mask.data_ptr<bool>();
  }
  const dim3 grid(static_cast<unsigned int>(batch * 32), static_cast<unsigned int>(n), static_cast<unsigned int>(n / 32));
  pack_block32_to_mxfp8_lhs_mask_kernel<<<grid, 32, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __nv_fp8_storage_t*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      mask_ptr,
      reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
      output_scale_fp32.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(n),
      channel_group,
      transpose);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto output_scale = swizzle_scale_rowwise(output_scale_fp32.to(at::kFloat8_e8m0fnu));
  return {output, output_scale};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> pack_block32_to_mxfp8_lhs_pair(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& mask_opt) {
  const int64_t batch = payload.size(0);
  const int64_t n = payload.size(1);
  auto output0 = at::empty({batch * 32, n, n}, payload.options());
  auto output1 = at::empty({batch * 32, n, n}, payload.options());
  auto output0_scale_fp32 = at::empty({batch * 32, n, n / 32}, scale.options().dtype(at::kFloat));
  auto output1_scale_fp32 = at::empty({batch * 32, n, n / 32}, scale.options().dtype(at::kFloat));
  const bool* mask_ptr = nullptr;
  at::Tensor mask;
  if (mask_opt.has_value()) {
    mask = mask_opt.value();
    check_cuda_tensor(mask, "mask");
    TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool, "mask must be bool");
    TORCH_CHECK(
        mask.dim() == 3 && mask.size(0) == batch && mask.size(1) == n && mask.size(2) == n,
        "mask shape must match [B, N, N]");
    mask_ptr = mask.data_ptr<bool>();
  }
  const dim3 grid(static_cast<unsigned int>(batch * 32), static_cast<unsigned int>(n), static_cast<unsigned int>(n / 32));
  pack_block32_to_mxfp8_lhs_pair_mask_kernel<<<grid, 32, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __nv_fp8_storage_t*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      mask_ptr,
      reinterpret_cast<__nv_fp8_storage_t*>(output0.data_ptr()),
      output0_scale_fp32.data_ptr<float>(),
      reinterpret_cast<__nv_fp8_storage_t*>(output1.data_ptr()),
      output1_scale_fp32.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto output0_scale = swizzle_scale_rowwise(output0_scale_fp32.to(at::kFloat8_e8m0fnu));
  auto output1_scale = swizzle_scale_rowwise(output1_scale_fp32.to(at::kFloat8_e8m0fnu));
  return {output0, output0_scale, output1, output1_scale};
}

std::tuple<at::Tensor, at::Tensor> pack_block32_to_mxfp8_rhs(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& mask_opt,
    int channel_group) {
  const int64_t batch = payload.size(0);
  const int64_t n = payload.size(1);
  auto output = at::empty({batch * 32, n, n}, payload.options());
  auto output_scale_fp32 = at::empty({batch * 32, n / 32, n}, scale.options().dtype(at::kFloat));
  const bool* mask_ptr = nullptr;
  at::Tensor mask;
  if (mask_opt.has_value()) {
    mask = mask_opt.value();
    check_cuda_tensor(mask, "mask");
    TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool, "mask must be bool");
    TORCH_CHECK(
        mask.dim() == 3 && mask.size(0) == batch && mask.size(1) == n && mask.size(2) == n,
        "mask shape must match [B, N, N]");
    mask_ptr = mask.data_ptr<bool>();
  }
  const dim3 grid(static_cast<unsigned int>(batch * 32), static_cast<unsigned int>(n / 32), static_cast<unsigned int>(n));
  pack_block32_to_mxfp8_rhs_mask_kernel<<<grid, 32, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __nv_fp8_storage_t*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      mask_ptr,
      reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
      output_scale_fp32.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(n),
      channel_group);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto output_scale = swizzle_scale_rowwise(output_scale_fp32.to(at::kFloat8_e8m0fnu).transpose(1, 2).contiguous());
  return {output, output_scale};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
pack_block32_to_mxfp8_fused(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& mask_opt) {
  const int64_t batch = payload.size(0);
  const int64_t n = payload.size(1);
  const int64_t padded_rows = ((n + 127) / 128) * 128;
  const int64_t padded_cols = (((n / 32) + 3) / 4) * 4;
  auto a1 = at::empty({batch * 32, n, n}, payload.options());
  auto b1 = at::empty({batch * 32, n, n}, payload.options());
  auto a2_t = at::empty({batch * 32, n, n}, payload.options());
  auto b2_rhs = at::empty({batch * 32, n, n}, payload.options());
  auto a1_scale = at::full({batch * 32, padded_rows, padded_cols}, kMinPow2Scale, scale.options().dtype(at::kFloat8_e8m0fnu));
  auto b1_scale = at::full({batch * 32, padded_rows, padded_cols}, kMinPow2Scale, scale.options().dtype(at::kFloat8_e8m0fnu));
  auto a2_t_scale = at::full({batch * 32, padded_rows, padded_cols}, kMinPow2Scale, scale.options().dtype(at::kFloat8_e8m0fnu));
  auto b2_rhs_scale = at::full({batch * 32, padded_rows, padded_cols}, kMinPow2Scale, scale.options().dtype(at::kFloat8_e8m0fnu));
  const bool* mask_ptr = nullptr;
  at::Tensor mask;
  if (mask_opt.has_value()) {
    mask = mask_opt.value();
    check_cuda_tensor(mask, "mask");
    TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool, "mask must be bool");
    TORCH_CHECK(
        mask.dim() == 3 && mask.size(0) == batch && mask.size(1) == n && mask.size(2) == n,
        "mask shape must match [B, N, N]");
    mask_ptr = mask.data_ptr<bool>();
  }
  const dim3 grid(static_cast<unsigned int>(batch * 4), static_cast<unsigned int>((n + 1) / 2), static_cast<unsigned int>(n / 32));
  pack_block32_to_mxfp8_fused_kernel<<<grid, 512, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __nv_fp8_storage_t*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      mask_ptr,
      reinterpret_cast<__nv_fp8_storage_t*>(a1.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(a1_scale.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(b1.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(b1_scale.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(a2_t.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(a2_t_scale.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(b2_rhs.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(b2_rhs_scale.data_ptr()),
      static_cast<int>(batch),
      static_cast<int>(n),
      static_cast<int>(padded_rows),
      static_cast<int>(padded_cols));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a1, a1_scale, b1, b1_scale, a2_t, a2_t_scale, b2_rhs, b2_rhs_scale};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
gate_sigmoid_mul_pack_to_mxfp8_impl(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const c10::optional<at::Tensor>& lhs_bias_opt,
    const c10::optional<at::Tensor>& rhs_bias_opt,
    const at::Tensor& mask,
    bool use_warp_kernel) {
  TORCH_CHECK(lhs.scalar_type() == c10::ScalarType::BFloat16 && rhs.scalar_type() == c10::ScalarType::BFloat16,
              "gate_sigmoid_mul_pack_to_mxfp8_fused expects bfloat16 inputs");
  TORCH_CHECK(lhs.sizes() == rhs.sizes(), "lhs and rhs must match");
  TORCH_CHECK(lhs.dim() == 3 && lhs.size(0) == 1, "gate_sigmoid_mul_pack_to_mxfp8_fused expects lhs/rhs with shape [1, rows, 128]");
  TORCH_CHECK(lhs.size(2) == 128, "gate_sigmoid_mul_pack_to_mxfp8_fused expects width 128");
  check_cuda_tensor(mask, "mask");
  TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool, "mask must be bool");
  TORCH_CHECK(mask.dim() == 3 && mask.size(1) == mask.size(2), "mask must have shape [B, N, N]");

  at::Tensor lhs_bias;
  at::Tensor rhs_bias;
  const __nv_bfloat16* lhs_bias_ptr = nullptr;
  const __nv_bfloat16* rhs_bias_ptr = nullptr;
  if (lhs_bias_opt.has_value()) {
    lhs_bias = lhs_bias_opt.value();
    check_cuda_tensor(lhs_bias, "lhs_bias");
    TORCH_CHECK(lhs_bias.dim() == 1 && lhs_bias.size(0) == 128, "lhs_bias must be 1D with width 128");
    lhs_bias_ptr = reinterpret_cast<const __nv_bfloat16*>(lhs_bias.data_ptr());
  }
  if (rhs_bias_opt.has_value()) {
    rhs_bias = rhs_bias_opt.value();
    check_cuda_tensor(rhs_bias, "rhs_bias");
    TORCH_CHECK(rhs_bias.dim() == 1 && rhs_bias.size(0) == 128, "rhs_bias must be 1D with width 128");
    rhs_bias_ptr = reinterpret_cast<const __nv_bfloat16*>(rhs_bias.data_ptr());
  }
  TORCH_CHECK(lhs_bias_ptr != nullptr && rhs_bias_ptr != nullptr,
              "gate_sigmoid_mul_pack_to_mxfp8_fused requires lhs_bias and rhs_bias");

  const int64_t batch = mask.size(0);
  const int64_t n = mask.size(1);
  TORCH_CHECK(lhs.size(1) == batch * n * n, "lhs/rhs rows must equal batch * n * n");
  const int64_t padded_rows = ((n + 127) / 128) * 128;
  const int64_t padded_cols = (((n / 32) + 3) / 4) * 4;
  auto a1 = at::empty({batch * 32, n, n}, lhs.options().dtype(at::kFloat8_e4m3fn));
  auto b1 = at::empty({batch * 32, n, n}, lhs.options().dtype(at::kFloat8_e4m3fn));
  auto a2_t = at::empty({batch * 32, n, n}, lhs.options().dtype(at::kFloat8_e4m3fn));
  auto b2_rhs = at::empty({batch * 32, n, n}, lhs.options().dtype(at::kFloat8_e4m3fn));
  auto a1_scale = at::full({batch * 32, padded_rows, padded_cols}, kMinPow2Scale, lhs.options().dtype(at::kFloat8_e8m0fnu));
  auto b1_scale = at::full({batch * 32, padded_rows, padded_cols}, kMinPow2Scale, lhs.options().dtype(at::kFloat8_e8m0fnu));
  auto a2_t_scale = at::full({batch * 32, padded_rows, padded_cols}, kMinPow2Scale, lhs.options().dtype(at::kFloat8_e8m0fnu));
  auto b2_rhs_scale = at::full({batch * 32, padded_rows, padded_cols}, kMinPow2Scale, lhs.options().dtype(at::kFloat8_e8m0fnu));
  if (use_warp_kernel) {
    const dim3 grid(
        static_cast<unsigned int>(batch * 32),
        static_cast<unsigned int>(n),
        static_cast<unsigned int>(n / 32));
    gate_sigmoid_mul_pack_to_mxfp8_warp_kernel<<<grid, 128, 0, current_cuda_stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(lhs.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(rhs.data_ptr()),
        lhs_bias_ptr,
        rhs_bias_ptr,
        mask.data_ptr<bool>(),
        reinterpret_cast<__nv_fp8_storage_t*>(a1.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(a1_scale.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(b1.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(b1_scale.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(a2_t.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(a2_t_scale.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(b2_rhs.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(b2_rhs_scale.data_ptr()),
        static_cast<int>(batch),
        static_cast<int>(n),
        static_cast<int>(padded_rows),
        static_cast<int>(padded_cols));
  } else {
    const dim3 grid(
        static_cast<unsigned int>(batch * 4),
        static_cast<unsigned int>((n + 1) / 2),
        static_cast<unsigned int>(n / 32));
    gate_sigmoid_mul_pack_to_mxfp8_fused_kernel<<<grid, 512, 0, current_cuda_stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(lhs.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(rhs.data_ptr()),
        lhs_bias_ptr,
        rhs_bias_ptr,
        mask.data_ptr<bool>(),
        reinterpret_cast<__nv_fp8_storage_t*>(a1.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(a1_scale.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(b1.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(b1_scale.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(a2_t.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(a2_t_scale.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(b2_rhs.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(b2_rhs_scale.data_ptr()),
        static_cast<int>(batch),
        static_cast<int>(n),
        static_cast<int>(padded_rows),
        static_cast<int>(padded_cols));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a1, a1_scale, b1, b1_scale, a2_t, a2_t_scale, b2_rhs, b2_rhs_scale};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
gate_sigmoid_mul_pack_to_mxfp8_fused(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const c10::optional<at::Tensor>& lhs_bias_opt,
    const c10::optional<at::Tensor>& rhs_bias_opt,
    const at::Tensor& mask) {
  const bool use_warp_kernel = get_gatepack_use_warp();
  return gate_sigmoid_mul_pack_to_mxfp8_impl(lhs, rhs, lhs_bias_opt, rhs_bias_opt, mask, use_warp_kernel);
}

std::tuple<at::Tensor, at::Tensor> tri_pair_to_block32_carrier(
    const at::Tensor& x1,
    const at::Tensor& x2,
    int64_t batch) {
  TORCH_CHECK(x1.scalar_type() == c10::ScalarType::Half && x2.scalar_type() == c10::ScalarType::Half,
              "tri_pair_to_block32_carrier currently expects float16 inputs");
  TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must match");
  const int64_t d_chunk = x1.size(0) / batch;
  const int64_t n = x1.size(1);
  TORCH_CHECK(d_chunk == 32, "tri_pair_to_block32_carrier expects d_chunk=32, got ", d_chunk);
  auto output = at::empty({batch, n, n, d_chunk * 2}, x1.options().dtype(at::kFloat8_e4m3fn));
  auto output_scale = at::empty({batch, n, n, 2}, x1.options().dtype(at::kFloat));
  const dim3 grid(static_cast<unsigned int>(batch * n), static_cast<unsigned int>((n + 15) / 16));
  tri_pair_to_block32_carrier_tiled_kernel<<<grid, 256, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __half*>(x1.data_ptr()),
      reinterpret_cast<const __half*>(x2.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
      output_scale.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(n),
      static_cast<int>(d_chunk));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, output_scale};
}

std::tuple<at::Tensor, at::Tensor> tri_pair_layernorm_to_block32_carrier(
    const at::Tensor& x1,
    const at::Tensor& x2,
    int64_t batch,
    const at::Tensor& weight,
    const at::Tensor& bias,
    float eps) {
  TORCH_CHECK(x1.scalar_type() == c10::ScalarType::Half && x2.scalar_type() == c10::ScalarType::Half,
              "tri_pair_layernorm_to_block32_carrier expects float16 inputs");
  TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must match");
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 && weight.size(0) == 64 && bias.size(0) == 64,
              "output layernorm weight/bias must have width 64");
  TORCH_CHECK(weight.scalar_type() == c10::ScalarType::BFloat16 && bias.scalar_type() == c10::ScalarType::BFloat16,
              "output layernorm weight/bias must be bfloat16");
  const int64_t d_chunk = x1.size(0) / batch;
  const int64_t n = x1.size(1);
  TORCH_CHECK(d_chunk == 32, "tri_pair_layernorm_to_block32_carrier expects d_chunk=32, got ", d_chunk);
  auto output = at::empty({batch, n, n, d_chunk * 2}, x1.options().dtype(at::kFloat8_e4m3fn));
  auto output_scale = at::empty({batch, n, n, 2}, x1.options().dtype(at::kFloat));
  const int64_t total_positions = batch * n * n;
  tri_pair_layernorm_to_block32_half_kernel<<<(total_positions + 1) / 2, 128, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __half*>(x1.data_ptr()),
      reinterpret_cast<const __half*>(x2.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
      output_scale.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(n),
      eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, output_scale};
}

void validate_block32_carrier(const at::Tensor& payload, const at::Tensor& scale);

std::vector<at::Tensor> debug_tri_input_norm_gate_block32_stages_impl(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& input_norm_weight,
    const at::Tensor& input_norm_bias,
    double input_norm_eps,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const at::Tensor& mask,
    const std::string& tri_out_dtype_str,
    bool use_warp_kernel) {
  const auto tri_out_dtype = parse_out_dtype(tri_out_dtype_str);
  TORCH_CHECK(tri_out_dtype == c10::ScalarType::Half,
              "debug tri input-norm gate stages currently support tri_out_dtype='float16' only");
  validate_block32_carrier(payload, scale);
  TORCH_CHECK(payload.size(3) == 128, "debug tri input-norm gate stages expect payload width 128");
  check_cuda_tensor(input_norm_weight, "input_norm_weight");
  check_cuda_tensor(input_norm_bias, "input_norm_bias");
  TORCH_CHECK(input_norm_weight.dim() == 1 && input_norm_weight.size(0) == 128,
              "input_norm_weight must have width 128");
  TORCH_CHECK(input_norm_bias.dim() == 1 && input_norm_bias.size(0) == 128,
              "input_norm_bias must have width 128");
  TORCH_CHECK(
      input_norm_weight.scalar_type() == c10::ScalarType::BFloat16 &&
          input_norm_bias.scalar_type() == c10::ScalarType::BFloat16,
      "input_norm weight and bias must be bfloat16");
  check_cuda_tensor(lhs_b_t, "lhs_b_t");
  check_cuda_tensor(lhs_scale_swizzled, "lhs_scale_swizzled");
  check_cuda_tensor(rhs_b_t, "rhs_b_t");
  check_cuda_tensor(rhs_scale_swizzled, "rhs_scale_swizzled");
  check_cuda_tensor(mask, "mask");
  TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool && mask.dim() == 3 && mask.size(1) == mask.size(2),
              "mask must have shape [B, N, N] and bool dtype");
  if (lhs_bias.has_value()) {
    check_cuda_tensor(lhs_bias.value(), "lhs_bias");
  }
  if (rhs_bias.has_value()) {
    check_cuda_tensor(rhs_bias.value(), "rhs_bias");
  }

  set_cuda_device(payload);
  validate_sm100_plus();
  const int64_t batch = payload.size(0);
  const int64_t n = payload.size(1);
  const int64_t cols = payload.size(3);
  const int64_t rows = batch * n * n;
  TORCH_CHECK(rows % 32 == 0, "debug tri input-norm gate stages require rows divisible by 32, got ", rows);

  auto a = at::empty({1, rows, cols}, payload.options().dtype(at::kFloat8_e4m3fn));
  const int64_t padded_rows = ((rows + 127) / 128) * 128;
  auto a_scale_swizzled = at::full(
      {1, padded_rows, 4},
      kMinPow2Scale,
      payload.options().dtype(at::kFloat8_e8m0fnu));
  const int layernorm_blocks = static_cast<int>((rows + 3) / 4);
  transition_norm_to_mxfp8_input_kernel<<<layernorm_blocks, 128, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __nv_fp8_storage_t*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(input_norm_weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(input_norm_bias.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(a.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(a_scale_swizzled.data_ptr()),
      static_cast<int>(rows),
      static_cast<float>(input_norm_eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto lhs_dense = run_mxfp8_cublaslt_bmm(
      a, lhs_b_t, a_scale_swizzled, lhs_scale_swizzled, c10::ScalarType::BFloat16, false);
  auto rhs_dense = run_mxfp8_cublaslt_bmm(
      a, rhs_b_t, a_scale_swizzled, rhs_scale_swizzled, c10::ScalarType::BFloat16, false);
  auto packed = gate_sigmoid_mul_pack_to_mxfp8_impl(lhs_dense, rhs_dense, lhs_bias, rhs_bias, mask, use_warp_kernel);
  auto x1 = run_mxfp8_cublaslt_bmm(
      std::get<0>(packed), std::get<2>(packed), std::get<1>(packed), std::get<3>(packed), tri_out_dtype, false);
  auto x2 = run_mxfp8_cublaslt_bmm(
      std::get<4>(packed), std::get<6>(packed), std::get<5>(packed), std::get<7>(packed), tri_out_dtype, true);
  auto output = tri_pair_to_block32_carrier(x1, x2, batch);

  return {
      a,
      a_scale_swizzled,
      lhs_dense,
      rhs_dense,
      std::get<0>(packed),
      std::get<1>(packed),
      std::get<2>(packed),
      std::get<3>(packed),
      std::get<4>(packed),
      std::get<5>(packed),
      std::get<6>(packed),
      std::get<7>(packed),
      x1,
      x2,
      std::get<0>(output),
      std::get<1>(output),
  };
}

void validate_block32_carrier(const at::Tensor& payload, const at::Tensor& scale) {
  check_cuda_tensor(payload, "payload");
  check_cuda_tensor(scale, "scale");
  TORCH_CHECK(payload.scalar_type() == c10::ScalarType::Float8_e4m3fn, "payload must use float8_e4m3fn");
  TORCH_CHECK(scale.scalar_type() == c10::ScalarType::Float, "scale must use float32");
  TORCH_CHECK(payload.dim() == 4, "payload must have shape [B, N, N, D]");
  TORCH_CHECK(scale.dim() == 4, "scale must have shape [B, N, N, D/32]");
  TORCH_CHECK(payload.size(0) == scale.size(0) && payload.size(1) == scale.size(1) && payload.size(2) == scale.size(2),
              "payload and scale batch/spatial dimensions must match");
  TORCH_CHECK(payload.size(1) == payload.size(2), "payload must use square spatial dimensions");
  TORCH_CHECK(payload.size(3) % 32 == 0, "payload channel dim must be divisible by 32");
  TORCH_CHECK(scale.size(3) == payload.size(3) / 32, "scale last dim must be payload channels / 32");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
debug_gate_sigmoid_mul_pack_to_mxfp8_reference_cuda(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Tensor& lhs_bias,
    const at::Tensor& rhs_bias,
    const at::Tensor& mask) {
  return gate_sigmoid_mul_pack_to_mxfp8_impl(lhs, rhs, lhs_bias, rhs_bias, mask, false);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
debug_gate_sigmoid_mul_pack_to_mxfp8_warp_cuda(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Tensor& lhs_bias,
    const at::Tensor& rhs_bias,
    const at::Tensor& mask) {
  return gate_sigmoid_mul_pack_to_mxfp8_impl(lhs, rhs, lhs_bias, rhs_bias, mask, true);
}

std::vector<at::Tensor> debug_tri_input_norm_gate_block32_reference_stages_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& input_norm_weight,
    const at::Tensor& input_norm_bias,
    double input_norm_eps,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const at::Tensor& mask,
    const std::string& tri_out_dtype) {
  return debug_tri_input_norm_gate_block32_stages_impl(
      payload,
      scale,
      input_norm_weight,
      input_norm_bias,
      input_norm_eps,
      lhs_b_t,
      lhs_scale_swizzled,
      lhs_bias,
      rhs_b_t,
      rhs_scale_swizzled,
      rhs_bias,
      mask,
      tri_out_dtype,
      false);
}

std::vector<at::Tensor> debug_tri_input_norm_gate_block32_warp_stages_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& input_norm_weight,
    const at::Tensor& input_norm_bias,
    double input_norm_eps,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const at::Tensor& mask,
    const std::string& tri_out_dtype) {
  return debug_tri_input_norm_gate_block32_stages_impl(
      payload,
      scale,
      input_norm_weight,
      input_norm_bias,
      input_norm_eps,
      lhs_b_t,
      lhs_scale_swizzled,
      lhs_bias,
      rhs_b_t,
      rhs_scale_swizzled,
      rhs_bias,
      mask,
      tri_out_dtype,
      true);
}

std::tuple<at::Tensor, at::Tensor> linear_block32_fused_cuda(
    const at::Tensor& a,
    const at::Tensor& b_t,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const c10::optional<at::Tensor>& bias,
    const std::string& out_dtype_str,
    bool apply_relu,
    bool direct_fp8_output,
    bool fuse_bias_epilogue,
    const c10::optional<at::Tensor>& residual_payload,
    const c10::optional<at::Tensor>& residual_scale) {
  const auto out_dtype = parse_out_dtype(out_dtype_str);
  TORCH_CHECK(out_dtype == c10::ScalarType::BFloat16,
              "MiniFold native fused linear path currently supports out_dtype='bfloat16' only");

  check_cuda_tensor(a, "a");
  check_cuda_tensor(b_t, "b_t");
  check_cuda_tensor(a_scale_swizzled, "a_scale_swizzled");
  check_cuda_tensor(b_scale_swizzled, "b_scale_swizzled");
  TORCH_CHECK(a.dim() == 3 && b_t.dim() == 3, "linear_block32_fused expects 3D tensors");
  TORCH_CHECK(
      a.scalar_type() == c10::ScalarType::Float8_e4m3fn || a.scalar_type() == c10::ScalarType::Float8_e5m2,
      "a must be FP8");
  TORCH_CHECK(b_t.scalar_type() == a.scalar_type(), "a and b_t must have matching FP8 dtype");
  TORCH_CHECK(a_scale_swizzled.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
              "a_scale_swizzled must use torch.float8_e8m0fnu");
  TORCH_CHECK(b_scale_swizzled.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
              "b_scale_swizzled must use torch.float8_e8m0fnu");
  TORCH_CHECK(a.size(0) == b_t.size(0), "batch dimensions must match");
  TORCH_CHECK(a.size(2) == b_t.size(2), "K dimensions must match");
  TORCH_CHECK(a.size(1) % 32 == 0 && a.size(2) % 32 == 0 && b_t.size(1) % 32 == 0,
              "MiniFold native fused linear path requires M, N, and K divisible by 32");

  set_cuda_device(a);
  validate_sm100_plus();
  check_same_device(a, b_t, a_scale_swizzled, b_scale_swizzled);
  if (bias.has_value()) {
    check_cuda_tensor(bias.value(), "bias");
    TORCH_CHECK(bias.value().device() == a.device(), "bias must be on the same CUDA device as a");
  }
  if (residual_payload.has_value() || residual_scale.has_value()) {
    TORCH_CHECK(residual_payload.has_value() && residual_scale.has_value(),
                "residual_payload and residual_scale must be provided together");
    check_cuda_tensor(residual_payload.value(), "residual_payload");
    check_cuda_tensor(residual_scale.value(), "residual_scale");
    TORCH_CHECK(residual_payload.value().device() == a.device() && residual_scale.value().device() == a.device(),
                "residual tensors must be on the same CUDA device as a");
  }

  const bool use_direct_output = direct_fp8_output && bias.has_value() && !apply_relu;
  const bool use_fused_bias_epilogue = fuse_bias_epilogue && bias.has_value();
  const auto epilogue =
      use_fused_bias_epilogue ? (apply_relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS)
                              : CUBLASLT_EPILOGUE_DEFAULT;
  auto dense = run_mxfp8_cublaslt_bmm(
      a,
      b_t,
      a_scale_swizzled,
      b_scale_swizzled,
      out_dtype,
      false,
      (use_direct_output || use_fused_bias_epilogue) ? bias : c10::nullopt,
      epilogue);
  if (use_direct_output || use_fused_bias_epilogue) {
    return quantize_block32_bf16(dense, residual_payload, residual_scale);
  }
  return quantize_block32_bf16_with_bias(dense, bias, residual_payload, residual_scale, apply_relu);
}

std::tuple<at::Tensor, at::Tensor> transition_norm_fc1_block32_fused_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& norm_weight,
    const at::Tensor& norm_bias,
    double norm_eps,
    const at::Tensor& b_cutlass_col,
    const at::Tensor& b_scale_swizzled,
    const c10::optional<at::Tensor>& bias) {
  validate_block32_carrier(payload, scale);
  TORCH_CHECK(payload.size(3) == 128, "transition_norm_fc1_block32_fused expects payload width 128");
  check_cuda_tensor(norm_weight, "norm_weight");
  check_cuda_tensor(norm_bias, "norm_bias");
  TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.size(0) == 128, "norm_weight must have width 128");
  TORCH_CHECK(norm_bias.dim() == 1 && norm_bias.size(0) == 128, "norm_bias must have width 128");
  TORCH_CHECK(
      norm_weight.scalar_type() == c10::ScalarType::BFloat16 && norm_bias.scalar_type() == c10::ScalarType::BFloat16,
      "norm weight and bias must be bfloat16");
  check_cuda_tensor(b_cutlass_col, "b_cutlass_col");
  check_cuda_tensor(b_scale_swizzled, "b_scale_swizzled");
  TORCH_CHECK(
      b_cutlass_col.scalar_type() == c10::ScalarType::Float8_e4m3fn || b_cutlass_col.scalar_type() == c10::ScalarType::Float8_e5m2,
      "b_cutlass_col must be FP8");
  TORCH_CHECK(b_scale_swizzled.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
              "b_scale_swizzled must use torch.float8_e8m0fnu");
  TORCH_CHECK(b_cutlass_col.dim() == 3, "b_cutlass_col must be 3D");
  TORCH_CHECK(b_cutlass_col.size(0) == 1 && b_cutlass_col.size(1) == 128 && b_cutlass_col.size(2) == 512,
              "transition_norm_fc1_block32_fused expects b_cutlass_col with shape [1, 128, 512]");
  if (bias.has_value()) {
    check_cuda_tensor(bias.value(), "bias");
    TORCH_CHECK(
        bias.value().dim() == 1 && bias.value().size(0) == 512 && bias.value().scalar_type() == c10::ScalarType::BFloat16,
        "fc1 bias must be bfloat16 with width 512");
  }

  set_cuda_device(payload);
  validate_sm100_plus();
  TORCH_CHECK(payload.device() == b_cutlass_col.device() && payload.device() == b_scale_swizzled.device(),
              "all transition_norm_fc1 tensors must be on the same CUDA device");
  TORCH_CHECK(bias.has_value(), "transition_norm_fc1_block32_fused requires bias for canonical fc1 direct output");
  const int64_t batch = payload.size(0);
  const int64_t n = payload.size(1);
  const int64_t cols = payload.size(3);
  const int64_t rows = batch * n * n;
  TORCH_CHECK(rows % 32 == 0, "transition_norm_fc1_block32_fused requires rows divisible by 32, got ", rows);
  const int64_t kChunkRows = get_transition_norm_fc1_chunk_rows();
  auto payload_2d = payload.view({rows, cols});
  auto scale_2d = scale.view({rows, cols / 32});
  auto output_2d = at::empty({rows, 512}, payload.options().dtype(at::kFloat8_e4m3fn));
  auto output_scale_2d = at::empty({rows, 16}, payload.options().dtype(at::kFloat));

  for (int64_t start = 0; start < rows; start += kChunkRows) {
    const int64_t chunk_rows = std::min<int64_t>(kChunkRows, rows - start);
    auto payload_chunk = payload_2d.narrow(0, start, chunk_rows);
    auto scale_chunk = scale_2d.narrow(0, start, chunk_rows);
    auto a = at::empty({1, chunk_rows, cols}, payload.options().dtype(at::kFloat8_e4m3fn));
    const int64_t padded_rows = ((chunk_rows + 127) / 128) * 128;
    auto a_scale_swizzled = at::full(
        {1, padded_rows, 4},
        kMinPow2Scale,
        payload.options().dtype(at::kFloat8_e8m0fnu));
    const int layernorm_blocks = static_cast<int>((chunk_rows + 3) / 4);
    transition_norm_to_mxfp8_input_kernel<<<layernorm_blocks, 128, 0, current_cuda_stream()>>>(
        reinterpret_cast<const __nv_fp8_storage_t*>(payload_chunk.data_ptr()),
        scale_chunk.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(norm_bias.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(a.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(a_scale_swizzled.data_ptr()),
        static_cast<int>(chunk_rows),
        static_cast<float>(norm_eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto output_chunk = output_2d.narrow(0, start, chunk_rows).view({1, chunk_rows, 512});
    auto output_scale_swizzled = at::full(
        {1, padded_rows, 4},
        kMinPow2Scale,
        payload.options().dtype(at::kFloat8_e8m0fnu));
    linear_block32_fc1_direct_into_cuda(
        a,
        b_cutlass_col,
        a_scale_swizzled,
        b_scale_swizzled,
        bias.value(),
        output_chunk,
        output_scale_swizzled);
    output_scale_2d.narrow(0, start, chunk_rows).copy_(
        unswizzle_scale_rowwise(output_scale_swizzled, chunk_rows, 16).to(at::kFloat));
  }
  return {output_2d.view({1, rows, 512}), output_scale_2d.view({1, rows, 16})};
}

std::tuple<at::Tensor, at::Tensor> gate_sigmoid_mul_block32_fused_cuda(
    const at::Tensor& a,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const std::string& out_dtype_str,
    const c10::optional<at::Tensor>& residual_payload,
    const c10::optional<at::Tensor>& residual_scale) {
  const auto out_dtype = parse_out_dtype(out_dtype_str);
  TORCH_CHECK(out_dtype == c10::ScalarType::BFloat16,
              "gate_sigmoid_mul_block32_fused currently supports out_dtype='bfloat16' only");
  check_cuda_tensor(a, "a");
  check_cuda_tensor(lhs_b_t, "lhs_b_t");
  check_cuda_tensor(rhs_b_t, "rhs_b_t");
  check_cuda_tensor(a_scale_swizzled, "a_scale_swizzled");
  check_cuda_tensor(lhs_scale_swizzled, "lhs_scale_swizzled");
  check_cuda_tensor(rhs_scale_swizzled, "rhs_scale_swizzled");
  TORCH_CHECK(a.dim() == 3 && lhs_b_t.dim() == 3 && rhs_b_t.dim() == 3, "gate_sigmoid_mul_block32_fused expects 3D tensors");
  TORCH_CHECK(a.size(0) == lhs_b_t.size(0) && a.size(0) == rhs_b_t.size(0), "batch dimensions must match");
  TORCH_CHECK(a.size(2) == lhs_b_t.size(2) && a.size(2) == rhs_b_t.size(2), "K dimensions must match");
  TORCH_CHECK(lhs_b_t.size(1) == rhs_b_t.size(1), "lhs and rhs gate outputs must have matching width");
  if (residual_payload.has_value() || residual_scale.has_value()) {
    TORCH_CHECK(residual_payload.has_value() && residual_scale.has_value(),
                "residual_payload and residual_scale must be provided together");
    check_cuda_tensor(residual_payload.value(), "residual_payload");
    check_cuda_tensor(residual_scale.value(), "residual_scale");
    TORCH_CHECK(residual_payload.value().device() == a.device() && residual_scale.value().device() == a.device(),
                "residual tensors must be on the same CUDA device as a");
  }

  set_cuda_device(a);
  validate_sm100_plus();
  auto lhs_dense = run_mxfp8_cublaslt_bmm(a, lhs_b_t, a_scale_swizzled, lhs_scale_swizzled, out_dtype, false);
  auto rhs_dense = run_mxfp8_cublaslt_bmm(a, rhs_b_t, a_scale_swizzled, rhs_scale_swizzled, out_dtype, false);
  return gate_sigmoid_mul_quantize_bf16(lhs_dense, rhs_dense, lhs_bias, rhs_bias, residual_payload, residual_scale);
}

std::tuple<at::Tensor, at::Tensor> tri_mul_pair_from_block32_carrier_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& mask,
    const std::string& out_dtype_str) {
  const auto out_dtype = parse_out_dtype(out_dtype_str);
  TORCH_CHECK(out_dtype == c10::ScalarType::Half,
              "tri_mul_pair_from_block32_carrier currently supports out_dtype='float16' only");
  validate_block32_carrier(payload, scale);

  set_cuda_device(payload);
  validate_sm100_plus();
  if (mask.has_value()) {
    check_cuda_tensor(mask.value(), "mask");
    TORCH_CHECK(mask.value().scalar_type() == c10::ScalarType::Bool, "mask must be bool");
  }
  auto packed = pack_block32_to_mxfp8_fused(payload, scale, mask);
  auto x1 = run_mxfp8_cublaslt_bmm(
      std::get<0>(packed), std::get<2>(packed), std::get<1>(packed), std::get<3>(packed), out_dtype, false);
  auto x2 = run_mxfp8_cublaslt_bmm(
      std::get<4>(packed), std::get<6>(packed), std::get<5>(packed), std::get<7>(packed), out_dtype, true);
  return tri_pair_to_block32_carrier(x1, x2, payload.size(0));
}

std::tuple<at::Tensor, at::Tensor> tri_gate_block32_fused_cuda(
    const at::Tensor& a,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const at::Tensor& mask,
    const std::string& tri_out_dtype_str) {
  const auto tri_out_dtype = parse_out_dtype(tri_out_dtype_str);
  TORCH_CHECK(tri_out_dtype == c10::ScalarType::Half,
              "tri_gate_block32_fused currently supports tri_out_dtype='float16' only");
  check_cuda_tensor(a, "a");
  check_cuda_tensor(a_scale_swizzled, "a_scale_swizzled");
  check_cuda_tensor(lhs_b_t, "lhs_b_t");
  check_cuda_tensor(lhs_scale_swizzled, "lhs_scale_swizzled");
  check_cuda_tensor(rhs_b_t, "rhs_b_t");
  check_cuda_tensor(rhs_scale_swizzled, "rhs_scale_swizzled");
  check_cuda_tensor(mask, "mask");
  TORCH_CHECK(a.dim() == 3 && a.size(0) == 1 && a.size(2) == 128,
              "tri_gate_block32_fused expects a with shape [1, rows, 128]");
  TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool && mask.dim() == 3 && mask.size(1) == mask.size(2),
              "mask must have shape [B, N, N] and bool dtype");
  if (lhs_bias.has_value()) {
    check_cuda_tensor(lhs_bias.value(), "lhs_bias");
  }
  if (rhs_bias.has_value()) {
    check_cuda_tensor(rhs_bias.value(), "rhs_bias");
  }

  set_cuda_device(a);
  validate_sm100_plus();
  const int64_t batch = mask.size(0);
  const int64_t n = mask.size(1);
  TORCH_CHECK(a.size(1) == batch * n * n, "a rows must equal batch * n * n");

  auto lhs_dense = run_mxfp8_cublaslt_bmm(a, lhs_b_t, a_scale_swizzled, lhs_scale_swizzled, c10::ScalarType::BFloat16, false);
  auto rhs_dense = run_mxfp8_cublaslt_bmm(a, rhs_b_t, a_scale_swizzled, rhs_scale_swizzled, c10::ScalarType::BFloat16, false);
  auto packed = gate_sigmoid_mul_pack_to_mxfp8_fused(lhs_dense, rhs_dense, lhs_bias, rhs_bias, mask);
  auto x1 = run_mxfp8_cublaslt_bmm(
      std::get<0>(packed), std::get<2>(packed), std::get<1>(packed), std::get<3>(packed), tri_out_dtype, false);
  auto x2 = run_mxfp8_cublaslt_bmm(
      std::get<4>(packed), std::get<6>(packed), std::get<5>(packed), std::get<7>(packed), tri_out_dtype, true);
  return tri_pair_to_block32_carrier(x1, x2, batch);
}

std::tuple<at::Tensor, at::Tensor> tri_input_norm_gate_block32_fused_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& input_norm_weight,
    const at::Tensor& input_norm_bias,
    double input_norm_eps,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const at::Tensor& mask,
    const std::string& tri_out_dtype_str) {
  const auto tri_out_dtype = parse_out_dtype(tri_out_dtype_str);
  TORCH_CHECK(tri_out_dtype == c10::ScalarType::Half,
              "tri_input_norm_gate_block32_fused currently supports tri_out_dtype='float16' only");
  validate_block32_carrier(payload, scale);
  TORCH_CHECK(payload.size(3) == 128, "tri_input_norm_gate_block32_fused expects payload width 128");
  check_cuda_tensor(input_norm_weight, "input_norm_weight");
  check_cuda_tensor(input_norm_bias, "input_norm_bias");
  TORCH_CHECK(input_norm_weight.dim() == 1 && input_norm_weight.size(0) == 128,
              "input_norm_weight must have width 128");
  TORCH_CHECK(input_norm_bias.dim() == 1 && input_norm_bias.size(0) == 128,
              "input_norm_bias must have width 128");
  TORCH_CHECK(
      input_norm_weight.scalar_type() == c10::ScalarType::BFloat16 &&
          input_norm_bias.scalar_type() == c10::ScalarType::BFloat16,
      "input_norm weight and bias must be bfloat16");
  check_cuda_tensor(lhs_b_t, "lhs_b_t");
  check_cuda_tensor(lhs_scale_swizzled, "lhs_scale_swizzled");
  check_cuda_tensor(rhs_b_t, "rhs_b_t");
  check_cuda_tensor(rhs_scale_swizzled, "rhs_scale_swizzled");
  check_cuda_tensor(mask, "mask");
  TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool && mask.dim() == 3 && mask.size(1) == mask.size(2),
              "mask must have shape [B, N, N] and bool dtype");
  if (lhs_bias.has_value()) {
    check_cuda_tensor(lhs_bias.value(), "lhs_bias");
  }
  if (rhs_bias.has_value()) {
    check_cuda_tensor(rhs_bias.value(), "rhs_bias");
  }

  set_cuda_device(payload);
  validate_sm100_plus();
  const int64_t batch = payload.size(0);
  const int64_t n = payload.size(1);
  const int64_t cols = payload.size(3);
  const int64_t rows = batch * n * n;
  TORCH_CHECK(rows % 32 == 0, "tri_input_norm_gate_block32_fused requires rows divisible by 32, got ", rows);

  auto a = at::empty({1, rows, cols}, payload.options().dtype(at::kFloat8_e4m3fn));
  const int64_t padded_rows = ((rows + 127) / 128) * 128;
  auto a_scale_swizzled = at::full(
      {1, padded_rows, 4},
      kMinPow2Scale,
      payload.options().dtype(at::kFloat8_e8m0fnu));
  const int layernorm_blocks = static_cast<int>((rows + 3) / 4);
  transition_norm_to_mxfp8_input_kernel<<<layernorm_blocks, 128, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __nv_fp8_storage_t*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(input_norm_weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(input_norm_bias.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(a.data_ptr()),
      reinterpret_cast<__nv_fp8_storage_t*>(a_scale_swizzled.data_ptr()),
      static_cast<int>(rows),
      static_cast<float>(input_norm_eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto lhs_dense = run_mxfp8_cublaslt_bmm(a, lhs_b_t, a_scale_swizzled, lhs_scale_swizzled, c10::ScalarType::BFloat16, false);
  auto rhs_dense = run_mxfp8_cublaslt_bmm(a, rhs_b_t, a_scale_swizzled, rhs_scale_swizzled, c10::ScalarType::BFloat16, false);
  auto packed = gate_sigmoid_mul_pack_to_mxfp8_fused(lhs_dense, rhs_dense, lhs_bias, rhs_bias, mask);
  auto x1 = run_mxfp8_cublaslt_bmm(
      std::get<0>(packed), std::get<2>(packed), std::get<1>(packed), std::get<3>(packed), tri_out_dtype, false);
  auto x2 = run_mxfp8_cublaslt_bmm(
      std::get<4>(packed), std::get<6>(packed), std::get<5>(packed), std::get<7>(packed), tri_out_dtype, true);
  return tri_pair_to_block32_carrier(x1, x2, batch);
}

std::tuple<at::Tensor, at::Tensor> tri_gate_layernorm_block32_fused_cuda(
    const at::Tensor& a,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const at::Tensor& mask,
    const at::Tensor& output_norm_weight,
    const at::Tensor& output_norm_bias,
    double output_norm_eps,
    const std::string& tri_out_dtype_str) {
  const auto tri_out_dtype = parse_out_dtype(tri_out_dtype_str);
  TORCH_CHECK(tri_out_dtype == c10::ScalarType::Half,
              "tri_gate_layernorm_block32_fused currently supports tri_out_dtype='float16' only");
  check_cuda_tensor(a, "a");
  check_cuda_tensor(a_scale_swizzled, "a_scale_swizzled");
  check_cuda_tensor(lhs_b_t, "lhs_b_t");
  check_cuda_tensor(lhs_scale_swizzled, "lhs_scale_swizzled");
  check_cuda_tensor(rhs_b_t, "rhs_b_t");
  check_cuda_tensor(rhs_scale_swizzled, "rhs_scale_swizzled");
  check_cuda_tensor(mask, "mask");
  check_cuda_tensor(output_norm_weight, "output_norm_weight");
  check_cuda_tensor(output_norm_bias, "output_norm_bias");
  TORCH_CHECK(a.dim() == 3 && a.size(0) == 1 && a.size(2) == 128,
              "tri_gate_layernorm_block32_fused expects a with shape [1, rows, 128]");
  TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool && mask.dim() == 3 && mask.size(1) == mask.size(2),
              "mask must have shape [B, N, N] and bool dtype");
  TORCH_CHECK(output_norm_weight.scalar_type() == c10::ScalarType::BFloat16 &&
                  output_norm_bias.scalar_type() == c10::ScalarType::BFloat16,
              "output_norm weight/bias must be bfloat16");
  TORCH_CHECK(output_norm_weight.dim() == 1 && output_norm_weight.size(0) == 64, "output_norm_weight must have width 64");
  TORCH_CHECK(output_norm_bias.dim() == 1 && output_norm_bias.size(0) == 64, "output_norm_bias must have width 64");
  if (lhs_bias.has_value()) {
    check_cuda_tensor(lhs_bias.value(), "lhs_bias");
  }
  if (rhs_bias.has_value()) {
    check_cuda_tensor(rhs_bias.value(), "rhs_bias");
  }

  set_cuda_device(a);
  validate_sm100_plus();
  const int64_t batch = mask.size(0);
  const int64_t n = mask.size(1);
  TORCH_CHECK(a.size(1) == batch * n * n, "a rows must equal batch * n * n");

  auto lhs_dense = run_mxfp8_cublaslt_bmm(a, lhs_b_t, a_scale_swizzled, lhs_scale_swizzled, c10::ScalarType::BFloat16, false);
  auto rhs_dense = run_mxfp8_cublaslt_bmm(a, rhs_b_t, a_scale_swizzled, rhs_scale_swizzled, c10::ScalarType::BFloat16, false);
  auto packed = gate_sigmoid_mul_pack_to_mxfp8_fused(lhs_dense, rhs_dense, lhs_bias, rhs_bias, mask);
  auto x1 = run_mxfp8_cublaslt_bmm(
      std::get<0>(packed), std::get<2>(packed), std::get<1>(packed), std::get<3>(packed), tri_out_dtype, false);
  auto x2 = run_mxfp8_cublaslt_bmm(
      std::get<4>(packed), std::get<6>(packed), std::get<5>(packed), std::get<7>(packed), tri_out_dtype, true);
  return tri_pair_layernorm_to_block32_carrier(
      x1,
      x2,
      batch,
      output_norm_weight,
      output_norm_bias,
      static_cast<float>(output_norm_eps));
}

std::tuple<at::Tensor, at::Tensor> relu_block32_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale) {
  return unary_block32_impl(payload, scale, 0);
}

std::tuple<at::Tensor, at::Tensor> add_block32_cuda(
    const at::Tensor& lhs_payload,
    const at::Tensor& lhs_scale,
    const at::Tensor& rhs_payload,
    const at::Tensor& rhs_scale) {
  return binary_block32_impl(lhs_payload, lhs_scale, rhs_payload, rhs_scale, 0);
}

std::tuple<at::Tensor, at::Tensor> layernorm_block32_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps) {
  return layernorm_block32_impl(payload, scale, weight, bias, static_cast<float>(eps));
}

}  // namespace minifold_native_ext
