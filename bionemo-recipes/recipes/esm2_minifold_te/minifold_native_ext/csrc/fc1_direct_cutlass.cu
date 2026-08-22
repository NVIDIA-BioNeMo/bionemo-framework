#include "common.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace minifold_native_ext {

namespace {

using namespace cute;

constexpr float kMinPow2Scale = 5.877471754111438e-39f;

void check_cuda_tensor(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void set_cuda_device(const at::Tensor& t) {
  TORCH_CHECK(cudaSetDevice(static_cast<int>(t.get_device())) == cudaSuccess, "failed to set CUDA device");
}

void record_tensor_on_current_stream(const at::Tensor& t) {
  if (!t.defined() || !t.is_cuda() || !t.has_storage()) {
    return;
  }
  c10::cuda::CUDACachingAllocator::recordStream(
      t.storage().data_ptr(),
      c10::cuda::getCurrentCUDAStream(t.get_device()));
}

void validate_sm100_plus() {
  auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props != nullptr, "could not query CUDA device properties");
  const int sm = props->major * 10 + props->minor;
  TORCH_CHECK(sm >= 100, "MiniFold CUTLASS fc1 direct path requires SM100+, found SM", sm);
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

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
using ElementInput = cutlass::float_e4m3_t;
using ElementA = cutlass::mx_float8_t<ElementInput>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

using ElementB = cutlass::mx_float8_t<ElementInput>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16;

using ElementC = void;
using LayoutCTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 1;

using ElementD = cutlass::float_e4m3_t;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 16;

using ElementAccumulator = float;
using ElementCompute = float;
using ElementBias = cutlass::bfloat16_t;
using ElementSFD = cutlass::float_ue8m0_t;

constexpr int OutputSFVectorSize = 16;

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTileShape = Shape<_256, _256, _256>;
using ClusterShape = Shape<_2, _4, _1>;

using FusionOperation = cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
    cutlass::epilogue::thread::ReLu,
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD,
    cutlass::layout::RowMajor,
    ElementBias,
    void,
    ElementCompute>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    MmaTileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutCTag,
    AlignmentC,
    ElementD,
    LayoutDTag,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutATag,
    AlignmentA,
    ElementB,
    LayoutBTag,
    AlignmentB,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using Sm1xxBlockScaledOutputConfig = cutlass::detail::Sm1xxBlockScaledOutputConfig<
    OutputSFVectorSize,
    cute::UMMA::Major::K>;

template <typename StrideType, typename ShapeType>
auto make_packed_stride(ShapeType shape) {
  return cutlass::make_cute_packed_stride(StrideType{}, shape);
}
#endif

}  // namespace

namespace {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
void run_linear_block32_fc1_direct(
    const at::Tensor& a,
    const at::Tensor& b_cutlass_col,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const at::Tensor& bias,
    const at::Tensor& out,
    const at::Tensor& out_scale_swizzled) {
  check_cuda_tensor(a, "a");
  check_cuda_tensor(b_cutlass_col, "b_cutlass_col");
  check_cuda_tensor(a_scale_swizzled, "a_scale_swizzled");
  check_cuda_tensor(b_scale_swizzled, "b_scale_swizzled");
  check_cuda_tensor(bias, "bias");
  check_cuda_tensor(out, "out");
  check_cuda_tensor(out_scale_swizzled, "out_scale_swizzled");
  TORCH_CHECK(a.scalar_type() == c10::ScalarType::Float8_e4m3fn, "a must use float8_e4m3fn");
  TORCH_CHECK(b_cutlass_col.scalar_type() == c10::ScalarType::Float8_e4m3fn, "b_cutlass_col must use float8_e4m3fn");
  TORCH_CHECK(a_scale_swizzled.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
              "a_scale_swizzled must use float8_e8m0fnu");
  TORCH_CHECK(b_scale_swizzled.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
              "b_scale_swizzled must use float8_e8m0fnu");
  TORCH_CHECK(bias.scalar_type() == c10::ScalarType::BFloat16, "bias must be bfloat16");
  TORCH_CHECK(out.scalar_type() == c10::ScalarType::Float8_e4m3fn, "out must use float8_e4m3fn");
  TORCH_CHECK(out_scale_swizzled.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
              "out_scale_swizzled must use float8_e8m0fnu");
  TORCH_CHECK(a.device() == b_cutlass_col.device() && a.device() == a_scale_swizzled.device() &&
                  a.device() == b_scale_swizzled.device() && a.device() == bias.device() &&
                  a.device() == out.device() && a.device() == out_scale_swizzled.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(a.dim() == 3 && a.size(0) == 1, "a must have shape [1, M, 128]");
  TORCH_CHECK(a.size(2) == 128, "a must have width 128");
  TORCH_CHECK(b_cutlass_col.dim() == 3 && b_cutlass_col.size(0) == 1 && b_cutlass_col.size(1) == 128 &&
                  b_cutlass_col.size(2) == 512,
              "b_cutlass_col must have shape [1, 128, 512]");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == 512, "bias must have shape [512]");

  set_cuda_device(a);
  validate_sm100_plus();

  const int64_t batch = a.size(0);
  const int64_t m = a.size(1);
  const int64_t k = a.size(2);
  const int64_t n = b_cutlass_col.size(2);
  TORCH_CHECK(batch == 1, "linear_block32_fc1_direct supports batch dimension 1 only");
  TORCH_CHECK(m % 32 == 0, "linear_block32_fc1_direct requires M divisible by 32, got ", m);

  const int64_t groups = n / 32;
  const int64_t padded_rows = ((m + 127) / 128) * 128;
  const int64_t padded_cols = ((groups + 3) / 4) * 4;
  TORCH_CHECK(out.dim() == 3 && out.size(0) == batch && out.size(1) == m && out.size(2) == n, "out has incompatible shape");
  TORCH_CHECK(
      out_scale_swizzled.dim() == 3 && out_scale_swizzled.size(0) == batch && out_scale_swizzled.size(1) == padded_rows &&
          out_scale_swizzled.size(2) == padded_cols,
      "out_scale_swizzled has incompatible shape");

  auto norm_constant = at::ones({1}, a.options().dtype(at::kFloat));

  auto stride_a = make_packed_stride<StrideA>(cute::make_shape(static_cast<int>(m), static_cast<int>(k), 1));
  auto stride_b = make_packed_stride<StrideB>(cute::make_shape(static_cast<int>(n), static_cast<int>(k), 1));
  auto stride_d = make_packed_stride<StrideD>(cute::make_shape(static_cast<int>(m), static_cast<int>(n), 1));
  auto stride_c = StrideC{};
  auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  auto layout_sfd = Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(
      cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  static_cast<void>(layout_sfd);

  Gemm gemm;
  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device());
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.bias_ptr = reinterpret_cast<ElementBias const*>(bias.data_ptr());
  fusion_args.block_scale_factor_ptr = reinterpret_cast<ElementSFD*>(out_scale_swizzled.data_ptr());
  fusion_args.norm_constant_ptr = norm_constant.data_ptr<float>();

  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      {
          reinterpret_cast<ElementA::DataType const*>(a.data_ptr()),
          stride_a,
          reinterpret_cast<ElementB::DataType const*>(b_cutlass_col.data_ptr()),
          stride_b,
          reinterpret_cast<ElementA::ScaleFactorType const*>(a_scale_swizzled.data_ptr()),
          layout_sfa,
          reinterpret_cast<ElementB::ScaleFactorType const*>(b_scale_swizzled.data_ptr()),
          layout_sfb,
      },
      {
          fusion_args,
          nullptr,
          stride_c,
          reinterpret_cast<ElementD*>(out.data_ptr()),
          stride_d,
      }};
  arguments.scheduler.max_swizzle_size = 1;

  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS fc1 direct-output can_implement failed: ", cutlassGetStatusString(status));

  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)}, a.options().dtype(at::kByte));
  }

  status = gemm.initialize(arguments, workspace_size > 0 ? workspace.data_ptr() : nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS fc1 direct-output initialize failed: ", cutlassGetStatusString(status));

  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS fc1 direct-output run failed: ", cutlassGetStatusString(status));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUTLASS fc1 direct-output launch failed after run()");

  record_tensor_on_current_stream(norm_constant);
  record_tensor_on_current_stream(out_scale_swizzled);
  record_tensor_on_current_stream(out);
  if (workspace_size > 0) {
    record_tensor_on_current_stream(workspace);
  }
}
#endif

}  // namespace

std::tuple<at::Tensor, at::Tensor> linear_block32_fc1_direct_cuda(
    const at::Tensor& a,
    const at::Tensor& b_cutlass_col,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const at::Tensor& bias) {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  TORCH_CHECK(false, "CUTLASS SM100 support is unavailable in this build");
#else
  const int64_t batch = a.size(0);
  const int64_t m = a.size(1);
  const int64_t n = b_cutlass_col.size(2);
  const int64_t groups = n / 32;
  const int64_t padded_rows = ((m + 127) / 128) * 128;
  const int64_t padded_cols = ((groups + 3) / 4) * 4;
  auto out = at::empty({batch, m, n}, a.options().dtype(at::kFloat8_e4m3fn));
  auto out_scale_swizzled = at::full(
      {batch, padded_rows, padded_cols},
      kMinPow2Scale,
      a.options().dtype(at::kFloat8_e8m0fnu));
  run_linear_block32_fc1_direct(a, b_cutlass_col, a_scale_swizzled, b_scale_swizzled, bias, out, out_scale_swizzled);
  auto out_scale = unswizzle_scale_rowwise(out_scale_swizzled, m, groups).to(at::kFloat).contiguous();
  return {out, out_scale};
#endif
}

void linear_block32_fc1_direct_into_cuda(
    const at::Tensor& a,
    const at::Tensor& b_cutlass_col,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const at::Tensor& bias,
    const at::Tensor& out,
    const at::Tensor& out_scale_swizzled) {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  TORCH_CHECK(false, "CUTLASS SM100 support is unavailable in this build");
#else
  run_linear_block32_fc1_direct(a, b_cutlass_col, a_scale_swizzled, b_scale_swizzled, bias, out, out_scale_swizzled);
#endif
}

}  // namespace minifold_native_ext
