#pragma once

#include <ATen/ATen.h>

#include <string>
#include <vector>

namespace bmm_ext {

enum class BlockScaledFormat {
  kMXFP8,
  kNVFP4,
};

inline BlockScaledFormat parse_format(const std::string& format) {
  if (format == "mxfp8") {
    return BlockScaledFormat::kMXFP8;
  }
  if (format == "nvfp4") {
    return BlockScaledFormat::kNVFP4;
  }
  throw std::invalid_argument("format must be 'mxfp8' or 'nvfp4'");
}

inline c10::ScalarType parse_out_dtype(const std::string& dtype) {
  if (dtype == "float16") {
    return c10::ScalarType::Half;
  }
  if (dtype == "bfloat16") {
    return c10::ScalarType::BFloat16;
  }
  if (dtype == "float32") {
    return c10::ScalarType::Float;
  }
  throw std::invalid_argument("out_dtype must be one of: float16, bfloat16, float32");
}

std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t>& dims);

at::Tensor bmm_block_scaled_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale,
    const at::Tensor& b_scale,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const std::string& format,
    const std::string& out_dtype,
    int64_t sf_vec_size,
    std::vector<int64_t> a_shape_override,
    std::vector<int64_t> b_shape_override,
    bool a_rhs_transposed,
    bool b_rhs_transposed);

at::Tensor mxfp8_cublaslt_bmm_cuda(
    const at::Tensor& a,
    const at::Tensor& b_t,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const std::string& out_dtype);

at::Tensor mxfp8_cublaslt_bmm_rhs_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const std::string& out_dtype);

std::vector<at::Tensor> mxfp8_cublaslt_tri_mul_pair_cuda(
    const at::Tensor& a1,
    const at::Tensor& b1,
    const at::Tensor& a2_t,
    const at::Tensor& b2_rhs,
    const at::Tensor& a1_scale_swizzled,
    const at::Tensor& b1_scale_swizzled,
    const at::Tensor& a2_t_scale_swizzled,
    const at::Tensor& b2_rhs_scale_swizzled,
    const std::string& out_dtype);

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
    const at::Tensor& b2_scale_swizzled);

}  // namespace bmm_ext
