#pragma once

#include <ATen/ATen.h>

#include <string>

namespace tri_mul_ext {

at::Tensor tri_mul_fused_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t k_dim,
    const std::string& out_dtype);

std::vector<at::Tensor> tri_mul_fused_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t k_dim);

at::Tensor tri_mul_bdnn_cublas_cuda(
    const at::Tensor& a,
    const at::Tensor& b,
    int64_t k_dim,
    const std::string& out_dtype);

at::Tensor tri_mul_xbdnn_cublas_cuda(
    const at::Tensor& x_bdnn,
    const std::string& out_dtype);

}  // namespace tri_mul_ext
