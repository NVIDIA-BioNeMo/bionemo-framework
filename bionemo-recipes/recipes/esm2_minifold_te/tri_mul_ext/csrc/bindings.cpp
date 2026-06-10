#include <torch/extension.h>

#include "common.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "tri_mul_fused",
      &tri_mul_ext::tri_mul_fused_cuda,
      "Triangular multiplication fused-kernel scaffold",
      pybind11::arg("a"),
      pybind11::arg("b"),
      pybind11::arg("k_dim"),
      pybind11::arg("out_dtype"));
  m.def(
      "tri_mul_fused_backward",
      &tri_mul_ext::tri_mul_fused_backward_cuda,
      "Triangular multiplication fused backward",
      pybind11::arg("grad"),
      pybind11::arg("a"),
      pybind11::arg("b"),
      pybind11::arg("k_dim"));
  m.def(
      "tri_mul_bdnn_cublas",
      &tri_mul_ext::tri_mul_bdnn_cublas_cuda,
      "Triangular multiplication on (B, D, N, N) via cuBLAS strided batched GEMM",
      pybind11::arg("a"),
      pybind11::arg("b"),
      pybind11::arg("k_dim"),
      pybind11::arg("out_dtype"));
  m.def(
      "tri_mul_xbdnn_cublas",
      &tri_mul_ext::tri_mul_xbdnn_cublas_cuda,
      "Compute both triangular contractions from one contiguous (B, 128, N, N) tensor via cuBLAS batched GEMM",
      pybind11::arg("x_bdnn"),
      pybind11::arg("out_dtype"));
}
