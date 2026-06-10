#include <torch/extension.h>

#include "common.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "bmm_block_scaled",
      &bmm_ext::bmm_block_scaled_cuda,
      "Block-scaled BMM for MXFP8 and NVFP4",
      pybind11::arg("a"),
      pybind11::arg("b"),
      pybind11::arg("a_scale"),
      pybind11::arg("b_scale"),
      pybind11::arg("a_amax"),
      pybind11::arg("b_amax"),
      pybind11::arg("format"),
      pybind11::arg("out_dtype"),
      pybind11::arg("sf_vec_size"),
      pybind11::arg("a_shape_override") = std::vector<int64_t>{},
      pybind11::arg("b_shape_override") = std::vector<int64_t>{},
      pybind11::arg("a_rhs_transposed") = false,
      pybind11::arg("b_rhs_transposed") = false);
  m.def(
      "mxfp8_cublaslt_bmm",
      &bmm_ext::mxfp8_cublaslt_bmm_cuda,
      "Raw cuBLASLt MXFP8 strided batched GEMM",
      pybind11::arg("a"),
      pybind11::arg("b_t"),
      pybind11::arg("a_scale_swizzled"),
      pybind11::arg("b_scale_swizzled"),
      pybind11::arg("out_dtype"));
  m.def(
      "mxfp8_cublaslt_bmm_rhs",
      &bmm_ext::mxfp8_cublaslt_bmm_rhs_cuda,
      "Raw cuBLASLt MXFP8 strided batched GEMM with RHS-packed B operand",
      pybind11::arg("a"),
      pybind11::arg("b"),
      pybind11::arg("a_scale_swizzled"),
      pybind11::arg("b_scale_swizzled"),
      pybind11::arg("out_dtype"));
  m.def(
      "mxfp8_cublaslt_tri_mul_pair",
      &bmm_ext::mxfp8_cublaslt_tri_mul_pair_cuda,
      "Paired raw cuBLASLt MXFP8 tri-mul forward contractions",
      pybind11::arg("a1"),
      pybind11::arg("b1"),
      pybind11::arg("a2_t"),
      pybind11::arg("b2_rhs"),
      pybind11::arg("a1_scale_swizzled"),
      pybind11::arg("b1_scale_swizzled"),
      pybind11::arg("a2_t_scale_swizzled"),
      pybind11::arg("b2_rhs_scale_swizzled"),
      pybind11::arg("out_dtype"));
  m.def(
      "mxfp8_cublaslt_tri_mul_pair_backward",
      &bmm_ext::mxfp8_cublaslt_tri_mul_pair_backward_cuda,
      "Paired raw cuBLASLt MXFP8 tri-mul backward contractions",
      pybind11::arg("g1"),
      pybind11::arg("g1_t"),
      pybind11::arg("g2"),
      pybind11::arg("g2_t"),
      pybind11::arg("a1_t"),
      pybind11::arg("b1_t"),
      pybind11::arg("a2"),
      pybind11::arg("b2"),
      pybind11::arg("g1_scale_swizzled"),
      pybind11::arg("g1_t_scale_swizzled"),
      pybind11::arg("g2_scale_swizzled"),
      pybind11::arg("g2_t_scale_swizzled"),
      pybind11::arg("a1_t_scale_swizzled"),
      pybind11::arg("b1_t_scale_swizzled"),
      pybind11::arg("a2_scale_swizzled"),
      pybind11::arg("b2_scale_swizzled"));
}
