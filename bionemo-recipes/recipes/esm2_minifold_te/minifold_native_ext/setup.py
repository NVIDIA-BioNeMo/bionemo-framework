from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CUTLASS_ROOT = ROOT / "third_party" / "cutlass"


setup(
    name="minifold_native_ext",
    version="0.1.0",
    description="MiniFold-specific native FP8 kernels",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="minifold_native_ext._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/fc1_direct_cutlass.cu",
                "csrc/linear_block32_cuda.cu",
            ],
            include_dirs=[
                str(ROOT / "csrc"),
                str(CUTLASS_ROOT / "include"),
                str(CUTLASS_ROOT / "tools" / "util" / "include"),
                "/usr/local/cuda/include",
                "/usr/local/cuda/targets/x86_64-linux/include",
            ],
            library_dirs=[
                "/usr/local/cuda/lib64",
                "/usr/local/cuda/targets/x86_64-linux/lib",
            ],
            runtime_library_dirs=[
                "/usr/local/cuda/lib64",
                "/usr/local/cuda/targets/x86_64-linux/lib",
            ],
            libraries=["cublasLt", "cublas"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
