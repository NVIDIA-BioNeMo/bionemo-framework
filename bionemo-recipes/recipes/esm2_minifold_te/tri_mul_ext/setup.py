from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent


setup(
    name="tri_mul_ext",
    version="0.1.0",
    description="Scaffold for fused triangular multiplication kernels",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="tri_mul_ext._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/tri_mul_fused_bf16.cu",
            ],
            include_dirs=[str(ROOT / "csrc")],
            libraries=["cublas"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
