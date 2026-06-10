from pathlib import Path
import site

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
TE_ROOT = Path("/usr/local/lib/python3.12/dist-packages/transformer_engine")
SITE_INCLUDE = next(
    (Path(p) / "include" for p in site.getsitepackages() if (Path(p) / "include").exists()),
    None,
)
if SITE_INCLUDE is None:
    raise RuntimeError("Could not locate site-packages include directory.")


setup(
    name="bmm_ext",
    version="0.1.0",
    description="Block-scaled batched matmul for MXFP8 and NVFP4",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="bmm_ext._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/bmm_block_scaled_cuda.cu",
            ],
            include_dirs=[
                str(ROOT / "csrc"),
                str(SITE_INCLUDE),
                str(TE_ROOT / "common/include"),
                "/usr/local/cuda/include",
            ],
            library_dirs=[
                "/usr/local/cuda/lib64",
                "/usr/local/cuda/targets/x86_64-linux/lib",
                str(TE_ROOT),
            ],
            runtime_library_dirs=[
                str(TE_ROOT),
                "/usr/local/cuda/lib64",
                "/usr/local/cuda/targets/x86_64-linux/lib",
            ],
            libraries=["transformer_engine", "nvrtc", "cublasLt", "cublas"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
