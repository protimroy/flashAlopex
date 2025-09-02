# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(__file__)
cuda_dir = os.path.join(this_dir, "cuda")

setup(
    name="flashalopex",
    version="0.1.0",
    description="FlashAlopex - CUDA fused Alopex optimizer",
    packages=["flashalopex"],
    ext_modules=[
        CUDAExtension(
            name="flashalopex_cuda",
            sources=[
                os.path.join(cuda_dir, "alopex_binding.cpp"),
                os.path.join(cuda_dir, "alopex_kernel.cu")
            ],
            extra_compile_args={
                "cxx": ["-O3", "-g"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
