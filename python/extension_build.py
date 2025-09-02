# flashalopex/extension_build.py
import os
import torch
from torch.utils.cpp_extension import load

_here = os.path.dirname(__file__)
cuda_src = os.path.join(_here, "..", "cuda")
_build_dir = os.path.join(_here, "..", ".torch_ext_build")  # define build dir

def build_jit(name: str = "flashalopex_cuda", verbose: bool = True):
    os.makedirs(_build_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot build flashalopex_cuda.")

    # Set a specific arch to speed up builds if not provided
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    sources = [
        os.path.join(cuda_src, "alopex_kernel.cu"),
        os.path.join(cuda_src, "alopex_binding.cpp"),
    ]

    module = load(
        name=name,
        sources=sources,
        verbose=verbose,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcurand"],  # link cuRAND
        build_directory=_build_dir,
        with_cuda=True,
        is_python_module=True,
    )
    return module

if __name__ == "__main__":
    build_jit()