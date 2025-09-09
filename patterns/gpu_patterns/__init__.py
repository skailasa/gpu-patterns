import os
import pathlib

import numpy as np
import torch
from torch.utils.cpp_extension import load

root = pathlib.Path(__file__).parent
kernels = root / "kernels"
sources = [*(kernels.glob("*.cpp")), *(kernels.glob("*.cu"))]
sources = [str(p) for p in sources]

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

kernels = load(
    name="gpu_kernels",
    sources=sources,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)
