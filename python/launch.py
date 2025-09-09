import os
import pathlib

import numpy as np
import torch
from torch.utils.cpp_extension import load

root = pathlib.Path(__file__).parent.parent
kernels = root / "kernels"
sources = [*(kernels.glob("*.cpp")), *(kernels.glob("*.cu"))]
sources = [str(p) for p in sources]

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

ext = load(
    name="my_ext",
    sources=sources,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# Example use
x = torch.randn(1_000_000, device="cuda", dtype=torch.float32)
y = ext.axpby(x, 2.0, 0.5)
print(y[:5])