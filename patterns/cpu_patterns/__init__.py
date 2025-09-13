import os
import pathlib

from torch.utils.cpp_extension import load

root = pathlib.Path(__file__).parent
kernels = root / "kernels"
sources = [*(kernels.glob("*.cpp"))]
sources = [str(p) for p in sources]

kernels = load(
    name="cpu_kernels",
    sources=sources,
    extra_cflags=["-O3", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=True,
)
