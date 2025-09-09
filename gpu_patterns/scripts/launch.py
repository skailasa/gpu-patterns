import torch
from gpu_patterns import kernels

# Example use
x = torch.randn(1_000_000, device="cuda", dtype=torch.float32)
y = kernels.axpby(x, 2.0, 0.5)
print(y[:5])