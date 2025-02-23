import torch
import triton.profiler as proton

from fwht._fwht_triton import fwht

a = torch.randn(16, 2048, 256, device='cuda', dtype=torch.float32)

for i in range(10):
    b = fwht(a)
