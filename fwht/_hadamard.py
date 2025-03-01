import torch
import triton
import triton.language as tl
    
def _reference_fwht(a, scale=1.0):
    r"""
    Copied from wikipedia:
    https://en.wikipedia.org/wiki/Fast_Walsh–Hadamard_transform
    """
    h = 1
    size = a.size(-1)
    while h < size:
        for i in range(0, size, h * 2):
            for j in range(i, i + h):
                x = a[..., j].clone()
                y = a[..., j + h].clone()
                a[..., j] = x + y
                a[..., j + h] = x - y
        h *= 2
    return a * scale

def hadamard(n, device):
    return _reference_fwht(torch.eye(n, device=device, dtype=torch.float64))