import math
from functools import partial

import einops
import torch

from fwht import fast_hadamard_transform
from fwht._fhwt_triton_base import (
    hadamard_16
)

DEVICE = 'cuda'

def _build_hadamard_2(device):
    return torch.tensor([[1, 1], [1, -1]], device=device, dtype=torch.float64)

def _build_hadamard_16(device):
    H_16 = hadamard_16(device=device, dtype=torch.float64)
    return H_16

def _build_hadamard_256(device):
    H_16 = hadamard_16(device=device, dtype=torch.float64)
    return torch.kron(H_16, H_16)

def _build_hadamard_512(device):
    H_2 = _build_hadamard_2(device)
    H_16 = hadamard_16(device=device, dtype=torch.float64)
    return torch.kron(H_2, torch.kron(H_16, H_16))

def _build_hadamard_1024(device):
    H_2 = _build_hadamard_2(device)
    H_512 = _build_hadamard_512(device)
    return torch.kron(H_2, H_512)

def _build_hadamard_2048(device):
    H_2 = _build_hadamard_2(device)
    H_1024 = _build_hadamard_1024(device)
    return torch.kron(H_2, H_1024)

def _build_hadamard_4096(device):
    H_16 = hadamard_16(device=device, dtype=torch.float64)
    return torch.kron(H_16, torch.kron(H_16, H_16))

def _build_hadamard_8192(device):
    H_2 = _build_hadamard_2(device)
    H_4096 = _build_hadamard_4096(device)
    return torch.kron(H_2, H_4096)

def rand_ones(size, device):
    zeros = torch.zeros(size, device=device)
    batch_size = size[0]
    nonzero = size[1] // 3
    indices = torch.randint(0, size[1] - 1, (nonzero,), device=device)
    zeros[:, indices] = 1
    return zeros

_INPUT_GENERATORS = (
    (partial(torch.ones, device=DEVICE), 1e-3),
    (lambda size: -torch.ones(size, device=DEVICE), 1e-3),
    (partial(torch.randn, device=DEVICE), 1),
    (partial(rand_ones, device=DEVICE), 1e-3)
    #(partial(torch.rand, device=DEVICE), 1),
)

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

def _recursive_fwht(a):
    size = a.size(-1)
    if size == 1:
        return a
    left = a[..., :size // 2]
    right = a[..., size // 2:]
    a_left = _recursive_fwht(left).clone()
    a_right = _recursive_fwht(right).clone()
    a[..., :size // 2] = a_left + a_right
    a[..., size // 2:] = a_left - a_right
    return a

def test_reference_recursive():
    a = torch.tensor([1, 0, 1, 0, 0, 1, 1, 0], dtype=torch.float32)
    rec = _recursive_fwht(a.clone())
    ref = _reference_fwht(a.clone())
    assert torch.allclose(rec, ref)
    
def test_reference_wikipedia():
    a = torch.tensor([1, 0, 1, 0, 0, 1, 1, 0], dtype=torch.float32)
    b = torch.tensor([4, 2, 0, -2, 0, 2, 0, 2], dtype=torch.float32)
    for ref in (_recursive_fwht, _reference_fwht):
        assert torch.allclose(ref(a.clone()), b)

def test_reference_orthogonal():
    for size in [8, 16, 32, 64]:
        a = torch.eye(size, dtype=torch.float32)
        for ref in (_recursive_fwht, _reference_fwht):
            H = ref(a.clone())
            assert torch.allclose(H @ H.T, a * size) 

def fwht_wrapper(size, H, scale=1):
    for gen, atol in _INPUT_GENERATORS:
        a = gen((8, size))
        #expected1 = einops.einsum(H, a.double(), 'r c, b c -> b r').float()
        expected2 = _reference_fwht(a.clone()) * scale
        actual = fast_hadamard_transform(a, scale)
        #assert torch.allclose(expected1, actual, atol=atol * scale)
        assert torch.allclose(expected2, actual, atol=atol * scale)

def test_fwht_256():
    H_256 = _build_hadamard_256(DEVICE)
    fwht_wrapper(256, H_256)

def test_fwht_512():
    H_512 = _build_hadamard_512(DEVICE)
    fwht_wrapper(512, H_512)

def test_fwht_1024():
    H_1024 = _build_hadamard_1024(DEVICE)
    fwht_wrapper(1024, H_1024)
    
def test_fwht_2048():
    H_2048 = _build_hadamard_2048(DEVICE)
    fwht_wrapper(2048, H_2048)

def test_fwht_4096():
    H_4096 = _build_hadamard_4096(DEVICE)
    fwht_wrapper(4096, H_4096)
        
def test_fwht_4096_scale():
    scale = 1e-4
    H_4096 = scale * _build_hadamard_4096(DEVICE)
    fwht_wrapper(4096, H_4096, scale)
    
def test_fwht_8192_scale():
    H_8192 = _build_hadamard_4096(DEVICE)
    fwht_wrapper(8192, H_8192)
    
def right_zero_pad(a, size):
    zeros = torch.zeros(a.size(0), size, device=DEVICE)
    zeros[:, :a.size(1)] = a
    return zeros
    
def test_fwht_276_implicit_pad():
    size = 272
    H = _build_hadamard_512(DEVICE)
    a = torch.ones((2, size), device=DEVICE)
    expected1 = einops.einsum(
        H, right_zero_pad(a.clone(), 512).double(), 'r c, b c -> b r').float()
    actual = fast_hadamard_transform(a)
    assert torch.allclose(expected1[:, :size], actual, atol=1e-3)
    
def test_fwht_4096_f16():
    size = 4096
    scale = 1
    a = torch.randn(8, size, device=DEVICE, dtype=torch.float16)
    expected = _reference_fwht(a.clone())
    actual = fast_hadamard_transform(a, scale)
    assert torch.allclose(expected, actual, atol=1 * scale)

def test_fwht_4096_backward():
    size = 4096
    scale = 1
    a = torch.randn(8, size, device=DEVICE, dtype=torch.float16, requires_grad=True)
    out = fast_hadamard_transform(a, scale)
    loss = out.sum()
    loss.backward()
    
def test_fwht_small():
    for size in (2, 4, 8, 16):
        a = rand_ones((8, size), device=DEVICE).to(torch.float32)
        expected = _reference_fwht(a.clone())
        actual = fast_hadamard_transform(a)
        assert torch.allclose(expected, actual, atol=1e-3)