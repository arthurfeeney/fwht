import math
from functools import partial

import einops
import torch

from fwht import fast_hadamard_transform
from fwht._hadamard import (
    _reference_fwht,
    hadamard
)

DEVICE = 'cuda'

def rand_ones(size, device):
    zeros = torch.zeros(size, device=device)
    nonzero = size[1] // 3
    indices = torch.randint(0, size[1] - 1, (nonzero,), device=device)
    zeros[:, indices] = 1
    return zeros

_INPUT_GENERATORS = [
    (partial(torch.ones, device=DEVICE), 1e-3),
    (lambda size: -torch.ones(size, device=DEVICE), 1e-3),
    (partial(torch.randn, device=DEVICE), 1),
    (partial(rand_ones, device=DEVICE), 1e-3)
]

_SCALES = [
    
]

def test_reference_orthogonal():
    for size in [8, 16, 32, 64]:
        a = torch.eye(size, dtype=torch.float32, device=DEVICE)
        H = hadamard(size, DEVICE).float()
        assert torch.allclose(H @ H.T, a * size) 

def fwht_wrapper(size, H, scale=1):
    for gen, atol in _INPUT_GENERATORS:
        a = gen((8, size))
        expected1 = einops.einsum(H, a.double(), 'r c, b c -> b r').float() * scale
        expected2 = _reference_fwht(a.clone(), scale=scale)
        actual = fast_hadamard_transform(a, scale)
        assert torch.allclose(expected1, actual, atol=atol * scale)
        assert torch.allclose(expected2, actual, atol=atol * scale)

def test_fwht_256():
    H_256 = hadamard(256, DEVICE)
    fwht_wrapper(256, H_256)

def test_fwht_512():
    H_512 = hadamard(512, DEVICE)
    fwht_wrapper(512, H_512)

def test_fwht_1024():
    H_1024 = hadamard(1024, DEVICE)
    fwht_wrapper(1024, H_1024)
    
def test_fwht_2048():
    H_2048 = hadamard(2048, DEVICE)
    fwht_wrapper(2048, H_2048)

def test_fwht_4096():
    H_4096 = hadamard(4096, DEVICE)
    fwht_wrapper(4096, H_4096)
        
def test_fwht_4096_scale():
    scale = 1e-4
    H_4096 = hadamard(4096, DEVICE)
    fwht_wrapper(4096, H_4096, scale)
    
def test_fwht_8192_scale():
    H_8192 = hadamard(8192, DEVICE)
    fwht_wrapper(8192, H_8192)
    
def right_zero_pad(a, size):
    zeros = torch.zeros(a.size(0), size, device=DEVICE)
    zeros[:, :a.size(1)] = a
    return zeros
    
def test_fwht_276_implicit_pad():
    size = 272
    H = hadamard(512, DEVICE)
    a = torch.ones((2, size), device=DEVICE)
    expected1 = einops.einsum(
        H, right_zero_pad(a.clone(), 512).double(), 'r c, b c -> b r').float()
    actual = fast_hadamard_transform(a)
    assert torch.allclose(expected1[:, :size], actual, atol=1e-3)

"""    
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
        
def test_fwht_medium():
    for size in (32,):
        a = rand_ones((8, size), device=DEVICE).to(torch.float32)
        expected = _reference_fwht(a.clone())
        actual = fast_hadamard_transform(a)
        assert torch.allclose(expected, actual, atol=1e-3)
"""