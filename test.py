import math

import einops
import torch

from fwht._fwht_triton import fwht
from fwht._fhwt_triton_base import (
    hadamard_16_f32
)

def _build_hadamard_2(device):
    return torch.tensor([[1, 1], [1, -1]], device=device)

def _build_hadamard_256(device):
    H_16 = hadamard_16_f32(device)
    return torch.kron(H_16, H_16)

def _build_hadamard_512(device):
    H_2 = _build_hadamard_2(device)
    H_16 = hadamard_16_f32(device)
    return torch.kron(H_2, torch.kron(H_16, H_16))

def _build_hadamard_4096(device):
    H_16 = hadamard_16_f32(device)
    return torch.kron(H_16, torch.kron(H_16, H_16))

_INPUT_GENERATORS = (
    torch.ones,
    lambda size: -torch.ones(size),
    torch.randn,
    torch.rand,
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

def test_fwht_256():
    H_256 = _build_hadamard_256('cpu')
    for gen in _INPUT_GENERATORS:
        a = gen((8, 256))
        expected1 = einops.einsum(H_256, a, 'r c, b c -> b r')
        expected2 = _reference_fwht(a.clone())
        actual = fwht(a)
        assert torch.allclose(expected1, actual, atol=1e-2)
        assert torch.allclose(expected2, actual, atol=1e-2)

def test_fwht_512():
    H_512 = _build_hadamard_512('cpu')
    for gen in _INPUT_GENERATORS:
        a = gen((8, 512))
        expected1 = einops.einsum(H_512, a, 'r c, b c -> b r')
        expected2 = _reference_fwht(a.clone())
        actual = fwht(a)
        print(expected1)
        print(expected2)
        print(actual)
        
        assert torch.allclose(expected1, actual, atol=1e-2)
        assert torch.allclose(expected2, actual, atol=1e-2)

def test_fwht_4096():
    H_4096 = _build_hadamard_4096('cpu')
    for gen in _INPUT_GENERATORS:
        a = gen((8, 4096))
        expected = einops.einsum(H_4096, a, 'r c, b c -> b r')
        actual = fwht(a)
        assert torch.allclose(expected, actual, atol=1e-2)
        
def test_fwht_4096_scale():
    scale = 1e-4
    H_4096 = scale * _build_hadamard_4096('cpu')
    for gen in _INPUT_GENERATORS:
        a = gen((8, 4096))
        expected = einops.einsum(H_4096, a, 'r c, b c -> b r')
        actual = fwht(a, scale=scale)
        assert torch.allclose(expected, actual, atol=1e-2)
        
        wrong = fwht(a, scale=1)
        assert not torch.allclose(expected, wrong, atol=1e-2)