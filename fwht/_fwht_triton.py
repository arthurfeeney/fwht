import math
import torch
import triton
import triton.language as tl

from fwht._hadamard import (
    hadamard_16,
    hadamard_8,
    hadamard_4,
    hadamard_2
)

def power_of_16_less_than(n):
    assert n > 0
    assert n < 16 ** 4
    if n < 16: return 1
    if n < 256: return 16
    if n < 4096: return 256
    # we do not support powers of 16 above 16**3
    else: return 4096

@triton.jit
def build_H(SIZE: tl.constexpr, dtype: tl.constexpr):
    r"""
    Construct small Hadamard matrices, in such a way that Triton can optimize the code away.
    This uses the identity $H_{i,j} = (-1)^{i \cdot j}$, 
    where the operation $\cdot$ is the BITWISE dot product of integers.
    This is necessary because
        1. Triton does not allow manually writing out the tensor inside a kernel.
        2. This avoids an additional load from global memory and seems to make the
            kernels a bit faster.
    """
    tl.static_assert(0 < SIZE)
    tl.static_assert(SIZE <= 16)

    i = tl.arange(0, SIZE)
    j = tl.arange(0, SIZE)    
    matching_bits = (i[:, None] & j)

    bit_sum = tl.zeros_like(matching_bits)
    for i in tl.static_range(5):
        bit_sum += matching_bits & 1
        matching_bits >>= 1

    # map odd to -1, even to 1
    H = 2 * ((bit_sum % 2) == 0) - 1
    return H.cast(dtype)

@triton.jit
def fwht_small_kernel(
    a_ptr,
    scale,
    batch_size: int,
    CHUNK_SIZE: tl.constexpr,
    IN_SIZE: tl.constexpr,
    WORK_SIZE: tl.constexpr,
    BASE_SIZE: tl.constexpr,
    POWER_OF_2: tl.constexpr,
):
    r"""
    Kernel for small fwht of size 32, 64, or 128).
    """
    tl.static_assert(WORK_SIZE >= 16)
    tl.static_assert(WORK_SIZE % BASE_SIZE == 0)
    tl.static_assert(WORK_SIZE >= IN_SIZE)
    
    batch_idx = tl.program_id(axis=0) * CHUNK_SIZE * IN_SIZE
    
    chunk_idx = batch_idx * CHUNK_SIZE * IN_SIZE + (tl.arange(0, CHUNK_SIZE) * IN_SIZE)[:, None]
    a_idx = tl.arange(0, WORK_SIZE)[None, :]
    a_ptrs = (chunk_idx % batch_size) + (a_idx % IN_SIZE)
    mask = chunk_idx < batch_size and a_idx < IN_SIZE
    
    a = tl.load(a_ptr, mask=mask, other=0.0)
    base = build_H(BASE_SIZE, a.dtype)
    
    inner_size: tl.constexpr = WORK_SIZE // BASE_SIZE
    ar = a.reshape(CHUNK_SIZE, inner_size, BASE_SIZE)
    br = base.expand_dims(0).broadcast_to(inner_size, BASE_SIZE, BASE_SIZE)
    a = tl.sum(ar[:, :, None] * br[None, :, :], axis=1)

    if POWER_OF_2 > 1:
        H = build_H(POWER_OF_2, a.dtype)
        inner_size = WORK_SIZE // POWER_OF_2
        mat = a.reshape(CHUNK_SIZE, POWER_OF_2, inner_size)
        mat = tl.sum(H[:, :, None] * mat[None, :, :], axis=1)      
        a = mat.reshape(WORK_SIZE)

    tl.store(a_ptrs, a * scale, mask=mask)

@triton.jit
def fwht_256_2step_kernel(
    a: tl.tensor,
    base: tl.tensor,
    A_SIZE: tl.constexpr,
    BASE_SIZE: tl.constexpr
):
    batch_size: tl.constexpr = A_SIZE // (BASE_SIZE ** 2)
    ar = a.reshape(batch_size, BASE_SIZE, BASE_SIZE)
    br = base.expand_dims(0).broadcast_to(batch_size, BASE_SIZE, BASE_SIZE)
    left = tl.dot(br, ar, out_dtype=a.dtype)
    return tl.dot(left, br, out_dtype=a.dtype).reshape(A_SIZE)
    
@triton.autotune(configs=[
        triton.Config(kwargs={}, num_warps=4),
    ],
    key=['WORK_SIZE'])
@triton.jit
def fwht_256_kernel(
    a_ptr,
    scale,
    IN_SIZE: tl.constexpr,
    WORK_SIZE: tl.constexpr,
    BASE_SIZE: tl.constexpr,
    POWER_OF_2: tl.constexpr,
):
    tl.static_assert(WORK_SIZE >= 256)
    tl.static_assert(WORK_SIZE <= (2 ** 3) * (16 ** 3))
    tl.static_assert(WORK_SIZE % BASE_SIZE == 0)
    tl.static_assert(WORK_SIZE >= IN_SIZE)
    
    batch_idx = tl.program_id(axis=0)
    a_ptrs = a_ptr + batch_idx * IN_SIZE + (tl.arange(0, WORK_SIZE) % IN_SIZE)
    mask = tl.arange(0, WORK_SIZE) < IN_SIZE
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    
    base = build_H(BASE_SIZE, a.dtype)
    
    BASE_SIZE_SQUARED: tl.constexpr = BASE_SIZE ** 2
    BASE_SIZE_CUBED: tl.constexpr = BASE_SIZE ** 3
    
    # case 1: kron(base, base)a
    if BASE_SIZE_SQUARED <= WORK_SIZE:
        a = fwht_256_2step_kernel(a, base, WORK_SIZE, BASE_SIZE)
    
    # case 2: using result of case 2, kron(base, kron(base, base))a
    if BASE_SIZE_CUBED <= WORK_SIZE:
        BATCH_SIZE: tl.constexpr = WORK_SIZE // BASE_SIZE_CUBED
        mat = a.reshape(BATCH_SIZE, BASE_SIZE, BASE_SIZE_SQUARED)
        mat = tl.dot(
            base.expand_dims(0).broadcast_to(BATCH_SIZE, BASE_SIZE, BASE_SIZE), 
            mat,
            out_dtype=a.dtype
        )
        a = mat.reshape(WORK_SIZE)

    # the prior cases only work for powers of 16,
    # this step lets us work with more general powers of 2.
    if POWER_OF_2 > 1:
        H = build_H(POWER_OF_2, a.dtype)
        mat = a.reshape(POWER_OF_2, WORK_SIZE // POWER_OF_2)
        mat = tl.sum(H[:, :, None] * mat[None, :, :], axis=1)      
        a = mat.reshape(WORK_SIZE)

    tl.store(a_ptrs, a * scale, mask=mask)

def fwht(
    a, 
    scale=1.0,
    inplace=False
):
    if not inplace:
        a = a.clone()
    
    a_flat = a.view(-1, a.size(-1))
    a_size = a_flat.size(1)

    assert a_size >= 2
    assert a_size <= 16 ** 3 * 2 ** 3        
    
    # next power of 2 larger than a_size
    work_size = int(2 ** math.ceil(math.log2(a_size)))
    power_of_16 = power_of_16_less_than(work_size)
    power_of_2 = work_size // power_of_16
    assert power_of_2 in (1, 2, 4, 8)

    H_16 = hadamard_16(a.dtype, a.device)
    lookup = {
        8: hadamard_8,
        4: hadamard_4,
        2: hadamard_2,
    }
    if power_of_2 == 1:
        H_power_of_2 = None
    else:
        H_power_of_2 = lookup[power_of_2](a.dtype, a.device)
    
    # Small sizes all memory bound, just directly apply hadamard matrix.
    if a_size == 16:
        a_flat.copy_(torch.mm(a_flat, H_16))
        return a
    elif a_size in (2, 4, 8):
        a_flat.copy_(torch.mm(a_flat, H_power_of_2))
        return a

    if work_size in (32, 64, 128):
        CHUNK_SIZE = 16
        grid = (a_flat.size(0) // CHUNK_SIZE, work_size)
        fwht_small_kernel[grid](
            a_flat,
            scale,
            base_ptr=H_16,
            H_power_of_2_ptr=H_power_of_2,
            IN_SIZE=a_size,
            WORK_SIZE=work_size,
            BASE_SIZE=16,
            POWER_OF_2=power_of_2,
            CHUNK_SIZE=CHUNK_SIZE
        )
        return a

    grid = (a_flat.size(0),) 
    fwht_256_kernel[grid](
        a_flat,
        scale,
        IN_SIZE=a_size,
        WORK_SIZE=work_size,
        BASE_SIZE=16,
        POWER_OF_2=power_of_2,
    )
    
    return a