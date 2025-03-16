import math

import triton
import triton.language as tl

from ._build_hadamard import build_H

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
    tl.static_assert(WORK_SIZE >= 16)
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
        tl.static_assert(WORK_SIZE % BASE_SIZE_SQUARED == 0)
        a = fwht_256_2step_kernel(a, base, WORK_SIZE, BASE_SIZE)
    
    # case 2: using result of case 2, kron(base, kron(base, base))a
    if BASE_SIZE_CUBED <= WORK_SIZE:
        tl.static_assert(WORK_SIZE % BASE_SIZE_CUBED == 0)
        BATCH_SIZE: tl.constexpr = WORK_SIZE // BASE_SIZE_CUBED
        mat = a.reshape(BATCH_SIZE, BASE_SIZE, BASE_SIZE_SQUARED)
        mat = tl.dot(
            base.expand_dims(0).broadcast_to(BATCH_SIZE, BASE_SIZE, BASE_SIZE), 
            mat,
            out_dtype=a.dtype
        )
        a = mat.reshape(WORK_SIZE)

    # use cuda cores for smaller cases than 256
    if WORK_SIZE < BASE_SIZE_SQUARED:
        INNER_SIZE: tl.constexpr = WORK_SIZE // BASE_SIZE    
        ar = a.reshape(INNER_SIZE, BASE_SIZE)
        ar = tl.sum(ar[:, :, None] * base[None, :, :], axis=1)
        a = ar.reshape(WORK_SIZE)

    # the prior cases only work for powers of BASE_SIZE,
    # this step lets us work with more general powers of 2.
    if POWER_OF_2 > 1:
        H = build_H(POWER_OF_2, a.dtype)
        mat = a.reshape(POWER_OF_2, WORK_SIZE // POWER_OF_2)
        mat = tl.sum(H[:, :, None] * mat[None, :, :], axis=1)      
        a = mat.reshape(WORK_SIZE)

    tl.store(a_ptrs, a * scale, mask=mask)

def power_of_16_less_than(n):
    assert n > 0
    assert n < 16 ** 4
    if n < 16: return 1
    if n < 256: return 16
    if n < 4096: return 256
    # we do not support powers of 16 above 16**3
    else: return 4096
    
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