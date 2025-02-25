import math
import torch
import triton
import triton.language as tl

from fwht._fhwt_triton_base import (
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
def load_H_ptr(H_ptr, SIZE: tl.constexpr):
    H_block_ptr = tl.make_block_ptr(
        base=H_ptr,
        shape=(SIZE, SIZE),
        strides=(SIZE, 1),
        offsets=(0, 0),
        block_shape=(SIZE, SIZE),
        order=(1, 0)
    )
    H = tl.load(H_block_ptr, eviction_policy="evict_last")
    return H

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
        #triton.Config(kwargs={}, num_warps=2),
        triton.Config(kwargs={}, num_warps=4),
        #triton.Config(kwargs={}, num_warps=8)
    ],
    key=['WORK_SIZE'])
@triton.jit
def fwht_256_kernel(
    a_ptr,
    scale,
    base_ptr,
    H_power_of_2_ptr,
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
    
    base = load_H_ptr(base_ptr, BASE_SIZE)
    
    BASE_SIZE_SQUARED: tl.constexpr = BASE_SIZE ** 2
    BASE_SIZE_CUBED: tl.constexpr = BASE_SIZE ** 3
        
    # case 2: kron(base, base)a
    if BASE_SIZE_SQUARED <= WORK_SIZE:
        a = fwht_256_2step_kernel(a, base, WORK_SIZE, BASE_SIZE)
    
    # case 3: using result of case 2, kron(base, kron(base, base))a
    if BASE_SIZE_CUBED <= WORK_SIZE:
        BATCH_SIZE: tl.constexpr = WORK_SIZE // BASE_SIZE_CUBED
        mat = a.reshape(BATCH_SIZE, BASE_SIZE, BASE_SIZE_SQUARED)
        mat = tl.dot(
            base.expand_dims(0).broadcast_to(BATCH_SIZE, BASE_SIZE, BASE_SIZE), 
            mat,
            out_dtype=a.dtype
        )
        a = mat.reshape(WORK_SIZE)

    # the three prior cases only work for powers of 16,
    # this next step lets us work with more general powers of 2.
    if POWER_OF_2 > 1:
        H = load_H_ptr(H_power_of_2_ptr, POWER_OF_2)
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

    # ideally hadamard matrices would be constructed in the kernel as needed...
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
    
    # if the input is really small, can just directly apply hadamard matrix.
    if a_size == 16:
        a_flat.copy_(torch.mm(a_flat, H_16))
        return a
    elif a_size in (2, 4, 8):
        a_flat.copy_(torch.mm(a_flat, H_power_of_2))
        return a
    
    if work_size in (32, 64, 128):
        pass

    grid = (a_flat.size(0),) 
    fwht_256_kernel[grid](
        a_flat,
        scale,
        base_ptr=H_16,
        H_power_of_2_ptr=H_power_of_2,
        IN_SIZE=a_size,
        WORK_SIZE=work_size,
        BASE_SIZE=16,
        POWER_OF_2=power_of_2,
    )
    
    return a