import math
import torch
import triton
import triton.language as tl

from fwht._fhwt_triton_base import (
    hadamard_16_f32,
    hadamard_8_f32,
    hadamard_4_f32,
    hadamard_2_f32
)

@triton.jit
def load_H_ptr(H_ptr, size: tl.constexpr):
    H_block_ptr = tl.make_block_ptr(
            base=H_ptr,
            shape=(size, size),
            strides=(size, 1),
            offsets=(0, 0),
            block_shape=(size, size),
            order=(1, 0)
        )
    H = tl.load(H_block_ptr)
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
    left = tl.dot(br, ar)
    return tl.dot(left, br).reshape(A_SIZE)
    
@triton.jit
def fwht_256_kernel(
    a_ptr,
    a_batch_stride,
    a_vector_stride,
    scale,
    base_ptr,
    IN_SIZE: tl.constexpr,
    A_SIZE: tl.constexpr,
    BASE_SIZE: tl.constexpr,
    # TODO: these may not get used, but IDK how else
    # to access them inside the kernel...
    H_8_ptr,
    H_4_ptr,
    H_2_ptr
):
    tl.static_assert(A_SIZE >= 256)
    tl.static_assert(A_SIZE <= (2 ** 3) * (16 ** 3))
    tl.static_assert(A_SIZE % BASE_SIZE == 0)
    tl.static_assert(A_SIZE >= IN_SIZE)
    
    batch_idx = tl.program_id(axis=0)
    a_ptrs = a_ptr + batch_idx * IN_SIZE + (tl.arange(0, A_SIZE) % IN_SIZE)
    mask = tl.arange(0, A_SIZE) < IN_SIZE    
    a = tl.load(a_ptrs, mask=mask, other=0)
    
    base_block_ptr = tl.make_block_ptr(
        base=base_ptr,
        shape=(BASE_SIZE, BASE_SIZE),
        strides=(BASE_SIZE, 1),
        offsets=(0, 0),
        block_shape=(BASE_SIZE, BASE_SIZE),
        order=(1, 0)
    )
    base = tl.load(base_block_ptr)
    
    a = fwht_256_2step_kernel(a, base, A_SIZE, BASE_SIZE)
    
    BASE_SIZE_SQUARED: tl.constexpr = BASE_SIZE ** 2
    BASE_SIZE_CUBED: tl.constexpr = BASE_SIZE ** 3
    if BASE_SIZE_CUBED <= A_SIZE:
        mat = a.reshape(A_SIZE // BASE_SIZE_CUBED, BASE_SIZE, BASE_SIZE_SQUARED)
        mat = tl.dot(
            base.expand_dims(0).broadcast_to(A_SIZE // BASE_SIZE_CUBED, BASE_SIZE, BASE_SIZE), 
            mat
        )
        a = mat.reshape(A_SIZE)

    POWER_OF_16 = 1
    while POWER_OF_16 * 16 <= A_SIZE:
        POWER_OF_16 *= 16
      
    POWER_OF_2 = A_SIZE // POWER_OF_16
    tl.device_assert(POWER_OF_2 >= 1 and POWER_OF_2 <= 8)
    
    if POWER_OF_2 > 1:
        # 1. This uses cuda cores because I can't figure out
        #    the block-diagonal version using tensor cores...
        # 2. POWER_OF_2 could be constexpr, and so the if-else
        #    chain can be condensend. But it's annoying :-)
        if POWER_OF_2 == 8:
            H8 = load_H_ptr(H_8_ptr, 8)
            mat8 = a.reshape(8, A_SIZE // 8)
            mat8 = tl.sum(H8[:, :, None] * mat8[None, :, :], axis=1)      
            a = mat8.reshape(A_SIZE)
        elif POWER_OF_2 == 4:
            H4 = load_H_ptr(H_4_ptr, 4)
            mat4 = a.reshape(4, A_SIZE // 4)
            mat4 = tl.sum(H4[:, :, None] * mat4[None, :, :], axis=1)      
            a = mat4.reshape(A_SIZE)
        else:
            H2 = load_H_ptr(H_2_ptr, 2)
            mat2 = a.reshape(2, A_SIZE // 2)
            mat2 = tl.sum(H2[:, :, None] * mat2[None, :, :], axis=1)      
            a = mat2.reshape(A_SIZE)

    tl.store(a_ptrs, a * scale, mask=mask)

def fwht_inplace(a, scale=None):
    r"""
    Computes a Fast Walsh-Hadamard transform of `a` in-place.
    We assume it is applied along the last dimension of a.
    Args:
        a: [..., v]
        scale: Optional normalizing factor. Note: this is applied after
        the full transform has been computed, rather than in each
        step.
    """
    if scale is None:
        scale = 1

    a_flat = a.view(-1, a.size(-1))
    a_size = a_flat.size(1)
    
    # ideally these would be constructed in the kernel as needed...
    base_matrix = hadamard_16_f32(a.device)
    H_8 = hadamard_8_f32(a.device)
    H_4 = hadamard_4_f32(a.device)
    H_2 = hadamard_2_f32(a.device)
    
    grid = (a_flat.size(0),) 
    fwht_256_kernel[grid](
        a_flat,
        a_flat.stride(0),
        a_flat.stride(1),
        scale,
        base_matrix,
        IN_SIZE=a_size,
        A_SIZE=int(2 ** math.ceil(math.log2(a_size))),
        BASE_SIZE=16,
        H_8_ptr=H_8,
        H_4_ptr=H_4,
        H_2_ptr=H_2
    )
    
def fwht(a, scale=None):
    cpy = a.clone()
    fwht_inplace(cpy, scale)
    return cpy