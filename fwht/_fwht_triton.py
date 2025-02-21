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
def fwht_kernel(
    a_ptr,
    a_batch_stride,
    a_vector_stride,
    scale,
    base_ptr,
    A_SIZE: tl.constexpr,
    BASE_SIZE: tl.constexpr,
    # TODO: these may not get used, but IDK how else
    # to access them inside the kernel...
    H_8_ptr,
    H_4_ptr,
    H_2_ptr
):
    tl.static_assert(A_SIZE >= 256,
                     msg="The input should be at least 16*16==256")
    tl.static_assert(A_SIZE)
    tl.static_assert(A_SIZE % BASE_SIZE == 0,
                     msg="A_SIZE must be a multiple of BASE_SIZE")
 
    batch_idx = tl.program_id(axis=0)
    a_ptrs = a_ptr + batch_idx * A_SIZE + tl.arange(0, A_SIZE)
    a = tl.load(a_ptrs)

    # TODO: ideally, base should just be like a constexpr thing...?
    # it just has a fixed structure, so it could just get plopped into
    # shared memory and/or just be created in registers...
    base_block_ptr = tl.make_block_ptr(
        base=base_ptr,
        shape=(BASE_SIZE, BASE_SIZE),
        strides=(BASE_SIZE, 1),
        offsets=(0, 0),
        block_shape=(BASE_SIZE, BASE_SIZE),
        order=(1, 0)
    )
    base = tl.load(base_block_ptr)
    
    a = tl.dot(
        a.reshape((A_SIZE // BASE_SIZE, BASE_SIZE)), base
    ).reshape(A_SIZE)

    BASE_SIZE_SQUARED: tl.constexpr = BASE_SIZE ** 2
    if A_SIZE >= BASE_SIZE_SQUARED:
        mat = a.reshape(A_SIZE // BASE_SIZE_SQUARED, BASE_SIZE, BASE_SIZE)
        mat = tl.dot(base.expand_dims(0).broadcast_to(A_SIZE // BASE_SIZE_SQUARED, BASE_SIZE, BASE_SIZE), 
                     mat)
        a = mat.reshape(A_SIZE)

    BASE_SIZE_CUBED: tl.constexpr = BASE_SIZE ** 3
    if A_SIZE >= BASE_SIZE_CUBED:
        mat = a.reshape(A_SIZE // BASE_SIZE_CUBED, BASE_SIZE, BASE_SIZE_SQUARED)
        mat = tl.dot(base.expand_dims(0).broadcast_to(A_SIZE // BASE_SIZE_CUBED, BASE_SIZE, BASE_SIZE), 
                     mat)
        a = mat.reshape(A_SIZE)

    # if A_SIZE is not a power of 16, we do one more step using a power of 2.
    largest_power_of_16 = 1
    while largest_power_of_16 * 16 <= A_SIZE:
        largest_power_of_16 = largest_power_of_16 * 16
    REMAINING_POWER_OF_2 = A_SIZE // largest_power_of_16
    tl.static_assert(REMAINING_POWER_OF_2 in (1, 2, 4, 8))
    
    if REMAINING_POWER_OF_2 > 1:
        if REMAINING_POWER_OF_2 == 8:
            H_rem_ptr = H_8_ptr
        elif REMAINING_POWER_OF_2 == 4:
            H_rem_ptr = H_4_ptr
        else:
            H_rem_ptr = H_2_ptr
            
        H_block_ptr = tl.make_block_ptr(
            base=H_rem_ptr,
            shape=(BASE_SIZE, BASE_SIZE),
            strides=(BASE_SIZE, 1),
            offsets=(0, 0),
            block_shape=(BASE_SIZE, BASE_SIZE),
            order=(1, 0)
        )
        H = tl.load(H_block_ptr)
        a = tl.dot(
            #a.reshape(A_SIZE // BASE_SIZE, BASE_SIZE), H
            H.expand_dims(0).broadcast_to(A_SIZE // BASE_SIZE_SQUARED, BASE_SIZE, BASE_SIZE),
            a.reshape(A_SIZE // BASE_SIZE_SQUARED, BASE_SIZE, BASE_SIZE).trans(0, 2, 1)
        ).trans(0, 2, 1).reshape(A_SIZE)
            
    tl.store(a_ptrs, a * scale)

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
    
    # ideally these will be constructed in the kernel...
    base_matrix = hadamard_16_f32(a.device)
    H_8 = hadamard_8_f32(a.device)
    H_4 = hadamard_4_f32(a.device)
    H_2 = hadamard_2_f32(a.device)
    
    grid = (a_flat.size(0),) 
    fwht_kernel[grid](
        a_flat,
        a_flat.stride(0),
        a_flat.stride(1),
        scale,
        base_matrix,
        A_SIZE=a_size,
        BASE_SIZE=16,
        H_8_ptr=H_8,
        H_4_ptr=H_4,
        H_2_ptr=H_2
    )
    
def fwht(a, scale=None):
    cpy = a.clone()
    fwht_inplace(cpy, scale)
    return cpy