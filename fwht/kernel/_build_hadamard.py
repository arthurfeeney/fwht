import triton
import triton.language as tl

@triton.jit
def build_H(SIZE: tl.constexpr, dtype: tl.constexpr):
    r"""
    Construct small Hadamard matrices, in such a way that Triton can optimize the code away.
    This uses the identity $H_{i,j} = (-1)^{i \cdot j}$, 
    where the operation $\cdot$ is the BITWISE dot product of integers.
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