# torch-fwht

This is an implementation of the Fast Walsh-Hadamard Transform (FWHT)
in triton.

This is a work-in-progress!

## Notes on Implementation

The implementation relies on two things:
1. Hadamard transforms `H_pq` can be expressed as the
   Kronecker product of two smaller Hadamard transforms `H_pq = kron(H_p, H_q)`.
2. The matrix-vector product of a kronecker product satisfies 
   `kron(B^T, A) col_vec(m)` = `col_vec(AmB)`, where `col_vec` unravels 
   the columns of a matrix to form a vector. (So, basically the kronecker 
   product can be implemented using GEMMs and that's good because triton likes
   big beautiful matrix multiplications.) Since Hadamard matrices are symmetric, you can just
   use `kron(H, H) col_vec(m) = col_vec(HmH)` and `kron(H, H) row_vec(m) = row_vec(HmH)`, 
   without worrying about transposes.


These ideas basically yoinked from https://arxiv.org/pdf/1304.7054.

[HadaCore](https://arxiv.org/pdf/2412.08832v1) seems to be doing pretty much the same strategy, but does way more low-level stuff.

## Caveats

1. If the input is not a power of 2, the kernel has to explicitly zero pad it to the next power of 2. This happens inside the kernel, so it does not need to allocate in global memory, just doing extra compute. Triton requires block sizes to be powers of 2, so I'm not sure if there's a way to work around this.
2. This assumes the input can be factorized as a power of 16 times a power of 2. (Meaning the input size needs to be `16^m * 2^n`.)
3. The maximum supported size is currently `16**3 * 2 ** 3`, which allows things to fit in GPU shared memory. Going beyond this size is likely uncommon in machine learning.