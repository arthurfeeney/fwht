# triton-fwht

This is an implementation of the Fast Walsh-Hadamard Transform (FWHT)
in triton.

This is a work-in-progress.

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

Combining these, you can basically implement larger Hadamard transforms using
smaller ones as base cases. I.e., `H_256 a = kron(H_16, H_16) a = col_vec(H_16 mat(a) H_16)`
(where `mat(a)` is basically reshaping the vector `a` into a matrix). This basically just
repeats: `H_4096 a = kron(H_16, kron(H_16, H_16)) a`.

This can also be used to make funkier sizes: 
    - `H_512 a = kron(H_2, kron(H_16, H_16)) a`
    - `H_128 a = kron(H_8, H_16) a`

These ideas basically yoinked from https://arxiv.org/pdf/1304.7054 

[HadaCore](https://arxiv.org/pdf/2412.08832v1) seems to be doing pretty much the same strategy,
but does way more low-level stuff. (I also copy the way the handle powers of 2---doing an extra iteration with smaller block diagonal matrices). The implementation from [Dao-AILab](https://github.com/Dao-AILab/fast-hadamard-transform/tree/master) seems to work with chunks of size 8 and does not use tensor cores.