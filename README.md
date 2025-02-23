# triton-fwht

This is an implementation of the Fast Walsh-Hadamard Transform (FWHT)
in triton.

This is a work-in-progress.

TODO:
1. setup and test lower precisions (fp16, bf16, fp8)
2. Much more heavily test weird sizes. 
3. float32 performance is a little disappointing. (3x slower than torch.clone.)
4. It seems surprisingly imprecise relative to a referance implementation on randn inputs :shrug:

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

[HadaCore](https://arxiv.org/pdf/2412.08832v1) seems to be doing pretty much the same strategy, but does way more low-level stuff.

## Caveats

1. If the input is not a power of 2, the kernel has to explicitly zero pad it to the next power of 2. This happens inside the kernel, so it does not need to allocate in global memory, just doing extra compute. Triton requires block sizes to be powers of 2, so I'm not sure if there's a way to work around this.
2. If the input size is not a power of 2, it does an extra iteration applying a hadamard
transform with order that is a power of 2. This step could be done with tensor cores, but I haven't gotten it working yet.