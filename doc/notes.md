# Notes on Implementation

The implementation relies on a few interesting things:
1. Hadamard transforms $H_{pq}$ can be expressed as the
   Kronecker product of two smaller Hadamard transforms $H_{pq} = \text{kron}(H_p, H_q)$.
2. The matrix-vector product of a kronecker product satisfies 
   $\text{kron}(B^T, A) \text{colvec}(m)$ = $\text{colvec}(AmB)$, where colvec unravels 
   the columns of a matrix to form a vector. (So, basically the kronecker 
   product can be implemented using GEMMs and that's good because triton likes
   big beautiful matrix multiplications.)
3. A way to construct a Hadamard matrix is the identity $(H_n)_{i,j} = -1^{i \cdot j}$, where
   $i\cdot j$ is the bit-wise dot product of integers $i$ and $j$. This is useful for
   building the small Hadamard matrices used in the base case, instead of loading them from gmem.
   Triton is able to optimize most of this most of the generation away during compilation. 