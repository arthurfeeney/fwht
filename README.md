# torch-fwht

This is an implementation of the Fast Walsh-Hadamard Transform (FWHT) for pytorch.
The GPU code is implemented with triton. The CPU code is a simple C++ extension.

This is a work-in-progress!

## Installation

The only dependencies are triton and pytorch.

```console
git clone https://github.com/arthurfeeney/fwht && cd fwht
pip install --editable .
```

To run the tests,

```console
pip install pytest
python -m pytest test/
```

## Example

```python
from fwht import (
   fast_hadamard_transform,
   # can also use as a module
   # FastHadamardTransform
)

data = torch.ones(32, device='cuda')
fast_hadamard_transform(data, inplace=True)
assert data[0] = 32
assert torch.all(data[1:] == 0)
```

## Caveats

1. If the input is not a power of 2, the kernel has to explicitly zero pad it to the next power of two. This happens inside the kernel, so it does not need to allocate in global memory, just doing extra compute. Triton requires block sizes to be powers of 2, so I'm not sure if there's a way to work around this.
2. This assumes the input can be factorized as a power of 16 times a power of 2. (Meaning the input size needs to be $16^m * 2^n$.) Otherwise, it will get padded.
3. The maximum supported size is currently $16^3 * 2^3$, which allows things to fit in GPU shared memory. Going beyond this size is likely uncommon in machine learning.

## Notes on Implementation

The implementation relies on a few things:
1. Hadamard transforms $H_{pq}$ can be expressed as the
   Kronecker product of two smaller Hadamard transforms $H_{pq} = \text{kron}(H_p, H_q)$.
2. The matrix-vector product of a kronecker product satisfies 
   $\text{kron}(B^T, A) \text{colvec}(m)$ = $\text{colvec}(AmB)$, where colvec unravels 
   the columns of a matrix to form a vector. Since the Hadamard matrices are symmetric, 
   it is even simpler and doesn't require transposes.
3. A way to construct a Hadamard matrix is the identity $(H_n)_{i,j} = -1^{i \cdot j}$, where
   $i\cdot j$ is the bit-wise dot product of integers $i$ and $j$. This is useful for
   building the small Hadamard matrices used in the base case, instead of loading them from gmem.
   Triton should optimize most of this most of this away during compilation. 