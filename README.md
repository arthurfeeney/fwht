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

## Caveats

1. If the input is not a power of 2, the kernel has to explicitly zero pad it to the next power of 2. This happens inside the kernel, so it does not need to allocate in global memory, just doing extra compute. Triton requires block sizes to be powers of 2, so I'm not sure if there's a way to work around this.
2. This assumes the input can be factorized as a power of 16 times a power of 2. (Meaning the input size needs to be $16^m * 2^n$.) Otherwise, it will get padded.
3. The maximum supported size is currently $16**3 * 2 ** 3$, which allows things to fit in GPU shared memory. Going beyond this size is likely uncommon in machine learning.