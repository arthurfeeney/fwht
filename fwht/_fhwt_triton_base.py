import torch
import triton
import triton.language as tl

def sign_to_number(x):
    return 1 if x == "+" else -1

# all sign strings are yoinked from http://neilsloane.com/hadamard/
signs_2 = """++
+-""".split('\n')

signs_4 = """++++
+-+-
++--
+--+""".split('\n')

signs_8 = """++++++++
+-+-+-+-
++--++--
+--++--+
++++----
+-+--+-+
++----++
+--+-++-""".split('\n')

signs_16 = """++++++++++++++++
+-+-+-+-+-+-+-+-
++--++--++--++--
+--++--++--++--+
++++----++++----
+-+--+-++-+--+-+
++----++++----++
+--+-++-+--+-++-
++++++++--------
+-+-+-+--+-+-+-+
++--++----++--++
+--++--+-++--++-
++++--------++++
+-+--+-+-+-++-+-
++----++--++++--
+--+-++--++-+--+""".split('\n')

def hadamard_2(device, dtype):
    H_2 = torch.tensor(
        [[sign_to_number(s) for s in row] for row in signs_2], dtype=dtype, device=device
    )
    return H_2

def hadamard_4(device, dtype):
    H_4 = torch.tensor(
        [[sign_to_number(s) for s in row] for row in signs_4], dtype=dtype, device=device
    )
    return H_4

def hadamard_8(device, dtype):
    H_8 = torch.tensor(
        [[sign_to_number(s) for s in row] for row in signs_8], dtype=dtype, device=device
    )
    return H_8

def hadamard_16(device, dtype):
    return torch.tensor(
        [[sign_to_number(s) for s in row] for row in signs_16], dtype=dtype, device=device
    )