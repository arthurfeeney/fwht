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

def hadamard_2_f32(device):
    H_2 = torch.tensor(
        [[sign_to_number(s) for s in row] for row in signs_2], dtype=torch.float32, device=device
    )
    return H_2
    #return torch.block_diag(H_2, H_2, H_2, H_2, H_2, H_2, H_2, H_2)

def hadamard_4_f32(device):
    H_4 = torch.tensor(
        [[sign_to_number(s) for s in row] for row in signs_4], dtype=torch.float32, device=device
    )
    return H_4
    #return torch.block_diag(H_4, H_4, H_4, H_4)

def hadamard_8_f32(device):
    H_8 = torch.tensor(
        [[sign_to_number(s) for s in row] for row in signs_8], dtype=torch.float32, device=device
    )
    return H_8
    return torch.block_diag(H_8, H_8)

def hadamard_16_f32(device):
    return torch.tensor(
        [[sign_to_number(s) for s in row] for row in signs_16], dtype=torch.float32, device=device
    )