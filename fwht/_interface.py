import torch
from torch import nn

from fwht._fwht_triton import fwht

class HadamardTransformAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, inplace):
        ctx._scale = scale
        ctx._inplace = inplace
        return fwht(input, scale, inplace)
    
    @staticmethod
    def backward(ctx, grad_output):
        return fwht(grad_output, ctx._scale, ctx._inplace), None, None
    
def fast_hadamard_transform(input, scale=1.0, inplace=False):
    return HadamardTransformAutograd.apply(input, scale, inplace)

class FastHadamardTransform(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, a, scale=1.0):
        return fast_hadamard_transform(a, scale)