import torch

from fwht._fwht_triton import fwht

class HadamardTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx._scale = scale
        return fwht(input, scale)
    
    @staticmethod
    def backward(ctx, grad_output):
        return fwht(grad_output, scale=ctx._scale), None
    
def fast_hadamard_transform(input, scale=1.0):
    return HadamardTransform.apply(input, scale)