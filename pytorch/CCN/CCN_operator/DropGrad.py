import torch
import torch.nn as nn

class DropGrad_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, drop):
        ctx.drop = drop
        return x.clone()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.drop, None
    

class DropGrad(nn.Module):
    def __init__(self, drop = True):
        super(DropGrad, self).__init__()
        self.drop = 0 if drop else 1
    def forward(self, x):
        res = DropGrad_AF.apply(x, self.drop)
        return res
