import torch
import torch.nn as nn


class MASKDATA_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, keep_channel, with_zero):
        if not with_zero:
            outputs = x[:,:keep_channel]
        else:
            outputs = torch.zeros_like(x)
            outputs[:,:keep_channel] = x[:,:keep_channel]
        ctx.keep_channel = keep_channel
        ctx.save_for_backward(x)
        return outputs
        
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        res = torch.zeros_like(x)
        res[:,:ctx.keep_channel]=grad_output[:,:ctx.keep_channel]
        return res, None, None
    
class MaskData(nn.Module):
    def __init__(self, channel, with_zero=False):
        super(MaskData, self).__init__()
        self.keep_channel = channel
        self.with_zero = with_zero
    def forward(self, x):
        return MASKDATA_AF.apply(x, self.keep_channel, self.with_zero)

