import torch
import torch.nn as nn
import CCN
from CCN_operator.BaseOpModule import BaseOpModule

class ContextReshape_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None
    

class ContextReshape(BaseOpModule):
    def __init__(self, ngroup, device = 0, time_it = False):
        super(ContextReshape, self).__init__(device)
        self.op = {gid:CCN.ContextReshapeOp(ngroup, gid, time_it) for gid in self.device_list}
        
    def forward(self, x):
        res = ContextReshape_AF.apply(x, self.op)
        return res
