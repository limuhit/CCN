import torch
import torch.nn as nn
import CCN
from CCN_operator.BaseOpModule import BaseOpModule

class DOutput_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0],outputs[1]
        
    @staticmethod
    def backward(ctx, grad_output, grad_output1):
        return None, None
    

class DOutput(BaseOpModule):
    
    def __init__(self, ngroup, total_region=65536, device = 0, time_it = False):
        super(DOutput, self).__init__(device)
        self.op = { gid : CCN.DOutputOp(ngroup,total_region, gid, time_it) for gid in self.device_list}
        

    def forward(self, x):
        res = DOutput_AF.apply(x, self.op)
        return res
