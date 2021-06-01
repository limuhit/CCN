import torch
import torch.nn as nn
import CCN
from CCN_operator.BaseOpModule import BaseOpModule

class Dquant_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, op):
        gid = x.device.index
        outputs = op[gid].forward(x, weight)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None
    

class Dquant(BaseOpModule):
    
    def __init__(self, channel,bin_num, device = 0, time_it = False):
        super(Dquant, self).__init__(device)
        self.op = { gid : CCN.DquantOp(channel,bin_num, gid, time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.zeros((channel,bin_num),dtype=torch.float32).to('cuda:%d'%self.device_list[0]))
        

    def forward(self, x):
        res = Dquant_AF.apply(x, self.weight, self.op)
        return res
