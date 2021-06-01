import torch
import torch.nn as nn
import CCN
from CCN_operator.BaseOpModule import BaseOpModule

class EntropyGmm_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, delta, mean, label, op):
        gid = weight.device.index
        outputs = op[gid].forward(weight, delta, mean, label)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], outputs[1], outputs[2], outputs[3], None
    

class EntropyGmm(BaseOpModule):

    def __init__(self, num_gaussian=3,ignore_label=0, device = 0, time_it = False):
        super(EntropyGmm, self).__init__(device)
        self.op = {gid:CCN.EntropyGmmOp(num_gaussian,ignore_label, gid, time_it) for gid in self.device_list}
        

    def forward(self, weight, delta, mean, label):
        res = EntropyGmm_AF.apply(weight, delta, mean, label, self.op)
        return res
