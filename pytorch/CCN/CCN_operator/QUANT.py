import torch
import torch.nn as nn
import CCN
import math
from CCN_operator.BaseOpModule import BaseOpModule

class QUANT_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, quant_op, training):
        gid = x.device.index
        outputs = quant_op[gid].forward(x, weight, training)
        ctx.save_for_backward(x, outputs[0])
        ctx.quant_op = quant_op
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs[0],outputs[1]
        
    @staticmethod
    def backward(ctx, *grad_output):
        x,out = ctx.saved_tensors
        gid = x.device.index
        outputs  = ctx.quant_op[gid].backward(list(grad_output),x,out)
        return outputs[0], outputs[1], None, None
    

class QUANT(BaseOpModule):

    def __init__(self, channel, bin_num, check_iters=100, weight_decay=0.9, ntop=1, top_alpha=0.1, device_id=0, time_flag=False):
        super(QUANT, self).__init__(device_id)
        ta = 1./(bin_num+1)
        tb = math.log(ta)
        self.weight = nn.Parameter(torch.zeros((channel,bin_num),dtype=torch.float32).to('cuda:%d'%self.device_list[0]))
        self.weight.data[:,0] = ta
        self.weight.data[:,1:] = tb
        self.op = {gid:CCN.QuantOp(channel, bin_num, weight_decay, check_iters, ntop, top_alpha, gid, time_flag) for gid in self.device_list}

    def forward(self, x):
        res = QUANT_AF.apply(x, self.weight,self.op, self.training)
        return res

