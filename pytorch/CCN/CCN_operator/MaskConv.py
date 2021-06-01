import torch
import torch.nn as nn
import CCN
from torch.autograd import Variable
from CCN_operator.BaseOpModule import BaseOpModule

class MaskConstrain_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        op[gid].forward(x)
        ctx.op = op
        return x
        
    @staticmethod
    def backward(ctx, grad_output):
        gid = grad_output.device.index
        ctx.op[gid].backward(grad_output)
        return grad_output, None
    

class MConv(BaseOpModule):

    def __init__(self, ngroup, c_in, c_out, kernel_size, hidden = False, device = 0, time_it = False):
        super(MConv, self).__init__(device)
        constrain = 6 if hidden else 5
        self.op = {gid:CCN.MaskConstrainOp(constrain, ngroup, gid, time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.empty((c_out*ngroup,c_in*ngroup,kernel_size,kernel_size),dtype=torch.float32))
        torch.nn.init.kaiming_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(c_out*ngroup,dtype=torch.float32))
        self.kernel_size = kernel_size

    def forward(self, x):
        self.weight.data = MaskConstrain_AF.apply(self.weight.data, self.op)
        res = nn.functional.conv2d(x,self.weight,self.bias, padding=self.kernel_size//2)
        return res
