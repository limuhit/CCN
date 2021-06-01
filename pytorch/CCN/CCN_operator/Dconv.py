import torch
import torch.nn as nn
import CCN
from CCN_operator.BaseOpModule import BaseOpModule

class Dconv_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x,  weight, bias, op):
        gid = x.device.index
        outputs = op[gid].forward(x, weight, bias)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
       return None,None,None,None
    

class DConv(BaseOpModule):
    # ngroup, c_in, c_out
    def __init__(self, ngroup, c_in, c_out, kernel_size, hidden=False, device = 0, time_it = False):
        super(DConv, self).__init__(device)
        constrain = 6 if hidden else 5
        channel, nout = ngroup*c_in, ngroup*c_out
        self.op = { gid : CCN.DconvOp(channel,ngroup,nout,kernel_size,constrain, gid, time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.rand((nout,channel,kernel_size,kernel_size),dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((nout),dtype=torch.float32))
        

    def forward(self, x):
        res = Dconv_AF.apply(x, self.weight, self.bias, self.op)
        return res

if __name__ == '__main__':
    data = torch.rand((2,2,6,6),dtype=torch.float32).to('cuda:0')
    dc = DConv(2,2,4,5,5).to('cuda:0')
    y2 = dc(data)
    pass