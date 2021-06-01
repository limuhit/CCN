import torch
from CCN_operator import MConv, DConv

if __name__ == '__main__':
    data = torch.rand((2,2,32,32),dtype=torch.float32).to('cuda:0')
    mc = MConv(2,1,2,5).to('cuda:0')
    dc = DConv(2,1,2,5).to('cuda:0')
    mc.bias.data.fill_(0.1)
    y1 = mc(data)
    dc.weight = mc.weight
    dc.bias = mc.bias
    for _ in range(32+32+2-2):
        y2 = dc(data)
    print(torch.mean(torch.abs(y1-y2)))