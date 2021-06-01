import torch
from CCN_operator import Dquant, QUANT


if __name__ == '__main__':
    data = torch.rand((2,4,64,64)).type(torch.float32).to('cuda:0')
    qt = QUANT(4,8,ntop=2).to('cuda:0')
    dqt = Dquant(4,8).to('cuda:0')
    y,z = qt(data)
    dqt.weight = qt.weight
    y2 = dqt(z)
    print(torch.mean(torch.abs(y-y2)))