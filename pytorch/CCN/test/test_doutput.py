import torch
from CCN_operator import DOutput

if __name__ == '__main__':
    x = torch.range(0,15,dtype=torch.float32).view(1,4,2,2).to("cuda:0")
    print(x)
    ext = DOutput(2).to("cuda:0")
    for i in range(4):
        y = ext(x)
        print('iter:',i)
        print(y)
