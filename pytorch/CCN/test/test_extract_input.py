import torch
from CCN_operator import DInput, DExtract

if __name__ == '__main__':
    x = torch.range(0,15,dtype=torch.float32).view(1,2,2,4).to("cuda:0")
    ext = DExtract(False).to("cuda:0")
    pt = DInput(2).to("cuda:0")
    #print(x)
    for i in range(7):
        y = ext(x)
        z = pt(y)
        #print('iter:',i)
        #print(y)
        #print(z)