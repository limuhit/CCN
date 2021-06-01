import torch
import torch.nn as nn

class ConstScale(nn.Module):
    
    def __init__(self, scale, bias):
        super(ConstScale,self).__init__()
        self.scale = scale 
        self.bias = bias

    def forward(self,x):
        return x * self.scale + self.bias

