import torch
import torch.nn as nn

class GELU(nn.Module):

    def __init__(self):
        super.__init__()

    def forward(self, x):
        gelu_x = 0.5 * x (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x + 0.044715*torch.pow(x,3))))

        return gelu_x
    


class Tanh(nn.module):

    def __init__(self):
        super.__init__()

    def forward(self, x):
        tanh_x = torch.tanh(x)

        return tanh_x

    
