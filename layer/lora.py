import torch
import torch.nn as nn

class Lora_Layer(nn.Module):
    def __init__(self, embed_dim, rank, alpha):
        super().__init__()

        self.alpha = alpha

        std_dev = 1/torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(embed_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, embed_dim))


    def forward(self, x):
        out = self.alpha * (x @ self.A @ self.B)

        return out
    

class Linear_with_Lora(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()

        self.linear = linear
        self.lora = Lora_Layer(linear.embed_dim, rank, alpha)

    def forward(self, x):
        out = self.linear(x) + self.lora(x)

        return out