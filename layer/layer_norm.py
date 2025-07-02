import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.eps = 1e-9
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))


    def forward(self, x):
        
        mean_x = x.mean(dim=-1, keep_dim=True)
        var_x = x.var(dim=-1, keep_dim=True, unbiased=False)

        std_x = torch.sqrt(var_x)

        x_norm = (x - mean_x) / (std_x + self.eps)

        x_norm = self.scale * x_norm + self.shift

        return x_norm
