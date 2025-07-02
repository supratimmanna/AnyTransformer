import torch
import torch.nn as nn

from layer.activation import *

class Feed_Forwad(nn.Module):

    def __init__(self, embed_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()

        if dropout:
            self.ff_layers = nn.Sequential([
                nn.Linear(embed_dim, hidden_dim),
                GELU(),
                nn.Dropout(),
                nn.Linear(hidden_dim, out_dim),
            ])

    def forward(self, x):
        out = self.ff_layers(x)

        return out