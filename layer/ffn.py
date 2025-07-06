import torch
import torch.nn as nn

from layer.activation import *

class Feed_Forwad(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()

        self.ff_layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        out = self.ff_layers(x)

        return out