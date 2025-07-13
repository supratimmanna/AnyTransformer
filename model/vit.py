import torch
import torch.nn as nn

from layer.vit_embedding import Patch_Embedding
from layer.vit_transformer import ViT_Transformer_Block
from layer.layer_norm import Layer_Norm


class ViT_Model(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.input_embedding = Patch_Embedding(config)

        self.transformer_blocks = nn.Sequential(
            *[ViT_Transformer_Block(config) for _ in range(config['num_layers'])]
        )

        self.final_norm = Layer_Norm(config['embed_dim'])

        self.lm_head = nn.Linear(config['embed_dim'], config['num_class'], bias=False)




    def forward(self, x):

        x = self.input_embedding(x)

        x = self.transformer_blocks(x)
        x = self.final_norm(x)

        logits = self.lm_head(x[:,0,:])

        return logits



