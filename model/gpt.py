import torch
import torch.nn as nn

from layer.embedding import gpt_input_Embedding
from layer.transformer import GPT_Transformer_Block
from layer.layer_norm import Layer_Norm


class GPT_Model(nn.Module):
    def __init__(self, config, weight_tie=True):
        super().__init__()

        self.input_embedding = gpt_input_Embedding(config)

        self.transformer_blocks = nn.Sequential(
            *[GPT_Transformer_Block(config) for _ in range(config['num_layers'])]
        )

        self.final_norm = Layer_Norm(config['embed_dim'])

        self.lm_head = nn.Linear(config['embed_dim'], config['vocab_size'], bias=False)

        # Tie the weights
        if weight_tie:
            self.lm_head.weight = self.input_embedding.token_embedding.weight


    def forward(self, x):

        x = self.input_embedding(x)

        x = self.transformer_blocks(x)
        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits



