import torch
import torch.nn as nn

class gpt_input_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_embedding = nn.Embedding(config['context_length'], config['embed_dim'])
        self.dropout = nn.Dropout(config['input_embed_dropout'])

    def forward(self, tok_idx):

        bs, seq_len = tok_idx.shape
        tok_embeds = self.token_embedding(tok_idx)
        pos_embeds = self.pos_embedding(torch.arange(seq_len))

        input_embeds = tok_embeds + pos_embeds

        input_embeds = self.dropout(input_embeds)

        return input_embeds