import torch
import torch.nn as nn

class gpt_input_Embedding(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(context_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tok_idx):

        bs, seq_len = tok_idx.shape
        tok_embeds = self.token_embedding(tok_idx)
        pos_embeds = self.pos_embedding(torch.arrange(seq_len))

        input_embeds = tok_embeds + pos_embeds

        input_embeds = self.dropout(input_embeds)

        return input_embeds