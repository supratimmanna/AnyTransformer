import torch
import torch.nn as nn
import math

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
    

class Sinusoidal_Positional_Embedding(nn.Module):

    def __init__(self, context_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(context_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, context_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    


class bert_input_Embedding(nn.Module):
    def __init__(self, config, pos_encd_type = 'sinusoidal'):
        super().__init__()

        self.pos_encd_type = pos_encd_type
        self.token_embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])

        if pos_encd_type == 'sinusoidal':
            self.pos_embedding = Sinusoidal_Positional_Embedding(config['context_length'], config['embed_dim'])

        elif pos_encd_type == 'learned':
            self.pos_embedding = nn.Embedding(config['context_length'], config['embed_dim'])

        else:
            print('Value of pos_encd_type must be from [sinusoidal, learned]')

        self.seg_embedding = nn.Embedding(config['num_segment'], config['embed_dim'])

        self.dropout = nn.Dropout(config['input_embed_dropout'])

    def forward(self, tok_idx, segment_label):

        bs, seq_len = tok_idx.shape
        tok_embeds = self.token_embedding(tok_idx)

        if self.pos_encd_type == 'learned':
            pos_embeds = self.pos_embedding(torch.arange(seq_len))

        elif self.pos_encd_type == 'sinusoidal':
            pos_embeds = self.pos_embedding(tok_idx)
        
        else:
            # will update some more positional encoding like RoPE
            pass

        seg_embeds = self.seg_embedding(segment_label)

        input_embeds = tok_embeds + pos_embeds + seg_embeds

        input_embeds = self.dropout(input_embeds)

        return input_embeds