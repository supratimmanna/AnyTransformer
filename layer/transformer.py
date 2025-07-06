import torch
import torch.nn as nn

from layer.attention import *
from layer.ffn import *
from layer.layer_norm import *

class GPT_Transformer_Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention_block = MultiHead_Attention(in_dim=config['embed_dim'], out_dim=config['embed_dim'],
                                                   context_length=config['context_length'], num_head=config['num_head'],
                                                   dropout=config['dropout'], qkv_bias=config['qkv_bias'],
                                                   causal_attention=config['causal_attention'])
        
        self.ffn = Feed_Forwad(in_dim=config['embed_dim'], hidden_dim=config['ff_hidden_dim'], out_dim=config['embed_dim'],
                               dropout=config['dropout'])
        
        self.layer_norm_1 = Layer_Norm(embed_dim=config['embed_dim'])
        self.layer_norm_2 = Layer_Norm(embed_dim=config['embed_dim'])

        self.dropout_residual = nn.Dropout(config['dropout'])


    def forward(self, x):

        # Shortcut connection for attention block
        residual = x

        x = self.layer_norm_1(x)
        x = self.attention_block(x) # Shape [batch_size, num_tokens, emb_size]
        x = self.dropout_residual(x)

        # Add the original input back
        x += residual

        residual = x

        x = self.layer_norm_2(x)
        x = self.ffn(x)
        x = self.dropout_residual(x)
    
        x += residual

        return x
    
