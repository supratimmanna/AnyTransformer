import torch
import torch.nn as nn

from layer.embedding import bert_input_Embedding
from layer.transformer import BERT_Transformer_Block
from layer.layer_norm import Layer_Norm
from layer.activation import GELU


class BERT_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_embedding = bert_input_Embedding(config)

        self.transformer_blocks = nn.ModuleList(
            [BERT_Transformer_Block(config) for _ in range(config['num_layers'])]
        )



    def forward(self, x, segment_label, attention_mask):

        x = self.input_embedding(x, segment_label)

        for block in self.transformer_blocks:
            x = block(x, attention_mask)  # each block gets both x and attention_mask

        return x


class BERT_Pretraining_Head(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()

        ## masked language model head
        self.mlm = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            GELU(),
            Layer_Norm(embed_dim=embed_dim)
        )

        self.mlm_decoder = nn.Linear(embed_dim, vocab_size, bias=False)

        ## NSP (Next sentence prediction head)
        self.nsp_classifier = nn.Linear(embed_dim, 2)



    def forward(self, seq_output):

        # sequence_output: [B, T, D]
        # cls_output: [B, D] (output[:, 0, :])

        mlm_hidden = self.mlm(seq_output)
        mlm_logits = self.mlm_decoder(mlm_hidden) # [B, T, vocab_size]

        # First token: [CLS] which is reponsible for the sentence classification
        nsp_hidden = seq_output[:,0,:]
        nsp_logits = self.nsp_classifier(nsp_hidden) # [B, 2]

        return mlm_logits, nsp_logits


class BERT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self. bert_encoder = BERT_Encoder(config)
        self.bert_pretrain_head = BERT_Pretraining_Head(config['embed_dim'], config['vocab_size'])

    
    def forward(self, x, segment_label, attention_mask):

        output = self.bert_encoder(x, segment_label, attention_mask)
        mlm_logits, nsp_logits = self.bert_pretrain_head(output)

        return mlm_logits, nsp_logits

