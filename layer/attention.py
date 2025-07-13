import torch
import torch.nn as nn


class Scaled_DotProduct_Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, head_dim, dropout, mask_bool=None):    

        #  Dot product for each head
        attention_score = (Q @ K.transpose(2,3)) / torch.sqrt(torch.tensor(head_dim))

        # Use the mask to fill attention scores
        if mask_bool is not None:
            attention_score.masked_fill_(mask_bool, -torch.inf)

        attention_weight = torch.softmax(attention_score, dim=1)
        attention_weight = dropout(attention_weight)

        # Shape: (b, token_num, num_heads, head_dim)
        context_vector = (attention_weight @ V).transpose(1,2)

        return context_vector






class MultiHead_Attention(nn.Module):
    def __init__(self, in_dim, out_dim, num_head, context_length=1024, dropout=0.0, qkv_bias = False, causal_attention=True):
        super().__init__()

        assert out_dim % num_head ==0, "out_dim must be visible by num_head"

        self.out_dim = out_dim
        self.num_head = num_head
        self.head_dim = out_dim//num_head
        self.causal_attention = causal_attention

        ## query, key and value projection
        self.w_q = nn.Linear(in_dim, out_dim, bias = qkv_bias)
        self.w_k = nn.Linear(in_dim, out_dim, bias = qkv_bias)
        self.w_v = nn.Linear(in_dim, out_dim, bias = qkv_bias)

        ## project of the head outputs to the final output
        self.out_proj = nn.Linear(out_dim, out_dim, bias = qkv_bias)
        
        self.dropout = nn.Dropout(dropout)

        if self.causal_attention:
            self.register_buffer('causal_attention_mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, attention_mask=None):

        # if not self.causal_attention and attention_mask is None:
        #     raise ValueError('Either maske causal_attention=True or provide a attention_mask but not both')
        
        b, token_num, in_dim = x.shape

        ## shape of Q,K,V: (b, token_num, out_dim)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x) 

        # Unroll last dim: (b, token_num, d_out) -> (b, token_num, num_head, head_dim) -> (b, num_head, token_num, head_dim)
        Q = Q.view(b, token_num, self.num_head, self.head_dim).transpose(1,2)
        K = K.view(b, token_num, self.num_head, self.head_dim).transpose(1,2)
        V = V.view(b, token_num, self.num_head, self.head_dim).transpose(1,2)

        # Use the mask to fill attention scores
        if self.causal_attention:
            mask_bool = self.causal_attention_mask.bool()[:token_num, :token_num]

        else:
            if attention_mask is not None:
                mask_bool = attention_mask.bool()[:token_num, :token_num]

                if len(mask_bool.shape)==2:
                    mask_bool = mask_bool.unsqueeze(1).unsqueeze(1)

                if len(mask_bool.shape)==3:
                    mask_bool = mask_bool.unsqueeze(2)
                    
            else:
                mask_bool = attention_mask


        context_vector = Scaled_DotProduct_Attention()(Q, K, V, self.head_dim, self.dropout, mask_bool)
        
        # Combine heads, where self.out_dim = self.num_head * self.head_dim
        context_vector = context_vector.contiguous().view(b, token_num, self.out_dim)

        context_vector = self.out_proj(context_vector)

        return context_vector
        


class MultiHead_Group_Query_Attention(nn.Module):
    def __init__(self, in_dim, out_dim, num_q_head, num_kv_head, context_length=1024, dropout=0.0, qkv_bias = False, causal_attention=True):
        super().__init__()

        ## if num_kv_head =1 then it is a Multi-query attention
        
        assert out_dim % num_q_head ==0, "out_dim must be visible by num_q_head"
        assert num_q_head % num_kv_head ==0, "num_q_head must be visible by num_kv_head"

        self.out_dim = out_dim
        self.num_q_head = num_q_head
        self.num_kv_head = num_kv_head
        self.head_dim = out_dim//num_q_head
        self.kv_group_size = num_q_head // num_kv_head
        self.causal_attention = causal_attention

        ## query, key and value projection
        self.w_q = nn.Linear(in_dim, out_dim, bias = qkv_bias)
        self.w_k = nn.Linear(in_dim, self.head_dim*self.num_kv_head, bias = qkv_bias)
        self.w_v = nn.Linear(in_dim, self.head_dim*self.num_kv_head, bias = qkv_bias)

        ## project of the head outputs to the final output
        self.out_proj = nn.Linear(out_dim, out_dim, bias = qkv_bias)
        
        self.dropout = nn.Dropout(dropout)

        if self.causal_attention:
            self.register_buffer('causal_attention_mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, attention_mask=None):

        if self.causal_attention and attention_mask is not None:
            raise ValueError('Either maske causal_attention=True or provide a attention_mask but not both')
        
        b, token_num, in_dim = x.shape

        ## shape of Q,K,V: (b, token_num, out_dim)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x) 

        # Unroll last dim: (b, token_num, d_out) -> (b, token_num, num_head, head_dim) -> (b, num_head, token_num, head_dim)
        Q = Q.view(b, token_num, self.num_q_head, self.head_dim).transpose(1,2)
        K = K.view(b, token_num, self.num_kv_head, self.head_dim).transpose(1,2)
        V = V.view(b, token_num, self.num_kv_head, self.head_dim).transpose(1,2)


        # Expand k, v to match number of query heads
        # Repeat each k/v head for kv_group_size
        K = K.repeat_interleave(self.kv_group_size, dim=1)  # [B, num_q_heads, T, head_dim]
        V = V.repeat_interleave(self.kv_group_size, dim=1)

        # Use the mask to fill attention scores
        if self.causal_attention:
            mask_bool = self.causal_attention_mask.bool()[:token_num, :token_num]

        else:

            if attention_mask is not None:
                mask_bool = attention_mask[:token_num, :token_num]

                if len(mask_bool.shape)==2:
                    mask_bool = mask_bool.unsqueeze(1).unsqueeze(1)

                if len(mask_bool.shape)==3:
                    mask_bool = mask_bool.unsqueeze(2)

            else:
                mask_bool = attention_mask

        context_vector = Scaled_DotProduct_Attention()(Q, K, V, self.head_dim, self.dropout, mask_bool)

        # Combine heads, where self.out_dim = self.num_head * self.head_dim
        context_vector = context_vector.contiguous().view(b, token_num, self.out_dim)

        context_vector = self.out_proj(context_vector)

        return context_vector
        

        



