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




########################################################################################################################

### Multi-head Self-Attention for both Causal and Non-Causal Attention ################

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
        


########################################################################################################################

### Multi-head Group Query Attention and Multi-Head Multi Query Attention for both Causal and Non-Causal Attention ################

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
        



### ##########################################################################################

### Multi-head Local Attention for both Causal and Non-Causal Attention ##################################

class MultiHead_Local_Global_Attention(nn.Module):
    def __init__(self, in_dim, out_dim, num_head, window_size=64, dropout=0.0, qkv_bias = False, causal_attention=True):
        super().__init__()

        assert out_dim % num_head ==0, "out_dim must be visible by num_head"

        self.out_dim = out_dim
        self.num_head = num_head
        self.head_dim = out_dim//num_head
        self.causal_attention = causal_attention
        self.window_size = window_size

        ## query, key and value projection
        self.w_q = nn.Linear(in_dim, out_dim, bias = qkv_bias)
        self.w_k = nn.Linear(in_dim, out_dim, bias = qkv_bias)
        self.w_v = nn.Linear(in_dim, out_dim, bias = qkv_bias)

        ## project of the head outputs to the final output
        self.out_proj = nn.Linear(out_dim, out_dim, bias = qkv_bias)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, attention_mask=None, global_attention = None):

        ## if global_attention is None then it performs only local attention
        ## if attention_mask is None then it performs only local attention without discarding the masked token
        
        
        b, token_num, in_dim = x.shape
        device = x.device

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
            causal_local_global_attention_mask = self.local_global_causal_mask(b, token_num, self.window_size, 
                                                                        global_attention = global_attention, device = device)
            
            mask_bool = causal_local_global_attention_mask.bool()[:token_num, :token_num].unsqueeze(1)

        else:
            # create local symmetric mask
            mask_bool_local_global_symmetry = self.symmetric_local_global_mask(b, token_num, self.window_size, 
                                                                        global_attention=global_attention,device=device)
            
            mask_bool_local_global_symmetry = mask_bool_local_global_symmetry.unsqueeze(1)

            if attention_mask is not None:
                ## if already an attention mask provided for masked token that take that also into consideration 
                ## along with local attention mask by perfroming AND logic between two amsks.

                mask_bool_provided = attention_mask[:token_num, :token_num].unsqueeze(1).unsqueeze(1)
                mask_bool = torch.logical_or(mask_bool_local_global_symmetry, mask_bool_provided)

                
            else:
                mask_bool = mask_bool_local_global_symmetry.bool()

        dotproduct_attention = Scaled_DotProduct_Attention()
        context_vector = dotproduct_attention(Q, K, V, self.head_dim, self.dropout, mask_bool)
        
        # Combine heads, where self.out_dim = self.num_head * self.head_dim
        context_vector = context_vector.contiguous().view(b, token_num, self.out_dim)

        context_vector = self.out_proj(context_vector)

        return context_vector
    

    def local_global_causal_mask(self, batch_size, seq_len, window_size, global_attention=None, device=None):

        all_masks = []
        for i in range(batch_size):
            idx = torch.arange(seq_len, device=device).unsqueeze(1)  # shape [T, 1]
            jdx = torch.arange(seq_len, device=device).unsqueeze(0)  # shape [1, T]

            # Original condition — positions allowed
            allowed = (idx - jdx < window_size) & (idx >= jdx)

            # Invert it: 0 for allowed positions, 1 for masked positions
            local_mask = torch.logical_not(allowed).float()  

            ## if global attention token mask (0:attend the token for attention score 1: do not attend) is provided
            ## then bring those golbal token into consideration for attention score calculation.
            ## For causal the gobal atention tokens must be from the previous time stamp

            if global_attention is not None:
                # Global mask: 0s for global rows/cols
                global_mask = torch.ones_like(local_mask)

                for j,g in enumerate(global_attention[i]):
                    # print(j,g)
                    
                    if g.item()==True:
                        global_mask[j, :] = 0
                        global_mask[:, j] = 0

                final_mask = torch.logical_and(local_mask, global_mask)  # [T, T]

            else:
                final_mask = local_mask

            all_masks.append(final_mask)
        
        return torch.stack(all_masks, dim=0).float()

    

    def symmetric_local_global_mask(self, batch_size, seq_len, window_size, global_attention=None, device=None):
        assert window_size % 2 == 1, "window_size must be odd for symmetric mask"
        half_window = window_size // 2

        all_masks = []

        for i in range(batch_size):

                idx = torch.arange(seq_len, device=device).unsqueeze(1)  # [T, 1]
                jdx = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]

                # Mask 0 where |i - j| <= half_window, else 1
                local_mask = (torch.abs(idx - jdx) > half_window).float()

                ## if global attention token mask (0:attend the token for attention score 1: do not attend) is provided
                ## then bring those golbal token into consideration for attention score calculation.
                if global_attention is not None:
                        # Global mask: 0s for global rows/cols
                        global_mask = torch.ones_like(local_mask)

                        for j,g in enumerate(global_attention[i]):
                                if g.item()==True:
                                        global_mask[j, :] = 0
                                        global_mask[:, j] = 0

                        final_mask = torch.logical_and(local_mask, global_mask)  # [T, T]

                else:
                        final_mask = local_mask

                all_masks.append(final_mask)

        return torch.stack(all_masks, dim=0).float()



