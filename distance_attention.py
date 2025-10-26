import math
import torch
from torch import nn
from d2l import torch as d2l

def masked_softmax(X, valid_lens):
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class distancetAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shapeof queries: (batch_size, no. of queries, d)
    # Shape of keys: (batchZ_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def  forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        #norm_k term
        norm_k = torch.linalg.norm(keys,p=2, dim=-1).pow(2)
        # B, K -> B, 1, K to same with scores B, Q, K 
        norm_k = norm_k.unsqueeze(1)
        # Swap the last two dimension of keys with keys.transpose(1,2)
        #add_norm term
        scores = torch.bmm(queries, keys.transpose(1,2)) + norm_k*(-0.5)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
  
  
