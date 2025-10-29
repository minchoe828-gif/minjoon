class MyAdditiveAttention(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.W_q = nn.LazyLinear(num_hiddens)
        self.W_k = nn.LazyLinear(num_hiddens)
        self.w_v = nn.LazyLinear(1)

    def forward(queries, keys, values, valid_lens):
        # queries_shape = (Batch_size, num_of_queries, queries_dim)
        # keys_shape = (Batch_size, num_of_keys, keys_dim)
        # queries_dim & keys_dim -> num_hiddens
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        #queries.shape = (Batch_size, num_of_queries, 1, num_hiddens)
        #keys.shape = (Batch_size, 1, num_of_keys, num_hiddens)
        queries=queries.unsqueeze(2)
        keys=keys.unsqueeze(1)
        features=nn.functional.tanh(queries+keys)
        scores=self.w_v(features)
        #scores.shape=(Batch_size, num_of_quereis, num_of_keys)
        scores=scores.squeeze(-1)
        self.attention_weight = masked_softmax(scores, valid_lens)
        return torch.bmm(scores, values)

class AdditiveAttention(nn.Module):
    def __init__(self, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
