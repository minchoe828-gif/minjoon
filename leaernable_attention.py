class DotProductAttention(nn.Module):
    def __init__(self,q_dim, k_dim, dropout, epsilon=1e-3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.M = nn.Parameter(torch.randn(q_dim, k_dim))
        nn.init.xavier_uniform_(self.M)

    def forward(self, queries, keys, values, valid_lens=None):
        transformed_queries = torch.matmul(queries, self.M)
        scores = torch.bmm(transformed_queries, keys.transpose(1,2))
        # normalizing by frobenius norm not sqrt(d) cause deviation is frobenus norm of M
        scores /= torch.linalg.norm(self.M, ord='fro')+epsilon
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
