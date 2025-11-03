class MyMultiheadAttention(nn.Module):
    def __init__(self,head_num, num_hiddens):
        super().__init__()
        self.head_num, self.num_hiddens = head_num, num_hiddens
        self.W_q = nn.LazyLinear(self.head_num*self.num_hiddens, bias=False)
        self.W_k = nn.LazyLinear(self.head_num*self.num_hiddens, bias=False)
        self.W_v = nn.LazyLinear(self.head_num*self.num_hiddens, bias=False)
        self.W_o = nn.LazyLinear(self.head_num*self.num_hiddens, bias=False)
    def forward(self, queries, keys, values):
        trans_q = self.transform_qkv(self.W_q(queries))
        trans_k = self.transform_qkv(self.W_k(keys))
        trans_v = self.transform_qkv(self.W_v(values))
        attention_heads = torch.nn.functional.scaled_dot_product_attention(trans_q, trans_k, trans_v)
        
