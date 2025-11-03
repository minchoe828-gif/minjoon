class MyMultiheadAttention(nn.Module):
    def __init__(self,head_num, num_hiddens):
        super().__init__()
        self.head_num, self.num_hiddens = head_num, num_hiddens
        # (Batch_size, number_of_qkv, features_dimension) -> (Batch_size, number_of_qkv, head*hidden)
        self.W_q = nn.LazyLinear(self.head_num*self.num_hiddens, bias=False)
        self.W_k = nn.LazyLinear(self.head_num*self.num_hiddens, bias=False)
        self.W_v = nn.LazyLinear(self.head_num*self.num_hiddens, bias=False)
        # (Batch_size, number_of_q, head*hidden) -> (Batch_size, number_of_q, head*hidden)
        self.W_o = nn.LazyLinear(self.head_num*self.num_hiddens, bias=False)
    def forward(self, queries, keys, values):
        #(Batch_size, number_of_qkv,head*hidden) -> (Batch_size*head, number_of_qkv, hidden)
        trans_q = self.transform_qkv(self.W_q(queries))
        trans_k = self.transform_qkv(self.W_k(keys))
        trans_v = self.transform_qkv(self.W_v(values))
        #(Batch_size*head, number_of_q, hidden)
        attention_heads = torch.nn.functional.scaled_dot_product_attention(trans_q, trans_k, trans_v)
        #(Batch_size*head, number_of_q, hidden) -> (Batch_size, number_of_q, head*hidden)
        trans_attention_heads = self.transform_output(attention_heads)
        return self.W_o(trans_attention_heads)

    def transform_qkv(self, input):
        batch_size, num_of_qkv, _ = input.shape
        input = input.reshape((batch_size, num_of_qkv, -1, self.num_hiddens))
        input = input.permute(0,2,1,3)
        input = input.reshape((batch_size*self.head_num, num_of_qkv, self.num_hiddens))
        return input

    def transform_output(self, input):
        input = input.reshape((-1, self.head_num, input.shape[1], input.shape[2]))
        input = input.permute(0,2,1,3)
        input = input.reshape((input.shape[0], input.shape[1],self.head_num*self.num_hiddens))
        return input 
        
        
        
