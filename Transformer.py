class Transformer_encoding_block(nn.Module):
    def __init__(self, num_head, num_hidden, num_qkv, bias):
        super().__init__()
        self.attention = Multihead_Attention(num_head, num_hidden, bias)
        self.addnorm_1 = AddNorm(num_head, num_hidden)
        self.positionwiseffn = PositionwiseFFN()
        self.addnorm_2 = AddNorm()
    def forward(self, X):
        return self.addnorm_2(self.positionwiseffn(self.addnorm_1(self.attention(X))))

class Multihead_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(










class AddNorm(nn.Module):

class PositinowiseFFN(nn.Module):

