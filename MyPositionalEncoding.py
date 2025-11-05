class MyPositionalEncoding(nn.Module):
    def __init__(self, max_sequences, num_embed):
        super().__init__()
        self.P = torch.zeros((max_sequences, num_embed))
        X_row = torch.arange(max_sequences).unsqueeze(1)
        X_column = torch.pow(10000, torch.arange(0,num_embed,2)/num_embed)
        X = X_row/X_column
        self.P[:, 0::2] = torch.sin(X)
        self.P[:, 1::2] = torch.cos(X)

    def forward(self, X):
        X += self.P[:X.shape[1],:].to(X.device).unsqueeze(0)
        return X

class MyLearnablePositionalEncoding(nn.Module):
    def __init__(self, max_sequences, num_embed):
        super().__init__()
        P = torch.zeros((max_sequences, num_embed))
        X_row = torch.arange(max_sequences).unsqueeze(1)
        X_column = torch.pow(10000, torch.arange(0,num_embed,2)/num_embed)
        X = X_row/X_column
        P[:, 0::2] = torch.sin(X)
        P[:, 1::2] = torch.cos(X)
        self.P = nn.Parameter(P)
        self.linear = nn.LazyLinear(num_embed)
    def forward(self, X):
        X += self.linear(self.P[:X.shape[1],:].to(X.device).unsqueeze(0))
        return X
        
