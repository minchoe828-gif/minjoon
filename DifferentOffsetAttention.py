class DifferentOffsetAttention(nn.Module):
    def __init__(self, num_of_queries, num_of_keys_values):
            super().__init__()
            offset_i=torch.arange(num_of_keys_values).unsqueeze(0)
            offset_j=torch.arange(num_of_queries).unsqueeze(1)
            offset = offset_i-offset_j
            self.offset = nn.Parameter(offset)

    def forward(self, Q, K, V):
        return torch.softmax(Q@K.transpose(-2,-1)/(K.shape[-1]**0.5)+self.offset.unsqueeze(0), dim=-1)@V
    
        
    
        
