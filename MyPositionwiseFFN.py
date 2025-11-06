class MyPositionwiseFFN(nn.Module):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.linear_1 = nn.Linear(input_dimension, hidden_dimension)
        self.linear_2 = nn.Linear(hidden_dimension, input_dimension)

    def forward(self, input):
        #input.shape: (Batch,sequence, feature)
        return self.linear_2(torch.relu(self.linear_1(input)))
    
