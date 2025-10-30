class MyLSTMSeq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self):
        super().__init__(self, embed_size, vocab_size, num_hiddens, num_layers, dropout)
        self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers, dropout=dropout) 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)

    def forward(self, X, state):
        #X:(batch, sequence) -> (batch, sequence, embed_size)
        X=self.embedding(X)
        #X:(b,s,e) -> (s, b, e) 
        X.permute(1,0,2)
        #enc_output:(sequence, batch, hidden)
        #hidden_state:(layer, batch, hidden)
        enc_output, hidden_state, enc_valid_lens = state
        for x in X:
            #query:(batch, hidden) -> (batch, num of query=1, hidden)
            query = hidden_state[-1].unsqueeze(1)
            #context : (Batch, num of query=1, hidden)
            context = self.attention(query, enc_output, enc_output, enc_valid_lens)
            x=torch.cat((context, x.unsqueeze(1)), dim=-1)
            
        
        
