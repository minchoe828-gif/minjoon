class MyLSTMSeq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, embed_size, vocab_size, num_hiddens, num_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(embed_size+num_hiddens, num_hiddens, num_layers, dropout=dropout) 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        outputs, (hidden_state, memory_cell) = enc_outputs
        return (outputs.permute(1,0,2), hidden_state, memory_cell, enc_valid_lens)
    def forward(self, X, state):
        #X:(batch, sequence) -> (batch, sequence, embed_size) -> (s,b,e)
        X=self.embedding(X).permute(1,0,2)
        #enc_output:(batch,sequence, hidden)
        #hidden_state & memory_cell:(layer, batch, hidden)
        enc_output, hidden_state, memory_cell, enc_valid_lens = state
        outputs, _attention_weight = [], []
        for x in X:
            #query:(batch, hidden) -> (batch, num of query=1, hidden)
            query = hidden_state[-1].unsqueeze(1)
            #context : (Batch, num of query=1, hidden)
            context = self.attention(query, enc_output, enc_output, enc_valid_lens)
            x=torch.cat((context, x.unsqueeze(1)), dim=-1)
            out, (hidden_state, memory_cell) = self.rnn(x.permute(1,0,2), (hidden_state, memory_cell))
            outputs.append(out)
            _attention_weight.append(self.attention.attention_weight)
        #outputs:(sequence, batch, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1,0,2), [enc_output, hidden_state, memory_cell, enc_valid_lens]
        
        
        
