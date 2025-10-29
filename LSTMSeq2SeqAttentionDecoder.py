class MyLSTMSeq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self):
        super().__init__(self, embed_size, vocab_size, num_hiddens, num_layers, dropout)
        self.rnn = nn.LSTM(embed_size, num_hiddens, num_layers, dropout=dropout) 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)

    
