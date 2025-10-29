class AttentionDecoder(d2l.Decoder):
    def __init__(self):
        super().__init__()

    @property
    def attention_weight(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, embed_size, num_hiddens_rnn, num_hiddens_attention, num_layers, vocab_size, dropout):
        super().__init__()
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens_rnn, num_layers)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dense = nn.LazyLinear(vocab_size)
        self.attention = d2l.AdditiveAttention(num_hiddens_attention, dropout=dropout)  

    def forward(self, X, state):
        enc_outputs, hidden_states, enc_valid_lens = state
        X = self.embedding(X).permute(1,0,2)
        outputs, self._attention_weight=[], []
        for x in X:
            x=x.unsqueeze(0)
            queries = hidden_states[-1].unsqueeze(1)
            attention = self.attention(queries, enc_outputs, enc_outputs, enc_valid_lens)
            input = torch.cat((x, attention), dim=0)
            out, hidden_state = self.rnn(input, hidden_state)
            ouputs.append(out)
            self._attention_weight.append(self.attention.attention_weight)
        outputs=torch.cat(outputs, dim=0)
        return self.dense(outputs)
    @property
    def attention_weight(self):
        return self._attentions_weight
