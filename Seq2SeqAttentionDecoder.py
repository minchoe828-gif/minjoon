class AttentionDecoder(d2l.Decoder):
    def __init__(self):
        super().__init__()

    @property
    def attention_weight(self):
        raise NotImplementedError

class MySeq2SeqAttentionDecoder(AttentionDecoder):
    #num_hiddens를 둘로 나누었다. 하나는 attention, 하나는 rnn 굳이 두 대상을 동일하게 다루는지 잘 모르겠다 .
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
            #x의 unsqueeze는 1 에서 수행되어야 한다. attention.shape : (batch_size, 1, num_hiddens)
            x=x.unsqueeze(0)
            queries = hidden_states[-1].unsqueeze(1)
            attention = self.attention(queries, enc_outputs, enc_outputs, enc_valid_lens)
            input = torch.cat((x, attention), dim=0)
            #해당 input을 바로 넣으면 안되고 차원을 permute 시켜서 rnn에 맞게 넣어야 한다. 
            #따라서 input.shape : (batch_size, 1, num_hiddens+ embed_size) -> (1, batch_size, num_hiddens + embed_size)
            out, hidden_states = self.rnn(input, hidden_states)
            ouputs.append(out)
            self._attention_weight.append(self.attention.attention_weight)
        outputs=torch.cat(outputs, dim=0)
        #permute 를 통해서 num_step, batch_size, vocab_size -> batch_size, num_step, vocab_size 로 만들어야 한다.
        return self.dense(outputs)
    @property
    def attention_weight(self):
        return self._attentions_weight

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size+num_hiddens, num_hiddens, num_layers, dropout = dropout)
        self.dense = nn.LazyLinear(vocab_size)
        #빠트린 부분 
        #모델 초기화 과정에서 모델 파라미터들의 초기화는 필수적이다. 여기서 모델 파라미터를 activation에 맞는 다양한 initializing 방법으로 초기화 할 수 있기 때문에 내가 원하는 초기화 방식을 메서드화시킨 d2l.init_seq2seq를 사용한다.
        self.apply(d2l.init_seq2seq)

    def forward(self, X, state):
        #tensor shape comment is good 
        #enc_outputs.shape : (batch_size, num_steps, num_hiddens)
        #hiddens_state.shape : (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        #shape of X : (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1,0,2)
        outputs, self._attention_weights = [], [] 
        for x in X:
            #shape of query : (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hiddens_state[-1], dim=1)
            #shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens
            )
            #concatenate
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1,0,2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weigths)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1,0,2), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
        
            
            
