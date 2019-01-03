import torch.nn as nn


class RNNClassifier(nn.Module):

    def __init__(self, 
                 input_size, 
                 word_vec_dim, 
                 hidden_size, 
                 n_classes,
                 n_layers=4, 
                 dropout_p=.3
                 ):
        self.input_size = input_size  # vocabulary_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_dim)
        self.rnn = nn.LSTM(input_size=word_vec_dim,
                           hidden_size=hidden_size, # 256
                           num_layers=n_layers, # 4
                           dropout=dropout_p, # 0.5
                           batch_first=True,
                           bidirectional=True
                           )
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x): #forward에서 정의함
        # |x| = (batch_size, length) -> index
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_dim)
        x, _ = self.rnn(x) # _: hidden state 무시함
        # |x| = (batch_size, length, hidden_size * 2) #bi direction 이므로..
        y = self.activation(self.generator(x[:, -1])) #(bs, hs*2) ->
        # |y| = (batch_size, n_classes)

        return y
