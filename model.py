import torch.nn as nn

class GenericRNN(nn.Module):
    """
        A generic class for LSTM and Vanilla RNN
    """
    def __init__(self, in_dim, hidden_dim, out_dim, type='LSTM'):
        # use type == 'LSTM' to set to LSTM
        # use type == 'vanilla' to set to VanillaRNN
        super(GenericRNN, self).__init__()
        if type == 'LSTM':
            self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, \
                                     num_layers=1, bias=True, batch_first=False, dropout=0, \
                                    bidirectional=False)
        elif type == 'vanilla':
            # here nonlinearity is tanh, to keep the same as LSTM
            self.rnn = nn.RNN(input_size=in_dim, hidden_size=hidden_dim, \
                                    num_layers=1, nonlinearity='tanh', bias=True, \
                                    batch_first=False, dropout=0, bidirectional=False)
        # connect the hidden layer of lstm to fc layer
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, state_0=None):
        """
            -- input:
                sequence input: x
                initial state of hidden layer: state_0
                    For LSTM: state_0 = (h_0, c_0)
                    For VanillaRNN: state_0 = h_0
            -- output:
                sequence output
                last state of hidden layer: state_n
                    For LSTM: state_n = (h_n, c_n)
                    For VanillaRNN: state_n = h_n
            assume input is of shape (chunk_size, in_size)
            RNN needs shape (seq_size, batch_size, in_size)
        """
        x = x.view(len(x), 1, -1)
        # return the sequence output, hidden state and cell state of last data in sequence
        x, state_n = self.rnn(x, state_0)
        # convert back to shape (chunk_size, out_size)
        # and do linear layer
        x = self.fc(x.squeeze(1))
        # do softmax on x to get the logistic prediction (classification problem)
        x = self.softmax(x)
        return x, state_n
    def predict(self, x, state_0=None):
        # only return the sequential output, discard the hidden states
        return self(x)[0]
