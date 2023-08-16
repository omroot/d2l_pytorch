
from torch import nn


from omd2l.models.base.Module import Module
from omd2l.models.RNN.RNN import RNN


class LSTM(RNN):
    def __init__(self, num_inputs, num_hiddens):
        Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_inputs, num_hiddens)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)




