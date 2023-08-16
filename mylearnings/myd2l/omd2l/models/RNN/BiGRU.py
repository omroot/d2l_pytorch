
from torch import nn



from omd2l.models.base.Module import Module
from omd2l.models.RNN.RNN import RNN

class BiGRU(RNN):
    def __init__(self, num_inputs, num_hiddens):
        Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        self.num_hiddens *= 2

