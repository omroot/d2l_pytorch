
from torch import nn

import omd2l.utils.compute  as compute
from omd2l.models.RNN.RNNLMScratch   import RNNLMScratch


class RNNLM(RNNLMScratch):
    """Defined in :numref:`sec_rnn-concise`"""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
    def output_layer(self, hiddens):
        return compute.swapaxes(self.linear(hiddens), 0, 1)