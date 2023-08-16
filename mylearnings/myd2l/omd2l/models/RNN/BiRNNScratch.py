
import torch


from omd2l.models.base.Module import Module

from omd2l.models.RNN.RNNScratch import RNNScratch

class BiRNNScratch(Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.f_rnn = RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled

    def forward(self, inputs, Hs=None):
        f_H, b_H = Hs if Hs is not None else (None, None)
        f_outputs, f_H = self.f_rnn(inputs, f_H)
        b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
        outputs = [torch.cat((f, b), -1) for f, b in zip(
            f_outputs, reversed(b_outputs))]
        return outputs, (f_H, b_H)