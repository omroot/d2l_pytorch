
import torch
from torch import nn
from torch.nn import functional as F

from omd2l.models.base.Classifier import Classifier
from omd2l.models.RNN import RNN
import omd2l.utils.compute  as compute

class RNNLMScratch(Classifier):
    """Defined in :numref:`sec_rnn-scratch`"""
    def __init__(self, 
                 rnn: RNN,
                 vocab_size: int,
                 lr: float=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(
            compute.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(compute.zeros(self.vocab_size))
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', compute.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', compute.exp(l), train=False)

    def one_hot(self, X):
        """Defined in :numref:`sec_rnn-scratch`"""
        # Output shape: (num_steps, batch_size, vocab_size)
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)

    def output_layer(self, rnn_outputs):
        """Defined in :numref:`sec_rnn-scratch`"""
        outputs = [compute.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return compute.stack(outputs, 1)


    def forward(self, X, state=None):
        """Defined in :numref:`sec_rnn-scratch`"""
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def predict(self, prefix, num_preds, vocab, device=None):
        """Defined in :numref:`sec_rnn-scratch`"""
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = compute.tensor([[outputs[-1]]], device=device)
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1:  # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:  # Predict `num_preds` steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(compute.reshape(compute.argmax(Y, axis=2), 1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])
