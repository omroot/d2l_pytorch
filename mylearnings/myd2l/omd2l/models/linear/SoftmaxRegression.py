

import torch
from torch import nn
nn_Module = nn.Module
from torch.nn import functional as F
from omd2l.models.base.Classifier import Classifier

class SoftmaxRegression(Classifier):
    """Defined in :numref:`sec_softmax_concise`"""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))
    def forward(self, X):
        return self.net(X)

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')

class SoftmaxRegressionTabular(Classifier):
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net =  nn.LazyLinear(num_outputs)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
    def forward(self, X):
        y_pred = torch.sigmoid(self.net(X))
        return y_pred

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')