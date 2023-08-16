

import torch
from torch import nn

nn_Module = nn.Module


from omd2l.models.base.Classifier import Classifier


class DropoutMLPScratch(Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def dropout_layer(self, X, dropout):
        assert 0 <= dropout <= 1
        if dropout == 1: return torch.zeros_like(X)
        mask = (torch.rand(X.shape) > dropout).float()
        return mask * X / (1.0 - dropout)

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:
            H1 = self.dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = self.dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)