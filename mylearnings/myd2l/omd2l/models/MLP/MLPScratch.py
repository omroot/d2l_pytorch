

import torch
from torch import nn

nn_Module = nn.Module


from omd2l.models.base.Classifier import Classifier



class MLPScratch(Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))

    def relu(self,X):
        a = torch.zeros_like(X)
        return torch.max(X, a)
    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))
        A = torch.matmul(X, self.W1) + self.b1
        H = self.relu(A)
        return torch.matmul(H, self.W2) + self.b2


