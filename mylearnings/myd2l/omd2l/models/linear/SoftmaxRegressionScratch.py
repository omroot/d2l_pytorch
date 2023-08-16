
import torch

from torch import nn


nn_Module = nn.Module


from omd2l.models.base.Classifier import Classifier

class SoftmaxRegressionScratch(Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

    def softmax(self, X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdims=True)
        return X_exp / partition  # The broadcasting mechanism is applied here

    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat[list(range(len(y_hat))), y]).mean()


    def loss(self, y_hat, y):
        return self.cross_entropy(y_hat, y)

    def forward(self, X):
        return self.softmax(torch.matmul(X.reshape((-1, self.W.shape[0])), self.W) + self.b)



