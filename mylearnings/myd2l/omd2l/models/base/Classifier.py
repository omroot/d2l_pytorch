

from torch import nn
nn_Module = nn.Module

from torch.nn import functional as F

from omd2l.models.base import Module
import omd2l.utils.compute  as compute

class Classifier(Module):
    """Defined in :numref:`sec_classification`"""

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions.

        Defined in :numref:`sec_classification`"""
        Y_hat = compute.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = compute.astype(compute.argmax(Y_hat, axis=1), Y.dtype)
        compare = compute.astype(preds == compute.reshape(Y, -1), compute.float32)
        return compute.reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = compute.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = compute.reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = compute.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)