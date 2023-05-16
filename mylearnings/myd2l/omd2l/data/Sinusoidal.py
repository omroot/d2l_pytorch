import collections
import random
import re
import torch


from omd2l.data.DataModule import DataModule
from omd2l.utils.io_utils import download
from omd2l.nlp.Vocab import Vocab
import omd2l.config as cfg


class SinusoidalData(DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = torch.arange(1, T + 1, dtype=torch.float32)
        self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2


    def get_dataloader(self, train):
        features = [self.x[i: self.T - self.tau + i] for i in range(self.tau)]
        self.features = torch.stack(features, 1)
        self.labels = self.x[self.tau:].reshape((-1, 1))
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.features, self.labels], train, i)



