
import pickle
import pandas as pd

import torch
from torch import nn
from torch.utils import data
nn_Module = nn.Module





from torch.utils.data import Dataset

from torch.utils.data import TensorDataset
from omd2l.data.DataModule import DataModule


class MyData(Dataset):
    def __init__(self, data):
        self.data = data


    def __getitem__(self, index):
        return self.data.iloc[index]


    def __len__(self):
        return len(self.data)


class StocksData(DataModule):
    def __init__(self, batch_size, features, response, dataset=None, raw_train= None, raw_val = None,train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.raw_train is None:
            with open("./feat_alpha_df.pickle", "rb") as handle:
                dataset = pickle.load(handle)
            dataset['ff_alpha_direction'] = 1.0*(dataset['ff_alpha']>0)
            dataset['tradeDate'] = pd.to_datetime(dataset['date']).dt.date
            dataset.sort_values(by = 'tradeDate',ascending=True, inplace = True)
            dataset.reset_index(drop = True, inplace = True)
            self.dataset = dataset
            self.raw_train =  dataset.head(round(0.6*dataset.shape[0]))
            self.raw_val = dataset.tail(round(0.39*dataset.shape[0]))


    def preprocess(self):
        self.train = self.raw_train.groupby('tradeDate')[self.features].apply(
            lambda x: (x - x.mean()) / (x.std())).copy()
        self.train[self.response] = self.raw_train[self.response]
        self.val = self.raw_val.groupby('tradeDate')[self.features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        self.val[self.response] = self.raw_val[self.response]

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val

        tensors = TensorDataset(torch.from_numpy(data[self.features].values).to(torch.float) ,  # X
                                torch.from_numpy(data[self.response].values).to(torch.long) )   # Y

        return torch.utils.data.DataLoader( tensors,
                                           self.batch_size,
                                           shuffle=train,
                                           num_workers=self.num_workers)

