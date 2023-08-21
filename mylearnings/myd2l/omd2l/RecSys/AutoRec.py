import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoRec(nn.Module):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_users, num_hidden)
        self.decoder = nn.Linear(num_hidden, num_users)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)


    def forward(self, input):
        hidden = self.dropout(F.sigmoid(self.encoder(input)))
        pred = self.decoder(hidden)
        if self.training:  # Mask the gradient during training
            return pred * torch.sign(input)
        else:
            return pred
