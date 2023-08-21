import torch
import torch.nn as nn

class HingeLossbRec(nn.Module):
    def __init__(self):
        super(HingeLossbRec, self).__init__()

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = torch.sum(torch.maximum(-distances + margin, torch.tensor(0.0)))
        return loss
