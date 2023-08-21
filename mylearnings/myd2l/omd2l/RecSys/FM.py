import torch
import torch.nn as nn

class FM(nn.Module):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        embedding_x = self.embedding(x)
        square_of_sum = torch.sum(embedding_x, dim=1) ** 2
        sum_of_square = torch.sum(embedding_x ** 2, dim=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdim=True)
        x = torch.sigmoid(x)
        return x
