import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Linear(num_inputs, 1, bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add_module(f'dense_{dim}', nn.Linear(input_dim, dim))
            self.mlp.add_module(f'relu_{dim}', nn.ReLU())
            self.mlp.add_module(f'dropout_{dim}', nn.Dropout(p=drop_rate))
            input_dim = dim
        self.mlp.add_module(f'dense_output', nn.Linear(input_dim, 1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        inputs = embed_x.view(-1, self.embed_output_dim)
        x = self.linear_layer(self.fc(x).sum(dim=1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True) \
            + self.mlp(inputs)
        x = torch.sigmoid(x)
        return x
