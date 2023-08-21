import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, num_factors, num_users, num_items):
        super(MF, self).__init__()
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self._init_weights()


    def _init_weights(self):
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)

        # Use torch.sum and .squeeze() instead of np.sum and np.squeeze
        outputs = (P_u * Q_i).sum(dim=1) + b_u.squeeze() + b_i.squeeze()
        return outputs.flatten()
#
# import mxnet as mx
# from mxnet import autograd, gluon, np, npx
# from mxnet.gluon import nn
# from d2l import mxnet as d2l
#
# npx.set_np()
# class DeepFM(nn.Block):
#     def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
#         super(DeepFM, self).__init__()
#         num_inputs = int(sum(field_dims))
#         self.embedding = nn.Embedding(num_inputs, num_factors)
#         self.fc = nn.Embedding(num_inputs, 1)
#         self.linear_layer = nn.Dense(1, use_bias=True)
#         input_dim = self.embed_output_dim = len(field_dims) * num_factors
#         self.mlp = nn.Sequential()
#         for dim in mlp_dims:
#             self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
#             self.mlp.add(nn.Dropout(rate=drop_rate))
#             input_dim = dim
#         self.mlp.add(nn.Dense(in_units=input_dim, units=1))
#
#     def forward(self, x):
#         embed_x = self.embedding(x)
#         square_of_sum = np.sum(embed_x, axis=1) ** 2
#         sum_of_square = np.sum(embed_x ** 2, axis=1)
#         inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
#         x = self.linear_layer(self.fc(x).sum(1)) \
#             + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
#             + self.mlp(inputs)
#         x = npx.sigmoid(x)
#         return x
#
# batch_size = 2048
# data_dir = d2l.download_extract('ctr')
# train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
# test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
#                            feat_mapper=train_data.feat_mapper,
#                            defaults=train_data.defaults)
# train_iter = gluon.data.DataLoader(
#     train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
#     num_workers=d2l.get_dataloader_workers())
# test_iter = gluon.data.DataLoader(
#     test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
#     num_workers=d2l.get_dataloader_workers())