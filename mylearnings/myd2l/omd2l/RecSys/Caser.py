import torch
import torch.nn as nn
import torch.nn.functional as F

class Caser(nn.Module):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05):
        super(Caser, self).__init__()
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Vertical convolution layer
        self.conv_v = nn.Conv2d(1, d_prime, (L, 1))
        # Horizontal convolution layers
        h = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList()
        self.max_pool = nn.ModuleList()
        for i in h:
            self.conv_h.append(nn.Conv2d(1, d, (i, num_factors)))
            self.max_pool.append(nn.MaxPool1d(L - i + 1))
        # Fully connected layer
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Linear(d_prime * num_factors + d * L, num_factors)
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = self.Q(seq).unsqueeze(1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(out_v.size(0), self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = conv(item_embs).squeeze(3)
                t = maxp(conv_out)
                pool_out = t.squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, dim=1)
        out = torch.cat([out_v, out_h], dim=1)
        z = self.fc(self.dropout(out))
        x = torch.cat([z, user_emb], dim=1)
        q_prime_i = self.Q_prime(item_id).squeeze()
        b = self.b(item_id).squeeze()
        res = torch.sum(x * q_prime_i, dim=1) + b
        return res
