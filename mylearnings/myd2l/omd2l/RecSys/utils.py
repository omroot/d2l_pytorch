
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from omd2l.utils.io_utils import download_extract

def read_data_ml100k():
    data_dir = download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items



def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data

def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter


def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()  # Assuming you have a function to read the data

    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)  # Assuming you have a function to split the data

    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)  # Assuming you have a function to load the data
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)  # Assuming you have a function to load the data

    train_u_tensor = torch.tensor(train_u, dtype=torch.long)
    train_i_tensor = torch.tensor(train_i, dtype=torch.long)
    train_r_tensor = torch.tensor(train_r, dtype=torch.float32)
    test_u_tensor = torch.tensor(test_u, dtype=torch.long)
    test_i_tensor = torch.tensor(test_i, dtype=torch.long)
    test_r_tensor = torch.tensor(test_r, dtype=torch.float32)

    train_set = TensorDataset(train_u_tensor, train_i_tensor, train_r_tensor)
    test_set = TensorDataset(test_u_tensor, test_i_tensor, test_r_tensor)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return num_users, num_items, train_loader, test_loader