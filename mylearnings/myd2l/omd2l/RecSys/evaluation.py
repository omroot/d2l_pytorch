
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset

def evaluator(net, test_loader, device):
    mse_loss = torch.nn.MSELoss()  # Mean Squared Error (MSE) loss
    rmse_list = []

    net.eval()  # Set the network to evaluation mode

    with torch.no_grad():
        for users, items, ratings in test_loader:
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            r_hat = net(users, items)
            rmse = torch.sqrt(mse_loss(r_hat, ratings))
            rmse_list.append(rmse.item())

    net.train()  # Set the network back to training mode

    return float(sum(rmse_list) / len(rmse_list))


def evaluator_autorec(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = [torch.tensor(values).to(device) for device in devices]
        scores.extend([network(i).cpu().detach().numpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)



#@save
def hit_and_auc(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if val in set(test_matrix)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if val in set(test_matrix)]
    max = len(rankedlist) - 1
    auc = 1.0 * (max - hits_all[0][0]) / max if len(hits_all) > 0 else 0
    return len(hits_k), auc


def evaluate_ranking(net, test_input, seq, candidates, num_users, num_items,
                     devices):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_users)])
    for u in range(num_users):
        neg_items = list(all_items - set(candidates[int(u)]))
        user_ids, item_ids, x, scores = [], [], [], []
        item_ids.extend(neg_items)
        user_ids.extend([u] * len(neg_items))
        x.extend([torch.tensor(user_ids)])
        if seq is not None:
            x.append(torch.tensor(seq[user_ids, :]))
        x.extend([torch.tensor(item_ids)])
        test_data_iter = DataLoader(
            TensorDataset(*x), shuffle=False, batch_size=1024)
        for index, values in enumerate(test_data_iter):
            x = [v.to(devices[0]) for v in values]
            scores.extend([list(net(*t).cpu().detach().numpy()) for t in zip(*x)])
        scores = [item for sublist in scores for item in sublist]
        item_scores = list(zip(item_ids, scores))
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        temp = hit_and_auc(ranked_items[u], test_input[u], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    return np.mean(np.array(hit_rate)), np.mean(np.array(auc))
