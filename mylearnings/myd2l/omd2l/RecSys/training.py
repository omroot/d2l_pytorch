import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omd2l.utils.display import Animator
from omd2l.utils.time import Timer
from omd2l.utils.compute import try_gpu
from omd2l.utils.Accumulator import Accumulator
def train_recsys_rating(net, train_loader, test_loader, loss_fn, optimizer,
                        num_epochs, device=try_gpu(), evaluator=None,
                        **kwargs):
    net.to(device)
    timer = Timer()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    net.to(device)

    for epoch in range(num_epochs):
        metric, l = Accumulator(3), 0.
        for i, values in enumerate(train_loader):
            timer.start()
            input_data = [v.to(device) for v in values]
            train_feat = input_data[:-1]
            train_label = input_data[-1]

            optimizer.zero_grad()
            preds = net(*train_feat)
            ls = [loss_fn(p, s) for p, s in zip(preds, train_label)]
            loss = sum(ls)
            loss.backward()
            optimizer.step()

            l += loss.item() / len(train_loader)
            metric.add(l, train_label.size(0), train_label.numel())
            timer.stop()

        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_loader, kwargs['inter_mat'],
                                  device)
        else:
            test_rmse = evaluator(net, test_loader, device)

        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))

    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
