import torch
import torchvision

from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import IPython
ipython = IPython.get_ipython()
ipython.magic('matplotlib')

## parameters
base_dir = '/home/phillipl/0_para/3_resources/PyTorch'
val_pct = 0.2
batch_size = 1000

## Preparing the data

dataset = MNIST(root = base_dir + '/MNIST',
                train = True,
                transform = torchvision.transforms.ToTensor())

def split_indices(n, val_pct):
    n_val = int(n*val_pct)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

train_idxs, val_idxs = split_indices(len(dataset), val_pct)

train_sampler = torch.utils.data.SubsetRandomSampler(train_idxs)
val_sampler = torch.utils.data.SubsetRandomSampler(val_idxs)

train_dl = torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler = train_sampler)
val_dl = torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler = val_sampler)

class MnistModel(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        out = self.linear1(xb)
        out = torch.nn.functional.relu(out)
        out = self.linear2(out)
        return out
        
input_size = 784
num_classes = 10

model = MnistModel(input_size, hidden_size = 32,
                   out_size = num_classes)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()

def to_device(data, device):
    """Move tensor(s) to device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

for images, labels in train_dl:
    print(images.shape)
    images = to_device(images, device)
    print(images.device)
    break

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

for xb, yb in train_dl:
    print(xb.device)
    break

def loss_batch(model, loss_fn, xb, yb, opt = None, metric = None):
    pred = model(xb)
    loss = loss_fn(pred, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    metric_result = None
    if metric is not None:
        metric_result = metric(pred, yb)
    return loss.item(), len(xb), metric_result

def evaluate(model, loss_fn, valid_dl, metric):
    result = [loss_batch(model, loss_fn, xb, yb, metric = metric) for xb, yb in valid_dl]
    losses, totals, metrics = zip(*result)
    avg_loss = np.sum(np.multiply(losses, totals)) / np.sum(totals)
    avg_metric = np.sum(np.multiply(metrics, totals)) / np.sum(totals)
    return avg_loss, np.sum(totals), avg_metric

def fit(epochs, lr, model, loss_fn, train_dl, val_dl, opt_fn = None, metric = None):
    losses, metrics = [], []

    if opt_fn is None: opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(), lr = lr)

    for epoch in range(epochs):
        for xb, yb in train_dl:
            loss_batch(model, loss_fn, xb, yb, opt)

        result = evaluate(model, loss_fn, val_dl, metric)
        val_loss, total, val_metric = result

        losses.append(val_loss)
        metrics.append(val_metric)

        print([val_loss, val_metric])

    return losses, metrics

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim = 1)
    return torch.sum(preds == targets).item() / len(preds)

loss_fn = torch.nn.functional.cross_entropy
to_device(model, device)

fit(10, 0.01, model, loss_fn, train_dl, val_dl, metric = accuracy)







