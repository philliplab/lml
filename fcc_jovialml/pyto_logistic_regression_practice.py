import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchvision

import IPython
ipython = IPython.get_ipython()
ipython.magic('matplotlib')

from torchvision.datasets import MNIST

base_dir = '/home/phillipl/0_para/3_resources/PyTorch'
batch_size = 100
val_percentage = 0.2
n_features = 28*28
n_targets = 10
learning_rate = 0.001

dataset = MNIST(root = base_dir + '/MNIST',
                train = True,
                transform = torchvision.transforms.ToTensor())

shuffled_indexes = np.random.permutation(range(len(dataset)))
val_indexes = shuffled_indexes[:int(val_percentage*len(dataset))] 
train_indexes = shuffled_indexes[int(val_percentage*len(dataset)):] 

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
train_dl = torch.utils.data.DataLoader(dataset, 
                                       batch_size = batch_size,
                                       sampler = train_sampler)

val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indexes)
val_dl = torch.utils.data.DataLoader(dataset,
                                     batch_size = batch_size,
                                     sampler = val_sampler)

class MnistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_targets)

    def forward(self, xb):
        xbr = xb.reshape(-1, n_features)
        preds = self.linear(xbr)
        return preds

def loss_batch(model, loss_fn, xb, yb, opt = None, metric_fn = None):
    preds = model(xb)
    loss = loss_fn(preds, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    metric = None
    if metric_fn is not None:
        metric = metric_fn(preds, yb)
    return loss.item(), len(xb), metric

def evaluate(model, loss_fn, opt, metric_fn):
    with torch.no_grad():
        result = [loss_batch(model, loss_fn, xb, yb, opt = None, metric_fn = metric_fn) for xb, yb in val_dl]
        losses, nums, metrics = zip(*result)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums))/total
        avg_metric = np.sum(np.multiply(metrics, nums))/total
    return avg_loss, total, avg_metric

#outputs = preds
#labels = yb
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.sum(preds == labels).item() / len(preds)

def fit(epochs, model, loss_fn, opt, metric_fn):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            avg_loss, total, avg_metric = loss_batch(model, loss_fn, xb, yb, opt, metric_fn)
        print(evaluate(model, loss_fn, opt, metric_fn))

model = MnistModel()
loss_fn = torch.nn.functional.cross_entropy
opt = torch.optim.SGD(model.parameters(), lr = learning_rate)
metric_fn = accuracy
for xb, yb in train_dl:
    print(xb.shape)
    break

fit(10, model, loss_fn, opt, metric_fn)








