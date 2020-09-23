import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid

import glob
import os
import tarfile

import IPython
ipython = IPython.get_ipython()
ipython.magic('matplotlib')

base_dir = '/home/phillipl/0_para/3_resources/PyTorch'
val_pct = 0.2
batch_size = 128*4*16
train_losses, val_losses, val_metrics = [], [], []

# Get data (and check)
# dataset_url = 'https://files.fast.ai/data/examples/cifar10.tgz'
# download_url(dataset_url, root = base_dir +'/cifar10') 
# Download fails via python but works via browser
# need to get past cloudflare protection - it is complicated
# glob.glob(base_dir + '/cifar10/train/truck/*')

dataset = ImageFolder(base_dir + '/cifar10/train', transform = ToTensor())

def show_example(img, label):
    print(f"Label: {dataset.classes[label]} ({label})")
    plt.imshow(img.permute(1, 2, 0))

show_example(*dataset[1000])

def split_indices(n, val_pct = 0.1):
    indexes = np.random.permutation(n)
    n_val = int(n * val_pct)
    val_idxs, train_idxs = indexes[:n_val], indexes[n_val:]
    return train_idxs, val_idxs

train_idxs, val_idxs = split_indices(len(dataset), val_pct)

train_sampler = SubsetRandomSampler(train_idxs)
train_dl = DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      sampler = train_sampler)

valid_sampler = SubsetRandomSampler(val_idxs)
valid_dl = DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      sampler = valid_sampler)

for xb, yb in train_dl:
    print(xb.size())
    break

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, 10).permute(1, 2, 0))
        break

show_batch(train_dl)
show_batch(valid_dl)

model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # output: bs x 16 x 16 x 16

        nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # output: bs x 16 x 8 x 8

        nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # output: bs x 16 x 4 x 4

        nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # output: bs x 16 x 2 x 2

        nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # output: bs x 16 x 1 x 1

        nn.Flatten(), # output bs x 16
        nn.Linear(16, 10) # output bs x 10

)

for images, labels in train_dl:
    print(images.shape)
    out = model(images)
    print(out.shape)
    break

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

device = get_default_device()
device

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
to_device(model, device)

def loss_batch(model, loss_func, xb, yb, opt = None, metric = None):
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result

def evaluate(model, loss_func, valid_dl, metric = None):
    with torch.no_grad():
        result = [loss_batch(model, loss_func, xb, yb, metric = metric) for xb, yb in valid_dl]
        losses, totals, metrics = zip(*result)

        total = np.sum(totals)
        avg_loss = np.sum(np.multiply(losses, totals)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, totals)) / total
        
    return avg_loss, total, avg_metric

def fit(epochs, model, loss_fn, train_dl, valid_dl,
        opt_fn = None, lr = None, metric = None):
    train_losses, val_losses, val_metrics = [], [], []

    if opt_fn is None: opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(), lr = 0.001 if lr is None else lr)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            train_loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt)

        model.eval()
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        print([epoch, epochs, train_loss, val_loss, val_metric])
    return train_losses, val_losses, val_metrics

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim = 1)
    return torch.sum(preds == targets).item() / len(preds)

val_loss, _, val_metric = evaluate(model, F.cross_entropy, valid_dl, accuracy)

num_epochs = 10
opt_fn = torch.optim.Adam
lr = 0.005

history = fit(num_epochs, model, F.cross_entropy,
              train_dl, valid_dl, opt_fn, lr, accuracy)
train_losses += history[0] 
val_losses += history[1]
val_metrics += history[2]

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb.to(device))
    _, preds = torch.max(yb, dim = 1)
    return dataset.classes[preds[0].item()]

test_dataset = ImageFolder(base_dir + '/cifar10/test', transform = ToTensor())
test_dl = DeviceDataLoader(DataLoader(test_dataset, batch_size = batch_size), device)
test_loss, _, test_acc = evaluate(model, F.cross_entropy, test_dl, accuracy)

