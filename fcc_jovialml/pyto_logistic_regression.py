import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import IPython
ipython = IPython.get_ipython()
ipython.magic('matplotlib')

base_dir = "/home/phillipl/0_para/3_resources/PyTorch/MNIST"

dataset = MNIST(root = base_dir + '/',
        train = True,
        transform = transforms.ToTensor())

img_tensor = dataset[1][0]
plt.imshow(img_tensor[0, :, : ], cmap = 'gray')

indx_perm = np.random.permutation(len(dataset))
train_idx, val_idx = indx_perm[:int(0.8*len(dataset))], indx_perm[int(0.8*len(dataset)):]

batch_size = 100

train_sampler = SubsetRandomSampler(train_idx)
train_loader = DataLoader(dataset,
                          batch_size,
                          sampler = train_sampler)

val_sampler = SubsetRandomSampler(val_idx)
val_loader = DataLoader(dataset,
                        batch_size,
                        sampler = val_sampler)

input_size = 28*28
num_classes = 10

class MnistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

def loss_batch(model, loss_fn, xb, yb, opt = None, metric = None):
    preds = model(xb)
    loss = loss_fn(preds, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result
    
def evaluate(model, loss_fn, valid_dl, metric = None):
    with torch.no_grad():
        results = [loss_batch(model, loss_fn, xb, yb, metric = metric) for xb, yb in valid_dl]
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.sum(preds == labels).item() / len(preds)

# 2:42:00

def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric = None):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt)

result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        print(f"{epoch} - {epochs} - {val_loss} - {val_metric}")

model = MnistModel()

loss_fn = F.cross_entropy

learning_rate = 0.001
opt = torch.optim.SGD(model.parameters(), lr = learning_rate)

fit(5, model, loss_fn, opt, train_loader, val_loader, accuracy)

test_dataset = MNIST(root = base_dir + '/',
        train = False,
        transform = transforms.ToTensor())

img, label = test_dataset[0]
plt.imshow(img[0, :, :], cmap = 'gray')

torch.max(model(img), dim = 1)

test_loader = DataLoader(test_dataset, batch_size = 200)
test_loss, total, test_acc = evaluate(model, loss_fn, test_loader, metric = accuracy)

torch.save(model.state_dict(), base_dir + '/mnist-logistic.pth')

model2 = MnistModel()
model2.load_state_dict(torch.load(base_dir + '/mnist-logistic.pth'))
model2.state_dict()

test_loss, total, test_acc = evaluate(model2, loss_fn, test_loader, metric = accuracy)













