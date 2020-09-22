import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
batch_size = 100

# Get data (and check)
# dataset_url = 'https://files.fast.ai/data/examples/cifar10.tgz'
# download_url(dataset_url, root = base_dir +'/cifar10') 
# Download fails via python but works via browser
# need to get past cloudflare protection - it is complicated
# glob.glob(base_dir + '/cifar10/train/truck/*')

dataset = ImageFolder(base_dir + '/cifar10/train', transform = ToTensor())

dataset[19000][0].shape

plt.imshow(dataset[19000][0].numpy())

dataset[19000][0].permute(1, 2, 0).shape

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

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, 10).permute(1, 2, 0))
        break

#show_batch(train_dl)
#show_batch(valid_dl)










