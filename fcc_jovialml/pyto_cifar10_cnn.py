import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import glob
import os
import tarfile

import IPython
ipython = IPython.get_ipython()
ipython.magic('matplotlib')

base_dir = '/home/phillipl/0_para/3_resources/PyTorch'

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



