import torch
import numpy as np
import IPython
ipython = IPython.get_ipython()
ipython.magic('matplotlib')
import matplotlib.pyplot as plt

x = torch.arange(-5, 5, 0.1)
y = torch.sigmoid(x)
plt.plot(x.numpy(), y.numpy())

