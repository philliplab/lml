import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch.nn.functional as F

x = torch.tensor([[3., 4, 5]])
w = torch.tensor([[1.], [2], [3]], requires_grad = True)
b = torch.tensor(5., requires_grad = True)

y = w * x + b
y = torch.mm(x, w) + b

y.backward()

x.grad
w.grad
b.grad

# linear regression


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.randn(2, 3, requires_grad = True)
b = torch.randn(2, requires_grad = True)

def model(x):
    return x @ w.t() + b

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

w.grad.zero_()
b.grad.zero_() 

for i in range(10000):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    print(loss)
    
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

print(w)
print(w.grad)
print(b)
print(b.grad)


## continue from 1:09:40.

import torch.nn as nn

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], 
                    [22, 37], [103, 119], [56, 70], 
                    [81, 101], [119, 133], [22, 37], 
                    [103, 119], [56, 70], [81, 101], 
                    [119, 133], [22, 37], [103, 119]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Define dataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]


batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle = True)

i = 0
for xb, yb in train_dl:
    print(xb)
    print(yb)
    print(i)
    i += 1
    if i == 10:
        break

# 1:13:52
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)

preds = model(inputs)
preds

loss_fn = F.mse_loss
loss = loss_fn(targets, preds)

# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr = 1e-5)

def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss}")

fit(10, model = model, loss_fn = loss_fn, opt = opt)

model(inputs) - targets
