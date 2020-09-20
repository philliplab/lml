import torch
import numpy as np

n_features = 2
x = np.array(np.random.normal(0, 1, (1000, n_features)), dtype = 'float32')
err = np.array(np.random.normal(0, 1, (1000, 1)), dtype = 'float32')
tw = np.array(np.random.random((n_features, 1)), dtype = 'float32')*n_features - 5

y = np.array(np.matmul(x, tw) + err, dtype = 'float32')

x = torch.from_numpy(x)
y = torch.from_numpy(y)

train_ds = torch.utils.data.TensorDataset(x, y)
train_dl = torch.utils.data.DataLoader(train_ds, 100, shuffle = True)

model = torch.nn.Linear(n_features, 1)
loss_fn = torch.nn.functional.mse_loss
opt = torch.optim.SGD(model.parameters(), lr = 1e-2)
print(model.weight)
print(model.bias)

preds = model(x)

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


#all_results = {}
#for i in range(10):
    
for epoch in range(10):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
    print(epoch, loss)
#    all_results[i] = {'p':list(model.parameters()),
#                      'l':loss}

[i[1]['l'].detach().numpy() for i in all_results.items()]

list(model.parameters())
tw

from sklearn.linear_model import LinearRegression
reg = LinearRegression(fit_intercept = True).fit(x, y)
reg.coef_
reg.intercept_

import pandas as pd

px = pd.DataFrame(x.numpy())
px.columns = [chr(ord('A')+i) for i in range(n_features)]
cm = px.corr()
