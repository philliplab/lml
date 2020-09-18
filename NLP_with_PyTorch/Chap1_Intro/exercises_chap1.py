import torch

#1
x = torch.Tensor([[1,2,3], [4,5,6]])
x
y = x.unsqueeze(dim =0) 
y

#2
y.squeeze(dim=0)

#3
torch.rand(5,3)*4 + 3

#4
torch.randn(2,2)

#5
(torch.Tensor([1,1,1,0,1,0]) -1).nonzero()

#6
x = torch.Tensor([2,3,4])
torch.stack([x, x, x])
x = x.unsqueeze(dim = 1)
x.expand(3,4)

x = torch.rand(3, 1)
x.expand(3,4)

#7
torch.bmm(torch.rand(3,5,4), torch.rand(3,4,5))

#8
x = torch.rand(4,5)
x = torch.stack([x,x,x])
torch.bmm(torch.rand(3,5,4), x)
