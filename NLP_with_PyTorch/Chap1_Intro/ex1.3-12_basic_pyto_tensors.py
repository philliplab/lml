import torch
import numpy as np

def describe(x):
    print(f"Shape: {x.shape}")
    print(f"Type: {x.type()}")
    print(f"Values:\n {x}")


describe( torch.Tensor(2, 3, 4) )


describe( torch.rand(2, 3) )
describe( torch.randn(3, 2) )


describe( torch.ones(2, 3) )
describe( torch.zeros(2, 3) )


describe( torch.Tensor([[1,2,3], [4,5,6]]) )


describe( torch.from_numpy( np.random.normal(5, 10, (3,3) ) ) )


describe( torch.Tensor([[1,2,3], [4,5,6]]).long() )


describe( torch.tensor([[1,2,3], [4,5,6]], dtype = torch.int64) )


x = torch.rand(2, 3)
y = torch.randn(3, 2)

torch.add(x, x)
torch.mm(x, y)
torch.mm(y, x)


torch.Tensor(range(9)).view(3,3)

torch.sum(x, dim = 0)
torch.sum(x, axis = 0)

torch.transpose(x, 0, 1)
torch.t(x)


x
x[1,1]
x[:1,1]
x[1,:1]
x[:1,:2]



