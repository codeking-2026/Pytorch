import torch
x = torch.tensor([[1,2]])
y = torch.tensor([[1],[2]])

print(x.shape)
# torch.Size([1,2]) # one entity of two items
print(y.shape)
# torch.Size([2,1]) # two entities of one item each
print(x.dtype)
# torch.int64

x = torch.tensor([False, 1, 2.0])
print(x)
# tensor([0., 1., 2.])

torch.zeros((3, 4))

torch.ones((3, 4))
torch.randint(low=0, high=10, size=(3,4))
torch.rand(3, 4)
torch.randn((3,4))

import numpy as np
x = np.array([[10,20,30],[2,3,4]])
y = torch.tensor(x)
print(type(x), type(y))