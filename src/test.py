import torch
from torch.autograd import grad
from torch import nn, tensor

x = torch.randn(2, 1)
y = torch.randn(2, 1)
A = torch.cat((x, y), 1)
print(x)
print(y)
print(A)
print(A.size())