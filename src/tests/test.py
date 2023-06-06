import sys
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import grad
from torch.func import functional_call
from torch.autograd.functional import hessian
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

import torch

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out * 2 * x

x = torch.randn(2, 1, requires_grad=True, dtype=torch.double)
print("x:", x)
res = (Square.apply(x))
print("res:", res)
res.backward()
print("x.grad:", x.grad)
#print(torch.autograd.gradcheck(Square.apply, x))
#print(torch.autograd.gradgradcheck(Square.apply, x))