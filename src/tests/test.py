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

A = torch.randn(3, 1)
B = torch.randn(3, 1)
res = torch.einsum('ij,ij->', A, B)
print("A:", A[0:2])
print("B:", B)
print("res:", res)
print("dot prod.:", A.T @ B)

