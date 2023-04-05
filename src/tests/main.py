import sys
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import grad
from torch.autograd.functional import hessian
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src/')

from model.InnerSolution.InnerSolution import InnerSolution
from model.BilevelProblem.BilevelProblem import BilevelProblem
from model.utils import set_seed, plot_1D_iterations, plot_2D_functions, plot_loss

class Data(Dataset):
	"""
	A class for input data.
	"""
	def __init__(self, X, y):
		self.X = X
		self.y = y
		self.len = len(self.y)

	def __getitem__(self, index):
		return self.X[index], self.y[index]

	def __len__(self):
		return self.len

def load_data(self, X_inner, y_inner, X_outer=None, y_outer=None):
    """
    Loads data into a type suitable for batch training.
        param X_inner: data of the inner objective
        param y_inner: labels of the inner objective
        param X_outer: data of the outer objective
        param y_outer: labels of the outer objective
    """
    self.X_inner = X_inner
    self.y_inner = y_inner
    self.inner_data = Data(X_inner, y_inner)
    self.inner_dataloader = DataLoader(dataset=self.inner_data, batch_size=self.batch_size, shuffle=True)
    self.X_outer = X_outer
    self.y_outer = y_outer
    if not (self.X_outer is None and self.y_outer is None):
        self.outer_data = Data(X_outer, y_outer)
        self.outer_dataloader = DataLoader(dataset=self.outer_data, batch_size=self.batch_size, shuffle=True)

# Setting the random seed.
set_seed()

# Setting the directory to save figures to.
figures_dir = "figures/"

# Setting the device to GPUs if available.
if torch.cuda.is_available():
    device = "cuda"
    print("All good, switching to GPUs.")
else:
    device = "cpu"
    print("No GPUs found, setting the device to CPU.")

# Initialize dimesnions
n, m, m_out, m_in, batch = 2, 100000, 30000, 70000, 100
# The coefficient tensor of size (n,1) filled with values uniformally sampled from the range (0,1)
coef = np.array([[1],[1]]).astype('float32')#np.random.uniform(size=(n,1)).astype('float32')
coef_harm = np.array([[2],[-4]]).astype('float32')#np.random.uniform(size=(n,1)).astype('float32')
# The data tensor of size (m,n) filled with values uniformally sampled from the range (0,1)
X = np.random.uniform(size=(m, n)).astype('float32')
# True h_star
h_true = lambda X: X @ coef
h_harm = lambda X: X @ coef_harm
y_main = h_true(X)+np.random.normal(scale=1.2, size=(m,1)).astype('float32')
y_aux1 = h_true(X)+np.random.normal(size=(m,1)).astype('float32')
y_aux2 = h_harm(X)+np.random.normal(size=(m,1)).astype('float32')
y = np.hstack((y_main, y_aux1, y_aux2))
# Split X into 2 tensors with sizes [m_in, m_out] along dimension 0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
# Convert everything to PyTorch tensors
X_train, X_val, y_train, y_val, coef = (torch.from_numpy(X_train)), (torch.from_numpy(X_val)), (torch.from_numpy(y_train)), (torch.from_numpy(y_val)), (torch.from_numpy(coef))
#X_val_t = X_val.to(device)
#y_val_t = y_val.to(device)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("True coeficients:", coef)
print("X training data:", X_train[1:5])
print("y training labels:", y_train[1:5])
print()

max_epochs = 1
mu_0_value = 1.
mu0 = (torch.full((2,1), mu_0_value)).to(device)
inner_data = Data(X_train, y_train)
inner_dataloader = DataLoader(dataset=inner_data, batch_size=batch, shuffle=True)
outer_data = Data(X_val, y_val)
outer_dataloader = DataLoader(dataset=outer_data, batch_size=batch, shuffle=True)


#########################################################
############ NEURAL IMPLICIT DIFFERENTIATION ############
#########################################################

# Objective functions
MSE = lambda h, X, y: (1/2)*torch.mean(torch.pow((h(X) - y),2)) 
fo = lambda mu, h_X_out, y_out: ((1/2)*torch.mean(torch.pow((h_X_out - torch.reshape(y_out[:,0], (len(y_out),1))),2))) + 0*(torch.sum(mu))
fi = lambda mu, h_X_in, y_in: ((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,0], (len(y_in),1))),2))) + mu[0]*((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,1], (len(y_in),1))),2))) + mu[1]*((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,2], (len(y_in),1))),2)))

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, outer_dataloader, inner_dataloader, device, batch_size=batch)#, X_val_t=X_val_t, y_val_t=y_val_t)
outer_optimizer = torch.optim.SGD([mu0], lr=0.05)
nb_iters, iters, losses, times = bp_neural.optimize(mu0, outer_optimizer, max_epochs=max_epochs)

# Show results
print("NEURAL IMPLICIT DIFFERENTIATION")
print("Number of iterations:", nb_iters)
print("Outer variable values:", iters)
print("Outer loss values:", losses)
print("Average iteration time:", np.average(times))
print()

plot_loss(figures_dir+"out_loss_NID", losses, title="Outer loss of neur. im. diff.")