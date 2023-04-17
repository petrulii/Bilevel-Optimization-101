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

from model.InnerSolution.InnerSolution import InnerSolution
from model.BilevelProblem.BilevelProblem import BilevelProblem
from model.NeuralNetworks.NeuralNetworkOuterModel import NeuralNetworkOuterModel
from model.NeuralNetworks.NeuralNetworkInnerModel import NeuralNetworkInnerModel
from model.NeuralNetworks.NeuralNetworkInnerDualModel import NeuralNetworkInnerDualModel
from model.utils import set_seed, plot_loss, tensor_to_state_dict

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
n, m, m_out, m_in, batch = 2, 100000, 30000, 70000, 1000
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

# Inner model
layer_sizes = [2, 10, 20, 10, 1]
# Neural network to approximate the function h*
inner_model = (NeuralNetworkInnerModel(layer_sizes)).to(device)
# Optimizer that improves the approximation of h*
inner_optimizer = torch.optim.SGD(inner_model.parameters(), lr=0.01)

# Inner dual model
# Neural network to approximate the function a*
inner_dual_model = (NeuralNetworkInnerDualModel(layer_sizes)).to(device)
# Optimizer that improves the approximation of a*
inner_dual_optimizer = torch.optim.SGD(inner_dual_model.parameters(), lr=0.01)

# Outer model
layer_sizes = [2, 1]
# The outer neural network parametrized by the outer variable mu
outer_model = (NeuralNetworkOuterModel(layer_sizes)).to(device)
# Optimizer that improves the outer variable mu
outer_optimizer = torch.optim.SGD([mu0], lr=0.1)

# Gather all models
inner_models = (inner_model, inner_optimizer, inner_dual_model, inner_dual_optimizer)
outer_models = (outer_model, outer_optimizer)

# Objective functions 
fo = lambda mu, inner_value, y_out: ((1/2)*torch.mean(torch.pow((inner_value - torch.reshape(y_out[:,0], (len(y_out),1))),2))) + 0*functional_call(outer_model, tensor_to_state_dict(outer_model, mu), torch.hstack((((1/2)*torch.mean(torch.pow((inner_value - torch.reshape(y_out[:,1], (len(y_out),1))),2))),((1/2)*torch.mean(torch.pow((inner_value - torch.reshape(y_out[:,2], (len(y_out),1))),2))))))
fi = lambda mu, inner_value, y_in: ((1/2)*torch.mean(torch.pow((inner_value - torch.reshape(y_in[:,0], (len(y_in),1))),2))) + functional_call(outer_model, tensor_to_state_dict(outer_model, mu), torch.hstack((((1/2)*torch.mean(torch.pow((inner_value - torch.reshape(y_in[:,1], (len(y_in),1))),2))),((1/2)*torch.mean(torch.pow((inner_value - torch.reshape(y_in[:,2], (len(y_in),1))),2))))))

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, outer_dataloader, inner_dataloader, outer_models, inner_models, device, batch_size=batch)
nb_iters, iters, losses, times = bp_neural.optimize(mu0, max_epochs=max_epochs)

# Show results
print("NEURAL IMPLICIT DIFFERENTIATION")
print("Number of iterations:", nb_iters)
print("Outer variable values:", iters)
print("Outer loss values:", losses)
print("Average iteration time:", np.average(times))
print()

plot_loss(figures_dir+"out_loss_NID", losses, title="Outer loss of neur. im. diff.")