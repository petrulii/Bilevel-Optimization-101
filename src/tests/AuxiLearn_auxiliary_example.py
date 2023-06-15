import sys
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import autograd
from torch.autograd import grad
from torch.func import functional_call
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd.functional import hessian
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import os
from pathlib import Path
import torchvision
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model.InnerSolution.InnerSolution import InnerSolution
from model.BilevelProblem.BilevelProblem import BilevelProblem
from model.NeuralNetworks.NeuralNetworkOuterModel import NeuralNetworkOuterModel
from model.NeuralNetworks.NeuralNetworkInnerModel import NeuralNetworkInnerModel
from model.utils import *

# Add AuxiLearn project directory path
sys.path.append('/home/clear/ipetruli/projects/AuxiLearn')

from auxilearn.hypernet import MonoLinearHyperNet
from auxilearn.optim import MetaOptimizer

# Set logging
wandb.init(group="AuxiLearn_auxiliary_example", job_type="eval")

# Setting the directory to save figures to.
figures_dir = "figures/"

# Setting the device to GPUs if available.
if torch.cuda.is_available():
    device = "cuda"
    print("All good, switching to GPUs.")
else:
    device = "cpu"
    print("No GPUs found, setting the device to CPU.")

# Setting hyper-parameters
nb_aux_tasks = 2
batch_size = 128
max_epochs = 100
max_outer_iters = 100
max_inner_dual_iters, max_inner_iters = 3, 15
eval_every_n = 7
n_meta_loss_accum = 1

# Get data
n, m, m_out, m_in, X_train, X_val, y_train, y_val, coef = auxiliary_toy_data()

# Dataloaders for inner and outer data
inner_data = Data(X_train, y_train)
inner_dataloader = DataLoader(dataset=inner_data, batch_size=batch_size, shuffle=True)
outer_data = Data(X_val, y_val)
outer_dataloader = DataLoader(dataset=outer_data, batch_size=batch_size, shuffle=True)
test_dataloader = outer_dataloader

# Inner model
# Neural network to approximate the function h*
inner_model = (NeuralNetworkInnerModel([2, 10, 20, 10, 3])).to(device)
# Optimizer that improves the approximation of h*
inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-4)
inner_scheduler = None
# Neural network to approximate the function a*
inner_dual_model = (NeuralNetworkInnerModel([2, 10, 20, 10, 3])).to(device)
# Optimizer that improves the approximation of a*
inner_dual_optimizer = torch.optim.Adam(inner_dual_model.parameters(), lr=1e-5)
inner_dual_scheduler = None

# Outer model
# The outer neural network parametrized by the outer variable mu
outer_model = (NeuralNetworkOuterModel([nb_aux_tasks, 1])).to(device)
# Initialize mu
mu0 = (torch.ones((nb_aux_tasks,1))).to(device)
#mu0.normal_(mean=1, std=1)
# Optimizer that improves the outer variable mu
outer_optimizer = torch.optim.Adam(outer_model.parameters(), lr=1e-3)
outer_scheduler = None

meta_optimizer = MetaOptimizer(meta_optimizer=outer_optimizer, hpo_lr=1e-4, truncate_iter=max_inner_dual_iters, max_grad_norm=25)

# Print configuration
suffix = "Auxiliary_example"
print("Run configuration:", suffix)

wandb.watch((inner_model),log_freq=100)

# Gather all models
inner_models = (inner_model, inner_optimizer, inner_scheduler, inner_dual_model, inner_dual_optimizer, inner_dual_scheduler)
outer_models = (outer_model, outer_optimizer, outer_scheduler)

# Loss helper functions
relu = torch.nn.ReLU()
MSE = nn.MSELoss()

# Outer objective function
def fo(inner_value, labels):
    (main_pred, aux_pred) = inner_value
    (main_label, aux_label) = labels
    loss = MSE(main_pred, main_label)
    return loss

# Inner objective function
def fi(h_X_in, y_in):
    # Keep mu positive
    (main_pred, aux_pred) = h_X_in
    aux_pred = torch.sigmoid(aux_pred)
    (main_label, aux_label) = y_in
    aux_label = aux_label.T
    loss = MSE(main_pred, main_label)
    aux_loss_vector = torch.zeros((nb_aux_tasks,1)).to(device)
    for task in range(nb_aux_tasks):
        aux_loss_vector[task] = MSE(aux_pred[:,task], aux_label[:,task])
    # Essentially dot product with mu
    loss += outer_model(aux_loss_vector.T)[0,0]
    return loss

def hyperstep():
    """
    Compute hyper-parameter gradient.
    """
    # Get loss for outer data
    iters = 0
    total_meta_val_loss = .0
    for data in outer_dataloader:
        X_outer, main_label, aux_label, data_id = data
        aux_label = torch.stack(aux_label)
        y_outer = (main_label.to(device, dtype=torch.float), aux_label.to(device, dtype=torch.float))
        X_outer = X_outer.to(device, dtype=torch.float)
        inner_value = inner_model(X_outer)
        loss = fo(inner_value, y_outer)
        wandb.log({"out. loss": loss.item()})
        iters += 1
        total_meta_val_loss += loss
        meta_val_loss = total_meta_val_loss/iters
        if iters>=n_meta_loss_accum:
            break
    # Get loss for inner data
    iters = 0
    total_meta_train_loss = .0
    for data in inner_dataloader:
        X_inner, main_label, aux_label, data_id = data
        aux_label = torch.stack(aux_label)
        y_inner = (main_label.to(device, dtype=torch.float), aux_label.to(device, dtype=torch.float))
        X_inner = X_inner.to(device, dtype=torch.float)
        inner_value = inner_model(X_outer)
        loss = fi(inner_value, y_outer)
        iters += 1
        total_meta_train_loss += loss
        meta_train_loss = total_meta_train_loss/iters
        if iters>=n_meta_loss_accum:
            break
    # Hyperparam step
    hypergrads = meta_optimizer.step(
        val_loss=meta_val_loss,
        train_loss=total_meta_train_loss,
        aux_params=list(outer_model.parameters()),
        parameters=list(inner_model.parameters()),
        return_grads=True
    )
    return hypergrads

def optimize_inner():
    """
    Optimization loop for the inner-level model that approximates h*.
    """
    epoch_loss, epoch_iters = 0, 0
    for data in inner_dataloader:
        X_inner, main_label, aux_label, data_id = data
        aux_label = torch.stack(aux_label)
        y_inner = (main_label.to(device, dtype=torch.float), aux_label.to(device, dtype=torch.float))
        X_inner = X_inner.to(device, dtype=torch.float)
        h_X_i = inner_model(X_inner)
        # Compute the loss
        loss = fi(h_X_i, y_inner)
        # Backpropagation
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
        epoch_loss += loss.item()
        epoch_iters += 1
        if epoch_iters >= max_inner_iters:
            break
    inner_loss = epoch_loss/epoch_iters
    return inner_loss

def optimize_outer(max_epochs=10, max_iters=100, eval_every_n=10):
    """
    Find the optimal outer solution.
        param maxiter: maximum number of iterations
    """
    iters, outer_losses, inner_losses, val_accs, times, evaluated, acc_smooth = 0, [], [], [], [], False, 10
    # Making sure gradient of mu is computed.
    for epoch in range(max_epochs):
        epoch_iters = 0
        epoch_loss = 0
        for data in outer_dataloader:
            X_outer, main_label, aux_label, data_id = data
            aux_label = torch.stack(aux_label)
            y_outer = (main_label.to(device, dtype=torch.float), aux_label.to(device, dtype=torch.float))
            start = time.time()
            # Move data to GPU
            X_outer = X_outer.to(device, dtype=torch.float)
            # Inner value corresponds to h*(X_outer)
            forward_start = time.time()
            inner_loss = optimize_inner()
            inner_value = inner_model(X_outer)
            wandb.log({"duration of forward": time.time() - forward_start})
            loss = fo(inner_value, y_outer)
            wandb.log({"inn. loss": inner_loss})
            wandb.log({"out. loss": loss.item()})
            # Backpropagation
            backward_start = time.time()
            curr_hypergrads = hyperstep()
            print("mu:\n", outer_model.layer_1.weight)
            wandb.log({"duration of backward": time.time() - backward_start})
            # Make sure all weights of the single linear layer are positive or null
            for p in outer_model.parameters():
                p.data.clamp_(0)
                wandb.log({"outer var. norm": torch.norm(p.data).item()})
            # Update loss and iteration count
            epoch_loss += loss.item()
            epoch_iters += 1
            iters += 1
            duration = time.time() - start
            wandb.log({"iter. time": duration})
            times.append(duration)
            # Inner losses
            inner_losses.append(inner_loss)
            # Outer losses
            outer_losses.append(loss.item())
            if epoch_iters >= max_iters:
                break
        _, class_pred = torch.max(inner_value[0], dim=1)
    return iters, outer_losses, inner_losses, val_accs, times

# Optimize using classical implicit differention
iters, outer_losses, inner_losses, val_accs, times = optimize_outer(max_epochs=max_epochs, max_iters=max_outer_iters, eval_every_n=eval_every_n)

# Show results
print("mu:\n", outer_model.layer_1.weight)
print("\nNumber of iterations:", iters)
print("Average iteration time:", np.average(times))