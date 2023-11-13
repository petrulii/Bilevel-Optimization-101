import sys
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import MSELoss

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model.utils import *
# The experiment-specific functions
from model_previous.FunctionApproximator.FunctionApproximator import FunctionApproximator
from model_previous.BilevelProblem.BilevelProblem import BilevelProblem
from my_data.dsprite.dspriteKernel import *

# Set seed
seed = 42
set_seed(seed)

# Setting the device to GPUs if available.
"""if torch.cuda.is_available():
    device = "cuda"
    print("All good, switching to GPUs.")
else:"""
device = "cpu"
print("No GPUs found, setting the device to CPU.")


# Setting hyper-parameters
max_epochs = 10000
max_outer_iters = 100
max_inner_iters = 20
eval_every_n = 100
lam1 = 0.1
lam2 = 0.1
batch_size = 245
alpha = 5.0
gamma = 1.0

# Get data
test_data = generate_test_dsprite(device=device)
train_data, validation_data = generate_train_dsprite(data_size=500, rand_seed=seed, device=device, val_size=10)
inner_data, outer_data = split_train_data(train_data, split_ratio=0.5)
instrumental_in, treatment_in, outcome_in, instrumental_out, treatment_out, outcome_out, treatment_test, outcome_test = inner_data.instrumental, inner_data.treatment, inner_data.outcome, outer_data.instrumental, outer_data.treatment, outer_data.outcome, test_data.treatment, test_data.structural
treatment_test, structural_test = test_data.treatment, test_data.structural
#instrumental_val, treatment_val, outcome_val = validation_data.instrumental, validation_data.treatment, validation_data.outcome

# Dataloaders for inner and outer data
data = (torch.from_numpy(instrumental_in), torch.from_numpy(treatment_in), torch.from_numpy(outcome_in), torch.from_numpy(instrumental_out), torch.from_numpy(treatment_out), torch.from_numpy(outcome_out), torch.from_numpy(treatment_test), torch.from_numpy(structural_test))
#test_data = (treatment_test, outcome_test)
#validation_data = (instrumental_val, treatment_val, outcome_val)

# Set logging
wandb.init(group="Dsprites_KernelIV")

# Loss helper functions
MSE = nn.MSELoss()

# Outer objective function
def fo(outer_param, g_z_out, Y):
    loss = 1/2*torch.norm(Y - g_z_out)**2
    return loss

# Inner objective function
def fi(outer_param, g_z_in, K_X):
    f_x_in = K_X @ outer_param
    loss = 1/2*torch.norm(f_x_in - g_z_in)**2
    return loss

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, "kernel_implicit_diff", data, batch_size=batch_size, alpha=alpha, gamma=gamma)
mu_opt, iters, n_iters, times, inner_loss, outer_loss, h_star = bp_neural.train(mu0=0, maxiter=max_epochs, step=1e-4)
print("mu:", mu_opt)