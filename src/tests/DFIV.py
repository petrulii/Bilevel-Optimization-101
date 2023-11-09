import sys
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import MSELoss
#from torch.optim.lr_scheduler import StepLR

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model.utils import *
# The experiment-specific functions
from my_data.dsprite.trainer import DFIVTrainer
from my_data.dsprite.dspriteDFIV_original import *

# Set seed
seed = 42
set_seed(seed)

# Set logging
wandb.init(group="Dsprites_DFIV")

# Setting the device to GPUs if available.
if torch.cuda.is_available():
    device = "cuda"
    print("All good, switching to GPUs.")
else:
    device = "cpu"
    print("No GPUs found, setting the device to CPU.")

# Setting hyper-parameters
max_epochs = 10000
max_outer_iters = 100
max_inner_iters = 20
eval_every_n = 100
lam1 = 0.1
lam2 = 0.1

# Get data
#inner_data, outer_data, test_data = generate_dsprite_data(train_size=6, val_size=6, device=device)
test_data = generate_test_dsprite(device=device)
train_data = generate_train_dsprite(data_size=5000, rand_seed=seed, device=device)
inner_data, outer_data = split_train_data(train_data, split_ratio=0.5)

# Neural networks for dsprites data
inner_model, outer_model = build_net_for_dsprite(seed)
inner_model.to(device)
outer_model.to(device)
print("First inner layer:", list(inner_model[0].parameters())[0])
print("First outer layer:", list(outer_model[0].parameters())[0])

# Optimizer that improves the approximation of h*
inner_lr = 1e-4
inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=inner_lr, weight_decay=lam1)
inner_scheduler = None

# Optimizer that improves the outer variable
outer_lr = 1e-4
outer_optimizer = torch.optim.Adam(outer_model.parameters(), lr=outer_lr, weight_decay=lam2)
outer_scheduler = None

# Print configuration
suffix = "DFIC_example"+"_inner_lr"+str(inner_lr)+"_outer_lr"+str(outer_lr)
print("Run configuration:", suffix)

# Initialize and train the DFIVTrainer
dfiv_trainer = DFIVTrainer(
    outer_model, inner_model, outer_optimizer, inner_optimizer,
    train_params = {"stage1_iter": max_inner_iters, "stage2_iter": 1, "max_epochs": max_epochs, "lam1": lam1, "lam2": lam2}
)
# Solve the two-stage regression problem
dfiv_trainer.train(inner_data, outer_data, test_data)