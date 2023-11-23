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
from model.BilevelProblem.BilevelProblem import BilevelProblem
from my_data.dsprite.dspriteBilevel import *

# Set seed
seed = 42
set_seed(seed)

# Setting the device to GPUs if available.
if torch.cuda.is_available():
    device = "cuda"
    print("All good, switching to GPUs.")
else:
    device = "cpu"
    print("No GPUs found, setting the device to CPU.")

# Setting hyper-parameters
batch_size = 2400
max_epochs = 100
max_outer_iters = 100
max_inner_dual_iters, max_inner_iters = 1, 1
eval_every_n = 1
lam1 = 0.
lam2 = 1e-3
u_dim = 33

# Get data
#instrumental_train, treatment_train, outcome_train, instrumental_val, treatment_val, outcome_val, treatment_test, outcome_test = generate_dsprite_data(train_size=6, val_size=6)
test_data = generate_test_dsprite(device=device)
train_data, validation_data = generate_train_dsprite(data_size=5000, rand_seed=seed, device=device, val_size=200)
inner_data, outer_data = split_train_data(train_data, split_ratio=0.5)
instrumental_in, treatment_in, outcome_in, instrumental_out, treatment_out, outcome_out, treatment_test, outcome_test = inner_data.instrumental, inner_data.treatment, inner_data.outcome, outer_data.instrumental, outer_data.treatment, outer_data.outcome, test_data.treatment, test_data.structural
instrumental_val, treatment_val, outcome_val = validation_data.instrumental, validation_data.treatment, validation_data.outcome

# Dataloaders for inner and outer data
inner_data = DspritesData(instrumental_in, treatment_in, outcome_in)
inner_dataloader = DataLoader(dataset=inner_data, batch_size=batch_size, shuffle=True, drop_last=True)
outer_data = DspritesData(instrumental_out, treatment_out, outcome_out)
outer_dataloader = DataLoader(dataset=outer_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_data = DspritesTestData(treatment_test, outcome_test)
validation_data = DspritesData(instrumental_val, treatment_val, outcome_val)

# Neural networks for dsprites data
inner_model, inner_dual_model, outer_model = build_net_for_dsprite(seed)
inner_model.to(device)
inner_dual_model.to(device)
outer_model.to(device)
print("First inner layer:", list(inner_model[0].parameters())[0])
print("First outer layer:", list(outer_model[0].parameters())[0])

# Optimizer that improves the approximation of h*
inner_lr = 1e-5
inner_wd = 1e-3
inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=inner_lr, weight_decay=inner_wd)
inner_scheduler = None

# Optimizer that improves the approximation of a*
inner_dual_lr = 1e-5
inner_dual_wd = 5e-3
inner_dual_optimizer = torch.optim.Adam(inner_dual_model.parameters(), lr=inner_dual_lr, weight_decay=inner_dual_wd)
inner_dual_scheduler = None

# The outer neural network parametrized by the outer variable
outer_param = state_dict_to_tensor(outer_model, device)#torch.cat((torch.rand(u_dim).to(device), state_dict_to_tensor(outer_model, device)), 0)
outer_lr = 1e-5
outer_wd = 1e-3
# Optimizer that improves the outer variable
outer_optimizer = torch.optim.Adam([outer_param], lr=outer_lr, weight_decay=outer_wd)
outer_scheduler = None

# Print configuration
run_name = "Dsprites bilevel::="+" inner_lr:"+str(inner_lr)+", dual_lr:"+str(inner_dual_lr)+", outer_lr:"+str(outer_lr)+" inner_wd:"+str(inner_wd)+", dual_wd:"+str(inner_dual_wd)+", outer_wd:"+str(outer_wd)+", max_inner_dual_iters:"+str(max_inner_dual_iters)+", max_inner_iters:"+str(max_inner_iters)+", lam1:"+str(lam1)+", lam2:"+str(lam2)+", batch_size:"+str(batch_size)+", max_epochs:"+str(max_epochs)
print("Run configuration:", run_name)

# Set logging
wandb.init(group="Dsprites_bilevelIV", name=run_name)

# Gather all models
inner_models = (inner_model, inner_optimizer, inner_scheduler, inner_dual_model, inner_dual_optimizer, inner_dual_scheduler)
outer_models = (outer_model, outer_optimizer, outer_scheduler)

# Loss helper functions
MSE = nn.MSELoss()

# Outer objective function
def fo(outer_param, g_z_out, Y):
    feature = augment_stage2_feature(g_z_out)
    #u = torch.reshape(outer_param[:u_dim], (u_dim,1))
    #u = fit_linear(Y, feature, lam2)
    #u.detach()
    #pred = linear_reg_pred(feature, u)
    #loss = MSE(pred, Y) + lam2 * torch.norm(u) ** 2
    loss = MSE(g_z_out, Y)
    wandb.log({"out. loss term1": loss.item()})
    #wandb.log({"out. loss term2": (lam2 * torch.norm(u) ** 2).item()})
    return loss

# Inner objective function
def fi(outer_param, g_z_in, X):
    #outer_param_without_u = outer_param[u_dim:]
    #outer_NN_dic = tensor_to_state_dict(outer_model, outer_param_without_u, device)
    outer_NN_dic = tensor_to_state_dict(outer_model, outer_param, device)
    treatment_feature = torch.func.functional_call(outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X)
    #feature = augment_stage1_feature(g_z_in)
    #V = fit_linear(treatment_feature, feature, lam1)
    #pred = linear_reg_pred(feature, V)
    #V = outer_param[-u_dim:]
    loss = MSE(g_z_in, treatment_feature)
    #loss = torch.norm((treatment_feature - pred)) ** 2 + lam1 * torch.norm(V) ** 2
    return loss

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, outer_dataloader, inner_dataloader, outer_models, inner_models, device, batch_size=batch_size, max_inner_iters=max_inner_iters, max_inner_dual_iters=max_inner_dual_iters)
# Solve the bilevel problem
iters, outer_losses, inner_losses, test_losses, times = bp_neural.optimize(outer_param, max_epochs=max_epochs, max_iters=max_outer_iters, eval_every_n=eval_every_n, validation_data=validation_data, test_data=test_data)

# Show results
print("Number of iterations:", iters)
print("Average iteration time:", np.average(times))