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
max_epochs = 50000
max_outer_iters = 100
max_inner_iters = 20
eval_every_n = 100
lam1 = 1.0
lam2 = 1.0
batch_size = 245
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

mu0 = torch.tensor([[-1],
            [-1],
            [-0.1936],
            [-0.2549],
            [-0.2398],
            [ 0.0568],
            [ 0.0102],
            [ 0.1438],
            [-1],
            [-0.0106],
            [-0.0145],
            [-0.1287],
            [ 0.0147],
            [-0.0500],
            [-0.1694],
            [ 0.2733],
            [ 0.0049],
            [-0.4836],
            [ 0.1613],
            [-0.1334],
            [-0.0213],
            [ 0.3209],
            [-0.2843],
            [ 0.3120],
            [ 0.0494],
            [-0.2016],
            [-0.2083],
            [-0.2541],
            [-0.3407],
            [-0.1016],
            [-0.0185],
            [ 0.1352],
            [ 0.2620],
            [ 0.0968],
            [-0.2102],
            [ 0.1765],
            [ 0.0816],
            [ 0.3882],
            [-0.1479],
            [ 0.0700],
            [-0.1068],
            [ 0.2427],
            [ 0.3761],
            [ 1],
            [-0.1132],
            [-0.0236],
            [-0.5959],
            [ 0.3509],
            [-0.1458],
            [-0.4946],
            [ 0.5354],
            [-0.0014],
            [-0.0801],
            [ 0.1768],
            [ 0.1935],
            [ 0.2672],
            [-0.4116],
            [ 0.0573],
            [-0.1249],
            [ 0.0352],
            [ 0.4159],
            [ 0.1120],
            [-0.0172],
            [-0.1130],
            [ 0.2348],
            [ 0.1994],
            [ 0.1386],
            [ 0.0694],
            [ 0.1005],
            [-0.3106],
            [ 0.1230],
            [-0.2588],
            [ 0.5352],
            [-0.3976],
            [-0.2496],
            [ 0.0984],
            [-0.0594],
            [ 0.3887],
            [ 0.1712],
            [ 0.0705],
            [ 0.1936],
            [-0.4611],
            [-0.2031],
            [-0.3255],
            [-0.0104],
            [-0.1851],
            [ 0.3059],
            [-0.3509],
            [-0.0010],
            [ 0.0482],
            [ 0.5818],
            [ 0.1354],
            [-0.4248],
            [ 0.1905],
            [ 0.5630],
            [-0.1173],
            [ 0.3660],
            [-0.2433],
            [-0.2240],
            [ 0.2800],
            [-0.1255],
            [ 0.1919],
            [-0.3415],
            [-0.2889],
            [ 0.1127],
            [-0.1557],
            [-0.4432],
            [ 0.0934],
            [-0.0541],
            [-0.2349],
            [-0.2892],
            [ 0.2841],
            [ 0.2271],
            [ 0.2782],
            [-0.1030],
            [-0.1840],
            [-0.0122],
            [ 0.3368],
            [ 0.3833],
            [-0.1078],
            [-0.1809],
            [-0.0242],
            [-0.1811],
            [ 0.4078],
            [-0.2301],
            [-0.2248],
            [ 0.2437],
            [ 0.2644],
            [ 0.2950],
            [-0.0688],
            [ 0.0582],
            [-0.0409],
            [ 0.0542],
            [ 0.1625],
            [-0.2331],
            [-0.1059],
            [-0.0806],
            [-0.1526],
            [ 0.5352],
            [-0.0030],
            [ 0.0084],
            [-0.3829],
            [ 0.5614],
            [ 0.4018],
            [-0.6061],
            [-0.2680],
            [-0.1979],
            [-0.2425],
            [ 0.3718],
            [-0.2548],
            [ 0.4871],
            [ 0.3345],
            [ 0.1678],
            [-0.0932],
            [-0.4039],
            [-0.1408],
            [-0.1547],
            [ 0.4132],
            [-0.3619],
            [-0.2702],
            [-0.4176],
            [ 0.5104],
            [-0.4315],
            [ 0.2210],
            [-0.3342],
            [ 0.3023],
            [-0.1431],
            [ 0.0534],
            [ 0.0223],
            [-0.2669],
            [ 0.0121],
            [ 0.0164],
            [-0.3230],
            [ 0.3959],
            [ 0.1442],
            [-0.1549],
            [ 0.0433],
            [-0.2449],
            [-0.0475],
            [-0.0081],
            [ 0.1485],
            [ 0.2961],
            [-0.1593],
            [ 0.0327],
            [-0.4066],
            [-0.1613],
            [ 0.0258],
            [ 0.1702],
            [-0.1437],
            [ 0.0706],
            [ 0.0259],
            [-0.1765],
            [-0.0469],
            [-0.4045],
            [-0.1209],
            [-0.4585],
            [-0.0345],
            [-0.2656],
            [-0.4058],
            [-0.0322],
            [-0.2274],
            [ 0.3759],
            [ 0.0446],
            [ 0.0796],
            [-0.3083],
            [-0.1394],
            [ 0.5732],
            [ 0.2342],
            [ 0.3726],
            [-0.0675],
            [ 0.5527],
            [ 0.0300],
            [ 0.1190],
            [-0.5791],
            [-0.0278],
            [ 0.2919],
            [ 0.1754],
            [-0.1782],
            [-0.2427],
            [-0.1874],
            [-0.2490],
            [ 0.2079],
            [ 0.5285],
            [-0.0434],
            [-0.1490],
            [-0.0400],
            [-0.0071],
            [-0.2727],
            [-0.0553],
            [-0.1582],
            [-0.3657],
            [-0.2485],
            [-0.1678],
            [-0.3608],
            [ 0.5078],
            [-0.0500],
            [-0.3958],
            [-0.1094],
            [ 0.0521],
            [ 0.2535],
            [ 0.0244],
            [ 0.1072],
            [-0.2962],
            [-0.2563],
            [-0.1914]], dtype=torch.float64)

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, "kernel_implicit_diff", data, batch_size=batch_size, reg_param=(lam1, lam2), gamma=gamma)
mu_opt, iters, n_iters, times, inner_loss, outer_loss, h_star = bp_neural.train(mu0=mu0, maxiter=max_epochs, step=1e-4)
print("mu:", mu_opt)