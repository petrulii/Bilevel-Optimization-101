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
from torch.nn import functional as func
from torch.autograd.functional import hessian
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model.InnerSolution.InnerSolution import InnerSolution
from model.BilevelProblem.BilevelProblem import BilevelProblem
from model.utils import set_seed, plot_loss, tensor_to_state_dict

from experiments.NYuv2.experiments.nyuv2.data import nyu_dataloaders
from experiments.NYuv2.experiments.nyuv2.metrics import compute_iou, compute_miou
from experiments.NYuv2.experiments.nyuv2.model import SegNetSplit
from experiments.NYuv2.auxilearn.hypernet import MonoHyperNet, MonoLinearHyperNet, MonoNonlinearHyperNet, MonoNoFCCNNHyperNet
from experiments.NYuv2.auxilearn.optim import MetaOptimizer

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

# Setting hyper-parameters
batch_size = 16
max_epochs = 1

# Dataloaders for inner and outer data
inner_dataloader, outer_dataloader, test_dataloader = nyu_dataloaders(
    datapath="/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/NYuv2/data",
    validation_indices="/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/NYuv2/hpo_validation_indices.json",
    aux_set=False,
    batch_size=batch_size,
    val_batch_size=batch_size
)

#########################################################
############ NEURAL IMPLICIT DIFFERENTIATION ############
#########################################################

# Helper function for objectives
def calc_loss(seg_pred, seg, depth_pred, depth, pred_normal, normal):
    """
    Per-pixel loss, i.e., loss image
    """
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(depth, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(depth.device)
    # semantic loss: depth-wise cross entropy
    seg_loss = func.nll_loss(seg_pred, seg, ignore_index=-1, reduction='none')
    # depth loss: l1 norm
    depth_loss = torch.sum(torch.abs(depth_pred - depth) * binary_mask, dim=1)
    # normal loss: dot product
    normal_loss = 1 - torch.sum((pred_normal * normal) * binary_mask, dim=1)
    return (seg_loss, depth_loss, normal_loss)

# Inner model
# Neural network to approximate the function h*
inner_model = SegNetSplit(logsigma=False).to(device)
# Optimizer that improves the approximation of h*
inner_optimizer = torch.optim.SGD(inner_model.parameters(), lr=0.001)

# Inner dual model
# Neural network to approximate the function a*
inner_dual_model = SegNetSplit(logsigma=False).to(device)
# Optimizer that improves the approximation of a*
inner_dual_optimizer = torch.optim.SGD(inner_model.parameters(), lr=0.001)

# Outer model
# The outer neural network parametrized by the outer variable mu
outer_model = MonoNoFCCNNHyperNet(main_task=0, reduction='mean').to(device)

# Objective functions
def fo(mu, h_X_out, y_out, accuracy=False):
    seg, depth, normal = y_out
    seg_pred, depth_pred, pred_normal = h_X_out
    seg = seg.type(torch.LongTensor).to(device)
    # Display accuracy
    if accuracy:
        _, predicted = torch.max(seg_pred.data, 1)
        value = seg.detach().flatten().cpu()
        pred = predicted.detach().flatten().cpu()
        print("Accuracy:", torch.sum(value == pred)/value.size()[0])
    seg_loss, depth_loss, normal_loss = calc_loss(seg_pred, seg, depth_pred, depth, pred_normal, normal)
    loss_image = torch.stack((seg_loss, depth_loss, normal_loss), dim=1)
    total_loss = loss_image.mean(dim=(0, 2, 3))
    loss = total_loss[0].mean(0)
    return loss

def fi(mu, h_X_in, y_in):
    seg, depth, normal = y_in
    seg_pred, depth_pred, pred_normal = h_X_in
    seg_loss, depth_loss, normal_loss = calc_loss(seg_pred, seg, depth_pred, depth, pred_normal, normal)
    loss_image = torch.stack((seg_loss, depth_loss, normal_loss), dim=1)
    loss = functional_call(outer_model, tensor_to_state_dict(outer_model, mu, device), loss_image)
    return loss

# Initialize mu
summary(outer_model, input_size=(3, 288, 384), device=device)
mu0 = torch.randn((1408, 1))
# Optimizer that improves the outer variable mu
outer_optimizer = torch.optim.SGD([mu0], lr=0.1)

# Gather all models
inner_models = (inner_model, inner_optimizer, inner_dual_model, inner_dual_optimizer)
outer_models = (outer_model, outer_optimizer)

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, outer_dataloader, inner_dataloader, outer_models, inner_models, device, batch_size=batch_size)
nb_iters, iters, losses, times = bp_neural.optimize(mu0, max_epochs=max_epochs)

# Show results
print("Number of iterations:", nb_iters)
#print("Outer variable values:", iters)
print("Outer loss values:", losses)
print("Average iteration time:", np.average(times))
print()

plot_loss(figures_dir+"out_loss_NID", losses, title="Outer loss of neur. im. diff.")

test_losses = []
for mu in iters:
    nb_iters, losses = bp_neural.evaluate(test_dataloader, mu)
    print("Losses for mu iteration:", losses)
    test_losses.append(losses[-1])
print("Test loss values:", test_losses)
plot_loss(figures_dir+"test_loss_NID", test_losses, title="Test loss of neur. im. diff.")

#Use gpu 23/28 rather than 21, more memory