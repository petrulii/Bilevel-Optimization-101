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
sys.path.insert(1, '/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/birds')
from Bird_dataset import Bird_dataset_naive
from torchvision import transforms

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model.InnerSolution.InnerSolution import InnerSolution
from model.BilevelProblem.BilevelProblem import BilevelProblem
from model.NeuralNetworks.BirdsResNet import ResNet
from model.NeuralNetworks.NeuralNetworkOuterModel import NeuralNetworkOuterModel
from model.utils import *

# Setting the random seed.
set_seed()

# Paths to data
train_image_file = '/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/birds/CUB_200_2011/preprocess_data/full_train_set.json'
valid_image_file = '/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/birds/CUB_200_2011/preprocess_data/valid_set.json'
test_image_file = '/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/birds/CUB_200_2011/preprocess_data/test_set.json'
label_file = '/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/birds/CUB_200_2011/preprocess_data/image_dictionary.json'
image_root = '/home/clear/ipetruli/projects/bilevel-optimization/src/experiments/birds/CUB_200_2011/CUB_200_2011/images'

# Data pre-processing
train_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

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
nb_aux_tasks = 312
batch_size = 64
max_epochs = 100
max_outer_iters = 100
max_inner_iters = 15

# Dataloaders for inner and outer data
train_dataset = Bird_dataset_naive(train_image_file,label_file,image_root,transform=train_transform, finegrain=True)
inner_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = Bird_dataset_naive(valid_image_file,label_file,image_root,transform=test_transform, finegrain=True)
outer_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_dataset = Bird_dataset_naive(test_image_file,label_file,image_root,transform=test_transform, finegrain=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Inner model
# Neural network to approximate the function h*
inner_model = ResNet().to(device)
# Show summary
#summary(inner_model, input_size=(3, 224, 224))
#for name, param in inner_model.named_parameters():
#    print(name)
# Optimizer that improves the approximation of h*
inner_optimizer = torch.optim.SGD(inner_model.parameters(), lr=6e-1)
inner_scheduler = lr_scheduler.LinearLR(inner_optimizer, start_factor=1.0, end_factor=0.001, total_iters=600)

# Inner dual model
# Neural network to approximate the function a*
inner_dual_model = ResNet().to(device)
# Optimizer that improves the approximation of a*
inner_dual_optimizer = torch.optim.SGD(inner_dual_model.parameters(), lr=6e-1)
inner_dual_scheduler = lr_scheduler.LinearLR(inner_dual_optimizer, start_factor=1.0, end_factor=0.001, total_iters=600)

# Outer model
# The outer neural network parametrized by the outer variable mu
outer_model = NeuralNetworkOuterModel(layer_sizes=[nb_aux_tasks,1]).to(device)
# Show summary
#summary(outer_model, input_size=(2,1))
# Initialize mu
mu0 = torch.ones((nb_aux_tasks,1)).to(device)
# Optimizer that improves the outer variable mu
outer_optimizer = torch.optim.SGD([mu0], lr=1e+1)#lr=1e-3, weight_decay=5e-3, momentum=.9)
outer_scheduler = lr_scheduler.LinearLR(outer_optimizer, start_factor=1.0, end_factor=0.01, total_iters=600)

# Gather all models
inner_models = (inner_model, inner_optimizer, inner_scheduler, inner_dual_model, inner_dual_optimizer, inner_dual_scheduler)
outer_models = (outer_model, outer_optimizer, outer_scheduler)

# Loss helper functions
binary_loss = nn.BCELoss(reduction='mean')
class_loss = nn.CrossEntropyLoss(reduction='mean')
relu = torch.nn.ReLU()

# Outer objective function
def fo(mu, inner_value, labels):
    (main_pred, aux_pred) = inner_value
    (main_label, aux_label) = labels
    loss = class_loss(main_pred, main_label.to(device, dtype=torch.long))
    return loss

# Inner objective function
def fi(mu, h_X_in, y_in):
    # Keep mu positive
    mu = relu(mu)
    (main_pred, aux_pred) = h_X_in
    aux_pred = torch.sigmoid(aux_pred)
    (main_label, aux_label) = y_in
    aux_label = aux_label.T
    loss = class_loss(main_pred, main_label.to(device, dtype=torch.long))
    aux_loss_vector = torch.zeros((nb_aux_tasks,1)).to(device)
    for task in range(nb_aux_tasks):
        aux_loss_vector[task] = binary_loss(aux_pred[:,task], aux_label[:,task].to(device, dtype=torch.float))
    # Essentially dot product with mu
    loss += (mu.T @ aux_loss_vector)[0,0]#functional_call(outer_model, tensor_to_state_dict(outer_model, mu, device), aux_loss_vector.T)[0,0]
    return loss

# Optimize using neural implicit differention
bp_neural = BilevelProblem(fo, fi, outer_dataloader, inner_dataloader, outer_models, inner_models, device, batch_size=batch_size, max_inner_iters=max_inner_iters)
iters, outer_losses, inner_losses, test_losses, times = bp_neural.optimize(mu0, max_epochs=max_epochs, test_dataloader=test_dataloader, max_iters=max_outer_iters)

# Show results
print("Number of iterations:", iters)
print("Average iteration time:", np.average(times))

plot_loss(figures_dir+"loss_birds_inner_batch64_100epochs_15inner", train_loss=inner_losses, title="Inner loss of neur. im. diff.")
plot_loss(figures_dir+"loss_birds_outer_batch64_100epochs_15inner", val_loss=outer_losses, title="Outer oss of neur. im. diff.")
plot_loss(figures_dir+"accuracy_birds_test_batch64_100epochs_15inner", test_loss=test_losses, title="Test acc. of neur. im. diff.")

#Use gpu 23 or 28 rather than 21, more memory