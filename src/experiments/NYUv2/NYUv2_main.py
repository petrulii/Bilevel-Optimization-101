import sys
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from pprint import pprint
from datasets import load_dataset

# Add main project directory path
src_path = "/home/clear/ipetruli/projects/bilevel-optimization/src/"
sys.path.append(src_path)

from model.utils import set_seed
from model.BilevelProblem.BilevelProblem import BilevelProblem
from model.utils import plot_1D_iterations, plot_2D_functions, plot_loss
from experiments.NYUv2.data import nyu_dataloaders

# Setting the random seed.
set_seed()

# Setting the directory to save figures to.
figures_dir = src_path+"figures/"

# Dataloaders for NYUv2 data.
ds = load_dataset(src_path+"experiments/NYUv2/dataset/nyu_depth_v2.py")
pprint(vars(ds))
exit(0)
dataset_train, dataset_val = nyu_dataloaders(datapath=src_path+'experiments/NYUv2/dataset/nyu_depth_v2.py')

dataset = [X_val,y_val,X_train,y_train]
maxiter = 2
step = 0.1
mu_0_value = 1.
mu0 = (torch.full((2,1), mu_0_value))

#########################################################
############ NEURAL IMPLICIT DIFFERENTIATION ############
#########################################################

# Objective functions
MSE = lambda h, X, y: (1/2)*torch.mean(torch.pow((h(X) - y),2)) 
fo = lambda mu, h, X_out, y_out: MSE(h,X_out,torch.reshape(y_out[:,0], (len(y_out),1))) + 0*torch.sum(mu)
fo_h_X = lambda mu, h_X_out, y_out: ((1/2)*torch.mean(torch.pow((h_X_out - torch.reshape(y_out[:,0], (len(y_out),1))),2))) + 0*torch.sum(mu)
fi = lambda mu, h, X_in, y_in: MSE(h,X_in,torch.reshape(y_in[:,0], (len(y_in),1))) + mu[0]*MSE(h,X_in,torch.reshape(y_in[:,1], (len(y_in),1))) + mu[1]*MSE(h,X_in,torch.reshape(y_in[:,2], (len(y_in),1)))
fi_h_X = lambda mu, h_X_in, y_in: ((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,0], (len(y_in),1))),2))) + mu[0]*((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,1], (len(y_in),1))),2))) + mu[1]*((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,2], (len(y_in),1))),2)))

# Manual gradients
#og1 = lambda mu, h, X_out, y_out: torch.full(mu.size(), 0.)
#og2 = lambda mu, h, X_out, y_out: 1/(X_out.size()[0]) * ((h(X_out) - torch.reshape(y_out[:,0], (len(y_out),1))))
#ig22 = lambda mu, h, X_in, y_in: 1/(X_in.size()[0]) * (torch.eye(len(y_in)).to(torch.device("cuda")) + mu[0]*(torch.eye(len(y_in)).to(torch.device("cuda"))) + mu[1]*(torch.eye(len(y_in)).to(torch.device("cuda"))))
#ig12 = lambda mu, h, X_in, y_in: 1/(X_in.size()[0]) * (torch.cat(((h(X_in) - torch.reshape(y_in[:,1], (len(y_in),1))), (h(X_in) - torch.reshape(y_in[:,2], (len(y_in),1)))),1))
#grads = [og1,og2,ig22,ig12]

# Optimize using neural implicit differention
bp_neural = BilevelProblem(outer_objective=fo, inner_objective=fi, method="neural_implicit_diff", data=dataset, fo_h_X=fo_h_X, fi_h_X=fi_h_X)
mu_opt_n, iters_n, n_iters, times, inner_loss, outer_loss, h_star_n = bp_neural.optimize(mu0, maxiter=maxiter, step=step)
plot_loss(figures_dir+"inn_loss_NID", inner_loss, title="Inner loss of neur. im. diff.")
plot_loss(figures_dir+"out_loss_NID", outer_loss, title="Outer loss of neur. im. diff.")

# Show results
print("NEURAL IMPLICIT DIFFERENTIATION")
print("Argmin of the outer objective:", mu_opt_n)
print("Last value of outer loss:", outer_loss[-1])
print("Last value of inner loss:", inner_loss[-1])
print("Average iteration time:", np.average(times))
print("Number of iterations:", n_iters)
print()


############################################################
############## CLASSICAL vs NEURAL COMPARISON ##############
############################################################

h_true = lambda X : X @ coef
h_theta = lambda theta : lambda X : X @ theta
h_star_c = bp_classic.get_h_star(h_theta)

f_c = lambda mu : torch.pow(torch.norm((h_star_c(X_train) - y_train)),2) + (mu.clone().detach())*torch.pow(torch.norm(h_star_c(X_train)),2)
f_n = lambda mu : torch.pow(torch.norm((h_star_n(X_val) - y_val)),2) + mu*torch.pow(torch.norm(h_star_n(X_val)),2)
plot_2D_functions(figures_dir+"pred_function", h_true, h_star_c, h_star_n, points=None, plot_x_lim=[0,mu_0_value], plot_y_lim=[0,1], plot_nb_contours=80, titles=["True Imp. Diff.","Classical Imp. Diff.","Neural Imp. Diff."])

"""
# Testing a*
X_test = torch.from_numpy(np.random.uniform(size=(100, n)).astype('float32'))
y_test = h_true(X_test)
h_theta_grad = lambda X : X
outer_grad2_h = lambda mu, h, X_out, y_out: (h(X_out) - y_out)
a_star_c = bp_classic.a_star(mu_opt_c, outer_grad2_h, h_star_c, h_theta_grad)
a_star_n = bp_neural.a_star
print(torch.norm(a_star_c(X_test, y_test) - a_star_n(X_test)))
"""