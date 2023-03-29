import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from BilevelProblem.BilevelProblem import BilevelProblem
from utils import plot_1D_iterations, plot_2D_functions, plot_loss

# Setting the random seed.
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

# Setting the directory to safe figures to.
figures_dir = "/home/clear/ipetruli/projects/bilevel-optimization/src/figures/"

# Initialize dimesnions
n, m, m_out, m_in, batch = 2, 10000, 3000, 7000, 32
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

dataset = [X_val,y_val,X_train,y_train]
maxiter = 2
step = 0.1
mu_0_value = 1.
mu0 = (torch.full((2,1), mu_0_value))

############################################################
####### NEURAL IMPLICIT DIFFERENTIATION AUTOGRAD ###########
############################################################

# Objective functions
MSE = lambda h, X, y: (1/2)*torch.mean(torch.pow((h(X) - y),2)) 
fo = lambda mu, h, X_out, y_out: MSE(h,X_out,torch.reshape(y_out[:,0], (len(y_out),1))) + 0*torch.sum(mu)
fo_h_X = lambda mu, h_X_out, y_out: ((1/2)*torch.mean(torch.pow((h_X_out - torch.reshape(y_out[:,0], (len(y_out),1))),2))) + 0*torch.sum(mu)
fi = lambda mu, h, X_in, y_in: MSE(h,X_in,y_in[:,0]) + mu[0]*MSE(h,X_in,torch.reshape(y_in[:,1], (len(y_in),1))) + mu[1]*MSE(h,X_in,torch.reshape(y_in[:,2], (len(y_in),1)))
fi_h_X = lambda mu, h_X_in, y_in: ((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,0], (len(y_in),1))),2))) + mu[0]*((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,1], (len(y_in),1))),2))) + mu[1]*((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,2], (len(y_in),1))),2)))

# Optimize using neural implicit differention
bp_neural_aut = BilevelProblem(outer_objective=fo, inner_objective=fi, method="neural_implicit_diff", data=dataset, fo_h_X=fo_h_X, fi_h_X=fi_h_X)
mu_opt_n_aut, iters_n_aut, n_iters_aut, times_aut, inner_loss_aut, outer_loss_aut, h_star_n_aut = bp_neural_aut.optimize(mu0, maxiter=maxiter, step=step)
plot_loss(figures_dir+"inn_loss_NID_aut", inner_loss_aut, title="Inner loss of neur. im. diff. + aut")
plot_loss(figures_dir+"out_loss_NID_aut", outer_loss_aut, title="Outer loss of neur. im. diff. + aut")

# Show results
print("NEURAL IMPLICIT DIFFERENTIATION AUTOGRAD")
print("Argmin of the outer objective:", mu_opt_n_aut)
print("Last value of outer loss:", outer_loss_aut[-1])
print("Last value of inner loss:", inner_loss_aut[-1])
print("Average iteration time:", np.average(times_aut))
print("Number of iterations:", n_iters_aut)
print()
