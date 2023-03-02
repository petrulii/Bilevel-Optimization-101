import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from utils import plot_1D_iterations, plot_2D_functions
from BilevelProblem.BilevelProblem import BilevelProblem

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Initialize dimesnions
n, m, m_v, m_t, batch = 2, 10000, 3000, 7000, 64
# The coefficient tensor of size (n,1) filled with values uniformally sampled from the range (0,1)
coef = np.random.uniform(size=(n,1)).astype('float32')
# The data tensor of size (m,n) filled with values uniformally sampled from the range (0,1)
X = np.random.uniform(size=(m, n)).astype('float32')
# True h_star
h_true = lambda X: X @ coef
y = h_true(X)
# Split X into 2 tensors with sizes [m_t, m_v] along dimension 0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
# Convert everything to PyTorch tensors
X_train, X_val, y_train, y_val, coef = torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(coef)
print("True coeficients:", coef)
print("X training data:", X_train[1:5])
print("y training labels:", y_train[1:5])
print()


############################################################
############ NEURAL IMPLICIT DIFFERENTIATION ############
############################################################

fo = lambda mu, h, w, v: None
fi = lambda mu, h, w, v: None
# Gradient wrt mu of f
og1 = lambda mu, h, X_out, y_out: torch.tensor([[0]])
# Gradient wrt h of f
og2 = lambda mu, h, X_out, y_out: (h(X_out) - y_out)
# Hessian wrt h of g
ig22 = lambda mu, h, X_in, y_in: torch.eye(len(y_in)) * (1+2*mu)
# Gradient wrt mu of gradient wrt h of g
ig12 = lambda mu, h, X_in, y_in: 2*h(X_in)

# Optimize using neural implicit differention
bp = BilevelProblem(outer_objective=fo, inner_objective=fi, method="neural_implicit_diff", outer_grad1=og1, outer_grad2=og2, inner_grad22=ig22, inner_grad12=ig12, X_outer=X_val, y_outer=y_val, X_inner=X_train, y_inner=y_train)#, coef=coef)
mu0 = torch.full((1,1), 1.)
mu_opt, iters_n, n_iters, times, h_star_n = bp.optimize(mu0, maxiter=100, step=0.1)
# Show results
print("NEURAL IMPLICIT DIFFERENTIATION")
print("Argmin of the outer objective:", mu_opt)
print("Average iteration time:", np.average(times))
print("Number of iterations:", n_iters)
print()
############################################################
########## FRENKENSTEIN IMPLICIT DIFFERENTIATION ###########
############################################################
"""
def find_h_star(X, y, mu):
    return torch.linalg.inv((1+2*mu) * X.T @ X) @ X.T @ y

# Optimize using neural implicit differention
bp = BilevelProblem(outer_objective=fo, inner_objective=fi, method="frankenstein", outer_grad1=og1, outer_grad2=og2, inner_grad22=ig22, inner_grad12=ig12, find_h_star=find_h_star, X_outer=X_val, y_outer=y_val, X_inner=X_train, y_inner=y_train)#, coef=coef)
mu0 = torch.full((1,1), 1.)
mu_opt, iters_n, n_iters, times, h_star_n = bp.optimize(mu0, maxiter=100, step=0.1)
print("FRENKENSTEIN IMPLICIT DIFFERENTIATION")
print("Argmin of the outer objective:", mu_opt)
print("Average iteration time:", np.average(times))
print("Number of iterations:", n_iters)
print()"""


############################################################
############ CLASSICAL IMPLICIT DIFFERENTIATION ############
############################################################

def find_h_star(X, y, mu):
    """
    Find a closed form solution of theta for a fixed mu.
    """
    return torch.linalg.inv((1+2*mu) * X.T @ X) @ X.T @ y

# Gradient wrt mu of f
og1 = lambda mu, h, X_out, y_out: torch.tensor([[0]])
# Gradient wrt h of f
og2 = lambda mu, h, X_out, y_out: X_out.T @ (X_out @ h - y_out)
# Hessian wrt h of g
ig22 = lambda mu, h, X_in, y_in: X_in.T @ X_in + 2 * mu * X_in.T @ X_in
# Gradient wrt mu of gradient wrt h of g
ig12 = lambda mu, h, X_in, y_in: 2 * X_in.T @ X_in @ h

# Optimize using neural implicit differention
bp = BilevelProblem(outer_objective=fo, inner_objective=fi, method="implicit_diff", outer_grad1=og1, outer_grad2=og2, inner_grad22=ig22, inner_grad12=ig12, find_h_star=find_h_star, X_outer=X_val, y_outer=y_val, X_inner=X_train, y_inner=y_train)#, coef=coef)
mu0 = torch.full((1,1), 1.)
mu_opt, iters_c, n_iters, times, theta = bp.optimize(mu0, maxiter=100, step=0.1)
print("CLASSICAL IMPLICIT DIFFERENTIATION")
print("Argmin of the outer objective:", mu_opt)
print("Average iteration time:", np.average(times))
print("Number of iterations:", n_iters)
print()
print("True coefficients:", coef)
print("Fitted coefficients:", theta)

h_true = lambda x : x @ coef
h_star_c = lambda x : x @ theta
#f_c = lambda mu : torch.pow(torch.norm((h_star_c(X_train) - y_train)),2) + mu*torch.pow(torch.norm(h_star_c(X_train)),2)
def f_c(mu):
    n1 = torch.norm(h_star_c(X_train) - y_train)
    t1 = torch.pow(n1,2)
    n2 = torch.norm(h_star_c(X_train))
    t2 = torch.tensor(mu)*torch.pow(n2,2)
    return t1+t2
f_n = lambda mu : torch.pow(torch.norm((h_star_n(X_val) - y_val)),2) + mu*torch.pow(torch.norm(h_star_n(X_val)),2)
plot_2D_functions(h_true, h_star_c, h_star_n, points=None, plot_x_lim=[0,1], plot_y_lim=[0,1], plot_nb_contours=80)
plot_1D_iterations(iters_c, iters_n, f_c, f_n, plot_x_lim=[0,1])