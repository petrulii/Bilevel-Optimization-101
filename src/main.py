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

dataset = [X_val,y_val,X_train,y_train]
maxiter = 100
step = 0.1
mu_0_value = 1.
mu0 = torch.full((1,1), mu_0_value)

############################################################
############ CLASSICAL IMPLICIT DIFFERENTIATION ############
############################################################

# A function to find optimal or closed form solution of h*
def find_theta_star(X, y, mu):
    """
    Find a closed form solution of theta for a fixed mu.
    """
    return torch.linalg.inv((1+2*mu) * X.T @ X) @ X.T @ y

# Objective functions
fo = lambda mu, theta, X, y: None
fi = lambda mu, theta, X, y: None

# Gradients
og1 = lambda mu, theta, X_out, y_out: torch.tensor([[0]])
og2 = lambda mu, theta, X_out, y_out: X_out.T @ (X_out @ theta - y_out)
ig22 = lambda mu, theta, X_in, y_in: X_in.T @ X_in + 2 * mu * X_in.T @ X_in
ig12 = lambda mu, theta, X_in, y_in: 2 * X_in.T @ X_in @ theta

# Optimize using classical implicit differention
bp_classic = BilevelProblem(outer_objective=fo, inner_objective=fi, method="implicit_diff", data=dataset, gradients=[og1,og2,ig22,ig12], find_theta_star=find_theta_star)
mu_opt_c, iters_c, n_iters, times, theta = bp_classic.optimize(mu0, maxiter=maxiter, step=step)

# Show results
print("CLASSICAL IMPLICIT DIFFERENTIATION")
print("Argmin of the outer objective:", mu_opt_c)
print("Average iteration time:", np.average(times))
print("Number of iterations:", n_iters)
print("True coefficients theta:", coef)
print("Fitted coefficients theta:", theta)


############################################################
############ NEURAL IMPLICIT DIFFERENTIATION ############
############################################################

# Objective functions
fo = lambda mu, h, w, v: None
fi = lambda mu, h, w, v: None

# Gradients
og1 = lambda mu, h, X_out, y_out: torch.tensor([[0]])
og2 = lambda mu, h, X_out, y_out: (h(X_out) - y_out)
ig22 = lambda mu, h, X_in, y_in: torch.eye(len(y_in)) * (1+2*mu)
ig12 = lambda mu, h, X_in, y_in: 2*h(X_in)

# Optimize using neural implicit differention
bp_neural = BilevelProblem(outer_objective=fo, inner_objective=fi, method="neural_implicit_diff", data=dataset, gradients=[og1,og2,ig22,ig12])
mu_opt_n, iters_n, n_iters, times, h_star_n = bp_neural.optimize(mu0, maxiter=maxiter, step=step)

# Show results
print("NEURAL IMPLICIT DIFFERENTIATION")
print("Argmin of the outer objective:", mu_opt_n)
print("Average iteration time:", np.average(times))
print("Number of iterations:", n_iters)
print()


############################################################
############## CLASSICAL vs NEURAL COMPARISON ##############
############################################################

h_true = lambda X : X @ coef
h_theta = lambda theta : lambda X : X @ theta
h_star_c = bp_classic.h_star(h_theta)

f_c = lambda mu : torch.pow(torch.norm((h_star_c(X_train) - y_train)),2) + (mu.clone().detach())*torch.pow(torch.norm(h_star_c(X_train)),2)
f_n = lambda mu : torch.pow(torch.norm((h_star_n(X_val) - y_val)),2) + mu*torch.pow(torch.norm(h_star_n(X_val)),2)
#plot_2D_functions(h_true, h_star_c, h_star_n, points=None, plot_x_lim=[0,mu_0_value], plot_y_lim=[0,1], plot_nb_contours=80, titles=["True Imp. Diff.","Classical Imp. Diff.","Neural Imp. Diff."])
#plot_1D_iterations(iters_c, iters_n, f_c, f_n, plot_x_lim=[0,mu_0_value], titles=["Classical Imp. Diff.","Neural Imp. Diff."])


# Testing a*
X_test = torch.from_numpy(np.random.uniform(size=(100, n)).astype('float32'))
y_test = h_true(X_test)
h_theta_grad = lambda X : X
outer_grad2_h = lambda mu, h, X_out, y_out: (h(X_out) - y_out)
a_star_c = bp_classic.a_star(mu_opt_c, outer_grad2_h, h_star_c, h_theta_grad)
a_star_n = bp_neural.a_star
print(torch.norm(a_star_c(X_test, y_test) - a_star_n(X_test)))