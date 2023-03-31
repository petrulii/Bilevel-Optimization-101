import sys
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src/')

from model.utils import set_seed
from model.BilevelProblem.BilevelProblem import BilevelProblem
from model.utils import plot_1D_iterations, plot_2D_functions, plot_loss

# Setting the random seed.
set_seed()

# Setting the directory to safe figures to.
figures_dir = "figures/"

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
############ CLASSICAL IMPLICIT DIFFERENTIATION ############
############################################################

# A function to find optimal or closed form solution of h*
def find_theta_star(X, y, mu):
    """
    Find a closed form solution of theta for a fixed mu.
    """
    term1 = X.T @ X + mu[0]*X.T @ X + mu[1]*X.T @ X
    term2 = X.T @ y[:,0] + mu[0]*X.T @ y[:,1] + mu[1]*X.T @ y[:,2]
    theta = torch.linalg.solve(term1, term2)
    return torch.reshape(theta, (theta.size()[0], 1))#?

# Objective functions
MSE = lambda th, X, y: (1/2)*torch.mean(torch.pow(((X @ th) - y),2))
fo = lambda mu, theta, X_out, y_out: MSE(theta,X_out,torch.reshape(y_out[:,0], (y_out[:,0].size()[0],1))) + 0*torch.sum(mu)
fi = lambda mu, theta, X_in, y_in: MSE(theta,X_in,torch.reshape(y_in[:,0], (y_in[:,0].size()[0],1))) + mu[0]*MSE(theta,X_in,torch.reshape(y_in[:,1], (y_in[:,1].size()[0],1))) + mu[1]*MSE(theta,X_in,torch.reshape(y_in[:,2], (y_in[:,2].size()[0],1)))

# Gradients
#og1 = lambda mu, theta, X_out, y_out: (torch.full((n,1), 0.))
#og2 = lambda mu, theta, X_out, y_out: 1/(X_out.size()[0]) * (X_out.T @ (X_out @ theta - torch.reshape(y_out[:,0], (y_out[:,0].size()[0],1))))
#ig22 = lambda mu, theta, X_in, y_in: 1/(X_in.size()[0]) * (X_in.T @ X_in + mu[0]*X_in.T @ X_in + mu[1]*X_in.T @ X_in)
#ig12 = lambda mu, theta, X_in, y_in: 1/(X_in.size()[0]) * (torch.cat((X_in.T @ (X_in @ theta - torch.reshape(y_in[:,1], (y_in[:,1].size()[0],1))), X_in.T @ (X_in @ theta - torch.reshape(y_in[:,2], (y_in[:,2].size()[0],1)))),1))
#grads = [og1,og2,ig22,ig12]

# Optimize using classical implicit differention
bp_classic = BilevelProblem(outer_objective=fo, inner_objective=fi, method="implicit_diff", data=dataset, find_theta_star=find_theta_star)
mu_opt_c, iters_c, n_iters, times, inner_loss, outer_loss, theta = bp_classic.optimize(mu0, maxiter=maxiter, step=step)
plot_loss(figures_dir+"inn_loss_CID", inner_loss, title="Inner loss of clas. im. diff.")
plot_loss(figures_dir+"out_loss_CID", outer_loss, title="Outer loss of clas. im. diff.")

# Show results
print("CLASSICAL IMPLICIT DIFFERENTIATION")
print("Argmin of the outer objective:", mu_opt_c)
print("Average iteration time:", np.average(times))
print("Number of iterations:", n_iters)
print("True coefficients theta:", coef)
print("Fitted coefficients theta:", theta)
print()


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