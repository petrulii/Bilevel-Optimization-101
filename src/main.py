import torch
import random
import numpy as np
from utils import sample
from torch.utils.data import random_split
from BilevelProblem.BilevelProblem import BilevelProblem
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Initialize dimesnions
n, m, m_v, m_t, batch = 2, 1000, 300, 700, 64
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

#fig = plt.figure(figsize=(4,8))
#ax = fig.add_subplot(projection='3d')
#ax.scatter(X_train[:,0], X_train[:,1], y_train, marker='.')
#plt.show()


############ NEURAL IMPLICIT DIFFERENTIATION ############

fo = lambda mu, h, w, v: None
fi = lambda mu, h, w, v: None
# Gradient wrt mu of f
og1 = lambda mu, h, X_out, y_out: torch.tensor([[0]])#torch.mean(torch.pow(h(sample(X_out,batch)),2.0))
# Gradient wrt h of f
og2 = lambda mu, h, X_out, y_out: (h(X_out) - y_out)# + torch.mul(h(X_out),torch.mul(mu,2.0))
# Hessian wrt h of g
ig22 = lambda mu, h, X_in, y_in: torch.eye(len(y_in))# + torch.mul(torch.eye(len(y_in)),torch.mul(mu,2.0))
# Gradient wrt mu of gradient wrt h of g
ig12 = lambda mu, h, X_in, y_in: h(X_in) * 2

# Optimize using neural implicit differention
bp = BilevelProblem(outer_objective=fo, inner_objective=fi, method="neural_implicit_diff", outer_grad1=og1, outer_grad2=og2, inner_grad22=ig22, inner_grad12=ig12, X_outer=X_val, y_outer=y_val, X_inner=X_train, y_inner=y_train)
mu0 = torch.full((1,1), 0.2)
mu_opt, iters, n_iters = bp.optimize(mu0, maxiter=1000, step=0.1)
# Show results
print("neural_implicit_diff Argmin of the outer objective:", mu_opt)
print("neural_implicit_diff Number of iterations:", n_iters)


############ CLASSICAL IMPLICIT DIFFERENTIATION ############

def find_h_star(X_in, y_in, mu_old):
    """
    Find a closed form solution of h_star for a fixed mu.
      param mu_old: old value of the outer variable
    """
    mu = (mu_old.cpu().detach().numpy())[0,0]
    clf = Ridge(alpha=mu, solver='cholesky')
    clf.fit(X_in.cpu().detach().numpy(), y_in.cpu().detach().numpy())
    return torch.from_numpy(clf.coef_.T)

# Gradient wrt mu of f
og1 = lambda mu, h, X_out, y_out: torch.tensor([[0]])
# Gradient wrt h of f
og2 = lambda mu, h, X_out, y_out: torch.transpose(X_out,0,1) @ (X_out @ h - y_out) + torch.mul(h,torch.mul(mu,2.0))
# Hessian wrt h of g
ig22 = lambda mu, h, X_in, y_in: torch.transpose(X_in,0,1) @ X_in + torch.mul((torch.transpose(X_in,0,1) @ X_in),torch.mul(mu,2.0))
# Gradient wrt mu of gradient wrt h of g
ig12 = lambda mu, h, X_in, y_in: torch.mul(h,2.0)

# Optimize using neural implicit differention
bp = BilevelProblem(outer_objective=fo, inner_objective=fi, method="implicit_diff", outer_grad1=og1, outer_grad2=og2, inner_grad22=ig22, inner_grad12=ig12, find_h_star=find_h_star, X_outer=X_val, y_outer=y_val, X_inner=X_train, y_inner=y_train)
mu0 = torch.full((1,1), 0.2)
mu_opt, iters, n_iters = bp.optimize(mu0, maxiter=10000, step=0.1)
# Show results
print("implicit_diff Argmin of the outer objective:", mu_opt)
print("implicit_diff Number of iterations:", n_iters)
print("True coeficients:", coef)