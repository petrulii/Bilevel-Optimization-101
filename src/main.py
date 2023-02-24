import torch
import numpy as np
from BilevelProblem.BilevelProblem import BilevelProblem
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

np.random.seed(0)

n = 2
m_v = 300
m_t = 700
m= m_v + m_t
#X, y, coef = make_regression(n_samples=m, n_features=n, n_informative=n, n_targets=1, noise=0, coef=True, random_state=10)
#X, y, coef = X*1/10, y*1/10, coef*1/10
coef = np.random.uniform(size=(n,1))
X = np.random.uniform(size=(m, n))
h_true = lambda X: X @ coef#(1+X) @ coef.T
y = h_true(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
print("True coeficients:", coef)

#fig = plt.figure(figsize=(4,8))
#ax = fig.add_subplot(projection='3d')
#ax.scatter(X_train[:,0], X_train[:,1], y_train, marker='.')
#plt.show()

fo = lambda x, h, w, v: (1/m_v)*np.sum(np.array([((1/2)*np.sum(h(w)-v)**2+x*np.sum(h)**2) for (w, v) in (X_val, y_val)]))
fi = lambda x, h, w, v: (1/m_t)*np.sum(np.array([((1/2)*np.sum(h(w)-v)**2+x*np.sum(h)**2) for (w, v) in (X_train, y_train)]))
# Gradient wrt x of f
og1 = lambda x, h, X_out, y_out: np.mean(np.power((h(X_out[np.random.randint(X_out.shape[0], size=64), :])).detach().numpy(),2))
# Gradient wrt h of f
og2 = lambda x, h, X_out, y_out: (h(X_out) - y_out) + torch.mul(h(X_out),torch.mul(x,2.0))
# Hessian wrt h of g
ig22 = lambda x, h, X_in, y_in: torch.eye(len(y_in)) + torch.mul(torch.eye(len(y_in)),torch.mul(x,2.0))
# Gradient wrt x of gradient wrt h of g
ig12 = lambda x, h, X_in, y_in: (h(X_in)).detach().numpy() * 2

bp = BilevelProblem(outer_objective=fo, inner_objective=fi, method="neural_implicit_diff", outer_grad1=og1, outer_grad2=og2, inner_grad22=ig22, inner_grad12=ig12, X_outer=X_val, y_outer=y_val, X_inner=X_train, y_inner=y_train)
# Optimize using classical implicit differention.
x0 = np.array([[1.]])
x_opt, iters, n_iters = bp.optimize(x0, maxiter=10, step=0.1)
print("Argmin of the outer objective:", x_opt)
print("Number of iterations:", n_iters)