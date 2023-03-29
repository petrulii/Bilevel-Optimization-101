import torch
from torch import nn, tensor
from torch.autograd import grad
from torch.autograd.functional import hessian


# Objective functions
MSE = lambda h, X, y: (1/2)*torch.mean(torch.pow((h(X) - y),2)) 
fo = lambda mu, h, X_out, y_out: MSE(h,X_out,torch.reshape(y_out[:,0], (len(y_out),1))) + 0*torch.sum(mu)
fo_h_X = lambda mu, h_X_out, y_out: ((1/2)*torch.mean(torch.pow((h_X_out - torch.reshape(y_out[:,0], (len(y_out),1))),2))) + 0*torch.sum(mu)
fi = lambda mu, h, X_in, y_in: MSE(h,X_in,y_in[:,0]) + mu[0]*MSE(h,X_in,torch.reshape(y_in[:,1], (len(y_in),1))) + mu[1]*MSE(h,X_in,torch.reshape(y_in[:,2], (len(y_in),1)))
fi_h_X = lambda mu, h_X_in, y_in: ((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,0], (len(y_in),1))),2))) + mu[0]*((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,1], (len(y_in),1))),2))) + mu[1]*((1/2)*torch.mean(torch.pow((h_X_in - torch.reshape(y_in[:,2], (len(y_in),1))),2)))

# Manual gradients
og1 = lambda mu, h, X_out, y_out: torch.full((mu.size()[0],1), 0.)
og2 = lambda mu, h, X_out, y_out: 1/(X_out.size()[0]) * ((h(X_out) - torch.reshape(y_out[:,0], (len(y_out),1))))
ig22 = lambda mu, h, X_in, y_in: 1/(X_in.size()[0]) * (torch.eye(len(y_in)) + mu[0]*(torch.eye(len(y_in))) + mu[1]*(torch.eye(len(y_in))))
ig12 = lambda mu, h, X_in, y_in: 1/(X_in.size()[0]) * (torch.cat(((h(X_in) - torch.reshape(y_in[:,1], (len(y_in),1))), (h(X_in) - torch.reshape(y_in[:,2], (len(y_in),1)))),1))

# Data
X = torch.randn(10,2)
y = torch.randn(10,3)
theta = torch.randn(2,1)
mu = torch.randn(2,1)
h = lambda X : X @ theta

# Autograd test
#h_X = h(X)
#h_X.detach()
# Outer grad. 2 aut.
#h_X.requires_grad = True
#h_X.retain_grad()
#fo_h_X(mu, h_X, y).backward()
#grad1 = h_X.grad
# Outer grad. 1 aut.
mu.requires_grad = True
mu.retain_grad()
fo(mu, h, X, y).backward()
#fo_h_X(mu, h_X, y).backward()
grad1 = torch.reshape(mu.grad, mu.size())
#h_X.requires_grad = True
#h_X.retain_grad()
#f = lambda arg1, arg2: fi_h_X(arg1, arg2, y)
# Inner grad. 12 aut.
#grad1 = hessian(f, (mu, h_X))[0][1]
#grad1 = (torch.reshape(grad1, (mu.size()[0],h_X.size()[0]))).T
# Inner grad. 22 aut.
#grad1 = hessian(f, (mu, h_X))[1][1]
#grad1 = torch.reshape(grad1, (h_X.size()[0],h_X.size()[0]))
print("Autograd:")
print(grad1)

# Manual test
# Outer grad. 2 man.
#grad2 = og2(mu, h, X, y)
# Outer grad. 1 man.
grad2 = og1(mu, h, X, y)
# Inner grad. 12 man.
#grad2 = ig12(mu, h, X, y)
# Inner grad. 22 man.
#grad2 = ig22(mu, h, X, y)
print("Manual grad:")
print(grad2)
