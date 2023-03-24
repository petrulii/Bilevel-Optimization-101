import torch
from torch.autograd import grad
from torch import nn, tensor

MSE = lambda th, X, y: (1/2)*torch.mean(torch.pow(((X @ th) - y),2))

class fo(nn.Module):
    def __init__(self):
        super(fo, self).__init__()

    def forward(self, mu):
        #res = nn.functional.mse_loss(X_out @ theta, y_out)
        #res = MSE(theta,X_out,y_out[:,0])
        return torch.tensor([0])

mu_0_value = 1.
mu = torch.full((1,1), mu_0_value)
y = torch.full((2,1), mu_0_value)
X = torch.full((2,2), mu_0_value)
th = torch.full((2,1), mu_0_value)
mu.requires_grad=True
f = fo()
f(mu).backward()
print(mu.grad)