import sys
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd.functional import hessian

# Add main project directory path
sys.path.append('/home/ipetruli/Bureau/bilevel-optimization/src')



class InnerSolution(nn.Module):
  """
  Instanciates the inner solution of the bilevel problem.
  """

  def __init__(self, inner_loss, inner_dataloader, device):
    """
    Init method.
      param inner_loss: inner level objective function
      param inner_dataloader: data loader for inner data
      param device: CPU or GPU
    """
    self.__input_check__(inner_loss, inner_dataloader, device)
    super(InnerSolution, self).__init__()
    self.inner_loss = inner_loss
    self.inner_dataloader = inner_dataloader
    layer_sizes = [2, 10, 20, 10, 1]
    # Neural network to approximate the function h*
    self.model = (NeuralNetworkModel(layer_sizes)).to(device)
    # Neural network to approximate the function a*
    self.dual_model = (NeuralNetworkDualModel(layer_sizes)).to(device)
    # Optimizer that improves the approximation of h*
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
    # Optimizer that improves the approximation of a*
    self.dual_optimizer = torch.optim.SGD(self.dual_model.parameters(), lr=0.01)
    self.device = device
  
  def __input_check__(self, inner_loss, inner_dataloader, device):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (inner_loss is None):
      raise AttributeError("You must specify the inner objective.")
    if (inner_dataloader is None):
      raise AttributeError("You must give the inner data loader.")
    if (device is None):
      raise AttributeError("You must specify a device: GPU/CPU.")

  def forward(self, mu, X_outer, y_outer):
    """
    Forward pass of a neural network that approximates the function h* for Neur. Imp. Diff.
      param mu: the current outer variable
      param X_outer: the outer data that the dual model needs access to
    """
    #with torch.enable_grad():
    # We use an intermediate ArgMinOp because we can only write a custom backward for functions
    # of type torch.autograd.Function, nn.Module doesn't allow to custumize the backward.
    opt_inner_val = ArgMinOp.apply(self, mu, X_outer, y_outer)
    return opt_inner_val

  def optimize(self, mu, max_epochs=1):
    """
    Optimization loop for the inner-level model that approximates h*.
    """
    nb_iters, losses = 0, []
    for epoch in range(max_epochs):
      for X_inner, y_inner in self.inner_dataloader:
        # Move to GPU
        X_inner = X_inner.to(self.device)
        y_inner = y_inner.to(self.device)
        # Compute prediction and loss
        h_X_i = self.model.forward(X_inner)
        loss = self.inner_loss(mu, h_X_i, y_inner)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        losses.append(loss.data)
        nb_iters += 1
    return nb_iters, losses
  
  def optimize_dual(self, mu, X_outer, y_outer, outer_grad2, max_epochs=1):
    """
    Optimization loop for the inner-level model that approximates a*.
    """
    nb_iters, losses = 0, []
    for epoch in range(max_epochs):
      for X_inner, y_inner in self.inner_dataloader:
        # Move to GPU
        X_inner = X_inner.to(self.device)
        y_inner = y_inner.to(self.device)
        # Compute prediction and loss
        a_X_i = self.dual_model.forward(X_inner)
        h_X_i = self.model.forward(X_inner)
        a_X_o = self.dual_model.forward(X_outer)
        # Here autograd outer_grad2?
        loss = self.loss_H(mu, h_X_i, a_X_i, a_X_o, y_inner, outer_grad2)
        # Backpropagation
        self.dual_optimizer.zero_grad()
        loss.backward()
        self.dual_optimizer.step()
        losses.append(loss.data)
        nb_iters += 1
    return nb_iters, losses

  def loss_H(self, mu, h_X_i, a_X_i, a_X_o, y_inner, outer_grad2):
    """
    Loss function for optimizing a*.
    """
    f = lambda arg1, arg2: self.inner_loss(arg1, arg2, y_inner)
    v = (mu.detach().clone()*0, a_X_i)
    hessvp = autograd.functional.hvp(f, (mu, h_X_i), v)[1][1]
    return (1/2)*torch.mean(a_X_i.T @ hessvp)+(1/2)*torch.mean(a_X_o.T @ outer_grad2)

  def compute_hessian_vector_prod(self, mu, X_outer, y_outer, inner_value, dual_val):
    """
    Computing B*a where a is dual_val=a(X_outer) and B is the functional derivative delta_mu delta_h g(mu,h*).
    """
    f = lambda arg1, arg2: self.inner_loss(arg1, arg2, y_outer)
    v = (mu.detach().clone()*0, dual_val)
    hessvp = autograd.functional.hvp(f, (mu, inner_value), v)[1][0]
    return hessvp

  def compute_hessian(self, mu, inner_value, y_inner):
    """
    Returns the hessian of the inner objective wrt to the second argument h/theta.
    """
    f = lambda arg1, arg2: self.inner_loss(arg1, arg2, y_inner)
    hess = hessian(f, (mu, inner_value))[1][1]
    dim = inner_value.size()[0]
    hess = torch.reshape(hess, (dim, dim))
    return hess


class ArgMinOp(torch.autograd.Function):
  """
  A pure function that approximates h*.
  """

  @staticmethod
  def forward(ctx, inner_solution, mu, X_outer, y_outer):
    """
    Forward pass of a function that approximates h* for Neur. Imp. Diff.
    """
    # In forward autograd is disabled by default but we use it in optimize(mu).
    with torch.enable_grad():
      # Train the model to approximate h* at mu_k
      inner_solution.optimize(mu)
    # Remember the value h*(X_outer)
    inner_value = inner_solution.model(X_outer)
    # Context ctx allows to communicate from forward to backward
    ctx.inner_solution = inner_solution
    ctx.inner_value = inner_value
    ctx.save_for_backward(mu, X_outer, y_outer)
    return inner_value

  @staticmethod
  def backward(ctx, outer_grad2):
    """
    Backward pass of a function that approximates h* for Neur. Imp. Diff.
    """
    # Context ctx allows to communicate from forward to backward
    inner_solution = ctx.inner_solution
    inner_value = ctx.inner_value
    mu, X_outer, y_outer = ctx.saved_tensors
    ## Solve a system Ax+b=0 by minimizing min_a 0.5 aTAa + aTb
    # A: the hessian of the inner objective
    # b: outer_grad2
    # Need to enable it because I call autograd in optimize_dual.
    with torch.enable_grad():
      # Here the model approximating a* needs to be trained on the same X_inner batches
      # as the h* model was trained on and on X_outter batches that h was evaluated on
      # in the outer loop where we optimize the outer objective g(mu, h).
      inner_solution.optimize_dual(mu, X_outer, y_outer, outer_grad2)
      dual_value = inner_solution.dual_model(X_outer)
      grad = inner_solution.compute_hessian_vector_prod(mu, X_outer, y_outer, inner_value, dual_value)
    return None, grad, None, None



class NeuralNetworkDualModel(nn.Module):
  """
  A neural network to approximate the function a* for Neur. Imp. Diff.
  """
  def __init__(self, layer_sizes):
    super(NeuralNetworkDualModel, self).__init__()
    self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
    nn.init.kaiming_uniform_(self.layer_1.weight)
    self.layer_2 = nn.Linear(layer_sizes[1], layer_sizes[2])
    nn.init.kaiming_uniform_(self.layer_2.weight)
    self.layer_3 = nn.Linear(layer_sizes[2], layer_sizes[3])
    nn.init.kaiming_uniform_(self.layer_3.weight)
    self.layer_4 = nn.Linear(layer_sizes[3], layer_sizes[4])

  def forward(self, x):
    x = torch.relu(self.layer_1(x))
    x = torch.tanh(self.layer_2(x))
    x = torch.tanh(self.layer_3(x))
    x = self.layer_4(x)
    return x



class NeuralNetworkModel(nn.Module):
  """
  A neural network to approximate the function h* for Neur. Imp. Diff.
  """
  def __init__(self, layer_sizes):
    super(NeuralNetworkModel, self).__init__()
    self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
    nn.init.kaiming_uniform_(self.layer_1.weight)
    self.layer_2 = nn.Linear(layer_sizes[1], layer_sizes[2])
    nn.init.kaiming_uniform_(self.layer_2.weight)
    self.layer_3 = nn.Linear(layer_sizes[2], layer_sizes[3])
    nn.init.kaiming_uniform_(self.layer_3.weight)
    self.layer_4 = nn.Linear(layer_sizes[3], layer_sizes[4])

  def forward(self, x):
    x = self.layer_1(x)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)
    return x
  
# Look at Function doc for enable_grad default value.
# Check forward once.
# Make sure gradients are not added but zeroed before.