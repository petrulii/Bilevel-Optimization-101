import sys
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd.functional import hessian

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')


class InnerSolution(nn.Module):
  """
  Instanciates the inner solution of the bilevel problem.
  """

  def __init__(self, inner_loss, inner_dataloader, inner_models, device, optimizer=None):
    """
    Init method.
      param inner_loss: inner level objective function
      param inner_dataloader: data loader for inner data
      param device: CPU or GPU
    """
    super(InnerSolution, self).__init__()
    self.inner_loss = inner_loss
    self.inner_dataloader = inner_dataloader
    self.model, self.optimizer, self.dual_model, self.dual_optimizer = inner_models
    self.device = device
  
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
    running_loss, nb_iters, losses = 0, 0, []
    for epoch in range(max_epochs):
      #for X_inner, y_inner in self.inner_dataloader:
      # _____NYUv2 start_____
      for item in self.inner_dataloader:
        eval_data, eval_label, eval_depth, eval_normal = item
        X_inner = eval_data
        y_inner = (eval_label, eval_depth, eval_normal)
      # _____NYUv2 end_____
        # Move to GPU
        X_inner = X_inner.to(self.device)
        if type(y_inner) is tuple:
          nb_items = len(y_inner)
          for i in range(nb_items):
            l = list(y_inner)
            l[i] = l[i].type(torch.LongTensor)
            l[i] = l[i].to(self.device)
            y_inner = tuple(l)
        else:
          y_inner = y_inner.to(self.device)
        # Compute prediction and loss
        h_X_i = self.model.forward(X_inner)
        loss = self.inner_loss(mu, h_X_i, y_inner)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item()
        nb_iters += 1
        losses.append(running_loss/nb_iters)
        if nb_iters >= 3:#test
          break
    return nb_iters, losses
  
  def optimize_dual(self, mu, X_outer, y_outer, outer_grad2, max_epochs=1):
    """
    Optimization loop for the inner-level model that approximates a*.
    """
    running_loss, nb_iters, losses = 0, 0, []
    for epoch in range(max_epochs):
      #for X_inner, y_inner in self.inner_dataloader:
      # _____NYUv2 start_____
      for item in self.inner_dataloader:
        eval_data, eval_label, eval_depth, eval_normal = item
        X_inner = eval_data
        y_inner = (eval_label, eval_depth, eval_normal)
      # _____NYUv2 end_____
        # Move to GPU
        X_inner = X_inner.to(self.device)
        if type(y_inner) is tuple:
          nb_items = len(y_inner)
          for i in range(nb_items):
            l = list(y_inner)
            l[i] = l[i].type(torch.LongTensor)
            l[i] = l[i].to(self.device)
            y_inner = tuple(l)
        else:
          y_inner = y_inner.to(self.device)
        # Compute prediction and loss
        a_X_i = self.dual_model.forward(X_inner)
        h_X_i = self.model.forward(X_inner)
        a_X_o = self.dual_model.forward(X_outer)
        loss = self.loss_H(mu, h_X_i, a_X_i, a_X_o, y_inner, outer_grad2)
        # Backpropagation
        self.dual_optimizer.zero_grad()
        loss.backward()
        self.dual_optimizer.step()
        running_loss += loss.item()
        nb_iters += 1
        losses.append(running_loss/nb_iters)
        if nb_iters >= 3:#test
          break
    return nb_iters, losses

  def loss_H(self, mu, h_X_i, a_X_i, a_X_o, y_inner, outer_grad2):
    """
    Loss function for optimizing a*.
    """
    # Make sure a*(X) is a tuple, if not, wrap as tuple.
    if not(type(a_X_i) is tuple):
      a_X_i = (a_X_i,)
    if not(type(a_X_o) is tuple):
      a_X_o = (a_X_o,)
    nb_items = len(a_X_i)
    # Make sure h*(X) is a tuple, if not, wrap as tuple.
    if not(type(h_X_i) is tuple):
      h_X_i = (h_X_i,)
    if not(type(outer_grad2) is tuple):
      outer_grad2 = (outer_grad2,)
    # Specifying the inner objective as a function of h*(X)
    f = lambda h_X: self.inner_loss(mu, h_X, y_inner)
    # Find the product with the hessian wrt h*(X)
    hessvp = autograd.functional.hvp(f, h_X_i, a_X_i)[1]
    # Compute the loss
    term1, term2 = 0, 0
    for i in range(nb_items):
      term1 += torch.einsum('ij,ij->',a_X_i[i],hessvp[i])
    for i in range(nb_items):
      term2 += torch.einsum('ij,ij->',a_X_o[i],outer_grad2[i])
    return (1/2)*term1+(1/2)*term2

  def compute_hessian_vector_prod(self, mu, X_outer, y_outer, inner_value, dual_val):
    """
    Computing B*a where a is dual_val=a(X_outer) and B is the functional derivative delta_mu delta_h g(mu,h*).
    """
    # Specifying the inner objective as a function of mu and h*(X)
    f = lambda arg1, arg2: self.inner_loss(arg1, arg2, y_outer)
    # Make sure a*(X) is a tuple, if not, wrap as tuple.
    if not(type(dual_val) is tuple):
      dual_val = (dual_val,)
    # Make sure h*(X) is a tuple, if not, wrap as tuple.
    if not(type(inner_value) is tuple):
      inner_value = (inner_value,)
    # Here v has to be a tuple, so we concatinate mu with a*(X).
    v = (mu.detach().clone()*0,) + dual_val
    # Here args has to be a tuple, so we concatinate mu with h*(X).
    args = (mu,) + inner_value
    # Similar to H_loss eigsum
    hessvp = autograd.functional.hvp(f, args, v)[1][0]
    # Here extract the tuple again
    return hessvp


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
    ctx.mu = mu
    ctx.X_outer = X_outer
    ctx.y_outer = y_outer
    return inner_value

  @staticmethod
  def backward(ctx, outer_grad2):
    """
    Backward pass of a function that approximates h* for Neur. Imp. Diff.
    """
    # Context ctx allows to communicate from forward to backward
    inner_solution = ctx.inner_solution
    inner_value = ctx.inner_value
    mu = ctx.mu
    X_outer = ctx.X_outer
    y_outer = ctx.y_outer
    # Need to enable_grad because we use autograd in optimize_dual (disabled in backward() by default).
    with torch.enable_grad():
      # Here the model approximating a* needs to be trained on the same X_inner batches
      # as the h* model was trained on and on X_outter batches that h was evaluated on
      # in the outer loop where we optimize the outer objective g(mu, h).
      inner_solution.optimize_dual(mu, X_outer, y_outer, outer_grad2)
      dual_value = inner_solution.dual_model(X_outer)
      grad = inner_solution.compute_hessian_vector_prod(mu, X_outer, y_outer, inner_value, dual_value)
    return None, grad, None, None