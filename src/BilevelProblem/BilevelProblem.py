import math
import time
import torch
from torch.autograd import grad
from torch.autograd.functional import hessian
from utils import sample_X, sample_X_y
from FunctionApproximator.FunctionApproximator import FunctionApproximator


class BilevelProblem:
  """
  The BilevelProblem object instanciates the bilevel problem.
  """

  def __init__(self, outer_objective, inner_objective, method, data, fo_h_X=None, fi_h_X=None, find_theta_star=None, batch_size=64, device=None, gradients=None):
    """
    Init method.
      param inner_objective: inner level objective function
      param outer_objective: outer level objective function
      param X_outer: feature data for the outer objective
      param y_outer: label data for the outer objective
      param X_inner: feature data for the inner objective
      param y_inner: label data for the inner objective
      param find_theta_star: method to find the optimal parameter vector for class. imp. diff.
    """
    self.outer_objective = outer_objective
    self.inner_objective = inner_objective
    self.method = method
    self.X_outer = data[0]
    self.y_outer = data[1]
    self.X_inner = data[2]
    self.y_inner = data[3]
    self.dim_x = self.X_inner.size()[1]
    # If analytical gradients are provided.
    if gradients != None:
      self.outer_grad1 = gradients[0]
      self.outer_grad2 = gradients[1]
      self.inner_grad22 = gradients[2]
      self.inner_grad12 = gradients[3]
    if self.method=="neural_implicit_diff":
      self.fo_h_X = fo_h_X
      self.fi_h_X = fi_h_X
      dim_y = 1
      layer_sizes = [self.dim_x, 10, 20, 10, dim_y]
      # Neural network to approximate the function h*
      self.NN_h = FunctionApproximator(layer_sizes, loss_G=inner_objective, function='h', device=device)
      self.NN_h.load_data(self.X_inner, self.y_inner)
      # Neural network to approximate the function a*
      self.NN_a = FunctionApproximator(layer_sizes, function='a', device=device)
      self.NN_a.load_data(self.X_inner, self.y_inner, self.X_outer, self.y_outer)
    elif self.method=="implicit_diff":
      self.find_theta_star = find_theta_star
    self.device = device
    self.batch_size = batch_size
    self.__input_check__()

  def __input_check__(self):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (self.outer_objective is None) or (self.inner_objective is None):
      raise AttributeError("You must specify the inner and outer objectives")
    if (self.inner_grad22 is None) and (self.fo_h_X is None or self.fi_h_X is None):
      raise AttributeError("You must specify objectives with respect to h(X)")
    if (self.method == "implicit_diff")  and (self.find_theta_star is None):
      raise AttributeError("You must specify the closed form solution of the inner problem for classical imp. diff.")
    if not (self.method == "implicit_diff" or self.method == "neural_implicit_diff"):
      raise ValueError("Invalid method for solving the bilevel problem")

  def outer_grad1(self, mu, h, X_out, y_out):
    """
    Returns the gradient of the outer objective wrt to the first variable.
    """
    if self.method=="implicit_diff":
      theta = h
      X_out.detach()
      y_out.detach()
      theta.detach()
      assert(theta.requires_grad == X_out.requires_grad == y_out.requires_grad == False)
    elif self.method=="neural_implicit_diff":
      assert(X_out.requires_grad == y_out.requires_grad == False)
    else:
      raise ValueError("Unkown method for solving a bilevel problem")
    mu.detach()
    mu.requires_grad=True
    mu.retain_grad()
    self.outer_objective(mu, h, X_out, y_out).backward()
    gradient = torch.reshape(mu.grad, mu.size())
    if gradient is None:
      gradient = mu.detach().clone() * 0
    mu.requires_grad=False
    return gradient

  def outer_grad2(self, mu, h, X_out, y_out):
    """
    Returns the gradient of the outer objective wrt to the second variable.
    """
    assert(mu.requires_grad == X_out.requires_grad == y_out.requires_grad == False)
    if self.method=="implicit_diff":
      theta = h
      X_out.detach()
      y_out.detach()
      theta.detach()
      mu.detach()
      theta.requires_grad=True
      theta.retain_grad()
      self.outer_objective(mu, theta, X_out, y_out).backward()
      gradient = torch.reshape(theta.grad, theta.size())
      theta.requires_grad=False
    elif self.method=="neural_implicit_diff":
      h_X = h(X_out)
      h_X.retain_grad()
      self.fo_h_X(mu, h_X, y_out).backward()
      gradient = h_X.grad
    else:
      raise ValueError("Unkown method for solving a bilevel problem")
    return gradient

  def inner_grad22(self, mu, h, X_in, y_in):
    """
    Returns the hessian of the inner objective wrt to the second variable.
    """
    assert(mu.requires_grad == X_in.requires_grad == y_in.requires_grad == False)
    if self.method=="implicit_diff":
      theta = h
      X_in.detach()
      y_in.detach()
      theta.detach()
      mu.detach()
      theta.requires_grad=True
      theta.retain_grad()
      f = lambda arg1, arg2: self.inner_objective(arg1, arg2, X_in, y_in)
      hess = torch.reshape(hessian(f, (mu, theta))[1][1], (self.dim_x, self.dim_x))
      #tensor([[6881.8682, 5165.3926],
      #        [5165.3926, 7028.7324]], device='cuda:0')
      theta.requires_grad=False
    elif self.method=="neural_implicit_diff":
      h_X = h(X_in)
      h_X.retain_grad()
      f = lambda arg1, arg2: self.fi_h_X(arg1, arg2, y_in)
      hess = hessian(f, (mu, h_X))[1][1]
      hess = torch.reshape(hess, (h_X.size()[0],h_X.size()[0]))
    else:
      raise ValueError("Unkown method for solving a bilevel problem")
    return hess

  def inner_grad12(self, mu, h, X_in, y_in):
    """
    Returns part of the hessian of the inner objective wrt to both variables.
    """
    assert(X_in.requires_grad == y_in.requires_grad == False)
    mu.requires_grad=True
    if self.method=="implicit_diff":
      theta = h
      X_in.detach()
      y_in.detach()
      theta.detach()
      mu.detach()
      theta.requires_grad=True
      theta.retain_grad()
      f = lambda arg1, arg2: self.inner_objective(arg1, arg2, X_in, y_in)
      hess = hessian(f, (mu, theta))[0][1]
      hess = torch.reshape(hess, (mu.size()[0],theta.size()[0])).T
      theta.requires_grad=False
    elif self.method=="neural_implicit_diff":
      h_X = h(X_in)
      h_X.retain_grad()
      f = lambda arg1, arg2: self.fi_h_X(arg1, arg2, y_in)
      hess = hessian(f, (mu, h_X))[0][1]
      hess = torch.reshape(hess, (mu.size()[0],h_X.size()[0])).T
    else:
      raise ValueError("Unkown method for solving a bilevel problem")
    mu.requires_grad=False
    return hess

  def optimize(self, mu0, maxiter=100, step=0.1):
    """
    Find the optimal solution.
      param mu0: initial value of the outer variable
      param maxiter: maximum number of iterations
      param step: stepsize for gradient descent on the outer variable
    """
    if not isinstance(mu0, (torch.Tensor)):
      raise TypeError("Invalid input type for mu0, should be a tensor")
    mu_new = mu0
    n_iters, iters, converged, inner_loss, outer_loss, times = 0, [mu_new], False, [], [], []
    while n_iters < maxiter and not converged:
      mu_old = mu_new.clone()
      start = time.time()
      mu_new, h_star = self.find_mu_new(mu_old, step)
      inner_loss.append(self.inner_objective(mu_new, h_star, self.X_inner, self.y_inner))
      outer_loss.append(self.outer_objective(mu_new, h_star, self.X_outer, self.y_outer))
      times.append(time.time() - start)
      converged = self.check_convergence(mu_old, mu_new)
      iters.append(mu_new)
      n_iters += 1
    return mu_new, iters, n_iters, times, inner_loss, outer_loss, h_star

  def find_mu_new(self, mu_old, step):
    """
    Find the next value in gradient descent.
      param mu_old: old value of the outer variable
      param step: stepsize for gradient descent
    """
    if self.method=="implicit_diff":
      X_in, y_in = self.X_inner, self.y_inner
      X_out, y_out = self.X_outer, self.y_outer
      # 1) Find a parameter vector h* the argmin of the inner objective G(mu,h)
      theta_star = self.find_theta_star(self.X_inner, self.y_inner, mu_old)
      # 2) Get Jh* the Jacobian of the inner objective wrt h
      Jac = -1*torch.linalg.solve((self.inner_grad22(mu_old, theta_star, X_in, y_in)), (self.inner_grad12(mu_old, theta_star, X_in, y_in)))
      # 3) Compute grad L(mu): the gradient of L(mu) wrt mu
      grad = self.outer_grad1(mu_old, theta_star, X_out, y_out) + Jac.T @ self.outer_grad2(mu_old, theta_star, X_out, y_out)
      self.theta_star = theta_star
      h_star = theta_star
    elif self.method=="neural_implicit_diff":
      # 1) Find a function that approximates h*(x)
      h_star, loss_values = self.NN_h.train(mu_k=mu_old, num_epochs = 10)
      # 2) Find a function that approximates a*(x)
      a_star, loss_values = self.NN_a.train(self.inner_grad22, self.outer_grad2, mu_k=mu_old, h_k=h_star, num_epochs = 10)
      # 3) Compute grad L(mu): the gradient of L(mu) wrt mu
      X_out, y_out = sample_X_y(self.X_outer, self.y_outer, self.batch_size)
      X_in, y_in = sample_X_y(self.X_inner, self.y_inner, self.batch_size)
      B = self.B_star(mu_old, h_star, X_in, y_in)
      grad = self.outer_grad1(mu_old, h_star, X_out, y_out) + B.T @ (a_star(X_in))
      self.h_star = h_star
      self.a_star = a_star
    else:
      raise ValueError("Unkown method for solving a bilevel problem")
    # 4) Compute the next iterate x_{k+1} = x_k - grad L(x)
    mu_new = mu_old-step*grad
    # 5) Enforce x positive
    mu_new = torch.nn.functional.relu(mu_new)
    # Remove the associated autograd
    mu_new = mu_new.detach()
    return mu_new, h_star

  def h_star(self, h_theta=None):
    """
    Return the function h* for neural imp. diff. or h*_theta for classical imp. diff.
      param h_theta: a method that return a function parametrized by theta*
    """
    if self.method == "neural_implicit_diff":
      h = self.NN_h
    elif self.method == "implicit_diff":
      h = h_theta(self.theta_star)
    return h

  def a_star(self, mu=0, outer_grad2_h=None, h_theta=None, h_theta_grad=None):
    """
    Return the function a* for neural imp. diff. and an equivalent for classical imp. diff.
      param inner_grad22: the hessian of the inner objective
      param h_theta_grad: the gradient of a function parametrized by theta*
    """
    if self.method == "implicit_diff":
      # Need X_in and X_ou here, no?
      a = lambda X, y: h_theta_grad(X) @ torch.linalg.solve(self.inner_grad22(mu, self.theta_star, X, y), h_theta_grad(X).T) @ outer_grad2_h(mu, h_theta, X, y)
    return a

  def B_star(self, mu_old, h_star, X_in, y_in):
    """
    Computes the matrix B*(x).
    """
    return self.inner_grad12(mu_old, h_star, X_in, y_in)

  def check_convergence(self, mu_old, mu_new):
    """
    Checks convergence of the algorithm based on equality of last two iterates.
    """
    return torch.norm(mu_old-mu_new)<5.3844e-04
  