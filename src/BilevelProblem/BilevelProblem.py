import math
import torch
import numpy as np
from utils import plot_loss
from utils import sample
import matplotlib.pyplot as plt
from FunctionApproximator.FunctionApproximator import FunctionApproximator


class BilevelProblem:
  """
  The BilevelProblem object instanciates the bilevel problem.
  """

  def __init__(self, outer_objective, inner_objective, method, outer_grad1=None, outer_grad2=None, inner_grad22=None, inner_grad12=None, find_h_star=None, X_outer=None, y_outer=None, X_inner=None, y_inner=None, batch_size=64):
    """
    Init method.
      param inner_objective: inner level objective function
      param outer_objective: outer level objective function
      param outer_grad1: gradient wrt first variable of the outer objective
      param outer_grad2: gradient wrt second variable of the outer objective
      param inner_grad22: hessian wrt second variable of the inner objective
      param inner_grad12: hessian wrt first and second variables of the inner objective
      param X_outer: feature data for the outer objective
      param y_outer: label data for the outer objective
      param X_inner: feature data for the inner objective
      param y_inner: label data for the inner objective
    """
    self.outer_objective = outer_objective
    self.inner_objective = inner_objective
    self.method = method
    self.outer_grad1 = outer_grad1
    self.outer_grad2 = outer_grad2
    self.inner_grad22 = inner_grad22
    self.inner_grad12 = inner_grad12
    self.X_outer = X_outer
    self.y_outer = y_outer
    self.X_inner = X_inner
    self.y_inner = y_inner
    self.batch_size = batch_size
    self.find_h_star = find_h_star
    if self.method=="neural_implicit_diff":
      self.NN_h = FunctionApproximator()
      self.NN_h.load_data(self.X_inner, self.y_inner)
      self.NN_a = FunctionApproximator()
      self.NN_a.load_data(self.X_inner, self.y_inner, self.X_outer, self.y_outer)
    self.__input_check__()

  def __input_check__(self):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (self.outer_objective is None) or (self.inner_objective is None):
      raise AttributeError("You must specify the inner and outer objectives")
    if (self.outer_grad1 is None) or (self.outer_grad2 is None) or (self.inner_grad22 is None) or (self.inner_grad12 is None):
      raise AttributeError("You must specify each of the necessary gradients")
    if (self.method == "implicit_diff")  and (self.find_h_star is None):
      raise AttributeError("You must specify the closed form solution of the inner problem for classical imp. diff.")
    if not (self.method == "implicit_diff" or self.method == "neural_implicit_diff")  or (self.method is None):
      raise ValueError("Invalid method for solving the bilevel problem")

  def optimize(self, mu0, maxiter=100, step=0.1):
    """
    Find the optimal solution.
      param mu0: initial value of the outer variable
      param maxiter: maximum number of iterations
      param step: stepsize for gradient descent on the outer variable
    """
    if not isinstance(mu0, (torch.Tensor)):
      raise TypeError("Invalid input type for mu0, should be a tensor")
    n_iters = 0
    converged = False
    mu_new = mu0
    iters = [mu_new]
    while n_iters < maxiter and converged is False:
      mu_old = mu_new.clone()
      mu_new = self.find_mu_new(mu_old, step)
      converged = self.check_convergence(mu_old, mu_new)
      iters.append(mu_new)
      n_iters += 1
    return mu_new, iters, n_iters

  def find_mu_new(self, mu_old, step):
    """
    Find the next value in gradient descent.
      param mu_old: old value of the outer variable
      param step: stepsize for gradient descent
    """
    if self.method=="implicit_diff":
      X_in, y_in = self.X_inner, self.y_inner
      X_out, y_out = X_in, y_in#self.X_outer, self.y_outer
      # 1) Find a function that approximates h*(x) the argmin of the inner objective G(x,y)
      h_star = self.find_h_star(X_in, y_in, mu_old)
      # 2) Get Jh*(x) the Jacobian
      Jac = (-1*torch.linalg.inv(self.inner_grad22(mu_old, h_star, X_in, y_in))) @ (self.inner_grad12(mu_old, h_star, X_in, y_in))
      # 3) Compute grad L(mu): the gradient of L(mu) wrt mu
      term1 = self.outer_grad1(mu_old, h_star, X_out, y_out)
      term2 = torch.transpose(Jac,0,1) @ self.outer_grad2(mu_old, h_star, X_out, y_out)# 1*700 @ 300*1
      grad = (term1 + term2)
    elif self.method=="neural_implicit_diff":
      # 1) Find a function that approximates h*(x)
      h_star, loss_values = self.NN_h.train(mu_k=mu_old, num_epochs = 10)
      #plot_loss(loss_values)
      # 2) Find a function that approximates a*(x)
      a_star, loss_values = self.NN_a.train(self.inner_grad22, self.outer_grad2, mu_k=mu_old, h_k=h_star, num_epochs = 10)
      #plot_loss(loss_values)
      # 3) Compute grad L(x): the gradient of L(x) wrt x
      X_in = sample(self.X_inner, self.batch_size)
      X_out = sample(self.X_outer, self.batch_size)
      term1 = self.outer_grad1(mu_old, h_star, X_out, None)
      term2 = self.B_star(mu_old, h_star, X_in, None)
      term3 = a_star(X_in)
      grad = term1 + term2.T @ term3
    else:
      raise ValueError("Unkown method for solving a bilevel problem")
    # 4) Compute the next iterate x_{k+1} = x_k - grad L(x)
    mu_new = mu_old-step*grad
    # 5) Enforce x positive
    mu_new = torch.full((1,1), 0.) if mu_new[0,0]<0 else mu_new
    # Remove the associated autograd
    mu_new = mu_new.detach()
    return mu_new

  def B_star(self, mu_old, h_star, X_in, y_in):
    """
    Computes the matrix B*(x).
    """
    return self.inner_grad12(mu_old, h_star, X_in, y_in)

  def check_convergence(self, mu_old, mu_new):
    """
    Checks convergence of the algorithm based on equality of last two iterates.
    """
    return math.isclose(mu_old, mu_new, rel_tol=1e-5)

  #TODO
  def visualize(self, intermediate_points, plot_x_lim=5, plot_y_lim=5, plot_nb_contours=50):
    """
    Plot the intermediate steps of bilevel optimization.
      param intermediate_points: intermediate points of gradient descent
    """
    xlist = np.linspace(-1*plot_x_lim, plot_x_lim, plot_nb_contours)
    ylist = np.linspace(-1*plot_y_lim, plot_y_lim, plot_nb_contours)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros_like(X)
    for i in range(0, len(X)):
        for j in range(0, len(X)):
            Z[i, j] = self.outer_objective(X[i, j], Y[i, j])
    plt.clf()
    cs = plt.contour(X, Y, Z, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    for i in intermediate_points:
      plt.scatter(i, 0, marker='.', color='#495CD5')
    plt.scatter(intermediate_points[-1], 0, marker='*', color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-plot_x_lim, plot_x_lim)
    plt.ylim(-plot_y_lim, plot_y_lim)
    plt.show()
