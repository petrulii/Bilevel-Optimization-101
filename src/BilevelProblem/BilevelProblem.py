"""
The BilevelProblem object instanciates the bilevel problem.
"""
import numpy as np
import matplotlib.pyplot as plt
import FunctionApproximator

class BilevelProblem:

  def __init__(self, outer_objective, inner_objective, method, outer_grad1=None, outer_grad2=None, inner_grad22=None, inner_grad12=None, X_outer=None, y_outer=None, X_inner=None, y_inner=None):
    """
    Required inputs inner and outer level objectives of the bilevel problem.
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

  def __input_check__(self):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (self.outer_objective is None) or (self.inner_objective is None):
      raise AttributeError("You must specify the inner and outer objectives")
    if (self.outer_grad1 is None) or (self.outer_grad2 is None) or (self.inner_grad22 is None) or (self.inner_grad12 is None):
      raise AttributeError("You must specify each of the necessary gradients")
    if not (self.method == "implicit_diff" or self.method == "neural_implicit_diff")  or (self.method is None):
      raise ValueError("Invalid method for solving the bilevel problem")


  def optimize(self, x0, y0, maxiter=100, step=0.1):
    """
    Find the optimal solution.
      param x0: initial value for inner variable x
      param y0: initial value for outer variable x
      param maxiter: maximum number of iterations
    """
    if not isinstance(x0, (np.ndarray)):
      raise TypeError("Invalid input type for x0, should be a numpy ndarray")
    if not isinstance(y0, (np.ndarray)):
      raise TypeError("Invalid input type for y0, should be a numpy ndarray")
    if len(self.x0) < 1:
      raise ValueError("x0 must have length > 0")
    if len(self.y0) < 1:
      raise ValueError("y0 must have length > 0")
    n_iters = 0
    converged = False
    x_new = x0
    iters = [x_new]
    while n_iters < maxiter and converged is False:
      x_old = x_new.copy()
      x_new = self.find_x_new(x_old, step)
      converged = self.check_convergence()
      iters.append(x_new)
      n_iters += 1
    return x_new, iters, n_iters

  def find_x_new(self, x_old, step):
    """
    Find the optimal solution.
      param x0: initial value for inner variable x
      param y0: initial value for outer variable x
    """
    if self.method=="implicit_diff":
      # Get y*(x) the argmin of the inner objective g(x,y)
      # Get Jy*(x) the Jacobian
      # Compute the gradient of the outer objective L(x)=f(x,y*(x))
      # Compute the next iterate x_{k+1} = x_k - grad L(x)
      # Compute the Hessian of g(x,y*(x))
      y_opt = -x_old
      Jac = -1#(-np.invert(self.inner_grad22(x_old,y_opt))) * (self.inner_grad12(x_old,y_opt))
      grad = self.outer_grad1(x_old,y_opt) + Jac * self.outer_grad2(x_old,y_opt)#Jac.T * self.outer_grad2(x_old,y_opt)
      #print(x_old, self.outer_grad1(x_old,y_opt), Jac, Jac * self.outer_grad2(x_old,y_opt))
      x_new = x_old-step*grad
    elif self.method=="neural_implicit_diff":
      # Fit a neural network to y*_x: w -> y*_x(w)
      # Get y_k the arg min of y*_{x_k} where x_k is the current x
      # Fit a neural network to A^{-T} H f(x_k, _): w -> A^{-T} H f(x_k, w)
      # Get the Jacobian of y*_{x_k} where x_k is the current x
      # Compute the gradient of L(x) using the obtained y_k and the Jacobian
      # Compute the next iterate x_{k+1} = x_k - grad L(x)
      # Instanciate a function approximator with the training data.
      # Instanciate a function approximator with the training data.
      
      NN_h = FunctionApproximator()
      NN_h.load_data(self.X_inner, self.y_inner)
      # Find a function that approximates the data.
      NN_h.train()# Specify objective G
      h_star = NN_h.approximate_function()

      # Get y_k the minimizer of h_star(y)

      NN_a = FunctionApproximator()
      NN_a.load_data(self.X_inner, self.y_inner, self.X_outer, self.y_outer)
      # Find a function that approximates the data.
      NN_a.train()# Specify objective H
      a_star = NN_a.approximate_function()

      # Compute the gradient of L(x) wrt x
      grad = None
      x_new = x_old-step*grad
      x_new = None
    else:
      raise ValueError("Unkown method for solving a bilevel problem")   
    return x_new

  #TODO
  def check_convergence(self):
    """
    Checks convergence of the alorithm based on chosen convergence criateria.
    """
    return False

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
