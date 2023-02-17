"""
The BilevelProblem object instanciates the bilevel problem.
"""
import numpy as np
import matplotlib.pyplot as plt

class BilevelProblem:

  def __init__(self, outer_objective, inner_objective):
    """
    Required inputs inner and outer level objectives of the bilevel problem.
      param inner_objective: inner level objective function
      param outer_objective: outer level objective function
    """
    self.outer_objective = outer_objective
    self.inner_objective = inner_objective

  def optimize(self, method, x0, y0, outer_grad1=None, outer_grad2=None, inner_grad22=None, inner_grad12=None, maxiter=1000): # Possibly move method to find_x_new
    """
    Find the optimal solution.
      param x0: initial value for inner variable x
      param y0: initial value for outer variable x
    """
    self.x0=x0
    self.y0=y0
    self.maxiter=maxiter
    if method=="implicit_diff":
      # Get y*(x) the argmin of the inner objective g(x,y)
      # Get Jy*(x) the Jacobian
      # Compute the gradient of the outer objective L(x)=f(x,y*(x))
      # Compute the next iterate x_{k+1} = x_k - grad L(x)
      self.outer_grad1 = outer_grad1
      self.outer_grad2 = outer_grad2
      self.inner_grad22 = inner_grad22
      self.inner_grad12 = inner_grad12
      n_iters = 0
      converged = False
      x_new = self.x0
      iters = [x_new]
      while n_iters < self.maxiter and converged is False:
        x_old = x_new#.copy()
        x_new = self.find_x_new(x_old)
        converged = self.check_convergence()
        iters.append(x_new)
        n_iters += 1
      x_opt = x_new
    elif method=="neural_implicit_diff":
      # Fit a neural network to y*_x: w -> y*_x(w)
      # Get y_k the arg min of y*_{x_k} where x_k is the current x
      # Fit a neural network to A^{-T} H f(x_k, _): w -> A^{-T} H f(x_k, w)
      # Get the Jacobian of y*_{x_k} where x_k is the current x
      # Compute the gradient of L(x) using the obtained y_k and the Jacobian
      # Compute the next iterate x_{k+1} = x_k - grad L(x)
      n_iters = 0
      x_opt = iters = None
    else:
      raise ValueError("Unkown method for solving a bilevel problem")        
    return x_opt, iters, n_iters

  def find_x_new(self, x_old, stepsize=0.1):
    """
    Find the optimal solution.
      param x0: initial value for inner variable x
      param y0: initial value for outer variable x
    """
    # Compute the Hessian of g(x,y*(x))
    y_opt = 0
    Jac = (-np.invert(self.inner_grad22(x_old,y_opt))) * (self.inner_grad12(x_old,y_opt))
    grad = self.outer_grad1(x_old,y_opt) + Jac.T * self.outer_grad2(x_old,y_opt)
    x_new = x_old-stepsize*grad
    return x_new
    
  def check_convergence(self):
    return False
    # Visualize iterations.

  def visualize(self, intermediate_points, plot_x_lim=5, plot_y_lim=5, plot_nb_contours=50):
    """
    Plot the intermediate steps of bilevel optimization.
      param intermediate_points: intermediate points of gradient descent
    """
    points = np.asarray(intermediate_points)
    xlist = np.linspace(-1*plot_x_lim, plot_x_lim, plot_nb_contours)
    ylist = np.linspace(-1*plot_y_lim, plot_y_lim, plot_nb_contours)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros_like(X)
    for i in range(0, len(X)):
        for j in range(0, len(X)):
            Z[i, j] = self.outer_objective(X[i, j], Y[i, j])
    plt.clf()
    cs = plt.contour(X, Y, Z, plot_nb_contours, cmap=plt.cm.magma, alpha=0.8, extend='both')
    # Show contour values.
    #plt.clabel(cs, cs.levels, inline=True, fontsize=10)
    #plt.scatter(points[0, 0], points[0, 1], marker='.', color='#495CD5')
    #plt.scatter(points[1:, 0], points[1:, 1], marker='.', color='#5466DE')
    #plt.scatter(points[-1, 0], points[-1, 1], marker='*', color='r')
    for i in intermediate_points:
      plt.scatter(i, 0, marker='.', color='#495CD5')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-plot_x_lim, plot_x_lim)
    plt.ylim(-plot_y_lim, plot_y_lim)
    plt.show()
