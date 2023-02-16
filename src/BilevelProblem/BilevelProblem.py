"""
The BilevelProblem object instanciates the bilevel problem.
"""
import numpy as np
import matplotlib.pyplot as plt

class BilevelProblem:

    def __init__(self, inner_objective, outer_objective):
        """
        Required inputs inner and outer level objectives of the bilevel problem.
          param inner_objective: inner level objective function
          param outer_objective: outer level objective function
        """
        self.inner_objective = inner_objective
        self.outer_objective = outer_objective

    def solve(self, x0=0, y0=0):
        """
        Find the optimal solution.
          param x0: initial value for inner variable x
          param y0: initial value for outer variable x
        """
        # Fit a neural network to y*_x: w -> y*_x(w)
        # Get y_k the arg min of y*_{x_k} where x_k is the current x
        # Fit a neural network to A^{-T} H f(x_k, _): w -> A^{-T} H f(x_k, w)
        # Get the Jacobian of y*_{x_k} where x_k is the current x
        # Compute the gradient of L(x) using the obtained y_k and the Jacobian
        # Compute the next iteration x_{k+1} = x_k - grad L(x)

        x_opt, iters, n_iters = 0
        
        return x_opt, iters, n_iters

    def visualize(self, iterations):
        """
        Visualize the optimization procedure.
          param iterations: iteration points
        """
        if self.x0.ndim!=2 and self.y0.ndim!=1:
            raise AttributeError("Visualizing iterations is only possible with 2D x and 1D y.") 
        plt.savefig("figures/bilevel_optimization.png", format="png")
        plt.show()
