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

    def visualize(self, iterations):
        """
        Visualize the optimization procedure.
          param iterations: iteration points
        """
        if self.x0.ndim!=2 and self.y0.ndim!=1:
            raise AttributeError("Visualizing iterations is only possible in the 2D x and 1D y case.") 
        plt.savefig("figures/bilevel_optimization.png", format="png")
        plt.show()
