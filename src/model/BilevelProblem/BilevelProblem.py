import sys
import math
import time
import random
import numpy as np
import torch

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src/')

from model.utils import set_seed
from model.InnerSolution.InnerSolution import InnerSolution
from model.utils import plot_1D_iterations, plot_2D_functions, plot_loss


class BilevelProblem:
  """
  Instanciates the bilevel problem and solves it using Neural Implicit Differentiation.
  """

  def __init__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, device, batch_size=64, X_val_t=None, y_val_t=None):
    """
    Init method.
      param outer_loss: outer level loss function
      param inner_loss: inner level loss function
      param outer_dataloader: input data and labels for the outer loss
      param inner_dataloader: input data and labels for the inner loss
      param batch_size: batch size for both inner and outer optimizers
      param device: CPU or GPU
    """
    self.__input_check__(outer_loss, inner_loss, outer_dataloader, inner_dataloader, device, batch_size)
    self.outer_loss = outer_loss
    self.inner_loss = inner_loss
    self.outer_dataloader = outer_dataloader
    self.inner_dataloader = inner_dataloader
    self.batch_size = batch_size
    self.device = device
    self.inner_solution = InnerSolution(inner_loss, inner_dataloader, device)
    self.X_val_t=X_val_t
    self.y_val_t=y_val_t

  def __input_check__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, device, batch_size):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (outer_loss is None) or (inner_loss is None):
      raise AttributeError("You must specify inner and outer losses.")
    if (outer_dataloader is None) or (inner_dataloader is None):
      raise AttributeError("You must give outer and inner dataloaders.")
    if (device is None):
      raise AttributeError("You must specify a device: GPU/CPU.")
    if not (type(batch_size) is int):
      raise TypeError("Batch size must be an integer value.")

  def optimize(self, mu, outer_optimizer, max_epochs=1):
    """
    Find the optimal outer solution.
      param mu: initial value of the outer variable
      param maxiter: maximum number of iterations
    """
    nb_iters, iters, losses, times = 0, [], [], []
    old_loss = None
    for epoch in range(max_epochs):
      for X_outer, y_outer in self.outer_dataloader:
        start = time.time()
        # Move to GPU
        X_outer = X_outer.to(self.device)
        y_outer = y_outer.to(self.device)
        # Inner value corresponds to h*(X_outer)
        inner_value = self.inner_solution(mu, X_outer, y_outer)
        # Making sure gradient of mu is computed.
        mu.requires_grad = True
        loss = self.outer_loss(mu, inner_value, y_outer)
        if not(old_loss is None):
          if torch.allclose(old_loss, loss):
            break
        # Backpropagation
        outer_optimizer.zero_grad()
        loss.backward()
        outer_optimizer.step()
        times.append(time.time() - start)
        iters.append(mu)
        losses.append(loss)
        nb_iters += 1
        #tmp_sol = self.inner_solution(mu, self.X_val_t, self.y_val_t)
        #tmp = self.outer_loss(mu, tmp_sol, self.y_val_t)
        old_loss = loss.clone()
    return nb_iters, iters, losses, times