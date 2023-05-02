import sys
import time
import torch
from torch.nn import functional as func
import gc

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model.InnerSolution.InnerSolution import InnerSolution

from sklearn.metrics import accuracy_score

from torchviz import make_dot

from model.utils import get_memory_info

class BilevelProblem:
  """
  Instanciates the bilevel problem and solves it using Neural Implicit Differentiation.
  """

  def __init__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_models, device, batch_size=64):
    """
    Init method.
      param outer_loss: outer level loss function
      param inner_loss: inner level loss function
      param outer_dataloader: input data and labels for the outer loss
      param inner_dataloader: input data and labels for the inner loss
      param batch_size: batch size for both inner and outer optimizers
      param device: CPU or GPU
    """
    self.__input_check__(outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_models, device, batch_size)
    self.outer_loss = outer_loss
    self.inner_loss = inner_loss
    self.outer_dataloader = outer_dataloader
    self.inner_dataloader = inner_dataloader
    self.batch_size = batch_size
    self.device = device
    self.outer_model, self.outer_optimizer, self.outer_scheduler = outer_model
    self.inner_solution = InnerSolution(inner_loss, inner_dataloader, inner_models, device)

  def __input_check__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_models, device, batch_size):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (outer_loss is None) or (inner_loss is None):
      raise AttributeError("You must specify inner and outer losses.")
    if (outer_dataloader is None) or (inner_dataloader is None):
      raise AttributeError("You must give outer and inner dataloaders.")
    if (outer_model is None) or not(len(outer_model)==3):
      raise AttributeError("You must specify the outer model and its optimizer.")
    if (inner_models is None) or not(len(inner_models)==6):
      raise AttributeError("You must specify both inner models and their optimizers.")
    if (device is None):
      raise AttributeError("You must specify a device: GPU/CPU.")
    if not (type(batch_size) is int):
      raise TypeError("Batch size must be an integer value.")

  def optimize(self, mu, max_epochs=1, outer_optimizer=None, test_dataloader=None, max_iters=100):
    """
    Find the optimal outer solution.
      param mu: initial value of the outer variable
      param maxiter: maximum number of iterations
    """
    iters, outer_losses, inner_losses, test_losses, times = 0, [], [], [], []
    # Making sure gradient of mu is computed.
    for epoch in range(max_epochs):
      epoch_iters = 0
      epoch_loss = 0
      #for X_outer, y_outer in self.outer_dataloader:
      for data in self.outer_dataloader:
        print("Outer epoch:", epoch, "iteration:", iters)
        X_outer, main_label, aux_label, data_id = data
        aux_label = torch.stack(aux_label)
        y_outer = (main_label, aux_label)
        start = time.time()
        # Move data to GPU
        X_outer = X_outer.to(self.device, dtype=torch.float)
        for task in y_outer:
          task.to(self.device, dtype=torch.float)
        #y_outer = y_outer.to(self.device, dtype=torch.float)
        # Inner value corresponds to h*(X_outer)
        mu.requires_grad = True
        inner_value = self.inner_solution(mu, X_outer, y_outer)
        for pred in inner_value:
          pred.retain_grad()
        loss = self.outer_loss(mu, inner_value, y_outer)
        print("Inner loss:", self.inner_solution.loss)
        print("Outer loss:", loss.item())
        #make_dot(loss, show_attrs=True, show_saved=True).render("graph_loss", format="png")
        # Backpropagation
        self.outer_optimizer.zero_grad()
        loss.backward()
        self.outer_optimizer.step()
        # Update loss and iteration count
        epoch_loss += loss.item()
        epoch_iters += 1
        iters += 1
        duration = time.time() - start
        times.append(duration)
        # Inner losses
        inner_losses.append(self.inner_solution.loss)
        # Outer losses
        outer_losses.append(epoch_loss/epoch_iters)
        # Test losses
        if not (test_dataloader is None):
          test_loss = self.evaluate(test_dataloader, mu)
          test_losses.append(test_loss)
        else:
          test_losses.append(0)
        if epoch_iters >= max_iters:
          break
      self.outer_scheduler.step()
      print("Outer iteration avg. loss:", epoch_loss/epoch_iters)
      print("Inner iteration avg. loss:", self.inner_solution.loss)
      print("Inner dual iteration avg. loss:", self.inner_solution.dual_loss)
    return iters, outer_losses, inner_losses, test_losses, times
  
  def evaluate(self, test_dataloader, mu, max_iters=10):
    """
    Find the optimal outer solution.
      param test_dataloader: test data
      param mu: outer variable
    """
    total_loss, iters = 0, 0
    #for X_test, y_test in self.test_dataloader:
    for data in test_dataloader:
      X_outer, main_label, aux_label, data_id = data
      aux_label = torch.stack(aux_label)
      y_outer = (main_label, aux_label)
      # Move data to GPU
      X_outer = X_outer.to(self.device, dtype=torch.float)
      for task in y_outer:
        task.to(self.device, dtype=torch.float)
      #y_outer = y_outer.to(self.device, dtype=torch.float)
      # Inner value corresponds to h*(X_outer)
      inner_value = self.inner_solution(mu, X_outer, y_outer)
      loss = self.outer_loss(mu, inner_value, y_outer)
      total_loss += loss.item()
      iters += 1
      if iters >= max_iters:
        break
    print("Test avg. loss:", total_loss/iters)
    return total_loss/iters