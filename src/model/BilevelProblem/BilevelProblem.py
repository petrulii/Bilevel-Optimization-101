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

  def optimize(self, mu, max_epochs=1, outer_optimizer=None, test_dataloader=None):
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
      for item in self.outer_dataloader:
        # Move data to GPU
        start = time.time()
        eval_data, eval_label, eval_depth, eval_normal = item
        X_outer = eval_data.to(self.device)
        y_outer = (eval_label.to(self.device), eval_depth.to(self.device), eval_normal.to(self.device))
        # Inner value corresponds to h*(X_outer)
        mu.requires_grad = True
        inner_value = self.inner_solution(mu, X_outer, y_outer)
        loss = self.outer_loss(mu, inner_value, y_outer)
        #make_dot(loss, params=dict([('mu', mu)])).render("graph_loss", format="png")
        # Backpropagation
        self.outer_optimizer.zero_grad()
        loss.backward()
        self.outer_optimizer.step()
        self.outer_scheduler.step()
        # Update loss and iteration count
        epoch_loss += loss.item()
        epoch_iters += 1
        iters += 1
        duration = time.time() - start
        times.append(duration)
      # Evaluate at the end of every epoch
      outer_losses.append(epoch_loss/epoch_iters)
      print("Outer iteration:", epoch, "|", iters)
      print("Outer iteration avg. loss:", epoch_loss/epoch_iters)
      print("Inner iteration avg. loss:", self.inner_solution.loss)
      inner_losses.append(self.inner_solution.loss)
      print("Inner dual iteration avg. loss:", self.inner_solution.dual_loss)
      test_loss, accuracy = self.evaluate(test_dataloader, mu)
      test_losses.append(test_loss)
      print("Test avg. loss:", test_loss)
      print("Test avg. acc.:", accuracy)
      print("Outer variable:", mu)
    return iters, outer_losses, inner_losses, test_losses, times
  
  def evaluate(self, test_dataloader, mu, max_iters=10):
    """
    Find the optimal outer solution.
      param test_dataloader: test data
      param mu: outer variable
    """
    total_loss, total_acc, iters = 0, 0, 0
    #for X_test, y_test in self.test_dataloader:
    # _____NYUv2 start_____
    for item in test_dataloader:
      eval_data, eval_label, eval_depth, eval_normal = item
      X_outer = eval_data
      y_outer = (eval_label, eval_depth, eval_normal)
      # _____NYUv2 end_____
      # Move to GPU
      X_outer = X_outer.to(self.device)
      if type(y_outer) is tuple:
        nb_items = len(y_outer)
        for i in range(nb_items):
          l = list(y_outer)
          l[i] = l[i].to(self.device)
          y_outer = tuple(l)
      else:
        y_outer = y_outer.to(self.device)
      # Inner value corresponds to h*(X_outer)
      inner_value = self.inner_solution(mu, X_outer, y_outer)
      # Compute accuracy
      seg, depth, normal = y_outer
      seg_pred, depth_pred, pred_normal = inner_value
      _, predicted = torch.max(seg_pred.data, 1)
      value = seg.detach().flatten().cpu()
      pred = predicted.detach().flatten().cpu()
      correct = torch.sum(value==pred)
      total = torch.sum(value!=-1)
      acc = correct/total
      total_acc += acc
      # Compute loss
      loss = self.outer_loss(mu, inner_value, y_outer)
      total_loss += loss.item()
      iters += 1
      if iters >= max_iters:
        break
    return total_loss/iters, total_acc/iters