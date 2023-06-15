import sys
import time
import torch
from torch.nn import functional as func
import gc
import wandb
import numpy as np
from statistics import mean

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model.InnerSolution.InnerSolution import InnerSolution

from sklearn.metrics import accuracy_score

#from torchviz import make_dot

from model.utils import get_memory_info, get_accuracy

class BilevelProblem:
  """
  Instanciates the bilevel problem and solves it using Neural Implicit Differentiation.
  """

  def __init__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_models, device, batch_size=64, max_inner_iters=200, max_inner_dual_iters=5, aux_dataloader=None):
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
    self.inner_solution = InnerSolution(inner_loss, inner_dataloader, inner_models, device, max_iters=max_inner_iters, max_dual_iters=max_inner_dual_iters)
    if aux_dataloader is None:
      self.aux_dataloader = outer_dataloader
    else:
      self.aux_dataloader = aux_dataloader

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

  def optimize(self, mu, max_epochs=10, max_iters=100, eval_every_n=10, test_dataloader=None):
    """
    Find the optimal outer solution.
      param mu: initial value of the outer variable
      param maxiter: maximum number of iterations
    """
    iters, outer_losses, inner_losses, val_accs, times, evaluated, acc_smooth = 0, [], [], [], [], False, 10
    # Making sure gradient of mu is computed.
    for epoch in range(max_epochs):
      epoch_iters = 0
      epoch_loss = 0
      #for X_outer, y_outer in self.outer_dataloader:
      for data in self.aux_dataloader:
        #print("Outer epoch:", epoch, "iteration:", iters)
        X_outer, main_label, aux_label, data_id = data
        aux_label = torch.stack(aux_label)
        y_outer = (main_label.to(self.device, dtype=torch.float), aux_label.to(self.device, dtype=torch.float))
        start = time.time()
        # Move data to GPU
        X_outer = X_outer.to(self.device, dtype=torch.float)
        #y_outer = y_outer.to(self.device, dtype=torch.float)
        # Inner value corresponds to h*(X_outer)
        mu.requires_grad = True
        forward_start = time.time()
        inner_value = self.inner_solution(mu, X_outer, y_outer)
        wandb.log({"duration of forward": time.time() - forward_start})
        #for pred in inner_value:
        #  pred.retain_grad()
        loss = self.outer_loss(mu, inner_value, y_outer)
        wandb.log({"inn. loss": self.inner_solution.loss})
        wandb.log({"inn. dual loss": self.inner_solution.dual_loss})
        wandb.log({"out. loss": loss.item()})
        wandb.log({"outer var. norm": torch.norm(mu).item()})
        #make_dot(loss, show_attrs=True, show_saved=True).render("graph_loss", format="png")
        # Backpropagation
        self.outer_optimizer.zero_grad()
        backward_start = time.time()
        loss.backward()
        wandb.log({"outer var. grad norm": torch.norm(mu.grad).item()})
        wandb.log({"duration of backward": time.time() - backward_start})
        self.outer_optimizer.step()
        # Update loss and iteration count
        epoch_loss += loss.item()
        epoch_iters += 1
        iters += 1
        duration = time.time() - start
        wandb.log({"iter. time": duration})
        print("mu:", mu[0:10])
        times.append(duration)
        # Inner losses
        inner_losses.append(self.inner_solution.loss)
        # Outer losses
        outer_losses.append(epoch_loss/epoch_iters)
        # Evaluate
        if (iters % eval_every_n == 0):
          _, val_acc = self.evaluate(self.outer_dataloader, mu)
          val_accs.append(val_acc)
          wandb.log({"acc": val_acc})
          if (test_dataloader!=None) and (not evaluated) and (len(val_accs)>acc_smooth) and (mean(val_accs[-acc_smooth:]) <= mean(val_accs[-(acc_smooth*2):-(acc_smooth)])):
            test_loss, test_acc = self.evaluate(test_dataloader, mu)
            wandb.log({"test loss": test_loss})
            wandb.log({"test acc": test_acc})
            evaluated = True
        if epoch_iters >= max_iters:
          break
      # Print predictions and labels for a sanity check
      _, class_pred = torch.max(inner_value[0], dim=1)
      print("Train prediction:", class_pred[0:6])
      print("Train label:", y_outer[0][0:6])
    return iters, outer_losses, inner_losses, val_accs, times
  
  def evaluate(self, dataloader, mu, max_iters=100):
    """
    Evaluate the prediction quality on the test dataset.
      param dataloader: data to evaluate on
      param mu: outer variable
    """
    self.inner_solution.eval = True
    total_loss, total_acc, iters = 0, 0, 0
    for data in dataloader:
      X_outer, main_label, aux_label, data_id = data
      aux_label = torch.stack(aux_label)
      y_outer = (main_label.to(self.device, dtype=torch.float), aux_label.to(self.device, dtype=torch.float))
      # Move data to GPU
      X_outer = X_outer.to(self.device, dtype=torch.float)
      # Inner value corresponds to h*(X_outer)
      inner_value = self.inner_solution(mu, X_outer, y_outer)
      # Compute loss
      loss = self.outer_loss(mu, inner_value, y_outer)
      # Compute accuracy
      _, class_pred = torch.max(inner_value[0], dim=1)
      class_pred = class_pred.to(self.device)
      accuracy = get_accuracy(class_pred, y_outer[0])
      total_acc += accuracy.item()
      total_loss += loss.item()
      iters += 1
    self.inner_solution.eval = False
    return total_loss/iters, total_acc/iters