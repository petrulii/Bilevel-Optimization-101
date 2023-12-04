import sys
import time
import torch
import torch.nn as nn
import wandb
import numpy as np
from torch.nn import functional as func
from statistics import mean

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')
sys.path.append('/home/clear/ipetruli/packages/miniconda3/lib/python3.10/site-packages')

from model.InnerSolution.InnerSolution import InnerSolution
from sklearn.metrics import accuracy_score
from model.utils import get_memory_info, get_accuracy, tensor_to_state_dict, find_V_opt, augment_stage1_feature, augment_stage2_feature, fit_linear, linear_reg_pred
from torchviz import make_dot

# Loss helper functions
MSE = nn.MSELoss()

class BilevelProblem:
  """
  Instanciates the bilevel problem and solves it using Neural Implicit Differentiation.
  """

  def __init__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_models, device, batch_size=64, max_inner_epochs=100, max_inner_dual_epochs=100):
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
    self.inner_solution = InnerSolution(inner_loss, inner_dataloader, inner_models, device, batch_size, max_epochs=max_inner_epochs, max_dual_epochs=max_inner_dual_epochs)

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
      raise AttributeError("You must specify both inner and dual inner models and their optimizers.")
    if (device is None):
      raise AttributeError("You must specify a device: GPU/CPU.")
    if not (type(batch_size) is int):
      raise TypeError("Batch size must be an integer value.")

  def optimize(self, outer_param, max_epochs=100, eval_every_n=10, validation_data=None, test_data=None):
    """
    Find the optimal outer solution.
      param outer_param: initial value of the outer variable
      param maxiter: maximum number of iterations
    """
    iters, outer_losses, inner_losses, val_losses, times, evaluated, acc_smooth = 0, [], [], [], [], False, 10
    for epoch in range(max_epochs):
      for data in self.outer_dataloader:
        start = time.time()
        Z, X, Y = data
        # Move data to GPU
        Z = Z.to(self.device, dtype=torch.float)
        X = X.to(self.device, dtype=torch.float)
        Y = Y.to(self.device, dtype=torch.float)
        # Inner value corresponds to h*(Z)
        outer_param.requires_grad = True
        forward_start = time.time()
        #V = find_V_opt(self.outer_model, outer_param, g_Z=None, X_in, lam1, self.device, inner_dataloader=self.inner_dataloader, inner_solution=self.inner_solution)
        inner_value = self.inner_solution(outer_param, Z, X)
        wandb.log({"duration of forward": time.time() - forward_start})
        #feature = augment_stage1_feature(inner_value)
        #predicted_treatment_feature = linear_reg_pred(feature, V)
        #loss = self.outer_loss(outer_param, predicted_treatment_feature, Y)
        loss = self.outer_loss(outer_param, inner_value, Y)
        # For checking the computational <autograd> graph.
        #make_dot(loss, params={ "outer param.":outer_param}, show_attrs=True, show_saved=True).render("graph_loss", format="png")
        #exit(0)
        wandb.log({"inn. loss": self.inner_solution.loss})
        wandb.log({"inn. dual loss": self.inner_solution.dual_loss})
        wandb.log({"out. loss": loss.item()})
        wandb.log({"outer var. norm": torch.norm(outer_param).item()})
        wandb.log({"outer var. grad. norm": torch.norm(outer_param.grad).item()})
        # Backpropagation
        self.outer_optimizer.zero_grad()
        backward_start = time.time()
        loss.backward()
        self.outer_optimizer.step()
        wandb.log({"duration of backward": time.time() - backward_start})
        # Update loss and iteration count
        iters += 1
        duration = time.time() - start
        wandb.log({"iter. time": duration})
        times.append(duration)
        # Inner losses
        inner_losses.append(self.inner_solution.loss)
        # Outer losses
        outer_losses.append(loss.item())
        # Evaluate on validation data and check the stopping condition
        if (validation_data!=None) and (iters % eval_every_n == 0):#(not evaluated) and (len(val_accs)>acc_smooth) and (mean(val_accs[-acc_smooth:]) <= mean(val_accs[-(acc_smooth*2):-(acc_smooth)])):
          feature = augment_stage2_feature(inner_value)
          #u = fit_linear(Y, feature, 0.1)
          wandb.log({"val. loss": self.evaluate(validation_data, outer_param)})
          if (test_data!=None):
            wandb.log({"test loss": self.evaluate(test_data, outer_param)})
            #evaluated = True
      if not(self.outer_scheduler is None):
        self.outer_scheduler.step()
        wandb.log({"outer lr": self.outer_optimizer.param_groups[0]['lr']})
    return iters, outer_losses, inner_losses, val_losses, times

  def evaluate(self, data, outer_param):#, u):
    """
    Evaluate the prediction quality on the test dataset.
      param data: data to evaluate on
      param outer_param: outer variable
    """
    with torch.no_grad():
      self.outer_model.train(False)
      self.inner_solution.model.train(False)
      self.inner_solution.dual_model.train(False)
      self.inner_solution.eval = True
      #u_dim = 33
      Y = (torch.from_numpy(data.outcome)).to(self.device, dtype=torch.float)
      X = (torch.from_numpy(data.treatment)).to(self.device, dtype=torch.float)
      #u = torch.reshape(outer_param[:u_dim], (u_dim,1))
      #outer_param_without_u = outer_param[u_dim:]
      #outer_NN_dic = tensor_to_state_dict(self.outer_model, outer_param_without_u, self.device)
      outer_NN_dic = tensor_to_state_dict(self.outer_model, outer_param, self.device)
      # Get the value of f(X)
      treatment_feature = torch.func.functional_call(self.outer_model, parameter_and_buffer_dicts=outer_NN_dic, args=X)
      #loss = MSE((treatment_feature @ u[:-1] + u[-1]), Y)
      loss = MSE(treatment_feature, Y)
      self.outer_model.train(True)
      self.inner_solution.model.train(True)
      self.inner_solution.dual_model.train(True)
      self.inner_solution.eval = False
      return loss.item()