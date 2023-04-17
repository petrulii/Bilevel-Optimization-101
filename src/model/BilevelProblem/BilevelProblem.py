import sys
import time
import torch

# Add main project directory path
sys.path.append('/home/clear/ipetruli/projects/bilevel-optimization/src')

from model.InnerSolution.InnerSolution import InnerSolution

from sklearn.metrics import accuracy_score


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
    self.outer_model, self.outer_optimizer = outer_model
    self.inner_solution = InnerSolution(inner_loss, inner_dataloader, inner_models, device)

  def __input_check__(self, outer_loss, inner_loss, outer_dataloader, inner_dataloader, outer_model, inner_models, device, batch_size):
    """
    Ensure that the inputs are of the right form and all necessary inputs have been supplied.
    """
    if (outer_loss is None) or (inner_loss is None):
      raise AttributeError("You must specify inner and outer losses.")
    if (outer_dataloader is None) or (inner_dataloader is None):
      raise AttributeError("You must give outer and inner dataloaders.")
    if (outer_model is None) or not(len(outer_model)==2):
      raise AttributeError("You must specify the outer model and its optimizer.")
    if (inner_models is None) or not(len(inner_models)==4):
      raise AttributeError("You must specify both inner models and their optimizers.")
    if (device is None):
      raise AttributeError("You must specify a device: GPU/CPU.")
    if not (type(batch_size) is int):
      raise TypeError("Batch size must be an integer value.")

  def optimize(self, mu, max_epochs=1, outer_optimizer=None):
    """
    Find the optimal outer solution.
      param mu: initial value of the outer variable
      param maxiter: maximum number of iterations
    """
    running_loss, nb_iters, iters, losses, times = 0, 0, [], [], []
    # Making sure gradient of mu is computed.
    for epoch in range(max_epochs):
      #for X_outer, y_outer in self.outer_dataloader:
      # _____NYUv2 start_____
      for item in self.outer_dataloader:
        eval_data, eval_label, eval_depth, eval_normal = item
        X_outer = eval_data
        y_outer = (eval_label, eval_depth, eval_normal)
      # _____NYUv2 end_____
        start = time.time()
        iters.append(torch.flatten(mu.detach().clone()))
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
        mu.requires_grad = True
        inner_value = self.inner_solution(mu, X_outer, y_outer)
        for item in inner_value:
          item.requires_grad=True
        loss = self.outer_loss(mu, inner_value, y_outer)
        # Backpropagation
        self.outer_optimizer.zero_grad()
        loss.backward()
        self.outer_optimizer.step()
        times.append(time.time() - start)
        running_loss += loss.item()
        nb_iters += 1
        print("Outer train iteration:", nb_iters, "\nRunning loss:", running_loss/nb_iters)
        losses.append(running_loss/nb_iters)
    return nb_iters, iters, losses, times
  
  def evaluate(self, test_dataloader, mu):
    """
    Find the optimal outer solution.
      param mu: outer variable
      param test_dataloader: test data
    """
    running_loss, nb_iters, losses = 0, 0, []
    #for X_outer, y_outer in self.outer_dataloader:
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
        if nb_iters<4:
          loss = self.outer_loss(mu, inner_value, y_outer)
        else:
          loss = self.outer_loss(mu, inner_value, y_outer, accuracy=True)
        running_loss += loss.item()
        nb_iters += 1
        #print("Outer test iteration:", nb_iters, "\nRunning loss:", running_loss/nb_iters)
        losses.append(running_loss/nb_iters)
    return nb_iters, losses